import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union
from transformers import BertModel,BertTokenizer
# from pytorch_pretrained_bert import BertTokenizer, BertModel
from torchvision import transforms
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib
matplotlib.rcParams['axes.unicode_minus']=False


import torch
from torch import nn
from packaging.version import parse as V
from typeguard import check_argument_types
import torch.nn.functional as F
from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.transducer.error_calculator import ErrorCalculatorTransducer
from espnet2.asr_transducer.utils import get_transducer_task_io
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (  # noqa: H301
    LabelSmoothingLoss,
)
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
import numpy as np
from espnet.nets.pytorch_backend.conformer.encoder_layer import EncoderLayer
# from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
    LegacyRelPositionMultiHeadedAttention,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.nets_utils import get_activation
from espnet.nets.pytorch_backend.transformer.embedding import (
    PositionalEncoding,  # noqa: H301
    ScaledPositionalEncoding,  # noqa: H301
    RelPositionalEncoding,  # noqa: H301
    LegacyRelPositionalEncoding,  # noqa: H301
)
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield

class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

#然后送入这个
class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, 512),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class ESPnetASRModel(AbsESPnetModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        bias_encoder: Optional[AbsEncoder],
        postencoder: Optional[AbsPostEncoder],
        decoder: AbsDecoder,
        ctc: CTC,
        joint_network: Optional[torch.nn.Module],
        ctc_weight: float = 0.5,
        interctc_weight: float = 0.0,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        extract_feats_in_collect_stats: bool = True,
    ):
        assert check_argument_types()
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert 0.0 <= interctc_weight < 1.0, interctc_weight

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.blank_id = 0
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.interctc_weight = interctc_weight
        self.token_list = token_list.copy()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.postencoder = postencoder
        self.encoder = encoder
        self.bias_encoder = bias_encoder

        if not hasattr(self.encoder, "interctc_use_conditioning"):
            self.encoder.interctc_use_conditioning = False
        if self.encoder.interctc_use_conditioning:
            self.encoder.conditioning_layer = torch.nn.Linear(
                vocab_size, self.encoder.output_size()
            )

        self.use_transducer_decoder = joint_network is not None

        self.error_calculator = None

        if self.use_transducer_decoder:
            from warprnnt_pytorch import RNNTLoss

            self.decoder = decoder
            self.joint_network = joint_network

            self.criterion_transducer = RNNTLoss(
                blank=self.blank_id,
                fastemit_lambda=0.0,
            )

            if report_cer or report_wer:
                self.error_calculator_trans = ErrorCalculatorTransducer(
                    decoder,
                    joint_network,
                    token_list,
                    sym_space,
                    sym_blank,
                    report_cer=report_cer,
                    report_wer=report_wer,
                )
            else:
                self.error_calculator_trans = None

                if self.ctc_weight != 0:
                    self.error_calculator = ErrorCalculator(
                        token_list, sym_space, sym_blank, report_cer, report_wer
                    )
        else:
            # we set self.decoder = None in the CTC mode since
            # self.decoder parameters were never used and PyTorch complained
            # and threw an Exception in the multi-GPU experiment.
            # thanks Jeff Farris for pointing out the issue.
            if ctc_weight == 1.0:
                self.decoder = None
            else:
                self.decoder = decoder

            self.criterion_att = LabelSmoothingLoss(
                size=vocab_size,
                padding_idx=ignore_id,
                smoothing=lsm_weight,
                normalize_length=length_normalized_loss,
            )

            if report_cer or report_wer:
                self.error_calculator = ErrorCalculator(
                    token_list, sym_space, sym_blank, report_cer, report_wer
                )

        if ctc_weight == 0.0:
            self.ctc = None
        else:
            self.ctc = ctc
        
        drop_rate=0.0
        activation = get_activation("swish")
        
        self.context_embedding1 = torch.nn.Embedding(
            num_embeddings=5000,
            embedding_dim=512,
        )
        
        encoder_selfattn_layer = RelPositionMultiHeadedAttention
        encoder_selfattn_layer_args=(4, 512, drop_rate, False)

 
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args=(512, 1024, drop_rate, activation)

        self.embed =RelPositionalEncoding(512,drop_rate)
        self.context_encoder= repeat(
                5,
                lambda lnum: EncoderLayer(
                    512,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    None,
                    None,
                    drop_rate,
                    True,
                    False,
                    0.0,
            ),
        )

        self._bias_encoder= repeat(
                1,
                lambda lnum: EncoderLayer(
                    512,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args),
                    None,
                    None,
                    drop_rate,
                    True,
                    False,
                    0.0,
            ),
        )
        attention_dim=512
        attention_heads=4
        attention_dropout_rate=0
        dropout_rate=0
        linear_units=1024

        self._encoders = EncoderLayer(
            attention_dim,
            MultiHeadedAttention(attention_heads, attention_dim, attention_dropout_rate),
            PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
            dropout_rate,
            True,
            False,
            )

        self.after_norm = torch.nn.LayerNorm(512)

        

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

        self.MHA = MultiHeadedAttention(
                4, 512, 0
            )

        dim = 1024
        mult = 4
        dropout = 0.5
        

        self.drop = nn.Dropout(dropout)
        ninp = 1024
        # self.embed = nn.Embedding(5000, 512, padding_idx=ignore_id)
        self.feed_forward = FeedForward(dim=1024)
        self.linear = nn.Linear(512,512)
        self.feed_forward_bert = FeedForward(dim=768)
        # self.context_encoder = nn.LSTM(input_size=512,
        #                                 hidden_size=1024,
        #                                 num_layers=4,
        #                                 dropout=dropout,
        #                                 batch_first=True,
        #                                 bidirectional=True)
        
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.bert = self.bert.requires_grad_(False)
        self.tokennizer_bert = BertTokenizer.from_pretrained("bert-base-cased")

        # bert = BertModel.from_pretrained("bert-base-cased")
        # word_embeddings = bert.embeddings.word_embeddings
        # self.bert_model = word_embeddings
        # self.bert_model = torch.nn.Sequential(bert,torch.nn.Linear(768, 1))
        # weight = next(self.context_encoder.parameters())
    # def init_hidden(self,bsz):
    #     weight = next(self.parameters())
    #     # return (weight.new_zeros(self.context_encoder.num_layers, bsz, self.context_encoder.hidden_size),
    #     #         weight.new_zeros(self.context_encoder.num_layers, bsz, self.context_encoder.hidden_size))
    #     return (weight.new_zeros(self.context_encoder.num_layers, bsz, self.context_encoder.hidden_size))
        
    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
            kwargs: "utt_id" is among the input.
        """
        # 当kwargs中包含context_text时，把他取出来，参与下面的计算：
        if kwargs.__contains__("context_text"):
            context_text = kwargs.get("context_text")
            context_text_lengths = kwargs.get("context_text_lengths")
            assert context_text_lengths.dim() == 1, context_text_lengths.shape
            assert text_lengths.dim() == 1, text_lengths.shape
            # Check that batch_size is unified
            assert (
                speech.shape[0]
                == speech_lengths.shape[0]
                == text.shape[0]
                == text_lengths.shape[0]
                == context_text.shape[0]
                == context_text_lengths.shape[0]
            ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape, context_text.shape, context_text_lengths.shape)
            batch_size = speech.shape[0]

            # for data-parallel

            #________________________________________________________________________________________
            text = text[:, : text_lengths.max()]


            context_text = context_text[:, : context_text_lengths.max()]
            # self.bert.to("cuda")
            # output = self.bert(context_text)[0]
            context_embedding1 = self.context_embedding1.weight.unsqueeze(0).repeat(batch_size,1,1)
            
            context_in_pad,context_out_pad = add_sos_eos(context_text, 0, 0, 0)



            context_in_pad_flag=context_in_pad
            context_mask=context_in_pad.ge(0).unsqueeze(1)
            context_in_pad=context_in_pad+(~context_in_pad.ge(0))
            context_text_lengths = context_text_lengths + 1

            '''
            scores2=torch.nn.functional.one_hot(context_in_pad,num_classes=5000).to(torch.float32)
            
            context_embedding=torch.matmul(scores2, context_embedding1)
            # # 对 coontext_in_pad进行mask
            dcontext_in_pad_masked=context_in_pad.clone().detach()
            n = dcontext_in_pad_masked.numel()
            m = int(round(n*0.25))

            indices = np.random.choice(n, m, replace=False)
            dcontext_in_pad_masked.flatten()[indices] = 0

            mask1=dcontext_in_pad_masked.ge(1).unsqueeze(-1).repeat(1,1,512)
            mask2=~dcontext_in_pad_masked.ge(1).unsqueeze(-1).repeat(1,1,512)

            zero_context=torch.flip(context_embedding.clone(),[0])

            context_embedding=(context_embedding*mask1+zero_context*mask2)
            context_embedding,pos_emp=self.embed(context_embedding)
            context_embedding, context_mask = self.context_encoder((context_embedding,pos_emp),context_mask)

            context_embedding=context_embedding[0]

            context_embedding=self.after_norm(context_embedding).cuda()
            '''
            #________________________________________________________________________________________________
            # self.context_encoder.to("cpu")
            # text = F.pad(text, [1, 0], "constant", self.eos)

            # 没修改成功的LSTMEncoder 
            # context_in_pad,context_out_pad = add_sos_eos(context_text, 0, 0, 0)
            # text_lengths = text_lengths + 1
            # embed_text = self.embed(context_in_pad)
            # embedding = self.drop(embed_text)
            # h = torch.zeros((self.context_encoder.num_layers*2,embedding.size()[0], self.context_encoder.hidden_size), dtype=torch.float)
            # c = torch.zeros((self.context_encoder.num_layers*2,embedding.size()[0], self.context_encoder.hidden_size), dtype=torch.float)
            # hidden = h, c
            # # hidden = (weight.new_zeros(4,batch_size,1024),weight.new_zeros(4,batch_size,1024))
            # output,hidden = self.context_encoder(embedding,hidden)
            # self.context_encoder.to("cuda")
            # # embedded = self.drop(F.relu(self.embed(context_text)))
            # output, (hidden, cell)= self.context_encoder(context_embedding) 


            # 1. Encoder
            encoder_out, encoder_out_lens= self.encode(speech, speech_lengths)

            # self.MHA = MultiHeadedAttention(
            #     4, 512, 0
            # )

            
            # #将MHA的设置为和encoder_out相同的device#
            devices = encoder_out.device

            with torch.no_grad():
                '''
                test_text = [0,2,629,12,565,4,306]
                input_ids = torch.tensor(test_text).cuda()
                # self.bert_model=self.bert_model.to(devices)
                # self.bert_model=self.bert_model.eval()
                # embeddings1 = self.bert_model(input_ids,1)
                self.bert=self.bert.to(devices)
                self.bert=self.bert.eval()
                embeddings = self.bert(input_ids)
                '''
                
                self.bert = self.bert.to(devices)
                context_in_pad = context_in_pad.squeeze(dim=0).cuda()
                if context_in_pad.dim() < 2:
                    context_in_pad = context_in_pad.unsqueeze(dim=0).cuda()
                context_embedding = self.bert(context_in_pad)[0]

                '''
                # 将Context Embedding 的 Tensor 画出来
                uploader = transforms.ToPILImage()
                image = context_embedding.cpu().clone()
                image = image.squeeze(0)
                image = uploader(image)
                image.save("/home/yachao001/research_espnet/egs2/librispeech/asr1/exp/asr_12-14/embedding.jpg")
                context_embedding = context_embedding.detach()
                context_embedding = self.feed_forward_bert(context_embedding)
                # 将Context Embedding 的 Tensor 画出来结束
                '''
                # TNSE画图
                A = context_embedding[0].cpu().clone()
                context_embedding_cpu = context_embedding.cpu().clone()
                plt.imshow(A, cmap=plt.get_cmap('coolwarm'), vmin=A.min(), vmax=A.max())
                plt.colorbar()
                plt.show()
                plt.savefig("/home/yachao001/research_espnet/egs2/librispeech/asr1/exp/asr_12-14/embedding.jpg")





                #TNSE画图结束


            self.MHA = self.MHA.to(devices)

            MHA_out = self.MHA(encoder_out,context_embedding,context_embedding,mask=None)
            
            
            
            # Combiner Layer
            encoder_out = self.after_norm(encoder_out)
            MHA_out = self.after_norm(MHA_out)
            concat=torch.cat((encoder_out,MHA_out), -1)

            concat = self.feed_forward(concat)

            encoder_out = concat
            
            # n1 = encoder_out.size()[1]
            # m1 = concat.size()[1]
            # # encoder_out = self.linear_layer(concat,m1,n1)
            # concat = concat.permute(0,2,1)
            # self.linear = torch.nn.Linear(m1,n1).to(device=devices)
            # concat = self.linear(concat).permute(0,2,1)


            # attention_dim=512
            # dropout_rate=0
            # linear_units=1024

            # positionwise_layer = PositionwiseFeedForward
            # positionwise_layer_args = (
            #     attention_dim,
            #     linear_units,
            #     dropout_rate,
            # )


            # self.positionwise_layer=positionwise_layer(*positionwise_layer_args)

            # self.positionwise_layer=self.positionwise_layer.to(device)

            # encoder_out = self.positionwise_layer(concat)
            



            intermediate_outs = None
            if isinstance(encoder_out, tuple):
                intermediate_outs = encoder_out[1]
                encoder_out = encoder_out[0]

            loss_att, acc_att, cer_att, wer_att = None, None, None, None
            loss_ctc, cer_ctc = None, None
            loss_transducer, cer_transducer, wer_transducer = None, None, None
            stats = dict()

            # 1. CTC branch

            

            if self.ctc_weight != 0.0:
                loss_ctc, cer_ctc = self._calc_ctc_loss(
                    encoder_out, encoder_out_lens, text, text_lengths
                )

                # Collect CTC branch stats
                stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
                stats["cer_ctc"] = cer_ctc

            # Intermediate CTC (optional)
            loss_interctc = 0.0
            if self.interctc_weight != 0.0 and intermediate_outs is not None:
                for layer_idx, intermediate_out in intermediate_outs:
                    # we assume intermediate_out has the same length & padding
                    # as those of encoder_out
                    loss_ic, cer_ic = self._calc_ctc_loss(
                        intermediate_out, encoder_out_lens, text, text_lengths
                    )
                    loss_interctc = loss_interctc + loss_ic

                    # Collect Intermedaite CTC stats
                    stats["loss_interctc_layer{}".format(layer_idx)] = (
                        loss_ic.detach() if loss_ic is not None else None
                    )
                    stats["cer_interctc_layer{}".format(layer_idx)] = cer_ic

                loss_interctc = loss_interctc / len(intermediate_outs)

                # calculate whole encoder loss
                loss_ctc = (
                    1 - self.interctc_weight
                ) * loss_ctc + self.interctc_weight * loss_interctc

            if self.use_transducer_decoder:
                # 2a. Transducer decoder branch
                (
                    loss_transducer,
                    cer_transducer,
                    wer_transducer,
                ) = self._calc_transducer_loss(
                    encoder_out,
                    encoder_out_lens,
                    text,
                )

                if loss_ctc is not None:
                    loss = loss_transducer + (self.ctc_weight * loss_ctc)
                else:
                    loss = loss_transducer

                # Collect Transducer branch stats
                stats["loss_transducer"] = (
                    loss_transducer.detach() if loss_transducer is not None else None
                )
                stats["cer_transducer"] = cer_transducer
                stats["wer_transducer"] = wer_transducer

            else:
                # 2b. Attention decoder branch
                if self.ctc_weight != 1.0:
                    loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                        encoder_out, encoder_out_lens, text, text_lengths
                    )

                # 3. CTC-Att loss definition
                if self.ctc_weight == 0.0:
                    loss = loss_att
                elif self.ctc_weight == 1.0:
                    loss = loss_ctc
                else:
                    loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

                # Collect Attn branch stats
                stats["loss_att"] = loss_att.detach() if loss_att is not None else None
                stats["acc"] = acc_att
                stats["cer"] = cer_att
                stats["wer"] = wer_att

            # Collect total loss stats
            stats["loss"] = loss.detach()

            # force_gatherable: to-device and to-tensor if scalar for DataParallel
            loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
            return loss, stats, weight

        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
        loss_transducer, cer_transducer, wer_transducer = None, None, None
        stats = dict()

        # 1. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )

            # Collect CTC branch stats
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc

        # Intermediate CTC (optional)
        loss_interctc = 0.0
        if self.interctc_weight != 0.0 and intermediate_outs is not None:
            for layer_idx, intermediate_out in intermediate_outs:
                # we assume intermediate_out has the same length & padding
                # as those of encoder_out
                loss_ic, cer_ic = self._calc_ctc_loss(
                    intermediate_out, encoder_out_lens, text, text_lengths
                )
                loss_interctc = loss_interctc + loss_ic

                # Collect Intermedaite CTC stats
                stats["loss_interctc_layer{}".format(layer_idx)] = (
                    loss_ic.detach() if loss_ic is not None else None
                )
                stats["cer_interctc_layer{}".format(layer_idx)] = cer_ic

            loss_interctc = loss_interctc / len(intermediate_outs)

            # calculate whole encoder loss
            loss_ctc = (
                1 - self.interctc_weight
            ) * loss_ctc + self.interctc_weight * loss_interctc

        if self.use_transducer_decoder:
            # 2a. Transducer decoder branch
            (
                loss_transducer,
                cer_transducer,
                wer_transducer,
            ) = self._calc_transducer_loss(
                encoder_out,
                encoder_out_lens,
                text,
            )

            if loss_ctc is not None:
                loss = loss_transducer + (self.ctc_weight * loss_ctc)
            else:
                loss = loss_transducer

            # Collect Transducer branch stats
            stats["loss_transducer"] = (
                loss_transducer.detach() if loss_transducer is not None else None
            )
            stats["cer_transducer"] = cer_transducer
            stats["wer_transducer"] = wer_transducer

        else:
            # 2b. Attention decoder branch
            if self.ctc_weight != 1.0:
                loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                    encoder_out, encoder_out_lens, text, text_lengths
                )

            # 3. CTC-Att loss definition
            if self.ctc_weight == 0.0:
                loss = loss_att
            elif self.ctc_weight == 1.0:
                loss = loss_ctc
            else:
                loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

            # Collect Attn branch stats
            stats["loss_att"] = loss_att.detach() if loss_att is not None else None
            stats["acc"] = acc_att
            stats["cer"] = cer_att
            stats["wer"] = wer_att

        # Collect total loss stats
        stats["loss"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if self.extract_feats_in_collect_stats:
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        else:
            # Generate dummy stats if extract_feats_in_collect_stats is False
            logging.warning(
                "Generating dummy stats for feats and feats_lengths, "
                "because encoder_conf.extract_feats_in_collect_stats is "
                f"{self.extract_feats_in_collect_stats}"
            )
            feats, feats_lengths = speech, speech_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        if self.encoder.interctc_use_conditioning:
            encoder_out, encoder_out_lens, _ = self.encoder(
                feats, feats_lengths, ctc=self.ctc
            )
        else:
            encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        # Post-encoder, e.g. NLU
        if self.postencoder is not None:
            encoder_out, encoder_out_lens = self.postencoder(
                encoder_out, encoder_out_lens
            )

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        if intermediate_outs is not None:
            return (encoder_out, intermediate_outs), encoder_out_lens

        return encoder_out, encoder_out_lens

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    def nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute negative log likelihood(nll) from transformer-decoder

        Normally, this function is called in batchify_nll.

        Args:
            encoder_out: (Batch, Length, Dim)
            encoder_out_lens: (Batch,)
            ys_pad: (Batch, Length)
            ys_pad_lens: (Batch,)
        """
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )  # [batch, seqlen, dim]
        batch_size = decoder_out.size(0)
        decoder_num_class = decoder_out.size(2)
        # nll: negative log-likelihood
        nll = torch.nn.functional.cross_entropy(
            decoder_out.view(-1, decoder_num_class),
            ys_out_pad.view(-1),
            ignore_index=self.ignore_id,
            reduction="none",
        )
        nll = nll.view(batch_size, -1)
        nll = nll.sum(dim=1)
        assert nll.size(0) == batch_size
        return nll

    def batchify_nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        batch_size: int = 100,
    ):
        """Compute negative log likelihood(nll) from transformer-decoder

        To avoid OOM, this fuction seperate the input into batches.
        Then call nll for each batch and combine and return results.
        Args:
            encoder_out: (Batch, Length, Dim)
            encoder_out_lens: (Batch,)
            ys_pad: (Batch, Length)
            ys_pad_lens: (Batch,)
            batch_size: int, samples each batch contain when computing nll,
                        you may change this to avoid OOM or increase
                        GPU memory usage
        """
        total_num = encoder_out.size(0)
        if total_num <= batch_size:
            nll = self.nll(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)
        else:
            nll = []
            start_idx = 0
            while True:
                end_idx = min(start_idx + batch_size, total_num)
                batch_encoder_out = encoder_out[start_idx:end_idx, :, :]
                batch_encoder_out_lens = encoder_out_lens[start_idx:end_idx]
                batch_ys_pad = ys_pad[start_idx:end_idx, :]
                batch_ys_pad_lens = ys_pad_lens[start_idx:end_idx]
                batch_nll = self.nll(
                    batch_encoder_out,
                    batch_encoder_out_lens,
                    batch_ys_pad,
                    batch_ys_pad_lens,
                )
                nll.append(batch_nll)
                start_idx = end_idx
                if start_idx == total_num:
                    break
            nll = torch.cat(nll)
        assert nll.size(0) == total_num
        return nll

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def _calc_transducer_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Compute Transducer loss.

        Args:
            encoder_out: Encoder output sequences. (B, T, D_enc)
            encoder_out_lens: Encoder output sequences lengths. (B,)
            labels: Label ID sequences. (B, L)

        Return:
            loss_transducer: Transducer loss value.
            cer_transducer: Character error rate for Transducer.
            wer_transducer: Word Error Rate for Transducer.

        """
        decoder_in, target, t_len, u_len = get_transducer_task_io(
            labels,
            encoder_out_lens,
            ignore_id=self.ignore_id,
            blank_id=self.blank_id,
        )

        self.decoder.set_device(encoder_out.device)
        decoder_out = self.decoder(decoder_in)

        joint_out = self.joint_network(
            encoder_out.unsqueeze(2), decoder_out.unsqueeze(1)
        )

        loss_transducer = self.criterion_transducer(
            joint_out,
            target,
            t_len,
            u_len,
        )

        cer_transducer, wer_transducer = None, None
        if not self.training and self.error_calculator_trans is not None:
            cer_transducer, wer_transducer = self.error_calculator_trans(
                encoder_out, target
            )

        return loss_transducer, cer_transducer, wer_transducer

    def _calc_batch_ctc_loss(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ):
        if self.ctc is None:
            return
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        # Calc CTC loss
        do_reduce = self.ctc.reduce
        self.ctc.reduce = False
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, text, text_lengths)
        self.ctc.reduce = do_reduce
        return loss_ctc
