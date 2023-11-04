# SO-VITS-SVC源码解析 16

# `vencoder/wavlm/WavLM.py`

这段代码是一个用于语音识别任务的PyTorch实现，它使用了来自微软研究院和《Large-Scale Self-Supervised Pre-training for Full Stack Speech Processing》论文中的WavLM模型。该代码的目的是训练一个大规模的高质量语音数据集，以提高模型的性能。这个实现包括以下组件：

1. 引入相关库和模块：PyTorch库、NumPy数组、math库以及一些其他用于数据处理和训练的模块。
2. 设置日志记录：定义一个logger变量，用于记录模型的训练和验证信息。
3. 加载数据集：从公共GitHub仓库中下载预训练的训练数据和验证数据，并使用Transformers库的TextDataset类加载数据。
4. 数据预处理：对数据进行清洗和处理，包括去除噪声、对文本进行编码以及对数据进行划分和划分训练集和验证集。
5. 训练模型：使用WavLM模型训练模型，包括设置模型的架构、优化器和损失函数，以及训练和验证模型的参数。
6. 输出模型：使用测试数据集评估模型的性能，并输出模型的预测结果。
7. 记录损失：使用 accumulated_accuracy 函数记录模型的损失。
8. 设置超参数：设置训练参数，包括学习率、批大小、训练轮数等。
9. 加载数据：从文件中读取数据，并使用map函数对数据进行处理。
10. 输出数据：使用map函数对数据进行处理，并将数据输出为torch.device函数。


```py
# --------------------------------------------------------
# WavLM: Large-Scale Self-Supervised  Pre-training  for Full Stack Speech Processing (https://arxiv.org/abs/2110.13900.pdf)
# Github source: https://github.com/microsoft/unilm/tree/master/wavlm
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq code bases
# https://github.com/pytorch/fairseq
# --------------------------------------------------------

import logging
import math
from typing import List, Optional, Tuple

import numpy as np
import torch
```

这段代码定义了一个基于BERT模型的自编码器，其中包括一个BERT前馈网络，一个layer norm层，以及一些有用的功能函数。

具体来说，这个自编码器由两个子网络组成，一个是输入层，另一个是输出层。输入层接受一个长达128个时间的序列，而输出层则输出一个长达256个时间的序列，每个时间步都可能是多种声音。

在BERT模型的预处理阶段，对输入文本应用了Bert的“norm”类组件，其中包括了Fp32GroupNorm和Fp32LayerNorm。

在网络的前半部分，对输入序列应用了GLU线性激活，并使用TransposeLast对最后一个时间步进行变换。

在网络的后半部分，应用了MultiheadAttention和SamePad，对输入序列中的每个时间步进行加权平均和保留最短的输入序列。

通过get_activation_fn函数，在网络的输出层应用了一个激活函数，用于对加权平均的输出进行非线性变换。

最后，通过init_bert_params函数，初始化了BERT模型的参数。

这个自编码器的目的是实现BERT模型的非线性映射，将其应用于语音合成的任务中。


```py
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from vencoder.wavlm.modules import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    GLU_Linear,
    GradMultiply,
    MultiheadAttention,
    SamePad,
    TransposeLast,
    get_activation_fn,
    init_bert_params,
)

```

This is a Python implementation of the CSS offset reset function. It creates a function called `基於掩码的掩码` which is used to return a `mask` object that overrides the default CSS offset reset behavior.

Here is the implementation of the function:
```pypython
def 基於掩碼的掩码(mask, offset, min_len):
   if len(mask) == 0:
       return mask
   max_prob = 0
   for idx, probs in enumerate(mask):
       prob = probs / np.sum(probs)
       if idx == len(mask) - 1:
           max_prob = prob
       else:
           max_prob = max(max_prob, prob)
   return mask, max_prob
掩码func = func poisson_示出血的概率，平均概率和事件发生概率
```
The function takes three arguments: `mask`, `offset`, and `min_len`.

The function first checks if the length of the mask is 0, in which case it returns the default behavior of the offset reset.

If the length of the mask is greater than 0, the function calculates the probability of each element in the mask using the `poisson_` function from the `scipy.stats` library, which calculates the cumulative distribution function of the Poisson distribution.

The function then finds the index of the maximum probability among all elements with the same index, and returns the mask object with the corresponding `max_prob` value.

If the function cannot find any elements with the same `max_prob` value, it returns the default behavior of the offset reset.

The function also takes care of the arguments `offset` and `min_len`.

The `offset` argument specifies the CSS offset reset value, and `min_len` is the minimum length of the mask.


```py
logger = logging.getLogger(__name__)


def compute_mask_indices(
    shape: Tuple[int, int],
    padding_mask: Optional[torch.Tensor],
    mask_prob: float,
    mask_length: int,
    mask_type: str = "static",
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape

    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
            poisson = sample from possion distribution with lambda = mask length
        min_masks: minimum number of masked spans
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
    """

    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    all_num_mask = int(
        # add a random number for probabilistic rounding
        mask_prob * all_sz / float(mask_length)
        + np.random.rand()
    )

    all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * sz / float(mask_length)
                + np.random.rand()
            )
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask

        if mask_type == "static":
            lengths = np.full(num_mask, mask_length)
        elif mask_type == "uniform":
            lengths = np.random.randint(mask_other, mask_length * 2 + 1, size=num_mask)
        elif mask_type == "normal":
            lengths = np.random.normal(mask_length, mask_other, size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == "poisson":
            lengths = np.random.poisson(mask_length, size=num_mask)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise Exception("unknown mask selection " + mask_type)

        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = np.random.randint(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - keep_length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.fromiter(
                    (e - s if e - s >= length + min_space else 0 for s, e in parts),
                    np.int,
                )
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = np.random.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            min_len = min(lengths)
            if sz - min_len <= num_mask:
                min_len = sz - num_mask - 1

            mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)

            mask_idc = np.asarray(
                [
                    mask_idc[j] + offset
                    for j in range(len(mask_idc))
                    for offset in range(lengths[j])
                ]
            )

        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))

    min_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        mask[i, mask_idc] = True

    return mask


```

This is a PyTorch implementation of a masked language modeling task. It appears to include several parameters for customizing themask length, coverage, and other aspects of the masked language model. The model is designed to handle input sequences of arbitrary length, but the visible output sequences are limited to 256 tokens.


```py
class WavLMConfig:
    def __init__(self, cfg=None):
        self.extractor_mode: str = "default"     # mode for feature extractor. default has a single group norm with d groups in the first conv block, whereas layer_norm has layer norms in every block (meant to use with normalize=True)
        self.encoder_layers: int = 12     # num encoder layers in the transformer

        self.encoder_embed_dim: int = 768     # encoder embedding dimension
        self.encoder_ffn_embed_dim: int = 3072     # encoder embedding dimension for FFN
        self.encoder_attention_heads: int = 12     # num encoder attention heads
        self.activation_fn: str = "gelu"     # activation function to use

        self.layer_norm_first: bool = False     # apply layernorm first in the transformer
        self.conv_feature_layers: str = "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2"     # string describing convolutional feature extraction layers in form of a python list that contains [(dim, kernel_size, stride), ...]
        self.conv_bias: bool = False     # include bias in conv encoder
        self.feature_grad_mult: float = 1.0     # multiply feature extractor var grads by this

        self.normalize: bool = False  # normalize input to have 0 mean and unit variance during training

        # dropouts
        self.dropout: float = 0.1     # dropout probability for the transformer
        self.attention_dropout: float = 0.1     # dropout probability for attention weights
        self.activation_dropout: float = 0.0     # dropout probability after activation in FFN
        self.encoder_layerdrop: float = 0.0     # probability of dropping a tarnsformer layer
        self.dropout_input: float = 0.0     # dropout to apply to the input (after feat extr)
        self.dropout_features: float = 0.0     # dropout to apply to the features (after feat extr)

        # masking
        self.mask_length: int = 10     # mask length
        self.mask_prob: float = 0.65     # probability of replacing a token with mask
        self.mask_selection: str = "static"     # how to choose mask length
        self.mask_other: float = 0     # secondary mask argument (used for more complex distributions), see help in compute_mask_indicesh
        self.no_mask_overlap: bool = False     # whether to allow masks to overlap
        self.mask_min_space: int = 1     # min space between spans (if no overlap is enabled)

        # channel masking
        self.mask_channel_length: int = 10     # length of the mask for features (channels)
        self.mask_channel_prob: float = 0.0     # probability of replacing a feature with 0
        self.mask_channel_selection: str = "static"     # how to choose mask length for channel masking
        self.mask_channel_other: float = 0     # secondary mask argument (used for more complex distributions), see help in compute_mask_indices
        self.no_mask_channel_overlap: bool = False     # whether to allow channel masks to overlap
        self.mask_channel_min_space: int = 1     # min space between spans (if no overlap is enabled)

        # positional embeddings
        self.conv_pos: int = 128     # number of filters for convolutional positional embeddings
        self.conv_pos_groups: int = 16     # number of groups for convolutional positional embedding

        # relative position embedding
        self.relative_position_embedding: bool = False     # apply relative position embedding
        self.num_buckets: int = 320     # number of buckets for relative position embedding
        self.max_distance: int = 1280     # maximum distance for relative position embedding
        self.gru_rel_pos: bool = False     # apply gated relative position embedding

        if cfg is not None:
            self.update(cfg)

    def update(self, cfg: dict):
        self.__dict__.update(cfg)


```

This is a class that wraps the neural network's output for the target有害 justice social organization. It inherits from the `torch.nn.Module` class and implements the following methods:

* `forward_features(features)`: This method applies the frontend (1x feature) to the input (2x feature). It is a custom method to perform this operation.
* `no_grad(self)`: This method applies the no-grad (1x feature) to the input (2x feature). It is a custom method to perform this operation.
* `GradMultiply(features, factor)`: This method applies the multi-scale feature to the input. It is a custom method to perform this operation.
* `apply_mask(features, mask_indices)`: This method applies the mask to the input. It takes the input (2x feature) and the mask (1x feature).
* `encoder(x, mask_grad, layer)`: This method applies the layers to the input. It takes the input (Bx feature), the mask gradient (1x feature), and the layer index (integer).


```py
class WavLM(nn.Module):
    def __init__(
        self,
        cfg: WavLMConfig,
    ) -> None:
        super().__init__()
        logger.info(f"WavLM Config: {cfg.__dict__}")

        self.cfg = cfg
        feature_enc_layers = eval(cfg.conv_feature_layers)
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        )

        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )

        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)

    def apply_mask(self, x, padding_mask):
        B, T, C = x.shape
        if self.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob,
                self.mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices

    def forward_padding_mask(
            self, features: torch.Tensor, padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(
            padding_mask.size(0), features.size(1), -1
        )
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        ret_conv: bool = False,
        output_layer: Optional[int] = None,
        ret_layer_results: bool = False,
    ):

        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)

        features = features.transpose(1, 2)
        features = self.layer_norm(features)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)

        if mask:
            x, mask_indices = self.apply_mask(
                features, padding_mask
            )
        else:
            x = features

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        x, layer_results = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1
        )

        res = {"x": x, "padding_mask": padding_mask, "features": features, "layer_results": layer_results}

        feature = res["features"] if ret_conv else res["x"]
        if ret_layer_results:
            feature = (feature, res["layer_results"])
        return feature, res["padding_mask"]


```

This is a PyTorch implementation of a convolutional neural network (CNN) model. It appears to be a custom implementation of a 3-stage convolutional neural network (CNN) model. The `CustomCNN` class is继承 from the `torchvision.models.detection.FasterRCNN` class and defines the custom architecture.

The custom architecture includes a 2D convolutional neural network (CNN) with a "custom" 3x3 convolutional neural network (CNN), followed by a layer normalization operation. The 2D CNN has 64 filters with astride 2,0, and a padding of 1. The custom 3x3 CNN has 32 filters with astride 2,0, and a padding of 1. The layer normalization operations are defined as `torch.nn.LayerNorm([dim, idim])` and `torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)` respectively.

The model takes an input of shape `(batch_size, input_shape)` and returns a tensor of shape `(batch_size, output_shape)`. The `output_shape` is determined by the custom 3x3 CNN, followed by the `layer_norm` operation, and then the `ReLU` activation function.


```py
class ConvFeatureExtractionModel(nn.Module):
    def __init__(
            self,
            conv_layers: List[Tuple[int, int, int]],
            dropout: float = 0.0,
            mode: str = "default",
            conv_bias: bool = False,
            conv_type: str = "default"
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
                n_in,
                n_out,
                k,
                stride,
                is_layer_norm=False,
                is_group_norm=False,
                conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (is_layer_norm and is_group_norm) is False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        self.conv_type = conv_type
        if self.conv_type == "default":
            in_d = 1
            self.conv_layers = nn.ModuleList()
            for i, cl in enumerate(conv_layers):
                assert len(cl) == 3, "invalid conv definition: " + str(cl)
                (dim, k, stride) = cl

                self.conv_layers.append(
                    block(
                        in_d,
                        dim,
                        k,
                        stride,
                        is_layer_norm=mode == "layer_norm",
                        is_group_norm=mode == "default" and i == 0,
                        conv_bias=conv_bias,
                    )
                )
                in_d = dim
        elif self.conv_type == "conv2d":
            in_d = 1
            self.conv_layers = nn.ModuleList()
            for i, cl in enumerate(conv_layers):
                assert len(cl) == 3
                (dim, k, stride) = cl

                self.conv_layers.append(
                    torch.nn.Conv2d(in_d, dim, k, stride)
                )
                self.conv_layers.append(torch.nn.ReLU())
                in_d = dim
        elif self.conv_type == "custom":
            in_d = 1
            idim = 80
            self.conv_layers = nn.ModuleList()
            for i, cl in enumerate(conv_layers):
                assert len(cl) == 3
                (dim, k, stride) = cl
                self.conv_layers.append(
                    torch.nn.Conv2d(in_d, dim, k, stride, padding=1)
                )
                self.conv_layers.append(
                    torch.nn.LayerNorm([dim, idim])
                )
                self.conv_layers.append(torch.nn.ReLU())
                in_d = dim
                if (i + 1) % 2 == 0:
                    self.conv_layers.append(
                        torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
                    )
                    idim = int(math.ceil(idim / 2))
        else:
            pass

    def forward(self, x, mask=None):

        # BxT -> BxCxT
        x = x.unsqueeze(1)
        if self.conv_type == "custom":
            for conv in self.conv_layers:
                if isinstance(conv, nn.LayerNorm):
                    x = x.transpose(1, 2)
                    x = conv(x).transpose(1, 2)
                else:
                    x = conv(x)
            x = x.transpose(2, 3).contiguous()
            x = x.view(x.size(0), -1, x.size(-1))
        else:
            for conv in self.conv_layers:
                x = conv(x)
            if self.conv_type == "conv2d":
                b, c, t, f = x.size()
                x = x.transpose(2, 3).contiguous().view(b, c * f, t)
        return x


```

This is a PyTorch implementation of a masked language modeling model called "MaskedLingProb". It takes a input tensor `x` of size `(batch_size, sequence_length, feature_dim)` and applies masking according to the value in the `mask` parameter. It also applies a learnable positional encoding based on the `layer` parameter.

The model has an additional layer `layer` which applies a masked ling masking process before the input tensor is passed through the layers.

The `extract_features` method applies the masked ling masking process and returns the masked tensor and the results of the maskedling prob task.

The `pos_conv` function applies a position-wise convolution to the input tensor.

The `mask_norm` function applies a learned position-wise normalization to the input tensor based on the `layer` parameter.

The `mask_norm_first` flag is set to `True` if `layer` is `None`.


```py
class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim

        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=args.conv_pos,
            padding=args.conv_pos // 2,
            groups=args.conv_pos_groups,
        )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (args.conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(args.conv_pos), nn.GELU())

        if hasattr(args, "relative_position_embedding"):
            self.relative_position_embedding = args.relative_position_embedding
            self.num_buckets = args.num_buckets
            self.max_distance = args.max_distance
        else:
            self.relative_position_embedding = False
            self.num_buckets = 0
            self.max_distance = 0

        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=args.encoder_ffn_embed_dim,
                    num_attention_heads=args.encoder_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=args.attention_dropout,
                    activation_dropout=args.activation_dropout,
                    activation_fn=args.activation_fn,
                    layer_norm_first=args.layer_norm_first,
                    has_relative_attention_bias=(self.relative_position_embedding and i == 0),
                    num_buckets=self.num_buckets,
                    max_distance=self.max_distance,
                    gru_rel_pos=args.gru_rel_pos,
                )
                for i in range(args.encoder_layers)
            ]
        )

        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop

        self.apply(init_bert_params)

    def forward(self, x, padding_mask=None, streaming_mask=None, layer=None):
        x, layer_results = self.extract_features(x, padding_mask, streaming_mask, layer)

        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)

        return x, layer_results

    def extract_features(self, x, padding_mask=None, streaming_mask=None, tgt_layer=None):

        if padding_mask is not None:
            x[padding_mask] = 0

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        layer_results = []
        z = None
        if tgt_layer is not None:
            layer_results.append((x, z))
        r = None
        pos_bias = None
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, z, pos_bias = layer(x, self_attn_padding_mask=padding_mask, need_weights=False,
                                       self_attn_mask=streaming_mask, pos_bias=pos_bias)
            if tgt_layer is not None:
                layer_results.append((x, z))
            if i == tgt_layer:
                r = x
                break

        if r is not None:
            x = r

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, layer_results


```

This is a Python implementation of a neural network model called Multi-Head self-attention layer. It consists of multiple layers of self-attention mechanism and a final layer with residual connection. The self-attention mechanism allows the model to focus on different parts of the input sequence when computing the attention scores.

The first layer has a parameter `self_attn_layer_norm` which normalizes the input features before attending them. The second layer also has a parameter `self_attn_layer_norm` and a variable `residual` which is added to the input. This allows the model to store previous information.

The `activation_name` is a string that specifies the activation function to use. The `activation_fn` is a function that returns the activation function.

The output of this model is the final attention weights and the residual values.

Please note that this is just an example implementation and may not work perfectly in all cases.


```py
class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
            self,
            embedding_dim: float = 768,
            ffn_embedding_dim: float = 3072,
            num_attention_heads: float = 8,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            activation_fn: str = "relu",
            layer_norm_first: bool = False,
            has_relative_attention_bias: bool = False,
            num_buckets: int = 0,
            max_distance: int = 0,
            rescale_init: bool = False,
            gru_rel_pos: bool = False,
    ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_name = activation_fn
        self.activation_fn = get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            has_relative_attention_bias=has_relative_attention_bias,
            num_buckets=num_buckets,
            max_distance=max_distance,
            rescale_init=rescale_init,
            gru_rel_pos=gru_rel_pos,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)

        if self.activation_name == "glu":
            self.fc1 = GLU_Linear(self.embedding_dim, ffn_embedding_dim, "swish")
        else:
            self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim)

    def forward(
            self,
            x: torch.Tensor,
            self_attn_mask: torch.Tensor = None,
            self_attn_padding_mask: torch.Tensor = None,
            need_weights: bool = False,
            pos_bias=None
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn, pos_bias = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
                attn_mask=self_attn_mask,
                position_bias=pos_bias
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            if self.activation_name == "glu":
                x = self.fc1(x)
            else:
                x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn, pos_bias = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
                attn_mask=self_attn_mask,
                position_bias=pos_bias
            )

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x
            if self.activation_name == "glu":
                x = self.fc1(x)
            else:
                x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, attn, pos_bias


```

# `vencoder/whisper/audio.py`

这段代码使用了Python中的一些机器学习库：FFmpeg，NumPy，PyTorch以及librosa。它们用于实现以下功能：

1. 从PyTorch中引入了lru_cache库，以在需要时缓存计算结果，以避免重复计算。
2. 从ffmpeg中引入了ffmpeg库，以执行音频文件的录制和处理。
3. 从typing中引入了Union类型，以表示需要的不同类型。
4. 从librosa中引入了librosa_mel_fn函数，用于将MEL（ Mel-Frequency 数据）转换为音频信号。
5. 在函数内部，使用exact_div函数对音频信号进行预分频，以便在训练过程中实现频率降权。


```py
from functools import lru_cache
from typing import Union

import ffmpeg
import numpy as np
import torch
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn

from .utils import exact_div

# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
```

这段代码定义了两个变量，HOP_LENGTH和CHUNK_LENGTH，以及一个常量N_SAMPLES和N_FRAMES。N_SAMPLES是CHUNK_LENGTH的两倍，因此每个CHUNK_LENGTH包含480000个样本，而每个N_FRAMES是N_SAMPLES除以HOP_LENGTH得到的商，也就是3000个。

接着定义了一个名为load_audio的函数，该函数接受一个音频文件和采样率参数。它通过尝试使用FFMPEG命令行工具读取音频文件并将其解码，然后将其转换为NumPy数组。解码过程中，如果出现错误，该函数将抛出一个RuntimeError。

最后，函数的实现通过返回NumPy数组中的音频波形，其单位是float32。


```py
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000: number of samples in a chunk
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000: number of frames in a mel spectrogram input


def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


```

这段代码定义了一个名为pad_or_trim的函数，用于对输入的音频信号进行填充或截短，以使其长度符合预期编码器的输入要求。

函数的第一个参数是一个音频信号的numpy数组，第二个参数是一个整数，表示要保留的音频样本数。第三个参数是一个整数或-1，表示要填充或截短的轴(在-1的情况下，表示截短轴)。

如果输入的音频信号是numpy数组，函数首先检查该数组是否为浮点数，如果是，则执行以下操作：

1. 如果输入的音频信号长度不等于要保留的音频样本数，则在输入数组中从指定轴(默认为-1)开始，向后或向前填充缺失的样本，以使输入数组的总长度等于要保留的音频样本数。

2. 如果输入的音频信号是稀疏数组(即包含稀疏值和稠密值)，则在输入数组中从指定轴(默认为-1)开始，向后或向前填充缺失的样本，以使输入数组的总长度等于要保留的音频样本数。如果输入数组本身就是稀疏数组，则直接返回输入数组。

3. 如果输入的音频信号是numpy数组，且指定轴是-1，则执行以下操作：

- 从输入数组中删除所有元素，使得输入数组中没有一个稀疏值。
- 在指定的轴(默认为-1)处，创建一个包含输入数组长度-1的新的空数组，用于存储填充或截短的后的音频样本数。
- 将新数组中的元素设置为输入数组中对应位置的元素的值，其中值小于0的元素被替换为零，值大于0的元素保持不变。
- 返回新数组。

如果输入的音频信号是numpy数组，且指定轴是0，则执行以下操作：

- 从输入数组中删除所有元素，使得输入数组中没有一个稀疏值。
- 在指定的轴(默认为0)处，创建一个包含输入数组长度-1的新的空数组，用于存储填充或截短的后的音频样本数。
- 将新数组中的元素设置为输入数组中对应位置的元素的值，其中值小于0的元素被替换为零，值大于0的元素保持不变。
- 返回新数组。

如果输入的音频信号是numpy数组，且指定轴不是-1或0，则执行以下操作：

- 如果输入的音频信号长度不等于要保留的音频样本数，则在输入数组中从指定轴(指定为-1或0)开始，向后或向前填充缺失的样本，以使输入数组的总长度等于要保留的音频样本数。

- 如果输入的音频信号是稀疏数组，则在输入数组中从指定轴(指定为-1或0)开始，向后或向前填充缺失的样本，以使输入数组的总长度等于要保留的音频样本数。如果输入数组本身就是稀疏数组，则直接返回输入数组。

- 如果输入的音频信号是numpy数组，且指定轴是-1或0，则在输入数组中从指定轴(指定为-1或0)开始，向后或向前填充缺失的样本，以使输入数组的总长度等于要保留的音频样本数。

- 如果输入的音频信号是稀疏数组，则在输入数组中从指定轴(指定为-1或0)开始，向后或向前填充缺失的样本，以使输入数组的总长度等于要保留的音频样本数。如果输入数组本身就是稀疏数组，则直接返回输入数组。


```py
def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(dim=axis, index=torch.arange(length, device=array.device))

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


```

这段代码定义了一个名为 "mel\_filters" 的函数，它接受一个名为 "device" 的输入参数和一个名为 "n\_mels" 的输入参数（默认值为 80）。

函数实现中使用了 LRU 缓存机制来优化存储，其中使用了 librosa 库中的 mel\_fn 函数将一个采样率 SAR（此处为 16000，注意是采样率而非帧率）转换为一个 mel 分频图像，然后传送到 device 指定的设备上，最后将结果保存到设备地 numpy 数组中，也可以通过文件系统 np.savez_compressed 进行压缩存储。

总结：该函数的作用是加载一个 mel 分频图像，并将其存储到 device 指定的位置，可以用于将 librosa 中处理的 STFT 数据投影到 mel 分频空间，从而实现将声音信号与图像信号分离的目标。


```py
@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int = N_MELS) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        )
    """
    assert n_mels == 80, f"Unsupported n_mels: {n_mels}"
    return torch.from_numpy(librosa_mel_fn(sr=SAMPLE_RATE,n_fft=N_FFT,n_mels=n_mels)).to(device)


```

这段代码定义了一个名为 `log_mel_spectrogram` 的函数，它的输入参数是一个音频信号，以及一个表示滤波器数量常数的整数。函数返回一个包含 80 个 Mel-频率带宽的音频信号。

具体来说，函数首先检查输入音频是否是字符串类型，如果是，函数将加载音频并将其转换为 PyTorch 中的 Tensor 类型。如果输入音频不是字符串类型，函数将直接将其转换为 PyTorch 中的 Tensor 类型。

然后函数对输入音频进行快速傅里叶变换（FFT），并提取出音频信号的幅度信息。接下来，函数根据设定的滤波器数量从音频信号中提取 Mel-频率带宽的音频信号。函数将提取的音频信号与预设的滤波器相乘，并计算 Mel-频率带宽的幅度信息。最后，函数将幅度信息对 10 的幂函数进行开根号，并将得到的结果取对数，得到一个在对数形式的 Mel-频率带宽的音频信号。

函数的实现中，使用了两个函数 mel\_filters 和 magnitudes。其中，mel\_filters 函数根据设定的滤波器数量从音频信号中提取 Mel-频率带宽的音频信号，而 magnitudes 函数则计算了音频信号的幅度信息。这两个函数的具体实现并未在函数中给出，因此无法提供更多信息。


```py
def log_mel_spectrogram(audio: Union[str, np.ndarray, torch.Tensor], n_mels: int = N_MELS):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    Returns
    -------
    torch.Tensor, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)

    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec

```

# `vencoder/whisper/decoding.py`

这段代码定义了一个名为 `Whisper` 的类，并从两个数据类中定义了一些属性和方法。

首先从 `dataclasses` 模块中定义了 `Whisper` 类，定义了 `Whisper` 类中需要使用的属性和方法。然后从 `typing` 模块中定义了一些类型，以便在代码中使用。

接下来是定义了一些变量，包括从 `numpy` 模块中导入的 `CHUNK_LENGTH` 变量，它定义了每个音频块的长度。接下来从 `torch` 模块中导入了一些函数，包括 `F.torch.tensor` 函数，用于创建一个 `Tensor` 对象。

然后从 `torch.distributions` 模块中导入了一个名为 `Categorical` 的函数，定义了一种类型的分布，稍后会在代码中使用。

接下来定义了一个名为 `compression_ratio` 的函数，用于计算压缩比率。

接着是定义了一个名为 `Whisper` 的类，包含了一些用于实现 `Whisper` 类的方法和属性。

最后导入了一些内部类和函数，以便在 `Whisper` 类中实现更具体的逻辑。


```py
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical

from .audio import CHUNK_LENGTH
from .tokenizer import Tokenizer, get_tokenizer
from .utils import compression_ratio

if TYPE_CHECKING:
    from .model import Whisper


```

This is a function that performs language identification on an input audio signal. It takes in a pre-trained language model, the audio signal, and the encoding format used for the audio signal (e.g., 2D or 3D).

First, the audio signal is converted to a 2D format, if it is not already in that format. The audio signal is then passed through the pre-trained language model to obtain the logits. These logits are then used to compute the probability distribution over all languages. The final output is a tuple of the detected language tokens and the corresponding probability distribution over all languages.

If the input audio signal is encoded in 3D format, the function first splits the audio signal into left and right half-舱，然后将左半舱的音频信号输入到预训练语言模型中。最后，对左右半舱的输出进行拼接，再根据需要对检测到的语言进行概率加权。


```py
@torch.no_grad()
def detect_language(model: "Whisper", mel: Tensor, tokenizer: Tokenizer = None) -> Tuple[Tensor, List[dict]]:
    """
    Detect the spoken language in the audio, and return them as list of strings, along with the ids
    of the most probable language tokens and the probability distribution over all language tokens.
    This is performed outside the main decode loop in order to not interfere with kv-caching.

    Returns
    -------
    language_tokens : Tensor, shape = (n_audio,)
        ids of the most probable language tokens, which appears after the startoftranscript token.
    language_probs : List[Dict[str, float]], length = n_audio
        list of dictionaries containing the probability distribution over all languages.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer(model.is_multilingual)
    if tokenizer.language is None or tokenizer.language_token not in tokenizer.sot_sequence:
        raise ValueError("This model doesn't have language tokens so it can't perform lang id")

    single = mel.ndim == 2
    if single:
        mel = mel.unsqueeze(0)

    # skip encoder forward pass if already-encoded audio features were given
    if mel.shape[-2:] != (model.dims.n_audio_ctx, model.dims.n_audio_state):
        mel = model.encoder(mel)

    # forward pass using a single token, startoftranscript
    n_audio = mel.shape[0]
    x = torch.tensor([[tokenizer.sot]] * n_audio).to(mel.device)  # [n_audio, 1]
    logits = model.logits(x, mel)[:, 0]

    # collect detected languages; suppress all non-language tokens
    mask = torch.ones(logits.shape[-1], dtype=torch.bool)
    mask[list(tokenizer.all_language_tokens)] = False
    logits[:, mask] = -np.inf
    language_tokens = logits.argmax(dim=-1)
    language_token_probs = logits.softmax(dim=-1).cpu()
    language_probs = [
        {
            c: language_token_probs[i, j].item()
            for j, c in zip(tokenizer.all_language_tokens, tokenizer.all_language_codes)
        }
        for i in range(n_audio)
    ]

    if single:
        language_tokens = language_tokens[0]
        language_probs = language_probs[0]

    return language_tokens, language_probs


```

`@dataclass(frozen=True)` is a dataclass that follows the CSS dataclass naming conventions. `DecodingOptions` is the class that represents the decoding options for an audio transcription task.

The `task` field is a string that specifies whether to perform X->X "transcribe" or X->English "translate" operations. The `language` field is an optional string that indicates the language of the audio. If it is not specified, the task's language is inferred from the detected language.

The `temperature`, `sample_len`, `best_of`, `beam_size`, `patience`, `length_penalty`, `prompt`, `prefix`, `suppress_blank`, `suppress_tokens`, `timestamp`, `without_timestamps`, `max_initial_timestamp`, `fp16`, and `implementation_details` fields are all optional and have default values or are set to `True` to enable certain features.

The `DecodingOptions` class is used to configure the settings of an audio transcription task.


```py
@dataclass(frozen=True)
class DecodingOptions:
    task: str = "transcribe"  # whether to perform X->X "transcribe" or X->English "translate"
    language: Optional[str] = None  # language that the audio is in; uses detected language if None

    # sampling-related options
    temperature: float = 0.0
    sample_len: Optional[int] = None  # maximum number of tokens to sample
    best_of: Optional[int] = None     # number of independent samples to collect, when t > 0
    beam_size: Optional[int] = None   # number of beams in beam search, when t == 0
    patience: Optional[float] = None  # patience in beam search (https://arxiv.org/abs/2204.05424)

    # options for ranking generations (either beams or best-of-N samples)
    length_penalty: Optional[float] = None   # "alpha" in Google NMT, None defaults to length norm

    # prompt, prefix, and token suppression
    prompt: Optional[Union[str, List[int]]] = None   # text or tokens for the previous context
    prefix: Optional[Union[str, List[int]]] = None   # text or tokens to prefix the current context
    suppress_blank: bool = True                      # this will suppress blank outputs

    # list of tokens ids (or comma-separated token ids) to suppress
    # "-1" will suppress a set of symbols as defined in `tokenizer.non_speech_tokens()`
    suppress_tokens: Optional[Union[str, Iterable[int]]] = "-1"

    # timestamp sampling options
    without_timestamps: bool = False              # use <|notimestamps|> to sample text tokens only
    max_initial_timestamp: Optional[float] = 1.0  # the initial timestamp cannot be later than this

    # implementation details
    fp16: bool = True  # use fp16 for most of the calculation


```



这是一个名为 DecodingResult 的类，用于表示语音识别中的解码结果。类中定义了以下字段：

- audio_features：一个音频特征向量，可能是从听觉模型中获得的。
- language：当前正在处理的语言，可能是来自说话者的。
- language_probs：一个字典，其中包含当前正在处理的语言的概率分布。
- tokens：一个列表，包含当前正在处理的字节。
- text：当前正在生成的文本，可能是从说话者那里获得的。
- avg_logprob：一个平均逻辑概率，表示整个解码过程的总体概率。
- no_speech_prob：一个表示说话者是否说话的概率，可能是从另一个通道获得的。
- temperature：一个表示解码过程中使用的温度，可能是从另一个通道获得的。
- compression_ratio：一个表示压缩效果的比值，可能是从另一个通道获得的。

此外，还定义了两个子类 Inference 和 Decoding，用于执行实际的解码操作和数据预处理。其中 Inference 类包含一个方法 logits，该方法接受两个参数：当前正在处理的字节和音频特征向量。Decoding 类包含一个方法 cleanup_caching，用于清理在解码过程中产生的任何资源或钩子。


```py
@dataclass(frozen=True)
class DecodingResult:
    audio_features: Tensor
    language: str
    language_probs: Optional[Dict[str, float]] = None
    tokens: List[int] = field(default_factory=list)
    text: str = ""
    avg_logprob: float = np.nan
    no_speech_prob: float = np.nan
    temperature: float = np.nan
    compression_ratio: float = np.nan


class Inference:
    def logits(self, tokens: Tensor, audio_features: Tensor) -> Tensor:
        """Perform a forward pass on the decoder and return per-token logits"""
        raise NotImplementedError

    def rearrange_kv_cache(self, source_indices) -> None:
        """Update the key-value cache according to the updated beams"""
        raise NotImplementedError

    def cleanup_caching(self) -> None:
        """Clean up any resources or hooks after decoding is finished"""
        pass


```

这段代码定义了一个名为 PyTorchInference 的类，用于在 PyTorch 中进行基于 Whisper 模型的隐式推理。

在初始化函数 `__init__` 中，首先创建一个将 `model`、`initial_token_length` 和自定义的 hooks 存储在同一个类中的实例。其中，`model` 是一个 Whisper 模型，是一个需要加载预训练模型的 Whisper 类实例；`initial_token_length` 是初始化时文本的最低长度，即不包括该长度之后的文本将不再被考虑为可读文本；`kv_cache` 和 `hooks` 是用于保存已经计算出的 KV 数据和自定义的钩子的数据。

在 `logits` 函数中，首先检查 `kv_cache` 是否存在，如果不存在，则创建它并将 `model.install_kv_cache_hooks()` 方法中的钩子添加到 `kv_cache` 中。然后，检查输入 `tokens` 和 `audio_features` 的大小，如果 `tokens` 的形状比 `initial_token_length` 大，则 `tokens` 中的前 `initial_token_length` 个元素将被丢弃。接着，使用 `model.decoder` 方法对 `tokens` 和 `audio_features` 进行前馈，其中使用了 `kv_cache` 来存储已经计算出来的 KV 数据。

在 `cleanup_caching` 函数中，移除所有自定义的钩子，并清除 `kv_cache`。然后，在每次需要使用 Whisper 模型时，将 `model.install_kv_cache_hooks()` 方法中的钩子添加到 `kv_cache` 中。

在 `rearrange_kv_cache` 函数中，将 `kv_cache` 中存储的模块和对应的 `tensor` 中的数据按照 `source_indices` 进行重新排序，使得生成的音频能够涵盖所有已学习的音频特征。


```py
class PyTorchInference(Inference):
    def __init__(self, model: "Whisper", initial_token_length: int):
        self.model: "Whisper" = model
        self.initial_token_length = initial_token_length
        self.kv_cache = {}
        self.hooks = []

    def logits(self, tokens: Tensor, audio_features: Tensor) -> Tensor:
        if not self.kv_cache:
            self.kv_cache, self.hooks = self.model.install_kv_cache_hooks()

        if tokens.shape[-1] > self.initial_token_length:
            # only need to use the last token except in the first forward pass
            tokens = tokens[:, -1:]

        return self.model.decoder(tokens, audio_features, kv_cache=self.kv_cache)

    def cleanup_caching(self):
        for hook in self.hooks:
            hook.remove()

        self.kv_cache = {}
        self.hooks = []

    def rearrange_kv_cache(self, source_indices):
        for module, tensor in self.kv_cache.items():
            # update the key/value cache to contain the selected sequences
            self.kv_cache[module] = tensor[source_indices].detach()


```

这段代码定义了一个名为 `SequenceRanker` 的类，它实现了对输入序列的排名，并返回每个样本的索引。接着定义了一个名为 `MaximumLikelihoodRanker` 的类，它继承了 `SequenceRanker` 类，并实现了一个简单的长度加权最小化模型。

具体来说，`SequenceRanker` 类中的 `rank` 方法接收一个包含样本列表和每个样本的对数值列表的输入，然后按照以下步骤进行排名：

1. 对于每个样本，计算其在所有其他样本中的排名，即当前样本与所有其他样本的距离（通过 `欧几里得距离` 计算或者 `Sum of the absolute differences`）。
2. 计算当前样本与所有其他样本的分数，其中当前样本的分数为其在当前排名上的得分除以当前样本与所有其他样本的距离的乘积。
3. 使用当前样本的得分最高的样本的得分作为当前样本的最终得分。

`MaximumLikelihoodRanker` 类中的 `rank` 方法执行了 `SequenceRanker` 类中的 `rank` 方法，并使用一个可选的 `length_penalty` 参数对当前样本的评分进行惩罚。具体来说，如果 `length_penalty` 为 `None`，则当前样本的评分为其在所有其他样本中的得分。否则，当前样本的评分为其在所有其他样本中的得分加上一个根据其长度加权计算出的分数，该分数根据 Google NMT 论文中提出的比例系数 `5 + length / 6` 计算，其中 `length` 是样本的长度。


```py
class SequenceRanker:
    def rank(self, tokens: List[List[Tensor]], sum_logprobs: List[List[float]]) -> List[int]:
        """
        Given a list of groups of samples and their cumulative log probabilities,
        return the indices of the samples in each group to select as the final result
        """
        raise NotImplementedError


class MaximumLikelihoodRanker(SequenceRanker):
    """
    Select the sample with the highest log probabilities, penalized using either
    a simple length normalization or Google NMT paper's length penalty
    """

    def __init__(self, length_penalty: Optional[float]):
        self.length_penalty = length_penalty

    def rank(self, tokens: List[List[Tensor]], sum_logprobs: List[List[float]]):
        def scores(logprobs, lengths):
            result = []
            for logprob, length in zip(logprobs, lengths):
                if self.length_penalty is None:
                    penalty = length
                else:
                    # from the Google NMT paper
                    penalty = ((5 + length) / 6) ** self.length_penalty
                result.append(logprob / penalty)
            return result

        # get the sequence with the highest score
        lengths = [[len(t) for t in s] for s in tokens]
        return [np.argmax(scores(p, l)) for p, l in zip(sum_logprobs, lengths)]


```



This is a simple implementation of a Neural TTS model in PyTorch. It takes an input sequence of tokens, a couting log probabilities for each token, and a list of audio signals. The audio signals are assumed to be one-hot encoded, and the couting log probabilities are calculated based on the attention mechanism. The model initializes its state with the first token and zero log probabilities for the rest of the audio signals. It also initializes the output audio signals with zeros for the rest of the audio duration. The reset() method is defined to reset the internal state after an input sequence has been processed. The update() method is defined to select the next token based on the current log probabilities and the attention mechanism. The finalize() method is defined to finalize the search by selecting the best candidates based on the couting log probabilities and return them as a list of sequences.


```py
class TokenDecoder:
    def reset(self):
        """Initialize any stateful variables for decoding a new sequence"""

    def update(self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor) -> Tuple[Tensor, bool]:
        """Specify how to select the next token, based on the current trace and logits

        Parameters
        ----------
        tokens : Tensor, shape = (n_batch, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence tokens

        logits : Tensor, shape = (n_batch, vocab_size)
            per-token logits of the probability distribution at the current step

        sum_logprobs : Tensor, shape = (n_batch)
            cumulative log probabilities for each sequence

        Returns
        -------
        tokens : Tensor, shape = (n_batch, current_sequence_length + 1)
            the tokens, appended with the selected next token

        completed : bool
            True if all sequences has reached the end of text

        """
        raise NotImplementedError

    def finalize(
        self, tokens: Tensor, sum_logprobs: Tensor
    ) -> Tuple[Sequence[Sequence[Tensor]], List[List[float]]]:
        """Finalize search and return the final candidate sequences

        Parameters
        ----------
        tokens : Tensor, shape = (n_audio, n_group, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence

        sum_logprobs : Tensor, shape = (n_audio, n_group)
            cumulative log probabilities for each sequence

        Returns
        -------
        tokens : Sequence[Sequence[Tensor]], length = n_audio
            sequence of Tensors containing candidate token sequences, for each audio input

        sum_logprobs : List[List[float]], length = n_audio
            sequence of cumulative log probabilities corresponding to the above

        """
        raise NotImplementedError


```

这段代码定义了一个名为 "GreedyDecoder" 的类，它继承自名为 "TokenDecoder" 的类。

在这个类的初始化方法 "__init__" 中，首先创建了两个变量：

- "temperature": 温度参数，用于控制模型的胡乱程度。温度越高，胡乱程度越大。
- "eot": 一个整数参数，表示一个结束标记(eos)。当解码过程中某个位置的最后一个标记是 "eos"，则认为整个序列的编码成功。

接着，在 "update" 方法中，接收三个输入参数：

- "logits": 一个带有标签的二维张量，表示模型的当前输出。
- "sum_logprobs": 一个带有标签的二维张量，表示模型的当前输出对应的概率分布。
- "tokens": 一个带有标签的一维张量，表示当前的输入序列。

在这个方法的实现中，首先判断 "temperature" 是否为 0，如果是，则输出一个未经过乱序的 "logits" 向量。否则，使用 "Categorical" 对 "logits" 进行聚类，并输出聚类后的结果。

接着，计算当前输入序列的 "logprobs"，并使用 log 函数计算模型的胡乱程度。然后，根据当前的 "tokens"，输出一个已完成标记的 "logits" 向量，并使用 "all" 函数计算该向量是否包含最后一个标记 "eos"。

最后，使用 "pad" 函数对输入序列进行填充，以保证每个序列至少包含一个 "eos" 标记。

在 "finalize" 方法中，首先对输入序列中的最后一个标记 "eos" 进行填充，然后返回两个值：

- 一个未经过乱序的 "logits" 向量。
- 一个带有标签的二维张量，表示当前的输入序列和最后一个标记 "eos" 对应的概率分布。


```py
class GreedyDecoder(TokenDecoder):
    def __init__(self, temperature: float, eot: int):
        self.temperature = temperature
        self.eot = eot

    def update(self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor) -> Tuple[Tensor, bool]:
        temperature = self.temperature
        if temperature == 0:
            next_tokens = logits.argmax(dim=-1)
        else:
            next_tokens = Categorical(logits=logits / temperature).sample()

        logprobs = F.log_softmax(logits.float(), dim=-1)
        current_logprobs = logprobs[torch.arange(logprobs.shape[0]), next_tokens]
        sum_logprobs += current_logprobs * (tokens[:, -1] != self.eot)

        next_tokens[tokens[:, -1] == self.eot] = self.eot
        tokens = torch.cat([tokens, next_tokens[:, None]], dim=-1)

        completed = (tokens[:, -1] == self.eot).all()
        return tokens, completed

    def finalize(self, tokens: Tensor, sum_logprobs: Tensor):
        # make sure each sequence has at least one EOT token at the end
        tokens = F.pad(tokens, (0, 1), value=self.eot)
        return tokens, sum_logprobs.tolist()


```

It seems like `ange_kv_cache` is a function that uses the KV cache to store the probability tensors of a model for later usage. The function takes two arguments: `source_indices`, which is a list of indices indicating the start positions of the tokens in the source sentence, and `max_candidates`, which is an integer representing the maximum number of candidates that can be kept in the candidate list for each token.

The function returns two values: `tokens` and `completed`. `tokens` is a list of lists, where each inner list contains the finished token sequences for each token in the source sentence. The last inner list in each list contains the unfinished token sequence, if there are not enough candidates for that token yet.

The `completed` variable is a boolean indicating whether all audio sources have enough samples to be considered completed.

It appears that the function uses a recursive approach to fill in the `tokens` list. For each unfinished token sequence, the function generates all possible candidate solutions and keeps the highest one. If there are not enough candidates, the function continues to the next unfinished token sequence. The candidates are stored in the `candidates` list, which is defined with the `max_candidates` value.

After all unfinished token sequences have been processed, the function adds the finished token sequences to the `tokens` list.

Note that the audio samples are assumed to be stored in the `source_indices` list, which is passed as an argument. It is not clear from the code what happens to the `source_indices` list after it is passed to `ange_kv_cache`.


```py
class BeamSearchDecoder(TokenDecoder):
    def __init__(self, beam_size: int, eot: int, inference: Inference, patience: Optional[float] = None):
        self.beam_size = beam_size
        self.eot = eot
        self.inference = inference
        self.patience = patience or 1.0
        self.max_candidates: int = round(beam_size * self.patience)
        self.finished_sequences = None

        assert self.max_candidates > 0, f"Invalid beam size ({beam_size}) or patience ({patience})"

    def reset(self):
        self.finished_sequences = None

    def update(self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor) -> Tuple[Tensor, bool]:
        if tokens.shape[0] % self.beam_size != 0:
            raise ValueError(f"{tokens.shape}[0] % {self.beam_size} != 0")

        n_audio = tokens.shape[0] // self.beam_size
        if self.finished_sequences is None:  # for the first update
            self.finished_sequences = [{} for _ in range(n_audio)]

        logprobs = F.log_softmax(logits.float(), dim=-1)
        next_tokens, source_indices, finished_sequences = [], [], []
        for i in range(n_audio):
            scores, sources, finished = {}, {}, {}

            # STEP 1: calculate the cumulative log probabilities for possible candidates
            for j in range(self.beam_size):
                idx = i * self.beam_size + j
                prefix = tokens[idx].tolist()
                for logprob, token in zip(*logprobs[idx].topk(self.beam_size + 1)):
                    new_logprob = (sum_logprobs[idx] + logprob).item()
                    sequence = tuple(prefix + [token.item()])
                    scores[sequence] = new_logprob
                    sources[sequence] = idx

            # STEP 2: rank the candidates and keep the top beam_size sequences for each audio
            saved = 0
            for sequence in sorted(scores, key=scores.get, reverse=True):
                if sequence[-1] == self.eot:
                    finished[sequence] = scores[sequence]
                else:
                    sum_logprobs[len(next_tokens)] = scores[sequence]
                    next_tokens.append(sequence)
                    source_indices.append(sources[sequence])

                    saved += 1
                    if saved == self.beam_size:
                        break

            finished_sequences.append(finished)

        tokens = torch.tensor(next_tokens, device=tokens.device)
        self.inference.rearrange_kv_cache(source_indices)

        # add newly finished sequences to self.finished_sequences
        assert len(self.finished_sequences) == len(finished_sequences)
        for previously_finished, newly_finished in zip(self.finished_sequences, finished_sequences):
            for seq in sorted(newly_finished, key=newly_finished.get, reverse=True):
                if len(previously_finished) >= self.max_candidates:
                    break  # the candidate list is full
                previously_finished[seq] = newly_finished[seq]

        # mark as completed if all audio has enough number of samples
        completed = all(
            len(sequences) >= self.max_candidates for sequences in self.finished_sequences
        )
        return tokens, completed

    def finalize(self, preceding_tokens: Tensor, sum_logprobs: Tensor):
        # collect all finished sequences, including patience, and add unfinished ones if not enough
        sum_logprobs = sum_logprobs.cpu()
        for i, sequences in enumerate(self.finished_sequences):
            if len(sequences) < self.beam_size:  # when not enough sequences are finished
                for j in list(np.argsort(sum_logprobs[i]))[::-1]:
                    sequence = preceding_tokens[i, j].tolist() + [self.eot]
                    sequences[tuple(sequence)] = sum_logprobs[i][j].item()
                    if len(sequences) >= self.beam_size:
                        break

        tokens: List[List[Tensor]] = [
            [torch.tensor(seq) for seq in sequences.keys()] for sequences in self.finished_sequences
        ]
        sum_logprobs: List[List[float]] = [
            list(sequences.values()) for sequences in self.finished_sequences
        ]
        return tokens, sum_logprobs


```

这段代码定义了一个名为 `LogitFilter` 的类，其作用是在给定的 `logits` 和 `tokens` 输入数据上应用任何过滤或掩码操作，并将结果保存回 `logits` 中。

具体来说，这个类包含一个名为 `apply` 的方法，该方法接受两个参数：`logits` 和 `tokens`。在方法中，首先对 `logits` 进行处理，如果定义了一个 `apply_mask` 方法，则使用该方法对 `logits` 进行掩码操作。否则，直接将 `mask` 设置为 `True`，这样 `logits` 中的所有元素都将被认为是 active(即，具有非零概率)。

然后，将处理后的 `logits` 数据与 `tokens` 进行拼接，并将结果保存回 `logits` 中。这个 `apply` 方法仅在接收两个 `Tensor` 类型的参数时有效，因此它适用于输入数据是长文本序列的情况。


```py
class LogitFilter:
    def apply(self, logits: Tensor, tokens: Tensor) -> None:
        """Apply any filtering or masking to logits in-place

        Parameters
        ----------
        logits : Tensor, shape = (n_batch, vocab_size)
            per-token logits of the probability distribution at the current step

        tokens : Tensor, shape = (n_batch, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence tokens

        """
        raise NotImplementedError


```

这两 class 是PyTorch中的两个类，属于一个细粒度的数据增强类。主要用于对输入数据进行处理。

第一个类：SuppressBlank(LogitFilter)

这个类的作用是对文本数据中的空白字符进行处理，具体实现如下：
1. 初始化：构造函数以需要处理的文本数据为参数，同时记录下样本的起始位置。
2. apply：在apply方法中，将输入logits和对应的tokens作为参数。首先，获取当前的sample_begin位置，如果该位置离开始文本位置还有空格，就将其对应的值替换为-inf，以便在计算的时候知道哪些位置是无需计算的。然后，遍历输入logits和对应tokens的每一个位置，替换掉周围的空白字符（即从空白字符开始到end_of_word的位置）。最后，将修改后的logits和对应的tokens返回。

第二个类：SuppressTokens(LogitFilter)

这个类的作用是对需要进行预测的文本数据中的token进行处理，具体实现如下：
1. 初始化：构造函数以需要抑制的token序列为参数。
2. apply：在apply方法中，将输入logits和对应的tokens作为参数。首先，将suppress_tokens中的每一个token复制到logits中，同时对logits进行修改，使得对应位置的值替换为-inf。这样，当模型在进行预测的时候，suppress_tokens中所有的token都将被忽略，从而减小模型的预测误差。


```py
class SuppressBlank(LogitFilter):
    def __init__(self, tokenizer: Tokenizer, sample_begin: int):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin

    def apply(self, logits: Tensor, tokens: Tensor):
        if tokens.shape[1] == self.sample_begin:
            logits[:, self.tokenizer.encode(" ") + [self.tokenizer.eot]] = -np.inf


class SuppressTokens(LogitFilter):
    def __init__(self, suppress_tokens: Sequence[int]):
        self.suppress_tokens = list(suppress_tokens)

    def apply(self, logits: Tensor, tokens: Tensor):
        logits[:, self.suppress_tokens] = -np.inf


```

This code appears to be a masked language model pre-training step. Here is a summary of what it does:

1. It first loops through each token in the `tokens` array, except for the very last token.

2. For each token, it finds the sequence of token indices that correspond to the masked portion of the token.

3. If there is only one part of the token that is a timestamp (i.e., it was added to the original token by someone), the method checks whether the penultimate part of the token sequence was also a timestamp.

4. If the penultimate part of the token sequence was also a timestamp, the method performs some additional processing to take into account. This includes either normalizing the probabilities to avoid taking tokens that are too short or removing the token from the sequence altogether if it is not a timestamp.

5. Next, it applies a mask to the entire token sequence.

6. Finally, it returns the modified `logits` array.

The `masked_token_index` variable is a boolean mask that determines which tokens get masked. This allows the code to apply the mask to certain tokens while leaving others unmodified.


```py
class ApplyTimestampRules(LogitFilter):
    def __init__(
        self, tokenizer: Tokenizer, sample_begin: int, max_initial_timestamp_index: Optional[int]
    ):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin
        self.max_initial_timestamp_index = max_initial_timestamp_index

    def apply(self, logits: Tensor, tokens: Tensor):
        # suppress <|notimestamps|> which is handled by without_timestamps
        if self.tokenizer.no_timestamps is not None:
            logits[:, self.tokenizer.no_timestamps] = -np.inf

        # timestamps have to appear in pairs, except directly before EOT; mask logits accordingly
        for k in range(tokens.shape[0]):
            seq = [t for t in tokens[k, self.sample_begin :].tolist()]
            last_was_timestamp = len(seq) >= 1 and seq[-1] >= self.tokenizer.timestamp_begin
            penultimate_was_timestamp = len(seq) < 2 or seq[-2] >= self.tokenizer.timestamp_begin

            if last_was_timestamp:
                if penultimate_was_timestamp:  # has to be non-timestamp
                    logits[k, self.tokenizer.timestamp_begin :] = -np.inf
                else:  # cannot be normal text tokens
                    logits[k, : self.tokenizer.eot] = -np.inf

        if tokens.shape[1] == self.sample_begin:
            # suppress generating non-timestamp tokens at the beginning
            logits[:, : self.tokenizer.timestamp_begin] = -np.inf

            # apply the `max_initial_timestamp` option
            if self.max_initial_timestamp_index is not None:
                last_allowed = self.tokenizer.timestamp_begin + self.max_initial_timestamp_index
                logits[:, last_allowed + 1 :] = -np.inf

        # if sum of probability over timestamps is above any other token, sample timestamp
        logprobs = F.log_softmax(logits.float(), dim=-1)
        for k in range(tokens.shape[0]):
            timestamp_logprob = logprobs[k, self.tokenizer.timestamp_begin :].logsumexp(dim=-1)
            max_text_token_logprob = logprobs[k, : self.tokenizer.timestamp_begin].max()
            if timestamp_logprob > max_text_token_logprob:
                logits[k, : self.tokenizer.timestamp_begin] = -np.inf


```

 audio\_features: List[Tensor], audio features extracted from the audio signals

languages: List[str], languages of the text

tokens: List[Tuple[Tensor, int, int]], the raw tokens (text, index, rate) for each sample

audio\_features: List[Tuple[Tensor, int, int]], the audio features of each sample (number of samples, rate, etc.)

language: str, the language of the text

avg\_logprob: List[float], the average log probability of each sample

noise\_prob: List[float], the noise probability of each sample (normalized to [0, 1])

compression\_ratio: float, the compression ratio of the text (number of tokens / audio\_features)



```py
class DecodingTask:
    inference: Inference
    sequence_ranker: SequenceRanker
    decoder: TokenDecoder
    logit_filters: List[LogitFilter]

    def __init__(self, model: "Whisper", options: DecodingOptions):
        self.model = model

        language = options.language or "en"
        tokenizer = get_tokenizer(model.is_multilingual, language=language, task=options.task)
        self.tokenizer: Tokenizer = tokenizer
        self.options: DecodingOptions = self._verify_options(options)

        self.n_group: int = options.beam_size or options.best_of or 1
        self.n_ctx: int = model.dims.n_text_ctx
        self.sample_len: int = options.sample_len or model.dims.n_text_ctx // 2

        self.sot_sequence: Tuple[int] = tokenizer.sot_sequence
        if self.options.without_timestamps:
            self.sot_sequence = tokenizer.sot_sequence_including_notimestamps

        self.initial_tokens: Tuple[int] = self._get_initial_tokens()
        self.sample_begin: int = len(self.initial_tokens)
        self.sot_index: int = self.initial_tokens.index(tokenizer.sot)

        # inference: implements the forward pass through the decoder, including kv caching
        self.inference = PyTorchInference(model, len(self.initial_tokens))

        # sequence ranker: implements how to rank a group of sampled sequences
        self.sequence_ranker = MaximumLikelihoodRanker(options.length_penalty)

        # decoder: implements how to select the next tokens, given the autoregressive distribution
        if options.beam_size is not None:
            self.decoder = BeamSearchDecoder(
                options.beam_size, tokenizer.eot, self.inference, options.patience
            )
        else:
            self.decoder = GreedyDecoder(options.temperature, tokenizer.eot)

        # logit filters: applies various rules to suppress or penalize certain tokens
        self.logit_filters = []
        if self.options.suppress_blank:
            self.logit_filters.append(SuppressBlank(self.tokenizer, self.sample_begin))
        if self.options.suppress_tokens:
            self.logit_filters.append(SuppressTokens(self._get_suppress_tokens()))
        if not options.without_timestamps:
            precision = CHUNK_LENGTH / model.dims.n_audio_ctx  # usually 0.02 seconds
            max_initial_timestamp_index = None
            if options.max_initial_timestamp:
                max_initial_timestamp_index = round(self.options.max_initial_timestamp / precision)
            self.logit_filters.append(
                ApplyTimestampRules(tokenizer, self.sample_begin, max_initial_timestamp_index)
            )

    def _verify_options(self, options: DecodingOptions) -> DecodingOptions:
        if options.beam_size is not None and options.best_of is not None:
            raise ValueError("beam_size and best_of can't be given together")
        if options.temperature == 0:
            if options.best_of is not None:
                raise ValueError("best_of with greedy sampling (T=0) is not compatible")
        if options.patience is not None and options.beam_size is None:
            raise ValueError("patience requires beam_size to be given")
        if options.length_penalty is not None and not (0 <= options.length_penalty <= 1):
            raise ValueError("length_penalty (alpha) should be a value between 0 and 1")

        return options

    def _get_initial_tokens(self) -> Tuple[int]:
        tokens = list(self.sot_sequence)
        prefix = self.options.prefix
        prompt = self.options.prompt

        if prefix:
            prefix_tokens = (
                self.tokenizer.encode(" " + prefix.strip()) if isinstance(prefix, str) else prefix
            )
            if self.sample_len is not None:
                max_prefix_len = self.n_ctx // 2 - self.sample_len
                prefix_tokens = prefix_tokens[-max_prefix_len:]
            tokens = tokens + prefix_tokens

        if prompt:
            prompt_tokens = (
                self.tokenizer.encode(" " + prompt.strip()) if isinstance(prompt, str) else prompt
            )
            tokens = [self.tokenizer.sot_prev] + prompt_tokens[-(self.n_ctx // 2 - 1) :] + tokens

        return tuple(tokens)

    def _get_suppress_tokens(self) -> Tuple[int]:
        suppress_tokens = self.options.suppress_tokens

        if isinstance(suppress_tokens, str):
            suppress_tokens = [int(t) for t in suppress_tokens.split(",")]

        if -1 in suppress_tokens:
            suppress_tokens = [t for t in suppress_tokens if t >= 0]
            suppress_tokens.extend(self.tokenizer.non_speech_tokens)
        elif suppress_tokens is None or len(suppress_tokens) == 0:
            suppress_tokens = []  # interpret empty string as an empty list
        else:
            assert isinstance(suppress_tokens, list), "suppress_tokens must be a list"

        suppress_tokens.extend(
            [self.tokenizer.sot, self.tokenizer.sot_prev, self.tokenizer.sot_lm]
        )
        if self.tokenizer.no_speech is not None:
            # no-speech probability is collected separately
            suppress_tokens.append(self.tokenizer.no_speech)

        return tuple(sorted(set(suppress_tokens)))

    def _get_audio_features(self, mel: Tensor):
        if self.options.fp16:
            mel = mel.half()

        if mel.shape[-2:] == (self.model.dims.n_audio_ctx, self.model.dims.n_audio_state):
            # encoded audio features are given; skip audio encoding
            print("encoded audio features are given; skip audio encoding")
            audio_features = mel
        else:
            print(mel.shape)
            print("===============================")
            audio_features = self.model.encoder(mel)

        if audio_features.dtype != (torch.float16 if self.options.fp16 else torch.float32):
            return TypeError(f"audio_features has an incorrect dtype: {audio_features.dtype}")

        return audio_features

    def _detect_language(self, audio_features: Tensor, tokens: Tensor):
        languages = [self.options.language] * audio_features.shape[0]
        lang_probs = None

        if self.options.language is None or self.options.task == "lang_id":
            lang_tokens, lang_probs = self.model.detect_language(audio_features, self.tokenizer)
            languages = [max(probs, key=probs.get) for probs in lang_probs]
            if self.options.language is None:
                tokens[:, self.sot_index + 1] = lang_tokens  # write language tokens

        return languages, lang_probs

    def _main_loop(self, audio_features: Tensor, tokens: Tensor):
        assert audio_features.shape[0] == tokens.shape[0]
        n_batch = tokens.shape[0]
        sum_logprobs: Tensor = torch.zeros(n_batch, device=audio_features.device)
        no_speech_probs = [np.nan] * n_batch

        try:
            for i in range(self.sample_len):
                logits = self.inference.logits(tokens, audio_features)

                if i == 0 and self.tokenizer.no_speech is not None:  # save no_speech_probs
                    probs_at_sot = logits[:, self.sot_index].float().softmax(dim=-1)
                    no_speech_probs = probs_at_sot[:, self.tokenizer.no_speech].tolist()

                # now we need to consider the logits at the last token only
                logits = logits[:, -1]

                # apply the logit filters, e.g. for suppressing or applying penalty to
                for logit_filter in self.logit_filters:
                    logit_filter.apply(logits, tokens)

                # expand the tokens tensor with the selected next tokens
                tokens, completed = self.decoder.update(tokens, logits, sum_logprobs)

                if completed or tokens.shape[-1] > self.n_ctx:
                    break
        finally:
            self.inference.cleanup_caching()

        return tokens, sum_logprobs, no_speech_probs

    @torch.no_grad()
    def run(self, mel: Tensor) -> List[DecodingResult]:
        self.decoder.reset()
        tokenizer: Tokenizer = self.tokenizer
        n_audio: int = mel.shape[0]

        audio_features: Tensor = self._get_audio_features(mel)  # encoder forward pass
        tokens: Tensor = torch.tensor([self.initial_tokens]).repeat(n_audio, 1)

        # detect language if requested, overwriting the language token
        languages, language_probs = self._detect_language(audio_features, tokens)
        if self.options.task == "lang_id":
            return [
                DecodingResult(audio_features=features, language=language, language_probs=probs)
                for features, language, probs in zip(audio_features, languages, language_probs)
            ]

        # repeat the audio & text tensors by the group size, for beam search or best-of-n sampling
        audio_features = audio_features.repeat_interleave(self.n_group, dim=0)
        tokens = tokens.repeat_interleave(self.n_group, dim=0).to(audio_features.device)

        # call the main sampling loop
        tokens, sum_logprobs, no_speech_probs = self._main_loop(audio_features, tokens)

        # reshape the tensors to have (n_audio, n_group) as the first two dimensions
        audio_features = audio_features[:: self.n_group]
        no_speech_probs = no_speech_probs[:: self.n_group]
        assert audio_features.shape[0] == len(no_speech_probs) == n_audio

        tokens = tokens.reshape(n_audio, self.n_group, -1)
        sum_logprobs = sum_logprobs.reshape(n_audio, self.n_group)

        # get the final candidates for each group, and slice between the first sampled token and EOT
        tokens, sum_logprobs = self.decoder.finalize(tokens, sum_logprobs)
        tokens: List[List[Tensor]] = [
            [t[self.sample_begin : (t == tokenizer.eot).nonzero()[0, 0]] for t in s] for s in tokens
        ]

        # select the top-ranked sample in each group
        selected = self.sequence_ranker.rank(tokens, sum_logprobs)
        tokens: List[List[int]] = [t[i].tolist() for i, t in zip(selected, tokens)]
        texts: List[str] = [tokenizer.decode(t).strip() for t in tokens]

        sum_logprobs: List[float] = [lp[i] for i, lp in zip(selected, sum_logprobs)]
        avg_logprobs: List[float] = [lp / (len(t) + 1) for t, lp in zip(tokens, sum_logprobs)]

        fields = (texts, languages, tokens, audio_features, avg_logprobs, no_speech_probs)
        if len(set(map(len, fields))) != 1:
            raise RuntimeError(f"inconsistent result lengths: {list(map(len, fields))}")

        return [
            DecodingResult(
                audio_features=features,
                language=language,
                tokens=tokens,
                text=text,
                avg_logprob=avg_logprob,
                no_speech_prob=no_speech_prob,
                temperature=self.options.temperature,
                compression_ratio=compression_ratio(text),
            )
            for text, language, tokens, features, avg_logprob, no_speech_prob in zip(*fields)
        ]


```

这段代码定义了一个名为 `decode` 的函数，它接受一个名为 `model` 的 Whisper 模型实例、一个名为 `mel` 的 Tensor，以及一个名为 `options` 的 DecodingOptions 实例。这个函数返回一个 Union 类型变量，即可能是一个 DecodingResult，也可能是由多个 DecodingResult 组成的列表。

首先，函数检查传入的 Mel 是否是一个标量，如果是，则将 Mel 转换为张量，然后将 Mel 张量末尾添加一个维度，使其形状为 (80, 3000)。这样可以确保在输入 Mel 数据时，可以按秒计算时间。

接下来，函数使用一个名为 `DecodingTask` 的类，这个类实现了 `run` 方法。在 `run` 方法中，函数将输入的 Mel 张量传递给 Whisper 模型，并返回可能是一个 DecodingResult 实例的结果。

最后，函数根据输入的 Mel 张量是否有多个维度来决定返回哪种类型。如果 Mel 张量只有两个维度，那么函数将直接返回结果。如果 Mel 张量有多个维度，那么函数会将结果封装为一个可能包含多个 DecodingResult 实例的列表中。


```py
@torch.no_grad()
def decode(model: "Whisper", mel: Tensor, options: DecodingOptions = DecodingOptions()) -> Union[DecodingResult, List[DecodingResult]]:
    """
    Performs decoding of 30-second audio segment(s), provided as Mel spectrogram(s).

    Parameters
    ----------
    model: Whisper
        the Whisper model instance

    mel: torch.Tensor, shape = (80, 3000) or (*, 80, 3000)
        A tensor containing the Mel spectrogram(s)

    options: DecodingOptions
        A dataclass that contains all necessary options for decoding 30-second segments

    Returns
    -------
    result: Union[DecodingResult, List[DecodingResult]]
        The result(s) of decoding contained in `DecodingResult` dataclass instance(s)
    """
    single = mel.ndim == 2
    if single:
        mel = mel.unsqueeze(0)
    result = DecodingTask(model, options).run(mel)
    
    if single:
        result = result[0]

    return result

```