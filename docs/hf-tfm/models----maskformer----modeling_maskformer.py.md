# `.\models\maskformer\modeling_maskformer.py`

```
# coding=utf-8
# Copyright 2022 Meta Platforms, Inc. and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

### PyTorch MaskFormer模型的基础元类
""" 
import math
from dataclasses import dataclass  
from numbers import Number
from typing import Dict, List, Optional, Tuple

import numpy as np  
import torch  
from torch import Tensor, nn  

from ...activations import ACT2FN                 
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask  
from ...modeling_outputs import BaseModelOutputWithCrossAttentions  
from ...modeling_utils import PreTrainedModel  
from ...pytorch_utils import is_torch_greater_or_equal_than_2_1  
from ...utils import (
    ModelOutput,                                   # 基本模型输出类，用于封装模型输出项
    add_start_docstrings,                         # 用于添加模型的开始说明文档
    add_start_docstrings_to_model_forward,        # 用于添加模型输入参数的文档
    is_accelerate_available,                      # 检查加速器模块是否可用
    is_scipy_available,                           # 检查科学计算库是否可用
    logging,                                     # 日志记录模块
    replace_return_docstrings,                    # 替换返回文档说明的函数
    requires_backends,                           # 要求特定后端支持的装饰器
)    
from ...utils.backbone_utils import load_backbone  
from ..detr import DetrConfig  
from .configuration_maskformer import MaskFormerConfig  
from .configuration_maskformer_swin import MaskFormerSwinConfig  

if is_accelerate_available():                      # 检查加速器模块是否存在
    from accelerate import PartialState     
    from accelerate.utils import reduce  

if is_scipy_available():                           # 检查科学计算库存在且可用
    from scipy.optimize import linear_sum_assignment  

logger = logging.get_logger(__name__)               # 创建日志记录器

# "MaskFormerConfig"类实例，用于指定模型配置
_CONFIG_FOR_DOC = "MaskFormerConfig"
# "facebook/maskformer-swin-base-ade"模型的预训练模型地址
_CHECKPOINT_FOR_DOC = "facebook/maskformer-swin-base-ade"

# MaskFormer预训练模型列表
MASKFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = ["facebook/maskformer-swin-base-ade",]

@dataclass                
# 定义"DetrDecoderOutput"类，扩展了"BaseModelOutputWithCrossAttentions"类，用于处理"DETR"解码器的输出项
class DetrDecoderOutput(BaseModelOutputWithCrossAttentions):
    """
    "DetrDecoderOutput"类继承自"BaseModelOutputWithCrossAttentions"类，用于封装"DETREncoder"模块的输出。
    接收一个"CrossAttentions"对象作为属性，并在此基础上添加了一个可选的解码器中间层激活堆栈。
    用于单辅助解码器损失训练时提供额外的特征信息。
    """
    
    # 定义函数的参数列表，包括最后一个隐藏层的输出状态
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层的隐藏状态的序列输出。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            可选参数，当传递 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回，
            包含元组中的 `torch.FloatTensor`（一个用于嵌入层输出，每层输出一个）的形状为 `(batch_size, sequence_length, hidden_size)`。
            模型每一层的隐藏状态，以及初始嵌入层的输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            可选参数，当传递 `output_attentions=True` 或 `config.output_attentions=True` 时返回，
            包含元组中的 `torch.FloatTensor`（每一层一个）的形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
            经过注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            可选参数，当同时传递 `output_attentions=True` 和 `config.add_cross_attention=True` 或 `config.output_attentions=True` 时返回，
            包含元组中的 `torch.FloatTensor`（每一层一个）的形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
            解码器交叉注意力层的注意力权重，经过注意力 softmax 后，用于计算交叉注意力头中的加权平均值。
        intermediate_hidden_states (`torch.FloatTensor` of shape `(config.decoder_layers, batch_size, num_queries, hidden_size)`, *optional*, returned when `config.auxiliary_loss=True`):
            可选参数，当传递 `config.auxiliary_loss=True` 时返回，
            形状为 `(config.decoder_layers, batch_size, num_queries, hidden_size)` 的中间解码器激活状态。
            每个解码器层的中间激活状态，每个状态经过了层归一化。
    """
    
    # intermediate_hidden_states 变量定义为可选的 `torch.FloatTensor` 类型，表示中间隐藏状态，默认为 None
    intermediate_hidden_states: Optional[torch.FloatTensor] = None
# 定义一个数据类 `MaskFormerPixelLevelModuleOutput`，继承自 `ModelOutput`，表示 MaskFormer 的像素级模块的输出
@dataclass
class MaskFormerPixelLevelModuleOutput(ModelOutput):
    """
    MaskFormer's pixel level module output. It returns both the last and (optionally) the hidden states from the
    `encoder` and `decoder`. By default, the `encoder` is a MaskFormerSwin Transformer and the `decoder` is a Feature
    Pyramid Network (FPN).

    The `encoder_last_hidden_state` are referred on the paper as **images features**, while `decoder_last_hidden_state`
    as **pixel embeddings**

    Args:
        encoder_last_hidden_state (`torch.FloatTensor` of shape`(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the encoder.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the model at
            the output of each stage.
        decoder_last_hidden_state (`torch.FloatTensor` of shape`(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the decoder.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the model at
            the output of each stage.
    """

    # 定义属性 `encoder_last_hidden_state`，表示编码器最后一个隐藏状态的张量，形状为 `(batch_size, num_channels, height, width)`
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 定义属性 `decoder_last_hidden_state`，表示解码器最后一个隐藏状态的张量，形状为 `(batch_size, num_channels, height, width)`
    decoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 定义属性 `encoder_hidden_states`，表示编码器的隐藏状态的元组，每个元素是一个形状为 `(batch_size, num_channels, height, width)` 的张量
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义属性 `decoder_hidden_states`，表示解码器的隐藏状态的元组，每个元素是一个形状为 `(batch_size, num_channels, height, width)` 的张量
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class MaskFormerPixelDecoderOutput(ModelOutput):
    """
    MaskFormer's pixel decoder module output, practically a Feature Pyramid Network. It returns the last hidden state
    and (optionally) the hidden states.
    """
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            模型最后阶段的最后隐藏状态（最终特征图）。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, 当 `output_hidden_states=True` 时返回或当 `config.output_hidden_states=True` 时返回):
            包含多个元素的元组，每个元素是 `torch.FloatTensor`，形状为 `(batch_size, num_channels, height, width)`。
            模型在每一层输出的隐藏状态，还包括初始嵌入的输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, 当 `output_attentions=True` 时返回或当 `config.output_attentions=True` 时返回):
            包含多个元素的元组，每个元素是 `torch.FloatTensor`，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
            Detr 解码器中注意力权重经过 attention softmax 后的输出，用于计算自注意力头中的加权平均值。
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# 定义一个数据类，用于存储 [`MaskFormerModel`] 的输出。这个类返回计算 logits 所需的所有隐藏状态。

@dataclass
class MaskFormerModelOutput(ModelOutput):
    """
    Class for outputs of [`MaskFormerModel`]. This class returns all the needed hidden states to compute the logits.

    Args:
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the encoder model (backbone).
        pixel_decoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the pixel decoder model (FPN).
        transformer_decoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Last hidden states (final feature map) of the last stage of the transformer decoder model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the encoder
            model at the output of each stage.
        pixel_decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the pixel
            decoder model at the output of each stage.
        transformer_decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states (also called feature maps) of the
            transformer decoder at the output of each stage.
        hidden_states `tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` containing `encoder_hidden_states`, `pixel_decoder_hidden_states` and
            `decoder_hidden_states`
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights from Detr's decoder after the attention softmax, used to compute the
            weighted average in the self-attention heads.
    """
    # 定义可选的 torch.FloatTensor 类型变量，用于存储编码器的最后隐藏状态
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 定义可选的 torch.FloatTensor 类型变量，用于存储像素解码器的最后隐藏状态
    pixel_decoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 定义可选的 torch.FloatTensor 类型变量，用于存储变换器解码器的最后隐藏状态
    transformer_decoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 定义可选的 Tuple[torch.FloatTensor] 类型变量，用于存储编码器的隐藏状态序列
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义可选的 Tuple[torch.FloatTensor] 类型变量，用于存储像素解码器的隐藏状态序列
    pixel_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义可选的 Tuple[torch.FloatTensor] 类型变量，用于存储变换器解码器的隐藏状态序列
    transformer_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义可选的 Tuple[torch.FloatTensor] 类型变量，用于存储隐藏状态序列
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义可选的 Tuple[torch.FloatTensor] 类型变量，用于存储注意力分布序列
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# 数据类装饰器，用于定义实例分割输出的数据结构，继承自ModelOutput类
@dataclass
class MaskFormerForInstanceSegmentationOutput(ModelOutput):
    """
    Class for outputs of [`MaskFormerForInstanceSegmentation`].

    This output can be directly passed to [`~MaskFormerImageProcessor.post_process_semantic_segmentation`] or or
    [`~MaskFormerImageProcessor.post_process_instance_segmentation`] or
    [`~MaskFormerImageProcessor.post_process_panoptic_segmentation`] depending on the task. Please, see
    [`~MaskFormerImageProcessor] for details regarding usage.

    """

    # 损失值，可选的浮点张量
    loss: Optional[torch.FloatTensor] = None
    # 类别查询的逻辑张量
    class_queries_logits: torch.FloatTensor = None
    # 掩码查询的逻辑张量
    masks_queries_logits: torch.FloatTensor = None
    # 辅助逻辑张量
    auxiliary_logits: torch.FloatTensor = None
    # 编码器最后隐藏状态，可选的浮点张量
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 像素解码器最后隐藏状态，可选的浮点张量
    pixel_decoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 变换器解码器最后隐藏状态，可选的浮点张量
    transformer_decoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 编码器隐藏状态，可选的浮点张量元组
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 像素解码器隐藏状态，可选的浮点张量元组
    pixel_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 变换器解码器隐藏状态，可选的浮点张量元组
    transformer_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 隐藏状态，可选的浮点张量元组
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 注意力分数，可选的浮点张量元组
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# 重新实现自原始实现的函数
def upsample_like(pixel_values: Tensor, like: Tensor, mode: str = "bilinear") -> Tensor:
    """
    An utility function that upsamples `pixel_values` to match the dimension of `like`.

    Args:
        pixel_values (`torch.Tensor`):
            The tensor we wish to upsample.
        like (`torch.Tensor`):
            The tensor we wish to use as size target.
        mode (str, *optional*, defaults to `"bilinear"`):
            The interpolation mode.

    Returns:
        `torch.Tensor`: The upsampled tensor
    """
    # 获取`like`张量的高度和宽度维度
    _, _, height, width = like.shape
    # 使用双线性插值法对`pixel_values`进行上采样，使其大小与`like`相匹配
    upsampled = nn.functional.interpolate(pixel_values, size=(height, width), mode=mode, align_corners=False)
    return upsampled


# 计算DICE损失的函数
def dice_loss(inputs: Tensor, labels: Tensor, num_masks: int) -> Tensor:
    r"""
    Compute the DICE loss, similar to generalized IOU for masks as follows:

    $$ \mathcal{L}_{\text{dice}(x, y) = 1 - \frac{2 * x \cap y }{x \cup y + 1}} $$

    In practice, since `labels` is a binary mask, (only 0s and 1s), dice can be computed as follow

    $$ \mathcal{L}_{\text{dice}(x, y) = 1 - \frac{2 * x * y }{x + y + 1}} $$

    Args:
        inputs (`torch.Tensor`):
            A tensor representing a mask.
        labels (`torch.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).
        num_masks (`int`):
            The number of masks present in the current batch, used for normalization.

    Returns:
        `torch.Tensor`: The computed loss.
    """
    # 对输入进行sigmoid操作并展平为一维张量，得到预测概率
    probs = inputs.sigmoid().flatten(1)
    # 计算DICE损失的分子部分：2 * 预测概率 * 真实标签的交集
    numerator = 2 * (probs * labels).sum(-1)
    # 计算概率和标签在最后一个维度上的和，分别求和
    denominator = probs.sum(-1) + labels.sum(-1)
    # 计算损失值，使用给定的数值计算公式
    loss = 1 - (numerator + 1) / (denominator + 1)
    # 将所有损失值求和并除以遮罩数量，得到平均损失
    loss = loss.sum() / num_masks
    # 返回计算得到的平均损失值
    return loss
# 从原始实现重构而来的函数，计算逐对的 Sigmoid Focal Loss
def sigmoid_focal_loss(
    inputs: Tensor, labels: Tensor, num_masks: int, alpha: float = 0.25, gamma: float = 2
) -> Tensor:
    r"""
    Focal loss，最初在 [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) 中提出，最初用于 RetinaNet。该损失计算如下：

    $$ \mathcal{L}_{\text{focal loss}} = -(1 - p_t)^{\gamma}\log{(p_t)} $$

    其中 \\(CE(p_t) = -\log{(p_t)}}\\)，CE 是标准交叉熵损失。

    请参考论文中的方程式 (1,2,3) 以获得更好的理解。

    Args:
        inputs (`torch.Tensor`):
            任意形状的浮点张量。
        labels (`torch.Tensor`):
            与 inputs 相同形状的张量。存储每个元素的二元分类标签 (0 表示负类，1 表示正类)。
        num_masks (`int`):
            当前批次中存在的掩码数量，用于归一化。
        alpha (float, *可选*, 默认为 0.25):
            在范围 (0,1) 内的加权因子，用于平衡正负例。
        gamma (float, *可选*, 默认为 2.0):
            调整因子 \\(1 - p_t\\) 的指数，用于平衡简单与困难的例子。

    Returns:
        `torch.Tensor`: 计算得到的损失。
    """
    # 使用带 logits 的二元交叉熵损失，不进行归一化
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    # 对输入进行 sigmoid 操作得到概率
    probs = inputs.sigmoid()
    # 计算标准交叉熵损失
    cross_entropy_loss = criterion(inputs, labels)
    # 计算 p_t
    p_t = probs * labels + (1 - probs) * (1 - labels)
    # 计算 focal loss
    loss = cross_entropy_loss * ((1 - p_t) ** gamma)

    # 如果 alpha 大于等于 0，计算 alpha_t
    if alpha >= 0:
        alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
        loss = alpha_t * loss

    # 计算平均损失并进行归一化
    loss = loss.mean(1).sum() / num_masks
    return loss


# 从原始实现重构而来的函数，计算逐对的 Dice Loss
def pair_wise_dice_loss(inputs: Tensor, labels: Tensor) -> Tensor:
    """
    Dice Loss 的逐对版本，参见 `dice_loss` 以了解用法。

    Args:
        inputs (`torch.Tensor`):
            表示掩码的张量
        labels (`torch.Tensor`):
            与 inputs 相同形状的张量。存储每个元素的二元分类标签 (0 表示负类，1 表示正类)。

    Returns:
        `torch.Tensor`: 每对之间计算得到的损失。
    """
    # 对输入进行 sigmoid 操作并展平为一维
    inputs = inputs.sigmoid().flatten(1)
    # 计算分子，使用矩阵乘法
    numerator = 2 * torch.matmul(inputs, labels.T)
    # 使用广播获取 [num_queries, NUM_CLASSES] 矩阵
    # 计算分母
    denominator = inputs.sum(-1)[:, None] + labels.sum(-1)[None, :]
    # 计算 Dice Loss
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


# 从原始实现重构而来的函数，计算逐对的 Sigmoid Focal Loss
def pair_wise_sigmoid_focal_loss(inputs: Tensor, labels: Tensor, alpha: float = 0.25, gamma: float = 2.0) -> Tensor:
    r"""
    Sigmoid Focal Loss 的逐对版本，参见 `sigmoid_focal_loss` 以了解用法。
    ```
    # 如果alpha小于0，则引发值错误异常
    if alpha < 0:
        raise ValueError("alpha must be positive")

    # 获取输入张量的高度和宽度（假设输入是一个二维张量）
    height_and_width = inputs.shape[1]

    # 使用二元交叉熵损失函数，但是禁止自动平均（即不对每个样本的损失求平均）
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    # 计算输入张量的sigmoid函数值，即转换为概率
    prob = inputs.sigmoid()

    # 计算正样本的交叉熵损失
    cross_entropy_loss_pos = criterion(inputs, torch.ones_like(inputs))

    # 计算焦点损失的正样本部分，用于聚焦于困难的样本
    focal_pos = ((1 - prob) ** gamma) * cross_entropy_loss_pos
    focal_pos *= alpha

    # 计算负样本的交叉熵损失
    cross_entropy_loss_neg = criterion(inputs, torch.zeros_like(inputs))

    # 计算焦点损失的负样本部分，用于聚焦于容易的样本
    focal_neg = (prob**gamma) * cross_entropy_loss_neg
    focal_neg *= 1 - alpha

    # 计算最终的损失值，分别乘以标签的转置以加权正负样本
    loss = torch.matmul(focal_pos, labels.T) + torch.matmul(focal_neg, (1 - labels).T)

    # 返回归一化后的损失，即平均每个元素的损失
    return loss / height_and_width
# Copied from transformers.models.detr.modeling_detr.DetrAttention
class DetrAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.

    Here, we add position embeddings to the queries and keys (as explained in the DETR paper).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim  # 初始化注意力机制的嵌入维度
        self.num_heads = num_heads  # 初始化注意力头的数量
        self.dropout = dropout  # 初始化dropout率
        self.head_dim = embed_dim // num_heads  # 计算每个注意力头的维度
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {num_heads})."
            )
        self.scaling = self.head_dim**-0.5  # 缩放因子，用于注意力计算

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 用于投影键的线性层
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 用于投影值的线性层
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 用于投影查询的线性层
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 输出投影的线性层

    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        # 重塑张量形状，以便进行多头注意力操作

    def with_pos_embed(self, tensor: torch.Tensor, object_queries: Optional[Tensor], **kwargs):
        position_embeddings = kwargs.pop("position_embeddings", None)

        if kwargs:
            raise ValueError(f"Unexpected arguments {kwargs.keys()}")

        if position_embeddings is not None and object_queries is not None:
            raise ValueError(
                "Cannot specify both position_embeddings and object_queries. Please use just object_queries"
            )

        if position_embeddings is not None:
            logger.warning_once(
                "position_embeddings has been deprecated and will be removed in v4.34. Please use object_queries instead"
            )
            object_queries = position_embeddings

        return tensor if object_queries is None else tensor + object_queries
        # 添加位置嵌入到输入张量中的查询，支持使用对象查询

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        object_queries: Optional[torch.Tensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
        spatial_position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs,
    ):
        # 前向传播函数，实现注意力机制的计算过程
    # 初始化方法，用于初始化一个DetrDecoderLayer对象
    def __init__(self, config: DetrConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 设置嵌入维度等于配置文件中的d_model值
        self.embed_dim = config.d_model

        # 创建一个自注意力层对象，使用DetrAttention类实现
        self.self_attn = DetrAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        # 设置Dropout层的概率为配置文件中的dropout值
        self.dropout = config.dropout
        # 设置激活函数为配置文件中指定的激活函数
        self.activation_fn = ACT2FN[config.activation_function]
        # 设置激活函数后的Dropout概率为配置文件中的activation_dropout值
        self.activation_dropout = config.activation_dropout

        # 对自注意力层输出进行LayerNorm归一化处理
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        
        # 创建一个编码器注意力层对象，使用DetrAttention类实现
        self.encoder_attn = DetrAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        # 对编码器注意力层输出进行LayerNorm归一化处理
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        
        # 使用线性层进行特征变换，输入维度为embed_dim，输出维度为配置文件中的decoder_ffn_dim
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        # 对fc1层输出进行线性变换，输出维度为embed_dim
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        # 对最终输出进行LayerNorm归一化处理
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    # 前向传播方法，定义了如何处理输入数据的流程
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        object_queries: Optional[torch.Tensor] = None,
        query_position_embeddings: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
class DetrDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`DetrDecoderLayer`].

    The decoder updates the query embeddings through multiple self-attention and cross-attention layers.

    Some small tweaks for DETR:

    - object_queries and query_position_embeddings are added to the forward pass.
    - if self.config.auxiliary_loss is set to True, also returns a stack of activations from all decoding layers.

    Args:
        config: DetrConfig
    """

    def __init__(self, config: DetrConfig):
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop

        # Initialize layers as a list of DetrDecoderLayer modules based on config.decoder_layers
        self.layers = nn.ModuleList([DetrDecoderLayer(config) for _ in range(config.decoder_layers)])
        
        # Apply LayerNorm to the output of the last decoder layer
        self.layernorm = nn.LayerNorm(config.d_model)

        # Gradient checkpointing is disabled by default
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        object_queries=None,
        query_position_embeddings=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        # Forward pass through the decoder layers
        # Each layer updates the query embeddings using self-attention and cross-attention mechanisms
        # object_queries and query_position_embeddings are incorporated if provided
        # If auxiliary_loss is True, also returns hidden states from all decoding layers
        # The method returns a dictionary of output values
        pass  # Placeholder for actual implementation


# refactored from original implementation
class MaskFormerHungarianMatcher(nn.Module):
    """This class computes an assignment between the labels and the predictions of the network.

    For efficiency reasons, the labels don't include the no_object. Because of this, in general, there are more
    predictions than labels. In this case, we do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1.0, cost_mask: float = 1.0, cost_dice: float = 1.0):
        """Creates the matcher

        Params:
            cost_class (float, *optional*, defaults to 1.0):
                This is the relative weight of the classification error in the matching cost.
            cost_mask (float, *optional*,  defaults to 1.0):
                This is the relative weight of the focal loss of the binary mask in the matching cost.
            cost_dice (float, *optional*, defaults to 1.0):
                This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        if cost_class == 0 and cost_mask == 0 and cost_dice == 0:
            raise ValueError("All costs cant be 0")
        
        # Initialize the relative weights for classification, mask focal loss, and dice loss
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

    @torch.no_grad()
    def forward(self):
        pass  # Placeholder for actual implementation
    # 返回对象的字符串表示形式，用于调试和显示
    def __repr__(self):
        # 构建字符串的头部，表示对象的类和名称
        head = "Matcher " + self.__class__.__name__
        # 构建字符串的主体部分，包括成本类、掩码和Dice成本的信息
        body = [
            f"cost_class: {self.cost_class}",  # 显示成本类的数值
            f"cost_mask: {self.cost_mask}",    # 显示成本掩码的数值
            f"cost_dice: {self.cost_dice}",    # 显示Dice成本的数值
        ]
        _repr_indent = 4  # 设置缩进量
        # 将头部和主体部分结合起来，每一行前面加上指定的缩进量
        lines = [head] + [" " * _repr_indent + line for line in body]
        # 将所有行连接成一个多行字符串并返回
        return "\n".join(lines)
# 从原始实现中复制并调整
class MaskFormerLoss(nn.Module):
    def __init__(
        self,
        num_labels: int,
        matcher: MaskFormerHungarianMatcher,
        weight_dict: Dict[str, float],
        eos_coef: float,
    ):
        """
        MaskFormer Loss类。损失计算与DETR非常类似。过程分为两步：
        1) 计算真实标签掩码与模型输出之间的匈牙利分配
        2) 监督每对匹配的真实标签/预测（监督类别和掩码）

        Args:
            num_labels (`int`):
                类别数量。
            matcher (`MaskFormerHungarianMatcher`):
                计算预测和标签之间分配的Torch模块。
            weight_dict (`Dict[str, float]`):
                不同损失要应用的权重字典。
            eos_coef (`float`):
                应用于空类别的权重。
        """

        super().__init__()
        requires_backends(self, ["scipy"])
        self.num_labels = num_labels
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        # 创建一个权重张量，包含所有类别和一个额外的EOS类别
        empty_weight = torch.ones(self.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def _max_by_axis(self, the_list: List[List[int]]) -> List[int]:
        # 按轴找到列表中的最大值
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def _pad_images_to_max_in_batch(self, tensors: List[Tensor]) -> Tuple[Tensor, Tensor]:
        # 获取批次中的最大尺寸
        max_size = self._max_by_axis([list(tensor.shape) for tensor in tensors])
        batch_size = len(tensors)
        # 计算最终形状
        batch_shape = [batch_size] + max_size
        b, _, h, w = batch_shape
        # 获取元数据
        dtype = tensors[0].dtype
        device = tensors[0].device
        # 创建零填充的张量和填充掩码
        padded_tensors = torch.zeros(batch_shape, dtype=dtype, device=device)
        padding_masks = torch.ones((b, h, w), dtype=torch.bool, device=device)
        # 将张量填充到最大尺寸
        for tensor, padded_tensor, padding_mask in zip(tensors, padded_tensors, padding_masks):
            padded_tensor[: tensor.shape[0], : tensor.shape[1], : tensor.shape[2]].copy_(tensor)
            padding_mask[: tensor.shape[1], : tensor.shape[2]] = False

        return padded_tensors, padding_masks

    def loss_labels(
        self, class_queries_logits: Tensor, class_labels: List[Tensor], indices: Tuple[np.array]
    ) -> Dict[str, Tensor]:
        """Compute the losses related to the labels using cross entropy.

        Args:
            class_queries_logits (`torch.Tensor`):
                A tensor of shape `batch_size, num_queries, num_labels`
            class_labels (`List[torch.Tensor]`):
                List of class labels of shape `(labels)`.
            indices (`Tuple[np.array])`:
                The indices computed by the Hungarian matcher.

        Returns:
            `Dict[str, Tensor]`: A dict of `torch.Tensor` containing the following key:
            - **loss_cross_entropy** -- The loss computed using cross entropy on the predicted and ground truth labels.
        """

        pred_logits = class_queries_logits
        batch_size, num_queries, _ = pred_logits.shape
        # Define CrossEntropyLoss criterion with empty_weight
        criterion = nn.CrossEntropyLoss(weight=self.empty_weight)
        # Obtain indices for permutation based on the Hungarian matcher output
        idx = self._get_predictions_permutation_indices(indices)
        # Concatenate target classes for each query in the batch
        # Shape after concatenation: (batch_size, num_queries)
        target_classes_o = torch.cat([target[j] for target, (_, j) in zip(class_labels, indices)])
        # Initialize target_classes tensor with default values
        # Shape: (batch_size, num_queries)
        target_classes = torch.full(
            (batch_size, num_queries), fill_value=self.num_labels, dtype=torch.int64, device=pred_logits.device
        )
        # Update target_classes tensor using the permutation indices
        target_classes[idx] = target_classes_o
        # Transpose pred_logits from "batch_size x num_queries x num_labels" to "batch_size x num_labels x num_queries"
        pred_logits_transposed = pred_logits.transpose(1, 2)
        # Compute cross entropy loss between transposed pred_logits and target_classes
        loss_ce = criterion(pred_logits_transposed, target_classes)
        # Prepare losses dictionary with cross entropy loss
        losses = {"loss_cross_entropy": loss_ce}
        return losses
    ) -> Dict[str, Tensor]:
        """Compute the losses related to the masks using focal and dice loss.

        Args:
            masks_queries_logits (`torch.Tensor`):
                A tensor of shape `batch_size, num_queries, height, width`
            mask_labels (`torch.Tensor`):
                List of mask labels of shape `(labels, height, width)`.
            indices (`Tuple[np.array])`:
                The indices computed by the Hungarian matcher.
            num_masks (`int)`:
                The number of masks, used for normalization.

        Returns:
            `Dict[str, Tensor]`: A dict of `torch.Tensor` containing two keys:
            - **loss_mask** -- The loss computed using sigmoid focal loss on the predicted and ground truth masks.
            - **loss_dice** -- The loss computed using dice loss on the predicted and ground truth
              masks.
        """
        # Get permutation indices for predictions based on Hungarian matcher results
        src_idx = self._get_predictions_permutation_indices(indices)
        # Get permutation indices for targets based on Hungarian matcher results
        tgt_idx = self._get_targets_permutation_indices(indices)

        # Select predicted masks using the permutation indices
        pred_masks = masks_queries_logits[src_idx]

        # Pad and stack target masks to match the shape of predictions
        target_masks, _ = self._pad_images_to_max_in_batch(mask_labels)
        target_masks = target_masks[tgt_idx]

        # Upsample predicted masks to match the size of target masks
        pred_masks = nn.functional.interpolate(
            pred_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False
        )
        # Flatten the predictions and targets for loss computation
        pred_masks = pred_masks[:, 0].flatten(1)
        target_masks = target_masks.flatten(1)

        # Compute losses using sigmoid focal loss and dice loss
        losses = {
            "loss_mask": sigmoid_focal_loss(pred_masks, target_masks, num_masks),
            "loss_dice": dice_loss(pred_masks, target_masks, num_masks),
        }
        return losses

    def _get_predictions_permutation_indices(self, indices):
        # Concatenate batch indices for predictions
        batch_indices = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        # Concatenate prediction indices based on permutation results
        predictions_indices = torch.cat([src for (src, _) in indices])
        return batch_indices, predictions_indices

    def _get_targets_permutation_indices(self, indices):
        # Concatenate batch indices for targets
        batch_indices = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        # Concatenate target indices based on permutation results
        target_indices = torch.cat([tgt for (_, tgt) in indices])
        return batch_indices, target_indices

    def forward(
        self,
        masks_queries_logits: Tensor,
        class_queries_logits: Tensor,
        mask_labels: List[Tensor],
        class_labels: List[Tensor],
        auxiliary_predictions: Optional[Dict[str, Tensor]] = None,
        """
        This performs the loss computation.

        Args:
            masks_queries_logits (`torch.Tensor`):
                A tensor of shape `batch_size, num_queries, height, width`
                表示查询掩码的logits张量，形状为 `batch_size, num_queries, height, width`
            class_queries_logits (`torch.Tensor`):
                A tensor of shape `batch_size, num_queries, num_labels`
                表示查询类别的logits张量，形状为 `batch_size, num_queries, num_labels`
            mask_labels (`torch.Tensor`):
                List of mask labels of shape `(labels, height, width)`.
                掩码标签列表，形状为 `(labels, height, width)`
            class_labels (`List[torch.Tensor]`):
                List of class labels of shape `(labels)`.
                类别标签列表，形状为 `(labels)`
            auxiliary_predictions (`Dict[str, torch.Tensor]`, *optional*):
                if `use_auxiliary_loss` was set to `true` in [`MaskFormerConfig`], then it contains the logits from the
                inner layers of the Detr's Decoder.
                可选参数，如果在 `MaskFormerConfig` 中设置了 `use_auxiliary_loss` 为 `true`，则包含来自 Detr 解码器内部层的logits。

        Returns:
            `Dict[str, Tensor]`: A dict of `torch.Tensor` containing two keys:
            - **loss_cross_entropy** -- The loss computed using cross entropy on the predicted and ground truth labels.
              使用交叉熵计算预测标签和真实标签之间的损失。
            - **loss_mask** -- The loss computed using sigmoid focal loss on the predicted and ground truth masks.
              使用sigmoid focal loss计算预测掩码和真实掩码之间的损失。
            - **loss_dice** -- The loss computed using dice loss on the predicted and ground truth masks.
              使用dice loss计算预测掩码和真实掩码之间的损失。
            if `use_auxiliary_loss` was set to `true` in [`MaskFormerConfig`], the dictionary contains additional losses
            for each auxiliary predictions.
            如果在 [`MaskFormerConfig`] 中设置了 `use_auxiliary_loss` 为 `true`，则字典包含每个辅助预测的额外损失。
        """

        # retrieve the matching between the outputs of the last layer and the labels
        # 获取最后一层输出与标签之间的匹配
        indices = self.matcher(masks_queries_logits, class_queries_logits, mask_labels, class_labels)

        # compute the average number of target masks for normalization purposes
        # 计算平均目标掩码数量，用于归一化
        num_masks: Number = self.get_num_masks(class_labels, device=class_labels[0].device)

        # get all the losses
        # 获取所有的损失
        losses: Dict[str, Tensor] = {
            **self.loss_masks(masks_queries_logits, mask_labels, indices, num_masks),
            **self.loss_labels(class_queries_logits, class_labels, indices),
        }

        # in case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        # 如果存在辅助损失，则对每个中间层的输出重复此过程。
        if auxiliary_predictions is not None:
            for idx, aux_outputs in enumerate(auxiliary_predictions):
                masks_queries_logits = aux_outputs["masks_queries_logits"]
                class_queries_logits = aux_outputs["class_queries_logits"]
                loss_dict = self.forward(masks_queries_logits, class_queries_logits, mask_labels, class_labels)
                loss_dict = {f"{key}_{idx}": value for key, value in loss_dict.items()}
                losses.update(loss_dict)

        return losses
    # 定义一个方法，计算批次中目标掩码的平均数量，用于归一化目的。
    def get_num_masks(self, class_labels: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Computes the average number of target masks across the batch, for normalization purposes.
        计算批次中目标掩码的平均数量，用于归一化目的。
        """
        # 计算所有类别标签中的掩码总数
        num_masks = sum([len(classes) for classes in class_labels])
        
        # 将掩码总数转换为张量，并指定数据类型和设备
        num_masks = torch.as_tensor(num_masks, dtype=torch.float, device=device)
        
        # 默认单进程世界大小
        world_size = 1
        
        # 如果加速库可用
        if is_accelerate_available():
            # 如果共享状态非空
            if PartialState._shared_state != {}:
                # 使用共享状态中的减少功能处理掩码总数
                num_masks = reduce(num_masks)
                # 获取部分状态对象的进程数量
                world_size = PartialState().num_processes
        
        # 将掩码总数除以进程数量进行归一化，并确保至少为1
        num_masks = torch.clamp(num_masks / world_size, min=1)
        
        # 返回归一化后的掩码数量
        return num_masks
class MaskFormerFPNConvLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, kernel_size: int = 3, padding: int = 1):
        """
        A basic module that executes conv - norm - in sequence used in MaskFormer.

        Args:
            in_features (`int`):
                The number of input features (channels).
            out_features (`int`):
                The number of outputs features (channels).
        """
        super().__init__()
        # Define layers for convolution, group normalization, and ReLU activation
        self.layers = [
            nn.Conv2d(in_features, out_features, kernel_size=kernel_size, padding=padding, bias=False),
            nn.GroupNorm(32, out_features),
            nn.ReLU(inplace=True),
        ]
        # Add each layer to the module and name them with their index
        for i, layer in enumerate(self.layers):
            self.add_module(str(i), layer)

    def forward(self, input: Tensor) -> Tensor:
        # Apply each layer sequentially to the input tensor
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class MaskFormerFPNLayer(nn.Module):
    def __init__(self, in_features: int, lateral_features: int):
        """
        A Feature Pyramid Network Layer (FPN) layer. It creates a feature map by aggregating features from the previous
        and backbone layer. Due to the spatial mismatch, the tensor coming from the previous layer is upsampled.

        Args:
            in_features (`int`):
                The number of input features (channels).
            lateral_features (`int`):
                The number of lateral features (channels).
        """
        super().__init__()
        # Project features from the lateral connection to match in_features using 1x1 convolution and group normalization
        self.proj = nn.Sequential(
            nn.Conv2d(lateral_features, in_features, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(32, in_features),
        )
        # Create a convolutional block for further processing of features
        self.block = MaskFormerFPNConvLayer(in_features, in_features)

    def forward(self, down: Tensor, left: Tensor) -> Tensor:
        # Project features from the lateral connection
        left = self.proj(left)
        # Upsample the downsampled features to match the size of the lateral features
        down = nn.functional.interpolate(down, size=left.shape[-2:], mode="nearest")
        # Aggregate features by element-wise addition
        down += left
        # Process the aggregated features using the convolutional block
        down = self.block(down)
        return down


class MaskFormerFPNModel(nn.Module):
    # This class definition continues in the actual code and is incomplete here.
    pass
    # 初始化方法，定义特征金字塔网络的结构
    def __init__(self, in_features: int, lateral_widths: List[int], feature_size: int = 256):
        """
        Feature Pyramid Network, given an input tensor and a set of feature maps of different feature/spatial sizes,
        it creates a list of feature maps with the same feature size.

        Args:
            in_features (`int`):
                The number of input features (channels).
            lateral_widths (`List[int]`):
                A list with the feature (channel) sizes of each lateral connection.
            feature_size (int, *optional*, defaults to 256):
                The feature (channel) size of the resulting feature maps.
        """
        # 调用父类的初始化方法
        super().__init__()
        # 定义特征金字塔网络的起始卷积层
        self.stem = MaskFormerFPNConvLayer(in_features, feature_size)
        # 定义特征金字塔网络的中间层序列，每层是一个MaskFormerFPNLayer对象
        self.layers = nn.Sequential(
            *[MaskFormerFPNLayer(feature_size, lateral_width) for lateral_width in lateral_widths[::-1]]
        )

    # 前向传播方法，计算特征金字塔网络的输出特征图列表
    def forward(self, features: List[Tensor]) -> List[Tensor]:
        # 初始化一个空列表，用于存储特征金字塔网络的输出特征图
        fpn_features = []
        # 获取最后一个特征图
        last_feature = features[-1]
        # 获取除了最后一个特征图外的其他特征图列表
        other_features = features[:-1]
        # 将最后一个特征图送入起始卷积层stem计算
        output = self.stem(last_feature)
        # 逐层处理特征金字塔网络的每一层
        for layer, left in zip(self.layers, other_features[::-1]):
            # 使用当前层处理输出特征图和对应的左侧特征图，得到新的输出特征图
            output = layer(output, left)
            # 将处理后的特征图加入到特征金字塔网络输出列表中
            fpn_features.append(output)
        # 返回特征金字塔网络的所有输出特征图列表
        return fpn_features
# 定义了一个名为 MaskFormerPixelDecoder 的神经网络模块类
class MaskFormerPixelDecoder(nn.Module):
    # 初始化方法，设置模块的参数和属性
    def __init__(self, *args, feature_size: int = 256, mask_feature_size: int = 256, **kwargs):
        r"""
        Pixel Decoder Module proposed in [Per-Pixel Classification is Not All You Need for Semantic
        Segmentation](https://arxiv.org/abs/2107.06278). It first runs the backbone's features into a Feature Pyramid
        Network creating a list of feature maps. Then, it projects the last one to the correct `mask_size`.

        Args:
            feature_size (`int`, *optional*, defaults to 256):
                The feature size (channel dimension) of the FPN feature maps.
            mask_feature_size (`int`, *optional*, defaults to 256):
                The features (channels) of the target masks size \\(C_{\epsilon}\\) in the paper.
        """
        super().__init__()

        # 创建 MaskFormerFPNModel 实例，用于生成特征金字塔网络的特征图列表
        self.fpn = MaskFormerFPNModel(*args, feature_size=feature_size, **kwargs)
        # 使用卷积层将最后一个特征图投影到正确的 mask 尺寸
        self.mask_projection = nn.Conv2d(feature_size, mask_feature_size, kernel_size=3, padding=1)

    # 前向传播方法，处理输入数据并返回输出
    def forward(
        self, features: List[Tensor], output_hidden_states: bool = False, return_dict: bool = True
    ) -> MaskFormerPixelDecoderOutput:
        # 使用特征金字塔网络处理输入特征列表，生成特征金字塔的特征图列表
        fpn_features = self.fpn(features)
        # 获取最后一个特征图并进行投影
        last_feature_projected = self.mask_projection(fpn_features[-1])

        # 根据 return_dict 参数返回不同形式的输出
        if not return_dict:
            return (last_feature_projected, tuple(fpn_features)) if output_hidden_states else (last_feature_projected,)

        # 如果 return_dict 为 True，则返回 MaskFormerPixelDecoderOutput 对象
        return MaskFormerPixelDecoderOutput(
            last_hidden_state=last_feature_projected, hidden_states=tuple(fpn_features) if output_hidden_states else ()
        )


# 复制并改编自原始实现，与 DetrSinePositionEmbedding 实现几乎相同
class MaskFormerSinePositionEmbedding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    # 初始化方法，设置位置嵌入的参数和属性
    def __init__(
        self, num_pos_feats: int = 64, temperature: int = 10000, normalize: bool = False, scale: Optional[float] = None
    ):
        super().__init__()
        # 如果指定了 scale 参数但未设置 normalize 参数，则抛出异常
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        # 初始化位置特征数量、温度参数、标准化标志和缩放比例
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale
    # 实现 Transformer 模型中的位置编码生成函数
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # 如果没有给定掩码，创建一个全零的掩码张量，与输入张量的维度匹配
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        
        # 计算反掩码，将掩码取反，转换为输入张量的数据类型
        not_mask = (~mask).to(x.dtype)
        
        # 在不被掩码遮挡的区域上计算累积和，作为位置编码的一部分
        y_embed = not_mask.cumsum(1)  # 在第二个维度上进行累积和
        x_embed = not_mask.cumsum(2)  # 在第三个维度上进行累积和
        
        # 如果需要归一化位置编码
        if self.normalize:
            eps = 1e-6
            # 对 y 轴和 x 轴的位置编码进行归一化处理，并乘以缩放因子 self.scale
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # 生成维度张量，用于计算位置编码
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.int64, device=x.device).type_as(x)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        # 根据维度张量计算 x 和 y 的位置编码
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        # 使用正弦和余弦函数堆叠 x 和 y 的位置编码
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        # 将 x 和 y 的位置编码连接起来，并将维度顺序转换为 (batch, channels, height, width)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        
        # 返回位置编码张量
        return pos
class PredictionBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation: nn.Module) -> None:
        super().__init__()
        # 创建一个包含线性层和激活函数的层列表
        self.layers = [nn.Linear(in_dim, out_dim), activation]
        # 将每个层作为子模块添加到当前模块中，以便在 forward 方法中能够正确调用
        for i, layer in enumerate(self.layers):
            self.add_module(str(i), layer)

    def forward(self, input: Tensor) -> Tensor:
        hidden_state = input
        # 逐层应用网络层和激活函数
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class MaskformerMLPPredictionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3):
        """
        A classic Multi Layer Perceptron (MLP).

        Args:
            input_dim (`int`):
                输入维度。
            hidden_dim (`int`):
                隐藏层维度。
            output_dim (`int`):
                输出维度。
            num_layers (int, *optional*, defaults to 3):
                层数。
        """
        super().__init__()
        # 构建输入和输出维度的列表，用于每个预测块的创建
        in_dims = [input_dim] + [hidden_dim] * (num_layers - 1)
        out_dims = [hidden_dim] * (num_layers - 1) + [output_dim]

        self.layers = []
        # 根据给定维度创建预测块列表
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            # 对于除了最后一层外的每一层使用ReLU激活函数，最后一层使用恒等激活函数
            activation = nn.ReLU() if i < num_layers - 1 else nn.Identity()
            # 创建预测块对象
            layer = PredictionBlock(in_dim, out_dim, activation=activation)
            self.layers.append(layer)
            # 将预测块作为子模块添加到当前模块中，使用索引作为名称
            self.add_module(str(i), layer)

    def forward(self, input: Tensor) -> Tensor:
        hidden_state = input
        # 逐层应用预测块
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class MaskFormerPixelLevelModule(nn.Module):
    pass  # 空模块，暂无具体实现
    def __init__(self, config: MaskFormerConfig):
        """
        Pixel Level Module proposed in [Per-Pixel Classification is Not All You Need for Semantic
        Segmentation](https://arxiv.org/abs/2107.06278). It runs the input image through a backbone and a pixel
        decoder, generating an image feature map and pixel embeddings.

        Args:
            config ([`MaskFormerConfig`]):
                The configuration used to instantiate this model.
        """
        super().__init__()  # 调用父类的初始化方法

        # 检查配置中是否有`backbone_config`属性，并且其`model_type`为"swin"时
        if getattr(config, "backbone_config") is not None and config.backbone_config.model_type == "swin":
            # 为了向后兼容，创建一个新的`backbone_config`，并从字典形式转换而来
            backbone_config = config.backbone_config
            backbone_config = MaskFormerSwinConfig.from_dict(backbone_config.to_dict())
            # 设置新的`out_features`，这里设置为固定的阶段名称列表
            backbone_config.out_features = ["stage1", "stage2", "stage3", "stage4"]
            config.backbone_config = backbone_config

        # 加载指定配置的背骨网络
        self.encoder = load_backbone(config)

        # 获取背骨网络最后一层的特征通道数
        feature_channels = self.encoder.channels

        # 初始化像素级解码器，传入参数为最后一个特征层的通道数、FPN特征大小、掩码特征大小、以及其它特征层的宽度
        self.decoder = MaskFormerPixelDecoder(
            in_features=feature_channels[-1],
            feature_size=config.fpn_feature_size,
            mask_feature_size=config.mask_feature_size,
            lateral_widths=feature_channels[:-1],
        )

    def forward(
        self, pixel_values: Tensor, output_hidden_states: bool = False, return_dict: bool = True
    ) -> MaskFormerPixelLevelModuleOutput:
        # 将像素值传入编码器，获取特征映射
        features = self.encoder(pixel_values).feature_maps

        # 将特征映射传入解码器，获取解码器的输出
        decoder_output = self.decoder(features, output_hidden_states, return_dict=return_dict)

        # 如果`return_dict`为False，返回特定格式的输出元组
        if not return_dict:
            last_hidden_state = decoder_output[0]  # 解码器输出的最后隐藏状态
            outputs = (features[-1], last_hidden_state)  # 输出包括编码器的最后一个特征映射和解码器的最后隐藏状态
            if output_hidden_states:
                hidden_states = decoder_output[1]  # 解码器的所有隐藏状态
                outputs = outputs + (tuple(features),) + (hidden_states,)  # 输出扩展为包括所有特征映射和隐藏状态
            return outputs

        # 如果`return_dict`为True，构造并返回`MaskFormerPixelLevelModuleOutput`对象
        return MaskFormerPixelLevelModuleOutput(
            encoder_last_hidden_state=features[-1],  # 编码器的最后一个特征映射
            decoder_last_hidden_state=decoder_output.last_hidden_state,  # 解码器的最后隐藏状态
            encoder_hidden_states=tuple(features) if output_hidden_states else (),  # 所有编码器特征映射
            decoder_hidden_states=decoder_output.hidden_states if output_hidden_states else (),  # 所有解码器隐藏状态
        )
class MaskFormerTransformerModule(nn.Module):
    """
    The MaskFormer's transformer module.
    """

    def __init__(self, in_features: int, config: MaskFormerConfig):
        super().__init__()
        hidden_size = config.decoder_config.hidden_size
        should_project = in_features != hidden_size
        # 初始化位置编码器，用于对象查询的位置信息嵌入
        self.position_embedder = MaskFormerSinePositionEmbedding(num_pos_feats=hidden_size // 2, normalize=True)
        # 初始化查询的嵌入层，根据配置的查询数量和隐藏大小
        self.queries_embedder = nn.Embedding(config.decoder_config.num_queries, hidden_size)
        # 如果输入特征与隐藏大小不同，进行卷积投影
        self.input_projection = nn.Conv2d(in_features, hidden_size, kernel_size=1) if should_project else None
        # 初始化解码器
        self.decoder = DetrDecoder(config=config.decoder_config)

    def forward(
        self,
        image_features: Tensor,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: Optional[bool] = None,
    ) -> DetrDecoderOutput:
        if self.input_projection is not None:
            # 如果存在输入投影层，对图像特征进行投影
            image_features = self.input_projection(image_features)
        # 生成对象查询的位置嵌入
        object_queries = self.position_embedder(image_features)
        # 重复查询嵌入以匹配批次大小
        batch_size = image_features.shape[0]
        queries_embeddings = self.queries_embedder.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        # 初始化输入嵌入（用零填充），将会被模型修改
        inputs_embeds = torch.zeros_like(queries_embeddings, requires_grad=True)

        batch_size, num_channels, height, width = image_features.shape
        # 重新排列图像特征和对象查询的维度以便匹配解码器的输入格式
        image_features = image_features.view(batch_size, num_channels, height * width).permute(0, 2, 1)
        object_queries = object_queries.view(batch_size, num_channels, height * width).permute(0, 2, 1)

        # 调用解码器进行前向传播
        decoder_output: DetrDecoderOutput = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            encoder_hidden_states=image_features,
            encoder_attention_mask=None,
            object_queries=object_queries,
            query_position_embeddings=queries_embeddings,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 返回解码器的输出结果
        return decoder_output


注释：
    # Args 定义了此函数的输入参数
    Args:
        # `pixel_values` 是一个 FloatTensor，表示像素值，形状为 `(batch_size, num_channels, height, width)`
        # 像素值可以通过 `AutoImageProcessor` 获得。详见 `MaskFormerImageProcessor.__call__` 的说明。
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
        
        # `pixel_mask` 是一个 LongTensor，形状为 `(batch_size, height, width)`，可选参数
        # 用于避免在填充像素值上执行注意力操作。掩码的取值范围为 `[0, 1]`：
        #
        # - 1 表示真实像素（即 **未掩码**），
        # - 0 表示填充像素（即 **已掩码**）。
        #
        # [什么是注意力掩码？](../glossary#attention-mask)
        pixel_mask (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
        
        # `output_hidden_states` 是一个布尔值，可选参数
        # 是否返回所有层的隐藏状态。更多细节请参见返回的张量中的 `hidden_states`。
        output_hidden_states (`bool`, *optional*):
        
        # `output_attentions` 是一个布尔值，可选参数
        # 是否返回 Detr 解码器注意力层的注意力张量。
        output_attentions (`bool`, *optional*):
        
        # `return_dict` 是一个布尔值，可选参数
        # 是否返回 `~MaskFormerModelOutput` 而不是普通的元组。
        return_dict (`bool`, *optional*):
"""
Defines the MaskFormerModel class which extends MaskFormerPreTrainedModel.

@add_start_docstrings(
    "The bare MaskFormer Model outputting raw hidden-states without any specific head on top.",
    MASKFORMER_START_DOCSTRING,
)
"""
class MaskFormerModel(MaskFormerPreTrainedModel):
    """
    Initializes a MaskFormerModel instance.

    Args:
        config (MaskFormerConfig): Configuration object specifying model parameters.

    Inherits:
        MaskFormerPreTrainedModel: Base class for MaskFormerModel, pre-trained model.

    Attributes:
        pixel_level_module (MaskFormerPixelLevelModule): Pixel-level module for MaskFormer.
        transformer_module (MaskFormerTransformerModule): Transformer module for MaskFormer.
    """

    def __init__(self, config: MaskFormerConfig):
        """
        Constructor for MaskFormerModel.

        Args:
            config (MaskFormerConfig): Configuration object specifying model parameters.

        Calls super() to initialize from MaskFormerPreTrainedModel, initializes:
            - pixel_level_module (MaskFormerPixelLevelModule): Module for pixel-level operations.
            - transformer_module (MaskFormerTransformerModule): Transformer module for MaskFormer.

        Post-initialization handled by self.post_init().
        """
        super().__init__(config)
        self.pixel_level_module = MaskFormerPixelLevelModule(config)
        self.transformer_module = MaskFormerTransformerModule(
            in_features=self.pixel_level_module.encoder.channels[-1], config=config
        )

        self.post_init()
    # 定义一个方法 `forward`，用于模型的前向传播
    def forward(
        # 输入参数 `pixel_values`，类型为 Tensor，表示输入的像素值
        self,
        # 输入参数 `pixel_mask`，可选的 Tensor 类型，表示像素的掩码，用于指示哪些像素是有效的
        pixel_values: Tensor,
        # 输入参数 `output_hidden_states`，可选的布尔值，控制是否输出隐藏状态
        output_hidden_states: Optional[bool] = None,
        # 输入参数 `output_attentions`，可选的布尔值，控制是否输出注意力权重
        output_attentions: Optional[bool] = None,
        # 输入参数 `return_dict`，可选的布尔值，控制是否返回字典形式的输出
        return_dict: Optional[bool] = None,
class MaskFormerForInstanceSegmentation(MaskFormerPreTrainedModel):
    def __init__(self, config: MaskFormerConfig):
        super().__init__(config)
        # 初始化 MaskFormerModel 模型
        self.model = MaskFormerModel(config)
        # 从配置中获取隐藏层大小
        hidden_size = config.decoder_config.hidden_size
        # 创建一个线性层用于类别预测，输出维度为 num_labels + 1（增加一个“空”类别）
        self.class_predictor = nn.Linear(hidden_size, config.num_labels + 1)
        # 创建 MaskformerMLPPredictionHead 实例，用于掩码嵌入
        self.mask_embedder = MaskformerMLPPredictionHead(hidden_size, hidden_size, config.mask_feature_size)

        # 创建 MaskFormerHungarianMatcher 实例，用于匹配器
        self.matcher = MaskFormerHungarianMatcher(
            cost_class=1.0, cost_dice=config.dice_weight, cost_mask=config.mask_weight
        )

        # 设置损失权重字典，用于损失函数 MaskFormerLoss
        self.weight_dict: Dict[str, float] = {
            "loss_cross_entropy": config.cross_entropy_weight,
            "loss_mask": config.mask_weight,
            "loss_dice": config.dice_weight,
        }

        # 创建 MaskFormerLoss 损失函数实例
        self.criterion = MaskFormerLoss(
            config.num_labels,
            matcher=self.matcher,
            weight_dict=self.weight_dict,
            eos_coef=config.no_object_weight,
        )

        # 运行初始化后处理方法
        self.post_init()

    # 计算并返回损失字典
    def get_loss_dict(
        self,
        masks_queries_logits: Tensor,
        class_queries_logits: Tensor,
        mask_labels: Tensor,
        class_labels: Tensor,
        auxiliary_logits: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        loss_dict: Dict[str, Tensor] = self.criterion(
            masks_queries_logits, class_queries_logits, mask_labels, class_labels, auxiliary_logits
        )
        # 根据权重字典调整每个损失值
        for key, weight in self.weight_dict.items():
            for loss_key, loss in loss_dict.items():
                if key in loss_key:
                    loss *= weight

        return loss_dict

    # 计算并返回总损失值
    def get_loss(self, loss_dict: Dict[str, Tensor]) -> Tensor:
        return sum(loss_dict.values())

    # 前向传播函数，接受多个输入和输出参数，包括像素值、掩码和类别标签等
    @add_start_docstrings_to_model_forward(MASKFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MaskFormerForInstanceSegmentationOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Tensor,
        mask_labels: Optional[List[Tensor]] = None,
        class_labels: Optional[List[Tensor]] = None,
        pixel_mask: Optional[Tensor] = None,
        output_auxiliary_logits: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 省略了前向传播函数的其余部分，因为没有要注释的代码
        pass
```