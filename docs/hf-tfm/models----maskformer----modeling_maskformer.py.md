# `.\transformers\models\maskformer\modeling_maskformer.py`

```py
# 设置文件编码为 UTF-8
# 版权信息
# 2022 年版权归 Meta Platforms, Inc. 和 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本许可
# 除非符合许可，否则不得使用此文件
# 您可以在以下网址获取许可证复本 http://www.apache.org/licenses/LICENSE-2.0
# 除非法律要求或书面同意，否则分发的软件是基于“原样”分发的，不带任何保证或条件，不论是明示的还是默示的
# 有关具体语言的权限和限制，请参阅许可证
# PyTorch MaskFormer 模型

# 导入需要的库
import math
from dataclasses import dataclass
from numbers import Number
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

# 导入 Hugging Face 的相关模块和工具函数
from ... import AutoBackbone
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_scipy_available,
    logging,
    replace_return_docstrings,
    requires_backends,
)
from ..detr import DetrConfig
from .configuration_maskformer import MaskFormerConfig
from .configuration_maskformer_swin import MaskFormerSwinConfig

# 如果 scipy 可用，则导入 linear_sum_assignment 函数
if is_scipy_available():
    from scipy.optimize import linear_sum_assignment

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 文档中引用的配置和检查点
_CONFIG_FOR_DOC = "MaskFormerConfig"
_CHECKPOINT_FOR_DOC = "facebook/maskformer-swin-base-ade"

# 预训练 MaskFormer 模型的存档列表
MASKFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/maskformer-swin-base-ade",
    # 查看所有 MaskFormer 模型 https://huggingface.co/models?filter=maskformer
]

# 数据类注释，用于 DETR 解码器输出
@dataclass
class DetrDecoderOutput(BaseModelOutputWithCrossAttentions):
    """
    DETR 解码器输出的基类。该类在 BaseModelOutputWithCrossAttentions 中添加了一个属性，
    即可选的中间解码器激活堆栈，即每个解码器层的输出，每个输出都经过一个 layernorm。
    在使用辅助解码损失训练模型时非常有用。
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
        intermediate_hidden_states (`torch.FloatTensor` of shape `(config.decoder_layers, batch_size, num_queries, hidden_size)`, *optional*, returned when `config.auxiliary_loss=True`):
            Intermediate decoder activations, i.e. the output of each decoder layer, each of them gone through a
            layernorm.
    """

    intermediate_hidden_states: Optional[torch.FloatTensor] = None


注释：定义了一个可选参数 `intermediate_hidden_states`，它是一个 `torch.FloatTensor` 对象，形状为 `(config.decoder_layers, batch_size, num_queries, hidden_size)`。这个参数是解码器每个层的中间激活状态（通过 layernorm 处理后的结果）。
# 定义 MaskFormer 的像素级模块输出类，继承自 ModelOutput 类
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

    # 定义属性 encoder_last_hidden_state，默认为 None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 定义属性 decoder_last_hidden_state，默认为 None
    decoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 定义属性 encoder_hidden_states，默认为 None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义属性 decoder_hidden_states，默认为 None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


# 定义 MaskFormer 的像素解码器输出类，继承自 ModelOutput 类
@dataclass
class MaskFormerPixelDecoderOutput(ModelOutput):
    """
    MaskFormer's pixel decoder module output, practically a Feature Pyramid Network. It returns the last hidden state
    and (optionally) the hidden states.
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights from Detr's decoder after the attention softmax, used to compute the
            weighted average in the self-attention heads.
    """

    # 定义变量并初始化为None，用于存储最后一个隐藏状态、隐藏状态和注意力权重
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# 定义 MaskFormerModelOutput 类，用于存储 MaskFormerModel 的输出结果
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
    # 定义可选的变量 encoder_last_hidden_state，用于存储编码器的最后隐藏状态
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 定义可选的变量 pixel_decoder_last_hidden_state，用于存储像素解码器的最后隐藏状态
    pixel_decoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 定义可选的变量 transformer_decoder_last_hidden_state，用于存储变换器解码器的最后隐藏状态
    transformer_decoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 定义可选的变量 encoder_hidden_states，用于存储编码器的隐藏状态元组
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义可选的变量 pixel_decoder_hidden_states，用于存储像素解码器的隐藏状态元组
    pixel_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义可选的变量 transformer_decoder_hidden_states，用于存储变换器解码器的隐藏状态元组
    transformer_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义可选的变量 hidden_states，用于存储隐藏状态元组
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义可选的变量 attentions，用于存储注意力权重元组
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# 定义一个数据类，用于存储实例分割的输出结果
@dataclass
class MaskFormerForInstanceSegmentationOutput(ModelOutput):
    """
    Class for outputs of [`MaskFormerForInstanceSegmentation`].

    This output can be directly passed to [`~MaskFormerImageProcessor.post_process_semantic_segmentation`] or or
    [`~MaskFormerImageProcessor.post_process_instance_segmentation`] or
    [`~MaskFormerImageProcessor.post_process_panoptic_segmentation`] depending on the task. Please, see
    [`~MaskFormerImageProcessor] for details regarding usage.

    """

    # 定义各种输出结果的变量
    loss: Optional[torch.FloatTensor] = None
    class_queries_logits: torch.FloatTensor = None
    masks_queries_logits: torch.FloatTensor = None
    auxiliary_logits: torch.FloatTensor = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    pixel_decoder_last_hidden_state: Optional[torch.FloatTensor] = None
    transformer_decoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    pixel_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    transformer_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# 定义一个函数，将像素值上采样到与指定张量相同的维度
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
    # 获取目标张量的高度和宽度
    _, _, height, width = like.shape
    # 使用双线性插值对像素值进行上采样
    upsampled = nn.functional.interpolate(pixel_values, size=(height, width), mode=mode, align_corners=False)
    return upsampled


# 从原始实现重构的函数，计算 DICE 损失
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
    # 计算概率
    probs = inputs.sigmoid().flatten(1)
    # 计算分子
    numerator = 2 * (probs * labels).sum(-1)
    # 计算每个样本的分母，即概率之和与标签之和
    denominator = probs.sum(-1) + labels.sum(-1)
    # 计算损失值，1 减去（分子加 1）除以（分母加 1）
    loss = 1 - (numerator + 1) / (denominator + 1)
    # 对损失值进行求和并除以掩码数量，得到平均损失值
    loss = loss.sum() / num_masks
    # 返回平均损失值
    return loss
# 从原始实现重构而来的 Sigmoid Focal Loss 函数
def sigmoid_focal_loss(
    inputs: Tensor, labels: Tensor, num_masks: int, alpha: float = 0.25, gamma: float = 2
) -> Tensor:
    r"""
    在 [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) 中提出的 Focal Loss 最初用于 RetinaNet。该损失计算如下：

    $$ \mathcal{L}_{\text{focal loss} = -(1 - p_t)^{\gamma}\log{(p_t)} $$

    其中 \\(CE(p_t) = -\log{(p_t)}}\\)，CE 是标准的交叉熵损失

    请参考论文中的方程式 (1,2,3) 以获得更好的理解。

    Args:
        inputs (`torch.Tensor`):
            任意形状的浮点张量。
        labels (`torch.Tensor`):
            与 inputs 具有相同形状的张量。存储每个元素的二元分类标签 (0 表示负类，1 表示正类)。
        num_masks (`int`):
            当前批次中存在的掩码数量，用于归一化。
        alpha (float, *可选*, 默认为 0.25):
            用于平衡正负样本的权重因子，取值范围为 (0,1)。
        gamma (float, *可选*, 默认为 2.0):
            用于平衡易样本与难样本的调制因子的指数 \\(1 - p_t\\)。

    Returns:
        `torch.Tensor`: 计算得到的损失。
    """
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    probs = inputs.sigmoid()
    cross_entropy_loss = criterion(inputs, labels)
    p_t = probs * labels + (1 - probs) * (1 - labels)
    loss = cross_entropy_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
        loss = alpha_t * loss

    loss = loss.mean(1).sum() / num_masks
    return loss


# 从原始实现重构而来的 Pair Wise Dice Loss 函数
def pair_wise_dice_loss(inputs: Tensor, labels: Tensor) -> Tensor:
    """
    Dice Loss 的一对一版本，参见 `dice_loss` 以了解用法。

    Args:
        inputs (`torch.Tensor`):
            表示掩码的张量
        labels (`torch.Tensor`):
            与 inputs 具有相同形状的张量。存储每个元素的二元分类标签 (0 表示负类，1 表示正类)。

    Returns:
        `torch.Tensor`: 每对之间计算得到的损失。
    """
    inputs = inputs.sigmoid().flatten(1)
    numerator = 2 * torch.matmul(inputs, labels.T)
    # 使用广播获取一个 [num_queries, NUM_CLASSES] 矩阵
    denominator = inputs.sum(-1)[:, None] + labels.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


# 从原始实现重构而来的 Pair Wise Sigmoid Focal Loss 函数
def pair_wise_sigmoid_focal_loss(inputs: Tensor, labels: Tensor, alpha: float = 0.25, gamma: float = 2.0) -> Tensor:
    r"""
    Focal Loss 的一对一版本，参见 `sigmoid_focal_loss` 以了解用法。
    Args:
        inputs (`torch.Tensor`):
            代表一个掩码的张量。
        labels (`torch.Tensor`):
            与输入张量具有相同形状的张量。存储每个输入元素的二元分类标签（0表示负类，1表示正类）。
        alpha (float, *optional*, 默认为0.25):
            在范围(0,1)内的加权因子，用于平衡正类与负类示例。
        gamma (float, *optional*, 默认为2.0):
            调节因子 \\(1 - p_t\\) 的指数，用于平衡简单与困难示例。

    Returns:
        `torch.Tensor`: 每对之间计算的损失。
    """
    如果 alpha 小于0，则引发 ValueError 异常
    height_and_width = inputs.shape[1]

    使用"none"作为减少方式创建二元交叉熵损失函数
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    计算输入的 sigmoid 函数值
    prob = inputs.sigmoid()
    计算正类的交叉熵损失
    cross_entropy_loss_pos = criterion(inputs, torch.ones_like(inputs))
    计算焦点损失中的正类部分
    focal_pos = ((1 - prob) ** gamma) * cross_entropy_loss_pos
    focal_pos *= alpha

    计算负类的交叉熵损失
    cross_entropy_loss_neg = criterion(inputs, torch.zeros_like(inputs))

    计算焦点损失中的负类部分
    focal_neg = (prob**gamma) * cross_entropy_loss_neg
    focal_neg *= 1 - alpha

    计算损失，包括正类和负类部分
    loss = torch.matmul(focal_pos, labels.T) + torch.matmul(focal_neg, (1 - labels).T)

    返回损失除以高度和宽度
    return loss / height_and_width
# 从transformers.models.detr.modeling_detr.DetrAttention中复制过来的类
class DetrAttention(nn.Module):
    """
    多头注意力机制，源自 'Attention Is All You Need' 论文。

    在这里，我们为查询和键添加位置嵌入（如DETR论文中所解释的）。
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim必须能被num_heads整除（得到 `embed_dim`: {self.embed_dim} 和 `num_heads`: {num_heads}）."
            )
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def with_pos_embed(self, tensor: torch.Tensor, object_queries: Optional[Tensor], **kwargs):
        position_embeddings = kwargs.pop("position_embeddings", None)

        if kwargs:
            raise ValueError(f"意外的参数 {kwargs.keys()}")

        if position_embeddings is not None and object_queries is not None:
            raise ValueError(
                "不能同时指定position_embeddings和object_queries。请只使用object_queries"
            )

        if position_embeddings is not None:
            logger.warning_once(
                "position_embeddings已被弃用，并将在v4.34中删除。请改用object_queries"
            )
            object_queries = position_embeddings

        return tensor if object_queries is None else tensor + object_queries

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        object_queries: Optional[torch.Tensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
        spatial_position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs,
# 从transformers.models.detr.modeling_detr.DetrDecoderLayer中复制过来的类
class DetrDecoderLayer(nn.Module):
    # 初始化函数，接受一个DetrConfig类型的参数config
    def __init__(self, config: DetrConfig):
        # 调用父类的初始化函数
        super().__init__()
        # 设置嵌入维度为config中的d_model值
        self.embed_dim = config.d_model

        # 创建self-attention层，设置参数包括嵌入维度、注意力头数和dropout率
        self.self_attn = DetrAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        # 设置dropout率
        self.dropout = config.dropout
        # 设置激活函数
        self.activation_fn = ACT2FN[config.activation_function]
        # 设置激活函数的dropout率
        self.activation_dropout = config.activation_dropout

        # 创建self-attention层的LayerNorm层
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 创建encoder-attention层，设置参数包括嵌入维度、注意力头数和dropout率
        self.encoder_attn = DetrAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        # 创建encoder-attention层的LayerNorm层
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 创建全连接层fc1，输入维度为嵌入维度，输出维度为config中的decoder_ffn_dim
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        # 创建全连接层fc2，输入维度为config中的decoder_ffn_dim，输出维度为嵌入维度
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        # 创建最终的LayerNorm层
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    # 前向传播函数，接受多个参数，包括隐藏状态、注意力掩码、目标查询、查询位置嵌入、编码器隐藏状态、编码器注意力掩码、是否输出注意力权重等
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

        self.layers = nn.ModuleList([DetrDecoderLayer(config) for _ in range(config.decoder_layers)])
        # in DETR, the decoder uses layernorm after the last decoder layer output
        self.layernorm = nn.LayerNorm(config.d_model)

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
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

    @torch.no_grad()
    # 定义对象的字符串表示形式，返回对象的描述信息
    def __repr__(self):
        # 定义头部信息，包含类名
        head = "Matcher " + self.__class__.__name__
        # 定义主体信息，包含各项成本参数
        body = [
            f"cost_class: {self.cost_class}",
            f"cost_mask: {self.cost_mask}",
            f"cost_dice: {self.cost_dice}",
        ]
        # 定义缩进量
        _repr_indent = 4
        # 将头部和主体信息组合成完整的描述信息
        lines = [head] + [" " * _repr_indent + line for line in body]
        # 将描述信息按行组合成字符串并返回
        return "\n".join(lines)
# 从原始实现中复制并调整的类
class MaskFormerLoss(nn.Module):
    def __init__(
        self,
        num_labels: int,
        matcher: MaskFormerHungarianMatcher,
        weight_dict: Dict[str, float],
        eos_coef: float,
    ):
        """
        MaskFormer Loss。损失计算与DETR非常相似。过程分为两步：1) 计算真实掩码与模型输出之间的匈牙利分配 2) 监督每对匹配的真实/预测（监督类别和掩码）

        Args:
            num_labels (`int`):
                类别数量。
            matcher (`MaskFormerHungarianMatcher`):
                一个计算预测和标签之间分配的torch模块。
            weight_dict (`Dict[str, float]`):
                一个包含不同损失应用的权重的字典。
            eos_coef (`float`):
                应用于空类别的权重。
        """

        super().__init__()
        requires_backends(self, ["scipy"])
        self.num_labels = num_labels
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        empty_weight = torch.ones(self.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def _max_by_axis(self, the_list: List[List[int]]) -> List[int]:
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def _pad_images_to_max_in_batch(self, tensors: List[Tensor]) -> Tuple[Tensor, Tensor]:
        # 获取批次中的最大尺寸
        max_size = self._max_by_axis([list(tensor.shape) for tensor in tensors])
        batch_size = len(tensors)
        # 计算最终尺寸
        batch_shape = [batch_size] + max_size
        b, _, h, w = batch_shape
        # 获取元数据
        dtype = tensors[0].dtype
        device = tensors[0].device
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
        criterion = nn.CrossEntropyLoss(weight=self.empty_weight)
        idx = self._get_predictions_permutation_indices(indices)
        # shape = (batch_size, num_queries)
        target_classes_o = torch.cat([target[j] for target, (_, j) in zip(class_labels, indices)])
        # shape = (batch_size, num_queries)
        target_classes = torch.full(
            (batch_size, num_queries), fill_value=self.num_labels, dtype=torch.int64, device=pred_logits.device
        )
        target_classes[idx] = target_classes_o
        # target_classes is a (batch_size, num_labels, num_queries), we need to permute pred_logits "b q c -> b c q"
        pred_logits_transposed = pred_logits.transpose(1, 2)
        loss_ce = criterion(pred_logits_transposed, target_classes)
        losses = {"loss_cross_entropy": loss_ce}
        return losses

    def loss_masks(
        self, masks_queries_logits: Tensor, mask_labels: List[Tensor], indices: Tuple[np.array], num_masks: int
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
            - **loss_dice** -- The loss computed using dice loss on the predicted on the predicted and ground truth
              masks.
        """
        # Get the permutation indices for predictions based on the Hungarian matcher
        src_idx = self._get_predictions_permutation_indices(indices)
        tgt_idx = self._get_targets_permutation_indices(indices)
        # Extract predicted masks based on permutation indices
        pred_masks = masks_queries_logits[src_idx]
        # Pad and stack target masks to match the number of labels
        target_masks, _ = self._pad_images_to_max_in_batch(mask_labels)
        target_masks = target_masks[tgt_idx]
        # Upsample predictions to target size using bilinear interpolation
        pred_masks = nn.functional.interpolate(
            pred_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False
        )
        pred_masks = pred_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        # Compute losses using sigmoid focal loss and dice loss
        losses = {
            "loss_mask": sigmoid_focal_loss(pred_masks, target_masks, num_masks),
            "loss_dice": dice_loss(pred_masks, target_masks, num_masks),
        }
        return losses

    def _get_predictions_permutation_indices(self, indices):
        # Permute predictions following the indices
        batch_indices = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        predictions_indices = torch.cat([src for (src, _) in indices])
        return batch_indices, predictions_indices

    def _get_targets_permutation_indices(self, indices):
        # Permute labels following the indices
        batch_indices = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        target_indices = torch.cat([tgt for (_, tgt) in indices])
        return batch_indices, target_indices

    def forward(
        self,
        masks_queries_logits: Tensor,
        class_queries_logits: Tensor,
        mask_labels: List[Tensor],
        class_labels: List[Tensor],
        auxiliary_predictions: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """
        This performs the loss computation.

        Args:
            masks_queries_logits (`torch.Tensor`):
                A tensor of shape `batch_size, num_queries, height, width`
            class_queries_logits (`torch.Tensor`):
                A tensor of shape `batch_size, num_queries, num_labels`
            mask_labels (`torch.Tensor`):
                List of mask labels of shape `(labels, height, width)`.
            class_labels (`List[torch.Tensor]`):
                List of class labels of shape `(labels)`.
            auxiliary_predictions (`Dict[str, torch.Tensor]`, *optional*):
                if `use_auxiliary_loss` was set to `true` in [`MaskFormerConfig`], then it contains the logits from the
                inner layers of the Detr's Decoder.

        Returns:
            `Dict[str, Tensor]`: A dict of `torch.Tensor` containing two keys:
            - **loss_cross_entropy** -- The loss computed using cross entropy on the predicted and ground truth labels.
            - **loss_mask** -- The loss computed using sigmoid focal loss on the predicted and ground truth masks.
            - **loss_dice** -- The loss computed using dice loss on the predicted on the predicted and ground truth
              masks.
            if `use_auxiliary_loss` was set to `true` in [`MaskFormerConfig`], the dictionary contains addional losses
            for each auxiliary predictions.
        """

        # retrieve the matching between the outputs of the last layer and the labels
        indices = self.matcher(masks_queries_logits, class_queries_logits, mask_labels, class_labels)
        # compute the average number of target masks for normalization purposes
        num_masks: Number = self.get_num_masks(class_labels, device=class_labels[0].device)
        # get all the losses
        losses: Dict[str, Tensor] = {
            **self.loss_masks(masks_queries_logits, mask_labels, indices, num_masks),
            **self.loss_labels(class_queries_logits, class_labels, indices),
        }
        # in case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if auxiliary_predictions is not None:
            for idx, aux_outputs in enumerate(auxiliary_predictions):
                masks_queries_logits = aux_outputs["masks_queries_logits"]
                class_queries_logits = aux_outputs["class_queries_logits"]
                loss_dict = self.forward(masks_queries_logits, class_queries_logits, mask_labels, class_labels)
                loss_dict = {f"{key}_{idx}": value for key, value in loss_dict.items()}
                losses.update(loss_dict)

        return losses
    # 定义一个方法，用于计算批次中目标掩模的平均数量，以便进行归一化处理
    def get_num_masks(self, class_labels: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Computes the average number of target masks across the batch, for normalization purposes.
        """
        # 计算批次中目标掩模的总数量
        num_masks = sum([len(classes) for classes in class_labels])
        # 将目标掩模数量转换为 PyTorch 张量，并指定数据类型和设备
        num_masks_pt = torch.as_tensor(num_masks, dtype=torch.float, device=device)
        # 返回目标掩模数量的 PyTorch 张量
        return num_masks_pt
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
        # 初始化函数，定义一个执行卷积 - 归一化 - 激活的基本模块
        super().__init__()
        self.layers = [
            nn.Conv2d(in_features, out_features, kernel_size=kernel_size, padding=padding, bias=False),
            nn.GroupNorm(32, out_features),
            nn.ReLU(inplace=True),
        ]
        for i, layer in enumerate(self.layers):
            # 为了向后兼容，当类继承自 nn.Sequential 时，层的名称是其在序列中的索引
            # 在 nn.Module 子类中，它们派生自分配给实例属性的名称，例如 self.my_layer_name = Layer()
            # 不能给实例属性整数名称，因此需要显式注册
            self.add_module(str(i), layer)

    def forward(self, input: Tensor) -> Tensor:
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
        # 初始化函数，定义一个特征金字塔网络层，通过聚合来自前一个和主干层的特征来创建特征图
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(lateral_features, in_features, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(32, in_features),
        )

        self.block = MaskFormerFPNConvLayer(in_features, in_features)

    def forward(self, down: Tensor, left: Tensor) -> Tensor:
        left = self.proj(left)
        down = nn.functional.interpolate(down, size=left.shape[-2:], mode="nearest")
        down += left
        down = self.block(down)
        return down


class MaskFormerFPNModel(nn.Module):
    def __init__(self, in_features: int, lateral_widths: List[int], feature_size: int = 256):
        """
        初始化函数，创建特征金字塔网络，根据输入张量和不同特征/空间大小的特征图集合，生成具有相同特征大小的特征图列表。

        Args:
            in_features (`int`):
                输入特征的数量（通道数）。
            lateral_widths (`List[int]`):
                包含每个侧连接的特征（通道）大小的列表。
            feature_size (int, *optional*, defaults to 256):
                结果特征图的特征（通道）数。
        """
        super().__init__()
        # 创建 MaskFormerFPNConvLayer 实例作为特征金字塔网络的起始层
        self.stem = MaskFormerFPNConvLayer(in_features, feature_size)
        # 创建包含多个 MaskFormerFPNLayer 实例的序列作为特征金字塔网络的中间层
        self.layers = nn.Sequential(
            *[MaskFormerFPNLayer(feature_size, lateral_width) for lateral_width in lateral_widths[::-1]]
        )

    def forward(self, features: List[Tensor]) -> List[Tensor]:
        # 初始化空列表用于存储特征金字塔网络的特征图
        fpn_features = []
        # 获取最后一个特征图
        last_feature = features[-1]
        # 获取除最后一个特征图外的其他特征图
        other_features = features[:-1]
        # 将最后一个特征图传递给起始层，并获取输出
        output = self.stem(last_feature)
        # 遍历中间层，并将每层的输出添加到特征金字塔网络的特征图列表中
        for layer, left in zip(self.layers, other_features[::-1]):
            output = layer(output, left)
            fpn_features.append(output)
        # 返回特征金字塔网络的特征图列表
        return fpn_features
# 定义一个名为 MaskFormerPixelDecoder 的类，继承自 nn.Module
class MaskFormerPixelDecoder(nn.Module):
    # 初始化函数，接受一些参数，包括 feature_size 和 mask_feature_size
    def __init__(self, *args, feature_size: int = 256, mask_feature_size: int = 256, **kwargs):
        # 调用父类的初始化函数
        super().__init__()

        # 创建 MaskFormerFPNModel 实例，并传入参数 feature_size
        self.fpn = MaskFormerFPNModel(*args, feature_size=feature_size, **kwargs)
        # 创建一个卷积层，将 feature_size 映射到 mask_feature_size
        self.mask_projection = nn.Conv2d(feature_size, mask_feature_size, kernel_size=3, padding=1)

    # 前向传播函数，接受 features、output_hidden_states 和 return_dict 参数，返回 MaskFormerPixelDecoderOutput
    def forward(
        self, features: List[Tensor], output_hidden_states: bool = False, return_dict: bool = True
    ) -> MaskFormerPixelDecoderOutput:
        # 将 features 输入到 FPN 模型中，得到 fpn_features
        fpn_features = self.fpn(features)
        # 获取最后一个特征图，并通过 mask_projection 进行投影
        last_feature_projected = self.mask_projection(fpn_features[-1])

        # 如果 return_dict 为 False，则返回元组，否则返回 MaskFormerPixelDecoderOutput 对象
        if not return_dict:
            return (last_feature_projected, tuple(fpn_features)) if output_hidden_states else (last_feature_projected,)

        return MaskFormerPixelDecoderOutput(
            last_hidden_state=last_feature_projected, hidden_states=tuple(fpn_features) if output_hidden_states else ()
        )


# 定义一个名为 MaskFormerSinePositionEmbedding 的类，继承自 nn.Module
class MaskFormerSinePositionEmbedding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    # 初始化函数，接受 num_pos_feats、temperature、normalize 和 scale 参数
    def __init__(
        self, num_pos_feats: int = 64, temperature: int = 10000, normalize: bool = False, scale: Optional[float] = None
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 如果 scale 不为 None 且 normalize 为 False，则抛出 ValueError
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        # 初始化一些参数
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale
    # 定义一个前向传播函数，接受输入张量 x 和可选的掩码张量 mask，返回处理后的张量
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # 如果没有提供掩码张量，则创建一个与输入张量相同大小的全零张量作为掩码
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        # 计算非掩码部分，即取反操作
        not_mask = (~mask).to(x.dtype)
        # 在非掩码部分上进行累积操作，得到 y 方向和 x 方向的位置编码
        y_embed = not_mask.cumsum(1)
        x_embed = not_mask.cumsum(2)
        # 如果需要进行归一化处理
        if self.normalize:
            eps = 1e-6
            # 对 y 和 x 方向的位置编码进行归一化处理
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # 生成位置编码的维度序列
        dim_t = torch.arange(self.num_pos_feats, dtype=x.dtype, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        # 计算 x 和 y 方向的位置编码
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # 对 x 和 y 方向的位置编码进行正弦和余弦变换，并拼接在一起
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # 将 x 和 y 方向的位置编码拼接在一起，并进行维度置换
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        # 返回位置编码张量
        return pos
class PredictionBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation: nn.Module) -> None:
        super().__init__()
        self.layers = [nn.Linear(in_dim, out_dim), activation]
        # 维护子模块的索引，使其看起来像是 Sequential 块的一部分
        for i, layer in enumerate(self.layers):
            self.add_module(str(i), layer)

    def forward(self, input: Tensor) -> Tensor:
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class MaskformerMLPPredictionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3):
        """
        一个经典的多层感知器（MLP）。

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
        in_dims = [input_dim] + [hidden_dim] * (num_layers - 1)
        out_dims = [hidden_dim] * (num_layers - 1) + [output_dim]

        self.layers = []
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            activation = nn.ReLU() if i < num_layers - 1 else nn.Identity()
            layer = PredictionBlock(in_dim, out_dim, activation=activation)
            self.layers.append(layer)
            # 为了向后兼容，当类继承自 nn.Sequential 时
            # 在 nn.Sequential 子类中，给定的层的名称是其在序列中的索引。
            # 在 nn.Module 子类中，它们派生自分配给它们的实例属性，例如
            # self.my_layer_name = Layer()
            # 我们不能给实例属性整数名称，即 self.0 是不允许的，因此需要显式注册
            self.add_module(str(i), layer)

    def forward(self, input: Tensor) -> Tensor:
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class MaskFormerPixelLevelModule(nn.Module):
    def __init__(self, config: MaskFormerConfig):
        """
        Pixel Level Module proposed in [Per-Pixel Classification is Not All You Need for Semantic
        Segmentation](https://arxiv.org/abs/2107.06278). It runs the input image through a backbone and a pixel
        decoder, generating an image feature map and pixel embeddings.

        Args:
            config ([`MaskFormerConfig`]):
                The configuration used to instantiate this model.
        """
        # 初始化函数，接受一个配置对象作为参数
        super().__init__()

        # TODD: add method to load pretrained weights of backbone
        # 获取配置对象中的骨干网络配置
        backbone_config = config.backbone_config
        # 如果骨干网络类型为"swin"
        if backbone_config.model_type == "swin":
            # 为了向后兼容，将配置对象转换为MaskFormerSwinConfig类型
            backbone_config = MaskFormerSwinConfig.from_dict(backbone_config.to_dict())
            # 设置输出特征为["stage1", "stage2", "stage3", "stage4"]
            backbone_config.out_features = ["stage1", "stage2", "stage3", "stage4"]
        # 根据配置对象创建自动骨干网络
        self.encoder = AutoBackbone.from_config(backbone_config)

        # 获取骨干网络的通道数
        feature_channels = self.encoder.channels
        # 创建MaskFormerPixelDecoder对象
        self.decoder = MaskFormerPixelDecoder(
            in_features=feature_channels[-1],
            feature_size=config.fpn_feature_size,
            mask_feature_size=config.mask_feature_size,
            lateral_widths=feature_channels[:-1],
        )

    def forward(
        self, pixel_values: Tensor, output_hidden_states: bool = False, return_dict: bool = True
    ) -> MaskFormerPixelLevelModuleOutput:
        # 将输入像素值通过骨干网络得到特征图
        features = self.encoder(pixel_values).feature_maps
        # 将特征图通过解码器得到输出
        decoder_output = self.decoder(features, output_hidden_states, return_dict=return_dict)

        # 如果不返回字典
        if not return_dict:
            # 获取解码器输出的最后一个隐藏状态
            last_hidden_state = decoder_output[0]
            # 组装输出
            outputs = (features[-1], last_hidden_state)
            # 如果需要输出隐藏状态
            if output_hidden_states:
                hidden_states = decoder_output[1]
                outputs = outputs + (tuple(features),) + (hidden_states,)
            return outputs

        # 返回MaskFormerPixelLevelModuleOutput对象
        return MaskFormerPixelLevelModuleOutput(
            # 最后一个特征实际上是最后一层的输出
            encoder_last_hidden_state=features[-1],
            decoder_last_hidden_state=decoder_output.last_hidden_state,
            encoder_hidden_states=tuple(features) if output_hidden_states else (),
            decoder_hidden_states=decoder_output.hidden_states if output_hidden_states else (),
        )
class MaskFormerTransformerModule(nn.Module):
    """
    The MaskFormer's transformer module.
    """

    def __init__(self, in_features: int, config: MaskFormerConfig):
        # 初始化 MaskFormerTransformerModule 类
        super().__init__()
        hidden_size = config.decoder_config.hidden_size
        should_project = in_features != hidden_size
        # 初始化位置编码器
        self.position_embedder = MaskFormerSinePositionEmbedding(num_pos_feats=hidden_size // 2, normalize=True)
        # 初始化查询嵌入器
        self.queries_embedder = nn.Embedding(config.decoder_config.num_queries, hidden_size)
        # 如果需要进行投影，则初始化输入投影层
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
        # 如果存在输入投影层，则对图像特征进行投影
        if self.input_projection is not None:
            image_features = self.input_projection(image_features)
        # 获取对象查询
        object_queries = self.position_embedder(image_features)
        # 重复查询嵌入 "q c -> b q c"
        batch_size = image_features.shape[0]
        queries_embeddings = self.queries_embedder.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        inputs_embeds = torch.zeros_like(queries_embeddings, requires_grad=True)

        batch_size, num_channels, height, width = image_features.shape
        # 重新排列图像特征和对象查询 "b c h w -> b (h w) c"
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
        return decoder_output


MASKFORMER_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`MaskFormerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

MASKFORMER_INPUTS_DOCSTRING = r"""
    Args:
        # 输入参数 pixel_values 是一个 torch.FloatTensor，表示像素值，形状为 (batch_size, num_channels, height, width)
        # 像素值可以使用 AutoImageProcessor 获取。详细信息请参考 MaskFormerImageProcessor.__call__
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
        # 输入参数 pixel_mask 是一个 torch.LongTensor，表示遮罩，形状为 (batch_size, height, width)，可选参数
        # 用于避免在填充像素值上执行注意力操作。遮罩值选在 [0, 1] 范围内：
        # - 1 表示真实像素（即**未遮罩**），
        # - 0 表示填充像素（即**遮罩**）。
        # [什么是注意力遮罩？](../glossary#attention-mask)
        pixel_mask (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
        # 输出参数 output_hidden_states 是一个布尔值，可选参数
        # 是否返回所有层的隐藏状态。有关更多细节，请查看返回的张量中的 hidden_states
        output_hidden_states (`bool`, *optional*):
        # 输出参数 output_attentions 是一个布尔值，可选参数
        # 是否返回 Detr 解码器注意力层的注意力张量
        output_attentions (`bool`, *optional*):
        # 输出参数 return_dict 是一个布尔值，可选参数
        # 是否返回一个 MaskFormerModelOutput 而不是一个普通元组
        return_dict (`bool`, *optional*):
# 定义 MaskFormerPreTrainedModel 类，继承自 PreTrainedModel 类
class MaskFormerPreTrainedModel(PreTrainedModel):
    # 设置配置类为 MaskFormerConfig
    config_class = MaskFormerConfig
    # 设置基础模型前缀为 "model"
    base_model_prefix = "model"
    # 设置主输入名称为 "pixel_values"
    main_input_name = "pixel_values"

    # 初始化权重函数，接受一个 nn.Module 类型的参数
    def _init_weights(self, module: nn.Module):
        # 获取配置中的初始化参数
        xavier_std = self.config.init_xavier_std
        std = self.config.init_std
        # 如果模块是 MaskFormerTransformerModule 类型
        if isinstance(module, MaskFormerTransformerModule):
            # 如果模块的输入投影不为空
            if module.input_projection is not None:
                # 使用 xavier 均匀分布初始化输入投影的权重
                nn.init.xavier_uniform_(module.input_projection.weight, gain=xavier_std)
                # 将输入投影的偏置初始化为 0
                nn.init.constant_(module.input_projection.bias, 0)
        # 如果模块是 MaskFormerFPNModel 类型
        elif isinstance(module, MaskFormerFPNModel):
            # 使用 xavier 均匀分布初始化 FPN 模型的权重
            nn.init.xavier_uniform_(module.stem.get_submodule("0").weight, gain=xavier_std)
        # 如果模块是 MaskFormerFPNLayer 类型
        elif isinstance(module, MaskFormerFPNLayer):
            # 使用 xavier 均匀分布初始化 FPN 层的权重
            nn.init.xavier_uniform_(module.proj[0].weight, gain=xavier_std)
        # 如果模块是 MaskFormerFPNConvLayer 类型
        elif isinstance(module, MaskFormerFPNConvLayer):
            # 使用 xavier 均匀分布初始化 FPN 卷积层的权重
            nn.init.xavier_uniform_(module.get_submodule("0").weight, gain=xavier_std)
        # 如果模块是 MaskformerMLPPredictionHead 类型
        elif isinstance(module, MaskformerMLPPredictionHead):
            # 对 MLP 头部进行初始化
            # 在原始实现中找不到正确的初始化器，使用 xavier 初始化
            for submodule in module.modules():
                if isinstance(submodule, nn.Linear):
                    nn.init.xavier_uniform_(submodule.weight, gain=xavier_std)
                    nn.init.constant_(submodule.bias, 0)
        # 如果模块是 nn.LayerNorm 类型
        elif isinstance(module, nn.LayerNorm):
            # 将 LayerNorm 的偏置初始化为 0
            module.bias.data.zero_()
            # 将 LayerNorm 的权重初始化为 1
            module.weight.data.fill_(1.0)
        # 从 DETR 复制过来的初始化方式
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            # 稍微不同于 TF 版本，使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是 nn.Embedding 类型
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果有填充索引，则将对应权重初始化为 0
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

# 添加文档字符串描述 MaskFormerModel 类
@add_start_docstrings(
    "The bare MaskFormer Model outputting raw hidden-states without any specific head on top.",
    MASKFORMER_START_DOCSTRING,
)
# 定义 MaskFormerModel 类，继承自 MaskFormerPreTrainedModel 类
class MaskFormerModel(MaskFormerPreTrainedModel):
    # 初始化函数，接受一个 MaskFormerConfig 类型的参数
    def __init__(self, config: MaskFormerConfig):
        # 调用父类的初始化函数
        super().__init__(config)
        # 创建像素级模块
        self.pixel_level_module = MaskFormerPixelLevelModule(config)
        # 创建变换器模块
        self.transformer_module = MaskFormerTransformerModule(
            in_features=self.pixel_level_module.encoder.channels[-1], config=config
        )

        # 调用后续初始化函数
        self.post_init()

    # 添加文档字符串描述模型前向传播
    @add_start_docstrings_to_model_forward(MASKFORMER_INPUTS_DOCSTRING)
    # 替换返回文档字符串描述
    @replace_return_docstrings(output_type=MaskFormerModelOutput, config_class=_CONFIG_FOR_DOC)
    # 定义一个前向传播函数，接受像素数值和像素掩码作为输入
    # 可选参数output_hidden_states用于指定是否输出隐藏状态
    # 可选参数output_attentions用于指定是否输出注意力权重
    # 可选参数return_dict用于指定是否返回字典形式的结果
class MaskFormerForInstanceSegmentation(MaskFormerPreTrainedModel):
    # 定义一个用于实例分割的 MaskFormer 模型类，继承自 MaskFormerPreTrainedModel
    def __init__(self, config: MaskFormerConfig):
        # 初始化方法，接受一个 MaskFormerConfig 类型的参数 config
        super().__init__(config)
        # 调用父类的初始化方法
        self.model = MaskFormerModel(config)
        # 创建一个 MaskFormerModel 模型对象
        hidden_size = config.decoder_config.hidden_size
        # 获取隐藏层大小
        # + 1 because we add the "null" class
        self.class_predictor = nn.Linear(hidden_size, config.num_labels + 1)
        # 创建一个线性层用于类别预测
        self.mask_embedder = MaskformerMLPPredictionHead(hidden_size, hidden_size, config.mask_feature_size)
        # 创建一个 MaskformerMLPPredictionHead 对象用于 mask 嵌入

        self.matcher = MaskFormerHungarianMatcher(
            cost_class=1.0, cost_dice=config.dice_weight, cost_mask=config.mask_weight
        )
        # 创建一个 MaskFormerHungarianMatcher 对象用于匹配

        self.weight_dict: Dict[str, float] = {
            "loss_cross_entropy": config.cross_entropy_weight,
            "loss_mask": config.mask_weight,
            "loss_dice": config.dice_weight,
        }
        # 定义一个权重字典，包含交叉熵损失、mask 损失和 dice 损失的权重

        self.criterion = MaskFormerLoss(
            config.num_labels,
            matcher=self.matcher,
            weight_dict=self.weight_dict,
            eos_coef=config.no_object_weight,
        )
        # 创建一个 MaskFormerLoss 对象用于计算损失

        self.post_init()
        # 调用后续初始化方法

    def get_loss_dict(
        self,
        masks_queries_logits: Tensor,
        class_queries_logits: Tensor,
        mask_labels: Tensor,
        class_labels: Tensor,
        auxiliary_logits: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        # 定义一个方法用于获取损失字典
        loss_dict: Dict[str, Tensor] = self.criterion(
            masks_queries_logits, class_queries_logits, mask_labels, class_labels, auxiliary_logits
        )
        # 计算损失字典

        # weight each loss by `self.weight_dict[<LOSS_NAME>]` including auxiliary losses
        for key, weight in self.weight_dict.items():
            for loss_key, loss in loss_dict.items():
                if key in loss_key:
                    loss *= weight
        # 根据权重字典中的权重对每个损失进行加权处理

        return loss_dict
        # 返回损失字典

    def get_loss(self, loss_dict: Dict[str, Tensor]) -> Tensor:
        # 定义一个方法用于获取总损失
        return sum(loss_dict.values())
        # 返回所有损失的总和
    # 获取模型输出中的像素解码器的最后隐藏状态，即像素嵌入
    pixel_embeddings = outputs.pixel_decoder_last_hidden_state
    # 获取辅助预测（每个解码器层一个）
    auxiliary_logits: List[str, Tensor] = []
    # 如果配置中使用辅助损失，则返回多个预测结果
    if self.config.use_auxiliary_loss:
        # 将所有解码器隐藏状态堆叠起来
        stacked_transformer_decoder_outputs = torch.stack(outputs.transformer_decoder_hidden_states)
        # 使用类别预测器获取类别
        classes = self.class_predictor(stacked_transformer_decoder_outputs)
        # 获取类别查询的logits
        class_queries_logits = classes[-1]
        # 获取掩码嵌入
        mask_embeddings = self.mask_embedder(stacked_transformer_decoder_outputs)

        # 计算二进制掩码
        num_embeddings, batch_size, num_queries, num_channels = mask_embeddings.shape
        _, _, height, width = pixel_embeddings.shape
        binaries_masks = torch.zeros(
            (num_embeddings, batch_size, num_queries, height, width), device=mask_embeddings.device
        )
        for c in range(num_channels):
            binaries_masks += mask_embeddings[..., c][..., None, None] * pixel_embeddings[None, :, None, c]

        masks_queries_logits = binaries_masks[-1]
        # 遍历除最后一个外的所有元素
        for aux_binary_masks, aux_classes in zip(binaries_masks[:-1], classes[:-1]):
            auxiliary_logits.append(
                {"masks_queries_logits": aux_binary_masks, "class_queries_logits": aux_classes}
            )

    else:
        # 如果不使用辅助损失，则直接获取类别预测和掩码嵌入
        transformer_decoder_hidden_states = outputs.transformer_decoder_last_hidden_state
        classes = self.class_predictor(transformer_decoder_hidden_states)
        class_queries_logits = classes
        mask_embeddings = self.mask_embedder(transformer_decoder_hidden_states)

        # 计算掩码查询的logits
        batch_size, num_queries, num_channels = mask_embeddings.shape
        _, _, height, width = pixel_embeddings.shape
        masks_queries_logits = torch.zeros((batch_size, num_queries, height, width), device=mask_embeddings.device)
        for c in range(num_channels):
            masks_queries_logits += mask_embeddings[..., c][..., None, None] * pixel_embeddings[:, None, c]

    # 返回类别查询logits、掩码查询logits和辅助logits
    return class_queries_logits, masks_queries_logits, auxiliary_logits

@add_start_docstrings_to_model_forward(MASKFORMER_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=MaskFormerForInstanceSegmentationOutput, config_class=_CONFIG_FOR_DOC)
    # 定义一个前向传播函数，接受像素数值、掩码标签、类别标签、像素掩码、输出辅助日志、输出隐藏状态、输出注意力、返回字典等参数
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
```