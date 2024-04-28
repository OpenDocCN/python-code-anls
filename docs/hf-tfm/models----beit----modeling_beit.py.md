# `.\transformers\models\beit\modeling_beit.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 Microsoft Research 和 The HuggingFace Inc. 团队所有
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可;
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件根据“原样”分发，
# 无论是明示的还是默示的，软件不提供任何形式的担保或条件。
# 有关特定语言的权限，请参阅许可证。
""" PyTorch BEiT 模型。"""

# 导入所需的库
import collections.abc
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入所需的模块和类
from ...activations import ACT2FN
from ...modeling_outputs import (
    BackboneOutput,
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
    MaskedLMOutput,
    SemanticSegmenterOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ...utils.backbone_utils import BackboneMixin
from .configuration_beit import BeitConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的通用字符串
_CONFIG_FOR_DOC = "BeitConfig"

# 基本文档字符串
_CHECKPOINT_FOR_DOC = "microsoft/beit-base-patch16-224-pt22k"
_EXPECTED_OUTPUT_SHAPE = [1, 197, 768]

# 图像分类文档字符串
_IMAGE_CLASS_CHECKPOINT = "microsoft/beit-base-patch16-224"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

# BEiT 预训练模型存档列表
BEIT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/beit-base-patch16-224",
    # 查看所有 BEiT 模型 https://huggingface.co/models?filter=beit
]

@dataclass
class BeitModelOutputWithPooling(BaseModelOutputWithPooling):
    """
    [`BeitModel`] 的输出类。
    """
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            # 模型最后一层的隐藏状态的序列输出。形状为(batch_size, sequence_length, hidden_size)。
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            # 如果*config.use_mean_pooling*设置为True，则为裁剪的令牌（不包括*[CLS]*标记）的最后一层隐藏状态的平均值。如果设置为False，则返回*[CLS]*标记的最终隐藏状态。
            # 形状为(batch_size, hidden_size)。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            # 当`output_hidden_states=True`传递或者`config.output_hidden_states=True`时返回的可选参数。
            # 包含了模型在每一层输出的隐藏状态的元组。形状为(batch_size, sequence_length, hidden_size)。
            # 包含了模型在每一层输出的隐藏状态以及初始嵌入层输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            # 当`output_attentions=True`传递或者`config.output_attentions=True`时返回的可选参数。
            # 包含了每一层的注意力权重的元组。形状为(batch_size, num_heads, sequence_length, sequence_length)。
            # 用于计算自注意力头中的加权平均值的注意力权重。
    """
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    # 如果 drop_prob 为 0 或者不处于训练状态，则直接返回输入
    if drop_prob == 0.0 or not training:
        return input
    # 计算保留概率
    keep_prob = 1 - drop_prob
    # 根据输入的维度创建形状
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    # 生成随机张量
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    # 将随机张量二值化
    random_tensor.floor_()
    # 计算输出
    output = input.div(keep_prob) * random_tensor
    return output


class BeitDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


# Based on timm implementation, which can be found here:
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
class BeitEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.

    """

    def __init__(self, config: BeitConfig) -> None:
        super().__init__()

        # 创建 CLS token 参数
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        # 如果使用 mask token，则创建 mask token 参数
        if config.use_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        else:
            self.mask_token = None
        # 创建 patch embeddings
        self.patch_embeddings = BeitPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        # 如果使用绝对位置嵌入，则创建位置嵌入参数
        if config.use_absolute_position_embeddings:
            self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
        else:
            self.position_embeddings = None
        # 创建 dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    # 定义 forward 方法，用于前向传播，接受像素值和可选的掩码位置张量，返回嵌入向量和补丁的高度和宽度
    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: Optional[torch.BoolTensor] = None) -> torch.Tensor:
        # 使用 patch_embeddings 方法将像素值转换为嵌入向量，并获取补丁的高度和宽度
        embeddings, (patch_height, patch_width) = self.patch_embeddings(
            pixel_values, self.position_embeddings[:, 1:, :] if self.position_embeddings is not None else None
        )
        # 获取批次大小、序列长度和嵌入向量的维度
        batch_size, seq_len, _ = embeddings.size()

        # 如果存在掩码位置张量
        if bool_masked_pos is not None:
            # 将掩码的可视标记替换为掩码标记
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            # 将掩码位置张量扩展成与嵌入向量相同类型的张量
            w = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            # 将嵌入向量中掩码位置处的值替换为掩码标记
            embeddings = embeddings * (1 - w) + mask_tokens * w

        # 扩展 CLS 标记以匹配批次大小，并将其与位置嵌入相加
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        if self.position_embeddings is not None:
            cls_tokens = cls_tokens + self.position_embeddings[:, :1, :]

        # 将 CLS 标记与嵌入向量连接起来
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # 对连接后的嵌入向量应用 dropout
        embeddings = self.dropout(embeddings)

        # 返回嵌入向量和补丁的高度和宽度
        return embeddings, (patch_height, patch_width)
class BeitPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        # 从配置中获取图像大小和补丁大小
        image_size, patch_size = config.image_size, config.patch_size
        # 从配置中获取通道数和隐藏状态大小
        num_channels, hidden_size = config.num_channels, config.hidden_size

        # 如果图像大小和补丁大小不是可迭代对象，则转换为元组
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        # 计算图像分割为补丁后的数量
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        # 补丁形状为图像高度除以补丁高度，图像宽度除以补丁宽度
        patch_shape = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.patch_shape = patch_shape

        # 创建卷积层，用于将通道映射为隐藏状态大小
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor, position_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 获取像素值的形状
        batch_size, num_channels, height, width = pixel_values.shape
        # 如果通道数与配置中的不匹配，则引发错误
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )

        # 通过投影将像素值转换为补丁嵌入
        embeddings = self.projection(pixel_values)
        # 获取补丁的高度和宽度
        patch_height, patch_width = embeddings.shape[2], embeddings.shape[3]

        if position_embedding is not None:
            # 将位置嵌入插值到相应的大小
            position_embedding = position_embedding.view(1, self.patch_shape[0], self.patch_shape[1], -1).permute(
                0, 3, 1, 2
            )
            position_embedding = nn.functional.interpolate(
                position_embedding, size=(patch_height, patch_width), mode="bicubic"
            )
            embeddings = embeddings + position_embedding

        # 将嵌入展平并转置，以匹配 Transformer 的输入形状
        embeddings = embeddings.flatten(2).transpose(1, 2)

        return embeddings, (patch_height, patch_width)


class BeitSelfAttention(nn.Module):
    # 初始化方法，接受配置和窗口大小参数
    def __init__(self, config: BeitConfig, window_size: Optional[tuple] = None) -> None:
        # 调用父类的初始化方法
        super().__init__()
        # 如果隐藏层大小不能被注意力头数整除，并且配置中没有嵌入大小，则引发值错误异常
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        # 设置注意力头数和每个注意力头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 创建线性层用于查询、键和值的映射
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 创建用于掩码的丢弃层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # 如果存在窗口大小，则创建相对位置偏置对象，否则为 None
        if window_size:
            self.relative_position_bias = BeitRelativePositionBias(config, window_size=window_size)
        else:
            self.relative_position_bias = None

    # 用于调整张量形状以适应注意力计算的格式
    def transpose_for_scores(self, x):
        # 创建新的张量形状，将注意力头和注意力头大小放置到适当的位置
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # 重新调整张量形状
        x = x.view(*new_x_shape)
        # 交换张量的维度顺序，以符合注意力计算的格式
        return x.permute(0, 2, 1, 3)

    # 前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        relative_position_bias: Optional["BeitRelativePositionBias"] = None,
        ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        # 使用 query 网络层对隐藏状态进行处理
        mixed_query_layer = self.query(hidden_states)

        # 使用 key 网络层对隐藏状态进行处理，并转置以便计算注意力分数
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        # 使用 value 网络层对隐藏状态进行处理，并转置以便计算注意力分数
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        # 使用 mixed_query_layer 转置以便计算注意力分数
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 计算原始注意力分数，即 query 和 key 的点积
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # 对注意力分数进行缩放
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 如果存在相对位置偏置，则添加到注意力分数中
        if self.relative_position_bias is not None:
            attention_scores = attention_scores + self.relative_position_bias().unsqueeze(0)

        # 如果提供了共享的相对位置偏置，则添加到注意力分数中
        if relative_position_bias is not None:
            attention_scores = attention_scores + relative_position_bias

        # 将注意力分数归一化为概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 使用 dropout 随机丢弃一些 token 的注意力概率
        attention_probs = self.dropout(attention_probs)

        # 如果需要，对头部进行掩码操作
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算上下文向量，即注意力概率与 value 层的乘积
        context_layer = torch.matmul(attention_probs, value_layer)

        # 调整上下文向量的维度顺序
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # 根据是否需要输出注意力权重，返回不同的输出结果
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
class BeitSelfOutput(nn.Module):
    """
    The residual connection is defined in BeitLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """
    # 定义 BeitSelfOutput 类，用于处理自注意力机制输出
    def __init__(self, config: BeitConfig) -> None:
        super().__init__()
        # 创建线性层，用于自注意力机制输出的线性变换
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建 dropout 层，用于对输出进行随机失活
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor, gamma=None) -> torch.Tensor:
        # 通过线性层对隐藏状态进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的结果进行随机失活
        hidden_states = self.dropout(hidden_states)
        # 返回处理后的结果
        return hidden_states


class BeitAttention(nn.Module):
    # 定义 BeitAttention 类，用于处理注意力机制
    def __init__(self, config: BeitConfig, window_size: Optional[tuple] = None) -> None:
        super().__init__()
        # 创建自注意力机制实例
        self.attention = BeitSelfAttention(config, window_size=window_size)
        # 创建自注意力机制输出层实例
        self.output = BeitSelfOutput(config)
        # 初始化被修剪的头集合为空
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 查找可修剪头的索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # 修剪线性层
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储修剪的头
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        relative_position_bias: Optional["BeitRelativePositionBias"] = None,
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        # 通过自注意力机制计算输出
        self_outputs = self.attention(hidden_states, head_mask, output_attentions, relative_position_bias)

        # 将自注意力机制的输出输入到自注意力机制输出层
        attention_output = self.output(self_outputs[0], hidden_states)

        # 如果需要输出注意力权重，则将其添加到输出中
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BeitIntermediate(nn.Module):
    # 定义 BeitIntermediate 类，用于处理中间层
    def __init__(self, config: BeitConfig) -> None:
        super().__init__()
        # 创建线性层，用于中间层的线性变换
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 根据配置选择激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
    # 前向传播函数，用于将输入的隐藏状态向前传播一层
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将隐藏状态通过全连接层进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的隐藏状态应用激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)

        # 返回处理后的隐藏状态
        return hidden_states
class BeitOutput(nn.Module):
    def __init__(self, config: BeitConfig) -> None:
        super().__init__()
        # 创建一个全连接层，输入维度为config.intermediate_size，输出维度为config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个dropout层，概率为config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入hidden_states传入全连接层
        hidden_states = self.dense(hidden_states)
        # 将全连接层的输出传入dropout层
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class BeitLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: BeitConfig, window_size: Optional[tuple] = None, drop_path_rate: float = 0.0) -> None:
        super().__init__()
        # 设置feed forward的chunk大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 设置序列长度的维度
        self.seq_len_dim = 1
        # 创建一个BeitAttention对象
        self.attention = BeitAttention(config, window_size=window_size)
        # 创建一个BeitIntermediate对象
        self.intermediate = BeitIntermediate(config)
        # 创建一个BeitOutput对象
        self.output = BeitOutput(config)
        # 创建LayerNorm层，用于在attention之前进行归一化
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 如果drop_path_rate大于0，则创建一个BeitDropPath对象，否则创建一个Identity对象
        self.drop_path = BeitDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        # 创建LayerNorm层，用于在attention之后进行归一化
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 初始化lambda_1和lambda_2参数
        init_values = config.layer_scale_init_value
        if init_values > 0:
            self.lambda_1 = nn.Parameter(init_values * torch.ones((config.hidden_size)), requires_grad=True)
            self.lambda_2 = nn.Parameter(init_values * torch.ones((config.hidden_size)), requires_grad=True)
        else:
            self.lambda_1, self.lambda_2 = None, None

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        relative_position_bias: Optional["BeitRelativePositionBias"] = None,
    # 定义函数的输入和输出类型注解
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        # 使用 self.attention 处理隐藏状态，得到自注意力输出
        self_attention_outputs = self.attention(
            # 在 BEiT 模型中，先对隐藏状态进行 layernorm 处理，然后再进行自注意力计算
            self.layernorm_before(hidden_states),
            head_mask,
            output_attentions=output_attentions,
            relative_position_bias=relative_position_bias,
        )
        # 从自注意力输出中提取自注意力结果
        attention_output = self_attention_outputs[0]
        # 如果输出注意力权重，则添加到输出元组中
        outputs = self_attention_outputs[1:]

        # 如果存在 lambda_1，则对自注意力输出进行缩放
        if self.lambda_1 is not None:
            attention_output = self.lambda_1 * attention_output

        # 第一个残差连接
        hidden_states = self.drop_path(attention_output) + hidden_states

        # 在 BEiT 模型中，自注意力计算后也应用 layernorm
        layer_output = self.layernorm_after(hidden_states)

        # 经过 intermediate 层处理
        layer_output = self.intermediate(layer_output)
        # 经过 output 层处理
        layer_output = self.output(layer_output)

        # 如果存在 lambda_2，则对输出进行缩放
        if self.lambda_2 is not None:
            layer_output = self.lambda_2 * layer_output

        # 第二个残差连接
        layer_output = self.drop_path(layer_output) + hidden_states

        # 将 layer_output 加入到输出元组中
        outputs = (layer_output,) + outputs

        # 返回输出元组
        return outputs
class BeitRelativePositionBias(nn.Module):
    def __init__(self, config: BeitConfig, window_size: tuple) -> None:
        super().__init__()
        self.window_size = window_size
        # 计算相对位置偏置表的大小
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        # 创建用于存储相对位置偏置的可学习参数，形状为(num_relative_distance, num_attention_heads)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, config.num_attention_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # 获取每个窗口内每个标记的配对相对位置索引
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # 将坐标移到以0为起点
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        # 创建用于存储相对位置索引的张量
        relative_position_index = torch.zeros(
            size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype
        )
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        # 注册相对位置索引为缓冲区，但不进行梯度计算
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

    def forward(self) -> torch.Tensor:
        # 获取相对位置偏置，根据相对位置索引和相对位置偏置表
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] + 1, self.window_size[0] * self.window_size[1] + 1, -1
        )  # Wh*Ww,Wh*Ww,nH

        # 调整维度顺序并返回相对位置偏置张量
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww


class BeitEncoder(nn.Module):
    # 初始化函数，接受配置和窗口大小参数，设置相对位置偏置对象或者为 None
    def __init__(self, config: BeitConfig, window_size: Optional[tuple] = None) -> None:
        # 调用父类的初始化函数
        super().__init__()
        # 保存配置信息
        self.config = config
        # 如果配置要使用共享的相对位置偏置，则创建相对位置偏置对象，否则为 None
        if config.use_shared_relative_position_bias:
            self.relative_position_bias = BeitRelativePositionBias(config, window_size=window_size)
        else:
            self.relative_position_bias = None

        # 根据随机深度衰减规则生成 drop path rate 列表
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
        # 创建多层 BeitLayer 模块列表
        self.layer = nn.ModuleList(
            [
                BeitLayer(
                    config,
                    window_size=window_size if config.use_relative_position_bias else None,
                    drop_path_rate=dpr[i],
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        # 是否使用梯度检查点
        self.gradient_checkpointing = False

    # 前向传播函数，接受隐藏状态、头部掩码、是否输出注意力、是否输出隐藏状态、是否返回字典等参数
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        # 初始化所有隐藏状态和自注意力
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # 遍历每一层的 BeitLayer 模块
        for i, layer_module in enumerate(self.layer):
            # 如果要输出隐藏状态，则保存当前隐藏状态
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 如果使用梯度检查点且处于训练模式，则调用梯度检查点函数
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                # 获取相对位置偏置
                relative_position_bias = (
                    self.relative_position_bias() if self.relative_position_bias is not None else None
                )
                # 调用当前层的前向传播函数
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions, relative_position_bias)

            # 更新隐藏状态
            hidden_states = layer_outputs[0]

            # 如果要输出注意力，则保存当前层的自注意力
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果要输出隐藏状态，则保存最终隐藏状态
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典，则返回非空的结果元组
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 返回 BaseModelOutput 对象
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
BEIT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`BeitConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

Explanation: This variable stores the documentation string (docstring) for the `BEIT_START_DOCSTRING`. It provides information about the purpose of the `BeitPreTrainedModel` class and its usage, including details on how to initialize the model and where to find further documentation.


BEIT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`BeitImageProcessor.__call__`] for details.

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

Explanation: This variable stores the documentation string (docstring) for the `BEIT_INPUTS_DOCSTRING`. It provides information about the input arguments expected by the `BeitPreTrainedModel.forward` method, including details on the pixel values, head mask, and optional arguments like `output_attentions`, `output_hidden_states`, and `return_dict`.
    # 创建一个字符串对象，描述了Beit模型的基本信息，输出原始的隐藏状态，没有特定的头部结构
    "The bare Beit Model transformer outputting raw hidden-states without any specific head on top.",
    # BEIT_START_DOCSTRING 用于引用 BEIT 模型文档字符串的起始标记
    BEIT_START_DOCSTRING,
# 定义一个名为BeitModel的类，继承自BeitPreTrainedModel类
class BeitModel(BeitPreTrainedModel):
    # 初始化方法，接受一个BeitConfig类型的config参数和一个布尔类型的add_pooling_layer参数
    def __init__(self, config: BeitConfig, add_pooling_layer: bool = True) -> None:
        # 调用父类的初始化方法
        super().__init__(config)
        # 将config参数保存到self.config中
        self.config = config

        # 创建BeitEmbeddings对象并保存到self.embeddings中
        self.embeddings = BeitEmbeddings(config)
        # 创建BeitEncoder对象并保存到self.encoder中，传入window_size参数
        self.encoder = BeitEncoder(config, window_size=self.embeddings.patch_embeddings.patch_shape)

        # 根据config中的use_mean_pooling属性选择不同的LayerNorm层
        self.layernorm = (
            nn.Identity() if config.use_mean_pooling else nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        )
        # 根据add_pooling_layer参数选择是否创建BeitPooler对象并保存到self.pooler中
        self.pooler = BeitPooler(config) if add_pooling_layer else None

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入层的方法
    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    # 剪枝模型中的注意力头
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 前向传播方法，接受多个参数
    @add_start_docstrings_to_model_forward(BEIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BeitModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, BeitModelOutputWithPooling]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        # 如果未提供 output_attentions 参数，则使用配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未提供 output_hidden_states 参数，则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未提供 return_dict 参数，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果未提供 pixel_values 参数，则引发 ValueError 异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 准备头部遮罩（head mask）如果需要
        # 在头部遮罩中，1.0 表示保留该头部
        # attention_probs 的形状为 bsz x n_heads x N x N
        # 输入的 head_mask 的形状为 [num_heads] 或 [num_hidden_layers x num_heads]
        # 将 head_mask 转换为形状 [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # 使用像素值和可选的 bool_masked_pos 参数来嵌入输入
        # 返回嵌入后的输出以及补丁的高度和宽度
        embedding_output, (patch_height, patch_width) = self.embeddings(pixel_values, bool_masked_pos)

        # 对嵌入后的输入进行编码
        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取编码器的序列输出
        sequence_output = encoder_outputs[0]
        # 对序列输出进行 LayerNormalization
        sequence_output = self.layernorm(sequence_output)
        # 如果存在池化器，则对序列输出进行池化
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        # 如果 return_dict 为 False，则返回序列输出以及可能的池化输出和其他编码器输出
        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        # 如果 return_dict 为 True，则返回包含序列输出、池化输出、隐藏状态和注意力分布的 BeitModelOutputWithPooling 对象
        return BeitModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 定义一个名为BeitPooler的类，继承自nn.Module
class BeitPooler(nn.Module):
    # 初始化方法，接受一个BeitConfig类型的参数config
    def __init__(self, config: BeitConfig) -> None:
        super().__init__()
        # 如果config中指定使用均值池化，则创建一个LayerNorm层
        self.layernorm = (
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) if config.use_mean_pooling else None
        )

    # 前向传播方法，接受一个torch.Tensor类型的hidden_states参数，返回一个torch.Tensor类型的结果
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 如果存在layernorm层
        if self.layernorm is not None:
            # 对patch tokens的最终隐藏状态进行均值池化
            patch_tokens = hidden_states[:, 1:, :]
            pooled_output = self.layernorm(patch_tokens.mean(1))
        else:
            # 通过简单地取[CLS]标记的最终隐藏状态进行池化
            pooled_output = hidden_states[:, 0]

        return pooled_output

# 定义一个名为BeitForMaskedImageModeling的类，继承自BeitPreTrainedModel
@add_start_docstrings(
    """Beit Model transformer with a 'language' modeling head on top. BEiT does masked image modeling by predicting
    visual tokens of a Vector-Quantize Variational Autoencoder (VQ-VAE), whereas other vision models like ViT and DeiT
    predict RGB pixel values. As a result, this class is incompatible with [`AutoModelForMaskedImageModeling`], so you
    will need to use [`BeitForMaskedImageModeling`] directly if you wish to do masked image modeling with BEiT.""",
    BEIT_START_DOCSTRING,
)
class BeitForMaskedImageModeling(BeitPreTrainedModel):
    # 初始化方法，接受一个BeitConfig类型的参数config
    def __init__(self, config: BeitConfig) -> None:
        super().__init__(config)

        # 设置num_labels属性为config中的num_labels
        self.num_labels = config.num_labels
        # 创建一个BeitModel实例，不添加池化层
        self.beit = BeitModel(config, add_pooling_layer=False)

        # 分类器头部
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法，接受多个参数，返回一个MaskedLMOutput类型的结果
    @add_start_docstrings_to_model_forward(BEIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, MaskedLMOutput]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).

        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, BeitForMaskedImageModeling
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
        >>> model = BeitForMaskedImageModeling.from_pretrained("microsoft/beit-base-patch16-224-pt22k")

        >>> num_patches = (model.config.image_size // model.config.patch_size) ** 2
        >>> pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
        >>> # create random boolean mask of shape (batch_size, num_patches)
        >>> bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()

        >>> outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
        >>> loss, logits = outputs.loss, outputs.logits
        >>> list(logits.shape)
        [1, 196, 8192]
        ```py"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 BEiT 模型进行前向传播
        outputs = self.beit(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取序列输出
        sequence_output = outputs[0]
        # 对序列输出进行 LayerNormalization
        sequence_output = self.layernorm(sequence_output)
        # 通过 lm_head 进行预测得分计算
        prediction_scores = self.lm_head(sequence_output[:, 1:])

        masked_lm_loss = None
        if labels is not None:
            # 定义交叉熵损失函数
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            # 计算 Masked Language Model 损失
            masked_lm_loss = loss_fct(prediction_scores[bool_masked_pos], labels)

        if not return_dict:
            # 如果不需要返回字典，则返回预测得分和其他输出
            output = (prediction_scores,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 返回 MaskedLMOutput 对象
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用 BEiT 模型进行图像分类，顶部有一个图像分类头部（线性层位于补丁标记的最终隐藏状态的平均值之上），例如用于 ImageNet 数据集
@add_start_docstrings(
    """
    Beit Model transformer with an image classification head on top (a linear layer on top of the average of the final
    hidden states of the patch tokens) e.g. for ImageNet.
    """,
    BEIT_START_DOCSTRING,
)
class BeitForImageClassification(BeitPreTrainedModel):
    # 初始化函数，接受一个 BeitConfig 类型的参数
    def __init__(self, config: BeitConfig) -> None:
        # 调用父类的初始化函数
        super().__init__(config)

        # 获取标签数量
        self.num_labels = config.num_labels
        # 创建 BEiT 模型，添加池化层
        self.beit = BeitModel(config, add_pooling_layer=True)

        # 分类器头部
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数
    @add_start_docstrings_to_model_forward(BEIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 确定是否返回字典格式的输出，如果未指定则根据配置决定是否返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 调用 BEiT 模型进行推断
        outputs = self.beit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果返回字典格式的输出，则获取池化后的输出；否则获取第二个输出
        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        # 使用分类器对池化后的输出进行分类
        logits = self.classifier(pooled_output)

        loss = None
        # 如果标签不为空，则计算损失
        if labels is not None:
            # 如果问题类型未指定，则根据标签类型和类别数量确定问题类型
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型选择相应的损失函数进行计算
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        # 如果不返回字典格式的输出，则构造输出元组
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回图像分类器输出
        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```  
class BeitConvModule(nn.Module):
    """
    A convolutional block that bundles conv/norm/activation layers. This block simplifies the usage of convolution
    layers, which are commonly used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).

    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int], str] = 0,
        bias: bool = False,
        dilation: Union[int, Tuple[int, int]] = 1,
    ) -> None:
        # 调用父类初始化方法
        super().__init__()
        # 定义卷积层
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            dilation=dilation,
        )
        # 定义批归一化层
        self.bn = nn.BatchNorm2d(out_channels)
        # 定义激活函数层
        self.activation = nn.ReLU()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 输入经过卷积层
        output = self.conv(input)
        # 输出经过批归一化层
        output = self.bn(output)
        # 输出经过激活函数层
        output = self.activation(output)

        return output


class BeitPyramidPoolingBlock(nn.Module):
    def __init__(self, pool_scale: int, in_channels: int, channels: int) -> None:
        super().__init__()
        # 创建一个池化模块和一个 BeiTConvModule 模块组成的层列表
        self.layers = [
            nn.AdaptiveAvgPool2d(pool_scale),
            BeitConvModule(in_channels, channels, kernel_size=1),
        ]
        # 将创建的层逐个加入模块
        for i, layer in enumerate(self.layers):
            self.add_module(str(i), layer)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 初始隐藏状态为输入
        hidden_state = input
        # 逐层计算隐藏状态
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class BeitPyramidPoolingModule(nn.Module):
    """
    Pyramid Pooling Module (PPM) used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        align_corners (bool): align_corners argument of F.interpolate.

    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    def __init__(self, pool_scales: Tuple[int, ...], in_channels: int, channels: int, align_corners: bool) -> None:
        super().__init__()
        # 初始化池化尺度、对齐角点和输入输出通道数
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.blocks = []
        # 对于每个池化尺度，创建一个 BeitPyramidPoolingBlock 模块，并加入到模块列表中
        for i, pool_scale in enumerate(pool_scales):
            block = BeitPyramidPoolingBlock(pool_scale=pool_scale, in_channels=in_channels, channels=channels)
            self.blocks.append(block)
            self.add_module(str(i), block)
    # 定义前向传播方法，接受一个张量 x 作为输入，并返回一个张量列表
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # 初始化用于存储每个尺度的特征的列表
        ppm_outs = []
        # 遍历每个空洞池化模块
        for ppm in self.blocks:
            # 将输入张量 x 传递给当前的空洞池化模块，得到输出
            ppm_out = ppm(x)
            # 对空洞池化模块的输出进行上采样，使其与输入张量 x 的尺寸相匹配
            upsampled_ppm_out = nn.functional.interpolate(
                ppm_out, size=x.size()[2:], mode="bilinear", align_corners=self.align_corners
            )
            # 将上采样后的输出添加到 ppm_outs 列表中
            ppm_outs.append(upsampled_ppm_out)
        # 返回所有空洞池化模块输出的列表
        return ppm_outs
class BeitUperHead(nn.Module):
    """
    Unified Perceptual Parsing for Scene Understanding. This head is the implementation of
    [UPerNet](https://arxiv.org/abs/1807.10221).

    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    def __init__(self, config: BeitConfig) -> None:
        super().__init__()

        # 初始化对象属性
        self.pool_scales = config.pool_scales  # e.g. (1, 2, 3, 6)
        self.in_channels = [config.hidden_size] * 4  # e.g. [768, 768, 768, 768]
        self.channels = config.hidden_size
        self.align_corners = False
        self.classifier = nn.Conv2d(self.channels, config.num_labels, kernel_size=1)

        # PSP Module
        # 初始化 PSP 模块
        self.psp_modules = BeitPyramidPoolingModule(
            self.pool_scales,
            self.in_channels[-1],
            self.channels,
            align_corners=self.align_corners,
        )
        # 瓶颈层
        self.bottleneck = BeitConvModule(
            self.in_channels[-1] + len(self.pool_scales) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
        )
        # FPN Module
        # 初始化 FPN 模块
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            # Lateral Convolutional Module
            l_conv = BeitConvModule(in_channels, self.channels, kernel_size=1)
            # Feature Pyramid Network Convolutional Module
            fpn_conv = BeitConvModule(self.channels, self.channels, kernel_size=3, padding=1)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # FPN 瓶颈层
        self.fpn_bottleneck = BeitConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
        )

    # PSP 模块的前向传播
    def psp_forward(self, inputs):
        x = inputs[-1]
        psp_outs = [x]
        # 对输入进行 PSP 操作
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        # 使用 PSP 模块的输出进行瓶颈处理
        output = self.bottleneck(psp_outs)

        return output
    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        # 构建侧向连接
        laterals = [lateral_conv(encoder_hidden_states[i]) for i, lateral_conv in enumerate(self.lateral_convs)]

        # 添加 PSP 模块的输出到侧向连接结果中
        laterals.append(self.psp_forward(encoder_hidden_states))

        # 构建自顶向下路径
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # 获取前一层特征图的形状
            prev_shape = laterals[i - 1].shape[2:]
            # 对当前层的特征图进行上采样，使其尺寸与前一层相同
            laterals[i - 1] = laterals[i - 1] + nn.functional.interpolate(
                laterals[i], size=prev_shape, mode="bilinear", align_corners=self.align_corners
            )

        # 构建输出
        # 对侧向连接中除最后一层外的每一层应用卷积层
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)]
        # 将 PSP 特征也加入到输出中
        fpn_outs.append(laterals[-1])

        # 对除最后一层外的每一层特征图进行上采样，使其尺寸与最底层相同
        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = nn.functional.interpolate(
                fpn_outs[i], size=fpn_outs[0].shape[2:], mode="bilinear", align_corners=self.align_corners
            )
        
        # 拼接所有特征图
        fpn_outs = torch.cat(fpn_outs, dim=1)
        # 使用 FPN 瓶颈层处理特征图
        output = self.fpn_bottleneck(fpn_outs)
        # 使用分类器进行最终的分类
        output = self.classifier(output)

        return output
class BeitFCNHead(nn.Module):
    """
    Fully Convolution Networks for Semantic Segmentation. This head is implemented of
    [FCNNet](https://arxiv.org/abs/1411.4038>).

    Args:
        config (BeitConfig): Configuration.
        in_channels
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        dilation (int): The dilation rate for convs in the head. Default: 1.


    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    def __init__(
        self, config: BeitConfig, in_index: int = 2, kernel_size: int = 3, dilation: Union[int, Tuple[int, int]] = 1
    ) -> None:
        # 初始化函数，设置头部的卷积网络结构
        super().__init__()
        self.in_channels = config.hidden_size
        self.channels = config.auxiliary_channels
        self.num_convs = config.auxiliary_num_convs
        self.concat_input = config.auxiliary_concat_input
        self.in_index = in_index

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        # 创建卷积模块列表
        convs.append(
            BeitConvModule(
                self.in_channels, self.channels, kernel_size=kernel_size, padding=conv_padding, dilation=dilation
            )
        )
        for i in range(self.num_convs - 1):
            convs.append(
                BeitConvModule(
                    self.channels, self.channels, kernel_size=kernel_size, padding=conv_padding, dilation=dilation
                )
            )
        if self.num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = BeitConvModule(
                self.in_channels + self.channels, self.channels, kernel_size=kernel_size, padding=kernel_size // 2
            )

        self.classifier = nn.Conv2d(self.channels, config.num_labels, kernel_size=1)

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        # 前向传播函数，对编码器隐藏状态进行处理并返回输出
        # 只取相关的特征图
        hidden_states = encoder_hidden_states[self.in_index]
        output = self.convs(hidden_states)
        if self.concat_input:
            output = self.conv_cat(torch.cat([hidden_states, output], dim=1))
        output = self.classifier(output)
        return output


@add_start_docstrings(
    """
    Beit Model transformer with a semantic segmentation head on top e.g. for ADE20k, CityScapes.
    """,
    BEIT_START_DOCSTRING,
)
class BeitForSemanticSegmentation(BeitPreTrainedModel):
    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config: BeitConfig) -> None:
        # 调用父类的初始化方法
        super().__init__(config)

        # 将配置对象中的标签数量赋给实例变量
        self.num_labels = config.num_labels
        # 使用配置对象初始化 BEiT 模型，不添加池化层
        self.beit = BeitModel(config, add_pooling_layer=False)

        # FPNs
        # 检查 config.out_indices 是否包含 4 个整数，指定了从骨干网中使用哪些特征。在基本尺寸的架构中，可以使用 [3, 5, 7, 11]
        if len(self.config.out_indices) != 4:
            # 如果不是，抛出数值错误
            raise ValueError(
                "BeitForSemanticSegmentation requires config.out_indices to be a list of 4 integers, "
                "specifying which features to use from the backbone. One can use [3, 5, 7, 11] in case of "
                "a base-sized architecture."
            )
        # 第一个特征金字塔网络
        self.fpn1 = nn.Sequential(
            # 转置卷积层，将输入大小扩大两倍
            nn.ConvTranspose2d(config.hidden_size, config.hidden_size, kernel_size=2, stride=2),
            # 批归一化层
            nn.BatchNorm2d(config.hidden_size),
            # GELU 激活函数
            nn.GELU(),
            # 转置卷积层，将输入大小再次扩大两倍
            nn.ConvTranspose2d(config.hidden_size, config.hidden_size, kernel_size=2, stride=2),
        )
        # 第二个特征金字塔网络
        self.fpn2 = nn.Sequential(
            # 转置卷积层，将输入大小扩大两倍
            nn.ConvTranspose2d(config.hidden_size, config.hidden_size, kernel_size=2, stride=2),
        )
        # 第三个特征金字塔网络，恒等映射，即不进行任何操作
        self.fpn3 = nn.Identity()
        # 第四个特征金字塔网络，最大池化层，将输入大小缩小两倍
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 语义分割头部
        # 解码头部
        self.decode_head = BeitUperHead(config)
        # 辅助头部，如果配置中使用辅助头部，则初始化，否则为 None
        self.auxiliary_head = BeitFCNHead(config) if config.use_auxiliary_head else None

        # 初始化权重并应用最终处理
        self.post_init()

    # 计算损失函数
    def compute_loss(self, logits, auxiliary_logits, labels):
        # 将 logits 上采样到原始图像大小
        upsampled_logits = nn.functional.interpolate(
            logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )
        if auxiliary_logits is not None:
            # 如果有辅助 logits，将其上采样到原始图像大小
            upsampled_auxiliary_logits = nn.functional.interpolate(
                auxiliary_logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
        # 计算加权损失
        loss_fct = CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)
        # 计算主要损失
        main_loss = loss_fct(upsampled_logits, labels)
        loss = main_loss
        if auxiliary_logits is not None:
            # 如果有辅助 logits，计算辅助损失并加到总损失上
            auxiliary_loss = loss_fct(upsampled_auxiliary_logits, labels)
            loss += self.config.auxiliary_loss_weight * auxiliary_loss

        return loss

    # 前向传播方法
    @add_start_docstrings_to_model_forward(BEIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SemanticSegmenterOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 使用 BEiT 骨干网络，可用于类似 DETR 和 MaskFormer 的框架
@add_start_docstrings(
    """
    BEiT backbone, to be used with frameworks like DETR and MaskFormer.
    """,
    BEIT_START_DOCSTRING,
)
class BeitBackbone(BeitPreTrainedModel, BackboneMixin):
    def __init__(self, config):
        # 调用父类构造函数
        super().__init__(config)
        # 初始化骨干网络
        super()._init_backbone(config)

        # 设置特征数量为隐藏层大小，根据隐藏层数量
        self.num_features = [config.hidden_size for _ in range(config.num_hidden_layers + 1)]
        # 创建 BEiT 嵌入层
        self.embeddings = BeitEmbeddings(config)
        # 创建 BEiT 编码器
        self.encoder = BeitEncoder(config, window_size=self.embeddings.patch_embeddings.patch_shape)

        # 如果需要添加 FPN
        if config.add_fpn:
            # 检查输出索引是否为4个整数
            if len(self.config.out_indices) != 4:
                raise ValueError(
                    "BeitBackbone requires config.out_indices to be a list of 4 integers, "
                    "specifying which features to use from the backbone. One can use [3, 5, 7, 11] in case of "
                    "a base-sized architecture."
                )
            hidden_size = config.hidden_size
            # 创建 FPN1
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size=2, stride=2),
                nn.BatchNorm2d(hidden_size, eps=config.batch_norm_eps),
                nn.GELU(),
                nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size=2, stride=2),
            )

            # 创建 FPN2
            self.fpn2 = nn.Sequential(nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size=2, stride=2))
            # 创建 FPN3
            self.fpn3 = nn.Identity()
            # 创建 FPN4
            self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    # 前向传播函数
    @add_start_docstrings_to_model_forward(BEIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BackboneOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Tensor,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> BackboneOutput:
        """
        返回：BackboneOutput对象，包含特征图、隐藏状态和注意力权重信息

        示例：

        ```py
        >>> from transformers import AutoImageProcessor, AutoBackbone
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> processor = AutoImageProcessor.from_pretrained("microsoft/beit-base-patch16-224")
        >>> model = AutoBackbone.from_pretrained(
        ...     "microsoft/beit-base-patch16-224", out_features=["stage1", "stage2", "stage3", "stage4"]
        ... )

        >>> inputs = processor(image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> feature_maps = outputs.feature_maps
        >>> list(feature_maps[-1].shape)
        [1, 768, 14, 14]
        ```"""

        # 如果return_dict参数不为None，则使用参数值；否则使用模型配置中的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果output_hidden_states参数不为None，则使用参数值；否则使用模型配置中的值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果output_attentions参数不为None，则使用参数值；否则使用模型配置中的值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        # 获取像素值的批次大小
        batch_size = pixel_values.shape[0]
        # 对像素值进行嵌入处理，得到嵌入输出和每个补丁的高度和宽度
        embedding_output, (patch_height, patch_width) = self.embeddings(pixel_values)

        # 对嵌入输出进行编码，返回隐藏状态和注意力权重
        outputs = self.encoder(
            embedding_output, output_hidden_states=True, output_attentions=output_attentions, return_dict=return_dict
        )

        # 如果return_dict为True，则将隐藏状态作为字典返回；否则作为元组返回
        hidden_states = outputs.hidden_states if return_dict else outputs[1]

        # 初始化特征图的空元组
        feature_maps = ()
        # 遍历阶段名称和隐藏状态，生成特征图
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                # 如果配置允许，对隐藏状态进行重塑
                if self.config.reshape_hidden_states:
                    hidden_state = hidden_state[:, 1:, :]
                    hidden_state = hidden_state.permute(0, 2, 1)
                    hidden_state = hidden_state.reshape(batch_size, -1, patch_height, patch_width)

                # 将特征图添加到特征图元组中
                feature_maps += (hidden_state,)

        # 如果配置中包含添加FPN，则进行FPN处理
        if self.config.add_fpn:
            feature_maps = [
                self.fpn1(feature_maps[0]),
                self.fpn2(feature_maps[1]),
                self.fpn3(feature_maps[2]),
                self.fpn4(feature_maps[3]),
            ]
            feature_maps = tuple(feature_maps)

        # 如果return_dict为False，则根据输出是否包含隐藏状态返回输出
        if not return_dict:
            if output_hidden_states:
                output = (feature_maps,) + outputs[1:]
            else:
                output = (feature_maps,) + outputs[2:]
            return output

        # 返回BackboneOutput对象，包含特征图、隐藏状态和注意力权重信息
        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )
```