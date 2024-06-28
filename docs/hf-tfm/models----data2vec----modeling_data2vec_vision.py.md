# `.\models\data2vec\modeling_data2vec_vision.py`

```
# coding=utf-8
# 声明版权信息，此文件版权归 Meta Platforms 和 The HuggingFace Inc. 团队所有
#
# 根据 Apache 许可证版本 2.0 进行许可，除非符合许可证的要求，否则不得使用此文件
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发本软件
# 本软件不附带任何明示或暗示的担保或条件
# 有关具体的语言授权，请参阅许可证
""" PyTorch Data2VecVision 模型。"""


import collections.abc  # 导入 collections.abc 模块
import math  # 导入 math 模块
from dataclasses import dataclass  # 从 dataclasses 模块导入 dataclass 装饰器
from typing import List, Optional, Tuple, Union  # 导入类型提示

import torch  # 导入 PyTorch 库
import torch.utils.checkpoint  # 导入 PyTorch 的 checkpoint 模块
from torch import nn  # 从 PyTorch 导入 nn 模块
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss  # 从 nn 导入三种损失函数

from ...activations import ACT2FN  # 从本地导入 ACT2FN 激活函数
from ...modeling_outputs import (  # 导入模型输出相关的类
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
    SemanticSegmenterOutput,
)
from ...modeling_utils import PreTrainedModel  # 从 modeling_utils 导入 PreTrainedModel 类
from ...pytorch_utils import (  # 导入 PyTorch 工具函数
    find_pruneable_heads_and_indices,
    meshgrid,
    prune_linear_layer,
)
from ...utils import (  # 导入通用实用函数
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_data2vec_vision import Data2VecVisionConfig  # 导入 Data2VecVisionConfig 配置类


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

# General docstring
_CONFIG_FOR_DOC = "Data2VecVisionConfig"  # 文档字符串中使用的配置名称

# Base docstring
_CHECKPOINT_FOR_DOC = "facebook/data2vec-vision-base"  # 文档字符串中使用的基础检查点名称
_EXPECTED_OUTPUT_SHAPE = [1, 197, 768]  # 预期输出的形状为 1x197x768

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "facebook/data2vec-vision-base-ft1k"  # 图像分类使用的检查点名称
_IMAGE_CLASS_EXPECTED_OUTPUT = "remote control, remote"  # 图像分类的预期输出描述

DATA2VEC_VISION_PRETRAINED_MODEL_ARCHIVE_LIST = [  # Data2VecVision 预训练模型的存档列表
    "facebook/data2vec-vision-base-ft1k",
    # 查看所有 Data2VecVision 模型，请访问 https://huggingface.co/models?filter=data2vec-vision
]


@dataclass
# 从 transformers.models.beit.modeling_beit.BeitModelOutputWithPooling 复制的 Data2VecVisionModelOutputWithPooling 类定义
class Data2VecVisionModelOutputWithPooling(BaseModelOutputWithPooling):
    """
    [`Data2VecVisionModel`] 的输出类。
    """
    pass  # 此处为占位符，表示暂无额外实现
    # 将最后一层模型的隐藏状态作为输入，用于特征提取或下游任务的输入
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            # 模型最后一层的隐藏状态序列，形状为 `(batch_size, sequence_length, hidden_size)`
            Sequence of hidden-states at the output of the last layer of the model.
    
    # 如果 *config.use_mean_pooling* 设置为 True，则返回除 *[CLS]* 标记外的补丁标记的最后一层隐藏状态的平均值；
    # 如果设置为 False，则返回 *[CLS]* 标记的最终隐藏状态。
    pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
        # 如果 *config.use_mean_pooling* 设置为 True，则返回补丁标记的最后一层隐藏状态的平均值。
        # 如果设置为 False，则返回 *[CLS]* 标记的最终隐藏状态。
        Average of the last layer hidden states of the patch tokens (excluding the *[CLS]* token) if
        *config.use_mean_pooling* is set to True. If set to False, then the final hidden state of the *[CLS]* token
        will be returned.
    
    # 可选参数，当 `output_hidden_states=True` 时返回，或当 `config.output_hidden_states=True` 时返回，
    # 返回模型每一层的隐藏状态，包括初始嵌入输出。
    hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
        # 可选参数，当 `output_hidden_states=True` 时返回，或当 `config.output_hidden_states=True` 时返回，
        # 包含 `torch.FloatTensor` 的元组（一个用于嵌入的输出 + 一个用于每层输出），
        # 形状为 `(batch_size, sequence_length, hidden_size)`。
        Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
        shape `(batch_size, sequence_length, hidden_size)`.
    
    # 可选参数，当 `output_attentions=True` 时返回，或当 `config.output_attentions=True` 时返回，
    # 返回每层的注意力权重，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
    attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
        # 可选参数，当 `output_attentions=True` 时返回，或当 `config.output_attentions=True` 时返回，
        # 包含 `torch.FloatTensor` 的元组（每层一个），形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
        Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
        sequence_length)`.
    
        Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
        heads.
# Copied from transformers.models.beit.modeling_beit.drop_path
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


# Copied from transformers.models.beit.modeling_beit.BeitDropPath with Beit->Data2VecVision
class Data2VecVisionDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 调用全局函数 drop_path，传入当前实例的 drop_prob 和训练状态 self.training
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        # 返回描述当前实例 drop_prob 的字符串表示
        return "p={}".format(self.drop_prob)


# Copied from transformers.models.beit.modeling_beit.BeitEmbeddings with Beit->Data2VecVision
class Data2VecVisionEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.

    """

    def __init__(self, config: Data2VecVisionConfig) -> None:
        super().__init__()

        # 定义 CLS token 参数作为可学习参数
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        
        # 如果配置中启用了 mask token，则定义 mask token 参数作为可学习参数
        if config.use_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        else:
            self.mask_token = None
        
        # 初始化 patch embeddings，根据配置确定是否包含绝对位置 embeddings
        self.patch_embeddings = Data2VecVisionPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        
        # 如果配置中启用了绝对位置 embeddings，则定义位置 embeddings 参数作为可学习参数
        if config.use_absolute_position_embeddings:
            self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
        else:
            self.position_embeddings = None
        
        # 定义 dropout 层，使用配置中的隐藏层 dropout 概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: Optional[torch.BoolTensor] = None) -> torch.Tensor:
        # 使用 patch_embeddings 方法得到嵌入向量和补丁的高度和宽度信息
        embeddings, (patch_height, patch_width) = self.patch_embeddings(
            pixel_values, self.position_embeddings[:, 1:, :] if self.position_embeddings is not None else None
        )
        # 获取批量大小、序列长度和嵌入向量的维度
        batch_size, seq_len, _ = embeddings.size()

        if bool_masked_pos is not None:
            # 根据掩码位置替换被掩码的视觉令牌为 mask_tokens
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            w = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1 - w) + mask_tokens * w

        # 扩展 cls_token 以匹配当前批次的维度
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # 如果存在位置嵌入，则将其添加到 cls_tokens 中
        if self.position_embeddings is not None:
            cls_tokens = cls_tokens + self.position_embeddings[:, :1, :]

        # 将 cls_tokens 和 embeddings 沿着序列长度维度拼接
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # 对 embeddings 应用 dropout
        embeddings = self.dropout(embeddings)

        # 返回嵌入向量和补丁的高度和宽度信息
        return embeddings, (patch_height, patch_width)
# Copied from transformers.models.beit.modeling_beit.BeitPatchEmbeddings with Beit->Data2VecVision
class Data2VecVisionPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        # Extract configuration parameters
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        # Ensure image_size and patch_size are tuples
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)

        # Calculate number of patches and patch shape
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        patch_shape = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])

        # Store parameters as attributes
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.patch_shape = patch_shape

        # Projection layer to generate patch embeddings
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor, position_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Extract dimensions from input pixel values
        batch_size, num_channels, height, width = pixel_values.shape

        # Check if number of channels matches the configuration
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )

        # Project pixel values into patch embeddings
        embeddings = self.projection(pixel_values)
        patch_height, patch_width = embeddings.shape[2], embeddings.shape[3]

        # Add position embeddings if provided
        if position_embedding is not None:
            # Reshape and interpolate position embeddings to match patch size
            position_embedding = position_embedding.view(1, self.patch_shape[0], self.patch_shape[1], -1).permute(
                0, 3, 1, 2
            )
            position_embedding = nn.functional.interpolate(
                position_embedding, size=(patch_height, patch_width), mode="bicubic"
            )
            embeddings = embeddings + position_embedding

        # Flatten embeddings and transpose for further processing
        embeddings = embeddings.flatten(2).transpose(1, 2)

        return embeddings, (patch_height, patch_width)


# Copied from transformers.models.beit.modeling_beit.BeitSelfAttention with Beit->Data2VecVision
class Data2VecVisionSelfAttention(nn.Module):
    # 初始化函数，接受一个配置对象和一个可选的窗口大小参数
    def __init__(self, config: Data2VecVisionConfig, window_size: Optional[tuple] = None) -> None:
        # 调用父类的初始化方法
        super().__init__()
        
        # 检查隐藏层大小是否能被注意力头数整除，同时不存在嵌入大小属性
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            # 如果不满足条件，抛出数值错误异常
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )
        
        # 设置注意力头数和每个注意力头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 定义查询、键、值的线性变换层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 定义注意力概率的丢弃层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # 如果存在窗口大小参数，初始化相对位置偏置对象
        if window_size:
            self.relative_position_bias = Data2VecVisionRelativePositionBias(config, window_size=window_size)
        else:
            self.relative_position_bias = None

    # 将输入张量 x 转换为注意力分数的形状
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播函数，接受隐藏状态张量等输入，可选的头部掩码、是否输出注意力矩阵、相对位置偏置参数
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        relative_position_bias: Optional["Data2VecVisionRelativePositionBias"] = None,
        ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        # 从隐藏状态生成混合查询层
        mixed_query_layer = self.query(hidden_states)

        # 使用self.key处理隐藏状态并为得分转置以获取键层
        key_layer = self.transpose_for_scores(self.key(hidden_states))

        # 使用self.value处理隐藏状态并为得分转置以获取值层
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # 为混合查询层转置以获取查询层
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 计算原始注意力分数，即查询与键的点积
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # 根据注意力头大小对得分进行缩放
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 如果存在相对位置偏置，则添加到注意力分数中
        if self.relative_position_bias is not None:
            attention_scores = attention_scores + self.relative_position_bias().unsqueeze(0)

        # 如果提供了共享的相对位置偏置，则也添加到注意力分数中
        if relative_position_bias is not None:
            attention_scores = attention_scores + relative_position_bias

        # 将注意力分数归一化为注意力概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 对注意力概率进行Dropout处理，实际上是随机丢弃整个token的注意力
        attention_probs = self.dropout(attention_probs)

        # 如果需要，应用头部掩码
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算上下文向量，将注意力概率与值层相乘
        context_layer = torch.matmul(attention_probs, value_layer)

        # 对上下文向量进行维度重排，以便与Transformer模型中的预期形状一致
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # 根据需要输出注意力分数
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
# 从 transformers.models.beit.modeling_beit.BeitSelfOutput 复制而来，将 Beit 替换为 Data2VecVision
class Data2VecVisionSelfOutput(nn.Module):
    """
    在 Data2VecVisionLayer 中定义了残差连接，而不是在这里（像其他模型一样），这是因为在每个块之前应用了 layernorm。
    """

    def __init__(self, config: Data2VecVisionConfig) -> None:
        super().__init__()
        # 创建一个全连接层，输入和输出大小都为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建一个 dropout 层，使用的 dropout 概率为 config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor, gamma=None) -> torch.Tensor:
        # 使用全连接层处理 hidden_states
        hidden_states = self.dense(hidden_states)
        # 对处理后的 hidden_states 应用 dropout
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# 从 transformers.models.beit.modeling_beit.BeitAttention 复制而来，将 Beit 替换为 Data2VecVision
class Data2VecVisionAttention(nn.Module):
    def __init__(self, config: Data2VecVisionConfig, window_size: Optional[tuple] = None) -> None:
        super().__init__()
        # 创建一个 Data2VecVisionSelfAttention 实例，传入 config 和可选的 window_size 参数
        self.attention = Data2VecVisionSelfAttention(config, window_size=window_size)
        # 创建一个 Data2VecVisionSelfOutput 实例，传入 config
        self.output = Data2VecVisionSelfOutput(config)
        # 创建一个空集合，用于存储被修剪的注意力头
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 调用 find_pruneable_heads_and_indices 函数找到可修剪的头部及其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # 修剪线性层
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储修剪的头部
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        relative_position_bias: Optional["Data2VecVisionRelativePositionBias"] = None,
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        # 调用 self.attention 进行注意力计算
        self_outputs = self.attention(hidden_states, head_mask, output_attentions, relative_position_bias)

        # 使用 self.output 处理 self_outputs[0] 和 hidden_states，得到注意力输出
        attention_output = self.output(self_outputs[0], hidden_states)

        # 构建输出元组，如果需要输出注意力，则添加到元组中
        outputs = (attention_output,) + self_outputs[1:]  # 如果需要输出注意力，则添加到元组中
        return outputs


# 从 transformers.models.beit.modeling_beit.BeitIntermediate 复制而来，将 Beit 替换为 Data2VecVision
class Data2VecVisionIntermediate(nn.Module):
    # 初始化方法，用于创建一个新的Data2VecVisionConfig对象的实例
    def __init__(self, config: Data2VecVisionConfig) -> None:
        # 调用父类（nn.Module）的初始化方法
        super().__init__()
        # 创建一个线性层，输入维度是config.hidden_size，输出维度是config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果config.hidden_act是字符串类型，则从ACT2FN字典中获取对应的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            # 否则直接使用config.hidden_act作为激活函数
            self.intermediate_act_fn = config.hidden_act

    # 前向传播方法，接收一个张量hidden_states作为输入，返回一个张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入张量通过线性层self.dense进行线性变换
        hidden_states = self.dense(hidden_states)
        # 将线性变换后的张量通过激活函数self.intermediate_act_fn进行非线性变换
        hidden_states = self.intermediate_act_fn(hidden_states)

        # 返回经过线性和非线性变换后的张量作为输出
        return hidden_states
# Copied from transformers.models.beit.modeling_beit.BeitOutput with Beit->Data2VecVision
class Data2VecVisionOutput(nn.Module):
    def __init__(self, config: Data2VecVisionConfig) -> None:
        super().__init__()
        # 定义一个全连接层，将输入特征维度转换为隐藏状态特征维度
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 定义一个dropout层，用于随机将输入张量中部分元素设为0，以减少过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入张量经过全连接层进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的张量进行dropout操作
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# Copied from transformers.models.beit.modeling_beit.BeitLayer with Beit->Data2VecVision,BEiT->Data2VecVision
class Data2VecVisionLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(
        self, config: Data2VecVisionConfig, window_size: Optional[tuple] = None, drop_path_rate: float = 0.0
    ) -> None:
        super().__init__()
        # 设置前馈chunk的大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度维度为1
        self.seq_len_dim = 1
        # 定义注意力层，包括自注意力和相对位置编码
        self.attention = Data2VecVisionAttention(config, window_size=window_size)
        # 定义中间层，包括全连接和dropout操作
        self.intermediate = Data2VecVisionIntermediate(config)
        # 定义输出层，包括全连接和dropout操作
        self.output = Data2VecVisionOutput(config)
        # 定义LayerNorm层，在特定维度上对输入进行归一化
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 根据drop_path_rate的值，定义DropPath层或者恒等映射
        self.drop_path = Data2VecVisionDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        # 定义LayerNorm层，在特定维度上对输入进行归一化
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 根据配置初始化lambda_1和lambda_2参数
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
        relative_position_bias: Optional["Data2VecVisionRelativePositionBias"] = None,
        # 相对位置偏置，用于考虑局部和全局的关系
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        # 使用 self.attention 对 hidden_states 进行自注意力计算
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # 在 Data2VecVision 中，self-attention 之前应用 layernorm
            head_mask,
            output_attentions=output_attentions,
            relative_position_bias=relative_position_bias,
        )
        # 获取自注意力计算的输出
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # 如果输出注意力权重，将注意力也加入到输出中

        # 如果存在 lambda_1，则对 attention_output 应用缩放
        if self.lambda_1 is not None:
            attention_output = self.lambda_1 * attention_output

        # 第一个残差连接
        hidden_states = self.drop_path(attention_output) + hidden_states

        # 在 Data2VecVision 中，self-attention 之后也应用 layernorm
        layer_output = self.layernorm_after(hidden_states)

        # 应用中间层和输出层的变换
        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output)

        # 如果存在 lambda_2，则对 layer_output 应用缩放
        if self.lambda_2 is not None:
            layer_output = self.lambda_2 * layer_output

        # 第二个残差连接
        layer_output = self.drop_path(layer_output) + hidden_states

        # 将最终输出组装成 outputs 元组
        outputs = (layer_output,) + outputs

        return outputs
# 从 transformers.models.beit.modeling_beit.BeitRelativePositionBias 复制代码，并将 Beit 改为 Data2VecVision
class Data2VecVisionRelativePositionBias(nn.Module):
    def __init__(self, config: Data2VecVisionConfig, window_size: tuple) -> None:
        super().__init__()
        self.window_size = window_size
        # 计算相对位置偏置表的大小
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, config.num_attention_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # 获取窗口内每个标记的成对相对位置索引
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # 从0开始移动
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = torch.zeros(
            size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype
        )
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

    def forward(self) -> torch.Tensor:
        # 获取相对位置偏置，形状为 nH, Wh*Ww, Wh*Ww
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] + 1, self.window_size[0] * self.window_size[1] + 1, -1
        )  # Wh*Ww,Wh*Ww,nH

        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww


# 从 transformers.models.beit.modeling_beit.BeitEncoder 复制代码，并将 Beit 改为 Data2VecVision
class Data2VecVisionEncoder(nn.Module):
    def __init__(self, config: Data2VecVisionConfig, window_size: Optional[tuple] = None) -> None:
        super().__init__()
        self.config = config
        # 根据配置决定是否使用共享的相对位置偏置
        if config.use_shared_relative_position_bias:
            # 如果使用共享的相对位置偏置，则创建相应的对象
            self.relative_position_bias = Data2VecVisionRelativePositionBias(config, window_size=window_size)
        else:
            # 否则将相对位置偏置设为 None
            self.relative_position_bias = None

        # 计算随机深度衰减规则，生成一个列表
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
        # 创建一个 nn.ModuleList，包含多个 Data2VecVisionLayer 对象，每个对象使用不同的随机深度衰减率
        self.layer = nn.ModuleList(
            [
                Data2VecVisionLayer(
                    config,
                    window_size=window_size if config.use_relative_position_bias else None,
                    drop_path_rate=dpr[i],
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        # 设置梯度检查点为 False
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        # 如果需要输出隐藏状态，则初始化一个空的元组用于存储所有隐藏状态
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力权重，则初始化一个空的元组用于存储所有自注意力权重
        all_self_attentions = () if output_attentions else None

        # 遍历每个层次的模块
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前的隐藏状态加入到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 如果梯度检查点为开启且处于训练阶段，则使用梯度检查点函数进行前向传播
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                # 获取相对位置偏置（如果可用）
                relative_position_bias = (
                    self.relative_position_bias() if self.relative_position_bias is not None else None
                )
                # 对当前层进行前向传播，获取输出
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions, relative_position_bias)

            # 更新隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力权重，则将当前层的自注意力权重加入到 all_self_attentions 中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果需要输出隐藏状态，则将最终的隐藏状态加入到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果 return_dict 为 False，则返回一个元组，包含非空的结果项
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 否则返回一个 BaseModelOutput 对象，包含最终的隐藏状态、所有隐藏状态和所有自注意力权重
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
# 从 transformers.models.data2vec_vision.modeling_data2vec_vision 中复制代码，将 BeitPreTrainedModel 替换为 Data2VecVisionPreTrainedModel，beit 替换为 data2vec_vision
class Data2VecVisionPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 使用 Data2VecVisionConfig 作为配置类
    config_class = Data2VecVisionConfig
    # 基础模型前缀为 "data2vec_vision"
    base_model_prefix = "data2vec_vision"
    # 主输入名称为 "pixel_values"
    main_input_name = "pixel_values"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            # 使用正态分布初始化权重，均值为 0，标准差为 self.config.initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                # 如果存在偏置，则初始化为 0
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重，均值为 0，标准差为 self.config.initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                # 如果存在 padding_idx，则将对应位置的权重初始化为 0
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 初始化 LayerNorm 层的偏置为 0，权重为 1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


# DATA2VEC_VISION_START_DOCSTRING 的注释部分，提供了该模型的基本信息和使用说明
DATA2VEC_VISION_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`Data2VecVisionConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# DATA2VEC_VISION_INPUTS_DOCSTRING 暂时为空，用于描述模型的输入信息
DATA2VEC_VISION_INPUTS_DOCSTRING = r"""
    # 定义函数签名和参数说明
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
@add_start_docstrings(
    "The bare Data2VecVision Model transformer outputting raw hidden-states without any specific head on top.",
    DATA2VEC_VISION_START_DOCSTRING,
)
# 从 transformers.models.beit.modeling_beit.BeitModel 复制过来，将 BEIT->DATA2VEC_VISION, Beit->Data2VecVision, True->False
class Data2VecVisionModel(Data2VecVisionPreTrainedModel):
    def __init__(self, config: Data2VecVisionConfig, add_pooling_layer: bool = False) -> None:
        super().__init__(config)
        self.config = config

        # 初始化 Data2VecVisionModel
        self.embeddings = Data2VecVisionEmbeddings(config)  # 初始化视觉嵌入层
        self.encoder = Data2VecVisionEncoder(config, window_size=self.embeddings.patch_embeddings.patch_shape)  # 初始化编码器

        # 如果 config.use_mean_pooling 为 True，则使用 nn.Identity()；否则使用 nn.LayerNorm 初始化 layernorm
        self.layernorm = (
            nn.Identity() if config.use_mean_pooling else nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        )

        # 如果 add_pooling_layer 为 True，则初始化 Data2VecVisionPooler；否则设置为 None
        self.pooler = Data2VecVisionPooler(config) if add_pooling_layer else None

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings  # 返回输入嵌入层的 patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 对模型的注意力头进行修剪
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(DATA2VEC_VISION_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Data2VecVisionModelOutputWithPooling,
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
        ) -> Union[tuple, Data2VecVisionModelOutputWithPooling]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        # 根据传入参数或者配置确定是否返回注意力矩阵
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 根据传入参数或者配置确定是否返回隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 根据传入参数或者配置确定是否返回一个字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果像素值为 None，则抛出值错误异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 准备头部掩码（如果需要）
        # 在头部掩码中，1.0 表示保留该头部
        # attention_probs 的形状为 bsz x n_heads x N x N
        # 输入的 head_mask 形状为 [num_heads] 或者 [num_hidden_layers x num_heads]
        # head_mask 被转换为形状 [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # 使用嵌入层处理像素值和可选的布尔掩码位置
        embedding_output, (patch_height, patch_width) = self.embeddings(pixel_values, bool_masked_pos)

        # 使用编码器处理嵌入输出
        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 提取序列输出
        sequence_output = encoder_outputs[0]
        # 应用层归一化到序列输出
        sequence_output = self.layernorm(sequence_output)
        # 如果存在池化器，则将序列输出池化
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        # 如果不返回字典，则返回序列输出和池化输出的元组
        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        # 如果返回字典，则返回一个包含序列输出、池化输出、隐藏状态和注意力的数据结构
        return Data2VecVisionModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# Copied from transformers.models.beit.modeling_beit.BeitPooler with Beit->Data2VecVision
class Data2VecVisionPooler(nn.Module):
    def __init__(self, config: Data2VecVisionConfig) -> None:
        super().__init__()
        # 初始化层归一化层，如果配置中使用均值池化，则创建层归一化层
        self.layernorm = (
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) if config.use_mean_pooling else None
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.layernorm is not None:
            # 如果存在层归一化层，则对补丁令牌的最终隐藏状态进行均值池化
            patch_tokens = hidden_states[:, 1:, :]
            pooled_output = self.layernorm(patch_tokens.mean(1))
        else:
            # 否则，通过简单地取[CLS]令牌的最终隐藏状态来进行池化
            pooled_output = hidden_states[:, 0]

        return pooled_output


@add_start_docstrings(
    """
    Data2VecVision Model transformer with an image classification head on top (a linear layer on top of the average of
    the final hidden states of the patch tokens) e.g. for ImageNet.
    """,
    DATA2VEC_VISION_START_DOCSTRING,
)
# Copied from transformers.models.beit.modeling_beit.BeitForImageClassification with BEIT->DATA2VEC_VISION,Beit->Data2VecVision,beit->data2vec_vision
class Data2VecVisionForImageClassification(Data2VecVisionPreTrainedModel):
    def __init__(self, config: Data2VecVisionConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        # 创建 Data2VecVision 模型，添加池化层
        self.data2vec_vision = Data2VecVisionModel(config, add_pooling_layer=True)

        # 分类器头部
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(DATA2VEC_VISION_INPUTS_DOCSTRING)
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
        # 此处省略了函数的最后部分，因为要注意不要更改或省略任何部分
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 根据返回字典是否为空，确定是否使用预设的返回字典配置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 使用数据2向量视觉编码器处理像素值，并返回结果
        outputs = self.data2vec_vision(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果使用预设的返回字典配置，则从输出中获取汇聚输出；否则，从输出元组中获取第二个元素
        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        # 使用分类器模型对汇聚输出进行分类，得到预测的逻辑回归值
        logits = self.classifier(pooled_output)

        # 初始化损失值为None
        loss = None
        # 如果提供了标签
        if labels is not None:
            # 如果问题类型未定义，则根据标签数据类型和标签数量设置问题类型
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型计算对应的损失函数
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    # 对于单标签回归任务，计算均方误差损失
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    # 对于多标签回归任务，计算均方误差损失
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                # 对于单标签分类任务，使用交叉熵损失函数
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # 对于多标签分类任务，使用带logits的二元交叉熵损失函数
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        
        # 如果不使用预设的返回字典配置，则输出包含损失值在内的元组；否则，只输出模型预测的逻辑回归值
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 使用自定义的输出对象构建并返回结果，包括损失值、逻辑回归值、隐藏状态和注意力权重
        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# Copied from transformers.models.beit.modeling_beit.BeitConvModule with Beit->Data2VecVision
class Data2VecVisionConvModule(nn.Module):
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
        super().__init__()
        # 定义卷积层，输入通道数、输出通道数、卷积核大小、填充方式、是否包含偏置、扩张率
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            dilation=dilation,
        )
        # 定义批归一化层，对输出通道数进行归一化处理
        self.bn = nn.BatchNorm2d(out_channels)
        # 定义激活函数层，使用ReLU激活函数
        self.activation = nn.ReLU()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 执行前向传播过程，依次经过卷积层、批归一化层和激活函数层
        output = self.conv(input)
        output = self.bn(output)
        output = self.activation(output)

        return output


# Copied from transformers.models.beit.modeling_beit.BeitPyramidPoolingBlock with Beit->Data2VecVision
class Data2VecVisionPyramidPoolingBlock(nn.Module):
    def __init__(self, pool_scale: int, in_channels: int, channels: int) -> None:
        super().__init__()
        # 定义池化模块，使用自适应平均池化进行特征提取
        # 和卷积模块，通过Data2VecVisionConvModule定义的卷积、归一化和ReLU激活层处理
        self.layers = [
            nn.AdaptiveAvgPool2d(pool_scale),
            Data2VecVisionConvModule(in_channels, channels, kernel_size=1),
        ]
        # 将定义的每一层作为模块添加到当前模块中
        for i, layer in enumerate(self.layers):
            self.add_module(str(i), layer)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        hidden_state = input
        # 执行前向传播过程，依次经过池化层和卷积模块层
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


# Copied from transformers.models.beit.modeling_beit.BeitPyramidPoolingModule with Beit->Data2VecVision
class Data2VecVisionPyramidPoolingModule(nn.Module):
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

    # 空白，等待进一步实现
    pass
    # 初始化函数，设置池化尺度、输入通道数、输出通道数、对齐角点标志
    def __init__(self, pool_scales: Tuple[int, ...], in_channels: int, channels: int, align_corners: bool) -> None:
        # 调用父类的初始化函数
        super().__init__()
        # 将参数赋值给对象的属性
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.blocks = []
        
        # 遍历池化尺度列表，创建数据到向量视觉金字塔池化块
        for i, pool_scale in enumerate(pool_scales):
            block = Data2VecVisionPyramidPoolingBlock(
                pool_scale=pool_scale, in_channels=in_channels, channels=channels
            )
            # 将创建的块添加到块列表中
            self.blocks.append(block)
            # 通过 add_module 方法将块添加为当前模块的子模块，使用索引 i 作为名称
            self.add_module(str(i), block)

    # 前向传播函数，接收输入张量 x，返回列表形式的多个上采样后的池化输出张量
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        ppm_outs = []
        # 遍历每个池化块
        for ppm in self.blocks:
            # 对输入 x 执行当前池化块的前向传播
            ppm_out = ppm(x)
            # 使用双线性插值上采样池化块的输出，保持与输入 x 相同的尺寸
            upsampled_ppm_out = nn.functional.interpolate(
                ppm_out, size=x.size()[2:], mode="bilinear", align_corners=self.align_corners
            )
            # 将上采样后的输出添加到 ppm_outs 列表中
            ppm_outs.append(upsampled_ppm_out)
        # 返回包含所有池化块输出的列表
        return ppm_outs
# 从transformers.models.data2vec.modeling_data2vec.Data2VecVisionUperHead复制而来，将Beit替换为Data2VecVision
class Data2VecVisionUperHead(nn.Module):
    """
    Unified Perceptual Parsing for Scene Understanding. This head is the implementation of
    [UPerNet](https://arxiv.org/abs/1807.10221).

    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    def __init__(self, config: Data2VecVisionConfig) -> None:
        super().__init__()

        self.pool_scales = config.pool_scales  # 例如 (1, 2, 3, 6)，池化尺度列表
        self.in_channels = [config.hidden_size] * 4  # 例如 [768, 768, 768, 768]，输入通道数列表
        self.channels = config.hidden_size  # 隐藏层大小，通常等于输入通道数
        self.align_corners = False  # 是否对齐角落像素
        self.classifier = nn.Conv2d(self.channels, config.num_labels, kernel_size=1)  # 分类器，1x1卷积层

        # PSP模块
        self.psp_modules = Data2VecVisionPyramidPoolingModule(
            self.pool_scales,
            self.in_channels[-1],
            self.channels,
            align_corners=self.align_corners,
        )
        self.bottleneck = Data2VecVisionConvModule(
            self.in_channels[-1] + len(self.pool_scales) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
        )
        # FPN模块
        self.lateral_convs = nn.ModuleList()  # 横向卷积列表
        self.fpn_convs = nn.ModuleList()  # FPN卷积列表
        for in_channels in self.in_channels[:-1]:  # 跳过顶层
            l_conv = Data2VecVisionConvModule(in_channels, self.channels, kernel_size=1)  # 横向卷积层
            fpn_conv = Data2VecVisionConvModule(self.channels, self.channels, kernel_size=3, padding=1)  # FPN卷积层
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = Data2VecVisionConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
        )

    def psp_forward(self, inputs):
        x = inputs[-1]  # 获取输入中的最后一个张量
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))  # 执行PSP模块
        psp_outs = torch.cat(psp_outs, dim=1)  # 在通道维度上拼接输出
        output = self.bottleneck(psp_outs)  # 应用瓶颈层处理

        return output
    # 定义前向传播函数，接收编码器隐藏状态作为输入，返回处理后的张量
    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        # 构建侧向连接
        laterals = [lateral_conv(encoder_hidden_states[i]) for i, lateral_conv in enumerate(self.lateral_convs)]

        # 将 PSP 模块的输出添加到侧向连接中
        laterals.append(self.psp_forward(encoder_hidden_states))

        # 构建自顶向下路径
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + nn.functional.interpolate(
                laterals[i], size=prev_shape, mode="bilinear", align_corners=self.align_corners
            )

        # 构建输出
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)]
        
        # 将 PSP 特征也加入到输出中
        fpn_outs.append(laterals[-1])

        # 对所有层级的输出进行自底向上插值
        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = nn.functional.interpolate(
                fpn_outs[i], size=fpn_outs[0].shape[2:], mode="bilinear", align_corners=self.align_corners
            )

        # 在通道维度上拼接所有的输出
        fpn_outs = torch.cat(fpn_outs, dim=1)
        
        # 使用 FPN 瓶颈网络处理拼接后的特征
        output = self.fpn_bottleneck(fpn_outs)
        
        # 使用分类器对处理后的特征进行分类
        output = self.classifier(output)

        return output
# 从 transformers.models.beit.modeling_beit.BeitFCNHead 复制而来，将 Beit 替换为 Data2VecVision
class Data2VecVisionFCNHead(nn.Module):
    """
    基于 Fully Convolution Networks 的语义分割头部。此头部的实现基于 FCNNet。

    Args:
        config (Data2VecVisionConfig): 配置参数。
        in_channels: 输入通道数。
        kernel_size (int): 头部卷积层的内核大小。默认为 3。
        dilation (int): 头部卷积层的扩张率。默认为 1。

    基于 OpenMMLab 实现，详情请见 https://github.com/open-mmlab/mmsegmentation。
    """

    def __init__(
        self,
        config: Data2VecVisionConfig,
        in_index: int = 2,
        kernel_size: int = 3,
        dilation: Union[int, Tuple[int, int]] = 1,
    ) -> None:
        super().__init__()
        self.in_channels = config.hidden_size  # 输入通道数为配置中的隐藏大小
        self.channels = config.auxiliary_channels  # 辅助通道数为配置中的辅助通道数
        self.num_convs = config.auxiliary_num_convs  # 卷积层数为配置中的卷积层数
        self.concat_input = config.auxiliary_concat_input  # 是否拼接输入为配置中的拼接输入标志
        self.in_index = in_index  # 输入索引为给定的输入索引

        conv_padding = (kernel_size // 2) * dilation  # 计算卷积填充大小
        convs = []
        convs.append(
            Data2VecVisionConvModule(
                self.in_channels, self.channels, kernel_size=kernel_size, padding=conv_padding, dilation=dilation
            )  # 添加第一个卷积模块
        )
        for i in range(self.num_convs - 1):
            convs.append(
                Data2VecVisionConvModule(
                    self.channels, self.channels, kernel_size=kernel_size, padding=conv_padding, dilation=dilation
                )  # 根据配置添加额外的卷积模块
            )
        if self.num_convs == 0:
            self.convs = nn.Identity()  # 如果卷积层数为0，使用恒等映射
        else:
            self.convs = nn.Sequential(*convs)  # 否则创建卷积层的序列模块
        if self.concat_input:
            self.conv_cat = Data2VecVisionConvModule(
                self.in_channels + self.channels, self.channels, kernel_size=kernel_size, padding=kernel_size // 2
            )  # 如果拼接输入为真，创建拼接卷积模块

        self.classifier = nn.Conv2d(self.channels, config.num_labels, kernel_size=1)  # 创建分类器卷积层

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        # 取出相关特征图
        hidden_states = encoder_hidden_states[self.in_index]
        output = self.convs(hidden_states)  # 经过卷积模块
        if self.concat_input:
            output = self.conv_cat(torch.cat([hidden_states, output], dim=1))  # 如果拼接输入，进行特征拼接
        output = self.classifier(output)  # 最终分类器输出
        return output


@add_start_docstrings(
    """
    带有语义分割头部的 Data2VecVision 模型变压器，例如用于 ADE20k、CityScapes 等数据集。
    """,
    DATA2VEC_VISION_START_DOCSTRING,
)
# 从 transformers.models.beit.modeling_beit.BeitForSemanticSegmentation 复制而来，将 BEIT->DATA2VEC_VISION,Beit->Data2VecVision,microsoft/beit-base-finetuned-ade-640-640->facebook/data2vec-vision-base,beit->data2vec_vision
class Data2VecVisionForSemanticSegmentation(Data2VecVisionPreTrainedModel):
    def __init__(self, config: Data2VecVisionConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.data2vec_vision = Data2VecVisionModel(config, add_pooling_layer=False)

        # FPNs
        # 检查 config.out_indices 是否包含了四个整数，若不是则抛出数值错误
        if len(self.config.out_indices) != 4:
            raise ValueError(
                "Data2VecVisionForSemanticSegmentation requires config.out_indices to be a list of 4 integers, "
                "specifying which features to use from the backbone. One can use [3, 5, 7, 11] in case of "
                "a base-sized architecture."
            )
        
        # 创建第一个特征金字塔网络（FPN）
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(config.hidden_size, config.hidden_size, kernel_size=2, stride=2),
            nn.BatchNorm2d(config.hidden_size),
            nn.GELU(),
            nn.ConvTranspose2d(config.hidden_size, config.hidden_size, kernel_size=2, stride=2),
        )
        
        # 创建第二个特征金字塔网络（FPN）
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(config.hidden_size, config.hidden_size, kernel_size=2, stride=2),
        )
        
        # 创建第三个特征金字塔网络（FPN），是一个恒等映射（Identity mapping）
        self.fpn3 = nn.Identity()
        
        # 创建第四个特征金字塔网络（FPN），使用最大池化操作
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Semantic segmentation head(s)
        # 初始化解码头部和辅助头部（如果启用）
        self.decode_head = Data2VecVisionUperHead(config)
        self.auxiliary_head = Data2VecVisionFCNHead(config) if config.use_auxiliary_head else None

        # Initialize weights and apply final processing
        # 执行后续初始化步骤
        self.post_init()

    def compute_loss(self, logits, auxiliary_logits, labels):
        # upsample logits to the images' original size
        # 将 logits 上采样至原始图像大小
        upsampled_logits = nn.functional.interpolate(
            logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )
        if auxiliary_logits is not None:
            upsampled_auxiliary_logits = nn.functional.interpolate(
                auxiliary_logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
        
        # compute weighted loss
        # 计算加权损失
        loss_fct = CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)
        main_loss = loss_fct(upsampled_logits, labels)
        loss = main_loss
        
        if auxiliary_logits is not None:
            auxiliary_loss = loss_fct(upsampled_auxiliary_logits, labels)
            loss += self.config.auxiliary_loss_weight * auxiliary_loss

        return loss

    @add_start_docstrings_to_model_forward(DATA2VEC_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SemanticSegmenterOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
```