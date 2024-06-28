# `.\models\donut\modeling_donut_swin.py`

```py
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch Donut Swin Transformer model.

This implementation is identical to a regular Swin Transformer, without final layer norm on top of the final hidden
states."""

import collections.abc
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from .configuration_donut_swin import DonutSwinConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "DonutSwinConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "https://huggingface.co/naver-clova-ix/donut-base"
_EXPECTED_OUTPUT_SHAPE = [1, 49, 768]

DONUT_SWIN_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "naver-clova-ix/donut-base",
    # See all Donut Swin models at https://huggingface.co/models?filter=donut
]


@dataclass
# 数据类，定义了一个用于存储数据的类
# 从transformers.models.swin.modeling_swin.SwinEncoderOutput复制而来，仅将Swin替换为DonutSwin
class DonutSwinEncoderOutput(ModelOutput):
    """
    DonutSwin encoder's outputs, with potential hidden states and attentions.
    """
    # DonutSwin编码器的输出，可能包含隐藏状态和注意力
    # 最后一层模型的隐藏状态，形状为 `(batch_size, sequence_length, hidden_size)`。
    last_hidden_state: torch.FloatTensor = None
    
    # 模型每层的隐藏状态的元组，如果设置了 `output_hidden_states=True` 或者 `config.output_hidden_states=True` 则返回。
    # 元组中包含 `torch.FloatTensor`，形状为 `(batch_size, sequence_length, hidden_size)`。
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    
    # 注意力权重的元组，如果设置了 `output_attentions=True` 或者 `config.output_attentions=True` 则返回。
    # 元组中包含 `torch.FloatTensor`，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    
    # 模型每层的隐藏状态的元组，如果设置了 `output_hidden_states=True` 或者 `config.output_hidden_states=True` 则返回。
    # 元组中包含 `torch.FloatTensor`，形状为 `(batch_size, hidden_size, height, width)`，表示包括空间维度在内的每层的隐藏状态。
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
# 通过 @dataclass 装饰器定义一个数据类 DonutSwinModelOutput，继承自 ModelOutput
# 从 transformers.models.swin.modeling_swin.SwinModelOutput 复制，将 Swin 替换为 DonutSwin
@dataclass
class DonutSwinModelOutput(ModelOutput):
    """
    DonutSwin model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`, *optional*, returned when `add_pooling_layer=True` is passed):
            Average pooling of the last layer hidden-state.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each stage) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, hidden_size, height, width)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
            include the spatial dimensions.
    """

    # 定义类的成员变量，用于存储模型输出的不同部分
    last_hidden_state: torch.FloatTensor = None
    pooler_output: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None


# 从 transformers.models.swin.modeling_swin.window_partition 复制的函数
def window_partition(input_feature, window_size):
    """
    Partitions the given input into windows.
    """
    # 获取输入特征的形状信息
    batch_size, height, width, num_channels = input_feature.shape
    # 将输入特征按窗口大小分割成小窗口，存储在 input_feature 中
    input_feature = input_feature.view(
        batch_size, height // window_size, window_size, width // window_size, window_size, num_channels
    )
    # 调整分割后的窗口顺序，并重新整理为一个扁平化的张量
    windows = input_feature.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, num_channels)
    # 返回分割后的窗口张量
    return windows


# 从 transformers.models.swin.modeling_swin.window_reverse 复制的函数
# 定义一个函数window_reverse，用于合并窗口以生成更高分辨率的特征
def window_reverse(windows, window_size, height, width):
    # 获取窗口的通道数
    num_channels = windows.shape[-1]
    # 重塑窗口张量的形状，以便进行后续操作
    windows = windows.view(-1, height // window_size, width // window_size, window_size, window_size, num_channels)
    # 调整张量的维度顺序，以便后续操作的连续性
    windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, height, width, num_channels)
    # 返回处理后的窗口张量
    return windows


# 从transformers.models.swin.modeling_swin.SwinEmbeddings复制并修改为DonutSwinEmbeddings
class DonutSwinEmbeddings(nn.Module):
    """
    构建补丁和位置嵌入。可选择添加掩码令牌。
    """

    def __init__(self, config, use_mask_token=False):
        super().__init__()

        # 初始化补丁嵌入
        self.patch_embeddings = DonutSwinPatchEmbeddings(config)
        # 获取补丁数量和网格大小
        num_patches = self.patch_embeddings.num_patches
        self.patch_grid = self.patch_embeddings.grid_size
        # 如果需要使用掩码令牌，则初始化掩码令牌
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim)) if use_mask_token else None

        # 根据配置决定是否使用绝对位置嵌入
        if config.use_absolute_embeddings:
            self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.embed_dim))
        else:
            self.position_embeddings = None

        # 初始化LayerNorm和Dropout
        self.norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, pixel_values: Optional[torch.FloatTensor], bool_masked_pos: Optional[torch.BoolTensor] = None
    ) -> Tuple[torch.Tensor]:
        # 获取补丁嵌入和输出维度
        embeddings, output_dimensions = self.patch_embeddings(pixel_values)
        # 对嵌入进行LayerNorm
        embeddings = self.norm(embeddings)
        # 获取批量大小、序列长度和嵌入维度
        batch_size, seq_len, _ = embeddings.size()

        # 如果存在掩码位置信息，则用掩码令牌替换掩码的视觉令牌
        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            # 将掩码应用到嵌入张量中
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # 如果存在位置嵌入，则添加到嵌入张量中
        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings

        # 应用Dropout到嵌入张量
        embeddings = self.dropout(embeddings)

        # 返回处理后的嵌入张量和输出维度
        return embeddings, output_dimensions


# 从transformers.models.swin.modeling_swin.SwinPatchEmbeddings复制并修改为DonutSwinPatchEmbeddings
class DonutSwinPatchEmbeddings(nn.Module):
    """
    将形状为(batch_size, num_channels, height, width)的像素值转换为Transformer可消耗的初始隐藏状态（补丁嵌入），
    形状为(batch_size, seq_length, hidden_size)。
    """
    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 从配置对象中获取图像大小和patch大小
        image_size, patch_size = config.image_size, config.patch_size
        # 从配置对象中获取通道数和嵌入维度大小
        num_channels, hidden_size = config.num_channels, config.embed_dim
        # 如果图像大小和patch大小不是可迭代对象，则转换为元组
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        # 计算图像中patch的数量
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        # 设置对象的属性
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        # 计算图像网格大小
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])

        # 创建一个卷积层，用于将输入的通道转换为隐藏维度的输出
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    # 对输入的像素值进行可能的填充，使其能够被patch大小整除
    def maybe_pad(self, pixel_values, height, width):
        # 如果宽度不能被patch的宽度整除，则进行填充
        if width % self.patch_size[1] != 0:
            pad_values = (0, self.patch_size[1] - width % self.patch_size[1])
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        # 如果高度不能被patch的高度整除，则进行填充
        if height % self.patch_size[0] != 0:
            pad_values = (0, 0, 0, self.patch_size[0] - height % self.patch_size[0])
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        # 返回填充后的像素值
        return pixel_values

    # 前向传播函数，接受一个可选的torch.FloatTensor类型的像素值作为输入，返回嵌入向量和输出维度
    def forward(self, pixel_values: Optional[torch.FloatTensor]) -> Tuple[torch.Tensor, Tuple[int]]:
        # 获取输入张量的形状信息
        _, num_channels, height, width = pixel_values.shape
        # 如果通道数不等于设定的通道数，则抛出值错误异常
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # 将输入进行填充，使其能够被patch大小整除
        pixel_values = self.maybe_pad(pixel_values, height, width)
        # 将填充后的像素值通过投影卷积层转换为嵌入向量
        embeddings = self.projection(pixel_values)
        # 获取嵌入向量的形状信息
        _, _, height, width = embeddings.shape
        # 计算输出的维度信息
        output_dimensions = (height, width)
        # 将嵌入向量展平，并在特定维度上转置
        embeddings = embeddings.flatten(2).transpose(1, 2)

        # 返回嵌入向量和输出维度
        return embeddings, output_dimensions
# Copied from transformers.models.swin.modeling_swin.SwinPatchMerging
class DonutSwinPatchMerging(nn.Module):
    """
    Patch Merging Layer.

    Args:
        input_resolution (`Tuple[int]`):
            Resolution of input feature.
        dim (`int`):
            Number of input channels.
        norm_layer (`nn.Module`, *optional*, defaults to `nn.LayerNorm`):
            Normalization layer class.
    """

    def __init__(self, input_resolution: Tuple[int], dim: int, norm_layer: nn.Module = nn.LayerNorm) -> None:
        super().__init__()
        # 初始化 Patch Merging 层，保存输入分辨率和通道数
        self.input_resolution = input_resolution
        self.dim = dim
        # 创建线性层，用于降维操作，从4倍的通道数到2倍的通道数，无偏置
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        # 初始化归一化层，对4倍的通道数进行归一化处理
        self.norm = norm_layer(4 * dim)

    def maybe_pad(self, input_feature, height, width):
        # 判断是否需要对输入特征进行填充，使得高度和宽度均为偶数
        should_pad = (height % 2 == 1) or (width % 2 == 1)
        if should_pad:
            pad_values = (0, 0, 0, width % 2, 0, height % 2)
            input_feature = nn.functional.pad(input_feature, pad_values)

        return input_feature

    def forward(self, input_feature: torch.Tensor, input_dimensions: Tuple[int, int]) -> torch.Tensor:
        height, width = input_dimensions
        # 获取输入特征的批大小、通道数以及特征图的数量
        batch_size, dim, num_channels = input_feature.shape

        # 将输入特征重塑为四维张量 [batch_size, height, width, num_channels]
        input_feature = input_feature.view(batch_size, height, width, num_channels)
        # 如果需要，对输入特征进行填充，使得高度和宽度均为偶数
        input_feature = self.maybe_pad(input_feature, height, width)

        # 下采样操作，将特征图划分成四个区域，分别对应输入特征的四分之一大小
        input_feature_0 = input_feature[:, 0::2, 0::2, :]
        input_feature_1 = input_feature[:, 1::2, 0::2, :]
        input_feature_2 = input_feature[:, 0::2, 1::2, :]
        input_feature_3 = input_feature[:, 1::2, 1::2, :]
        
        # 将四个区域的特征按通道拼接起来，形成新的特征图
        input_feature = torch.cat([input_feature_0, input_feature_1, input_feature_2, input_feature_3], -1)
        # 将新的特征图重塑为三维张量 [batch_size, height/2*width/2, 4*num_channels]
        input_feature = input_feature.view(batch_size, -1, 4 * num_channels)

        # 对新的特征图进行归一化处理
        input_feature = self.norm(input_feature)
        # 使用线性层进行降维操作，输出特征维度为 [batch_size, height/2*width/2, 2*dim]
        input_feature = self.reduction(input_feature)

        return input_feature


# Copied from transformers.models.beit.modeling_beit.drop_path
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    """

    # 实现样本级别的路径丢弃（随机深度），在残差块的主路径中应用
    # 输入参数包括：input - 输入张量，drop_prob - 丢弃概率（默认为0.0），training - 是否处于训练模式（默认为False）
    # 详细讨论可以参考 Ross Wightman 的评论和链接中的讨论
    # 如果 drop_prob 为 0.0 或者不处于训练模式，则直接返回输入
    if drop_prob == 0.0 or not training:
        return input
    # 计算保留的概率
    keep_prob = 1 - drop_prob
    # 确定输出张量的形状，适用于不同维度的张量，不仅限于二维卷积网络
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)
    # 生成与输入张量相同设备和数据类型的随机张量，值在 [keep_prob, 1.0) 之间
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    # 将随机张量向下取整，将其二值化
    random_tensor.floor_()
    # 计算输出，将输入张量除以保留概率，再乘以随机张量
    output = input.div(keep_prob) * random_tensor
    # 返回处理后的输出张量
    return output
# 从transformers.models.swin.modeling_swin.SwinDropPath复制而来，定义了DonutSwinDropPath类，用于实现每个样本的Drop Path（随机深度）机制。
class DonutSwinDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob  # 初始化Drop Path的概率

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)  # 执行Drop Path操作

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)  # 返回Drop Path概率的描述字符串


# 从transformers.models.swin.modeling_swin.SwinSelfAttention复制而来，定义了DonutSwinSelfAttention类，实现Swin Transformer的自注意力机制，被修改为适应新的Donut模型。
class DonutSwinSelfAttention(nn.Module):
    def __init__(self, config, dim, num_heads, window_size):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"The hidden size ({dim}) is not a multiple of the number of attention heads ({num_heads})"
            )

        self.num_attention_heads = num_heads  # 注意力头的数量
        self.attention_head_size = int(dim / num_heads)  # 每个注意力头的大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 所有注意力头的总大小
        self.window_size = (
            window_size if isinstance(window_size, collections.abc.Iterable) else (window_size, window_size)
        )  # 窗口大小，用于相对位置编码

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads)
        )  # 相对位置偏置表格，用作注意力矩阵的偏置

        # 获取窗口内每个标记的成对相对位置索引
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(meshgrid([coords_h, coords_w], indexing="ij"))  # 构建网格坐标
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 计算成对相对坐标
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)  # 注册成对相对位置索引为模型的缓冲区

        self.query = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)  # 查询变换器
        self.key = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)  # 键变换器
        self.value = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)  # 值变换器

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)  # 注意力概率的dropout机制

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)  # 转置矩阵，以适应多头注意力的计算
    # 定义前向传播方法，用于处理输入隐藏状态和注意力掩码等参数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 获取输入隐藏状态的维度信息
        batch_size, dim, num_channels = hidden_states.shape
        # 通过 self.query 对隐藏状态进行查询操作，生成混合的查询层
        mixed_query_layer = self.query(hidden_states)

        # 使用 self.key 对隐藏状态进行键操作，并转换维度以便进行注意力计算
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        # 使用 self.value 对隐藏状态进行值操作，并转换维度以便进行注意力计算
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        # 对混合的查询层进行维度转换以便进行注意力计算
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 计算 "查询" 和 "键" 之间的点积，得到原始的注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # 将注意力分数除以 sqrt(注意力头的大小)，以归一化
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 获取相对位置偏置，并按照特定方式重塑其形状以便加到注意力分数上
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attention_scores = attention_scores + relative_position_bias.unsqueeze(0)

        # 如果提供了注意力掩码，则应用掩码
        if attention_mask is not None:
            # 将注意力掩码重塑为适合注意力分数张量的形状
            mask_shape = attention_mask.shape[0]
            attention_scores = attention_scores.view(
                batch_size // mask_shape, mask_shape, self.num_attention_heads, dim, dim
            )
            attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(0)
            attention_scores = attention_scores.view(-1, self.num_attention_heads, dim, dim)

        # 对注意力分数进行 softmax 归一化，得到注意力概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 使用 dropout 随机丢弃整个令牌的注意力概率
        attention_probs = self.dropout(attention_probs)

        # 如果提供了头部掩码，则应用头部掩码
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算上下文层，将注意力概率乘以值层，并重塑其形状
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # 根据输出标志决定返回的输出，可能包括注意力概率
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
# Copied from transformers.models.swin.modeling_swin.SwinSelfOutput
class DonutSwinSelfOutput(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        # 初始化一个线性层，输入和输出维度均为 dim
        self.dense = nn.Linear(dim, dim)
        # 初始化一个 dropout 层，使用 config 中的 dropout 概率
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将输入的 hidden_states 通过线性层 self.dense 进行变换
        hidden_states = self.dense(hidden_states)
        # 对变换后的 hidden_states 使用 dropout 进行随机置零
        hidden_states = self.dropout(hidden_states)
        # 返回处理后的 hidden_states
        return hidden_states


# Copied from transformers.models.swin.modeling_swin.SwinAttention with Swin->DonutSwin
class DonutSwinAttention(nn.Module):
    def __init__(self, config, dim, num_heads, window_size):
        super().__init__()
        # 初始化 DonutSwinSelfAttention 层
        self.self = DonutSwinSelfAttention(config, dim, num_heads, window_size)
        # 初始化 DonutSwinSelfOutput 层
        self.output = DonutSwinSelfOutput(config, dim)
        # 初始化一个空集合，用于存储被修剪的注意力头
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 根据给定的 heads 进行注意力头修剪，并返回修剪后的 heads 和索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 对线性层进行修剪
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储被修剪的注意力头
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 调用 self 层进行自注意力计算
        self_outputs = self.self(hidden_states, attention_mask, head_mask, output_attentions)
        # 将 self 层的输出经过 output 层处理
        attention_output = self.output(self_outputs[0], hidden_states)
        # 将输出打包成元组，如果需要输出注意力权重，则加入输出中
        outputs = (attention_output,) + self_outputs[1:]
        # 返回处理后的输出
        return outputs


# Copied from transformers.models.swin.modeling_swin.SwinIntermediate
class DonutSwinIntermediate(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        # 初始化一个线性层，输入维度为 dim，输出维度为 config.mlp_ratio * dim
        self.dense = nn.Linear(dim, int(config.mlp_ratio * dim))
        # 根据 config 中的 hidden_act 初始化激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入的 hidden_states 通过线性层 self.dense 进行变换
        hidden_states = self.dense(hidden_states)
        # 使用预定义的激活函数对 hidden_states 进行非线性变换
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回处理后的 hidden_states
        return hidden_states


# Copied from transformers.models.swin.modeling_swin.SwinOutput
# 此处代码被省略，需要在此处补充相关注释
# 定义名为 DonutSwinOutput 的神经网络模块，继承自 nn.Module 类
class DonutSwinOutput(nn.Module):
    # 初始化函数，接收 config 和 dim 两个参数
    def __init__(self, config, dim):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个线性层，输入大小为 config.mlp_ratio * dim，输出大小为 dim
        self.dense = nn.Linear(int(config.mlp_ratio * dim), dim)
        # 创建一个 Dropout 层，使用配置中的 hidden_dropout_prob 参数
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数，接收输入 hidden_states，返回经过处理的张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入 hidden_states 通过 self.dense 线性层处理
        hidden_states = self.dense(hidden_states)
        # 对处理后的张量应用 dropout 操作
        hidden_states = self.dropout(hidden_states)
        # 返回处理后的张量作为输出
        return hidden_states


# 从 transformers.models.swin.modeling_swin.SwinLayer 复制代码，将 Swin 改为 DonutSwin
class DonutSwinLayer(nn.Module):
    # 初始化函数，接收 config、dim、input_resolution、num_heads 和可选的 shift_size 参数
    def __init__(self, config, dim, input_resolution, num_heads, shift_size=0):
        # 调用父类的初始化函数
        super().__init__()
        # 设置分块大小为 config.chunk_size_feed_forward
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 设置 shift_size
        self.shift_size = shift_size
        # 设置窗口大小为 config.window_size
        self.window_size = config.window_size
        # 设置输入分辨率
        self.input_resolution = input_resolution
        # 在层归一化之前应用层归一化，设置归一化的 epsilon 为 config.layer_norm_eps
        self.layernorm_before = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        # 创建自注意力机制（Attention）层 DonutSwinAttention 对象
        self.attention = DonutSwinAttention(config, dim, num_heads, window_size=self.window_size)
        # 如果 drop_path_rate 大于 0.0，则创建 DropPath 层 DonutSwinDropPath 对象，否则创建 Identity 层
        self.drop_path = DonutSwinDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
        # 在层归一化之后应用层归一化，设置归一化的 epsilon 为 config.layer_norm_eps
        self.layernorm_after = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        # 创建中间层 DonutSwinIntermediate 对象
        self.intermediate = DonutSwinIntermediate(config, dim)
        # 创建输出层 DonutSwinOutput 对象
        self.output = DonutSwinOutput(config, dim)

    # 设置 shift_size 和 window_size 的函数，根据输入分辨率 input_resolution 进行调整
    def set_shift_and_window_size(self, input_resolution):
        # 如果输入分辨率中最小的尺寸小于等于窗口大小 window_size，则不分割窗口
        if min(input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(input_resolution)

    # 根据给定的高度和宽度生成注意力掩码的函数
    def get_attn_mask(self, height, width, dtype):
        # 如果 shift_size 大于 0，则计算 SW-MSA 的注意力掩码
        if self.shift_size > 0:
            # 创建一个高度为 1，宽度为 height 和 width，通道数为 1 的零张量 img_mask
            img_mask = torch.zeros((1, height, width, 1), dtype=dtype)
            # 定义高度和宽度的切片
            height_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            width_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            count = 0
            # 遍历高度和宽度切片，并在 img_mask 上进行相应的标记
            for height_slice in height_slices:
                for width_slice in width_slices:
                    img_mask[:, height_slice, width_slice, :] = count
                    count += 1

            # 将 img_mask 分割为窗口并展平成二维张量
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            # 构建注意力掩码，使对角线上的元素为 0，其他位置为 -100.0
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        # 返回生成的注意力掩码
        return attn_mask
    # 在可能的情况下，对隐藏状态进行填充，以保证其尺寸能够被窗口大小整除
    def maybe_pad(self, hidden_states, height, width):
        # 计算右边和底部需要填充的像素数，确保能够被窗口大小整除
        pad_right = (self.window_size - width % self.window_size) % self.window_size
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size
        # 定义填充值的元组：(前, 后, 上, 右, 下, 左)，这里只在右边和底部进行填充
        pad_values = (0, 0, 0, pad_right, 0, pad_bottom)
        # 对隐藏状态进行填充操作
        hidden_states = nn.functional.pad(hidden_states, pad_values)
        # 返回填充后的隐藏状态和填充值的元组
        return hidden_states, pad_values

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        always_partition: Optional[bool] = False,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 如果不总是分区，设置偏移量和窗口大小
        if not always_partition:
            self.set_shift_and_window_size(input_dimensions)
        else:
            # 否则，什么也不做
            pass
        # 获取输入维度的高度和宽度
        height, width = input_dimensions
        # 获取隐藏状态的批量大小、通道数和维度
        batch_size, _, channels = hidden_states.size()
        # 备份隐藏状态
        shortcut = hidden_states

        # 在层归一化之前应用层归一化
        hidden_states = self.layernorm_before(hidden_states)

        # 将隐藏状态重塑为四维张量 (batch_size, height, width, channels)
        hidden_states = hidden_states.view(batch_size, height, width, channels)

        # 使用 maybe_pad 方法对隐藏状态进行填充，使其大小为窗口大小的倍数
        hidden_states, pad_values = self.maybe_pad(hidden_states, height, width)

        # 获取填充后的高度和宽度
        _, height_pad, width_pad, _ = hidden_states.shape

        # 如果有循环偏移量，将隐藏状态进行循环移位
        if self.shift_size > 0:
            shifted_hidden_states = torch.roll(hidden_states, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_hidden_states = hidden_states

        # 分区窗口，将移位后的隐藏状态分割成窗口
        hidden_states_windows = window_partition(shifted_hidden_states, self.window_size)
        hidden_states_windows = hidden_states_windows.view(-1, self.window_size * self.window_size, channels)

        # 获取注意力遮罩，以排除填充区域的影响
        attn_mask = self.get_attn_mask(height_pad, width_pad, dtype=hidden_states.dtype)
        if attn_mask is not None:
            attn_mask = attn_mask.to(hidden_states_windows.device)

        # 使用注意力机制处理窗口
        attention_outputs = self.attention(
            hidden_states_windows, attn_mask, head_mask, output_attentions=output_attentions
        )

        # 获取注意力机制的输出
        attention_output = attention_outputs[0]

        # 将注意力输出重塑为四维张量 (batch_size, height, width, channels)
        attention_windows = attention_output.view(-1, self.window_size, self.window_size, channels)

        # 反向操作，将窗口重排成原始形状
        shifted_windows = window_reverse(attention_windows, self.window_size, height_pad, width_pad)

        # 如果有循环偏移量，对注意力窗口进行反向循环移位
        if self.shift_size > 0:
            attention_windows = torch.roll(shifted_windows, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attention_windows = shifted_windows

        # 如果存在填充，则裁剪注意力窗口以匹配原始图像尺寸
        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
            attention_windows = attention_windows[:, :height, :width, :].contiguous()

        # 将注意力窗口重塑为三维张量 (batch_size, height*width, channels)
        attention_windows = attention_windows.view(batch_size, height * width, channels)

        # 将快捷连接和注意力窗口加和，并应用 drop_path
        hidden_states = shortcut + self.drop_path(attention_windows)

        # 在层归一化之后应用层归一化
        layer_output = self.layernorm_after(hidden_states)

        # 应用中间层
        layer_output = self.intermediate(layer_output)

        # 应用输出层
        layer_output = hidden_states + self.output(layer_output)

        # 返回层输出，如果需要输出注意力权重则包含在内
        layer_outputs = (layer_output, attention_outputs[1]) if output_attentions else (layer_output,)
        return layer_outputs
# 从 transformers.models.swin.modeling_swin.SwinStage 复制而来，将 Swin 替换为 DonutSwin
class DonutSwinStage(nn.Module):
    def __init__(self, config, dim, input_resolution, depth, num_heads, drop_path, downsample):
        super().__init__()
        self.config = config
        self.dim = dim
        # 创建包含多个 DonutSwinLayer 的模块列表，根据给定的深度
        self.blocks = nn.ModuleList(
            [
                DonutSwinLayer(
                    config=config,
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    # 根据奇偶性确定 shift_size 的值
                    shift_size=0 if (i % 2 == 0) else config.window_size // 2,
                )
                for i in range(depth)
            ]
        )

        # 如果有 downsample 参数，创建 patch merging 层；否则为 None
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=nn.LayerNorm)
        else:
            self.downsample = None

        self.pointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        always_partition: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        height, width = input_dimensions
        # 遍历每个 DonutSwinLayer 模块进行前向传播计算
        for i, layer_module in enumerate(self.blocks):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 调用当前层的前向传播方法，得到该层的输出
            layer_outputs = layer_module(
                hidden_states, input_dimensions, layer_head_mask, output_attentions, always_partition
            )

            # 更新隐藏状态为当前层的输出
            hidden_states = layer_outputs[0]

        hidden_states_before_downsampling = hidden_states
        # 如果存在 downsample 层，对隐藏状态进行下采样处理
        if self.downsample is not None:
            height_downsampled, width_downsampled = (height + 1) // 2, (width + 1) // 2
            output_dimensions = (height, width, height_downsampled, width_downsampled)
            hidden_states = self.downsample(hidden_states_before_downsampling, input_dimensions)
        else:
            output_dimensions = (height, width, height, width)

        # 生成 stage 的输出，包括隐藏状态、下采样前的隐藏状态和输出维度信息
        stage_outputs = (hidden_states, hidden_states_before_downsampling, output_dimensions)

        # 如果开启了输出注意力机制信息的选项，则将每个层的注意力信息添加到输出中
        if output_attentions:
            stage_outputs += layer_outputs[1:]
        return stage_outputs


# 从 transformers.models.swin.modeling_swin.SwinEncoder 复制而来，将 Swin 替换为 DonutSwin
class DonutSwinEncoder(nn.Module):
    # 初始化函数，用于初始化一个 DonutSwin 模型实例
    def __init__(self, config, grid_size):
        # 调用父类（nn.Module）的初始化方法
        super().__init__()
        # 计算模型层数
        self.num_layers = len(config.depths)
        # 保存配置对象
        self.config = config
        # 计算每层的 drop path rate
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]
        # 创建模型的层列表，每层是一个 DonutSwinStage 实例
        self.layers = nn.ModuleList(
            [
                DonutSwinStage(
                    config=config,
                    # 设置每层的输入维度
                    dim=int(config.embed_dim * 2**i_layer),
                    # 设置输入分辨率
                    input_resolution=(grid_size[0] // (2**i_layer), grid_size[1] // (2**i_layer)),
                    # 设置层的深度（即重复次数）
                    depth=config.depths[i_layer],
                    # 设置注意力头的数量
                    num_heads=config.num_heads[i_layer],
                    # 设置当前层的 drop path rates
                    drop_path=dpr[sum(config.depths[:i_layer]) : sum(config.depths[: i_layer + 1])],
                    # 如果不是最后一层，则使用 DonutSwinPatchMerging 进行下采样
                    downsample=DonutSwinPatchMerging if (i_layer < self.num_layers - 1) else None,
                )
                # 循环创建每一层的实例
                for i_layer in range(self.num_layers)
            ]
        )

        # 是否使用梯度检查点，默认为 False
        self.gradient_checkpointing = False

    # 前向传播函数，计算模型的输出
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        output_hidden_states_before_downsampling: Optional[bool] = False,
        always_partition: Optional[bool] = False,
        return_dict: Optional[bool] = True,
# 从 transformers.models.swin.modeling_swin.SwinPreTrainedModel 复制并修改为 DonutSwinPreTrainedModel 类
class DonutSwinPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定配置类为 DonutSwinConfig
    config_class = DonutSwinConfig
    # 基础模型的前缀为 "swin"
    base_model_prefix = "swin"
    # 主输入名称为 "pixel_values"
    main_input_name = "pixel_values"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 对线性层和卷积层使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                # 如果有偏置，则将偏置初始化为零
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            # 对 LayerNorm 层，初始化偏置为零，权重为1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


SWIN_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`DonutSwinConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

SWIN_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`DonutImageProcessor.__call__`] for details.
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
    "The bare Donut Swin Model transformer outputting raw hidden-states without any specific head on top.",
    SWIN_START_DOCSTRING,
)
# 定义 DonutSwinModel 类，继承自 DonutSwinPreTrainedModel
class DonutSwinModel(DonutSwinPreTrainedModel):
    pass  # 这里省略了具体实现，仅作为示例展示类的定义和继承关系
    def __init__(self, config, add_pooling_layer=True, use_mask_token=False):
        # 调用父类的初始化方法，传入配置信息
        super().__init__(config)
        # 将配置信息保存到实例变量中
        self.config = config
        # 计算编码器层数量
        self.num_layers = len(config.depths)
        # 计算特征数量
        self.num_features = int(config.embed_dim * 2 ** (self.num_layers - 1))

        # 初始化嵌入层对象
        self.embeddings = DonutSwinEmbeddings(config, use_mask_token=use_mask_token)
        # 初始化编码器对象，传入嵌入层的补丁网格
        self.encoder = DonutSwinEncoder(config, self.embeddings.patch_grid)

        # 如果需要添加池化层，则初始化自适应平均池化层，否则设为 None
        self.pooler = nn.AdaptiveAvgPool1d(1) if add_pooling_layer else None

        # 调用初始化权重和应用最终处理的方法
        self.post_init()

    def get_input_embeddings(self):
        # 返回嵌入层的补丁嵌入
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 遍历需要剪枝的层和对应的注意力头信息
        for layer, heads in heads_to_prune.items():
            # 调用编码器对象中相应层的注意力模块的剪枝方法
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(SWIN_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=DonutSwinModelOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 此处应该包含模型前向传播的详细说明文档和示例代码，但在注释中无法展示具体内容
        pass
        ) -> Union[Tuple, DonutSwinModelOutput]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        # 设置输出是否包含注意力权重，默认与模型配置一致
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置输出是否包含隐藏状态，默认与模型配置一致
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置返回对象类型，默认与模型配置一致
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果未提供像素值，则抛出数值错误
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 准备头部掩码（如果需要）
        # 头部掩码中的 1 表示保留该头部
        # attention_probs 的形状为 bsz x n_heads x N x N
        # 输入的 head_mask 形状为 [num_heads] 或 [num_hidden_layers x num_heads]
        # 将 head_mask 转换为形状 [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, len(self.config.depths))

        # 将像素值和布尔掩码位置作为输入，传递给嵌入层获取嵌入输出和输入维度
        embedding_output, input_dimensions = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)

        # 将嵌入输出传递给编码器，获取编码器的输出
        encoder_outputs = self.encoder(
            embedding_output,
            input_dimensions,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取编码器的序列输出
        sequence_output = encoder_outputs[0]

        # 如果存在池化器，则对序列输出进行池化和扁平化处理
        pooled_output = None
        if self.pooler is not None:
            pooled_output = self.pooler(sequence_output.transpose(1, 2))
            pooled_output = torch.flatten(pooled_output, 1)

        # 如果不返回字典，则返回元组格式的输出
        if not return_dict:
            output = (sequence_output, pooled_output) + encoder_outputs[1:]
            return output

        # 如果返回字典，则返回特定的模型输出对象
        return DonutSwinModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            reshaped_hidden_states=encoder_outputs.reshaped_hidden_states,
        )
```