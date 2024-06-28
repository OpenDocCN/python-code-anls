# `.\models\cvt\modeling_cvt.py`

```py
# coding=utf-8
# Copyright 2022 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch CvT model."""

import collections.abc
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_outputs import ImageClassifierOutputWithNoAttention, ModelOutput
from ...modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import logging
from .configuration_cvt import CvtConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "CvtConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "microsoft/cvt-13"
_EXPECTED_OUTPUT_SHAPE = [1, 384, 14, 14]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "microsoft/cvt-13"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

# CvT预训练模型存档列表
CVT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/cvt-13",
    "microsoft/cvt-13-384",
    "microsoft/cvt-13-384-22k",
    "microsoft/cvt-21",
    "microsoft/cvt-21-384",
    "microsoft/cvt-21-384-22k",
    # See all Cvt models at https://huggingface.co/models?filter=cvt
]

@dataclass
class BaseModelOutputWithCLSToken(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        cls_token_value (`torch.FloatTensor` of shape `(batch_size, 1, hidden_size)`):
            Classification token at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
    """

    last_hidden_state: torch.FloatTensor = None
    cls_token_value: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
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


# Copied from transformers.models.beit.modeling_beit.BeitDropPath
class CvtDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class CvtEmbeddings(nn.Module):
    """
    Construct the CvT embeddings.
    """

    def __init__(self, patch_size, num_channels, embed_dim, stride, padding, dropout_rate):
        super().__init__()
        self.convolution_embeddings = CvtConvEmbeddings(
            patch_size=patch_size, num_channels=num_channels, embed_dim=embed_dim, stride=stride, padding=padding
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, pixel_values):
        hidden_state = self.convolution_embeddings(pixel_values)
        hidden_state = self.dropout(hidden_state)
        return hidden_state


class CvtConvEmbeddings(nn.Module):
    """
    Image to Conv Embedding.
    """

    def __init__(self, patch_size, num_channels, embed_dim, stride, padding):
        super().__init__()
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        self.patch_size = patch_size
        self.projection = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.normalization = nn.LayerNorm(embed_dim)
    # 定义前向传播函数，接受像素值作为输入
    def forward(self, pixel_values):
        # 使用投影函数对像素值进行处理
        pixel_values = self.projection(pixel_values)
        # 获取输入张量的维度信息：批大小，通道数，高度，宽度
        batch_size, num_channels, height, width = pixel_values.shape
        # 计算隐藏层大小
        hidden_size = height * width
        # 将输入张量重新排列为 "批大小 (高度 * 宽度) 通道数"
        pixel_values = pixel_values.view(batch_size, num_channels, hidden_size).permute(0, 2, 1)
        # 如果启用了归一化函数，则对像素值进行归一化处理
        if self.normalization:
            pixel_values = self.normalization(pixel_values)
        # 将张量重新排列为 "批大小 通道数 高度 宽度"
        pixel_values = pixel_values.permute(0, 2, 1).view(batch_size, num_channels, height, width)
        # 返回处理后的像素值张量
        return pixel_values
# 定义自注意力模块的卷积投影类
class CvtSelfAttentionConvProjection(nn.Module):
    def __init__(self, embed_dim, kernel_size, padding, stride):
        super().__init__()
        # 创建一个二维卷积层对象
        self.convolution = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=False,
            groups=embed_dim,  # 设定卷积的分组数量为 embed_dim，用于深度可分离卷积
        )
        # 创建二维批归一化层对象
        self.normalization = nn.BatchNorm2d(embed_dim)

    def forward(self, hidden_state):
        # 对输入的隐藏状态进行卷积操作
        hidden_state = self.convolution(hidden_state)
        # 对卷积后的结果进行批归一化处理
        hidden_state = self.normalization(hidden_state)
        return hidden_state


# 定义自注意力模块的线性投影类
class CvtSelfAttentionLinearProjection(nn.Module):
    def forward(self, hidden_state):
        # 获取输入隐藏状态的维度信息
        batch_size, num_channels, height, width = hidden_state.shape
        # 计算隐藏状态的大小
        hidden_size = height * width
        # 重新排列张量的维度顺序，转换为 "b (h w) c" 的形式
        hidden_state = hidden_state.view(batch_size, num_channels, hidden_size).permute(0, 2, 1)
        return hidden_state


# 定义自注意力模块的投影类
class CvtSelfAttentionProjection(nn.Module):
    def __init__(self, embed_dim, kernel_size, padding, stride, projection_method="dw_bn"):
        super().__init__()
        # 根据投影方法选择不同的投影方式
        if projection_method == "dw_bn":
            # 使用深度可分离卷积进行投影
            self.convolution_projection = CvtSelfAttentionConvProjection(embed_dim, kernel_size, padding, stride)
        # 创建线性投影对象
        self.linear_projection = CvtSelfAttentionLinearProjection()

    def forward(self, hidden_state):
        # 使用卷积投影对隐藏状态进行处理
        hidden_state = self.convolution_projection(hidden_state)
        # 使用线性投影对卷积后的结果进行进一步处理
        hidden_state = self.linear_projection(hidden_state)
        return hidden_state


# 定义自注意力模块的主类
class CvtSelfAttention(nn.Module):
    def __init__(
        self,
        num_heads,
        embed_dim,
        kernel_size,
        padding_q,
        padding_kv,
        stride_q,
        stride_kv,
        qkv_projection_method,
        qkv_bias,
        attention_drop_rate,
        with_cls_token=True,
        **kwargs,
    ):
        # 在这里进行初始化，省略了部分参数
        pass  # 实际初始化内容可以根据需要添加
        super().__init__()
        # 调用父类的初始化方法

        self.scale = embed_dim**-0.5
        # 初始化缩放因子，用于缩放注意力分数

        self.with_cls_token = with_cls_token
        # 是否包含类别标记的标志

        self.embed_dim = embed_dim
        # 嵌入维度大小

        self.num_heads = num_heads
        # 注意力头的数量

        self.convolution_projection_query = CvtSelfAttentionProjection(
            embed_dim,
            kernel_size,
            padding_q,
            stride_q,
            projection_method="linear" if qkv_projection_method == "avg" else qkv_projection_method,
        )
        # 创建用于查询的卷积投影对象，根据指定的投影方法

        self.convolution_projection_key = CvtSelfAttentionProjection(
            embed_dim, kernel_size, padding_kv, stride_kv, projection_method=qkv_projection_method
        )
        # 创建用于键的卷积投影对象，根据指定的投影方法

        self.convolution_projection_value = CvtSelfAttentionProjection(
            embed_dim, kernel_size, padding_kv, stride_kv, projection_method=qkv_projection_method
        )
        # 创建用于值的卷积投影对象，根据指定的投影方法

        self.projection_query = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        # 创建查询的线性投影层

        self.projection_key = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        # 创建键的线性投影层

        self.projection_value = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        # 创建值的线性投影层

        self.dropout = nn.Dropout(attention_drop_rate)
        # 创建用于注意力掩码的dropout层

    def rearrange_for_multi_head_attention(self, hidden_state):
        batch_size, hidden_size, _ = hidden_state.shape
        head_dim = self.embed_dim // self.num_heads
        # 计算每个注意力头的维度

        # 重新排列张量以用于多头注意力计算，形式为 'b t (h d) -> b h t d'
        return hidden_state.view(batch_size, hidden_size, self.num_heads, head_dim).permute(0, 2, 1, 3)
        # 返回重新排列后的张量，以便进行多头注意力计算
    # 定义一个前向传播方法，接受隐藏状态、高度和宽度作为参数
    def forward(self, hidden_state, height, width):
        # 如果设置了包含CLS token，则将隐藏状态分割为CLS token和其余部分
        if self.with_cls_token:
            cls_token, hidden_state = torch.split(hidden_state, [1, height * width], 1)
        
        # 获取批大小、隐藏大小和通道数
        batch_size, hidden_size, num_channels = hidden_state.shape
        
        # 重新排列隐藏状态的维度以适应多头注意力机制的需求："b (h w) c -> b c h w"
        hidden_state = hidden_state.permute(0, 2, 1).view(batch_size, num_channels, height, width)

        # 使用卷积投影函数对隐藏状态进行键、查询和值的投影
        key = self.convolution_projection_key(hidden_state)
        query = self.convolution_projection_query(hidden_state)
        value = self.convolution_projection_value(hidden_state)

        # 如果设置了包含CLS token，则将CLS token拼接到查询、键和值中
        if self.with_cls_token:
            query = torch.cat((cls_token, query), dim=1)
            key = torch.cat((cls_token, key), dim=1)
            value = torch.cat((cls_token, value), dim=1)

        # 计算每个头的维度
        head_dim = self.embed_dim // self.num_heads

        # 使用投影后的结果重新排列以适应多头注意力机制："b t (h d)"
        query = self.rearrange_for_multi_head_attention(self.projection_query(query))
        key = self.rearrange_for_multi_head_attention(self.projection_key(key))
        value = self.rearrange_for_multi_head_attention(self.projection_value(value))

        # 计算注意力分数
        attention_score = torch.einsum("bhlk,bhtk->bhlt", [query, key]) * self.scale
        # 计算注意力概率并应用dropout
        attention_probs = torch.nn.functional.softmax(attention_score, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # 计算上下文向量
        context = torch.einsum("bhlt,bhtv->bhlv", [attention_probs, value])
        
        # 重新排列上下文向量的维度："b h t d -> b t (h d)"
        _, _, hidden_size, _ = context.shape
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, hidden_size, self.num_heads * head_dim)
        
        # 返回上下文向量作为前向传播的输出结果
        return context
class CvtSelfOutput(nn.Module):
    """
    The residual connection is defined in CvtLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, embed_dim, drop_rate):
        super().__init__()
        # 定义全连接层，输入和输出维度为 embed_dim
        self.dense = nn.Linear(embed_dim, embed_dim)
        # 定义 dropout 层，应用于全连接层的输出
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, hidden_state, input_tensor):
        # 输入 hidden_state 经过全连接层
        hidden_state = self.dense(hidden_state)
        # 对全连接层输出应用 dropout
        hidden_state = self.dropout(hidden_state)
        return hidden_state


class CvtAttention(nn.Module):
    def __init__(
        self,
        num_heads,
        embed_dim,
        kernel_size,
        padding_q,
        padding_kv,
        stride_q,
        stride_kv,
        qkv_projection_method,
        qkv_bias,
        attention_drop_rate,
        drop_rate,
        with_cls_token=True,
    ):
        super().__init__()
        # 创建自注意力模块，参数由传入的参数决定
        self.attention = CvtSelfAttention(
            num_heads,
            embed_dim,
            kernel_size,
            padding_q,
            padding_kv,
            stride_q,
            stride_kv,
            qkv_projection_method,
            qkv_bias,
            attention_drop_rate,
            with_cls_token,
        )
        # 创建自定义的输出模块，包括全连接层和 dropout
        self.output = CvtSelfOutput(embed_dim, drop_rate)
        # 存储被修剪的注意力头的集合
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 寻找可修剪的注意力头，并获取相应的索引
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

    def forward(self, hidden_state, height, width):
        # 使用自注意力模块处理 hidden_state，height 和 width 是额外的参数
        self_output = self.attention(hidden_state, height, width)
        # 使用自定义的输出模块处理 self_output 和 hidden_state
        attention_output = self.output(self_output, hidden_state)
        return attention_output


class CvtIntermediate(nn.Module):
    def __init__(self, embed_dim, mlp_ratio):
        super().__init__()
        # 定义全连接层，输入维度为 embed_dim，输出维度为 embed_dim * mlp_ratio
        self.dense = nn.Linear(embed_dim, int(embed_dim * mlp_ratio))
        # 定义 GELU 激活函数
        self.activation = nn.GELU()

    def forward(self, hidden_state):
        # 输入 hidden_state 经过全连接层
        hidden_state = self.dense(hidden_state)
        # 对全连接层输出应用 GELU 激活函数
        hidden_state = self.activation(hidden_state)
        return hidden_state


class CvtOutput(nn.Module):
    # 定义初始化函数，接受嵌入维度、MLP比率和丢弃率作为参数
    def __init__(self, embed_dim, mlp_ratio, drop_rate):
        # 调用父类构造函数进行初始化
        super().__init__()
        # 创建一个全连接层，将输入维度乘以MLP比率得到输出维度，输出维度为embed_dim
        self.dense = nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        # 创建一个丢弃层，丢弃率为drop_rate，用于在训练过程中随机丢弃部分神经元以减少过拟合
        self.dropout = nn.Dropout(drop_rate)

    # 定义前向传播函数，接受隐藏状态和输入张量作为输入，返回处理后的隐藏状态
    def forward(self, hidden_state, input_tensor):
        # 将隐藏状态通过全连接层dense进行线性变换
        hidden_state = self.dense(hidden_state)
        # 对线性变换后的隐藏状态进行dropout操作，随机丢弃一部分神经元
        hidden_state = self.dropout(hidden_state)
        # 将dropout后的隐藏状态与输入张量相加，实现残差连接
        hidden_state = hidden_state + input_tensor
        # 返回处理后的隐藏状态作为输出
        return hidden_state
class CvtLayer(nn.Module):
    """
    CvtLayer composed by attention layers, normalization and multi-layer perceptrons (mlps).
    """

    def __init__(
        self,
        num_heads,
        embed_dim,
        kernel_size,
        padding_q,
        padding_kv,
        stride_q,
        stride_kv,
        qkv_projection_method,
        qkv_bias,
        attention_drop_rate,
        drop_rate,
        mlp_ratio,
        drop_path_rate,
        with_cls_token=True,
    ):
        super().__init__()
        
        # 初始化自注意力层
        self.attention = CvtAttention(
            num_heads,
            embed_dim,
            kernel_size,
            padding_q,
            padding_kv,
            stride_q,
            stride_kv,
            qkv_projection_method,
            qkv_bias,
            attention_drop_rate,
            drop_rate,
            with_cls_token,
        )

        # 初始化中间层
        self.intermediate = CvtIntermediate(embed_dim, mlp_ratio)
        
        # 初始化输出层
        self.output = CvtOutput(embed_dim, mlp_ratio, drop_rate)
        
        # 如果有drop_path_rate大于0，则初始化drop path层；否则使用恒等映射
        self.drop_path = CvtDropPath(drop_prob=drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        
        # 初始化前层归一化层
        self.layernorm_before = nn.LayerNorm(embed_dim)
        
        # 初始化后层归一化层
        self.layernorm_after = nn.LayerNorm(embed_dim)

    def forward(self, hidden_state, height, width):
        # 对隐藏状态进行前层归一化后，通过自注意力层计算自注意力输出
        self_attention_output = self.attention(
            self.layernorm_before(hidden_state),  # in Cvt, layernorm is applied before self-attention
            height,
            width,
        )
        attention_output = self_attention_output
        
        # 对自注意力输出应用DropPath层
        attention_output = self.drop_path(attention_output)

        # 第一个残差连接
        hidden_state = attention_output + hidden_state

        # 对隐藏状态进行后层归一化后，通过中间层计算层输出
        layer_output = self.layernorm_after(hidden_state)
        layer_output = self.intermediate(layer_output)

        # 第二个残差连接
        layer_output = self.output(layer_output, hidden_state)
        
        # 对层输出应用DropPath层
        layer_output = self.drop_path(layer_output)
        
        return layer_output
    # 初始化函数，用于初始化一个CvtTransformer对象
    def __init__(self, config, stage):
        # 调用父类的初始化方法
        super().__init__()
        # 将输入的配置参数和阶段保存到对象的属性中
        self.config = config
        self.stage = stage
        # 如果配置中指定了要使用的类别标记（cls_token），则创建一个可学习的类别标记参数
        if self.config.cls_token[self.stage]:
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.config.embed_dim[-1]))

        # 创建嵌入层对象，用于将输入的隐藏状态映射到嵌入空间
        self.embedding = CvtEmbeddings(
            patch_size=config.patch_sizes[self.stage],           # 补丁大小
            stride=config.patch_stride[self.stage],              # 步长
            num_channels=config.num_channels if self.stage == 0 else config.embed_dim[self.stage - 1],  # 输入通道数
            embed_dim=config.embed_dim[self.stage],              # 嵌入维度
            padding=config.patch_padding[self.stage],            # 填充
            dropout_rate=config.drop_rate[self.stage],           # 丢弃率
        )

        # 计算每个层的丢弃路径率
        drop_path_rates = [x.item() for x in torch.linspace(0, config.drop_path_rate[self.stage], config.depth[stage])]

        # 创建包含多个CvtLayer层的顺序容器
        self.layers = nn.Sequential(
            *[
                CvtLayer(
                    num_heads=config.num_heads[self.stage],       # 头数
                    embed_dim=config.embed_dim[self.stage],       # 嵌入维度
                    kernel_size=config.kernel_qkv[self.stage],    # QKV核大小
                    padding_q=config.padding_q[self.stage],      # Q填充
                    padding_kv=config.padding_kv[self.stage],    # KV填充
                    stride_kv=config.stride_kv[self.stage],      # KV步长
                    stride_q=config.stride_q[self.stage],        # Q步长
                    qkv_projection_method=config.qkv_projection_method[self.stage],  # QKV投影方法
                    qkv_bias=config.qkv_bias[self.stage],        # QKV偏置
                    attention_drop_rate=config.attention_drop_rate[self.stage],  # 注意力丢弃率
                    drop_rate=config.drop_rate[self.stage],       # 丢弃率
                    drop_path_rate=drop_path_rates[self.stage],   # 丢弃路径率
                    mlp_ratio=config.mlp_ratio[self.stage],       # MLP比率
                    with_cls_token=config.cls_token[self.stage],  # 是否包含类别标记
                )
                # 根据配置的深度创建多个CvtLayer对象
                for _ in range(config.depth[self.stage])
            ]
        )

    # 前向传播函数，定义了如何计算输入隐藏状态的转换过程
    def forward(self, hidden_state):
        cls_token = None
        # 将输入的隐藏状态通过嵌入层进行转换
        hidden_state = self.embedding(hidden_state)
        batch_size, num_channels, height, width = hidden_state.shape
        # 将转换后的张量重新排列为 batch_size x (height * width) x num_channels
        hidden_state = hidden_state.view(batch_size, num_channels, height * width).permute(0, 2, 1)
        # 如果配置中指定了要使用的类别标记，则将类别标记和转换后的隐藏状态连接起来
        if self.config.cls_token[self.stage]:
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            hidden_state = torch.cat((cls_token, hidden_state), dim=1)

        # 逐层对输入的隐藏状态进行变换，通过CvtLayer层处理
        for layer in self.layers:
            layer_outputs = layer(hidden_state, height, width)
            hidden_state = layer_outputs

        # 如果配置中指定了要使用的类别标记，则从最终的隐藏状态中分离出类别标记
        if self.config.cls_token[self.stage]:
            cls_token, hidden_state = torch.split(hidden_state, [1, height * width], 1)
        # 将最终的隐藏状态重新排列为 batch_size x num_channels x height x width 的形式
        hidden_state = hidden_state.permute(0, 2, 1).view(batch_size, num_channels, height, width)
        # 返回转换后的隐藏状态和类别标记（如果有的话）
        return hidden_state, cls_token
        """
        # CVTEncoder 类，用于实现一个可变形视觉 Transformer 编码器模型

        def __init__(self, config):
            # 初始化函数，继承自 nn.Module
            super().__init__()
            # 保存模型配置信息
            self.config = config
            # 初始化模型的多个编码阶段
            self.stages = nn.ModuleList([])
            # 根据配置中的深度信息，逐个创建并添加 CvtStage 实例到 stages 中
            for stage_idx in range(len(config.depth)):
                self.stages.append(CvtStage(config, stage_idx))

        def forward(self, pixel_values, output_hidden_states=False, return_dict=True):
            # 初始化隐藏状态和额外输出
            all_hidden_states = () if output_hidden_states else None
            hidden_state = pixel_values

            # 初始化分类标记
            cls_token = None
            # 遍历每个阶段的模块
            for _, (stage_module) in enumerate(self.stages):
                # 通过阶段模块处理隐藏状态，同时获取分类标记
                hidden_state, cls_token = stage_module(hidden_state)
                # 如果需要输出所有隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_state,)

            # 如果不返回字典形式的输出，则返回非空的隐藏状态、分类标记和所有隐藏状态
            if not return_dict:
                return tuple(v for v in [hidden_state, cls_token, all_hidden_states] if v is not None)

            # 返回包含分类标记、最终隐藏状态和所有隐藏状态的 BaseModelOutputWithCLSToken 对象
            return BaseModelOutputWithCLSToken(
                last_hidden_state=hidden_state,
                cls_token_value=cls_token,
                hidden_states=all_hidden_states,
            )
        """



CVT_PRETRAINED_MODEL_DOCSTRING = r"""
    This model is an abstract class for handling weights initialization and providing a simple interface for downloading
    and loading pretrained models.

    Configuration:
        config_class (`CvtConfig`): The configuration class holding all parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. To load weights, use the `~PreTrainedModel.from_pretrained` method.

    Attributes:
        base_model_prefix (`str`): Prefix applied to base model attributes.
        main_input_name (`str`): Name of the main input attribute for the model, typically 'pixel_values'.
"""

CVT_INIT_WEIGHTS_DOCSTRING = r"""
    Initialize the weights of the model module.

    Args:
        module (`nn.Module`): The PyTorch module for which weights need to be initialized.
"""

CVT_INIT_WEIGHTS_FUNCTION_DOCSTRING = r"""
        Initialize the weights of the provided module.

        Args:
            module (`nn.Module`): The PyTorch module for which weights need to be initialized.
"""

CVT_INIT_WEIGHTS_LINEAR_CONV2D_DOCSTRING = r"""
            Initialize the weights of a Linear or Conv2d module.

            Args:
                module (`nn.Module`): The PyTorch Linear or Conv2d module.
"""

CVT_INIT_WEIGHTS_LAYERNORM_DOCSTRING = r"""
            Initialize the weights of a LayerNorm module.

            Args:
                module (`nn.Module`): The PyTorch LayerNorm module.
"""

CVT_INIT_WEIGHTS_CVTSTAGE_DOCSTRING = r"""
            Initialize the weights of a CvtStage module.

            Args:
                module (`CvtStage`): The CvtStage module.
"""

CVT_START_DOCSTRING,
CVT_INPUTS_DOCSTRING,
CVT_PRETRAINED_MODEL_DOCSTRING,
CVT_INIT_WEIGHTS_DOCSTRING,
CVT_INIT_WEIGHTS_FUNCTION_DOCSTRING,
CVT_INIT_WEIGHTS_LINEAR_CONV2D_DOCSTRING,
CVT_INIT_WEIGHTS_LAYERNORM_DOCSTRING,
CVT_INIT_WEIGHTS_CVTSTAGE_DOCSTRING```
        """
        # CVTEncoder 类，用于实现一个可变形视觉 Transformer 编码器模型

        def __init__(self, config):
            # 初始化函数，继承自 nn.Module
            super().__init__()
            # 保存模型配置信息
            self.config = config
            # 初始化模型的多个编码阶段
            self.stages = nn.ModuleList([])
            # 根据配置中的深度信息，逐个创建并添加 CvtStage 实例到 stages 中
            for stage_idx in range(len(config.depth)):
                self.stages.append(CvtStage(config, stage_idx))

        def forward(self, pixel_values, output_hidden_states=False, return_dict=True):
            # 初始化隐藏状态和额外输出
            all_hidden_states = () if output_hidden_states else None
            hidden_state = pixel_values

            # 初始化分类标记
            cls_token = None
            # 遍历每个阶段的模块
            for _, (stage_module) in enumerate(self.stages):
                # 通过阶段模块处理隐藏状态，同时获取分类标记
                hidden_state, cls_token = stage_module(hidden_state)
                # 如果需要输出所有隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_state,)

            # 如果不返回字典形式的输出，则返回非空的隐藏状态、分类标记和所有隐藏状态
            if not return_dict:
                return tuple(v for v in [hidden_state, cls_token, all_hidden_states] if v is not None)

            # 返回包含分类标记、最终隐藏状态和所有隐藏状态的 BaseModelOutputWithCLSToken 对象
            return BaseModelOutputWithCLSToken(
                last_hidden_state=hidden_state,
                cls_token_value=cls_token,
                hidden_states=all_hidden_states,
            )
        """



CVT_PRETRAINED_MODEL_DOCSTRING = r"""
    This model is an abstract class for handling weights initialization and providing a simple interface for downloading
    and loading pretrained models.

    Configuration:
        config_class (`CvtConfig`): The configuration class holding all parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. To load weights, use the `~PreTrainedModel.from_pretrained` method.

    Attributes:
        base_model_prefix (`str`): Prefix applied to base model attributes.
        main_input_name (`str`): Name of the main input attribute for the model, typically 'pixel_values'.
"""

CVT_INIT_WEIGHTS_DOCSTRING = r"""
    Initialize the weights of the model module.

    Args:
        module (`nn.Module`): The PyTorch module for which weights need to be initialized.
"""

CVT_INIT_WEIGHTS_FUNCTION_DOCSTRING = r"""
        Initialize the weights of the provided module.

        Args:
            module (`nn.Module`): The PyTorch module for which weights need to be initialized.
"""

CVT_INIT_WEIGHTS_LINEAR_CONV2D_DOCSTRING = r"""
            Initialize the weights of a Linear or Conv2d module.

            Args:
                module (`nn.Module`): The PyTorch Linear or Conv2d module.
"""

CVT_INIT_WEIGHTS_LAYERNORM_DOCSTRING = r"""
            Initialize the weights of a LayerNorm module.

            Args:
                module (`nn.Module`): The PyTorch LayerNorm module.
"""

CVT_INIT_WEIGHTS_CVTSTAGE_DOCSTRING = r"""
            Initialize the weights of a CvtStage module.

            Args:
                module (`CvtStage`): The CvtStage module.
"""

CVT_START_DOCSTRING,
CVT_INPUTS_DOCSTRING,
CVT_PRETRAINED_MODEL_DOCSTRING,
CVT_INIT_WEIGHTS_DOCSTRING,
CVT_INIT_WEIGHTS_FUNCTION_DOCSTRING,
CVT_INIT_WEIGHTS_LINEAR_CONV2D_DOCSTRING,
CVT_INIT_WEIGHTS_LAYERNORM_DOCSTRING,
CVT_INIT_WEIGHTS_CVTSTAGE_DOCSTRING
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            # 输入的像素数值。可以使用 `AutoImageProcessor` 获取像素值。详见 `CvtImageProcessor.__call__`。
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`CvtImageProcessor.__call__`]
            for details.
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。返回的张量中包含 `hidden_states`，详见返回的张量部分。
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            # 是否返回一个 `~file_utils.ModelOutput` 而不是一个普通的元组。
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""
Cvt Model transformer outputting raw hidden-states without any specific head on top.
"""
class CvtModel(CvtPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.encoder = CvtEncoder(config)  # 初始化 CvtEncoder 模块
        self.post_init()

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)  # 对指定层的注意力头进行修剪

    @add_start_docstrings_to_model_forward(CVT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithCLSToken,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithCLSToken]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")  # 如果 pixel_values 为 None，则抛出数值错误异常

        encoder_outputs = self.encoder(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]  # 获取编码器输出的序列输出

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]  # 返回序列输出以及额外的编码器输出

        return BaseModelOutputWithCLSToken(
            last_hidden_state=sequence_output,
            cls_token_value=encoder_outputs.cls_token_value,
            hidden_states=encoder_outputs.hidden_states,
        )  # 返回包含 CLS 标记值的基础模型输出类型

"""
Cvt Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
the [CLS] token) e.g. for ImageNet.
"""
class CvtForImageClassification(CvtPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels  # 存储标签数目
        self.cvt = CvtModel(config, add_pooling_layer=False)  # 初始化 CvtModel 模块，不添加池化层
        self.layernorm = nn.LayerNorm(config.embed_dim[-1])  # 应用 LayerNorm 到最后一个嵌入维度
        # 分类器头部
        self.classifier = (
            nn.Linear(config.embed_dim[-1], config.num_labels) if config.num_labels > 0 else nn.Identity()
        )  # 如果存在标签数目，则创建线性层作为分类器头部，否则创建恒等映射

        # Initialize weights and apply final processing
        self.post_init()
    
    @add_start_docstrings_to_model_forward(CVT_INPUTS_DOCSTRING)
    # 使用装饰器添加代码示例的文档字符串，指定模型使用的检查点、输出类型、配置类和预期输出
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    # 定义模型的前向传播方法，接受像素值张量、标签张量、是否返回隐藏状态和是否返回字典作为参数
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageClassifierOutputWithNoAttention]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 根据需要确定是否返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 调用转换器进行处理像素值，根据需要返回隐藏状态，结果存储在outputs中
        outputs = self.cvt(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取序列输出和CLS token
        sequence_output = outputs[0]
        cls_token = outputs[1]
        # 如果配置中的CLS token标记为最后一个
        if self.config.cls_token[-1]:
            # 应用层归一化到CLS token
            sequence_output = self.layernorm(cls_token)
        else:
            # 获取序列输出的维度信息
            batch_size, num_channels, height, width = sequence_output.shape
            # 重新排列 "b c h w -> b (h w) c"
            sequence_output = sequence_output.view(batch_size, num_channels, height * width).permute(0, 2, 1)
            # 应用层归一化到重新排列的序列输出
            sequence_output = self.layernorm(sequence_output)

        # 计算序列输出的平均值
        sequence_output_mean = sequence_output.mean(dim=1)
        # 将平均值输入分类器得到logits
        logits = self.classifier(sequence_output_mean)

        # 初始化损失为None
        loss = None
        # 如果存在标签
        if labels is not None:
            # 如果问题类型尚未确定
            if self.config.problem_type is None:
                # 根据标签数确定问题类型
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.config.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型计算损失
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.config.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 如果不需要返回字典，则返回logits和额外的隐藏状态
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回ImageClassifierOutputWithNoAttention对象，包括损失、logits和隐藏状态
        return ImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)
```