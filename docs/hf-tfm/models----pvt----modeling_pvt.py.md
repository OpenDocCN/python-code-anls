# `.\models\pvt\modeling_pvt.py`

```py
# coding=utf-8
# 上面的行指定了文件的编码格式为 UTF-8，确保可以正确处理所有字符
# Copyright 2023 Authors: Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan,
# Kaitao Song, Ding Liang, Tong Lu, Ping Luo, Ling Shao and The HuggingFace Inc. team.
# 版权声明，列出了代码的版权信息及贡献者
# All rights reserved.
# 版权声明，保留所有权利
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache License, Version 2.0 授权许可，允许在遵守许可的情况下使用本文件
# you may not use this file except in compliance with the License.
# 除非符合许可证的要求，否则不得使用本文件
# You may obtain a copy of the License at
# 可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# 除非适用法律要求或书面同意，否则软件按"AS IS"分发，
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 在许可的情况下，按"AS IS"分发，不提供任何明示或暗示的保证或条件
# See the License for the specific language governing permissions and
# limitations under the License.
# 查看许可证以了解特定语言下的权限和限制
""" PyTorch PVT model."""
# 模型的简短描述

import collections
# 导入 collections 模块，用于操作集合数据类型
import math
# 导入 math 模块，提供数学运算函数
from typing import Iterable, Optional, Tuple, Union
# 导入类型提示的模块，用于声明函数和变量的类型

import torch
# 导入 PyTorch 库
import torch.nn.functional as F
# 导入 PyTorch 的函数模块
import torch.utils.checkpoint
# 导入 PyTorch 的检查点模块
from torch import nn
# 从 PyTorch 中导入神经网络模块
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
# 从 PyTorch 中导入损失函数

from ...activations import ACT2FN
# 从本地模块导入 ACT2FN 激活函数
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput
# 从本地模块导入模型输出类
from ...modeling_utils import PreTrainedModel
# 从本地模块导入预训练模型的基类
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
# 从本地模块导入模型剪枝相关的函数
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
# 从本地模块导入其他实用工具函数和日志函数

from .configuration_pvt import PvtConfig
# 从当前目录下的配置文件中导入 PVT 模型的配置类

logger = logging.get_logger(__name__)
# 获取当前模块的日志记录器

_CONFIG_FOR_DOC = "PvtConfig"
# 文档中用于说明的配置变量名

_CHECKPOINT_FOR_DOC = "Zetatech/pvt-tiny-224"
# 文档中用于说明的检查点变量名

_EXPECTED_OUTPUT_SHAPE = [1, 50, 512]
# 预期的模型输出形状

_IMAGE_CLASS_CHECKPOINT = "Zetatech/pvt-tiny-224"
# 图像分类检查点的名称

_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"
# 预期的图像分类输出

PVT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Zetatech/pvt-tiny-224"
    # PVT 预训练模型存档列表，包含一个模型路径
    # See all PVT models at https://huggingface.co/models?filter=pvt
    # 可以在指定的网址查看所有的 PVT 模型
]


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
    # 如果 drop_prob 为 0 或者不处于训练模式，则直接返回输入
    if drop_prob == 0.0 or not training:
        return input
    # 计算保留概率
    keep_prob = 1 - drop_prob
    # 创建随机张量，与输入张量相同的形状
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # 将随机张量二值化
    # 计算输出，按照保留概率调整输入张量的值
    output = input.div(keep_prob) * random_tensor
    return output
# 从 transformers.models.convnext.modeling_convnext.ConvNextDropPath 复制过来的类，用于在残差块的主路径中每个样本上应用Drop Path（随机深度）。
class PvtDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 调用 drop_path 函数，对输入的 hidden_states 执行 Drop Path 操作，根据当前模型是否处于训练状态来决定是否应用 Drop Path。
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        # 返回一个描述对象的额外信息的字符串，格式为 "p=drop_prob"，其中 drop_prob 是初始化时传入的概率值。
        return "p={}".format(self.drop_prob)


class PvtPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(
        self,
        config: PvtConfig,
        image_size: Union[int, Iterable[int]],
        patch_size: Union[int, Iterable[int]],
        stride: int,
        num_channels: int,
        hidden_size: int,
        cls_token: bool = False,
    ):
        super().__init__()
        # 初始化函数，用于将输入的像素值 `pixel_values` 转换成 Transformer 模型可用的 patch embeddings。
        self.config = config
        # 将 image_size 和 patch_size 转换为 Iterable 类型，如果它们不是的话。
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        # 计算图像中的 patch 数量。
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        # 初始化位置编码张量，形状为 (1, num_patches + 1 if cls_token else num_patches, hidden_size)，用于 Transformer 模型中位置编码。
        self.position_embeddings = nn.Parameter(
            torch.randn(1, num_patches + 1 if cls_token else num_patches, hidden_size)
        )
        # 如果 cls_token 为 True，则初始化一个形状为 (1, 1, hidden_size) 的可学习的类别令牌（class token）张量。
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size)) if cls_token else None
        # 使用卷积操作将输入的像素值转换成 hidden_size 维度的 patch embeddings。
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=stride, stride=patch_size)
        # 使用 LayerNorm 对隐藏状态进行归一化，eps 是 LayerNorm 的 epsilon 参数。
        self.layer_norm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        # 使用 Dropout 进行隐藏状态的随机丢弃，p 是 Dropout 概率。
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        # 插值位置编码，将输入的位置编码张量 embeddings 插值到新的高度和宽度。
        num_patches = height * width
        # 如果当前输入的 patch 数量等于配置中指定的图像总 patch 数量，则直接返回位置编码张量。
        if num_patches == self.config.image_size * self.config.image_size:
            return self.position_embeddings
        # 将输入的 embeddings 重塑为 (1, height, width, -1)，然后进行维度置换。
        embeddings = embeddings.reshape(1, height, width, -1).permute(0, 3, 1, 2)
        # 使用双线性插值将 embeddings 插值到新的高度和宽度。
        interpolated_embeddings = F.interpolate(embeddings, size=(height, width), mode="bilinear")
        # 重塑插值后的 embeddings 为 (1, -1, height * width)，然后进行维度置换。
        interpolated_embeddings = interpolated_embeddings.reshape(1, -1, height * width).permute(0, 2, 1)
        return interpolated_embeddings
    # 定义一个方法，接受一个张量 pixel_values，返回元组 (embeddings, height, width)
    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        # 获取输入张量的维度信息：batch_size, num_channels, height, width
        batch_size, num_channels, height, width = pixel_values.shape
        
        # 检查输入张量的通道数是否与模型要求的通道数一致
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        
        # 使用 projection 方法对输入张量进行投影，得到 patch_embed
        patch_embed = self.projection(pixel_values)
        
        # 忽略前面的维度，获取 patch_embed 的最后两个维度的大小作为新的 height 和 width
        *_, height, width = patch_embed.shape
        
        # 将 patch_embed 进行展平处理，然后交换维度 1 和 2
        patch_embed = patch_embed.flatten(2).transpose(1, 2)
        
        # 对 patch_embed 进行 layer normalization
        embeddings = self.layer_norm(patch_embed)
        
        # 如果存在 cls_token，则在 embeddings 前面添加 cls_token
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            embeddings = torch.cat((cls_token, embeddings), dim=1)
            
            # 使用 interpolate_pos_encoding 方法插值生成位置编码，并在前面添加初始位置编码
            position_embeddings = self.interpolate_pos_encoding(self.position_embeddings[:, 1:], height, width)
            position_embeddings = torch.cat((self.position_embeddings[:, :1], position_embeddings), dim=1)
        else:
            # 如果不存在 cls_token，则直接使用 interpolate_pos_encoding 方法生成位置编码
            position_embeddings = self.interpolate_pos_encoding(self.position_embeddings, height, width)
        
        # 将 embeddings 和 position_embeddings 相加，并应用 dropout
        embeddings = self.dropout(embeddings + position_embeddings)

        # 返回计算得到的 embeddings，以及当前的 height 和 width
        return embeddings, height, width
    # PvtSelfOutput 类定义，继承自 nn.Module
    class PvtSelfOutput(nn.Module):
        def __init__(self, config: PvtConfig, hidden_size: int):
            super().__init__()
            # 初始化一个全连接层，用于线性变换 hidden_states
            self.dense = nn.Linear(hidden_size, hidden_size)
            # 初始化一个 Dropout 层，用于随机失活 hidden_states 的部分神经元
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            # 对输入的 hidden_states 进行线性变换
            hidden_states = self.dense(hidden_states)
            # 对变换后的 hidden_states 进行随机失活
            hidden_states = self.dropout(hidden_states)
            return hidden_states

    # PvtEfficientSelfAttention 类定义，继承自 nn.Module
    class PvtEfficientSelfAttention(nn.Module):
        """Efficient self-attention mechanism with reduction of the sequence [PvT paper](https://arxiv.org/abs/2102.12122)."""

        def __init__(
            self, config: PvtConfig, hidden_size: int, num_attention_heads: int, sequences_reduction_ratio: float
        ):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_attention_heads = num_attention_heads

            # 检查隐藏大小是否可以被注意力头的数量整除
            if self.hidden_size % self.num_attention_heads != 0:
                raise ValueError(
                    f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                    f"heads ({self.num_attention_heads})"
                )

            # 计算每个注意力头的大小
            self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
            self.all_head_size = self.num_attention_heads * self.attention_head_size

            # 初始化查询、键、值的线性层，并指定是否使用偏置
            self.query = nn.Linear(self.hidden_size, self.all_head_size, bias=config.qkv_bias)
            self.key = nn.Linear(self.hidden_size, self.all_head_size, bias=config.qkv_bias)
            self.value = nn.Linear(self.hidden_size, self.all_head_size, bias=config.qkv_bias)

            # 初始化 Dropout 层，用于注意力分数的随机失活
            self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

            # 设置序列缩减比率，如果比率大于1，则初始化一个二维卷积层和 LayerNorm 层
            self.sequences_reduction_ratio = sequences_reduction_ratio
            if sequences_reduction_ratio > 1:
                self.sequence_reduction = nn.Conv2d(
                    hidden_size, hidden_size, kernel_size=sequences_reduction_ratio, stride=sequences_reduction_ratio
                )
                self.layer_norm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)

        def transpose_for_scores(self, hidden_states: int) -> torch.Tensor:
            # 重新形状 hidden_states，以便进行多头注意力计算
            new_shape = hidden_states.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
            hidden_states = hidden_states.view(new_shape)
            return hidden_states.permute(0, 2, 1, 3)

        def forward(
            self,
            hidden_states: torch.Tensor,
            height: int,
            width: int,
            output_attentions: bool = False,
    ) -> Tuple[torch.Tensor]:
        # 使用 self.query 对隐藏状态进行查询操作，并调整维度以匹配注意力分数计算所需的格式
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        # 如果指定了序列缩减比率大于1，则进行以下操作
        if self.sequences_reduction_ratio > 1:
            batch_size, seq_len, num_channels = hidden_states.shape
            # 将隐藏状态重塑为 (batch_size, num_channels, height, width) 的格式
            hidden_states = hidden_states.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)
            # 应用序列缩减操作
            hidden_states = self.sequence_reduction(hidden_states)
            # 将隐藏状态重塑回 (batch_size, seq_len, num_channels) 的格式
            hidden_states = hidden_states.reshape(batch_size, num_channels, -1).permute(0, 2, 1)
            # 应用层归一化操作
            hidden_states = self.layer_norm(hidden_states)

        # 使用 self.key 对隐藏状态进行键操作，并调整维度以匹配注意力分数计算所需的格式
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        
        # 使用 self.value 对隐藏状态进行值操作，并调整维度以匹配注意力分数计算所需的格式
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # 计算 "查询" 和 "键" 之间的点积，得到原始注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # 将注意力分数除以缩放因子，以防止数值不稳定
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 将注意力分数归一化为注意力概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 使用 dropout 随机丢弃一些注意力概率，以减少过拟合风险
        attention_probs = self.dropout(attention_probs)

        # 计算上下文向量，将注意力概率与值进行加权求和
        context_layer = torch.matmul(attention_probs, value_layer)

        # 调整上下文向量的维度以匹配后续网络层的输入要求
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # 根据需要返回不同的输出，包括上下文向量和注意力概率
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
class PvtAttention(nn.Module):
    def __init__(
        self, config: PvtConfig, hidden_size: int, num_attention_heads: int, sequences_reduction_ratio: float
    ):
        super().__init__()
        # 初始化自注意力层，使用给定的配置、隐藏大小、注意力头数和序列缩减比例
        self.self = PvtEfficientSelfAttention(
            config,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            sequences_reduction_ratio=sequences_reduction_ratio,
        )
        # 初始化自注意力输出层，使用给定的配置和隐藏大小
        self.output = PvtSelfOutput(config, hidden_size=hidden_size)
        # 初始化一个空集合，用于存储被修剪的注意力头
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 找到可修剪的注意力头和其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 对线性层进行修剪
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储修剪后的注意力头
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self, hidden_states: torch.Tensor, height: int, width: int, output_attentions: bool = False
    ) -> Tuple[torch.Tensor]:
        # 执行前向传播，获取自注意力层的输出
        self_outputs = self.self(hidden_states, height, width, output_attentions)

        # 将自注意力层的输出作为输入，经过输出层得到注意力输出
        attention_output = self.output(self_outputs[0])
        # 如果需要输出注意力权重，则将它们加入到输出中
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class PvtFFN(nn.Module):
    def __init__(
        self,
        config: PvtConfig,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
    ):
        super().__init__()
        # 根据输入和输出特征大小初始化第一个线性层
        out_features = out_features if out_features is not None else in_features
        self.dense1 = nn.Linear(in_features, hidden_features)
        # 初始化中间激活函数，根据配置中的隐藏激活函数选择相应的函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        # 初始化第二个线性层，输出特征大小根据给定或默认的大小确定
        self.dense2 = nn.Linear(hidden_features, out_features)
        # 初始化一个Dropout层，使用给定的隐藏丢弃概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过第一个线性层传播输入的隐藏状态
        hidden_states = self.dense1(hidden_states)
        # 经过中间激活函数处理隐藏状态
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 应用Dropout层处理隐藏状态
        hidden_states = self.dropout(hidden_states)
        # 通过第二个线性层传播隐藏状态
        hidden_states = self.dense2(hidden_states)
        # 再次应用Dropout层处理最终的隐藏状态并返回
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class PvtLayer(nn.Module):
    # 初始化函数，用于初始化一个 PvtLayer 对象
    def __init__(
        self,
        config: PvtConfig,
        hidden_size: int,
        num_attention_heads: int,
        drop_path: float,
        sequences_reduction_ratio: float,
        mlp_ratio: float,
    ):
        super().__init__()
        # 初始化第一个 LayerNorm 层，用于对输入进行归一化处理
        self.layer_norm_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        # 初始化注意力机制模块 PvtAttention
        self.attention = PvtAttention(
            config=config,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            sequences_reduction_ratio=sequences_reduction_ratio,
        )
        # 根据 drop_path 参数初始化 PvtDropPath，若 drop_path > 0.0 则使用 PvtDropPath，否则使用 nn.Identity()
        self.drop_path = PvtDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        # 初始化第二个 LayerNorm 层，用于对输入进行归一化处理
        self.layer_norm_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        # 计算 MLP 的隐藏层大小
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        # 初始化 MLP 模块 PvtFFN
        self.mlp = PvtFFN(config=config, in_features=hidden_size, hidden_features=mlp_hidden_size)

    # 前向传播函数，处理输入并返回输出
    def forward(self, hidden_states: torch.Tensor, height: int, width: int, output_attentions: bool = False):
        # 第一步，对输入 hidden_states 进行 LayerNorm 归一化处理，并输入到 self.attention 模块中
        self_attention_outputs = self.attention(
            hidden_states=self.layer_norm_1(hidden_states),
            height=height,
            width=width,
            output_attentions=output_attentions,
        )
        # 从 self_attention_outputs 中取出经过注意力机制处理后的输出
        attention_output = self_attention_outputs[0]
        # 获取其余的输出，这些输出可以用于进一步分析，如输出注意力分数等
        outputs = self_attention_outputs[1:]

        # 应用 drop_path 操作，根据 drop_path 的值决定是否应用 drop path
        attention_output = self.drop_path(attention_output)
        # 将经过 drop path 处理后的注意力输出与原始输入 hidden_states 相加，得到新的 hidden_states
        hidden_states = attention_output + hidden_states

        # 对经过注意力层处理后的 hidden_states 再次进行 LayerNorm 归一化处理
        mlp_output = self.mlp(self.layer_norm_2(hidden_states))

        # 应用 drop_path 操作到 MLP 输出上
        mlp_output = self.drop_path(mlp_output)
        # 将 MLP 处理后的输出与之前的 hidden_states 相加，得到最终层的输出 layer_output
        layer_output = hidden_states + mlp_output

        # 将最终的层输出和之前的其他输出一起返回
        outputs = (layer_output,) + outputs

        return outputs
# 定义一个私有编码器的神经网络模块，继承自 nn.Module 类
class PvtEncoder(nn.Module):
    # 初始化方法，接受一个 PvtConfig 类型的配置对象作为参数
    def __init__(self, config: PvtConfig):
        super().__init__()
        # 将配置对象保存在模块的属性中
        self.config = config

        # 使用线性空间生成随机深度衰减规则列表，用于随机深度路径(drop path)
        drop_path_decays = torch.linspace(0, config.drop_path_rate, sum(config.depths)).tolist()

        # 补丁嵌入
        embeddings = []

        # 循环创建编码器块的嵌入层
        for i in range(config.num_encoder_blocks):
            embeddings.append(
                PvtPatchEmbeddings(
                    config=config,
                    # 如果是第一个块，则使用完整的图像尺寸，否则根据块的索引减少图像尺寸
                    image_size=config.image_size if i == 0 else self.config.image_size // (2 ** (i + 1)),
                    patch_size=config.patch_sizes[i],
                    stride=config.strides[i],
                    num_channels=config.num_channels if i == 0 else config.hidden_sizes[i - 1],
                    hidden_size=config.hidden_sizes[i],
                    # 如果是最后一个块，则设置为 True，否则为 False
                    cls_token=i == config.num_encoder_blocks - 1,
                )
            )
        # 将嵌入层列表转换为 nn.ModuleList，并保存在模块的属性中
        self.patch_embeddings = nn.ModuleList(embeddings)

        # Transformer 块
        blocks = []
        cur = 0
        # 循环创建编码器块的 Transformer 层
        for i in range(config.num_encoder_blocks):
            # 每个块由多个层组成
            layers = []
            if i != 0:
                cur += config.depths[i - 1]
            for j in range(config.depths[i]):
                layers.append(
                    PvtLayer(
                        config=config,
                        hidden_size=config.hidden_sizes[i],
                        num_attention_heads=config.num_attention_heads[i],
                        drop_path=drop_path_decays[cur + j],
                        sequences_reduction_ratio=config.sequence_reduction_ratios[i],
                        mlp_ratio=config.mlp_ratios[i],
                    )
                )
            # 将层列表转换为 nn.ModuleList，并保存在块列表中
            blocks.append(nn.ModuleList(layers))

        # 将块列表转换为 nn.ModuleList，并保存在模块的属性中
        self.block = nn.ModuleList(blocks)

        # 层归一化
        # 创建一个归一化层，对最后一个隐藏层进行归一化处理
        self.layer_norm = nn.LayerNorm(config.hidden_sizes[-1], eps=config.layer_norm_eps)

    # 前向传播方法，接受像素值张量和可选的输出参数，并返回模型的输出
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        # 前向传播方法的参数包括像素值张量和可选的输出参数
        ) -> Union[Tuple, BaseModelOutput]:
        # 如果不需要输出隐藏状态，则初始化为空元组；否则设置为 None
        all_hidden_states = () if output_hidden_states else None
        # 如果不需要输出注意力权重，则初始化为空元组；否则设置为 None
        all_self_attentions = () if output_attentions else None

        # 获取输入张量的批大小
        batch_size = pixel_values.shape[0]
        # 获取 Transformer 模型的块数
        num_blocks = len(self.block)
        # 初始化隐藏状态为输入像素值
        hidden_states = pixel_values

        # 迭代每个嵌入层和对应的块
        for idx, (embedding_layer, block_layer) in enumerate(zip(self.patch_embeddings, self.block)):
            # 第一步，获得图像块的嵌入表示
            hidden_states, height, width = embedding_layer(hidden_states)
            
            # 第二步，将嵌入表示送入 Transformer 块中
            for block in block_layer:
                # 调用 Transformer 块计算输出
                layer_outputs = block(hidden_states, height, width, output_attentions)
                hidden_states = layer_outputs[0]
                # 如果需要输出注意力权重，则累加到 all_self_attentions 中
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果需要输出隐藏状态，则累加到 all_hidden_states 中
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)
            
            # 如果不是最后一个块，则调整隐藏状态的形状和顺序
            if idx != num_blocks - 1:
                hidden_states = hidden_states.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
        
        # 对最终的隐藏状态进行 LayerNorm 处理
        hidden_states = self.layer_norm(hidden_states)

        # 如果需要输出隐藏状态，则将最终隐藏状态加入到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # 如果不需要以字典形式返回结果，则将所有结果打包成元组返回
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        
        # 如果需要以字典形式返回结果，则创建一个 BaseModelOutput 对象返回
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
PVT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`~PvtConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

PVT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`PvtImageProcessor.__call__`]
            for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""
    PVT_START_DOCSTRING,


注释：


# PVT_START_DOCSTRING 是一个常量或者预处理器指令，用于标识文档字符串的起始位置。
# 在一些代码风格中，PVT_START_DOCSTRING 可能用于指示文档字符串块的开始。
# 这种约定有助于程序员识别和管理代码中的文档说明部分。
)
# PvtModel 类的定义，继承自 PvtPreTrainedModel 类
class PvtModel(PvtPreTrainedModel):
    def __init__(self, config: PvtConfig):
        super().__init__(config)
        self.config = config

        # hierarchical Transformer encoder
        # 使用给定的配置初始化 PvtEncoder 对象作为编码器
        self.encoder = PvtEncoder(config)

        # Initialize weights and apply final processing
        # 执行初始化权重和最终处理步骤
        self.post_init()

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 遍历需要修剪的层和头部信息，逐层修剪注意力头
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(PVT_INPUTS_DOCSTRING.format("(batch_size, channels, height, width)"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    # 定义前向传播方法，接受像素值和可选的输出参数，返回模型输出
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将像素值和其它参数传递给编码器，并接收编码器的输出
        encoder_outputs = self.encoder(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取编码器的序列输出
        sequence_output = encoder_outputs[0]

        # 如果不使用返回字典，则返回序列输出和其它编码器输出
        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        # 使用 BaseModelOutput 类构造返回的输出字典
        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@add_start_docstrings(
    """
    Pvt Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.
    """,
    PVT_START_DOCSTRING,
)
# PvtForImageClassification 类的定义，继承自 PvtPreTrainedModel 类
class PvtForImageClassification(PvtPreTrainedModel):
    def __init__(self, config: PvtConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        # 使用给定的配置初始化 PvtModel 对象
        self.pvt = PvtModel(config)

        # Classifier head
        # 根据配置选择线性分类器或者恒等映射
        self.classifier = (
            nn.Linear(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # Initialize weights and apply final processing
        # 执行初始化权重和最终处理步骤
        self.post_init()

    @add_start_docstrings_to_model_forward(PVT_INPUTS_DOCSTRING.format("(batch_size, channels, height, width)"))
    # 重写前向传播方法的文档注释，描述输入参数的形状
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    # 使用装饰器添加代码样本的文档字符串，指定了检查点、输出类型、配置类和预期输出
    def forward(
        self,
        pixel_values: Optional[torch.Tensor],
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
        # 如果 return_dict 为 None，则根据 self.config.use_return_dict 来设定 return_dict 的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 PVT 模型进行前向传播，获取输出结果
        outputs = self.pvt(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从输出中获取序列输出
        sequence_output = outputs[0]

        # 对序列输出的第一个位置进行分类器的运算，得到 logits
        logits = self.classifier(sequence_output[:, 0, :])

        # 初始化 loss 为 None
        loss = None
        # 如果 labels 不为 None，则计算损失函数
        if labels is not None:
            # 如果 self.config.problem_type 未定义，则根据 num_labels 和 labels 的数据类型来定义问题类型
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型选择相应的损失函数
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

        # 如果 return_dict 为 False，则返回 tuple 类型的输出
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则返回 ImageClassifierOutput 类型的对象
        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```