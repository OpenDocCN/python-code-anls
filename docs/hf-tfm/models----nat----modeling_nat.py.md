# `.\models\nat\modeling_nat.py`

```
# coding=utf-8
# Copyright 2022 SHI Labs and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch Neighborhood Attention Transformer model."""

# Importing necessary libraries and modules
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# Importing activation function mappings
from ...activations import ACT2FN
# Importing output classes
from ...modeling_outputs import BackboneOutput
# Importing utility functions for model handling
from ...modeling_utils import PreTrainedModel
# Importing pruning utilities for linear layers
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
# Importing various utility functions and classes
from ...utils import (
    ModelOutput,
    OptionalDependencyNotAvailable,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_natten_available,
    logging,
    replace_return_docstrings,
    requires_backends,
)
# Importing backbone utility functions
from ...utils.backbone_utils import BackboneMixin
# Importing configuration class for Nat model
from .configuration_nat import NatConfig

# Checking availability of external module 'natten'
if is_natten_available():
    # Importing specific functions from 'natten.functional'
    from natten.functional import natten2dav, natten2dqkrpb
else:
    # Define placeholder functions if 'natten' is not available
    def natten2dqkrpb(*args, **kwargs):
        raise OptionalDependencyNotAvailable()

    def natten2dav(*args, **kwargs):
        raise OptionalDependencyNotAvailable()

# Setting up logging for the module
logger = logging.get_logger(__name__)

# General documentation strings
_CONFIG_FOR_DOC = "NatConfig"

# Base documentation strings
_CHECKPOINT_FOR_DOC = "shi-labs/nat-mini-in1k-224"
_EXPECTED_OUTPUT_SHAPE = [1, 7, 7, 512]

# Image classification documentation strings
_IMAGE_CLASS_CHECKPOINT = "shi-labs/nat-mini-in1k-224"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tiger cat"

# List of pretrained model archives for Nat model
NAT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "shi-labs/nat-mini-in1k-224",
    # See all Nat models at https://huggingface.co/models?filter=nat
]

# Definition of dataclass for NatEncoderOutput
@dataclass
class NatEncoderOutput(ModelOutput):
    """
    Nat encoder's outputs, with potential hidden states and attentions.
"""
    # 定义函数的参数及其类型注解，以下为函数的输入参数
    
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层的隐藏状态序列。
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            模型在每一层输出的隐藏状态的元组。
            Tuple of `torch.FloatTensor` representing hidden-states of the model at the output of each layer.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            模型在每一层输出的注意力权重的元组，用于计算自注意力头中的加权平均值。
            Tuple of `torch.FloatTensor` representing attention weights after attention softmax.
        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            模型在每一层输出的隐藏状态的元组，包含了空间维度重塑后的输出。
            Tuple of `torch.FloatTensor` representing hidden-states of the model at the output of each layer, reshaped to include spatial dimensions.
    """
    
    # 初始化函数的参数，默认值为None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
# 创建一个数据类 NatModelOutput，继承自 ModelOutput 类，用于表示 NAT 模型的输出，包括最后隐藏状态的汇总信息。
@dataclass
class NatModelOutput(ModelOutput):
    """
    Nat model's outputs that also contains a pooling of the last hidden states.

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

    # 定义类的属性，用于存储 NAT 模型的输出信息
    last_hidden_state: torch.FloatTensor = None
    pooler_output: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None


# 创建一个数据类 NatImageClassifierOutput，继承自 ModelOutput 类，用于表示 NAT 图像分类的输出。
@dataclass
class NatImageClassifierOutput(ModelOutput):
    """
    Nat outputs for image classification.
    """
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            分类（如果 `config.num_labels==1` 则为回归）损失。
            分类损失或回归损失的张量。
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            分类（如果 `config.num_labels==1` 则为回归）得分（SoftMax 之前）。
            模型输出的分类或回归得分（经过 SoftMax 之前的）。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            `torch.FloatTensor` 元组（当 `output_hidden_states=True` 传入或 `config.output_hidden_states=True` 时返回），
            包含形状为 `(batch_size, sequence_length, hidden_size)` 的张量。

            每个层的模型隐藏状态以及初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            `torch.FloatTensor` 元组（当 `output_attentions=True` 传入或 `config.output_attentions=True` 时返回），
            包含形状为 `(batch_size, num_heads, sequence_length, sequence_length)` 的张量。

            经过注意力 softmax 后的注意力权重，用于计算自注意力头部的加权平均值。
        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            `torch.FloatTensor` 元组（当 `output_hidden_states=True` 传入或 `config.output_hidden_states=True` 时返回），
            包含形状为 `(batch_size, hidden_size, height, width)` 的张量。

            每个层的模型隐藏状态以及初始嵌入输出重塑为包括空间维度。
    """

    # 可选的损失张量，当提供 `labels` 时返回
    loss: Optional[torch.FloatTensor] = None
    # 分类或回归得分张量，形状为 `(batch_size, config.num_labels)`
    logits: torch.FloatTensor = None
    # 可选的隐藏状态张量元组，当 `output_hidden_states=True` 时返回
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 可选的注意力权重张量元组，当 `output_attentions=True` 时返回
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 可选的重塑后的隐藏状态张量元组，当 `output_hidden_states=True` 时返回
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
class NatEmbeddings(nn.Module):
    """
    Construct the patch and position embeddings.
    """

    def __init__(self, config):
        super().__init__()

        self.patch_embeddings = NatPatchEmbeddings(config)  # 创建补丁嵌入对象

        self.norm = nn.LayerNorm(config.embed_dim)  # 初始化 LayerNorm 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # 初始化 Dropout 层

    def forward(self, pixel_values: Optional[torch.FloatTensor]) -> Tuple[torch.Tensor]:
        embeddings = self.patch_embeddings(pixel_values)  # 获取补丁嵌入向量
        embeddings = self.norm(embeddings)  # 应用 LayerNorm
        embeddings = self.dropout(embeddings)  # 应用 Dropout

        return embeddings


class NatPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, height, width, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        patch_size = config.patch_size
        num_channels, hidden_size = config.num_channels, config.embed_dim
        self.num_channels = num_channels

        if patch_size == 4:
            pass  # 当 patch_size 为 4 时，无需额外操作
        else:
            # TODO: Support arbitrary patch sizes.
            raise ValueError("Dinat only supports patch size of 4 at the moment.")  # 报错：当前仅支持 patch 大小为 4

        self.projection = nn.Sequential(
            nn.Conv2d(self.num_channels, hidden_size // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(hidden_size // 2, hidden_size, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        )  # 初始化卷积投影层

    def forward(self, pixel_values: Optional[torch.FloatTensor]) -> torch.Tensor:
        _, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )  # 检查通道维度是否匹配配置中设置的通道维度
        embeddings = self.projection(pixel_values)  # 对输入进行投影
        embeddings = embeddings.permute(0, 2, 3, 1)  # 调整维度顺序，使得最后一维为 hidden_size

        return embeddings


class NatDownsampler(nn.Module):
    """
    Convolutional Downsampling Layer.

    Args:
        dim (`int`):
            Number of input channels.
        norm_layer (`nn.Module`, *optional*, defaults to `nn.LayerNorm`):
            Normalization layer class.
    """

    def __init__(self, dim: int, norm_layer: nn.Module = nn.LayerNorm) -> None:
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.norm = norm_layer(2 * dim)  # 初始化规范化层

    def forward(self, input_feature: torch.Tensor) -> torch.Tensor:
        input_feature = self.reduction(input_feature.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # 执行卷积降采样
        input_feature = self.norm(input_feature)  # 应用规范化层
        return input_feature
# 定义一个函数，用于在神经网络中按概率丢弃路径（随机深度）以减少模型复杂度
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    
    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    # 如果丢弃概率为0或者不处于训练模式，则直接返回输入
    if drop_prob == 0.0 or not training:
        return input
    # 计算保留的概率
    keep_prob = 1 - drop_prob
    # 为了支持不同维度的张量，创建与输入形状相同的随机张量
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # 将随机张量二值化
    # 应用随机深度丢弃操作并返回输出
    output = input.div(keep_prob) * random_tensor
    return output


# 从transformers.models.beit.modeling_beit.BeitDropPath中复制的类，修改为NatDropPath
class NatDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 调用前面定义的drop_path函数，传递当前对象的dropout概率和训练模式
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        # 返回描述对象的额外字符串表示，显示dropout概率
        return "p={}".format(self.drop_prob)


class NeighborhoodAttention(nn.Module):
    def __init__(self, config, dim, num_heads, kernel_size):
        super().__init__()
        # 检查隐藏层大小是否能被注意力头数量整除
        if dim % num_heads != 0:
            raise ValueError(
                f"The hidden size ({dim}) is not a multiple of the number of attention heads ({num_heads})"
            )

        self.num_attention_heads = num_heads
        self.attention_head_size = int(dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.kernel_size = kernel_size

        # rpb是可学习的相对位置偏置，与Swin中的概念相同
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * self.kernel_size - 1), (2 * self.kernel_size - 1)))

        # 创建用于查询、键、值的线性变换层
        self.query = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)

        # 使用配置中的注意力概率丢弃层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        # 调整张量形状以便计算注意力分数
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 3, 1, 2, 4)
    # 定义前向传播方法，接受隐藏状态和是否输出注意力矩阵作为参数，返回元组类型的输出
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 通过查询（query）权重网络处理隐藏状态，以备计算注意力得分
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        # 通过键（key）权重网络处理隐藏状态，以备计算注意力得分
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        # 通过值（value）权重网络处理隐藏状态，以备计算注意力得分
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # 在计算注意力权重之前应用缩放因子，通常更高效，因为注意力权重通常比查询向量更大
        # 由于矩阵乘法中标量是可交换的，因此结果相同
        query_layer = query_layer / math.sqrt(self.attention_head_size)

        # 计算“查询”和“键”的归一化注意力得分，并添加相对位置偏置
        attention_scores = natten2dqkrpb(query_layer, key_layer, self.rpb, self.kernel_size, 1)

        # 将注意力得分归一化为概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 通过丢弃操作随机地“丢弃”一些注意力概率，这在原始Transformer论文中有所提及
        attention_probs = self.dropout(attention_probs)

        # 计算加权后的值（value）以生成上下文向量
        context_layer = natten2dav(attention_probs, value_layer, self.kernel_size, 1)
        # 调整上下文向量的维度顺序，以便进一步处理
        context_layer = context_layer.permute(0, 2, 3, 1, 4).contiguous()
        # 将调整后的上下文向量形状改变为全头尺寸
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # 根据是否需要输出注意力矩阵，选择性地返回上下文向量和注意力概率
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        # 返回输出元组
        return outputs
# 定义一个邻域注意力输出模块的神经网络模型
class NeighborhoodAttentionOutput(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        # 初始化一个线性层，用于将输入维度为dim的张量线性变换为维度仍为dim的输出张量
        self.dense = nn.Linear(dim, dim)
        # 初始化一个dropout层，使用config中指定的概率进行dropout操作
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将输入张量hidden_states通过线性层self.dense进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的张量进行dropout操作
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# 定义一个邻域注意力模块的神经网络模型
class NeighborhoodAttentionModule(nn.Module):
    def __init__(self, config, dim, num_heads, kernel_size):
        super().__init__()
        # 初始化一个邻域自注意力模块，使用config、dim、num_heads和kernel_size参数进行初始化
        self.self = NeighborhoodAttention(config, dim, num_heads, kernel_size)
        # 初始化一个邻域注意力输出模块，使用config和dim参数进行初始化
        self.output = NeighborhoodAttentionOutput(config, dim)
        # 初始化一个集合，用于存储需要剪枝的注意力头
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 调用find_pruneable_heads_and_indices函数，找到可剪枝的注意力头和对应的索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储剪枝的注意力头
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 调用self.self的forward方法，对隐藏状态进行自注意力操作
        self_outputs = self.self(hidden_states, output_attentions)
        # 调用self.output的forward方法，将自注意力操作的输出和输入隐藏状态作为输入计算注意力输出
        attention_output = self.output(self_outputs[0], hidden_states)
        # 构造输出元组，如果需要输出注意力，将它们加入到输出元组中
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


# 定义一个邻域中间层的神经网络模型
class NatIntermediate(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        # 初始化一个线性层，用于将输入维度为dim的张量线性变换为维度为config.mlp_ratio * dim的输出张量
        self.dense = nn.Linear(dim, int(config.mlp_ratio * dim))
        # 根据config中指定的隐藏层激活函数类型，选择对应的激活函数ACT2FN进行初始化
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入张量hidden_states通过线性层self.dense进行线性变换
        hidden_states = self.dense(hidden_states)
        # 将线性变换后的张量通过选择的激活函数self.intermediate_act_fn进行非线性变换
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# 定义一个邻域输出层的神经网络模型
class NatOutput(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        # 初始化一个线性层，用于将输入维度为config.mlp_ratio * dim的张量线性变换为维度为dim的输出张量
        self.dense = nn.Linear(int(config.mlp_ratio * dim), dim)
        # 初始化一个dropout层，使用config中指定的概率进行dropout操作
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    # 定义一个方法 `forward`，用于模型的前向传播过程，接受隐藏状态作为输入张量，并返回处理后的张量作为输出
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入张量通过全连接层 `self.dense` 进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的张量进行 dropout 操作，以防止过拟合
        hidden_states = self.dropout(hidden_states)
        # 返回经过线性变换和 dropout 处理后的张量作为输出
        return hidden_states
class NatLayer(nn.Module):
    def __init__(self, config, dim, num_heads, drop_path_rate=0.0):
        super().__init__()
        # 设置前馈分块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 设置卷积核大小
        self.kernel_size = config.kernel_size
        # 应用层归一化在注意力模块之前
        self.layernorm_before = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        # 创建注意力模块
        self.attention = NeighborhoodAttentionModule(config, dim, num_heads, kernel_size=self.kernel_size)
        # 根据丢弃路径率设置丢弃路径层
        self.drop_path = NatDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        # 应用层归一化在注意力模块之后
        self.layernorm_after = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        # 创建自然语言中间层
        self.intermediate = NatIntermediate(config, dim)
        # 创建自然语言输出层
        self.output = NatOutput(config, dim)
        # 如果配置允许，创建层缩放参数
        self.layer_scale_parameters = (
            nn.Parameter(config.layer_scale_init_value * torch.ones((2, dim)), requires_grad=True)
            if config.layer_scale_init_value > 0
            else None
        )

    def maybe_pad(self, hidden_states, height, width):
        # 设置窗口大小为卷积核大小
        window_size = self.kernel_size
        pad_values = (0, 0, 0, 0, 0, 0)
        # 如果输入的高度或宽度小于窗口大小，则进行填充
        if height < window_size or width < window_size:
            pad_l = pad_t = 0
            pad_r = max(0, window_size - width)
            pad_b = max(0, window_size - height)
            pad_values = (0, 0, pad_l, pad_r, pad_t, pad_b)
            hidden_states = nn.functional.pad(hidden_states, pad_values)
        return hidden_states, pad_values

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 获取批处理大小、高度、宽度和通道数
        batch_size, height, width, channels = hidden_states.size()
        # 保存输入的快捷连接
        shortcut = hidden_states

        # 应用层归一化在注意力模块之前
        hidden_states = self.layernorm_before(hidden_states)
        # 如果输入的大小小于卷积核大小，则进行填充
        hidden_states, pad_values = self.maybe_pad(hidden_states, height, width)

        # 获取填充后的高度和宽度
        _, height_pad, width_pad, _ = hidden_states.shape

        # 应用注意力模块
        attention_outputs = self.attention(hidden_states, output_attentions=output_attentions)

        # 获取注意力输出
        attention_output = attention_outputs[0]

        # 检查是否进行了填充
        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
            # 如果进行了填充，则裁剪注意力输出以匹配原始输入的尺寸
            attention_output = attention_output[:, :height, :width, :].contiguous()

        # 如果存在层缩放参数，则应用第一个参数到注意力输出
        if self.layer_scale_parameters is not None:
            attention_output = self.layer_scale_parameters[0] * attention_output

        # 计算最终的隐藏状态，结合快捷连接和丢弃路径
        hidden_states = shortcut + self.drop_path(attention_output)

        # 应用层归一化在注意力模块之后
        layer_output = self.layernorm_after(hidden_states)
        # 经过中间层和输出层处理的最终输出
        layer_output = self.output(self.intermediate(layer_output))

        # 如果存在层缩放参数，则应用第二个参数到最终输出
        if self.layer_scale_parameters is not None:
            layer_output = self.layer_scale_parameters[1] * layer_output

        # 结合快捷连接和丢弃路径到最终输出
        layer_output = hidden_states + self.drop_path(layer_output)

        # 返回层输出，如果需要，还返回注意力输出
        layer_outputs = (layer_output, attention_outputs[1]) if output_attentions else (layer_output,)
        return layer_outputs
    # 初始化函数，用于初始化一个神经网络模型
    def __init__(self, config, dim, depth, num_heads, drop_path_rate, downsample):
        # 调用父类的初始化函数
        super().__init__()
        # 将传入的参数保存到对象的属性中
        self.config = config  # 保存配置信息
        self.dim = dim  # 保存输入的维度信息
        # 创建神经网络层的列表，每一层是一个 NatLayer 对象
        self.layers = nn.ModuleList(
            [
                NatLayer(
                    config=config,
                    dim=dim,
                    num_heads=num_heads,
                    drop_path_rate=drop_path_rate[i],
                )
                for i in range(depth)  # 根据指定的深度创建对应数量的层
            ]
        )

        # 如果存在下采样层，则创建该下采样层对象
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=nn.LayerNorm)
        else:
            self.downsample = None  # 否则置为 None

        # 初始化一个指示变量，用于标记指向状态
        self.pointing = False

    # 前向传播函数，用于定义数据在模型中的正向流动过程
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        _, height, width, _ = hidden_states.size()  # 获取输入张量的高度和宽度信息
        # 遍历所有层，并将输入张量按顺序传递给每一层进行处理
        for i, layer_module in enumerate(self.layers):
            layer_outputs = layer_module(hidden_states, output_attentions)
            hidden_states = layer_outputs[0]  # 更新隐藏状态

        hidden_states_before_downsampling = hidden_states  # 保存下采样之前的隐藏状态
        # 如果存在下采样层，则对隐藏状态进行下采样处理
        if self.downsample is not None:
            hidden_states = self.downsample(hidden_states_before_downsampling)

        stage_outputs = (hidden_states, hidden_states_before_downsampling)  # 构造阶段输出元组

        # 如果需要输出注意力权重，则将每一层的注意力权重也加入到输出中
        if output_attentions:
            stage_outputs += layer_outputs[1:]

        return stage_outputs  # 返回阶段输出元组
class NatEncoder(nn.Module):
    # 自然编码器的类定义，继承自nn.Module
    def __init__(self, config):
        super().__init__()
        # 初始化方法，接受一个配置对象作为参数

        # 获取层级深度列表的长度
        self.num_levels = len(config.depths)
        # 将配置对象保存在self.config中
        self.config = config
        # 根据drop_path_rate参数创建一个线性间隔的列表
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]
        
        # 创建一个模块列表，每个元素是一个NatStage对象
        self.levels = nn.ModuleList(
            [
                NatStage(
                    config=config,
                    # 设置每一层的嵌入维度
                    dim=int(config.embed_dim * 2**i_layer),
                    # 设置每一层的深度
                    depth=config.depths[i_layer],
                    # 设置每一层的头数
                    num_heads=config.num_heads[i_layer],
                    # 设置每一层的drop_path_rate值
                    drop_path_rate=dpr[sum(config.depths[:i_layer]) : sum(config.depths[: i_layer + 1])],
                    # 设置是否下采样，最后一层不进行下采样
                    downsample=NatDownsampler if (i_layer < self.num_levels - 1) else None,
                )
                # 对每一个层级进行循环
                for i_layer in range(self.num_levels)
            ]
        )

    # 前向传播方法定义
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        output_hidden_states_before_downsampling: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, NatEncoderOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_reshaped_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if output_hidden_states:
            # 将隐藏状态重排列为 b h w c -> b c h w
            reshaped_hidden_state = hidden_states.permute(0, 3, 1, 2)
            # 将当前隐藏状态添加到所有隐藏状态元组中
            all_hidden_states += (hidden_states,)
            # 将重排列后的隐藏状态添加到所有重排列隐藏状态元组中
            all_reshaped_hidden_states += (reshaped_hidden_state,)

        for i, layer_module in enumerate(self.levels):
            # 调用每个层模块处理隐藏状态和注意力输出
            layer_outputs = layer_module(hidden_states, output_attentions)

            # 更新当前隐藏状态为当前层的隐藏状态输出
            hidden_states = layer_outputs[0]
            # 更新在下采样之前的隐藏状态为当前层的下采样前隐藏状态输出
            hidden_states_before_downsampling = layer_outputs[1]

            if output_hidden_states and output_hidden_states_before_downsampling:
                # 将下采样前的隐藏状态重排列为 b h w c -> b c h w
                reshaped_hidden_state = hidden_states_before_downsampling.permute(0, 3, 1, 2)
                # 将下采样前的隐藏状态添加到所有隐藏状态元组中
                all_hidden_states += (hidden_states_before_downsampling,)
                # 将重排列后的下采样前的隐藏状态添加到所有重排列隐藏状态元组中
                all_reshaped_hidden_states += (reshaped_hidden_state,)
            elif output_hidden_states and not output_hidden_states_before_downsampling:
                # 将当前隐藏状态重排列为 b h w c -> b c h w
                reshaped_hidden_state = hidden_states.permute(0, 3, 1, 2)
                # 将当前隐藏状态添加到所有隐藏状态元组中
                all_hidden_states += (hidden_states,)
                # 将重排列后的当前隐藏状态添加到所有重排列隐藏状态元组中
                all_reshaped_hidden_states += (reshaped_hidden_state,)

            if output_attentions:
                # 将当前层的注意力输出添加到所有自注意力元组中
                all_self_attentions += layer_outputs[2:]

        if not return_dict:
            # 如果不返回字典形式的输出，则返回非空元组的组成部分
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        # 返回按照 NatEncoderOutput 结构组织的输出字典
        return NatEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            reshaped_hidden_states=all_reshaped_hidden_states,
        )
class NatPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定配置类为 NatConfig
    config_class = NatConfig
    # 基础模型前缀设定为 "nat"
    base_model_prefix = "nat"
    # 主输入名称设定为 "pixel_values"
    main_input_name = "pixel_values"

    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果是线性层或卷积层，使用正态分布初始化权重
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 与 TF 版本稍有不同，TF 使用截断正态分布进行初始化
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置，初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是 LayerNorm 层，初始化偏置为零，权重为全1
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


NAT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`NatConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# NatModel 类的文档字符串
NAT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`ViTImageProcessor.__call__`]
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

# 使用 add_start_docstrings 装饰 NatModel 类，添加其描述信息和参数文档字符串
@add_start_docstrings(
    "The bare Nat Model transformer outputting raw hidden-states without any specific head on top.",
    NAT_START_DOCSTRING,
)
class NatModel(NatPreTrainedModel):
    pass
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)

        # 确保模型需要的后端库已经加载
        requires_backends(self, ["natten"])

        # 设置配置信息
        self.config = config
        # 确定模型深度的层数
        self.num_levels = len(config.depths)
        # 计算特征向量的维度
        self.num_features = int(config.embed_dim * 2 ** (self.num_levels - 1))

        # 初始化自然语言嵌入层
        self.embeddings = NatEmbeddings(config)
        # 初始化自然语言编码器
        self.encoder = NatEncoder(config)

        # 使用指定的层归一化函数初始化
        self.layernorm = nn.LayerNorm(self.num_features, eps=config.layer_norm_eps)
        # 如果需要添加池化层，则初始化自适应平均池化层
        self.pooler = nn.AdaptiveAvgPool1d(1) if add_pooling_layer else None

        # 初始化权重并进行最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 返回嵌入层中的补丁嵌入
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 遍历需要修剪的层和对应的头部信息
        for layer, heads in heads_to_prune.items():
            # 在编码器中修剪指定层的注意力头
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(NAT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=NatModelOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 输出注意力张量，如果未指定则使用配置中的输出注意力设置
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    # 输出隐藏状态张量，如果未指定则使用配置中的输出隐藏状态设置
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    # 返回字典标志，如果未指定则使用配置中的返回字典设置
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # 如果未提供像素值，则抛出数值错误
    if pixel_values is None:
        raise ValueError("You have to specify pixel_values")

    # 将像素值嵌入到嵌入层中
    embedding_output = self.embeddings(pixel_values)

    # 编码器处理嵌入输出
    encoder_outputs = self.encoder(
        embedding_output,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    # 获取编码器的序列输出
    sequence_output = encoder_outputs[0]
    # 应用层归一化到序列输出
    sequence_output = self.layernorm(sequence_output)

    # 初始化池化输出为 None
    pooled_output = None
    # 如果存在池化器，则对序列输出进行池化操作
    if self.pooler is not None:
        pooled_output = self.pooler(sequence_output.flatten(1, 2).transpose(1, 2))
        pooled_output = torch.flatten(pooled_output, 1)

    # 如果不使用返回字典形式
    if not return_dict:
        # 返回序列输出，池化输出以及可能的其他编码器输出
        output = (sequence_output, pooled_output) + encoder_outputs[1:]
        return output

    # 如果使用返回字典形式，则返回自定义的模型输出对象
    return NatModelOutput(
        last_hidden_state=sequence_output,
        pooler_output=pooled_output,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
        reshaped_hidden_states=encoder_outputs.reshaped_hidden_states,
    )
# 使用装饰器为类添加文档字符串，描述其作为 Nat 模型变换器和图像分类头的功能，该头部是在 [CLS] 标记的最终隐藏状态之上的线性层
@add_start_docstrings(
    """
    Nat Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.
    """,
    NAT_START_DOCSTRING,
)
class NatForImageClassification(NatPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 检查所需的后端是否已加载
        requires_backends(self, ["natten"])

        # 初始化分类器的标签数量
        self.num_labels = config.num_labels
        # 创建 NatModel 实例
        self.nat = NatModel(config)

        # 分类器头部
        self.classifier = (
            nn.Linear(self.nat.num_features, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # 初始化权重并应用最终处理
        self.post_init()

    # 使用装饰器添加模型前向方法的文档字符串，描述其输入参数
    @add_start_docstrings_to_model_forward(NAT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=NatImageClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 如果 return_dict 不为 None，则使用 return_dict；否则使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用自然语言处理模型进行处理，根据参数设置输出注意力权重和隐藏状态
        outputs = self.nat(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取池化后的输出，通常用于分类任务
        pooled_output = outputs[1]

        # 使用分类器对池化后的输出进行分类，得到预测的 logits
        logits = self.classifier(pooled_output)

        # 初始化损失为 None
        loss = None

        # 如果提供了标签 labels，则计算损失
        if labels is not None:
            # 如果问题类型未指定，则根据标签类型确定问题类型
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型选择对应的损失函数
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    # 对于单标签回归任务，计算均方误差损失
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    # 对于多标签回归任务，计算均方误差损失
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                # 对于单标签分类任务，计算交叉熵损失
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # 对于多标签分类任务，计算二元交叉熵损失
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 如果不需要返回字典形式的输出，则按照元组的形式返回结果
        if not return_dict:
            output = (logits,) + outputs[2:]  # outputs[2:] 包含额外的隐藏状态信息
            return ((loss,) + output) if loss is not None else output

        # 返回自定义的输出类 NatImageClassifierOutput，包含损失、logits、隐藏状态和注意力权重等信息
        return NatImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            reshaped_hidden_states=outputs.reshaped_hidden_states,
        )
# 使用装饰器添加文档字符串，描述这个类的作用是提供 NAT 的主干结构，可用于 DETR 和 MaskFormer 等框架。
# NAT_START_DOCSTRING 是预定义的一部分文档字符串内容。
@add_start_docstrings(
    "NAT backbone, to be used with frameworks like DETR and MaskFormer.",
    NAT_START_DOCSTRING,
)
class NatBackbone(NatPreTrainedModel, BackboneMixin):
    def __init__(self, config):
        # 调用父类的构造函数，初始化 NAT 模型的配置
        super().__init__(config)
        # 调用父类的方法，初始化主干结构
        super()._init_backbone(config)

        # 检查并确保需要的后端支持模块 "natten" 已经加载
        requires_backends(self, ["natten"])

        # 初始化嵌入层
        self.embeddings = NatEmbeddings(config)
        # 初始化编码器
        self.encoder = NatEncoder(config)
        
        # 计算每个特征层的通道数，并将其保存在列表中
        self.num_features = [config.embed_dim] + [int(config.embed_dim * 2**i) for i in range(len(config.depths))]

        # 为输出特征层的隐藏状态添加层归一化层
        hidden_states_norms = {}
        for stage, num_channels in zip(self.out_features, self.channels):
            hidden_states_norms[stage] = nn.LayerNorm(num_channels)
        self.hidden_states_norms = nn.ModuleDict(hidden_states_norms)

        # 执行初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入层的方法
    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    # 重写 forward 方法，添加输入文档字符串和返回值文档字符串，指定了输入参数和返回输出类型
    @add_start_docstrings_to_model_forward(NAT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BackboneOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ):
        """
        根据输入的参数返回BackboneOutput对象。

        Parameters:
            return_dict (bool, optional): 控制是否返回字典形式的输出，默认从self.config.use_return_dict获取。
            output_hidden_states (bool, optional): 控制是否输出隐藏状态，默认从self.config.output_hidden_states获取。
            output_attentions (bool, optional): 控制是否输出注意力，默认从self.config.output_attentions获取。

        Returns:
            BackboneOutput: 包含特征图、隐藏状态和注意力的输出对象。

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoBackbone
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> processor = AutoImageProcessor.from_pretrained("shi-labs/nat-mini-in1k-224")
        >>> model = AutoBackbone.from_pretrained(
        ...     "shi-labs/nat-mini-in1k-224", out_features=["stage1", "stage2", "stage3", "stage4"]
        ... )

        >>> inputs = processor(image, return_tensors="pt")

        >>> outputs = model(**inputs)

        >>> feature_maps = outputs.feature_maps
        >>> list(feature_maps[-1].shape)
        [1, 512, 7, 7]
        ```
        """
        # 如果未提供return_dict参数，则使用self.config.use_return_dict的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果未提供output_hidden_states参数，则使用self.config.output_hidden_states的设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未提供output_attentions参数，则使用self.config.output_attentions的设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        # 对输入的像素值进行嵌入处理，获取嵌入输出
        embedding_output = self.embeddings(pixel_values)

        # 将嵌入输出传入编码器，获取编码器的输出
        outputs = self.encoder(
            embedding_output,
            output_attentions=output_attentions,
            output_hidden_states=True,
            output_hidden_states_before_downsampling=True,
            return_dict=True,
        )

        # 获取重塑后的隐藏状态
        hidden_states = outputs.reshaped_hidden_states

        # 初始化特征图为空元组
        feature_maps = ()
        # 遍历阶段名称和隐藏状态，将符合输出特征要求的阶段的处理后的隐藏状态添加到特征图中
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                # TODO can we simplify this? 可以简化这部分代码吗？
                batch_size, num_channels, height, width = hidden_state.shape
                hidden_state = hidden_state.permute(0, 2, 3, 1).contiguous()
                hidden_state = hidden_state.view(batch_size, height * width, num_channels)
                hidden_state = self.hidden_states_norms[stage](hidden_state)
                hidden_state = hidden_state.view(batch_size, height, width, num_channels)
                hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()
                feature_maps += (hidden_state,)

        # 如果不需要返回字典形式的输出，则构造输出元组
        if not return_dict:
            output = (feature_maps,)
            if output_hidden_states:
                output += (outputs.hidden_states,)
            return output

        # 返回BackboneOutput对象，包含特征图、隐藏状态和注意力
        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )
```