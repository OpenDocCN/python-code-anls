# `.\models\vit_mae\modeling_vit_mae.py`

```py
# coding=utf-8
# Copyright 2022 Facebook AI and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch ViT MAE (masked autoencoder) model."""


import collections.abc
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN  # 导入激活函数映射表
from ...modeling_outputs import BaseModelOutput  # 导入基础模型输出类
from ...modeling_utils import PreTrainedModel  # 导入预训练模型基类
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer  # 导入相关的PyTorch工具函数
from ...utils import (  # 导入常用工具函数和类
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_vit_mae import ViTMAEConfig  # 导入ViT MAE模型的配置类


logger = logging.get_logger(__name__)  # 获取日志记录器

_CONFIG_FOR_DOC = "ViTMAEConfig"  # 用于文档的配置类名称
_CHECKPOINT_FOR_DOC = "facebook/vit-mae-base"  # 用于文档的预训练模型名称

VIT_MAE_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/vit-mae-base",
    # See all ViTMAE models at https://huggingface.co/models?filter=vit_mae
]
    # `last_hidden_state`参数：模型最后一层的隐藏状态输出，形状为`(batch_size, sequence_length, hidden_size)`
    last_hidden_state: torch.FloatTensor = None
    
    # `mask`参数：指示哪些补丁被屏蔽（1）和哪些没有（0）的张量，形状为`(batch_size, sequence_length)`
    mask: torch.LongTensor = None
    
    # `ids_restore`参数：包含（打乱后的）屏蔽补丁的原始索引的张量，形状为`(batch_size, sequence_length)`
    ids_restore: torch.LongTensor = None
    
    # `hidden_states`参数（可选）：元组的`torch.FloatTensor`（如果`output_hidden_states=True`或`config.output_hidden_states=True`时返回），
    # 包含模型每一层的隐藏状态输出加上初始嵌入输出，形状为`(batch_size, sequence_length, hidden_size)`
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    
    # `attentions`参数（可选）：元组的`torch.FloatTensor`（如果`output_attentions=True`或`config.output_attentions=True`时返回），
    # 包含每一层的注意力权重，形状为`(batch_size, num_heads, sequence_length, sequence_length)`。
    # 这些是经过注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
    attentions: Optional[Tuple[torch.FloatTensor]] = None
@dataclass
class ViTMAEDecoderOutput(ModelOutput):
    """
    ViTMAEDecoder的输出类，包含潜在的隐藏状态和注意力。

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, patch_size ** 2 * num_channels)`):
            像素重建的logits。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, 当 `output_hidden_states=True` 时返回或当 `config.output_hidden_states=True` 时返回):
            `torch.FloatTensor` 的元组（一个用于嵌入的输出 + 每层的输出），形状为 `(batch_size, sequence_length, hidden_size)`。
            模型每层输出的隐藏状态以及初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, 当 `output_attentions=True` 时返回或当 `config.output_attentions=True` 时返回):
            `torch.FloatTensor` 的元组（每层一个）形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
            经过注意力softmax后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class ViTMAEForPreTrainingOutput(ModelOutput):
    """
    ViTMAEForPreTraining的输出类，包含潜在的隐藏状态和注意力。

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`):
            像素重建损失。
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, patch_size ** 2 * num_channels)`):
            像素重建的logits。
        mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            指示哪些补丁被屏蔽（1）哪些没有（0）的张量。
        ids_restore (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            包含（打乱的）屏蔽补丁的原始索引的张量。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, 当 `output_hidden_states=True` 时返回或当 `config.output_hidden_states=True` 时返回):
            `torch.FloatTensor` 的元组（一个用于嵌入的输出 + 每层的输出），形状为 `(batch_size, sequence_length, hidden_size)`。
            模型每层输出的隐藏状态以及初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, 当 `output_attentions=True` 时返回或当 `config.output_attentions=True` 时返回):
            `torch.FloatTensor` 的元组（每层一个）形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
            经过注意力softmax后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    loss: Optional[torch.FloatTensor] = None
    # 定义变量 logits，类型为 torch.FloatTensor，初始值为 None
    logits: torch.FloatTensor = None
    # 定义变量 mask，类型为 torch.LongTensor，初始值为 None
    mask: torch.LongTensor = None
    # 定义变量 ids_restore，类型为 torch.LongTensor，初始值为 None
    ids_restore: torch.LongTensor = None
    # 定义变量 hidden_states，类型为 Optional[Tuple[torch.FloatTensor]]，初始值为 None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义变量 attentions，类型为 Optional[Tuple[torch.FloatTensor]]，初始值为 None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
def get_2d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    """
    Create 2D sin/cos positional embeddings.

    Args:
        embed_dim (`int`):
            Embedding dimension.
        grid_size (`int`):
            The grid height and width.
        add_cls_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add a classification (CLS) token.

    Returns:
        (`torch.FloatTensor` of shape (grid_size*grid_size, embed_dim) or (1+grid_size*grid_size, embed_dim): the
        position embeddings (with or without classification token)
    """
    # Generate a grid of height and width using numpy arrays
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    # Reshape the grid to prepare for positional embeddings calculation
    grid = grid.reshape([2, 1, grid_size, grid_size])
    # Compute positional embeddings using grid coordinates
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    
    # Optionally add a classification token to the positional embeddings
    if add_cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    Generate 2D sin/cos positional embeddings from grid coordinates.

    Args:
        embed_dim (`int`):
            Embedding dimension.
        grid (`numpy.ndarray`):
            Grid coordinates of shape (2, 1, grid_size, grid_size).

    Returns:
        (`numpy.ndarray`): Positional embeddings of shape (grid_size*grid_size, embed_dim)
    """
    # Ensure embedding dimension is even
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    # Generate sin/cos positional embeddings separately for height and width
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    # Concatenate embeddings for height and width to form 2D embeddings
    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Generate 1D sin/cos positional embeddings from positions.

    Args:
        embed_dim (`int`):
            Embedding dimension.
        pos (`numpy.ndarray`):
            Positions to be encoded, shape (M,).

    Returns:
        (`numpy.ndarray`): Positional embeddings of shape (M, embed_dim)
    """
    # Ensure embedding dimension is even
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    # Generate frequencies for sin/cos functions
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    # Reshape positions for matrix multiplication
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    # Compute sin and cos embeddings
    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    # Concatenate sin and cos embeddings
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class ViTMAEEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.

    """

    def __init__(self, config):
        """
        Initializes ViTMAEEmbeddings module.

        Args:
            config (`object`):
                Configuration object containing model parameters.
        """
        super().__init__()

        # Define CLS token as a learnable parameter
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        # Initialize patch embeddings using ViTMAEPatchEmbeddings module
        self.patch_embeddings = ViTMAEPatchEmbeddings(config)
        # Obtain number of patches from patch_embeddings
        self.num_patches = self.patch_embeddings.num_patches
        # Fixed sin-cos positional embeddings for patches and CLS token
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, config.hidden_size), requires_grad=False
        )
        self.config = config
        self.initialize_weights()

    def initialize_weights(self):
        """
        Initialize weights of the module.
        """
        # Implementation details for weight initialization can be added here
        pass
    def initialize_weights(self):
        # 使用 sin-cos 嵌入初始化（并冻结）位置嵌入
        pos_embed = get_2d_sincos_pos_embed(
            self.position_embeddings.shape[-1], int(self.patch_embeddings.num_patches**0.5), add_cls_token=True
        )
        self.position_embeddings.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # 使用类似 nn.Linear 的方式初始化 patch_embeddings（而不是 nn.Conv2d）
        w = self.patch_embeddings.projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm 的 trunc_normal_(std=.02) 实际上是 normal_(std=0.02)，因为截断过大（2.）
        torch.nn.init.normal_(self.cls_token, std=self.config.initializer_range)

    def random_masking(self, sequence, noise=None):
        """
        执行每个样本的随机掩码操作，通过每个样本的排序随机噪声来进行。排序随机噪声通过 argsort 实现。

        Args:
            sequence (`torch.LongTensor` of shape `(batch_size, sequence_length, dim)`)
                输入序列张量
            noise (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *可选*) 
                主要用于测试目的，控制随机性以及保持可重现性
        """
        batch_size, seq_length, dim = sequence.shape
        len_keep = int(seq_length * (1 - self.config.mask_ratio))

        if noise is None:
            noise = torch.rand(batch_size, seq_length, device=sequence.device)  # 噪声范围在 [0, 1]

        # 对每个样本的噪声进行排序
        ids_shuffle = torch.argsort(noise, dim=1)  # 升序：小的保留，大的移除
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # 保留第一个子集
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_unmasked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))

        # 生成二进制掩码：0 表示保留，1 表示移除
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        # 解除排序以获得二进制掩码
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return sequence_unmasked, mask, ids_restore

    def forward(self, pixel_values, noise=None):
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values)

        # 添加位置嵌入，不包括 cls token
        embeddings = embeddings + self.position_embeddings[:, 1:, :]

        # 掩码操作：长度变为 length * config.mask_ratio
        embeddings, mask, ids_restore = self.random_masking(embeddings, noise)

        # 添加 cls token
        cls_token = self.cls_token + self.position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        return embeddings, mask, ids_restore
# 定义一个名为 ViTMAEPatchEmbeddings 的类，继承自 nn.Module，用于将形状为 `(batch_size, num_channels, height, width)` 的像素值转换成形状为 `(batch_size, seq_length, hidden_size)` 的初始隐藏状态（patch embeddings），以供 Transformer 使用。
class ViTMAEPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        # 从配置中获取图像大小和patch大小
        image_size, patch_size = config.image_size, config.patch_size
        # 从配置中获取通道数和隐藏大小
        num_channels, hidden_size = config.num_channels, config.hidden_size
        # 如果图像大小和patch大小不是可迭代对象，则转换为元组
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        # 计算图像中的patch数量
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        # 使用 nn.Conv2d 定义投影层，将输入通道数转换为隐藏大小，使用patch大小的卷积核和步幅
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values):
        # 获取输入张量的维度信息
        batch_size, num_channels, height, width = pixel_values.shape
        # 如果输入通道数与配置中的不匹配，则抛出 ValueError 异常
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # 如果输入图像尺寸与配置中的不匹配，则抛出 ValueError 异常
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        # 将输入张量通过投影层进行处理，然后展平成二维张量，并交换维度顺序
        x = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return x


# 从 transformers.models.vit.modeling_vit.ViTSelfAttention 模型复制并重命名为 ViTMAESelfAttention
class ViTMAESelfAttention(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        # 检查隐藏大小是否可以被注意力头数整除，如果不是，则抛出 ValueError 异常
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        # 初始化注意力头数和每个注意力头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 定义查询、键和值的线性变换层
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        # 定义 dropout 层，用于注意力概率的随机丢弃
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    # 将输入张量 x 进行维度重塑，以便进行注意力计算
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 模型的前向传播方法
    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 计算混合查询向量
        mixed_query_layer = self.query(hidden_states)

        # 对键值对进行维度重塑，为了计算注意力得分
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 计算注意力得分，即查询向量与键向量的点积
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # 缩放注意力得分
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 将注意力得分归一化为概率分布
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 对注意力概率进行 dropout 处理
        attention_probs = self.dropout(attention_probs)

        # 如果指定了头部掩码，应用头部掩码
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算上下文向量，即注意力概率与值向量的加权和
        context_layer = torch.matmul(attention_probs, value_layer)

        # 将上下文向量维度重塑为 [batch_size, seq_length, all_head_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # 准备输出结果，包括上下文层和（可选的）注意力概率
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
# Copied from transformers.models.vit.modeling_vit.ViTSelfOutput with ViT->ViTMAE
class ViTMAESelfOutput(nn.Module):
    """
    The residual connection is defined in ViTMAELayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        # 定义一个全连接层，将输入特征空间转换为隐藏状态大小
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义一个dropout层，以减少过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 输入经过全连接层
        hidden_states = self.dense(hidden_states)
        # 经过dropout层
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTAttention with ViT->ViTMAE
class ViTMAEAttention(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        # 创建一个ViTMAESelfAttention对象
        self.attention = ViTMAESelfAttention(config)
        # 创建一个ViTMAESelfOutput对象
        self.output = ViTMAESelfOutput(config)
        # 初始化一个空集合，用于存储需要被修剪的注意力头
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        # 寻找需要被修剪的头的索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # 修剪线性层
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储修剪后的注意力头
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 调用self.attention进行自注意力计算
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        # 将self输出传递给self.output进行进一步处理
        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # 如果需要输出注意力权重，则添加到outputs中
        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTIntermediate ViT->ViTMAE
class ViTMAEIntermediate(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        # 创建一个线性层，将隐藏状态大小转换为中间层大小
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果隐藏激活函数是字符串，则使用相应的激活函数映射；否则使用给定的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
    # 定义一个前向传播方法，接收隐藏状态作为输入张量，并返回处理后的张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用全连接层对隐藏状态进行线性变换
        hidden_states = self.dense(hidden_states)
        # 应用激活函数到线性变换后的隐藏状态张量
        hidden_states = self.intermediate_act_fn(hidden_states)

        # 返回处理后的隐藏状态张量作为输出
        return hidden_states
# 从 transformers.models.vit.modeling_vit.ViTOutput 复制而来，被重命名为 ViTMAEOutput
class ViTMAEOutput(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        # 创建一个全连接层，将输入维度转换为 config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 定义一个 dropout 层，使用 config.hidden_dropout_prob 的概率进行随机失活
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将输入的 hidden_states 通过全连接层 dense 进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的结果进行 dropout 操作
        hidden_states = self.dropout(hidden_states)

        # 将 dropout 后的结果与输入的 input_tensor 相加，实现残差连接
        hidden_states = hidden_states + input_tensor

        return hidden_states


# 从 transformers.models.vit.modeling_vit.ViTLayer 复制而来，被重命名为 ViTMAELayer
class ViTMAELayer(nn.Module):
    """对应 timm 实现中的 Block 类。"""

    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        # 设置 chunk_size_feed_forward 和 seq_len_dim 参数
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        # 初始化 self-attention、中间层和输出层
        self.attention = ViTMAEAttention(config)
        self.intermediate = ViTMAEIntermediate(config)
        self.output = ViTMAEOutput(config)
        # 在 self-attention 前应用 layernorm
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 在 self-attention 后再次应用 layernorm
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 在 ViTMAE 中，在 self-attention 前应用 layernorm
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # 如果输出注意力权重，则添加自注意力权重

        # 第一个残差连接
        hidden_states = attention_output + hidden_states

        # 在 ViTMAE 中，self-attention 后再次应用 layernorm
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # 第二个残差连接在此处完成
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


# 从 transformers.models.vit.modeling_vit.ViTEncoder 复制而来，被重命名为 ViTMAEEncoder
class ViTMAEEncoder(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.config = config
        # 使用 ViTMAELayer 构建层的列表，重复 config.num_hidden_layers 次
        self.layer = nn.ModuleList([ViTMAELayer(config) for _ in range(config.num_hidden_layers)])
        # 设置梯度检查点为 False
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    # 定义函数的返回类型为一个元组或BaseModelOutput类型
    ) -> Union[tuple, BaseModelOutput]:
        # 如果不输出隐藏状态，则初始化为空元组；否则为None
        all_hidden_states = () if output_hidden_states else None
        # 如果不输出注意力权重，则初始化为空元组；否则为None
        all_self_attentions = () if output_attentions else None
    
        # 遍历模型的每一层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态
            if output_hidden_states:
                # 将当前层的隐藏状态添加到all_hidden_states元组中
                all_hidden_states = all_hidden_states + (hidden_states,)
    
            # 如果给定了head_mask，则使用当前层的head_mask；否则为None
            layer_head_mask = head_mask[i] if head_mask is not None else None
    
            # 如果开启了梯度检查点并且处于训练模式
            if self.gradient_checkpointing and self.training:
                # 使用梯度检查点函数来调用当前层的forward方法
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                # 否则直接调用当前层的forward方法
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)
    
            # 获取当前层的输出隐藏状态
            hidden_states = layer_outputs[0]
    
            # 如果需要输出注意力权重
            if output_attentions:
                # 将当前层的注意力权重添加到all_self_attentions元组中
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
    
        # 如果需要输出隐藏状态
        if output_hidden_states:
            # 将最终的隐藏状态添加到all_hidden_states元组中
            all_hidden_states = all_hidden_states + (hidden_states,)
    
        # 如果不需要返回字典格式的结果
        if not return_dict:
            # 返回非None的元组，包括隐藏状态、所有隐藏状态和所有注意力权重
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        
        # 否则，返回BaseModelOutput类型的对象，包括最终的隐藏状态、所有隐藏状态和所有注意力权重
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
# 使用 `add_start_docstrings` 装饰器为 `ViTMAEModel` 类添加文档字符串，描述其为一个输出原始隐藏状态的 ViTMAE 模型变压器，没有特定输出头部。
# 包含关于模型使用和行为的一般信息，建议用户参考 PyTorch 文档进行详细了解。

@add_start_docstrings(
    "The bare ViTMAE Model transformer outputting raw hidden-states without any specific head on top.",
    VIT_MAE_START_DOCSTRING,
)
class ViTMAEModel(ViTMAEPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置配置类，用于模型配置参数的初始化
    config_class = ViTMAEConfig
    # 基础模型前缀，用于标识模型
    base_model_prefix = "vit"
    # 主要输入名称，指定模型的主要输入是像素值
    main_input_name = "pixel_values"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        # 初始化模型权重
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 对于线性层和卷积层，使用正态分布初始化权重，均值为 0，标准差为配置文件中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置，则将偏置初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            # 对于 LayerNorm 层，初始化偏置为零，权重为 1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    # 初始化函数，接受一个配置参数 config
    def __init__(self, config):
        # 调用父类的初始化函数，传入配置参数 config
        super().__init__(config)
        # 将配置参数 config 存储在对象的属性中
        self.config = config

        # 初始化 ViTMAE 模型的嵌入层和编码器
        self.embeddings = ViTMAEEmbeddings(config)
        self.encoder = ViTMAEEncoder(config)

        # 初始化 LayerNorm 层，用于规范化隐藏层的输出
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 调用模型的后初始化方法，初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入层的方法
    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    # 剪枝模型中的注意力头部方法
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 遍历需要剪枝的层及其对应的注意力头部列表
        for layer, heads in heads_to_prune.items():
            # 调用编码器中特定层的注意力机制对象的剪枝方法
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 前向传播方法，实现模型的推理过程
    @add_start_docstrings_to_model_forward(VIT_MAE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ViTMAEModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        noise: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果 output_attentions 不为 None，则使用其值；否则使用模型配置中的 output_attentions 值

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果 output_hidden_states 不为 None，则使用其值；否则使用模型配置中的 output_hidden_states 值

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果 return_dict 不为 None，则使用其值；否则使用模型配置中的 use_return_dict 值

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        # 如果 pixel_values 为 None，则抛出 ValueError 异常提示需要指定 pixel_values

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        # 根据需要准备头部掩码
        # head_mask 是一个形状为 [num_hidden_layers x batch x num_heads x seq_length x seq_length] 的掩码数组

        embedding_output, mask, ids_restore = self.embeddings(pixel_values, noise=noise)
        # 将 pixel_values 通过 embeddings 方法转换为嵌入输出 embedding_output，同时生成 mask 和 ids_restore

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 使用编码器处理嵌入输出，可以选择传入头部掩码、注意力输出、隐藏状态输出和是否使用返回字典

        sequence_output = encoder_outputs[0]
        # 从编码器输出中获取序列输出

        sequence_output = self.layernorm(sequence_output)
        # 应用 layernorm 对序列输出进行归一化处理

        if not return_dict:
            return (sequence_output, mask, ids_restore) + encoder_outputs[1:]
        # 如果不使用返回字典，则返回序列输出、mask、ids_restore，以及编码器输出的其余部分

        return ViTMAEModelOutput(
            last_hidden_state=sequence_output,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
        # 使用 ViTMAEModelOutput 封装输出，包括最终隐藏状态、mask、ids_restore、隐藏状态数组和注意力数组
    # ViTMAE 解码器模型类，继承自 nn.Module
    class ViTMAEDecoder(nn.Module):
        def __init__(self, config, num_patches):
            super().__init__()
            # 初始化解码器的嵌入层，将隐藏大小转换为解码器隐藏大小
            self.decoder_embed = nn.Linear(config.hidden_size, config.decoder_hidden_size, bias=True)
            # 定义掩码令牌作为可学习参数
            self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_hidden_size))
            # 初始化解码器位置嵌入，使用固定的正弦-余弦嵌入
            self.decoder_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + 1, config.decoder_hidden_size), requires_grad=False
            )  # fixed sin-cos embedding

            # 复制配置并调整以用于解码器
            decoder_config = deepcopy(config)
            decoder_config.hidden_size = config.decoder_hidden_size
            decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
            decoder_config.num_attention_heads = config.decoder_num_attention_heads
            decoder_config.intermediate_size = config.decoder_intermediate_size
            # 创建解码器层列表
            self.decoder_layers = nn.ModuleList(
                [ViTMAELayer(decoder_config) for _ in range(config.decoder_num_hidden_layers)]
            )

            # 初始化解码器层归一化层
            self.decoder_norm = nn.LayerNorm(config.decoder_hidden_size, eps=config.layer_norm_eps)
            # 定义解码器的预测线性层，将解码器隐藏大小映射为图像块的像素数和通道数
            self.decoder_pred = nn.Linear(
                config.decoder_hidden_size, config.patch_size**2 * config.num_channels, bias=True
            )  # encoder to decoder
            # 是否使用梯度检查点，默认为 False
            self.gradient_checkpointing = False
            # 存储模型配置
            self.config = config
            # 初始化权重
            self.initialize_weights(num_patches)

        def initialize_weights(self, num_patches):
            # 使用正弦-余弦嵌入初始化（并冻结）位置嵌入
            decoder_pos_embed = get_2d_sincos_pos_embed(
                self.decoder_pos_embed.shape[-1], int(num_patches**0.5), add_cls_token=True
            )
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

            # 使用 timm 的截断正态分布初始化掩码令牌
            # timm's trunc_normal_(std=.02) 实际上相当于 normal_(std=0.02)，因为截断值太大（2.）
            torch.nn.init.normal_(self.mask_token, std=self.config.initializer_range)

        def forward(
            self,
            hidden_states,
            ids_restore,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        # embed tokens
        # 使用解码器的嵌入层将隐藏状态转换为嵌入表示
        x = self.decoder_embed(hidden_states)

        # append mask tokens to sequence
        # 将掩码标记追加到序列中
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        # 根据恢复的标识索引重新排列张量
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        # 添加位置嵌入
        hidden_states = x + self.decoder_pos_embed

        # apply Transformer layers (blocks)
        # 应用 Transformer 层（块）
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.decoder_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                # 使用梯度检查点函数处理层调用（用于内存效率）
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    None,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, head_mask=None, output_attentions=output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # normalize output using layer norm
        # 使用层归一化对隐藏状态进行标准化
        hidden_states = self.decoder_norm(hidden_states)

        # predictor projection
        # 预测器投影
        logits = self.decoder_pred(hidden_states)

        # remove cls token
        # 移除 cls 标记
        logits = logits[:, 1:, :]

        if not return_dict:
            # 如果不返回字典形式的输出，按顺序返回结果元组
            return tuple(v for v in [logits, all_hidden_states, all_self_attentions] if v is not None)
        # 返回 ViTMAEDecoderOutput 对象，包含 logits、hidden_states 和 attentions
        return ViTMAEDecoderOutput(
            logits=logits,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
@add_start_docstrings(
    """The ViTMAE Model transformer with the decoder on top for self-supervised pre-training.

    <Tip>

    Note that we provide a script to pre-train this model on custom data in our [examples
    directory](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining).

    </Tip>

    """,
    VIT_MAE_START_DOCSTRING,
)
class ViTMAEForPreTraining(ViTMAEPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Initialize the ViTMAE model with the provided configuration
        self.vit = ViTMAEModel(config)

        # Initialize the ViTMAE decoder using the config and number of patches from the embeddings
        self.decoder = ViTMAEDecoder(config, num_patches=self.vit.embeddings.num_patches)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        # Return the patch embeddings from the ViTMAE model's embeddings
        return self.vit.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model.

        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            # Prune heads in the attention mechanism of the encoder layer
            self.encoder.layer[layer].attention.prune_heads(heads)

    def patchify(self, pixel_values):
        """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Patchified pixel values.
        """
        patch_size, num_channels = self.config.patch_size, self.config.num_channels
        
        # Perform sanity checks on pixel values
        if (pixel_values.shape[2] != pixel_values.shape[3]) or (pixel_values.shape[2] % patch_size != 0):
            raise ValueError("Make sure the pixel values have a squared size that is divisible by the patch size")
        if pixel_values.shape[1] != num_channels:
            raise ValueError(
                "Make sure the number of channels of the pixel values is equal to the one set in the configuration"
            )

        # Patchify the pixel values
        batch_size = pixel_values.shape[0]
        num_patches_one_direction = pixel_values.shape[2] // patch_size
        patchified_pixel_values = pixel_values.reshape(
            batch_size, num_channels, num_patches_one_direction, patch_size, num_patches_one_direction, patch_size
        )
        patchified_pixel_values = torch.einsum("nchpwq->nhwpqc", patchified_pixel_values)
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size, num_patches_one_direction * num_patches_one_direction, patch_size**2 * num_channels
        )
        return patchified_pixel_values
    def unpatchify(self, patchified_pixel_values):
        """
        Args:
            patchified_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Patchified pixel values.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`:
                Pixel values.
        """
        # 从配置中获取补丁大小和通道数
        patch_size, num_channels = self.config.patch_size, self.config.num_channels
        
        # 计算单个方向上的补丁数量
        num_patches_one_direction = int(patchified_pixel_values.shape[1] ** 0.5)
        
        # 检查补丁数量是否可以完全平方
        if num_patches_one_direction**2 != patchified_pixel_values.shape[1]:
            raise ValueError("Make sure that the number of patches can be squared")
        
        # 对补丁化的像素值进行重塑，以进行反补丁化
        batch_size = patchified_pixel_values.shape[0]
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_patches_one_direction,
            num_patches_one_direction,
            patch_size,
            patch_size,
            num_channels,
        )
        
        # 使用 `einsum` 函数重新排列张量维度，完成反补丁化
        patchified_pixel_values = torch.einsum("nhwpqc->nchpwq", patchified_pixel_values)
        
        # 最终重塑像素值张量，以恢复原始图像形状
        pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_channels,
            num_patches_one_direction * patch_size,
            num_patches_one_direction * patch_size,
        )
        
        return pixel_values

    def forward_loss(self, pixel_values, pred, mask):
        """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values.
            pred (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Predicted pixel values.
            mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Tensor indicating which patches are masked (1) and which are not (0).

        Returns:
            `torch.FloatTensor`: Pixel reconstruction loss.
        """
        # 对目标像素值进行补丁化
        target = self.patchify(pixel_values)
        
        # 如果配置中指定了像素归一化损失，则进行像素归一化
        if self.config.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5
        
        # 计算像素重建损失
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        
        # 根据掩码计算被移除补丁的平均损失
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        
        return loss

    @add_start_docstrings_to_model_forward(VIT_MAE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ViTMAEForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        noise: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 设置返回字典的选项，如果未指定则使用配置中的默认值

        outputs = self.vit(
            pixel_values,
            noise=noise,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 使用 Vision Transformer 处理器进行前向传播，生成模型的输出

        latent = outputs.last_hidden_state
        # 获取模型输出的最后隐藏状态作为潜变量

        ids_restore = outputs.ids_restore
        # 获取模型输出中的 ids_restore 属性，用于恢复图像补丁

        mask = outputs.mask
        # 获取模型输出中的 mask 属性，用于掩码处理

        decoder_outputs = self.decoder(latent, ids_restore)
        # 使用解码器对潜变量和 ids_restore 进行解码

        logits = decoder_outputs.logits  # shape (batch_size, num_patches, patch_size*patch_size*num_channels)
        # 从解码器的输出中获取 logits，表示重建图像的预测值

        loss = self.forward_loss(pixel_values, logits, mask)
        # 计算损失，用于评估重建图像的准确性

        if not return_dict:
            output = (logits, mask, ids_restore) + outputs[2:]
            # 如果不使用返回字典，则返回一个包含 logits、mask 和 ids_restore 的元组，以及可能的额外输出
            return ((loss,) + output) if loss is not None else output
            # 如果存在损失，则将损失加入返回结果中

        return ViTMAEForPreTrainingOutput(
            loss=loss,
            logits=logits,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # 如果使用返回字典，则将损失、logits、mask、ids_restore、隐藏状态和注意力作为 ViTMAEForPreTrainingOutput 的实例返回
```