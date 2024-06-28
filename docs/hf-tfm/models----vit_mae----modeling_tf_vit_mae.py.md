# `.\models\vit_mae\modeling_tf_vit_mae.py`

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
""" TF 2.0 ViT MAE (masked autoencoder) model."""

# Importing necessary modules and libraries
from __future__ import annotations  # Allow forward references in type annotations

import collections.abc  # Import for abstract base classes
import math  # Import for mathematical functions
from copy import deepcopy  # Import for deep copying objects
from dataclasses import dataclass  # Import for creating structured data classes
from typing import Optional, Tuple, Union  # Import for type hints

import numpy as np  # Import for numerical operations with arrays
import tensorflow as tf  # Import TensorFlow library

# Importing specific functions and classes from custom modules
from ...activations_tf import get_tf_activation  # Import activation function retriever
from ...file_utils import (
    ModelOutput,  # Import base class for model outputs
    add_start_docstrings,  # Import function for adding docstrings to functions
    add_start_docstrings_to_model_forward,  # Import function for adding docstrings to model forward pass
    replace_return_docstrings,  # Import function for replacing return docstrings
)
from ...modeling_tf_outputs import TFBaseModelOutput  # Import base model output class for TensorFlow
from ...modeling_tf_utils import (
    TFModelInputType,  # Import type hint for model input in TensorFlow
    TFPreTrainedModel,  # Import base class for pre-trained models in TensorFlow
    get_initializer,  # Import function for getting weight initializers
    keras,  # Import Keras submodule from TensorFlow
    keras_serializable,  # Import decorator for serializing Keras layers
    unpack_inputs,  # Import function for unpacking model inputs
)
from ...tf_utils import shape_list, stable_softmax  # Import utility functions for TensorFlow
from ...utils import logging  # Import logging utilities
from .configuration_vit_mae import ViTMAEConfig  # Import configuration class for ViT MAE model


logger = logging.get_logger(__name__)  # Get logger instance for current module

_CONFIG_FOR_DOC = "ViTMAEConfig"  # Documentation string for configuration class
_CHECKPOINT_FOR_DOC = "facebook/vit-mae-base"  # Documentation string for model checkpoint

@dataclass
class TFViTMAEModelOutput(ModelOutput):
    """
    Class for TFViTMAEModel's outputs, with potential hidden states and attentions.
    """
    # 定义函数的参数及其类型注解
    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层的隐藏状态序列的张量。
        mask (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            指示哪些补丁被掩码（1）和哪些未被掩码（0）的张量。
        ids_restore (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            包含（打乱后的）掩码补丁的原始索引的张量。
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            元组的 `tf.Tensor` （一个用于嵌入输出 + 每层输出的一个）的形状为 `(batch_size, sequence_length, hidden_size)`。
            模型在每一层输出的隐藏状态以及初始嵌入输出。
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            元组的 `tf.Tensor` （每层一个）的形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
            注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
    """
    
    # 初始化函数的参数为默认值为 None
    last_hidden_state: tf.Tensor = None
    mask: tf.Tensor = None
    ids_restore: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None
@dataclass
class TFViTMAEDecoderOutput(ModelOutput):
    """
    TFViTMAEDecoderOutput 类用于存储 TFViTMAEDecoder 的输出结果，可能包含隐藏状态和注意力权重。

    Args:
        logits (`tf.Tensor` of shape `(batch_size, sequence_length, patch_size ** 2 * num_channels)`):
            像素重建的逻辑回归结果。
        hidden_states (`tuple(tf.Tensor)`, *optional*, 当 `output_hidden_states=True` 时返回或 `config.output_hidden_states=True` 时返回):
            包含 `tf.Tensor` 元组（一个用于嵌入的输出 + 每层的一个输出），形状为 `(batch_size, sequence_length, hidden_size)`。
            模型每层的隐藏状态以及初始嵌入的输出。
        attentions (`tuple(tf.Tensor)`, *optional*, 当 `output_attentions=True` 时返回或 `config.output_attentions=True` 时返回):
            包含 `tf.Tensor` 元组（每层一个），形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
            经过注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    logits: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None


@dataclass
class TFViTMAEForPreTrainingOutput(ModelOutput):
    """
    TFViTMAEForPreTrainingOutput 类用于存储 TFViTMAEForPreTraining 的输出结果，可能包含隐藏状态和注意力权重。

    Args:
        loss (`tf.Tensor` of shape `(1,)`):
            像素重建损失。
        logits (`tf.Tensor` of shape `(batch_size, sequence_length, patch_size ** 2 * num_channels)`):
            像素重建的逻辑回归结果。
        mask (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            指示哪些补丁被掩盖（1）和哪些没有（0）的张量。
        ids_restore (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            包含（打乱的）掩盖补丁的原始索引的张量。
        hidden_states (`tuple(tf.Tensor)`, *optional*, 当 `output_hidden_states=True` 时返回或 `config.output_hidden_states=True` 时返回):
            包含 `tf.Tensor` 元组（一个用于嵌入的输出 + 每层的一个输出），形状为 `(batch_size, sequence_length, hidden_size)`。
            模型每层的隐藏状态以及初始嵌入的输出。
        attentions (`tuple(tf.Tensor)`, *optional*, 当 `output_attentions=True` 时返回或 `config.output_attentions=True` 时返回):
            包含 `tf.Tensor` 元组（每层一个），形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
            经过注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    loss: tf.Tensor | None = None
    logits: tf.Tensor = None
    mask: tf.Tensor = None
    ids_restore: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    # attentions 是一个变量，类型是 Tuple[tf.Tensor] 或者 None
    attentions: Tuple[tf.Tensor] | None = None
# 创建二维 sin/cos 位置嵌入的函数
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
        (`tf.Tensor` of shape (grid_size*grid_size, embed_dim) or (1+grid_size*grid_size, embed_dim): the position
        embeddings (with or without classification token)
    """
    # 创建高度和宽度的网格
    grid_h = tf.range(grid_size, dtype=tf.float32)
    grid_w = tf.range(grid_size, dtype=tf.float32)
    grid = tf.meshgrid(grid_w, grid_h)  # 这里宽度先行
    grid = tf.stack(grid, axis=0)

    grid = tf.reshape(grid, [2, 1, grid_size, grid_size])
    # 从网格获取二维 sin/cos 位置嵌入
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if add_cls_token:
        # 如果需要添加 CLS token，则在位置嵌入前面加一个全零向量
        pos_embed = tf.concat([tf.zeros((1, embed_dim)), pos_embed], axis=0)
    return pos_embed


# 从网格获取二维 sin/cos 位置嵌入
def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    # 使用一半维度来编码 grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = tf.concat([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


# 从网格获取一维 sin/cos 位置嵌入
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    omega = tf.range(embed_dim // 2, dtype="float32")
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = tf.reshape(pos, [-1])  # (M,)
    out = tf.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    # 一半位置获取正弦模式，另一半获取余弦模式，然后串联起来
    emb_sin = tf.sin(out)  # (M, D/2)
    emb_cos = tf.cos(out)  # (M, D/2)

    emb = tf.concat([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class TFViTMAEEmbeddings(keras.layers.Layer):
    """
    构建 CLS token、位置和补丁嵌入。
    """

    def __init__(self, config: ViTMAEConfig, **kwargs):
        super().__init__(**kwargs)

        self.patch_embeddings = TFViTMAEPatchEmbeddings(config, name="patch_embeddings")
        self.num_patches = self.patch_embeddings.num_patches

        self.config = config
    # 在神经网络层的建立过程中，创建一个用于分类特殊令牌的权重矩阵，形状为 (1, 1, 隐藏层大小)
    self.cls_token = self.add_weight(
        shape=(1, 1, self.config.hidden_size),
        initializer=tf.random_normal_initializer(stddev=self.config.initializer_range),
        trainable=True,
        name="cls_token",
    )
    
    # 创建位置嵌入矩阵，形状为 (1, num_patches + 1, 隐藏层大小)，使用零值初始化
    self.position_embeddings = self.add_weight(
        shape=(1, self.num_patches + 1, self.config.hidden_size),
        initializer="zeros",
        trainable=False,  # 固定的正弦-余弦位置嵌入
        name="position_embeddings",
    )
    
    # 调用函数 `get_2d_sincos_pos_embed` 生成二维正弦-余弦位置嵌入
    pos_embed = get_2d_sincos_pos_embed(
        self.position_embeddings.shape[-1],
        int(self.patch_embeddings.num_patches**0.5),
        add_cls_token=True,
    )[None, ...]
    
    # 将生成的位置嵌入赋值给 self.position_embeddings
    self.position_embeddings.assign(pos_embed)

    # 如果模型已经建立完成，则直接返回
    if self.built:
        return
    
    # 标记模型已经建立
    self.built = True
    
    # 如果 self.patch_embeddings 属性存在，则调用它的 build 方法
    if getattr(self, "patch_embeddings", None) is not None:
        with tf.name_scope(self.patch_embeddings.name):
            self.patch_embeddings.build(None)

def random_masking(self, sequence: tf.Tensor, noise: tf.Tensor | None = None):
    """
    执行每个样本的随机遮盖，通过每个样本的乱序实现。每个样本的乱序由参数 argsort 的随机噪声完成。

    Args:
        sequence (`tf.Tensor` of shape `(batch_size, sequence_length, dim)`)
        noise (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*)，主要用于测试目的，
            控制随机性并保持可重现性
    """
    # 获取 sequence 的形状信息：batch_size, sequence_length, dim
    batch_size, seq_length, dim = shape_list(sequence)
    
    # 计算保留的长度，以保证不被遮盖的部分占比为 self.config.mask_ratio
    len_keep = int(seq_length * (1 - self.config.mask_ratio))

    # 如果没有提供噪声数据，则生成一个均匀分布在 [0, 1) 区间的随机噪声
    if noise is None:
        noise = tf.random.uniform(shape=(batch_size, seq_length), minval=0.0, maxval=1.0)  # 噪声范围在 [0, 1)

    # 对每个样本的噪声进行排序
    ids_shuffle = tf.argsort(noise, axis=1)  # 升序排序：小的表示保留，大的表示移除
    ids_restore = tf.argsort(ids_shuffle, axis=1)

    # 保留前 len_keep 部分的序号
    ids_keep = ids_shuffle[:, :len_keep]
    sequence_unmasked = tf.gather(
        sequence,
        axis=1,
        batch_dims=1,
        indices=ids_keep,
    )

    # 生成二进制遮罩：0 表示保留，1 表示移除
    # 这个方法是必需的，因为 TF 的 EagerTensors 不支持直接的赋值操作
    mask_keep = tf.zeros((batch_size, len_keep))
    mask_remove = tf.ones((batch_size, seq_length - len_keep))
    mask = tf.concat([mask_keep, mask_remove], axis=-1)

    # 根据 ids_restore 恢复原始顺序，得到最终的二进制遮罩
    mask = tf.gather(mask, axis=1, batch_dims=1, indices=ids_restore)

    return sequence_unmasked, mask, ids_restore
    def call(self, pixel_values: tf.Tensor, noise: tf.Tensor = None) -> tf.Tensor:
        # 使用 patch_embeddings 方法将像素值转换为嵌入向量
        embeddings = self.patch_embeddings(pixel_values)

        # 添加位置嵌入，不包括 cls 标记
        embeddings = embeddings + self.position_embeddings[:, 1:, :]

        # 执行随机遮蔽：将 embeddings 进行部分遮蔽，生成 mask，并记录遮蔽前的位置 ids_restore
        embeddings, mask, ids_restore = self.random_masking(embeddings, noise)

        # 添加 cls 标记
        # 从 self.cls_token 和 self.position_embeddings 中获取 cls 标记，并复制到每个样本序列的开头
        cls_token = self.cls_token + self.position_embeddings[:, :1, :]
        cls_tokens = tf.tile(cls_token, (shape_list(embeddings)[0], 1, 1))
        # 将 cls 标记与 embeddings 拼接起来
        embeddings = tf.concat([cls_tokens, embeddings], axis=1)

        # 返回处理后的 embeddings、mask 和 ids_restore
        return embeddings, mask, ids_restore
class TFViTMAEPatchEmbeddings(keras.layers.Layer):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config: ViTMAEConfig, **kwargs):
        super().__init__(**kwargs)
        # 从配置中获取图像大小和patch大小
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size
        # 如果图像大小和patch大小不是可迭代对象，转换为元组形式
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        # 计算图像中的patch数量
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_channels = num_channels
        self.config = config

        # 定义卷积层，用于将输入的像素值转换为patch embeddings
        self.projection = keras.layers.Conv2D(
            filters=hidden_size,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            data_format="channels_last",
            kernel_initializer="glorot_uniform",  # 使用glorot_uniform初始化卷积核
            bias_initializer="zeros",  # 使用零初始化偏置
            name="projection",
        )

    def call(self, pixel_values: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 获取输入张量的形状信息
        batch_size, num_channels, height, width = shape_list(pixel_values)
        
        # 在动态执行模式下，检查通道数是否与配置中设置的一致
        if tf.executing_eagerly():
            if num_channels != self.num_channels:
                raise ValueError(
                    "Make sure that the channel dimension of the pixel values match with the one set in the"
                    " configuration."
                )
            # 检查输入图像的尺寸是否与配置中设置的一致
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )

        # 在CPU上运行时，keras.layers.Conv2D不支持NCHW格式，需要将输入格式从NCHW转换为NHWC
        # shape = (batch_size, in_height, in_width, in_channels=num_channels)
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))

        # 将输入像素值投影到隐藏空间中
        projection = self.projection(pixel_values)

        # 将2D空间维度变换为单一的时间维度
        # shape = (batch_size, num_patches, out_channels=embed_dim)
        num_patches = (width // self.patch_size[1]) * (height // self.patch_size[0])
        x = tf.reshape(tensor=projection, shape=(batch_size, num_patches, -1))

        return x
    # 定义 build 方法，用于构建神经网络层的结构
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回，避免重复构建
        if self.built:
            return
        # 标记当前层已经构建
        self.built = True
        # 如果存在投影层，则构建投影层
        if getattr(self, "projection", None) is not None:
            # 在 TensorFlow 中，使用 name_scope 可以定义操作的命名空间
            with tf.name_scope(self.projection.name):
                # 构建投影层，输入的形状是 [None, None, None, self.num_channels]
                self.projection.build([None, None, None, self.num_channels])
# 从transformers.models.vit.modeling_tf_vit.TFViTSelfAttention复制到TFViTMAESelfAttention，并修改为ViT->ViTMAE
class TFViTMAESelfAttention(keras.layers.Layer):
    def __init__(self, config: ViTMAEConfig, **kwargs):
        super().__init__(**kwargs)

        # 检查隐藏大小是否能被注意力头数整除
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_attention_heads})"
            )

        # 初始化注意力头数和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

        # 创建用于查询、键、值的全连接层，并初始化权重
        self.query = keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        self.key = keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key"
        )
        self.value = keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )
        # 添加 dropout 层，用于注意力概率的随机失活
        self.dropout = keras.layers.Dropout(rate=config.attention_probs_dropout_prob)
        self.config = config

    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        # 将形状从 [batch_size, seq_length, all_head_size] 重塑为 [batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))

        # 将张量从 [batch_size, seq_length, num_attention_heads, attention_head_size] 转置为 [batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        # 获取隐藏状态张量的批量大小
        batch_size = shape_list(hidden_states)[0]
        # 对隐藏状态进行查询操作，生成混合查询层
        mixed_query_layer = self.query(inputs=hidden_states)
        # 对隐藏状态进行键操作，生成混合键层
        mixed_key_layer = self.key(inputs=hidden_states)
        # 对隐藏状态进行值操作，生成混合值层
        mixed_value_layer = self.value(inputs=hidden_states)
        # 将混合查询层转置以便计算注意力分数
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        # 将混合键层转置以便计算注意力分数
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        # 将混合值层转置以便计算注意力分数
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        # 计算查询与键之间的点积，得到原始注意力分数
        # 形状为(batch size, num_heads, seq_len_q, seq_len_k)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        # 计算注意力分数的缩放系数 dk
        dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(attention_scores, dk)

        # 将注意力分数归一化为概率
        attention_probs = stable_softmax(logits=attention_scores, axis=-1)

        # 对注意力概率进行 dropout 处理
        attention_probs = self.dropout(inputs=attention_probs, training=training)

        # 如果存在头部掩码，则应用头部掩码
        if head_mask is not None:
            attention_probs = tf.multiply(attention_probs, head_mask)

        # 计算注意力输出值
        attention_output = tf.matmul(attention_probs, value_layer)
        # 调整注意力输出值的维度顺序
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])

        # 将注意力输出值重新形状为(batch_size, seq_len_q, all_head_size)
        attention_output = tf.reshape(tensor=attention_output, shape=(batch_size, -1, self.all_head_size))
        # 构建输出元组，根据需要包含注意力概率
        outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)

        return outputs

    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在查询层，构建查询层
        if getattr(self, "query", None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.config.hidden_size])
        # 如果存在键层，构建键层
        if getattr(self, "key", None) is not None:
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.config.hidden_size])
        # 如果存在值层，构建值层
        if getattr(self, "value", None) is not None:
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.config.hidden_size])
# Copied from transformers.models.vit.modeling_tf_vit.TFViTSelfOutput with ViT->ViTMAE
class TFViTMAESelfOutput(keras.layers.Layer):
    """
    The residual connection is defined in TFViTMAELayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: ViTMAEConfig, **kwargs):
        super().__init__(**kwargs)

        # 定义一个全连接层，用于将输入的隐藏状态转换为指定大小的输出
        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 定义一个 dropout 层，用于在训练时随机置零部分神经元，防止过拟合
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 使用全连接层对隐藏状态进行转换
        hidden_states = self.dense(inputs=hidden_states)
        # 在训练时对转换后的输出应用 dropout
        hidden_states = self.dropout(inputs=hidden_states, training=training)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已定义全连接层，构建全连接层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# Copied from transformers.models.vit.modeling_tf_vit.TFViTAttention with ViT->ViTMAE
class TFViTMAEAttention(keras.layers.Layer):
    def __init__(self, config: ViTMAEConfig, **kwargs):
        super().__init__(**kwargs)

        # 定义注意力层对象，用于处理自注意力机制
        self.self_attention = TFViTMAESelfAttention(config, name="attention")
        # 定义输出层对象，负责接收注意力层输出并进行处理
        self.dense_output = TFViTMAESelfOutput(config, name="output")

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(
        self,
        input_tensor: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        # 调用自注意力层，处理输入张量，返回处理结果和可能的注意力分布（如果输出的话）
        self_outputs = self.self_attention(
            hidden_states=input_tensor, head_mask=head_mask, output_attentions=output_attentions, training=training
        )
        # 调用输出层，接收自注意力层的输出和输入张量，并返回处理后的结果
        attention_output = self.dense_output(
            hidden_states=self_outputs[0], input_tensor=input_tensor, training=training
        )
        # 将输出整合为一个元组，包括处理后的注意力输出和可能的注意力分布
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已定义自注意力层，构建自注意力层
        if getattr(self, "self_attention", None) is not None:
            with tf.name_scope(self.self_attention.name):
                self.self_attention.build(None)
        # 如果已定义输出层，构建输出层
        if getattr(self, "dense_output", None) is not None:
            with tf.name_scope(self.dense_output.name):
                self.dense_output.build(None)


# Copied from transformers.models.vit.modeling_tf_vit.TFViTIntermediate with ViT->ViTMAE
class TFViTMAEIntermediate(keras.layers.Layer):
    # 初始化函数，用于创建一个新的 ViTMAE 层实例
    def __init__(self, config: ViTMAEConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 创建一个密集连接层，用于处理输入特征
        self.dense = keras.layers.Dense(
            units=config.intermediate_size,  # 设置层的输出维度
            kernel_initializer=get_initializer(config.initializer_range),  # 使用指定的初始化器初始化权重矩阵
            name="dense"  # 设置层的名称
        )

        # 根据配置选择激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)  # 获取指定名称的 TensorFlow 激活函数
        else:
            self.intermediate_act_fn = config.hidden_act  # 直接使用给定的激活函数
        self.config = config  # 保存配置对象

    # 调用函数，用于定义层的正向传播逻辑
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)  # 将输入数据传递给密集连接层
        hidden_states = self.intermediate_act_fn(hidden_states)  # 应用中间激活函数

        return hidden_states  # 返回处理后的特征表示

    # 构建函数，用于构建层的参数
    def build(self, input_shape=None):
        if self.built:  # 如果已经构建过，直接返回
            return
        self.built = True  # 标记为已构建
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):  # 使用名称空间管理密集连接层
                self.dense.build([None, None, self.config.hidden_size])  # 构建密集连接层的参数
# Copied from transformers.models.vit.modeling_tf_vit.TFViTOutput with ViT->ViTMAE
class TFViTMAEOutput(keras.layers.Layer):
    def __init__(self, config: ViTMAEConfig, **kwargs):
        super().__init__(**kwargs)

        # 定义一个全连接层，输出维度为 config.hidden_size，使用指定的初始化方法
        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 定义一个 Dropout 层，使用给定的 dropout 率
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 对输入的 hidden_states 进行全连接操作
        hidden_states = self.dense(inputs=hidden_states)
        # 在训练时对全连接结果进行 dropout 处理
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 将 dropout 后的结果与 input_tensor 相加
        hidden_states = hidden_states + input_tensor

        return hidden_states

    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 self.dense 层，则根据输入形状构建全连接层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])


# Copied from transformers.models.vit.modeling_tf_vit.TFViTLayer with ViT->ViTMAE
class TFViTMAELayer(keras.layers.Layer):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: ViTMAEConfig, **kwargs):
        super().__init__(**kwargs)

        # 初始化 TFViTMAEAttention 层
        self.attention = TFViTMAEAttention(config, name="attention")
        # 初始化 TFViTMAEIntermediate 层
        self.intermediate = TFViTMAEIntermediate(config, name="intermediate")
        # 初始化 TFViTMAEOutput 层
        self.vit_output = TFViTMAEOutput(config, name="output")

        # 初始化 layernorm 层，在每个 block 的开始和结束进行归一化
        self.layernorm_before = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm_before")
        self.layernorm_after = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm_after")
        self.config = config

    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
        **kwargs
    ) -> tf.Tensor:
        # 调用 self.attention 层处理输入 hidden_states
        attention_output = self.attention(
            hidden_states, head_mask, output_attentions=output_attentions, training=training
        )
        # 将 attention 输出与 hidden_states 相加，并进行 layernorm 处理
        hidden_states = self.layernorm_before(attention_output + hidden_states)
        # 调用 self.intermediate 层处理 layernorm 后的结果
        intermediate_output = self.intermediate(hidden_states)
        # 将 intermediate 输出与 hidden_states 相加，并进行 layernorm 处理
        hidden_states = self.layernorm_after(intermediate_output + hidden_states)
        # 调用 self.vit_output 层处理最终的输出
        output = self.vit_output(hidden_states, attention_output, training=training)

        return output
    ) -> Tuple[tf.Tensor]:
        # 调用 self.attention 进行注意力计算，ViTMAE 中在 self-attention 前应用 layernorm
        attention_outputs = self.attention(
            input_tensor=self.layernorm_before(inputs=hidden_states),  # 在 self-attention 前应用 layernorm
            head_mask=head_mask,
            output_attentions=output_attentions,
            training=training,
        )
        attention_output = attention_outputs[0]

        # 第一个残差连接
        hidden_states = attention_output + hidden_states

        # ViTMAE 中在 self-attention 后同样应用 layernorm
        layer_output = self.layernorm_after(inputs=hidden_states)

        # 使用 intermediate 层处理输出
        intermediate_output = self.intermediate(hidden_states=layer_output)

        # 第二个残差连接在此处完成
        layer_output = self.vit_output(
            hidden_states=intermediate_output, input_tensor=hidden_states, training=training
        )
        outputs = (layer_output,) + attention_outputs[1:]  # 如果有需要，添加注意力信息到输出中

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经构建过，直接返回

        # 构建 attention 层
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)

        # 构建 intermediate 层
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)

        # 构建 vit_output 层
        if getattr(self, "vit_output", None) is not None:
            with tf.name_scope(self.vit_output.name):
                self.vit_output.build(None)

        # 构建 layernorm_before 层
        if getattr(self, "layernorm_before", None) is not None:
            with tf.name_scope(self.layernorm_before.name):
                self.layernorm_before.build([None, None, self.config.hidden_size])

        # 构建 layernorm_after 层
        if getattr(self, "layernorm_after", None) is not None:
            with tf.name_scope(self.layernorm_after.name):
                self.layernorm_after.build([None, None, self.config.hidden_size])
# 从 transformers.models.vit.modeling_tf_vit.TFViTEncoder 复制代码，将 ViT 更改为 ViTMAE
class TFViTMAEEncoder(keras.layers.Layer):
    def __init__(self, config: ViTMAEConfig, **kwargs):
        super().__init__(**kwargs)

        # 初始化编码器的多层子模块 TFViTMAELayer，并命名为"layer_._{i}"
        self.layer = [TFViTMAELayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]

    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        # 初始化存储所有隐藏状态的元组，如果不需要输出隐藏状态则为 None
        all_hidden_states = () if output_hidden_states else None
        # 初始化存储所有注意力权重的元组，如果不需要输出注意力权重则为 None
        all_attentions = () if output_attentions else None

        # 遍历每一层编码器
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                # 如果需要输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 调用当前层的编码器模块，计算输出
            layer_outputs = layer_module(
                hidden_states=hidden_states,
                head_mask=head_mask[i],
                output_attentions=output_attentions,
                training=training,
            )
            # 更新当前隐藏状态为当前层的输出
            hidden_states = layer_outputs[0]

            if output_attentions:
                # 如果需要输出注意力权重，则将当前层的注意力权重添加到 all_attentions 中
                all_attentions = all_attentions + (layer_outputs[1],)

        # 添加最后一层的隐藏状态到 all_hidden_states
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要返回字典格式的输出，则返回所有非空的元组值
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

        # 返回 TFBaseModelOutput 类的对象，包含最后的隐藏状态、所有隐藏状态和所有注意力权重
        return TFBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )

    def build(self, input_shape=None):
        # 如果已经构建过则直接返回
        if self.built:
            return
        self.built = True
        # 构建每一层的编码器模块
        if getattr(self, "layer", None) is not None:
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    layer.build(None)


@keras_serializable
class TFViTMAEMainLayer(keras.layers.Layer):
    config_class = ViTMAEConfig

    def __init__(self, config: ViTMAEConfig, **kwargs):
        super().__init__(**kwargs)

        # 初始化 ViTMAE 主层的配置
        self.config = config

        # 初始化 ViTMAE 主层的嵌入层 TFViTMAEEmbeddings，并命名为"embeddings"
        self.embeddings = TFViTMAEEmbeddings(config, name="embeddings")
        # 初始化 ViTMAE 主层的编码器 TFViTMAEEncoder，并命名为"encoder"
        self.encoder = TFViTMAEEncoder(config, name="encoder")
        # 初始化 ViTMAE 主层的层归一化层 LayerNormalization，使用指定的 epsilon 值，并命名为"layernorm"
        self.layernorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm")

    def get_input_embeddings(self) -> keras.layers.Layer:
        # 返回嵌入层 TFViTMAEEmbeddings 的补丁嵌入
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 抛出未实现的错误，子类需要实现具体的头部修剪逻辑
        raise NotImplementedError

    @unpack_inputs
    # 定义一个方法 `call`，用于执行模型推断或训练
    def call(
        self,
        pixel_values: TFModelInputType | None = None,  # 输入像素值，可以为空
        noise: tf.Tensor = None,  # 噪声张量，默认为空
        head_mask: np.ndarray | tf.Tensor | None = None,  # 头部掩码，可以是 NumPy 数组、张量或空
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可选布尔值
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选布尔值
        return_dict: Optional[bool] = None,  # 是否以字典形式返回结果，可选布尔值
        training: bool = False,  # 是否处于训练模式，默认为 False
    ) -> Union[TFViTMAEModelOutput, Tuple[tf.Tensor]]:
        # 调用嵌入层的方法获取嵌入输出、掩码和恢复的 IDs
        embedding_output, mask, ids_restore = self.embeddings(
            pixel_values=pixel_values, training=training, noise=noise
        )

        # 如果需要，准备头部掩码
        # 在头部掩码中，1.0 表示保留该头部
        # attention_probs 的形状为 bsz x n_heads x N x N
        # 输入的 head_mask 形状为 [num_heads] 或 [num_hidden_layers x num_heads]
        # 并且 head_mask 被转换为形状 [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            # 如果存在头部掩码，但当前未实现如何处理
            raise NotImplementedError
        else:
            # 如果头部掩码为空，则创建一个空列表，长度为隐藏层数
            head_mask = [None] * self.config.num_hidden_layers

        # 使用编码器处理嵌入输出
        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 获取编码器输出的序列输出
        sequence_output = encoder_outputs[0]

        # 应用层归一化到序列输出
        sequence_output = self.layernorm(inputs=sequence_output)

        # 如果不要求以字典形式返回结果，则返回元组
        if not return_dict:
            return (sequence_output, mask, ids_restore) + encoder_outputs[1:]

        # 以 TFViTMAEModelOutput 对象形式返回结果，包括最后的隐藏状态、掩码、恢复的 IDs、隐藏状态和注意力权重
        return TFViTMAEModelOutput(
            last_hidden_state=sequence_output,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    # 定义构建方法 build，用于在需要时构建模型
    def build(self, input_shape=None):
        # 如果已经构建过模型，则直接返回
        if self.built:
            return

        # 设置标记为已构建
        self.built = True

        # 如果存在嵌入层，构建嵌入层
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)

        # 如果存在编码器，构建编码器
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)

        # 如果存在层归一化，构建层归一化，设置形状为 [None, None, self.config.hidden_size]
        if getattr(self, "layernorm", None) is not None:
            with tf.name_scope(self.layernorm.name):
                self.layernorm.build([None, None, self.config.hidden_size])
"""
    Documenting the expected input formats for ViT-MAE models when using TensorFlow. This docstring serves as a guide
    for users on how to provide inputs to the model.

    TensorFlow models in `transformers` support two input formats:
    - Passing all inputs as keyword arguments.
    - Passing all inputs in a list, tuple, or dictionary as the first positional argument.

    This flexibility ensures compatibility with TensorFlow's Keras API and other functional usage scenarios.

    Args:
        pixel_values (Tensor): Input pixel values representing the image.
        attention_mask (Tensor, optional): Mask to avoid performing attention on padding tokens.
        token_type_ids (Tensor, optional): Segment token indices to distinguish different parts of the input.

    Usage Examples:
        - Using keyword arguments: `model(pixel_values=inputs)`
        - Using a list or tuple for positional argument:
          `model([pixel_values, attention_mask])` or `model([pixel_values, attention_mask, token_type_ids])`
        - Using a dictionary with input names: `model({"pixel_values": pixel_values, "token_type_ids": token_type_ids})`

    Note:
        For custom layers or models using Keras Functional API, ensure inputs match the documented formats.

    Reference:
        [Transformers documentation](https://huggingface.co/transformers/model_doc/vit.html)
"""
    Args:
        pixel_values (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` ``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`ViTImageProcessor.__call__`]
            for details.
        head_mask (`np.ndarray` or `tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
            config will be used instead.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
            used instead.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple. This argument can be used
            in eager mode, in graph mode the value will always be set to True.
        training (`bool`, *optional*, defaults to `False``):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).


注释：
"""
Transformer 模型的 ViTMAE 版本的 TensorFlow 实现，输出原始隐藏状态而不带特定的输出头部。

Args:
    config (ViTMAEConfig): ViTMAE 模型的配置对象。
    *inputs: 可变长度的输入参数。
    **kwargs: 关键字参数。

Attributes:
    vit (TFViTMAEMainLayer): ViTMAE 主层对象。

Methods:
    get_input_embeddings(): 获取输入嵌入层的方法。
    call(): 模型的前向传播方法，接受多种参数并返回模型输出。
    build(input_shape=None): 构建模型的方法，用于初始化网络层。

Examples:
    ```
    >>> from transformers import AutoImageProcessor, TFViTMAEModel
    >>> from PIL import Image
    >>> import requests

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
    >>> model = TFViTMAEModel.from_pretrained("facebook/vit-mae-base")

    >>> inputs = image_processor(images=image, return_tensors="tf")
    >>> outputs = model(**inputs)
    >>> last_hidden_states = outputs.last_hidden_state
    ```
"""
class TFViTMAEModel(TFViTMAEPreTrainedModel):
    def __init__(self, config: ViTMAEConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 初始化 ViTMAE 主层对象
        self.vit = TFViTMAEMainLayer(config, name="vit")

    def get_input_embeddings(self):
        # 调用 ViTMAE 主层对象的输入嵌入层方法
        return self.vit.get_input_embeddings()

    @unpack_inputs
    @add_start_docstrings_to_model_forward(VIT_MAE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFViTMAEModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        pixel_values: TFModelInputType | None = None,
        noise: tf.Tensor = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFViTMAEModelOutput, Tuple[tf.Tensor]]:
        r"""
        模型的前向传播方法，接受多种参数并返回模型输出。

        Args:
            pixel_values (TFModelInputType | None): 输入的像素值，可以为 None。
            noise (tf.Tensor): 噪声张量，默认为 None。
            head_mask (np.ndarray | tf.Tensor | None): 头部掩码，可以为 None。
            output_attentions (Optional[bool]): 是否输出注意力权重。
            output_hidden_states (Optional[bool]): 是否输出隐藏状态。
            return_dict (Optional[bool]): 是否返回字典形式的输出。
            training (bool): 是否处于训练模式。

        Returns:
            Union[TFViTMAEModelOutput, Tuple[tf.Tensor]]: 模型的输出结果。

        Examples:
            ```
            >>> from transformers import AutoImageProcessor, TFViTMAEModel
            >>> from PIL import Image
            >>> import requests

            >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            >>> image = Image.open(requests.get(url, stream=True).raw)

            >>> image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
            >>> model = TFViTMAEModel.from_pretrained("facebook/vit-mae-base")

            >>> inputs = image_processor(images=image, return_tensors="tf")
            >>> outputs = model(**inputs)
            >>> last_hidden_states = outputs.last_hidden_state
            ```
        """
        # 调用 ViTMAE 主层对象的前向传播方法
        outputs = self.vit(
            pixel_values=pixel_values,
            noise=noise,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        return outputs

    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 设置已构建标志为 True
        self.built = True
        # 如果存在 ViTMAE 主层对象，则在命名作用域内构建它
        if getattr(self, "vit", None) is not None:
            with tf.name_scope(self.vit.name):
                self.vit.build(None)
    # 初始化函数，用于创建对象实例时的初始化操作
    def __init__(self, config, num_patches, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        
        # 创建一个全连接层作为解码器的嵌入层，用于将输入映射到解码器的隐藏大小
        self.decoder_embed = keras.layers.Dense(config.decoder_hidden_size, name="decoder_embed")

        # 深拷贝配置对象，用于配置解码器层的参数
        decoder_config = deepcopy(config)
        decoder_config.hidden_size = config.decoder_hidden_size
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        
        # 创建多层解码器，每层使用相同的配置
        self.decoder_layers = [
            TFViTMAELayer(decoder_config, name=f"decoder_layers.{j}") for j in range(config.decoder_num_hidden_layers)
        ]

        # 创建层归一化层，用于归一化解码器的输出
        self.decoder_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="decoder_norm")
        
        # 创建解码器预测层，将解码器输出映射回原始图像块的大小和通道数
        self.decoder_pred = keras.layers.Dense(
            config.patch_size**2 * config.num_channels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="decoder_pred",
        )  # encoder to decoder

        # 保存配置对象和图像块数量
        self.config = config
        self.num_patches = num_patches

    # 构建模型，用于在图层创建完成后的初始化和构建操作
    def build(self, input_shape=None):
        # 创建一个权重，用作掩码令牌，形状为 (1, 1, 解码器隐藏大小)
        self.mask_token = self.add_weight(
            shape=(1, 1, self.config.decoder_hidden_size),
            initializer=tf.random_normal_initializer(stddev=self.config.initializer_range),
            trainable=True,
            name="mask_token",
        )
        
        # 创建解码器位置嵌入权重，形状为 (1, 图像块数量+1, 解码器隐藏大小)，初始化为零
        self.decoder_pos_embed = self.add_weight(
            shape=(1, self.num_patches + 1, self.config.decoder_hidden_size),
            initializer="zeros",
            trainable=False,
            name="decoder_pos_embed",
        )
        
        # 使用函数生成二维正弦余弦位置嵌入，并将结果赋值给解码器位置嵌入权重
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.num_patches**0.5),
            add_cls_token=True,
        )[None, ...]
        self.decoder_pos_embed.assign(decoder_pos_embed)

        # 如果已经构建完成则直接返回
        if self.built:
            return
        self.built = True
        
        # 如果存在解码器嵌入层，则构建该层
        if getattr(self, "decoder_embed", None) is not None:
            with tf.name_scope(self.decoder_embed.name):
                self.decoder_embed.build([None, None, self.config.hidden_size])
        
        # 如果存在解码器归一化层，则构建该层
        if getattr(self, "decoder_norm", None) is not None:
            with tf.name_scope(self.decoder_norm.name):
                self.decoder_norm.build([None, None, self.config.decoder_hidden_size])
        
        # 如果存在解码器预测层，则构建该层
        if getattr(self, "decoder_pred", None) is not None:
            with tf.name_scope(self.decoder_pred.name):
                self.decoder_pred.build([None, None, self.config.decoder_hidden_size])
        
        # 如果存在解码器层，则分别构建每一层
        if getattr(self, "decoder_layers", None) is not None:
            for layer in self.decoder_layers:
                with tf.name_scope(layer.name):
                    layer.build(None)

    # 调用模型，实现模型的前向计算
    def call(
        self,
        hidden_states,
        ids_restore,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        # 其他未列出的参数，用于控制前向计算的行为
        # 嵌入标记tokens到隐藏状态
        x = self.decoder_embed(hidden_states)

        # 将mask tokens附加到序列
        mask_tokens = tf.tile(
            self.mask_token,
            (shape_list(x)[0], shape_list(ids_restore)[1] + 1 - shape_list(x)[1], 1),
        )
        x_ = tf.concat([x[:, 1:, :], mask_tokens], axis=1)  # 没有cls token
        x_ = tf.gather(x_, axis=1, batch_dims=1, indices=ids_restore)  # 取消洗牌
        x = tf.concat([x[:, :1, :], x_], axis=1)  # 添加cls token

        # 添加位置嵌入
        hidden_states = x + self.decoder_pos_embed

        # 应用Transformer层（块）
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.decoder_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                head_mask=None,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 对隐藏状态进行归一化
        hidden_states = self.decoder_norm(hidden_states)

        # 预测器投影
        logits = self.decoder_pred(hidden_states)

        # 移除cls token
        logits = logits[:, 1:, :]

        # 根据return_dict决定返回的内容
        if not return_dict:
            return tuple(v for v in [logits, all_hidden_states, all_self_attentions] if v is not None)
        return TFViTMAEDecoderOutput(logits=logits, hidden_states=all_hidden_states, attentions=all_self_attentions)
@add_start_docstrings(
    "The ViTMAE Model transformer with the decoder on top for self-supervised pre-training.",
    VIT_MAE_START_DOCSTRING,
)
class TFViTMAEForPreTraining(TFViTMAEPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # 初始化 ViT 主层
        self.vit = TFViTMAEMainLayer(config, name="vit")
        
        # 初始化解码器，传入配置和从 ViT 主层获取的补丁数
        self.decoder = TFViTMAEDecoder(
            config,
            num_patches=self.vit.embeddings.num_patches,
            name="decoder",
        )

    def get_input_embeddings(self):
        # 返回 ViT 主层的输入嵌入
        return self.vit.get_input_embeddings()

    def _prune_heads(self, heads_to_prune):
        # 抛出未实现错误，用于剪枝操作
        raise NotImplementedError

    def patchify(self, pixel_values):
        """
        Args:
            pixel_values (`tf.Tensor` of shape `(batch_size, height, width, num_channels)` or `(batch_size, num_channels, height, width)`):
                Pixel values.

        Returns:
            `tf.Tensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Patchified pixel values.
        """
        patch_size, num_channels = self.config.patch_size, self.config.num_channels
        
        # 确保通道在最后一个维度
        if shape_list(pixel_values)[1] == num_channels:
            pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))

        # 断言检查
        tf.debugging.assert_equal(
            shape_list(pixel_values)[1],
            shape_list(pixel_values)[2],
            message="Make sure the pixel values have a squared size",
        )
        tf.debugging.assert_equal(
            shape_list(pixel_values)[1] % patch_size,
            0,
            message="Make sure the pixel values have a size that is divisible by the patch size",
        )
        tf.debugging.assert_equal(
            shape_list(pixel_values)[3],
            num_channels,
            message=(
                "Make sure the number of channels of the pixel values is equal to the one set in the configuration"
            ),
        )

        # 补丁化处理
        batch_size = shape_list(pixel_values)[0]
        num_patches_one_direction = shape_list(pixel_values)[2] // patch_size
        patchified_pixel_values = tf.reshape(
            pixel_values,
            (batch_size, num_patches_one_direction, patch_size, num_patches_one_direction, patch_size, num_channels),
        )
        patchified_pixel_values = tf.einsum("nhpwqc->nhwpqc", patchified_pixel_values)
        patchified_pixel_values = tf.reshape(
            patchified_pixel_values,
            (batch_size, num_patches_one_direction * num_patches_one_direction, patch_size**2 * num_channels),
        )
        return patchified_pixel_values
    def unpatchify(self, patchified_pixel_values):
        """
        Args:
            patchified_pixel_values (`tf.Tensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Patchified pixel values.

        Returns:
            `tf.Tensor` of shape `(batch_size, height, width, num_channels)`:
                Pixel values.
        """
        # 从patchified_pixel_values中获取patch大小和通道数
        patch_size, num_channels = self.config.patch_size, self.config.num_channels
        # 计算每个方向上的patch数量，应该是整数
        num_patches_one_direction = int(shape_list(patchified_pixel_values)[1] ** 0.5)
        # 进行健全性检查，确保patch数量是可以平方的
        tf.debugging.assert_equal(
            num_patches_one_direction * num_patches_one_direction,
            shape_list(patchified_pixel_values)[1],
            message="Make sure that the number of patches can be squared",
        )

        # 解除patchification
        batch_size = shape_list(patchified_pixel_values)[0]
        patchified_pixel_values = tf.reshape(
            patchified_pixel_values,
            (batch_size, num_patches_one_direction, num_patches_one_direction, patch_size, patch_size, num_channels),
        )
        patchified_pixel_values = tf.einsum("nhwpqc->nhpwqc", patchified_pixel_values)
        # 重新组织成完整的像素值形状
        pixel_values = tf.reshape(
            patchified_pixel_values,
            (batch_size, num_patches_one_direction * patch_size, num_patches_one_direction * patch_size, num_channels),
        )
        return pixel_values

    def forward_loss(self, pixel_values, pred, mask):
        """
        Args:
            pixel_values (`tf.Tensor` of shape `(batch_size, height, width, num_channels)`):
                Pixel values.
            pred (`tf.Tensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Predicted pixel values.
            mask (`tf.Tensor` of shape `(batch_size, sequence_length)`):
                Tensor indicating which patches are masked (1) and which are not (0).

        Returns:
            `tf.Tensor`: Pixel reconstruction loss.
        """
        # 将像素值进行patchify处理
        target = self.patchify(pixel_values)
        # 如果设置了像素损失的归一化，则进行归一化处理
        if self.config.norm_pix_loss:
            mean = tf.reduce_mean(target, axis=-1, keepdims=True)
            var = tf.math.reduce_variance(target, axis=-1, keepdims=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        # 计算损失，即预测值与目标值之间的平方差
        loss = (pred - target) ** 2
        loss = tf.reduce_mean(loss, axis=-1)  # [batch_size, num_patches], mean loss per patch

        # 计算仅在掩码处损失的平均损失
        loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)  # mean loss on removed patches
        loss = tf.reshape(loss, (1,))
        return loss

    @unpack_inputs
    @add_start_docstrings_to_model_forward(VIT_MAE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFViTMAEForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        pixel_values: TFModelInputType | None = None,
        noise: tf.Tensor = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFViTMAEForPreTrainingOutput, Tuple[tf.Tensor]]:
        r"""
        Returns:

        Examples:

        ```
        >>> from transformers import AutoImageProcessor, TFViTMAEForPreTraining
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
        >>> model = TFViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> loss = outputs.loss
        >>> mask = outputs.mask
        >>> ids_restore = outputs.ids_restore
        ```"""

        # 根据传入的参数设置是否返回字典格式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 Vision Transformer 模型进行前向传播
        outputs = self.vit(
            pixel_values=pixel_values,
            noise=noise,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 从输出中获取最后隐藏状态、恢复的图像标识和掩码
        latent = outputs.last_hidden_state
        ids_restore = outputs.ids_restore
        mask = outputs.mask

        # 使用解码器生成的 logits 计算前向损失
        decoder_outputs = self.decoder(latent, ids_restore)  # [batch_size, num_patches, patch_size**2*3]
        logits = decoder_outputs.logits
        loss = self.forward_loss(pixel_values, logits, mask)

        # 根据是否返回字典格式决定返回的输出形式
        if not return_dict:
            output = (logits, mask, ids_restore) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TFViTMAEForPreTrainingOutput 对象，包含损失、logits、掩码、恢复的图像标识、隐藏状态和注意力矩阵
        return TFViTMAEForPreTrainingOutput(
            loss=loss,
            logits=logits,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return

        # 标记模型已经构建
        self.built = True

        # 如果已定义 Vision Transformer 模型，构建其结构
        if getattr(self, "vit", None) is not None:
            with tf.name_scope(self.vit.name):
                self.vit.build(None)

        # 如果已定义解码器模型，构建其结构
        if getattr(self, "decoder", None) is not None:
            with tf.name_scope(self.decoder.name):
                self.decoder.build(None)
```