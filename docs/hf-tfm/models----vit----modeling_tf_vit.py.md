# `.\models\vit\modeling_tf_vit.py`

```py
# coding=utf-8
# Copyright 2021 Google AI, Ross Wightman, The HuggingFace Inc. team. All rights reserved.
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
""" TF 2.0 ViT model."""

from __future__ import annotations

import collections.abc
import math
from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling, TFSequenceClassifierOutput
from ...modeling_tf_utils import (
    TFModelInputType,
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    get_initializer,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import shape_list, stable_softmax
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_vit import ViTConfig

# Logger setup for this module
logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "ViTConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "google/vit-base-patch16-224-in21k"
_EXPECTED_OUTPUT_SHAPE = [1, 197, 768]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "google/vit-base-patch16-224"
_IMAGE_CLASS_EXPECTED_OUTPUT = "Egyptian cat"

class TFViTEmbeddings(keras.layers.Layer):
    """
    Construct the CLS token, position and patch embeddings.
    """

    def __init__(self, config: ViTConfig, **kwargs):
        super().__init__(**kwargs)
        
        # Initialize patch embeddings using TFViTPatchEmbeddings layer
        self.patch_embeddings = TFViTPatchEmbeddings(config, name="patch_embeddings")
        # Dropout layer with rate from configuration
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def build(self, input_shape=None):
        # Get number of patches from patch embeddings layer
        num_patches = self.patch_embeddings.num_patches
        
        # Initialize CLS token embedding
        self.cls_token = self.add_weight(
            shape=(1, 1, self.config.hidden_size),
            initializer=get_initializer(self.config.initializer_range),
            trainable=True,
            name="cls_token",
        )
        
        # Initialize position embeddings based on number of patches
        self.position_embeddings = self.add_weight(
            shape=(1, num_patches + 1, self.config.hidden_size),
            initializer=get_initializer(self.config.initializer_range),
            trainable=True,
            name="position_embeddings",
        )

        # Check if layer is already built to avoid re-building
        if self.built:
            return
        
        self.built = True
        
        # Build patch embeddings layer if it exists
        if getattr(self, "patch_embeddings", None) is not None:
            with tf.name_scope(self.patch_embeddings.name):
                self.patch_embeddings.build(None)
    def interpolate_pos_encoding(self, embeddings, height, width) -> tf.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        # 获取嵌入张量的形状信息：batch_size为批大小，seq_len为序列长度，dim为嵌入维度
        batch_size, seq_len, dim = shape_list(embeddings)
        # 计算图像分块的数量（即序列长度减去1）
        num_patches = seq_len - 1

        # 获取预训练位置编码张量的形状信息：num_positions为位置编码的数量
        _, num_positions, _ = shape_list(self.position_embeddings)
        num_positions -= 1

        # 如果图像分块数量等于位置编码数量且图像高度等于宽度，则直接返回位置编码张量
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        
        # 从位置编码张量中分离出类别位置编码
        class_pos_embed = self.position_embeddings[:, :1]
        # 从位置编码张量中分离出图像分块位置编码
        patch_pos_embed = self.position_embeddings[:, 1:]
        # 计算新的图像分块数量
        h0 = height // self.config.patch_size
        w0 = width // self.config.patch_size
        # 使用双三次插值法对图像分块位置编码进行调整
        patch_pos_embed = tf.image.resize(
            images=tf.reshape(
                patch_pos_embed, shape=(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
            ),
            size=(h0, w0),
            method="bicubic",
        )

        # 检查调整后的图像分块位置编码张量形状是否与预期一致
        shape = shape_list(patch_pos_embed)
        assert h0 == shape[-3] and w0 == shape[-2]
        # 重新整形调整后的图像分块位置编码张量
        patch_pos_embed = tf.reshape(tensor=patch_pos_embed, shape=(1, -1, dim))
        # 将类别位置编码和调整后的图像分块位置编码拼接在一起作为最终的位置编码张量
        return tf.concat(values=(class_pos_embed, patch_pos_embed), axis=1)

    def call(
        self, pixel_values: tf.Tensor, interpolate_pos_encoding: bool = False, training: bool = False
    ) -> tf.Tensor:
        # 获取输入像素张量的形状信息：batch_size为批大小，num_channels为通道数，height为高度，width为宽度
        batch_size, num_channels, height, width = shape_list(pixel_values)
        # 将像素值转换为嵌入向量，并根据需要进行位置编码的插值
        embeddings = self.patch_embeddings(
            pixel_values, interpolate_pos_encoding=interpolate_pos_encoding, training=training
        )

        # 将[CLS]令牌添加到嵌入的补丁令牌中
        cls_tokens = tf.repeat(self.cls_token, repeats=batch_size, axis=0)
        embeddings = tf.concat((cls_tokens, embeddings), axis=1)

        # 如果需要插值位置编码，则将插值后的位置编码添加到每个令牌中
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            # 否则，将原始位置编码添加到每个令牌中
            embeddings = embeddings + self.position_embeddings

        # 在训练时对嵌入向量应用丢弃操作
        embeddings = self.dropout(embeddings, training=training)

        # 返回最终的嵌入向量
        return embeddings
# 基于 timm 实现，可以在此处找到：
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
class TFViTPatchEmbeddings(keras.layers.Layer):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config: ViTConfig, **kwargs):
        super().__init__(**kwargs)
        
        # 从配置中获取图像大小和补丁大小
        image_size, patch_size = config.image_size, config.patch_size
        # 从配置中获取通道数和隐藏大小
        num_channels, hidden_size = config.num_channels, config.hidden_size

        # 如果图像大小和补丁大小不是可迭代对象，则转换为元组
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        
        # 计算图像中的补丁数量
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        
        # 设置实例变量
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_channels = num_channels
        self.config = config

        # 创建卷积层，用于将图像补丁投影到隐藏大小的向量空间
        self.projection = keras.layers.Conv2D(
            filters=hidden_size,                              # 输出通道数为隐藏大小
            kernel_size=patch_size,                           # 卷积核大小设为补丁大小
            strides=patch_size,                               # 步幅设为补丁大小
            padding="valid",                                  # 使用有效填充
            data_format="channels_last",                       # 输入格式为通道在后
            use_bias=True,                                    # 使用偏置项
            kernel_initializer=get_initializer(self.config.initializer_range),  # 卷积核初始化器
            bias_initializer="zeros",                         # 偏置项初始化器设为零
            name="projection",                                # 层的名称为“projection”
        )

    def call(
        self, pixel_values: tf.Tensor, interpolate_pos_encoding: bool = False, training: bool = False
        # 输入函数，将像素值转换为图像补丁嵌入向量
        # pixel_values: 输入的像素值张量
        # interpolate_pos_encoding: 是否插值位置编码
        # training: 是否在训练模式下
    ) -> tf.Tensor:
        # 获取输入张量的形状信息：batch_size, num_channels, height, width
        batch_size, num_channels, height, width = shape_list(pixel_values)
        
        # 如果在即时执行模式下，并且像素值的通道数不等于配置中设置的通道数，则引发值错误
        if tf.executing_eagerly() and num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        
        # 如果不需要插值位置编码
        if not interpolate_pos_encoding:
            if tf.executing_eagerly():
                # 如果高度或宽度与模型期望的图像尺寸不匹配，则引发值错误
                if height != self.image_size[0] or width != self.image_size[1]:
                    raise ValueError(
                        f"Input image size ({height}*{width}) doesn't match model"
                        f" ({self.image_size[0]}*{self.image_size[1]})."
                    )

        # 当在 CPU 上运行时，`keras.layers.Conv2D` 不支持 `NCHW` 格式，因此将输入格式从 `NCHW` 转换为 `NHWC`
        # 形状变为：(batch_size, in_height, in_width, in_channels=num_channels)
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))

        # 对输入进行投影操作
        projection = self.projection(pixel_values)

        # 将二维空间维度转换为单一的时间维度
        # 形状变为：(batch_size, num_patches, out_channels=embed_dim)
        num_patches = (width // self.patch_size[1]) * (height // self.patch_size[0])
        embeddings = tf.reshape(tensor=projection, shape=(batch_size, num_patches, -1))

        return embeddings

    def build(self, input_shape=None):
        # 如果已经构建过网络，则直接返回
        if self.built:
            return
        
        # 标记网络已构建
        self.built = True
        
        # 如果已存在投影层，则构建投影层
        if getattr(self, "projection", None) is not None:
            with tf.name_scope(self.projection.name):
                # 构建投影层，输入形状为 [None, None, None, self.num_channels]
                self.projection.build([None, None, None, self.num_channels])
class TFViTSelfAttention(keras.layers.Layer):
    # 定义一个名为TFViTSelfAttention的自定义Layer类
    def __init__(self, config: ViTConfig, **kwargs):
        # 初始化函数，接受一个ViTConfig类型的config对象和其他可选参数
        super().__init__(**kwargs)
        # 调用父类的初始化函数

        if config.hidden_size % config.num_attention_heads != 0:
            # 如果隐藏层大小不能被注意力头的数量整除
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        # 设置注意力头的数量
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        # 计算每个注意力头的大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # 计算所有注意力头的总大小
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)
        # 计算注意力头大小的平方根

        self.query = keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        # 创建查询矩阵
        self.key = keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key"
        )
        # 创建键矩阵
        self.value = keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )
        # 创建值矩阵
        self.dropout = keras.layers.Dropout(rate=config.attention_probs_dropout_prob)
        # 创建dropout层
        self.config = config
        # 保存config对象

    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        # 定义一个函数，将输入的tensor进行维度变换，返回变换后的tensor
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))
        # 将tensor从[batch_size, seq_length, all_head_size]形状变换为[batch_size, seq_length, num_attention_heads, attention_head_size]

        return tf.transpose(tensor, perm=[0, 2, 1, 3])
        # 将tensor从[batch_size, seq_length, num_attention_heads, attention_head_size]形状变换为[batch_size, num_attention_heads, seq_length, attention_head_size]

    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
        ```
    ) -> Tuple[tf.Tensor]:
        # 获取隐藏状态张量的批量大小
        batch_size = shape_list(hidden_states)[0]
        # 通过 self.query 对象计算混合的查询层
        mixed_query_layer = self.query(inputs=hidden_states)
        # 通过 self.key 对象计算混合的键层
        mixed_key_layer = self.key(inputs=hidden_states)
        # 通过 self.value 对象计算混合的值层
        mixed_value_layer = self.value(inputs=hidden_states)
        # 将混合的查询层转置以便进行注意力分数计算
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        # 将混合的键层转置以便进行注意力分数计算
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        # 将混合的值层转置以便进行注意力分数计算
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        # 计算查询与键的点积，得到原始注意力分数
        # 形状为 (batch size, num_heads, seq_len_q, seq_len_k)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        # 计算缩放系数 dk，并将注意力分数进行缩放
        dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(attention_scores, dk)

        # 将注意力分数归一化为概率
        attention_probs = stable_softmax(logits=attention_scores, axis=-1)

        # 使用 dropout 随机屏蔽注意力概率中的部分内容，用于模型训练中的稳定性
        attention_probs = self.dropout(inputs=attention_probs, training=training)

        # 如果有头部掩码 head_mask，则将其应用到注意力概率中
        if head_mask is not None:
            attention_probs = tf.multiply(attention_probs, head_mask)

        # 计算注意力输出值
        attention_output = tf.matmul(attention_probs, value_layer)
        # 调整输出张量的维度顺序
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])

        # 重新整形注意力输出张量的形状为 (batch_size, seq_len_q, all_head_size)
        attention_output = tf.reshape(tensor=attention_output, shape=(batch_size, -1, self.all_head_size))
        # 输出包含注意力输出张量和注意力概率的元组，如果输出注意力分布则包含注意力概率
        outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)

        return outputs

    def build(self, input_shape=None):
        # 如果已经构建过网络层，则直接返回
        if self.built:
            return
        # 设置网络层为已构建状态
        self.built = True
        # 如果 self.query 存在，则构建查询层
        if getattr(self, "query", None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.config.hidden_size])
        # 如果 self.key 存在，则构建键层
        if getattr(self, "key", None) is not None:
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.config.hidden_size])
        # 如果 self.value 存在，则构建值层
        if getattr(self, "value", None) is not None:
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.config.hidden_size])
class TFViTSelfOutput(keras.layers.Layer):
    """
    The residual connection is defined in TFViTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: ViTConfig, **kwargs):
        super().__init__(**kwargs)

        # 定义一个全连接层，用于变换隐藏状态的维度
        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 定义一个dropout层，用于随机失活隐藏状态
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        # 保存配置信息
        self.config = config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 对隐藏状态进行全连接层变换
        hidden_states = self.dense(inputs=hidden_states)
        # 在训练时对全连接层输出进行dropout处理
        hidden_states = self.dropout(inputs=hidden_states, training=training)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                # 构建全连接层，指定输入形状和隐藏单元数
                self.dense.build([None, None, self.config.hidden_size])


class TFViTAttention(keras.layers.Layer):
    def __init__(self, config: ViTConfig, **kwargs):
        super().__init__(**kwargs)

        # 定义自注意力层，用于计算注意力分数
        self.self_attention = TFViTSelfAttention(config, name="attention")
        # 定义输出层，处理自注意力层的输出
        self.dense_output = TFViTSelfOutput(config, name="output")

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(
        self,
        input_tensor: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        # 进行自注意力计算，得到自注意力层的输出
        self_outputs = self.self_attention(
            hidden_states=input_tensor, head_mask=head_mask, output_attentions=output_attentions, training=training
        )
        # 经过输出层处理自注意力层的输出
        attention_output = self.dense_output(
            hidden_states=self_outputs[0], input_tensor=input_tensor, training=training
        )
        # 构建最终的输出元组，包括注意力输出和可能的其他返回值
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "self_attention", None) is not None:
            with tf.name_scope(self.self_attention.name):
                # 构建自注意力层
                self.self_attention.build(None)
        if getattr(self, "dense_output", None) is not None:
            with tf.name_scope(self.dense_output.name):
                # 构建输出层
                self.dense_output.build(None)


class TFViTIntermediate(keras.layers.Layer):
    # 此处需要继续完成 TFViTIntermediate 类的注释
    # 初始化方法，用于创建一个新的ViTLayer对象
    def __init__(self, config: ViTConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 创建一个全连接层对象，设置单元数为config.intermediate_size，
        # 内核初始化器为config.initializer_range指定的初始化器，层名为"dense"
        self.dense = keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        # 如果config.hidden_act是字符串类型，则通过get_tf_activation函数获取对应的激活函数
        # 否则直接使用config.hidden_act作为中间激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        
        # 将配置信息保存在self.config中
        self.config = config

    # 调用方法，用于执行实际的前向传播操作
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 将输入hidden_states传入全连接层self.dense，得到输出hidden_states
        hidden_states = self.dense(inputs=hidden_states)
        # 将全连接层的输出hidden_states通过中间激活函数self.intermediate_act_fn进行激活处理
        hidden_states = self.intermediate_act_fn(hidden_states)

        # 返回处理后的hidden_states作为本层的输出
        return hidden_states

    # 构建方法，用于构建层的参数和状态
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        
        # 设置标志位built为True，表示已经构建过
        self.built = True
        
        # 如果self.dense层存在，则使用tf.name_scope设置作用域为self.dense.name，
        # 并构建全连接层self.dense，输入形状为[None, None, self.config.hidden_size]
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
class TFViTOutput(keras.layers.Layer):
    def __init__(self, config: ViTConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，用于将输入转换到隐藏大小
        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建一个 dropout 层，用于在训练时随机失活部分神经元
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 使用全连接层对隐藏状态进行线性变换
        hidden_states = self.dense(inputs=hidden_states)
        # 在训练时对转换后的隐藏状态应用 dropout
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 将 dropout 后的隐藏状态与输入张量相加，实现残差连接
        hidden_states = hidden_states + input_tensor

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            # 使用 dense 层的名称作为命名空间，构建其权重
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])


class TFViTLayer(keras.layers.Layer):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: ViTConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建注意力机制层
        self.attention = TFViTAttention(config, name="attention")
        # 创建中间层
        self.intermediate = TFViTIntermediate(config, name="intermediate")
        # 创建 ViT 输出层
        self.vit_output = TFViTOutput(config, name="output")

        # 创建前层归一化层，用于 ViT 中的前置处理
        self.layernorm_before = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm_before")
        # 创建后层归一化层，用于 ViT 中的后置处理
        self.layernorm_after = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm_after")
        self.config = config

    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        # 对隐藏状态应用前层归一化
        attention_outputs = self.attention(
            input_tensor=self.layernorm_before(inputs=hidden_states),
            head_mask=head_mask,
            output_attentions=output_attentions,
            training=training,
        )
        attention_output = attention_outputs[0]

        # 第一个残差连接
        hidden_states = attention_output + hidden_states

        # 对输出应用后层归一化
        layer_output = self.layernorm_after(inputs=hidden_states)

        intermediate_output = self.intermediate(hidden_states=layer_output)

        # 第二个残差连接
        layer_output = self.vit_output(
            hidden_states=intermediate_output, input_tensor=hidden_states, training=training
        )
        outputs = (layer_output,) + attention_outputs[1:]  # 如果输出注意力信息，则添加到输出中

        return outputs
    # 在构建模型时调用的方法，用于设置模型的各个组件
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回，避免重复构建
        if self.built:
            return
        # 设置标志位，表示模型已经构建
        self.built = True
        
        # 如果存在注意力组件，则构建注意力组件
        if getattr(self, "attention", None) is not None:
            # 在命名空间中构建注意力组件
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        
        # 如果存在中间层组件，则构建中间层组件
        if getattr(self, "intermediate", None) is not None:
            # 在命名空间中构建中间层组件
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        
        # 如果存在ViT输出组件，则构建ViT输出组件
        if getattr(self, "vit_output", None) is not None:
            # 在命名空间中构建ViT输出组件
            with tf.name_scope(self.vit_output.name):
                self.vit_output.build(None)
        
        # 如果存在层归一化前组件，则构建层归一化前组件
        if getattr(self, "layernorm_before", None) is not None:
            # 在命名空间中构建层归一化前组件，输入形状为 [None, None, self.config.hidden_size]
            with tf.name_scope(self.layernorm_before.name):
                self.layernorm_before.build([None, None, self.config.hidden_size])
        
        # 如果存在层归一化后组件，则构建层归一化后组件
        if getattr(self, "layernorm_after", None) is not None:
            # 在命名空间中构建层归一化后组件，输入形状为 [None, None, self.config.hidden_size]
            with tf.name_scope(self.layernorm_after.name):
                self.layernorm_after.build([None, None, self.config.hidden_size])
class TFViTEncoder(keras.layers.Layer):
    def __init__(self, config: ViTConfig, **kwargs):
        super().__init__(**kwargs)

        # 初始化 ViT 编码器的各个层
        self.layer = [TFViTLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]

    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        # 初始化输出变量
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # 遍历每个编码器层
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                # 如果需要输出隐藏状态，则记录当前隐藏状态
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 调用当前编码器层的计算
            layer_outputs = layer_module(
                hidden_states=hidden_states,
                head_mask=head_mask[i],
                output_attentions=output_attentions,
                training=training,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                # 如果需要输出注意力权重，则记录当前层的注意力权重
                all_attentions = all_attentions + (layer_outputs[1],)

        # 添加最后一个编码器层的隐藏状态输出
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 根据 return_dict 参数决定返回结果的形式
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

        # 返回 TFBaseModelOutput 对象，包含最后隐藏状态、所有隐藏状态和所有注意力权重
        return TFBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )

    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        self.built = True
        # 构建每个编码器层
        if getattr(self, "layer", None) is not None:
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    layer.build(None)


@keras_serializable
class TFViTMainLayer(keras.layers.Layer):
    config_class = ViTConfig

    def __init__(self, config: ViTConfig, add_pooling_layer: bool = True, **kwargs):
        super().__init__(**kwargs)

        # 初始化 ViT 主层的配置
        self.config = config

        # 初始化 ViT 主层的各个子层
        self.embeddings = TFViTEmbeddings(config, name="embeddings")
        self.encoder = TFViTEncoder(config, name="encoder")
        self.layernorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm")
        self.pooler = TFViTPooler(config, name="pooler") if add_pooling_layer else None

    def get_input_embeddings(self) -> keras.layers.Layer:
        # 返回输入嵌入层的 patch embeddings
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 剪枝模型的注意力头，具体实现未给出
        raise NotImplementedError

    @unpack_inputs
    def call(
        self,
        pixel_values: TFModelInputType | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        # 检查是否提供了 pixel_values 参数，如果没有则抛出数值错误
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 通过嵌入层处理输入的像素值，包括插值位置编码和训练模式
        embedding_output = self.embeddings(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            training=training,
        )

        # 如果需要，准备头部遮罩
        # 在 head_mask 中为 1.0 表示保留对应的注意力头部
        # attention_probs 的形状为 bsz x n_heads x N x N
        # 输入的 head_mask 形状为 [num_heads] 或 [num_hidden_layers x num_heads]
        # 将 head_mask 转换为形状 [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            raise NotImplementedError
        else:
            # 如果 head_mask 为 None，则创建一个与隐藏层数量相同的空列表
            head_mask = [None] * self.config.num_hidden_layers

        # 使用编码器处理嵌入的输出，传入头部遮罩和其他可选参数
        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 获取编码器的序列输出
        sequence_output = encoder_outputs[0]
        # 对序列输出进行 LayerNormalization 处理
        sequence_output = self.layernorm(inputs=sequence_output)
        # 如果定义了池化器，则对序列输出进行池化操作
        pooled_output = self.pooler(hidden_states=sequence_output) if self.pooler is not None else None

        # 如果 return_dict 为 False，则返回一个包含序列输出和池化输出的元组，以及其他编码器输出
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        # 如果 return_dict 为 True，则构建 TFBaseModelOutputWithPooling 对象并返回
        return TFBaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 embeddings 属性，则构建 embeddings
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        # 如果存在 encoder 属性，则构建 encoder
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 如果存在 layernorm 属性，则根据配置的隐藏大小构建 layernorm
        if getattr(self, "layernorm", None) is not None:
            with tf.name_scope(self.layernorm.name):
                self.layernorm.build([None, None, self.config.hidden_size])
        # 如果存在 pooler 属性，则构建 pooler
        if getattr(self, "pooler", None) is not None:
            with tf.name_scope(self.pooler.name):
                self.pooler.build(None)
    """
    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a [keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
    behavior.

    <Tip>

    TensorFlow models and layers in `transformers` accept two formats as input:

    - having all inputs as keyword arguments (like PyTorch models), or
    - having all inputs as a list, tuple or dict in the first positional argument.

    The reason the second format is supported is that Keras methods prefer this format when passing inputs to models
    and layers. Because of this support, when using methods like `model.fit()` things should "just work" for you - just
    pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second
    format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with
    the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first
    positional argument:

    - a single Tensor with `pixel_values` only and nothing else: `model(pixel_values)`
    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
    `model([pixel_values, attention_mask])` or `model([pixel_values, attention_mask, token_type_ids])`
    - a dictionary with one or several input Tensors associated to the input names given in the docstring:
    `model({"pixel_values": pixel_values, "token_type_ids": token_type_ids})`

    Note that when creating models and layers with
    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don't need to worry
    about any of this, as you can just pass inputs like you would to any other Python function!

    </Tip>

    Args:
        config ([`ViTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
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
        interpolate_pos_encoding (`bool`, *optional*):
            Whether to interpolate the pre-trained position encodings.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in
            eager mode, in graph mode the value will always be set to True.
        training (`bool`, *optional*, defaults to `False``):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
@add_start_docstrings(
    """
    ViT Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.
    """,
    VIT_START_DOCSTRING,
)
class TFViTModel(TFViTPreTrainedModel):
    def __init__(self, config: ViTConfig, *inputs, add_pooling_layer=True, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # Initialize the main ViT layer using TFViTMainLayer with optional pooling
        self.vit = TFViTMainLayer(config, add_pooling_layer=add_pooling_layer, name="vit")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def call(
        self,
        pixel_values: TFModelInputType | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        # Pass inputs to the ViT model and return outputs
        outputs = self.vit(
            pixel_values=pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
            training=training,
        )

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "vit", None) is not None:
            with tf.name_scope(self.vit.name):
                self.vit.build(None)


class TFViTPooler(keras.layers.Layer):
    def __init__(self, config: ViTConfig, **kwargs):
        super().__init__(**kwargs)

        # Initialize a dense layer for pooling with specified units, activation, and initializer
        self.dense = keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # Pooling operation by taking the hidden state of the first token
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(inputs=first_token_tensor)

        return pooled_output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
    """
    <Tip>

        Note that it's possible to fine-tune ViT on higher resolution images than the ones it has been trained on, by
        setting `interpolate_pos_encoding` to `True` in the forward of the model. This will interpolate the pre-trained
        position embeddings to the higher resolution.

    </Tip>
    """,
    VIT_START_DOCSTRING,
# 定义一个图像分类模型，继承自TFViTPreTrainedModel和TFSequenceClassificationLoss类
class TFViTForImageClassification(TFViTPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config: ViTConfig, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 设置模型的标签数量
        self.num_labels = config.num_labels
        # 创建一个ViT主层对象，不包含池化层
        self.vit = TFViTMainLayer(config, add_pooling_layer=False, name="vit")

        # 分类器头部
        self.classifier = keras.layers.Dense(
            units=config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="classifier",
        )
        # 保存配置对象
        self.config = config

    # 调用模型的前向传播方法，处理输入参数并返回输出结果
    @unpack_inputs
    @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def call(
        self,
        pixel_values: TFModelInputType | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    ) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        # 调用ViT主层的前向传播，获取输出结果
        outputs = self.vit(
            pixel_values=pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
            training=training,
        )
        # 从ViT输出中提取序列输出
        sequence_output = outputs[0]
        # 将序列输出传递给分类器获取logits
        logits = self.classifier(inputs=sequence_output[:, 0, :])
        # 如果存在标签，计算损失
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        # 如果不返回字典格式的输出，按照元组格式构建返回结果
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回TFSequenceClassifierOutput对象，包含损失、logits、隐藏状态和注意力权重
        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    # 定义神经网络层的构建方法，input_shape 参数表示输入形状，默认为 None
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回，避免重复构建
        if self.built:
            return
        # 标记该层已经构建
        self.built = True
        
        # 如果存在名为 "vit" 的属性，执行以下操作
        if getattr(self, "vit", None) is not None:
            # 在 TensorFlow 中创建名为 self.vit.name 的命名空间
            with tf.name_scope(self.vit.name):
                # 调用 self.vit 对象的 build 方法，参数为 None，即不指定输入形状
                self.vit.build(None)
        
        # 如果存在名为 "classifier" 的属性，执行以下操作
        if getattr(self, "classifier", None) is not None:
            # 在 TensorFlow 中创建名为 self.classifier.name 的命名空间
            with tf.name_scope(self.classifier.name):
                # 调用 self.classifier 对象的 build 方法，参数为 [None, None, self.config.hidden_size]
                # 表示指定输入形状为 [None, None, self.config.hidden_size]
                self.classifier.build([None, None, self.config.hidden_size])
```