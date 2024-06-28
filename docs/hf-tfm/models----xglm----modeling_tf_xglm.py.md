# `.\models\xglm\modeling_tf_xglm.py`

```py
# coding=utf-8
# Copyright 2021 The Fairseq Authors The HuggingFace Inc. team. All rights reserved.
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
""" TF 2.0 XGLM model."""

# 导入所需的模块和库
from __future__ import annotations

import math
import random
from typing import Any, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...activations_tf import get_tf_activation

# Public API
from ...file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_tf_outputs import TFBaseModelOutputWithPastAndCrossAttentions, TFCausalLMOutputWithCrossAttentions
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFModelInputType,
    TFPreTrainedModel,
    TFSharedEmbeddings,
    get_initializer,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import logging
from .configuration_xglm import XGLMConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的模型检查点和配置
_CHECKPOINT_FOR_DOC = "facebook/xglm-564M"
_CONFIG_FOR_DOC = "XGLMConfig"

# 预训练模型的存档列表
TF_XGLM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/xglm-564M",
    # See all XGLM models at https://huggingface.co/models?filter=xglm
]

# 定义一个大的负数常量
LARGE_NEGATIVE = -1e8

# 创建正弦位置编码
def create_sinusoidal_positions(num_positions: int, embedding_dim: int, padding_idx: Optional[int]) -> tf.Tensor:
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = tf.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
    emb = tf.expand_dims(tf.range(num_positions, dtype=tf.float32), axis=1) * tf.expand_dims(emb, axis=0)
    emb = tf.reshape(tf.concat([tf.sin(emb), tf.cos(emb)], axis=1), (num_positions, -1))
    if embedding_dim % 2 == 1:
        # 如果embedding_dim是奇数，需要在末尾补零
        emb = tf.concat([emb, tf.zeros((num_positions, 1))], axis=1)
    if padding_idx is not None:
        # 创建用于填充位置的掩码，确保填充位置的位置编码为零
        _padding_mask = tf.concat(
            [
                tf.ones((padding_idx, shape_list(emb)[1])),
                tf.zeros((1, shape_list(emb)[1])),
                tf.ones((shape_list(emb)[0] - padding_idx - 1, shape_list(emb)[1])),
            ],
            axis=0,
        )
        emb *= _padding_mask

    return tf.constant(emb, name="embed_positions")


# 从输入ID创建位置ID
def _create_position_ids_from_input_ids(
    input_ids: tf.Tensor, past_key_values_length: int, padding_idx: Optional[int]
) -> tf.Tensor:
    """
    根据输入的token IDs创建位置 IDs

    Args:
        input_ids (tf.Tensor): 输入的token IDs
        past_key_values_length (int): 过去key values的长度
        padding_idx (Optional[int]): 填充的索引位置

    Returns:
        tf.Tensor: 对应的位置 IDs
    """
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.
    """
    # 使用 TensorFlow 的 where 函数创建一个掩码，标记非填充符号位置为1，填充符号位置为0
    mask = tf.where(input_ids != padding_idx, 1, 0)
    # 计算增量索引，累积非填充位置的数量，并加上过去键值长度，乘以掩码确保只对非填充符号操作
    incremental_indices = (tf.cast(tf.cumsum(mask, axis=1), dtype=mask.dtype) + past_key_values_length) * mask
    # 将增量索引转换为 int64 类型，并加上填充索引，以得到最终的位置编码
    return tf.cast(incremental_indices, dtype=tf.int64) + padding_idx
# 定义一个函数，根据输入的嵌入向量和过去的键值对长度，生成位置ID张量
def _create_position_ids_from_inputs_embeds(
    inputs_embeds: tf.Tensor, past_key_values_length: int, padding_idx: Optional[int]
) -> tf.Tensor:
    """
    Args:
        inputs_embeds: 直接提供的嵌入向量张量
    Returns:
        tf.Tensor: 生成的位置ID张量
    """
    # 获取输入嵌入向量的形状
    input_shape = shape_list(inputs_embeds)[:-1]
    # 获取序列长度
    sequence_length = input_shape[1]

    # 生成从padding_idx + 1到sequence_length + padding_idx + 1的序列，数据类型为tf.int64
    position_ids = tf.range(padding_idx + 1, sequence_length + padding_idx + 1, dtype=tf.int64)

    # 将位置ID张量扩展为与输入形状相同的广播形式，并加上过去键值对长度
    return tf.broadcast_to(tf.expand_dims(position_ids, axis=0), input_shape) + past_key_values_length


# 从transformers.models.bart.modeling_tf_bart._make_causal_mask复制而来
def _make_causal_mask(input_ids_shape: tf.TensorShape, past_key_values_length: int = 0):
    """
    创建用于双向自注意力的因果掩码。
    """
    # 获取批量大小
    bsz = input_ids_shape[0]
    # 目标长度为输入形状的第二维度长度
    tgt_len = input_ids_shape[1]
    # 创建一个初始化为-LARGE_NEGATIVE的全1掩码
    mask = tf.ones((tgt_len, tgt_len)) * LARGE_NEGATIVE
    mask_cond = tf.range(shape_list(mask)[-1])

    # 将掩码中对角线以下的元素设为0
    mask = tf.where(mask_cond < tf.reshape(mask_cond + 1, (shape_list(mask)[-1], 1)), 0.0, mask)

    # 如果过去的键值对长度大于0，则在掩码的左侧填充0
    if past_key_values_length > 0:
        mask = tf.concat([tf.zeros((tgt_len, past_key_values_length)), mask], axis=-1)

    return tf.tile(mask[None, None, :, :], (bsz, 1, 1, 1))


# 从transformers.models.bart.modeling_tf_bart._expand_mask复制而来
def _expand_mask(mask: tf.Tensor, tgt_len: Optional[int] = None):
    """
    将注意力掩码从 `[bsz, seq_len]` 扩展到 `[bsz, 1, tgt_seq_len, src_seq_len]`。
    """
    # 获取源序列长度
    src_len = shape_list(mask)[1]
    # 如果未提供目标序列长度，则默认为源序列长度
    tgt_len = tgt_len if tgt_len is not None else src_len
    one_cst = tf.constant(1.0)
    mask = tf.cast(mask, dtype=one_cst.dtype)
    # 在第二维度上扩展掩码
    expanded_mask = tf.tile(mask[:, None, None, :], (1, 1, tgt_len, 1))

    return (one_cst - expanded_mask) * LARGE_NEGATIVE


# 从transformers.models.bart.modeling_tf_bart.TFXGLMAttention（将Bart更改为XGLM）复制而来
class TFXGLMAttention(keras.layers.Layer):
    """来自"Attention Is All You Need"的多头注意力"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        **kwargs,
        ):
            # 调用父类初始化函数，并传递所有关键字参数
            super().__init__(**kwargs)
            # 设置嵌入维度
            self.embed_dim = embed_dim

            # 设置注意力头数
            self.num_heads = num_heads
            # 创建一个丢弃层，用于在训练时随机丢弃输入单元
            self.dropout = keras.layers.Dropout(dropout)
            # 计算每个注意力头的维度
            self.head_dim = embed_dim // num_heads
            # 如果 embed_dim 不能被 num_heads 整除，抛出数值错误
            if (self.head_dim * num_heads) != self.embed_dim:
                raise ValueError(
                    f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                    f" and `num_heads`: {num_heads})."
                )
            # 缩放因子，用于缩放注意力分数
            self.scaling = self.head_dim**-0.5
            # 是否为解码器的标志
            self.is_decoder = is_decoder

            # 创建用于键、查询、值和输出的全连接层
            self.k_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="k_proj")
            self.q_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="q_proj")
            self.v_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="v_proj")
            self.out_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="out_proj")

        # 将张量重塑为指定形状的私有方法
        def _shape(self, tensor: tf.Tensor, seq_len: int, bsz: int):
            return tf.transpose(tf.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))

        # 前向传播函数，接收多个输入张量，并返回输出张量
        def call(
            self,
            hidden_states: tf.Tensor,
            key_value_states: tf.Tensor | None = None,
            past_key_value: Tuple[Tuple[tf.Tensor]] | None = None,
            attention_mask: tf.Tensor | None = None,
            layer_head_mask: tf.Tensor | None = None,
            training: Optional[bool] = False,
        ):
            # 如果已经构建过，则直接返回
            if self.built:
                return
            # 标记为已构建
            self.built = True
            # 如果存在 k_proj 属性，则构建 k_proj 层
            if getattr(self, "k_proj", None) is not None:
                with tf.name_scope(self.k_proj.name):
                    self.k_proj.build([None, None, self.embed_dim])
            # 如果存在 q_proj 属性，则构建 q_proj 层
            if getattr(self, "q_proj", None) is not None:
                with tf.name_scope(self.q_proj.name):
                    self.q_proj.build([None, None, self.embed_dim])
            # 如果存在 v_proj 属性，则构建 v_proj 层
            if getattr(self, "v_proj", None) is not None:
                with tf.name_scope(self.v_proj.name):
                    self.v_proj.build([None, None, self.embed_dim])
            # 如果存在 out_proj 属性，则构建 out_proj 层
            if getattr(self, "out_proj", None) is not None:
                with tf.name_scope(self.out_proj.name):
                    self.out_proj.build([None, None, self.embed_dim])
# TFXGLMDecoderLayer 类的构造函数，初始化一个解码器层
class TFXGLMDecoderLayer(keras.layers.Layer):
    def __init__(self, config: XGLMConfig, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # 设置嵌入维度为配置中的模型维度
        self.embed_dim = config.d_model
        # 初始化自注意力层对象，用于处理解码器自注意力机制
        self.self_attn = TFXGLMAttention(
            embed_dim=self.embed_dim,
            num_heads=config.attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            name="self_attn",
        )
        # 初始化 dropout 层，用于在训练过程中进行随机失活
        self.dropout = keras.layers.Dropout(config.dropout)
        # 根据配置获取激活函数，并设置激活函数对象
        self.activation_fn = get_tf_activation(config.activation_function)
        # 初始化激活函数后的dropout层，用于激活函数后进行随机失活
        self.activation_dropout = keras.layers.Dropout(config.activation_dropout)

        # 如果配置要求添加跨注意力，初始化编码器注意力层对象
        if config.add_cross_attention:
            self.encoder_attn = TFXGLMAttention(
                embed_dim=self.embed_dim,
                num_heads=config.attention_heads,
                dropout=config.attention_dropout,
                is_decoder=True,
                name="encoder_attn",
            )
            # 初始化编码器注意力层后的 LayerNormalization 层
            self.encoder_attn_layer_norm = keras.layers.LayerNormalization(
                epsilon=1e-5, name="encoder_attn_layer_norm"
            )

        # 初始化自注意力层后的 LayerNormalization 层
        self.self_attn_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        # 初始化全连接层1，用于前馈神经网络的第一层
        self.fc1 = keras.layers.Dense(config.ffn_dim, name="fc1")
        # 初始化全连接层2，用于前馈神经网络的第二层，输出维度为嵌入维度
        self.fc2 = keras.layers.Dense(self.embed_dim, name="fc2")
        # 初始化最终的 LayerNormalization 层
        self.final_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")
        # 保存配置对象到实例变量中
        self.config = config

    # Copied from transformers.models.mbart.modeling_tf_mbart.TFMBartDecoderLayer.call
    # 定义层的调用方法，接受多个输入和参数
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None = None,
        encoder_hidden_states: tf.Tensor | None = None,
        encoder_attention_mask: tf.Tensor | None = None,
        layer_head_mask: tf.Tensor | None = None,
        cross_attn_layer_head_mask: tf.Tensor | None = None,
        past_key_value: Tuple[tf.Tensor] | None = None,
        training: Optional[bool] = False,
        # 返回层的调用结果

        # 隐藏状态：当前层的输入张量
        hidden_states: tf.Tensor,
        # 注意力掩码：用于指定哪些位置的元素需要被注意，哪些不需要
        attention_mask: tf.Tensor | None = None,
        # 编码器隐藏状态：编码器层的输出张量
        encoder_hidden_states: tf.Tensor | None = None,
        # 编码器注意力掩码：编码器层的注意力掩码张量
        encoder_attention_mask: tf.Tensor | None = None,
        # 层头掩码：指定每个注意力头的掩码张量
        layer_head_mask: tf.Tensor | None = None,
        # 跨注意力层头掩码：用于跨注意力的每个注意力头的掩码张量
        cross_attn_layer_head_mask: tf.Tensor | None = None,
        # 过去的键值对：用于缓存的过去键值对元组
        past_key_value: Tuple[tf.Tensor] | None = None,
        # 训练标志：指定是否处于训练模式
        training: Optional[bool] = False,
        # 返回：当前层的调用结果

        # 隐藏状态：当前层的输入张量
        hidden_states: tf.Tensor,
        # 注意力掩码：用于指定哪些位置的元素需要被注意，哪些不需要
        attention_mask: tf.Tensor | None = None,
        # 编码器隐藏状态：编码器层的输出张量
        encoder_hidden_states: tf.Tensor | None = None,
        # 编码器注意力掩码：编码器层的注意力掩码张量
        encoder_attention_mask: tf.Tensor | None = None,
        # 层头掩码：指定每个注意力头的掩码张量
        layer_head_mask: tf.Tensor | None = None,
        # 跨注意力层头掩码：用于跨注意力的每个注意力头的掩码张量
        cross_attn_layer_head_mask: tf.Tensor | None = None,
        # 过去的键值对：用于缓存的过去键值对元组
        past_key_value: Tuple[tf.Tensor] | None = None,
        # 训练标志：指定是否处于训练模式，默认为 False
        training: Optional[bool] = False,
    ) -> tf.Tensor:
        # 返回当前层的调用结果，后续逻辑在具体模型调用时处理
        pass


注释：
    # 如果模型已经构建，则直接返回，避免重复构建
    def build(self, input_shape=None):
        if self.built:
            return
        
        # 标记模型已经构建
        self.built = True
        
        # 如果存在 self_attn 属性，则构建 self_attn 层，并设置作用域名称
        if getattr(self, "self_attn", None) is not None:
            with tf.name_scope(self.self_attn.name):
                self.self_attn.build(None)
        
        # 如果存在 self_attn_layer_norm 属性，则构建 self_attn_layer_norm 层，并设置作用域名称
        if getattr(self, "self_attn_layer_norm", None) is not None:
            with tf.name_scope(self.self_attn_layer_norm.name):
                self.self_attn_layer_norm.build([None, None, self.embed_dim])
        
        # 如果存在 fc1 属性，则构建 fc1 层，并设置作用域名称
        if getattr(self, "fc1", None) is not None:
            with tf.name_scope(self.fc1.name):
                self.fc1.build([None, None, self.embed_dim])
        
        # 如果存在 fc2 属性，则构建 fc2 层，并设置作用域名称
        if getattr(self, "fc2", None) is not None:
            with tf.name_scope(self.fc2.name):
                self.fc2.build([None, None, self.config.ffn_dim])
        
        # 如果存在 final_layer_norm 属性，则构建 final_layer_norm 层，并设置作用域名称
        if getattr(self, "final_layer_norm", None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.embed_dim])
        
        # 如果存在 encoder_attn 属性，则构建 encoder_attn 层，并设置作用域名称
        if getattr(self, "encoder_attn", None) is not None:
            with tf.name_scope(self.encoder_attn.name):
                self.encoder_attn.build(None)
        
        # 如果存在 encoder_attn_layer_norm 属性，则构建 encoder_attn_layer_norm 层，并设置作用域名称
        if getattr(self, "encoder_attn_layer_norm", None) is not None:
            with tf.name_scope(self.encoder_attn_layer_norm.name):
                self.encoder_attn_layer_norm.build([None, None, self.embed_dim])
# 使用 keras_serializable 装饰器将类标记为可序列化，以便可以序列化和反序列化
@keras_serializable
class TFXGLMMainLayer(keras.layers.Layer):
    # 设置配置类，用于该层的配置信息
    config_class = XGLMConfig

    # 初始化方法，接受配置对象和其他参数，继承父类的初始化方法
    def __init__(
        self, config: XGLMConfig, embed_tokens: Optional[TFSharedEmbeddings] = None, *inputs, **kwargs: Any
    ) -> None:
        super().__init__(*inputs, **kwargs)

        # 将配置对象保存为类属性
        self.config = config
        # 设置填充标记的索引
        self.padding_idx = config.pad_token_id
        # 设置最大目标位置
        self.max_target_positions = config.max_position_embeddings
        # 如果配置为缩放嵌入，则计算嵌入比例
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        # 如果提供了嵌入标记，则使用提供的；否则创建一个新的共享嵌入层
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = TFSharedEmbeddings(
                config.vocab_size, config.d_model, self.padding_idx, name="embed_tokens"
            )

        # 设置偏移量为2，用于嵌入位置
        self.offset = 2
        # 创建正弦位置嵌入的权重矩阵
        self._embed_positions_weights = create_sinusoidal_positions(
            num_positions=config.max_position_embeddings + self.offset,
            embedding_dim=config.d_model,
            padding_idx=config.pad_token_id,
        )

        # 设置丢弃层，用于模型训练时的随机丢弃
        self.dropout = keras.layers.Dropout(config.dropout)
        # 创建多层解码器层的列表
        self.layers = [TFXGLMDecoderLayer(config, name=f"layers.{i}") for i in range(config.num_layers)]
        # 设置层丢弃率
        self.layerdrop = config.layerdrop
        # 创建层归一化层，用于归一化层输出
        self.layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="layer_norm")

    # 获取输入嵌入层对象
    def get_input_embeddings(self) -> TFSharedEmbeddings:
        return self.embed_tokens

    # 设置输入嵌入层对象
    def set_input_embeddings(self, value: TFSharedEmbeddings) -> None:
        self.embed_tokens = value

    # 准备解码器的注意力掩码
    def _prepare_decoder_attention_mask(
        self,
        attention_mask: tf.Tensor | None,
        input_shape: tf.TensorShape,
        past_key_values_length: int,
    ) -> tf.Tensor:
        # 创建因果掩码
        combined_attention_mask = _make_causal_mask(input_shape, past_key_values_length)
        # 如果输入序列长度大于1，则使用创建的掩码；否则创建一个全1的掩码
        combined_attention_mask = tf.cond(
            input_shape[-1] > 1, lambda: combined_attention_mask, lambda: tf.ones_like(combined_attention_mask)
        )
        # 如果没有提供额外的注意力掩码，则直接返回组合的注意力掩码
        if attention_mask is None:
            return combined_attention_mask
        # 否则，根据目标序列长度扩展提供的注意力掩码，并与组合的注意力掩码相加
        expand_attention_mask = _expand_mask(attention_mask, tgt_len=input_shape[-1])
        return expand_attention_mask + combined_attention_mask

    # 嵌入位置信息到输入中
    def embed_positions(self, position_ids: np.ndarray | tf.Tensor | None = None) -> tf.Tensor:
        # 将位置 IDs 偏移量加到输入中
        position_ids += self.offset
        # 从位置权重矩阵中根据位置 IDs 获取对应的位置嵌入
        positions = tf.gather(self._embed_positions_weights, position_ids, axis=0)
        return positions

    # 解包输入的装饰器，用于解析传入的输入参数
    @unpack_inputs
    # 定义一个方法 `call`，用于执行模型推断或训练的操作，接受多个输入参数
    def call(
        self,
        input_ids: TFModelInputType | None = None,  # 输入的 token IDs，可以是空值
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 注意力掩码，可以是 NumPy 数组或 TensorFlow 张量，也可以是空值
        position_ids: np.ndarray | tf.Tensor | None = None,  # 位置 IDs，可以是 NumPy 数组或 TensorFlow 张量，也可以是空值
        encoder_hidden_states: np.ndarray | tf.Tensor | None = None,  # 编码器隐藏状态，可以是 NumPy 数组或 TensorFlow 张量，也可以是空值
        encoder_attention_mask: np.ndarray | tf.Tensor | None = None,  # 编码器注意力掩码，可以是 NumPy 数组或 TensorFlow 张量，也可以是空值
        head_mask: np.ndarray | tf.Tensor | None = None,  # 头部掩码，可以是 NumPy 数组或 TensorFlow 张量，也可以是空值
        cross_attn_head_mask: np.ndarray | tf.Tensor | None = None,  # 交叉注意力头部掩码，可以是 NumPy 数组或 TensorFlow 张量，也可以是空值
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,  # 过去的键值对，可以是空值或包含 NumPy 数组或 TensorFlow 张量的元组
        inputs_embeds: np.ndarray | tf.Tensor | None = None,  # 输入嵌入向量，可以是 NumPy 数组或 TensorFlow 张量，也可以是空值
        use_cache: Optional[bool] = None,  # 是否使用缓存，可以是空值
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可以是空值
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可以是空值
        return_dict: Optional[bool] = None,  # 是否返回字典形式的结果，可以是空值
        training: Optional[bool] = False,  # 是否处于训练模式，默认为 False
        **kwargs: Any,  # 其它未指定参数，以字典形式收集
    ):
        # 模型建造方法，如果已经建造过则直接返回
        def build(self, input_shape=None):
            if self.built:
                return
            self.built = True
            # 如果存在层归一化，则进行层归一化的建造
            if getattr(self, "layer_norm", None) is not None:
                with tf.name_scope(self.layer_norm.name):
                    self.layer_norm.build([None, None, self.config.d_model])
            # 如果存在嵌入标记，则进行嵌入标记的建造
            if getattr(self, "embed_tokens", None) is not None:
                with tf.name_scope(self.embed_tokens.name):
                    self.embed_tokens.build(None)
            # 如果存在多层，则遍历每一层并建造
            if getattr(self, "layers", None) is not None:
                for layer in self.layers:
                    with tf.name_scope(layer.name):
                        layer.build(None)
# 导入必要的库和模块
class TFXGLMPreTrainedModel(TFPreTrainedModel):
    # 设置配置类，用于此模型的配置参数
    config_class = XGLMConfig
    # 模型基础名称前缀，通常是 "model"
    base_model_prefix = "model"


# XGLM_START_DOCSTRING 是一个原始字符串，用于文档化模型的基本信息和使用方法
XGLM_START_DOCSTRING = r"""
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

    - a single Tensor with `input_ids` only and nothing else: `model(input_ids)`
    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
    `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`
    - a dictionary with one or several input Tensors associated to the input names given in the docstring:
    `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    Note that when creating models and layers with
    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don't need to worry
    about any of this, as you can just pass inputs like you would to any other Python function!

    </Tip>

    Args:
        config ([`XGLMConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""

# XGLM_INPUTS_DOCSTRING 是一个原始字符串，用于文档化模型输入的信息，但在提供的代码中没有内容

# 使用装饰器 add_start_docstrings 将类的文档字符串与特定描述组合起来
@add_start_docstrings(
    "The bare XGLM Model transformer outputting raw hidden-states without any specific head on top.",
    XGLM_START_DOCSTRING,
)
# TFXGLMModel 类继承自 TFXGLMPreTrainedModel 类，表示一个 Transformer 解码器模型
class TFXGLMModel(TFXGLMPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_layers* layers. Each layer is a [`TFXGLMDecoderLayer`]
    """
    """
    初始化函数，设置模型的配置和嵌入层参数，继承父类的初始化方法。

    Args:
        config: XGLMConfig 类型的配置对象
        embed_tokens: 可选的 TFSharedEmbeddings 类型的嵌入层参数
        *inputs: 可变数量的输入参数
        **kwargs: 可变数量的关键字参数
    """
    super().__init__(config, *inputs, **kwargs)

    # 使用给定的配置和嵌入层参数创建 TF 模型的主层
    self.model = TFXGLMMainLayer(config, embed_tokens=embed_tokens, name="model")

@unpack_inputs
@add_start_docstrings_to_model_forward(XGLM_INPUTS_DOCSTRING)
@add_code_sample_docstrings(
    checkpoint=_CHECKPOINT_FOR_DOC,
    output_type=TFBaseModelOutputWithPastAndCrossAttentions,
    config_class=_CONFIG_FOR_DOC,
)
def call(
    self,
    input_ids: TFModelInputType | None = None,
    attention_mask: np.ndarray | tf.Tensor | None = None,
    position_ids: np.ndarray | tf.Tensor | None = None,
    encoder_hidden_states: np.ndarray | tf.Tensor | None = None,
    encoder_attention_mask: np.ndarray | tf.Tensor | None = None,
    head_mask: np.ndarray | tf.Tensor | None = None,
    cross_attn_head_mask: np.ndarray | tf.Tensor | None = None,
    past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
    inputs_embeds: np.ndarray | tf.Tensor | None = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    training: Optional[bool] = False,
    **kwargs: Any,
) -> Union[TFBaseModelOutputWithPastAndCrossAttentions, Tuple[tf.Tensor]]:
    """
    调用方法，用于模型的前向推断。

    Args:
        input_ids: TFModelInputType 类型或 None，输入的 token IDs
        attention_mask: np.ndarray 或 tf.Tensor 或 None，注意力遮罩
        position_ids: np.ndarray 或 tf.Tensor 或 None，位置 IDs
        encoder_hidden_states: np.ndarray 或 tf.Tensor 或 None，编码器隐藏状态
        encoder_attention_mask: np.ndarray 或 tf.Tensor 或 None，编码器注意力遮罩
        head_mask: np.ndarray 或 tf.Tensor 或 None，注意力头部遮罩
        cross_attn_head_mask: np.ndarray 或 tf.Tensor 或 None，跨注意力头部遮罩
        past_key_values: 可选的 Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]，过去的键值
        inputs_embeds: np.ndarray 或 tf.Tensor 或 None，输入的嵌入
        use_cache: 可选的 bool 类型，是否使用缓存
        output_attentions: 可选的 bool 类型，是否输出注意力权重
        output_hidden_states: 可选的 bool 类型，是否输出隐藏状态
        return_dict: 可选的 bool 类型，是否返回字典格式的输出
        training: 可选的 bool 类型，默认为 False，是否处于训练模式
        **kwargs: 其他关键字参数

    Returns:
        模型输出，可以是 TFBaseModelOutputWithPastAndCrossAttentions 类型或 tf.Tensor 的元组
    """
    # 调用模型的前向计算，传递所有参数
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        head_mask=head_mask,
        cross_attn_head_mask=cross_attn_head_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        training=training,
    )

    return outputs

def build(self, input_shape=None):
    """
    构建方法，用于建立模型的层次结构。

    Args:
        input_shape: 可选的输入形状信息
    """
    if self.built:
        return
    self.built = True
    # 如果存在模型对象，则在其命名空间下建立模型
    if getattr(self, "model", None) is not None:
        with tf.name_scope(self.model.name):
            self.model.build(None)
# 使用装饰器给类添加文档字符串，描述其作为带语言建模头部的 XGLM 模型转换器的特性
@add_start_docstrings(
    """
    The XGLM Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    XGLM_START_DOCSTRING,
)
class TFXGLMForCausalLM(TFXGLMPreTrainedModel, TFCausalLanguageModelingLoss):
    # 模型在加载时忽略的键列表，用于处理缺失的情况
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"model.embed_positions.weights",
        r"lm_head.weight",
    ]
    # 模型在保存时忽略的键列表，用于避免保存不必要的参数
    _keys_to_ignore_on_save = [
        r"model.embed_positions.weights",
    ]

    def __init__(
        self, config: XGLMConfig, embed_tokens: Optional[TFSharedEmbeddings] = None, *inputs: Any, **kwargs: Any
    ) -> None:
        # 调用父类的初始化方法，传递配置和其他参数
        super().__init__(config, *inputs, **kwargs)

        # 创建模型主体层，并命名为 "model"
        self.model = TFXGLMMainLayer(config, embed_tokens=embed_tokens, name="model")
        
        # 创建语言建模头部，使用 Dense 层，不使用偏置，使用指定的初始化器初始化权重
        self.lm_head = keras.layers.Dense(
            config.vocab_size,
            use_bias=False,
            kernel_initializer=get_initializer(config.init_std),
            name="lm_head",
        )
        
        # 保存配置对象
        self.config = config

    # 返回语言建模头部
    def get_output_embeddings(self):
        return self.lm_head

    # 设置新的输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 准备生成时的输入，根据参数和过去的键值决定输入的形式
    def prepare_inputs_for_generation(self, inputs, past_key_values=None, use_cache=None, **kwargs):
        # 如果有过去的键值，只使用输入的最后一个标记
        if past_key_values:
            inputs = tf.expand_dims(inputs[:, -1], -1)

        # 获取位置标识和注意力掩码
        position_ids = kwargs.get("position_ids", None)
        attention_mask = kwargs.get("attention_mask", None)

        # 如果有注意力掩码但没有位置标识，根据掩码计算位置标识
        if attention_mask is not None and position_ids is None:
            position_ids = tf.math.cumsum(attention_mask, axis=-1, exclusive=True)
            if past_key_values:
                position_ids = tf.expand_dims(position_ids[:, -1], -1)

        # 返回准备好的输入字典
        return {
            "input_ids": inputs,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    # 使用装饰器来解包输入，并添加模型前向传播的文档字符串
    @unpack_inputs
    @add_start_docstrings_to_model_forward(XGLM_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFCausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFCausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义一个方法 `call`，用于调用当前类的实例
    def call(
        # 输入模型的标识符，可以是 TensorFlow 模型输入类型或 None
        self,
        input_ids: TFModelInputType | None = None,
        # 注意力掩码，可以是 NumPy 数组或 TensorFlow 张量或 None
        attention_mask: np.ndarray | tf.Tensor | None = None,
        # 位置编码，可以是 NumPy 数组或 TensorFlow 张量或 None
        position_ids: np.ndarray | tf.Tensor | None = None,
        # 编码器隐藏状态，可以是 NumPy 数组或 TensorFlow 张量或 None
        encoder_hidden_states: np.ndarray | tf.Tensor | None = None,
        # 编码器注意力掩码，可以是 NumPy 数组或 TensorFlow 张量或 None
        encoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        # 头部掩码，可以是 NumPy 数组或 TensorFlow 张量或 None
        head_mask: np.ndarray | tf.Tensor | None = None,
        # 跨注意力头部掩码，可以是 NumPy 数组或 TensorFlow 张量或 None
        cross_attn_head_mask: np.ndarray | tf.Tensor | None = None,
        # 过去键值对，类型为可选的元组，每个元组包含 NumPy 数组或 TensorFlow 张量
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        # 输入嵌入，可以是 NumPy 数组或 TensorFlow 张量或 None
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        # 标签，可以是 NumPy 数组或 TensorFlow 张量或 None
        labels: np.ndarray | tf.Tensor | None = None,
        # 是否使用缓存，可以是布尔值或 None
        use_cache: Optional[bool] = None,
        # 是否输出注意力，可以是布尔值或 None
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，可以是布尔值或 None
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典形式的结果，可以是布尔值或 None
        return_dict: Optional[bool] = None,
        # 是否处于训练模式，可以是布尔值，默认为 False
        training: Optional[bool] = False,
        # 其他参数，类型为任意
        **kwargs: Any,
    ) -> Union[TFCausalLMOutputWithCrossAttentions, Tuple[tf.Tensor]]:
        r"""
        labels (`np.ndarray` or `tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """

        # 调用模型进行前向传播，生成输出结果
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 从模型输出的第一个元素中获取隐藏状态
        hidden_states = outputs[0]
        # 使用语言模型头部生成语言模型的逻辑(logits)
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # 将标签向左移动一位，并且截断最后一个逻辑(token)
            labels = tf.concat(
                [labels[:, 1:], tf.fill((labels.shape[0], 1), tf.cast(self.config.pad_token_id, labels.dtype))],
                axis=-1,
            )
            # 计算损失
            loss = self.hf_compute_loss(labels, lm_logits)

        if not return_dict:
            # 如果不返回字典，按顺序返回结果
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回带有交叉注意力的 TF 语言模型输出对象
        return TFCausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        # 标记模型已构建
        self.built = True
        if getattr(self, "model", None) is not None:
            # 构建模型
            with tf.name_scope(self.model.name):
                self.model.build(None)
        if getattr(self, "lm_head", None) is not None:
            # 构建语言模型头部
            with tf.name_scope(self.lm_head.name):
                self.lm_head.build([None, None, self.config.hidden_size])

    def tf_to_pt_weight_rename(self, tf_weight):
        if tf_weight == "lm_head.weight":
            # 重命名权重，将 tf 的 lm_head.weight 映射到 PyTorch 的 model.embed_tokens.weight
            return tf_weight, "model.embed_tokens.weight"
        else:
            return (tf_weight,)
```