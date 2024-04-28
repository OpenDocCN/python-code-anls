# `.\transformers\models\bart\modeling_tf_bart.py`

```
# coding=utf-8
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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
""" TF 2.0 Bart model."""


from __future__ import annotations

import random
from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFBaseModelOutputWithPastAndCrossAttentions,
    TFSeq2SeqLMOutput,
    TFSeq2SeqModelOutput,
    TFSeq2SeqSequenceClassifierOutput,
)

# Public API
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFModelInputType,
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_bart import BartConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "facebook/bart-large"
_CONFIG_FOR_DOC = "BartConfig"


LARGE_NEGATIVE = -1e8


def shift_tokens_right(input_ids: tf.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input tokens to the right, inserting decoder start token at the beginning and padding token at the end.

    Args:
        input_ids (:obj:`tf.Tensor`): The input tensor of token ids.
        pad_token_id (:obj:`int`): The id of the padding token.
        decoder_start_token_id (:obj:`int`): The id of the decoder start token.

    Returns:
        :obj:`tf.Tensor`: The shifted input tensor.
    """
    # Cast pad_token_id and decoder_start_token_id to the same type as input_ids
    pad_token_id = tf.cast(pad_token_id, input_ids.dtype)
    decoder_start_token_id = tf.cast(decoder_start_token_id, input_ids.dtype)
    # Create a tensor filled with decoder_start_token_id, of shape (batch_size, 1)
    start_tokens = tf.fill(
        (shape_list(input_ids)[0], 1), tf.convert_to_tensor(decoder_start_token_id, input_ids.dtype)
    )
    # Shift input_ids to the right by one position
    shifted_input_ids = tf.concat([start_tokens, input_ids[:, :-1]], -1)
    # Replace -100 values in labels with pad_token_id
    shifted_input_ids = tf.where(
        shifted_input_ids == -100,
        tf.fill(shape_list(shifted_input_ids), tf.convert_to_tensor(pad_token_id, input_ids.dtype)),
        shifted_input_ids,
    )

    # Assert that all values in shifted_input_ids are greater than or equal to 0
    assert_gte0 = tf.debugging.assert_greater_equal(shifted_input_ids, tf.constant(0, dtype=input_ids.dtype))

    # Ensure the assertion op is called by wrapping the result in an identity no-op
    with tf.control_dependencies([assert_gte0]):
        shifted_input_ids = tf.identity(shifted_input_ids)

    return shifted_input_ids


def _make_causal_mask(input_ids_shape: tf.TensorShape, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.

    Args:
        input_ids_shape (:obj:`tf.TensorShape`): The shape of the input tensor.
        past_key_values_length (:obj:`int`, optional, defaults to 0): The length of past key values.

    Returns:
        :obj:`tf.Tensor`: The causal mask tensor.
    """
    # Create a mask tensor with shape (1, input_seq_length, input_seq_length)
    # where the lower diagonal elements are 1 and upper diagonal elements are 0
    # 获取输入张量的批量大小
    bsz = input_ids_shape[0]
    # 获取输入张量的目标长度
    tgt_len = input_ids_shape[1]
    # 创建一个形状为(tgt_len, tgt_len)的全为LARGE_NEGATIVE的张量
    mask = tf.ones((tgt_len, tgt_len)) * LARGE_NEGATIVE
    # 生成一个长度为tgt_len的1维张量，内容为0到tgt_len-1的连续整数
    mask_cond = tf.range(shape_list(mask)[-1])

    # 将mask中小于mask_cond+1的位置设为0
    mask = tf.where(mask_cond < tf.reshape(mask_cond + 1, (shape_list(mask)[-1], 1)), 0.0, mask)

    # 如果过去的键值长度大于0，则在mask左侧添加相应长度的零向量
    if past_key_values_length > 0:
        mask = tf.concat([tf.zeros((tgt_len, past_key_values_length)), mask], axis=-1)

    # 在批量维度上复制mask，使得其与输入张量的批量大小相匹配
    return tf.tile(mask[None, None, :, :], (bsz, 1, 1, 1))
    # 将注意力掩码从 `[bsz, seq_len]` 扩展到 `[bsz, 1, tgt_seq_len, src_seq_len]`
def _expand_mask(mask: tf.Tensor, tgt_len: Optional[int] = None):
    # 获取掩码的源序列长度
    src_len = shape_list(mask)[1]
    # 如果未提供目标序列长度，则将目标序列长度设置为源序列长度
    tgt_len = tgt_len if tgt_len is not None else src_len
    # 创建一个值为 1 的常量张量
    one_cst = tf.constant(1.0)
    # 将掩码张量转换为与常量张量相同的数据类型
    mask = tf.cast(mask, dtype=one_cst.dtype)
    # 在第二个维度上复制掩码，使其形状变为 `[bsz, 1, tgt_seq_len, src_seq_len]`
    expanded_mask = tf.tile(mask[:, None, None, :], (1, 1, tgt_len, 1))

    # 返回扩展后的掩码，其中未覆盖的位置用 `LARGE_NEGATIVE` 表示
    return (one_cst - expanded_mask) * LARGE_NEGATIVE


class TFBartLearnedPositionalEmbedding(tf.keras.layers.Embedding):
    # 该模块学习位置嵌入，直到固定的最大尺寸
    def __init__(self, num_embeddings: int, embedding_dim: int, **kwargs):
        # Bart 被设置为如果指定了 padding_idx，则将嵌入的 id 偏移 2，并相应调整 num_embeddings。其他模型没有这个 hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim, **kwargs)

    def call(
        self,
        input_shape: Optional[tf.TensorShape] = None,
        past_key_values_length: int = 0,
        position_ids: tf.Tensor | None = None,
    ):
        # 输入预期为大小为 [bsz x seqlen]
        if position_ids is None:
            # 如果未提供位置 id，则根据输入形状生成位置 id
            seq_len = input_shape[1]
            position_ids = tf.range(seq_len, delta=1, name="range")
            position_ids += past_key_values_length

        # 如果位置 id 是张量，则将偏移值的数据类型与位置 id 保持一致
        offset_dtype = position_ids.dtype if isinstance(position_ids, tf.Tensor) else tf.int32
        # 调用父类的 call 方法，并在返回前将偏移值添加到位置 id 上
        return super().call(position_ids + tf.constant(self.offset, dtype=offset_dtype))


class TFBartAttention(tf.keras.layers.Layer):
    # "Attention Is All You Need" 中的多头注意力机制
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.head_dim = embed_dim // num_heads
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        # 初始化线性变换层
        self.k_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="k_proj")
        self.q_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="q_proj")
        self.v_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="v_proj")
        self.out_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="out_proj")

    # 将张量重塑为适合多头注意力计算的形状
    def _shape(self, tensor: tf.Tensor, seq_len: int, bsz: int):
        return tf.transpose(tf.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))
    # 定义Transformer的call方法，用于执行self-attention操作
    def call(
        self,
        # 输入的隐藏状态张量，shape为[batch_size, seq_length, embed_dim]
        hidden_states: tf.Tensor,
        # key和value的状态张量，shape同hidden_states，若为None则默认为None
        key_value_states: tf.Tensor | None = None,
        # 过去的key和value状态的元组，若为None则默认为None
        past_key_value: Tuple[Tuple[tf.Tensor]] | None = None,
        # 注意力遮罩张量，shape为[batch_size, 1, seq_length, seq_length]，若为None则默认为None
        attention_mask: tf.Tensor | None = None,
        # 层级头掩码张量，shape为[batch_size, num_heads, seq_length, seq_length]，若为None则默认为None
        layer_head_mask: tf.Tensor | None = None,
        # 是否处于训练模式的布尔值，若为None则默认为False
        training: Optional[bool] = False,
    # 构建Transformer层，初始化权重
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记已经构建
        self.built = True
        # 构建k_proj子层（可选）
        if getattr(self, "k_proj", None) is not None:
            with tf.name_scope(self.k_proj.name):
                # 构建k_proj层，输入形状为[None, None, embed_dim]
                self.k_proj.build([None, None, self.embed_dim])
        # 构建q_proj子层（可选）
        if getattr(self, "q_proj", None) is not None:
            with tf.name_scope(self.q_proj.name):
                # 构建q_proj层，输入形状为[None, None, embed_dim]
                self.q_proj.build([None, None, self.embed_dim])
        # 构建v_proj子层（可选）
        if getattr(self, "v_proj", None) is not None:
            with tf.name_scope(self.v_proj.name):
                # 构建v_proj层，输入形状为[None, None, embed_dim]
                self.v_proj.build([None, None, self.embed_dim])
        # 构建out_proj子层（可选）
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                # 构建out_proj层，输入形状为[None, None, embed_dim]
                self.out_proj.build([None, None, self.embed_dim])
# 定义了一个 TF BART 编码器层的类
class TFBartEncoderLayer(tf.keras.layers.Layer):
    # 初始化方法，接受一个配置对象和其他参数
    def __init__(self, config: BartConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 设置嵌入维度为配置对象中的模型维度
        self.embed_dim = config.d_model
        # 创建自注意力层对象，使用 TF BART 注意力类，并指定名称
        self.self_attn = TFBartAttention(
            self.embed_dim, config.encoder_attention_heads, dropout=config.attention_dropout, name="self_attn"
        )
        # 创建自注意力层的 LayerNormalization 层，指定名称
        self.self_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        # 创建丢弃层，用于执行丢弃操作
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        # 获取激活函数对象，根据配置中的激活函数类型
        self.activation_fn = get_tf_activation(config.activation_function)
        # 创建激活函数后的丢弃层
        self.activation_dropout = tf.keras.layers.Dropout(config.activation_dropout)
        # 创建第一个全连接层，指定输出维度和名称
        self.fc1 = tf.keras.layers.Dense(config.encoder_ffn_dim, name="fc1")
        # 创建第二个全连接层，输出维度与嵌入维度相同，指定名称
        self.fc2 = tf.keras.layers.Dense(self.embed_dim, name="fc2")
        # 创建最终的 LayerNormalization 层，指定名称
        self.final_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")
        # 保存配置对象
        self.config = config

    # 调用方法，执行编码器层的前向传播
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: np.ndarray | tf.Tensor | None,
        layer_head_mask: tf.Tensor | None,
        training: Optional[bool] = False,
    ) -> tf.Tensor:
        """
        Args:
            hidden_states (`tf.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`tf.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`tf.Tensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`
        """
        # 保存输入的隐藏状态作为残差连接的起始点
        residual = hidden_states
        # 使用自注意力层处理输入的隐藏状态
        hidden_states, self_attn_weights, _ = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, layer_head_mask=layer_head_mask
        )
        # 断言自注意力层的输出形状与残差连接前的形状相同
        tf.debugging.assert_equal(
            shape_list(hidden_states),
            shape_list(residual),
            message=f"Self attn modified the shape of query {shape_list(residual)} to {shape_list(hidden_states)}",
        )
        # 对自注意力层的输出进行丢弃操作
        hidden_states = self.dropout(hidden_states, training=training)
        # 将丢弃后的结果与残差连接起来
        hidden_states = residual + hidden_states
        # 对连接后的结果执行自注意力层的 LayerNormalization
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 保存连接后的结果作为残差连接的起始点
        residual = hidden_states
        # 使用激活函数对第一个全连接层的输出进行激活
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 对激活后的结果执行丢弃操作
        hidden_states = self.activation_dropout(hidden_states, training=training)
        # 使用第二个全连接层进行线性变换
        hidden_states = self.fc2(hidden_states)
        # 对第二个全连接层的输出执行丢弃操作
        hidden_states = self.dropout(hidden_states, training=training)
        # 将第二个全连接层的输出与残差连接起来
        hidden_states = residual + hidden_states
        # 对连接后的结果执行最终的 LayerNormalization
        hidden_states = self.final_layer_norm(hidden_states)

        # 返回编码器层的输出和自注意力权重
        return hidden_states, self_attn_weights
    # 构建模型，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在 self_attn 属性，则构建 self_attn 层
        if getattr(self, "self_attn", None) is not None:
            with tf.name_scope(self.self_attn.name):
                self.self_attn.build(None)
        # 如果存在 self_attn_layer_norm 属性，则构建 self_attn_layer_norm 层
        if getattr(self, "self_attn_layer_norm", None) is not None:
            with tf.name_scope(self.self_attn_layer_norm.name):
                self.self_attn_layer_norm.build([None, None, self.embed_dim])
        # 如果存在 fc1 属性，则构建 fc1 层
        if getattr(self, "fc1", None) is not None:
            with tf.name_scope(self.fc1.name):
                self.fc1.build([None, None, self.embed_dim])
        # 如果存在 fc2 属性，则构建 fc2 层
        if getattr(self, "fc2", None) is not None:
            with tf.name_scope(self.fc2.name):
                self.fc2.build([None, None, self.config.encoder_ffn_dim])
        # 如果存在 final_layer_norm 属性，则构建 final_layer_norm 层
        if getattr(self, "final_layer_norm", None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.embed_dim])
class TFBartDecoderLayer(tf.keras.layers.Layer):
    # 初始化方法，接受配置参数和其他关键字参数
    def __init__(self, config: BartConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 设置嵌入维度为配置中的模型维度
        self.embed_dim = config.d_model
        # 创建自注意力层对象
        self.self_attn = TFBartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            name="self_attn",
            is_decoder=True,
        )
        # 创建Dropout层
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        # 获取激活函数
        self.activation_fn = get_tf_activation(config.activation_function)
        # 创建激活函数的Dropout层
        self.activation_dropout = tf.keras.layers.Dropout(config.activation_dropout)

        # 创建自注意力层的LayerNormalization层
        self.self_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        # 创建编码器注意力层对象
        self.encoder_attn = TFBartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            name="encoder_attn",
            is_decoder=True,
        )
        # 创建编码器注意力层的LayerNormalization层
        self.encoder_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="encoder_attn_layer_norm")
        # 创建全连接层1
        self.fc1 = tf.keras.layers.Dense(config.decoder_ffn_dim, name="fc1")
        # 创建全连接层2
        self.fc2 = tf.keras.layers.Dense(self.embed_dim, name="fc2")
        # 创建最终的LayerNormalization层
        self.final_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")
        # 保存配置参数
        self.config = config

    # 调用方法，接受隐藏状态、注意力掩码等参数
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        encoder_hidden_states: np.ndarray | tf.Tensor | None = None,
        encoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        layer_head_mask: tf.Tensor | None = None,
        cross_attn_layer_head_mask: tf.Tensor | None = None,
        past_key_value: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        training: Optional[bool] = False,
    # 构建模型方法，用于构建Transformer层
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回
        if self.built:
            return
        # 设置模型构建标志为已构建
        self.built = True
        # 如果存在self_attn属性，则构建self attention层
        if getattr(self, "self_attn", None) is not None:
            with tf.name_scope(self.self_attn.name):
                self.self_attn.build(None)
        # 如果存在self_attn_layer_norm属性，则构建self attention层的layer normalization层
        if getattr(self, "self_attn_layer_norm", None) is not None:
            with tf.name_scope(self.self_attn_layer_norm.name):
                self.self_attn_layer_norm.build([None, None, self.embed_dim])
        # 如果存在encoder_attn属性，则构建encoder-decoder attention层
        if getattr(self, "encoder_attn", None) is not None:
            with tf.name_scope(self.encoder_attn.name):
                self.encoder_attn.build(None)
        # 如果存在encoder_attn_layer_norm属性，则构建encoder-decoder attention层的layer normalization层
        if getattr(self, "encoder_attn_layer_norm", None) is not None:
            with tf.name_scope(self.encoder_attn_layer_norm.name):
                self.encoder_attn_layer_norm.build([None, None, self.embed_dim])
        # 如果存在fc1属性，则构建第一个全连接层
        if getattr(self, "fc1", None) is not None:
            with tf.name_scope(self.fc1.name):
                self.fc1.build([None, None, self.embed_dim])
        # 如果存在fc2属性，则构建第二个全连接层
        if getattr(self, "fc2", None) is not None:
            with tf.name_scope(self.fc2.name):
                self.fc2.build([None, None, self.config.decoder_ffn_dim])
        # 如果存在final_layer_norm属性，则构建最终的layer normalization层
        if getattr(self, "final_layer_norm", None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.embed_dim])
class TFBartClassificationHead(tf.keras.layers.Layer):
    """Head for sentence-level classification tasks."""

    def __init__(self, inner_dim: int, num_classes: int, pooler_dropout: float, name: str, **kwargs):
        # 初始化分类头部，设置内部维度、类别数量、池化层的dropout率、名称等参数
        super().__init__(name=name, **kwargs)
        # 全连接层，将输入维度转换为内部维度
        self.dense = tf.keras.layers.Dense(inner_dim, name="dense")
        # Dropout层，用于防止过拟合
        self.dropout = tf.keras.layers.Dropout(pooler_dropout)
        # 输出投影层，将内部维度转换为类别数量
        self.out_proj = tf.keras.layers.Dense(num_classes, name="out_proj")
        # 输入维度等于内部维度
        self.input_dim = inner_dim
        self.inner_dim = inner_dim

    def call(self, inputs):
        # 对输入进行dropout操作
        hidden_states = self.dropout(inputs)
        # 全连接层计算
        hidden_states = self.dense(hidden_states)
        # Tanh激活函数
        hidden_states = tf.keras.activations.tanh(hidden_states)
        # 再次进行dropout操作
        hidden_states = self.dropout(hidden_states)
        # 输出投影计算
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经构建过，则直接返回
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                # 构建全连接层
                self.dense.build([None, None, self.input_dim])
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                # 构建输出投影层
                self.out_proj.build([None, None, self.inner_dim])


class TFBartPretrainedModel(TFPreTrainedModel):
    config_class = BartConfig
    base_model_prefix = "model"

    @property
    def dummy_inputs(self):
        dummy_inputs = super().dummy_inputs
        # Dummy inputs should not contain the default val of 1
        # as this is the padding token and some assertions check it
        dummy_inputs["input_ids"] = dummy_inputs["input_ids"] * 2
        if "decoder_input_ids" in dummy_inputs:
            dummy_inputs["decoder_input_ids"] = dummy_inputs["decoder_input_ids"] * 2
        return dummy_inputs

    def tf_to_pt_weight_rename(self, tf_weight):
        if tf_weight == "model.shared.weight":
            return tf_weight, "model.decoder.embed_tokens.weight"
        else:
            return (tf_weight,)


BART_START_DOCSTRING = r"""
    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
    behavior.

    <Tip>

    TensorFlow models and layers in `transformers` accept two formats as input:

    - having all inputs as keyword arguments (like PyTorch models), or
    - having all inputs as a list, tuple or dict in the first positional argument.

    The reason the second format is supported is that Keras methods prefer this format when passing inputs to models

"""
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
        config ([`BartConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.



# 此部分是代码中的注释，提供了关于如何使用模型的输入以及参数配置的说明
# BART_GENERATION_EXAMPLE 是一个包含 BART 模型生成摘要和填充掩码的示例
# 这里展示了如何使用 BART 模型生成摘要和填充掩码
# 首先导入必要的库和模型
# 创建 BART 模型和分词器
# 定义要进行摘要的文章
# 使用分词器对文章进行编码
# 生成摘要
# 打印生成的摘要

# BART_INPUTS_DOCSTRING 是一个空字符串，用于文档字符串的占位符

# TFBartEncoder 是一个 Transformer 编码器层，由多个自注意力层组成
# 初始化方法接受配置参数和嵌入标记作为输入
# 初始化过程中设置了一些属性，如 dropout、层丢弃率、填充索引、最大位置编码长度、嵌入缩放因子等
# 创建了嵌入标记和位置编码的层
# 创建了多个编码器层，并设置了层名称
# 创建了一个用于归一化嵌入的层
# 设置了嵌入维度
    # 定义一个方法用于调用模型
    def call(
        self,
        input_ids: TFModelInputType | None = None,  # 输入的 token IDs，可以是 TensorFlow 模型输入类型或者 None
        inputs_embeds: np.ndarray | tf.Tensor | None = None,  # 输入的嵌入向量，可以是 numpy 数组、张量或者 None
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 注意力掩码，可以是 numpy 数组、张量或者 None
        head_mask: np.ndarray | tf.Tensor | None = None,  # 注意力头的掩码，可以是 numpy 数组、张量或者 None
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可选布尔值，默认为 None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选布尔值，默认为 None
        return_dict: Optional[bool] = None,  # 是否返回字典格式的输出，可选布尔值，默认为 None
        training: Optional[bool] = False,  # 是否处于训练模式，可选布尔值，默认为 False
    # 构建模型，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 标记模型为已构建
        self.built = True
        # 如果存在嵌入位置，则构建嵌入位置
        if getattr(self, "embed_positions", None) is not None:
            with tf.name_scope(self.embed_positions.name):
                self.embed_positions.build(None)
        # 如果存在层归一化，则构建层归一化
        if getattr(self, "layernorm_embedding", None) is not None:
            with tf.name_scope(self.layernorm_embedding.name):
                self.layernorm_embedding.build([None, None, self.embed_dim])
        # 如果存在层，则遍历每一层并构建
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)
# 使用 keras_serializable 装饰器将类标记为可序列化的
@keras_serializable
# 定义一个 TFBartDecoder 类，继承自 tf.keras.layers.Layer
class TFBartDecoder(tf.keras.layers.Layer):
    # 设置 config_class 属性为 BartConfig
    config_class = BartConfig
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`TFBartDecoderLayer`]

    Args:
        config: BartConfig
        embed_tokens: output embedding
    """

    # 初始化方法，接受 config 和 embed_tokens 作为参数
    def __init__(self, config: BartConfig, embed_tokens: Optional[tf.keras.layers.Embedding] = None, **kwargs):
        super().__init__(**kwargs)
        # 设置 config 属性为传入的 config 参数
        self.config = config
        # 设置 padding_idx 属性为 config 中的 pad_token_id
        self.padding_idx = config.pad_token_id
        # 设置 embed_tokens 属性为传入的 embed_tokens 参数
        self.embed_tokens = embed_tokens
        # 设置 layerdrop 属性为 config 中的 decoder_layerdrop
        self.layerdrop = config.decoder_layerdrop
        # 创建 embed_positions 属性为 TFBartLearnedPositionalEmbedding 对象
        self.embed_positions = TFBartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            name="embed_positions",
        )
        # 设置 embed_scale 属性为 config 中的 d_model 的平方根，如果 scale_embedding 为真则为 1.0
        self.embed_scale = tf.math.sqrt(float(config.d_model)) if config.scale_embedding else 1.0
        # 创建 layers 属性为包含多个 TFBartDecoderLayer 对象的列表
        self.layers = [TFBartDecoderLayer(config, name=f"layers.{i}") for i in range(config.decoder_layers)]
        # 创建 layernorm_embedding 属性为 LayerNormalization 对象
        self.layernorm_embedding = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layernorm_embedding")

        # 创建 dropout 属性为 Dropout 对象
        self.dropout = tf.keras.layers.Dropout(config.dropout)

    # 使用 unpack_inputs 装饰器处理输入参数
    @unpack_inputs
    # 定义 call 方法，接受多个输入参数
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        encoder_hidden_states: np.ndarray | tf.Tensor | None = None,
        encoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        cross_attn_head_mask: np.ndarray | tf.Tensor | None = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    # 定义 build 方法，接受 input_shape 参数，默认为 None
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在 embed_positions 属性，则构建它
        if getattr(self, "embed_positions", None) is not None:
            with tf.name_scope(self.embed_positions.name):
                self.embed_positions.build(None)
        # 如果存在 layernorm_embedding 属性，则构建它
        if getattr(self, "layernorm_embedding", None) is not None:
            with tf.name_scope(self.layernorm_embedding.name):
                self.layernorm_embedding.build([None, None, self.config.d_model])
        # 如果存在 layers 属性，则逐个构建其中的每个层
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)


# 使用 keras_serializable 装饰器将类标记为可序列化的
@keras_serializable
# 定义一个 TFBartMainLayer 类，继承自 tf.keras.layers.Layer
class TFBartMainLayer(tf.keras.layers.Layer):
    # 设置 config_class 属性为 BartConfig
    config_class = BartConfig
    # 初始化方法，接受BartConfig对象作为参数，以及可选的load_weight_prefix和其他关键字参数
    def __init__(self, config: BartConfig, load_weight_prefix=None, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 将传入的config对象保存为实例属性
        self.config = config
        # 创建一个共享的嵌入层，用于编码器和解码器共享
        self.shared = tf.keras.layers.Embedding(
            input_dim=config.vocab_size,
            output_dim=config.d_model,
            # 使用TruncatedNormal初始化器初始化嵌入层的权重
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=self.config.init_std),
            # 给嵌入层命名
            name="model.shared",
        )
        # 设置嵌入层的load_weight_prefix属性，用于指定层的预期名称范围（用于加载/存储权重）
        self.shared.load_weight_prefix = "model.shared" if load_weight_prefix is None else load_weight_prefix

        # 创建Bart编码器，传入config和共享的嵌入层
        self.encoder = TFBartEncoder(config, self.shared, name="encoder")
        # 创建Bart解码器，传入config和共享的嵌入层
        self.decoder = TFBartDecoder(config, self.shared, name="decoder")

    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.shared

    # 设置输入嵌入层
    def set_input_embeddings(self, new_embeddings):
        # 更新共享的嵌入层
        self.shared = new_embeddings
        # 更新编码器的嵌入层
        self.encoder.embed_tokens = self.shared
        # 更新解码器的嵌入层
        self.decoder.embed_tokens = self.shared

    # 调用方法，实现了模型的前向传播
    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        decoder_input_ids: np.ndarray | tf.Tensor | None = None,
        decoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        decoder_position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        decoder_head_mask: np.ndarray | tf.Tensor | None = None,
        cross_attn_head_mask: np.ndarray | tf.Tensor | None = None,
        encoder_outputs: Optional[Union[Tuple, TFBaseModelOutput]] = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        decoder_inputs_embeds: np.ndarray | tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
        **kwargs,
    ):
        # 方法内部的实现已省略，主要用于模型的前向传播

    # 构建模型，主要用于初始化模型的各个层
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回
        if self.built:
            return
        # 将模型标记为已构建
        self.built = True
        # 对共享嵌入层进行构建
        with tf.name_scope(self.shared.load_weight_prefix + "/" + self.shared.name + "/"):
            self.shared.build(None)
        # 如果存在编码器，则对编码器进行构建
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 如果存在解码器，则对解码器进行构建
        if getattr(self, "decoder", None) is not None:
            with tf.name_scope(self.decoder.name):
                self.decoder.build(None)
# 添加起始文档字符串，描述该类是一个裸的 BART 模型，输出原始的隐藏状态而没有特定的头部
# 引用了 BART_START_DOCSTRING
class TFBartModel(TFBartPretrainedModel):
    # 需要加载权重前缀
    _requires_load_weight_prefix = True

    # 初始化方法，接受配置对象和加载权重前缀作为参数
    def __init__(self, config: BartConfig, load_weight_prefix=None, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)
        
        # 创建 TF BART 主层对象
        self.model = TFBartMainLayer(config, load_weight_prefix=load_weight_prefix, name="model")

    # 获取编码器
    def get_encoder(self):
        return self.model.encoder

    # 获取解码器
    def get_decoder(self):
        return self.model.decoder

    # 调用方法，接受多个输入参数，并返回 TFBaseModelOutput 或 tf.Tensor 类型的结果
    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFSeq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        decoder_input_ids: np.ndarray | tf.Tensor | None = None,
        decoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        decoder_position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        decoder_head_mask: np.ndarray | tf.Tensor | None = None,
        cross_attn_head_mask: np.ndarray | tf.Tensor | None = None,
        encoder_outputs: Optional[Union[Tuple, TFBaseModelOutput]] = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        decoder_inputs_embeds: np.ndarray | tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
        **kwargs,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        # 调用 TF BART 主层对象的方法，传入各种参数
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 返回模型输出
        return outputs
    # 定义一个方法，用于处理模型输出
    def serving_output(self, output):
        # 如果配置了使用缓存，从模型输出中获取过去键值对，取第二个值（当前的键值对）
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        # 如果配置了输出隐藏状态，将输出的解码器隐藏状态转换为张量
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置了输出注意力权重，将输出的解码器注意力权重转换为张量
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        # 如果配置了输出注意力权重，将输出的交叉注意力权重转换为张量
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        # 如果配置了输出隐藏状态，将输出的编码器隐藏状态转换为张量
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置了输出注意力权重，将输出的编码器注意力权重转换为张量
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None

        # 返回一个 TFSeq2SeqModelOutput 对象，包含了处理后的模型输出
        return TFSeq2SeqModelOutput(
            last_hidden_state=output.last_hidden_state,
            past_key_values=pkv,
            decoder_hidden_states=dec_hs,
            decoder_attentions=dec_attns,
            cross_attentions=cross_attns,
            encoder_last_hidden_state=output.encoder_last_hidden_state,
            encoder_hidden_states=enc_hs,
            encoder_attentions=enc_attns,
        )

    # 构建方法，用于构建模型
    def build(self, input_shape=None):
        # 如果已经构建过模型，直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果模型存在
        if getattr(self, "model", None) is not None:
            # 在模型的命名空间下构建模型，传入输入形状为 None
            with tf.name_scope(self.model.name):
                self.model.build(None)
class BiasLayer(tf.keras.layers.Layer):
    """
    Bias as a layer. It is used for serialization purposes: `tf.keras.Model.save_weights` stores on a per-layer basis,
    so all weights have to be registered in a layer.
    """

    def __init__(self, shape, initializer, trainable, name, **kwargs):
        # 调用父类的初始化方法
        super().__init__(name=name, **kwargs)
        # 添加一个可训练的权重变量作为偏置
        # 注意：当进行序列化时，该变量的名称不会被添加作用域，因此它不会出现在格式为"outer_layer/inner_layer/.../name:0"的命名中。
        # 而是直接为"name:0"。更多细节请参考：https://github.com/huggingface/transformers/pull/18833#issuecomment-1233090214
        self.bias = self.add_weight(name=name, shape=shape, initializer=initializer, trainable=trainable)

    def call(self, x):
        # 返回输入张量加上偏置的结果
        return x + self.bias


@add_start_docstrings(
    "The BART Model with a language modeling head. Can be used for summarization.",
    BART_START_DOCSTRING,
)
class TFBartForConditionalGeneration(TFBartPretrainedModel, TFCausalLanguageModelingLoss):
    # 需要在加载时忽略的键列表
    _keys_to_ignore_on_load_missing = [r"final_logits_bias"]
    # 需要加载权重时的前缀标记
    _requires_load_weight_prefix = True

    def __init__(self, config, load_weight_prefix=None, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)
        # 创建 BART 主模型层
        self.model = TFBartMainLayer(config, load_weight_prefix=load_weight_prefix, name="model")
        # 标记是否使用缓存
        self.use_cache = config.use_cache
        # 创建一个偏置层，用于添加到最终的逻辑回归中
        # final_bias_logits 在 pytorch 中被注册为缓冲区，为了保持一致性，设置为不可训练。
        self.bias_layer = BiasLayer(
            name="final_logits_bias", shape=[1, config.vocab_size], initializer="zeros", trainable=False
        )

    def get_decoder(self):
        # 获取解码器
        return self.model.decoder

    def get_encoder(self):
        # 获取编码器
        return self.model.encoder

    def get_output_embeddings(self):
        # 获取输出嵌入层
        return self.get_input_embeddings()

    def set_output_embeddings(self, value):
        # 设置输出嵌入层
        self.set_input_embeddings(value)

    def get_bias(self):
        # 获取偏置
        return {"final_logits_bias": self.bias_layer.bias}

    def set_bias(self, value):
        # 替换包含偏置的现有层，以进行正确的（反）序列化
        # 确定词汇表大小
        vocab_size = value["final_logits_bias"].shape[-1]
        # 创建新的偏置层
        self.bias_layer = BiasLayer(
            name="final_logits_bias", shape=[1, vocab_size], initializer="zeros", trainable=False
        )
        # 将值分配给偏置层的权重变量
        self.bias_layer.bias.assign(value["final_logits_bias"])

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(BART_GENERATION_EXAMPLE)
    @unpack_inputs
    # 定义一个方法，用于调用Transformer模型
    def call(
        self,
        # 输入序列的token IDs，可以为None
        input_ids: TFModelInputType | None = None,
        # 输入序列的注意力掩码，可以为NumPy数组或TensorFlow张量，也可以为None
        attention_mask: np.ndarray | tf.Tensor | None = None,
        # 解码器输入序列的token IDs，可以为NumPy数组或TensorFlow张量，也可以为None
        decoder_input_ids: np.ndarray | tf.Tensor | None = None,
        # 解码器输入序列的注意力掩码，可以为NumPy数组或TensorFlow张量，也可以为None
        decoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        # 解码器输入序列的位置IDs，可以为NumPy数组或TensorFlow张量，也可以为None
        decoder_position_ids: np.ndarray | tf.Tensor | None = None,
        # 编码器自注意力机制的头掩码，可以为NumPy数组或TensorFlow张量，也可以为None
        head_mask: np.ndarray | tf.Tensor | None = None,
        # 解码器自注意力机制的头掩码，可以为NumPy数组或TensorFlow张量，也可以为None
        decoder_head_mask: np.ndarray | tf.Tensor | None = None,
        # 编码器-解码器交叉注意力机制的头掩码，可以为NumPy数组或TensorFlow张量，也可以为None
        cross_attn_head_mask: np.ndarray | tf.Tensor | None = None,
        # 编码器输出的对象，包含了各层的输出等信息，默认为None
        encoder_outputs: Optional[TFBaseModelOutput] = None,
        # 编码器历史键值对的元组，可以为None
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        # 输入的嵌入式表示，可以为NumPy数组或TensorFlow张量，也可以为None
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        # 解码器输入的嵌入式表示，可以为NumPy数组或TensorFlow张量，也可以为None
        decoder_inputs_embeds: np.ndarray | tf.Tensor | None = None,
        # 是否使用缓存，可以为True、False或None
        use_cache: Optional[bool] = None,
        # 是否输出注意力权重，可以为True或None
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，可以为True或None
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典格式的输出，可以为True或None
        return_dict: Optional[bool] = None,
        # 标签张量，可以为TensorFlow张量或None
        labels: tf.Tensor | None = None,
        # 是否处于训练模式，可以为True、False或None，默认为False
        training: Optional[bool] = False,
        ) -> Union[TFSeq2SeqLMOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        """

        # 处理标签，将pad_token_id的标签替换为-100，其余不变
        if labels is not None:
            labels = tf.where(
                labels == self.config.pad_token_id,  # 检查是否为pad_token_id
                tf.cast(tf.fill(shape_list(labels), -100), labels.dtype),  # 如果是，用-100填充
                labels,  # 否则保持原样
            )
            # 如果没有提供decoder_input_ids或decoder_inputs_embeds，则根据labels创建decoder_input_ids
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id  # 向右移动tokens
                )

        # 将输入传递给模型进行处理
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 计算LM logits
        lm_logits = tf.matmul(outputs[0], self.model.shared.weights, transpose_b=True)
        lm_logits = self.bias_layer(lm_logits)  # 应用偏置层
        # 计算masked language modeling loss
        masked_lm_loss = None if labels is None else self.hf_compute_loss(labels, lm_logits)

        # 根据return_dict决定返回值形式
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        return TFSeq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,  # 过去的key values，来自输出的索引1
            decoder_hidden_states=outputs.decoder_hidden_states,  # decoder隐藏状态，来自输出的索引2
            decoder_attentions=outputs.decoder_attentions,  # decoder注意力权重，来自输出的索引3
            cross_attentions=outputs.cross_attentions,  # 交叉注意力权重，来自输出的索引4
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,  # encoder最后隐藏状态，来自encoder输出的索引0
            encoder_hidden_states=outputs.encoder_hidden_states,  # encoder隐藏状态，来自encoder输出的索引1
            encoder_attentions=outputs.encoder_attentions,  # encoder注意力权重，来自encoder输出的索引2
        )
    # 定义一个方法用于处理模型输出，准备用于服务的输出
    def serving_output(self, output):
        # 如果配置中使用了缓存，从输出的过去键值对中获取过去键值对的第二个元素
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        # 如果配置中输出了隐藏状态，将解码器隐藏状态转换为张量
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置中输出了注意力，将解码器注意力转换为张量
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        # 如果配置中输出了注意力，将交叉注意力转换为张量
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        # 如果配置中输出了隐藏状态，将编码器隐藏状态转换为张量
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置中输出了注意力，将编码器注意力转换为张量
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None

        # 返回一个包含处理后的输出的 TFSeq2SeqLMOutput 对象
        return TFSeq2SeqLMOutput(
            logits=output.logits,
            past_key_values=pkv,
            decoder_hidden_states=dec_hs,
            decoder_attentions=dec_attns,
            cross_attentions=cross_attns,
            encoder_last_hidden_state=output.encoder_last_hidden_state,
            encoder_hidden_states=enc_hs,
            encoder_attentions=enc_attns,
        )

    # 准备用于生成的输入
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # 如果使用了过去的键值对，只保留解码器输入的最后一个 token
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        # 根据条件设置解码器位置 id
        if decoder_attention_mask is not None:  # xla
            # 使用累积和计算解码器位置 id
            decoder_position_ids = tf.math.cumsum(decoder_attention_mask, axis=-1, exclusive=True)[:, -1:]
        elif past_key_values is not None:  # no xla + past_key_values
            # 如果没有 xla 且有过去的键值对，设置解码器位置 id 为过去键值对的长度
            decoder_position_ids = past_key_values[0][0].shape[2]
        else:  # no xla + no past_key_values
            # 如果没有 xla 且没有过去的键值对，设置解码器位置 id 为序列长度范围
            decoder_position_ids = tf.range(decoder_input_ids.shape[1])

        # 返回准备好的输入字典
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_position_ids": decoder_position_ids,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    # 从标签准备解码器输入 id
    def prepare_decoder_input_ids_from_labels(self, labels: tf.Tensor):
        # 将标签向右移动一位，用于准备解码器的输入
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)
    # 构建模型，如果已经构建过则直接返回
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在模型属性，则构建模型
        if getattr(self, "model", None) is not None:
            # 使用模型的名称创建命名空间
            with tf.name_scope(self.model.name):
                # 构建模型
                self.model.build(None)
        # 如果存在偏置层属性，则构建偏置层
        if getattr(self, "bias_layer", None) is not None:
            # 使用偏置层的名称创建命名空间
            with tf.name_scope(self.bias_layer.name):
                # 构建偏置层
                self.bias_layer.build(None)
# 添加文档字符串，描述这是一个在Bart模型基础上加入序列分类/头的类，例如用于GLUE任务
@add_start_docstrings(
    """
    Bart model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE
    tasks.
    """,
    BART_START_DOCSTRING,
)
class TFBartForSequenceClassification(TFBartPretrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config: BartConfig, load_weight_prefix=None, *inputs, **kwargs):
        # 调用父类的构造函数
        super().__init__(config, *inputs, **kwargs)
        # 创建Bart主层模型
        self.model = TFBartMainLayer(config, load_weight_prefix=load_weight_prefix, name="model")
        # 创建分类头部
        self.classification_head = TFBartClassificationHead(
            config.d_model, config.num_labels, config.classifier_dropout, name="classification_head"
        )

    # 定义模型调用方法，根据给定的输入返回模型输出
    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        decoder_input_ids: np.ndarray | tf.Tensor | None = None,
        decoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        decoder_position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        decoder_head_mask: np.ndarray | tf.Tensor | None = None,
        cross_attn_head_mask: np.ndarray | tf.Tensor | None = None,
        encoder_outputs: Optional[TFBaseModelOutput] = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        decoder_inputs_embeds: np.ndarray | tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: tf.Tensor | None = None,
        training: Optional[bool] = False,
        # 其他参数用于模型调用和训练过程
    # 定义一个方法用于处理模型输出，将输出转换为 TensorFlow 张量，并根据配置选择性地提取不同的信息
    def serving_output(self, output):
        # 将模型输出的logits转换为张量
        logits = tf.convert_to_tensor(output.logits)
        # 如果配置中使用缓存，则提取缓存的过去键值（past key values），否则置为None
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        # 如果配置中要输出隐藏状态，则将decoder_hidden_states转换为张量，否则置为None
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置中要输出decoder的注意力权重，则将decoder_attentions转换为张量，否则置为None
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        # 如果配置中要输出交叉注意力权重，则将cross_attentions转换为张量，否则置为None
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        # 如果配置中要输出encoder的隐藏状态，则将encoder_hidden_states转换为张量，否则置为None
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置中要输出encoder的注意力权重，则将encoder_attentions转换为张量，否则置为None
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None

        # 返回转换后的模型输出
        return TFSeq2SeqSequenceClassifierOutput(
            logits=logits,
            past_key_values=pkv,
            decoder_hidden_states=dec_hs,
            decoder_attentions=dec_attns,
            cross_attentions=cross_attns,
            encoder_last_hidden_state=output.encoder_last_hidden_state,
            encoder_hidden_states=enc_hs,
            encoder_attentions=enc_attns,
        )

    # 构建方法用于构建模型
    def build(self, input_shape=None):
        # 如果已经构建过则直接返回
        if self.built:
            return
        # 将标志置为已构建
        self.built = True
        # 如果存在模型则构建模型
        if getattr(self, "model", None) is not None:
            with tf.name_scope(self.model.name):
                self.model.build(None)
        # 如果存在分类头则构建分类头
        if getattr(self, "classification_head", None) is not None:
            with tf.name_scope(self.classification_head.name):
                self.classification_head.build(None)
```