# `.\models\whisper\modeling_tf_whisper.py`

```
# 设置文件编码为 UTF-8
# Copyright 2022 The OpenAI Authors and The HuggingFace Inc. team. All rights reserved.
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
""" TensorFlow Whisper model."""


from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

# 导入自定义模块
from ...activations_tf import get_tf_activation
from ...generation.configuration_utils import GenerationConfig
from ...generation.tf_logits_process import TFLogitsProcessorList
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFBaseModelOutputWithPastAndCrossAttentions,
    TFSeq2SeqLMOutput,
    TFSeq2SeqModelOutput,
)
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFModelInputType,
    TFPreTrainedModel,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_whisper import WhisperConfig
from .tokenization_whisper import TASK_IDS, TO_LANGUAGE_CODE

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的配置信息
_CONFIG_FOR_DOC = "WhisperConfig"

# 预训练模型的存档列表
TF_WHISPER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "openai/whisper-base",
    # 查看所有 Whisper 模型：https://huggingface.co/models?filter=whisper
]

# 定义一个大负数常量
LARGE_NEGATIVE = -1e8


def sinusoidal_embedding_init(shape, dtype=tf.float32) -> tf.Tensor:
    """Returns sinusoids for positional embedding"""
    # 解构形状元组
    length, channels = shape
    # 如果通道数不能被2整除，则抛出异常
    if channels % 2 != 0:
        raise ValueError(
            f"Number of channels has to be divisible by 2 for sinusoidal positional embeddings, got {channels} channels."
        )
    # 计算时间尺度增量的对数
    log_timescale_increment = math.log(10000) / (channels // 2 - 1)
    # 计算时间尺度的倒数
    inv_timescales = tf.exp(-log_timescale_increment * tf.range(channels // 2, dtype=tf.float32))
    # 缩放时间
    scaled_time = tf.reshape(tf.range(length, dtype=tf.float32), (-1, 1)) * tf.reshape(inv_timescales, (1, -1))
    # 合并正弦和余弦的时间编码
    return tf.cast(tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1), dtype)


# 从 transformers.models.bart.modeling_tf_bart.shift_tokens_right 复制的函数
def shift_tokens_right(input_ids: tf.Tensor, pad_token_id: int, decoder_start_token_id: int):
    # 将 pad_token_id 和 decoder_start_token_id 转换为与 input_ids 相同的数据类型
    pad_token_id = tf.cast(pad_token_id, input_ids.dtype)
    decoder_start_token_id = tf.cast(decoder_start_token_id, input_ids.dtype)
    # 创建起始标记的张量，用于decoder输入的起始
    start_tokens = tf.fill(
        (shape_list(input_ids)[0], 1), tf.convert_to_tensor(decoder_start_token_id, input_ids.dtype)
    )
    
    # 将输入的ids向左移动一个位置，用于生成decoder输入序列
    shifted_input_ids = tf.concat([start_tokens, input_ids[:, :-1]], -1)
    
    # 将labels中可能的-100值替换为`pad_token_id`
    shifted_input_ids = tf.where(
        shifted_input_ids == -100,
        tf.fill(shape_list(shifted_input_ids), tf.convert_to_tensor(pad_token_id, input_ids.dtype)),
        shifted_input_ids,
    )

    # 断言`shifted_input_ids`中的值大于等于0或者为-100
    assert_gte0 = tf.debugging.assert_greater_equal(shifted_input_ids, tf.constant(0, dtype=input_ids.dtype))

    # 确保断言操作被调用，通过在结果外包装一个identity no-op
    with tf.control_dependencies([assert_gte0]):
        shifted_input_ids = tf.identity(shifted_input_ids)

    return shifted_input_ids
# 从transformers库中复制的函数，用于生成用于自注意力的因果遮罩
def _make_causal_mask(input_ids_shape: tf.TensorShape, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    # 获取批量大小
    bsz = input_ids_shape[0]
    # 获取目标序列长度
    tgt_len = input_ids_shape[1]
    # 创建一个形状为(tgt_len, tgt_len)的矩阵，并用大负数初始化
    mask = tf.ones((tgt_len, tgt_len)) * LARGE_NEGATIVE
    # 创建一个与tgt_len长度相等的序列
    mask_cond = tf.range(shape_list(mask)[-1])

    # 将 mask 中小于 mask_cond + 1 的位置置为0
    mask = tf.where(mask_cond < tf.reshape(mask_cond + 1, (shape_list(mask)[-1], 1)), 0.0, mask)

    # 如果过去键值长度大于0，则在mask的左侧连接一个形状为(tgt_len, past_key_values_length)的零矩阵
    if past_key_values_length > 0:
        mask = tf.concat([tf.zeros((tgt_len, past_key_values_length)), mask], axis=-1)

    # 返回形状为(bsz, 1, tgt_len, tgt_len)的mask矩阵
    return tf.tile(mask[None, None, :, :], (bsz, 1, 1, 1))


# 从transformers库中复制的函数，用于将注意力遮罩从[bsz, seq_len]扩展到[bsz, 1, tgt_seq_len, src_seq_len]
def _expand_mask(mask: tf.Tensor, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    # 获取输入mask的源序列长度
    src_len = shape_list(mask)[1]
    # 如果未提供tgt_len，则使用源序列长度
    tgt_len = tgt_len if tgt_len is not None else src_len
    # 创建一个常数张量，值为1.0
    one_cst = tf.constant(1.0)
    # 将mask转换为与one_cst相同的数据类型
    mask = tf.cast(mask, dtype=one_cst.dtype)
    # 在第二维上将mask扩展为形状为[bsz, 1, tgt_len, src_len]的张量
    expanded_mask = tf.tile(mask[:, None, None, :], (1, 1, tgt_len, 1))

    # 返回形状为[bsz, 1, tgt_len, src_len]的扩展后的遮罩，其中未覆盖区域的值乘以一个大负数
    return (one_cst - expanded_mask) * LARGE_NEGATIVE


class TFWhisperPositionalEmbedding(keras.layers.Layer):
    def __init__(
        self,
        num_positions: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        embedding_initializer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_positions = num_positions
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.embedding_initializer = keras.initializers.get(embedding_initializer)

    def build(self, input_shape):
        # 添加名为'weight'的权重，形状为[num_positions, embedding_dim]，由embedding_initializer初始化
        self.weight = self.add_weight(
            name="weight",
            shape=[self.num_positions, self.embedding_dim],
            initializer=self.embedding_initializer,
            trainable=True,
        )
        super().build(input_shape)

    def call(self, input_ids, past_key_values_length=0):
        # 将past_key_values_length转换为tf.int32类型
        past_key_values_length = tf.cast(past_key_values_length, tf.int32)
        # 创建一个序列，从past_key_values_length开始，步长为1，长度为input_ids的第二个维度的长度
        gather_indices = tf.range(tf.shape(input_ids)[1], delta=1) + past_key_values_length
        # 返回根据gather_indices从self.weight中收集的张量
        return tf.gather(self.weight, gather_indices)


class TFWhisperAttention(keras.layers.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

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
        self.dropout = keras.layers.Dropout(dropout)
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        # 初始化用于处理输入的线性层，不使用偏置
        self.k_proj = keras.layers.Dense(embed_dim, use_bias=False, name="k_proj")
        self.v_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="v_proj")
        self.q_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="q_proj")
        self.out_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="out_proj")

    # 从 transformers.models.bart.modeling_tf_bart.TFBartAttention._shape 复制而来，用于整形张量
    def _shape(self, tensor: tf.Tensor, seq_len: int, bsz: int):
        return tf.transpose(tf.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))

    # 从 transformers.models.bart.modeling_tf_bart.TFBartAttention.call 复制而来，用于执行注意力计算
    def call(
        self,
        hidden_states: tf.Tensor,
        key_value_states: tf.Tensor | None = None,
        past_key_value: Tuple[Tuple[tf.Tensor]] | None = None,
        attention_mask: tf.Tensor | None = None,
        layer_head_mask: tf.Tensor | None = None,
        training: Optional[bool] = False,
    ):
        # 如果已经构建则直接返回
        if self.built:
            return
        self.built = True
        # 构建各个线性层
        if getattr(self, "k_proj", None) is not None:
            with tf.name_scope(self.k_proj.name):
                self.k_proj.build([None, None, self.embed_dim])
        if getattr(self, "v_proj", None) is not None:
            with tf.name_scope(self.v_proj.name):
                self.v_proj.build([None, None, self.embed_dim])
        if getattr(self, "q_proj", None) is not None:
            with tf.name_scope(self.q_proj.name):
                self.q_proj.build([None, None, self.embed_dim])
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build([None, None, self.embed_dim])
# 从transformers.models.speech_to_text.modeling_tf_speech_to_text.TFSpeech2TextEncoderLayer复制并修改为Whisper
class TFWhisperEncoderLayer(keras.layers.Layer):
    def __init__(self, config: WhisperConfig, **kwargs):
        super().__init__(**kwargs)
        # 初始化层参数
        self.embed_dim = config.d_model  # 设置嵌入维度为config中的d_model
        # 创建自注意力层对象，使用Whisper的注意力头数和dropout参数
        self.self_attn = TFWhisperAttention(
            self.embed_dim, config.encoder_attention_heads, dropout=config.attention_dropout, name="self_attn"
        )
        # 创建LayerNormalization层，用于自注意力层的归一化
        self.self_attn_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        # 创建dropout层，用于全连接层之前的随机失活
        self.dropout = keras.layers.Dropout(config.dropout)
        # 获取激活函数
        self.activation_fn = get_tf_activation(config.activation_function)
        # 创建dropout层，用于激活函数之后的随机失活
        self.activation_dropout = keras.layers.Dropout(config.activation_dropout)
        # 创建全连接层1，输入维度为config中的encoder_ffn_dim，输出维度保持一致
        self.fc1 = keras.layers.Dense(config.encoder_ffn_dim, name="fc1")
        # 创建全连接层2，输入和输出维度都为嵌入维度
        self.fc2 = keras.layers.Dense(self.embed_dim, name="fc2")
        # 创建LayerNormalization层，用于最终层的归一化
        self.final_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")
        # 保存配置信息
        self.config = config

    def call(
        self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, layer_head_mask: tf.Tensor, training: bool = False
    ):
        """
        Args:
            hidden_states (`tf.Tensor`): 输入到层的张量，形状为`(batch, seq_len, embed_dim)`
            attention_mask (`tf.Tensor`): 注意力遮罩，形状为`(batch, 1, tgt_len, src_len)`，用极大的负值表示填充元素
            layer_head_mask (`tf.Tensor`): 给定层中注意力头的遮罩，形状为`(encoder_attention_heads,)`
            training (bool): 是否处于训练模式
        """
        residual = hidden_states  # 保存输入张量作为残差连接的起始点
        hidden_states = self.self_attn_layer_norm(hidden_states)  # 对输入进行自注意力归一化处理
        # 调用自注意力层，获取输出张量、注意力权重和未使用的附加信息
        hidden_states, self_attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            training=training,
        )

        # 断言自注意力层没有改变查询张量的形状
        tf.debugging.assert_equal(
            shape_list(hidden_states),
            shape_list(residual),
            message=f"Self attn modified the shape of query {shape_list(residual)} to {shape_list(hidden_states)}",
        )

        hidden_states = self.dropout(hidden_states, training=training)  # 应用dropout到自注意力层输出
        hidden_states = residual + hidden_states  # 执行残差连接

        residual = hidden_states  # 更新残差连接的起始点
        hidden_states = self.final_layer_norm(hidden_states)  # 对输出进行最终归一化处理
        hidden_states = self.activation_fn(self.fc1(hidden_states))  # 应用激活函数到第一个全连接层
        hidden_states = self.activation_dropout(hidden_states, training=training)  # 应用dropout到激活函数输出
        hidden_states = self.fc2(hidden_states)  # 应用第二个全连接层
        hidden_states = self.dropout(hidden_states, training=training)  # 应用dropout到第二个全连接层输出
        hidden_states = residual + hidden_states  # 执行最终的残差连接

        return hidden_states, self_attn_weights  # 返回处理后的张量和自注意力权重
    # 在构建网络层之前，检查是否已经构建过，如果已构建则直接返回，避免重复构建
    def build(self, input_shape=None):
        if self.built:
            return
        
        # 设置标志位，表示网络已经构建
        self.built = True
        
        # 如果存在 self_attn 属性，则构建 self attention 层
        if getattr(self, "self_attn", None) is not None:
            with tf.name_scope(self.self_attn.name):
                # 调用 self attention 层的 build 方法，传入 None 作为输入形状
                self.self_attn.build(None)
        
        # 如果存在 self_attn_layer_norm 属性，则构建 layer normalization 层
        if getattr(self, "self_attn_layer_norm", None) is not None:
            with tf.name_scope(self.self_attn_layer_norm.name):
                # 调用 layer normalization 层的 build 方法，传入形状为 [None, None, self.embed_dim]
                self.self_attn_layer_norm.build([None, None, self.embed_dim])
        
        # 如果存在 fc1 属性，则构建第一个全连接层
        if getattr(self, "fc1", None) is not None:
            with tf.name_scope(self.fc1.name):
                # 调用第一个全连接层的 build 方法，传入形状为 [None, None, self.embed_dim]
                self.fc1.build([None, None, self.embed_dim])
        
        # 如果存在 fc2 属性，则构建第二个全连接层
        if getattr(self, "fc2", None) is not None:
            with tf.name_scope(self.fc2.name):
                # 调用第二个全连接层的 build 方法，传入形状为 [None, None, self.config.encoder_ffn_dim]
                self.fc2.build([None, None, self.config.encoder_ffn_dim])
        
        # 如果存在 final_layer_norm 属性，则构建最终的 layer normalization 层
        if getattr(self, "final_layer_norm", None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                # 调用最终 layer normalization 层的 build 方法，传入形状为 [None, None, self.embed_dim]
                self.final_layer_norm.build([None, None, self.embed_dim])
# 从transformers.models.speech_to_text.modeling_tf_speech_to_text.TFSpeech2TextDecoderLayer复制而来，更名为TFWhisperDecoderLayer
class TFWhisperDecoderLayer(keras.layers.Layer):
    def __init__(self, config: WhisperConfig, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = config.d_model  # 设置嵌入维度为config中的d_model值

        # 创建自注意力层，用于处理decoder自身的注意力机制
        self.self_attn = TFWhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            name="self_attn",
            is_decoder=True,
        )
        self.dropout = keras.layers.Dropout(config.dropout)  # dropout层，用于模型训练过程中的随机失活
        self.activation_fn = get_tf_activation(config.activation_function)  # 激活函数，根据config中的激活函数类型获取对应的激活函数
        self.activation_dropout = keras.layers.Dropout(config.activation_dropout)  # 激活函数后的dropout层

        self.self_attn_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")  # 自注意力层的归一化层

        # 创建与encoder交互的注意力层，用于decoder与encoder交互信息
        self.encoder_attn = TFWhisperAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            name="encoder_attn",
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="encoder_attn_layer_norm")  # 与encoder交互注意力层的归一化层

        self.fc1 = keras.layers.Dense(config.decoder_ffn_dim, name="fc1")  # 全连接层1
        self.fc2 = keras.layers.Dense(self.embed_dim, name="fc2")  # 全连接层2，输出维度与嵌入维度相同

        self.final_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")  # 最终的归一化层
        self.config = config  # 保存配置信息

    def call(
        self,
        hidden_states,
        attention_mask: tf.Tensor | None = None,
        encoder_hidden_states: tf.Tensor | None = None,
        encoder_attention_mask: tf.Tensor | None = None,
        layer_head_mask: tf.Tensor | None = None,
        cross_attn_layer_head_mask: tf.Tensor | None = None,
        past_key_value: Tuple[tf.Tensor] | None = None,
        training=False,
    # 如果模型已经建立，则直接返回，不再重复建立
    if self.built:
        return
    
    # 设置标志位，表示模型已经建立
    self.built = True
    
    # 如果存在自注意力层，则构建自注意力层
    if getattr(self, "self_attn", None) is not None:
        with tf.name_scope(self.self_attn.name):
            self.self_attn.build(None)
    
    # 如果存在自注意力层归一化层，则构建该层，输入形状为[None, None, self.embed_dim]
    if getattr(self, "self_attn_layer_norm", None) is not None:
        with tf.name_scope(self.self_attn_layer_norm.name):
            self.self_attn_layer_norm.build([None, None, self.embed_dim])
    
    # 如果存在编码器注意力层，则构建编码器注意力层
    if getattr(self, "encoder_attn", None) is not None:
        with tf.name_scope(self.encoder_attn.name):
            self.encoder_attn.build(None)
    
    # 如果存在编码器注意力层归一化层，则构建该层，输入形状为[None, None, self.embed_dim]
    if getattr(self, "encoder_attn_layer_norm", None) is not None:
        with tf.name_scope(self.encoder_attn_layer_norm.name):
            self.encoder_attn_layer_norm.build([None, None, self.embed_dim])
    
    # 如果存在第一个全连接层，则构建该层，输入形状为[None, None, self.embed_dim]
    if getattr(self, "fc1", None) is not None:
        with tf.name_scope(self.fc1.name):
            self.fc1.build([None, None, self.embed_dim])
    
    # 如果存在第二个全连接层，则构建该层，输入形状为[None, None, self.config.decoder_ffn_dim]
    if getattr(self, "fc2", None) is not None:
        with tf.name_scope(self.fc2.name):
            self.fc2.build([None, None, self.config.decoder_ffn_dim])
    
    # 如果存在最终归一化层，则构建该层，输入形状为[None, None, self.embed_dim]
    if getattr(self, "final_layer_norm", None) is not None:
        with tf.name_scope(self.final_layer_norm.name):
            self.final_layer_norm.build([None, None, self.embed_dim])
class TFWhisperPreTrainedModel(TFPreTrainedModel):
    # 指定配置类为WhisperConfig，用于配置模型参数
    config_class = WhisperConfig
    # 模型基础名称前缀为"model"
    base_model_prefix = "model"
    # 主要输入名称为"input_features"
    main_input_name = "input_features"

    def _get_feat_extract_output_lengths(self, input_lengths: tf.Tensor) -> int:
        """
        计算卷积层的输出长度
        """
        # 根据公式计算卷积层的输出长度
        input_lengths = (input_lengths - 1) // 2 + 1

        return input_lengths

    @property
    def dummy_inputs(self) -> Dict[str, tf.Tensor]:
        """
        构建网络所需的虚拟输入

        Returns:
            `Dict[str, tf.Tensor]`: 虚拟输入字典
        """
        return {
            # 创建形状为[1, num_mel_bins, max_source_positions * 2 - 1]的均匀分布随机张量
            self.main_input_name: tf.random.uniform(
                [1, self.config.num_mel_bins, self.config.max_source_positions * 2 - 1], dtype=tf.float32
            ),
            # 固定形状为[[1, 3]]的整数张量作为decoder的输入id
            "decoder_input_ids": tf.constant([[1, 3]], dtype=tf.int32),
        }

    @property
    def input_signature(self):
        # 定义输入签名，指定输入张量的形状和数据类型
        return {
            "input_features": tf.TensorSpec((None, self.config.num_mel_bins, None), tf.float32, name="input_features"),
            "decoder_input_ids": tf.TensorSpec((None, None), tf.int32, name="decoder_input_ids"),
            "decoder_attention_mask": tf.TensorSpec((None, None), tf.int32, name="decoder_attention_mask"),
        }


WHISPER_START_DOCSTRING = r"""
    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a [keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`WhisperConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""

WHISPER_INPUTS_DOCSTRING = r"""
"""


@keras_serializable
class TFWhisperEncoder(keras.layers.Layer):
    # 指定配置类为WhisperConfig，用于配置编码器参数
    config_class = WhisperConfig
    """
    Transformer编码器，包含config.encoder_layers个自注意力层。每一层是一个[`TFWhisperEncoderLayer`].

    Args:
        config: WhisperConfig
        embed_tokens (TFWhisperEmbedding): 输出嵌入
    """
    # 初始化方法，接收一个WhisperConfig对象和其他关键字参数
    def __init__(self, config: WhisperConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 将传入的config对象保存到self.config中
        self.config = config
        # 从config对象中获取encoder_layerdrop属性并保存到self.layerdrop中
        self.layerdrop = config.encoder_layerdrop

        # 从config对象中获取d_model属性并保存到self.embed_dim中
        self.embed_dim = config.d_model
        # 从config对象中获取num_mel_bins属性并保存到self.num_mel_bins中
        self.num_mel_bins = config.num_mel_bins
        # 从config对象中获取pad_token_id属性并保存到self.padding_idx中
        self.padding_idx = config.pad_token_id
        # 从config对象中获取max_source_positions属性并保存到self.max_source_positions中
        self.max_source_positions = config.max_source_positions
        # 如果config对象中的scale_embedding为True，则计算并保存self.embed_scale为self.embed_dim的平方根，否则为1.0
        self.embed_scale = math.sqrt(self.embed_dim) if config.scale_embedding else 1.0

        # 在call()方法中添加填充以匹配PyTorch实现
        # 创建第一个卷积层，设置卷积核大小为3，步长为1，padding方式为"valid"，并命名为"conv1"
        self.conv1 = keras.layers.Conv1D(self.embed_dim, kernel_size=3, strides=1, padding="valid", name="conv1")
        # 创建第二个卷积层，设置卷积核大小为3，步长为2，padding方式为"valid"，并命名为"conv2"
        self.conv2 = keras.layers.Conv1D(self.embed_dim, kernel_size=3, strides=2, padding="valid", name="conv2")

        # 创建位置嵌入层TFWhisperPositionalEmbedding对象，设置位置数量为self.max_source_positions，嵌入维度为self.embed_dim，
        # 使用sinusoidal_embedding_init作为初始化方法，并命名为"embed_positions"
        self.embed_positions = TFWhisperPositionalEmbedding(
            num_positions=self.max_source_positions,
            embedding_dim=self.embed_dim,
            embedding_initializer=sinusoidal_embedding_init,
            name="embed_positions",
        )
        # 设置位置嵌入层为不可训练状态
        self.embed_positions.trainable = False

        # 创建编码器层列表，包含config.encoder_layers个TFWhisperEncoderLayer对象，每个对象命名为"layers.{i}"，其中i为层的索引
        self.encoder_layers = [TFWhisperEncoderLayer(config, name=f"layers.{i}") for i in range(config.encoder_layers)]
        
        # 创建LayerNormalization层，设置epsilon为1e-5，并命名为"layer_norm"
        self.layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="layer_norm")

        # 创建Dropout层，设置dropout率为config.dropout
        self.dropout = keras.layers.Dropout(config.dropout)

    # 解包输入参数的装饰器函数，定义在call()方法上
    @unpack_inputs
    # 定义call()方法，接收多个参数，用于模型的前向传播
    def call(
        self,
        input_features=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
@keras_serializable
class TFWhisperDecoder(keras.layers.Layer):
    config_class = WhisperConfig
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`TFWhisperDecoderLayer`]

    Args:
        config: WhisperConfig
    """

    def __init__(self, config: WhisperConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.dropout = keras.layers.Dropout(config.dropout)  # 初始化一个丢弃层，使用配置中的丢弃率
        self.layerdrop = config.decoder_layerdrop  # 设置层级丢弃率
        self.padding_idx = config.pad_token_id  # 设置填充标记索引
        self.max_target_positions = config.max_target_positions  # 最大目标位置数
        self.max_source_positions = config.max_source_positions  # 最大源位置数
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0  # 如果配置中启用了嵌入缩放，则计算嵌入缩放值，否则为1.0

        self.embed_tokens = keras.layers.Embedding(
            input_dim=config.vocab_size,
            output_dim=config.d_model,
            embeddings_initializer=keras.initializers.TruncatedNormal(stddev=self.config.init_std),
            name="embed_tokens",
        )  # 初始化嵌入层，用于将输入标记映射到向量空间

        self.embed_positions = TFWhisperPositionalEmbedding(
            self.max_target_positions, config.d_model, name="embed_positions"
        )  # 初始化位置编码器，用于为输入位置信息生成嵌入向量

        self.decoder_layers = [TFWhisperDecoderLayer(config, name=f"layers.{i}") for i in range(config.decoder_layers)]
        # 初始化多层解码器层，每一层是一个 TFWhisperDecoderLayer 对象，索引命名为 layers.{i}

        self.layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="layer_norm")  # 初始化层归一化层

    def get_input_embeddings(self):
        return self.embed_tokens  # 返回输入嵌入层对象

    def set_input_embeddings(self, value):
        self.embed_tokens = value  # 设置输入嵌入层对象为指定值

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        batch_size, seq_len = input_shape[0], input_shape[1]

        combined_attention_mask = tf.cond(
            tf.math.greater(seq_len, 1),
            lambda: _make_causal_mask(input_shape, past_key_values_length=past_key_values_length),
            lambda: _expand_mask(tf.ones((batch_size, seq_len + past_key_values_length)), tgt_len=seq_len),
        )  # 根据输入形状和过去键值长度生成解码器注意力掩码

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )  # 如果存在输入的注意力掩码，则扩展和组合注意力掩码
        return combined_attention_mask

    @unpack_inputs
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        encoder_hidden_states=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
    # 构建模型，如果已经构建过，则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        
        # 设置标志为已构建
        self.built = True
        
        # 如果存在嵌入词向量（embed_tokens）属性，则构建它
        if getattr(self, "embed_tokens", None) is not None:
            with tf.name_scope(self.embed_tokens.name):
                self.embed_tokens.build(None)
        
        # 如果存在嵌入位置信息（embed_positions）属性，则构建它
        if getattr(self, "embed_positions", None) is not None:
            with tf.name_scope(self.embed_positions.name):
                self.embed_positions.build(None)
        
        # 如果存在层归一化（layer_norm）属性，则构建它
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                # 构建层归一化，传入的形状为 [None, None, self.config.d_model]
                self.layer_norm.build([None, None, self.config.d_model])
        
        # 如果存在解码器层（decoder_layers）属性，则依次构建每一层
        if getattr(self, "decoder_layers", None) is not None:
            for layer in self.decoder_layers:
                with tf.name_scope(layer.name):
                    # 构建当前解码器层，传入的形状为 None（未指定具体输入形状）
                    layer.build(None)
# 添加模型的文档字符串，描述该层的输出是裸的隐藏状态，没有特定的头部信息
@add_start_docstrings(
    "The bare Whisper Model outputting raw hidden-states without any specific head on top.",
    WHISPER_START_DOCSTRING,
)
# 使该类可以序列化为Keras模型
@keras_serializable
class TFWhisperMainLayer(keras.layers.Layer):
    # 指定配置类为WhisperConfig
    config_class = WhisperConfig

    # 初始化方法，接受WhisperConfig对象作为参数
    def __init__(self, config: WhisperConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        # 创建Whisper编码器对象
        self.encoder = TFWhisperEncoder(config, name="encoder")
        # 创建Whisper解码器对象
        self.decoder = TFWhisperDecoder(config, name="decoder")

    # 返回解码器的嵌入层
    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    # 设置解码器的嵌入层
    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    # 返回编码器对象
    def get_encoder(self):
        return self.encoder

    # 返回解码器对象
    def get_decoder(self):
        return self.decoder

    # 模型前向传播方法，处理输入特征和解码器的各种输入及其掩码
    @add_start_docstrings_to_model_forward(WHISPER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @unpack_inputs
    def call(
        self,
        input_features=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_position_ids=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
    ):
        # 方法内部建立模型结构，确保只建立一次
        def build(self, input_shape=None):
            if self.built:
                return
            self.built = True
            # 如果存在编码器对象，则在名称作用域内建立编码器
            if getattr(self, "encoder", None) is not None:
                with tf.name_scope(self.encoder.name):
                    self.encoder.build(None)
            # 如果存在解码器对象，则在名称作用域内建立解码器
            if getattr(self, "decoder", None) is not None:
                with tf.name_scope(self.decoder.name):
                    self.decoder.build(None)


# 添加模型的文档字符串，描述该模型输出裸的隐藏状态，没有特定的头部信息
@add_start_docstrings(
    "The bare Whisper Model outputting raw hidden-states without any specific head on top.",
    WHISPER_START_DOCSTRING,
)
# TFWhisperModel继承自TFWhisperPreTrainedModel类
class TFWhisperModel(TFWhisperPreTrainedModel):
    # 初始化方法，接受WhisperConfig对象作为参数
    def __init__(self, config: WhisperConfig, **kwargs):
        super().__init__(config, **kwargs)
        # 创建TFWhisperMainLayer模型对象作为该模型的一部分
        self.model = TFWhisperMainLayer(config, name="model")

    # 返回解码器的嵌入层
    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    # 设置解码器的嵌入层
    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    # 返回模型的编码器对象
    def get_encoder(self):
        return self.model.encoder

    # 返回模型的解码器对象
    def get_decoder(self):
        return self.model.decoder

    # 返回模型的解码器对象
    def decoder(self):
        return self.model.decoder

    # 返回模型的编码器对象
    def encoder(self):
        return self.model.encoder

    # 模型前向传播方法，处理输入特征和解码器的各种输入及其掩码
    @add_start_docstrings_to_model_forward(WHISPER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    @unpack_inputs
    def call(
        self,
        input_features: TFModelInputType | None = None,  # 输入特征，可以是 TensorFlow 模型的输入类型或空
        decoder_input_ids: np.ndarray | tf.Tensor | None = None,  # 解码器输入的 token IDs，可以是 NumPy 数组、TensorFlow 张量或空
        decoder_attention_mask: np.ndarray | tf.Tensor | None = None,  # 解码器的注意力掩码，可以是 NumPy 数组、TensorFlow 张量或空
        decoder_position_ids: np.ndarray | tf.Tensor | None = None,  # 解码器的位置 IDs，可以是 NumPy 数组、TensorFlow 张量或空
        head_mask: np.ndarray | tf.Tensor | None = None,  # 头部掩码，可以是 NumPy 数组、TensorFlow 张量或空
        decoder_head_mask: np.ndarray | tf.Tensor | None = None,  # 解码器头部掩码，可以是 NumPy 数组、TensorFlow 张量或空
        cross_attn_head_mask: np.ndarray | tf.Tensor | None = None,  # 跨注意力头部掩码，可以是 NumPy 数组、TensorFlow 张量或空
        encoder_outputs: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,  # 编码器的输出，可选，包含 NumPy 数组或 TensorFlow 张量的元组的元组
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,  # 过去的键值对，可选，包含 NumPy 数组或 TensorFlow 张量的元组的元组
        decoder_inputs_embeds: Optional[Tuple[Union[np.ndarray, tf.Tensor]]] = None,  # 解码器的嵌入输入，可选，包含 NumPy 数组或 TensorFlow 张量的元组
        use_cache: Optional[bool] = None,  # 是否使用缓存，可选布尔值
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可选布尔值
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选布尔值
        return_dict: Optional[bool] = None,  # 是否返回字典格式结果，可选布尔值
        training: bool = False,  # 是否处于训练模式，默认为 False
    ) -> Union[Tuple[tf.Tensor], TFSeq2SeqModelOutput]:  # 返回值可以是 TensorFlow 张量的元组或 TFSeq2SeqModelOutput 类型

        """
        Returns:
        
        Example:
        
         ```python
         >>> import tensorflow as tf
         >>> from transformers import TFWhisperModel, AutoFeatureExtractor
         >>> from datasets import load_dataset

         >>> model = TFWhisperModel.from_pretrained("openai/whisper-base")
         >>> feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
         >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
         >>> inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="tf")
         >>> input_features = inputs.input_features
         >>> decoder_input_ids = tf.convert_to_tensor([[1, 1]]) * model.config.decoder_start_token_id
         >>> last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
         >>> list(last_hidden_state.shape)
         [1, 2, 512]
         ```
        """
        
        outputs = self.model(  # 调用模型的主体部分，传入各种参数进行计算
            input_features=input_features,  # 输入特征
            decoder_input_ids=decoder_input_ids,  # 解码器输入的 token IDs
            decoder_attention_mask=decoder_attention_mask,  # 解码器的注意力掩码
            decoder_position_ids=decoder_position_ids,  # 解码器的位置 IDs
            head_mask=head_mask,  # 头部掩码
            decoder_head_mask=decoder_head_mask,  # 解码器头部掩码
            cross_attn_head_mask=cross_attn_head_mask,  # 跨注意力头部掩码
            encoder_outputs=encoder_outputs,  # 编码器的输出
            past_key_values=past_key_values,  # 过去的键值对
            decoder_inputs_embeds=decoder_inputs_embeds,  # 解码器的嵌入输入
            use_cache=use_cache,  # 是否使用缓存
            output_attentions=output_attentions,  # 是否输出注意力权重
            output_hidden_states=output_hidden_states,  # 是否输出隐藏状态
            return_dict=return_dict,  # 是否返回字典格式结果
            training=training,  # 是否处于训练模式
        )
        return outputs  # 返回模型计算的结果
    # 定义一个方法用于生成服务端输出
    def serving_output(self, output):
        # 如果配置要求使用缓存，则获取输出中的过去键值对的第二个元素
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        # 如果配置要求输出隐藏状态，则将输出中的解码器隐藏状态转换为张量
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置要求输出注意力分布，则将输出中的解码器注意力分布转换为张量
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        # 如果配置要求输出交叉注意力分布，则将输出中的交叉注意力分布转换为张量
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        # 如果配置要求输出隐藏状态，则将输出中的编码器隐藏状态转换为张量
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置要求输出注意力分布，则将输出中的编码器注意力分布转换为张量
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None

        # 返回一个 TFSeq2SeqModelOutput 对象，包含以下属性
        return TFSeq2SeqModelOutput(
            last_hidden_state=output.last_hidden_state,  # 最后一个隐藏状态
            past_key_values=pkv,  # 过去的键值对
            decoder_hidden_states=dec_hs,  # 解码器隐藏状态
            decoder_attentions=dec_attns,  # 解码器注意力分布
            cross_attentions=cross_attns,  # 交叉注意力分布
            encoder_last_hidden_state=output.encoder_last_hidden_state,  # 编码器最后一个隐藏状态
            encoder_hidden_states=enc_hs,  # 编码器隐藏状态
            encoder_attentions=enc_attns,  # 编码器注意力分布
        )

    # 构建方法用于创建模型
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 设置已构建标志为 True
        self.built = True
        # 如果存在模型对象
        if getattr(self, "model", None) is not None:
            # 使用模型的名称空间，构建模型
            with tf.name_scope(self.model.name):
                self.model.build(None)
@add_start_docstrings(
    "The Whisper Model with a language modeling head. Can be used for automatic speech recognition.",
    WHISPER_START_DOCSTRING,
)
class TFWhisperForConditionalGeneration(TFWhisperPreTrainedModel, TFCausalLanguageModelingLoss):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"encoder.version",
        r"decoder.version",
        r"proj_out.weight",
    ]
    _keys_to_ignore_on_save = [
        r"proj_out.weight",
    ]

    def __init__(self, config: WhisperConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = TFWhisperMainLayer(config, name="model")

    # 返回模型的编码器部分
    def get_encoder(self):
        return self.model.get_encoder()

    # 返回模型的解码器部分
    def get_decoder(self):
        return self.model.get_decoder()

    # 返回模型的输出嵌入层
    def get_output_embeddings(self):
        return self.get_input_embeddings()

    # 设置模型的输出嵌入层
    def set_output_embeddings(self, value):
        self.set_input_embeddings(value)

    # 调整模型的Token嵌入层大小
    def resize_token_embeddings(self, new_num_tokens: int) -> keras.layers.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        return new_embeddings

    # 模型前向传播函数，用于生成输出序列
    @add_start_docstrings_to_model_forward(WHISPER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @unpack_inputs
    def call(
        self,
        input_features: TFModelInputType | None = None,
        decoder_input_ids: np.ndarray | tf.Tensor | None = None,
        decoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        decoder_position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        decoder_head_mask: np.ndarray | tf.Tensor | None = None,
        cross_attn_head_mask: np.ndarray | tf.Tensor | None = None,
        encoder_outputs: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        decoder_inputs_embeds: Optional[Tuple[Union[np.ndarray, tf.Tensor]]] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ):
        # 此处实现模型的具体前向计算逻辑，生成对应的输出

    # 生成函数，用于生成模型的输出序列
    def generate(
        self,
        inputs: Optional[tf.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[TFLogitsProcessorList] = None,
        seed: Optional[List[int]] = None,
        return_timestamps: Optional[bool] = None,
        task: Optional[str] = None,
        language: Optional[str] = None,
        is_multilingual: Optional[bool] = None,
        prompt_ids: Optional[tf.Tensor] = None,
        return_token_timestamps=None,
        **kwargs,
    ):
        # 此处实现生成函数的逻辑，用于根据输入生成模型的输出序列
    def serving_output(self, output):
        # 如果配置要求使用缓存，则从输出的过去键值对中获取第一个元素，否则为 None
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        # 如果配置要求输出隐藏状态，则将输出的解码器隐藏状态转换为张量，否则为 None
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置要求输出注意力权重，则将输出的解码器注意力权重转换为张量，否则为 None
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        # 如果配置要求输出注意力权重，则将输出的交叉注意力权重转换为张量，否则为 None
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        # 如果配置要求输出隐藏状态，则将输出的编码器隐藏状态转换为张量，否则为 None
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置要求输出注意力权重，则将输出的编码器注意力权重转换为张量，否则为 None
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None

        # 返回 TFSeq2SeqLMOutput 对象，其中包括 logits、过去键值对、解码器隐藏状态、解码器注意力权重、
        # 交叉注意力权重、编码器最后隐藏状态、编码器隐藏状态、编码器注意力权重
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

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        use_cache=None,
        encoder_outputs=None,
        attention_mask=None,
        decoder_attention_mask=None,
        **kwargs,
    ):
        # 如果 past_key_values 不为 None，则仅保留 decoder_input_ids 的最后一个位置的标记
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        # 如果存在 decoder_attention_mask，则使用累积和计算 decoder_position_ids 的最后一个位置
        if decoder_attention_mask is not None:  # xla
            decoder_position_ids = tf.math.cumsum(decoder_attention_mask, axis=-1, exclusive=True)[:, -1:]
        # 如果没有 xla 并且存在 past，则使用 past_key_values 中的信息计算 decoder_position_ids
        elif past_key_values is not None:  # no xla + past
            decoder_position_ids = past_key_values[0][0].shape[2]
        # 否则，计算 decoder_position_ids 为 decoder_input_ids 的长度范围
        else:  # no xla + no past
            decoder_position_ids = tf.range(decoder_input_ids.shape[1])
        # 将 decoder_position_ids 广播到与 decoder_input_ids 形状相同
        decoder_position_ids = tf.broadcast_to(decoder_position_ids, decoder_input_ids.shape)

        # 返回输入生成的准备数据字典，包括输入特征、编码器输出、过去键值对、解码器输入标识、缓存使用情况、
        # 解码器注意力掩码、解码器位置标识
        return {
            "input_features": None,  # 传递 None 是为了满足 Keras.layer.__call__ 的要求
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "use_cache": use_cache,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_position_ids": decoder_position_ids,
        }

    def build(self, input_shape=None):
        # 如果已经构建，则直接返回
        if self.built:
            return
        # 标记为已经构建
        self.built = True
        # 如果模型存在，则在其命名作用域内构建模型
        if getattr(self, "model", None) is not None:
            with tf.name_scope(self.model.name):
                self.model.build(None)
```