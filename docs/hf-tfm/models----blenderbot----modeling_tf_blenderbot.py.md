# `.\models\blenderbot\modeling_tf_blenderbot.py`

```py
# coding=utf-8
# 版权所有 2021 年 Facebook, Inc 和 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）授权；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“现状”分发的软件
# 没有任何明示或暗示的保证或条件。
# 有关特定语言的详细信息，请参阅许可证。
""" TF 2.0 Blenderbot 模型。"""


from __future__ import annotations

import os
import random
import warnings
from typing import List, Optional, Tuple, Union

import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFBaseModelOutputWithPastAndCrossAttentions,
    TFSeq2SeqLMOutput,
    TFSeq2SeqModelOutput,
)

# 公共 API
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFPreTrainedModel,
    keras,
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
from .configuration_blenderbot import BlenderbotConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "facebook/blenderbot-400M-distill"
_CONFIG_FOR_DOC = "BlenderbotConfig"


LARGE_NEGATIVE = -1e8


# 从 transformers.models.bart.modeling_tf_bart.shift_tokens_right 复制而来
# 将输入的 token 向右移动，用于生成过程中的输入
def shift_tokens_right(input_ids: tf.Tensor, pad_token_id: int, decoder_start_token_id: int):
    pad_token_id = tf.cast(pad_token_id, input_ids.dtype)
    decoder_start_token_id = tf.cast(decoder_start_token_id, input_ids.dtype)
    # 创建以 decoder_start_token_id 填充的张量，作为起始 token
    start_tokens = tf.fill(
        (shape_list(input_ids)[0], 1), tf.convert_to_tensor(decoder_start_token_id, input_ids.dtype)
    )
    # 向右移动输入的 token ids
    shifted_input_ids = tf.concat([start_tokens, input_ids[:, :-1]], -1)
    # 将标签中可能存在的 -100 值替换为 pad_token_id
    shifted_input_ids = tf.where(
        shifted_input_ids == -100,
        tf.fill(shape_list(shifted_input_ids), tf.convert_to_tensor(pad_token_id, input_ids.dtype)),
        shifted_input_ids,
    )

    # 确保 `labels` 只包含正值和 -100
    assert_gte0 = tf.debugging.assert_greater_equal(shifted_input_ids, tf.constant(0, dtype=input_ids.dtype))

    # 确保断言操作在调用时不会被优化掉
    with tf.control_dependencies([assert_gte0]):
        shifted_input_ids = tf.identity(shifted_input_ids)

    return shifted_input_ids

# 从 transformers.models.bart.modeling_tf_bart._make_causal_mask 复制而来
# 创建用于自注意力的因果遮罩，用于单向解码器自注意力机制
def _make_causal_mask(input_ids_shape: tf.TensorShape, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    创建用于双向自注意力的因果遮罩。
    """
    # 获取批量大小
    bsz = input_ids_shape[0]
    # 获取目标序列长度
    tgt_len = input_ids_shape[1]
    # 创建初始遮罩，设定为非常大的负数
    mask = tf.ones((tgt_len, tgt_len)) * LARGE_NEGATIVE
    # 生成一个序列长度的范围
    mask_cond = tf.range(shape_list(mask)[-1])

    # 将遮罩设定为只对当前位置之前的位置可见，其余为不可见
    mask = tf.where(mask_cond < tf.reshape(mask_cond + 1, (shape_list(mask)[-1], 1)), 0.0, mask)

    # 如果过去的键值长度大于零，则在遮罩的左侧添加零值部分
    if past_key_values_length > 0:
        mask = tf.concat([tf.zeros((tgt_len, past_key_values_length)), mask], axis=-1)

    # 返回扩展后的遮罩，用于模型的自注意力机制
    return tf.tile(mask[None, None, :, :], (bsz, 1, 1, 1))


# Copied from transformers.models.bart.modeling_tf_bart._expand_mask
def _expand_mask(mask: tf.Tensor, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    将注意力遮罩从 `[bsz, seq_len]` 扩展到 `[bsz, 1, tgt_seq_len, src_seq_len]`。
    """
    # 获取源序列的长度
    src_len = shape_list(mask)[1]
    # 如果未指定目标序列的长度，则使用源序列的长度
    tgt_len = tgt_len if tgt_len is not None else src_len
    # 创建一个常数张量为 1.0
    one_cst = tf.constant(1.0)
    # 将遮罩转换为指定数据类型的张量
    mask = tf.cast(mask, dtype=one_cst.dtype)
    # 在遮罩的第二维度上进行扩展，使其变为 `[bsz, 1, tgt_seq_len, src_seq_len]`
    expanded_mask = tf.tile(mask[:, None, None, :], (1, 1, tgt_len, 1))

    # 返回扩展后的遮罩，用于模型的注意力机制
    return (one_cst - expanded_mask) * LARGE_NEGATIVE


class TFBlenderbotLearnedPositionalEmbedding(keras.layers.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    此模块学习位置嵌入，最多到固定的最大大小。
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, **kwargs):
        super().__init__(num_embeddings, embedding_dim, **kwargs)

    def call(
        self, input_shape: tf.TensorShape, past_key_values_length: int = 0, position_ids: tf.Tensor | None = None
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        # 如果未提供位置 ID，则根据输入序列长度生成位置 ID
        if position_ids is None:
            seq_len = input_shape[1]
            position_ids = tf.range(seq_len, delta=1, name="range")
            position_ids += past_key_values_length

        # 调用父类的 call 方法，使用位置 ID 生成位置嵌入
        return super().call(tf.cast(position_ids, dtype=tf.int32))


# Copied from transformers.models.bart.modeling_tf_bart.TFBartAttention with Bart->Blenderbot
class TFBlenderbotAttention(keras.layers.Layer):
    """Multi-headed attention from "Attention Is All You Need"
    多头注意力机制，源自于《Attention Is All You Need》"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        **kwargs,
        # 其他参数
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

        self.k_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="k_proj")
        self.q_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="q_proj")
        self.v_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="v_proj")
        self.out_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="out_proj")


    def _shape(self, tensor: tf.Tensor, seq_len: int, bsz: int):
        # Reshape and transpose the input tensor to match the expected multi-head shape
        return tf.transpose(tf.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))


    def call(
        self,
        hidden_states: tf.Tensor,
        key_value_states: tf.Tensor | None = None,
        past_key_value: Tuple[Tuple[tf.Tensor]] | None = None,
        attention_mask: tf.Tensor | None = None,
        layer_head_mask: tf.Tensor | None = None,
        training: Optional[bool] = False,
    ):
        # Main call function defining how the transformer layer processes inputs


    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "k_proj", None) is not None:
            with tf.name_scope(self.k_proj.name):
                # Build the linear transformation layer for keys
                self.k_proj.build([None, None, self.embed_dim])
        if getattr(self, "q_proj", None) is not None:
            with tf.name_scope(self.q_proj.name):
                # Build the linear transformation layer for queries
                self.q_proj.build([None, None, self.embed_dim])
        if getattr(self, "v_proj", None) is not None:
            with tf.name_scope(self.v_proj.name):
                # Build the linear transformation layer for values
                self.v_proj.build([None, None, self.embed_dim])
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                # Build the final output projection layer
                self.out_proj.build([None, None, self.embed_dim])
# Copied from transformers.models.mbart.modeling_tf_mbart.TFMBartEncoderLayer with MBart->Blenderbot
class TFBlenderbotEncoderLayer(keras.layers.Layer):
    def __init__(self, config: BlenderbotConfig, **kwargs):
        super().__init__(**kwargs)
        # 初始化层的参数，包括嵌入维度和注意力机制相关组件
        self.embed_dim = config.d_model
        self.self_attn = TFBlenderbotAttention(
            self.embed_dim, config.encoder_attention_heads, dropout=config.attention_dropout, name="self_attn"
        )
        self.self_attn_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        self.dropout = keras.layers.Dropout(config.dropout)
        self.activation_fn = get_tf_activation(config.activation_function)
        self.activation_dropout = keras.layers.Dropout(config.activation_dropout)
        self.fc1 = keras.layers.Dense(config.encoder_ffn_dim, name="fc1")
        self.fc2 = keras.layers.Dense(self.embed_dim, name="fc2")
        self.final_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")
        self.config = config

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        layer_head_mask: tf.Tensor,
        training: Optional[bool] = False,
    ):
        """
        Args:
            hidden_states (`tf.Tensor`): 输入到该层的张量，形状为 *(batch, seq_len, embed_dim)*
            attention_mask (`tf.Tensor`): 注意力掩码张量，大小为 *(batch, 1, tgt_len, src_len)*，其中填充元素由非常大的负值表示。
            layer_head_mask (`tf.Tensor`): 给定层中注意力头的掩码张量，大小为 *(encoder_attention_heads,)*
        """
        residual = hidden_states
        # 对输入的 hidden_states 进行 LayerNormalization 处理
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # 使用 self_attn 处理隐藏状态，得到新的 hidden_states 和注意力权重 self_attn_weights
        hidden_states, self_attn_weights, _ = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, layer_head_mask=layer_head_mask
        )

        # 断言确保 self attn 操作没有改变查询的形状
        tf.debugging.assert_equal(
            shape_list(hidden_states),
            shape_list(residual),
            message=f"Self attn modified the shape of query {shape_list(residual)} to {shape_list(hidden_states)}",
        )

        # 应用 dropout，并将残差连接到处理后的 hidden_states
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        # 再次进行 LayerNormalization 处理
        hidden_states = self.final_layer_norm(hidden_states)
        # 应用激活函数和 dropout 到全连接层 fc1
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.activation_dropout(hidden_states, training=training)
        # 经过全连接层 fc2
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        # 将残差连接到最终的 hidden_states 输出
        hidden_states = residual + hidden_states

        return hidden_states, self_attn_weights
    # 定义模型构建方法，设置输入形状为可选，通常用于构建神经网络模型的层次结构
    def build(self, input_shape=None):
        # 如果模型已经构建完成，则直接返回，避免重复构建
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        
        # 如果存在 self_attn 属性，进行相应的构建操作
        if getattr(self, "self_attn", None) is not None:
            # 在 TensorFlow 中使用命名空间，用于区分不同的操作和变量
            with tf.name_scope(self.self_attn.name):
                # 构建 self_attn 层
                self.self_attn.build(None)
        
        # 如果存在 self_attn_layer_norm 属性，进行相应的构建操作
        if getattr(self, "self_attn_layer_norm", None) is not None:
            with tf.name_scope(self.self_attn_layer_norm.name):
                # 构建 self_attn_layer_norm 层，并指定输入形状
                self.self_attn_layer_norm.build([None, None, self.embed_dim])
        
        # 如果存在 fc1 属性，进行相应的构建操作
        if getattr(self, "fc1", None) is not None:
            with tf.name_scope(self.fc1.name):
                # 构建 fc1 层，并指定输入形状
                self.fc1.build([None, None, self.embed_dim])
        
        # 如果存在 fc2 属性，进行相应的构建操作
        if getattr(self, "fc2", None) is not None:
            with tf.name_scope(self.fc2.name):
                # 构建 fc2 层，并指定输入形状为 encoder_ffn_dim
                self.fc2.build([None, None, self.config.encoder_ffn_dim])
        
        # 如果存在 final_layer_norm 属性，进行相应的构建操作
        if getattr(self, "final_layer_norm", None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                # 构建 final_layer_norm 层，并指定输入形状
                self.final_layer_norm.build([None, None, self.embed_dim])
# 从transformers.models.mbart.modeling_tf_mbart.TFMBartDecoderLayer复制而来，将MBart->Blenderbot
class TFBlenderbotDecoderLayer(keras.layers.Layer):
    def __init__(self, config: BlenderbotConfig, **kwargs):
        super().__init__(**kwargs)
        # 初始化层，设定嵌入维度为config中的d_model
        self.embed_dim = config.d_model
        # 创建self attention层，使用TFBlenderbotAttention
        self.self_attn = TFBlenderbotAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            name="self_attn",
            is_decoder=True,
        )
        # Dropout层，使用配置中的dropout率
        self.dropout = keras.layers.Dropout(config.dropout)
        # 激活函数，根据配置获取对应的TensorFlow激活函数
        self.activation_fn = get_tf_activation(config.activation_function)
        # 激活函数的dropout层，使用配置中的activation_dropout率
        self.activation_dropout = keras.layers.Dropout(config.activation_dropout)

        # LayerNormalization层，用于self attention后的归一化
        self.self_attn_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        # 创建encoder-decoder attention层，使用TFBlenderbotAttention
        self.encoder_attn = TFBlenderbotAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            name="encoder_attn",
            is_decoder=True,
        )
        # LayerNormalization层，用于encoder-decoder attention后的归一化
        self.encoder_attn_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="encoder_attn_layer_norm")
        # 全连接层1，使用配置中的decoder_ffn_dim作为units数目
        self.fc1 = keras.layers.Dense(config.decoder_ffn_dim, name="fc1")
        # 全连接层2，输出维度与嵌入维度相同
        self.fc2 = keras.layers.Dense(self.embed_dim, name="fc2")
        # 最终的LayerNormalization层，用于全连接层输出的归一化
        self.final_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")
        # 保存配置参数
        self.config = config

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
        # 该方法定义了Blenderbot解码器层的前向传播逻辑，包括self attention和encoder-decoder attention的处理
    # 构建函数，用于构建神经网络层次结构
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回，不进行重复构建
        if self.built:
            return
        # 将标志位设置为已构建状态
        self.built = True
        
        # 如果存在 self_attn 属性，则构建 self_attention 层
        if getattr(self, "self_attn", None) is not None:
            # 使用 self_attn 的名称作为命名空间
            with tf.name_scope(self.self_attn.name):
                # 调用 self_attn 对象的 build 方法
                self.self_attn.build(None)
        
        # 如果存在 self_attn_layer_norm 属性，则构建 self_attention 层的 layer normalization
        if getattr(self, "self_attn_layer_norm", None) is not None:
            # 使用 self_attn_layer_norm 的名称作为命名空间
            with tf.name_scope(self.self_attn_layer_norm.name):
                # 调用 self_attn_layer_norm 对象的 build 方法，指定输入形状
                self.self_attn_layer_norm.build([None, None, self.embed_dim])
        
        # 如果存在 encoder_attn 属性，则构建 encoder_attention 层
        if getattr(self, "encoder_attn", None) is not None:
            # 使用 encoder_attn 的名称作为命名空间
            with tf.name_scope(self.encoder_attn.name):
                # 调用 encoder_attn 对象的 build 方法
                self.encoder_attn.build(None)
        
        # 如果存在 encoder_attn_layer_norm 属性，则构建 encoder_attention 层的 layer normalization
        if getattr(self, "encoder_attn_layer_norm", None) is not None:
            # 使用 encoder_attn_layer_norm 的名称作为命名空间
            with tf.name_scope(self.encoder_attn_layer_norm.name):
                # 调用 encoder_attn_layer_norm 对象的 build 方法，指定输入形状
                self.encoder_attn_layer_norm.build([None, None, self.embed_dim])
        
        # 如果存在 fc1 属性，则构建第一个全连接层
        if getattr(self, "fc1", None) is not None:
            # 使用 fc1 的名称作为命名空间
            with tf.name_scope(self.fc1.name):
                # 调用 fc1 对象的 build 方法，指定输入形状
                self.fc1.build([None, None, self.embed_dim])
        
        # 如果存在 fc2 属性，则构建第二个全连接层
        if getattr(self, "fc2", None) is not None:
            # 使用 fc2 的名称作为命名空间
            with tf.name_scope(self.fc2.name):
                # 调用 fc2 对象的 build 方法，指定输入形状为解码器的 FFN 维度
                self.fc2.build([None, None, self.config.decoder_ffn_dim])
        
        # 如果存在 final_layer_norm 属性，则构建最终的 layer normalization 层
        if getattr(self, "final_layer_norm", None) is not None:
            # 使用 final_layer_norm 的名称作为命名空间
            with tf.name_scope(self.final_layer_norm.name):
                # 调用 final_layer_norm 对象的 build 方法，指定输入形状
                self.final_layer_norm.build([None, None, self.embed_dim])
# TFBlenderbotPreTrainedModel 类继承自 TFPreTrainedModel 类，是 Blenderbot 模型的 TensorFlow 2.0 Keras 实现。
class TFBlenderbotPreTrainedModel(TFPreTrainedModel):
    # 指定配置类为 BlenderbotConfig
    config_class = BlenderbotConfig
    # 模型的基础名称前缀为 "model"
    base_model_prefix = "model"
    # 打印人类输入的文本 UTTERANCE
    >>> UTTERANCE = "My friends are cool but they eat too many carbs."
    >>> print("Human: ", UTTERANCE)
    
    # 使用 tokenizer 对输入的文本进行处理，并返回 TensorFlow 张量格式的输入
    >>> inputs = tokenizer([UTTERANCE], return_tensors="tf")
    
    # 使用预训练模型生成回复文本的 ID
    >>> reply_ids = model.generate(**inputs)
    
    # 打印机器人生成的回复文本，跳过特殊符号解码
    >>> print("Bot: ", tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0])
    
    # 打印人类输入的 REPLY 文本
    >>> REPLY = "I'm not sure"
    >>> print("Human: ", REPLY)
    
    # 构建下一个对话文本 NEXT_UTTERANCE，包含前一个对话内容和新的问题
    >>> NEXT_UTTERANCE = (
    ...     "My friends are cool but they eat too many carbs.</s> <s>That's unfortunate. "
    ...     "Are they trying to lose weight or are they just trying to be healthier?</s> "
    ...     "<s> I'm not sure."
    ... )
    
    # 使用 tokenizer 对下一个对话文本进行处理，并返回 TensorFlow 张量格式的输入
    >>> inputs = tokenizer([NEXT_UTTERANCE], return_tensors="tf")
    
    # 使用预训练模型生成下一个对话文本的 ID
    >>> next_reply_ids = model.generate(**inputs)
    
    # 打印机器人生成的下一个对话文本，跳过特殊符号解码
    >>> print("Bot: ", tokenizer.batch_decode(next_reply_ids, skip_special_tokens=True)[0])
"""

BLENDERBOT_INPUTS_DOCSTRING = r"""
"""


@keras_serializable
class TFBlenderbotEncoder(keras.layers.Layer):
    config_class = BlenderbotConfig
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`TFBlenderbotEncoderLayer`].

    Args:
        config: BlenderbotConfig
    """

    def __init__(self, config: BlenderbotConfig, embed_tokens: Optional[keras.layers.Embedding] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.dropout = keras.layers.Dropout(config.dropout)  # 初始化一个丢弃层，用于在训练过程中随机丢弃输入
        self.layerdrop = config.encoder_layerdrop  # 从配置中获取层丢弃率，表示在每个训练步骤中丢弃编码器层的概率
        self.padding_idx = config.pad_token_id  # 获取填充标记的索引，用于处理输入序列的填充
        self.max_source_positions = config.max_position_embeddings  # 获取最大源序列位置数，用于限制输入序列的最大长度
        self.embed_scale = tf.math.sqrt(float(config.d_model)) if config.scale_embedding else 1.0  # 根据配置是否缩放嵌入向量的大小

        self.embed_tokens = embed_tokens  # 用于输入序列的嵌入令牌
        self.embed_positions = TFBlenderbotLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            name="embed_positions",
        )  # 初始化学习的位置嵌入层，用于将输入序列的位置编码成向量
        self.layers = [TFBlenderbotEncoderLayer(config, name=f"layers.{i}") for i in range(config.encoder_layers)]  # 创建多层编码器层
        self.layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="layer_norm")  # 初始化层归一化层，用于每个层输出的归一化处理

    def get_embed_tokens(self):
        return self.embed_tokens  # 返回当前嵌入令牌

    def set_embed_tokens(self, embed_tokens):
        self.embed_tokens = embed_tokens  # 设置新的嵌入令牌

    @unpack_inputs
    def call(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
    ):  # 定义 Transformer 编码器的前向传播函数
        """
        参数：
            input_ids: 输入的 token IDs
            inputs_embeds: 替代的嵌入输入
            attention_mask: 注意力掩码，用于指示哪些位置需要注意哪些位置不需要
            head_mask: 多头注意力机制的掩码
            output_attentions: 是否输出注意力权重
            output_hidden_states: 是否输出所有隐藏状态
            return_dict: 是否返回字典格式的输出
            training: 是否处于训练模式
        返回：
            根据配置返回不同格式的输出
        """
        # 以下是前向传播的具体实现，根据输入参数进行不同的计算和处理

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "embed_positions", None) is not None:
            with tf.name_scope(self.embed_positions.name):
                self.embed_positions.build(None)  # 构建位置嵌入层
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.d_model])  # 构建层归一化层
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)  # 构建每一层的编码器层


@keras_serializable
class TFBlenderbotDecoder(keras.layers.Layer):
    config_class = BlenderbotConfig
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`TFBlenderbotDecoderLayer`]

    Args:
        config: BlenderbotConfig
        embed_tokens: output embedding
    """
    # 初始化方法，用于创建一个新的TFBlenderbotDecoder对象
    def __init__(self, config: BlenderbotConfig, embed_tokens: Optional[keras.layers.Embedding] = None, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 将传入的配置对象保存到实例变量中
        self.config = config
        # 将配置中的填充标记ID保存到实例变量中
        self.padding_idx = config.pad_token_id
        # 将传入的嵌入层对象保存到实例变量中
        self.embed_tokens = embed_tokens
        # 从配置中获取解码器层dropout的比例并保存到实例变量中
        self.layerdrop = config.decoder_layerdrop
        # 创建一个学习的位置嵌入对象并保存到实例变量中
        self.embed_positions = TFBlenderbotLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            name="embed_positions",
        )
        # 如果配置中指定了缩放嵌入，则计算并保存嵌入缩放因子；否则设置为1.0
        self.embed_scale = tf.math.sqrt(float(config.d_model)) if config.scale_embedding else 1.0
        # 创建解码器层列表，每个解码器层都使用给定的配置对象进行初始化，并保存到实例变量中
        self.layers = [TFBlenderbotDecoderLayer(config, name=f"layers.{i}") for i in range(config.decoder_layers)]
        # 创建一个层归一化层对象，设置epsilon为1e-5，并保存到实例变量中
        self.layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="layer_norm")

        # 创建一个dropout层对象，并保存到实例变量中，使用配置中的dropout比例
        self.dropout = keras.layers.Dropout(config.dropout)

    # 获取嵌入层对象的方法
    def get_embed_tokens(self):
        return self.embed_tokens

    # 设置嵌入层对象的方法
    def set_embed_tokens(self, embed_tokens):
        self.embed_tokens = embed_tokens

    # 使用@unpack_inputs装饰器标记的调用方法，定义了Blenderbot解码器的前向传播逻辑
    @unpack_inputs
    def call(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
    ):
        # 省略了前向传播的具体实现，根据参数配置实现解码器的逻辑

    # 构建方法，在第一次调用call方法时被调用，用于构建模型的层次结构
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回
        if self.built:
            return
        # 设置模型已构建标志为True
        self.built = True
        # 如果实例中存在embed_positions属性，则构建embed_positions对象
        if getattr(self, "embed_positions", None) is not None:
            with tf.name_scope(self.embed_positions.name):
                self.embed_positions.build(None)
        # 如果实例中存在layer_norm属性，则构建layer_norm对象
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.d_model])
        # 遍历解码器层列表中的每一层，分别构建每一层解码器层对象
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)
# 使用装饰器将类标记为可序列化，适用于Keras
@keras_serializable
class TFBlenderbotMainLayer(keras.layers.Layer):
    # 配置类为BlenderbotConfig
    config_class = BlenderbotConfig

    # 初始化方法，接收BlenderbotConfig实例和其他关键字参数
    def __init__(self, config: BlenderbotConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 将传入的配置对象保存为属性
        self.config = config

        # 创建共享的嵌入层，用于编码器和解码器共享的词汇表和模型尺寸
        self.shared = keras.layers.Embedding(
            input_dim=config.vocab_size,  # 输入维度为词汇表大小
            output_dim=config.d_model,     # 输出维度为模型维度
            embeddings_initializer=keras.initializers.TruncatedNormal(stddev=self.config.init_std),  # 初始化嵌入层的权重
            name="model.shared",  # 层的名称
        )
        
        # 附加属性，指定层的预期名称范围（用于加载/存储权重）
        self.shared.load_weight_prefix = "model.shared"

        # 创建编码器对象，传入配置对象和共享的嵌入层
        self.encoder = TFBlenderbotEncoder(config, self.shared, name="encoder")

        # 创建解码器对象，传入配置对象和共享的嵌入层
        self.decoder = TFBlenderbotDecoder(config, self.shared, name="decoder")

    # 获取输入嵌入层的方法
    def get_input_embeddings(self):
        return self.shared

    # 设置输入嵌入层的方法
    def set_input_embeddings(self, new_embeddings):
        # 更新共享的嵌入层
        self.shared = new_embeddings
        # 更新编码器和解码器中的嵌入层
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    # 使用装饰器解包输入参数的方法
    @unpack_inputs
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_position_ids=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs: Optional[Union[Tuple, TFBaseModelOutput]] = None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
        **kwargs,
        ):
            # 如果用户没有提供隐藏状态的输出，则使用模型配置中的默认设置
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )

            # 如果没有提供编码器输出，则调用编码器进行前向传播
            if encoder_outputs is None:
                encoder_outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    training=training,
                )
            # 如果 return_dict=True 并且用户传递了一个元组作为 encoder_outputs，则将其包装在 TFBaseModelOutput 中
            elif return_dict and not isinstance(encoder_outputs, TFBaseModelOutput):
                encoder_outputs = TFBaseModelOutput(
                    last_hidden_state=encoder_outputs[0],
                    hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                    attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                )
            # 如果 return_dict=False 并且用户传递了 TFBaseModelOutput 作为 encoder_outputs，则将其包装在元组中
            elif not return_dict and not isinstance(encoder_outputs, tuple):
                encoder_outputs = encoder_outputs.to_tuple()

            # 使用解码器进行解码操作
            decoder_outputs = self.decoder(
                decoder_input_ids,
                attention_mask=decoder_attention_mask,
                position_ids=decoder_position_ids,
                encoder_hidden_states=encoder_outputs[0],
                encoder_attention_mask=attention_mask,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                training=training,
            )

            # 如果 return_dict=False，则将解码器输出和编码器输出合并并返回
            if not return_dict:
                return decoder_outputs + encoder_outputs

            # 如果 return_dict=True，则将解码器输出和编码器输出合并为 TFSeq2SeqModelOutput 类型并返回
            return TFSeq2SeqModelOutput(
                last_hidden_state=decoder_outputs.last_hidden_state,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )
    # 定义模型的构建方法，当输入形状为None时表示该方法可接受任意输入形状
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回，避免重复构建
        if self.built:
            return
        # 标记模型为已构建状态
        self.built = True
        
        # 共享/共同权重期望在模型基础命名空间中
        # 在 tf.name_scope 的末尾添加 "/"（但不是开头！）将其放置在根命名空间而不是当前命名空间
        with tf.name_scope(self.shared.load_weight_prefix + "/" + self.shared.name + "/"):
            # 构建共享部分模型
            self.shared.build(None)
        
        # 如果存在编码器部分，进入编码器的命名空间并构建
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        
        # 如果存在解码器部分，进入解码器的命名空间并构建
        if getattr(self, "decoder", None) is not None:
            with tf.name_scope(self.decoder.name):
                self.decoder.build(None)
# 添加模型的文档字符串，说明这是一个输出原始隐藏状态的 BLENDERBOT 模型，没有特定的输出头部分
@add_start_docstrings(
    "The bare BLENDERBOT Model outputting raw hidden-states without any specific head on top.",
    BLENDERBOT_START_DOCSTRING,
)
class TFBlenderbotModel(TFBlenderbotPreTrainedModel):
    def __init__(self, config: BlenderbotConfig, *inputs, **kwargs):
        # 调用父类的初始化方法，传入配置和其他输入参数
        super().__init__(config, *inputs, **kwargs)

        # 创建 TFBlenderbotMainLayer 实例作为模型的主要组成部分
        self.model = TFBlenderbotMainLayer(config, name="model")

    # 返回编码器部分的方法
    def get_encoder(self):
        return self.model.encoder

    # 返回解码器部分的方法
    def get_decoder(self):
        return self.model.decoder

    @classmethod
    # 从预训练模型加载模型的类方法
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        if pretrained_model_name_or_path == "facebook/blenderbot-90M":
            # 如果加载的是 facebook/blenderbot-90M 模型，则发出未来警告
            from ..blenderbot_small import TFBlenderbotSmallModel

            warnings.warn(
                "The checkpoint `facebook/blenderbot-90M` is deprecated. In the future, please use the identical"
                " checkpoint `facebook/small_blenderbot-90M` with"
                " `TFBlenderbotSmallForConditionalGeneration.from_pretrained('facebook/small_blenderbot-90M')`"
                " instead.",
                FutureWarning,
            )
            # 返回 TFBlenderbotSmallModel 的预训练模型
            return TFBlenderbotSmallModel.from_pretrained(pretrained_model_name_or_path)

        # 否则调用父类的 from_pretrained 方法加载模型
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    @unpack_inputs
    @add_start_docstrings_to_model_forward(BLENDERBOT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFSeq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义模型的调用方法，接收多个输入参数
    def call(
        self,
        input_ids: tf.Tensor | None = None,
        attention_mask: tf.Tensor | None = None,
        decoder_input_ids: tf.Tensor | None = None,
        decoder_attention_mask: tf.Tensor | None = None,
        decoder_position_ids: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        decoder_head_mask: tf.Tensor | None = None,
        cross_attn_head_mask: tf.Tensor | None = None,
        encoder_outputs: Optional[Union[Tuple, TFBaseModelOutput]] = None,
        past_key_values: List[tf.Tensor] | None = None,
        inputs_embeds: tf.Tensor | None = None,
        decoder_inputs_embeds: tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
        **kwargs,
    ) -> Union[Tuple[tf.Tensor], TFSeq2SeqModelOutput]:
        # 调用模型的方法，传入以下参数，并接收返回的输出
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

        # 返回模型的输出
        return outputs

    # 从 transformers.models.bart.modeling_tf_bart.TFBartModel.serving_output 复制而来
    def serving_output(self, output):
        # 根据配置判断是否需要处理过去键值（past_key_values）
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        # 根据配置判断是否需要输出解码器隐藏状态（decoder_hidden_states）
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        # 根据配置判断是否需要输出解码器注意力权重（decoder_attentions）
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        # 根据配置判断是否需要输出交叉注意力权重（cross_attentions）
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        # 根据配置判断是否需要输出编码器隐藏状态（encoder_hidden_states）
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        # 根据配置判断是否需要输出编码器注意力权重（encoder_attentions）
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None

        # 构建并返回 TFSeq2SeqModelOutput 对象
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

    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果模型存在，使用模型的名称构建
        if getattr(self, "model", None) is not None:
            with tf.name_scope(self.model.name):
                self.model.build(None)
# Copied from transformers.models.bart.modeling_tf_bart.BiasLayer
class BiasLayer(keras.layers.Layer):
    """
    Bias as a layer. It is used for serialization purposes: `keras.Model.save_weights` stores on a per-layer basis,
    so all weights have to be registered in a layer.
    """

    def __init__(self, shape, initializer, trainable, name, **kwargs):
        super().__init__(name=name, **kwargs)
        # Note: the name of this variable will NOT be scoped when serialized, i.e. it will not be in the format of
        # "outer_layer/inner_layer/.../name:0". Instead, it will be "name:0". For further details, see:
        # https://github.com/huggingface/transformers/pull/18833#issuecomment-1233090214
        # 添加偏置权重作为层的一部分，以便在模型保存和加载时能够正确处理
        self.bias = self.add_weight(name=name, shape=shape, initializer=initializer, trainable=trainable)

    def call(self, x):
        # 在输入张量 x 上添加偏置向量 self.bias
        return x + self.bias


@add_start_docstrings(
    "The BLENDERBOT Model with a language modeling head. Can be used for summarization.",
    BLENDERBOT_START_DOCSTRING,
)
class TFBlenderbotForConditionalGeneration(TFBlenderbotPreTrainedModel, TFCausalLanguageModelingLoss):
    _keys_to_ignore_on_load_unexpected = [
        r"model.encoder.embed_tokens.weight",
        r"model.decoder.embed_tokens.weight",
    ]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 创建 TFBlenderbotMainLayer 实例，并命名为 "model"，作为模型的核心组件
        self.model = TFBlenderbotMainLayer(config, name="model")
        # 根据配置中的参数设置是否使用缓存
        self.use_cache = config.use_cache
        # 创建 BiasLayer 实例作为模型输出的偏置向量，名为 "final_logits_bias"
        # 该偏置向量用于调整模型最终输出的 logits，设置为不可训练以保持一致性
        self.bias_layer = BiasLayer(
            name="final_logits_bias", shape=[1, config.vocab_size], initializer="zeros", trainable=False
        )

    def get_decoder(self):
        # 获取模型的解码器（decoder）部分
        return self.model.decoder

    def get_encoder(self):
        # 获取模型的编码器（encoder）部分
        return self.model.encoder

    def get_output_embeddings(self):
        # 获取模型的输出嵌入层
        return self.get_input_embeddings()

    def set_output_embeddings(self, value):
        # 设置模型的输出嵌入层
        self.set_input_embeddings(value)

    def get_bias(self):
        # 返回模型当前使用的偏置向量，以字典形式返回，键为 "final_logits_bias"
        return {"final_logits_bias": self.bias_layer.bias}

    def set_bias(self, value):
        # 用给定的偏置值替换当前模型中的偏置层，确保正确的序列化和反序列化过程
        vocab_size = value["final_logits_bias"].shape[-1]
        self.bias_layer = BiasLayer(
            name="final_logits_bias", shape=[1, vocab_size], initializer="zeros", trainable=False
        )
        self.bias_layer.bias.assign(value["final_logits_bias"])

    @classmethod
    # 根据预训练模型名称或路径加载模型，并传递给模型的参数和关键字参数
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        # 如果预训练模型名称或路径是特定的字符串
        if pretrained_model_name_or_path == "facebook/blenderbot-90M":
            # 从模块中导入 TFBlenderbotSmallForConditionalGeneration 类
            from ..blenderbot_small import TFBlenderbotSmallForConditionalGeneration

            # 发出警告，说明特定检查点已弃用，并建议新的检查点名称和使用方式
            warnings.warn(
                "The checkpoint `facebook/blenderbot-90M` is deprecated. In the future, please use the identical"
                " checkpoint `facebook/small_blenderbot-90M` with"
                " `TFBlenderbotSmallForConditionalGeneration.from_pretrained('facebook/small_blenderbot-90M')`"
                " instead.",
                FutureWarning,
            )
            # 返回从预训练模型加载的 TFBlenderbotSmallForConditionalGeneration 实例
            return TFBlenderbotSmallForConditionalGeneration.from_pretrained(pretrained_model_name_or_path)

        # 调用父类的 from_pretrained 方法，传递预训练模型名称或路径以及其他参数和关键字参数
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    # 将装饰器应用于 call 方法，以添加模型输入和输出的文档字符串
    @unpack_inputs
    @add_start_docstrings_to_model_forward(BLENDERBOT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(BLENDERBOT_GENERATION_EXAMPLE)
    def call(
        self,
        # 模型的输入张量，可以为 None
        input_ids: tf.Tensor | None = None,
        # 注意力遮罩张量，可以为 None
        attention_mask: tf.Tensor | None = None,
        # 解码器输入的 ID 张量，可以为 None
        decoder_input_ids: tf.Tensor | None = None,
        # 解码器的注意力遮罩张量，可以为 None
        decoder_attention_mask: tf.Tensor | None = None,
        # 解码器的位置 ID 张量，可以为 None
        decoder_position_ids: tf.Tensor | None = None,
        # 头部遮罩张量，可以为 None
        head_mask: tf.Tensor | None = None,
        # 解码器头部遮罩张量，可以为 None
        decoder_head_mask: tf.Tensor | None = None,
        # 跨注意力头部遮罩张量，可以为 None
        cross_attn_head_mask: tf.Tensor | None = None,
        # 编码器输出，可以为元组或 TFBaseModelOutput 类型
        encoder_outputs: Optional[Union[Tuple, TFBaseModelOutput]] = None,
        # 过去键值列表，可以为 None
        past_key_values: List[tf.Tensor] | None = None,
        # 输入嵌入张量，可以为 None
        inputs_embeds: tf.Tensor | None = None,
        # 解码器输入嵌入张量，可以为 None
        decoder_inputs_embeds: tf.Tensor | None = None,
        # 是否使用缓存，可以为 None
        use_cache: Optional[bool] = None,
        # 是否输出注意力权重，可以为 None
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，可以为 None
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典类型结果，可以为 None
        return_dict: Optional[bool] = None,
        # 标签张量，可以为 None
        labels: tf.Tensor | None = None,
        # 是否处于训练模式，默认为 False
        training: Optional[bool] = False,
    ) -> Union[Tuple[tf.Tensor], TFSeq2SeqLMOutput]:
        r"""
        labels (`tf.tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        """
        # 如果给定了标签，则处理标签，将所有标记为 pad_token_id 的标签改为 -100，其余保持不变
        if labels is not None:
            labels = tf.where(
                labels == self.config.pad_token_id,
                tf.cast(tf.fill(shape_list(labels), -100), labels.dtype),
                labels,
            )
            # 如果未提供解码器的输入，根据标签生成解码器的输入
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        # 使用模型进行前向传播
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
        
        # 计算语言模型的 logits
        lm_logits = tf.matmul(outputs[0], self.model.shared.weights, transpose_b=True)
        lm_logits = self.bias_layer(lm_logits)
        
        # 计算掩码语言模型的损失，如果没有标签则损失为 None
        masked_lm_loss = None if labels is None else self.hf_compute_loss(labels, lm_logits)

        # 如果 return_dict 为 False，则按照元组形式返回输出
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        
        # 如果 return_dict 为 True，则按照 TFSeq2SeqLMOutput 类的实例形式返回输出
        return TFSeq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,  # 索引 1 的 d outputs
            decoder_hidden_states=outputs.decoder_hidden_states,  # 索引 2 的 d outputs
            decoder_attentions=outputs.decoder_attentions,  # 索引 3 的 d outputs
            cross_attentions=outputs.cross_attentions,  # 索引 4 的 d outputs
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,  # 索引 0 的 encoder outputs
            encoder_hidden_states=outputs.encoder_hidden_states,  # 索引 1 的 e outputs
            encoder_attentions=outputs.encoder_attentions,  # 索引 2 的 e outputs
        )

    # 从 transformers.models.bart.modeling_tf_bart.TFBartForConditionalGeneration.serving_output 复制而来
    # 定义一个方法用于处理模型的输出，根据配置选择性地包含不同的输出信息
    def serving_output(self, output):
        # 如果配置要求使用缓存，则从输出中获取过去的键-值对
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        # 如果配置要求输出隐藏状态，则将解码器的隐藏状态转换为张量
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置要求输出注意力权重，则将解码器的注意力权重转换为张量
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        # 如果配置要求输出交叉注意力权重，则将交叉注意力权重转换为张量
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        # 如果配置要求输出隐藏状态，则将编码器的隐藏状态转换为张量
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置要求输出注意力权重，则将编码器的注意力权重转换为张量
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None

        # 返回一个 TFSeq2SeqLMOutput 对象，包含处理后的输出信息
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

    # 从 transformers 库中的 TFBartForConditionalGeneration 类的方法 prepare_inputs_for_generation 复制而来
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
        # 如果存在过去的键-值对，根据此情况截取 decoder_input_ids
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        # 如果有 decoder_attention_mask，使用 XLA 编译执行
        if decoder_attention_mask is not None:  # xla
            # 计算累积的位置 IDs，并取最后一个位置
            decoder_position_ids = tf.math.cumsum(decoder_attention_mask, axis=-1, exclusive=True)[:, -1:]
        # 如果没有 XLA + 存在过去的键-值对
        elif past_key_values is not None:  # no xla + past_key_values
            # 获取过去键-值对的第一个元素的第一个维度的长度作为位置 IDs
            decoder_position_ids = past_key_values[0][0].shape[2]
        else:  # 没有 XLA + 没有过去的键-值对
            # 创建 decoder_input_ids 的位置 IDs
            decoder_position_ids = tf.range(decoder_input_ids.shape[1])

        # 返回一个包含准备好用于生成的输入参数的字典
        return {
            "input_ids": None,  # encoder_outputs 已定义，不需要 input_ids
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_position_ids": decoder_position_ids,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # 更改此项以避免缓存（可能用于调试）
        }
    # 定义一个方法用于构建网络层，支持接收输入形状参数，如果已经构建过则直接返回
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回，避免重复构建
        if self.built:
            return
        # 标记为已构建
        self.built = True
        
        # 检查是否存在模型属性，如果存在，则使用 TensorFlow 的名称空间来构建模型
        if getattr(self, "model", None) is not None:
            with tf.name_scope(self.model.name):
                # 调用模型的build方法来构建模型，传入None表示不指定输入形状
                self.model.build(None)
        
        # 检查是否存在偏置层属性，如果存在，则使用 TensorFlow 的名称空间来构建偏置层
        if getattr(self, "bias_layer", None) is not None:
            with tf.name_scope(self.bias_layer.name):
                # 调用偏置层的build方法来构建偏置层，传入None表示不指定输入形状
                self.bias_layer.build(None)
```