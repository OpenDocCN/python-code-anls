# `.\models\blenderbot_small\modeling_tf_blenderbot_small.py`

```py
# coding=utf-8
# 版权所有 2021 年 Facebook, Inc 和 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）许可；
# 除非符合许可证的规定，否则您不能使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件是基于“按原样”分发的，
# 没有任何明示或暗示的担保或条件。
# 有关特定语言的详情，请参阅许可证。
""" TF 2.0 BlenderbotSmall 模型。"""


from __future__ import annotations

import random  # 导入随机数模块
from typing import List, Optional, Tuple, Union  # 导入类型提示模块

import numpy as np  # 导入 NumPy 库
import tensorflow as tf  # 导入 TensorFlow 库

from ...activations_tf import get_tf_activation  # 从本地导入 TensorFlow 激活函数
from ...modeling_tf_outputs import (  # 从本地导入 TensorFlow 模型输出类
    TFBaseModelOutput,
    TFBaseModelOutputWithPastAndCrossAttentions,
    TFSeq2SeqLMOutput,
    TFSeq2SeqModelOutput,
)

# 公共 API
from ...modeling_tf_utils import (  # 从本地导入 TensorFlow 模型工具类和函数
    TFCausalLanguageModelingLoss,
    TFPreTrainedModel,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax  # 从本地导入 TensorFlow 工具函数
from ...utils import (  # 从本地导入通用工具函数
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_blenderbot_small import BlenderbotSmallConfig  # 从本地导入 BlenderbotSmall 配置类


logger = logging.get_logger(__name__)  # 获取 logger 对象


_CHECKPOINT_FOR_DOC = "facebook/blenderbot_small-90M"  # 预训练模型检查点用于文档说明
_CONFIG_FOR_DOC = "BlenderbotSmallConfig"  # BlenderbotSmall 配置用于文档说明


LARGE_NEGATIVE = -1e8  # 设置一个大负数常量，用于某些计算中


# 从 transformers.models.bart.modeling_tf_bart.shift_tokens_right 复制而来
def shift_tokens_right(input_ids: tf.Tensor, pad_token_id: int, decoder_start_token_id: int):
    pad_token_id = tf.cast(pad_token_id, input_ids.dtype)  # 将 pad_token_id 转换为 input_ids 的数据类型
    decoder_start_token_id = tf.cast(decoder_start_token_id, input_ids.dtype)  # 将 decoder_start_token_id 转换为 input_ids 的数据类型
    start_tokens = tf.fill(
        (shape_list(input_ids)[0], 1),  # 填充形状为 (input_ids 的行数, 1) 的张量
        tf.convert_to_tensor(decoder_start_token_id, input_ids.dtype)  # 使用 decoder_start_token_id 填充
    )
    shifted_input_ids = tf.concat([start_tokens, input_ids[:, :-1]], -1)  # 将起始标记与 input_ids 右移一位进行连接
    # 将 labels 中可能的 -100 值替换为 pad_token_id
    shifted_input_ids = tf.where(
        shifted_input_ids == -100,
        tf.fill(shape_list(shifted_input_ids), tf.convert_to_tensor(pad_token_id, input_ids.dtype)),
        shifted_input_ids,
    )

    # "验证 labels 中仅包含正值和 -100"
    assert_gte0 = tf.debugging.assert_greater_equal(shifted_input_ids, tf.constant(0, dtype=input_ids.dtype))

    # 确保通过包装结果在一个空操作中调用断言操作
    with tf.control_dependencies([assert_gte0]):
        shifted_input_ids = tf.identity(shifted_input_ids)

    return shifted_input_ids  # 返回右移后的 input_ids


# 从 transformers.models.bart.modeling_tf_bart._make_causal_mask 复制而来
# 创建一个用于双向自注意力的因果（causal）掩码。
def _make_causal_mask(input_ids_shape: tf.TensorShape, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    # 获取批次大小
    bsz = input_ids_shape[0]
    # 获取目标序列长度
    tgt_len = input_ids_shape[1]
    # 创建初始掩码，所有元素为负无穷大（用于softmax后概率接近0）
    mask = tf.ones((tgt_len, tgt_len)) * LARGE_NEGATIVE
    # 创建掩码条件，形状为 [tgt_len]
    mask_cond = tf.range(shape_list(mask)[-1])

    # 将对角线以下的元素设置为0，保留对角线及以上的元素
    mask = tf.where(mask_cond < tf.reshape(mask_cond + 1, (shape_list(mask)[-1], 1)), 0.0, mask)

    # 如果有历史键值长度，则在掩码左侧添加0的列，使其与历史键值对齐
    if past_key_values_length > 0:
        mask = tf.concat([tf.zeros((tgt_len, past_key_values_length)), mask], axis=-1)

    # 在批次维度和其他维度上复制掩码，以匹配输入的形状
    return tf.tile(mask[None, None, :, :], (bsz, 1, 1, 1))


# 从 transformers.models.bart.modeling_tf_bart._expand_mask 复制过来
def _expand_mask(mask: tf.Tensor, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    # 获取源序列长度
    src_len = shape_list(mask)[1]
    # 如果未提供目标长度，则使用源序列长度
    tgt_len = tgt_len if tgt_len is not None else src_len
    # 创建常数张量1.0
    one_cst = tf.constant(1.0)
    # 将掩码转换为常数张量类型
    mask = tf.cast(mask, dtype=one_cst.dtype)
    # 在维度上复制掩码，以匹配目标长度
    expanded_mask = tf.tile(mask[:, None, None, :], (1, 1, tgt_len, 1))

    # 返回扩展后的掩码，其中将1减去掩码值乘以一个大负数（LARGE_NEGATIVE）
    return (one_cst - expanded_mask) * LARGE_NEGATIVE


# 从 transformers.models.blenderbot.modeling_tf_blenderbot.TFBlenderbotLearnedPositionalEmbedding 复制过来，将Blenderbot改为BlenderbotSmall
class TFBlenderbotSmallLearnedPositionalEmbedding(keras.layers.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, **kwargs):
        super().__init__(num_embeddings, embedding_dim, **kwargs)

    def call(
        self, input_shape: tf.TensorShape, past_key_values_length: int = 0, position_ids: tf.Tensor | None = None
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        # 如果未提供位置ID，则创建一个从0开始递增的序列，与历史键值长度相加
        if position_ids is None:
            seq_len = input_shape[1]
            position_ids = tf.range(seq_len, delta=1, name="range")
            position_ids += past_key_values_length

        # 调用父类的call方法，传递位置ID并转换为int32类型
        return super().call(tf.cast(position_ids, dtype=tf.int32))


# 从 transformers.models.bart.modeling_tf_bart.TFBartAttention 复制过来，将Bart改为BlenderbotSmall
class TFBlenderbotSmallAttention(keras.layers.Layer):
    """Multi-headed attention from "Attention Is All You Need"""

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
        self.head_dim = embed_dim // num_heads  # 计算每个注意力头的维度
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"  # 抛出异常，如果 embed_dim 不能被 num_heads 整除
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5  # 缩放因子的计算
        self.is_decoder = is_decoder

        self.k_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="k_proj")  # 创建用于 K 矩阵投影的 Dense 层
        self.q_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="q_proj")  # 创建用于 Q 矩阵投影的 Dense 层
        self.v_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="v_proj")  # 创建用于 V 矩阵投影的 Dense 层
        self.out_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="out_proj")  # 创建用于输出矩阵投影的 Dense 层

    def _shape(self, tensor: tf.Tensor, seq_len: int, bsz: int):
        return tf.transpose(tf.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))  # 重新形状化张量，以便多头注意力操作

    def call(
        self,
        hidden_states: tf.Tensor,
        key_value_states: tf.Tensor | None = None,
        past_key_value: Tuple[Tuple[tf.Tensor]] | None = None,
        attention_mask: tf.Tensor | None = None,
        layer_head_mask: tf.Tensor | None = None,
        training: Optional[bool] = False,
    ):
        # 模型的前向传播函数
        # hidden_states: 输入的隐藏状态张量
        # key_value_states: 可选的键值状态张量
        # past_key_value: 可选的过去键值张量
        # attention_mask: 可选的注意力掩码张量
        # layer_head_mask: 可选的层头掩码张量
        # training: 可选的训练模式标志

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "k_proj", None) is not None:
            with tf.name_scope(self.k_proj.name):
                self.k_proj.build([None, None, self.embed_dim])  # 构建 K 矩阵投影层
        if getattr(self, "q_proj", None) is not None:
            with tf.name_scope(self.q_proj.name):
                self.q_proj.build([None, None, self.embed_dim])  # 构建 Q 矩阵投影层
        if getattr(self, "v_proj", None) is not None:
            with tf.name_scope(self.v_proj.name):
                self.v_proj.build([None, None, self.embed_dim])  # 构建 V 矩阵投影层
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build([None, None, self.embed_dim])  # 构建输出矩阵投影层
# Copied from transformers.models.bart.modeling_tf_bart.TFBartEncoderLayer with Bart->BlenderbotSmall

class TFBlenderbotSmallEncoderLayer(keras.layers.Layer):
    def __init__(self, config: BlenderbotSmallConfig, **kwargs):
        super().__init__(**kwargs)
        # 初始化层的参数
        self.embed_dim = config.d_model  # 获取模型的嵌入维度
        # 创建自注意力层对象，用于处理自注意力机制
        self.self_attn = TFBlenderbotSmallAttention(
            self.embed_dim, config.encoder_attention_heads, dropout=config.attention_dropout, name="self_attn"
        )
        # 创建自注意力层后的层归一化层
        self.self_attn_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        # 随机失活层，用于在训练期间随机失活部分神经元
        self.dropout = keras.layers.Dropout(config.dropout)
        # 激活函数，根据配置选择合适的激活函数
        self.activation_fn = get_tf_activation(config.activation_function)
        # 激活函数后的激活层随机失活层
        self.activation_dropout = keras.layers.Dropout(config.activation_dropout)
        # 第一个全连接层，处理前馈神经网络的第一层变换
        self.fc1 = keras.layers.Dense(config.encoder_ffn_dim, name="fc1")
        # 第二个全连接层，处理前馈神经网络的第二层变换
        self.fc2 = keras.layers.Dense(self.embed_dim, name="fc2")
        # 最终层归一化层，处理前馈神经网络的输出
        self.final_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")
        # 保存配置信息
        self.config = config

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: np.ndarray | tf.Tensor | None,
        layer_head_mask: tf.Tensor | None,
        training: Optional[bool] = False,
    ) -> tf.Tensor:
        """
        Args:
            hidden_states (`tf.Tensor`): 输入到层的张量，形状为 `(batch, seq_len, embed_dim)`
            attention_mask (`tf.Tensor`): 注意力掩码，形状为 `(batch, 1, tgt_len, src_len)`，
                其中填充元素由非常大的负值表示。
            layer_head_mask (`tf.Tensor`): 给定层的注意力头部掩码，形状为 `(encoder_attention_heads,)`
        """
        # 保留输入的残差连接
        residual = hidden_states
        # 使用自注意力层处理输入张量
        hidden_states, self_attn_weights, _ = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, layer_head_mask=layer_head_mask
        )

        # 断言确保自注意力层不改变输入张量的形状
        tf.debugging.assert_equal(
            shape_list(hidden_states),
            shape_list(residual),
            message=f"Self attn modified the shape of query {shape_list(residual)} to {shape_list(hidden_states)}",
        )

        # 应用随机失活到处理后的张量
        hidden_states = self.dropout(hidden_states, training=training)
        # 残差连接与处理后的张量相加
        hidden_states = residual + hidden_states
        # 应用层归一化到残差连接后的张量
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 保留新的残差连接
        residual = hidden_states
        # 使用激活函数处理第一个全连接层
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 应用激活函数后的随机失活
        hidden_states = self.activation_dropout(hidden_states, training=training)
        # 使用第二个全连接层处理张量
        hidden_states = self.fc2(hidden_states)
        # 应用随机失活到处理后的张量
        hidden_states = self.dropout(hidden_states, training=training)
        # 残差连接与处理后的张量相加
        hidden_states = residual + hidden_states
        # 应用层归一化到残差连接后的张量
        hidden_states = self.final_layer_norm(hidden_states)

        # 返回处理后的张量以及自注意力权重
        return hidden_states, self_attn_weights
    # 构建方法用于建立模型的层结构，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        
        # 如果存在 self_attn 属性，则构建 self attention 层
        if getattr(self, "self_attn", None) is not None:
            with tf.name_scope(self.self_attn.name):
                # 使用 self attention 层的名称作为命名空间，构建该层
                self.self_attn.build(None)
        
        # 如果存在 self_attn_layer_norm 属性，则构建 layer normalization 层
        if getattr(self, "self_attn_layer_norm", None) is not None:
            with tf.name_scope(self.self_attn_layer_norm.name):
                # 使用 layer normalization 层的名称作为命名空间，构建该层
                self.self_attn_layer_norm.build([None, None, self.embed_dim])
        
        # 如果存在 fc1 属性，则构建第一个全连接层
        if getattr(self, "fc1", None) is not None:
            with tf.name_scope(self.fc1.name):
                # 使用第一个全连接层的名称作为命名空间，构建该层
                self.fc1.build([None, None, self.embed_dim])
        
        # 如果存在 fc2 属性，则构建第二个全连接层
        if getattr(self, "fc2", None) is not None:
            with tf.name_scope(self.fc2.name):
                # 使用第二个全连接层的名称作为命名空间，构建该层
                self.fc2.build([None, None, self.config.encoder_ffn_dim])
        
        # 如果存在 final_layer_norm 属性，则构建最终的 layer normalization 层
        if getattr(self, "final_layer_norm", None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                # 使用最终 layer normalization 层的名称作为命名空间，构建该层
                self.final_layer_norm.build([None, None, self.embed_dim])
# 从transformers.models.bart.modeling_tf_bart.TFBartDecoderLayer复制而来，将Bart改为BlenderbotSmall
class TFBlenderbotSmallDecoderLayer(keras.layers.Layer):
    def __init__(self, config: BlenderbotSmallConfig, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = config.d_model  # 设置嵌入维度为配置中的模型维度
        self.self_attn = TFBlenderbotSmallAttention(
            embed_dim=self.embed_dim,  # 自注意力层，使用设定的嵌入维度
            num_heads=config.decoder_attention_heads,  # 使用配置中的注意力头数
            dropout=config.attention_dropout,  # 使用配置中的注意力机制dropout率
            name="self_attn",  # 层名称为self_attn
            is_decoder=True,  # 标记为解码器自注意力层
        )
        self.dropout = keras.layers.Dropout(config.dropout)  # Dropout层，使用配置中的dropout率
        self.activation_fn = get_tf_activation(config.activation_function)  # 获取激活函数
        self.activation_dropout = keras.layers.Dropout(config.activation_dropout)  # 激活函数的dropout层

        self.self_attn_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        # 自注意力层后的LayerNormalization层

        self.encoder_attn = TFBlenderbotSmallAttention(
            self.embed_dim,  # 编码器注意力层，使用相同的嵌入维度
            config.decoder_attention_heads,  # 使用配置中的注意力头数
            dropout=config.attention_dropout,  # 使用配置中的注意力机制dropout率
            name="encoder_attn",  # 层名称为encoder_attn
            is_decoder=True,  # 标记为解码器编码器注意力层
        )
        self.encoder_attn_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="encoder_attn_layer_norm")
        # 编码器注意力层后的LayerNormalization层

        self.fc1 = keras.layers.Dense(config.decoder_ffn_dim, name="fc1")  # 第一个全连接层，使用配置中的FFN维度
        self.fc2 = keras.layers.Dense(self.embed_dim, name="fc2")  # 第二个全连接层，输出维度与嵌入维度相同
        self.final_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")
        # 最终的LayerNormalization层

        self.config = config  # 保存配置信息

    def call(
        self,
        hidden_states: tf.Tensor,  # 输入的隐藏状态张量
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 注意力掩码张量或数组，可选
        encoder_hidden_states: np.ndarray | tf.Tensor | None = None,  # 编码器隐藏状态张量或数组，可选
        encoder_attention_mask: np.ndarray | tf.Tensor | None = None,  # 编码器注意力掩码张量或数组，可选
        layer_head_mask: tf.Tensor | None = None,  # 层级头掩码张量，可选
        cross_attn_layer_head_mask: tf.Tensor | None = None,  # 跨注意力头掩码张量，可选
        past_key_value: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,  # 过去键值元组，可选
        training: Optional[bool] = False,  # 训练标志位，可选
    # 构建函数，用于构建模型的层结构，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 设置标志位，表示模型已经构建
        self.built = True
        
        # 如果存在 self_attn 属性，则构建 self attention 层
        if getattr(self, "self_attn", None) is not None:
            with tf.name_scope(self.self_attn.name):
                self.self_attn.build(None)
        
        # 如果存在 self_attn_layer_norm 属性，则构建 self attention 层的层归一化
        if getattr(self, "self_attn_layer_norm", None) is not None:
            with tf.name_scope(self.self_attn_layer_norm.name):
                self.self_attn_layer_norm.build([None, None, self.embed_dim])
        
        # 如果存在 encoder_attn 属性，则构建 encoder-decoder attention 层
        if getattr(self, "encoder_attn", None) is not None:
            with tf.name_scope(self.encoder_attn.name):
                self.encoder_attn.build(None)
        
        # 如果存在 encoder_attn_layer_norm 属性，则构建 encoder-decoder attention 层的层归一化
        if getattr(self, "encoder_attn_layer_norm", None) is not None:
            with tf.name_scope(self.encoder_attn_layer_norm.name):
                self.encoder_attn_layer_norm.build([None, None, self.embed_dim])
        
        # 如果存在 fc1 属性，则构建第一个全连接层
        if getattr(self, "fc1", None) is not None:
            with tf.name_scope(self.fc1.name):
                self.fc1.build([None, None, self.embed_dim])
        
        # 如果存在 fc2 属性，则构建第二个全连接层
        if getattr(self, "fc2", None) is not None:
            with tf.name_scope(self.fc2.name):
                self.fc2.build([None, None, self.config.decoder_ffn_dim])
        
        # 如果存在 final_layer_norm 属性，则构建最终的层归一化层
        if getattr(self, "final_layer_norm", None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.embed_dim])
# TFBlenderbotSmallPreTrainedModel 类的定义，继承自 TFPreTrainedModel。
class TFBlenderbotSmallPreTrainedModel(TFPreTrainedModel):
    # 配置类，指定为 BlenderbotSmallConfig
    config_class = BlenderbotSmallConfig
    # 模型基本前缀设置为 "model"
    base_model_prefix = "model"
    >>> UTTERANCE = "My friends are cool but they eat too many carbs."
    >>> print("Human: ", UTTERANCE)
    打印出人类的发言
    
    >>> inputs = tokenizer([UTTERANCE], return_tensors="tf")
    使用分词器对发言进行处理，返回模型输入的张量表示
    
    >>> reply_ids = model.generate(**inputs)
    使用模型生成回复
    
    >>> print("Bot: ", tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0])
    打印出生成的机器人回复，跳过特殊标记后的解码结果
    
    >>> REPLY = "I'm not sure"
    >>> print("Human: ", REPLY)
    打印出人类的回复
    
    >>> NEXT_UTTERANCE = (
    ...     "My friends are cool but they eat too many carbs.</s> "
    ...     "<s>what kind of carbs do they eat? i don't know much about carbs.</s> "
    ...     "<s>I'm not sure."
    ... )
    设置下一轮对话的文本
    
    >>> inputs = tokenizer([NEXT_UTTERANCE], return_tensors="tf")
    使用分词器处理下一轮对话文本，返回模型输入的张量表示
    
    >>> inputs.pop("token_type_ids")
    移除张量表示中的token_type_ids（标记类型标识符）
    
    >>> next_reply_ids = model.generate(**inputs)
    使用模型生成下一轮对话的回复
    
    >>> print("Bot: ", tokenizer.batch_decode(next_reply_ids, skip_special_tokens=True)[0])
    打印出生成的机器人回复，跳过特殊标记后的解码结果
"""

BLENDERBOT_SMALL_INPUTS_DOCSTRING = r"""
"""


@keras_serializable
class TFBlenderbotSmallEncoder(keras.layers.Layer):
    config_class = BlenderbotSmallConfig
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`TFBlenderbotSmallEncoderLayer`].

    Args:
        config: BlenderbotSmallConfig
    """

    def __init__(self, config: BlenderbotSmallConfig, embed_tokens: Optional[keras.layers.Embedding] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.dropout = keras.layers.Dropout(config.dropout)  # 初始化dropout层，根据配置设置dropout率
        self.layerdrop = config.encoder_layerdrop  # 获取配置中的layerdrop参数，用于层级别的dropout
        self.padding_idx = config.pad_token_id  # 获取配置中的pad_token_id，用于填充的特殊token
        self.max_source_positions = config.max_position_embeddings  # 获取配置中的max_position_embeddings，最大位置嵌入长度
        self.embed_scale = tf.math.sqrt(float(config.d_model)) if config.scale_embedding else 1.0  # 根据配置设置嵌入的缩放因子

        self.embed_tokens = embed_tokens  # 初始化嵌入token
        self.embed_positions = TFBlenderbotSmallLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            name="embed_positions",
        )  # 初始化位置嵌入
        self.layers = [TFBlenderbotSmallEncoderLayer(config, name=f"layers.{i}") for i in range(config.encoder_layers)]  # 创建多个编码层
        self.layernorm_embedding = keras.layers.LayerNormalization(epsilon=1e-5, name="layernorm_embedding")  # 初始化嵌入层归一化
        self.embed_dim = config.d_model  # 获取配置中的嵌入维度

    def get_embed_tokens(self):
        return self.embed_tokens  # 返回嵌入token

    def set_embed_tokens(self, embed_tokens):
        self.embed_tokens = embed_tokens  # 设置嵌入token

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
    ):
        """
        实现Layer的call方法，用于前向传播

        Args:
            input_ids: 输入的token ids
            inputs_embeds: 嵌入表示
            attention_mask: 注意力掩码
            head_mask: 多头注意力的掩码
            output_attentions: 是否输出注意力权重
            output_hidden_states: 是否输出隐藏状态
            return_dict: 是否返回字典格式结果
            training: 是否为训练模式

        Returns:
            根据配置返回相应的结果
        """
        # 省略具体实现细节，实现模型的前向传播逻辑

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "embed_positions", None) is not None:
            with tf.name_scope(self.embed_positions.name):
                self.embed_positions.build(None)  # 构建位置嵌入
        if getattr(self, "layernorm_embedding", None) is not None:
            with tf.name_scope(self.layernorm_embedding.name):
                self.layernorm_embedding.build([None, None, self.embed_dim])  # 构建嵌入层的归一化
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)  # 构建每个编码层


@keras_serializable
class TFBlenderbotSmallDecoder(keras.layers.Layer):
    config_class = BlenderbotSmallConfig
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`TFBlenderbotSmallDecoderLayer`]

    Args:
        config: BlenderbotSmallConfig
        embed_tokens: output embedding
    """
    # 使用给定的配置和嵌入标记初始化对象，继承父类的初始化方法
    def __init__(self, config: BlenderbotSmallConfig, embed_tokens: Optional[keras.layers.Embedding] = None, **kwargs):
        super().__init__(**kwargs)
        # 将配置保存在对象中
        self.config = config
        # 设置填充索引为配置中的填充标记 ID
        self.padding_idx = config.pad_token_id
        # 设置嵌入标记为给定的嵌入标记
        self.embed_tokens = embed_tokens
        # 设置层的丢弃率为配置中的解码器层丢弃率
        self.layerdrop = config.decoder_layerdrop
        # 使用给定的最大位置嵌入数量和模型维度创建位置嵌入对象
        self.embed_positions = TFBlenderbotSmallLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            name="embed_positions",
        )
        # 如果配置中设置了缩放嵌入，则计算并设置嵌入的缩放因子为模型维度的平方根，否则设为1.0
        self.embed_scale = tf.math.sqrt(float(config.d_model)) if config.scale_embedding else 1.0
        # 创建解码器层的列表，每一层使用给定的配置创建一个解码器层对象
        self.layers = [TFBlenderbotSmallDecoderLayer(config, name=f"layers.{i}") for i in range(config.decoder_layers)]
        # 创建用于嵌入层归一化的层归一化对象
        self.layernorm_embedding = keras.layers.LayerNormalization(epsilon=1e-5, name="layernorm_embedding")

        # 创建一个丢弃层，使用配置中的丢弃率
        self.dropout = keras.layers.Dropout(config.dropout)

    # 获取当前嵌入标记对象的方法
    def get_embed_tokens(self):
        return self.embed_tokens

    # 设置新的嵌入标记对象的方法
    def set_embed_tokens(self, embed_tokens):
        self.embed_tokens = embed_tokens

    # 装饰器，解包输入参数，用于处理call方法的输入参数
    @unpack_inputs
    # 模型的调用方法，处理输入并返回模型的输出
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
        # 方法体的具体实现将在下文注释中描述


注释：
@keras_serializable
class TFBlenderbotSmallMainLayer(keras.layers.Layer):
    # 设定配置类为 BlenderbotSmallConfig
    config_class = BlenderbotSmallConfig

    def __init__(self, config: BlenderbotSmallConfig, **kwargs):
        super().__init__(**kwargs)

        # 初始化函数，接收 BlenderbotSmallConfig 对象作为配置参数
        self.config = config
        
        # 创建一个共享的嵌入层，用于共享模型的词汇表和嵌入大小
        self.shared = keras.layers.Embedding(
            input_dim=config.vocab_size,
            output_dim=config.d_model,
            embeddings_initializer=keras.initializers.TruncatedNormal(stddev=self.config.init_std),
            name="model.shared",
        )
        # 添加一个额外的属性，用于指定层的预期名称范围（用于加载/存储权重）
        self.shared.load_weight_prefix = "model.shared"

        # 创建编码器和解码器层，使用 TFBlenderbotSmallEncoder 和 TFBlenderbotSmallDecoder 类
        self.encoder = TFBlenderbotSmallEncoder(config, self.shared, name="encoder")
        self.decoder = TFBlenderbotSmallDecoder(config, self.shared, name="decoder")

    # 返回共享的嵌入层对象
    def get_input_embeddings(self):
        return self.shared

    # 设置新的输入嵌入层对象，并更新编码器和解码器中的 embed_tokens 属性
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    # 使用装饰器 unpack_inputs，处理输入参数并调用模型
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
        # 如果输出隐藏状态参数为 None，则使用模型配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # 如果 encoder_outputs 为 None，则调用 encoder 进行编码
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
        # 如果 return_dict=True 且 encoder_outputs 是元组，则将其包装在 TFBaseModelOutput 中
        elif return_dict and not isinstance(encoder_outputs, TFBaseModelOutput):
            encoder_outputs = TFBaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        # 如果 return_dict=False 且 encoder_outputs 是 TFBaseModelOutput，则将其转换为元组
        elif not return_dict and not isinstance(encoder_outputs, tuple):
            encoder_outputs = encoder_outputs.to_tuple()

        # 调用 decoder 进行解码，使用 encoder 输出作为其中的一些参数
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

        # 如果 return_dict=False，则将 decoder 和 encoder 输出组合后返回
        if not return_dict:
            return decoder_outputs + encoder_outputs

        # 如果 return_dict=True，则根据 TFSeq2SeqModelOutput 的结构返回 decoder 和 encoder 的输出
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
    # 构建模型的方法，在输入形状为None时
    def build(self, input_shape=None):
        # 如果模型已经构建完成，则直接返回
        if self.built:
            return
        # 设置模型已构建的标志为True
        self.built = True
        
        # 共享/绑定的权重期望位于模型基本命名空间中
        # 将"/"添加到tf.name_scope的末尾（而不是开头！）会将其放置在根命名空间而不是当前命名空间中。
        with tf.name_scope(self.shared.load_weight_prefix + "/" + self.shared.name + "/"):
            # 构建共享/绑定模型
            self.shared.build(None)
        
        # 如果存在编码器(encoder)模型，则在其命名空间内构建
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        
        # 如果存在解码器(decoder)模型，则在其命名空间内构建
        if getattr(self, "decoder", None) is not None:
            with tf.name_scope(self.decoder.name):
                self.decoder.build(None)
# 为 TFBlenderbotSmallModel 类添加文档字符串，说明这是一个不带特定顶部头的原始隐藏状态输出的 BLENDERBOT_SMALL 模型。
@add_start_docstrings(
    "The bare BLENDERBOT_SMALL Model outputting raw hidden-states without any specific head on top.",
    BLENDERBOT_SMALL_START_DOCSTRING,
)
class TFBlenderbotSmallModel(TFBlenderbotSmallPreTrainedModel):
    def __init__(self, config: BlenderbotSmallConfig, *inputs, **kwargs):
        # 调用父类的构造函数，传递配置和其他输入参数
        super().__init__(config, *inputs, **kwargs)

        # 创建 TFBlenderbotSmallMainLayer 实例作为模型的主要层
        self.model = TFBlenderbotSmallMainLayer(config, name="model")

    # 返回模型的编码器部分
    def get_encoder(self):
        return self.model.encoder

    # 返回模型的解码器部分
    def get_decoder(self):
        return self.model.decoder

    # 定义模型的前向传播方法，接收多个输入参数，输出模型的结果
    @unpack_inputs
    @add_start_docstrings_to_model_forward(BLENDERBOT_SMALL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFSeq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
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
        # 调用模型的前向传播方法，将输入参数传递给模型并获取输出结果
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

        # 返回模型的输出结果
        return outputs

    # 从 transformers.models.bart.modeling_tf_bart.TFBartModel.serving_output 复制并注释
    # 该部分功能的具体内容未在提供的代码片段中给出，需要进一步补充
    # 定义一个方法用于处理模型的输出
    def serving_output(self, output):
        # 如果配置要求使用缓存，则从输出中获取过去键值对中的第二个元素；否则设为 None
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        # 如果配置要求输出隐藏状态，则将输出的解码器隐藏状态转换为张量；否则设为 None
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置要求输出注意力权重，则将输出的解码器注意力权重转换为张量；否则设为 None
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        # 如果配置要求输出交叉注意力权重，则将输出的交叉注意力权重转换为张量；否则设为 None
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        # 如果配置要求输出隐藏状态，则将输出的编码器隐藏状态转换为张量；否则设为 None
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置要求输出注意力权重，则将输出的编码器注意力权重转换为张量；否则设为 None
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None

        # 返回一个 TFSeq2SeqModelOutput 对象，包含不同类型的模型输出
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

    # 构建方法，用于建立模型结构
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 设定为已构建状态
        self.built = True
        # 如果已存在模型，则在指定的命名空间下构建模型
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
        # 添加权重到层中，用于偏置项，名称不会进行作用域化处理以便正确序列化
        self.bias = self.add_weight(name=name, shape=shape, initializer=initializer, trainable=trainable)

    def call(self, x):
        # 在输入张量 x 上加上偏置项
        return x + self.bias


@add_start_docstrings(
    "The BLENDERBOT_SMALL Model with a language modeling head. Can be used for summarization.",
    BLENDERBOT_SMALL_START_DOCSTRING,
)
class TFBlenderbotSmallForConditionalGeneration(TFBlenderbotSmallPreTrainedModel, TFCausalLanguageModelingLoss):
    _keys_to_ignore_on_load_unexpected = [
        r"model.encoder.embed_tokens.weight",
        r"model.decoder.embed_tokens.weight",
    ]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 创建 TFBlenderbotSmallMainLayer 实例作为模型主体，并命名为 "model"
        self.model = TFBlenderbotSmallMainLayer(config, name="model")
        # 从配置中获取是否使用缓存
        self.use_cache = config.use_cache
        # 创建 BiasLayer 实例作为模型的偏置项，用于最终的 logits，设置为不可训练以保持一致性
        self.bias_layer = BiasLayer(
            name="final_logits_bias", shape=[1, config.vocab_size], initializer="zeros", trainable=False
        )

    def get_decoder(self):
        # 返回模型的解码器部分
        return self.model.decoder

    def get_encoder(self):
        # 返回模型的编码器部分
        return self.model.encoder

    def get_output_embeddings(self):
        # 返回输入嵌入层
        return self.get_input_embeddings()

    def set_output_embeddings(self, value):
        # 设置输出嵌入层
        self.set_input_embeddings(value)

    def get_bias(self):
        # 返回偏置项字典
        return {"final_logits_bias": self.bias_layer.bias}

    def set_bias(self, value):
        # 替换已有的包含偏置项的层，以便正确序列化和反序列化
        vocab_size = value["final_logits_bias"].shape[-1]
        self.bias_layer = BiasLayer(
            name="final_logits_bias", shape=[1, vocab_size], initializer="zeros", trainable=False
        )
        # 将新的偏置值赋给偏置层
        self.bias_layer.bias.assign(value["final_logits_bias"])

    @unpack_inputs
    @add_start_docstrings_to_model_forward(BLENDERBOT_SMALL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(BLENDERBOT_SMALL_GENERATION_EXAMPLE)
    # 定义一个方法用于执行模型的前向传播。参数如下：

    # input_ids: 输入的张量，表示模型的输入序列的标识符
    input_ids: tf.Tensor | None = None,

    # attention_mask: 输入的张量，用于指示哪些位置的标识符需要被注意力层忽略
    attention_mask: tf.Tensor | None = None,

    # decoder_input_ids: 解码器的输入序列的标识符
    decoder_input_ids: tf.Tensor | None = None,

    # decoder_attention_mask: 解码器的输入张量，指示哪些位置的标识符需要被注意力层忽略
    decoder_attention_mask: tf.Tensor | None = None,

    # decoder_position_ids: 解码器的位置标识符
    decoder_position_ids: tf.Tensor | None = None,

    # head_mask: 指定哪些注意力头部应该被屏蔽的张量
    head_mask: tf.Tensor | None = None,

    # decoder_head_mask: 解码器的注意力头部的屏蔽张量
    decoder_head_mask: tf.Tensor | None = None,

    # cross_attn_head_mask: 交叉注意力的头部屏蔽张量
    cross_attn_head_mask: tf.Tensor | None = None,

    # encoder_outputs: 编码器输出的可选结果
    encoder_outputs: Optional[TFBaseModelOutput] = None,

    # past_key_values: 解码器过去的键值对列表
    past_key_values: List[tf.Tensor] | None = None,

    # inputs_embeds: 输入的嵌入张量
    inputs_embeds: tf.Tensor | None = None,

    # decoder_inputs_embeds: 解码器的输入嵌入张量
    decoder_inputs_embeds: tf.Tensor | None = None,

    # use_cache: 是否使用缓存的布尔值
    use_cache: Optional[bool] = None,

    # output_attentions: 是否输出注意力权重的布尔值
    output_attentions: Optional[bool] = None,

    # output_hidden_states: 是否输出隐藏状态的布尔值
    output_hidden_states: Optional[bool] = None,

    # return_dict: 是否返回字典格式的输出结果的布尔值
    return_dict: Optional[bool] = None,

    # labels: 标签张量，用于模型训练
    labels: tf.Tensor | None = None,

    # training: 是否为训练模式的布尔值，默认为False
    training: Optional[bool] = False,
    ) -> Union[Tuple[tf.Tensor], TFSeq2SeqLMOutput]:
        r"""
        labels (`tf.tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
            Returns a tuple containing either masked_lm_loss and model outputs or a TFSeq2SeqLMOutput object.

        """

        # Adjust labels to replace pad_token_id with -100, preserving dtype
        if labels is not None:
            labels = tf.where(
                labels == self.config.pad_token_id,
                tf.cast(tf.fill(shape_list(labels), -100), labels.dtype),
                labels,
            )
            # Set use_cache to False if decoder_input_ids or decoder_inputs_embeds are not provided
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                # Shift labels to the right and prepend decoder_start_token_id
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        # Pass inputs to the model for computation
        outputs = self.model(
            input_ids,
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
        
        # Compute logits and apply bias
        lm_logits = tf.matmul(outputs[0], self.model.shared.weights, transpose_b=True)
        lm_logits = self.bias_layer(lm_logits)
        # Compute masked language modeling loss if labels are provided
        masked_lm_loss = None if labels is None else self.hf_compute_loss(labels, lm_logits)

        # Return outputs based on return_dict flag
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        # Return TFSeq2SeqLMOutput object containing relevant model outputs
        return TFSeq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,  # index 1 of d outputs
            decoder_hidden_states=outputs.decoder_hidden_states,  # index 2 of d outputs
            decoder_attentions=outputs.decoder_attentions,  # index 3 of d outputs
            cross_attentions=outputs.cross_attentions,  # index 4 of d outputs
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,  # index 0 of encoder outputs
            encoder_hidden_states=outputs.encoder_hidden_states,  # index 1 of encoder outputs
            encoder_attentions=outputs.encoder_attentions,  # index 2 of encoder outputs
        )

    # Copied from transformers.models.bart.modeling_tf_bart.TFBartForConditionalGeneration.serving_output
    # 定义一个方法用于生成模型输出，将输入的输出对象output转换为TFSeq2SeqLMOutput对象
    def serving_output(self, output):
        # 如果配置允许使用缓存，则从output的过去键值对中获取第一个元素作为past_key_values
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        # 如果配置允许输出隐藏状态，则将output的解码器隐藏状态转换为张量dec_hs
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置允许输出注意力权重，则将output的解码器注意力转换为张量dec_attns
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        # 如果配置允许输出注意力权重，则将output的交叉注意力转换为张量cross_attns
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        # 如果配置允许输出隐藏状态，则将output的编码器隐藏状态转换为张量enc_hs
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置允许输出注意力权重，则将output的编码器注意力转换为张量enc_attns
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None

        # 返回一个TFSeq2SeqLMOutput对象，包含logits、past_key_values、decoder_hidden_states、decoder_attentions、
        # cross_attentions、encoder_last_hidden_state、encoder_hidden_states和encoder_attentions等属性
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

    # 从transformers库中复制的方法，用于生成生成过程的输入
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
        # 如果past_key_values不为None，则截取decoder_input_ids的最后一个标记作为输入
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        # 如果decoder_attention_mask不为None，则使用累积求和操作计算decoder_position_ids
        if decoder_attention_mask is not None:  # xla
            decoder_position_ids = tf.math.cumsum(decoder_attention_mask, axis=-1, exclusive=True)[:, -1:]
        # 否则如果past_key_values不为None，则根据past_key_values的形状获取decoder_position_ids
        elif past_key_values is not None:  # no xla + past_key_values
            decoder_position_ids = past_key_values[0][0].shape[2]
        # 否则使用tf.range生成decoder_input_ids的位置ids作为decoder_position_ids
        else:  # no xla + no past_key_values
            decoder_position_ids = tf.range(decoder_input_ids.shape[1])

        # 返回一个字典，包含生成过程中的所有输入参数
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
    # 定义一个方法 `build`，用于构建神经网络层次结构
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回，避免重复构建
        if self.built:
            return
        # 标记为已构建状态
        self.built = True
        
        # 如果存在名为 `model` 的属性且不为 None，则进入条件
        if getattr(self, "model", None) is not None:
            # 使用 `model` 的名字作为命名空间，构建模型
            with tf.name_scope(self.model.name):
                self.model.build(None)
        
        # 如果存在名为 `bias_layer` 的属性且不为 None，则进入条件
        if getattr(self, "bias_layer", None) is not None:
            # 使用 `bias_layer` 的名字作为命名空间，构建偏置层
            with tf.name_scope(self.bias_layer.name):
                self.bias_layer.build(None)
```