# `.\models\marian\modeling_tf_marian.py`

```py
# coding=utf-8
# 版权所有 2021 年 The Marian Team 作者和 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件是基于“按现状”提供的，
# 没有任何明示或暗示的保证或条件。
# 请参阅许可证以获取特定语言的权限和限制。
""" TF 2.0 Marian model."""

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
from .configuration_marian import MarianConfig

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "Helsinki-NLP/opus-mt-en-de"
_CONFIG_FOR_DOC = "MarianConfig"

LARGE_NEGATIVE = -1e8

# 从 transformers.models.bart.modeling_tf_bart.shift_tokens_right 复制而来
def shift_tokens_right(input_ids: tf.Tensor, pad_token_id: int, decoder_start_token_id: int):
    pad_token_id = tf.cast(pad_token_id, input_ids.dtype)
    decoder_start_token_id = tf.cast(decoder_start_token_id, input_ids.dtype)
    # 创建一个形状为 (input_ids 的行数, 1) 的张量，填充为 decoder_start_token_id
    start_tokens = tf.fill(
        (shape_list(input_ids)[0], 1), tf.convert_to_tensor(decoder_start_token_id, input_ids.dtype)
    )
    # 将 input_ids 右移一位，将 start_tokens 和 input_ids 的前 n-1 列拼接起来
    shifted_input_ids = tf.concat([start_tokens, input_ids[:, :-1]], -1)
    # 如果 labels 中存在 -100 的值，则用 pad_token_id 替换
    shifted_input_ids = tf.where(
        shifted_input_ids == -100,
        tf.fill(shape_list(shifted_input_ids), tf.convert_to_tensor(pad_token_id, input_ids.dtype)),
        shifted_input_ids,
    )

    # "验证 `labels` 只包含正值和 -100"
    assert_gte0 = tf.debugging.assert_greater_equal(shifted_input_ids, tf.constant(0, dtype=input_ids.dtype))

    # 确保断言操作被调用，通过将结果包装在 identity 操作中
    with tf.control_dependencies([assert_gte0]):
        shifted_input_ids = tf.identity(shifted_input_ids)

    return shifted_input_ids
# 创建一个用于双向自注意力的因果遮罩
def _make_causal_mask(input_ids_shape: tf.TensorShape, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    # 获取批次大小
    bsz = input_ids_shape[0]
    # 获取目标序列长度
    tgt_len = input_ids_shape[1]
    # 创建一个形状为 (tgt_len, tgt_len) 的全1矩阵，并乘以一个大负数以表示遮罩
    mask = tf.ones((tgt_len, tgt_len)) * LARGE_NEGATIVE
    # 创建一个序列长度的范围
    mask_cond = tf.range(shape_list(mask)[-1])

    # 将遮罩中对角线以下的元素设置为0
    mask = tf.where(mask_cond < tf.reshape(mask_cond + 1, (shape_list(mask)[-1], 1)), 0.0, mask)

    # 如果过去键值长度大于0，则在遮罩左侧添加零矩阵
    if past_key_values_length > 0:
        mask = tf.concat([tf.zeros((tgt_len, past_key_values_length)), mask], axis=-1)

    # 将遮罩扩展为四维张量并返回
    return tf.tile(mask[None, None, :, :], (bsz, 1, 1, 1))


# 从 transformers.models.bart.modeling_tf_bart._expand_mask 复制过来的函数
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
    # 将遮罩转换为浮点数类型
    mask = tf.cast(mask, dtype=one_cst.dtype)
    # 将遮罩在第二维上复制tgt_len次，以扩展为四维张量
    expanded_mask = tf.tile(mask[:, None, None, :], (1, 1, tgt_len, 1))

    # 返回扩展后的遮罩并应用大负数
    return (one_cst - expanded_mask) * LARGE_NEGATIVE


class TFMarianSinusoidalPositionalEmbedding(keras.layers.Layer):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, **kwargs):
        super().__init__(**kwargs)

        # 如果嵌入维度为奇数，则抛出错误
        if embedding_dim % 2 != 0:
            raise NotImplementedError(f"odd embedding_dim {embedding_dim} not supported")

        self.embedding_dim = embedding_dim
        self.num_positions = num_positions

    def build(self, input_shape: tf.TensorShape):
        """
        Build shared token embedding layer Shared weights logic adapted from
        https://github.com/tensorflow/models/blob/a009f4fb9d2fc4949e32192a944688925ef78659/official/transformer/v2/embedding_layer.py#L24
        """

        # 初始化位置编码权重
        weight = self._init_weight(self.num_positions, self.embedding_dim)

        # 添加权重张量到层中
        self.weight = self.add_weight(
            name="embeddings",
            shape=[self.num_positions, self.embedding_dim],
        )
        # 将初始化的权重转换为与self.weight相同的数据类型并分配给self.weight
        weight = tf.cast(weight, dtype=self.weight.dtype)
        self.weight.assign(weight)

        super().build(input_shape)

    @staticmethod
    def _init_weight(n_pos: int, dim: int):
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        # 创建位置编码矩阵，使用正弦和余弦函数
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        table = np.zeros_like(position_enc)
        # 第一列全为零
        table[:, 0 : dim // 2] = np.sin(position_enc[:, 0::2])
        # 第二列为余弦值
        table[:, dim // 2 :] = np.cos(position_enc[:, 1::2])
        # 将表格转换为张量
        table = tf.convert_to_tensor(table)
        # 停止梯度传播
        tf.stop_gradient(table)
        return table
    def call(
        self, input_shape: tf.TensorShape, past_key_values_length: int = 0, position_ids: tf.Tensor | None = None
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        # 如果未提供位置 ID，则根据输入形状和过去键值长度生成位置 ID
        if position_ids is None:
            # 获取输入的序列长度
            seq_len = input_shape[1]
            # 使用 TensorFlow 的 range 函数生成位置 ID，起始值为 past_key_values_length，
            # 终止值为 seq_len + past_key_values_length，步长为 1
            position_ids = tf.range(past_key_values_length, seq_len + past_key_values_length, delta=1, name="range")
        # 根据位置 ID 从 self.weight 中收集对应的权重值
        return tf.gather(self.weight, position_ids)
# 从 transformers.models.bart.modeling_tf_bart.TFBartAttention 复制并修改为 Bart->Marian
class TFMarianAttention(keras.layers.Layer):
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
        self.embed_dim = embed_dim  # 初始化注意力机制的嵌入维度

        self.num_heads = num_heads  # 头数，决定了注意力头的数量
        self.dropout = keras.layers.Dropout(dropout)  # dropout层，用于随机失活
        self.head_dim = embed_dim // num_heads  # 每个头的维度
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"  # 检查嵌入维度是否可以被头数整除
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5  # 缩放因子，用于调整注意力分数
        self.is_decoder = is_decoder  # 是否为解码器

        self.k_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="k_proj")  # k投影层，将输入投影到k空间
        self.q_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="q_proj")  # q投影层，将输入投影到q空间
        self.v_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="v_proj")  # v投影层，将输入投影到v空间
        self.out_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="out_proj")  # 输出投影层，将合并的注意力头投影到输出维度

    def _shape(self, tensor: tf.Tensor, seq_len: int, bsz: int):
        return tf.transpose(tf.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        # 重新整形张量以适应多头注意力的形状，包括张量的转置和重塑操作

    def call(
        self,
        hidden_states: tf.Tensor,
        key_value_states: tf.Tensor | None = None,
        past_key_value: Tuple[Tuple[tf.Tensor]] | None = None,
        attention_mask: tf.Tensor | None = None,
        layer_head_mask: tf.Tensor | None = None,
        training: Optional[bool] = False,
    ):
        # 定义层的前向传播逻辑
        ...

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "k_proj", None) is not None:
            with tf.name_scope(self.k_proj.name):
                self.k_proj.build([None, None, self.embed_dim])
        if getattr(self, "q_proj", None) is not None:
            with tf.name_scope(self.q_proj.name):
                self.q_proj.build([None, None, self.embed_dim])
        if getattr(self, "v_proj", None) is not None:
            with tf.name_scope(self.v_proj.name):
                self.v_proj.build([None, None, self.embed_dim])
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build([None, None, self.embed_dim])
        # 构建函数，用于按需创建投影层



# 从 transformers.models.bart.modeling_tf_bart.TFBartEncoderLayer 复制并修改为 Bart->Marian
class TFMarianEncoderLayer(keras.layers.Layer):
    # 编码器层类，适用于 Marian 模型，从 BART 模型修改而来
    ...
    # 初始化方法，接受一个MarianConfig对象和额外的关键字参数
    def __init__(self, config: MarianConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 设置嵌入维度为config中的模型维度d_model
        self.embed_dim = config.d_model
        # 创建自注意力层TFMarianAttention对象，使用config中的参数设置
        self.self_attn = TFMarianAttention(
            self.embed_dim, config.encoder_attention_heads, dropout=config.attention_dropout, name="self_attn"
        )
        # 创建自注意力层的LayerNormalization层
        self.self_attn_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        # 创建Dropout层，使用config中的dropout参数
        self.dropout = keras.layers.Dropout(config.dropout)
        # 获取激活函数，根据config中的激活函数类型
        self.activation_fn = get_tf_activation(config.activation_function)
        # 创建激活函数的Dropout层，使用config中的activation_dropout参数
        self.activation_dropout = keras.layers.Dropout(config.activation_dropout)
        # 创建全连接层fc1，输出维度为config中的encoder_ffn_dim
        self.fc1 = keras.layers.Dense(config.encoder_ffn_dim, name="fc1")
        # 创建全连接层fc2，输出维度为self.embed_dim
        self.fc2 = keras.layers.Dense(self.embed_dim, name="fc2")
        # 创建最终的LayerNormalization层
        self.final_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")
        # 保存传入的MarianConfig对象
        self.config = config

    # call方法用于执行实际的前向传播逻辑
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: np.ndarray | tf.Tensor | None,
        layer_head_mask: tf.Tensor | None,
        training: Optional[bool] = False,
    ) -> tf.Tensor:
        """
        Args:
            hidden_states (`tf.Tensor`): 输入层的张量，形状为 `(batch, seq_len, embed_dim)`
            attention_mask (`tf.Tensor`): 注意力掩码，形状为 `(batch, 1, tgt_len, src_len)`，
                                          其中填充元素由非常大的负值指示。
            layer_head_mask (`tf.Tensor`): 给定层的注意力头掩码，形状为 `(encoder_attention_heads,)`
            training (`Optional[bool]`, optional): 是否处于训练模式，默认为False。
        Returns:
            tf.Tensor: 返回处理后的张量，形状为 `(batch, seq_len, embed_dim)`
        """
        # 保存输入的隐藏状态作为残差连接的一部分
        residual = hidden_states
        # 使用自注意力层处理隐藏状态，得到处理后的隐藏状态、注意力权重和附加信息
        hidden_states, self_attn_weights, _ = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, layer_head_mask=layer_head_mask
        )

        # 断言：确保自注意力层没有修改查询的形状
        tf.debugging.assert_equal(
            shape_list(hidden_states),
            shape_list(residual),
            message=f"Self attn modified the shape of query {shape_list(residual)} to {shape_list(hidden_states)}",
        )

        # 对处理后的隐藏状态应用Dropout层，根据training参数决定是否使用训练模式
        hidden_states = self.dropout(hidden_states, training=training)
        # 将残差连接到处理后的隐藏状态上
        hidden_states = residual + hidden_states
        # 使用自注意力层的LayerNormalization层对结果进行归一化处理
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 保存处理后的隐藏状态作为新的残差连接的一部分
        residual = hidden_states
        # 使用激活函数处理第一个全连接层的结果
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 对处理后的结果应用激活函数的Dropout层，根据training参数决定是否使用训练模式
        hidden_states = self.activation_dropout(hidden_states, training=training)
        # 使用第二个全连接层处理结果
        hidden_states = self.fc2(hidden_states)
        # 对处理后的结果应用Dropout层，根据training参数决定是否使用训练模式
        hidden_states = self.dropout(hidden_states, training=training)
        # 将残差连接到处理后的结果上
        hidden_states = residual + hidden_states
        # 使用最终的LayerNormalization层对结果进行归一化处理
        hidden_states = self.final_layer_norm(hidden_states)

        # 返回处理后的隐藏状态和自注意力权重
        return hidden_states, self_attn_weights
    # 在神经网络层的建立函数中，用于构建网络结构
    def build(self, input_shape=None):
        # 如果已经建立过网络，则直接返回，不重复建立
        if self.built:
            return
        # 将标记设为已建立
        self.built = True
        
        # 如果存在 self_attn 属性，则构建 self_attention 层
        if getattr(self, "self_attn", None) is not None:
            # 使用 self_attn 层的名称作为命名空间
            with tf.name_scope(self.self_attn.name):
                # 调用 self_attn 层的建立函数
                self.self_attn.build(None)
        
        # 如果存在 self_attn_layer_norm 属性，则构建 self_attention 层的 LayerNormalization
        if getattr(self, "self_attn_layer_norm", None) is not None:
            # 使用 self_attn_layer_norm 层的名称作为命名空间
            with tf.name_scope(self.self_attn_layer_norm.name):
                # 调用 self_attn_layer_norm 层的建立函数，输入形状为 [None, None, self.embed_dim]
                self.self_attn_layer_norm.build([None, None, self.embed_dim])
        
        # 如果存在 fc1 属性，则构建第一个全连接层
        if getattr(self, "fc1", None) is not None:
            # 使用 fc1 层的名称作为命名空间
            with tf.name_scope(self.fc1.name):
                # 调用 fc1 层的建立函数，输入形状为 [None, None, self.embed_dim]
                self.fc1.build([None, None, self.embed_dim])
        
        # 如果存在 fc2 属性，则构建第二个全连接层
        if getattr(self, "fc2", None) is not None:
            # 使用 fc2 层的名称作为命名空间
            with tf.name_scope(self.fc2.name):
                # 调用 fc2 层的建立函数，输入形状为 [None, None, self.config.encoder_ffn_dim]
                self.fc2.build([None, None, self.config.encoder_ffn_dim])
        
        # 如果存在 final_layer_norm 属性，则构建最终的 LayerNormalization
        if getattr(self, "final_layer_norm", None) is not None:
            # 使用 final_layer_norm 层的名称作为命名空间
            with tf.name_scope(self.final_layer_norm.name):
                # 调用 final_layer_norm 层的建立函数，输入形状为 [None, None, self.embed_dim]
                self.final_layer_norm.build([None, None, self.embed_dim])
# 从 transformers.models.bart.modeling_tf_bart.TFBartDecoderLayer 复制而来，将 Bart 改为 Marian
class TFMarianDecoderLayer(keras.layers.Layer):
    def __init__(self, config: MarianConfig, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = config.d_model  # 初始化嵌入维度为配置中的模型维度
        self.self_attn = TFMarianAttention(  # 创建自注意力层对象
            embed_dim=self.embed_dim,  # 使用配置中的模型维度
            num_heads=config.decoder_attention_heads,  # 使用配置中的解码器注意力头数
            dropout=config.attention_dropout,  # 使用配置中的注意力丢弃率
            name="self_attn",  # 层名称为 self_attn
            is_decoder=True,  # 标记这是一个解码器注意力层
        )
        self.dropout = keras.layers.Dropout(config.dropout)  # 使用配置中的丢弃率创建丢弃层
        self.activation_fn = get_tf_activation(config.activation_function)  # 获取激活函数
        self.activation_dropout = keras.layers.Dropout(config.activation_dropout)  # 使用配置中的激活函数丢弃率创建丢弃层

        self.self_attn_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        # 创建自注意力层后的层归一化层

        self.encoder_attn = TFMarianAttention(  # 创建编码器注意力层对象
            self.embed_dim,  # 使用配置中的模型维度
            config.decoder_attention_heads,  # 使用配置中的解码器注意力头数
            dropout=config.attention_dropout,  # 使用配置中的注意力丢弃率
            name="encoder_attn",  # 层名称为 encoder_attn
            is_decoder=True,  # 标记这是一个解码器注意力层
        )
        self.encoder_attn_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="encoder_attn_layer_norm")
        # 创建编码器注意力层后的层归一化层

        self.fc1 = keras.layers.Dense(config.decoder_ffn_dim, name="fc1")  # 创建全连接层 fc1
        self.fc2 = keras.layers.Dense(self.embed_dim, name="fc2")  # 创建全连接层 fc2

        self.final_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")
        # 最终输出后的层归一化层

        self.config = config  # 保存配置对象

    def call(
        self,
        hidden_states: tf.Tensor,  # 输入隐藏状态张量
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 注意力掩码
        encoder_hidden_states: np.ndarray | tf.Tensor | None = None,  # 编码器隐藏状态张量或数组
        encoder_attention_mask: np.ndarray | tf.Tensor | None = None,  # 编码器注意力掩码
        layer_head_mask: tf.Tensor | None = None,  # 层头部掩码
        cross_attn_layer_head_mask: tf.Tensor | None = None,  # 跨注意力层头部掩码
        past_key_value: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,  # 过去的键值对（可选）
        training: Optional[bool] = False,  # 训练模式（可选，默认为 False）
    # 定义 build 方法，用于构建模型的层次结构
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回，避免重复构建
        if self.built:
            return
        # 设置标志位，表示模型已经构建
        self.built = True
        
        # 如果存在 self_attn 属性，则构建 self attention 层
        if getattr(self, "self_attn", None) is not None:
            # 在命名空间 self_attn 下，构建 self attention 层
            with tf.name_scope(self.self_attn.name):
                self.self_attn.build(None)
        
        # 如果存在 self_attn_layer_norm 属性，则构建 self attention 层的 layer normalization 层
        if getattr(self, "self_attn_layer_norm", None) is not None:
            # 在命名空间 self_attn_layer_norm 下，构建 layer normalization 层
            with tf.name_scope(self.self_attn_layer_norm.name):
                self.self_attn_layer_norm.build([None, None, self.embed_dim])
        
        # 如果存在 encoder_attn 属性，则构建 encoder attention 层
        if getattr(self, "encoder_attn", None) is not None:
            # 在命名空间 encoder_attn 下，构建 encoder attention 层
            with tf.name_scope(self.encoder_attn.name):
                self.encoder_attn.build(None)
        
        # 如果存在 encoder_attn_layer_norm 属性，则构建 encoder attention 层的 layer normalization 层
        if getattr(self, "encoder_attn_layer_norm", None) is not None:
            # 在命名空间 encoder_attn_layer_norm 下，构建 layer normalization 层
            with tf.name_scope(self.encoder_attn_layer_norm.name):
                self.encoder_attn_layer_norm.build([None, None, self.embed_dim])
        
        # 如果存在 fc1 属性，则构建第一个全连接层
        if getattr(self, "fc1", None) is not None:
            # 在命名空间 fc1 下，构建全连接层
            with tf.name_scope(self.fc1.name):
                self.fc1.build([None, None, self.embed_dim])
        
        # 如果存在 fc2 属性，则构建第二个全连接层
        if getattr(self, "fc2", None) is not None:
            # 在命名空间 fc2 下，构建全连接层，输入维度为 decoder_ffn_dim
            with tf.name_scope(self.fc2.name):
                self.fc2.build([None, None, self.config.decoder_ffn_dim])
        
        # 如果存在 final_layer_norm 属性，则构建最终的 layer normalization 层
        if getattr(self, "final_layer_norm", None) is not None:
            # 在命名空间 final_layer_norm 下，构建 layer normalization 层
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.embed_dim])
# 定义一个名为 TFMarianPreTrainedModel 的类，继承自 TFPreTrainedModel 类
class TFMarianPreTrainedModel(TFPreTrainedModel):
    # 指定配置类为 MarianConfig
    config_class = MarianConfig
    # 指定基础模型前缀为 "model"
    base_model_prefix = "model"


# 定义一个文档字符串常量 MARIAN_START_DOCSTRING
MARIAN_START_DOCSTRING = r"""
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
        config ([`MarianConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""
MARIAN_GENERATION_EXAMPLE = r"""
        TF version of marian-nmt's transformer.h (c++). Designed for the OPUS-NMT translation checkpoints. Available
        models are listed [here](https://huggingface.co/models?search=Helsinki-NLP).

        Examples:

        ```
        >>> from transformers import AutoTokenizer, TFMarianMTModel
        >>> from typing import List

        >>> src = "fr"  # source language
        >>> trg = "en"  # target language
        >>> sample_text = "où est l'arrêt de bus ?"
        >>> model_name = f"Helsinki-NLP/opus-mt-{src}-{trg}"

        >>> model = TFMarianMTModel.from_pretrained(model_name)
        >>> tokenizer = AutoTokenizer.from_pretrained(model_name)
        >>> batch = tokenizer([sample_text], return_tensors="tf")
        >>> gen = model.generate(**batch)
        >>> tokenizer.batch_decode(gen, skip_special_tokens=True)
        "Where is the bus stop ?"
        ```
"""

MARIAN_INPUTS_DOCSTRING = r"""
"""

# 自定义 Keras 层 `TFMarianEncoder`，标记为可序列化
@keras_serializable
class TFMarianEncoder(keras.layers.Layer):
    # 配置类为 MarianConfig
    config_class = MarianConfig

    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`TFMarianEncoderLayer`].

    Args:
        config: MarianConfig
    """
    
    # 初始化方法
    def __init__(self, config: MarianConfig, embed_tokens: Optional[keras.layers.Embedding] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config  # 存储配置信息
        self.dropout = keras.layers.Dropout(config.dropout)  # Dropout 层，使用配置的 dropout 比率
        self.layerdrop = config.encoder_layerdrop  # Encoder 层 dropout 比率
        self.padding_idx = config.pad_token_id  # 填充 token 的索引
        self.max_source_positions = config.max_position_embeddings  # 最大源位置数
        self.embed_scale = tf.math.sqrt(float(config.d_model)) if config.scale_embedding else 1.0  # 嵌入向量缩放因子

        self.embed_tokens = embed_tokens  # 嵌入 token
        self.embed_positions = TFMarianSinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            name="embed_positions",
        )  # Sinusoidal 位置嵌入

        # 创建多个 Transformer Encoder 层
        self.layers = [TFMarianEncoderLayer(config, name=f"layers.{i}") for i in range(config.encoder_layers)]

    # 获取嵌入 token
    def get_embed_tokens(self):
        return self.embed_tokens

    # 设置嵌入 token
    def set_embed_tokens(self, embed_tokens):
        self.embed_tokens = embed_tokens

    # 对输入进行解包并调用处理的方法
    @unpack_inputs
    def call(
        self,
        input_ids: tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        attention_mask: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        **kwargs,
    ):
    # 定义神经网络层的构建方法，用于在指定输入形状下构建网络层
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回，避免重复构建
        if self.built:
            return
        # 将网络标记为已构建状态
        self.built = True
        
        # 如果存在嵌入位置信息的属性，则构建嵌入位置信息
        if getattr(self, "embed_positions", None) is not None:
            # 使用该属性的命名空间创建名称作用域，并构建嵌入位置信息
            with tf.name_scope(self.embed_positions.name):
                self.embed_positions.build(None)
        
        # 如果存在多个层，则依次构建每个层
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                # 使用每个层的名称创建名称作用域，并构建该层
                with tf.name_scope(layer.name):
                    layer.build(None)
@keras_serializable
class TFMarianDecoder(keras.layers.Layer):
    # 指定配置类为MarianConfig
    config_class = MarianConfig
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`TFMarianDecoderLayer`]

    Args:
        config: MarianConfig  # 输入参数为MarianConfig类型的配置对象
        embed_tokens: output embedding  # 输出嵌入的标记
    """

    def __init__(self, config: MarianConfig, embed_tokens: Optional[keras.layers.Embedding] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config  # 保存配置对象
        self.padding_idx = config.pad_token_id  # 获取填充标记ID
        self.embed_tokens = embed_tokens  # 保存嵌入标记对象
        self.layerdrop = config.decoder_layerdrop  # 获取层丢弃概率
        self.embed_positions = TFMarianSinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            name="embed_positions",
        )  # 创建Sinusoidal位置嵌入对象
        self.embed_scale = tf.math.sqrt(float(config.d_model)) if config.scale_embedding else 1.0  # 根据scale_embedding决定缩放因子
        self.layers = [TFMarianDecoderLayer(config, name=f"layers.{i}") for i in range(config.decoder_layers)]  # 创建多层解码器层对象

        self.dropout = keras.layers.Dropout(config.dropout)  # 创建dropout层对象

    def get_embed_tokens(self):
        return self.embed_tokens  # 返回嵌入标记对象

    def set_embed_tokens(self, embed_tokens):
        self.embed_tokens = embed_tokens  # 设置嵌入标记对象

    @unpack_inputs
    def call(
        self,
        input_ids: tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        attention_mask: tf.Tensor | None = None,
        position_ids: tf.Tensor | None = None,
        encoder_hidden_states: tf.Tensor | None = None,
        encoder_attention_mask: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        cross_attn_head_mask: tf.Tensor | None = None,
        past_key_values: Tuple[Tuple[tf.Tensor]] | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ):
        # 定义Transformer解码器的前向传播过程
        ...

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "embed_positions", None) is not None:
            with tf.name_scope(self.embed_positions.name):
                self.embed_positions.build(None)  # 构建位置嵌入对象
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)  # 逐层构建解码器层对象


@keras_serializable
class TFMarianMainLayer(keras.layers.Layer):
    config_class = MarianConfig
    # 初始化函数，接受一个MarianConfig对象和其他关键字参数
    def __init__(self, config: MarianConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 将传入的配置对象存储在self.config中
        self.config = config

        # 创建一个共享的Embedding层，用于编码器和解码器共享
        self.shared = keras.layers.Embedding(
            input_dim=config.vocab_size,  # 词汇表大小，作为输入维度
            output_dim=config.d_model,     # 输出维度，通常是模型的维度
            embeddings_initializer=keras.initializers.TruncatedNormal(stddev=self.config.init_std),  # 使用截断正态分布初始化
            name="model.shared",           # 层的名称
        )
        
        # 添加一个额外的属性，指定层的预期名称作用域（用于加载/存储权重）
        self.shared.load_weight_prefix = "model.shared"

        # 创建编码器对象，使用TFMarianEncoder类，并传入配置对象和共享的Embedding层
        self.encoder = TFMarianEncoder(config, self.shared, name="encoder")

        # 创建解码器对象，使用TFMarianDecoder类，并传入配置对象和共享的Embedding层
        self.decoder = TFMarianDecoder(config, self.shared, name="decoder")

    # 获取输入Embedding层的方法
    def get_input_embeddings(self):
        return self.shared

    # 设置输入Embedding层的方法，传入新的Embedding层
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings  # 更新共享的Embedding层
        self.encoder.embed_tokens = self.shared  # 更新编码器的Embedding层
        self.decoder.embed_tokens = self.shared  # 更新解码器的Embedding层

    # 使用unpack_inputs装饰器定义的call方法，实现模型的调用过程
    @unpack_inputs
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
        past_key_values: Tuple[Tuple[tf.Tensor]] = None,
        inputs_embeds: tf.Tensor | None = None,
        decoder_inputs_embeds: tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        **kwargs,
    ):
        # 实现模型的前向传播逻辑，接收一系列输入张量和参数
        pass  # 该方法尚未实现具体的逻辑，只是定义了方法签名和参数
        ):
            # 如果没有提供解码器的输入 ID 和嵌入向量，则不使用缓存
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                use_cache = False
        
            # 如果没有指定输出隐藏状态，则使用模型配置中的默认设置
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
        
            # 如果没有提供编码器输出，则调用编码器进行前向传播计算
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
            # 如果 return_dict=True 并且用户传入的 encoder_outputs 是 tuple 类型，则将其包装成 TFBaseModelOutput 对象
            elif return_dict and not isinstance(encoder_outputs, TFBaseModelOutput):
                encoder_outputs = TFBaseModelOutput(
                    last_hidden_state=encoder_outputs[0],
                    hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                    attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                )
            # 如果 return_dict=False 并且用户传入的 encoder_outputs 是 TFBaseModelOutput 类型，则将其转换成 tuple 类型
            elif not return_dict and not isinstance(encoder_outputs, tuple):
                encoder_outputs = encoder_outputs.to_tuple()
        
            # 调用解码器进行解码操作
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
        
            # 如果 return_dict=False，则将解码器输出和编码器输出组合成一个 tuple 返回
            if not return_dict:
                return decoder_outputs + encoder_outputs
        
            # 如果 return_dict=True，则将解码器和编码器的输出组装成 TFSeq2SeqModelOutput 对象返回
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
    # 如果模型已经构建，则直接返回，不再重复构建
    if self.built:
        return
    # 设置标志位表示模型已经构建
    self.built = True
    
    # 在模型基础命名空间中设置共享/绑定权重的命名空间
    # 将 "/" 添加到名称作用域的末尾（而不是开头！）将其放置在根命名空间而不是当前命名空间中。
    with tf.name_scope(self.shared.load_weight_prefix + "/" + self.shared.name + "/"):
        # 构建共享权重模块
        self.shared.build(None)
    
    # 如果存在编码器（encoder）模块，则在其命名空间中构建
    if getattr(self, "encoder", None) is not None:
        with tf.name_scope(self.encoder.name):
            self.encoder.build(None)
    
    # 如果存在解码器（decoder）模块，则在其命名空间中构建
    if getattr(self, "decoder", None) is not None:
        with tf.name_scope(self.decoder.name):
            self.decoder.build(None)
# 以裸MARIAN模型为基础，输出未加特定头部的原始隐藏状态。
# 继承自TFMarianPreTrainedModel类，是MARIAN模型的TensorFlow实现。
class TFMarianModel(TFMarianPreTrainedModel):
    def __init__(self, config: MarianConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 使用TFMarianMainLayer初始化模型，命名为"model"
        self.model = TFMarianMainLayer(config, name="model")

    # 返回模型的编码器
    def get_encoder(self):
        return self.model.encoder

    # 返回模型的解码器
    def get_decoder(self):
        return self.model.decoder

    # 对模型进行调用的方法，接受多种输入参数，并根据需要进行处理
    @unpack_inputs
    @add_start_docstrings_to_model_forward(MARIAN_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        encoder_outputs: tf.Tensor | None = None,
        past_key_values: Tuple[Tuple[tf.Tensor]] | None = None,
        inputs_embeds: tf.Tensor | None = None,
        decoder_inputs_embeds: tf.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        training: bool = False,
        **kwargs,
    ) -> Tuple[tf.Tensor] | TFSeq2SeqModelOutput:
        # 将输入参数传递给模型的call方法，返回模型的输出
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

    # 从transformers.models.bart.modeling_tf_bart.TFBartModel.serving_output中复制而来
    # 定义一个方法，用于处理模型输出
    def serving_output(self, output):
        # 如果配置中使用缓存，则获取输出中的过去键值对中的第二个元素（过去的键值对）
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        # 如果配置中需要输出隐藏状态，则将输出中的解码器隐藏状态转换为张量
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置中需要输出注意力权重，则将输出中的解码器注意力权重转换为张量
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        # 如果配置中需要输出交叉注意力权重，则将输出中的交叉注意力权重转换为张量
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        # 如果配置中需要输出隐藏状态，则将输出中的编码器隐藏状态转换为张量
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置中需要输出注意力权重，则将输出中的编码器注意力权重转换为张量
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None

        # 返回一个 TFSeq2SeqModelOutput 对象，包含处理后的各种输出
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

    # 定义一个构建方法，用于构建模型
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记为已经构建
        self.built = True
        # 如果已经存在模型，则在模型的命名空间下构建模型（这里可能是指 TensorFlow 的命名空间）
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
        # 添加一个权重，用于该层的偏置，用于模型的序列化
        self.bias = self.add_weight(name=name, shape=shape, initializer=initializer, trainable=trainable)

    def call(self, x):
        # 在调用时，将偏置加到输入张量 x 上
        return x + self.bias


@add_start_docstrings(
    "The MARIAN Model with a language modeling head. Can be used for summarization.",
    MARIAN_START_DOCSTRING,
)
class TFMarianMTModel(TFMarianPreTrainedModel, TFCausalLanguageModelingLoss):
    _keys_to_ignore_on_load_unexpected = [
        r"model.encoder.embed_tokens.weight",
        r"model.decoder.embed_tokens.weight",
    ]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 创建 MARIAN 模型的主体部分，并命名为 "model"
        self.model = TFMarianMainLayer(config, name="model")
        self.use_cache = config.use_cache
        # 创建一个偏置层 BiasLayer，用于模型的最终 logits 的偏置，保持不可训练状态以保持一致性
        # 这里的 final_logits_bias 在 PyTorch 中作为缓冲区注册，因此在 TensorFlow 中保持不可训练以便正确序列化
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
        # 返回输入的嵌入层
        return self.get_input_embeddings()

    def set_output_embeddings(self, value):
        # 设置输入的嵌入层
        self.set_input_embeddings(value)

    def get_bias(self):
        # 返回模型的偏置信息，这里只包含最终 logits 的偏置
        return {"final_logits_bias": self.bias_layer.bias}

    def set_bias(self, value):
        # 替换已有的偏置层以正确（反）序列化包含偏置的层
        vocab_size = value["final_logits_bias"].shape[-1]
        self.bias_layer = BiasLayer(
            name="final_logits_bias", shape=[1, vocab_size], initializer="zeros", trainable=False
        )
        # 赋予新的偏置值
        self.bias_layer.bias.assign(value["final_logits_bias"])

    @unpack_inputs
    @add_start_docstrings_to_model_forward(MARIAN_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(MARIAN_GENERATION_EXAMPLE)
    # 定义一个方法 `call`，用于执行模型的前向传播或推断过程，支持以下参数：

    # 输入序列的 token IDs 张量，可以是 None（TensorFlow 张量或 None）
    input_ids: tf.Tensor | None = None,

    # 注意力掩码张量，用于指定模型关注哪些 token，可以是 None（TensorFlow 张量或 None）
    attention_mask: tf.Tensor | None = None,

    # 解码器输入序列的 token IDs 张量，可以是 None（TensorFlow 张量或 None）
    decoder_input_ids: tf.Tensor | None = None,

    # 解码器注意力掩码张量，用于指定解码器关注哪些 token，可以是 None（TensorFlow 张量或 None）
    decoder_attention_mask: tf.Tensor | None = None,

    # 解码器位置 IDs 张量，可以是 None（TensorFlow 张量或 None）
    decoder_position_ids: tf.Tensor | None = None,

    # 多头注意力掩码张量，用于指定哪些注意力头应该被屏蔽，可以是 None（TensorFlow 张量或 None）
    head_mask: tf.Tensor | None = None,

    # 解码器多头注意力掩码张量，用于指定解码器的哪些注意力头应该被屏蔽，可以是 None（TensorFlow 张量或 None）
    decoder_head_mask: tf.Tensor | None = None,

    # 交叉注意力头掩码张量，用于指定哪些注意力头应该被屏蔽，可以是 None（TensorFlow 张量或 None）
    cross_attn_head_mask: tf.Tensor | None = None,

    # 编码器输出对象，包含模型的编码器输出，可以是 None（TFBaseModelOutput 或 None）
    encoder_outputs: TFBaseModelOutput | None = None,

    # 过去的键值对，用于存储解码器在自回归生成中的过去内容，可以是 None（元组的元组，每个元组包含 TensorFlow 张量）
    past_key_values: Tuple[Tuple[tf.Tensor]] | None = None,

    # 输入嵌入张量，代替输入序列的 token IDs 张量，可以是 None（TensorFlow 张量或 None）
    inputs_embeds: tf.Tensor | None = None,

    # 解码器输入嵌入张量，代替解码器输入序列的 token IDs 张量，可以是 None（TensorFlow 张量或 None）
    decoder_inputs_embeds: tf.Tensor | None = None,

    # 是否使用缓存，用于指定是否在模型中使用缓存，可以是 None（布尔值或 None）
    use_cache: bool | None = None,

    # 是否输出注意力权重，用于指定是否返回模型中注意力权重，可以是 None（布尔值或 None）
    output_attentions: bool | None = None,

    # 是否输出隐藏状态，用于指定是否返回模型中的隐藏状态，可以是 None（布尔值或 None）
    output_hidden_states: bool | None = None,

    # 是否返回一个字典格式的结果，用于指定是否返回模型输出的字典格式结果，可以是 None（布尔值或 None）
    return_dict: bool | None = None,

    # 标签张量，用于指定训练时的标签值，可以是 None（TensorFlow 张量或 None）
    labels: tf.Tensor | None = None,

    # 是否处于训练模式，用于指定模型当前是否处于训练模式，默认为 False
    training: bool = False,
    ) -> Tuple[tf.Tensor] | TFSeq2SeqLMOutput:
        r"""
        labels (`tf.tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
            Depending on `return_dict`, either a tuple or `TFSeq2SeqLMOutput`.

        """

        # Adjust labels for masked language modeling loss computation
        if labels is not None:
            labels = tf.where(
                labels == self.config.pad_token_id,
                tf.fill(shape_list(labels), tf.cast(-100, labels.dtype)),
                labels,
            )
            # Reset `use_cache` flag if decoder inputs are not provided explicitly
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                # Shift labels to the right for decoder input
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        # Forward pass through the model with specified inputs and parameters
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

        # Compute logits for masked language modeling
        lm_logits = tf.matmul(outputs[0], self.model.shared.weights, transpose_b=True)
        lm_logits = self.bias_layer(lm_logits)
        masked_lm_loss = None if labels is None else self.hf_compute_loss(labels, lm_logits)

        # Return outputs either as a tuple or TFSeq2SeqLMOutput based on `return_dict`
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        return TFSeq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,  # Past key values from model outputs
            decoder_hidden_states=outputs.decoder_hidden_states,  # Decoder hidden states from model outputs
            decoder_attentions=outputs.decoder_attentions,  # Decoder attentions from model outputs
            cross_attentions=outputs.cross_attentions,  # Cross attentions from model outputs
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,  # Encoder last hidden state from encoder outputs
            encoder_hidden_states=outputs.encoder_hidden_states,  # Encoder hidden states from encoder outputs
            encoder_attentions=outputs.encoder_attentions,  # Encoder attentions from encoder outputs
        )

    # Copied from transformers.models.bart.modeling_tf_bart.TFBartForConditionalGeneration.serving_output
    # 定义一个方法，用于生成模型输出的结构化表示
    def serving_output(self, output):
        # 如果配置中启用缓存，则从输出中提取过去键值对的第二个元素作为 pkv，否则为 None
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        # 如果配置中启用输出隐藏状态，则将输出的解码器隐藏状态转换为张量，否则为 None
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置中启用输出注意力权重，则将输出的解码器注意力权重转换为张量，否则为 None
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        # 如果配置中启用输出交叉注意力权重，则将输出的交叉注意力权重转换为张量，否则为 None
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        # 如果配置中启用输出隐藏状态，则将输出的编码器隐藏状态转换为张量，否则为 None
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置中启用输出注意力权重，则将输出的编码器注意力权重转换为张量，否则为 None
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None

        # 返回一个 TFSeq2SeqLMOutput 对象，包括 logits、过去键值对、解码器隐藏状态、解码器注意力权重、
        # 交叉注意力权重、编码器最后隐藏状态、编码器隐藏状态和编码器注意力权重
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
        # 如果 past_key_values 不为 None，则仅保留 decoder_input_ids 的最后一个标记
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        # 如果存在 decoder_attention_mask，则计算其累积位置 IDs 的最后一个值；否则，根据 past_key_values 或 decoder_input_ids 的长度生成位置 IDs
        if decoder_attention_mask is not None:  # xla
            decoder_position_ids = tf.math.cumsum(decoder_attention_mask, axis=-1, exclusive=True)[:, -1:]
        elif past_key_values is not None:  # no xla + past_key_values
            decoder_position_ids = past_key_values[0][0].shape[2]
        else:  # no xla + no past_key_values
            decoder_position_ids = tf.range(decoder_input_ids.shape[1])

        # 返回一个包含模型生成所需输入的字典，包括 input_ids、encoder_outputs、past_key_values、decoder_input_ids、
        # attention_mask、decoder_attention_mask、decoder_position_ids、head_mask、decoder_head_mask、
        # cross_attn_head_mask 和 use_cache 参数
        return {
            "input_ids": None,  # encoder_outputs 已经定义，因此不需要 input_ids
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_position_ids": decoder_position_ids,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # 修改此处以避免缓存（可能用于调试目的）
        }

    # 定义一个方法，根据标签生成解码器的输入 IDs，向右移动标签，并用 pad_token_id 和 decoder_start_token_id 处理填充
    def prepare_decoder_input_ids_from_labels(self, labels: tf.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)
    # 定义神经网络层的构建方法，接受输入形状参数，如果已经构建过则直接返回
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回，不再重复构建
        if self.built:
            return
        # 将标志位设置为已构建
        self.built = True
        # 如果存在嵌套的模型对象，则在命名空间下构建该模型
        if getattr(self, "model", None) is not None:
            with tf.name_scope(self.model.name):
                # 使用空输入形状构建嵌套模型
                self.model.build(None)
        # 如果存在偏置层对象，则在命名空间下构建该偏置层
        if getattr(self, "bias_layer", None) is not None:
            with tf.name_scope(self.bias_layer.name):
                # 使用空输入形状构建偏置层
                self.bias_layer.build(None)
```