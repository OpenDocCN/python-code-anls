# `.\models\led\modeling_tf_led.py`

```py
# coding=utf-8
# 版权声明
#
# 根据 Apache 许可证版本 2.0（"许可证"）授权；除非符合许可证的条款，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件根据"原样"分发，无任何明示或暗示的担保或条件。
# 有关详细信息，请参阅许可证。
""" TF 2.0 LED 模型。"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutputWithPastAndCrossAttentions

# Public API
from ...modeling_tf_utils import (
    TFModelInputType,
    TFPreTrainedModel,
    get_initializer,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_led import LEDConfig

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "allenai/led-base-16384"
_CONFIG_FOR_DOC = "LEDConfig"

LARGE_NEGATIVE = -1e8

# 从 transformers.models.bart.modeling_tf_bart.shift_tokens_right 复制而来
def shift_tokens_right(input_ids: tf.Tensor, pad_token_id: int, decoder_start_token_id: int):
    # 将 pad_token_id 和 decoder_start_token_id 转换为与 input_ids 相同的数据类型
    pad_token_id = tf.cast(pad_token_id, input_ids.dtype)
    decoder_start_token_id = tf.cast(decoder_start_token_id, input_ids.dtype)
    
    # 创建起始 token，形状为 (batch_size, 1)，填充值为 decoder_start_token_id
    start_tokens = tf.fill(
        (shape_list(input_ids)[0], 1),
        tf.convert_to_tensor(decoder_start_token_id, input_ids.dtype)
    )
    
    # 将 input_ids 向右移动一位，将起始 token 放在最前面
    shifted_input_ids = tf.concat([start_tokens, input_ids[:, :-1]], -1)
    
    # 将 labels 中可能的 -100 值替换为 pad_token_id
    shifted_input_ids = tf.where(
        shifted_input_ids == -100,
        tf.fill(shape_list(shifted_input_ids), tf.convert_to_tensor(pad_token_id, input_ids.dtype)),
        shifted_input_ids,
    )
    
    # 断言 shifted_input_ids 中的值都大于等于 0
    assert_gte0 = tf.debugging.assert_greater_equal(shifted_input_ids, tf.constant(0, dtype=input_ids.dtype))
    
    # 确保断言操作被调用，通过在结果中包装一个身份 no-op
    with tf.control_dependencies([assert_gte0]):
        shifted_input_ids = tf.identity(shifted_input_ids)
    
    return shifted_input_ids


# 从 transformers.models.bart.modeling_tf_bart._make_causal_mask 复制而来
def _make_causal_mask(input_ids_shape: tf.TensorShape, past_key_values_length: int = 0):
    """
    # 创建用于双向自注意力的因果掩码。
    """
    # 获取输入张量的批大小
    bsz = input_ids_shape[0]
    # 获取目标长度（通常是序列长度）
    tgt_len = input_ids_shape[1]
    # 创建一个全为负无穷大的张量作为初始掩码
    mask = tf.ones((tgt_len, tgt_len)) * LARGE_NEGATIVE
    # 创建一个条件张量，其值为0到tgt_len-1的序列
    mask_cond = tf.range(shape_list(mask)[-1])

    # 根据条件张量设置掩码的值
    mask = tf.where(mask_cond < tf.reshape(mask_cond + 1, (shape_list(mask)[-1], 1)), 0.0, mask)

    # 如果过去的键值对长度大于0，则在掩码的左侧连接一列零张量
    if past_key_values_length > 0:
        mask = tf.concat([tf.zeros((tgt_len, past_key_values_length)), mask], axis=-1)

    # 使用 tf.tile 对掩码进行扩展，以匹配输入张量的批处理大小和维度
    return tf.tile(mask[None, None, :, :], (bsz, 1, 1, 1))
# 从transformers.models.bart.modeling_tf_bart._expand_mask复制而来的函数
def _expand_mask(mask: tf.Tensor, tgt_len: Optional[int] = None):
    """
    将注意力掩码从`[bsz, seq_len]`扩展到`[bsz, 1, tgt_seq_len, src_seq_len]`。
    """
    # 获取掩码张量的序列长度
    src_len = shape_list(mask)[1]
    # 如果未提供目标长度，则使用源长度作为目标长度
    tgt_len = tgt_len if tgt_len is not None else src_len
    # 创建一个常数张量，值为1.0
    one_cst = tf.constant(1.0)
    # 将掩码张量转换为与one_cst相同的数据类型
    mask = tf.cast(mask, dtype=one_cst.dtype)
    # 在掩码张量的第二维度上进行复制，复制tgt_len次，扩展为`[bsz, 1, tgt_len, src_len]`
    expanded_mask = tf.tile(mask[:, None, None, :], (1, 1, tgt_len, 1))

    # 返回扩展后的掩码张量与一个较大负数相乘的结果
    return (one_cst - expanded_mask) * LARGE_NEGATIVE


class TFLEDLearnedPositionalEmbedding(keras.layers.Embedding):
    """
    该模块学习固定最大大小的位置嵌入。
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, **kwargs):
        super().__init__(num_embeddings, embedding_dim, **kwargs)

    def call(self, input_shape: tf.TensorShape, past_key_values_length: int = 0):
        """
        输入预期为大小为[bsz x seqlen]的张量。
        """
        # 获取输入张量的序列长度
        seq_len = input_shape[1]
        # 创建一个序列长度的范围张量，以1为步长
        position_ids = tf.range(seq_len, delta=1, name="range")
        # 将过去键值对的长度添加到位置ID中
        position_ids += past_key_values_length

        # 调用父类的call方法，传入位置ID张量，并返回结果
        return super().call(tf.cast(position_ids, dtype=tf.int32))


# 从transformers.models.longformer.modeling_tf_longformer.TFLongformerSelfAttention复制而来，将TFLongformer改为TFLEDEncoder
class TFLEDEncoderSelfAttention(keras.layers.Layer):
    # 初始化函数，接受配置、层ID等参数，并调用父类的初始化方法
    def __init__(self, config, layer_id, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 将配置参数保存在对象中
        self.config = config

        # 检查隐藏层大小是否能被注意力头数整除，若不能则抛出数值错误异常
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads}"
            )

        # 初始化对象的属性：注意力头数、每个头的维度、嵌入维度
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size

        # 创建查询、键、值的Dense层，用于自注意力机制
        self.query = keras.layers.Dense(
            self.embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            name="query",
        )
        self.key = keras.layers.Dense(
            self.embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            name="key",
        )
        self.value = keras.layers.Dense(
            self.embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            name="value",
        )

        # 为具有全局注意力的标记创建独立的投影层
        self.query_global = keras.layers.Dense(
            self.embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            name="query_global",
        )
        self.key_global = keras.layers.Dense(
            self.embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            name="key_global",
        )
        self.value_global = keras.layers.Dense(
            self.embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            name="value_global",
        )

        # 创建Dropout层，用于注意力概率的随机丢弃
        self.dropout = keras.layers.Dropout(config.attention_probs_dropout_prob)
        self.global_dropout = keras.layers.Dropout(config.attention_probs_dropout_prob)

        # 将层ID保存在对象中
        self.layer_id = layer_id

        # 获取当前层的注意力窗口大小，并进行断言检查
        attention_window = config.attention_window[self.layer_id]

        assert (
            attention_window % 2 == 0
        ), f"`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}"
        assert (
            attention_window > 0
        ), f"`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}"

        # 计算单侧注意力窗口的大小
        self.one_sided_attn_window_size = attention_window // 2
    # 如果模型尚未构建，则构建查询(query_global)、键(key_global)和值(value_global)的全局作用域
    def build(self, input_shape=None):
        if not self.built:
            # 使用名字作用域创建查询(query_global)的组件
            with tf.name_scope("query_global"):
                self.query_global.build((self.config.hidden_size,))
            # 使用名字作用域创建键(key_global)的组件
            with tf.name_scope("key_global"):
                self.key_global.build((self.config.hidden_size,))
            # 使用名字作用域创建值(value_global)的组件
            with tf.name_scope("value_global"):
                self.value_global.build((self.config.hidden_size,))

        # 如果模型已构建，则直接返回
        if self.built:
            return
        # 设置模型构建状态为已构建
        self.built = True

        # 如果存在查询(query)属性，则使用其名字作用域构建查询组件
        if getattr(self, "query", None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.config.hidden_size])
        # 如果存在键(key)属性，则使用其名字作用域构建键组件
        if getattr(self, "key", None) is not None:
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.config.hidden_size])
        # 如果存在值(value)属性，则使用其名字作用域构建值组件
        if getattr(self, "value", None) is not None:
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.config.hidden_size])
        # 如果存在全局查询(query_global)属性，则使用其名字作用域构建全局查询组件
        if getattr(self, "query_global", None) is not None:
            with tf.name_scope(self.query_global.name):
                self.query_global.build([None, None, self.config.hidden_size])
        # 如果存在全局键(key_global)属性，则使用其名字作用域构建全局键组件
        if getattr(self, "key_global", None) is not None:
            with tf.name_scope(self.key_global.name):
                self.key_global.build([None, None, self.config.hidden_size])
        # 如果存在全局值(value_global)属性，则使用其名字作用域构建全局值组件
        if getattr(self, "value_global", None) is not None:
            with tf.name_scope(self.value_global.name):
                self.value_global.build([None, None, self.config.hidden_size])

    @staticmethod
    def _mask_invalid_locations(input_tensor, window_overlap):
        # 创建正确的上三角形布尔掩码
        mask_2d_upper = tf.reverse(
            tf.linalg.band_part(tf.ones(shape=(window_overlap, window_overlap + 1)), -1, 0),
            axis=[0],
        )

        # 对掩码进行填充以形成完整的矩阵
        padding = tf.convert_to_tensor(
            [[0, shape_list(input_tensor)[1] - window_overlap], [0, shape_list(input_tensor)[3] - window_overlap - 1]]
        )

        # 创建下三角形掩码
        mask_2d = tf.pad(mask_2d_upper, padding)

        # 将下三角形掩码与上三角形掩码合并
        mask_2d = mask_2d + tf.reverse(mask_2d, axis=[0, 1])

        # 将二维掩码扩展到四维矩阵
        mask_4d = tf.tile(mask_2d[None, :, None, :], (shape_list(input_tensor)[0], 1, 1, 1))

        # 用于掩蔽操作的负无穷大张量
        inf_tensor = -float("inf") * tf.ones_like(input_tensor)

        # 执行掩蔽操作
        input_tensor = tf.where(tf.math.greater(mask_4d, 0), inf_tensor, input_tensor)

        return input_tensor
    def _sliding_chunks_matmul_attn_probs_value(self, attn_probs, value, window_overlap):
        """
        Same as _sliding_chunks_query_key_matmul but for attn_probs and value tensors. Returned tensor will be of the
        same shape as `attn_probs`
        """

        # 获取 value 张量的形状信息：batch_size, seq_len, num_heads, head_dim
        batch_size, seq_len, num_heads, head_dim = shape_list(value)

        # 断言条件：seq_len 必须是 2 * window_overlap 的倍数
        tf.debugging.assert_equal(
            seq_len % (window_overlap * 2), 0, message="Seq_len has to be multiple of 2 * window_overlap"
        )
        
        # 断言条件：attn_probs 和 value 张量的前三个维度必须相同（除了 head_dim 维度）
        tf.debugging.assert_equal(
            shape_list(attn_probs)[:3],
            shape_list(value)[:3],
            message="value and attn_probs must have same dims (except head_dim)",
        )
        
        # 断言条件：attn_probs 张量的最后一个维度必须是 2 * window_overlap + 1
        tf.debugging.assert_equal(
            shape_list(attn_probs)[3],
            2 * window_overlap + 1,
            message="attn_probs last dim has to be 2 * window_overlap + 1",
        )

        # 计算 chunk 的数量，每个 chunk 的长度为 window_overlap
        chunks_count = seq_len // window_overlap - 1

        # 将 attn_probs 张量进行转置，并按照一定规则重新组织成 chunked_attn_probs 张量
        chunked_attn_probs = tf.reshape(
            tf.transpose(attn_probs, (0, 2, 1, 3)),
            (
                batch_size * num_heads,
                seq_len // window_overlap,
                window_overlap,
                2 * window_overlap + 1,
            ),
        )

        # 将 value 张量进行转置，并按照一定规则重新组织成 chunked_value 张量
        value = tf.reshape(
            tf.transpose(value, (0, 2, 1, 3)),
            (batch_size * num_heads, seq_len, head_dim),
        )

        # 在 seq_len 的两端各填充 window_overlap 个元素，值为 -1
        paddings = tf.convert_to_tensor([[0, 0], [window_overlap, window_overlap], [0, 0]])
        padded_value = tf.pad(value, paddings, constant_values=-1)

        # 将 padded_value 张量按照一定的窗口大小和跳跃步长进行切片
        frame_size = 3 * window_overlap * head_dim
        frame_hop_size = (shape_list(padded_value)[1] * head_dim - frame_size) // chunks_count
        chunked_value = tf.signal.frame(
            tf.reshape(padded_value, (batch_size * num_heads, -1)),
            frame_size,
            frame_hop_size,
        )
        chunked_value = tf.reshape(
            chunked_value,
            (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim),
        )

        # 断言条件：chunked_value 张量的形状必须是 [batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim]
        tf.debugging.assert_equal(
            shape_list(chunked_value),
            [batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim],
            message="Chunked value has the wrong shape",
        )

        # 对 chunked_attn_probs 和 chunked_value 进行张量乘法操作，得到上下文信息 context
        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)
        context = tf.einsum("bcwd,bcdh->bcwh", chunked_attn_probs, chunked_value)
        
        # 转置和重新组织 context 张量的维度，以符合预期的输出形状
        context = tf.transpose(
            tf.reshape(context, (batch_size, num_heads, seq_len, head_dim)),
            (0, 2, 1, 3),
        )

        # 返回计算得到的上下文 context 张量
        return context
    def _pad_and_transpose_last_two_dims(hidden_states_padded, paddings):
        """
        对最后两个维度进行填充和转置操作。

        Args:
        - hidden_states_padded: 填充后的隐藏状态张量
        - paddings: 填充的尺寸，用于指定在各维度上的填充数量

        Returns:
        - hidden_states_padded: 转置后的隐藏状态张量
        """
        hidden_states_padded = tf.pad(
            hidden_states_padded, paddings
        )  # 填充操作，具体填充的值并不重要，因为之后会被覆写
        batch_size, chunk_size, seq_length, hidden_dim = shape_list(hidden_states_padded)
        hidden_states_padded = tf.reshape(hidden_states_padded, (batch_size, chunk_size, hidden_dim, seq_length))

        return hidden_states_padded

    @staticmethod
    def _pad_and_diagonalize(chunked_hidden_states):
        """
        将每一行向右移动一个步长，将列转换为对角线。

        Args:
        - chunked_hidden_states: 分块的隐藏状态张量，每个块的形状为 (total_num_heads, num_chunks, window_overlap, hidden_dim)

        Returns:
        - chunked_hidden_states: 填充并对角化后的隐藏状态张量
        """
        total_num_heads, num_chunks, window_overlap, hidden_dim = shape_list(chunked_hidden_states)
        paddings = tf.convert_to_tensor([[0, 0], [0, 0], [0, 0], [0, window_overlap + 1]])
        chunked_hidden_states = tf.pad(
            chunked_hidden_states, paddings
        )  # 填充操作，具体填充的值并不重要，因为之后会被覆写
        chunked_hidden_states = tf.reshape(
            chunked_hidden_states, (total_num_heads, num_chunks, -1)
        )  # 将填充后的张量重新形状为 (total_num_heads, num_chunks, window_overlap + hidden_dim + window_overlap + 1)
        chunked_hidden_states = chunked_hidden_states[
            :, :, :-window_overlap
        ]  # 切片操作，去除填充后多余的部分，使得形状为 (total_num_heads, num_chunks, window_overlap + hidden_dim)
        chunked_hidden_states = tf.reshape(
            chunked_hidden_states,
            (total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim),
        )  # 将张量形状重新调整为 (total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim)
        chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]  # 去除最后一个维度的多余部分

        return chunked_hidden_states

    @staticmethod
    def _chunk(hidden_states, window_overlap):
        """convert into overlapping chunks. Chunk size = 2w, overlap size = w"""
        # 获取隐藏状态的形状信息：批量大小、序列长度、隐藏维度
        batch_size, seq_length, hidden_dim = shape_list(hidden_states)
        # 计算输出块的数量，每个块的大小为2w，重叠大小为w
        num_output_chunks = 2 * (seq_length // (2 * window_overlap)) - 1

        # 定义帧大小和帧步长（类似于卷积）
        frame_hop_size = window_overlap * hidden_dim
        frame_size = 2 * frame_hop_size
        # 将隐藏状态重塑为适合帧操作的形状
        hidden_states = tf.reshape(hidden_states, (batch_size, seq_length * hidden_dim))

        # 使用帧操作进行分块，带有重叠部分
        chunked_hidden_states = tf.signal.frame(hidden_states, frame_size, frame_hop_size)

        # 断言确保分块操作正确应用
        tf.debugging.assert_equal(
            shape_list(chunked_hidden_states),
            [batch_size, num_output_chunks, frame_size],
            message=(
                "Make sure chunking is correctly applied. `Chunked hidden states should have output dimension"
                f" {[batch_size, frame_size, num_output_chunks]}, but got {shape_list(chunked_hidden_states)}."
            ),
        )

        # 将分块后的隐藏状态重塑为所需的形状
        chunked_hidden_states = tf.reshape(
            chunked_hidden_states,
            (batch_size, num_output_chunks, 2 * window_overlap, hidden_dim),
        )

        return chunked_hidden_states

    @staticmethod
    def _get_global_attn_indices(is_index_global_attn):
        """compute global attn indices required throughout forward pass"""
        # 计算每个样本中非零全局注意力索引的数量
        num_global_attn_indices = tf.math.count_nonzero(is_index_global_attn, axis=1)
        num_global_attn_indices = tf.cast(num_global_attn_indices, dtype=tf.constant(1).dtype)

        # 批次中全局注意力索引的最大数量
        max_num_global_attn_indices = tf.reduce_max(num_global_attn_indices)

        # 获取所有非零全局注意力索引的位置
        is_index_global_attn_nonzero = tf.where(is_index_global_attn)

        # 创建帮助变量，指示哪些位置是全局注意力索引
        is_local_index_global_attn = tf.range(max_num_global_attn_indices) < tf.expand_dims(
            num_global_attn_indices, axis=-1
        )

        # 获取非填充值在全局注意力索引中的位置
        is_local_index_global_attn_nonzero = tf.where(is_local_index_global_attn)

        # 获取填充值在全局注意力索引中的位置
        is_local_index_no_global_attn_nonzero = tf.where(tf.math.logical_not(is_local_index_global_attn))

        return (
            max_num_global_attn_indices,
            is_index_global_attn_nonzero,
            is_local_index_global_attn_nonzero,
            is_local_index_no_global_attn_nonzero,
        )

    def _concat_with_global_key_attn_probs(
        self,
        attn_scores,
        key_vectors,
        query_vectors,
        max_num_global_attn_indices,
        is_index_global_attn_nonzero,
        is_local_index_global_attn_nonzero,
        is_local_index_no_global_attn_nonzero,
    ):
        batch_size = shape_list(key_vectors)[0]  # 获取key_vectors的批量大小

        # 选择全局key向量
        global_key_vectors = tf.gather_nd(key_vectors, is_index_global_attn_nonzero)

        # 创建仅包含全局key向量的张量
        key_vectors_only_global = tf.scatter_nd(
            is_local_index_global_attn_nonzero,
            global_key_vectors,
            shape=(
                batch_size,
                max_num_global_attn_indices,
                self.num_heads,
                self.head_dim,
            ),
        )

        # 计算来自全局key向量的注意力概率
        # 形状为 (batch_size, seq_len, num_heads, max_num_global_attn_indices)
        attn_probs_from_global_key = tf.einsum("blhd,bshd->blhs", query_vectors, key_vectors_only_global)

        # 转置以匹配形状 (batch_size, max_num_global_attn_indices, seq_len, num_heads)
        attn_probs_from_global_key_trans = tf.transpose(attn_probs_from_global_key, (0, 3, 1, 2))

        # 创建掩码形状
        mask_shape = (shape_list(is_local_index_no_global_attn_nonzero)[0],) + tuple(
            shape_list(attn_probs_from_global_key_trans)[-2:]
        )
        mask = tf.ones(mask_shape) * -10000.0  # 初始化掩码为较大的负数
        mask = tf.cast(mask, dtype=attn_probs_from_global_key_trans.dtype)  # 将掩码转换为与注意力概率相同的数据类型

        # 使用scatter_nd_update方法更新掩码
        attn_probs_from_global_key_trans = tf.tensor_scatter_nd_update(
            attn_probs_from_global_key_trans,
            is_local_index_no_global_attn_nonzero,
            mask,
        )

        # 再次转置以匹配形状 (batch_size, seq_len, num_heads, max_num_global_attn_indices)
        attn_probs_from_global_key = tf.transpose(attn_probs_from_global_key_trans, (0, 2, 3, 1))

        # 连接到注意力分数中
        # 形状为 (batch_size, seq_len, num_heads, extra attention count + 2*window+1)
        attn_scores = tf.concat((attn_probs_from_global_key, attn_scores), axis=-1)

        return attn_scores  # 返回最终的注意力分数张量

    def _compute_attn_output_with_global_indices(
        self,
        value_vectors,
        attn_probs,
        max_num_global_attn_indices,
        is_index_global_attn_nonzero,
        is_local_index_global_attn_nonzero,
    ):
        # 获取注意力概率张量的批量大小
        batch_size = shape_list(attn_probs)[0]

        # 仅保留全局注意力的部分概率值
        attn_probs_only_global = attn_probs[:, :, :, :max_num_global_attn_indices]

        # 根据全局注意力的非零索引，选择全局数值向量
        global_value_vectors = tf.gather_nd(value_vectors, is_index_global_attn_nonzero)

        # 创建仅包含全局数值向量的张量
        value_vectors_only_global = tf.scatter_nd(
            is_local_index_global_attn_nonzero,
            global_value_vectors,
            shape=(
                batch_size,
                max_num_global_attn_indices,
                self.num_heads,
                self.head_dim,
            ),
        )

        # 计算仅含全局注意力的注意力输出
        attn_output_only_global = tf.einsum("blhs,bshd->blhd", attn_probs_only_global, value_vectors_only_global)

        # 重新整形剩余的注意力概率张量
        attn_probs_without_global = attn_probs[:, :, :, max_num_global_attn_indices:]

        # 使用全局和局部注意力计算注意力输出
        attn_output_without_global = self._sliding_chunks_matmul_attn_probs_value(
            attn_probs_without_global, value_vectors, self.one_sided_attn_window_size
        )

        # 返回合并了全局和局部注意力输出的结果
        return attn_output_only_global + attn_output_without_global

    def _compute_global_attn_output_from_hidden(
        self,
        attn_output,
        hidden_states,
        max_num_global_attn_indices,
        layer_head_mask,
        is_local_index_global_attn_nonzero,
        is_index_global_attn_nonzero,
        is_local_index_no_global_attn_nonzero,
        is_index_masked,
        training,
    def reshape_and_transpose(self, vector, batch_size):
        # 将输入向量重新整形并转置，以便进行后续处理
        return tf.reshape(
            tf.transpose(
                tf.reshape(vector, (batch_size, -1, self.num_heads, self.head_dim)),
                (0, 2, 1, 3),
            ),
            (batch_size * self.num_heads, -1, self.head_dim),
        )
class TFLEDEncoderAttention(keras.layers.Layer):
    # 初始化编码器自注意力层
    def __init__(self, config, layer_id, **kwargs):
        super().__init__(**kwargs)
        # 初始化Longformer编码器自注意力层
        self.longformer_self_attn = TFLEDEncoderSelfAttention(config, layer_id=layer_id, name="longformer_self_attn")
        # 输出层，全连接层，输出维度为config中的d_model
        self.output_dense = keras.layers.Dense(config.d_model, use_bias=True, name="output")
        self.config = config

    # 调用函数，用于前向传播
    def call(self, inputs, training=False):
        (
            hidden_states,               # 编码器隐藏状态
            attention_mask,              # 注意力掩码
            layer_head_mask,             # 层头掩码
            is_index_masked,             # 是否对索引进行掩码
            is_index_global_attn,        # 是否全局注意力对索引进行掩码
            is_global_attn,              # 是否全局注意力
        ) = inputs

        # 调用Longformer编码器自注意力层
        self_outputs = self.longformer_self_attn(
            [hidden_states, attention_mask, layer_head_mask, is_index_masked, is_index_global_attn, is_global_attn],
            training=training,
        )

        # 经过输出全连接层
        attention_output = self.output_dense(self_outputs[0], training=training)
        outputs = (attention_output,) + self_outputs[1:]  # 输出结果包括注意力输出和其他信息

        return outputs

    # 构建层，用于初始化层的内部状态
    def build(self, input_shape=None):
        if self.built:  # 如果已经构建过，直接返回
            return
        self.built = True  # 标记已构建
        if getattr(self, "longformer_self_attn", None) is not None:
            with tf.name_scope(self.longformer_self_attn.name):
                self.longformer_self_attn.build(None)  # 构建Longformer编码器自注意力层
        if getattr(self, "output_dense", None) is not None:
            with tf.name_scope(self.output_dense.name):
                self.output_dense.build([None, None, self.config.d_model])  # 构建输出全连接层


class TFLEDDecoderAttention(keras.layers.Layer):
    """Multi-headed attention from "Attention Is All You Need""""

    # 初始化解码器注意力层
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
        self.embed_dim = embed_dim  # 嵌入维度

        self.num_heads = num_heads  # 注意力头数
        self.dropout = keras.layers.Dropout(dropout)  # Dropout层
        self.head_dim = embed_dim // num_heads  # 每个注意力头的维度
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5  # 缩放因子
        self.is_decoder = is_decoder  # 是否为解码器

        # 线性变换层，用于计算K、Q、V以及输出
        self.k_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="k_proj")
        self.q_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="q_proj")
        self.v_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="v_proj")
        self.out_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="out_proj")

    # 对张量进行形状变换
    def _shape(self, tensor: tf.Tensor, seq_len: int, bsz: int):
        return tf.transpose(tf.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))

    # 调用函数，用于前向传播
    def call(
        self,
        hidden_states: tf.Tensor,
        key_value_states: tf.Tensor | None = None,
        past_key_value: Tuple[Tuple[tf.Tensor]] | None = None,
        attention_mask: tf.Tensor | None = None,
        layer_head_mask: tf.Tensor | None = None,
        training=False,
    # 构建函数，用于构建模型的层次结构
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回，避免重复构建
        if self.built:
            return
        # 设置标志位表示已经构建
        self.built = True
        
        # 如果存在 k_proj 属性，则构建 k_proj 层
        if getattr(self, "k_proj", None) is not None:
            # 在命名空间下构建 k_proj 层
            with tf.name_scope(self.k_proj.name):
                self.k_proj.build([None, None, self.embed_dim])
        
        # 如果存在 q_proj 属性，则构建 q_proj 层
        if getattr(self, "q_proj", None) is not None:
            # 在命名空间下构建 q_proj 层
            with tf.name_scope(self.q_proj.name):
                self.q_proj.build([None, None, self.embed_dim])
        
        # 如果存在 v_proj 属性，则构建 v_proj 层
        if getattr(self, "v_proj", None) is not None:
            # 在命名空间下构建 v_proj 层
            with tf.name_scope(self.v_proj.name):
                self.v_proj.build([None, None, self.embed_dim])
        
        # 如果存在 out_proj 属性，则构建 out_proj 层
        if getattr(self, "out_proj", None) is not None:
            # 在命名空间下构建 out_proj 层
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build([None, None, self.embed_dim])
class TFLEDEncoderLayer(keras.layers.Layer):
    # 初始化编码器层，接受配置参数和层编号
    def __init__(self, config: LEDConfig, layer_id: int, **kwargs):
        super().__init__(**kwargs)
        # 设置嵌入维度为模型配置中的维度
        self.embed_dim = config.d_model
        # 初始化自注意力机制
        self.self_attn = TFLEDEncoderAttention(config, layer_id, name="self_attn")
        # 初始化自注意力层规范化
        self.self_attn_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        # 设置dropout层
        self.dropout = keras.layers.Dropout(config.dropout)
        # 获取激活函数
        self.activation_fn = get_tf_activation(config.activation_function)
        # 设置激活函数的dropout层
        self.activation_dropout = keras.layers.Dropout(config.activation_dropout)
        # 第一个全连接层，输出维度为编码器FFN的维度
        self.fc1 = keras.layers.Dense(config.encoder_ffn_dim, name="fc1")
        # 第二个全连接层，输出维度为嵌入维度
        self.fc2 = keras.layers.Dense(self.embed_dim, name="fc2")
        # 最终层规范化
        self.final_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")
        # 保存配置参数
        self.config = config

    # 定义调用方法，处理输入数据和各种掩码
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        layer_head_mask: tf.Tensor,
        is_index_masked: tf.Tensor,
        is_index_global_attn: tf.Tensor,
        is_global_attn: bool,
        training=False,
    ):
        """
        Args:
            hidden_states (`tf.Tensor`): 输入层的张量形状为 *(batch, seq_len, embed_dim)*
            attention_mask (`tf.Tensor`): 注意力掩码的形状为 *(batch, 1, tgt_len, src_len)*，
                其中填充元素由极大负值表示。
            layer_head_mask (`tf.Tensor`): 给定层中注意力头的掩码形状为 *(config.encoder_attention_heads,)*。
        """
        # 保留输入的残差连接
        residual = hidden_states
        # 进行自注意力计算
        layer_outputs = self.self_attn(
            [hidden_states, attention_mask, layer_head_mask, is_index_masked, is_index_global_attn, is_global_attn],
            training=training,
        )
        
        # 获取自注意力层的输出作为新的隐藏状态
        hidden_states = layer_outputs[0]

        # 断言自注意力是否修改了查询的形状
        tf.debugging.assert_equal(
            shape_list(hidden_states),
            shape_list(residual),
            message=f"Self attn modified the shape of query {shape_list(residual)} to {shape_list(hidden_states)}",
        )

        # 应用dropout层到隐藏状态
        hidden_states = self.dropout(hidden_states, training=training)
        # 添加残差连接到dropout后的隐藏状态
        hidden_states = residual + hidden_states
        # 应用自注意力层规范化
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # 保留更新后的残差连接
        residual = hidden_states
        # 应用激活函数到第一个全连接层
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 应用激活函数的dropout层
        hidden_states = self.activation_dropout(hidden_states, training=training)
        # 应用第二个全连接层
        hidden_states = self.fc2(hidden_states)
        # 应用dropout层到第二个全连接层
        hidden_states = self.dropout(hidden_states, training=training)
        # 添加残差连接到dropout后的第二个全连接层
        hidden_states = residual + hidden_states
        # 应用最终层规范化
        hidden_states = self.final_layer_norm(hidden_states)

        # 返回更新后的隐藏状态和其他层输出（如果有）
        return (hidden_states,) + layer_outputs[1:]
    # 构建函数，用于构建神经网络层的结构
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回，避免重复构建
        if self.built:
            return
        # 将构建状态标记为已构建
        self.built = True
        
        # 如果存在 self_attn 属性，构建 self attention 层
        if getattr(self, "self_attn", None) is not None:
            with tf.name_scope(self.self_attn.name):
                self.self_attn.build(None)
        
        # 如果存在 self_attn_layer_norm 属性，构建 self attention 层的 Layer Normalization
        if getattr(self, "self_attn_layer_norm", None) is not None:
            with tf.name_scope(self.self_attn_layer_norm.name):
                self.self_attn_layer_norm.build([None, None, self.embed_dim])
        
        # 如果存在 fc1 属性，构建第一个全连接层
        if getattr(self, "fc1", None) is not None:
            with tf.name_scope(self.fc1.name):
                self.fc1.build([None, None, self.embed_dim])
        
        # 如果存在 fc2 属性，构建第二个全连接层
        if getattr(self, "fc2", None) is not None:
            with tf.name_scope(self.fc2.name):
                self.fc2.build([None, None, self.config.encoder_ffn_dim])
        
        # 如果存在 final_layer_norm 属性，构建最终的 Layer Normalization 层
        if getattr(self, "final_layer_norm", None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.embed_dim])
class TFLEDDecoderLayer(keras.layers.Layer):
    # 定义 TFLED 解码器层，继承自 keras.layers.Layer

    def __init__(self, config: LEDConfig, **kwargs):
        # 初始化函数，接受 LEDConfig 类型的配置参数和其他关键字参数

        super().__init__(**kwargs)
        # 调用父类的初始化方法

        self.embed_dim = config.d_model
        # 设置嵌入维度为配置中的模型维度

        self.self_attn = TFLEDDecoderAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            name="self_attn",
            is_decoder=True,
        )
        # 创建自注意力机制对象，用于解码器自注意力层

        self.dropout = keras.layers.Dropout(config.dropout)
        # 创建 dropout 层，用于整个层的 dropout 操作

        self.activation_fn = get_tf_activation(config.activation_function)
        # 获取 TensorFlow 激活函数对象，根据配置中的激活函数类型

        self.activation_dropout = keras.layers.Dropout(config.activation_dropout)
        # 创建 dropout 层，用于激活函数的 dropout 操作

        self.self_attn_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        # 创建层归一化层，用于自注意力层的归一化

        self.encoder_attn = TFLEDDecoderAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            name="encoder_attn",
            is_decoder=True,
        )
        # 创建编码器注意力对象，用于解码器与编码器之间的注意力

        self.encoder_attn_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="encoder_attn_layer_norm")
        # 创建层归一化层，用于编码器注意力层的归一化

        self.fc1 = keras.layers.Dense(config.decoder_ffn_dim, name="fc1")
        # 创建全连接层，用于解码器的前馈神经网络

        self.fc2 = keras.layers.Dense(self.embed_dim, name="fc2")
        # 创建全连接层，用于解码器的前馈神经网络的输出层

        self.final_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")
        # 创建层归一化层，用于最终输出的归一化

        self.config = config
        # 保存配置对象

    def call(
        self,
        hidden_states,
        attention_mask: tf.Tensor | None = None,
        encoder_hidden_states: tf.Tensor | None = None,
        encoder_attention_mask: tf.Tensor | None = None,
        layer_head_mask: tf.Tensor | None = None,
        encoder_layer_head_mask: tf.Tensor | None = None,
        past_key_value: Tuple[tf.Tensor] | None = None,
        training=False,
        **kwargs
    ):
        # 定义层的调用函数，实现解码器层的前向传播逻辑
    # 构建方法用于构造模型的各个层，如果模型已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        
        # 设置标志表示模型已经构建完成
        self.built = True
        
        # 如果存在 self_attn 层，则构建 self_attn 层，并使用其名称作为命名空间
        if getattr(self, "self_attn", None) is not None:
            with tf.name_scope(self.self_attn.name):
                self.self_attn.build(None)
        
        # 如果存在 self_attn_layer_norm 层，则构建 self_attn_layer_norm 层，
        # 传入的形状是 [None, None, self.embed_dim]
        if getattr(self, "self_attn_layer_norm", None) is not None:
            with tf.name_scope(self.self_attn_layer_norm.name):
                self.self_attn_layer_norm.build([None, None, self.embed_dim])
        
        # 如果存在 encoder_attn 层，则构建 encoder_attn 层，并使用其名称作为命名空间
        if getattr(self, "encoder_attn", None) is not None:
            with tf.name_scope(self.encoder_attn.name):
                self.encoder_attn.build(None)
        
        # 如果存在 encoder_attn_layer_norm 层，则构建 encoder_attn_layer_norm 层，
        # 传入的形状是 [None, None, self.embed_dim]
        if getattr(self, "encoder_attn_layer_norm", None) is not None:
            with tf.name_scope(self.encoder_attn_layer_norm.name):
                self.encoder_attn_layer_norm.build([None, None, self.embed_dim])
        
        # 如果存在 fc1 层，则构建 fc1 层，传入的形状是 [None, None, self.embed_dim]
        if getattr(self, "fc1", None) is not None:
            with tf.name_scope(self.fc1.name):
                self.fc1.build([None, None, self.embed_dim])
        
        # 如果存在 fc2 层，则构建 fc2 层，传入的形状是 [None, None, self.config.decoder_ffn_dim]
        if getattr(self, "fc2", None) is not None:
            with tf.name_scope(self.fc2.name):
                self.fc2.build([None, None, self.config.decoder_ffn_dim])
        
        # 如果存在 final_layer_norm 层，则构建 final_layer_norm 层，
        # 传入的形状是 [None, None, self.embed_dim]
        if getattr(self, "final_layer_norm", None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.embed_dim])
# 定义 TFLEDPreTrainedModel 类，继承自 TFPreTrainedModel
class TFLEDPreTrainedModel(TFPreTrainedModel):
    # 设置配置类为 LEDConfig
    config_class = LEDConfig
    # 指定基础模型前缀为 "led"
    base_model_prefix = "led"

    # 定义 input_signature 属性，用于指定输入的签名
    @property
    def input_signature(self):
        # 调用父类的 input_signature 方法获取默认签名
        sig = super().input_signature
        # 添加全局注意力掩码的 TensorSpec 到签名中，形状为 (None, None)，数据类型为 tf.int32
        sig["global_attention_mask"] = tf.TensorSpec((None, None), tf.int32, name="global_attention_mask")
        # 返回更新后的签名
        return sig


# 使用 dataclass 装饰器定义 TFLEDEncoderBaseModelOutput 类
@dataclass
# 类的注释被省略，这里是 TFLongformerBaseModelOutput 类的修改版本，用于 TFLEDEncoder
class TFLEDEncoderBaseModelOutput(ModelOutput):
    """
    Base class for Longformer's outputs, with potential hidden states, local and global attentions.
    """
    # 定义函数参数 `last_hidden_state`，类型为 `tf.Tensor`，默认为 None
    last_hidden_state: tf.Tensor = None
    
    # 定义函数参数 `hidden_states`，类型为元组 `Tuple[tf.Tensor, ...]` 或者 None，当 `output_hidden_states=True` 时返回
    # 表示模型在每个层输出的隐藏状态以及初始嵌入输出
    hidden_states: Tuple[tf.Tensor, ...] | None = None
    
    # 定义函数参数 `attentions`，类型为元组 `Tuple[tf.Tensor, ...]` 或者 None，当 `output_attentions=True` 时返回
    # 表示每个层的本地注意力权重，形状为 `(batch_size, num_heads, sequence_length, x + attention_window + 1)`
    # 这些是经过注意力 softmax 后的本地注意力权重，用于计算自注意力头中的加权平均值
    attentions: Tuple[tf.Tensor, ...] | None = None
    
    # 定义函数参数 `global_attentions`，类型为元组 `Tuple[tf.Tensor, ...]` 或者 None，当 `output_attentions=True` 时返回
    # 表示每个层的全局注意力权重，形状为 `(batch_size, num_heads, sequence_length, x)`
    # 这些是经过注意力 softmax 后的全局注意力权重，用于计算自注意力头中的加权平均值
    global_attentions: Tuple[tf.Tensor, ...] | None = None
# 定义一个 TFLEDSeq2SeqModelOutput 类，继承自 ModelOutput 类，用于存储序列到序列模型的输出
@dataclass
class TFLEDSeq2SeqModelOutput(ModelOutput):
    """
    Base class for model encoder's outputs that also contains : pre-computed hidden states that can speed up sequential
    decoding.

    """

    # 最后一个隐藏状态，类型为 tf.Tensor，默认为 None
    last_hidden_state: tf.Tensor = None
    # 存储过去关键值的列表，类型为 List[tf.Tensor] 或者 None
    past_key_values: List[tf.Tensor] | None = None
    # 解码器的隐藏状态元组，类型为 Tuple[tf.Tensor, ...] 或者 None
    decoder_hidden_states: Tuple[tf.Tensor, ...] | None = None
    # 解码器的注意力权重元组，类型为 Tuple[tf.Tensor, ...] 或者 None
    decoder_attentions: Tuple[tf.Tensor, ...] | None = None
    # 交叉注意力权重元组，类型为 Tuple[tf.Tensor, ...] 或者 None
    cross_attentions: Tuple[tf.Tensor, ...] | None = None
    # 编码器最后一个隐藏状态，类型为 tf.Tensor 或者 None
    encoder_last_hidden_state: tf.Tensor | None = None
    # 编码器的隐藏状态元组，类型为 Tuple[tf.Tensor, ...] 或者 None
    encoder_hidden_states: Tuple[tf.Tensor, ...] | None = None
    # 编码器的注意力权重元组，类型为 Tuple[tf.Tensor, ...] 或者 None
    encoder_attentions: Tuple[tf.Tensor, ...] | None = None
    # 编码器的全局注意力权重元组，类型为 Tuple[tf.Tensor, ...] 或者 None
    encoder_global_attentions: Tuple[tf.Tensor, ...] | None = None


# 定义一个 TFLEDSeq2SeqLMOutput 类，继承自 ModelOutput 类，用于存储序列到序列语言模型的输出
@dataclass
class TFLEDSeq2SeqLMOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    """

    # 损失张量，类型为 tf.Tensor 或者 None
    loss: tf.Tensor | None = None
    # 预测的 logits 张量，类型为 tf.Tensor，默认为 None
    logits: tf.Tensor = None
    # 存储过去关键值的列表，类型为 List[tf.Tensor] 或者 None
    past_key_values: List[tf.Tensor] | None = None
    # 解码器的隐藏状态元组，类型为 Tuple[tf.Tensor, ...] 或者 None
    decoder_hidden_states: Tuple[tf.Tensor, ...] | None = None
    # 解码器的注意力权重元组，类型为 Tuple[tf.Tensor, ...] 或者 None
    decoder_attentions: Tuple[tf.Tensor, ...] | None = None
    # 交叉注意力权重元组，类型为 Tuple[tf.Tensor, ...] 或者 None
    cross_attentions: Tuple[tf.Tensor, ...] | None = None
    # 编码器最后一个隐藏状态，类型为 tf.Tensor 或者 None
    encoder_last_hidden_state: tf.Tensor | None = None
    # 编码器的隐藏状态元组，类型为 Tuple[tf.Tensor, ...] 或者 None
    encoder_hidden_states: Tuple[tf.Tensor, ...] | None = None
    # 编码器的注意力权重元组，类型为 Tuple[tf.Tensor, ...] 或者 None
    encoder_attentions: Tuple[tf.Tensor, ...] | None = None
    # 编码器的全局注意力权重元组，类型为 Tuple[tf.Tensor, ...] 或者 None
    encoder_global_attentions: Tuple[tf.Tensor, ...] | None = None


# LED_START_DOCSTRING 为一个原始字符串，用于描述 TFPreTrainedModel 类的文档字符串
LED_START_DOCSTRING = r"""
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
    - 在调用模型时，可以使用这种形式传入输入张量。如果模型有不同的输入名称（比如input_ids、attention_mask、token_type_ids），则需要按照相应的输入名称传递张量。

    `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`
    - 如果模型的输入需要按照名称显式传递，则可以使用这种字典形式传递输入张量，其中键对应于模型的输入名称，值对应于输入张量本身。

    Note that when creating models and layers with
    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don't need to worry
    about any of this, as you can just pass inputs like you would to any other Python function!
    - 如果使用子类化的方式创建模型和层，那么可以像调用任何其他Python函数一样传递输入张量，无需担心输入的名称和形式。

    Args:
        config ([`LEDConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
    - 参数说明部分，config参数接受一个`LEDConfig`类型的对象，该对象包含模型的所有参数配置。使用配置文件初始化时，并不会加载模型的权重，只会加载配置信息。可以查阅[`~TFPreTrainedModel.from_pretrained`]方法来加载模型的权重。
"""
LED_INPUTS_DOCSTRING = r"""
"""

@keras_serializable
class TFLEDEncoder(keras.layers.Layer):
    # 设置配置类为LEDConfig
    config_class = LEDConfig
    """
    Transformer encoder consisting of *config.encoder_layers* self-attention layers. Each layer is a
    [`TFLEDEncoderLayer`].

    Args:
        config: LEDConfig
    """

    def __init__(self, config: LEDConfig, embed_tokens: Optional[keras.layers.Embedding] = None, **kwargs):
        super().__init__(**kwargs)
        # 初始化配置参数
        self.config = config
        # 设置dropout层
        self.dropout = keras.layers.Dropout(config.dropout)
        # 如果启用了encoder_layerdrop，则记录警告信息
        if config.encoder_layerdrop > 0:
            logger.warning("Layerdrop is currently disabled in TFLED models.")
        # 设置layerdrop为0.0
        self.layerdrop = 0.0
        # 设置padding索引为config.pad_token_id
        self.padding_idx = config.pad_token_id

        # 如果config.attention_window为整数，则确认其为偶数且为正数，并复制给每个层
        if isinstance(config.attention_window, int):
            assert config.attention_window % 2 == 0, "`config.attention_window` has to be an even value"
            assert config.attention_window > 0, "`config.attention_window` has to be positive"
            config.attention_window = [config.attention_window] * config.num_hidden_layers  # one value per layer
        else:
            # 否则确认其长度与num_hidden_layers相等
            assert len(config.attention_window) == config.num_hidden_layers, (
                "`len(config.attention_window)` should equal `config.num_hidden_layers`. "
                f"Expected {config.num_hidden_layers}, given {len(config.attention_window)}"
            )

        # 设置attention_window为config.attention_window
        self.attention_window = config.attention_window
        # 设置embed_tokens为输入的embed_tokens
        self.embed_tokens = embed_tokens
        # 初始化位置编码层TFLEDLearnedPositionalEmbedding
        self.embed_positions = TFLEDLearnedPositionalEmbedding(
            config.max_encoder_position_embeddings,
            config.d_model,
            name="embed_positions",
        )
        # 创建transformer encoder层列表，每一层为TFLEDEncoderLayer
        self.layers = [TFLEDEncoderLayer(config, i, name=f"layers.{i}") for i in range(config.encoder_layers)]
        # 创建layernorm层
        self.layernorm_embedding = keras.layers.LayerNormalization(epsilon=1e-5, name="layernorm_embedding")
        # 设置embed_dim为config.d_model
        self.embed_dim = config.d_model

    # 获取embed_tokens方法
    def get_embed_tokens(self):
        return self.embed_tokens

    # 设置embed_tokens方法
    def set_embed_tokens(self, embed_tokens):
        self.embed_tokens = embed_tokens

    # 解包输入的装饰器函数
    @unpack_inputs
    # 定义call函数，处理Transformer编码器的前向传播
    def call(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        global_attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
    ):
        # 实际前向传播逻辑在TFLEDEncoderLayer中实现，这里仅定义函数签名和参数

    # 计算隐藏状态的函数，截取hidden_states以适配padding长度
    @tf.function
    def compute_hidden_states(self, hidden_states, padding_len):
        return hidden_states[:, :-padding_len] if padding_len > 0 else hidden_states

    # 填充到指定窗口大小的函数，处理输入以匹配指定的注意力窗口大小
    def _pad_to_window_size(
        self,
        input_ids,
        attention_mask,
        inputs_embeds,
        pad_token_id,
    ):
        """A helper function to pad tokens and mask to work with implementation of Longformer selfattention."""
        # padding
        attention_window = (
            self.attention_window if isinstance(self.attention_window, int) else max(self.attention_window)
        )

        assert attention_window % 2 == 0, f"`attention_window` should be an even value. Given {attention_window}"

        input_shape = shape_list(input_ids) if input_ids is not None else shape_list(inputs_embeds)
        batch_size, seq_len = input_shape[:2]
        padding_len = (attention_window - seq_len % attention_window) % attention_window

        if padding_len > 0:
            logger.warning_once(
                f"Input ids are automatically padded from {seq_len} to {seq_len + padding_len} to be a multiple of "
                f"`config.attention_window`: {attention_window}"
            )

        paddings = tf.convert_to_tensor([[0, 0], [0, padding_len]])

        if input_ids is not None:
            # Pad input_ids with pad_token_id according to calculated padding_len
            input_ids = tf.pad(input_ids, paddings, constant_values=pad_token_id)

        if inputs_embeds is not None:
            if padding_len > 0:
                # Create padding for input_ids and embed them to get inputs_embeds_padding
                input_ids_padding = tf.fill((batch_size, padding_len), pad_token_id)
                inputs_embeds_padding = self.embed_tokens(input_ids_padding)
                # Concatenate original inputs_embeds with inputs_embeds_padding along the sequence dimension
                inputs_embeds = tf.concat([inputs_embeds, inputs_embeds_padding], axis=-2)

        # Pad attention_mask with False (indicating no attention on padding tokens)
        attention_mask = tf.pad(attention_mask, paddings, constant_values=False)

        return (
            padding_len,       # Amount of padding added to input_ids and attention_mask
            input_ids,         # Padded input_ids
            attention_mask,    # Padded attention_mask
            inputs_embeds,     # Padded inputs_embeds
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "embed_positions", None) is not None:
            with tf.name_scope(self.embed_positions.name):
                self.embed_positions.build(None)
        if getattr(self, "layernorm_embedding", None) is not None:
            with tf.name_scope(self.layernorm_embedding.name):
                self.layernorm_embedding.build([None, None, self.embed_dim])
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)
# 使用 @keras_serializable 装饰器，将 TFLEDDecoder 类标记为可序列化的 Keras 层
@keras_serializable
class TFLEDDecoder(keras.layers.Layer):
    # 指定配置类为 LEDConfig
    config_class = LEDConfig
    
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`TFLEDDecoderLayer`]

    Args:
        config: LEDConfig
        embed_tokens: output embedding
    """

    # 初始化方法，接受 LEDConfig 和可选的嵌入标记作为参数
    def __init__(self, config: LEDConfig, embed_tokens: Optional[keras.layers.Embedding] = None, **kwargs):
        super().__init__(**kwargs)
        # 设置配置
        self.config = config
        # 设置填充索引为配置中的 pad_token_id
        self.padding_idx = config.pad_token_id
        # 设置嵌入标记
        self.embed_tokens = embed_tokens
        # 如果配置中启用了 layerdrop，则发出警告（当前未启用）
        if config.decoder_layerdrop > 0:
            logger.warning("Layerdrop is currently disabled in TFLED models.")
        # 设置 layerdrop 为 0.0
        self.layerdrop = 0.0
        # 创建位置嵌入层对象 TFLEDLearnedPositionalEmbedding
        self.embed_positions = TFLEDLearnedPositionalEmbedding(
            config.max_decoder_position_embeddings,
            config.d_model,
            name="embed_positions",
        )
        # 创建多个 TFLEDDecoderLayer 层对象组成的列表
        self.layers = [TFLEDDecoderLayer(config, name=f"layers.{i}") for i in range(config.decoder_layers)]
        # 创建层标准化层对象，用于嵌入层标准化处理
        self.layernorm_embedding = keras.layers.LayerNormalization(epsilon=1e-5, name="layernorm_embedding")

        # 创建丢弃层，使用配置中的 dropout 比率
        self.dropout = keras.layers.Dropout(config.dropout)

    # 设置嵌入标记的方法
    def set_embed_tokens(self, embed_tokens):
        self.embed_tokens = embed_tokens

    # 使用 @unpack_inputs 装饰器定义的调用方法，接受多个输入参数并返回结果
    def call(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        encoder_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
    ):
        # 方法实现略过，未提供具体实现

    # 构建方法，用于构建层的内部组件
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        
        # 如果存在 embed_positions 属性，则构建该对象
        if getattr(self, "embed_positions", None) is not None:
            with tf.name_scope(self.embed_positions.name):
                self.embed_positions.build(None)
        
        # 如果存在 layernorm_embedding 属性，则构建该对象
        if getattr(self, "layernorm_embedding", None) is not None:
            with tf.name_scope(self.layernorm_embedding.name):
                self.layernorm_embedding.build([None, None, self.config.d_model])
        
        # 遍历每个层并构建它们
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)
    # 初始化函数，用于创建一个新的LED模型实例
    def __init__(self, config: LEDConfig, **kwargs):
        # 调用父类的初始化方法，继承父类的属性和方法
        super().__init__(**kwargs)
        # 将传入的配置参数保存到实例属性中
        self.config = config
        # 创建一个共享的嵌入层，用于编码和解码器共享词嵌入
        self.shared = keras.layers.Embedding(
            input_dim=config.vocab_size,  # 词汇表大小，词嵌入的输入维度
            output_dim=config.d_model,     # 嵌入向量的输出维度
            embeddings_initializer=keras.initializers.TruncatedNormal(stddev=self.config.init_std),  # 初始化嵌入矩阵的方式
            name="led.shared",  # 嵌入层的名称
        )
        # 为共享的嵌入层添加额外属性，指定层的预期名称范围（用于加载/存储权重）
        self.shared.load_weight_prefix = "led.shared"

        # 创建LED模型的编码器，传入配置和共享的词嵌入层
        self.encoder = TFLEDEncoder(config, self.shared, name="encoder")
        # 创建LED模型的解码器，传入配置和共享的词嵌入层
        self.decoder = TFLEDDecoder(config, self.shared, name="decoder")

    # 返回模型的输入嵌入层（共享的词嵌入层）
    def get_input_embeddings(self):
        return self.shared

    # 设置模型的输入嵌入层为新的词嵌入层
    def set_input_embeddings(self, new_embeddings):
        # 更新共享的词嵌入层
        self.shared = new_embeddings
        # 更新编码器的词嵌入层
        self.encoder.embed_tokens = self.shared
        # 更新解码器的词嵌入层
        self.decoder.embed_tokens = self.shared

    # 使用装饰器将输入参数解包，处理模型的前向传播
    @unpack_inputs
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs: Optional[Union[Tuple, TFLEDEncoderBaseModelOutput]] = None,
        global_attention_mask=None,
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
            # 如果没有提供解码器的输入 ID 和嵌入向量，则不使用缓存
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                use_cache = False

            # 如果没有提供编码器的输出，则调用编码器来生成编码器的输出
            if encoder_outputs is None:
                encoder_outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    global_attention_mask=global_attention_mask,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    training=training,
                )
            # 如果用户传入了元组形式的编码器输出，并且设置了 return_dict=True，则将其包装在 TFLEDEncoderBaseModelOutput 中
            elif return_dict and not isinstance(encoder_outputs, TFLEDEncoderBaseModelOutput):
                encoder_outputs = TFLEDEncoderBaseModelOutput(
                    last_hidden_state=encoder_outputs[0],
                    hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                    attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                )
            # 如果用户传入了 TFLEDEncoderBaseModelOutput 形式的编码器输出，并且设置了 return_dict=False，则将其转换为元组形式
            elif not return_dict and not isinstance(encoder_outputs, tuple):
                encoder_outputs = encoder_outputs.to_tuple()

            # 调用解码器生成解码器的输出
            decoder_outputs = self.decoder(
                decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_outputs[0],
                encoder_attention_mask=attention_mask,
                head_mask=decoder_head_mask,
                encoder_head_mask=head_mask,
                past_key_values=past_key_values,
                inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                training=training,
            )

            # 如果 return_dict=False，则返回解码器和编码器的输出作为元组
            if not return_dict:
                return decoder_outputs + encoder_outputs

            # 如果 return_dict=True，则将解码器和编码器的输出包装在 TFLEDSeq2SeqModelOutput 中并返回
            return TFLEDSeq2SeqModelOutput(
                last_hidden_state=decoder_outputs.last_hidden_state,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
                encoder_global_attentions=encoder_outputs.global_attentions,
            )
    # 如果模型已经构建完成，则直接返回，不重复构建
    if self.built:
        return
    # 设置模型已经构建标志为True
    self.built = True
    
    # 共享/绑定的权重期望位于模型基础命名空间中
    # 将"/"添加到tf.name_scope的末尾（而不是开头！）将其放置在根命名空间而不是当前命名空间中
    with tf.name_scope(self.shared.load_weight_prefix + "/" + self.shared.name + "/"):
        # 构建共享模型，无输入形状
        self.shared.build(None)
    
    # 如果存在编码器对象
    if getattr(self, "encoder", None) is not None:
        # 使用编码器名称创建命名空间
        with tf.name_scope(self.encoder.name):
            # 构建编码器，无输入形状
            self.encoder.build(None)
    
    # 如果存在解码器对象
    if getattr(self, "decoder", None) is not None:
        # 使用解码器名称创建命名空间
        with tf.name_scope(self.decoder.name):
            # 构建解码器，无输入形状
            self.decoder.build(None)
# 添加文档字符串，说明这是一个不带顶部特定头的裸 LED 模型输出原始隐藏状态。
# 使用 TFLEDPreTrainedModel 的子类化来定义 TFLEDModel 类
class TFLEDModel(TFLEDPreTrainedModel):
    
    # 初始化方法，接受配置和其他参数
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)
        
        # 创建 TFLEDMainLayer 实例，并命名为 "led"
        self.led = TFLEDMainLayer(config, name="led")

    # 返回编码器的方法
    def get_encoder(self):
        return self.led.encoder

    # 返回解码器的方法
    def get_decoder(self):
        return self.led.decoder

    # call 方法，定义模型的前向传播过程
    @unpack_inputs
    # 添加文档字符串，描述输入的格式要求
    @add_start_docstrings_to_model_forward(LED_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 添加代码示例的文档字符串，指向预训练模型的检查点、输出类型和配置类
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFLEDSeq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 方法签名和参数描述，指定了输入和输出的类型及格式
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: tf.Tensor | None = None,
        decoder_input_ids: tf.Tensor | None = None,
        decoder_attention_mask: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        decoder_head_mask: tf.Tensor | None = None,
        encoder_outputs: tf.Tensor | None = None,
        global_attention_mask: tf.Tensor | None = None,
        past_key_values: Tuple[Tuple[tf.Tensor]] | None = None,
        inputs_embeds: tf.Tensor | None = None,
        decoder_inputs_embeds: tf.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        training: bool = False,
        **kwargs,
    ) -> Tuple[tf.Tensor] | TFLEDSeq2SeqModelOutput:
        # 调用 TFLEDMainLayer 实例的__call__方法，传递参数并接收输出
        outputs = self.led(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
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
    # 定义一个方法用于处理模型的输出，接受一个output对象作为参数
    def serving_output(self, output):
        # 如果配置中设置使用缓存，则从output的过去键值对中获取第二个元素作为pkv，否则为None
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        # 如果配置中设置输出隐藏状态，则将output的解码器隐藏状态转换为张量dec_hs，否则为None
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置中设置输出注意力，则将output的解码器注意力转换为张量dec_attns，否则为None
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        # 如果配置中设置输出注意力，则将output的交叉注意力转换为张量cross_attns，否则为None
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        # 如果配置中设置输出隐藏状态，则将output的编码器隐藏状态转换为张量enc_hs，否则为None
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置中设置输出注意力，则将output的编码器注意力转换为张量enc_attns，否则为None
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None
        # 如果配置中设置输出注意力，则将output的全局编码器注意力转换为张量enc_g_attns，否则为None
        enc_g_attns = tf.convert_to_tensor(output.encoder_global_attentions) if self.config.output_attentions else None

        # 返回一个TFLEDSeq2SeqModelOutput对象，包含处理后的各种张量
        return TFLEDSeq2SeqModelOutput(
            last_hidden_state=output.last_hidden_state,
            past_key_values=pkv,
            decoder_hidden_states=dec_hs,
            decoder_attentions=dec_attns,
            cross_attentions=cross_attns,
            encoder_last_hidden_state=output.encoder_last_hidden_state,
            encoder_hidden_states=enc_hs,
            encoder_attentions=enc_attns,
            encoder_global_attentions=enc_g_attns,
        )

    # 定义一个方法用于构建模型，接受一个输入形状参数，默认为None
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果对象中存在led属性
        if getattr(self, "led", None) is not None:
            # 在led的名字作用域内构建led对象，传入None作为构建参数
            with tf.name_scope(self.led.name):
                self.led.build(None)
# Copied from transformers.models.bart.modeling_tf_bart.BiasLayer
# BiasLayer 类的定义，用于添加偏置作为一个层。用于序列化目的：`keras.Model.save_weights` 按层存储权重，
# 因此所有权重都必须在一个层中注册。
class BiasLayer(keras.layers.Layer):
    """
    Bias as a layer. It is used for serialization purposes: `keras.Model.save_weights` stores on a per-layer basis,
    so all weights have to be registered in a layer.
    """

    def __init__(self, shape, initializer, trainable, name, **kwargs):
        super().__init__(name=name, **kwargs)
        # 注：当序列化时，此变量的名称不会被作用域化，即不会以“outer_layer/inner_layer/.../name:0”的格式。
        # 而是“name:0”。更多细节见：
        # https://github.com/huggingface/transformers/pull/18833#issuecomment-1233090214
        # 添加一个权重作为偏置，具有给定的形状、初始化器和是否可训练的参数。
        self.bias = self.add_weight(name=name, shape=shape, initializer=initializer, trainable=trainable)

    def call(self, x):
        # 在调用时，返回输入张量 x 加上偏置 self.bias
        return x + self.bias


@add_start_docstrings(
    "The LED Model with a language modeling head. Can be used for summarization.",
    LED_START_DOCSTRING,
)
# TFLEDForConditionalGeneration 类继承自 TFLEDPreTrainedModel，表示带有语言建模头部的 LED 模型，可用于摘要生成。
class TFLEDForConditionalGeneration(TFLEDPreTrainedModel):
    # 在加载时忽略的键列表，用于不期望的项
    _keys_to_ignore_on_load_unexpected = [
        r"led.encoder.embed_tokens.weight",
        r"led.decoder.embed_tokens.weight",
    ]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 初始化 LED 主层，用给定的配置，并命名为 "led"
        self.led = TFLEDMainLayer(config, name="led")
        # 是否使用缓存，从配置中获取
        self.use_cache = config.use_cache
        # final_bias_logits 在 PyTorch 中作为缓冲区注册，为保持一致性，设为不可训练。
        # 创建一个 BiasLayer 实例作为 final_logits_bias，形状为 [1, vocab_size]，初始化为零。
        self.bias_layer = BiasLayer(
            name="final_logits_bias", shape=[1, config.vocab_size], initializer="zeros", trainable=False
        )

        # TODO (Joao): investigate why LED has numerical issues in XLA generate
        # 是否支持 XLA 生成，默认为 False，需进一步调查为何在 XLA 生成中 LED 存在数值问题。
        self.supports_xla_generation = False

    # 获取解码器
    def get_decoder(self):
        return self.led.decoder

    # 获取编码器
    def get_encoder(self):
        return self.led.encoder

    # 获取偏置信息，返回包含 final_logits_bias 偏置的字典
    def get_bias(self):
        return {"final_logits_bias": self.bias_layer.bias}

    # 设置偏置，替换包含偏置的现有层以正确（反）序列化
    def set_bias(self, value):
        # 获取词汇表大小
        vocab_size = value["final_logits_bias"].shape[-1]
        # 创建一个 BiasLayer 实例作为 final_logits_bias，形状为 [1, vocab_size]，初始化为零，且不可训练。
        self.bias_layer = BiasLayer(
            name="final_logits_bias", shape=[1, vocab_size], initializer="zeros", trainable=False
        )
        # 将给定的偏置值赋给 self.bias_layer.bias
        self.bias_layer.bias.assign(value["final_logits_bias"])

    # 获取输出的嵌入层
    def get_output_embeddings(self):
        return self.get_input_embeddings()

    # 设置输出的嵌入层
    def set_output_embeddings(self, value):
        self.set_input_embeddings(value)

    # 对模型前向方法添加开始文档字符串，详见 LED_INPUTS_DOCSTRING，并替换返回值的文档字符串为 TFLEDSeq2SeqLMOutput
    # 用于 API 文档生成。
    @unpack_inputs
    @add_start_docstrings_to_model_forward(LED_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFLEDSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    # 定义一个方法，用于调用模型
    def call(
        # 输入序列的 token IDs，可以是 TFModelInputType 类型或者 None
        self,
        input_ids: TFModelInputType | None = None,
        # 注意力掩码，可以是 numpy 数组、张量或者 None
        attention_mask: np.ndarray | tf.Tensor | None = None,
        # 解码器输入的 token IDs，可以是 numpy 数组、张量或者 None
        decoder_input_ids: np.ndarray | tf.Tensor | None = None,
        # 解码器的注意力掩码，可以是 numpy 数组、张量或者 None
        decoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        # 头部掩码，可以是 numpy 数组、张量或者 None
        head_mask: np.ndarray | tf.Tensor | None = None,
        # 解码器的头部掩码，可以是 numpy 数组、张量或者 None
        decoder_head_mask: np.ndarray | tf.Tensor | None = None,
        # 编码器的输出，可以是 TFLEDEncoderBaseModelOutput 类型或者 None
        encoder_outputs: TFLEDEncoderBaseModelOutput | None = None,
        # 全局注意力掩码，可以是 numpy 数组、张量或者 None
        global_attention_mask: np.ndarray | tf.Tensor | None = None,
        # 历史键值对，类型为 Tuple[Tuple[Union[np.ndarray, tf.Tensor]]] 或者 None
        past_key_values: Tuple[Tuple[Union[np.ndarray, tf.Tensor]]] | None = None,
        # 输入嵌入，可以是 numpy 数组、张量或者 None
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        # 解码器的输入嵌入，可以是 numpy 数组、张量或者 None
        decoder_inputs_embeds: np.ndarray | tf.Tensor | None = None,
        # 是否使用缓存，布尔类型或者 None
        use_cache: bool | None = None,
        # 是否输出注意力，布尔类型或者 None
        output_attentions: bool | None = None,
        # 是否输出隐藏状态，布尔类型或者 None
        output_hidden_states: bool | None = None,
        # 是否返回字典形式的结果，布尔类型或者 None
        return_dict: bool | None = None,
        # 标签，张量类型或者 None
        labels: tf.Tensor | None = None,
        # 是否处于训练模式，布尔类型，默认为 False
        training: bool = False,
    ) -> Tuple[tf.Tensor] | TFLEDSeq2SeqLMOutput:
        """
        返回一个元组，包含 tf.Tensor 和 TFLEDSeq2SeqLMOutput 类型的对象。

        如果 labels 不为 None：
            设置 use_cache 为 False
            如果 decoder_input_ids 和 decoder_inputs_embeds 都为 None：
                使用 shift_tokens_right 函数将 labels 右移，并设置填充和解码起始令牌的 ID

        使用 self.led 方法处理以下参数：
            input_ids: 输入的 token IDs
            attention_mask: 注意力掩码
            decoder_input_ids: 解码器输入的 token IDs
            decoder_attention_mask: 解码器注意力掩码
            encoder_outputs: 编码器输出
            global_attention_mask: 全局注意力掩码
            head_mask: 头部注意力掩码
            decoder_head_mask: 解码器头部注意力掩码
            past_key_values: 过去的键值对
            inputs_embeds: 输入的嵌入向量
            decoder_inputs_embeds: 解码器输入的嵌入向量
            use_cache: 是否使用缓存
            output_attentions: 是否输出注意力权重
            output_hidden_states: 是否输出隐藏状态
            return_dict: 是否返回字典
            training: 是否训练模式

        计算 lm_logits：
            使用 self.led.shared.weights 对 outputs[0] 进行矩阵乘法，转置部分

        将 lm_logits 传递给 self.bias_layer 进行处理

        计算 masked_lm_loss：
            如果 labels 为 None，则 masked_lm_loss 为 None，否则调用 self.hf_compute_loss 计算损失

        如果 return_dict 为 False：
            组装输出元组 output，包括 lm_logits 和 outputs 的其余部分

            如果 masked_lm_loss 不为 None，则将其包含在输出中

            返回 output

        否则，以 TFLEDSeq2SeqLMOutput 对象的形式返回：
            loss: masked_lm_loss
            logits: lm_logits
            past_key_values: outputs 中的 past_key_values（索引为 1）
            decoder_hidden_states: outputs 中的 decoder_hidden_states（索引为 2）
            decoder_attentions: outputs 中的 decoder_attentions（索引为 3）
            cross_attentions: outputs 中的 cross_attentions（索引为 4）
            encoder_last_hidden_state: encoder_outputs 中的 encoder_last_hidden_state（索引为 0）
            encoder_hidden_states: encoder_outputs 中的 encoder_hidden_states（索引为 1）
            encoder_attentions: encoder_outputs 中的 encoder_attentions（索引为 2）
            encoder_global_attentions: encoder_global_attentions

        """
    # 定义一个方法用于生成模型的输出
    def serving_output(self, output):
        # 如果配置要求使用缓存，则提取输出中的过去键值（past_key_values）
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        # 如果配置要求输出隐藏状态，则将输出的解码器隐藏状态转换为张量
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置要求输出注意力权重，则将输出的解码器注意力权重转换为张量
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        # 如果配置要求输出交叉注意力权重，则将输出的交叉注意力权重转换为张量
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        # 如果配置要求输出隐藏状态，则将输出的编码器隐藏状态转换为张量
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置要求输出注意力权重，则将输出的编码器注意力权重转换为张量
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None
        # 如果配置要求输出全局编码器注意力权重，则将输出的全局编码器注意力权重转换为张量
        enc_g_attns = tf.convert_to_tensor(output.encoder_global_attentions) if self.config.output_attentions else None

        # 返回一个包含输出结果的 TFLEDSeq2SeqLMOutput 对象
        return TFLEDSeq2SeqLMOutput(
            logits=output.logits,
            past_key_values=pkv,
            decoder_hidden_states=dec_hs,
            decoder_attentions=dec_attns,
            cross_attentions=cross_attns,
            encoder_last_hidden_state=output.encoder_last_hidden_state,
            encoder_hidden_states=enc_hs,
            encoder_attentions=enc_attns,
            encoder_global_attentions=enc_g_attns,
        )

    # 定义一个方法，准备生成时的输入参数
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # 如果 past_key_values 不为 None，则截断 decoder_input_ids，只保留最后一个 token
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        # 返回一个字典，包含生成时需要的输入参数
        return {
            "input_ids": None,  # encoder_outputs 已经定义，不需要 input_ids
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "use_cache": use_cache,  # 更改此参数以避免缓存（推测是为了调试）
        }

    # 定义一个方法，从标签生成解码器的输入 token ids
    def prepare_decoder_input_ids_from_labels(self, labels: tf.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)
    def hf_compute_loss(self, labels, logits):
        """计算跨熵损失，忽略填充标记"""
        # 使用稀疏分类交叉熵损失函数，设置为从 logits 计算，不进行损失值缩减
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=keras.losses.Reduction.NONE)
        
        # 如果配置为使用旧版本 TensorFlow 的损失计算方式
        if self.config.tf_legacy_loss:
            # 将标签展平为一维张量
            melted_labels = tf.reshape(labels, (-1,))
            # 创建活跃损失掩码，排除填充标记
            active_loss = tf.not_equal(melted_labels, self.config.pad_token_id)
            # 使用掩码从 logits 中提取有效值
            reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, shape_list(logits)[2])), active_loss)
            # 使用掩码从标签中提取有效标签
            labels = tf.boolean_mask(melted_labels, active_loss)
            return loss_fn(labels, reduced_logits)

        # 在此处将负标签裁剪为零，以避免 NaN 和错误 - 这些位置将在后续被掩码处理
        unmasked_loss = loss_fn(tf.nn.relu(labels), logits)
        # 确保只有非填充标签影响损失
        loss_mask = tf.cast(labels != self.config.pad_token_id, dtype=unmasked_loss.dtype)
        # 应用损失掩码到未掩码的损失上
        masked_loss = unmasked_loss * loss_mask
        # 计算掩码后的损失的均值
        reduced_masked_loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(loss_mask)
        # 返回形状为 (1,) 的降维后的损失张量
        return tf.reshape(reduced_masked_loss, (1,))

    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 标记已构建
        self.built = True
        
        # 如果存在 LED 层，则构建 LED 层
        if getattr(self, "led", None) is not None:
            with tf.name_scope(self.led.name):
                self.led.build(None)
        
        # 如果存在偏置层，则构建偏置层
        if getattr(self, "bias_layer", None) is not None:
            with tf.name_scope(self.bias_layer.name):
                self.bias_layer.build(None)
```