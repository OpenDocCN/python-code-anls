# `.\transformers\models\led\modeling_tf_led.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" TF 2.0 LED model."""

# 导入必要的库
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
# 导入相关模块
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutputWithPastAndCrossAttentions
# 导入公共API
from ...modeling_tf_utils import (
    TFModelInputType,
    TFPreTrainedModel,
    get_initializer,
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

# 获取日志记录器
logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "allenai/led-base-16384"
_CONFIG_FOR_DOC = "LEDConfig"

LARGE_NEGATIVE = -1e8

# 从transformers.models.bart.modeling_tf_bart中复制函数
def shift_tokens_right(input_ids: tf.Tensor, pad_token_id: int, decoder_start_token_id: int):
    # 将pad_token_id和decoder_start_token_id转换为与input_ids相同的数据类型
    pad_token_id = tf.cast(pad_token_id, input_ids.dtype)
    decoder_start_token_id = tf.cast(decoder_start_token_id, input_ids.dtype)
    # 创建形状为(input_ids的行数, 1)的张量，其值为decoder_start_token_id
    start_tokens = tf.fill(
        (shape_list(input_ids)[0], 1), tf.convert_to_tensor(decoder_start_token_id, input_ids.dtype)
    )
    # 将input_ids向右移动一位，并在开头插入decoder_start_token_id，形成shifted_input_ids
    shifted_input_ids = tf.concat([start_tokens, input_ids[:, :-1]], -1)
    # 将shifted_input_ids中可能的-100值替换为pad_token_id
    shifted_input_ids = tf.where(
        shifted_input_ids == -100,
        tf.fill(shape_list(shifted_input_ids), tf.convert_to_tensor(pad_token_id, input_ids.dtype)),
        shifted_input_ids,
    )
    # 断言shifted_input_ids中的值均为非负数
    assert_gte0 = tf.debugging.assert_greater_equal(shifted_input_ids, tf.constant(0, dtype=input_ids.dtype))
    # 确保断言操作被调用，通过将结果包装在identity no-op中
    with tf.control_dependencies([assert_gte0]):
        shifted_input_ids = tf.identity(shifted_input_ids)
    return shifted_input_ids

# 从transformers.models.bart.modeling_tf_bart中复制函数
def _make_causal_mask(input_ids_shape: tf.TensorShape, past_key_values_length: int = 0):
    """
    # 计算双向自注意力所需的因果掩码
    """
    # 获取输入张量的批量大小
    bsz = input_ids_shape[0]
    # 获取目标序列长度
    tgt_len = input_ids_shape[1]
    # 创建一个全为负无穷大的张量，用作初始掩码
    mask = tf.ones((tgt_len, tgt_len)) * LARGE_NEGATIVE
    # 生成一个与掩码张量长度相同的一维张量，范围为 0 到掩码长度减一
    mask_cond = tf.range(shape_list(mask)[-1])

    # 使用掩码长度及其加一后的形状来更新掩码张量的值，将主对角线及其以下的值设置为 0.0
    mask = tf.where(mask_cond < tf.reshape(mask_cond + 1, (shape_list(mask)[-1], 1)), 0.0, mask)

    # 如果过去的键值对长度大于 0，则在掩码的左侧添加全零列，用于对齐过去的键值对
    if past_key_values_length > 0:
        mask = tf.concat([tf.zeros((tgt_len, past_key_values_length)), mask], axis=-1)

    # 将掩码张量在批次和头维度上进行复制，使其与模型输出的张量形状相匹配，并返回结果
    return tf.tile(mask[None, None, :, :], (bsz, 1, 1, 1))
# 从transformers.models.bart.modeling_tf_bart._expand_mask中复制得到的函数
def _expand_mask(mask: tf.Tensor, tgt_len: Optional[int] = None):
    """
    将注意力掩码从`[bsz, seq_len]`扩展为`[bsz, 1, tgt_seq_len, src_seq_len]`。
    """
    # 获取掩码的长度
    src_len = shape_list(mask)[1]
    # 如果tgt_len不为None，则使用tgt_len，否则使用src_len
    tgt_len = tgt_len if tgt_len is not None else src_len
    # 创建值为1的常量张量，并将mask转换为与常量相同的数据类型
    one_cst = tf.constant(1.0)
    mask = tf.cast(mask, dtype=one_cst.dtype)
    # 在特定维度上复制mask，维度为[bsz, 1, tgt_len, src_len]
    expanded_mask = tf.tile(mask[:, None, None, :], (1, 1, tgt_len, 1))

    return (one_cst - expanded_mask) * LARGE_NEGATIVE


class TFLEDLearnedPositionalEmbedding(tf.keras.layers.Embedding):
    """
    该模块学习位置嵌入，最大长度为固定值。
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, **kwargs):
        super().__init__(num_embeddings, embedding_dim, **kwargs)

    def call(self, input_shape: tf.TensorShape, past_key_values_length: int = 0):
        """输入大小应为[bsz x seqlen]。"""
        # 获取序列长度
        seq_len = input_shape[1]
        # 生成位置ID，从0到seq_len-1，然后加上历史键值的长度
        position_ids = tf.range(seq_len, delta=1, name="range")
        position_ids += past_key_values_length

        return super().call(tf.cast(position_ids, dtype=tf.int32))


# 从transformers.models.longformer.modeling_tf_longformer.TFLongformerSelfAttention中复制得到的类，将TFLongformer改为TFLEDEncoder
class TFLEDEncoderSelfAttention(tf.keras.layers.Layer):
    # 初始化函数，接受配置、层 ID 和其他关键字参数
    def __init__(self, config, layer_id, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)
        # 保存配置信息到对象中
        self.config = config

        # 检查隐藏层大小是否可以被注意力头的数量整除，否则抛出数值错误
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads}"
            )

        # 初始化注意力头数量和每个头的维度
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size
        # 创建查询、键、值的 Dense 层
        self.query = tf.keras.layers.Dense(
            self.embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            name="query",
        )
        self.key = tf.keras.layers.Dense(
            self.embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            name="key",
        )
        self.value = tf.keras.layers.Dense(
            self.embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            name="value",
        )

        # 为具有全局注意力的标记分配单独的投影层
        self.query_global = tf.keras.layers.Dense(
            self.embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            name="query_global",
        )
        self.key_global = tf.keras.layers.Dense(
            self.embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            name="key_global",
        )
        self.value_global = tf.keras.layers.Dense(
            self.embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            name="value_global",
        )
        # 创建丢弃层，用于注意力概率的丢弃
        self.dropout = tf.keras.layers.Dropout(config.attention_probs_dropout_prob)
        self.global_dropout = tf.keras.layers.Dropout(config.attention_probs_dropout_prob)
        # 保存层 ID
        self.layer_id = layer_id
        # 获取注意力窗口大小
        attention_window = config.attention_window[self.layer_id]

        # 断言注意力窗口大小为偶数
        assert (
            attention_window % 2 == 0
        ), f"`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}"
        # 断言注意力窗口大小为正数
        assert (
            attention_window > 0
        ), f"`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}"

        # 计算单侧注意力窗口大小
        self.one_sided_attn_window_size = attention_window // 2
    # 在模型建立时进行构建，如果还未构建
    def build(self, input_shape=None):
        if not self.built:
            # 构建查询、键和值的全局模型
            with tf.name_scope("query_global"):
                self.query_global.build((self.config.hidden_size,))
            with tf.name_scope("key_global"):
                self.key_global.build((self.config.hidden_size,))
            with tf.name_scope("value_global"):
                self.value_global.build((self.config.hidden_size,))

        if self.built:
            return
        self.built = True
        # 构建查询、键、值和全局查询、键、值
        if getattr(self, "query", None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.config.hidden_size])
        if getattr(self, "key", None) is not None:
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.config.hidden_size])
        if getattr(self, "value", None) is not None:
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.config.hidden_size])
        if getattr(self, "query_global", None) is not None:
            with tf.name_scope(self.query_global.name):
                self.query_global.build([None, None, self.config.hidden_size])
        if getattr(self, "key_global", None) is not None:
            with tf.name_scope(self.key_global.name):
                self.key_global.build([None, None, self.config.hidden_size])
        if getattr(self, "value_global", None) is not None:
            with tf.name_scope(self.value_global.name):
                self.value_global.build([None, None, self.config.hidden_size])

    # 模型调用函数
    def call(
        self,
        inputs,
        training=False,
    @staticmethod
    def _mask_invalid_locations(input_tensor, window_overlap):
        # 创建正确的上三角布尔掩码
        mask_2d_upper = tf.reverse(
            tf.linalg.band_part(tf.ones(shape=(window_overlap, window_overlap + 1)), -1, 0),
            axis=[0],
        )

        # 填充为完整矩阵
        padding = tf.convert_to_tensor(
            [[0, shape_list(input_tensor)[1] - window_overlap], [0, shape_list(input_tensor)[3] - window_overlap - 1]]
        )

        # 创建下三角掩码
        mask_2d = tf.pad(mask_2d_upper, padding)

        # 与上三角掩码结合
        mask_2d = mask_2d + tf.reverse(mask_2d, axis=[0, 1])

        # 广播为完整矩阵
        mask_4d = tf.tile(mask_2d[None, :, None, :], (shape_list(input_tensor)[0], 1, 1, 1))

        # 用于掩码的无穷张量
        inf_tensor = -float("inf") * tf.ones_like(input_tensor)

        # 掩码
        input_tensor = tf.where(tf.math.greater(mask_4d, 0), inf_tensor, input_tensor)

        return input_tensor
    def _sliding_chunks_matmul_attn_probs_value(self, attn_probs, value, window_overlap):
        """
        Same as _sliding_chunks_query_key_matmul but for attn_probs and value tensors. Returned tensor will be of the
        same shape as `attn_probs`
        """

        # 获取 value 张量的形状信息
        batch_size, seq_len, num_heads, head_dim = shape_list(value)

        # 断言 seq_len 必须是 2 * window_overlap 的倍数
        tf.debugging.assert_equal(
            seq_len % (window_overlap * 2), 0, message="Seq_len has to be multiple of 2 * window_overlap"
        )
        # 断言 attn_probs 和 value 张量的前三个维度必须相同（除了 head_dim）
        tf.debugging.assert_equal(
            shape_list(attn_probs)[:3],
            shape_list(value)[:3],
            message="value and attn_probs must have same dims (except head_dim)",
        )
        # 断言 attn_probs 张量的最后一个维度必须为 2 * window_overlap + 1
        tf.debugging.assert_equal(
            shape_list(attn_probs)[3],
            2 * window_overlap + 1,
            message="attn_probs last dim has to be 2 * window_overlap + 1",
        )

        # 计算分块的数量
        chunks_count = seq_len // window_overlap - 1

        # 将 batch_size 和 num_heads 维度合并，然后将 seq_len 按照 window_overlap 大小分块
        chunked_attn_probs = tf.reshape(
            tf.transpose(attn_probs, (0, 2, 1, 3)),
            (
                batch_size * num_heads,
                seq_len // window_overlap,
                window_overlap,
                2 * window_overlap + 1,
            ),
        )

        # 将 batch_size 和 num_heads 维度合并
        value = tf.reshape(
            tf.transpose(value, (0, 2, 1, 3)),
            (batch_size * num_heads, seq_len, head_dim),
        )

        # 在序列的开头和结尾分别填充 window_overlap 个数值
        paddings = tf.convert_to_tensor([[0, 0], [window_overlap, window_overlap], [0, 0]])
        padded_value = tf.pad(value, paddings, constant_values=-1)

        # 将 padded_value 按照 3 * window_overlap * head_dim 大小分块，并且窗口之间有 window_overlap 的重叠
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

        # 断言 chunked_value 张量的形状
        tf.debugging.assert_equal(
            shape_list(chunked_value),
            [batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim],
            message="Chunked value has the wrong shape",
        )

        # 对 chunked_attn_probs 和 chunked_value 执行矩阵乘法，并且进行对角线填充
        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)
        context = tf.einsum("bcwd,bcdh->bcwh", chunked_attn_probs, chunked_value)
        context = tf.transpose(
            tf.reshape(context, (batch_size, num_heads, seq_len, head_dim)),
            (0, 2, 1, 3),
        )

        return context

    @staticmethod
    # 定义一个私有方法，对隐藏状态进行填充并交换最后两个维度
    def _pad_and_transpose_last_two_dims(hidden_states_padded, paddings):
        """pads rows and then flips rows and columns"""
        # 使用给定的填充值对隐藏状态进行填充
        hidden_states_padded = tf.pad(
            hidden_states_padded, paddings
        )  # padding value is not important because it will be overwritten
        # 获取填充后的隐藏状态的形状信息
        batch_size, chunk_size, seq_length, hidden_dim = shape_list(hidden_states_padded)
        # 改变填充后的隐藏状态的维度顺序
        hidden_states_padded = tf.reshape(hidden_states_padded, (batch_size, chunk_size, hidden_dim, seq_length))

        return hidden_states_padded

    @staticmethod
    # 定义一个静态方法，对分块隐藏状态进行填充并转换为对角线形式
    def _pad_and_diagonalize(chunked_hidden_states):
        """
        shift every row 1 step right, converting columns into diagonals.

        Example:

        ```python
        chunked_hidden_states: [
            0.4983,
            2.6918,
            -0.0071,
            1.0492,
            -1.8348,
            0.7672,
            0.2986,
            0.0285,
            -0.7584,
            0.4206,
            -0.0405,
            0.1599,
            2.0514,
            -1.1600,
            0.5372,
            0.2629,
        ]
        window_overlap = num_rows = 4
        ```py

                     (pad & diagonalize) => [ 0.4983, 2.6918, -0.0071, 1.0492, 0.0000, 0.0000, 0.0000
                       0.0000, -1.8348, 0.7672, 0.2986, 0.0285, 0.0000, 0.0000 0.0000, 0.0000, -0.7584, 0.4206,
                       -0.0405, 0.1599, 0.0000 0.0000, 0.0000, 0.0000, 2.0514, -1.1600, 0.5372, 0.2629 ]
        """
        # 获取chunked_hidden_states的形状信息
        total_num_heads, num_chunks, window_overlap, hidden_dim = shape_list(chunked_hidden_states)
        # 创建用于填充的张量
        paddings = tf.convert_to_tensor([[0, 0], [0, 0], [0, 0], [0, window_overlap + 1]])
        # 使用给定的填充值对chunked_hidden_states进行填充
        chunked_hidden_states = tf.pad(
            chunked_hidden_states, paddings
        )  # total_num_heads x num_chunks x window_overlap x (hidden_dim+window_overlap+1). Padding value is not important because it'll be overwritten
        # 改变填充后的chunked_hidden_states的维度顺序
        chunked_hidden_states = tf.reshape(
            chunked_hidden_states, (total_num_heads, num_chunks, -1)
        )  # total_num_heads x num_chunks x window_overlapL+window_overlapwindow_overlap+window_overlap
        # 去除填充后的chunked_hidden_states的多余部分
        chunked_hidden_states = chunked_hidden_states[
            :, :, :-window_overlap
        ]  # total_num_heads x num_chunks x window_overlapL+window_overlapwindow_overlap
        # 改变维度顺序，使chunked_hidden_states形成对角线
        chunked_hidden_states = tf.reshape(
            chunked_hidden_states,
            (total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim),
        )  # total_num_heads x num_chunks, window_overlap x hidden_dim+window_overlap
        # 去除多余的列
        chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]

        return chunked_hidden_states

    @staticmethod
    # 将隐藏状态分成重叠的块，块大小为2倍的窗口重叠，重叠大小为窗口重叠
    def _chunk(hidden_states, window_overlap):
        """转换为重叠块，块大小=2w，重叠大小=w"""
        # 获取批次大小、序列长度和隐藏维度的形状信息
        batch_size, seq_length, hidden_dim = shape_list(hidden_states)
        # 计算输出块的数量，公式为 2 * (序列长度 // (2 * 窗口重叠)) - 1
        num_output_chunks = 2 * (seq_length // (2 * window_overlap)) - 1
    
        # 定义框架大小和框架步长（类似卷积操作）
        frame_hop_size = window_overlap * hidden_dim
        frame_size = 2 * frame_hop_size
        # 将隐藏状态重塑为（批次大小，序列长度 * 隐藏维度）
        hidden_states = tf.reshape(hidden_states, (batch_size, seq_length * hidden_dim))
    
        # 使用给定的框架大小和框架步长对隐藏状态进行分块
        chunked_hidden_states = tf.signal.frame(hidden_states, frame_size, frame_hop_size)
    
        # 确认分块后的形状是否与预期一致
        tf.debugging.assert_equal(
            shape_list(chunked_hidden_states),
            [batch_size, num_output_chunks, frame_size],
            message=(
                "确保分块正确应用。`分块后的隐藏状态应该具有维度 "
                f" {[batch_size, frame_size, num_output_chunks]}，但得到的是 {shape_list(chunked_hidden_states)}。"
            ),
        )
    
        # 将分块后的隐藏状态重塑为（批次大小，输出块数量，2 * 窗口重叠，隐藏维度）
        chunked_hidden_states = tf.reshape(
            chunked_hidden_states,
            (batch_size, num_output_chunks, 2 * window_overlap, hidden_dim),
        )
    
        # 返回分块后的隐藏状态
        return chunked_hidden_states
    
    # 计算前向传播过程中所需的全局注意力索引
    @staticmethod
    def _get_global_attn_indices(is_index_global_attn):
        """计算前向传播过程中所需的全局注意力索引"""
        # 辅助变量：计算每个批次中全局注意力索引的数量
        num_global_attn_indices = tf.math.count_nonzero(is_index_global_attn, axis=1)
        # 将其转换为常量1的类型
        num_global_attn_indices = tf.cast(num_global_attn_indices, dtype=tf.constant(1).dtype)
    
        # 批次中全局注意力索引的最大数量
        max_num_global_attn_indices = tf.reduce_max(num_global_attn_indices)
    
        # 全局注意力索引的索引
        is_index_global_attn_nonzero = tf.where(is_index_global_attn)
    
        # 辅助变量：找到全局注意力索引中非填充值的位置
        is_local_index_global_attn = tf.range(max_num_global_attn_indices) < tf.expand_dims(
            num_global_attn_indices, axis=-1
        )
    
        # 全局注意力索引中非填充值的位置
        is_local_index_global_attn_nonzero = tf.where(is_local_index_global_attn)
    
        # 全局注意力索引中填充值的位置
        is_local_index_no_global_attn_nonzero = tf.where(tf.math.logical_not(is_local_index_global_attn))
    
        # 返回全局注意力相关的多个值
        return (
            max_num_global_attn_indices,
            is_index_global_attn_nonzero,
            is_local_index_global_attn_nonzero,
            is_local_index_no_global_attn_nonzero,
        )
    
    # 将注意力得分与全局注意力相关的内容进行拼接
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
        # 获取键向量的批量大小
        batch_size = shape_list(key_vectors)[0]

        # 选择全局键向量
        global_key_vectors = tf.gather_nd(key_vectors, is_index_global_attn_nonzero)

        # 创建仅包含全局键向量的张量
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

        # 计算来自全局键的注意力概率
        # 形状为 (batch_size, seq_len, num_heads, max_num_global_attn_indices)
        attn_probs_from_global_key = tf.einsum("blhd,bshd->blhs", query_vectors, key_vectors_only_global)

        # 转置注意力概率以匹配形状 (batch_size, max_num_global_attn_indices, seq_len, num_heads)
        attn_probs_from_global_key_trans = tf.transpose(attn_probs_from_global_key, (0, 3, 1, 2))

        # 创建用于遮罩的张量
        mask_shape = (shape_list(is_local_index_no_global_attn_nonzero)[0],) + tuple(
            shape_list(attn_probs_from_global_key_trans)[-2:]
        )
        mask = tf.ones(mask_shape) * -10000.0  # 使用较大的负数作为遮罩
        mask = tf.cast(mask, dtype=attn_probs_from_global_key_trans.dtype)

        # 使用遮罩更新注意力概率
        attn_probs_from_global_key_trans = tf.tensor_scatter_nd_update(
            attn_probs_from_global_key_trans,
            is_local_index_no_global_attn_nonzero,
            mask,
        )

        # 将注意力概率转置以匹配形状 (batch_size, seq_len, num_heads, max_num_global_attn_indices)
        attn_probs_from_global_key = tf.transpose(attn_probs_from_global_key_trans, (0, 2, 3, 1))

        # 将全局注意力概率连接到注意力分数上
        # 形状为 (batch_size, seq_len, num_heads, extra attention count + 2*window+1)
        attn_scores = tf.concat((attn_probs_from_global_key, attn_scores), axis=-1)

        return attn_scores

    # 计算带有全局索引的注意力输出
    def _compute_attn_output_with_global_indices(
        self,
        value_vectors,
        attn_probs,
        max_num_global_attn_indices,
        is_index_global_attn_nonzero,
        is_local_index_global_attn_nonzero,
    ):
        # 获取批量大小
        batch_size = shape_list(attn_probs)[0]

        # 仅保留全局注意力概率
        attn_probs_only_global = attn_probs[:, :, :, :max_num_global_attn_indices]

        # 选择全局值向量
        global_value_vectors = tf.gather_nd(value_vectors, is_index_global_attn_nonzero)

        # 创建仅含全局数值向量的数组
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

        # 计算仅全局注意力输出
        attn_output_only_global = tf.einsum("blhs,bshd->blhd", attn_probs_only_global, value_vectors_only_global)

        # 重新塑形注意力概率
        attn_probs_without_global = attn_probs[:, :, :, max_num_global_attn_indices:]

        # 计算包含全局的注意力输出
        attn_output_without_global = self._sliding_chunks_matmul_attn_probs_value(
            attn_probs_without_global, value_vectors, self.one_sided_attn_window_size
        )

        # 返回全局和局部注意力输出之和
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
        # 重新塑形和转置向量
        return tf.reshape(
            tf.transpose(
                tf.reshape(vector, (batch_size, -1, self.num_heads, self.head_dim)),
                (0, 2, 1, 3),
            ),
            (batch_size * self.num_heads, -1, self.head_dim),
        )
class TFLEDEncoderAttention(tf.keras.layers.Layer):
    # 定义一个名为TFLEDEncoderAttention的类，继承自tf.keras.layers.Layer类
    def __init__(self, config, layer_id, **kwargs):
        # 初始化函数，接受config, layer_id以及其他参数
        super().__init__(**kwargs)
        # 调用父类的初始化函数
        self.longformer_self_attn = TFLEDEncoderSelfAttention(config, layer_id=layer_id, name="longformer_self_attn")
        # 创建一个TFLEDEncoderSelfAttention实例，并将其保存为self.longformer_self_attn
        self.output_dense = tf.keras.layers.Dense(config.d_model, use_bias=True, name="output")
        # 创建一个全连接层实例，并将其保存为self.output_dense
        self.config = config
        # 保存传入的config参数

    def call(self, inputs, training=False):
        # 定义call方法，接受inputs和training参数
        (
            hidden_states,
            attention_mask,
            layer_head_mask,
            is_index_masked,
            is_index_global_attn,
            is_global_attn,
        ) = inputs
        # 将inputs解包为多个变量

        self_outputs = self.longformer_self_attn(
            [hidden_states, attention_mask, layer_head_mask, is_index_masked, is_index_global_attn, is_global_attn],
            training=training,
        )
        # 调用self.longformer_self_attn的call方法，并将结果保存为self_outputs

        attention_output = self.output_dense(self_outputs[0], training=training)
        # 使用self.output_dense处理self_outputs的第一个元素，并保存结果为attention_output
        outputs = (attention_output,) + self_outputs[1:]
        # 将attention_output与self_outputs的其他元素组合成元组，并保存为outputs

        return outputs
        # 返回outputs作为方法的输出

    def build(self, input_shape=None):
        # 定义一个名为build的方法，接受input_shape参数，输入参数为空时，默认为None
        if self.built:
            return
        # 如果self.built为True，直接返回
        self.built = True
        # 将self.built设置为True
        if getattr(self, "longformer_self_attn", None) is not None:
            with tf.name_scope(self.longformer_self_attn.name):
                self.longformer_self_attn.build(None)
        # 如果self.longformer_self_attn存在，调用其build方法
        if getattr(self, "output_dense", None) is not None:
            with tf.name_scope(self.output_dense.name):
                self.output_dense.build([None, None, self.config.d_model])
        # 如果self.output_dense存在，调用其build方法


class TFLEDDecoderAttention(tf.keras.layers.Layer):
    # 定义一个名为TFLEDDecoderAttention的类，继承自tf.keras.layers.Layer类
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
        # 初始化函数，接受embed_dim, num_heads, dropout, is_decoder, bias和其他参数
        super().__init__(**kwargs)
        # 调用父类的初始化函数
        self.embed_dim = embed_dim
        # 保存传入的embed_dim参数

        self.num_heads = num_heads
        # 保存传入的num_heads参数
        self.dropout = tf.keras.layers.Dropout(dropout)
        # 创建一个Dropout层实例，并将其保存为self.dropout
        self.head_dim = embed_dim // num_heads
        # 计算头部维度
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        # 检查embed_dim是否是num_heads的倍数
        self.scaling = self.head_dim**-0.5
        # 计算头部维度的缩放系数
        self.is_decoder = is_decoder
        # 保存传入的is_decoder参数

        self.k_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="k_proj")
        # 创建一个全连接层实例，并将其保存为self.k_proj
        self.q_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="q_proj")
        # 创建一个全连接层实例，并将其保存为self.q_proj
        self.v_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="v_proj")
        # 创建一个全连接层实例，并将其保存为self.v_proj
        self.out_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="out_proj")
        # 创建一个全连接层实例，并将其保存为self.out_proj

    def _shape(self, tensor: tf.Tensor, seq_len: int, bsz: int):
        # 定义一个名为_shape的方法，接受tensor, seq_len, bsz参数
        return tf.transpose(tf.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        # 对tensor进行reshape和transpose操作，并返回结果
    # 定义一个 call 方法，用于实现模型的前向传播
    def call(
        self,
        hidden_states: tf.Tensor,
        key_value_states: tf.Tensor | None = None,
        past_key_value: Tuple[Tuple[tf.Tensor]] | None = None,
        attention_mask: tf.Tensor | None = None,
        layer_head_mask: tf.Tensor | None = None,
        training=False,
    # 在构建模型时调用的方法
    def build(self, input_shape=None):
        # 如果模型已经构建完成，则直接返回，不重复构建
        if self.built:
            return
        # 设置标志表示模型已经构建完成
        self.built = True
        # 如果存在 k_proj 属性，构建 k_proj 模型
        if getattr(self, "k_proj", None) is not None:
            with tf.name_scope(self.k_proj.name):
                self.k_proj.build([None, None, self.embed_dim])
        # 如果存在 q_proj 属性，构建 q_proj 模型
        if getattr(self, "q_proj", None) is not None:
            with tf.name_scope(self.q_proj.name):
                self.q_proj.build([None, None, self.embed_dim])
        # 如果存在 v_proj 属性，构建 v_proj 模型
        if getattr(self, "v_proj", None) is not None:
            with tf.name_scope(self.v_proj.name):
                self.v_proj.build([None, None, self.embed_dim])
        # 如果存在 out_proj 属性，构建 out_proj 模型
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build([None, None, self.embed_dim])
class TFLEDEncoderLayer(tf.keras.layers.Layer):
    # 初始化方法，设置层参数
    def __init__(self, config: LEDConfig, layer_id: int, **kwargs):
        super().__init__(**kwargs)
        # 获取配置中的嵌入维度
        self.embed_dim = config.d_model
        # 创建自注意力层
        self.self_attn = TFLEDEncoderAttention(config, layer_id, name="self_attn")
        # 创建自注意力层的 LayerNormalization 层
        self.self_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        # 创建 dropout 层
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        # 获取激活函数
        self.activation_fn = get_tf_activation(config.activation_function)
        # 创建激活函数的 dropout 层
        self.activation_dropout = tf.keras.layers.Dropout(config.activation_dropout)
        # 创建全连接层 fc1
        self.fc1 = tf.keras.layers.Dense(config.encoder_ffn_dim, name="fc1")
        # 创建全连接层 fc2
        self.fc2 = tf.keras.layers.Dense(self.embed_dim, name="fc2")
        # 创建最终的 LayerNormalization 层
        self.final_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")
        # 保存配置
        self.config = config

    # 前向传播方法
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
            hidden_states (`tf.Tensor`): input to the layer of shape *(batch, seq_len, embed_dim)*
            attention_mask (`tf.Tensor`): attention mask of size
                *(batch, 1, tgt_len, src_len)* where padding elements are indicated by very large negative values.
            layer_head_mask (`tf.Tensor`): mask for attention heads in a given layer of size
                *(config.encoder_attention_heads,)*.
        """
        # 保存残差连接
        residual = hidden_states
        # 调用自注意力层
        layer_outputs = self.self_attn(
            [hidden_states, attention_mask, layer_head_mask, is_index_masked, is_index_global_attn, is_global_attn],
            training=training,
        )
        # 获取自注意力层的输出
        hidden_states = layer_outputs[0]

        # 断言自注意力层的输出与残差连接的形状相同
        tf.debugging.assert_equal(
            shape_list(hidden_states),
            shape_list(residual),
            message=f"Self attn modified the shape of query {shape_list(residual)} to {shape_list(hidden_states)}",
        )

        # 对自注意力层的输出应用 dropout
        hidden_states = self.dropout(hidden_states, training=training)
        # 残差连接
        hidden_states = residual + hidden_states
        # 应用 LayerNormalization
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # 保存残差连接
        residual = hidden_states
        # 应用激活函数和全连接层 fc1
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 应用激活函数的 dropout
        hidden_states = self.activation_dropout(hidden_states, training=training)
        # 应用全连接层 fc2
        hidden_states = self.fc2(hidden_states)
        # 对全连接层的输出应用 dropout
        hidden_states = self.dropout(hidden_states, training=training)
        # 残差连接
        hidden_states = residual + hidden_states
        # 最终应用 LayerNormalization
        hidden_states = self.final_layer_norm(hidden_states)

        # 返回处理后的 hidden_states，以及其他层输出
        return (hidden_states,) + layer_outputs[1:]
    # 构建模型
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 将标志设置为已构建
        self.built = True
        # 如果存在 self_attn 属性
        if getattr(self, "self_attn", None) is not None:
            with tf.name_scope(self.self_attn.name):
                # 构建 self_attn 层
                self.self_attn.build(None)
        # 如果存在 self_attn_layer_norm 属性
        if getattr(self, "self_attn_layer_norm", None) is not None:
            with tf.name_scope(self.self_attn_layer_norm.name):
                # 构建 self_attn_layer_norm 层
                self.self_attn_layer_norm.build([None, None, self.embed_dim])
        # 如果存在 fc1 属性
        if getattr(self, "fc1", None) is not None:
            with tf.name_scope(self.fc1.name):
                # 构建 fc1 层
                self.fc1.build([None, None, self.embed_dim])
        # 如果存在 fc2 属性
        if getattr(self, "fc2", None) is not None:
            with tf.name_scope(self.fc2.name):
                # 构建 fc2 层
                self.fc2.build([None, None, self.config.encoder_ffn_dim])
        # 如果存在 final_layer_norm 属性
        if getattr(self, "final_layer_norm", None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                # 构建 final_layer_norm 层
                self.final_layer_norm.build([None, None, self.embed_dim])
class TFLEDDecoderLayer(tf.keras.layers.Layer):
    # 初始化方法，接收 LEDConfig 对象和其他关键字参数
    def __init__(self, config: LEDConfig, **kwargs):
        super().__init__(**kwargs)
        # 设置嵌入维度为 LEDConfig 中的 d_model
        self.embed_dim = config.d_model
        # 创建自注意力机制层，用于处理自注意力
        self.self_attn = TFLEDDecoderAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            name="self_attn",
            is_decoder=True,
        )
        # 添加 dropout 层，用于防止过拟合
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        # 获取激活函数
        self.activation_fn = get_tf_activation(config.activation_function)
        # 添加激活函数的 dropout 层
        self.activation_dropout = tf.keras.layers.Dropout(config.activation_dropout)

        # 添加自注意力层规范化层
        self.self_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        # 创建编码器-解码器注意力机制层，用于处理编码器-解码器之间的注意力
        self.encoder_attn = TFLEDDecoderAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            name="encoder_attn",
            is_decoder=True,
        )
        # 添加编码器-解码器注意力机制层的规范化层
        self.encoder_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="encoder_attn_layer_norm")
        # 添加全连接层 1
        self.fc1 = tf.keras.layers.Dense(config.decoder_ffn_dim, name="fc1")
        # 添加全连接层 2
        self.fc2 = tf.keras.layers.Dense(self.embed_dim, name="fc2")
        # 添加最终规范化层
        self.final_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")
        # 保存配置
        self.config = config

    # 调用方法，处理输入张量，并返回处理结果
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
    # 定义一个用于构建模型的方法，可以接受输入的形状参数
    def build(self, input_shape=None):
        # 如果模型已经构建过则直接返回
        if self.built:
            return
        # 设置模型已经构建的标识为 True
        self.built = True
        # 如果对象有 self_attn 属性，并且不为 None，则构建 self_attn
        if getattr(self, "self_attn", None) is not None:
            # 在命名空间下构建 self_attn
            with tf.name_scope(self.self_attn.name):
                self.self_attn.build(None)
        # 如果对象有 self_attn_layer_norm 属性，并且不为 None，则构建 self_attn_layer_norm
        if getattr(self, "self_attn_layer_norm", None) is not None:
            # 在命名空间下构建 self_attn_layer_norm
            with tf.name_scope(self.self_attn_layer_norm.name):
                self.self_attn_layer_norm.build([None, None, self.embed_dim])
        # 如果对象有 encoder_attn 属性，并且不为 None，则构建 encoder_attn
        if getattr(self, "encoder_attn", None) is not None:
            # 在命名空间下构建 encoder_attn
            with tf.name_scope(self.encoder_attn.name):
                self.encoder_attn.build(None)
        # 如果对象有 encoder_attn_layer_norm 属性，并且不为 None，则构建 encoder_attn_layer_norm
        if getattr(self, "encoder_attn_layer_norm", None) is not None:
            # 在命名空间下构建 encoder_attn_layer_norm
            with tf.name_scope(self.encoder_attn_layer_norm.name):
                self.encoder_attn_layer_norm.build([None, None, self.embed_dim])
        # 如果对象有 fc1 属性，并且不为 None，则构建 fc1
        if getattr(self, "fc1", None) is not None:
            # 在命名空间下构建 fc1
            with tf.name_scope(self.fc1.name):
                self.fc1.build([None, None, self.embed_dim])
        # 如果对象有 fc2 属性，并且不为 None，则构建 fc2
        if getattr(self, "fc2", None) is not None:
            # 在命名空间下构建 fc2
            with tf.name_scope(self.fc2.name):
                self.fc2.build([None, None, self.config.decoder_ffn_dim])
        # 如果对象有 final_layer_norm 属性，并且不为 None，则构建 final_layer_norm
        if getattr(self, "final_layer_norm", None) is not None:
            # 在命名空间下构建 final_layer_norm
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.embed_dim])
# TFLEDPreTrainedModel 类，继承自 TFPreTrainedModel 类，用于 LED 预训练模型
class TFLEDPreTrainedModel(TFPreTrainedModel):
    # 配置类为 LEDConfig
    config_class = LEDConfig
    # 基础模型前缀为 "led"
    base_model_prefix = "led"

    # 输入签名属性，用于指定模型的输入格式
    @property
    def input_signature(self):
        # 调用父类的输入签名方法
        sig = super().input_signature
        # 添加全局注意力掩码参数的输入签名，格式为 (None, None)，类型为 tf.int32
        sig["global_attention_mask"] = tf.TensorSpec((None, None), tf.int32, name="global_attention_mask")
        # 返回完整的输入签名
        return sig


# dataclass 装饰器用于创建数据类，这里用于定义 TFLEDEncoderBaseModelOutput 类
@dataclass
# 从 transformers.models.longformer.modeling_tf_longformer.TFLongformerBaseModelOutput 复制而来，将 TFLongformer->TFLEDEncoder
class TFLEDEncoderBaseModelOutput(ModelOutput):
    """
    Base class for Longformer's outputs, with potential hidden states, local and global attentions.
    # `last_hidden_state`参数: 形状为`(batch_size, sequence_length, hidden_size)`的张量，模型最后一层的隐藏状态序列。
    last_hidden_state: tf.Tensor = None
    # `hidden_states`参数: 可选的元组类型参数，当传递`output_hidden_states=True`或`config.output_hidden_states=True`时返回，
    # 包含形状为`(batch_size, sequence_length, hidden_size)`的张量（一个用于嵌入的输出，每层一个）。
    hidden_states: Tuple[tf.Tensor] | None = None
    # `attentions`参数: 可选的元组类型参数，当传递`output_attentions=True`或`config.output_attentions=True`时返回，
    # 包含形状为`(batch_size, num_heads, sequence_length, x + attention_window + 1)`的张量，其中`x`是全局注意力掩码中具有全局注意力的标记数。
    # 这是自注意力头中注意力 softmax 之后的局部注意力权重，用于计算自注意力头中的加权平均值。
    # 这些是序列中每个标记到具有全局注意力的每个标记（前`x`个值）和到注意力窗口中的每个标记（剩余`attention_window + 1`个值）的注意力权重。
    # 注意，前`x`个值是指文本中具有固定位置的标记，但剩余的`attention_window + 1`个值是指具有相对位置的标记：
    # 标记到其自身的注意力权重位于索引`x + attention_window / 2`，前（后）`attention_window / 2`个值是到前（后）`attention_window / 2`个标记的注意力权重。
    # 如果注意力窗口包含具有全局注意力的标记，则对应索引处的注意力权重设置为0；该值应从第一个`x`个注意力权重中获取。
    # 如果一个标记具有全局注意力，那么对`attentions`中的所有其他标记的注意力权重设置为0，该值应从`global_attentions`中获取。
    attentions: Tuple[tf.Tensor] | None = None
    # `global_attentions`参数: 可选的元组类型参数，当传递`output_attentions=True`或`config.output_attentions=True`时返回，
    # 包含形状为`(batch_size, num_heads, sequence_length, x)`的张量，其中`x`是具有全局注意力掩码的标记数。
    # 这是注意力 softmax 之后的全局注意力权重，用于计算自注意力头中的加权平均值。
    # 这些是具有全局注意力的每个标记到序列中的每个标记的注意力权重。
    global_attentions: Tuple[tf.Tensor] | None = None
from dataclasses import dataclass
from typing import List, Tuple
import tensorflow as tf
from transformers.modeling_tf_utils import ModelOutput, TFPreTrainedModel

# 定义 TFLEDSeq2SeqModelOutput 类，继承自 ModelOutput 类
@dataclass
class TFLEDSeq2SeqModelOutput(ModelOutput):
    """
    Base class for model encoder's outputs that also contains pre-computed hidden states that can speed up sequential
    decoding.
    """

    # 最后一个隐藏状态，类型为 TensorFlow 张量，默认为 None
    last_hidden_state: tf.Tensor = None
    # 过去的键值，类型为列表，其中元素为 TensorFlow 张量或 None
    past_key_values: List[tf.Tensor] | None = None
    # 解码器隐藏状态，类型为元组，其中元素为 TensorFlow 张量或 None
    decoder_hidden_states: Tuple[tf.Tensor] | None = None
    # 解码器注意力，类型为元组，其中元素为 TensorFlow 张量或 None
    decoder_attentions: Tuple[tf.Tensor] | None = None
    # 交叉注意力，类型为元组，其中元素为 TensorFlow 张量或 None
    cross_attentions: Tuple[tf.Tensor] | None = None
    # 编码器最后一个隐藏状态，类型为 TensorFlow 张量或 None
    encoder_last_hidden_state: tf.Tensor | None = None
    # 编码器隐藏状态，类型为元组，其中元素为 TensorFlow 张量或 None
    encoder_hidden_states: Tuple[tf.Tensor] | None = None
    # 编码器注意力，类型为元组，其中元素为 TensorFlow 张量或 None
    encoder_attentions: Tuple[tf.Tensor] | None = None
    # 编码器全局注意力，类型为元组，其中元素为 TensorFlow 张量或 None
    encoder_global_attentions: Tuple[tf.Tensor] | None = None


# 定义 TFLEDSeq2SeqLMOutput 类，继承自 ModelOutput 类
@dataclass
class TFLEDSeq2SeqLMOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.
    """

    # 损失，类型为 TensorFlow 张量或 None
    loss: tf.Tensor | None = None
    # 对数，类型为 TensorFlow 张量，默认为 None
    logits: tf.Tensor = None
    # 过去的键值，类型为列表，其中元素为 TensorFlow 张量或 None
    past_key_values: List[tf.Tensor] | None = None
    # 解码器隐藏状态，类型为元组，其中元素为 TensorFlow 张量或 None
    decoder_hidden_states: Tuple[tf.Tensor] | None = None
    # 解码器注意力，类型为元组，其中元素为 TensorFlow 张量或 None
    decoder_attentions: Tuple[tf.Tensor] | None = None
    # 交叉注意力，类型为元组，其中元素为 TensorFlow 张量或 None
    cross_attentions: Tuple[tf.Tensor] | None = None
    # 编码器最后一个隐藏状态，类型为 TensorFlow 张量或 None
    encoder_last_hidden_state: tf.Tensor | None = None
    # 编码器隐藏状态，类型为元组，其中元素为 TensorFlow 张量或 None
    encoder_hidden_states: Tuple[tf.Tensor] | None = None
    # 编码器注意力，类型为元组，其中元素为 TensorFlow 张量或 None
    encoder_attentions: Tuple[tf.Tensor] | None = None
    # 编码器全局注意力，类型为元组，其中元素为 TensorFlow 张量或 None
    encoder_global_attentions: Tuple[tf.Tensor] | None = None


# 定义 LED_START_DOCSTRING 常量
LED_START_DOCSTRING = r"""
    This model inherits from `TFPreTrainedModel`. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a `tf.keras.Model` subclass. Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0
    documentation for all matter related to general usage and behavior.

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
"""
    # 使用 model 方法传入 input_ids 和 attention_mask 或 input_ids、attention_mask 和 token_type_ids
    # 返回一个包含一个或多个输入张量与在文档字符串中给定的输入名称相关联的字典：
    # `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`
    
    # 请注意，当使用子类化创建模型和层时，无需担心任何这些，因为您可以像对待任何其他 Python 函数一样传递输入！
    
    # 参数：
    # config（[`LEDConfig`]）：包含模型所有参数的模型配置类。
    # 使用配置文件初始化不会加载与模型关联的权重，仅加载配置。
    # 请查看[`~TFPreTrainedModel.from_pretrained`]方法来加载模型权重。
"""

# 定义类的文档字符串
LED_INPUTS_DOCSTRING = r"""
"""

# 用`keras_serializable`装饰器装饰自定义层`TFLEDEncoder`
@keras_serializable
class TFLEDEncoder(tf.keras.layers.Layer):
    # 配置类是`LEDConfig`
    config_class = LEDConfig
    # 初始化方法，接受`LEDConfig`对象和其他关键字参数
    """
    Transformer 编码器由 *config.encoder_layers* 个自注意力层组成。每一层是一个`TFLEDEncoderLayer`。

    Args:
        config: LEDConfig
    """

    def __init__(self, config: LEDConfig, embed_tokens: Optional[tf.keras.layers.Embedding] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        # 如果`config.encoder_layerdrop`大于0，打印警告信息
        if config.encoder_layerdrop > 0:
            logger.warning("Layerdrop is currently disabled in TFLED models.")
        # `layerdrop`值初始化为0.0
        self.layerdrop = 0.0
        # `padding_idx`保存`config.pad_token_id`的值
        self.padding_idx = config.pad_token_id

        # 如果`config.attention_window`为整数型
        if isinstance(config.attention_window, int):
            # 断言`config.attention_window`为偶数
            assert config.attention_window % 2 == 0, "`config.attention_window` has to be an even value"
            # 断言`config.attention_window`大于0
            assert config.attention_window > 0, "`config.attention_window` has to be positive"
            # 将`config.attention_window`转为列表，长度为`config.num_hidden_layers`
            config.attention_window = [config.attention_window] * config.num_hidden_layers  # one value per layer
        else:
            # 断言`config.attention_window`的长度等于`config.num_hidden_layers`
            assert len(config.attention_window) == config.num_hidden_layers, (
                "`len(config.attention_window)` should equal `config.num_hidden_layers`. "
                f"Expected {config.num_hidden_layers}, given {len(config.attention_window)}"
            )

        # 保存`config.attention_window`
        self.attention_window = config.attention_window
        self.embed_tokens = embed_tokens
        # 初始化`embed_positions`，学习位置编码
        self.embed_positions = TFLEDLearnedPositionalEmbedding(
            config.max_encoder_position_embeddings,
            config.d_model,
            name="embed_positions",
        )
        # 初始化`layers`，由多个`TFLEDEncoderLayer`组成
        self.layers = [TFLEDEncoderLayer(config, i, name=f"layers.{i}") for i in range(config.encoder_layers)]
        # 初始化`layernorm_embedding`层
        self.layernorm_embedding = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layernorm_embedding")
        # 保存`config.d_model`到`embed_dim`
        self.embed_dim = config.d_model

    # 获取`embed_tokens`方法
    def get_embed_tokens(self):
        return self.embed_tokens

    # 设置`embed_tokens`方法
    def set_embed_tokens(self, embed_tokens):
        self.embed_tokens = embed_tokens

    # `call`方法，接受多个参数和关键字参数
    @unpack_inputs
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
    # `compute_hidden_states`方法，在`hidden_states`后剪切`padding_len`个元素
    @tf.function
    def compute_hidden_states(self, hidden_states, padding_len):
        return hidden_states[:, :-padding_len] if padding_len > 0 else hidden_states

    # `_pad_to_window_size`方法，用于填充输入以匹配注意力窗口大小
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

        # 获取输入数据的形状
        input_shape = shape_list(input_ids) if input_ids is not None else shape_list(inputs_embeds)
        batch_size, seq_len = input_shape[:2]
        # 计算需要填充的长度，使得序列长度能够整除 attention_window
        padding_len = (attention_window - seq_len % attention_window) % attention_window

        if padding_len > 0:
            # 如果需要填充，发出警告并填充输入数据
            logger.warning_once(
                f"Input ids are automatically padded from {seq_len} to {seq_len + padding_len} to be a multiple of "
                f"`config.attention_window`: {attention_window}"
            )

        # 构造填充张量
        paddings = tf.convert_to_tensor([[0, 0], [0, padding_len]])

        if input_ids is not None:
            # 对输入数据进行填充
            input_ids = tf.pad(input_ids, paddings, constant_values=pad_token_id)

        if inputs_embeds is not None:
            if padding_len > 0:
                # 如果需要填充，对输入的嵌入向量进行填充
                input_ids_padding = tf.fill((batch_size, padding_len), pad_token_id)
                inputs_embeds_padding = self.embed_tokens(input_ids_padding)
                inputs_embeds = tf.concat([inputs_embeds, inputs_embeds_padding], axis=-2)

        # 对注意力掩码进行填充，填充部分不应该受到注意
        attention_mask = tf.pad(attention_mask, paddings, constant_values=False)  # no attention on the padding tokens

        return (
            padding_len,
            input_ids,
            attention_mask,
            inputs_embeds,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果存在嵌入位置信息，构建嵌入位置信息层
        if getattr(self, "embed_positions", None) is not None:
            with tf.name_scope(self.embed_positions.name):
                self.embed_positions.build(None)
        # 如果存在层归一化层，构建归一化层
        if getattr(self, "layernorm_embedding", None) is not None:
            with tf.name_scope(self.layernorm_embedding.name):
                self.layernorm_embedding.build([None, None, self.embed_dim])
        # 如果存在多层，逐层构建
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)
# 使用 keras_serializable 装饰器将类 TFLEDDecoder 标记为可序列化的
@keras_serializable
class TFLEDDecoder(tf.keras.layers.Layer):
    # 将配置类 LEDConfig 赋值给 config_class 属性
    config_class = LEDConfig
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`TFLEDDecoderLayer`]

    Args:
        config: LEDConfig
        embed_tokens: output embedding
    """

    # 初始化方法
    def __init__(self, config: LEDConfig, embed_tokens: Optional[tf.keras.layers.Embedding] = None, **kwargs):
        # 调用父类初始化方法
        super().__init__(**kwargs)
        # 将配置信息和填充 token 的索引赋值给对应属性
        self.config = config
        self.padding_idx = config.pad_token_id
        self.embed_tokens = embed_tokens
        # 如果配置中的 decoder_layerdrop 大于 0，打印警告信息
        if config.decoder_layerdrop > 0:
            logger.warning("Layerdrop is currently disabled in TFLED models.")
        self.layerdrop = 0.0
        # 创建学习的位置嵌入对象
        self.embed_positions = TFLEDLearnedPositionalEmbedding(
            config.max_decoder_position_embeddings,
            config.d_model,
            name="embed_positions",
        )
        # 创建多个 TFLEDDecoderLayer 层
        self.layers = [TFLEDDecoderLayer(config, name=f"layers.{i}") for i in range(config.decoder_layers)]
        # 创建 layernorm_embedding 层
        self.layernorm_embedding = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layernorm_embedding")

        # 创建 dropout 层
        self.dropout = tf.keras.layers.Dropout(config.dropout)

    # 设置 embed_tokens 属性的方法
    def set_embed_tokens(self, embed_tokens):
        self.embed_tokens = embed_tokens

    # call 方法
    @unpack_inputs
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
        # 方法主体，暂略

    # build 方法
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记已经构建
        self.built = True
        # 构建 embed_positions 层
        if getattr(self, "embed_positions", None) is not None:
            with tf.name_scope(self.embed_positions.name):
                self.embed_positions.build(None)
        # 构建 layernorm_embedding 层
        if getattr(self, "layernorm_embedding", None) is not None:
            with tf.name_scope(self.layernorm_embedding.name):
                self.layernorm_embedding.build([None, None, self.config.d_model])
        # 构建 layers 中的每个层
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)


# 使用 keras_serializable 装饰器将类 TFLEDMainLayer 标记为可序列化的
@keras_serializable
class TFLEDMainLayer(tf.keras.layers.Layer):
    # 将配置类 LEDConfig 赋值给 config_class 属性
    config_class = LEDConfig
    # 初始化 LED 模型的实例
    def __init__(self, config: LEDConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 保存 LED 模型的配置
        self.config = config
        # 创建共享的嵌入层，用于处理输入和输出的词嵌入
        self.shared = tf.keras.layers.Embedding(
            input_dim=config.vocab_size,  # 输入词汇表大小
            output_dim=config.d_model,     # 输出词嵌入的维度
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=self.config.init_std),  # 初始化词嵌入的标准差
            name="led.shared",             # 层的名称
        )
        # 设置加载/保存权重时的前缀，以指定层的名称作为命名空间
        self.shared.load_weight_prefix = "led.shared"

        # 创建 LED 编码器实例
        self.encoder = TFLEDEncoder(config, self.shared, name="encoder")
        # 创建 LED 解码器实例
        self.decoder = TFLEDDecoder(config, self.shared, name="decoder")

    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.shared

    # 设置输入嵌入层
    def set_input_embeddings(self, new_embeddings):
        # 更新共享嵌入层
        self.shared = new_embeddings
        # 更新编码器的嵌入层
        self.encoder.embed_tokens = self.shared
        # 更新解码器的嵌入层
        self.decoder.embed_tokens = self.shared

    # LED 模型的调用方法，处理输入并生成输出
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
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            # 如果 decoder_input_ids 和 decoder_inputs_embeds 都为 None，则不使用缓存
            use_cache = False

        if encoder_outputs is None:
            # 如果 encoder_outputs 为 None，则调用 self.encoder() 方法生成 encoder_outputs
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
        # 如果用户为 encoder_outputs 传入了一个元组，在 return_dict=True 时，将其包装在 TFLEDEncoderBaseModelOutput 中
        elif return_dict and not isinstance(encoder_outputs, TFLEDEncoderBaseModelOutput):
            encoder_outputs = TFLEDEncoderBaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        # 如果用户为 encoder_outputs 传入了一个 TFLEDEncoderBaseModelOutput，在 return_dict=False 时，将其包装在一个元组中
        elif not return_dict and not isinstance(encoder_outputs, tuple):
            encoder_outputs = encoder_outputs.to_tuple()

        # 调用 self.decoder() 方法生成 decoder_outputs
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

        if not return_dict:
            # 如果 return_dict 为 False，则返回 decoder_outputs 与 encoder_outputs 的组合
            return decoder_outputs + encoder_outputs

        return TFLEDSeq2SeqModelOutput(
            # 返回 TFLEDSeq2SeqModelOutput 对象
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
    # 构建模型，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 共享/绑定的权重应该在模型基础命名空间中
        # 在 tf.name_scope 后面添加 "/"（而不是在开头）会把它放在根命名空间而不是当前命名空间
        with tf.name_scope(self.shared.load_weight_prefix + "/" + self.shared.name + "/"):
            self.shared.build(None)
        # 如果存在编码器，则构建编码器
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 如果存在解码器，则构建解码器
        if getattr(self, "decoder", None) is not None:
            with tf.name_scope(self.decoder.name):
                self.decoder.build(None)
# 导入必要的库或模块
@add_start_docstrings(
    "The bare LED Model outputting raw hidden-states without any specific head on top.",  # 添加 LED 模型的文档字符串
    LED_START_DOCSTRING,  # 添加 LED 模型的开始文档字符串
)
# 定义 TFLEDModel 类，继承自 TFLEDPreTrainedModel
class TFLEDModel(TFLEDPreTrainedModel):
    # 初始化方法
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 实例化 TFLEDMainLayer 类，用于 LED 模型主要层
        self.led = TFLEDMainLayer(config, name="led")

    # 获取编码器的方法
    def get_encoder(self):
        return self.led.encoder

    # 获取解码器的方法
    def get_decoder(self):
        return self.led.decoder

    # 调用方法，用于模型的前向传播
    @unpack_inputs
    @add_start_docstrings_to_model_forward(LED_INPUTS_DOCSTRING.format("batch_size, sequence_length"))  # 添加模型前向传播的开始文档字符串
    @add_code_sample_docstrings(  # 添加代码示例的文档字符串
        checkpoint=_CHECKPOINT_FOR_DOC,  # 添加检查点的文档字符串
        output_type=TFLEDSeq2SeqModelOutput,  # 添加输出类型的文档字符串
        config_class=_CONFIG_FOR_DOC,  # 添加配置类的文档字符串
    )
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
        # 调用 TFLEDMainLayer 的方法，进行模型的前向传播
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
    # 定义一个方法，用于处理模型的输出
    def serving_output(self, output):
        # 如果配置中使用了缓存，则从输出中获取过去的键值，否则设为 None
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        # 如果配置中输出了隐藏状态，则将输出中的解码器隐藏状态转换为张量，否则设为 None
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置中输出了注意力权重，则将输出中的解码器注意力权重转换为张量，否则设为 None
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        # 如果配置中输出了注意力权重，则将输出中的交叉注意力权重转换为张量，否则设为 None
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        # 如果配置中输出了隐藏状态，则将输出中的编码器隐藏状态转换为张量，否则设为 None
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置中输出了注意力权重，则将输出中的编码器注意力权重转换为张量，否则设为 None
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None
        # 如果配置中输出了注意力权重，则将输出中的全局编码器注意力权重转换为张量，否则设为 None
        enc_g_attns = tf.convert_to_tensor(output.encoder_global_attentions) if self.config.output_attentions else None

        # 返回 TFLEDSeq2SeqModelOutput 对象，包含了处理后的输出结果
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

    # 定义一个方法，用于构建模型
    def build(self, input_shape=None):
        # 如果已经构建过了，则直接返回
        if self.built:
            return
        # 设置标志为已构建
        self.built = True
        # 如果存在 LED 模型，则在 LED 模型的命名空间下构建
        if getattr(self, "led", None) is not None:
            with tf.name_scope(self.led.name):
                self.led.build(None)
# 创建 BiasLayer 类，用于添加偏置到模型
class BiasLayer(tf.keras.layers.Layer):
    """
    Bias as a layer. It is used for serialization purposes: `tf.keras.Model.save_weights` stores on a per-layer basis,
    so all weights have to be registered in a layer.
    """

    def __init__(self, shape, initializer, trainable, name, **kwargs):
        super().__init__(name=name, **kwargs)
        # 添加一个权重变量作为偏置
        # 注意：当序列化时，此变量的名称不会被作用域化，将不会被序列化为 "outer_layer/inner_layer/.../name:0" 的格式。
        # 而是会被序列化为 "name:0" 。详细信息可参考：
        # https://github.com/huggingface/transformers/pull/18833#issuecomment-1233090214
        self.bias = self.add_weight(name=name, shape=shape, initializer=initializer, trainable=trainable)

    def call(self, x):
        return x + self.bias


# 创建 TFLEDForConditionalGeneration 类，用于定义条件生成模型
@add_start_docstrings(
    "The LED Model with a language modeling head. Can be used for summarization.",
    LED_START_DOCSTRING,
)
class TFLEDForConditionalGeneration(TFLEDPreTrainedModel):
    # 定义一些在加载时需要忽略的键
    _keys_to_ignore_on_load_unexpected = [
        r"led.encoder.embed_tokens.weight",
        r"led.decoder.embed_tokens.weight",
    ]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 创建 LED 主层
        self.led = TFLEDMainLayer(config, name="led")
        # 是否使用缓存
        self.use_cache = config.use_cache
        # 声明 final_bias_logits 为一个缓冲区，在 pytorch 中不可训练，保持一致
        self.bias_layer = BiasLayer(
            name="final_logits_bias", shape=[1, config.vocab_size], initializer="zeros", trainable=False
        )

        # 待办事项（Joao）：调查为什么 LED 在 XLA 生成中存在数值问题
        self.supports_xla_generation = False

    # 获取解码器
    def get_decoder(self):
        return self.led.decoder

    # 获取编码器
    def get_encoder(self):
        return self.led.encoder

    # 获取偏置
    def get_bias(self):
        return {"final_logits_bias": self.bias_layer.bias}

    # 设置偏置
    def set_bias(self, value):
        # 替换现有包含偏置的层，以正确（反）序列化
        vocab_size = value["final_logits_bias"].shape[-1]
        self.bias_layer = BiasLayer(
            name="final_logits_bias", shape=[1, vocab_size], initializer="zeros", trainable=False
        )
        self.bias_layer.bias.assign(value["final_logits_bias"])

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.get_input_embeddings()

    # 设置输出嵌入
    def set_output_embeddings(self, value):
        self.set_input_embeddings(value)

    # 对模型进行前向传播
    @unpack_inputs
    @add_start_docstrings_to_model_forward(LED_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFLEDSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    # 定义 Transformer 模型的调用方法
    def call(
        # 输入的标识符序列，可以为 None
        input_ids: TFModelInputType | None = None,
        # 注意力掩码，可以为 NumPy 数组或 TensorFlow 张量，也可以为 None
        attention_mask: np.ndarray | tf.Tensor | None = None,
        # 解码器的输入标识符序列，可以为 NumPy 数组或 TensorFlow 张量，也可以为 None
        decoder_input_ids: np.ndarray | tf.Tensor | None = None,
        # 解码器的注意力掩码，可以为 NumPy 数组或 TensorFlow 张量，也可以为 None
        decoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        # 头掩码，可以为 NumPy 数组或 TensorFlow 张量，也可以为 None
        head_mask: np.ndarray | tf.Tensor | None = None,
        # 解码器的头掩码，可以为 NumPy 数组或 TensorFlow 张量，也可以为 None
        decoder_head_mask: np.ndarray | tf.Tensor | None = None,
        # 编码器输出，可以为 None
        encoder_outputs: TFLEDEncoderBaseModelOutput | None = None,
        # 全局注意力掩码，可以为 NumPy 数组或 TensorFlow 张量，也可以为 None
        global_attention_mask: np.ndarray | tf.Tensor | None = None,
        # 过去的键值对，可以为 None
        past_key_values: Tuple[Tuple[Union[np.ndarray, tf.Tensor]]] | None = None,
        # 输入的嵌入，可以为 NumPy 数组或 TensorFlow 张量，也可以为 None
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        # 解码器的输入嵌入，可以为 NumPy 数组或 TensorFlow 张量，也可以为 None
        decoder_inputs_embeds: np.ndarray | tf.Tensor | None = None,
        # 是否使用缓存，可以为 None
        use_cache: bool | None = None,
        # 是否输出注意力权重，可以为 None
        output_attentions: bool | None = None,
        # 是否输出隐藏状态，可以为 None
        output_hidden_states: bool | None = None,
        # 是否返回字典类型的输出，可以为 None
        return_dict: bool | None = None,
        # 标签，可以为 None
        labels: tf.Tensor | None = None,
        # 是否处于训练状态，默认为 False
        training: bool = False,
    ) -> Tuple[tf.Tensor] | TFLEDSeq2SeqLMOutput:
        """
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, TFLEDForConditionalGeneration
        >>> import tensorflow as tf

        >>> mname = "allenai/led-base-16384"
        >>> tokenizer = AutoTokenizer.from_pretrained(mname)
        >>> TXT = "My friends are <mask> but they eat too many carbs."
        >>> model = TFLEDForConditionalGeneration.from_pretrained(mname)
        >>> batch = tokenizer([TXT], return_tensors="tf")
        >>> logits = model(inputs=batch.input_ids).logits
        >>> probs = tf.nn.softmax(logits[0])
        >>> # probs[5] is associated with the mask token
        ```py"""

        # 如果有标签，则不使用缓存
        if labels is not None:
            use_cache = False
            # 如果没有给定解码器输入 ID 或嵌入向量，则将标签右移一位作为解码器输入 ID
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        # 调用 LED 模型，传入各种参数
        outputs = self.led(
            input_ids,
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
        # 计算逻辑回归层的输出
        lm_logits = tf.matmul(outputs[0], self.led.shared.weights, transpose_b=True)
        lm_logits = self.bias_layer(lm_logits)
        # 如果有标签，则计算掩码语言模型损失
        masked_lm_loss = None if labels is None else self.hf_compute_loss(labels, lm_logits)

        # 如果 return_dict 为 False，则返回一个元组
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        # 否则返回 TFLEDSeq2SeqLMOutput 对象
        return TFLEDSeq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,  # d 输出的索引 1
            decoder_hidden_states=outputs.decoder_hidden_states,  # d 输出的索引 2
            decoder_attentions=outputs.decoder_attentions,  # d 输出的索引 3
            cross_attentions=outputs.cross_attentions,  # d 输出的索引 4
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,  # 编码器输出的索引 0
            encoder_hidden_states=outputs.encoder_hidden_states,  # e 输出的索引 1
            encoder_attentions=outputs.encoder_attentions,  # e 输出的索引 2
            encoder_global_attentions=outputs.encoder_global_attentions,
        )
```  
    # 定义一个用于处理模型输出的方法，根据配置参数选择性地转换输出内容
    def serving_output(self, output):
        # 根据是否使用缓存选择性地获取过去的键值对
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        # 如果需要输出隐藏层状态，则转换decoder_hidden_states为tensor
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        # 如果需要输出decoder注意力分布，则转换decoder_attentions为tensor
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        # 如果需要输出cross attentions分布，则转换cross_attentions为tensor
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        # 如果需要输出encoder隐藏层状态，则转换encoder_hidden_states为tensor
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        # 如果需要输出encoder的注意力分布，则转换encoder_attentions为tensor
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None
        # 如果需要输出encoder全局注意力分布，则转换encoder_global_attentions为tensor
        enc_g_attns = tf.convert_to_tensor(output.encoder_global_attentions) if self.config.output_attentions else None

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

    # 准备用于生成数据的输入，根据是否使用past_key_values切割decoder_input_ids
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
        # 如果存在past_key_values，则只保留decoder_input_ids的最后一个token
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs已定义，input_ids不需要
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "use_cache": use_cache,  # 更改这个参数以避免缓存（可能用于调试）
        }

    # 从标签中准备decoder_input_ids
    def prepare_decoder_input_ids_from_labels(self, labels: tf.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)
    # 定义计算损失的函数，忽略填充标记的交叉熵损失
    def hf_compute_loss(self, labels, logits):
        # 定义交叉熵损失函数，参数设置为输出 logits, 并不进行 reduction
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        # 若配置为使用传统损失，则对标签数据做处理
        if self.config.tf_legacy_loss:
            # 将标签数据展平
            melted_labels = tf.reshape(labels, (-1,))
            # 找出非填充的位置
            active_loss = tf.not_equal(melted_labels, self.config.pad_token_id)
            # 从 logits 中过滤非填充位置的数据
            reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, shape_list(logits)[2])), active_loss)
            # 从标签数据中过滤非填充位置的数据
            labels = tf.boolean_mask(melted_labels, active_loss)
            # 计算损失
            return loss_fn(labels, reduced_logits)
    
        # 将负标签裁剪为零，以避免 NaN 和错误 - 这些位置最终会被掩蔽
        unmasked_loss = loss_fn(tf.nn.relu(labels), logits)
        # 确保只有非填充标签会影响损失
        loss_mask = tf.cast(labels != self.config.pad_token_id, dtype=unmasked_loss.dtype)
        masked_loss = unmasked_loss * loss_mask
        # 计算总损失，除以非填充标签的总数量
        reduced_masked_loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(loss_mask)
        return tf.reshape(reduced_masked_loss, (1,))
    
    # 构建模型
    def build(self, input_shape=None):
        # 如果已经构建好了，直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 LED 层，构建 LED 层
        if getattr(self, "led", None) is not None:
            with tf.name_scope(self.led.name):
                self.led.build(None)
        # 如果存在偏置层，构建偏置层
        if getattr(self, "bias_layer", None) is not None:
            with tf.name_scope(self.bias_layer.name):
                self.bias_layer.build(None)
```