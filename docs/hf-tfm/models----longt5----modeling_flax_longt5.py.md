# `.\models\longt5\modeling_flax_longt5.py`

```
# 导入所需的模块和类
import copy  # 导入copy模块，用于复制对象
from typing import Any, Callable, List, Optional, Tuple  # 导入类型提示相关的模块

import flax.linen as nn  # 导入Flax的linen模块，并命名为nn
import jax  # 导入JAX库
import jax.numpy as jnp  # 导入JAX中的NumPy模块，并命名为jnp
import numpy as np  # 导入NumPy库，并命名为np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze  # 从Flax中导入相关类和函数
from flax.linen import combine_masks, make_causal_mask  # 导入Flax的函数和类
from flax.linen import partitioning as nn_partitioning  # 导入Flax的partitioning模块，并命名为nn_partitioning
from flax.linen.attention import dot_product_attention_weights  # 导入注意力机制相关函数
from flax.traverse_util import flatten_dict, unflatten_dict  # 导入Flax的工具函数
from jax.random import PRNGKey  # 从JAX中导入PRNGKey类

from ...modeling_flax_outputs import (  # 导入模型输出相关的类
    FlaxBaseModelOutput,
    FlaxBaseModelOutputWithPastAndCrossAttentions,
    FlaxCausalLMOutputWithCrossAttentions,
    FlaxSeq2SeqLMOutput,
    FlaxSeq2SeqModelOutput,
)
from ...modeling_flax_utils import (  # 导入模型工具函数和类
    ACT2FN,
    FlaxPreTrainedModel,
    append_call_sample_docstring,
    append_replace_return_docstrings,
    overwrite_call_docstring,
)
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings  # 导入工具函数和类
from .configuration_longt5 import LongT5Config  # 导入LongT5Config配置类


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

_CHECKPOINT_FOR_DOC = "google/long-t5-local-base"  # 预训练模型的检查点名称，用于文档
_CONFIG_FOR_DOC = "LongT5Config"  # 配置文件的名称，用于文档

remat = nn_partitioning.remat  # 将nn_partitioning.remat函数赋值给remat变量


# 从transformers.models.bart.modeling_flax_bart.shift_tokens_right复制而来
def shift_tokens_right(input_ids: jnp.ndarray, pad_token_id: int, decoder_start_token_id: int) -> jnp.ndarray:
    """
    将输入的token向右移动一个位置。
    """
    shifted_input_ids = jnp.zeros_like(input_ids)  # 创建一个和input_ids相同形状的零数组
    shifted_input_ids = shifted_input_ids.at[:, 1:].set(input_ids[:, :-1])  # 将input_ids向右移动一个位置
    shifted_input_ids = shifted_input_ids.at[:, 0].set(decoder_start_token_id)  # 设置起始位置为decoder_start_token_id

    shifted_input_ids = jnp.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)  # 如果shifted_input_ids等于-100，则设置为pad_token_id
    return shifted_input_ids  # 返回移动后的input_ids


def _pad_to_multiple(x: jnp.ndarray, block_len: int, axis: int, pad_value: int = 0) -> jnp.ndarray:
    """将数组填充到长度为block_len的倍数"""
    pad_len = -x.shape[axis] % block_len  # 计算需要填充的长度
    pad = [(0, 0)] * x.ndim  # 创建填充元组列表，维度与x相同
    pad[axis] = (0, pad_len)  # 设置axis维度的填充范围
    x = jnp.pad(x, pad_width=pad, mode="constant", constant_values=pad_value)  # 使用常数值pad_value进行填充
    return x  # 返回填充后的数组


def _split_into_blocks(x: jnp.ndarray, block_len: int, axis: int) -> jnp.ndarray:
    """沿着指定轴将输入数组分割成指定长度的块"""
    # 如果维度长度
    # 如果张量 x 在指定轴上的长度不是 block_len 的倍数，将使用 pad_value 进行填充
    # pad tensor to multiple of block_len
    if x.shape[axis] % block_len != 0:
        x = _pad_to_multiple(x, block_len, axis, pad_value=0)
    
    # 计算张量 x 在指定轴上被分成的块数
    num_blocks = x.shape[axis] // block_len
    
    # 构建输出张量的形状，保持除了指定轴外的其他维度不变，将指定轴的长度划分为 num_blocks 个块，每块长度为 block_len
    output_shape = x.shape[:axis] + (num_blocks, block_len) + x.shape[(axis + 1):]
    
    # 返回重塑后的张量，以形成指定的 output_shape
    return x.reshape(output_shape)
# 定义一个函数，用于将输入的数组 x 按指定轴 block_axis 进行扩展，使其在该轴上长度增加 2
# 其他轴不变，用常数值 pad_value 进行填充
def _concatenate_3_blocks(x: jnp.ndarray, block_axis: int, sequence_axis: int, pad_value: int = 0) -> jnp.ndarray:
    """Concatenate three consecutive blocks for each input block for local attentiont.
    For more information, see: https://arxiv.org/pdf/2112.07916.pdf.
    """
    num_blocks = x.shape[block_axis]

    pad = [(0, 0)] * x.ndim
    pad[block_axis] = (1, 1)
    # [batch_size, num_blocks, block_len] -> [batch_size, num_blocks + 2, block_len]
    x = jnp.pad(x, pad_width=pad, mode="constant", constant_values=pad_value)

    blocks_list: List[np.array] = []
    for i in range(3):
        # 我们在这里使用索引的方法:
        # https://numpy.org/doc/stable/user/basics.indexing.html#dealing-with-variable-numbers-of-indices-within-programs
        indices = [slice(0, None)] * x.ndim
        indices[block_axis] = slice(i, i + num_blocks)
        indices = tuple(indices)
        blocks_list.append(x[indices])
    # 返回沿着 sequence_axis 轴连接后的数组
    return jnp.concatenate(blocks_list, axis=sequence_axis)  # [batch_size, num_blocks, 3 * block_len, ...]


def _make_3block_relative_position_ids(block_len: int) -> jnp.ndarray:
    """Makes 3-blocked relative position ids for local attention."""
    position_ids = jnp.arange(3 * block_len, dtype=jnp.int32)
    center_position_ids = position_ids[block_len:-block_len]
    relative_position_ids = position_ids[None, :] - center_position_ids[:, None]  # [block_len, 3 * block_len]
    return relative_position_ids


def _mask_local_attention_mask(local_attention_mask: np.ndarray, block_len: int) -> jnp.ndarray:
    """Mask local attention mask to enforce that tokens are not allowed to attend tokens farther than ``local_radius."""
    relative_position_ids = _make_3block_relative_position_ids(block_len)
    locality_mask = jnp.abs(relative_position_ids) < block_len
    locality_mask = locality_mask[None, None, :, :]
    return jnp.logical_and(local_attention_mask, locality_mask)


def _get_local_attention_mask(attention_mask: np.ndarray, block_len: int) -> jnp.ndarray:
    """Prepare attention mask to be applied for a local attention."""
    # [batch_size, num_blocks, block_len]
    _blocked_attention_mask = _split_into_blocks(attention_mask, block_len, axis=1)
    # [batch_size, num_block, 3 * block_len]
    _3blocked_attention_mask = _concatenate_3_blocks(_blocked_attention_mask, block_axis=1, sequence_axis=2)

    _blocked_attention_mask = _blocked_attention_mask[..., None]
    _3blocked_attention_mask = _3blocked_attention_mask[..., None, :]
    # [batch_size, num_block, block_len, 3 * block_len]
    local_attention_mask = jnp.logical_and(_blocked_attention_mask, _3blocked_attention_mask)
    local_attention_mask = _mask_local_attention_mask(local_attention_mask, block_len)
    # [batch_size, 1, num_block, block_len, 3 * block_len]
    return local_attention_mask[:, None, ...]


def _make_global_fixed_block_ids(attention_mask: np.ndarray, global_block_size: int) -> Tuple[jnp.ndarray, np.ndarray]:
    """Make global fixed block ids for global attention."""
    ...
    """Obtain the "fixed block" global id corresponding to each input token.

    This implementation is a simlified version of the original Flaxformr implementation adopted from:
    https://github.com/google/flaxformer/blob/main/flaxformer/architectures/longt5/long_attention.py.

    In our scenario, as we use this strategy only for a decoder, orphan tokens, i.e. those tokens which do not make for
    the whole fixed block, are assigned to the preceding block.

    Padding tokens from the original sequence are represented by -1.
    """
    # 获取注意力掩码的批量大小和序列长度
    batch_size, seq_len = attention_mask.shape[:2]

    # 处理孤立标记的函数，将孤立的标记分配给前一个块
    def handle_orphan_tokens(block_ids: np.ndarray) -> jnp.ndarray:
        # 计算每个块的结束位置
        block_ends = (jnp.arange(seq_len) % global_block_size) == global_block_size - 1
        # 确定真实的块结束位置，同时确保块ID非负数
        true_block_ends = jnp.logical_and(block_ends, block_ids >= 0)
        # 统计完整块的数量
        full_blocks = true_block_ends.sum(-1)[..., None]
        # 将块ID限制在完整块的数量范围内
        block_ids = jnp.minimum(block_ids, full_blocks - 1)
        return block_ids

    # 创建固定块掩码，每个位置上的值为全局块大小的倒数
    fixed_block_mask = jnp.ones_like(attention_mask) / global_block_size
    # 对固定块掩码进行累积求和，并调整每个位置的值
    fixed_block_mask = jnp.cumsum(fixed_block_mask, axis=1) - fixed_block_mask
    # 根据注意力掩码设置掩码数组，非零位置设为1.0，零位置设为-1000.0
    mask = jnp.where(attention_mask != 0.0, 1.0, -1000.0)
    # 计算全局块ID，最大值为累积和减1，至少为-1.0（与注意力掩码数据类型相同）
    global_block_ids = jnp.maximum(
        jnp.floor(mask + fixed_block_mask - 1.0), jnp.array(-1.0, dtype=attention_mask.dtype)
    )
    # 将填充标记设为-1
    global_block_ids = (global_block_ids * attention_mask) + (attention_mask - 1)
    # 对孤立标记进行处理，保证块ID的正确性
    global_block_ids = handle_orphan_tokens(global_block_ids)
    # 计算全局块的数量
    num_globals = seq_len // global_block_size

    # 计算全局段ID，维度为[batch_size, seq_len // global_block_size]
    if num_globals > 0:
        # 如果存在全局块，则将全局块ID的最大值重复到每个全局块的数量上
        _sequence_block_ids_max = jnp.repeat(global_block_ids.max(axis=-1)[:, None], repeats=num_globals, axis=1)
    else:
        # 如果不存在全局块，则创建零数组
        _sequence_block_ids_max = jnp.zeros((batch_size, 0), dtype=global_block_ids.dtype)
    # 计算全局段ID，通过累积求和方法生成，每个全局段ID小于等于对应的块ID设为1，否则设为0
    global_segment_ids = jnp.cumsum(jnp.ones((batch_size, num_globals)), axis=-1) - 1
    global_segment_ids = jnp.where(global_segment_ids <= _sequence_block_ids_max, 1, 0)
    # 返回全局块ID和全局段ID
    return global_block_ids, global_segment_ids
# 创建用于本地到全局注意力的相对位置张量
def _make_side_relative_position_ids(attention_mask: np.ndarray, global_block_size: int) -> np.ndarray:
    # 调用函数生成全局固定块 ID 和全局段 ID
    block_ids, global_segment_ids = _make_global_fixed_block_ids(attention_mask, global_block_size)
    # 获取全局序列长度
    global_seq_len = global_segment_ids.shape[-1]
    # 创建全局位置索引
    global_positions = jnp.arange(global_seq_len)
    # 计算侧向相对位置张量
    side_relative_position = global_positions - block_ids[..., None]
    return side_relative_position


# 计算通过对各个块进行求和得到的各个块聚合
def _create_global_aggregates(hidden_states: np.ndarray, block_ids: np.ndarray, global_seq_len: int) -> np.ndarray:
    """Compute individual block aggregates by summing over individual blocks."""
    # 创建块 ID 的独热编码张量
    one_hot_block_ids = jax.nn.one_hot(block_ids, global_seq_len)
    # 执行张量乘积以计算块聚合
    return jnp.einsum("...nd,...ng->...gd", hidden_states, one_hot_block_ids)


# 从 transformers.models.t5.modeling_flax_t5.FlaxT5LayerNorm 复制并将 T5 更改为 LongT5
class FlaxLongT5LayerNorm(nn.Module):
    hidden_size: int
    dtype: jnp.dtype = jnp.float32
    eps: float = 1e-6
    weight_init: Callable[..., np.ndarray] = jax.nn.initializers.ones

    def setup(self):
        self.weight = self.param("weight", self.weight_init, (self.hidden_size,))

    def __call__(self, hidden_states):
        """
        Construct a layernorm module in the LongT5 style; No bias and no subtraction of mean.
        """
        # 总是使用 float32 计算层归一化
        variance = jnp.power(hidden_states.astype("f4"), 2).mean(axis=-1, keepdims=True)
        # 计算标准差并进行归一化
        hidden_states = hidden_states / jnp.sqrt(variance + self.eps)

        return self.weight * hidden_states


# 从 transformers.models.t5.modeling_flax_t5.FlaxT5DenseActDense 复制并将 T5 更改为 LongT5
class FlaxLongT5DenseActDense(nn.Module):
    config: LongT5Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 初始化权重标准差
        wi_init_std = self.config.initializer_factor * (self.config.d_model**-0.5)
        wo_init_std = self.config.initializer_factor * (self.config.d_ff**-0.5)

        # 定义输入密集层（无偏置）
        self.wi = nn.Dense(
            self.config.d_ff,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(wi_init_std),
            dtype=self.dtype,
        )
        # 定义激活函数
        self.act = ACT2FN[self.config.dense_act_fn]
        # 定义丢弃层
        self.dropout = nn.Dropout(self.config.dropout_rate)
        # 定义输出密集层（无偏置）
        self.wo = nn.Dense(
            self.config.d_model,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(wo_init_std),
            dtype=self.dtype,
        )

    def __call__(self, hidden_states, deterministic=True):
        # 输入到输入密集层
        hidden_states = self.wi(hidden_states)
        # 应用激活函数
        hidden_states = self.act(hidden_states)
        # 使用丢弃层（如果不是确定性的）
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 输入到输出密集层
        hidden_states = self.wo(hidden_states)
        return hidden_states


# 从 transformers.models.t5.modeling_flax_t5.FlaxT5DenseGatedActDense 复制并将 T5 更改为 LongT5
# 定义一个名为 FlaxLongT5DenseGatedActDense 的类，继承自 nn.Module
class FlaxLongT5DenseGatedActDense(nn.Module):
    # 类变量 config，类型为 LongT5Config，表示配置信息
    config: LongT5Config
    # 类变量 dtype，默认为 jnp.float32，表示计算中使用的数据类型

    # 初始化方法 setup，用于设置网络层
    def setup(self):
        # 初始化权重矩阵的标准差，wi_init_std 和 wo_init_std 分别为 d_model 和 d_ff 的倒数乘以 initializer_factor 的结果
        wi_init_std = self.config.initializer_factor * (self.config.d_model**-0.5)
        wo_init_std = self.config.initializer_factor * (self.config.d_ff**-0.5)

        # 创建第一个全连接层 wi_0
        self.wi_0 = nn.Dense(
            self.config.d_ff,  # 输出维度为 d_ff
            use_bias=False,  # 不使用偏置
            kernel_init=jax.nn.initializers.normal(wi_init_std),  # 使用正态分布初始化权重
            dtype=self.dtype,  # 指定数据类型为 dtype
        )
        # 创建第二个全连接层 wi_1
        self.wi_1 = nn.Dense(
            self.config.d_ff,  # 输出维度为 d_ff
            use_bias=False,  # 不使用偏置
            kernel_init=jax.nn.initializers.normal(wi_init_std),  # 使用正态分布初始化权重
            dtype=self.dtype,  # 指定数据类型为 dtype
        )
        # 创建输出全连接层 wo
        self.wo = nn.Dense(
            self.config.d_model,  # 输出维度为 d_model
            use_bias=False,  # 不使用偏置
            kernel_init=jax.nn.initializers.normal(wo_init_std),  # 使用正态分布初始化权重
            dtype=self.dtype,  # 指定数据类型为 dtype
        )
        # 创建 Dropout 层，使用配置中的 dropout_rate
        self.dropout = nn.Dropout(self.config.dropout_rate)
        # 根据配置中的 dense_act_fn 选择激活函数，并赋值给 act 变量
        self.act = ACT2FN[self.config.dense_act_fn]

    # 实现 __call__ 方法，定义类的可调用行为
    def __call__(self, hidden_states, deterministic):
        # 计算使用激活函数处理后的 hidden_states
        hidden_gelu = self.act(self.wi_0(hidden_states))
        # 计算 hidden_states 经过第二个全连接层的结果
        hidden_linear = self.wi_1(hidden_states)
        # 计算最终的 hidden_states，是经过门控激活函数处理的结果
        hidden_states = hidden_gelu * hidden_linear
        # 对 hidden_states 应用 Dropout，根据 deterministic 参数确定是否确定性执行
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 计算输出结果，经过全连接层 wo 处理
        hidden_states = self.wo(hidden_states)
        # 返回处理后的 hidden_states
        return hidden_states


# 从 transformers.models.t5.modeling_flax_t5.FlaxT5LayerFF 复制并修改为使用 LongT5
# 定义一个名为 FlaxLongT5LayerFF 的类，继承自 nn.Module
class FlaxLongT5LayerFF(nn.Module):
    # 类变量 config，类型为 LongT5Config，表示配置信息
    config: LongT5Config
    # 类变量 dtype，默认为 jnp.float32，表示计算中使用的数据类型

    # 初始化方法 setup，用于设置网络层
    def setup(self):
        # 如果配置中指定使用门控激活函数
        if self.config.is_gated_act:
            # 创建使用门控激活函数的 Dense 层对象
            self.DenseReluDense = FlaxLongT5DenseGatedActDense(self.config, dtype=self.dtype)
        else:
            # 创建使用普通激活函数的 Dense 层对象
            self.DenseReluDense = FlaxLongT5DenseActDense(self.config, dtype=self.dtype)

        # 创建 Layer Norm 层对象，使用 LongT5 的配置信息
        self.layer_norm = FlaxLongT5LayerNorm(
            self.config.d_model,  # 归一化的维度为 d_model
            eps=self.config.layer_norm_epsilon,  # 设置 epsilon 参数
            dtype=self.dtype,  # 指定数据类型为 dtype
        )
        # 创建 Dropout 层，使用配置中的 dropout_rate
        self.dropout = nn.Dropout(self.config.dropout_rate)

    # 实现 __call__ 方法，定义类的可调用行为
    def __call__(self, hidden_states, deterministic=True):
        # 对输入的 hidden_states 进行 Layer Norm 处理
        forwarded_states = self.layer_norm(hidden_states)
        # 将处理后的 hidden_states 传入 DenseReluDense 层对象处理
        forwarded_states = self.DenseReluDense(forwarded_states, deterministic=deterministic)
        # 加上 Dropout 处理后的 forwarded_states，并将结果加回到原始的 hidden_states 上
        hidden_states = hidden_states + self.dropout(forwarded_states, deterministic=deterministic)
        # 返回处理后的 hidden_states
        return hidden_states


# 从 transformers.models.t5.modeling_flax_t5.FlaxT5Attention 复制并修改为使用 LongT5
# 定义一个名为 FlaxLongT5Attention 的类，继承自 nn.Module
class FlaxLongT5Attention(nn.Module):
    # 类变量 config，类型为 LongT5Config，表示配置信息
    config: LongT5Config
    # 类变量 has_relative_attention_bias，默认为 False，表示是否有相对位置编码的注意力偏置
    has_relative_attention_bias: bool = False
    # 类变量 causal，默认为 False，表示是否是因果（自回归）注意力机制
    causal: bool = False
    # 类变量 dtype，默认为 jnp.float32，表示计算中使用的数据类型
    # 设置模型的初始化参数和配置
    def setup(self):
        # 设置相对注意力机制的桶数，从配置中获取
        self.relative_attention_num_buckets = self.config.relative_attention_num_buckets
        # 设置相对注意力机制的最大距离，从配置中获取
        self.relative_attention_max_distance = self.config.relative_attention_max_distance
        # 设置模型的维度，从配置中获取
        self.d_model = self.config.d_model
        # 设置键值投影的维度，从配置中获取
        self.key_value_proj_dim = self.config.d_kv
        # 设置注意力头的数量，从配置中获取
        self.n_heads = self.config.num_heads
        # 设置 dropout 率，从配置中获取
        self.dropout = self.config.dropout_rate
        # 计算内部维度，注意力头数量乘以键值投影维度
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # 初始化查询向量的标准差
        q_init_std = self.config.initializer_factor * ((self.inner_dim * self.key_value_proj_dim) ** -0.5)
        # 初始化键值向量的标准差
        kv_init_std = self.config.initializer_factor * (self.inner_dim**-0.5)
        # 初始化输出向量的标准差
        o_init_std = self.config.initializer_factor * (self.inner_dim**-0.5)

        # 初始化查询向量的 Dense 层
        self.q = nn.Dense(
            self.inner_dim,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(q_init_std),
            dtype=self.dtype,
        )
        # 初始化键向量的 Dense 层
        self.k = nn.Dense(
            self.inner_dim,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(kv_init_std),
            dtype=self.dtype,
        )
        # 初始化值向量的 Dense 层
        self.v = nn.Dense(
            self.inner_dim,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(kv_init_std),
            dtype=self.dtype,
        )
        # 初始化输出向量的 Dense 层
        self.o = nn.Dense(
            self.d_model,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(o_init_std),
            dtype=self.dtype,
        )

        # 如果模型具有相对注意力偏置，则初始化相对注意力偏置的嵌入层
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embed(
                self.relative_attention_num_buckets,
                self.n_heads,
                embedding_init=jax.nn.initializers.normal(kv_init_std),
                dtype=self.dtype,
            )

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on
        """
        relative_buckets = 0
        if bidirectional:
            # If bidirectional, adjust the number of buckets and determine if relative_position is positive
            num_buckets //= 2
            relative_buckets += (relative_position > 0) * num_buckets
            relative_position = jnp.abs(relative_position)
        else:
            # If not bidirectional, ensure relative_position is non-positive
            relative_position = -jnp.clip(relative_position, a_max=0)
        
        # Ensure relative_position is in the range [0, inf)
        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # Compute bucket index for larger positions logarithmically
        relative_position_if_large = max_exact + (
            jnp.log(relative_position / max_exact) / jnp.log(max_distance / max_exact) * (num_buckets - max_exact)
        )
        relative_position_if_large = jnp.clip(relative_position_if_large, a_max=num_buckets - 1)

        # Determine final relative_bucket based on whether relative_position is small or large
        relative_buckets += jnp.where(is_small, relative_position, relative_position_if_large)

        return relative_buckets.astype("i4")

    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        # Create matrices of context and memory positions
        context_position = jnp.arange(query_length, dtype="i4")[:, None]
        memory_position = jnp.arange(key_length, dtype="i4")[None, :]

        # Compute relative position as memory_position - context_position
        relative_position = memory_position - context_position

        # Compute relative_position_bucket using _relative_position_bucket function
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=(not self.causal),  # Determine bidirectionality based on 'causal' attribute
            num_buckets=self.relative_attention_num_buckets,  # Number of buckets for relative positions
            max_distance=self.relative_attention_max_distance,  # Maximum distance for mapping to buckets
        )

        # Obtain relative_attention_bias values based on computed relative_position_bucket
        values = self.relative_attention_bias(relative_position_bucket)

        # Rearrange values to match expected dimensions
        values = values.transpose((2, 0, 1))[None, :, :, :]

        return values

    def _split_heads(self, hidden_states):
        # Reshape hidden_states to split into heads
        return hidden_states.reshape(hidden_states.shape[:2] + (self.n_heads, self.key_value_proj_dim))
    # 将隐藏状态重塑为指定形状，用于后续操作
    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.inner_dim,))

    # 使用Flax的装饰器定义一个紧凑的函数，将投影后的键值状态与查询状态连接到缓存的先前状态
    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slighly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # 检测是否通过缺少现有缓存数据来初始化
        is_initialized = self.has_variable("cache", "cached_key")
        # 初始化或获取缓存的键和值，使用零向量填充
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        # 获取缓存的索引，指示当前缓存的位置
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            # 解构缓存的形状以便更新
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # 使用新的1D空间切片更新键和值缓存
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = jax.lax.dynamic_update_slice(cached_key.value, key, indices)
            value = jax.lax.dynamic_update_slice(cached_value.value, value, indices)
            # 更新缓存中的键和值
            cached_key.value = key
            cached_value.value = value
            # 更新缓存索引，表示已更新的缓存向量数量
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # 生成用于缓存的自注意力掩码：单个查询位置只能注意到已生成和缓存的键位置，而非剩余的零元素。
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            # 合并填充掩码和输入掩码
            attention_mask = combine_masks(pad_mask, attention_mask)
        
        # 返回更新后的键、值和注意力掩码
        return key, value, attention_mask

    # 创建位置偏置，用于注意力机制中的位置编码
    def _create_position_bias(
        self, key_states, query_states, attention_mask, init_cache, seq_length, causal_attention_mask_shift
        ):
            # 检查缓存是否已填充，并且当前场景支持因果关注（causal），并且缓存中没有初始化（init_cache 为 False）
            cache_is_filled = self.causal and self.has_variable("cache", "cached_key") and (not init_cache)
            # 计算关键字状态的长度
            key_length = key_states.shape[1]
            # 如果缓存已填充，则查询长度等于关键字状态的长度，否则等于查询状态的长度
            query_length = key_length if cache_is_filled else query_states.shape[1]

            # 如果模型支持相对注意偏置，则计算位置偏置
            if self.has_relative_attention_bias:
                position_bias = self.compute_bias(query_length, key_length)
            # 否则，如果有注意力掩码，则创建与注意力掩码相同形状的全零数组作为位置偏置
            elif attention_mask is not None:
                position_bias = jnp.zeros_like(attention_mask)
            # 否则，创建形状为 (1, self.n_heads, query_length, key_length) 的全零数组作为位置偏置
            else:
                position_bias = jnp.zeros((1, self.n_heads, query_length, key_length), dtype=self.dtype)

            # 如果缓存已填充，则只需取最后一个查询位置的偏置
            if cache_is_filled:
                max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
                position_bias = jax.lax.dynamic_slice(
                    position_bias,
                    (0, 0, causal_attention_mask_shift, 0),
                    (1, self.n_heads, seq_length, max_decoder_length),
                )
            # 返回计算得到的位置偏置
            return position_bias

        # 对象调用函数
        def __call__(
            self,
            hidden_states,
            attention_mask=None,
            key_value_states=None,
            position_bias=None,
            use_cache=False,
            output_attentions=False,
            deterministic=True,
            init_cache=False,
class FlaxLongT5LocalAttention(nn.Module):
    config: LongT5Config
    has_relative_attention_bias: bool = False
    dtype: jnp.dtype = jnp.float32  # 计算中使用的数据类型

    def setup(self):
        self.relative_attention_num_buckets = self.config.relative_attention_num_buckets
        self.relative_attention_max_distance = self.config.relative_attention_max_distance
        self.d_model = self.config.d_model
        self.key_value_proj_dim = self.config.d_kv
        self.n_heads = self.config.num_heads
        self.local_radius = self.config.local_radius
        self.block_len = self.local_radius + 1
        self.dropout = self.config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        q_init_std = self.config.initializer_factor * ((self.inner_dim * self.key_value_proj_dim) ** -0.5)
        kv_init_std = self.config.initializer_factor * (self.inner_dim**-0.5)
        o_init_std = self.config.initializer_factor * (self.inner_dim**-0.5)

        # 创建查询权重矩阵，用于计算查询 Q
        self.q = nn.Dense(
            self.inner_dim,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(q_init_std),
            dtype=self.dtype,
        )
        # 创建键权重矩阵，用于计算键 K
        self.k = nn.Dense(
            self.inner_dim,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(kv_init_std),
            dtype=self.dtype,
        )
        # 创建值权重矩阵，用于计算值 V
        self.v = nn.Dense(
            self.inner_dim,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(kv_init_std),
            dtype=self.dtype,
        )
        # 创建输出权重矩阵，用于计算输出 O
        self.o = nn.Dense(
            self.d_model,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(o_init_std),
            dtype=self.dtype,
        )

        # 如果配置中需要相对注意力偏置，则创建相对注意力偏置 Embed 层
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embed(
                self.relative_attention_num_buckets,
                self.n_heads,
                embedding_init=jax.nn.initializers.normal(kv_init_std),
            )

    @staticmethod
    # 从 transformers.models.t5.modeling_flax_t5.FlaxT5Attention._relative_position_bucket 复制而来
    def _relative_position_bucket(x, max_distance: int, num_buckets: int, bidirectional: bool = True):
        """
        根据相对位置计算桶索引，用于生成相对位置偏置。

        Args:
            x: 相对位置
            max_distance: 最大距离
            num_buckets: 桶的数量
            bidirectional: 是否双向

        Returns:
            相对位置的桶索引
        """
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on
        """
        relative_buckets = 0
        # 如果是双向注意力机制，则将桶的数量减半，并根据相对位置的正负确定桶的位置
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0) * num_buckets
            relative_position = jnp.abs(relative_position)
        else:
            # 如果是单向注意力机制，则将相对位置修正为非正数
            relative_position = -jnp.clip(relative_position, a_max=0)
        # 现在 relative_position 范围为 [0, inf)

        # 将较小的相对位置映射到更小的桶，将较大的相对位置映射到更大的桶
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # 较大相对位置映射到更大的桶，使用对数增长来平衡较大的相对位置
        relative_position_if_large = max_exact + (
            jnp.log(relative_position / max_exact) / jnp.log(max_distance / max_exact) * (num_buckets - max_exact)
        )
        relative_position_if_large = jnp.clip(relative_position_if_large, a_max=num_buckets - 1)

        # 根据相对位置大小选择相应的桶
        relative_buckets += jnp.where(is_small, relative_position, relative_position_if_large)

        return relative_buckets.astype("i4")

    def compute_bias(self, block_length: int):
        """Compute binned relative position bias"""
        # 创建记忆位置和上下文位置数组
        memory_position = jnp.arange(3 * block_length, dtype="i4")
        context_position = memory_position[block_length:-block_length]

        # 计算相对位置并将其转换为相对位置桶
        relative_position = memory_position[None, :] - context_position[:, None]
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=True,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )

        # 获取相对注意力偏置的值
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.transpose((2, 0, 1))[None, None, :, :, :]
        return values

    def _split_heads(self, hidden_states):
        # 将隐藏状态张量按头数目拆分
        return hidden_states.reshape(hidden_states.shape[:2] + (self.n_heads, self.key_value_proj_dim))
    # 将隐藏状态重新形状化为 (batch_size, sequence_length, inner_dim)
    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[0], -1, self.inner_dim)

    # 创建位置偏置矩阵，用于注意力机制中的位置编码
    def _create_position_bias(self, block_len: int, attention_mask: Optional[np.ndarray]) -> np.ndarray:
        # position_bias 的形状: (1, 1, n_heads, block_len, 3 * block_len)
        if self.has_relative_attention_bias:
            # 如果模型支持相对注意力偏置，则计算相对偏置
            position_bias = self.compute_bias(block_len)
        elif attention_mask is not None:
            # 如果有注意力遮罩，则创建一个与其形状相同的零矩阵作为位置偏置
            position_bias = jnp.zeros_like(attention_mask)
        else:
            # 否则创建一个形状为 (1, 1, self.n_heads, block_len, 3 * block_len) 的零矩阵作为位置偏置
            position_bias = jnp.zeros((1, 1, self.n_heads, block_len, 3 * block_len), dtype=self.dtype)

        return position_bias

    # Transformer 模型的主要调用方法，执行前向传播计算
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        key_value_states=None,
        position_bias=None,
        output_attentions=False,
        deterministic=True,
class FlaxLongT5TransientGlobalAttention(nn.Module):
    config: LongT5Config
    has_relative_attention_bias: bool = False
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.relative_attention_num_buckets = self.config.relative_attention_num_buckets
        self.relative_attention_max_distance = self.config.relative_attention_max_distance
        self.d_model = self.config.d_model
        self.key_value_proj_dim = self.config.d_kv
        self.n_heads = self.config.num_heads
        self.local_radius = self.config.local_radius
        self.block_len = self.local_radius + 1
        self.global_block_size = self.config.global_block_size
        self.dropout = self.config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        q_init_std = self.config.initializer_factor * ((self.inner_dim * self.key_value_proj_dim) ** -0.5)
        kv_init_std = self.config.initializer_factor * (self.inner_dim**-0.5)
        o_init_std = self.config.initializer_factor * (self.inner_dim**-0.5)

        # Initialize query, key, value, and output dense layers with appropriate parameters
        self.q = nn.Dense(
            self.inner_dim,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(q_init_std),
            dtype=self.dtype,
        )
        self.k = nn.Dense(
            self.inner_dim,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(kv_init_std),
            dtype=self.dtype,
        )
        self.v = nn.Dense(
            self.inner_dim,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(kv_init_std),
            dtype=self.dtype,
        )
        self.o = nn.Dense(
            self.d_model,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(o_init_std),
            dtype=self.dtype,
        )

        if self.has_relative_attention_bias:
            # Initialize relative attention bias if enabled
            self.relative_attention_bias = nn.Embed(
                self.relative_attention_num_buckets,
                self.n_heads,
                embedding_init=jax.nn.initializers.normal(kv_init_std),
            )

        # Initialize global relative attention bias and layer normalization for global attention
        if self.has_relative_attention_bias:
            self.global_relative_attention_bias = nn.Embed(
                self.relative_attention_num_buckets,
                self.n_heads,
                embedding_init=jax.nn.initializers.normal(kv_init_std),
            )
        self.global_input_layer_norm = FlaxLongT5LayerNorm(
            self.config.d_model, eps=self.config.layer_norm_epsilon, dtype=self.dtype
        )

    @staticmethod
    # Static method to compute relative position bucket, adapted from transformers.models.t5.modeling_flax_t5.FlaxT5Attention._relative_position_bucket
    def _relative_position_bucket(
        x: jnp.ndarray,  # Input array for relative positions
        bidirectional: bool = True,  # Flag indicating bidirectional attention
        num_buckets: int = 32,  # Number of buckets for relative position embeddings
        max_distance: int = 128,  # Maximum distance for relative position
    ) -> jnp.ndarray:
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on
        """
        # 初始化相对位置的桶编号为0
        relative_buckets = 0
        
        # 如果是双向的相对位置注意力机制，调整桶数量，并根据正负性设置相对桶的偏移
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0) * num_buckets
            relative_position = jnp.abs(relative_position)
        else:
            # 如果是单向的相对位置注意力机制，将相对位置限制为非正数
            relative_position = -jnp.clip(relative_position, a_max=0)
        
        # 现在，relative_position 的范围在 [0, inf)

        # 设置小绝对相对位置的桶数为一半
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # 另一半的桶用于在位置增量上按对数方式增大，直到 max_distance
        relative_position_if_large = max_exact + (
            jnp.log(relative_position / max_exact) / jnp.log(max_distance / max_exact) * (num_buckets - max_exact)
        )
        relative_position_if_large = jnp.clip(relative_position_if_large, a_max=num_buckets - 1)

        # 根据相对位置大小选择合适的桶编号
        relative_buckets += jnp.where(is_small, relative_position, relative_position_if_large)

        # 将相对桶编号转换为整数类型并返回
        return relative_buckets.astype("i4")

    def compute_bias(self, block_length: int):
        """Compute binned relative position bias"""
        # 创建一个包含特定长度内存位置的数组
        memory_position = jnp.arange(3 * block_length, dtype="i4")
        
        # 从内存位置中选择与上下文相关的位置
        context_position = memory_position[block_length:-block_length]

        # 计算每对内存位置和上下文位置之间的相对位置
        relative_position = memory_position[None, :] - context_position[:, None]
        
        # 根据相对位置计算相对位置的桶编号
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=True,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )

        # 使用相对位置桶编号计算相对注意力偏置值
        values = self.relative_attention_bias(relative_position_bucket)
        
        # 调整数据维度以匹配模型要求并返回结果
        values = values.transpose((2, 0, 1))[None, None, :, :, :]
        return values
    def compute_side_bias(self, attention_mask: np.ndarray, global_segment_ids: np.ndarray) -> np.ndarray:
        # (batch_size, 1, 1, seq_len, global_seq_len)
        # 创建一个边缘注意力掩码，比较每个位置的注意力掩码和全局段落 ID 是否相等
        side_attention_mask = jnp.equal(attention_mask[..., None], global_segment_ids[:, None, :])[:, None, ...]
        # 根据边缘注意力掩码，选择性地应用注意力偏置值
        attention_side_bias = jax.lax.select(
            side_attention_mask > 0,
            jnp.full(side_attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(side_attention_mask.shape, -1e10).astype(self.dtype),
        )
        # (batch_size, seq_len, global_seq_len)
        # 计算侧边相对位置信息
        side_relative_position = _make_side_relative_position_ids(attention_mask, self.global_block_size)
        # 根据相对位置信息创建相对位置桶
        side_relative_position_bucket = self._relative_position_bucket(
            side_relative_position,
            bidirectional=True,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        # (batch_size, seq_len, global_seq_len, num_heads)
        # 计算全局相对注意力偏置
        side_bias = self.global_relative_attention_bias(side_relative_position_bucket)

        # (batch_size, 1, num_heads, seq_len, global_seq_len)
        # 调整维度顺序，以匹配注意力偏置的形状
        side_bias = jnp.transpose(side_bias, (0, 3, 1, 2))
        # (batch_size, num_heads, seq_len, global_seq_len)
        # 结合边缘注意力偏置和全局相对注意力偏置
        attention_side_bias = attention_side_bias + side_bias
        return attention_side_bias

    def _split_heads(self, hidden_states):
        # 将隐藏状态分割成多个头部
        return hidden_states.reshape(hidden_states.shape[:2] + (self.n_heads, self.key_value_proj_dim))

    def _merge_heads(self, hidden_states):
        # 将多个头部的隐藏状态合并
        return hidden_states.reshape(hidden_states.shape[0], -1, self.inner_dim)

    def _create_position_bias(self, block_len: int, attention_mask: Optional[np.ndarray]) -> np.ndarray:
        # position_bias shape: # (1, 1, n_heads, block_len, 3 * block_len)
        # 根据是否具有相对注意力偏置或注意力掩码，创建位置偏置矩阵
        if self.has_relative_attention_bias:
            position_bias = self.compute_bias(block_len)
        elif attention_mask is not None:
            position_bias = jnp.zeros_like(attention_mask)
        else:
            position_bias = jnp.zeros((1, 1, self.n_heads, block_len, 3 * block_len), dtype=self.dtype)

        return position_bias

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        key_value_states=None,
        position_bias=None,
        output_attentions=False,
        deterministic=True,
class FlaxLongT5LayerLocalSelfAttention(nn.Module):
    """Local self attention used in encoder"""

    config: LongT5Config  # 类型注解，指定配置类 LongT5Config 的实例变量
    has_relative_attention_bias: bool = False  # 是否使用相对注意力偏置，默认为 False
    dtype: jnp.dtype = jnp.float32  # 计算时使用的数据类型，默认为 jnp.float32

    def setup(self):
        self.LocalSelfAttention = FlaxLongT5LocalAttention(
            self.config, has_relative_attention_bias=self.has_relative_attention_bias, dtype=self.dtype
        )
        self.layer_norm = FlaxLongT5LayerNorm(
            self.config.d_model, eps=self.config.layer_norm_epsilon, dtype=self.dtype
        )
        self.dropout = nn.Dropout(self.config.dropout_rate)

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        output_attentions=False,
        deterministic=True,
        **kwargs: Any,  # 用于接受 init_cache 的其他参数
    ):
        normed_hidden_states = self.layer_norm(hidden_states)  # 对输入的 hidden_states 进行 layer normalization
        attention_output = self.LocalSelfAttention(
            normed_hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
            deterministic=deterministic,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0], deterministic=deterministic)
        outputs = (hidden_states,) + attention_output[1:]  # 如果输出注意力，将注意力添加到输出中
        return outputs


class FlaxLongT5LayerTransientGlobalSelfAttention(nn.Module):
    """Transient-Global self attention used in encoder"""

    config: LongT5Config  # 类型注解，指定配置类 LongT5Config 的实例变量
    has_relative_attention_bias: bool = False  # 是否使用相对注意力偏置，默认为 False
    dtype: jnp.dtype = jnp.float32  # 计算时使用的数据类型，默认为 jnp.float32

    def setup(self):
        self.TransientGlobalSelfAttention = FlaxLongT5TransientGlobalAttention(
            self.config, has_relative_attention_bias=self.has_relative_attention_bias, dtype=self.dtype
        )
        self.layer_norm = FlaxLongT5LayerNorm(
            self.config.d_model, eps=self.config.layer_norm_epsilon, dtype=self.dtype
        )
        self.dropout = nn.Dropout(self.config.dropout_rate)

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        output_attentions=False,
        deterministic=True,
        **kwargs: Any,  # 用于接受 init_cache 的其他参数
    ):
        normed_hidden_states = self.layer_norm(hidden_states)  # 对输入的 hidden_states 进行 layer normalization
        attention_output = self.TransientGlobalSelfAttention(
            normed_hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
            deterministic=deterministic,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0], deterministic=deterministic)
        outputs = (hidden_states,) + attention_output[1:]  # 如果输出注意力，将注意力添加到输出中
        return outputs


# 从 transformers.models.t5.modeling_flax_t5.FlaxT5LayerSelfAttention 复制，将 T5 替换为 LongT5
class FlaxLongT5LayerSelfAttention(nn.Module):
    config: LongT5Config
    has_relative_attention_bias: bool = False
    dtype: jnp.dtype = jnp.float32  # 计算过程中使用的数据类型

    def setup(self):
        # 初始化自注意力层，使用配置信息和设定的参数
        self.SelfAttention = FlaxLongT5Attention(
            self.config,
            has_relative_attention_bias=self.has_relative_attention_bias,
            causal=self.config.causal,
            dtype=self.dtype,
        )
        # 初始化层归一化模块，使用模型配置中的维度和层归一化的 epsilon 参数
        self.layer_norm = FlaxLongT5LayerNorm(
            self.config.d_model, eps=self.config.layer_norm_epsilon, dtype=self.dtype
        )
        # 初始化 Dropout 模块，使用模型配置中的丢弃率参数
        self.dropout = nn.Dropout(self.config.dropout_rate)

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        output_attentions=False,
        deterministic=True,
        init_cache=False,
    ):
        # 对输入的隐藏状态进行层归一化
        normed_hidden_states = self.layer_norm(hidden_states)
        # 使用自注意力层处理归一化后的隐藏状态
        attention_output = self.SelfAttention(
            normed_hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
            deterministic=deterministic,
            init_cache=init_cache,
        )
        # 将原始隐藏状态与经过 Dropout 处理后的注意力输出相加
        hidden_states = hidden_states + self.dropout(attention_output[0], deterministic=deterministic)
        # 构建输出元组，包括更新后的隐藏状态和可能的注意力输出（如果需要）
        outputs = (hidden_states,) + attention_output[1:]  # 如果输出注意力信息，将其添加到输出中
        return outputs


# 从 transformers.models.t5.modeling_flax_t5.FlaxT5LayerCrossAttention 复制并修改为 LongT5
class FlaxLongT5LayerCrossAttention(nn.Module):
    config: LongT5Config
    dtype: jnp.dtype = jnp.float32  # 计算过程中使用的数据类型

    def setup(self):
        # 初始化编码-解码注意力层，使用配置信息和设定的参数
        self.EncDecAttention = FlaxLongT5Attention(
            self.config, has_relative_attention_bias=False, causal=False, dtype=self.dtype
        )
        # 初始化层归一化模块，使用模型配置中的维度和层归一化的 epsilon 参数
        self.layer_norm = FlaxLongT5LayerNorm(
            self.config.d_model, eps=self.config.layer_norm_epsilon, dtype=self.dtype
        )
        # 初始化 Dropout 模块，使用模型配置中的丢弃率参数
        self.dropout = nn.Dropout(self.config.dropout_rate)

    def __call__(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        output_attentions=False,
        deterministic=True,
    ):
        # 对输入的隐藏状态进行层归一化
        normed_hidden_states = self.layer_norm(hidden_states)
        # 使用编码-解码注意力层处理归一化后的隐藏状态和键值状态
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            attention_mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            output_attentions=output_attentions,
        )
        # 将原始隐藏状态与经过 Dropout 处理后的注意力输出相加
        hidden_states = hidden_states + self.dropout(attention_output[0], deterministic=deterministic)
        # 构建输出元组，包括更新后的隐藏状态和可能的注意力输出（如果需要）
        outputs = (hidden_states,) + attention_output[1:]  # 如果输出注意力信息，将其添加到输出中
        return outputs


class FlaxLongT5Block(nn.Module):
    config: LongT5Config
    has_relative_attention_bias: bool = False
    dtype: jnp.dtype = jnp.float32  # 计算过程中使用的数据类型
    # 设置函数，用于初始化模型配置
    def setup(self):
        # 从配置中获取是否采用因果（causal）注意力机制
        self.causal = self.config.causal
        # 根据是否采用因果注意力机制选择不同的注意力层类型
        if self.causal:
            attention_layer = FlaxLongT5LayerSelfAttention
        elif self.config.encoder_attention_type == "local":
            attention_layer = FlaxLongT5LayerLocalSelfAttention
        elif self.config.encoder_attention_type == "transient-global":
            attention_layer = FlaxLongT5LayerTransientGlobalSelfAttention
        else:
            # 如果未知的注意力类型，则引发数值错误异常
            raise ValueError(
                "For encoder attention mechanism, either `local` or `transient-global` attention type is expected, "
                f"but got {self.config.encoder_attention_type}."
            )
        # 初始化模型的注意力层
        self.layer = (
            attention_layer(
                self.config,
                has_relative_attention_bias=self.has_relative_attention_bias,
                name=str(0),
                dtype=self.dtype,
            ),
        )
        # 初始化前馈神经网络索引
        feed_forward_index = 1
        # 如果采用因果注意力机制，则添加交叉注意力层
        if self.causal:
            self.layer += (FlaxLongT5LayerCrossAttention(self.config, name=str(1), dtype=self.dtype),)
            feed_forward_index += 1

        # 添加前馈神经网络层
        self.layer += (FlaxLongT5LayerFF(self.config, name=str(feed_forward_index), dtype=self.dtype),)

    # 从 transformers.models.t5.modeling_flax_t5.FlaxT5Block.__call__ 复制而来，修改为 LongT5 模型
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        output_attentions=False,
        return_dict=True,
        deterministic=True,
        init_cache=False,
        ):
            # 调用自注意力层处理隐藏状态，传入注意力掩码、位置偏置等参数
            self_attention_outputs = self.layer[0](
                hidden_states,
                attention_mask=attention_mask,
                position_bias=position_bias,
                output_attentions=output_attentions,
                deterministic=deterministic,
                init_cache=init_cache,
            )
            # 更新隐藏状态为自注意力层的输出
            hidden_states = self_attention_outputs[0]
            # 保留自注意力输出和相关位置权重
            attention_outputs = self_attention_outputs[1:]  # 保留自注意力输出和相关位置权重

            # 如果需要执行交叉注意力，且存在编码器的隐藏状态
            do_cross_attention = self.causal and encoder_hidden_states is not None
            if do_cross_attention:
                # 调用交叉注意力层处理隐藏状态，传入编码器的键值状态、注意力掩码等参数
                cross_attention_outputs = self.layer[1](
                    hidden_states,
                    key_value_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    position_bias=encoder_decoder_position_bias,
                    output_attentions=output_attentions,
                    deterministic=deterministic,
                )
                # 更新隐藏状态为交叉注意力层的输出
                hidden_states = cross_attention_outputs[0]

                # 保留交叉注意力输出和相关位置权重
                attention_outputs = attention_outputs + cross_attention_outputs[1:]

            # 应用前馈神经网络层处理隐藏状态
            hidden_states = self.layer[-1](hidden_states, deterministic=deterministic)

            # 组装输出元组，包含更新后的隐藏状态
            outputs = (hidden_states,)

            # 将注意力输出追加到输出元组中
            outputs = outputs + attention_outputs

            # 返回隐藏状态和可能的注意力相关数据
            # 返回的元组结构为：hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights),
            #                (cross-attention position bias), (cross-attention weights)
            return outputs
# 从transformers.models.t5.modeling_flax_t5.FlaxT5LayerCollection复制并将T5改为LongT5
class FlaxLongT5LayerCollection(nn.Module):
    config: LongT5Config  # 配置对象，包含LongT5模型的配置信息
    has_relative_attention_bias: bool  # 是否使用相对注意力偏置
    dtype: jnp.dtype = jnp.float32  # 计算时的数据类型

    def setup(self):
        self.layer = FlaxLongT5Block(  # 初始化FlaxLongT5Block层对象
            self.config, has_relative_attention_bias=self.has_relative_attention_bias, dtype=self.dtype
        )

    def __call__(
        self,
        hidden_states,  # 输入的隐藏状态
        attention_mask=None,  # 注意力掩码，控制模型关注的位置
        position_bias=None,  # 位置偏置，用于自注意力机制中的位置编码
        encoder_hidden_states=None,  # 编码器隐藏状态，用于编码-解码器注意力
        encoder_attention_mask=None,  # 编码器的注意力掩码
        encoder_decoder_position_bias=None,  # 编码器到解码器的位置偏置
        output_attentions=False,  # 是否输出注意力权重
        deterministic=True,  # 是否使用确定性推断
        init_cache=False,  # 是否初始化缓存
    ):
        return self.layer(  # 调用FlaxLongT5Block层对象进行前向传播
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            encoder_decoder_position_bias=encoder_decoder_position_bias,
            output_attentions=output_attentions,
            deterministic=deterministic,
            init_cache=init_cache,
        )


# 从transformers.models.t5.modeling_flax_t5.FlaxT5BlockCollection复制并将T5改为LongT5
class FlaxLongT5BlockCollection(nn.Module):
    config: LongT5Config  # 配置对象，包含LongT5模型的配置信息
    dtype: jnp.dtype = jnp.float32  # 计算时的数据类型
    gradient_checkpointing: bool = False  # 是否使用梯度检查点技术

    def setup(self):
        self.causal = self.config.causal  # 是否使用因果（自回归）模式
        if self.gradient_checkpointing:
            FlaxLongT5CheckpointLayer = remat(FlaxLongT5LayerCollection, static_argnums=(6, 7, 8))
            self.blocks = [
                FlaxLongT5CheckpointLayer(  # 初始化带梯度检查点的LongT5层对象
                    self.config,
                    has_relative_attention_bias=(i == 0),
                    dtype=self.dtype,
                    name=str(i),
                )
                for i in range(self.config.num_layers)
            ]
        else:
            self.blocks = [
                FlaxLongT5LayerCollection(  # 初始化LongT5层对象列表
                    self.config,
                    has_relative_attention_bias=(i == 0),
                    dtype=self.dtype,
                    name=str(i),
                )
                for i in range(self.config.num_layers)
            ]

    def __call__(
        self,
        hidden_states=None,  # 输入的隐藏状态
        attention_mask=None,  # 注意力掩码，控制模型关注的位置
        encoder_hidden_states=None,  # 编码器隐藏状态，用于编码-解码器注意力
        encoder_attention_mask=None,  # 编码器的注意力掩码
        output_attentions: bool = False,  # 是否输出注意力权重
        output_hidden_states: bool = False,  # 是否输出隐藏状态
        deterministic: bool = True,  # 是否使用确定性推断
        init_cache: bool = False,  # 是否初始化缓存
        # 准备需要的头部掩码
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.causal) else None
        position_bias = None
        encoder_decoder_position_bias = None

        # 遍历每个 Transformer 模块
        for i, layer_module in enumerate(self.blocks):
            if output_hidden_states:
                # 如果需要输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 调用当前层的 Transformer 模块进行前向传播
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                position_bias,
                encoder_hidden_states,
                encoder_attention_mask,
                encoder_decoder_position_bias,
                output_attentions,
                deterministic,
                init_cache,
            )

            # 更新隐藏状态为当前层输出的隐藏状态
            hidden_states = layer_outputs[0]

            # 更新位置偏置为当前层输出的自注意力位置偏置
            position_bias = layer_outputs[1]

            # 如果是因果的并且有编码器隐藏状态，则更新编码器-解码器位置偏置
            if self.causal and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[3 if output_attentions else 2]

            # 如果需要输出注意力权重，则将当前层的注意力权重添加到 all_attentions 中
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)
                # 如果是因果的，则将当前层的交叉注意力权重添加到 all_cross_attentions 中
                if self.causal:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[4],)

        # 返回经过所有 Transformer 层处理后的输出
        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )
# 从transformers.models.t5.modeling_flax_t5.FlaxT5Stack复制并修改为LongT5Stack
class FlaxLongT5Stack(nn.Module):
    # 配置参数
    config: LongT5Config
    # 词嵌入层
    embed_tokens: nn.Embed
    # 计算中使用的数据类型，默认为jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 梯度检查点，默认关闭
    gradient_checkpointing: bool = False

    # 初始化方法
    def setup(self):
        # 是否是因果（causal）模型
        self.causal = self.config.causal
        # 创建LongT5BlockCollection块
        self.block = FlaxLongT5BlockCollection(
            self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )
        # 最终层归一化
        self.final_layer_norm = FlaxLongT5LayerNorm(
            self.config.d_model, eps=self.config.layer_norm_epsilon, dtype=self.dtype
        )
        # Dropout层
        self.dropout = nn.Dropout(self.config.dropout_rate)

    # 调用方法，处理输入并生成输出
    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
        init_cache: bool = False,
    ):
        # 获取词嵌入表示
        hidden_states = self.embed_tokens(input_ids)
        # 应用Dropout
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)

        # 调用块对象处理隐藏状态
        outputs = self.block(
            hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            deterministic=deterministic,
            init_cache=init_cache,
        )

        # 从块的输出中获取隐藏状态
        hidden_states = outputs[0]

        # 应用最终层归一化
        hidden_states = self.final_layer_norm(hidden_states)
        # 再次应用Dropout
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)

        # 添加最后一层隐藏状态（用于返回所有隐藏状态时）
        all_hidden_states = None
        if output_hidden_states:
            all_hidden_states = outputs.hidden_states
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 根据返回类型构建输出
        if not return_dict:
            if output_hidden_states:
                return (
                    hidden_states,
                    all_hidden_states,
                ) + outputs[2:]  # 返回隐藏状态及额外输出
            return (hidden_states,) + outputs[1:]  # 仅返回隐藏状态

        # 返回带过去和交叉注意力的基本模型输出
        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


# 以下是LongT5编码器输入的文档字符串
LONGT5_ENCODE_INPUTS_DOCSTRING = r"""
    # 接收输入参数的函数定义，用于处理LongT5模型的输入数据
    Args:
        input_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
            输入序列标记在词汇表中的索引。LongT5模型具有相对位置嵌入，因此可以在序列的左右两侧进行填充。
    
            可以使用[`AutoTokenizer`]获取索引。详见[`PreTrainedTokenizer.encode`]和[`PreTrainedTokenizer.__call__`]。
    
            想要了解有关如何为预训练准备`input_ids`的更多信息，请查看[长T5训练](./longt5#training)。
        attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            遮罩，用于避免在填充标记索引上执行注意力操作。遮罩值选在 `[0, 1]` 之间：
    
            - 对于**未被遮罩**的标记，值为1，
            - 对于**被遮罩**的标记，值为0。
    
            [什么是注意力遮罩?](../glossary#attention-mask)
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。查看返回的张量中的`attentions`以获取更多详细信息。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。查看返回的张量中的`hidden_states`以获取更多详细信息。
        return_dict (`bool`, *optional*):
            是否返回[`~utils.ModelOutput`]而不是普通元组。
"""
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
"""

class FlaxLongT5PreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定配置类为 LongT5Config
    config_class = LongT5Config
    # 基础模型前缀为 "transformer"
    base_model_prefix = "transformer"
    # 模块类默认为空
    module_class: nn.Module = None
    # 初始化方法，用于创建一个 LongT5 模型实例
    def __init__(
        self,
        config: LongT5Config,
        input_shape: Tuple[int] = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        # 使用给定的配置和参数创建模型类实例
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 调用父类的初始化方法，传入配置、模型类实例以及其他参数
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    # 启用梯度检查点功能，重新设置模型实例以支持梯度检查点
    def enable_gradient_checkpointing(self):
        self._module = self.module_class(
            config=self.config,
            dtype=self.dtype,
            gradient_checkpointing=True,
        )

    # 初始化模型权重方法，使用随机数种子 rng 和输入形状 input_shape
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量 input_ids
        input_ids = jnp.zeros(input_shape, dtype="i4")

        # 创建 attention_mask，decoder_input_ids 和 decoder_attention_mask 张量
        attention_mask = jnp.ones_like(input_ids)
        decoder_input_ids = jnp.ones_like(input_ids)
        decoder_attention_mask = jnp.ones_like(input_ids)

        # 分割随机数种子 rng 为 params_rng 和 dropout_rng
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 使用初始化方法初始化模型参数 random_params
        random_params = self.module.init(
            rngs,
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
        )["params"]

        # 如果提供了初始参数 params，则将缺失的参数从 random_params 中补充到 params 中
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    # 重写 __call__ 方法，并添加了文档字符串装饰器 LONGT5_INPUTS_DOCSTRING
    @add_start_docstrings_to_model_forward(LONGT5_INPUTS_DOCSTRING)
    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        decoder_input_ids: jnp.ndarray = None,
        decoder_attention_mask: Optional[jnp.ndarray] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
        ):
            # 如果未指定output_attentions，则使用配置中的默认值
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            # 如果未指定output_hidden_states，则使用配置中的默认值
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            # 如果未指定return_dict，则使用配置中的默认值
            return_dict = return_dict if return_dict is not None else self.config.return_dict

            # 如果decoder_input_ids未提供，则抛出值错误
            if decoder_input_ids is None:
                raise ValueError(
                    "Make sure to provide both `input_ids` and `decoder_input_ids`. `decoder_input_ids` is not passed"
                    " here."
                )

            # 准备编码器输入的注意力掩码
            if attention_mask is None:
                attention_mask = jnp.ones_like(input_ids)

            # 准备解码器输入的注意力掩码
            if decoder_attention_mask is None:
                decoder_attention_mask = jnp.ones_like(decoder_input_ids)

            # 如果需要处理任何伪随机数生成器，则放入rngs字典中
            rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

            # 调用self.module.apply方法来执行模型
            return self.module.apply(
                {"params": params or self.params},  # 模型参数
                input_ids=jnp.array(input_ids, dtype="i4"),  # 编码器输入的token IDs
                attention_mask=jnp.array(attention_mask, dtype="i4"),  # 编码器输入的注意力掩码
                decoder_input_ids=jnp.array(decoder_input_ids, dtype="i4"),  # 解码器输入的token IDs
                decoder_attention_mask=jnp.array(decoder_attention_mask, dtype="i4"),  # 解码器输入的注意力掩码
                output_attentions=output_attentions,  # 是否返回注意力权重
                output_hidden_states=output_hidden_states,  # 是否返回隐藏状态
                return_dict=return_dict,  # 是否返回字典形式的输出
                deterministic=not train,  # 是否确定性执行（非训练状态）
                rngs=rngs,  # 伪随机数生成器字典
            )
    def init_cache(self, batch_size, max_length, encoder_outputs):
        r"""
        Args:
            batch_size (`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
            encoder_outputs (`Union[FlaxBaseModelOutput, tuple(tuple(jnp.ndarray)]`):
                `encoder_outputs` consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*:
                `attentions`). `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*)
                is a sequence of hidden-states at the output of the last layer of the encoder. Used in the
                cross-attention of the decoder.
        """
        # 初始化输入变量以检索缓存
        decoder_input_ids = jnp.ones((batch_size, max_length), dtype="i4")
        decoder_attention_mask = jnp.ones_like(decoder_input_ids)

        def _decoder_forward(module, decoder_input_ids, decoder_attention_mask, **kwargs):
            decoder_module = module._get_decoder_module()
            return decoder_module(
                decoder_input_ids,
                decoder_attention_mask,
                **kwargs,
            )

        # 使用指定的参数初始化模型，并获取初始化后的变量
        init_variables = self.module.init(
            jax.random.PRNGKey(0),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            init_cache=True,
            method=_decoder_forward,  # 只需调用解码器以初始化缓存
        )
        # 返回冻结的缓存变量
        return unfreeze(init_variables["cache"])

    @add_start_docstrings(LONGT5_ENCODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutput, config_class=LongT5Config)
    def encode(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
        ):
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, FlaxLongT5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
        >>> model = FlaxLongT5ForConditionalGeneration.from_pretrained("google/long-t5-local-base")

        >>> text = "My friends are cool but they eat too many carbs."
        >>> inputs = tokenizer(text, return_tensors="np")
        >>> encoder_outputs = model.encode(**inputs)
        ```"""
        # 如果没有显式指定，则使用默认配置中的值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 如果 attention_mask 为 None，则创建一个全为 1 的掩码
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        # 如果有 dropout_rng，则加入 RNG 字典中
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        # 定义内部函数 _encoder_forward，用于调用编码器模块
        def _encoder_forward(module, input_ids, attention_mask, **kwargs):
            encode_module = module._get_encoder_module()
            return encode_module(input_ids, attention_mask, **kwargs)

        # 调用 Flax 模块的 apply 方法进行前向传播
        return self.module.apply(
            {"params": params or self.params},  # 使用给定的参数或者当前实例的参数
            input_ids=jnp.array(input_ids, dtype="i4"),  # 转换输入 ids 到 JAX 数组格式
            attention_mask=jnp.array(attention_mask, dtype="i4"),  # 转换注意力掩码到 JAX 数组格式
            output_attentions=output_attentions,  # 是否返回注意力权重
            output_hidden_states=output_hidden_states,  # 是否返回隐藏状态
            return_dict=return_dict,  # 是否返回字典形式的输出
            deterministic=not train,  # 是否是确定性运行模式，非训练状态
            rngs=rngs,  # 随机数生成器字典
            method=_encoder_forward,  # 调用的方法，这里是编码器的前向传播函数
        )
    
    @add_start_docstrings(LONGT5_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutputWithPastAndCrossAttentions, config_class=LongT5Config)
    def decode(
        self,
        decoder_input_ids,
        encoder_outputs,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        decoder_attention_mask: Optional[jnp.ndarray] = None,
        past_key_values: dict = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
"""
    The LongT5 model was proposed in [LongT5: Efficient Text-To-Text Transformer for Long
    Sequences](https://arxiv.org/abs/2112.07916) by Mandy Guo, Joshua Ainslie, David Uthus, Santiago Ontanon, Jianmo
    Ni, Yun-Hsuan Sung and Yinfei Yang. It's an encoder-decoder transformer pre-trained in a text-to-text denoising
    generative setting. LongT5 model is an extension of T5 model, and it enables using one of the two different
    efficient attention mechanisms - (1) Local attention, or (2) Transient-Global attention.

    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a Flax Linen
    [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a
    regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.

    Finally, this model supports inherent JAX features such as:

    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

    Parameters:
        config ([`LongT5Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.
        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
            The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and
            `jax.numpy.bfloat16` (on TPUs).

            This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
            specified all the computation will be performed with the given `dtype`.

            **Note that this only specifies the dtype of the computation and does not influence the dtype of model
            parameters.**

            If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`] and
            [`~FlaxPreTrainedModel.to_bf16`].
"""

@add_start_docstrings(
    "The bare LONGT5 Model transformer outputting raw hidden-stateswithout any specific head on top.",
    LONGT5_START_DOCSTRING,
)
# Copied from transformers.models.t5.modeling_flax_t5.FlaxT5Module with T5->LongT5
class FlaxLongT5Module(nn.Module):
    """
    Flax module for the LongT5 model, extending the T5 architecture to support long sequences and different attention mechanisms.

    Inherits from `nn.Module`, enabling it to be used as a Flax Linen module. Refer to Flax documentation for usage details.

    Attributes:
        config (LongT5Config): Model configuration object containing all parameters.
        dtype (jnp.dtype): Data type for computation, defaulting to jnp.float32.
    """
    config: LongT5Config
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    # 梯度检查点标志，默认为 False
    gradient_checkpointing: bool = False
    
    # 获取编码器模块的方法
    def _get_encoder_module(self):
        return self.encoder
    
    # 获取解码器模块的方法
    def _get_decoder_module(self):
        return self.decoder
    
    # 设置方法，用于初始化和配置模型
    def setup(self):
        # 创建共享的嵌入层，用于输入和输出的词汇表
        self.shared = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.initializer_factor * 1.0),
            dtype=self.dtype,
        )
    
        # 复制编码器配置，并禁用因果关系，创建编码器对象
        encoder_config = copy.deepcopy(self.config)
        encoder_config.causal = False
        self.encoder = FlaxLongT5Stack(
            encoder_config,
            embed_tokens=self.shared,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
    
        # 复制解码器配置，并启用因果关系，根据配置创建解码器对象
        decoder_config = copy.deepcopy(self.config)
        decoder_config.causal = True
        decoder_config.num_layers = self.config.num_decoder_layers
        self.decoder = FlaxLongT5Stack(
            decoder_config,
            embed_tokens=self.shared,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
    
    # 调用方法，实现模型的前向传播
    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        deterministic: bool = True,
    ):
        # 如果未指定返回字典，则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # 编码器的前向传播，生成编码器输出
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )
    
        # 解码器的前向传播，生成解码器输出
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],  # 使用编码器的隐藏状态作为解码器的输入
            encoder_attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )
    
        # 如果不需要返回字典，则将编码器和解码器的输出合并返回
        if not return_dict:
            return decoder_outputs + encoder_outputs
    
        # 构造并返回序列到序列模型的输出
        return FlaxSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
# 从transformers.models.t5.modeling_flax_t5.FlaxT5Model复制代码，将T5替换为LongT5
class FlaxLongT5Model(FlaxLongT5PreTrainedModel):
    # 使用FlaxLongT5Module作为模块类
    module_class = FlaxLongT5Module

# 将FlaxLongT5Model的调用示例文档字符串附加到_CHECKPOINT_FOR_DOC并使用FlaxSeq2SeqModelOutput进行文档化
append_call_sample_docstring(FlaxLongT5Model, _CHECKPOINT_FOR_DOC, FlaxSeq2SeqModelOutput, _CONFIG_FOR_DOC)

# 定义FLAX_LONGT5_MODEL_DOCSTRING，包含函数的返回值和示例用法
FLAX_LONGT5_MODEL_DOCSTRING = """
    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, FlaxLongT5Model

    >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
    >>> model = FlaxLongT5Model.from_pretrained("google/long-t5-local-base")

    >>> input_ids = tokenizer(
    ...     "Studies have been shown that owning a dog is good for you", return_tensors="np"
    ... ).input_ids
    >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="np").input_ids

    >>> # forward pass
    >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
    >>> last_hidden_states = outputs.last_hidden_state
    ```
"""

# 重写FlaxLongT5Model的调用文档字符串，使用LONGT5_INPUTS_DOCSTRING和FLAX_LONGT5_MODEL_DOCSTRING
overwrite_call_docstring(FlaxLongT5Model, LONGT5_INPUTS_DOCSTRING + FLAX_LONGT5_MODEL_DOCSTRING)

# 将FlaxLongT5Model的返回文档字符串替换为FlaxSeq2SeqLMOutput，使用_CONFIG_FOR_DOC作为配置类
append_replace_return_docstrings(FlaxLongT5Model, output_type=FlaxSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)

@add_start_docstrings("""LONGT5 Model with a `language modeling` head on top.""", LONGT5_START_DOCSTRING)
# 从transformers.models.t5.modeling_flax_t5.FlaxT5ForConditionalGenerationModule复制代码，将T5替换为LongT5
class FlaxLongT5ForConditionalGenerationModule(nn.Module):
    # 配置为LongT5Config类型
    config: LongT5Config
    # 计算中的数据类型为jnp.float32
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    # 梯度检查点默认为False
    gradient_checkpointing: bool = False

    # 返回编码器模块
    def _get_encoder_module(self):
        return self.encoder

    # 返回解码器模块
    def _get_decoder_module(self):
        return self.decoder

    # 模块设置函数
    def setup(self):
        # 模型维度为配置中的d_model
        self.model_dim = self.config.d_model

        # 创建共享的嵌入层，使用正态分布初始化
        self.shared = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.initializer_factor),
            dtype=self.dtype,
        )

        # 复制编码器配置，并设定特定参数
        encoder_config = copy.deepcopy(self.config)
        encoder_config.causal = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # 创建LongT5编码器栈
        self.encoder = FlaxLongT5Stack(
            encoder_config, self.shared, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )

        # 复制解码器配置，并设定特定参数
        decoder_config = copy.deepcopy(self.config)
        decoder_config.causal = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = self.config.num_decoder_layers
        # 创建LongT5解码器栈
        self.decoder = FlaxLongT5Stack(
            decoder_config, self.shared, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )

        # 创建语言模型头部，全连接层
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_factor),
            dtype=self.dtype,
        )
    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        deterministic: bool = True,
    ):
        # 确保返回字典的设置正确
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode
        # 调用编码器进行编码
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        hidden_states = encoder_outputs[0]

        # Decode
        # 调用解码器进行解码
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            # 如果需要共享词嵌入，则按比例缩放输出
            # 参考：https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        if self.config.tie_word_embeddings:
            # 如果需要共享词嵌入，则应用共享的嵌入层
            shared_embedding = self.shared.variables["params"]["embedding"]
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_embedding.T}}, sequence_output)
        else:
            lm_logits = self.lm_head(sequence_output)

        if not return_dict:
            # 如果不需要返回字典，则返回元组形式的结果
            return (lm_logits,) + decoder_outputs[1:] + encoder_outputs

        # 如果需要返回字典，则构造 FlaxSeq2SeqLMOutput 对象
        return FlaxSeq2SeqLMOutput(
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
class FlaxLongT5ForConditionalGeneration(FlaxLongT5PreTrainedModel):
    # 模型类指定为 FlaxLongT5ForConditionalGenerationModule
    module_class = FlaxLongT5ForConditionalGenerationModule

    @add_start_docstrings(LONGT5_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxCausalLMOutputWithCrossAttentions, config_class=LongT5Config)
    def decode(
        self,
        decoder_input_ids,
        encoder_outputs,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        decoder_attention_mask: Optional[jnp.ndarray] = None,
        past_key_values: dict = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
    ):
        """
        解码函数，根据给定的输入和条件生成输出。

        Args:
            decoder_input_ids: 解码器的输入 ID。
            encoder_outputs: 编码器的输出。
            encoder_attention_mask: 可选，编码器的注意力遮罩。
            decoder_attention_mask: 可选，解码器的注意力遮罩。
            past_key_values: 可选，过去的键值对，用于加速生成。
            output_attentions: 可选，是否输出注意力权重。
            output_hidden_states: 可选，是否输出隐藏状态。
            return_dict: 可选，是否以字典形式返回输出。
            train: 是否处于训练模式。
            params: 可选，模型参数。
            dropout_rng: 可选，用于 dropout 的随机数生成器。

        Returns:
            解码后的输出结果。
        """
        ...

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        max_length,
        attention_mask: Optional[jax.Array] = None,
        decoder_attention_mask: Optional[jax.Array] = None,
        encoder_outputs=None,
        **kwargs,
    ):
        """
        为生成过程准备输入，初始化缓存并生成注意力掩码。

        Args:
            decoder_input_ids: 解码器的输入 ID。
            max_length: 生成的最大长度。
            attention_mask: 可选，编码器的注意力遮罩。
            decoder_attention_mask: 可选，解码器的注意力遮罩。
            encoder_outputs: 可选，编码器的输出。
            **kwargs: 其他关键字参数。

        Returns:
            包含生成过程所需输入的字典。
        """
        # 初始化缓存
        batch_size, seq_length = decoder_input_ids.shape
        past_key_values = self.init_cache(batch_size, max_length, encoder_outputs)

        # 创建扩展的注意力掩码
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if decoder_attention_mask is not None:
            extended_attention_mask = jax.lax.dynamic_update_slice(
                extended_attention_mask, decoder_attention_mask, (0, 0)
            )

        return {
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "encoder_attention_mask": attention_mask,
            "decoder_attention_mask": extended_attention_mask,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        """
        更新生成过程的输入，将过去的键值对更新为模型输出的过去键值对。

        Args:
            model_outputs: 模型的输出。
            model_kwargs: 模型的关键字参数。

        Returns:
            更新后的模型关键字参数。
        """
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        return model_kwargs


FLAX_LONGT5_CONDITIONAL_GENERATION_DOCSTRING = """
    Returns:
        生成的结果。

    Example:

    ```python
    >>> from transformers import AutoTokenizer, FlaxLongT5ForConditionalGeneration

    >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
    >>> model = FlaxLongT5ForConditionalGeneration.from_pretrained("google/long-t5-local-base")

    >>> ARTICLE_TO_SUMMARIZE = "summarize: My friends are cool but they eat too many carbs."
    >>> inputs = tokenizer([ARTICLE_TO_SUMMARIZE], return_tensors="np")

    >>> # 生成摘要
    >>> summary_ids = model.generate(inputs["input_ids"]).sequences
    >>> print(tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False))
    ```
"""

overwrite_call_docstring(
    # 导入 FlaxLongT5ForConditionalGeneration 类以及相关文档字符串
    FlaxLongT5ForConditionalGeneration, LONGT5_INPUTS_DOCSTRING + FLAX_LONGT5_CONDITIONAL_GENERATION_DOCSTRING
# 调用函数 `append_replace_return_docstrings`，传入参数 `FlaxLongT5ForConditionalGeneration` 作为第一个位置参数，
# `output_type=FlaxSeq2SeqLMOutput` 作为关键字参数 `output_type` 的值，
# `_CONFIG_FOR_DOC` 作为关键字参数 `config_class` 的值。
append_replace_return_docstrings(
    FlaxLongT5ForConditionalGeneration, output_type=FlaxSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC
)
```