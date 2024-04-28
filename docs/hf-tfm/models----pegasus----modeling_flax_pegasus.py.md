# `.\transformers\models\pegasus\modeling_flax_pegasus.py`

```
# 指定编码为 UTF-8
# 版权声明
# 版权所有 © 2021，Google 和 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版（“许可证”）获得许可;
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按“原样”提供，不提供任何形式的保证或条件，
# 包括但不限于适销性和特定用途适用性的任何保证或条件。
# 有关更多详细信息，请参阅许可证。
""" Flax PEGASUS 模型。"""


import math
import random
from functools import partial  # 导入 partial 函数，用于创建偏函数
from typing import Callable, Optional, Tuple  # 导入类型提示

import flax.linen as nn  # 导入 Flax 的 Linen 模块，用于定义模型
import jax  # 导入 JAX 库
import jax.numpy as jnp  # 导入 JAX 的 NumPy 模块，用于数组操作
import numpy as np  # 导入 NumPy 库
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze  # 导入冻结字典相关的函数
from flax.linen import combine_masks, make_causal_mask  # 导入用于创建掩码的函数
from flax.linen.attention import dot_product_attention_weights  # 导入点积注意力权重计算函数
from flax.traverse_util import flatten_dict, unflatten_dict  # 导入用于扁平化和反扁平化字典的函数
from jax import lax  # 导入 JAX 的 lax 模块，用于定义低级操作
from jax.random import PRNGKey  # 导入 PRNGKey 用于生成伪随机数

from ...modeling_flax_outputs import (
    FlaxBaseModelOutput,  # 导入基本模型输出类
    FlaxBaseModelOutputWithPastAndCrossAttentions,  # 导入包含过去和交叉注意力的基本模型输出类
    FlaxCausalLMOutputWithCrossAttentions,  # 导入包含交叉注意力的因果语言模型输出类
    FlaxSeq2SeqLMOutput,  # 导入序列到序列语言模型输出类
    FlaxSeq2SeqModelOutput,  # 导入序列到序列模型输出类
)
from ...modeling_flax_utils import (
    ACT2FN,  # 导入激活函数到函数映射的字典
    FlaxPreTrainedModel,  # 导入 Flax 预训练模型类
    add_start_docstrings_to_model_forward,  # 导入用于添加模型前向方法文档字符串的函数
    append_call_sample_docstring,  # 导入用于添加调用示例文档字符串的函数
    append_replace_return_docstrings,  # 导入用于添加替换返回文档字符串的函数
    overwrite_call_docstring,  # 导入用于覆盖调用文档字符串的函数
)
from ...utils import add_start_docstrings, logging, replace_return_docstrings  # 导入用于添加文档字符串的辅助函数

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

_CHECKPOINT_FOR_DOC = "google/pegasus-large"  # 用于文档的检查点名称
_CONFIG_FOR_DOC = "PegasusConfig"  # 用于文档的配置名称

PEGASUS_START_DOCSTRING = r"""
    此模型继承自 [`FlaxPreTrainedModel`]。请查看超类文档以获取库实现的通用方法（例如下载或保存、调整输入嵌入、修剪头等）。

    此模型还是一个 Flax Linen
    [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) 的子类。可以将其视为常规 Flax 模块，并参考 Flax 文档了解有关通用使用和行为的所有事项。

    最后，此模型支持 JAX 的内在特性，例如：

    - [即时编译 (JIT)](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [自动微分](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [向量化](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [并行化](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)
"""  # PEGASUS 模型的开始文档字符串，包含对模型功能和特性的介绍
   
# 该文档字符串描述了 PEGASUS 模型输入的参数。
PEGASUS_INPUTS_DOCSTRING = r"""
    Args:
        # 输入序列的词索引，用于表示输入序列。padding 会被默认忽略。
        input_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        # 注意力掩码，用于避免在 padding 位置上执行注意力计算。值为 0 表示 masked，1 表示 not masked。 
        attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        # 解码器输入序列的词索引，用于表示目标序列。
        decoder_input_ids (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)
        # 解码器注意力掩码，默认会忽略 padding 位置，并使用因果掩码。
        decoder_attention_mask (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.

            If you want to change padding behavior, you should modify to your needs. See diagram 1 in [the
            paper](https://arxiv.org/abs/1910.13461) for more information on the default strategy.
        # 输入序列和解码器输入序列的位置索引。
        position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.
        decoder_position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the
            range `[0, config.max_position_embeddings - 1]`.
        # 是否返回注意力权重和隐藏状态。
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        # 是否返回 ModelOutput 而不是普通元组。
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# 这是 PEGASUS 编码器输入的文档字符串。
PEGASUS_ENCODE_INPUTS_DOCSTRING = r"""
        Args:
            input_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
                输入序列标记在词汇表中的索引。默认情况下，填充将被忽略。

                可以使用 [`AutoTokenizer`] 获得这些索引。有关详情，请参阅 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`]。

                [什么是输入 ID？](../glossary#input-ids)
            attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
                避免在填充标记索引上执行注意力的掩码。掩码值选在 `[0, 1]` 范围内：

                - 1 表示**未屏蔽**的标记，
                - 0 表示**屏蔽**的标记。

                [什么是注意力掩码？](../glossary#attention-mask)
            position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
                每个输入序列标记在位置嵌入中的位置索引。选在范围 `[0, config.max_position_embeddings - 1]` 内。
            output_attentions (`bool`, *optional*):
                是否返回所有注意力层的注意力张量。有关更多详细信息，请参阅返回的张量中的 `attentions`。
            output_hidden_states (`bool`, *optional*):
                是否返回所有层的隐藏状态。有关更多详细信息，请参阅返回的张量中的 `hidden_states`。
            return_dict (`bool`, *optional*):
                是否返回 [`~utils.ModelOutput`] 而不是普通元组。
```  
"""
PEGASUS_DECODE_INPUTS_DOCSTRING = r"""
    Args:
        decoder_input_ids (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`):
            解码器输入序列标记在词汇表中的索引。

            可以使用 [`AutoTokenizer`] 获取索引。有关详细信息，请参阅 [`PreTrainedTokenizer.encode`] 和
            [`PreTrainedTokenizer.__call__`]。

            [什么是解码器输入 ID？](../glossary#decoder-input-ids)
        encoder_outputs (`tuple(tuple(jnp.ndarray)`):
            元组包括 (`last_hidden_state`, *可选*: `hidden_states`, *可选*: `attentions`)
            `last_hidden_state` 的形状为 `(batch_size, sequence_length, hidden_size)`，*可选*）是编码器最后一层的输出的隐藏状态序列。在解码器的交叉注意力中使用。
        encoder_attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *可选*):
            避免对填充标记索引进行注意力计算的掩码。掩码值选择在 `[0, 1]` 区间：

            - 对于**未掩码**的标记，取值为 1，
            - 对于**掩码**的标记，取值为 0。

            [什么是注意力掩码？](../glossary#attention-mask)
        decoder_attention_mask (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`, *可选*):
            默认行为：生成一个忽略 `decoder_input_ids` 中填充标记的张量。默认情况下也将使用因果掩码。

            如果要更改填充行为，应根据需求修改。有关默认策略的更多信息，请参见[论文中的图 1](https://arxiv.org/abs/1910.13461)。
        decoder_position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *可选*):
            每个解码器输入序列标记在位置嵌入中的位置索引。选择范围为 `[0, config.max_position_embeddings - 1]`。
        past_key_values (`Dict[str, np.ndarray]`, *可选*, 由 `init_cache` 返回或传递先前 `past_key_values` 时返回):
            预先计算的隐藏状态（注意力块中的键和值）的字典，可用于快速自回归解码。预先计算的键和值隐藏状态的形
# 将输入 IDs 向右移动一个位置
def shift_tokens_right(input_ids: jnp.ndarray, pad_token_id: int, decoder_start_token_id: int) -> jnp.ndarray:
    """
    将输入序列向右移动一个位置。
    """
    # 创建一个与 input_ids 形状相同的全零数组
    shifted_input_ids = jnp.zeros_like(input_ids)
    # 将 input_ids 的前 n-1 个元素复制到 shifted_input_ids 的第 2 到最后一个元素
    shifted_input_ids = shifted_input_ids.at[:, 1:].set(input_ids[:, :-1])
    # 将 decoder_start_token_id 设置为 shifted_input_ids 的第一个元素
    shifted_input_ids = shifted_input_ids.at[:, 0].set(decoder_start_token_id)
    # 将 shifted_input_ids 中值为 -100 的元素替换为 pad_token_id
    shifted_input_ids = jnp.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)
    return shifted_input_ids


# 创建正弦位置编码
def create_sinusoidal_positions(n_pos, dim):
    # 创建 n_pos 行 dim 列的位置编码矩阵
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    # 计算 sentinel 值，用于确定何时使用正弦和余弦
    sentinel = dim // 2 + dim % 2
    # 创建与 position_enc 相同形状的输出数组
    out = np.zeros_like(position_enc)
    # 将 position_enc 的偶数列设置为正弦值，奇数列设置为余弦值
    out[:, 0:sentinel] = np.sin(position_enc[:, 0::2])
    out[:, sentinel:] = np.cos(position_enc[:, 1::2])
    # 将输出转换为 jnp.array 并返回
    return jnp.array(out)


# 多头注意力机制
class FlaxPegasusAttention(nn.Module):
    config: PegasusConfig
    embed_dim: int
    num_heads: int
    dropout: float = 0.0
    causal: bool = False
    bias: bool = True
    dtype: jnp.dtype = jnp.float32  # 计算使用的数据类型

    def setup(self) -> None:
        # 计算每个注意力头的维度
        self.head_dim = self.embed_dim // self.num_heads
        # 如果 embed_dim 不能被 num_heads 整除，则抛出错误
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )

        # 创建用于计算查询、键和值的全连接层
        dense = partial(
            nn.Dense,
            self.embed_dim,
            use_bias=self.bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        self.q_proj, self.k_proj, self.v_proj = dense(), dense(), dense()
        # 创建用于计算输出的全连接层
        self.out_proj = dense()

        # 创建dropout层
        self.dropout_layer = nn.Dropout(rate=self.dropout)

        # 如果需要因果掩码，则创建掩码
        if self.causal:
            self.causal_mask = make_causal_mask(
                jnp.ones((1, self.config.max_position_embeddings), dtype="bool"), dtype="bool"
            )

    # 将输入拆分成多个注意力头
    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    # 将多个注意力头合并为一个输出
    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    @nn.compact
    # 将来自单个输入标记的投影键和值状态与先前步骤中缓存的状态连接起来
    # 这个函数是从官方的 Flax 仓库中稍微改编的
    # https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        # 检测我们是否通过缺少现有缓存数据来进行初始化
        is_initialized = self.has_variable("cache", "cached_key")
        # 如果已经初始化，则获取缓存的键和值，如果没有则创建新的缓存
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        # 获取或创建缓存的索引
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            # 获取缓存键的形状信息
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # 使用新的1D空间切片更新键，值缓存
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            # 更新键，值缓存
            cached_key.value = key
            cached_value.value = value
            # 更新缓存向量的数量，并更新缓存索引
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # 用于缓存的解码器 self-attention 的因果掩码：我们单个查询位置应该只关注已经生成并缓存的那些键位置，而不是剩下的零元素
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)
        # 返回连接过的键，值以及注意力掩码
        return key, value, attention_mask

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        key_value_states: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        deterministic: bool = True,
# 将 MBart 模型的编码器层转为 Pegasus 模型的编码器层
class FlaxPegasusEncoderLayer(nn.Module):
    # 定义 Pegasus 配置和数据类型
    config: PegasusConfig
    dtype: jnp.dtype = jnp.float32

    # 初始化函数
    def setup(self) -> None:
        # 获取嵌入维度
        self.embed_dim = self.config.d_model
        # 创建 Pegasus 注意力层对象
        self.self_attn = FlaxPegasusAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.encoder_attention_heads,
            dropout=self.config.attention_dropout,
            dtype=self.dtype,
        )
        # 创建层归一化对象
        self.self_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        # 创建 dropout 层
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)
        # 获取激活函数
        self.activation_fn = ACT2FN[self.config.activation_function]
        # 创建激活函数的 dropout 层
        self.activation_dropout_layer = nn.Dropout(rate=self.config.activation_dropout)
        # 创建全连接层1
        self.fc1 = nn.Dense(
            self.config.encoder_ffn_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        # 创建全连接层2
        self.fc2 = nn.Dense(
            self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std)
        )
        # 创建最终层归一化对象
        self.final_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    # 调用函数
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        output_attentions: bool = True,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray]:
        # 备份隐藏状态
        residual = hidden_states
        # 对隐藏状态进行层归一化
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # 进行自注意力计算
        hidden_states, attn_weights = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask)
        # 经过 dropout
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 残差连接
        hidden_states = residual + hidden_states

        # 备份隐藏状态
        residual = hidden_states
        # 对隐藏状态进行最终层归一化
        hidden_states = self.final_layer_norm(hidden_states)
        # 使用激活函数
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 经过激活函数的 dropout
        hidden_states = self.activation_dropout_layer(hidden_states, deterministic=deterministic)
        # 全连接层2
        hidden_states = self.fc2(hidden_states)
        # 进行 dropout
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 残差连接
        hidden_states = residual + hidden_states

        # 输出结果
        outputs = (hidden_states,)

        # 如果需要输出注意力权重
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# 将 Bart 模型的编码器层集合转为 Pegasus 模型的编码器层集合
class FlaxPegasusEncoderLayerCollection(nn.Module):
    # 定义 Pegasus 配置和数据类型
    config: PegasusConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    # 初始化函数
    def setup(self):
        # 创建 Pegasus 编码器层集合
        self.layers = [
            FlaxPegasusEncoderLayer(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.encoder_layers)
        ]
        # 获取编码层丢弃率
        self.layerdrop = self.config.encoder_layerdrop
    # 定义一个call方法用于模型的前向传播
    def __call__(
        self,
        hidden_states, # 输入的隐藏状态
        attention_mask, # 注意力掩码
        deterministic: bool = True, # 是否使用确定性推断
        output_attentions: bool = False, # 是否输出注意力权重
        output_hidden_states: bool = False, # 是否输出隐藏状态
        return_dict: bool = True, # 返回类型，是否以字典形式返回结果
    ):
        # 如果需要输出注意力权重，创建一个空的元组
        all_attentions = () if output_attentions else None
        # 如果需要输出隐藏状态，创建一个空的元组
        all_hidden_states = () if output_hidden_states else None

        # 对每个编码器层进行迭代
        for encoder_layer in self.layers:
            # 如果需要输出隐藏状态，将当前隐藏状态添加到all_hidden_states中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            # 添加LayerDrop（详见https://arxiv.org/abs/1909.11556的描述）
            dropout_probability = random.uniform(0, 1)
            # 如果不是确定性推断并且随机数小于layerdrop的概率，则跳过该层
            if not deterministic and (dropout_probability < self.layerdrop):
                layer_outputs = (None, None) # 将输出置为空
            else:
                # 否则，调用encoder_layer的__call__方法
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions,
                    deterministic,
                )
            hidden_states = layer_outputs[0] # 获取输出的隐藏状态
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],) # 将注意力权重添加到all_attentions中

        # 如果需要输出隐藏状态，将最后一个隐藏状态添加到all_hidden_states中
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 将最后的结果组合成一个元组
        outputs = (hidden_states, all_hidden_states, all_attentions)

        # 如果不需要以字典形式返回结果，则将元组中的非空值返回
        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        # 如果需要以字典形式返回结果，则返回FlaxBaseModelOutput类对象
        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )
# 从transformers.models.mbart.modeling_flax_mbart.FlaxMBartDecoderLayer复制代码并将MBart->Pegasus
class FlaxPegasusDecoderLayer(nn.Module):
    # 定义类属性config为PegasusConfig，dtype为jnp.float32
    config: PegasusConfig
    dtype: jnp.dtype = jnp.float32

    # 初始化方法
    def setup(self) -> None:
        # 定义self.embed_dim为config中的d_model属性
        self.embed_dim = self.config.d_model
        # 定义self.self_attn为FlaxPegasusAttention对象，传入config、embed_dim、num_heads、dropout、causal和dtype属性
        self.self_attn = FlaxPegasusAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.decoder_attention_heads,
            dropout=self.config.attention_dropout,
            causal=True,
            dtype=self.dtype,
        )
        # 定义self.dropout_layer为nn.Dropout对象，传入rate属性为config中的dropout属性
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)
        # 定义self.activation_fn为ACT2FN字典中config.activation_function对应的值
        self.activation_fn = ACT2FN[self.config.activation_function]
        # 定义self.activation_dropout_layer为nn.Dropout对象，传入rate属性为config中的activation_dropout属性
        self.activation_dropout_layer = nn.Dropout(rate=self.config.activation_dropout)

        # 定义self.self_attn_layer_norm为nn.LayerNorm对象，传入dtype属性为self.dtype，epsilon属性为1e-05
        self.self_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        # 定义self.encoder_attn为FlaxPegasusAttention对象，传入config、embed_dim、num_heads、dropout和dtype属性
        self.encoder_attn = FlaxPegasusAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.decoder_attention_heads,
            dropout=self.config.attention_dropout,
            dtype=self.dtype,
        )
        # 定义self.encoder_attn_layer_norm为nn.LayerNorm对象，传入dtype属性为self.dtype，epsilon属性为1e-05
        self.encoder_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        # 定义self.fc1为nn.Dense对象，传入config中的decoder_ffn_dim属性和dtype属性为self.dtype
        self.fc1 = nn.Dense(
            self.config.decoder_ffn_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        # 定义self.fc2为nn.Dense对象，传入embed_dim属性为self.embed_dim，dtype属性为self.dtype，kernel_init属性为通过jax.nn.initializers.normal初始化传入的self.config.init_std
        self.fc2 = nn.Dense(
            self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std)
        )
        # 定义self.final_layer_norm为nn.LayerNorm对象，传入dtype属性为self.dtype，epsilon属性为1e-05

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        output_attentions: bool = True,
        deterministic: bool = True,
        # 定义__call__方法，传入一系列参数
    ) -> Tuple[jnp.ndarray]:
        # 保存残差连接，以备后续加和
        residual = hidden_states
        # 对隐藏状态进行自注意力机制层规范化
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 自注意力机制
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, init_cache=init_cache
        )
        # 应用 dropout 层
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 加和残差连接
        hidden_states = residual + hidden_states

        # 交叉注意力机制块
        cross_attn_weights = None
        # 如果有编码器隐藏状态，则执行交叉注意力机制
        if encoder_hidden_states is not None:
            # 保存残差连接
            residual = hidden_states
            # 对隐藏状态进行编码器注意力机制层规范化
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            # 执行编码器注意力机制
            hidden_states, cross_attn_weights = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            # 应用 dropout 层
            hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
            # 加和残差连接
            hidden_states = residual + hidden_states

        # 全连接层
        # 保存残差连接
        residual = hidden_states
        # 对隐藏状态进行最终层规范化
        hidden_states = self.final_layer_norm(hidden_states)
        # 应用激活函数
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 应用 dropout 层
        hidden_states = self.activation_dropout_layer(hidden_states, deterministic=deterministic)
        # 应用全连接层
        hidden_states = self.fc2(hidden_states)
        # 应用 dropout 层
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 加和残差连接
        hidden_states = residual + hidden_states

        # 输出
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则添加到输出中
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        # 返回输出
        return outputs
# 创建一个自定义的 Pegasus 解码器层集合类，继承于 nn.Module
class FlaxPegasusDecoderLayerCollection(nn.Module):
    # 初始化方法，接收 Pegasus 配置和数据类型参数
    config: PegasusConfig
    dtype: jnp.dtype = jnp.float32  # 计算时的数据类型

    def setup(self):
        # 创建 Pegasus 解码器层集合，根据解码器层数量
        self.layers = [
            FlaxPegasusDecoderLayer(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.decoder_layers)
        ]
        # 获取层丢弃率
        self.layerdrop = self.config.decoder_layerdrop

    def __call__(
        self,
        hidden_states,
        attention_mask,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 初始化存储所有隐藏状态和注意力权重的变量
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        # 遍历所有解码器层
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                # 添加 LayerDrop，根据概率丢弃部分层的输出

            dropout_probability = random.uniform(0, 1)
            if not deterministic and (dropout_probability < self.layerdrop):
                layer_outputs = (None, None, None)
            else:
                # 运行解码器层
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    init_cache=init_cache,
                    output_attentions=output_attentions,
                    deterministic=deterministic,
                )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # 将最后一个解码器层的隐藏状态加入到所有隐藏状态中
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 组合所有输出结果
        outputs = [hidden_states, all_hidden_states, all_self_attns, all_cross_attentions]

        # 如果不要以字典形式返回，则返回输出元组
        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        # 以字典形式返回包含结果的对象
        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


# 创建一个自定义的 Pegasus 编码器类，继承于 nn.Module
class FlaxPegasusEncoder(nn.Module):
    # 初始化方法，接收 Pegasus 配置和嵌入令牌参数
    config: PegasusConfig
    embed_tokens: nn.Embed
    dtype: jnp.dtype = jnp.float32  # 计算时的数据类型
    # 初始化模型，设置各个成员变量和参数
    def setup(self):
        # 定义一个丢弃层，根据配置中的丢弃率设置丢弃概率
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)

        # 获取嵌入维度
        embed_dim = self.config.d_model
        # 设置填充索引
        self.padding_idx = self.config.pad_token_id
        # 设置最大源序列位置
        self.max_source_positions = self.config.max_position_embeddings
        # 根据是否缩放嵌入，设置嵌入缩放因子
        self.embed_scale = math.sqrt(embed_dim) if self.config.scale_embedding else 1.0

        # 创建 sinusoidal 位置编码
        self.embed_positions = create_sinusoidal_positions(self.config.max_position_embeddings, embed_dim)
        # 初始化编码层集合
        self.layers = FlaxPegasusEncoderLayerCollection(self.config, self.dtype)
        # 初始化层归一化层
        self.layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    # 模型的调用方法，用于进行前向推断
    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
    ):
        # 获取输入形状
        input_shape = input_ids.shape
        # 将输入展平为二维，保留最后一个维度
        input_ids = input_ids.reshape(-1, input_shape[-1])

        # 获取输入的嵌入表示并缩放
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # 嵌入位置编码
        embed_pos = jnp.take(self.embed_positions, position_ids, axis=0)
        # 显式地将位置编码转换为与输入相同的数据类型
        embed_pos = embed_pos.astype(inputs_embeds.dtype)

        # 输入的嵌入表示加上位置编码
        hidden_states = inputs_embeds + embed_pos
        # 应用丢弃层
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)

        # 调用层集合进行编码器层的前向计算
        outputs = self.layers(
            hidden_states,
            attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # 获取最后一个隐藏状态并应用层归一化
        last_hidden_state = outputs[0]
        last_hidden_state = self.layer_norm(last_hidden_state)

        # 更新 `hidden_states` 中的最后一个元素，应用了上面的 `layernorm`
        hidden_states = None
        if output_hidden_states:
            hidden_states = outputs[1]
            hidden_states = hidden_states[:-1] + (last_hidden_state,)

        # 根据是否返回字典形式的结果，构建输出
        if not return_dict:
            outputs = (last_hidden_state, hidden_states) + (outputs[2:] if output_hidden_states else outputs[1:])
            return tuple(v for v in outputs if v is not None)

        # 返回 FlaxBaseModelOutput 对象
        return FlaxBaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
        )
# 这是一个继承自 nn.Module 的 FlaxPegasusDecoder 类
class FlaxPegasusDecoder(nn.Module):
    # 定义了 config 和 embed_tokens 属性，其中 dtype 属性默认为 jnp.float32
    config: PegasusConfig
    embed_tokens: nn.Embed
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    # 在 setup 方法中进行初始化操作
    def setup(self):
        # 创建一个 nn.Dropout 层，dropout 率为 config.dropout
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)

        # 获取 embed_dim 和 padding_idx 等属性值
        embed_dim = self.config.d_model
        self.padding_idx = self.config.pad_token_id
        self.max_target_positions = self.config.max_position_embeddings
        self.embed_scale = math.sqrt(self.config.d_model) if self.config.scale_embedding else 1.0

        # 创建一个正弦位置编码
        self.embed_positions = create_sinusoidal_positions(self.config.max_position_embeddings, embed_dim)

        # 创建一个 FlaxPegasusDecoderLayerCollection 对象
        self.layers = FlaxPegasusDecoderLayerCollection(self.config, self.dtype)
        # 创建一个 nn.LayerNorm 层
        self.layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    # 定义了一个 __call__ 方法，用于接收输入并进行相应的处理
    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
    ):
        # 获取 input_ids 的 shape，并将其展平
        input_shape = input_ids.shape
        input_ids = input_ids.reshape(-1, input_shape[-1])

        # 根据 input_ids 获取嵌入向量，并乘以 embed_scale
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # 根据 position_ids 获取位置编码向量，并转换为与 inputs_embeds 相同的数据类型
        positions = jnp.take(self.embed_positions, position_ids, axis=0)
        positions = positions.astype(inputs_embeds.dtype)

        # 将嵌入向量和位置编码相加
        hidden_states = inputs_embeds + positions
        # 对 hidden_states 进行 dropout 处理
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)

        # 调用 self.layers 进行进一步的处理
        outputs = self.layers(
            hidden_states,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取最后一个隐藏状态
        last_hidden_state = outputs[0]
        # 使用 self.layer_norm 对最后一个隐藏状态进行归一化
        last_hidden_state = self.layer_norm(last_hidden_state)

        # 更新 hidden_states 中最后一个元素为归一化后的 last_hidden_state
        hidden_states = None
        if output_hidden_states:
            hidden_states = outputs[1]
            hidden_states = hidden_states[:-1] + (last_hidden_state,)

        # 根据 return_dict 的值返回不同格式的输出
        if not return_dict:
            outputs = (last_hidden_state, hidden_states) + (outputs[2:] if output_hidden_states else outputs[1:])
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=last_hidden_state,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )
# 从transformers.models.bart.modeling_flax_bart.FlaxBartModule复制代码，并将Bart->Pegasus进行替换
class FlaxPegasusModule(nn.Module):
    # 初始化函数，接受PegasusConfig作为参数，指定数据类型为jnp.float32
    config: PegasusConfig
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型

    def setup(self):
        # 创建共享的嵌入层，参数为vocab_size和d_model，使用正态分布初始化，指定数据类型为dtype
        self.shared = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.init_std),
            dtype=self.dtype,
        )

        # 创建Pegasus编码器和解码器，传入config和共享的embed_tokens
        self.encoder = FlaxPegasusEncoder(self.config, dtype=self.dtype, embed_tokens=self.shared)
        self.decoder = FlaxPegasusDecoder(self.config, dtype=self.dtype, embed_tokens=self.shared)

    def _get_encoder_module(self):
        # 获取编码器模块
        return self.encoder

    def _get_decoder_module(self):
        # 获取解码器模块
        return self.decoder

    def __call__(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        position_ids,
        decoder_position_ids,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
    ):
        # 获取编码器的输出
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        # 获取解码器的输出，传入编码器的隐藏状态和注意力掩码
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            position_ids=decoder_position_ids,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        # 返回FlaxSeq2SeqModelOutput对象，包含解码器和编码器的输出信息
        return FlaxSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class FlaxPegasusPreTrainedModel(FlaxPreTrainedModel):
    config_class = PegasusConfig
    base_model_prefix: str = "model"
    module_class: nn.Module = None

    def __init__(
        self,
        config: PegasusConfig,
        input_shape: Tuple[int] = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        # 通过配置和参数初始化模块对象
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 调用父类的初始化方法，传入配置、模块对象、输入形状、种子、数据类型和是否初始化的标记
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        decoder_input_ids = input_ids
        decoder_attention_mask = jnp.ones_like(input_ids)

        # 获取批量大小和序列长度
        batch_size, sequence_length = input_ids.shape
        # 创建位置向量
        position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))
        decoder_position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # 切分随机数生成器
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 使用初始化的随机参数初始化模块
        random_params = self.module.init(
            rngs,
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            position_ids,
            decoder_position_ids,
        )["params"]

        if params is not None:
            # 将参数展平并转为可变字典
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            # 将缺失的键从随机参数中添加到给定的参数中
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params
    # 初始化缓存，用于快速自回归解码
        def init_cache(self, batch_size, max_length, encoder_outputs):
            r"""
            Args:
                batch_size (`int`):
                    用于快速自回归解码的批量大小。定义了初始化缓存的批量大小。
                max_length (`int`):
                    自回归解码可能的最大长度。定义了初始化缓存的序列长度。
                encoder_outputs (`Union[FlaxBaseModelOutput, tuple(tuple(jnp.ndarray)]`):
                    `encoder_outputs` 包括（`last_hidden_state`，* 可选 *：`hidden_states`，* 可选 *：`attentions`）。`last_hidden_state` 的形状为 `(batch_size, sequence_length, hidden_size)`，* 可选 *）是编码器最后一层的隐藏状态输出序列。在解码器的交叉注意力中使用。
            """
            # 初始化输入变量以检索缓存
            decoder_input_ids = jnp.ones((batch_size, max_length), dtype="i4")
            decoder_attention_mask = jnp.ones_like(decoder_input_ids)
            decoder_position_ids = jnp.broadcast_to(
                jnp.arange(jnp.atleast_2d(decoder_input_ids).shape[-1]), decoder_input_ids.shape
            )
    
            def _decoder_forward(module, decoder_input_ids, decoder_attention_mask, decoder_position_ids, **kwargs):
                decoder_module = module._get_decoder_module()
                return decoder_module(
                    decoder_input_ids,
                    decoder_attention_mask,
                    decoder_position_ids,
                    **kwargs,
                )
    
            # 初始化模型的变量，以获取缓存
            init_variables = self.module.init(
                jax.random.PRNGKey(0),
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                decoder_position_ids=decoder_position_ids,
                encoder_hidden_states=encoder_outputs[0],
                init_cache=True,
                method=_decoder_forward,  # 我们只需要调用解码器来初始化缓存
            )
            # 返回解冻后的缓存值
            return unfreeze(init_variables["cache"])
    
        @add_start_docstrings(PEGASUS_ENCODE_INPUTS_DOCSTRING)
        @replace_return_docstrings(output_type=FlaxBaseModelOutput, config_class=PegasusConfig)
        def encode(
            self,
            input_ids: jnp.ndarray,
            attention_mask: Optional[jnp.ndarray] = None,
            position_ids: Optional[jnp.ndarray] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            train: bool = False,
            params: dict = None,
            dropout_rng: PRNGKey = None,
    # 以下为 decode 方法的实现
    # 此处的三个引号应该是代码写错了，应该是要注释整个函数的
    # 定义一个类方法，用于执行模型推理
    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        decoder_input_ids: Optional[jnp.ndarray] = None,
        decoder_attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        decoder_position_ids: Optional[jnp.ndarray] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
    ):
        # 根据传入的参数或模型配置决定是否输出注意力权重
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 根据传入的参数或模型配置决定是否输出隐藏层状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 根据传入的参数或模型配置决定是否返回字典格式的输出
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 准备编码器的输入
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        if position_ids is None:
            batch_size, sequence_length = input_ids.shape
            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # 准备解码器的输入
        if decoder_input_ids is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, decoder_start_token_id=self.config.decoder_start_token_id
            )
        if decoder_attention_mask is None:
            decoder_attention_mask = jnp.ones_like(decoder_input_ids)
        if decoder_position_ids is None:
            batch_size, sequence_length = decoder_input_ids.shape
            decoder_position_ids = jnp.broadcast_to(
                jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
            )

        # 如果需要处理任何 PRNG（伪随机数生成器）
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        # 调用模型的处理方法
        return self.module.apply(
            {"params": params or self.params},
            input_ids=jnp.array(input_ids, dtype="i4"),
            attention_mask=jnp.array(attention_mask, dtype="i4"),
            position_ids=jnp.array(position_ids, dtype="i4"),
            decoder_input_ids=jnp.array(decoder_input_ids, dtype="i4"),
            decoder_attention_mask=jnp.array(decoder_attention_mask, dtype="i4"),
            decoder_position_ids=jnp.array(decoder_position_ids, dtype="i4"),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=not train,
            rngs=rngs,
        )
# 使用自动文档生成函数添加模型实例的注释描述
@add_start_docstrings(
    "The bare Pegasus Model transformer outputting raw hidden-states without any specific head on top.",
    PEGASUS_START_DOCSTRING,
)
# 定义 FlaxPegasusModel 类，继承自 FlaxPegasusPreTrainedModel 类
class FlaxPegasusModel(FlaxPegasusPreTrainedModel):
    # 模型配置参数
    config: PegasusConfig
    # 计算中所使用的数据类型
    dtype: jnp.dtype = jnp.float32  
    # 模型使用的模块类别
    module_class = FlaxPegasusModule

# 使用函数添加调用示例的注释描述
append_call_sample_docstring(FlaxPegasusModel, _CHECKPOINT_FOR_DOC, FlaxSeq2SeqModelOutput, _CONFIG_FOR_DOC)

# 定义 FlaxPegasusForConditionalGenerationModule 类，继承自 nn.Module 类
# 此处的代码类似于从 FlaxBartForConditionalGenerationModule 复制并修改为 Pegasus
class FlaxPegasusForConditionalGenerationModule(nn.Module):
    # 模型配置参数
    config: PegasusConfig
    # 计算中所使用的数据类型
    dtype: jnp.dtype = jnp.float32
    # bias 初始化函数
    bias_init: Callable[..., jnp.ndarray] = jax.nn.initializers.zeros

    # 设置模型结构
    def setup(self):
        # 创建 Pegasus 模型实例
        self.model = FlaxPegasusModule(config=self.config, dtype=self.dtype)
        # 创建 LM 头部，连接输入和输出的全连接层
        self.lm_head = nn.Dense(
            self.model.shared.num_embeddings,
            use_bias=False,
            dtype=self.dtype,
            # 初始化权重矩阵
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        # 创建最终输出的 logits 偏置参数
        self.final_logits_bias = self.param("final_logits_bias", self.bias_init, (1, self.model.shared.num_embeddings))

    # 返回编码器模块
    def _get_encoder_module(self):
        return self.model.encoder

    # 返回解码器模块
    def _get_decoder_module(self):
        return self.model.decoder

    # 模型调用函数
    def __call__(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        position_ids,
        decoder_position_ids,
        output_attentions: bool = False,  # 输出注意力权重
        output_hidden_states: bool = False,  # 输出隐藏状态
        return_dict: bool = True,  # 返回字典形式结果
        deterministic: bool = True,  # 确定性计算
        # 使用模型进行前向传播，获取输出
        outputs = self.model(
            input_ids=input_ids,  # 输入序列的token IDs
            attention_mask=attention_mask,  # 输入序列的attention mask
            decoder_input_ids=decoder_input_ids,  # 解码器输入序列的token IDs
            decoder_attention_mask=decoder_attention_mask,  # 解码器输入序列的attention mask
            position_ids=position_ids,  # 输入序列的位置编码
            decoder_position_ids=decoder_position_ids,  # 解码器输入序列的位置编码
            output_attentions=output_attentions,  # 是否输出注意力权重
            output_hidden_states=output_hidden_states,  # 是否输出隐藏状态
            return_dict=return_dict,  # 是否返回字典形式的结果
            deterministic=deterministic,  # 是否使用确定性计算
        )

        # 从模型输出中提取隐藏状态
        hidden_states = outputs[0]

        # 如果配置了词嵌入共享
        if self.config.tie_word_embeddings:
            # 获取共享的词嵌入参数
            shared_embedding = self.model.variables["params"]["shared"]["embedding"]
            # 计算语言模型logits，采用共享的词嵌入参数
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
        else:
            # 计算语言模型logits
            lm_logits = self.lm_head(hidden_states)

        # 添加最终logits的偏置
        lm_logits += jax.lax.stop_gradient(self.final_logits_bias.astype(self.dtype))

        # 如果不返回字典形式的结果
        if not return_dict:
            # 构造输出元组
            output = (lm_logits,) + outputs[1:]
            return output

        # 返回字典形式的结果
        return FlaxSeq2SeqLMOutput(
            logits=lm_logits,  # 语言模型logits
            decoder_hidden_states=outputs.decoder_hidden_states,  # 解码器隐藏状态
            decoder_attentions=outputs.decoder_attentions,  # 解码器注意力权重
            cross_attentions=outputs.cross_attentions,  # 交叉注意力权重
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,  # 编码器最后一个隐藏状态
            encoder_hidden_states=outputs.encoder_hidden_states,  # 编码器所有隐藏状态
            encoder_attentions=outputs.encoder_attentions,  # 编码器注意力权重
        )
# 导入 PEGASUS 模型及其相关的预训练模型和文档字符串
@add_start_docstrings(
    "The PEGASUS Model with a language modeling head. Can be used for summarization.", PEGASUS_START_DOCSTRING
)
class FlaxPegasusForConditionalGeneration(FlaxPegasusPreTrainedModel):
    # 指定模块类为 FlaxPegasusForConditionalGenerationModule
    module_class = FlaxPegasusForConditionalGenerationModule
    # 设置数据类型为 float32
    dtype: jnp.dtype = jnp.float32

    # 解码输入的描述文档字符串
    @add_start_docstrings(PEGASUS_DECODE_INPUTS_DOCSTRING)
    # 替换返回文档字符串
    @replace_return_docstrings(output_type=FlaxCausalLMOutputWithCrossAttentions, config_class=PegasusConfig)
    # 解码函数
    def decode(
        self,
        decoder_input_ids,
        encoder_outputs,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        decoder_attention_mask: Optional[jnp.ndarray] = None,
        decoder_position_ids: Optional[jnp.ndarray] = None,
        past_key_values: dict = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        deterministic: bool = True,
        params: dict = None,
        dropout_rng: PRNGKey = None,

    # 准备生成输入的函数
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        max_length,
        attention_mask: Optional[jax.Array] = None,
        decoder_attention_mask: Optional[jax.Array] = None,
        encoder_outputs=None,
        **kwargs,
    ):
        # 初始化cache
        batch_size, seq_length = decoder_input_ids.shape
        past_key_values = self.init_cache(batch_size, max_length, encoder_outputs)

        # 创建扩展的 attention mask
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if decoder_attention_mask is not None:
            position_ids = decoder_attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, decoder_attention_mask, (0, 0))
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        return {
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "encoder_attention_mask": attention_mask,
            "decoder_attention_mask": extended_attention_mask,
            "decoder_position_ids": position_ids,
        }

    # 更新生成过程中的输入
    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["decoder_position_ids"] = model_kwargs["decoder_position_ids"][:, -1:] + 1
        return model_kwargs
    # 从预训练模型 'google/pegasus-large' 中加载条件生成模型
    >>> model = FlaxPegasusForConditionalGeneration.from_pretrained('google/pegasus-large')
    # 从预训练模型 'google/pegasus-large' 中加载分词器
    >>> tokenizer = AutoTokenizer.from_pretrained('google/pegasus-large')

    # 待进行摘要的文章内容
    >>> ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
    # 使用分词器对文章内容进行分词，并返回输入模型的数据
    >>> inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='np')

    # 生成摘要
    >>> summary_ids = model.generate(inputs['input_ids']).sequences
    # 打印解码后的摘要内容，跳过特殊标记，并保留分词空格
    >>> print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))
    ```

    # 填充掩码示例:

    ```python
    # 从 'google/pegasus-large' 预训练模型中加载分词器
    >>> tokenizer = AutoTokenizer.from_pretrained("google/pegasus-large")
    # 待填充掩码的文本
    >>> TXT = "My friends are <mask> but they eat too many carbs."

    # 从 'google/pegasus-large' 预训练模型中加载条件生成模型
    >>> model = FlaxPegasusForConditionalGeneration.from_pretrained("google/pegasus-large")
    # 使用分词器对文本进行分词，并返回输入模型的数据
    >>> input_ids = tokenizer([TXT], return_tensors="np")["input_ids"]
    # 模型生成对应的logits
    >>> logits = model(input_ids).logits

    # 获取掩码的索引
    >>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
    # 对logits进行softmax概率计算
    >>> probs = jax.nn.softmax(logits[0, masked_index], axis=0)
    # 获取前k个概率最大的值和对应的预测结果
    >>> values, predictions = jax.lax.top_k(probs)

    # 解码得到的预测结果，并以空格分隔
    >>> tokenizer.decode(predictions).split()
    ```
# 重写 `FlaxPegasusForConditionalGeneration` 的文档字符串，将 `PEGASUS_INPUTS_DOCSTRING` 和 `FLAX_PEGASUS_CONDITIONAL_GENERATION_DOCSTRING` 添加到其文档字符串中
overwrite_call_docstring(
    FlaxPegasusForConditionalGeneration, PEGASUS_INPUTS_DOCSTRING + FLAX_PEGASUS_CONDITIONAL_GENERATION_DOCSTRING
)
# 向 `FlaxPegasusForConditionalGeneration` 的返回值文档字符串中添加或替换文档字符串，指定输出类型为 `FlaxSeq2SeqLMOutput`，配置类为 `_CONFIG_FOR_DOC`
append_replace_return_docstrings(
    FlaxPegasusForConditionalGeneration, output_type=FlaxSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC
)
```