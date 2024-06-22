# `.\transformers\models\mbart\modeling_flax_mbart.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 Facebook AI Research Team 和 HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本，除非符合许可证，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”基础分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制
""" Flax MBart model."""

# 导入所需的库
import math
import random
from functools import partial
from typing import Callable, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from jax.random import PRNGKey

# 导入模型输出相关的类
from ...modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxBaseModelOutputWithPastAndCrossAttentions,
    FlaxCausalLMOutputWithCrossAttentions,
    FlaxSeq2SeqLMOutput,
    FlaxSeq2SeqModelOutput,
    FlaxSeq2SeqQuestionAnsweringModelOutput,
    FlaxSeq2SeqSequenceClassifierOutput,
)
# 导入模型工具相关的函数和类
from ...modeling_flax_utils import (
    ACT2FN,
    FlaxPreTrainedModel,
    append_call_sample_docstring,
    append_replace_return_docstrings,
    overwrite_call_docstring,
)
# 导入一些辅助函数
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
# 导入 MBart 配置类
from .configuration_mbart import MBartConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置信息
_CHECKPOINT_FOR_DOC = "facebook/mbart-large-cc25"
_CONFIG_FOR_DOC = "MBartConfig"

# MBart 模型的起始文档字符串
MBART_START_DOCSTRING = r"""
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
"""
    # 参数说明：
    # config ([`MBartConfig`]): 包含模型所有参数的模型配置类。
    # 用配置文件初始化不会加载与模型相关的权重，只会加载配置。查看 [`~FlaxPreTrainedModel.from_pretrained`] 方法以加载模型权重。
    # dtype (`jax.numpy.dtype`, *optional*, 默认为 `jax.numpy.float32`):
    # 计算的数据类型。可以是 `jax.numpy.float32`, `jax.numpy.float16` (在GPU上) 和 `jax.numpy.bfloat16` (在TPU上) 中的一种。
    # 这可以用于在GPU或TPU上启用混合精度训练或半精度推断。如果指定，所有计算将使用给定的 `dtype` 进行。
    # **请注意，这仅指定计算的数据类型，不会影响模型参数的数据类型。**
    # 如果希望更改模型参数的数据类型，请参阅 [`~FlaxPreTrainedModel.to_fp16`] 和 [`~FlaxPreTrainedModel.to_bf16`]。
"""

MBART_INPUTS_DOCSTRING = r"""
"""


MBART_ENCODE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
            输入序列标记在词汇表中的索引。默认情况下，将忽略填充。

            可以使用 [`AutoTokenizer`] 获取索引。有关详细信息，请参阅 [`PreTrainedTokenizer.encode`] 和
            [`PreTrainedTokenizer.__call__`]。

            [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            避免在填充标记索引上执行注意力的掩码。掩码值选在 `[0, 1]` 范围内：

            - 1 表示**未被掩码**的标记，
            - 0 表示**被掩码**的标记。

            [什么是注意力掩码？](../glossary#attention-mask)
        position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            输入序列标记在位置嵌入中的位置索引。选在 `[0, config.max_position_embeddings - 1]` 范围内。
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。有关更多详细信息，请参阅返回张量中的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。有关更多详细信息，请参阅返回张量中的 `hidden_states`。
        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""

MBART_DECODE_INPUTS_DOCSTRING = r"""
"""


def shift_tokens_right(input_ids: jnp.ndarray, pad_token_id: int) -> jnp.ndarray:
    """
    将输入 ID 向右移动一个标记，并包装最后一个非填充标记（<LID> 标记）。请注意，与其他类似 Bart 模型不同，MBart
    没有单个 `decoder_start_token_id`。
    """
    prev_output_tokens = jnp.array(input_ids).copy()

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")

    # 用 `pad_token_id` 替换标签中可能存在的 -100 值
    prev_output_tokens = jnp.where(prev_output_tokens == -100, pad_token_id, input_ids)
    index_of_eos = (jnp.where(prev_output_tokens != pad_token_id, 1, 0).sum(axis=-1) - 1).reshape(-1, 1)
    decoder_start_tokens = jnp.array(
        [prev_output_tokens[i, eos_idx] for i, eos_idx in enumerate(index_of_eos)], dtype=jnp.int32
    ).squeeze()

    prev_output_tokens = prev_output_tokens.at[:, 1:].set(prev_output_tokens[:, :-1])
    prev_output_tokens = prev_output_tokens.at[:, 0].set(decoder_start_tokens)

    return prev_output_tokens


# 从 transformers.models.bart.modeling_flax_bart.FlaxBartAttention 复制，将 Bart->MBart
class FlaxMBartAttention(nn.Module):
    config: MBartConfig
    embed_dim: int
    num_heads: int
    # 定义神经网络层的参数，包括dropout率、是否使用因果关系、是否使用偏置、数据类型
    dropout: float = 0.0
    causal: bool = False
    bias: bool = True
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型

    # 设置神经网络层
    def setup(self) -> None:
        # 计算每个头的维度
        self.head_dim = self.embed_dim // self.num_heads
        # 检查embed_dim是否能被num_heads整除
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )

        # 部分应用nn.Dense函数，设置参数
        dense = partial(
            nn.Dense,
            self.embed_dim,
            use_bias=self.bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )

        # 初始化q_proj、k_proj、v_proj、out_proj
        self.q_proj, self.k_proj, self.v_proj = dense(), dense(), dense()
        self.out_proj = dense()

        # 初始化dropout层
        self.dropout_layer = nn.Dropout(rate=self.dropout)

        # 如果使用因果关系，创建因果关系的mask
        if self.causal:
            self.causal_mask = make_causal_mask(
                jnp.ones((1, self.config.max_position_embeddings), dtype="bool"), dtype="bool"
            )

    # 将隐藏状态拆分成多个头
    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    # 将多个头合并成隐藏状态
    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    # 使用nn.compact装饰器
    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slightly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # 检测是否通过缺少现有缓存数据来初始化。
        is_initialized = self.has_variable("cache", "cached_key")
        # 获取缓存的键值对
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        # 获取缓存索引
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32)

        if is_initialized:
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # 使用新的1D空间切片更新键值缓存
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # 用于缓存解码器自注意力的因果掩码：我们的单个查询位置应仅关注已生成和缓存的键位置，而不是剩余的零元素。
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)
        return key, value, attention_mask
class FlaxMBartEncoderLayer(nn.Module):
    config: MBartConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        # 设置编码器层的维度为配置中的模型维度
        self.embed_dim = self.config.d_model
        # 创建自注意力机制对象
        self.self_attn = FlaxMBartAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.encoder_attention_heads,
            dropout=self.config.attention_dropout,
            dtype=self.dtype,
        )
        # 创建自注意力机制的 LayerNorm 层
        self.self_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        # 创建 dropout 层
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)
        # 设置激活函数
        self.activation_fn = ACT2FN[self.config.activation_function]
        # 创建激活函数的 dropout 层
        self.activation_dropout_layer = nn.Dropout(rate=self.config.activation_dropout)
        # 创建全连接层 fc1
        self.fc1 = nn.Dense(
            self.config.encoder_ffn_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        # 创建全连接层 fc2
        self.fc2 = nn.Dense(
            self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std)
        )
        # 创建最终的 LayerNorm 层
        self.final_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        output_attentions: bool = True,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray]:
        # 保存残差连接
        residual = hidden_states
        # 对隐藏状态进行 LayerNorm
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # 进行自注意力计算
        hidden_states, attn_weights = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask)
        # 对输出进行 dropout
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 添加残差连接
        hidden_states = residual + hidden_states

        # 保存残差连接
        residual = hidden_states
        # 对隐藏状态进行最终的 LayerNorm
        hidden_states = self.final_layer_norm(hidden_states)
        # 使用激活函数
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 对激活函数的输出进行 dropout
        hidden_states = self.activation_dropout_layer(hidden_states, deterministic=deterministic)
        # 使用全连接层 fc2
        hidden_states = self.fc2(hidden_states)
        # 对输出进行 dropout
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 添加残差连接
        hidden_states = residual + hidden_states

        # 将隐藏状态作为输出
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则添加到输出中
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# 从 transformers.models.bart.modeling_flax_bart.FlaxBartEncoderLayerCollection 复制并修改为 MBart
class FlaxMBartEncoderLayerCollection(nn.Module):
    config: MBartConfig
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型

    def setup(self):
        # 创建编码器层列表
        self.layers = [
            FlaxMBartEncoderLayer(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.encoder_layers)
        ]
        # 设置层级丢弃率
        self.layerdrop = self.config.encoder_layerdrop
    # 定义一个调用函数，接受隐藏状态、注意力掩码、是否确定性、是否输出注意力、是否输出隐藏状态、是否返回字典等参数
    def __call__(
        self,
        hidden_states,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 如果需要输出注意力，则初始化一个空元组
        all_attentions = () if output_attentions else None
        # 如果需要输出隐藏状态，则初始化一个空元组
        all_hidden_states = () if output_hidden_states else None

        # 遍历每个编码器层
        for encoder_layer in self.layers:
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到所有隐藏状态中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            # 添加LayerDrop（参见https://arxiv.org/abs/1909.11556的描述）
            dropout_probability = random.uniform(0, 1)
            # 如果不是确定性的且随机数小于层丢弃率，则跳过该层
            if not deterministic and (dropout_probability < self.layerdrop):
                layer_outputs = (None, None)
            else:
                # 否则调用编码器层的前向传播函数
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions,
                    deterministic,
                )
            # 更新隐藏状态为编码器层的输出的第一个元素
            hidden_states = layer_outputs[0]
            # 如果需要输出注意力，则将当前层的注意力添加到所有注意力中
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # 如果需要输出隐藏状态，则将最终隐藏状态添加到所有隐藏状态中
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 将隐藏状态、所有隐藏状态和所有注意力组成一个元组
        outputs = (hidden_states, all_hidden_states, all_attentions)

        # 如果不需要返回字典，则返回所有输出中不为None的元素
        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        # 返回FlaxBaseModelOutput对象，包含最终隐藏状态、所有隐藏状态和所有注意力
        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )
# 定义一个 FlaxMBartDecoderLayer 类，继承自 nn.Module
class FlaxMBartDecoderLayer(nn.Module):
    # 定义类属性 config，类型为 MBartConfig
    config: MBartConfig
    # 定义类属性 dtype，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 初始化方法
    def setup(self) -> None:
        # 设置 embed_dim 为 config 中的 d_model
        self.embed_dim = self.config.d_model
        # 初始化 self_attn 为 FlaxMBartAttention 对象
        self.self_attn = FlaxMBartAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.decoder_attention_heads,
            dropout=self.config.attention_dropout,
            causal=True,
            dtype=self.dtype,
        )
        # 初始化 dropout_layer 为 nn.Dropout 对象
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)
        # 根据配置选择激活函数
        self.activation_fn = ACT2FN[self.config.activation_function]
        # 初始化 activation_dropout_layer 为 nn.Dropout 对象
        self.activation_dropout_layer = nn.Dropout(rate=self.config.activation_dropout)

        # 初始化 self_attn_layer_norm 为 nn.LayerNorm 对象
        self.self_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        # 初始化 encoder_attn 为 FlaxMBartAttention 对象
        self.encoder_attn = FlaxMBartAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.decoder_attention_heads,
            dropout=self.config.attention_dropout,
            dtype=self.dtype,
        )
        # 初始化 encoder_attn_layer_norm 为 nn.LayerNorm 对象
        self.encoder_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        # 初始化 fc1 为 nn.Dense 对象
        self.fc1 = nn.Dense(
            self.config.decoder_ffn_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        # 初始化 fc2 为 nn.Dense 对象
        self.fc2 = nn.Dense(
            self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std)
        )
        # 初始化 final_layer_norm 为 nn.LayerNorm 对象
        self.final_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    # 定义 __call__ 方法
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        output_attentions: bool = True,
        deterministic: bool = True,
    # 定义函数的返回类型为包含一个 NumPy 数组的元组
    ) -> Tuple[jnp.ndarray]:
        # 保存残差连接
        residual = hidden_states
        # 对隐藏状态进行 Layer Normalization
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 自注意力机制
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, init_cache=init_cache
        )
        # 对隐藏状态进行 Dropout
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 添加残差连接
        hidden_states = residual + hidden_states

        # 交叉注意力块
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            # 保存残差连接
            residual = hidden_states

            # 对隐藏状态进行 Encoder Attention Layer Normalization
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            # 执行 Encoder Attention 操作
            hidden_states, cross_attn_weights = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            # 对隐藏状态进行 Dropout
            hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
            # 添加残差连接
            hidden_states = residual + hidden_states

        # 全连接层
        residual = hidden_states
        # 对隐藏状态进行最终的 Layer Normalization
        hidden_states = self.final_layer_norm(hidden_states)
        # 使用激活函数激活全连接层的第一层
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 对激活后的隐藏状态进行 Dropout
        hidden_states = self.activation_dropout_layer(hidden_states, deterministic=deterministic)
        # 使用全连接层的第二层
        hidden_states = self.fc2(hidden_states)
        # 对隐藏状态进行 Dropout
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 添加残差连接
        hidden_states = residual + hidden_states

        # 输出结果为包含隐藏状态的元组
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则将注意力权重添加到输出中
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs
# 从transformers.models.bart.modeling_flax_bart.FlaxBartDecoderLayerCollection复制并修改为MBart
class FlaxMBartDecoderLayerCollection(nn.Module):
    config: MBartConfig  # MBart配置信息
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型

    def setup(self):
        # 创建MBart解码器层集合，根据配置信息创建多个解码器层
        self.layers = [
            FlaxMBartDecoderLayer(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.decoder_layers)
        ]
        # 解码器层的丢弃率
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
        # 初始化输出的隐藏状态和注意力
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        # 遍历解码器层
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                # 添加LayerDrop（参考https://arxiv.org/abs/1909.11556）
            dropout_probability = random.uniform(0, 1)
            if not deterministic and (dropout_probability < self.layerdrop):
                layer_outputs = (None, None, None)
            else:
                # 调用解码器层
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

        # 添加最后一个解码器层的隐藏状态
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = [hidden_states, all_hidden_states, all_self_attns, all_cross_attentions]

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


# 从transformers.models.bart.modeling_flax_bart.FlaxBartClassificationHead复制并修改为MBart
class FlaxMBartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    # 定义一个类，用于句子级别分类任务的头部
    config: MBartConfig
    inner_dim: int
    num_classes: int
    pooler_dropout: float
    dtype: jnp.dtype = jnp.float32

    # 初始化方法
    def setup(self):
        # 创建一个全连接层，输入维度为inner_dim，输出维度为inner_dim，使用正态分布初始化
        self.dense = nn.Dense(
            self.inner_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std)
        )
        # 创建一个Dropout层，用于防止过拟合
        self.dropout = nn.Dropout(rate=self.pooler_dropout)
        # 创建一个全连接层，输入维度为inner_dim，输出维度为num_classes，使用正态分布初始化
        self.out_proj = nn.Dense(
            self.num_classes,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )

    # 调用方法
    def __call__(self, hidden_states: jnp.ndarray, deterministic: bool):
        # 对输入的hidden_states进行Dropout操作
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 将Dropout后的hidden_states输入到全连接层中
        hidden_states = self.dense(hidden_states)
        # 对全连接层的输出进行tanh激活函数处理
        hidden_states = jnp.tanh(hidden_states)
        # 再次对处理后的hidden_states进行Dropout操作
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 将Dropout后的hidden_states输入到输出全连接层中
        hidden_states = self.out_proj(hidden_states)
        # 返回最终的输出结果
        return hidden_states
# 定义一个FlaxMBartEncoder类，继承自nn.Module
class FlaxMBartEncoder(nn.Module):
    # 定义类属性config为MBartConfig类型
    config: MBartConfig
    # 定义类属性embed_tokens为nn.Embed类型
    embed_tokens: nn.Embed
    # 定义类属性dtype为jnp.float32类型，表示计算的数据类型

    # 定义setup方法
    def setup(self):
        # 初始化dropout_layer为一个Dropout层，dropout率为config中的dropout值
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)

        # 获取embed_dim为config中的d_model值
        embed_dim = self.config.d_model
        # 获取padding_idx为config中的pad_token_id值
        self.padding_idx = self.config.pad_token_id
        # 获取max_source_positions为config中的max_position_embeddings值
        self.max_source_positions = self.config.max_position_embeddings
        # 如果config中的scale_embedding为True，则设置embed_scale为embed_dim的平方根，否则为1.0
        self.embed_scale = math.sqrt(embed_dim) if self.config.scale_embedding else 1.0

        # MBart设置了一个偏移值为2，用于处理padding_idx的情况
        self.offset = 2
        # 初始化embed_positions为一个Embed层，包含max_position_embeddings + offset个位置，每个位置的维度为embed_dim
        self.embed_positions = nn.Embed(
            self.config.max_position_embeddings + self.offset,
            embed_dim,
            embedding_init=jax.nn.initializers.normal(self.config.init_std),
        )
        # 初始化layers为一个FlaxMBartEncoderLayerCollection对象
        self.layers = FlaxMBartEncoderLayerCollection(self.config, self.dtype)
        # 初始化layernorm_embedding为一个LayerNorm层，数据类型为dtype，epsilon为1e-05
        self.layernorm_embedding = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        # 初始化layer_norm为一个LayerNorm层，数据类型为dtype，epsilon为1e-05

    # 定义__call__方法
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
        # 获取input_ids的形状
        input_shape = input_ids.shape
        # 将input_ids重塑为二维数组
        input_ids = input_ids.reshape(-1, input_shape[-1])

        # 使用embed_tokens对input_ids进行嵌入，乘以embed_scale
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # 使用embed_positions对position_ids加上offset进行嵌入
        embed_pos = self.embed_positions(position_ids + self.offset)

        # 将inputs_embeds和embed_pos相加得到hidden_states
        hidden_states = inputs_embeds + embed_pos
        # 对hidden_states进行layernorm
        hidden_states = self.layernorm_embedding(hidden_states)
        # 对hidden_states进行dropout操作
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)

        # 使用layers处理hidden_states
        outputs = self.layers(
            hidden_states,
            attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取outputs中的最后一个hidden_state
        last_hidden_states = outputs[0]
        # 对last_hidden_states进行layernorm
        last_hidden_states = self.layer_norm(last_hidden_states)

        # 更新hidden_states中的最后一个元素，应用layernorm后
        hidden_states = None
        if output_hidden_states:
            hidden_states = outputs[1]
            hidden_states = hidden_states[:-1] + (last_hidden_states,)

        # 如果不返回字典形式的结果
        if not return_dict:
            # 重新组合outputs
            outputs = (last_hidden_states, hidden_states) + (outputs[2:] if output_hidden_states else outputs[1:])
            return tuple(v for v in outputs if v is not None)

        # 返回FlaxBaseModelOutput对象，包含last_hidden_state、hidden_states和attentions
        return FlaxBaseModelOutput(
            last_hidden_state=last_hidden_states,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
        )


# 定义一个FlaxMBartDecoder类，继承自nn.Module
class FlaxMBartDecoder(nn.Module):
    # 定义类属性config为MBartConfig类型
    config: MBartConfig
    # 定义类属性embed_tokens为nn.Embed类型
    embed_tokens: nn.Embed
    # 定义变量dtype为jnp.float32，表示计算的数据类型
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    # 初始化函数，设置Dropout层，Embedding维度等参数
    def setup(self):
        # 初始化Dropout层，设置丢弃率为config中的dropout参数
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)

        # 获取Embedding维度、填充索引、最大目标位置、Embedding缩放因子等参数
        embed_dim = self.config.d_model
        self.padding_idx = self.config.pad_token_id
        self.max_target_positions = self.config.max_position_embeddings
        self.embed_scale = math.sqrt(self.config.d_model) if self.config.scale_embedding else 1.0

        # 如果是MBart模型，则根据填充索引调整Embedding ids和num_embeddings
        # 其他模型不需要这个调整
        self.offset = 2
        self.embed_positions = nn.Embed(
            self.config.max_position_embeddings + self.offset,
            embed_dim,
            embedding_init=jax.nn.initializers.normal(self.config.init_std),
        )

        # 初始化MBart解码器层集合、Embedding层的LayerNorm和解码器层的LayerNorm
        self.layers = FlaxMBartDecoderLayerCollection(self.config, self.dtype)
        self.layernorm_embedding = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        self.layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    # 调用函数，接收输入ids、注意力掩码、位置ids、编码器隐藏状态等参数
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
        # 获取输入张量的形状
        input_shape = input_ids.shape
        # 重塑输入张量的形状
        input_ids = input_ids.reshape(-1, input_shape[-1])

        # 使用嵌入标记并乘以嵌入比例
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # 嵌入位置信息
        positions = self.embed_positions(position_ids + self.offset)

        # 将嵌入的标记和位置相加
        hidden_states = inputs_embeds + positions
        # 应用层归一化
        hidden_states = self.layernorm_embedding(hidden_states)

        # 应用丢弃层
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)

        # 将隐藏状态传递给层
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
        last_hidden_states = outputs[0]
        # 应用层归一化
        last_hidden_states = self.layer_norm(last_hidden_states)

        # 在上面应用`layernorm`后更新`hidden_states`中的最后一个元素
        hidden_states = None
        if output_hidden_states:
            hidden_states = outputs[1]
            hidden_states = hidden_states[:-1] + (last_hidden_states,)

        # 如果不返回字典，则重新组织输出
        if not return_dict:
            outputs = (last_hidden_states, hidden_states) + (outputs[2:] if output_hidden_states else outputs[1:])
            return tuple(v for v in outputs if v is not None)

        # 返回带有过去和交叉注意力的FlaxBaseModelOutput
        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=last_hidden_states,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )
# 从transformers.models.bart.modeling_flax_bart.FlaxBartModule复制而来，将Bart替换为MBart
class FlaxMBartModule(nn.Module):
    config: MBartConfig
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型

    def setup(self):
        # 创建共享的嵌入层，用于输入和输出的词汇表
        self.shared = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.init_std),
            dtype=self.dtype,
        )

        # 创建MBart编码器和解码器
        self.encoder = FlaxMBartEncoder(self.config, dtype=self.dtype, embed_tokens=self.shared)
        self.decoder = FlaxMBartDecoder(self.config, dtype=self.dtype, embed_tokens=self.shared)

    def _get_encoder_module(self):
        return self.encoder

    def _get_decoder_module(self):
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
        # 调用编码器得到编码器输出
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        # 调用解码器得到解码器输出
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

        # 如果不返回字典，则将解码器和编码器输出连接起来返回
        if not return_dict:
            return decoder_outputs + encoder_outputs

        # 返回FlaxSeq2SeqModelOutput对象，包含解码器和编码器的输出
        return FlaxSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class FlaxMBartPreTrainedModel(FlaxPreTrainedModel):
    config_class = MBartConfig
    base_model_prefix: str = "model"
    module_class: nn.Module = None

    def __init__(
        self,
        config: MBartConfig,
        input_shape: Tuple[int] = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        # 使用配置和关键字参数实例化模块对象
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 调用父类的初始化方法
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量
        input_ids = jnp.zeros(input_shape, dtype="i4")
        # 确保初始化通过适用于FlaxMBartForSequenceClassificationModule
        input_ids = input_ids.at[(..., -1)].set(self.config.eos_token_id)
        attention_mask = jnp.ones_like(input_ids)
        decoder_input_ids = input_ids
        decoder_attention_mask = jnp.ones_like(input_ids)

        batch_size, sequence_length = input_ids.shape
        position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))
        decoder_position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # 分割随机数生成器
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 初始化模型参数
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
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    # 从transformers.models.bart.modeling_flax_bart.FlaxBartPreTrainedModel.init_cache复制并将Bart->MBart
    # 初始化缓存，用于快速自回归解码
    def init_cache(self, batch_size, max_length, encoder_outputs):
        r"""
        Args:
            batch_size (`int`):
                用于快速自回归解码的批处理大小。定义了初始化缓存的批处理大小。
            max_length (`int`):
                自回归解码的最大可能长度。定义了初始化缓存的序列长度。
            encoder_outputs (`Union[FlaxBaseModelOutput, tuple(tuple(jnp.ndarray)]`):
                `encoder_outputs` 包括 (`last_hidden_state`, *可选*: `hidden_states`, *可选*: `attentions`)。
                `last_hidden_state` 的形状为 `(batch_size, sequence_length, hidden_size)`，*可选* 是编码器最后一层的隐藏状态的序列。
                在解码器的交叉注意力中使用。
        """
        # 初始化用于检索缓存的输入变量
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

        # 初始化模型变量以检索缓存
        init_variables = self.module.init(
            jax.random.PRNGKey(0),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            encoder_hidden_states=encoder_outputs[0],
            init_cache=True,
            method=_decoder_forward,  # 我们只需要调用解码器来初始化缓存
        )
        # 返回解冻后的缓存
        return unfreeze(init_variables["cache"])

    @add_start_docstrings(MBART_ENCODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutput, config_class=MBartConfig)
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
    ):
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, FlaxMBartForConditionalGeneration

        >>> model = FlaxMBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")

        >>> text = "My friends are cool but they eat too many carbs."
        >>> inputs = tokenizer(text, max_length=1024, return_tensors="jax")
        >>> encoder_outputs = model.encode(**inputs)
        ```py"""
        # 设置输出注意力权重，默认为配置中的值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置输出隐藏状态，默认为配置中的值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置返回字典，默认为配置中的值
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 如果注意力掩码为空，则创建一个全为1的注意力掩码
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        # 如果位置编码为空，则根据输入的input_ids创建位置编码
        if position_ids is None:
            batch_size, sequence_length = input_ids.shape
            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # 处理任何需要的伪随机数生成器
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        # 定义编码器前向传播函数
        def _encoder_forward(module, input_ids, attention_mask, position_ids, **kwargs):
            encode_module = module._get_encoder_module()
            return encode_module(input_ids, attention_mask, position_ids, **kwargs)

        # 应用模型的前向传播
        return self.module.apply(
            {"params": params or self.params},
            input_ids=jnp.array(input_ids, dtype="i4"),
            attention_mask=jnp.array(attention_mask, dtype="i4"),
            position_ids=jnp.array(position_ids, dtype="i4"),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=not train,
            rngs=rngs,
            method=_encoder_forward,
        )

    # 添加解码器输入文档字符串
    @add_start_docstrings(MBART_DECODE_INPUTS_DOCSTRING)
    # 替换返回文档字符串
    @replace_return_docstrings(output_type=FlaxBaseModelOutputWithPastAndCrossAttentions, config_class=MBartConfig)
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
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
    # 添加输入文档字符串到模型前向传播
    @add_start_docstrings_to_model_forward(MBART_INPUTS_DOCSTRING)
    # 定义一个调用函数，接受多个参数
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
        # 如果未指定输出注意力权重，则使用配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未指定输出隐藏状态，则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定返回字典，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 准备编码器输入
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        if position_ids is None:
            batch_size, sequence_length = input_ids.shape
            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # 准备解码器输入
        if decoder_input_ids is None:
            decoder_input_ids = shift_tokens_right(input_ids, self.config.pad_token_id)
        if decoder_attention_mask is None:
            decoder_attention_mask = jnp.ones_like(decoder_input_ids)
        if decoder_position_ids is None:
            batch_size, sequence_length = decoder_input_ids.shape
            decoder_position_ids = jnp.broadcast_to(
                jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
            )

        # 处理任何 PRNG（伪随机数生成器）如果需要
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        # 调用模块的应用方法，传入参数和输入数据
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
# 为 FlaxMBartModel 类添加起始文档字符串，描述其输出原始隐藏状态而不带特定头部的 MBart 模型转换器
# 并继承自 FlaxMBartPreTrainedModel 类
@add_start_docstrings(
    "The bare MBart Model transformer outputting raw hidden-states without any specific head on top.",
    MBART_START_DOCSTRING,
)
class FlaxMBartModel(FlaxMBartPreTrainedModel):
    # MBart 模型的配置
    config: MBartConfig
    # 计算的数据类型
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    # 模块类为 FlaxMBartModule
    module_class = FlaxMBartModule

# 为 FlaxMBartModel 类添加调用示例文档字符串
append_call_sample_docstring(FlaxMBartModel, _CHECKPOINT_FOR_DOC, FlaxSeq2SeqModelOutput, _CONFIG_FOR_DOC)

# 从 transformers.models.bart.modeling_flax_bart.FlaxBartForConditionalGenerationModule 复制代码，将 Bart 替换为 MBart
class FlaxMBartForConditionalGenerationModule(nn.Module):
    # MBart 模型的配置
    config: MBartConfig
    # 计算的数据类型
    dtype: jnp.dtype = jnp.float32
    # 偏置初始化函数
    bias_init: Callable[..., jnp.ndarray] = jax.nn.initializers.zeros

    # 设置函数
    def setup(self):
        # 创建 MBart 模型
        self.model = FlaxMBartModule(config=self.config, dtype=self.dtype)
        # 创建语言模型头部
        self.lm_head = nn.Dense(
            self.model.shared.num_embeddings,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        # 创建最终 logits 的偏置
        self.final_logits_bias = self.param("final_logits_bias", self.bias_init, (1, self.model.shared.num_embeddings))

    # 获取编码器模块
    def _get_encoder_module(self):
        return self.model.encoder

    # 获取解码器模块
    def _get_decoder_module(self):
        return self.model.decoder

    # 调用函数
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
        # 使用模型进行推理，传入输入和解码器相关参数
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            position_ids=position_ids,
            decoder_position_ids=decoder_position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        # 获取模型输出的隐藏状态
        hidden_states = outputs[0]

        # 如果配置了共享词嵌入，则使用共享的嵌入层
        if self.config.tie_word_embeddings:
            shared_embedding = self.model.variables["params"]["shared"]["embedding"]
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
        else:
            # 否则直接使用语言模型头部
            lm_logits = self.lm_head(hidden_states)

        # 添加最终的 logits 偏置
        lm_logits += jax.lax.stop_gradient(self.final_logits_bias.astype(self.dtype))

        # 如果不需要返回字典，则返回输出元组
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return output

        # 返回 FlaxSeq2SeqLMOutput 对象，包含各种输出信息
        return FlaxSeq2SeqLMOutput(
            logits=lm_logits,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
# 定义一个带有语言建模头的 MMBart 模型，可用于摘要生成
class FlaxMBartForConditionalGeneration(FlaxMBartPreTrainedModel):
    # 模型类别为 FlaxMBartForConditionalGenerationModule
    module_class = FlaxMBartForConditionalGenerationModule
    # 数据类型为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 解码方法，用于生成文本
    @add_start_docstrings(MBART_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxCausalLMOutputWithCrossAttentions, config_class=MBartConfig)
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
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
    # 为生成准备输入
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        max_length,
        attention_mask: Optional[jax.Array] = None,
        decoder_attention_mask: Optional[jax.Array] = None,
        encoder_outputs=None,
        **kwargs,
    ):
        # 初始化缓存
        batch_size, seq_length = decoder_input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length, encoder_outputs)
        # 注意通常需要在 attention_mask 中放入 0，以处理 x > input_ids.shape[-1] 和 x < cache_length 的情况
        # 但由于解码器使用因果掩码，这些位置已经被掩盖了
        # 因此我们可以在这里创建一个静态的 attention_mask，这对于编译更有效
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

    # 更新生��的输入
    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["decoder_position_ids"] = model_kwargs["decoder_position_ids"][:, -1:] + 1
        return model_kwargs


FLAX_MBART_CONDITIONAL_GENERATION_DOCSTRING = r"""
    返回：

    摘要生成示例：

    ```python
    >>> from transformers import AutoTokenizer, FlaxMBartForConditionalGeneration, MBartConfig
    # 从预训练模型中加载 FlaxMBartForConditionalGeneration 模型
    model = FlaxMBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")
    # 从预训练模型中加载 AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")

    # 待总结的文章内容
    ARTICLE_TO_SUMMARIZE = "Meine Freunde sind cool, aber sie essen zu viel Kuchen."
    # 使用 tokenizer 对文章进行编码，限制最大长度为1024，返回 numpy 格式
    inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="np")

    # 生成摘要
    summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=5).sequences
    # 打印解码后的摘要内容，跳过特殊标记并保留原始的分词空格
    print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))


Mask filling example:


    # 从 transformers 中导入 AutoTokenizer 和 FlaxMBartForConditionalGeneration
    from transformers import AutoTokenizer, FlaxMBartForConditionalGeneration

    # 从预训练模型中加载 FlaxMBartForConditionalGeneration 模型
    model = FlaxMBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")
    # 从预训练模型中加载 AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")

    # 德语的语言标识符 <LID> 为 de_DE
    TXT = "</s> Meine Freunde sind <mask> nett aber sie essen zu viel Kuchen. </s> de_DE"
    # 使用 tokenizer 对文本进行编码，不添加特殊标记，返回 numpy 格式的输入 ID
    input_ids = tokenizer([TXT], add_special_tokens=False, return_tensors="np")["input_ids"]

    # 获取模型的输出 logits
    logits = model(input_ids).logits
    # 找到被 mask 的位置索引
    masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero()[0].item()
    # 对 logits 进行 softmax 处理，得到概率分布
    probs = logits[0, masked_index].softmax(dim=0)
    # 获取概率最高的前5个值和对应的预测值
    values, predictions = probs.topk(5)

    # 解码预测值并以空格分割
    tokenizer.decode(predictions).split()
"""

# 覆盖调用文档字符串，将FlaxMBartForConditionalGeneration的文档字符串与MBART_INPUTS_DOCSTRING和FLAX_MBART_CONDITIONAL_GENERATION_DOCSTRING连接起来
overwrite_call_docstring(
    FlaxMBartForConditionalGeneration, MBART_INPUTS_DOCSTRING + FLAX_MBART_CONDITIONAL_GENERATION_DOCSTRING
)
# 追加替换返回文档字符串，将FlaxMBartForConditionalGeneration的输出类型设置为FlaxSeq2SeqLMOutput，配置类设置为_CONFIG_FOR_DOC
append_replace_return_docstrings(
    FlaxMBartForConditionalGeneration, output_type=FlaxSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC
)


# 从transformers.models.bart.modeling_flax_bart.FlaxBartForSequenceClassificationModule复制代码，将Bart->MBart
class FlaxMBartForSequenceClassificationModule(nn.Module):
    config: MBartConfig
    dtype: jnp.dtype = jnp.float32
    num_labels: Optional[int] = None

    def setup(self):
        # 初始化FlaxMBartModule和FlaxMBartClassificationHead
        self.model = FlaxMBartModule(config=self.config, dtype=self.dtype)
        self.classification_head = FlaxMBartClassificationHead(
            config=self.config,
            inner_dim=self.config.d_model,
            num_classes=self.num_labels if self.num_labels is not None else self.config.num_labels,
            pooler_dropout=self.config.classifier_dropout,
        )

    def _get_encoder_module(self):
        # 返回编码器模块
        return self.model.encoder

    def _get_decoder_module(self):
        # 返回解码器模块
        return self.model.decoder

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
            # 使用模型进行推理，传入输入的编码器和解码器的相关参数
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                position_ids=position_ids,
                decoder_position_ids=decoder_position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                deterministic=deterministic,
            )

            # 获取模型输出的最后一个隐藏状态
            hidden_states = outputs[0]  # last hidden state

            # 创建一个掩码，标记输入中的<eos> token
            eos_mask = jnp.where(input_ids == self.config.eos_token_id, 1, 0)

            # 处理特定情况以避免在 JIT 编译期间出现错误
            if type(eos_mask) != jax.interpreters.partial_eval.DynamicJaxprTracer:
                # 检查是否所有示例中的<eos> token数量相同
                if len(jnp.unique(eos_mask.sum(1))) > 1:
                    raise ValueError("All examples must have the same number of <eos> tokens.")

                # 检查是否输入中存在缺失的<eos> token
                if any(eos_mask.sum(1) == 0):
                    raise ValueError("There are missing <eos> tokens in input_ids")

                # 保证每个示例中只有最后一个<eos> token被标记为1
                eos_mask_noised = eos_mask + jnp.arange(eos_mask.shape[1]) * 1e-6
                eos_mask = jnp.where(eos_mask_noised == eos_mask_noised.max(1).reshape(-1, 1), 1, 0)

            # 计算句子表示，将隐藏状态与eos_mask相乘并求和
            sentence_representation = jnp.einsum("ijk, ij -> ijk", hidden_states, eos_mask).sum(1)
            # 使用分类头部对句子表示进行分类
            logits = self.classification_head(sentence_representation, deterministic=deterministic)

            # 如果不返回字典，则返回输出元组
            if not return_dict:
                output = (logits,) + outputs[1:]
                return output

            # 返回 FlaxSeq2SeqSequenceClassifierOutput 对象
            return FlaxSeq2SeqSequenceClassifierOutput(
                logits=logits,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )
# 为 FlaxMBartForSequenceClassification 类添加文档字符串，描述其作用是在 MBart 模型顶部添加一个序列分类/头部（在汇总输出之上的线性层），例如用于 GLUE 任务
@add_start_docstrings(
    """
    MBart model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE
    tasks.
    """,
    MBART_START_DOCSTRING,
)
# 设置 FlaxMBartForSequenceClassification 类的模块类为 FlaxMBartForSequenceClassificationModule
class FlaxMBartForSequenceClassification(FlaxMBartPreTrainedModel):
    module_class = FlaxMBartForSequenceClassificationModule
    dtype = jnp.float32

# 添加调用示例文档字符串到 FlaxMBartForSequenceClassification 类
append_call_sample_docstring(
    FlaxMBartForSequenceClassification,
    _CHECKPOINT_FOR_DOC,
    FlaxSeq2SeqSequenceClassifierOutput,
    _CONFIG_FOR_DOC,
)

# 从 transformers.models.bart.modeling_flax_bart.FlaxBartForQuestionAnsweringModule 复制代码，并将 Bart->MBart
class FlaxMBartForQuestionAnsweringModule(nn.Module):
    config: MBartConfig
    dtype: jnp.dtype = jnp.float32
    num_labels = 2

    def setup(self):
        self.model = FlaxMBartModule(config=self.config, dtype=self.dtype)
        self.qa_outputs = nn.Dense(
            self.num_labels, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std)
        )

    def _get_encoder_module(self):
        return self.model.encoder

    def _get_decoder_module(self):
        return self.model.decoder

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
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            position_ids=position_ids,
            decoder_position_ids=decoder_position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = jnp.split(logits, logits.shape[-1], axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return output

        return FlaxSeq2SeqQuestionAnsweringModelOutput(
            start_logits=start_logits,
            end_logits=end_logits,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

# 为 FlaxMBartForQuestionAnsweringModule 类添加文档字符串
@add_start_docstrings(
    """
    # MBart模型，顶部带有用于提取式问答任务（如SQuAD）的跨度分类头（在隐藏状态输出顶部的线性层，用于计算“跨度起始logits”和“跨度结束logits”）。
    # MBART_START_DOCSTRING
)
# 关闭函数定义

# 定义一个继承自FlaxMBartPreTrainedModel的类FlaxMBartForQuestionAnswering
class FlaxMBartForQuestionAnswering(FlaxMBartPreTrainedModel):
    module_class = FlaxMBartForQuestionAnsweringModule
    dtype = jnp.float32

# 调用一个函数，用于在文档中添加示例字符串
append_call_sample_docstring(
    FlaxMBartForQuestionAnswering,
    _CHECKPOINT_FOR_DOC,
    FlaxSeq2SeqQuestionAnsweringModelOutput,
    _CONFIG_FOR_DOC,
)
```