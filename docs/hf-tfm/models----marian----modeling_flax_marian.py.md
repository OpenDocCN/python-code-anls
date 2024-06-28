# `.\models\marian\modeling_flax_marian.py`

```py
# coding=utf-8
# 版权所有 2021 年 The Marian Team 作者和 Google Flax Team 作者以及 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）许可；
# 除非符合许可证的要求，否则您不能使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则不得以“原样”分发软件，
# 没有任何明示或暗示的保证或条件。请查阅许可证了解具体语言。
""" Flax Marian model."""

import math
import random
from functools import partial
from typing import Callable, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from jax.random import PRNGKey

from ...modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxBaseModelOutputWithPastAndCrossAttentions,
    FlaxCausalLMOutputWithCrossAttentions,
    FlaxSeq2SeqLMOutput,
    FlaxSeq2SeqModelOutput,
)
from ...modeling_flax_utils import (
    ACT2FN,
    FlaxPreTrainedModel,
    append_call_sample_docstring,
    append_replace_return_docstrings,
    overwrite_call_docstring,
)
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_marian import MarianConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "Helsinki-NLP/opus-mt-en-de"
_CONFIG_FOR_DOC = "MarianConfig"


MARIAN_START_DOCSTRING = r"""
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
    Parameters:
        config ([`MarianConfig`]): Model configuration class with all the parameters of the model.
            初始化模型配置类，包含所有模型参数。
            通过配置文件初始化不会加载与模型相关的权重，仅加载配置。
            参考 [`~FlaxPreTrainedModel.from_pretrained`] 方法以加载模型权重。

        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
            计算的数据类型。可以是 `jax.numpy.float32`、`jax.numpy.float16`（在GPU上）和 `jax.numpy.bfloat16`（在TPU上）之一。

            可用于在GPU或TPU上启用混合精度训练或半精度推断。
            如果指定，所有计算将使用给定的 `dtype` 执行。

            **注意，这仅指定计算的数据类型，不影响模型参数的数据类型。**

            如果要更改模型参数的数据类型，请参阅 [`~FlaxPreTrainedModel.to_fp16`] 和 [`~FlaxPreTrainedModel.to_bf16`]。
"""

MARIAN_INPUTS_DOCSTRING = r"""
"""


MARIAN_ENCODE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
            输入序列标记的索引，用于词汇表中的标记。默认情况下会忽略填充。

            可以使用 [`AutoTokenizer`] 获得这些索引。参见 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`] 了解详情。

            [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *可选*):
            遮罩，用于避免在填充的标记索引上进行注意力计算。遮罩值选在 `[0, 1]` 范围内：

            - 1 表示**不遮罩**的标记，
            - 0 表示**遮罩**的标记。

            [什么是注意力遮罩？](../glossary#attention-mask)
        position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *可选*):
            输入序列标记在位置嵌入中的位置索引。选择范围是 `[0, config.max_position_embeddings - 1]`。
        output_attentions (`bool`, *可选*):
            是否返回所有注意力层的注意力张量。更多细节请参见返回的张量中的 `attentions` 字段。
        output_hidden_states (`bool`, *可选*):
            是否返回所有层的隐藏状态。更多细节请参见返回的张量中的 `hidden_states` 字段。
        return_dict (`bool`, *可选*):
            是否返回一个 [`~utils.ModelOutput`] 而不是简单的元组。
"""

MARIAN_DECODE_INPUTS_DOCSTRING = r"""
"""


def create_sinusoidal_positions(n_pos, dim):
    """
    创建正弦位置编码。

    Args:
        n_pos (int): 位置数量。
        dim (int): 编码维度。

    Returns:
        jnp.ndarray: 正弦位置编码的数组。
    """
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    sentinel = dim // 2 + dim % 2
    out = np.zeros_like(position_enc)
    out[:, 0:sentinel] = np.sin(position_enc[:, 0::2])
    out[:, sentinel:] = np.cos(position_enc[:, 1::2])

    return jnp.array(out)


# Copied from transformers.models.bart.modeling_flax_bart.shift_tokens_right
def shift_tokens_right(input_ids: jnp.ndarray, pad_token_id: int, decoder_start_token_id: int) -> jnp.ndarray:
    """
    将输入的标记向右移动一位。
    """
    shifted_input_ids = jnp.zeros_like(input_ids)
    shifted_input_ids = shifted_input_ids.at[:, 1:].set(input_ids[:, :-1])
    shifted_input_ids = shifted_input_ids.at[:, 0].set(decoder_start_token_id)

    shifted_input_ids = jnp.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)
    return shifted_input_ids


# Copied from transformers.models.bart.modeling_flax_bart.FlaxBartAttention with Bart->Marian
class FlaxMarianAttention(nn.Module):
    """
    Marian 模型的注意力机制模块。
    """
    config: MarianConfig
    embed_dim: int
    num_heads: int
    dropout: float = 0.0
    causal: bool = False
    bias: bool = True
    dtype: jnp.dtype = jnp.float32  # 计算中使用的数据类型，默认为 jnp.float32

    def setup(self) -> None:
        self.head_dim = self.embed_dim // self.num_heads  # 计算每个头部的维度
        if self.head_dim * self.num_heads != self.embed_dim:  # 检查 embed_dim 是否能被 num_heads 整除
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )

        # 创建一个部分应用了 nn.Dense 的函数，用于创建全连接层
        dense = partial(
            nn.Dense,
            self.embed_dim,
            use_bias=self.bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )

        # 初始化查询、键、值、输出的全连接层
        self.q_proj, self.k_proj, self.v_proj = dense(), dense(), dense()
        self.out_proj = dense()

        # 初始化 dropout 层
        self.dropout_layer = nn.Dropout(rate=self.dropout)

        if self.causal:
            # 如果需要因果注意力，创建因果 mask
            self.causal_mask = make_causal_mask(
                jnp.ones((1, self.config.max_position_embeddings), dtype="bool"), dtype="bool"
            )

    def _split_heads(self, hidden_states):
        # 将隐藏状态张量按照头部数目和头部维度进行重塑
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        # 将分离的头部重新合并成原来的形状
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    @nn.compact
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slightly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # 检测是否初始化缓存数据
        is_initialized = self.has_variable("cache", "cached_key")
        # 获取缓存的键（key）并初始化为零数组，其形状和数据类型与输入的键（key）相同
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        # 获取缓存的值（value）并初始化为零数组，其形状和数据类型与输入的值（value）相同
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        # 获取缓存索引（index），如果不存在则初始化为零
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            # 获取批次维度的数量和最大长度、注意力头数、每个头部的深度
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # 更新键（key）和值（value）缓存，使用新的一维空间切片
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            # 更新缓存中的键（key）和值（value）
            cached_key.value = key
            cached_value.value = value
            # 更新缓存索引，增加已更新的缓存向量数
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # 对于缓存的解码器自注意力，创建因果掩码：我们的单个查询位置只能关注已生成和缓存的键位置，而不是剩余的零元素
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            # 合并因果掩码和输入的注意力掩码
            attention_mask = combine_masks(pad_mask, attention_mask)
        # 返回更新后的键（key）、值（value）和注意力掩码
        return key, value, attention_mask
# 从 transformers.models.bart.modeling_flax_bart.FlaxBartEncoderLayer 复制代码并将 Bart->Marian 替换
class FlaxMarianEncoderLayer(nn.Module):
    # Marian 模型配置
    config: MarianConfig
    # 计算的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 设置层的初始化操作
    def setup(self) -> None:
        # 设置嵌入维度为模型配置中的 d_model
        self.embed_dim = self.config.d_model
        # 定义自注意力层
        self.self_attn = FlaxMarianAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.encoder_attention_heads,
            dropout=self.config.attention_dropout,
            dtype=self.dtype,
        )
        # 自注意力层后的 Layer Normalization
        self.self_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        # Dropout 层
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)
        # 激活函数
        self.activation_fn = ACT2FN[self.config.activation_function]
        # 激活函数后的 Dropout 层
        self.activation_dropout_layer = nn.Dropout(rate=self.config.activation_dropout)
        # 第一个全连接层，使用 jax 的正态分布初始化
        self.fc1 = nn.Dense(
            self.config.encoder_ffn_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        # 第二个全连接层，输出维度为 embed_dim，同样使用正态分布初始化
        self.fc2 = nn.Dense(
            self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std)
        )
        # 最终的 Layer Normalization
        self.final_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    # 实现类的调用方法，对输入的隐藏状态进行编码处理
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        output_attentions: bool = True,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray]:
        # 保存残差连接
        residual = hidden_states
        # 应用自注意力机制，得到新的隐藏状态和注意力权重
        hidden_states, attn_weights = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask)

        # 应用 Dropout
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 残差连接和新隐藏状态相加
        hidden_states = residual + hidden_states
        # 应用自注意力层后的 Layer Normalization
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 保存残差连接
        residual = hidden_states
        # 应用激活函数和第一个全连接层
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 应用激活函数后的 Dropout
        hidden_states = self.activation_dropout_layer(hidden_states, deterministic=deterministic)
        # 应用第二个全连接层
        hidden_states = self.fc2(hidden_states)
        # 应用 Dropout
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 残差连接和新隐藏状态相加
        hidden_states = residual + hidden_states
        # 应用最终的 Layer Normalization
        hidden_states = self.final_layer_norm(hidden_states)

        # 输出为一个元组，包含最终的隐藏状态
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，加入到输出元组中
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# 从 transformers.models.bart.modeling_flax_bart.FlaxBartEncoderLayerCollection 复制代码并将 Bart->Marian 替换
class FlaxMarianEncoderLayerCollection(nn.Module):
    # Marian 模型配置
    config: MarianConfig
    # 计算的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    # 设置层的初始化操作
    def setup(self):
        # 创建编码层的集合，每个编码层使用 FlaxMarianEncoderLayer 创建
        self.layers = [
            FlaxMarianEncoderLayer(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.encoder_layers)
        ]
        # 编码层的 dropout 率
        self.layerdrop = self.config.encoder_layerdrop
    # 定义一个特殊方法 __call__，使得对象可以被调用
    def __call__(
        self,
        hidden_states,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 如果需要输出注意力权重，则初始化空的元组用于存储所有注意力权重
        all_attentions = () if output_attentions else None
        # 如果需要输出隐藏状态，则初始化空的元组用于存储所有隐藏状态
        all_hidden_states = () if output_hidden_states else None

        # 遍历所有的编码器层
        for encoder_layer in self.layers:
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            # 添加 LayerDrop 功能，参见论文 https://arxiv.org/abs/1909.11556 的描述
            dropout_probability = random.uniform(0, 1)
            # 如果非确定性且随机数小于层级丢弃率，则跳过当前层
            if not deterministic and (dropout_probability < self.layerdrop):
                layer_outputs = (None, None)  # 跳过层的输出
            else:
                # 否则，调用当前编码器层的前向传播函数
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions,
                    deterministic,
                )
            # 更新隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]
            # 如果需要输出注意力权重，则将当前层的注意力权重添加到 all_attentions 中
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # 如果需要输出隐藏状态，则将最后一个隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 将最终的输出整合为一个元组，根据 return_dict 决定返回类型
        outputs = (hidden_states, all_hidden_states, all_attentions)

        # 如果不需要以字典形式返回，则返回一个元组，去除其中为 None 的部分
        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        # 否则，以 FlaxBaseModelOutput 对象形式返回所有输出
        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )
# 从 transformers.models.bart.modeling_flax_bart.FlaxBartDecoderLayer 复制代码，并将 Bart 更改为 Marian
class FlaxMarianDecoderLayer(nn.Module):
    # 使用 MarianConfig 类型的配置参数 config
    config: MarianConfig
    # 默认数据类型为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 设置方法，初始化层的各项参数
    def setup(self) -> None:
        # 获取嵌入维度，等于配置中的 d_model
        self.embed_dim = self.config.d_model
        # 定义自注意力层，使用 FlaxMarianAttention 类
        self.self_attn = FlaxMarianAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.decoder_attention_heads,
            dropout=self.config.attention_dropout,
            causal=True,
            dtype=self.dtype,
        )
        # 定义 dropout 层，用于 self-attention 和全连接层之间
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)
        # 激活函数，根据配置中的激活函数选择对应的函数
        self.activation_fn = ACT2FN[self.config.activation_function]
        # 激活函数的 dropout 层
        self.activation_dropout_layer = nn.Dropout(rate=self.config.activation_dropout)

        # 自注意力层的 LayerNorm 层
        self.self_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        # 定义编码器注意力层，使用 FlaxMarianAttention 类
        self.encoder_attn = FlaxMarianAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.decoder_attention_heads,
            dropout=self.config.attention_dropout,
            dtype=self.dtype,
        )
        # 编码器注意力层的 LayerNorm 层
        self.encoder_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        # 第一个全连接层，输入维度为 decoder_ffn_dim，输出维度与嵌入维度相同
        self.fc1 = nn.Dense(
            self.config.decoder_ffn_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        # 第二个全连接层，输入维度与嵌入维度相同，输出维度也与嵌入维度相同
        self.fc2 = nn.Dense(
            self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std)
        )
        # 最终输出的 LayerNorm 层
        self.final_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    # 对象调用方法，定义层的前向传播逻辑
    def __call__(
        self,
        hidden_states: jnp.ndarray,  # 输入的隐藏状态
        attention_mask: jnp.ndarray,  # 注意力掩码
        encoder_hidden_states: Optional[jnp.ndarray] = None,  # 编码器的隐藏状态（可选）
        encoder_attention_mask: Optional[jnp.ndarray] = None,  # 编码器的注意力掩码（可选）
        init_cache: bool = False,  # 是否初始化缓存（默认为 False）
        output_attentions: bool = True,  # 是否输出注意力权重（默认为 True）
        deterministic: bool = True,  # 是否确定性计算（默认为 True）
    ) -> Tuple[jnp.ndarray]:
        residual = hidden_states

        # Self Attention
        # 使用自注意力机制处理隐藏状态，返回处理后的隐藏状态和注意力权重
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, init_cache=init_cache
        )
        # 应用 dropout 层，用于防止过拟合
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 添加残差连接
        hidden_states = residual + hidden_states
        # 对处理后的隐藏状态进行层归一化
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_weights = None
        # 如果有编码器隐藏状态，执行交叉注意力机制
        if encoder_hidden_states is not None:
            residual = hidden_states

            # 使用编码器注意力机制处理隐藏状态，返回处理后的隐藏状态和注意力权重
            hidden_states, cross_attn_weights = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            # 应用 dropout 层，用于防止过拟合
            hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
            # 添加残差连接
            hidden_states = residual + hidden_states
            # 对处理后的隐藏状态进行层归一化
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        # 应用激活函数和全连接层 fc1
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 应用 dropout 层，用于防止过拟合
        hidden_states = self.activation_dropout_layer(hidden_states, deterministic=deterministic)
        # 应用全连接层 fc2
        hidden_states = self.fc2(hidden_states)
        # 应用 dropout 层，用于防止过拟合
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 添加残差连接
        hidden_states = residual + hidden_states
        # 对处理后的隐藏状态进行层归一化
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        # 如果需要输出注意力权重，将自注意力和交叉注意力的权重添加到输出中
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        # 返回最终输出
        return outputs
# Copied from transformers.models.bart.modeling_flax_bart.FlaxBartDecoderLayerCollection with Bart->Marian
# 定义一个名为FlaxMarianDecoderLayerCollection的类，作为Marian模型的解码器层集合

class FlaxMarianDecoderLayerCollection(nn.Module):
    config: MarianConfig
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型

    def setup(self):
        # 初始化解码器层列表，每个解码器层使用FlaxMarianDecoderLayer构造，数量由配置文件self.config.decoder_layers决定
        self.layers = [
            FlaxMarianDecoderLayer(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.decoder_layers)
        ]
        # 设置LayerDrop的概率，从配置文件self.config.decoder_layerdrop中获取

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
        # decoder layers
        # 如果需要输出隐藏状态，则初始化all_hidden_states为一个空元组，否则为None
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力分布，则初始化all_self_attns为一个空元组，否则为None
        all_self_attns = () if output_attentions else None
        # 如果需要输出交叉注意力分布，并且encoder_hidden_states不为None，则初始化all_cross_attentions为一个空元组，否则为None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        # 遍历每个解码器层进行处理
        for decoder_layer in self.layers:
            if output_hidden_states:
                # 如果需要输出隐藏状态，将当前的hidden_states加入all_hidden_states中
                all_hidden_states += (hidden_states,)
                # 添加LayerDrop功能，详情见论文https://arxiv.org/abs/1909.11556

            # 生成一个0到1之间的随机数，作为Dropout的概率
            dropout_probability = random.uniform(0, 1)
            # 如果不是确定性的计算，并且随机数小于self.layerdrop，则不执行当前解码器层的计算
            if not deterministic and (dropout_probability < self.layerdrop):
                layer_outputs = (None, None, None)
            else:
                # 否则，执行当前解码器层的计算，传入相应的参数
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    init_cache=init_cache,
                    output_attentions=output_attentions,
                    deterministic=deterministic,
                )

            # 更新hidden_states为当前解码器层的输出中的第一个元素
            hidden_states = layer_outputs[0]
            if output_attentions:
                # 如果需要输出注意力分布，将当前解码器层的注意力分布加入all_self_attns中
                all_self_attns += (layer_outputs[1],)

                # 如果encoder_hidden_states不为None，则将当前解码器层的交叉注意力分布加入all_cross_attentions中
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # 将最后一个解码器层的隐藏状态加入all_hidden_states中
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 汇总所有的输出信息到outputs列表中
        outputs = [hidden_states, all_hidden_states, all_self_attns, all_cross_attentions]

        # 如果不需要以字典形式返回结果，则返回outputs中不为None的元素构成的元组
        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        # 否则，以FlaxBaseModelOutputWithPastAndCrossAttentions对象的形式返回结果
        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


# 定义一个名为FlaxMarianEncoder的类，作为Marian模型的编码器
class FlaxMarianEncoder(nn.Module):
    config: MarianConfig
    embed_tokens: nn.Embed
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型
    # 初始化模型的设置，包括dropout层和embedding相关的参数设置
    def setup(self):
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)

        # 设置embedding的维度
        embed_dim = self.config.d_model
        # 设置最大的位置编码长度
        self.max_source_positions = self.config.max_position_embeddings
        # 如果设置了scale_embedding标志位，则对embedding进行缩放
        self.embed_scale = math.sqrt(embed_dim) if self.config.scale_embedding else 1.0

        # 创建sinusoidal位置编码矩阵
        self.embed_positions = create_sinusoidal_positions(self.config.max_position_embeddings, embed_dim)
        # 初始化encoder层集合
        self.layers = FlaxMarianEncoderLayerCollection(self.config, self.dtype)

    # 模型的调用方法，输入参数和返回类型可选
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
        # 获取输入的形状信息
        input_shape = input_ids.shape
        # 重新整形输入id
        input_ids = input_ids.reshape(-1, input_shape[-1])

        # 对输入id进行embedding并缩放
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # 根据位置id从预先创建的位置编码中取出对应的位置信息
        positions = jnp.take(self.embed_positions, position_ids, axis=0)
        # 明确地将位置信息的数据类型转换为和输入embedding相同的数据类型
        positions = positions.astype(inputs_embeds.dtype)

        # 将embedding和位置信息相加得到最终的隐藏状态表示
        hidden_states = inputs_embeds + positions
        # 应用dropout层到隐藏状态
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)

        # 调用模型的encoder层进行前向传播
        outputs = self.layers(
            hidden_states,
            attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果不要求返回字典形式的输出，则直接返回模型的outputs对象
        if not return_dict:
            return outputs

        # 返回以FlaxBaseModelOutput对象封装的输出结果，包括最终的隐藏状态、所有隐藏状态以及注意力分布
        return FlaxBaseModelOutput(
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
class FlaxMarianDecoder(nn.Module):
    config: MarianConfig  # 类型注解，指定config属性为MarianConfig类型
    embed_tokens: nn.Embed  # 类型注解，指定embed_tokens属性为nn.Embed类型
    dtype: jnp.dtype = jnp.float32  # 计算中使用的数据类型，默认为jnp.float32

    def setup(self):
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)  # 初始化dropout层，使用config中的dropout率

        embed_dim = self.config.d_model  # 获取config中的d_model作为嵌入维度
        self.max_target_positions = self.config.max_position_embeddings  # 设置最大目标位置为config中的max_position_embeddings
        self.embed_scale = math.sqrt(self.config.d_model) if self.config.scale_embedding else 1.0  # 根据scale_embedding标志设置嵌入缩放因子

        self.embed_positions = create_sinusoidal_positions(self.config.max_position_embeddings, embed_dim)  # 创建正弦位置编码
        self.layers = FlaxMarianDecoderLayerCollection(self.config, self.dtype)  # 初始化解码器层集合

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
        input_shape = input_ids.shape  # 获取输入张量的形状
        input_ids = input_ids.reshape(-1, input_shape[-1])  # 将输入张量重新形状为二维张量

        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale  # 使用嵌入令牌和缩放因子对输入进行嵌入

        # 嵌入位置信息
        positions = jnp.take(self.embed_positions, position_ids, axis=0)
        # 明确地将位置转换为与inputs_embeds相同的数据类型，因为self.embed_positions未注册为参数
        positions = positions.astype(inputs_embeds.dtype)

        hidden_states = inputs_embeds + positions  # 将嵌入的输入和位置编码相加

        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)  # 应用dropout层

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
        )  # 将hidden_states传递给解码器层进行处理

        if not return_dict:
            return outputs

        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )  # 如果return_dict为True，则返回带有注意力信息的输出


class FlaxMarianModule(nn.Module):
    config: MarianConfig  # 类型注解，指定config属性为MarianConfig类型
    dtype: jnp.dtype = jnp.float32  # 计算中使用的数据类型，默认为jnp.float32

    def setup(self):
        self.shared = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.init_std),
        )  # 初始化共享的嵌入层，使用config中的词汇大小和d_model，并使用正态分布初始化器初始化

        self.encoder = FlaxMarianEncoder(self.config, dtype=self.dtype, embed_tokens=self.shared)  # 初始化编码器
        self.decoder = FlaxMarianDecoder(self.config, dtype=self.dtype, embed_tokens=self.shared)  # 初始化解码器

    def _get_encoder_module(self):
        return self.encoder  # 返回编码器模块
    # 返回解码器模块对象
    def _get_decoder_module(self):
        return self.decoder

    # 实现调用操作，执行序列到序列模型的前向传播
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
        # 使用编码器模型处理输入序列
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        # 使用解码器模型处理目标序列
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

        # 如果不需要返回字典格式的输出，则将编码器和解码器的输出拼接并返回
        if not return_dict:
            return decoder_outputs + encoder_outputs

        # 返回序列到序列模型的输出对象，其中包含解码器和编码器的相关隐藏状态和注意力权重
        return FlaxSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
class FlaxMarianPreTrainedModel(FlaxPreTrainedModel):
    # 使用 MarianConfig 作为配置类
    config_class = MarianConfig
    # 基础模型前缀为 "model"
    base_model_prefix: str = "model"
    # 模块类暂未定义
    module_class: nn.Module = None

    def __init__(
        self,
        config: MarianConfig,
        input_shape: Tuple[int] = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        # 创建模块实例，传入配置和其他参数
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 调用父类构造函数初始化模型
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量 input_ids
        input_ids = jnp.zeros(input_shape, dtype="i4")
        # 设置 input_ids 的最后一个位置为 eos_token_id
        input_ids = input_ids.at[(..., -1)].set(self.config.eos_token_id)
        # 初始化 attention_mask 为全1的张量
        attention_mask = jnp.ones_like(input_ids)
        # 将 decoder_input_ids 初始化为 input_ids
        decoder_input_ids = input_ids
        # 将 decoder_attention_mask 初始化为全1的张量
        decoder_attention_mask = jnp.ones_like(input_ids)

        # 获取 input_ids 的形状信息
        batch_size, sequence_length = input_ids.shape
        # 生成 position_ids，广播形状为 (batch_size, sequence_length)
        position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))
        # 生成 decoder_position_ids，广播形状为 (batch_size, sequence_length)
        decoder_position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # 分割随机数生成器 rng，返回 params_rng 和 dropout_rng
        params_rng, dropout_rng = jax.random.split(rng)
        # 构建随机数字典 rngs，包含 params_rng 和 dropout_rng
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 使用模块的初始化方法初始化模型参数，返回随机生成的参数 random_params
        random_params = self.module.init(
            rngs,
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            position_ids,
            decoder_position_ids,
        )["params"]

        # 如果传入了预定义的参数 params
        if params is not None:
            # 展平 random_params 和 params
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            # 处理缺失的键
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            # 清空缺失键集合
            self._missing_keys = set()
            # 冻结并返回 params
            return freeze(unflatten_dict(params))
        else:
            # 返回随机生成的参数 random_params
            return random_params
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
        # 初始化用于检索缓存的输入变量
        decoder_input_ids = jnp.ones((batch_size, max_length), dtype="i4")
        # 创建与decoder_input_ids相同形状的全1张量，用作解码器的注意力遮罩
        decoder_attention_mask = jnp.ones_like(decoder_input_ids)
        # 使用广播方式生成位置编码，形状与decoder_input_ids相同
        decoder_position_ids = jnp.broadcast_to(
            jnp.arange(jnp.atleast_2d(decoder_input_ids).shape[-1]), decoder_input_ids.shape
        )

        def _decoder_forward(module, decoder_input_ids, decoder_attention_mask, decoder_position_ids, **kwargs):
            # 获取解码器模块
            decoder_module = module._get_decoder_module()
            # 调用解码器模块进行前向传播
            return decoder_module(decoder_input_ids, decoder_attention_mask, decoder_position_ids, **kwargs)

        # 使用给定的输入参数初始化模型的变量
        init_variables = self.module.init(
            jax.random.PRNGKey(0),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            encoder_hidden_states=encoder_outputs[0],
            init_cache=True,
            method=_decoder_forward,  # 只需调用解码器以初始化缓存
        )
        # 返回解冻后的初始化变量中的缓存部分
        return unfreeze(init_variables["cache"])



    @add_start_docstrings(MARIAN_ENCODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutput, config_class=MarianConfig)
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
    @add_start_docstrings(MARIAN_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutputWithPastAndCrossAttentions, config_class=MarianConfig)
    # 使用指定的文档字符串注解这个方法，将其标记为用于解码的函数，并替换返回值的文档字符串
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
    ):
        r"""
        Returns:

        Example:

        ```
        >>> from transformers import AutoTokenizer, FlaxMarianMTModel

        >>> tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
        >>> model = FlaxMarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-de")

        >>> text = "My friends are cool but they eat too many carbs."
        >>> inputs = tokenizer(text, max_length=64, return_tensors="jax")
        >>> encoder_outputs = model.encode(**inputs)
        ```
        
        Defines whether to output attentions or not. Defaults to `True` if `output_attentions` is not `None`, else `False`.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        
        Defines whether to output hidden states or not. Defaults to `True` if `output_hidden_states` is not `None`, else `False`.
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        
        Defines whether to return outputs as a dictionary. Defaults to `True` if `return_dict` is not `None`, else `False`.
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # If no attention mask is provided, create one with all elements set to 1.
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        
        # If no position ids are provided, generate them based on input_ids dimensions.
        if position_ids is None:
            batch_size, sequence_length = input_ids.shape
            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        # Define the function to perform the forward pass through the encoder.
        def _encoder_forward(module, input_ids, attention_mask, position_ids, **kwargs):
            encode_module = module._get_encoder_module()
            return encode_module(input_ids, attention_mask, position_ids, **kwargs)

        # Apply the encoder module on the input data and return the outputs.
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
    # 定义一个特殊方法 __call__，使实例对象可以像函数一样调用
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
        # 确定是否输出注意力权重信息，默认使用配置中的设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 确定是否输出隐藏状态信息，默认使用配置中的设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 确定是否返回字典格式的输出，默认使用配置中的设置
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 准备编码器的输入
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)  # 如果未提供注意力掩码，则创建一个全为1的数组作为掩码
        if position_ids is None:
            batch_size, sequence_length = input_ids.shape
            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))
            # 如果未提供位置编码，则根据输入的长度创建位置编码

        # 准备解码器的输入
        if decoder_input_ids is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, decoder_start_token_id=self.config.decoder_start_token_id
            )
            # 如果未提供解码器输入的令牌 ID，则通过右移输入的令牌来创建解码器输入
        if decoder_attention_mask is None:
            decoder_attention_mask = jnp.ones_like(decoder_input_ids)
            # 如果未提供解码器的注意力掩码，则创建一个全为1的数组作为掩码
        if decoder_position_ids is None:
            batch_size, sequence_length = decoder_input_ids.shape
            decoder_position_ids = jnp.broadcast_to(
                jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
            )
            # 如果未提供解码器的位置编码，则根据解码器输入的长度创建位置编码

        # 处理可能需要的任何随机数生成器
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        # 调用内部模块的 apply 方法，传递参数和所有输入，以执行模型的前向传播
        return self.module.apply(
            {"params": params or self.params},  # 传递模型参数
            input_ids=jnp.array(input_ids, dtype="i4"),  # 转换输入令牌 ID 为 JAX 数组
            attention_mask=jnp.array(attention_mask, dtype="i4"),  # 转换注意力掩码为 JAX 数组
            position_ids=jnp.array(position_ids, dtype="i4"),  # 转换位置编码为 JAX 数组
            decoder_input_ids=jnp.array(decoder_input_ids, dtype="i4"),  # 转换解码器输入为 JAX 数组
            decoder_attention_mask=jnp.array(decoder_attention_mask, dtype="i4"),  # 转换解码器的注意力掩码为 JAX 数组
            decoder_position_ids=jnp.array(decoder_position_ids, dtype="i4"),  # 转换解码器的位置编码为 JAX 数组
            output_attentions=output_attentions,  # 是否输出注意力权重信息
            output_hidden_states=output_hidden_states,  # 是否输出隐藏状态信息
            return_dict=return_dict,  # 是否返回字典格式的输出
            deterministic=not train,  # 是否处于推理模式
            rngs=rngs,  # 随机数生成器的字典
        )
@add_start_docstrings(
    "The bare Marian Model transformer outputting raw hidden-states without any specific head on top.",
    MARIAN_START_DOCSTRING,
)
class FlaxMarianModel(FlaxMarianPreTrainedModel):
    config: MarianConfig
    dtype: jnp.dtype = jnp.float32  # 计算时使用的数据类型
    module_class = FlaxMarianModule


append_call_sample_docstring(FlaxMarianModel, _CHECKPOINT_FOR_DOC, FlaxSeq2SeqModelOutput, _CONFIG_FOR_DOC)


class FlaxMarianMTModule(nn.Module):
    config: MarianConfig
    dtype: jnp.dtype = jnp.float32
    bias_init: Callable[..., jnp.ndarray] = jax.nn.initializers.zeros

    def setup(self):
        self.model = FlaxMarianModule(config=self.config, dtype=self.dtype)  # 初始化Marian模型
        self.lm_head = nn.Dense(
            self.model.shared.num_embeddings,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )  # 初始化语言模型头部，用于生成LM预测
        self.final_logits_bias = self.param("final_logits_bias", self.bias_init, (1, self.model.shared.num_embeddings))  # 初始化最终的logits偏置项

    def _get_encoder_module(self):
        return self.model.encoder  # 返回编码器模块

    def _get_decoder_module(self):
        return self.model.decoder  # 返回解码器模块

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
        )  # 调用Marian模型进行正向传播计算

        hidden_states = outputs[0]  # 获取模型输出的隐藏状态

        if self.config.tie_word_embeddings:
            shared_embedding = self.model.variables["params"]["shared"]["embedding"]
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)  # 计算LM预测的logits

        lm_logits += self.final_logits_bias.astype(self.dtype)  # 加上最终的logits偏置项

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return output  # 返回LM预测logits和其他可能的输出

        return FlaxSeq2SeqLMOutput(
            logits=lm_logits,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )  # 返回Seq2Seq LM模型的输出
# 添加模型文档字符串，标识该类为带语言建模头的MARIAN模型，可用于翻译任务
@add_start_docstrings(
    "The MARIAN Model with a language modeling head. Can be used for translation.", MARIAN_START_DOCSTRING
)
# 定义FlaxMarianMTModel类，继承自FlaxMarianPreTrainedModel类
class FlaxMarianMTModel(FlaxMarianPreTrainedModel):
    # 指定模块类为FlaxMarianMTModule
    module_class = FlaxMarianMTModule
    # 数据类型为32位浮点数
    dtype: jnp.dtype = jnp.float32

    # 添加解码函数的文档字符串，引用MARIAN_DECODE_INPUTS_DOCSTRING
    @add_start_docstrings(MARIAN_DECODE_INPUTS_DOCSTRING)
    # 替换返回值文档字符串，指定输出类型为FlaxCausalLMOutputWithCrossAttentions，配置类为MarianConfig
    @replace_return_docstrings(output_type=FlaxCausalLMOutputWithCrossAttentions, config_class=MarianConfig)
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
    ):
        # 此函数用于调整logits，确保不生成填充标记
        def _adapt_logits_for_beam_search(self, logits):
            """This function enforces the padding token never to be generated."""
            logits = logits.at[:, :, self.config.pad_token_id].set(float("-inf"))
            return logits

        # 准备生成所需的输入
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

            # 使用init_cache方法初始化past_key_values
            past_key_values = self.init_cache(batch_size, max_length, encoder_outputs)

            # 通常情况下，需要在attention_mask中为x > input_ids.shape[-1]和x < cache_length的位置放置0，
            # 但由于解码器使用因果掩码，这些位置已经被掩码了。
            # 因此，可以在此处创建一个静态的attention_mask，这对编译更加高效。
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

        # 更新生成过程的输入
        def update_inputs_for_generation(self, model_outputs, model_kwargs):
            model_kwargs["past_key_values"] = model_outputs.past_key_values
            model_kwargs["decoder_position_ids"] = model_kwargs["decoder_position_ids"][:, -1:] + 1
            return model_kwargs
# 定义一个多行字符串常量，包含函数 `FlaxMarianMTModel` 的文档字符串。
FLAX_MARIAN_MT_DOCSTRING = """
    Returns:

    Example:

    ```
    >>> from transformers import AutoTokenizer, FlaxMarianMTModel

    >>> model = FlaxMarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    >>> tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")

    >>> text = "My friends are cool but they eat too many carbs."
    >>> input_ids = tokenizer(text, max_length=64, return_tensors="jax").input_ids

    >>> sequences = model.generate(input_ids, max_length=64, num_beams=2).sequences

    >>> outputs = tokenizer.batch_decode(sequences, skip_special_tokens=True)
    >>> # should give *Meine Freunde sind cool, aber sie essen zu viele Kohlenhydrate.*
    ```
"""

# 调用 `overwrite_call_docstring` 函数，将 `FlaxMarianMTModel` 的文档字符串修改为
# 既有的 `MARIAN_INPUTS_DOCSTRING` 和 `FLAX_MARIAN_MT_DOCSTRING` 的组合。
overwrite_call_docstring(
    FlaxMarianMTModel,
    MARIAN_INPUTS_DOCSTRING + FLAX_MARIAN_MT_DOCSTRING,
)

# 调用 `append_replace_return_docstrings` 函数，修改 `FlaxMarianMTModel` 类的返回文档字符串，
# 设置输出类型为 `FlaxSeq2SeqLMOutput`，配置类为 `_CONFIG_FOR_DOC`。
append_replace_return_docstrings(FlaxMarianMTModel, output_type=FlaxSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
```