# `.\models\mbart\modeling_flax_mbart.py`

```
# coding=utf-8
# Copyright 2021, The Facebook AI Research Team and The HuggingFace Inc. team. All rights reserved.
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
""" Flax MBart model."""

# 导入必要的库和模块
import math
import random
from functools import partial
from typing import Callable, Optional, Tuple

import flax.linen as nn  # 导入 Flax 的 linen 模块作为 nn 别名
import jax  # 导入 JAX 库
import jax.numpy as jnp  # 导入 JAX 的 NumPy 实现作为 jnp 别名
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze  # 导入 Flax 的 FrozenDict 和相关函数
from flax.linen import combine_masks, make_causal_mask  # 导入 Flax 的 combine_masks 和 make_causal_mask 函数
from flax.linen.attention import dot_product_attention_weights  # 导入 Flax 的 dot_product_attention_weights 函数
from flax.traverse_util import flatten_dict, unflatten_dict  # 导入 Flax 的 flatten_dict 和 unflatten_dict 函数
from jax import lax  # 导入 JAX 的 lax 模块
from jax.random import PRNGKey  # 导入 JAX 的 PRNGKey 类

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

# 导入模型工具函数和类
from ...modeling_flax_utils import (
    ACT2FN,
    FlaxPreTrainedModel,
    append_call_sample_docstring,
    append_replace_return_docstrings,
    overwrite_call_docstring,
)

# 导入工具函数和类
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings

# 导入 MBart 的配置类
from .configuration_mbart import MBartConfig

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 用于文档的预训练模型检查点和配置信息
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
    Parameters:
        config ([`MBartConfig`]): Model configuration class with all the parameters of the model.
            初始化模型配置类，包含模型的所有参数。
            使用配置文件初始化不会加载模型的权重，只加载配置。可以查看 [`~FlaxPreTrainedModel.from_pretrained`] 方法加载模型权重。

        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
            计算的数据类型。可以是 `jax.numpy.float32`、`jax.numpy.float16`（在GPU上）、`jax.numpy.bfloat16`（在TPU上）之一。

            可用于在GPU或TPU上启用混合精度训练或半精度推断。如果指定了dtype，则所有计算将使用给定的 `dtype`。

            **注意，这仅指定计算的dtype，并不影响模型参数的dtype。**

            如果要更改模型参数的dtype，请参阅 [`~FlaxPreTrainedModel.to_fp16`] 和 [`~FlaxPreTrainedModel.to_bf16`]。
"""
MBART_INPUTS_DOCSTRING = r"""
"""


MBART_ENCODE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
            输入序列标记在词汇表中的索引。默认情况下，将忽略填充部分。

            可以使用 [`AutoTokenizer`] 获取这些索引。详见 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`]。

            [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            避免在填充标记索引上执行注意力操作的掩码。掩码值选择在 `[0, 1]`：

            - 1 表示 **未被掩码** 的标记，
            - 0 表示 **被掩码** 的标记。

            [什么是注意力掩码？](../glossary#attention-mask)
        position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            每个输入序列标记在位置嵌入中的位置索引。选择范围是 `[0, config.max_position_embeddings - 1]`。
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。查看返回的张量下的 `attentions` 获取更多细节。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。查看返回的张量下的 `hidden_states` 获取更多细节。
        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`] 而不是普通的元组。
"""

MBART_DECODE_INPUTS_DOCSTRING = r"""
"""


def shift_tokens_right(input_ids: jnp.ndarray, pad_token_id: int) -> jnp.ndarray:
    """
    将输入 ID 向右移动一个标记，并包装最后一个非填充标记（<LID> 标记）。注意，与其他类似 Bart 模型不同，MBart 没有单一的 `decoder_start_token_id`。
    """
    prev_output_tokens = jnp.array(input_ids).copy()

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")

    # 用 `pad_token_id` 替换标签中可能的 -100 值
    prev_output_tokens = jnp.where(prev_output_tokens == -100, pad_token_id, input_ids)
    index_of_eos = (jnp.where(prev_output_tokens != pad_token_id, 1, 0).sum(axis=-1) - 1).reshape(-1, 1)
    decoder_start_tokens = jnp.array(
        [prev_output_tokens[i, eos_idx] for i, eos_idx in enumerate(index_of_eos)], dtype=jnp.int32
    ).squeeze()

    prev_output_tokens = prev_output_tokens.at[:, 1:].set(prev_output_tokens[:, :-1])
    prev_output_tokens = prev_output_tokens.at[:, 0].set(decoder_start_tokens)

    return prev_output_tokens


# 从 transformers.models.bart.modeling_flax_bart.FlaxBartAttention 复制，将 Bart 改为 MBart
class FlaxMBartAttention(nn.Module):
    config: MBartConfig
    embed_dim: int
    num_heads: int
    # 定义 dropout 参数，默认为 0.0
    dropout: float = 0.0
    # 定义 causal 参数，默认为 False，表示是否使用因果注意力
    causal: bool = False
    # 定义 bias 参数，默认为 True，表示是否使用偏置
    bias: bool = True
    # 定义 dtype 参数，默认为 jnp.float32，表示计算中使用的数据类型

    def setup(self) -> None:
        # 计算每个注意力头的维度
        self.head_dim = self.embed_dim // self.num_heads
        # 检查 embed_dim 必须能被 num_heads 整除，否则抛出 ValueError 异常
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )

        # 部分应用 nn.Dense 函数，创建对应的全连接层，并使用指定的参数
        dense = partial(
            nn.Dense,
            self.embed_dim,
            use_bias=self.bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )

        # 创建 q_proj, k_proj, v_proj 和 out_proj 四个全连接层
        self.q_proj, self.k_proj, self.v_proj = dense(), dense(), dense()
        self.out_proj = dense()

        # 创建 dropout 层，用于随机丢弃输入元素以防止过拟合
        self.dropout_layer = nn.Dropout(rate=self.dropout)

        # 如果设置了 causal=True，创建因果注意力掩码，限制注意力只能关注之前的位置
        if self.causal:
            self.causal_mask = make_causal_mask(
                jnp.ones((1, self.config.max_position_embeddings), dtype="bool"), dtype="bool"
            )

    # 将隐藏状态按注意力头拆分
    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    # 将拆分后的注意力头合并回原始维度
    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    # 使用 nn.compact 装饰器，标志着这是一个 JAX 用来定义层的函数
    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slighly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # 检测是否通过变量"cache"中的"cached_key"来初始化缓存数据。
        is_initialized = self.has_variable("cache", "cached_key")
        # 获取或初始化缓存的键和值，若不存在则创建并初始化为零矩阵，形状和数据类型与输入的key和value相同。
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        # 获取或初始化缓存的索引，若不存在则创建并初始化为0。
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            # 获取缓存的维度信息，包括批处理维度、最大长度、注意力头数和每头深度。
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # 使用新的1D空间切片更新键和值缓存。
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            # 更新缓存中的键和值。
            cached_key.value = key
            cached_value.value = value
            # 更新缓存索引，增加已更新的缓存向量数。
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # 生成用于缓存的因果掩码：单个查询位置仅应关注已生成并缓存的键位置，而不是剩余的零元素。
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            # 结合输入的注意力掩码和生成的填充掩码。
            attention_mask = combine_masks(pad_mask, attention_mask)
        # 返回更新后的键、值和注意力掩码。
        return key, value, attention_mask
# MBart 编码器层定义，继承自 nn.Module 类
class FlaxMBartEncoderLayer(nn.Module):
    # MBart 配置参数
    config: MBartConfig
    # 计算时的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 初始化方法
    def setup(self) -> None:
        # 设置嵌入维度为配置中的模型维度
        self.embed_dim = self.config.d_model
        # 创建 MBart 自注意力层对象
        self.self_attn = FlaxMBartAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.encoder_attention_heads,
            dropout=self.config.attention_dropout,
            dtype=self.dtype,
        )
        # 自注意力层后的 LayerNorm 层
        self.self_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        # 随机失活层
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)
        # 激活函数
        self.activation_fn = ACT2FN[self.config.activation_function]
        # 激活函数后的随机失活层
        self.activation_dropout_layer = nn.Dropout(rate=self.config.activation_dropout)
        # 第一个全连接层
        self.fc1 = nn.Dense(
            self.config.encoder_ffn_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        # 第二个全连接层，最终输出维度与嵌入维度相同
        self.fc2 = nn.Dense(
            self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std)
        )
        # 最终的 LayerNorm 层
        self.final_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    # 对象调用方法，执行编码器层的前向传播
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        output_attentions: bool = True,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray]:
        # 保存残差连接
        residual = hidden_states
        # 应用自注意力层的 LayerNorm
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # 执行自注意力计算，返回计算结果和注意力权重
        hidden_states, attn_weights = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask)
        # 应用随机失活层
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 添加残差连接
        hidden_states = residual + hidden_states

        # 保存残差连接
        residual = hidden_states
        # 应用最终的 LayerNorm
        hidden_states = self.final_layer_norm(hidden_states)
        # 应用激活函数和第一个全连接层
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 应用激活函数后的随机失活层
        hidden_states = self.activation_dropout_layer(hidden_states, deterministic=deterministic)
        # 应用第二个全连接层
        hidden_states = self.fc2(hidden_states)
        # 应用最终的随机失活层
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 添加残差连接
        hidden_states = residual + hidden_states

        # 构建输出元组，包含最终的隐藏状态
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，添加到输出元组中
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# 从 transformers.models.bart.modeling_flax_bart.FlaxBartEncoderLayerCollection 复制并修改为 MBart
class FlaxMBartEncoderLayerCollection(nn.Module):
    # MBart 配置参数
    config: MBartConfig
    # 计算时的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型

    # 初始化方法
    def setup(self):
        # 创建多层 MBart 编码器层列表
        self.layers = [
            FlaxMBartEncoderLayer(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.encoder_layers)
        ]
        # 编码器层的层级丢弃率
        self.layerdrop = self.config.encoder_layerdrop
    # 定义一个调用方法，用于执行模型的前向传播
    def __call__(
        self,
        hidden_states,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 初始化存储所有注意力权重的变量，如果不需要输出注意力权重则置为 None
        all_attentions = () if output_attentions else None
        # 初始化存储所有隐藏状态的变量，如果不需要输出隐藏状态则置为 None
        all_hidden_states = () if output_hidden_states else None

        # 遍历每个编码器层
        for encoder_layer in self.layers:
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            # 添加层级丢弃功能，参见论文 https://arxiv.org/abs/1909.11556
            dropout_probability = random.uniform(0, 1)
            # 如果不是确定性执行且随机数小于层级丢弃率，则跳过当前层
            if not deterministic and (dropout_probability < self.layerdrop):
                layer_outputs = (None, None)  # 表示跳过当前层，输出为空
            else:
                # 否则，执行当前编码器层的前向传播，获取输出
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

        # 如果需要输出隐藏状态，则将最终的隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 汇总模型输出，包括最终隐藏状态、所有隐藏状态和所有注意力权重
        outputs = (hidden_states, all_hidden_states, all_attentions)

        # 如果不需要以字典形式返回结果，则返回一个元组，排除 None 的部分
        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        # 否则，以 FlaxBaseModelOutput 对象形式返回结果，包括最终隐藏状态、所有隐藏状态和所有注意力权重
        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )
class FlaxMBartDecoderLayer(nn.Module):
    # 定义类变量 config，类型为 MBartConfig，用于存储配置信息
    config: MBartConfig
    # 定义类变量 dtype，默认为 jnp.float32，指定数据类型为 32 位浮点数

    def setup(self) -> None:
        # 初始化函数，设置层的参数和模型结构

        # 将 self.embed_dim 设置为 self.config.d_model，表示嵌入维度等于模型维度
        self.embed_dim = self.config.d_model

        # 初始化 self.self_attn 为 FlaxMBartAttention 类实例，用于自注意力机制
        self.self_attn = FlaxMBartAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.decoder_attention_heads,
            dropout=self.config.attention_dropout,
            causal=True,
            dtype=self.dtype,
        )

        # 初始化 self.dropout_layer 为 Dropout 层，用于随机失活
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)

        # 根据配置选择激活函数，并初始化 self.activation_fn
        self.activation_fn = ACT2FN[self.config.activation_function]

        # 初始化 self.activation_dropout_layer 为 Dropout 层，用于激活函数的随机失活
        self.activation_dropout_layer = nn.Dropout(rate=self.config.activation_dropout)

        # 初始化 self.self_attn_layer_norm 为 LayerNorm 层，用于自注意力机制后的归一化
        self.self_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

        # 初始化 self.encoder_attn 为 FlaxMBartAttention 类实例，用于编码器-解码器注意力机制
        self.encoder_attn = FlaxMBartAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.decoder_attention_heads,
            dropout=self.config.attention_dropout,
            dtype=self.dtype,
        )

        # 初始化 self.encoder_attn_layer_norm 为 LayerNorm 层，用于编码器-解码器注意力机制后的归一化
        self.encoder_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

        # 初始化 self.fc1 为 Dense（全连接）层，用于第一个前馈神经网络（FFN）层
        self.fc1 = nn.Dense(
            self.config.decoder_ffn_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )

        # 初始化 self.fc2 为 Dense 层，用于第二个前馈神经网络（FFN）层
        self.fc2 = nn.Dense(
            self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std)
        )

        # 初始化 self.final_layer_norm 为 LayerNorm 层，用于最终输出的归一化
        self.final_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        output_attentions: bool = True,
        deterministic: bool = True,
        # 定义 __call__ 方法，用于模型调用时的前向传播
    ) -> Tuple[jnp.ndarray]:  
        # 将输入的隐藏状态作为残差保存
        residual = hidden_states  
        # 对当前的隐藏状态进行自注意力层归一化处理
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 自注意力机制
        # 调用自注意力层处理隐藏状态，返回处理后的新隐藏状态和自注意力权重
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, init_cache=init_cache
        )
        # 对自注意力层输出的隐藏状态进行 dropout 处理
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 将残差与处理后的隐藏状态相加，形成新的隐藏状态
        hidden_states = residual + hidden_states

        # 跨注意力块
        cross_attn_weights = None
        # 如果存在编码器的隐藏状态
        if encoder_hidden_states is not None:
            # 将当前的隐藏状态作为残差保存
            residual = hidden_states

            # 对当前的隐藏状态进行编码器注意力层归一化处理
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            # 调用编码器注意力层处理隐藏状态，返回处理后的新隐藏状态和编码器注意力权重
            hidden_states, cross_attn_weights = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            # 对编码器注意力层输出的隐藏状态进行 dropout 处理
            hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
            # 将残差与处理后的隐藏状态相加，形成新的隐藏状态
            hidden_states = residual + hidden_states

        # 全连接层处理
        # 将当前的隐藏状态作为残差保存
        residual = hidden_states
        # 对当前的隐藏状态进行最终归一化处理
        hidden_states = self.final_layer_norm(hidden_states)
        # 应用激活函数到全连接层的第一层
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 对全连接层的第一层输出进行 dropout 处理
        hidden_states = self.activation_dropout_layer(hidden_states, deterministic=deterministic)
        # 继续传播到全连接层的第二层
        hidden_states = self.fc2(hidden_states)
        # 对全连接层的第二层输出进行 dropout 处理
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 将残差与处理后的隐藏状态相加，形成新的隐藏状态
        hidden_states = residual + hidden_states

        # 准备输出
        outputs = (hidden_states,)

        # 如果需要输出注意力权重信息，则将自注意力权重和编码器注意力权重添加到输出中
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        # 返回输出结果
        return outputs
# 从 transformers.models.bart.modeling_flax_bart.FlaxBartDecoderLayerCollection 复制代码并将 Bart 改为 MBart
class FlaxMBartDecoderLayerCollection(nn.Module):
    config: MBartConfig  # 类型注解，指定 MBartConfig 类型的 config 变量
    dtype: jnp.dtype = jnp.float32  # 计算过程中使用的数据类型，默认为 jnp.float32

    def setup(self):
        # 创建 self.layers 列表，包含 self.config.decoder_layers 个 FlaxMBartDecoderLayer 对象
        self.layers = [
            FlaxMBartDecoderLayer(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.decoder_layers)
        ]
        self.layerdrop = self.config.decoder_layerdrop  # 设置 layerdrop 参数为 config 中的 decoder_layerdrop 值

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
        all_hidden_states = () if output_hidden_states else None  # 如果不输出 hidden_states，则设置为 None
        all_self_attns = () if output_attentions else None  # 如果不输出 self-attention，则设置为 None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None  # 如果不输出 cross-attention 或者没有 encoder_hidden_states，则设置为 None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)  # 将当前 hidden_states 添加到 all_hidden_states 中
                # 添加 LayerDrop（参见 https://arxiv.org/abs/1909.11556 进行描述）

            dropout_probability = random.uniform(0, 1)  # 随机生成一个 dropout 概率
            if not deterministic and (dropout_probability < self.layerdrop):
                layer_outputs = (None, None, None)  # 如果未指定 deterministic 或者 dropout 概率小于 layerdrop，则输出为 None
            else:
                # 调用 decoder_layer 进行解码层计算
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    init_cache=init_cache,
                    output_attentions=output_attentions,
                    deterministic=deterministic,
                )

            hidden_states = layer_outputs[0]  # 更新 hidden_states 为当前层的输出第一个元素
            if output_attentions:
                all_self_attns += (layer_outputs[1],)  # 将当前层的 self-attention 输出添加到 all_self_attns 中

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)  # 如果存在 encoder_hidden_states，则将 cross-attention 输出添加到 all_cross_attentions 中

        # 添加来自最后解码层的 hidden states
        if output_hidden_states:
            all_hidden_states += (hidden_states,)  # 将最后一个 hidden_states 添加到 all_hidden_states 中

        # 构建输出列表
        outputs = [hidden_states, all_hidden_states, all_self_attns, all_cross_attentions]

        if not return_dict:
            return tuple(v for v in outputs if v is not None)  # 如果不返回字典，则返回非 None 的元组

        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )
    """Head for sentence-level classification tasks."""
    
    # 定义一个类用于处理句子级别的分类任务，以下是其成员变量和方法的定义和说明

    config: MBartConfig
    # 用于存储配置信息的变量，类型为 MBartConfig 类型的对象

    inner_dim: int
    # 用于存储中间维度大小的整数变量，表示神经网络中间层的维度

    num_classes: int
    # 用于存储分类类别数量的整数变量，表示分类任务的输出类别数目

    pooler_dropout: float
    # 用于存储池化层的dropout率的浮点数变量，控制神经网络在训练中的丢弃比例

    dtype: jnp.dtype = jnp.float32
    # 数据类型，默认为 jax 的浮点数类型 jnp.float32

    def setup(self):
        # 初始化方法，用于设置类的各个成员变量

        self.dense = nn.Dense(
            self.inner_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std)
        )
        # 创建一个全连接层对象 self.dense，设置输入维度为 inner_dim，数据类型为 dtype，并使用正态分布初始化权重

        self.dropout = nn.Dropout(rate=self.pooler_dropout)
        # 创建一个 dropout 层对象 self.dropout，设置丢弃率为 pooler_dropout

        self.out_proj = nn.Dense(
            self.num_classes,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        # 创建一个输出投影层对象 self.out_proj，设置输出维度为 num_classes，数据类型为 dtype，并使用正态分布初始化权重

    def __call__(self, hidden_states: jnp.ndarray, deterministic: bool):
        # 类的调用方法，用于实现类的前向传播过程

        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 对输入的 hidden_states 应用 dropout 操作，根据 deterministic 参数决定是否使用确定性丢弃

        hidden_states = self.dense(hidden_states)
        # 将经过 dropout 后的 hidden_states 输入到全连接层 self.dense 中进行线性变换

        hidden_states = jnp.tanh(hidden_states)
        # 对全连接层的输出应用双曲正切激活函数

        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 对激活后的 hidden_states 再次应用 dropout 操作

        hidden_states = self.out_proj(hidden_states)
        # 将经过 dropout 的 hidden_states 输入到输出投影层 self.out_proj 中进行线性变换，得到最终的分类结果

        return hidden_states
        # 返回处理后的结果 hidden_states
# 定义一个名为 FlaxMBartEncoder 的类，继承自 nn.Module，用于 MBart 编码器模型
class FlaxMBartEncoder(nn.Module):
    # 类属性：MBart 的配置对象
    config: MBartConfig
    # 类属性：嵌入层对象，用于输入的词嵌入
    embed_tokens: nn.Embed
    # 类属性：计算中使用的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32  # 计算时使用的数据类型

    # 定义初始化方法 setup()
    def setup(self):
        # 初始化 dropout 层，根据配置中的 dropout 率
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)

        # 获取嵌入维度大小
        embed_dim = self.config.d_model
        # 获取填充符号的索引
        self.padding_idx = self.config.pad_token_id
        # 获取源序列的最大位置编码长度
        self.max_source_positions = self.config.max_position_embeddings
        # 根据配置设置嵌入的缩放因子
        self.embed_scale = math.sqrt(embed_dim) if self.config.scale_embedding else 1.0

        # MBart 的特殊设置：如果指定了 padding_idx，则将嵌入的 id 偏移 2，并相应地调整 num_embeddings。其他模型不需要这个处理
        self.offset = 2
        # 初始化位置嵌入层
        self.embed_positions = nn.Embed(
            self.config.max_position_embeddings + self.offset,  # 位置嵌入层的大小，考虑了偏移量
            embed_dim,  # 嵌入的维度大小
            embedding_init=jax.nn.initializers.normal(self.config.init_std),  # 初始化方法，使用正态分布
        )
        # 初始化编码器层集合
        self.layers = FlaxMBartEncoderLayerCollection(self.config, self.dtype)
        # 初始化嵌入层的 LayerNorm
        self.layernorm_embedding = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        # 初始化输出层的 LayerNorm
        self.layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    # 定义调用方法 __call__()，用于执行编码器的前向传播
    def __call__(
        self,
        input_ids,  # 输入的 token ids
        attention_mask,  # 注意力掩码，用于指示哪些位置是填充的
        position_ids,  # 位置 ids
        output_attentions: bool = False,  # 是否输出注意力权重
        output_hidden_states: bool = False,  # 是否输出隐藏状态
        return_dict: bool = True,  # 是否以字典形式返回结果
        deterministic: bool = True,  # 是否使用确定性计算
    ):
        # 获取输入的形状信息
        input_shape = input_ids.shape
        # 将输入 ids 展平为二维张量
        input_ids = input_ids.reshape(-1, input_shape[-1])

        # 使用嵌入 tokens 和缩放因子来嵌入输入的 ids
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # 根据位置 ids 和偏移量获取位置嵌入
        embed_pos = self.embed_positions(position_ids + self.offset)

        # 将输入嵌入和位置嵌入相加得到隐藏状态
        hidden_states = inputs_embeds + embed_pos
        # 对输入嵌入的 LayerNorm 处理
        hidden_states = self.layernorm_embedding(hidden_states)
        # 应用 dropout
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)

        # 调用编码器层集合的前向传播
        outputs = self.layers(
            hidden_states,
            attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取编码器层集合的最后隐藏状态
        last_hidden_states = outputs[0]
        # 对最后隐藏状态进行 LayerNorm 处理
        last_hidden_states = self.layer_norm(last_hidden_states)

        # 如果需要输出隐藏状态，则更新 `hidden_states` 中的最后一个元素，应用上面的 `layernorm`
        hidden_states = None
        if output_hidden_states:
            hidden_states = outputs[1]
            hidden_states = hidden_states[:-1] + (last_hidden_states,)

        # 如果不以字典形式返回结果，则将结果组合成元组返回
        if not return_dict:
            outputs = (last_hidden_states, hidden_states) + (outputs[2:] if output_hidden_states else outputs[1:])
            return tuple(v for v in outputs if v is not None)

        # 以 FlaxBaseModelOutput 对象的形式返回结果
        return FlaxBaseModelOutput(
            last_hidden_state=last_hidden_states,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
        )


# 定义一个名为 FlaxMBartDecoder 的类，目前还未完整给出，继承自 nn.Module
class FlaxMBartDecoder(nn.Module):
    # 类属性：MBart 的配置对象
    config: MBartConfig
    # 类属性：嵌入层对象，用于输入的词嵌入
    embed_tokens: nn.Embed
    # 设置默认数据类型为 jnp.float32，用于计算过程中的数据类型
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    # 初始化函数，在对象创建时调用，用于设置各种属性和参数
    def setup(self):
        # 初始化一个丢弃层，根据配置中的丢弃率设置
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)

        # 获取嵌入维度，从配置中获取填充索引、最大目标位置和嵌入缩放因子
        embed_dim = self.config.d_model
        self.padding_idx = self.config.pad_token_id
        self.max_target_positions = self.config.max_position_embeddings
        self.embed_scale = math.sqrt(self.config.d_model) if self.config.scale_embedding else 1.0

        # 如果是 MBart 模型，根据填充索引偏移量设置嵌入位置
        # 其他模型不需要此偏移量的调整
        self.offset = 2
        self.embed_positions = nn.Embed(
            self.config.max_position_embeddings + self.offset,  # 设置嵌入的最大位置数量
            embed_dim,  # 嵌入的维度
            embedding_init=jax.nn.initializers.normal(self.config.init_std),  # 使用正态分布初始化嵌入
        )

        # 初始化多层解码器层集合
        self.layers = FlaxMBartDecoderLayerCollection(self.config, self.dtype)
        # 初始化嵌入层的 LayerNorm
        self.layernorm_embedding = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        # 初始化通用的 LayerNorm
        self.layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    # 对象调用函数，实现模型的前向传播
    def __call__(
        self,
        input_ids,  # 输入的 token id
        attention_mask,  # 注意力掩码
        position_ids,  # 位置 id
        encoder_hidden_states: Optional[jnp.ndarray] = None,  # 编码器的隐藏状态（可选）
        encoder_attention_mask: Optional[jnp.ndarray] = None,  # 编码器的注意力掩码（可选）
        init_cache: bool = False,  # 是否初始化缓存（默认为 False）
        output_attentions: bool = False,  # 是否输出注意力权重（默认为 False）
        output_hidden_states: bool = False,  # 是否输出隐藏状态（默认为 False）
        return_dict: bool = True,  # 是否以字典形式返回结果（默认为 True）
        deterministic: bool = True,  # 是否确定性计算（默认为 True）
        ):
            # 获取输入张量的形状
            input_shape = input_ids.shape
            # 将输入张量重塑为二维张量
            input_ids = input_ids.reshape(-1, input_shape[-1])

            # 使用模型的嵌入层对输入张量进行嵌入，并乘以嵌入缩放因子
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

            # 嵌入位置信息
            positions = self.embed_positions(position_ids + self.offset)

            # 将输入嵌入和位置嵌入相加
            hidden_states = inputs_embeds + positions
            # 应用嵌入层归一化
            hidden_states = self.layernorm_embedding(hidden_states)

            # 对隐藏状态应用 dropout
            hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)

            # 将隐藏状态传递给层堆栈进行处理
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

            # 获取输出中的最后隐藏状态
            last_hidden_states = outputs[0]
            # 对最后隐藏状态应用层归一化
            last_hidden_states = self.layer_norm(last_hidden_states)

            # 如果需要输出隐藏状态，更新 `hidden_states` 中的最后一个元素
            hidden_states = None
            if output_hidden_states:
                hidden_states = outputs[1]
                hidden_states = hidden_states[:-1] + (last_hidden_states,)

            # 如果不返回字典形式的结果，构建输出元组
            if not return_dict:
                outputs = (last_hidden_states, hidden_states) + (outputs[2:] if output_hidden_states else outputs[1:])
                return tuple(v for v in outputs if v is not None)

            # 返回带有过去和交叉注意力的 FlaxBaseModelOutputWithPastAndCrossAttentions 对象
            return FlaxBaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=last_hidden_states,
                hidden_states=hidden_states,
                attentions=outputs.attentions,
                cross_attentions=outputs.cross_attentions,
            )
# Copied from transformers.models.bart.modeling_flax_bart.FlaxBartModule with Bart->MBart
# 定义了 FlaxMBartModule 类，继承自 nn.Module
class FlaxMBartModule(nn.Module):
    # 类属性 config，指定为 MBartConfig 类型
    config: MBartConfig
    # 类属性 dtype，指定为 jnp.float32，用于计算的数据类型

    # 初始化方法 setup，用于设置模块内部的各个组件
    def setup(self):
        # 创建一个共享的嵌入层 nn.Embed 对象
        self.shared = nn.Embed(
            self.config.vocab_size,  # 嵌入层的词汇表大小
            self.config.d_model,      # 嵌入的维度大小
            embedding_init=jax.nn.initializers.normal(self.config.init_std),  # 嵌入层参数初始化方法
            dtype=self.dtype,         # 嵌入层的数据类型
        )

        # 创建 MBartEncoder 对象，用于编码输入数据
        self.encoder = FlaxMBartEncoder(self.config, dtype=self.dtype, embed_tokens=self.shared)
        # 创建 MBartDecoder 对象，用于解码器的解码过程
        self.decoder = FlaxMBartDecoder(self.config, dtype=self.dtype, embed_tokens=self.shared)

    # 获取编码器模块的方法
    def _get_encoder_module(self):
        return self.encoder

    # 获取解码器模块的方法
    def _get_decoder_module(self):
        return self.decoder

    # 实现 __call__ 方法，定义了模型的调用过程
    def __call__(
        self,
        input_ids,                 # 输入的编码器输入 ID
        attention_mask,            # 编码器的注意力掩码
        decoder_input_ids,         # 解码器的输入 ID
        decoder_attention_mask,    # 解码器的注意力掩码
        position_ids,              # 位置 ID
        decoder_position_ids,      # 解码器的位置 ID
        output_attentions: bool = False,         # 是否输出注意力权重
        output_hidden_states: bool = False,      # 是否输出隐藏状态
        return_dict: bool = True,                # 是否以字典形式返回结果
        deterministic: bool = True,              # 是否确定性计算结果
    ):
        # 编码器的前向传播过程，返回编码器的输出结果
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        # 解码器的前向传播过程，返回解码器的输出结果
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            position_ids=decoder_position_ids,
            encoder_hidden_states=encoder_outputs[0],  # 使用编码器的隐藏状态作为输入
            encoder_attention_mask=attention_mask,     # 使用编码器的注意力掩码
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        # 如果 return_dict 为 False，则将解码器和编码器的输出结果连接起来返回
        if not return_dict:
            return decoder_outputs + encoder_outputs

        # 如果 return_dict 为 True，则返回 FlaxSeq2SeqModelOutput 对象，包含了完整的模型输出信息
        return FlaxSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


# 定义 FlaxMBartPreTrainedModel 类，继承自 FlaxPreTrainedModel
class FlaxMBartPreTrainedModel(FlaxPreTrainedModel):
    # 类属性 config_class，指定为 MBartConfig 类
    config_class = MBartConfig
    # 类属性 base_model_prefix，指定为字符串 "model"
    base_model_prefix: str = "model"
    # 类属性 module_class，默认为 None，用于指定模型的主模块

    # 初始化方法，用于创建 FlaxMBartPreTrainedModel 对象
    def __init__(
        self,
        config: MBartConfig,                  # MBart 模型的配置
        input_shape: Tuple[int] = (1, 1),     # 输入形状，默认为 (1, 1)
        seed: int = 0,                        # 随机种子，默认为 0
        dtype: jnp.dtype = jnp.float32,       # 计算数据类型，默认为 jnp.float32
        _do_init: bool = True,                # 是否初始化，默认为 True
        **kwargs,                             # 其他关键字参数
    ):
        # 调用父类的初始化方法，初始化模型的基本配置
        super().__init__(config=config, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init, **kwargs)
    ):
        # 使用给定的配置和参数实例化模块对象
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 调用父类初始化方法，传递配置、模块对象、输入形状、种子、数据类型以及是否初始化的标志
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    # 初始化模型权重
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量，全零张量，数据类型为'i4'
        input_ids = jnp.zeros(input_shape, dtype="i4")
        # 确保初始化可以为FlaxMBartForSequenceClassificationModule正常工作
        input_ids = input_ids.at[(..., -1)].set(self.config.eos_token_id)
        # 创建注意力遮罩，与input_ids形状相同，全为1
        attention_mask = jnp.ones_like(input_ids)
        # 初始化decoder输入为input_ids
        decoder_input_ids = input_ids
        # decoder的注意力遮罩与input_ids形状相同，全为1
        decoder_attention_mask = jnp.ones_like(input_ids)

        # 获取批次大小和序列长度
        batch_size, sequence_length = input_ids.shape
        # 生成位置编码，广播形状为(batch_size, sequence_length)
        position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))
        # decoder的位置编码与position_ids相同
        decoder_position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # 划分随机数生成器，用于参数和dropout
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 使用初始化方法初始化模型，返回参数字典
        random_params = self.module.init(
            rngs,
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            position_ids,
            decoder_position_ids,
        )["params"]

        # 如果提供了params，则用随机参数填充缺失的键
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            # 冻结并返回填充后的参数字典
            return freeze(unflatten_dict(params))
        else:
            # 否则，返回随机生成的参数字典
            return random_params

    # 从transformers.models.bart.modeling_flax_bart.FlaxBartPreTrainedModel.init_cache复制，替换Bart为MBart
    # 初始化缓存以支持快速自回归解码。
    def init_cache(self, batch_size, max_length, encoder_outputs):
        r"""
        Args:
            batch_size (`int`):
                用于快速自回归解码的批大小。定义初始化缓存时的批大小。
            max_length (`int`):
                自回归解码的最大可能长度。定义初始化缓存时的序列长度。
            encoder_outputs (`Union[FlaxBaseModelOutput, tuple(tuple(jnp.ndarray)]`):
                `encoder_outputs` 包括 (`last_hidden_state`, *可选*: `hidden_states`, *可选*: `attentions`)。
                `last_hidden_state` 的形状为 `(batch_size, sequence_length, hidden_size)`，*可选* 是编码器最后一层的隐藏状态的序列，
                用于解码器的交叉注意力。
        """
        # 初始化用于检索缓存的输入变量
        decoder_input_ids = jnp.ones((batch_size, max_length), dtype="i4")
        decoder_attention_mask = jnp.ones_like(decoder_input_ids)
        decoder_position_ids = jnp.broadcast_to(
            jnp.arange(jnp.atleast_2d(decoder_input_ids).shape[-1]), decoder_input_ids.shape
        )

        def _decoder_forward(module, decoder_input_ids, decoder_attention_mask, decoder_position_ids, **kwargs):
            # 获取解码器模块
            decoder_module = module._get_decoder_module()
            return decoder_module(
                decoder_input_ids,
                decoder_attention_mask,
                decoder_position_ids,
                **kwargs,
            )

        # 使用给定的输入初始化模型变量，其中 `method` 指定了仅需调用解码器来初始化缓存
        init_variables = self.module.init(
            jax.random.PRNGKey(0),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            encoder_hidden_states=encoder_outputs[0],
            init_cache=True,
            method=_decoder_forward,
        )
        # 返回冻结的缓存部分
        return unfreeze(init_variables["cache"])

    @add_start_docstrings(MBART_ENCODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutput, config_class=MBartConfig)
    # 编码方法，根据输入的参数编码输入序列
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
        ```
        """
        # Determine whether to output attentions based on input or default configuration
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # Determine whether to output hidden states based on input or default configuration
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # Determine whether to return a dictionary of outputs based on input or default configuration
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # If attention mask is not provided, create a mask where all elements are 1
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        # If position IDs are not provided, create a broadcasted array from 0 to sequence length
        if position_ids is None:
            batch_size, sequence_length = input_ids.shape
            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # Handle any pseudo-random number generators needed for dropout
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        # Define a nested function to forward input through the encoder module
        def _encoder_forward(module, input_ids, attention_mask, position_ids, **kwargs):
            encode_module = module._get_encoder_module()
            return encode_module(input_ids, attention_mask, position_ids, **kwargs)

        # Apply the Flax module with specified parameters and inputs
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

    @add_start_docstrings(MBART_DECODE_INPUTS_DOCSTRING)
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
    ):
        r"""
        """
        # Function definition continues in the next segment
    # 定义一个特殊方法，使得对象可以像函数一样被调用
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
        # 如果未指定输出注意力机制，则使用配置中的默认设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未指定输出隐藏状态，则使用配置中的默认设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定返回字典形式，则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 准备编码器的输入
        if attention_mask is None:
            # 如果未提供注意力遮罩，则创建一个全为1的遮罩，形状与输入的input_ids相同
            attention_mask = jnp.ones_like(input_ids)
        if position_ids is None:
            # 如果未提供位置编码，则根据输入的input_ids的形状自动创建位置编码
            batch_size, sequence_length = input_ids.shape
            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # 准备解码器的输入
        if decoder_input_ids is None:
            # 如果未提供解码器输入的token_ids，则根据编码器的输入右移一位，同时使用配置中的pad_token_id填充
            decoder_input_ids = shift_tokens_right(input_ids, self.config.pad_token_id)
        if decoder_attention_mask is None:
            # 如果未提供解码器的注意力遮罩，则创建一个全为1的遮罩，形状与decoder_input_ids相同
            decoder_attention_mask = jnp.ones_like(decoder_input_ids)
        if decoder_position_ids is None:
            # 如果未提供解码器的位置编码，则根据decoder_input_ids的形状自动创建位置编码
            batch_size, sequence_length = decoder_input_ids.shape
            decoder_position_ids = jnp.broadcast_to(
                jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
            )

        # 处理可能需要的任何伪随机数生成器
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        # 调用self.module的apply方法，传递参数和输入，执行模型计算
        return self.module.apply(
            {"params": params or self.params},  # 参数字典，如果params为None则使用self.params
            input_ids=jnp.array(input_ids, dtype="i4"),  # 将input_ids转换为jnp的整型数组
            attention_mask=jnp.array(attention_mask, dtype="i4"),  # 将attention_mask转换为jnp的整型数组
            position_ids=jnp.array(position_ids, dtype="i4"),  # 将position_ids转换为jnp的整型数组
            decoder_input_ids=jnp.array(decoder_input_ids, dtype="i4"),  # 将decoder_input_ids转换为jnp的整型数组
            decoder_attention_mask=jnp.array(decoder_attention_mask, dtype="i4"),  # 将decoder_attention_mask转换为jnp的整型数组
            decoder_position_ids=jnp.array(decoder_position_ids, dtype="i4"),  # 将decoder_position_ids转换为jnp的整型数组
            output_attentions=output_attentions,  # 输出注意力权重的标志
            output_hidden_states=output_hidden_states,  # 输出隐藏状态的标志
            return_dict=return_dict,  # 返回字典的标志
            deterministic=not train,  # 是否确定性计算，如果为False则进行随机dropout
            rngs=rngs,  # 伪随机数生成器字典
        )
# 为 FlaxMBartModel 类添加文档字符串，描述其作用为在 MBart 模型上输出原始隐藏状态而无需特定的顶部头部。
@add_start_docstrings(
    "The bare MBart Model transformer outputting raw hidden-states without any specific head on top.",
    MBART_START_DOCSTRING,
)
# 声明 FlaxMBartModel 类，继承自 FlaxMBartPreTrainedModel
class FlaxMBartModel(FlaxMBartPreTrainedModel):
    # 使用 MBartConfig 作为配置
    config: MBartConfig
    # 计算中使用的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 模块的类别设定为 FlaxMBartModule
    module_class = FlaxMBartModule

# 调用函数 append_call_sample_docstring，为 FlaxMBartModel 类添加调用示例的文档字符串
append_call_sample_docstring(FlaxMBartModel, _CHECKPOINT_FOR_DOC, FlaxSeq2SeqModelOutput, _CONFIG_FOR_DOC)

# 从 transformers.models.bart.modeling_flax_bart.FlaxBartForConditionalGenerationModule 复制并修改为 FlaxMBartForConditionalGenerationModule 类
class FlaxMBartForConditionalGenerationModule(nn.Module):
    # 使用 MBartConfig 作为配置
    config: MBartConfig
    # 计算中使用的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 偏置初始化函数，使用 jax.nn.initializers.zeros
    bias_init: Callable[..., jnp.ndarray] = jax.nn.initializers.zeros

    # 模块的设置方法
    def setup(self):
        # 创建 FlaxMBartModule 模块，使用给定的配置和数据类型
        self.model = FlaxMBartModule(config=self.config, dtype=self.dtype)
        # 创建 lm_head 密集层，输出维度为 self.model.shared.num_embeddings，不使用偏置，使用指定的初始化器
        self.lm_head = nn.Dense(
            self.model.shared.num_embeddings,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        # 创建 final_logits_bias 参数，维度为 (1, self.model.shared.num_embeddings)，使用偏置初始化器
        self.final_logits_bias = self.param("final_logits_bias", self.bias_init, (1, self.model.shared.num_embeddings))

    # 获取编码器模块的方法
    def _get_encoder_module(self):
        return self.model.encoder

    # 获取解码器模块的方法
    def _get_decoder_module(self):
        return self.model.decoder

    # 类的调用方法，定义了模型的前向传播逻辑和相关参数
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
        # 调用模型进行前向推断，获取模型输出
        outputs = self.model(
            input_ids=input_ids,  # 输入的编码器输入 ID
            attention_mask=attention_mask,  # 编码器的注意力遮罩
            decoder_input_ids=decoder_input_ids,  # 解码器的输入 ID
            decoder_attention_mask=decoder_attention_mask,  # 解码器的注意力遮罩
            position_ids=position_ids,  # 位置 ID，用于编码器
            decoder_position_ids=decoder_position_ids,  # 解码器的位置 ID
            output_attentions=output_attentions,  # 是否输出注意力权重
            output_hidden_states=output_hidden_states,  # 是否输出隐藏状态
            return_dict=return_dict,  # 是否返回字典格式的输出
            deterministic=deterministic,  # 是否确定性运行（不随机性质）
        )

        # 获取模型的隐藏状态作为 LM 的 logits
        hidden_states = outputs[0]

        # 如果配置要求共享词嵌入
        if self.config.tie_word_embeddings:
            # 获取共享的嵌入矩阵
            shared_embedding = self.model.variables["params"]["shared"]["embedding"]
            # 应用共享的嵌入矩阵作为 LM 头的核心参数
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
        else:
            # 使用原始的 LM 头计算 logits
            lm_logits = self.lm_head(hidden_states)

        # 将最终 logits 加上偏置项（停止梯度传播）
        lm_logits += jax.lax.stop_gradient(self.final_logits_bias.astype(self.dtype))

        # 如果不要求返回字典格式的输出
        if not return_dict:
            # 组装输出，包括 logits 和其他模型输出
            output = (lm_logits,) + outputs[1:]
            return output

        # 返回 FlaxSeq2SeqLMOutput 类型的输出
        return FlaxSeq2SeqLMOutput(
            logits=lm_logits,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
@add_start_docstrings(
    "The MMBart Model with a language modeling head. Can be used for summarization.", MBART_START_DOCSTRING
)
class FlaxMBartForConditionalGeneration(FlaxMBartPreTrainedModel):
    module_class = FlaxMBartForConditionalGenerationModule
    dtype: jnp.dtype = jnp.float32

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
    ):
        """
        Performs decoding with the model.

        Args:
            decoder_input_ids: Input IDs for the decoder.
            encoder_outputs: Outputs from the encoder.
            encoder_attention_mask: Optional attention mask for the encoder outputs.
            decoder_attention_mask: Optional attention mask for the decoder inputs.
            decoder_position_ids: Optional position IDs for the decoder inputs.
            past_key_values: Cached key values for efficient generation.
            output_attentions: Whether to output attentions.
            output_hidden_states: Whether to output hidden states.
            return_dict: Whether to return a dictionary or a tuple.
            train: Whether in training mode.
            params: Optional parameters.
            dropout_rng: Dropout random number generator key.

        Returns:
            Model output with cross attentions.

        """
        # Function body omitted for brevity as it is straightforward with provided docstrings.

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
        Prepares inputs for generation.

        Args:
            decoder_input_ids: Input IDs for the decoder.
            max_length: Maximum length for generation.
            attention_mask: Optional attention mask for the encoder outputs.
            decoder_attention_mask: Optional attention mask for the decoder inputs.
            encoder_outputs: Outputs from the encoder.
            **kwargs: Additional keyword arguments.

        Returns:
            Dictionary with prepared inputs for generation.

        """
        # initializing the cache
        batch_size, seq_length = decoder_input_ids.shape

        # Initialize past key values for efficient generation
        past_key_values = self.init_cache(batch_size, max_length, encoder_outputs)

        # Create an extended attention mask
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if decoder_attention_mask is not None:
            # Adjust position IDs based on decoder_attention_mask
            position_ids = decoder_attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, decoder_attention_mask, (0, 0))
        else:
            # Use default position IDs if decoder_attention_mask is not provided
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        return {
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "encoder_attention_mask": attention_mask,
            "decoder_attention_mask": extended_attention_mask,
            "decoder_position_ids": position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        """
        Updates inputs for generation.

        Args:
            model_outputs: Model outputs from the generation.
            model_kwargs: Original model keyword arguments.

        Returns:
            Updated model keyword arguments.

        """
        # Update past_key_values and decoder_position_ids for next generation step
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["decoder_position_ids"] = model_kwargs["decoder_position_ids"][:, -1:] + 1
        return model_kwargs


FLAX_MBART_CONDITIONAL_GENERATION_DOCSTRING = r"""
    Returns:

    Summarization example:

    ```python
    >>> from transformers import AutoTokenizer, FlaxMBartForConditionalGeneration, MBartConfig
    >>> model = FlaxMBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")
    >>> tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")
    
    从预训练的MBart模型和tokenizer中加载Facebook的mbart-large-cc25模型和标记器。
    
    
    >>> ARTICLE_TO_SUMMARIZE = "Meine Freunde sind cool, aber sie essen zu viel Kuchen."
    >>> inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="np")
    
    定义要进行摘要的文章，并使用tokenizer将其转换为模型所需的输入格式。
    
    
    >>> # Generate Summary
    >>> summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=5).sequences
    >>> print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))
    
    使用模型生成文章的摘要，指定生成4个束（beam），最大长度为5，然后解码生成的摘要并打印。
    
    
    >>> from transformers import AutoTokenizer, FlaxMBartForConditionalGeneration
    
    >>> model = FlaxMBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")
    >>> tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")
    
    再次加载MBart模型和标记器，确保环境准备好用于示例。
    
    
    >>> # de_DE is the language symbol id <LID> for German
    >>> TXT = "</s> Meine Freunde sind <mask> nett aber sie essen zu viel Kuchen. </s> de_DE"
    >>> input_ids = tokenizer([TXT], add_special_tokens=False, return_tensors="np")["input_ids"]
    
    定义一个包含掩码填充的例子，`TXT`包含一个掩码标记`<mask>`，表示需要填充的位置。将`TXT`编码为模型可接受的输入格式。
    
    
    >>> logits = model(input_ids).logits
    >>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero()[0].item()
    >>> probs = logits[0, masked_index].softmax(dim=0)
    >>> values, predictions = probs.topk(5)
    
    使用模型预测掩码位置的概率分布，并选择最高的五个概率值。
    
    
    >>> tokenizer.decode(predictions).split()
    
    将预测的结果解码为文本序列，并分割为单词列表。
"""

# 调用函数`overwrite_call_docstring`，用于重写模型类的文档字符串
overwrite_call_docstring(
    FlaxMBartForConditionalGeneration, MBART_INPUTS_DOCSTRING + FLAX_MBART_CONDITIONAL_GENERATION_DOCSTRING
)
# 调用函数`append_replace_return_docstrings`，用于追加或替换模型类的返回值文档字符串
append_replace_return_docstrings(
    FlaxMBartForConditionalGeneration, output_type=FlaxSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC
)


# 从`transformers.models.bart.modeling_flax_bart.FlaxBartForSequenceClassificationModule`复制代码，将Bart改为MBart
class FlaxMBartForSequenceClassificationModule(nn.Module):
    config: MBartConfig  # 定义MBart配置
    dtype: jnp.dtype = jnp.float32  # 设置数据类型为32位浮点数
    num_labels: Optional[int] = None  # 可选的标签数量

    def setup(self):
        # 初始化MBart模型和分类头部
        self.model = FlaxMBartModule(config=self.config, dtype=self.dtype)
        self.classification_head = FlaxMBartClassificationHead(
            config=self.config,
            inner_dim=self.config.d_model,
            num_classes=self.num_labels if self.num_labels is not None else self.config.num_labels,
            pooler_dropout=self.config.classifier_dropout,
        )

    def _get_encoder_module(self):
        # 获取编码器模块
        return self.model.encoder

    def _get_decoder_module(self):
        # 获取解码器模块
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
        # 定义模型调用接口，接受多个输入参数和控制参数

        # 返回字典格式的结果，控制是否返回注意力权重和隐藏状态
        return self.model(
            input_ids,
            attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            position_ids=position_ids,
            decoder_position_ids=decoder_position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )
        ):
            # 调用模型进行推理
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
            hidden_states = outputs[0]  # 最后一个隐藏状态

            # 创建一个掩码，用于标记输入中的结束符（<eos>）
            eos_mask = jnp.where(input_ids == self.config.eos_token_id, 1, 0)

            # 处理 JAX 编译时的类型错误
            if type(eos_mask) != jax.interpreters.partial_eval.DynamicJaxprTracer:
                # 检查所有样本是否具有相同数量的 <eos> 标记
                if len(jnp.unique(eos_mask.sum(1))) > 1:
                    raise ValueError("所有示例必须具有相同数量的 <eos> 标记。")

                # 检查输入中是否有缺失的 <eos> 标记
                if any(eos_mask.sum(1) == 0):
                    raise ValueError("输入中缺少 <eos> 标记。")

                # 确保每个示例只保留最后一个 <eos> 标记
                eos_mask_noised = eos_mask + jnp.arange(eos_mask.shape[1]) * 1e-6
                eos_mask = jnp.where(eos_mask_noised == eos_mask_noised.max(1).reshape(-1, 1), 1, 0)

            # 使用 <eos> 标记计算句子表示
            sentence_representation = jnp.einsum("ijk, ij -> ijk", hidden_states, eos_mask).sum(1)

            # 使用分类头部对句子表示进行分类预测
            logits = self.classification_head(sentence_representation, deterministic=deterministic)

            # 如果不返回字典，则返回元组
            if not return_dict:
                output = (logits,) + outputs[1:]
                return output

            # 如果返回字典，则返回 FlaxSeq2SeqSequenceClassifierOutput 类的实例
            return FlaxSeq2SeqSequenceClassifierOutput(
                logits=logits,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )
@add_start_docstrings(
    """
    MBart model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE
    tasks.
    """,
    MBART_START_DOCSTRING,
)
"""
使用`add_start_docstrings`装饰器为`FlaxMBartForSequenceClassification`类添加文档字符串，描述其作为带有顶部序列分类/头部的MBart模型。
"""

class FlaxMBartForSequenceClassification(FlaxMBartPreTrainedModel):
    """
    MBart序列分类模型，继承自`FlaxMBartPreTrainedModel`。
    """
    module_class = FlaxMBartForSequenceClassificationModule
    dtype = jnp.float32

append_call_sample_docstring(
    FlaxMBartForSequenceClassification,
    _CHECKPOINT_FOR_DOC,
    FlaxSeq2SeqSequenceClassifierOutput,
    _CONFIG_FOR_DOC,
)
"""
使用`append_call_sample_docstring`函数为`FlaxMBartForSequenceClassification`类添加示例调用文档字符串。
"""

# Copied from transformers.models.bart.modeling_flax_bart.FlaxBartForQuestionAnsweringModule with Bart->MBart
"""
从`transformers.models.bart.modeling_flax_bart.FlaxBartForQuestionAnsweringModule`复制代码，并将Bart替换为MBart。
"""

class FlaxMBartForQuestionAnsweringModule(nn.Module):
    """
    MBart问答模块定义，继承自`nn.Module`。
    """
    config: MBartConfig
    dtype: jnp.dtype = jnp.float32
    num_labels = 2

    def setup(self):
        """
        设置方法，初始化模型和输出层。
        """
        self.model = FlaxMBartModule(config=self.config, dtype=self.dtype)
        self.qa_outputs = nn.Dense(
            self.num_labels, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std)
        )

    def _get_encoder_module(self):
        """
        获取编码器模块的私有方法。
        """
        return self.model.encoder

    def _get_decoder_module(self):
        """
        获取解码器模块的私有方法。
        """
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
        """
        模型调用方法，接受多个输入和参数，返回包含多个输出的字典或元组。

        Args:
            input_ids: 输入的编码器输入id。
            attention_mask: 编码器的注意力掩码。
            decoder_input_ids: 解码器输入id。
            decoder_attention_mask: 解码器的注意力掩码。
            position_ids: 输入的位置id。
            decoder_position_ids: 解码器的位置id。
            output_attentions: 是否输出注意力权重。
            output_hidden_states: 是否输出隐藏状态。
            return_dict: 是否以字典形式返回输出。
            deterministic: 是否确定性计算。

        Returns:
            根据return_dict返回不同结构的输出。
        """
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
    MBart Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    MBART_START_DOCSTRING,



# MBart 模型，使用顶部的跨度分类头部用于抽取式问答任务，如 SQuAD（在隐藏状态输出之上的线性层，用于计算“起始跨度对数”和“结束跨度对数”）。
MBart Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
"""
# 导入 MBART_START_DOCSTRING，可能是一个模型文档字符串的起始标记或常量
MBART_START_DOCSTRING,
)
# 在此处代码似乎存在语法错误，可能是由于括号未正确闭合引起的问题
class FlaxMBartForQuestionAnswering(FlaxMBartPreTrainedModel):
    # 将模块类指定为 FlaxMBartForQuestionAnsweringModule
    module_class = FlaxMBartForQuestionAnsweringModule
    # 指定数据类型为 jnp.float32
    dtype = jnp.float32


# 向 FlaxMBartForQuestionAnswering 类附加一个调用样本文档字符串的函数
append_call_sample_docstring(
    FlaxMBartForQuestionAnswering,
    _CHECKPOINT_FOR_DOC,
    FlaxSeq2SeqQuestionAnsweringModelOutput,
    _CONFIG_FOR_DOC,
)
```