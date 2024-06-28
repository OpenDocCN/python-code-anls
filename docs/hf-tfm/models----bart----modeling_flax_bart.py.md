# `.\models\bart\modeling_flax_bart.py`

```py
# coding=utf-8
# Copyright 2021 The Fairseq Authors and The Google Flax Team Authors And The HuggingFace Inc. team. All rights reserved.
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
""" Flax Bart model."""

import math  # 导入数学函数库
import random  # 导入随机数函数库
from functools import partial  # 导入偏函数模块
from typing import Callable, Optional, Tuple  # 导入类型提示

import flax.linen as nn  # 导入Flax的linen模块作为nn别名
import jax  # 导入JAX库
import jax.numpy as jnp  # 导入JAX的NumPy接口，并且用jnp作为别名
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze  # 导入冻结字典相关函数
from flax.linen import combine_masks, make_causal_mask  # 导入生成掩码相关函数
from flax.linen.attention import dot_product_attention_weights  # 导入注意力权重计算函数
from flax.traverse_util import flatten_dict, unflatten_dict  # 导入字典扁平化和还原相关函数
from jax import lax  # 导入JAX的lax库
from jax.random import PRNGKey  # 导入PRNGKey，伪随机数生成器

from ...modeling_flax_outputs import (
    FlaxBaseModelOutput,  # 导入基础模型输出
    FlaxBaseModelOutputWithPastAndCrossAttentions,  # 导入包含过去和交叉注意力的基础模型输出
    FlaxCausalLMOutputWithCrossAttentions,  # 导入包含交叉注意力的因果语言建模输出
    FlaxSeq2SeqLMOutput,  # 导入序列到序列语言建模输出
    FlaxSeq2SeqModelOutput,  # 导入序列到序列模型输出
    FlaxSeq2SeqQuestionAnsweringModelOutput,  # 导入序列到序列问答模型输出
    FlaxSeq2SeqSequenceClassifierOutput,  # 导入序列到序列序列分类器输出
)
from ...modeling_flax_utils import (
    ACT2FN,  # 导入激活函数到函数名称的映射
    FlaxPreTrainedModel,  # 导入Flax预训练模型基类
    append_call_sample_docstring,  # 导入追加调用样例文档字符串函数
    append_replace_return_docstrings,  # 导入追加替换返回文档字符串函数
    overwrite_call_docstring,  # 导入覆盖调用文档字符串函数
)
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings  # 导入工具函数和模型前向文档字符串处理函数
from .configuration_bart import BartConfig  # 导入BART配置

logger = logging.get_logger(__name__)  # 获取logger对象

_CHECKPOINT_FOR_DOC = "facebook/bart-base"  # 预训练模型的文档检查点
_CONFIG_FOR_DOC = "BartConfig"  # BART模型配置的文档

BART_START_DOCSTRING = r"""
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
    # 参数说明:
    # config ([`BartConfig`]): 模型配置类，包含模型的所有参数。
    #     使用配置文件初始化不会加载模型的权重，只加载配置信息。
    #     可查看 [`~FlaxPreTrainedModel.from_pretrained`] 方法来加载模型权重。
    # dtype (`jax.numpy.dtype`, *可选*, 默认为 `jax.numpy.float32`):
    #     计算时所用的数据类型。可以是 `jax.numpy.float32`, `jax.numpy.float16`（在GPU上）和 `jax.numpy.bfloat16`（在TPU上）之一。
    #
    #     这可以用于在GPU或TPU上启用混合精度训练或半精度推断。如果指定了dtype，则所有计算将使用给定的dtype进行。
    #
    #     **请注意，这仅指定计算时的数据类型，并不影响模型参数的数据类型。**
    #
    #     如果希望更改模型参数的数据类型，请参阅 [`~FlaxPreTrainedModel.to_fp16`] 和 [`~FlaxPreTrainedModel.to_bf16`]。
"""
定义 BART 输入文档字符串
"""
BART_INPUTS_DOCSTRING = r"""
"""


"""
定义 BART 编码输入文档字符串
Args:
    input_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
        输入序列标记在词汇表中的索引。默认情况下，将忽略填充。
        
        可以使用 [`AutoTokenizer`] 获取索引。详情请参阅 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`]。

        [什么是输入 ID？](../glossary#input-ids)
    attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
        避免在填充标记索引上执行注意力的掩码。掩码值选在 `[0, 1]`：

        - 1 表示**未屏蔽**的标记，
        - 0 表示**已屏蔽**的标记。

        [什么是注意力掩码？](../glossary#attention-mask)
    position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
        每个输入序列标记在位置嵌入中的位置索引。选在范围 `[0, config.max_position_embeddings - 1]`。
    output_attentions (`bool`, *optional*):
        是否返回所有注意力层的注意力张量。详见返回张量中的 `attentions`。
    output_hidden_states (`bool`, *optional*):
        是否返回所有层的隐藏状态。详见返回张量中的 `hidden_states`。
    return_dict (`bool`, *optional*):
        是否返回 [`~utils.ModelOutput`] 而非普通元组。
"""


"""
定义 BART 解码输入文档字符串
"""
BART_DECODE_INPUTS_DOCSTRING = r"""
"""


def shift_tokens_right(input_ids: jnp.ndarray, pad_token_id: int, decoder_start_token_id: int) -> jnp.ndarray:
    """
    将输入 ID 向右移动一个标记。
    """
    shifted_input_ids = jnp.zeros_like(input_ids)
    shifted_input_ids = shifted_input_ids.at[:, 1:].set(input_ids[:, :-1])  # 将输入向右移动一个位置
    shifted_input_ids = shifted_input_ids.at[:, 0].set(decoder_start_token_id)  # 在起始位置插入解码器起始标记

    shifted_input_ids = jnp.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)  # 替换特殊标记为填充标记
    return shifted_input_ids


class FlaxBartAttention(nn.Module):
    """
    FlaxBartAttention 类定义
    """
    config: BartConfig
    embed_dim: int
    num_heads: int
    dropout: float = 0.0
    causal: bool = False
    bias: bool = True
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型
    # 设置函数，初始化注意力头的维度
    def setup(self) -> None:
        # 计算每个注意力头的维度
        self.head_dim = self.embed_dim // self.num_heads
        # 检查 embed_dim 是否可以整除 num_heads，否则抛出数值错误异常
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )

        # 定义一个局部函数 dense，部分应用 Dense 层的参数
        dense = partial(
            nn.Dense,
            self.embed_dim,
            use_bias=self.bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )

        # 创建查询、键、值以及输出投影的 Dense 层
        self.q_proj, self.k_proj, self.v_proj = dense(), dense(), dense()
        self.out_proj = dense()

        # 初始化 Dropout 层
        self.dropout_layer = nn.Dropout(rate=self.dropout)

        # 如果启用因果注意力，创建一个因果掩码
        if self.causal:
            self.causal_mask = make_causal_mask(
                jnp.ones((1, self.config.max_position_embeddings), dtype="bool"), dtype="bool"
            )

    # 将隐藏状态按注意力头进行分割
    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    # 将分割后的注意力头合并回原始形状
    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    # 使用 JAX 编译这个类的方法
    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slightly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # 检测是否通过检查"cache"变量来初始化缓存数据。
        is_initialized = self.has_variable("cache", "cached_key")
        # 获取或初始化缓存中的键和值
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        # 获取或初始化缓存中的索引，作为当前缓存操作的起始位置
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            # 提取缓存键值张量的维度信息，其中包括批次维度、序列长度、注意力头数和每头注意力深度
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # 使用新的1D空间切片更新键和值的缓存
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            # 更新缓存中的键和值
            cached_key.value = key
            cached_value.value = value
            # 更新缓存索引，增加已更新缓存向量的数量
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # 生成用于缓存的因果掩码，确保每个查询位置只注意到已生成并缓存的键位置，而不是剩余的零元素。
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            # 合并现有的注意力掩码和生成的因果掩码
            attention_mask = combine_masks(pad_mask, attention_mask)
        
        # 返回更新后的键、值和注意力掩码
        return key, value, attention_mask
class FlaxBartEncoderLayer(nn.Module):
    config: BartConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        # 设置编码器层的嵌入维度为模型配置中的维度
        self.embed_dim = self.config.d_model
        # 初始化自注意力机制
        self.self_attn = FlaxBartAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.encoder_attention_heads,
            dropout=self.config.attention_dropout,
            dtype=self.dtype,
        )
        # 初始化自注意力层规范化
        self.self_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        # 初始化 dropout 层
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)
        # 设置激活函数
        self.activation_fn = ACT2FN[self.config.activation_function]
        # 初始化激活函数的 dropout 层
        self.activation_dropout_layer = nn.Dropout(rate=self.config.activation_dropout)
        # 第一个全连接层，映射到编码器的前馈神经网络维度
        self.fc1 = nn.Dense(
            self.config.encoder_ffn_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        # 第二个全连接层，映射回嵌入维度
        self.fc2 = nn.Dense(
            self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std)
        )
        # 最终层规范化
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
        # 使用自注意力机制计算新的隐藏状态和注意力权重
        hidden_states, attn_weights = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask)

        # 应用 dropout 到隐藏状态
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 添加残差连接
        hidden_states = residual + hidden_states
        # 应用自注意力层规范化
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 保存残差连接
        residual = hidden_states
        # 应用激活函数到第一个全连接层
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 应用激活函数的 dropout
        hidden_states = self.activation_dropout_layer(hidden_states, deterministic=deterministic)
        # 应用第二个全连接层
        hidden_states = self.fc2(hidden_states)
        # 应用 dropout 到第二个全连接层
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 添加残差连接
        hidden_states = residual + hidden_states
        # 应用最终层规范化
        hidden_states = self.final_layer_norm(hidden_states)

        # 返回隐藏状态作为输出
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，将它们添加到输出中
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class FlaxBartEncoderLayerCollection(nn.Module):
    config: BartConfig
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型

    def setup(self):
        # 初始化编码器层集合，每层使用不同的编号和数据类型
        self.layers = [
            FlaxBartEncoderLayer(self.config, name=str(i), dtype=self.dtype) for i in range(self.config.encoder_layers)
        ]
        # 设置层级丢弃率
        self.layerdrop = self.config.encoder_layerdrop

    def __call__(
        self,
        hidden_states,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        ):
            # 如果不需要输出注意力权重，则初始化一个空元组
            all_attentions = () if output_attentions else None
            # 如果不需要输出隐藏状态，则初始化一个空元组
            all_hidden_states = () if output_hidden_states else None

            # 遍历每个编码器层
            for encoder_layer in self.layers:
                if output_hidden_states:
                    # 如果需要输出隐藏状态，则将当前隐藏状态加入到所有隐藏状态的元组中
                    all_hidden_states = all_hidden_states + (hidden_states,)
                # 添加LayerDrop功能（参见https://arxiv.org/abs/1909.11556进行描述）
                dropout_probability = random.uniform(0, 1)
                # 如果非确定性且随机数小于层丢弃率，则跳过当前层
                if not deterministic and (dropout_probability < self.layerdrop):  # skip the layer
                    # 设置当前层输出为None
                    layer_outputs = (None, None)
                else:
                    # 否则，调用当前编码器层进行前向传播计算
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        output_attentions,
                        deterministic,
                    )
                # 更新当前隐藏状态为编码器层输出的第一个元素
                hidden_states = layer_outputs[0]
                # 如果需要输出注意力权重，则将当前层的注意力权重加入到所有注意力权重的元组中
                if output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

            # 如果需要输出隐藏状态，则将最终的隐藏状态加入到所有隐藏状态的元组中
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # 构建模型输出结果
            outputs = (hidden_states, all_hidden_states, all_attentions)

            # 如果不使用返回字典格式，则返回输出元组中非None的部分
            if not return_dict:
                return tuple(v for v in outputs if v is not None)

            # 使用FlaxBaseModelOutput类包装输出结果并以字典格式返回
            return FlaxBaseModelOutput(
                last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
            )
# 定义一个名为 FlaxBartDecoderLayer 的类，继承自 nn.Module，表示这是一个神经网络模块
class FlaxBartDecoderLayer(nn.Module):
    # 类变量 config，指定为 BartConfig 类型，用于配置模型参数
    config: BartConfig
    # 类变量 dtype，默认为 jnp.float32 类型
    dtype: jnp.dtype = jnp.float32

    # 初始化方法，设置类的初始状态
    def setup(self) -> None:
        # 设置类的嵌入维度为配置中的 d_model 参数
        self.embed_dim = self.config.d_model
        # 初始化 self_attn 层，使用 FlaxBartAttention 自定义类，实现自注意力机制
        self.self_attn = FlaxBartAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.decoder_attention_heads,
            dropout=self.config.attention_dropout,
            causal=True,
            dtype=self.dtype,
        )
        # 初始化 dropout_layer 层，用于随机断开神经元连接，防止过拟合
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)
        # 根据配置中的激活函数选择对应的激活函数
        self.activation_fn = ACT2FN[self.config.activation_function]
        # 初始化 activation_dropout_layer 层，对激活函数的输出进行随机断开
        self.activation_dropout_layer = nn.Dropout(rate=self.config.activation_dropout)

        # 初始化 self_attn_layer_norm 层，用 LayerNorm 进行归一化处理
        self.self_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        # 初始化 encoder_attn 层，实现编码器-解码器注意力机制
        self.encoder_attn = FlaxBartAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.decoder_attention_heads,
            dropout=self.config.attention_dropout,
            dtype=self.dtype,
        )
        # 初始化 encoder_attn_layer_norm 层，对编码器-解码器注意力输出进行归一化
        self.encoder_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        # 初始化 fc1 层，全连接层，将输入映射到更高维度的空间
        self.fc1 = nn.Dense(
            self.config.decoder_ffn_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        # 初始化 fc2 层，全连接层，将高维度的输出映射回原始维度
        self.fc2 = nn.Dense(
            self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std)
        )
        # 初始化 final_layer_norm 层，对最终输出进行归一化处理
        self.final_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    # 类的调用方法，定义类在被调用时的行为
    def __call__(
        self,
        hidden_states: jnp.ndarray,  # 输入的隐藏状态，使用 JAX 的数组表示
        attention_mask: jnp.ndarray,  # 注意力掩码，指定哪些位置需要注意力
        encoder_hidden_states: Optional[jnp.ndarray] = None,  # 编码器的隐藏状态，可选参数
        encoder_attention_mask: Optional[jnp.ndarray] = None,  # 编码器的注意力掩码，可选参数
        init_cache: bool = False,  # 是否初始化缓存，用于存储计算结果的中间状态
        output_attentions: bool = True,  # 是否输出注意力权重
        deterministic: bool = True,  # 是否使用确定性计算结果
    ) -> Tuple[jnp.ndarray]:
        # 保留原始输入作为残差连接的一部分
        residual = hidden_states

        # 自注意力机制
        # 调用 self_attn 方法进行自注意力计算
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, init_cache=init_cache
        )
        # 应用 dropout 层
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 添加残差连接
        hidden_states = residual + hidden_states
        # 应用自注意力层的 Layer Normalization
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 跨注意力块
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            # 保留当前隐藏状态作为残差连接的一部分
            residual = hidden_states

            # 调用 encoder_attn 方法进行跨注意力计算
            hidden_states, cross_attn_weights = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            # 应用 dropout 层
            hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
            # 添加残差连接
            hidden_states = residual + hidden_states
            # 应用跨注意力层的 Layer Normalization
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # 全连接层
        # 保留当前隐藏状态作为残差连接的一部分
        residual = hidden_states
        # 应用激活函数和第一个全连接层 fc1
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 应用激活函数后的 dropout 层
        hidden_states = self.activation_dropout_layer(hidden_states, deterministic=deterministic)
        # 应用第二个全连接层 fc2
        hidden_states = self.fc2(hidden_states)
        # 应用 dropout 层
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 添加残差连接
        hidden_states = residual + hidden_states
        # 应用最终的 Layer Normalization
        hidden_states = self.final_layer_norm(hidden_states)

        # 输出设置为一个包含隐藏状态的元组
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则将它们添加到输出中
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs
class FlaxBartDecoderLayerCollection(nn.Module):
    config: BartConfig
    dtype: jnp.dtype = jnp.float32  # 计算中使用的数据类型

    def setup(self):
        # 创建多个 FlaxBartDecoderLayer 实例作为层集合
        self.layers = [
            FlaxBartDecoderLayer(self.config, name=str(i), dtype=self.dtype) for i in range(self.config.decoder_layers)
        ]
        # 从配置中获取并设置层丢弃率
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
        # 初始化用于存储所有隐藏状态、自注意力、交叉注意力的元组，根据参数决定是否存储
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        # 遍历每个解码器层
        for decoder_layer in self.layers:
            if output_hidden_states:
                # 如果需要输出隐藏状态，则记录当前隐藏状态
                all_hidden_states += (hidden_states,)
                # 添加层丢弃 (LayerDrop) 描述，参考论文 https://arxiv.org/abs/1909.11556

            # 随机生成丢弃概率
            dropout_probability = random.uniform(0, 1)
            # 如果不是确定性计算且随机数小于层丢弃率，则将层输出置为None
            if not deterministic and (dropout_probability < self.layerdrop):
                layer_outputs = (None, None, None)
            else:
                # 否则调用解码器层进行计算
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    init_cache=init_cache,
                    output_attentions=output_attentions,
                    deterministic=deterministic,
                )

            # 更新隐藏状态为当前解码器层的输出的第一个元素
            hidden_states = layer_outputs[0]
            if output_attentions:
                # 如果需要输出注意力，记录自注意力分数
                all_self_attns += (layer_outputs[1],)

                # 如果有编码器的隐藏状态，记录交叉注意力分数
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # 如果需要输出隐藏状态，则添加最后一个解码器层的隐藏状态
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 组装输出结果，根据需求返回字典或元组
        outputs = [hidden_states, all_hidden_states, all_self_attns, all_cross_attentions]

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        # 返回带过去和交叉注意力的基础模型输出
        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class FlaxBartClassificationHead(nn.Module):
    """用于句子级分类任务的头部模块。"""

    config: BartConfig
    inner_dim: int
    num_classes: int
    pooler_dropout: float
    dtype: jnp.dtype = jnp.float32
    # 定义模型初始化方法
    def setup(self):
        # 初始化一个全连接层对象，设置输入维度为 self.inner_dim，数据类型为 self.dtype，
        # 使用正态分布初始化权重，标准差为 self.config.init_std
        self.dense = nn.Dense(
            self.inner_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std)
        )
        # 初始化一个 Dropout 层对象，设置丢弃率为 self.pooler_dropout
        self.dropout = nn.Dropout(rate=self.pooler_dropout)
        # 初始化一个全连接层对象，设置输出维度为 self.num_classes，数据类型为 self.dtype，
        # 使用正态分布初始化权重，标准差为 self.config.init_std
        self.out_proj = nn.Dense(
            self.num_classes,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )

    # 定义模型调用方法
    def __call__(self, hidden_states: jnp.ndarray, deterministic: bool):
        # 对输入 hidden_states 应用 Dropout 层，根据 deterministic 参数决定是否使用确定性推断
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 将经过 Dropout 处理后的 hidden_states 输入到全连接层 self.dense 中进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对 hidden_states 中的每个元素应用双曲正切函数
        hidden_states = jnp.tanh(hidden_states)
        # 再次对经过 tanh 函数处理后的 hidden_states 应用 Dropout 层
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 将经过 Dropout 处理后的 hidden_states 输入到全连接层 self.out_proj 中进行线性变换
        hidden_states = self.out_proj(hidden_states)
        # 返回处理后的 hidden_states
        return hidden_states
# 定义 FlaxBartEncoder 类，继承自 nn.Module
class FlaxBartEncoder(nn.Module):
    # 引入 BartConfig 类型的配置参数 config
    config: BartConfig
    # 嵌入词汇表的 nn.Embed 类型对象 embed_tokens
    embed_tokens: nn.Embed
    # 计算过程中使用的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    # 模型初始化方法
    def setup(self):
        # 根据配置参数中的 dropout 率创建 Dropout 层
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)

        # 获取模型的嵌入维度
        embed_dim = self.config.d_model
        # 设置填充索引，从配置参数中获取
        self.padding_idx = self.config.pad_token_id
        # 设置最大源序列长度，从配置参数中获取
        self.max_source_positions = self.config.max_position_embeddings
        # 设置嵌入缩放因子，根据配置参数是否需要缩放
        self.embed_scale = math.sqrt(embed_dim) if self.config.scale_embedding else 1.0

        # Bart 模型的特殊设置，如果指定了 padding_idx 则需要偏移嵌入 ids 2 个单位
        # 并相应调整 num_embeddings。其他模型没有这种特殊处理
        self.offset = 2
        # 初始化嵌入位置的 nn.Embed 层
        self.embed_positions = nn.Embed(
            self.config.max_position_embeddings + self.offset,  # 嵌入位置的最大长度加上偏移量
            embed_dim,  # 嵌入的维度
            embedding_init=jax.nn.initializers.normal(self.config.init_std),  # 初始化方法为正态分布
            dtype=self.dtype,  # 指定数据类型
        )
        # 创建包含多个编码器层的集合
        self.layers = FlaxBartEncoderLayerCollection(self.config, self.dtype)
        # 对嵌入层进行 LayerNorm 规范化
        self.layernorm_embedding = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    # 模型调用方法
    def __call__(
        self,
        input_ids,  # 输入的 token ids
        attention_mask,  # 注意力遮罩
        position_ids,  # 位置 ids
        output_attentions: bool = False,  # 是否输出注意力权重
        output_hidden_states: bool = False,  # 是否输出隐藏状态
        return_dict: bool = True,  # 是否以字典形式返回结果
        deterministic: bool = True,  # 是否确定性计算
    ):
        # 获取输入的形状信息
        input_shape = input_ids.shape
        # 将输入 ids 展平为二维张量
        input_ids = input_ids.reshape(-1, input_shape[-1])

        # 根据嵌入 ids 获取对应的嵌入向量，并乘以嵌入缩放因子
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # 根据位置 ids 获取嵌入的位置向量
        embed_pos = self.embed_positions(position_ids + self.offset)

        # 将输入的嵌入向量和位置向量相加作为初始隐藏状态
        hidden_states = inputs_embeds + embed_pos
        # 对隐藏状态进行嵌入层规范化
        hidden_states = self.layernorm_embedding(hidden_states)
        # 使用 Dropout 层对隐藏状态进行随机置零处理
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)

        # 将隐藏状态传递给多层编码器层处理
        outputs = self.layers(
            hidden_states,
            attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果不以字典形式返回结果，则直接返回 outputs
        if not return_dict:
            return outputs

        # 以 FlaxBaseModelOutput 类型的字典形式返回结果
        return FlaxBaseModelOutput(
            last_hidden_state=outputs.last_hidden_state,  # 最后的隐藏状态
            hidden_states=outputs.hidden_states,  # 隐藏状态列表
            attentions=outputs.attentions,  # 注意力权重列表
        )
    # 初始化方法，设置模型的一些基本属性和层
    def setup(self):
        # 定义一个dropout层，用于在训练过程中随机丢弃部分神经元，防止过拟合
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)

        # 获取嵌入向量的维度，填充标记的索引，以及目标位置的最大值
        embed_dim = self.config.d_model
        self.padding_idx = self.config.pad_token_id
        self.max_target_positions = self.config.max_position_embeddings
        # 根据配置是否对嵌入向量进行缩放
        self.embed_scale = math.sqrt(self.config.d_model) if self.config.scale_embedding else 1.0

        # 如果padding_idx被指定，则调整嵌入id，通过offset为2调整num_embeddings
        # 其他模型不需要此调整
        self.offset = 2
        # 初始化位置嵌入层，输入大小为最大位置嵌入加上偏移量，输出维度为embed_dim
        self.embed_positions = nn.Embed(
            self.config.max_position_embeddings + self.offset,
            embed_dim,
            embedding_init=jax.nn.initializers.normal(self.config.init_std),
            dtype=self.dtype,
        )

        # 初始化解码器层集合
        self.layers = FlaxBartDecoderLayerCollection(self.config, self.dtype)
        # 初始化层归一化层，用于归一化嵌入层的输出
        self.layernorm_embedding = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    # 调用方法，执行模型的前向计算过程
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
        # 获取输入的形状信息，并重新调整input_ids的形状
        input_shape = input_ids.shape
        input_ids = input_ids.reshape(-1, input_shape[-1])

        # 嵌入输入token
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # 嵌入位置信息
        positions = self.embed_positions(position_ids + self.offset)

        # 将嵌入的token和位置信息相加得到隐藏状态
        hidden_states = inputs_embeds + positions
        # 对隐藏状态进行层归一化
        hidden_states = self.layernorm_embedding(hidden_states)

        # 对隐藏状态应用dropout层，根据deterministic参数确定是否确定性操作
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)

        # 通过解码器层进行前向传播
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

        # 如果return_dict为False，则直接返回outputs
        if not return_dict:
            return outputs

        # 如果return_dict为True，则返回包含过去和交叉注意力的FlaxBaseModelOutputWithPastAndCrossAttentions对象
        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )
class FlaxBartModule(nn.Module):
    config: BartConfig
    dtype: jnp.dtype = jnp.float32  # 计算时的数据类型

    def setup(self):
        self.shared = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.init_std),
            dtype=self.dtype,
        )

        # 初始化编码器和解码器模块
        self.encoder = FlaxBartEncoder(self.config, dtype=self.dtype, embed_tokens=self.shared)
        self.decoder = FlaxBartDecoder(self.config, dtype=self.dtype, embed_tokens=self.shared)

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
        # 调用编码器并获取其输出
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        # 调用解码器并获取其输出
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

        # 根据 return_dict 决定返回类型
        if not return_dict:
            return decoder_outputs + encoder_outputs

        # 返回经过 Seq2Seq 模型输出后的结果
        return FlaxSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class FlaxBartPreTrainedModel(FlaxPreTrainedModel):
    config_class = BartConfig
    base_model_prefix: str = "model"
    module_class: nn.Module = None

    def __init__(
        self,
        config: BartConfig,
        input_shape: Tuple[int] = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        # 使用给定的配置初始化模块
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)
    # 初始化模型参数的函数，使用给定的随机数生成器 rng，输入形状 input_shape 和可选的参数 params
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量 input_ids，全零张量，数据类型为整数
        input_ids = jnp.zeros(input_shape, dtype="i4")
        # 确保初始化阶段适用于 FlaxBartForSequenceClassificationModule
        # 将 input_ids 的最后一个位置设为配置中的 eos_token_id
        input_ids = input_ids.at[(..., -1)].set(self.config.eos_token_id)
        # 创建注意力掩码，与 input_ids 大小相同，全为 1
        attention_mask = jnp.ones_like(input_ids)
        # 解码器输入与输入相同
        decoder_input_ids = input_ids
        # 解码器注意力掩码与输入相同
        decoder_attention_mask = jnp.ones_like(input_ids)

        # 获取批量大小和序列长度
        batch_size, sequence_length = input_ids.shape
        # 创建位置编码张量，形状与 input_ids 相同，内容为序列长度的广播值
        position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))
        # 解码器位置编码与输入相同
        decoder_position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # 分割随机数生成器 rng 以用于参数和 dropout
        params_rng, dropout_rng = jax.random.split(rng)
        # 组合随机数生成器
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 使用模型的初始化方法初始化随机参数
        random_params = self.module.init(
            rngs,
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            position_ids,
            decoder_position_ids,
        )["params"]

        # 如果提供了初始参数 params，则将随机生成的参数与其合并
        if params is not None:
            # 展平并解冻随机生成的参数
            random_params = flatten_dict(unfreeze(random_params))
            # 展平并解冻提供的参数
            params = flatten_dict(unfreeze(params))
            # 将缺失的键从随机参数复制到提供的参数中
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            # 清空缺失键集合
            self._missing_keys = set()
            # 冻结并返回合并后的参数
            return freeze(unflatten_dict(params))
        else:
            # 如果没有提供初始参数，则直接返回随机生成的参数
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

        # 使用模型的初始化方法初始化变量
        init_variables = self.module.init(
            jax.random.PRNGKey(0),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            encoder_hidden_states=encoder_outputs[0],  # 使用编码器输出的最后隐藏状态初始化
            init_cache=True,
            method=_decoder_forward,  # 我们只需调用解码器来初始化缓存
        )
        # 解冻缓存变量并返回
        return unfreeze(init_variables["cache"])

    @add_start_docstrings(BART_ENCODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutput, config_class=BartConfig)
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

        ```
        >>> from transformers import AutoTokenizer, FlaxBartForConditionalGeneration

        >>> model = FlaxBartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

        >>> text = "My friends are cool but they eat too many carbs."
        >>> inputs = tokenizer(text, max_length=1024, return_tensors="jax")
        >>> encoder_outputs = model.encode(**inputs)
        ```"""
        # 初始化输出配置，如果未指定则使用模型配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 如果 attention_mask 未提供，则使用全 1 的张量作为默认值
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        # 如果 position_ids 未提供，则根据 input_ids 的形状自动广播生成位置编码
        if position_ids is None:
            batch_size, sequence_length = input_ids.shape
            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # 处理可能存在的随机数生成器 PRNG
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        # 定义内部函数 _encoder_forward 用于编码器的前向传播
        def _encoder_forward(module, input_ids, attention_mask, position_ids, **kwargs):
            encode_module = module._get_encoder_module()
            return encode_module(input_ids, attention_mask, position_ids, **kwargs)

        # 调用 Flax 模型的 apply 方法进行编码器的正向传播
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

    @add_start_docstrings(BART_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutputWithPastAndCrossAttentions, config_class=BartConfig)
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
    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
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
        # 确定是否输出注意力权重，默认从配置中获取
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 确定是否输出隐藏状态，默认从配置中获取
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 确定是否返回字典形式的输出，默认从配置中获取
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 准备编码器的输入
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)  # 使用与输入相同形状的全1注意力掩码
        if position_ids is None:
            batch_size, sequence_length = input_ids.shape
            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))
            # 若未提供位置编码，生成一个默认的位置编码矩阵

        # 准备解码器的输入
        if decoder_input_ids is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, decoder_start_token_id=self.config.decoder_start_token_id
            )
            # 若未提供解码器输入，使用右移函数生成以pad_token_id开头的序列
        if decoder_attention_mask is None:
            decoder_attention_mask = jnp.ones_like(decoder_input_ids)
            # 使用与解码器输入相同形状的全1注意力掩码
        if decoder_position_ids is None:
            batch_size, sequence_length = decoder_input_ids.shape
            decoder_position_ids = jnp.broadcast_to(
                jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
            )
            # 若未提供解码器位置编码，生成一个默认的位置编码矩阵

        # 处理需要的随机数生成器
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        return self.module.apply(
            {"params": params or self.params},  # 提供参数字典，若未提供则使用默认参数self.params
            input_ids=jnp.array(input_ids, dtype="i4"),  # 转换输入ids为指定类型的JAX数组
            attention_mask=jnp.array(attention_mask, dtype="i4"),  # 转换注意力掩码为指定类型的JAX数组
            position_ids=jnp.array(position_ids, dtype="i4"),  # 转换位置编码为指定类型的JAX数组
            decoder_input_ids=jnp.array(decoder_input_ids, dtype="i4"),  # 转换解码器输入ids为指定类型的JAX数组
            decoder_attention_mask=jnp.array(decoder_attention_mask, dtype="i4"),  # 转换解码器注意力掩码为指定类型的JAX数组
            decoder_position_ids=jnp.array(decoder_position_ids, dtype="i4"),  # 转换解码器位置编码为指定类型的JAX数组
            output_attentions=output_attentions,  # 是否输出注意力权重
            output_hidden_states=output_hidden_states,  # 是否输出隐藏状态
            return_dict=return_dict,  # 是否返回字典形式的输出
            deterministic=not train,  # 是否为确定性计算，取决于train参数
            rngs=rngs,  # 随机数生成器字典
        )
# 为 FlaxBartModel 类添加文档字符串，描述其作为 Bart 模型的基础转换器，输出原始隐藏状态而无需特定的输出头。
class FlaxBartModel(FlaxBartPreTrainedModel):
    config: BartConfig
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型为 jnp.float32
    module_class = FlaxBartModule

# 向 FlaxBartModel 类附加调用示例的文档字符串，以及 BART 的起始文档字符串。
append_call_sample_docstring(FlaxBartModel, _CHECKPOINT_FOR_DOC, FlaxSeq2SeqModelOutput, _CONFIG_FOR_DOC)

# 定义 FlaxBartForConditionalGenerationModule 类
class FlaxBartForConditionalGenerationModule(nn.Module):
    config: BartConfig
    dtype: jnp.dtype = jnp.float32
    bias_init: Callable[..., jnp.ndarray] = jax.nn.initializers.zeros

    def setup(self):
        # 初始化模型为 FlaxBartModule 实例，使用给定的配置和数据类型
        self.model = FlaxBartModule(config=self.config, dtype=self.dtype)
        # 初始化 lm_head 作为全连接层，输出维度为模型共享词汇表大小，不使用偏置，使用给定的初始化器
        self.lm_head = nn.Dense(
            self.model.shared.num_embeddings,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        # 初始化 final_logits_bias 作为模型参数，维度为 (1, 模型共享词汇表大小)，使用给定的偏置初始化器
        self.final_logits_bias = self.param("final_logits_bias", self.bias_init, (1, self.model.shared.num_embeddings))

    # 获取编码器模块的方法
    def _get_encoder_module(self):
        return self.model.encoder

    # 获取解码器模块的方法
    def _get_decoder_module(self):
        return self.model.decoder

    # 定义类的调用方法，接收多个输入和控制参数，并返回条件生成模型的输出
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
            # 使用模型进行推理，传入输入参数：input_ids, attention_mask, decoder_input_ids等
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

            # 从模型输出中获取隐藏状态
            hidden_states = outputs[0]

            # 如果配置要求共享词嵌入，则获取共享的嵌入层，并应用到语言模型的输出上
            if self.config.tie_word_embeddings:
                shared_embedding = self.model.variables["params"]["shared"]["embedding"]
                lm_logits = self.lm_head.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
            else:
                # 否则直接使用语言模型头部生成预测logits
                lm_logits = self.lm_head(hidden_states)

            # 将最终logits偏置加到语言模型logits上
            lm_logits += jax.lax.stop_gradient(self.final_logits_bias.astype(self.dtype))

            # 如果不需要返回字典形式的输出，则返回一个元组，包含lm_logits和其余输出
            if not return_dict:
                output = (lm_logits,) + outputs[1:]
                return output

            # 返回一个FlaxSeq2SeqLMOutput对象，包含各种输出，如logits、隐藏状态、注意力等
            return FlaxSeq2SeqLMOutput(
                logits=lm_logits,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )
# 使用装饰器为类添加文档字符串，指定了BART模型带有语言建模头部，可用于摘要生成
@add_start_docstrings(
    "The BART Model with a language modeling head. Can be used for summarization.", BART_START_DOCSTRING
)
class FlaxBartForConditionalGeneration(FlaxBartPreTrainedModel):
    # 指定模块类为FlaxBartForConditionalGenerationModule
    module_class = FlaxBartForConditionalGenerationModule
    # 指定数据类型为jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 使用装饰器添加解码方法的文档字符串
    @add_start_docstrings(BART_DECODE_INPUTS_DOCSTRING)
    # 替换返回值文档字符串，指定输出类型为FlaxCausalLMOutputWithCrossAttentions，配置类为BartConfig
    @replace_return_docstrings(output_type=FlaxCausalLMOutputWithCrossAttentions, config_class=BartConfig)
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
        # 解码方法，用于生成输出
        pass

    # 为生成准备输入的方法，准备生成时需要的输入数据
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        max_length,
        attention_mask: Optional[jax.Array] = None,
        decoder_attention_mask: Optional[jax.Array] = None,
        encoder_outputs=None,
        **kwargs,
    ):
        # 初始化缓存，用于存储先前的键值对
        batch_size, seq_length = decoder_input_ids.shape
        past_key_values = self.init_cache(batch_size, max_length, encoder_outputs)
        
        # 根据解码器的注意力掩码生成扩展的注意力掩码
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if decoder_attention_mask is not None:
            position_ids = decoder_attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, decoder_attention_mask, (0, 0))
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        # 返回生成所需的输入数据字典
        return {
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "encoder_attention_mask": attention_mask,
            "decoder_attention_mask": extended_attention_mask,
            "decoder_position_ids": position_ids,
        }

    # 更新生成输入的方法，根据模型输出和模型关键字参数更新输入数据
    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["decoder_position_ids"] = model_kwargs["decoder_position_ids"][:, -1:] + 1
        return model_kwargs


# Flax BART 条件生成的文档字符串，描述了返回的摘要示例和使用的例子
FLAX_BART_CONDITIONAL_GENERATION_DOCSTRING = """
    Returns:

    Summarization example:

    ```
    >>> from transformers import AutoTokenizer, FlaxBartForConditionalGeneration
    >>> model = FlaxBartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    # 使用预训练的 FlaxBart 模型加载条件生成模型，用于生成文本摘要
    >>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    # 使用预训练的 tokenizer 加载 BART 模型的分词器
    
    >>> ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
    # 待摘要的文章内容
    >>> inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="np")
    # 使用分词器对文章进行分词，并封装成适合模型输入的格式
    
    >>> # Generate Summary
    # 生成摘要的过程
    >>> summary_ids = model.generate(inputs["input_ids"]).sequences
    # 使用模型生成输入文章的摘要序列
    >>> print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))
    # 打印生成的摘要，跳过特殊标记并保持分词时的空格处理方式
    
    
    Mask filling example:
    
    
    >>> import jax
    # 导入 JAX 库，用于高性能数值计算
    >>> from transformers import AutoTokenizer, FlaxBartForConditionalGeneration
    
    >>> model = FlaxBartForConditionalGeneration.from_pretrained("facebook/bart-large")
    # 使用预训练的 FlaxBart 模型加载条件生成模型
    >>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    # 使用预训练的 tokenizer 加载 BART 模型的分词器
    
    >>> TXT = "My friends are <mask> but they eat too many carbs."
    # 带有掩码填充的文本示例
    >>> input_ids = tokenizer([TXT], return_tensors="jax")["input_ids"]
    # 使用分词器对带有掩码的文本进行分词，并封装成适合模型输入的格式
    
    >>> logits = model(input_ids).logits
    # 通过模型生成输入文本的 logits，用于获取每个词的预测概率
    >>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero()[0].item()
    # 找到掩码位置的索引
    >>> probs = jax.nn.softmax(logits[0, masked_index], axis=0)
    # 对掩码位置的 logits 进行 softmax 处理，得到预测概率分布
    >>> values, predictions = jax.lax.top_k(probs, k=1)
    # 获取最高概率的预测值和其对应的索引
    
    >>> tokenizer.decode(predictions).split()
    # 解码预测的标记并拆分成词汇列表
"""
将调用文档字符串覆盖为 BART 输入文档字符串和 FLAX BART 条件生成文档字符串的组合
"""
overwrite_call_docstring(
    FlaxBartForConditionalGeneration, BART_INPUTS_DOCSTRING + FLAX_BART_CONDITIONAL_GENERATION_DOCSTRING
)
"""
追加并替换 FlaxBartForConditionalGeneration 类的返回文档字符串
"""
append_replace_return_docstrings(
    FlaxBartForConditionalGeneration, output_type=FlaxSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC
)

"""
定义一个用于序列分类的 FlaxBartForSequenceClassificationModule 类
"""
class FlaxBartForSequenceClassificationModule(nn.Module):
    """
    BART 的配置
    """
    config: BartConfig
    """
    数据类型，默认为 32 位浮点数
    """
    dtype: jnp.dtype = jnp.float32
    """
    可选的标签数目
    """
    num_labels: Optional[int] = None

    """
    模型的设置方法
    """
    def setup(self):
        """
        创建 BART 模型实例
        """
        self.model = FlaxBartModule(config=self.config, dtype=self.dtype)
        """
        创建用于分类的 BART 分类头
        """
        self.classification_head = FlaxBartClassificationHead(
            config=self.config,
            inner_dim=self.config.d_model,
            num_classes=self.num_labels if self.num_labels is not None else self.config.num_labels,
            pooler_dropout=self.config.classifier_dropout,
        )

    """
    获取编码器模块的私有方法
    """
    def _get_encoder_module(self):
        return self.model.encoder

    """
    获取解码器模块的私有方法
    """
    def _get_decoder_module(self):
        return self.model.decoder

    """
    定义类实例被调用时的行为
    """
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
        """
        输入序列分类模块的参数：
        input_ids: 输入的 token IDs
        attention_mask: 注意力遮罩
        decoder_input_ids: 解码器的输入 token IDs
        decoder_attention_mask: 解码器的注意力遮罩
        position_ids: 位置 IDs
        decoder_position_ids: 解码器的位置 IDs
        output_attentions: 是否输出注意力权重
        output_hidden_states: 是否输出隐藏状态
        return_dict: 是否返回字典格式的输出
        deterministic: 是否确定性运行
        """

            # 实例方法主体为空，由子类实现具体逻辑
            pass
        ):
            # 调用模型进行推理，获取输出结果
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

            # 获取模型输出中的最后一个隐藏状态
            hidden_states = outputs[0]  # 最后一个隐藏状态

            # 创建一个掩码，标记输入中的 <eos> 位置
            eos_mask = jnp.where(input_ids == self.config.eos_token_id, 1, 0)

            # 处理特定的 JAX 编译错误类型，确保避免 JIT 编译中的错误
            if type(eos_mask) != jax.interpreters.partial_eval.DynamicJaxprTracer:
                # 检查每个示例中 <eos> 标记的数量是否一致
                if len(jnp.unique(eos_mask.sum(1))) > 1:
                    raise ValueError("所有示例必须具有相同数量的 <eos> 标记。")

                # 检查是否有示例缺少 <eos> 标记
                if any(eos_mask.sum(1) == 0):
                    raise ValueError("输入中缺少 <eos> 标记。")

                # 为每个示例保留最后一个 <eos> 标记
                eos_mask_noised = eos_mask + jnp.arange(eos_mask.shape[1]) * 1e-6
                eos_mask = jnp.where(eos_mask_noised == eos_mask_noised.max(1).reshape(-1, 1), 1, 0)

            # 使用 eos_mask 对隐藏状态进行加权求和，以获得句子表示
            sentence_representation = jnp.einsum("ijk, ij -> ijk", hidden_states, eos_mask).sum(1)

            # 将句子表示传递给分类头，获取分类 logits
            logits = self.classification_head(sentence_representation, deterministic=deterministic)

            # 如果不需要返回字典，则返回输出的元组
            if not return_dict:
                output = (logits,) + outputs[1:]
                return output

            # 构造 FlaxSeq2SeqSequenceClassifierOutput 对象，封装模型输出
            return FlaxSeq2SeqSequenceClassifierOutput(
                logits=logits,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )
# 使用自定义的 docstring 添加起始注释给 FlaxBartForSequenceClassification 类，指定其用途和应用场景
@add_start_docstrings(
    """
    Bart model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE
    tasks.
    """,
    BART_START_DOCSTRING,  # 引用预定义的 Bart 模型的起始注释
)
class FlaxBartForSequenceClassification(FlaxBartPreTrainedModel):
    module_class = FlaxBartForSequenceClassificationModule  # 设定模型类
    dtype = jnp.float32  # 设置数据类型


# 向 FlaxBartForSequenceClassification 类添加调用样例的文档字符串
append_call_sample_docstring(
    FlaxBartForSequenceClassification,
    _CHECKPOINT_FOR_DOC,  # 引用检查点文档
    FlaxSeq2SeqSequenceClassifierOutput,  # 引用输出类文档
    _CONFIG_FOR_DOC,  # 引用配置文档
)


# 定义 FlaxBartForQuestionAnsweringModule 类，继承自 nn.Module
class FlaxBartForQuestionAnsweringModule(nn.Module):
    config: BartConfig  # 使用 BartConfig 配置
    dtype: jnp.dtype = jnp.float32  # 设置数据类型为 float32
    num_labels = 2  # 设定标签数量为 2

    def setup(self):
        self.model = FlaxBartModule(config=self.config, dtype=self.dtype)  # 使用配置和数据类型初始化模型
        self.qa_outputs = nn.Dense(  # 定义问题-回答输出层
            self.num_labels,  # 输出层标签数量
            dtype=self.dtype,  # 输出层数据类型
            kernel_init=jax.nn.initializers.normal(self.config.init_std),  # 使用正态分布初始化权重
        )

    def _get_encoder_module(self):
        return self.model.encoder  # 获取编码器模块

    def _get_decoder_module(self):
        return self.model.decoder  # 获取解码器模块

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
        # 调用模型进行正向传播
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

        sequence_output = outputs[0]  # 提取序列输出

        logits = self.qa_outputs(sequence_output)  # 通过问题-回答输出层计算 logits
        start_logits, end_logits = jnp.split(logits, logits.shape[-1], axis=-1)  # 分割 logits 得到起始和结束 logits
        start_logits = start_logits.squeeze(-1)  # 压缩起始 logits 的最后一维
        end_logits = end_logits.squeeze(-1)  # 压缩结束 logits 的最后一维

        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]  # 如果不返回字典，则将输出整合为元组
            return output

        # 返回字典格式的输出
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


# 使用自定义的 docstring 添加起始注释给 FlaxBartForSequenceClassification 类，指定其用途和应用场景
@add_start_docstrings(
    """
    BART Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    """,
    layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    # 创建一个层用于在隐藏状态输出的基础上计算“span起始位置logits”和“span结束位置logits”。
    """,
    BART_START_DOCSTRING,
    # 使用预定义的 BART_START_DOCSTRING 常量作为文档字符串的起始部分
)

# 定义一个类，继承自FlaxBartPreTrainedModel，用于问答任务
class FlaxBartForQuestionAnswering(FlaxBartPreTrainedModel):
    # 模块类设置为FlaxBartForQuestionAnsweringModule
    module_class = FlaxBartForQuestionAnsweringModule
    # 数据类型设置为32位浮点数
    dtype = jnp.float32

# 向FlaxBartForQuestionAnswering类附加一个函数调用样例的文档字符串
append_call_sample_docstring(
    FlaxBartForQuestionAnswering,
    _CHECKPOINT_FOR_DOC,
    FlaxSeq2SeqQuestionAnsweringModelOutput,
    _CONFIG_FOR_DOC,
)

# 定义一个类，继承自FlaxPreTrainedModel，用于BART解码器预训练模型
class FlaxBartDecoderPreTrainedModel(FlaxPreTrainedModel):
    # 配置类设置为BartConfig
    config_class = BartConfig
    # 基础模型前缀设置为"model"
    base_model_prefix: str = "model"
    # 模块类初始化为None
    module_class: nn.Module = None

    def __init__(
        self,
        config: BartConfig,
        input_shape: Tuple[int] = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        # 设置配置为解码器模式
        config.is_decoder = True
        # 设置不是编码器-解码器模式
        config.is_encoder_decoder = False
        # 使用配置和数据类型初始化模块
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 调用父类初始化方法
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)

        # 获取批量大小和序列长度
        batch_size, sequence_length = input_ids.shape
        # 生成位置编码张量
        position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # 分割随机数生成器
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}
        # 初始化编码器隐藏状态和注意力掩码
        encoder_hidden_states = jnp.zeros(input_shape + (self.config.d_model,))
        encoder_attention_mask = attention_mask
        # 调用模块的初始化方法
        module_init_outputs = self.module.init(
            rngs,
            input_ids,
            attention_mask,
            position_ids,
            encoder_hidden_states,
            encoder_attention_mask,
            return_dict=False,
        )
        # 返回模块初始化的参数
        return module_init_outputs["params"]

    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                用于快速自回归解码的批量大小，定义了初始化缓存的批量大小。
            max_length (`int`):
                自回归解码的最大可能长度，定义了初始化缓存的序列长度。
        """
        # 初始化用于检索缓存的输入变量
        input_ids = jnp.ones((batch_size, max_length), dtype="i4")
        attention_mask = jnp.ones_like(input_ids, dtype="i4")
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        # 调用模块的初始化方法，设置init_cache=True以获取缓存
        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        # 解冻并返回初始化变量的缓存部分
        return unfreeze(init_variables["cache"])

    @add_start_docstrings_to_model_forward(BART_DECODE_INPUTS_DOCSTRING)
    # 定义一个特殊方法 __call__，使得对象可以被调用
    def __call__(
        # 参数 input_ids: 接受一个 NumPy 数组，用于输入模型的标识符
        self,
        input_ids: jnp.ndarray,
        # 参数 attention_mask: 可选参数，用于指定哪些标识符需要被注意
        attention_mask: Optional[jnp.ndarray] = None,
        # 参数 position_ids: 可选参数，用于指定输入标识符的位置信息
        position_ids: Optional[jnp.ndarray] = None,
        # 参数 encoder_hidden_states: 可选参数，编码器的隐藏状态
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        # 参数 encoder_attention_mask: 可选参数，编码器的注意力掩码
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        # 参数 output_attentions: 可选参数，指示是否返回注意力权重
        output_attentions: Optional[bool] = None,
        # 参数 output_hidden_states: 可选参数，指示是否返回所有隐藏状态
        output_hidden_states: Optional[bool] = None,
        # 参数 return_dict: 可选参数，指示是否返回结果字典形式
        return_dict: Optional[bool] = None,
        # 参数 train: 布尔类型参数，指示当前是否处于训练模式
        train: bool = False,
        # 参数 params: 字典类型参数，用于存储额外的参数信息
        params: dict = None,
        # 参数 past_key_values: 字典类型参数，用于存储过去的键值信息
        past_key_values: dict = None,
        # 参数 dropout_rng: PRNGKey 类型参数，用于控制 dropout 行为的随机数生成器
        dropout_rng: PRNGKey = None,
        ):
            # 如果 output_attentions 参数未指定，则使用模型配置中的默认值
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            # 如果 output_hidden_states 参数未指定，则使用模型配置中的默认值
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            # 如果 return_dict 参数未指定，则使用模型配置中的默认值
            return_dict = return_dict if return_dict is not None else self.config.return_dict

            # 如果 encoder_hidden_states 存在且未提供 encoder_attention_mask，则创建一个全为 1 的注意力掩码
            if encoder_hidden_states is not None and encoder_attention_mask is None:
                batch_size, sequence_length = encoder_hidden_states.shape[:2]
                encoder_attention_mask = jnp.ones((batch_size, sequence_length))

            # 准备解码器的输入
            # 如果 attention_mask 未提供，则创建一个与 input_ids 形状相同的全为 1 的注意力掩码
            if attention_mask is None:
                attention_mask = jnp.ones_like(input_ids)
            # 如果 position_ids 未提供，则根据 input_ids 的形状创建位置 ID
            if position_ids is None:
                batch_size, sequence_length = input_ids.shape
                position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

            # 处理需要的随机数生成器（PRNG）
            rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

            inputs = {"params": params or self.params}

            # 如果传入了 past_key_values，则将其作为 cache 输入，同时设置 mutable 标志确保 cache 可变
            if past_key_values:
                inputs["cache"] = past_key_values
                mutable = ["cache"]
            else:
                mutable = False

            # 调用模型的 apply 方法，传递各种输入参数
            outputs = self.module.apply(
                inputs,
                input_ids=jnp.array(input_ids, dtype="i4"),
                attention_mask=jnp.array(attention_mask, dtype="i4"),
                position_ids=jnp.array(position_ids, dtype="i4"),
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                deterministic=not train,
                rngs=rngs,
                mutable=mutable,
            )

            # 将更新后的 cache 添加到模型输出中（仅在 return_dict=True 且 past_key_values 不为空时执行）
            if past_key_values is not None and return_dict:
                outputs, past_key_values = outputs
                outputs["past_key_values"] = unfreeze(past_key_values["cache"])
                return outputs
            elif past_key_values is not None and not return_dict:
                outputs, past_key_values = outputs
                # 在输出的第一个元素后添加解冻的 past_key_values["cache"]，用于非字典返回模式
                outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

            # 返回模型的输出
            return outputs
class FlaxBartDecoderWrapper(nn.Module):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when the causal language model is
    used in combination with the [`EncoderDecoderModel`] framework.
    """

    config: BartConfig  # 定义一个成员变量 config，类型为 BartConfig，用于存储模型的配置信息
    dtype: jnp.dtype = jnp.float32  # 定义一个成员变量 dtype，指定数据类型为 jnp.float32，默认值为 jnp.float32

    def setup(self):
        embed_dim = self.config.d_model  # 从 config 中获取模型的 embedding 维度
        embed_tokens = nn.Embed(  # 创建一个嵌入层，用于处理模型的词汇表和 embedding 维度
            self.config.vocab_size,
            embed_dim,
            embedding_init=jax.nn.initializers.normal(self.config.init_std),  # 使用正态分布初始化嵌入层权重
            dtype=self.dtype,
        )
        self.decoder = FlaxBartDecoder(config=self.config, embed_tokens=embed_tokens, dtype=self.dtype)
        # 初始化一个 FlaxBartDecoder 对象，传入配置、嵌入层和数据类型

    def __call__(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)
        # 调用 FlaxBartDecoder 对象的 __call__ 方法，将参数传递给 decoder


class FlaxBartForCausalLMModule(nn.Module):
    config: BartConfig  # 定义一个成员变量 config，类型为 BartConfig，用于存储模型的配置信息
    dtype: jnp.dtype = jnp.float32  # 定义一个成员变量 dtype，指定数据类型为 jnp.float32，默认值为 jnp.float32

    def setup(self):
        self.model = FlaxBartDecoderWrapper(config=self.config, dtype=self.dtype)
        # 初始化一个 FlaxBartDecoderWrapper 对象，传入配置和数据类型
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),  # 使用正态分布初始化 Dense 层的权重
        )

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
        outputs = self.model(  # 调用 self.model 对象，传递所有参数
            input_ids,
            attention_mask,
            position_ids,
            encoder_hidden_states,
            encoder_attention_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]  # 获取模型输出的隐藏状态

        if self.config.tie_word_embeddings:
            shared_embedding = self.model.variables["params"]["decoder"]["embed_tokens"]["embedding"]
            # 如果配置指定共享词嵌入，则从模型的变量中获取共享的嵌入层
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
            # 应用共享的嵌入层权重计算 LM logits
        else:
            lm_logits = self.lm_head(hidden_states)
            # 否则直接计算 LM logits

        if not return_dict:
            return (lm_logits,) + outputs[1:]
            # 如果不返回字典，则返回 LM logits 和其他输出项

        return FlaxCausalLMOutputWithCrossAttentions(
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )
        # 返回带交叉注意力的因果语言模型输出


@add_start_docstrings(
    """
    Bart Decoder Model with a language modeling head on top (linear layer with weights tied to the input embeddings)
    e.g for autoregressive tasks.
    """,
    BART_START_DOCSTRING,
)
class FlaxBartForCausalLM(FlaxBartDecoderPreTrainedModel):
    module_class = FlaxBartForCausalLMModule
    # 定义一个 FlaxBartForCausalLM 类，继承自 FlaxBartDecoderPreTrainedModel，指定模块类为 FlaxBartForCausalLMModule
    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = None):
        # initializing the cache
        # 获取输入张量的批量大小和序列长度
        batch_size, seq_length = input_ids.shape

        # 使用模型的方法初始化缓存，返回过去的键值对
        past_key_values = self.init_cache(batch_size, max_length)
        
        # 注意：通常需要为超出输入长度和缓存长度之外的位置在 attention_mask 中填入 0
        # 但由于解码器使用因果掩码，这些位置已经被掩码了
        # 因此，我们可以在这里创建一个静态的 attention_mask，这样更有效率
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        
        # 如果提供了 attention_mask，则计算位置 ids
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            # 使用 lax.dynamic_update_slice 将 attention_mask 更新到 extended_attention_mask 中
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            # 否则，广播创建位置 ids
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        # 返回准备好的输入字典，包括过去的键值对、扩展的注意力掩码和位置 ids
        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        # 更新生成阶段的输入参数
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs
# 调用函数 append_call_sample_docstring，用于为指定模型和相关对象添加示例文档字符串
append_call_sample_docstring(
    FlaxBartForCausalLM,               # 参数1: FlaxBartForCausalLM 模型类
    _CHECKPOINT_FOR_DOC,               # 参数2: _CHECKPOINT_FOR_DOC 常量，表示检查点
    FlaxCausalLMOutputWithCrossAttentions,  # 参数3: FlaxCausalLMOutputWithCrossAttentions 类，带有跨注意力的输出
    _CONFIG_FOR_DOC,                   # 参数4: _CONFIG_FOR_DOC 常量，表示配置
)
```