# `.\models\opt\modeling_flax_opt.py`

```
# coding=utf-8
# 文件编码设置为 UTF-8

# Copyright 2022 The Fairseq Authors and The Google Flax Team Authors And The HuggingFace Inc. team. All rights reserved.
# 版权声明，声明版权归属和保留的组织或个人

# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache License, Version 2.0 进行许可，允许使用该文件

# you may not use this file except in compliance with the License.
# 除非符合许可证，否则不得使用此文件

# You may obtain a copy of the License at
# 可以在下面链接获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 除非适用法律要求或书面同意，否则本许可下分发的软件是基于"AS IS"的基础分发的，不提供任何明示或暗示的担保或条件
"""
Flax OPT model.
Flax OPT 模型
"""

from functools import partial
# 导入 partial 函数，用于创建偏函数

from typing import Optional, Tuple
# 导入类型提示工具

import flax.linen as nn
# 导入 Flax 的 linen 模块，并命名为 nn

import jax
# 导入 jax 库

import jax.numpy as jnp
# 导入 jax 的 numpy 模块，并命名为 jnp

from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
# 从 flax.core.frozen_dict 导入 FrozenDict、freeze、unfreeze 函数

from flax.linen import combine_masks, make_causal_mask
# 从 flax.linen 导入 combine_masks、make_causal_mask 函数

from flax.linen.attention import dot_product_attention_weights
# 从 flax.linen.attention 导入 dot_product_attention_weights 函数

from flax.traverse_util import flatten_dict, unflatten_dict
# 从 flax.traverse_util 导入 flatten_dict、unflatten_dict 函数

from jax import lax
# 从 jax 库导入 lax 模块

from jax.random import PRNGKey
# 从 jax.random 导入 PRNGKey 类

from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxMaskedLMOutput
# 从 modeling_flax_outputs 模块导入 FlaxBaseModelOutput、FlaxMaskedLMOutput 类

from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring
# 从 modeling_flax_utils 模块导入 ACT2FN、FlaxPreTrainedModel、append_call_sample_docstring 函数

from ...utils import add_start_docstrings, logging
# 从 utils 模块导入 add_start_docstrings、logging 函数

from .configuration_opt import OPTConfig
# 从当前目录下的 configuration_opt 模块导入 OPTConfig 类

logger = logging.get_logger(__name__)
# 使用 logging 模块获取当前模块的日志记录器对象

_CHECKPOINT_FOR_DOC = "facebook/opt-350m"
# 预训练模型的检查点名称，用于文档说明

_CONFIG_FOR_DOC = "OPTConfig"
# 配置类的名称，用于文档说明

OPT_START_DOCSTRING = r"""
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
# OPT_START_DOCSTRING 字符串，包含了 OPT 模型的文档起始部分，提供了继承的类、Flax Linen 的说明以及 JAX 的特性说明
    Parameters:
        config ([`OPTConfig`]): Model configuration class with all the parameters of the model.
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

OPT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# Copied from transformers.models.bart.modeling_flax_bart.FlaxBartAttention with Bart->OPT
class FlaxOPTAttention(nn.Module):
    config: OPTConfig
    embed_dim: int
    num_heads: int
    dropout: float = 0.0
    causal: bool = False
    bias: bool = True
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self) -> None:
        # 计算每个注意力头的维度
        self.head_dim = self.embed_dim // self.num_heads
        # 检查是否能整除，若不能则抛出错误
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )

        # 部分应用全连接层函数，用于创建查询、键、值和输出投影
        dense = partial(
            nn.Dense,
            self.embed_dim,
            use_bias=self.bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )

        # 创建查询、键、值和输出投影层
        self.q_proj, self.k_proj, self.v_proj = dense(), dense(), dense()
        self.out_proj = dense()

        # 创建 dropout 层
        self.dropout_layer = nn.Dropout(rate=self.dropout)

        # 如果是因果注意力，创建因果掩码
        if self.causal:
            self.causal_mask = make_causal_mask(
                jnp.ones((1, self.config.max_position_embeddings), dtype="bool"), dtype="bool"
            )
    # 将隐藏状态按照指定维度重新形状化，以便分离注意力头
    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    # 将隐藏状态按照指定维度重新形状化，以合并注意力头
    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    # 使用 Flax 框架的 @nn.compact 装饰器定义一个方法，用于将单个输入令牌的投影键、值状态与前几步骤的缓存状态连接起来
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slighly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # 检测是否通过缺少现有缓存数据来进行初始化
        is_initialized = self.has_variable("cache", "cached_key")
        # 获取或创建缓存的键和值，并初始化为零矩阵
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        # 获取或创建缓存索引，并初始化为零
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            # 提取批处理维度、最大长度、头数和每个头的深度
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # 使用新的一维空间切片更新键和值缓存
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            # 更新缓存中的键和值
            cached_key.value = key
            cached_value.value = value
            # 更新缓存索引以反映更新的缓存向量数
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # 用于缓存的因果掩码：我们的单个查询位置只应与已生成和缓存的键位置进行关联，而不是剩余的零元素
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            # 组合当前的掩码和输入的注意力掩码
            attention_mask = combine_masks(pad_mask, attention_mask)
        
        # 返回更新后的键、值和注意力掩码
        return key, value, attention_mask
class FlaxOPTDecoderLayer(nn.Module):
    config: OPTConfig  # 定义一个类成员变量 config，类型为 OPTConfig
    dtype: jnp.dtype = jnp.float32  # 定义一个类成员变量 dtype，默认为 jnp.float32

    def setup(self) -> None:
        self.embed_dim = self.config.hidden_size  # 从 config 中获取 hidden_size 并赋给 embed_dim
        self.self_attn = FlaxOPTAttention(  # 初始化 self_attn，使用 FlaxOPTAttention 类
            config=self.config,  # 传入配置参数 config
            embed_dim=self.embed_dim,  # 传入 embed_dim 参数
            num_heads=self.config.num_attention_heads,  # 传入注意力头数
            dropout=self.config.attention_dropout,  # 传入注意力 dropout 率
            causal=True,  # 是否使用因果注意力
            dtype=self.dtype,  # 数据类型为类成员变量 dtype
        )
        self.do_layer_norm_before = self.config.do_layer_norm_before  # 是否在前面进行层归一化
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)  # 初始化 dropout 层
        self.activation_fn = ACT2FN[self.config.activation_function]  # 根据激活函数名称选择对应的激活函数

        self.self_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)  # 初始化自注意力层的 LayerNorm
        self.fc1 = nn.Dense(  # 初始化全连接层 fc1
            self.config.ffn_dim,  # 全连接层的输出维度
            dtype=self.dtype,  # 数据类型为类成员变量 dtype
            kernel_init=jax.nn.initializers.normal(self.config.init_std),  # 使用正态分布初始化权重
        )
        self.fc2 = nn.Dense(  # 初始化全连接层 fc2
            self.embed_dim,  # 全连接层的输出维度为 embed_dim
            dtype=self.dtype,  # 数据类型为类成员变量 dtype
            kernel_init=jax.nn.initializers.normal(self.config.init_std)  # 使用正态分布初始化权重
        )
        self.final_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)  # 初始化最终输出的 LayerNorm

    def __call__(
        self,
        hidden_states: jnp.ndarray,  # 输入的隐藏状态张量
        attention_mask: jnp.ndarray,  # 注意力掩码张量
        init_cache: bool = False,  # 是否初始化缓存
        output_attentions: bool = True,  # 是否输出注意力权重
        deterministic: bool = True,  # 是否使用确定性计算
    ) -> Tuple[jnp.ndarray]:
        residual = hidden_states  # 保存输入的隐藏状态作为残差连接的基础

        # 根据 self.do_layer_norm_before 的值判断是否在注意力机制之前应用层归一化
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # 自注意力机制
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            init_cache=init_cache,
            deterministic=deterministic,
        )
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)  # 应用 dropout

        hidden_states = residual + hidden_states  # 添加残差连接

        # 根据 self.do_layer_norm_before 的值判断是否在注意力机制之后应用层归一化
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # 全连接层
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])  # 将隐藏状态展平

        residual = hidden_states  # 更新残差连接基础

        # 根据 self.do_layer_norm_before 的值判断是否在全连接层之前应用层归一化
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)  # 应用第一个全连接层
        hidden_states = self.activation_fn(hidden_states)  # 应用激活函数

        hidden_states = self.fc2(hidden_states)  # 应用第二个全连接层
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)  # 应用 dropout

        hidden_states = (residual + hidden_states).reshape(hidden_states_shape)  # 添加残差连接并恢复形状

        # 根据 self.do_layer_norm_before 的值判断是否在全连接层之后应用层归一化
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)  # 准备输出结果

        if output_attentions:
            outputs += (self_attn_weights,)  # 如果需要输出注意力权重，则添加到输出中

        return outputs  # 返回模型的输出
class FlaxOPTDecoderLayerCollection(nn.Module):
    config: OPTConfig
    dtype: jnp.dtype = jnp.float32  # 计算时的数据类型

    def setup(self):
        # 创建多个解码器层，并按顺序存储在列表中
        self.layers = [
            FlaxOPTDecoderLayer(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.num_hidden_layers)
        ]
        # 从配置中获取层丢弃率
        self.layerdrop = self.config.layerdrop

    def __call__(
        self,
        hidden_states,
        attention_mask,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        # 如果需要输出隐藏状态，则初始化一个空元组
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力权重，则初始化一个空元组
        all_self_attns = () if output_attentions else None

        # 遍历每个解码器层
        for decoder_layer in self.layers:
            # 如果需要输出隐藏状态，则将当前隐藏状态加入到列表中
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # 调用当前解码器层，获取其输出
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                init_cache=init_cache,
                output_attentions=output_attentions,
                deterministic=deterministic,
            )

            # 更新隐藏状态为当前层的输出
            hidden_states = layer_outputs[0]
            # 如果需要输出注意力权重，则将当前层的注意力权重加入到列表中
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # 组装最终输出，包括最终隐藏状态、所有隐藏状态列表和所有注意力权重列表
        outputs = [hidden_states, all_hidden_states, all_self_attns]
        return outputs


class FlaxOPTLearnedPositionalEmbedding(nn.Embed):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def setup(self):
        # 设置位置偏移量
        self.offset = 2
        # 初始化位置嵌入矩阵参数
        self.embedding = self.param(
            "embedding", self.embedding_init, (self.num_embeddings + self.offset, self.features), self.param_dtype
        )

    def __call__(self, positions):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""

        # 调用父类的 __call__ 方法，并在输入位置上加上偏移量
        return super().__call__(positions + self.offset)


class FlaxOPTDecoder(nn.Module):
    config: OPTConfig
    dtype: jnp.dtype = jnp.float32  # 计算时的数据类型
    offset: int = 2
    # 设置方法用于初始化模型参数和各种配置
    def setup(self):
        # 初始化一个dropout层，用于随机失活以防止过拟合
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)

        # 从配置中获取隐藏层大小作为嵌入维度
        embed_dim = self.config.hidden_size
        # 从配置中获取填充 token 的索引
        self.padding_idx = self.config.pad_token_id
        # 从配置中获取最大目标位置
        self.max_target_positions = self.config.max_position_embeddings

        # 初始化词嵌入层，使用正态分布初始化方法
        self.embed_tokens = nn.Embed(
            self.config.vocab_size,
            self.config.word_embed_proj_dim,
            embedding_init=jax.nn.initializers.normal(self.config.init_std),
            dtype=self.dtype,
        )

        # 初始化学习位置嵌入层，使用正态分布初始化方法
        self.embed_positions = FlaxOPTLearnedPositionalEmbedding(
            self.config.max_position_embeddings,
            embed_dim,
            embedding_init=jax.nn.initializers.normal(self.config.init_std),
            dtype=self.dtype,
        )

        # 如果词嵌入投影维度不等于隐藏层大小，则初始化投影层
        if self.config.word_embed_proj_dim != self.config.hidden_size:
            self.project_in = nn.Dense(self.config.hidden_size, use_bias=False)
            self.project_out = nn.Dense(self.config.word_embed_proj_dim, use_bias=False)
        else:
            # 否则将投影层设置为 None
            self.project_in = None
            self.project_out = None

        # 检查是否需要在最后一层使用 LayerNorm，主要是为了向后兼容
        if self.config.do_layer_norm_before and not self.config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        else:
            # 如果不需要 LayerNorm 则将其设置为 None
            self.final_layer_norm = None

        # 初始化解码器层集合
        self.layers = FlaxOPTDecoderLayerCollection(self.config, self.dtype)

    # 模型调用方法，用于执行模型的前向传播
    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
        # 其他参数用于控制模型的行为，如是否输出注意力矩阵、隐藏状态等
        ):
        # 获取输入的张量形状
        input_shape = input_ids.shape
        # 将输入张量展平为二维张量
        input_ids = input_ids.reshape(-1, input_shape[-1])

        # 使用嵌入标记方法对输入张量进行嵌入
        inputs_embeds = self.embed_tokens(input_ids)
        # 如果存在输入投影层，则将嵌入结果投影
        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        # 使用嵌入位置方法生成位置嵌入张量
        positions = self.embed_positions(position_ids)

        # 将嵌入的输入张量和位置嵌入张量相加以得到隐藏状态张量
        hidden_states = inputs_embeds + positions

        # 调用多层模型的前向传播方法，获取隐藏状态、所有隐藏状态和注意力张量
        hidden_state, all_hidden_states, attentions = self.layers(
            hidden_states,
            attention_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # 如果存在最终层归一化，则对隐藏状态进行归一化
        if self.final_layer_norm is not None:
            hidden_state = self.final_layer_norm(hidden_state)

        # 如果存在输出投影层，则对隐藏状态进行投影
        if self.project_out is not None:
            hidden_state = self.project_out(hidden_state)

        # 如果要求输出所有隐藏状态，则将当前隐藏状态加入到所有隐藏状态列表中
        if output_hidden_states:
            all_hidden_states += (hidden_state,)

        # 根据返回值是否为字典形式，决定返回元组还是命名元组形式的输出
        outputs = [hidden_state, all_hidden_states, attentions]

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        # 返回命名元组形式的输出
        return FlaxBaseModelOutput(
            last_hidden_state=hidden_state,
            hidden_states=all_hidden_states,
            attentions=attentions,
        )
    # 定义一个继承自FlaxPreTrainedModel的类，用于OPT模型的预训练。
    class FlaxOPTPreTrainedModel(FlaxPreTrainedModel):
        # 指定配置类为OPTConfig
        config_class = OPTConfig
        # 指定基础模型前缀为"model"
        base_model_prefix: str = "model"
        # 模块类初始化为None
        module_class: nn.Module = None

        # 初始化函数，接受配置config、输入形状input_shape、种子seed、数据类型dtype等参数
        def __init__(
            self,
            config: OPTConfig,
            input_shape: Tuple[int] = (1, 1),
            seed: int = 0,
            dtype: jnp.dtype = jnp.float32,
            _do_init: bool = True,
            **kwargs,
        ):
            # 使用module_class创建模块对象module，传入config和其他kwargs参数
            module = self.module_class(config=config, dtype=dtype, **kwargs)
            # 调用父类初始化方法，传入config、module、input_shape、seed、dtype、_do_init等参数
            super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

        # 初始化权重函数，接受随机数生成器rng、输入形状input_shape、参数params等参数，返回初始化后的参数params
        def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
            # 初始化input_ids为全零数组，数据类型为"i4"
            input_ids = jnp.zeros(input_shape, dtype="i4")
            # 初始化attention_mask为与input_ids形状相同的全1数组
            attention_mask = jnp.ones_like(input_ids)

            # 获取batch_size和sequence_length
            batch_size, sequence_length = input_ids.shape
            # 初始化position_ids为广播形式的序列长度数组
            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

            # 拆分rng生成params_rng和dropout_rng
            params_rng, dropout_rng = jax.random.split(rng)
            # 构建随机数字典rngs，包含params_rng和dropout_rng
            rngs = {"params": params_rng, "dropout": dropout_rng}

            # 使用module的init方法初始化模型参数
            module_init_outputs = self.module.init(
                rngs,
                input_ids,
                attention_mask,
                position_ids,
                return_dict=False,
            )

            # 获取随机初始化的模型参数random_params
            random_params = module_init_outputs["params"]
            # 如果params不为None，则将随机参数和给定参数params进行扁平化处理并填充缺失键
            if params is not None:
                random_params = flatten_dict(unfreeze(random_params))
                params = flatten_dict(unfreeze(params))
                for missing_key in self._missing_keys:
                    params[missing_key] = random_params[missing_key]
                self._missing_keys = set()
                return freeze(unflatten_dict(params))
            else:
                return random_params

        # 初始化缓存函数，用于快速自回归解码
        def init_cache(self, batch_size, max_length):
            r"""
            Args:
                batch_size (`int`):
                    用于快速自回归解码的批量大小。定义了初始化缓存的批处理大小。
                max_length (`int`):
                    自动回归解码的最大可能长度。定义了初始化缓存的序列长度。
            """
            # 初始化input_ids为全1数组，形状为(batch_size, max_length)，数据类型为"i4"
            input_ids = jnp.ones((batch_size, max_length), dtype="i4")
            # 初始化attention_mask为与input_ids形状相同的全1数组，数据类型为"i4"
            attention_mask = jnp.ones_like(input_ids, dtype="i4")
            # 初始化position_ids为广播形式的input_ids的序列长度数组
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

            # 使用module的init方法初始化模型变量，设置init_cache为True以初始化缓存
            init_variables = self.module.init(
                jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
            )
            # 返回解除冻结后的缓存变量
            return unfreeze(init_variables["cache"])
    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        params: dict = None,
        past_key_values: dict = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        dropout_rng: PRNGKey = None,
        deterministic: bool = True,
    ):
        # 设置输出注意力机制的标志，如果未指定，则使用配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置输出隐藏状态的标志，如果未指定，则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置返回字典的标志，如果未指定，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 如果未提供注意力掩码，则创建一个全为1的掩码
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        # 如果未提供位置编码，则根据注意力掩码累积的结果生成位置编码
        if position_ids is None:
            position_ids = (attention_mask.cumsum(axis=1) * attention_mask) - 1

        # 处理可能需要的任何伪随机数生成器
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        # 准备模型输入字典
        inputs = {"params": params or self.params}

        # 如果提供了过去的键值对，则将其缓存放入输入中，并标记为可变
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        # 应用模型的前向传播
        outputs = self.module.apply(
            inputs,
            input_ids=jnp.array(input_ids, dtype="i4"),
            attention_mask=jnp.array(attention_mask, dtype="i4"),
            position_ids=jnp.array(position_ids, dtype="i4"),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
            rngs=rngs,
            mutable=mutable,
        )

        # 如果同时传递了过去的键值对和return_dict为True，则将更新后的缓存添加到模型输出中
        if past_key_values is not None and return_dict:
            outputs, past_key_values = outputs
            outputs["past_key_values"] = unfreeze(past_key_values["cache"])
            return outputs
        # 如果同时传递了过去的键值对和return_dict为False，则将更新后的缓存插入到模型输出的适当位置
        elif past_key_values is not None and not return_dict:
            outputs, past_key_values = outputs
            outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

        # 返回模型的输出结果
        return outputs
class FlaxOPTModule(nn.Module):
    config: OPTConfig
    dtype: jnp.dtype = jnp.float32  # 计算时使用的数据类型

    def setup(self):
        # 初始化解码器对象，使用给定的配置和数据类型
        self.decoder = FlaxOPTDecoder(self.config, dtype=self.dtype)

    def _get_decoder_module(self):
        return self.decoder

    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
        init_cache=False,
    ):
        # 调用解码器对象进行前向传播
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
            init_cache=init_cache,
        )

        if not return_dict:
            return decoder_outputs

        # 返回经过模型输出的结果，作为 FlaxBaseModelOutput 对象
        return FlaxBaseModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        )


# 从 transformers.models.bart.modeling_flax_bart.FlaxBartModel 复制而来，将 Bart 换成 OPT
class FlaxOPTModel(FlaxOPTPreTrainedModel):
    config: OPTConfig
    dtype: jnp.dtype = jnp.float32  # 计算时使用的数据类型
    module_class = FlaxOPTModule


# 添加函数签名的示例文档到 FlaxOPTModel 类中
append_call_sample_docstring(FlaxOPTModel, _CHECKPOINT_FOR_DOC, FlaxBaseModelOutput, _CONFIG_FOR_DOC)


@add_start_docstrings(
    "The bare OPT Model transformer outputting raw hidden-states without any specific head on top.",
    OPT_START_DOCSTRING,
)
class FlaxOPTForCausalLMModule(nn.Module):
    config: OPTConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 初始化 OPT 模型和语言模型头部
        self.model = FlaxOPTModule(config=self.config, dtype=self.dtype)
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )

    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
    ):
        # 调用模型进行前向传播
        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        # 如果不要求返回字典形式的输出，直接返回模型的输出
        if not return_dict:
            return model_outputs

        # 否则，返回 FlaxBaseModelOutput 对象，其中包含模型的隐藏状态、注意力等信息
        return FlaxBaseModelOutput(
            last_hidden_state=model_outputs.last_hidden_state,
            hidden_states=model_outputs.hidden_states,
            attentions=model_outputs.attentions,
        )
    ):
        # 调用模型进行推理
        outputs = self.model(
            input_ids,
            attention_mask,
            position_ids,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        # 从模型输出中获取隐藏状态
        hidden_states = outputs[0]

        # 如果配置要求共享词嵌入，则使用decoder的嵌入矩阵作为共享的嵌入
        if self.config.tie_word_embeddings:
            shared_embedding = self.model.variables["params"]["decoder"]["embed_tokens"]["embedding"]
            # 应用共享的词嵌入到隐藏状态得到语言模型的logits
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
        else:
            # 否则直接用语言模型头部处理隐藏状态得到logits
            lm_logits = self.lm_head(hidden_states)

        # 如果不要求返回字典形式的输出，则返回tuple形式的结果
        if not return_dict:
            return (lm_logits,) + outputs[1:]

        # 返回FlaxMaskedLMOutput对象，其中包含logits、隐藏状态和注意力权重
        return FlaxMaskedLMOutput(
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
"""
OPT Model with a language modeling head on top (linear layer with weights tied to the input embeddings) e.g for
autoregressive tasks.
"""
@add_start_docstrings(
    """
    OPT Model with a language modeling head on top (linear layer with weights tied to the input embeddings) e.g for
    autoregressive tasks.
    """,
    OPT_START_DOCSTRING,
)
class FlaxOPTForCausalLM(FlaxOPTPreTrainedModel):
    # 使用 FlaxOPTForCausalLMModule 作为模块类
    module_class = FlaxOPTForCausalLMModule

    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = None):
        # initializing the cache
        batch_size, seq_length = input_ids.shape

        # 初始化缓存，准备用于生成
        past_key_values = self.init_cache(batch_size, max_length)

        # 由于解码器使用因果掩码，attention_mask 通常只需要在 input_ids.shape[-1] 之外和 cache_length 之前的位置放置 0，
        # 但这些位置因为因果掩码而已经被屏蔽了。因此，我们可以在这里创建一个静态的 attention_mask，这样更有效率地进行编译。
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")

        if attention_mask is not None:
            # 计算位置 ids
            position_ids = attention_mask.cumsum(axis=1) - 1
            # 更新动态切片的 extended_attention_mask
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            # 如果没有传入 attention_mask，则广播生成位置 ids
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        # 更新生成过程中的输入参数，更新 past_key_values 和 position_ids
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs


# 向类添加调用示例文档字符串
append_call_sample_docstring(
    FlaxOPTForCausalLM,
    _CHECKPOINT_FOR_DOC,
    FlaxBaseModelOutput,
    _CONFIG_FOR_DOC,
)
```