# `.\models\xglm\modeling_flax_xglm.py`

```py
# coding=utf-8
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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
""" Flax XGLM model."""

# 导入需要的库和模块
import math
import random
from functools import partial
from typing import Optional, Tuple

import flax.linen as nn  # 导入Flax的linen模块，通常用来定义神经网络模型
import jax  # 导入JAX，用于自动求导和数组计算
import jax.numpy as jnp  # 导入JAX的NumPy接口，用于数组操作
import numpy as np  # 导入NumPy，通用的数值计算库
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze  # 导入Flax的FrozenDict，用于不可变字典的操作
from flax.linen import combine_masks, make_causal_mask  # 导入Flax的函数和类
from flax.linen.attention import dot_product_attention_weights  # 导入Flax的注意力机制函数
from flax.traverse_util import flatten_dict, unflatten_dict  # 导入Flax的工具函数，用于字典扁平化和反扁平化
from jax import lax  # 导入JAX的lax模块，用于定义和执行JAX原语
from jax.random import PRNGKey  # 导入JAX的随机数生成器

from ...modeling_flax_outputs import (
    FlaxBaseModelOutputWithPastAndCrossAttentions,
    FlaxCausalLMOutputWithCrossAttentions,
)  # 导入自定义的Flax模型输出类
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring  # 导入自定义的Flax模型和工具函数
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging  # 导入自定义的工具函数和日志模块
from .configuration_xglm import XGLMConfig  # 导入XGLM模型的配置文件

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

_CHECKPOINT_FOR_DOC = "facebook/xglm-564M"  # 文档中的预训练模型名
_CONFIG_FOR_DOC = "XGLMConfig"  # 文档中的配置文件名

XGLM_START_DOCSTRING = r"""
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
    # 定义一个函数，接受以下参数:
    #   config (`XGLMConfig`)：包含模型所有参数的配置类。
    #       使用配置文件初始化不会加载与模型关联的权重，只加载配置。
    #       可以查看 [`~FlaxPreTrainedModel.from_pretrained`] 方法来加载模型权重。
    #   dtype (`jax.numpy.dtype`, *可选*, 默认为 `jax.numpy.float32`)：
    #       计算的数据类型。可以是 `jax.numpy.float32`、`jax.numpy.float16`（在GPU上）、`jax.numpy.bfloat16`（在TPU上）之一。
    #
    #       这可以用于在GPU或TPU上启用混合精度训练或半精度推断。如果指定，则所有计算将使用给定的 `dtype` 执行。
    #
    #       **注意，这仅指定计算的dtype，不影响模型参数的dtype。**
    #
    #       如果要更改模型参数的dtype，请参阅 [`~FlaxPreTrainedModel.to_fp16`] 和 [`~FlaxPreTrainedModel.to_bf16`]。
"""
XGLM_INPUTS_DOCSTRING = r"""
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


def create_sinusoidal_positions(n_pos, dim, padding_idx=1):
    # Calculate half of the dimension for sinusoidal embedding
    half_dim = dim // 2
    # Compute the exponential term for sinusoidal embedding
    emb = math.log(10000) / (half_dim - 1)
    emb = np.exp(np.arange(half_dim) * -emb)
    # Expand dimensions to perform element-wise multiplication
    emb = np.expand_dims(np.arange(n_pos), 1) * np.expand_dims(emb, 0)
    # Concatenate sine and cosine transformations of embeddings
    emb = np.concatenate([np.sin(emb), np.cos(emb)], 1)
    # Reshape the embedding to match desired dimensions
    emb = np.reshape(emb, (n_pos, dim))

    # If padding index is specified, zero out its embedding
    if padding_idx is not None:
        emb[padding_idx, :] = 0

    # Convert embedding to JAX array
    return jnp.array(emb)


class FlaxXGLMAttention(nn.Module):
    config: XGLMConfig
    embed_dim: int
    num_heads: int
    dropout: float = 0.0
    causal: bool = False
    bias: bool = True
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    def setup(self) -> None:
        # 计算每个头部的维度
        self.head_dim = self.embed_dim // self.num_heads

        # 检查 embed_dim 是否能被 num_heads 整除，否则抛出数值错误
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} "
                f"and `num_heads`: {self.num_heads})."
            )

        # 定义部分应用了部分参数的 Dense 层构造函数
        dense = partial(
            nn.Dense,
            self.embed_dim,
            use_bias=self.bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )

        # 初始化查询、键、值、输出投影层
        self.q_proj, self.k_proj, self.v_proj = dense(), dense(), dense()
        self.out_proj = dense()

        # 初始化 Dropout 层
        self.dropout_layer = nn.Dropout(rate=self.dropout)

        # 如果需要引入因果注意力机制，则创建对应的因果掩码
        if self.causal:
            self.causal_mask = make_causal_mask(
                jnp.ones((1, self.config.max_position_embeddings), dtype="bool"), dtype="bool"
            )

    def _split_heads(self, hidden_states):
        # 将隐藏状态张量按头部分割
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        # 将分割后的头部重新合并
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slighly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # 检测是否正在初始化，通过检查是否存在缓存数据来判断
        is_initialized = self.has_variable("cache", "cached_key")
        # 获取或创建缓存的键值对应的变量，如果不存在则创建一个全零数组
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        # 获取或创建缓存的值对应的变量，如果不存在则创建一个全零数组
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        # 获取或创建缓存索引对应的变量，如果不存在则创建一个值为0的整数数组
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            # 提取当前缓存的维度信息，包括批次维度、最大长度、头数、每头深度
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # 使用新的一维空间切片更新键和值的缓存
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            # 更新缓存中的键和值
            cached_key.value = key
            cached_value.value = value
            # 更新缓存索引，增加已更新的缓存向量数目
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # 用于缓存的因果掩码：我们的单个查询位置应该只关注已生成和缓存的键位置，而不是剩余的零元素。
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            # 将因果掩码和传入的注意力掩码结合起来
            attention_mask = combine_masks(pad_mask, attention_mask)
        # 返回更新后的键、值和注意力掩码
        return key, value, attention_mask
    # 定义一个 FlaxXGLMDecoderLayer 类，继承自 nn.Module
    class FlaxXGLMDecoderLayer(nn.Module):
        # 类变量：XGLMConfig 类型的 config 变量
        config: XGLMConfig
        # 类变量：jnp.float32 类型的 dtype，默认为 jnp.float32
        dtype: jnp.dtype = jnp.float32

        # 初始化方法，无返回值
        def setup(self) -> None:
            # 实例变量：self.embed_dim 等于 config.d_model
            self.embed_dim = self.config.d_model
            # 实例变量：self.self_attn 是一个 FlaxXGLMAttention 实例
            # 根据给定的 config 参数进行初始化
            self.self_attn = FlaxXGLMAttention(
                config=self.config,
                embed_dim=self.embed_dim,
                num_heads=self.config.attention_heads,
                dropout=self.config.attention_dropout,
                causal=True,
                dtype=self.dtype,
            )
            # 实例变量：self.self_attn_layer_norm 是一个 LayerNorm 实例
            # 根据 dtype 参数进行初始化
            self.self_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
            # 实例变量：self.dropout_layer 是一个 Dropout 层实例
            # 根据 config.dropout 参数进行初始化
            self.dropout_layer = nn.Dropout(rate=self.config.dropout)
            # 实例变量：self.activation_fn 是一个激活函数，根据 config.activation_function 选择
            self.activation_fn = ACT2FN[self.config.activation_function]
            # 实例变量：self.activation_dropout_layer 是一个 Dropout 层实例
            # 根据 config.activation_dropout 参数进行初始化
            self.activation_dropout_layer = nn.Dropout(rate=self.config.activation_dropout)

            # 如果 config.add_cross_attention 为 True，则初始化下面的变量
            if self.config.add_cross_attention:
                # 实例变量：self.encoder_attn 是一个 FlaxXGLMAttention 实例
                # 根据给定的 config 参数进行初始化
                self.encoder_attn = FlaxXGLMAttention(
                    config=self.config,
                    embed_dim=self.embed_dim,
                    num_heads=self.config.decoder_attention_heads,
                    dropout=self.config.attention_dropout,
                    dtype=self.dtype,
                )
                # 实例变量：self.encoder_attn_layer_norm 是一个 LayerNorm 实例
                # 根据 dtype 参数进行初始化
                self.encoder_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

            # 实例变量：self.fc1 是一个全连接层实例
            # 输入维度为 self.config.ffn_dim，输出维度为 self.embed_dim
            # 根据 dtype 参数和 self.config.init_std 进行初始化
            self.fc1 = nn.Dense(
                self.config.ffn_dim,
                dtype=self.dtype,
                kernel_init=jax.nn.initializers.normal(self.config.init_std),
            )
            # 实例变量：self.fc2 是一个全连接层实例
            # 输入维度为 self.embed_dim，输出维度为 self.embed_dim
            # 根据 dtype 参数和 self.config.init_std 进行初始化
            self.fc2 = nn.Dense(
                self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std)
            )
            # 实例变量：self.final_layer_norm 是一个 LayerNorm 实例
            # 根据 dtype 参数进行初始化
            self.final_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

        # 重写 __call__ 方法，用于实例调用时的行为
        # 可以接收多种输入参数并处理
        # 来自 transformers.models.mbart.modeling_flax_mbart.FlaxMBartDecoderLayer.__call__
        def __call__(
            self,
            hidden_states: jnp.ndarray,  # 输入的隐藏状态，类型为 jnp.ndarray
            attention_mask: jnp.ndarray,  # 注意力掩码，类型为 jnp.ndarray
            encoder_hidden_states: Optional[jnp.ndarray] = None,  # 编码器的隐藏状态，可选参数，默认为 None
            encoder_attention_mask: Optional[jnp.ndarray] = None,  # 编码器的注意力掩码，可选参数，默认为 None
            init_cache: bool = False,  # 是否初始化缓存，类型为布尔值，默认为 False
            output_attentions: bool = True,  # 是否输出注意力权重，类型为布尔值，默认为 True
            deterministic: bool = True,  # 是否确定性计算，类型为布尔值，默认为 True
            # 返回值类型为 Tuple[jnp.ndarray, Optional[jnp.ndarray]]
            # 其中第一个元素为输出的隐藏状态，第二个元素为注意力权重，可选
            ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        ) -> Tuple[jnp.ndarray]:
        # 保存残差连接（Residual Connection）的输入隐藏状态
        residual = hidden_states
        # 应用自注意力机制前的层归一化
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 自注意力机制
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, init_cache=init_cache
        )
        # 应用 dropout
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 添加残差连接
        hidden_states = residual + hidden_states

        # 交叉注意力块
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            # 保存残差连接
            residual = hidden_states

            # 应用编码器注意力块前的层归一化
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            # 应用编码器注意力机制
            hidden_states, cross_attn_weights = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            # 应用 dropout
            hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
            # 添加残差连接
            hidden_states = residual + hidden_states

        # 全连接层
        residual = hidden_states
        # 应用最终层归一化
        hidden_states = self.final_layer_norm(hidden_states)
        # 应用激活函数
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 应用激活函数后的 dropout
        hidden_states = self.activation_dropout_layer(hidden_states, deterministic=deterministic)
        # 应用最后的线性变换
        hidden_states = self.fc2(hidden_states)
        # 应用 dropout
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 添加残差连接
        hidden_states = residual + hidden_states

        # 准备输出
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则添加到输出中
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs
class FlaxXGLMDecoderLayerCollection(nn.Module):
    config: XGLMConfig
    dtype: jnp.dtype = jnp.float32  # 计算时使用的数据类型

    def setup(self):
        # 初始化所有的解码器层，并根据配置添加到层列表中
        self.layers = [
            FlaxXGLMDecoderLayer(self.config, name=str(i), dtype=self.dtype) for i in range(self.config.num_layers)
        ]
        # 设置层间隔概率（LayerDrop）
        self.layerdrop = self.config.layerdrop

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
        # 如果需要输出隐藏状态，则初始化存储所有隐藏状态的元组
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力权重，则初始化存储所有自注意力权重的元组
        all_self_attns = () if output_attentions else None
        # 如果需要输出交叉注意力权重且存在编码器隐藏状态，则初始化存储所有交叉注意力权重的元组
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        # 遍历所有解码器层
        for decoder_layer in self.layers:
            if output_hidden_states:
                # 如果需要输出隐藏状态，则将当前隐藏状态添加到存储所有隐藏状态的元组中
                all_hidden_states += (hidden_states,)
                # 添加层间隔概率（LayerDrop），详见论文 https://arxiv.org/abs/1909.11556
            dropout_probability = random.uniform(0, 1)
            if not deterministic and (dropout_probability < self.layerdrop):
                # 如果不是确定性计算且随机丢弃概率小于层间隔概率，则设置层输出为None
                layer_outputs = (None, None, None)
            else:
                # 否则，调用当前解码器层进行前向计算
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    init_cache=init_cache,
                    output_attentions=output_attentions,
                    deterministic=deterministic,
                )

            # 更新当前隐藏状态为解码器层的输出的第一个元素
            hidden_states = layer_outputs[0]
            if output_attentions:
                # 如果需要输出注意力权重，则将当前解码器层的自注意力权重添加到存储所有自注意力权重的元组中
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    # 如果存在编码器隐藏状态，则将当前解码器层的交叉注意力权重添加到存储所有交叉注意力权重的元组中
                    all_cross_attentions += (layer_outputs[2],)

        # 添加来自最后一个解码器层的隐藏状态
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 构建模型输出，根据需要返回不同的数据结构
        outputs = (hidden_states, all_hidden_states, all_self_attns, all_cross_attentions)

        if not return_dict:
            # 如果不需要返回字典形式的输出，则只返回非空的元组元素
            return tuple(v for v in outputs if v is not None)

        # 否则，返回包含各类注意力权重和隐藏状态的字典形式的输出
        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class FlaxXGLMModule(nn.Module):
    config: XGLMConfig
    dtype: jnp.dtype = jnp.float32  # 计算时使用的数据类型
    # 设置模型的初始配置
    def setup(self):
        # 初始化 dropout 层
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)

        # 获取嵌入维度、填充索引、最大目标位置和嵌入缩放因子的配置信息
        embed_dim = self.config.d_model
        self.padding_idx = self.config.pad_token_id
        self.max_target_positions = self.config.max_position_embeddings
        self.embed_scale = math.sqrt(self.config.d_model) if self.config.scale_embedding else 1.0

        # 创建词嵌入矩阵，指定词汇表大小和嵌入维度，使用正态分布初始化
        self.embed_tokens = nn.Embed(
            self.config.vocab_size,
            embed_dim,
            embedding_init=jax.nn.initializers.normal(self.config.init_std),
        )

        # XGLM 模型的特殊设置：如果指定了填充索引，将嵌入 id 偏移 2，并相应调整 num_embeddings
        # 其他模型不需要此调整
        self.offset = 2
        # 创建 sinusoidal 位置嵌入，考虑偏移量和嵌入维度
        self.embed_positions = create_sinusoidal_positions(
            self.config.max_position_embeddings + self.offset, embed_dim
        )
        
        # 初始化 XGLM 解码器层集合
        self.layers = FlaxXGLMDecoderLayerCollection(self.config, self.dtype)
        # 初始化 LayerNorm 层，设置类型和 epsilon 值
        self.layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    # 定义模型调用方法
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
            # 将输入张量重新整形为二维张量，保留最后一个维度不变
            input_ids = input_ids.reshape(-1, input_shape[-1])

            # 使用模型的词嵌入层对输入张量进行嵌入，并乘以嵌入缩放因子
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

            # 嵌入位置信息
            position_ids = position_ids + self.offset
            positions = jnp.take(self.embed_positions, position_ids, axis=0)

            # 将词嵌入和位置嵌入相加得到隐藏状态
            hidden_states = inputs_embeds + positions
            # 使用 dropout 层对隐藏状态进行处理，根据 deterministic 参数确定是否使用确定性的 dropout
            hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)

            # 将隐藏状态传入模型的层中进行处理
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

            # 获取模型输出中的最后一个隐藏状态，并进行层归一化处理
            last_hidden_states = outputs[0]
            last_hidden_states = self.layer_norm(last_hidden_states)

            hidden_states = None
            # 如果需要输出所有隐藏状态，则将其从模型输出中提取并添加最后一个隐藏状态
            if output_hidden_states:
                hidden_states = outputs[1]
                hidden_states = hidden_states[:-1] + (last_hidden_states,)

            # 根据 return_dict 决定如何返回模型输出
            if not return_dict:
                # 如果不需要返回字典形式的结果，则根据需要组合输出
                outputs = (last_hidden_states, hidden_states) + (outputs[2:] if output_hidden_states else outputs[1:])
                # 过滤掉空值并返回元组形式的结果
                return tuple(v for v in outputs if v is not None)

            # 如果需要返回字典形式的结果，则构建 FlaxBaseModelOutputWithPastAndCrossAttentions 对象
            return FlaxBaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=last_hidden_states,
                hidden_states=hidden_states,
                attentions=outputs.attentions,
                cross_attentions=outputs.cross_attentions,
            )
    # 定义 FlaxXGLMPreTrainedModel 类，继承自 FlaxPreTrainedModel 类
    class FlaxXGLMPreTrainedModel(FlaxPreTrainedModel):
        # 指定配置类为 XGLMConfig
        config_class = XGLMConfig
        # 指定基础模型前缀为 "model"
        base_model_prefix: str = "model"
        # 模块类默认为空
        module_class: nn.Module = None

        # 初始化方法，接受配置、输入形状、种子、数据类型等参数
        def __init__(
            self,
            config: XGLMConfig,
            input_shape: Tuple[int] = (1, 1),
            seed: int = 0,
            dtype: jnp.dtype = jnp.float32,
            _do_init: bool = True,
            **kwargs,
        ):
            # 使用模块类和其他参数初始化模块
            module = self.module_class(config=config, dtype=dtype, **kwargs)
            # 调用父类的初始化方法，传入配置、模块、输入形状、种子、数据类型等参数
            super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

        # 初始化权重方法，接受随机数种子、输入形状和参数字典等参数
        def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
            # 初始化输入张量
            input_ids = jnp.zeros(input_shape, dtype="i4")
            # 创建与 input_ids 类型相同的全1张量作为 attention_mask
            attention_mask = jnp.ones_like(input_ids)
            # 根据 input_ids 的形状广播生成位置编码张量
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
            # 切分随机数种子为 params_rng 和 dropout_rng
            params_rng, dropout_rng = jax.random.split(rng)
            # 创建随机数字典 rngs，用于参数和 dropout
            rngs = {"params": params_rng, "dropout": dropout_rng}

            # 如果配置中包含跨注意力机制
            if self.config.add_cross_attention:
                # 创建与 input_shape 和配置的嵌入维度大小相同的全0隐藏状态张量
                encoder_hidden_states = jnp.zeros(input_shape + (self.config.n_embd,))
                # 将 attention_mask 用作编码器的注意力掩码
                encoder_attention_mask = attention_mask
                # 使用模块的初始化方法进行初始化，传入随机数字典、input_ids、attention_mask、position_ids、隐藏状态张量及其注意力掩码
                module_init_outputs = self.module.init(
                    rngs,
                    input_ids,
                    attention_mask,
                    position_ids,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    return_dict=False,
                )
            else:
                # 否则，只使用 input_ids、attention_mask、position_ids 进行模块的初始化
                module_init_outputs = self.module.init(rngs, input_ids, attention_mask, position_ids, return_dict=False)

            # 获取随机初始化的模型参数
            random_params = module_init_outputs["params"]

            # 如果提供了预定义的参数，将随机参数与已有参数进行合并
            if params is not None:
                # 展平并解冻随机参数和已有参数
                random_params = flatten_dict(unfreeze(random_params))
                params = flatten_dict(unfreeze(params))
                # 将随机参数中缺失的键加入已有参数中
                for missing_key in self._missing_keys:
                    params[missing_key] = random_params[missing_key]
                self._missing_keys = set()
                # 冻结并重新构造参数字典
                return freeze(unflatten_dict(params))
            else:
                # 否则，直接返回随机初始化的参数
                return random_params

        # 初始化缓存方法，用于快速自回归解码
        def init_cache(self, batch_size, max_length):
            """
            Args:
                batch_size (`int`):
                    用于快速自回归解码的批处理大小。定义初始化缓存的批处理大小。
                max_length (`int`):
                    自回归解码的最大可能长度。定义初始化缓存的序列长度。
            """
            # 初始化用于检索缓存的输入变量
            input_ids = jnp.ones((batch_size, max_length), dtype="i4")
            # 创建与 input_ids 类型相同的全1张量作为 attention_mask
            attention_mask = jnp.ones_like(input_ids, dtype="i4")
            # 根据 input_ids 的形状广播生成位置编码张量
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

            # 使用模块的初始化方法初始化变量，包括 input_ids、attention_mask、position_ids，并请求返回缓存
            init_variables = self.module.init(
                jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
            )
            # 返回解冻后的初始化缓存
            return unfreeze(init_variables["cache"])
    # 将模型的前向传播方法装饰为添加文档字符串，用于模型输入参数的说明
    @add_start_docstrings_to_model_forward(XGLM_INPUTS_DOCSTRING)
    # 定义模型的调用方法，接受多个参数作为输入
    def __call__(
        self,
        input_ids: jnp.ndarray,  # 输入的token IDs，作为模型的输入
        attention_mask: Optional[jnp.ndarray] = None,  # 可选的注意力掩码，指示哪些token需要注意
        position_ids: Optional[jnp.ndarray] = None,  # 可选的位置IDs，用于指示token的位置信息
        encoder_hidden_states: Optional[jnp.ndarray] = None,  # 可选的编码器隐藏状态，用于encoder-decoder模型
        encoder_attention_mask: Optional[jnp.ndarray] = None,  # 可选的编码器注意力掩码
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出所有层的隐藏状态
        return_dict: Optional[bool] = None,  # 是否以字典形式返回结果
        train: bool = False,  # 是否处于训练模式
        params: dict = None,  # 模型参数字典
        past_key_values: dict = None,  # 过去的键值，用于存储前一次的状态信息
        dropout_rng: PRNGKey = None,  # 随机数生成器，用于Dropout层的随机掩码
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if encoder_hidden_states is not None and encoder_attention_mask is None:
            batch_size, sequence_length = encoder_hidden_states.shape[:2]
            encoder_attention_mask = jnp.ones((batch_size, sequence_length))

        # 准备编码器的输入
        # 如果 attention_mask 为空，则使用与 input_ids 相同形状的全 1 数组
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        # 如果 position_ids 为空，则广播形状为 (batch_size, sequence_length) 的序列长度数组
        if position_ids is None:
            batch_size, sequence_length = input_ids.shape
            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # 如果需要处理任何伪随机数生成器 (PRNG)，则构建相应的字典
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        inputs = {"params": params or self.params}

        # 如果 past_key_values 被传递，则初始化了缓存，并传递一个私有标志 init_cache 以确保使用缓存。
        # 必须确保缓存被标记为可变，以便 FlaxXGLMAttention 模块可以更改它。
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        # 调用模块的 apply 方法，传递输入参数
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

        # 将更新后的缓存添加到模型输出中
        if past_key_values is not None and return_dict:
            outputs, past_key_values = outputs
            outputs["past_key_values"] = unfreeze(past_key_values["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs, past_key_values = outputs
            outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

        # 返回模型输出
        return outputs
# 为了给 FlaxXGLMModel 类添加文档字符串，指定它输出原始隐藏状态而没有特定的顶部头部。
@add_start_docstrings(
    "The bare XGLM Model transformer outputting raw hidden-states without any specific head on top.",
    XGLM_START_DOCSTRING,
)
class FlaxXGLMModel(FlaxXGLMPreTrainedModel):
    module_class = FlaxXGLMModule


# 添加调用示例的文档字符串给 FlaxXGLMModel 类
append_call_sample_docstring(
    FlaxXGLMModel,
    _CHECKPOINT_FOR_DOC,
    FlaxBaseModelOutputWithPastAndCrossAttentions,
    _CONFIG_FOR_DOC,
)


class FlaxXGLMForCausalLMModule(nn.Module):
    config: XGLMConfig
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型

    def setup(self):
        # 使用配置和数据类型初始化 FlaxXGLMModule 模型
        self.model = FlaxXGLMModule(self.config, self.dtype)
        # 初始化语言模型头部，是一个全连接层，不使用偏置
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
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
    ):
        # 调用模型进行前向传播
        outputs = self.model(
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

        hidden_states = outputs[0]

        # 如果配置要求词嵌入共享，则使用共享的嵌入层参数进行计算
        if self.config.tie_word_embeddings:
            shared_embedding = self.model.variables["params"]["embed_tokens"]["embedding"]
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
        else:
            # 否则直接使用语言模型头部进行计算
            lm_logits = self.lm_head(hidden_states)

        # 如果不需要返回字典格式，则返回元组
        if not return_dict:
            return (lm_logits,) + outputs[1:]

        # 返回带有交叉注意力输出的 FlaxCausalLMOutputWithCrossAttentions 对象
        return FlaxCausalLMOutputWithCrossAttentions(
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


# 为 FlaxXGLMForCausalLM 类添加文档字符串，描述其为带有语言建模头部的 XGLM 模型变换器
@add_start_docstrings(
    """
    The XGLM Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    XGLM_START_DOCSTRING,
)
class FlaxXGLMForCausalLM(FlaxXGLMPreTrainedModel):
    module_class = FlaxXGLMForCausalLMModule
    # 为生成准备输入数据，初始化缓存
    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = None):
        # 获取输入张量的批量大小和序列长度
        batch_size, seq_length = input_ids.shape

        # 使用 self.init_cache 方法初始化过去键值对
        past_key_values = self.init_cache(batch_size, max_length)
        
        # 创建一个扩展的注意力掩码，初始化为全1数组
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        
        # 如果给定了 attention_mask，则根据其累积和更新位置 ID，并将 attention_mask 的值复制到扩展的注意力掩码中对应位置
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            # 否则，根据序列长度广播生成位置 ID
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        # 返回包含过去键值对、扩展注意力掩码和位置 ID 的字典
        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    # 更新生成的输入数据，将模型输出的过去键值对和更新后的位置 ID 存入 model_kwargs 中
    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs
# 调用函数 `append_call_sample_docstring`，将以下参数传递给它：
# - FlaxXGLMForCausalLM: 作为第一个参数传递的类或函数
# - _CHECKPOINT_FOR_DOC: 作为第二个参数传递的变量或值
# - FlaxCausalLMOutputWithCrossAttentions: 作为第三个参数传递的类或函数
# - _CONFIG_FOR_DOC: 作为第四个参数传递的变量或值
append_call_sample_docstring(
    FlaxXGLMForCausalLM,  # 第一个参数，传递类或函数 FlaxXGLMForCausalLM
    _CHECKPOINT_FOR_DOC,   # 第二个参数，传递变量或值 _CHECKPOINT_FOR_DOC
    FlaxCausalLMOutputWithCrossAttentions,  # 第三个参数，传递类或函数 FlaxCausalLMOutputWithCrossAttentions
    _CONFIG_FOR_DOC,  # 第四个参数，传递变量或值 _CONFIG_FOR_DOC
)
```