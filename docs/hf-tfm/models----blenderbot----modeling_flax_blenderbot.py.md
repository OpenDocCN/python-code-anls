# `.\models\blenderbot\modeling_flax_blenderbot.py`

```
# coding=utf-8
# 版权 2021 年 Fairseq 作者和 Google Flax 团队作者以及 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）许可；
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件，
# 没有任何明示或暗示的担保或条件。
# 有关特定语言的说明，请参阅许可证。
""" Flax Blenderbot model."""

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
from .configuration_blenderbot import BlenderbotConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "BlenderbotConfig"
_CHECKPOINT_FOR_DOC = "facebook/blenderbot-400M-distill"


BLENDERBOT_START_DOCSTRING = r"""
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
        config ([`BlenderbotConfig`]): Model configuration class with all the parameters of the model.
            使用 BlenderbotConfig 类作为参数，这个类包含了模型的所有参数。
            初始化时，仅加载配置文件，并不加载与模型相关的权重。
            若要加载模型的权重，请参考 [`~FlaxPreTrainedModel.from_pretrained`] 方法。
"""

BLENDERBOT_INPUTS_DOCSTRING = r"""
"""


BLENDERBOT_ENCODE_INPUTS_DOCSTRING = r"""
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

BLENDERBOT_DECODE_INPUTS_DOCSTRING = r"""
"""


# Copied from transformers.models.bart.modeling_flax_bart.shift_tokens_right
def shift_tokens_right(input_ids: jnp.ndarray, pad_token_id: int, decoder_start_token_id: int) -> jnp.ndarray:
    """
    Shift input ids one token to the right.

    Args:
        input_ids (jnp.ndarray): Array of input token indices.
        pad_token_id (int): Index of the padding token in the vocabulary.
        decoder_start_token_id (int): Index of the start token for decoder input.

    Returns:
        jnp.ndarray: Shifted input token indices.

    This function shifts the input token indices to the right by one position,
    inserting the decoder start token at the beginning and handling padding tokens.
    """
    shifted_input_ids = jnp.zeros_like(input_ids)  # Initialize an array of zeros with the same shape as input_ids
    shifted_input_ids = shifted_input_ids.at[:, 1:].set(input_ids[:, :-1])  # Shift input_ids to the right
    shifted_input_ids = shifted_input_ids.at[:, 0].set(decoder_start_token_id)  # Set the start token

    shifted_input_ids = jnp.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)  # Handle padding tokens
    return shifted_input_ids


# Copied from transformers.models.bart.modeling_flax_bart.FlaxBartAttention with Bart->Blenderbot
class FlaxBlenderbotAttention(nn.Module):
    """
    Implementation of the attention mechanism in the Blenderbot model.

    Attributes:
        config (BlenderbotConfig): The configuration object for the Blenderbot model.
        embed_dim (int): Dimensionality of the token embeddings.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability.
        causal (bool): Whether the attention is causal (for decoding).
        bias (bool): Whether to include bias in the attention computation.
        dtype (jnp.dtype): Data type of the computation (default is jnp.float32).
    """

    config: BlenderbotConfig
    embed_dim: int
    num_heads: int
    dropout: float = 0.0
    causal: bool = False
    bias: bool = True
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    def setup(self) -> None:
        # 将头维度设置为嵌入维度除以头数
        self.head_dim = self.embed_dim // self.num_heads
        # 检查嵌入维度是否能被头数整除，否则抛出数值错误
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )

        # 定义一个偏函数，用于创建具有固定参数的全连接层
        dense = partial(
            nn.Dense,
            self.embed_dim,
            use_bias=self.bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )

        # 创建查询、键、值和输出的全连接层
        self.q_proj, self.k_proj, self.v_proj = dense(), dense(), dense()
        self.out_proj = dense()

        # 创建一个dropout层，用于随机失活输入
        self.dropout_layer = nn.Dropout(rate=self.dropout)

        # 如果启用因果注意力，创建一个因果掩码
        if self.causal:
            self.causal_mask = make_causal_mask(
                jnp.ones((1, self.config.max_position_embeddings), dtype="bool"), dtype="bool"
            )

    def _split_heads(self, hidden_states):
        # 将隐藏状态张量重新形状为多头注意力的形状
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        # 将多头注意力的张量重新形状为原始形状
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slighly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # 检测是否通过检查"cache"中的变量"cached_key"存在来初始化
        is_initialized = self.has_variable("cache", "cached_key")
        # 从缓存中初始化或创建一个变量，用于存储缓存关键字
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        # 从缓存中初始化或创建一个变量，用于存储缓存数值
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        # 从缓存中初始化或创建一个变量，用于存储缓存索引
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))
        
        # 如果已经初始化
        if is_initialized:
            # 获取缓存关键字的维度信息
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # 使用新的1d空间切片更新关键字、数值缓存
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            # 更新缓存关键字的值
            cached_key.value = key
            # 更新缓存数值的值
            cached_value.value = value
            # 更新缓存索引值
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # 用于缓存的decoder自注意力的因果掩膜：我们的单个查询位置应只关注已经生成和缓存的那些关键字位置，而不是剩余的零元素。
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)
        return key, value, attention_mask

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        key_value_states: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        deterministic: bool = True,
# 从transformers.models.mbart.modeling_flax_mbart.FlaxMBartEncoderLayer复制代码，并将MBart->Blenderbot
class FlaxBlenderbotEncoderLayer(nn.Module):
    # 使用BlenderbotConfig配置类
    config: BlenderbotConfig
    # 计算中使用的数据类型为jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 设置函数，初始化编码器层
    def setup(self) -> None:
        # 设定嵌入维度为配置中的d_model值
        self.embed_dim = self.config.d_model
        # 使用FlaxBlenderbotAttention创建自注意力层
        self.self_attn = FlaxBlenderbotAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.encoder_attention_heads,
            dropout=self.config.attention_dropout,
            dtype=self.dtype,
        )
        # 对自注意力输出进行层归一化
        self.self_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        # dropout层，以配置中的dropout率丢弃部分神经元
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)
        # 激活函数使用配置中的激活函数类型的对应函数
        self.activation_fn = ACT2FN[self.config.activation_function]
        # 使用激活函数对应的dropout层，以配置中的激活dropout率丢弃部分神经元
        self.activation_dropout_layer = nn.Dropout(rate=self.config.activation_dropout)
        # 第一个全连接层，输出维度为配置中的encoder_ffn_dim
        self.fc1 = nn.Dense(
            self.config.encoder_ffn_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        # 第二个全连接层，输出维度与嵌入维度相同
        self.fc2 = nn.Dense(
            self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std)
        )
        # 最终的层归一化
        self.final_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    # 调用函数，执行编码器层的前向传播
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        output_attentions: bool = True,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray]:
        # 记录残差连接
        residual = hidden_states
        # 对输入的隐藏状态进行自注意力层的归一化
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # 执行自注意力机制计算
        hidden_states, attn_weights = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask)
        # 使用dropout层，丢弃部分计算结果
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 残差连接
        hidden_states = residual + hidden_states

        # 记录残差连接
        residual = hidden_states
        # 最终层归一化
        hidden_states = self.final_layer_norm(hidden_states)
        # 使用激活函数激活第一个全连接层的输出
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 使用激活dropout层，丢弃部分计算结果
        hidden_states = self.activation_dropout_layer(hidden_states, deterministic=deterministic)
        # 第二个全连接层的计算
        hidden_states = self.fc2(hidden_states)
        # 使用dropout层，丢弃部分计算结果
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 残差连接
        hidden_states = residual + hidden_states

        # 输出结果只包含隐藏状态
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则将注意力权重也加入输出
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# 从transformers.models.bart.modeling_flax_bart.FlaxBartEncoderLayerCollection复制代码，并将Bart->Blenderbot
class FlaxBlenderbotEncoderLayerCollection(nn.Module):
    # 使用BlenderbotConfig配置类
    config: BlenderbotConfig
    # 计算中使用的数据类型为jnp.float32
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    # 设置函数，初始化编码器层集合
    def setup(self):
        # 创建编码器层列表，包括多个FlaxBlenderbotEncoderLayer实例
        self.layers = [
            FlaxBlenderbotEncoderLayer(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.encoder_layers)
        ]
        # 编码器层的层丢弃率，从配置中获取
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
        # 如果需要输出注意力权重，则初始化空元组；否则设为None
        all_attentions = () if output_attentions else None
        # 如果需要输出隐藏状态，则初始化空元组；否则设为None
        all_hidden_states = () if output_hidden_states else None

        # 遍历每个编码器层
        for encoder_layer in self.layers:
            # 如果需要输出隐藏状态，则将当前隐藏状态加入元组中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            # 添加 LayerDrop 功能（参见 https://arxiv.org/abs/1909.11556 ）
            dropout_probability = random.uniform(0, 1)
            # 如果非确定性计算且随机数小于层丢弃概率，则跳过当前层
            if not deterministic and (dropout_probability < self.layerdrop):
                layer_outputs = (None, None)
            else:
                # 否则，调用当前编码器层的前向传播方法
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions,
                    deterministic,
                )
            # 更新隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]
            # 如果需要输出注意力权重，则将当前层的注意力权重加入元组中
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # 如果需要输出隐藏状态，则将最终的隐藏状态加入元组中
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 构建输出元组，包括最终的隐藏状态、所有隐藏状态和所有注意力权重
        outputs = (hidden_states, all_hidden_states, all_attentions)

        # 如果不需要返回字典，则将输出中的None值过滤掉后返回
        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        # 否则，将输出作为 FlaxBaseModelOutput 类的实例返回
        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )
# 从transformers.models.mbart.modeling_flax_mbart.FlaxMBartDecoderLayer复制代码，并将MBart->Blenderbot
class FlaxBlenderbotDecoderLayer(nn.Module):
    # 定义配置为BlenderbotConfig
    config: BlenderbotConfig
    # 定义数据类型为jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 初始化函数，设置各层和模块
    def setup(self) -> None:
        # 设置嵌入维度为配置中的d_model
        self.embed_dim = self.config.d_model
        # 创建BlenderbotAttention自注意力机制实例
        self.self_attn = FlaxBlenderbotAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.decoder_attention_heads,
            dropout=self.config.attention_dropout,
            causal=True,
            dtype=self.dtype,
        )
        # Dropout层，使用配置中的dropout率
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)
        # 激活函数，使用配置中指定的激活函数
        self.activation_fn = ACT2FN[self.config.activation_function]
        # 激活函数的Dropout层，使用配置中的activation_dropout率
        self.activation_dropout_layer = nn.Dropout(rate=self.config.activation_dropout)

        # Layer normalization层，用于自注意力机制
        self.self_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        # 创建BlenderbotAttention编码器注意力机制实例
        self.encoder_attn = FlaxBlenderbotAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.decoder_attention_heads,
            dropout=self.config.attention_dropout,
            dtype=self.dtype,
        )
        # 编码器注意力机制的Layer normalization层
        self.encoder_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        
        # 第一个全连接层，输出维度为配置中的decoder_ffn_dim
        self.fc1 = nn.Dense(
            self.config.decoder_ffn_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        # 第二个全连接层，输出维度与嵌入维度相同
        self.fc2 = nn.Dense(
            self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std)
        )
        # 最终的Layer normalization层
        self.final_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    # 调用函数，定义模型的前向计算过程
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        output_attentions: bool = True,
        deterministic: bool = True,
        # 继续定义参数
        ) -> Tuple[jnp.ndarray]:
        # 将输入的隐藏状态保存为残差连接的一部分
        residual = hidden_states
        # 对当前隐藏状态进行 Layer Normalization
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 自注意力机制
        # 调用 self_attn 方法进行自注意力计算，同时返回注意力权重
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, init_cache=init_cache
        )
        # 应用 dropout 层，以防止过拟合
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 将残差连接添加到当前隐藏状态中
        hidden_states = residual + hidden_states

        # 交叉注意力块
        cross_attn_weights = None
        # 如果存在编码器的隐藏状态，则执行以下操作
        if encoder_hidden_states is not None:
            # 将输入的隐藏状态保存为残差连接的一部分
            residual = hidden_states

            # 对当前隐藏状态进行 Layer Normalization
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            # 调用 encoder_attn 方法进行交叉注意力计算，同时返回注意力权重
            hidden_states, cross_attn_weights = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            # 应用 dropout 层，以防止过拟合
            hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
            # 将残差连接添加到当前隐藏状态中
            hidden_states = residual + hidden_states

        # 全连接层
        # 将输入的隐藏状态保存为残差连接的一部分
        residual = hidden_states
        # 对当前隐藏状态进行 Layer Normalization
        hidden_states = self.final_layer_norm(hidden_states)
        # 应用激活函数到全连接层的第一层
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 应用 dropout 层，以防止过拟合
        hidden_states = self.activation_dropout_layer(hidden_states, deterministic=deterministic)
        # 应用全连接层的第二层
        hidden_states = self.fc2(hidden_states)
        # 应用 dropout 层，以防止过拟合
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 将残差连接添加到当前隐藏状态中
        hidden_states = residual + hidden_states

        # 返回输出结果
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则将自注意力和交叉注意力的权重添加到输出中
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs
# Copied from transformers.models.bart.modeling_flax_bart.FlaxBartDecoderLayerCollection with Bart->Blenderbot
class FlaxBlenderbotDecoderLayerCollection(nn.Module):
    config: BlenderbotConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        # Initialize a list of Blenderbot decoder layers based on the provided configuration
        self.layers = [
            FlaxBlenderbotDecoderLayer(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.decoder_layers)
        ]
        # Set the layer dropout probability from the configuration
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
        # Initialize containers for storing outputs
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        # Iterate through each decoder layer
        for decoder_layer in self.layers:
            if output_hidden_states:
                # Store hidden states for potential use in output
                all_hidden_states += (hidden_states,)
                # Implement LayerDrop regularization during training
                # (see https://arxiv.org/abs/1909.11556 for details)
            dropout_probability = random.uniform(0, 1)
            if not deterministic and (dropout_probability < self.layerdrop):
                # Skip computation of the layer outputs based on LayerDrop probability
                layer_outputs = (None, None, None)
            else:
                # Compute outputs of the current decoder layer
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    init_cache=init_cache,
                    output_attentions=output_attentions,
                    deterministic=deterministic,
                )

            # Update hidden states with the outputs of the current layer
            hidden_states = layer_outputs[0]
            if output_attentions:
                # Store self-attention outputs if specified
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    # Store cross-attention outputs if specified and encoder_hidden_states is provided
                    all_cross_attentions += (layer_outputs[2],)

        # Store hidden states from the last decoder layer if output_hidden_states is enabled
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Prepare outputs based on return_dict flag
        outputs = [hidden_states, all_hidden_states, all_self_attns, all_cross_attentions]

        if not return_dict:
            # Return outputs as a tuple omitting None values
            return tuple(v for v in outputs if v is not None)

        # Return outputs as a FlaxBaseModelOutputWithPastAndCrossAttentions named tuple
        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class FlaxBlenderbotEncoder(nn.Module):
    config: BlenderbotConfig
    embed_tokens: nn.Embed
    # 定义数据类型为 jnp.float32，用于计算的数据类型
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    # 执行初始化设置操作
    def setup(self):
        # 初始化一个丢弃层，根据配置中的丢弃率
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)

        # 设置嵌入维度为模型配置中的 d_model
        embed_dim = self.config.d_model
        # 设置填充索引为配置中的 pad_token_id
        self.padding_idx = self.config.pad_token_id
        # 设置最大源序列位置为配置中的 max_position_embeddings
        self.max_source_positions = self.config.max_position_embeddings
        # 如果配置中设置了 scale_embedding，则设置嵌入缩放因子为 embed_dim 的平方根，否则为 1.0
        self.embed_scale = math.sqrt(embed_dim) if self.config.scale_embedding else 1.0

        # 初始化位置嵌入层，使用正态分布初始化方法，标准差为配置中的 init_std
        self.embed_positions = nn.Embed(
            self.config.max_position_embeddings,
            embed_dim,
            embedding_init=jax.nn.initializers.normal(self.config.init_std),
        )
        # 初始化编码器层集合
        self.layers = FlaxBlenderbotEncoderLayerCollection(self.config, self.dtype)
        # 初始化层归一化层，数据类型为 self.dtype，设置 epsilon 为 1e-05
        self.layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    # 定义对象调用方法
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
        # 获取输入张量的形状
        input_shape = input_ids.shape
        # 将输入张量展平为二维张量，保留最后一个维度
        input_ids = input_ids.reshape(-1, input_shape[-1])

        # 对输入 token 嵌入进行缩放
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # 获取位置嵌入
        embed_pos = self.embed_positions(position_ids)

        # 将嵌入的 token 和位置加和得到隐藏状态
        hidden_states = inputs_embeds + embed_pos
        # 对隐藏状态应用丢弃层，根据 deterministic 参数确定是否确定性丢弃
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)

        # 对隐藏状态应用编码器层
        outputs = self.layers(
            hidden_states,
            attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取编码器层输出的最后一个隐藏状态
        last_hidden_states = outputs[0]
        # 对最后一个隐藏状态应用层归一化
        last_hidden_states = self.layer_norm(last_hidden_states)

        # 如果需要输出隐藏状态，则更新隐藏状态列表中的最后一个元素
        hidden_states = None
        if output_hidden_states:
            hidden_states = outputs[1]
            hidden_states = hidden_states[:-1] + (last_hidden_states,)

        # 如果不需要以字典形式返回结果，则将输出元组化并返回
        if not return_dict:
            outputs = (last_hidden_states, hidden_states) + (outputs[2:] if output_hidden_states else outputs[1:])
            return tuple(v for v in outputs if v is not None)

        # 以 FlaxBaseModelOutput 类的形式返回结果，包括最后的隐藏状态、隐藏状态和注意力权重（如果有）
        return FlaxBaseModelOutput(
            last_hidden_state=last_hidden_states,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
        )
class FlaxBlenderbotDecoder(nn.Module):
    config: BlenderbotConfig  # 定义配置对象的类型为 BlenderbotConfig
    embed_tokens: nn.Embed  # 定义嵌入层对象的类型为 nn.Embed
    dtype: jnp.dtype = jnp.float32  # 定义计算过程中的数据类型为 jnp.float32

    def setup(self):
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)  # 初始化丢弃层对象，并设定丢弃率为配置中的 dropout

        embed_dim = self.config.d_model  # 从配置中获取嵌入维度
        self.padding_idx = self.config.pad_token_id  # 获取填充标记的索引
        self.max_target_positions = self.config.max_position_embeddings  # 获取目标位置的最大数
        self.embed_scale = math.sqrt(self.config.d_model) if self.config.scale_embedding else 1.0  # 根据配置计算嵌入比例

        self.embed_positions = nn.Embed(
            self.config.max_position_embeddings,
            embed_dim,
            embedding_init=jax.nn.initializers.normal(self.config.init_std),
        )  # 初始化位置嵌入层对象，设定位置数量、嵌入维度和初始化方式

        self.layers = FlaxBlenderbotDecoderLayerCollection(self.config, self.dtype)  # 初始化解码器层集合对象
        self.layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)  # 初始化层归一化对象，设定数据类型和 epsilon 值

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
        input_shape = input_ids.shape  # 获取输入数据形状
        input_ids = input_ids.reshape(-1, input_shape[-1])  # 重塑输入数据的形状为二维矩阵

        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale  # 将输入数据嵌入到嵌入层中，并按比例缩放

        # 嵌入位置信息
        positions = self.embed_positions(position_ids)

        hidden_states = inputs_embeds + positions  # 将输入嵌入向量与位置嵌入相加得到隐藏状态
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)  # 应用丢弃层

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
        )  # 将隐藏状态传递给解码器层集合对象进行解码

        last_hidden_states = outputs[0]  # 获取最后一个隐藏状态
        last_hidden_states = self.layer_norm(last_hidden_states)  # 应用层归一化到最后一个隐藏状态

        # 更新 `hidden_states` 中应用 `layernorm` 后的最后一个元素
        hidden_states = None
        if output_hidden_states:
            hidden_states = outputs[1]  # 获取所有隐藏状态
            hidden_states = hidden_states[:-1] + (last_hidden_states,)  # 将最后一个隐藏状态添加到列表中

        if not return_dict:
            outputs = (last_hidden_states, hidden_states) + (outputs[2:] if output_hidden_states else outputs[1:])
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=last_hidden_states,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )  # 返回模型输出对象，包含最后的隐藏状态、所有隐藏状态、注意力分数和交叉注意力分数
# 定义一个自定义的 Flax 模型，继承自 nn.Module
class FlaxBlenderbotModule(nn.Module):
    # 声明类变量 config，类型为 BlenderbotConfig，用于配置模型
    config: BlenderbotConfig
    # 声明类变量 dtype，表示计算时的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32  # 计算时的数据类型

    # 模型的初始化方法
    def setup(self):
        # 初始化一个共享的嵌入层，vocab_size 和 d_model 来自于 config，初始化方式为正态分布
        self.shared = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.init_std),
            dtype=self.dtype,
        )

        # 初始化编码器，使用自定义的 FlaxBlenderbotEncoder 类
        self.encoder = FlaxBlenderbotEncoder(self.config, dtype=self.dtype, embed_tokens=self.shared)
        # 初始化解码器，使用自定义的 FlaxBlenderbotDecoder 类
        self.decoder = FlaxBlenderbotDecoder(self.config, dtype=self.dtype, embed_tokens=self.shared)

    # 获取编码器模块的方法
    def _get_encoder_module(self):
        return self.encoder

    # 获取解码器模块的方法
    def _get_decoder_module(self):
        return self.decoder

    # 定义模型的调用方法，实现模型的前向传播
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
        # 调用编码器进行前向传播
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        # 调用解码器进行前向传播，其中传入编码器的输出作为解码器的输入
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

        # 如果 return_dict 为 False，则返回解码器和编码器的输出
        if not return_dict:
            return decoder_outputs + encoder_outputs

        # 如果 return_dict 为 True，则将解码器和编码器的输出整合成 FlaxSeq2SeqModelOutput 类型的结果并返回
        return FlaxSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


# 继承自 FlaxPreTrainedModel 的预训练模型基类
class FlaxBlenderbotPreTrainedModel(FlaxPreTrainedModel):
    # 配置类为 BlenderbotConfig
    config_class = BlenderbotConfig
    # 模型前缀为 "model"
    base_model_prefix: str = "model"
    # 模块类为 nn.Module，在子类中定义
    module_class: nn.Module = None

    # 模型初始化方法
    def __init__(
        self,
        config: BlenderbotConfig,
        input_shape: Tuple[int] = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
        ):
            # 使用给定的配置、数据类型和其他关键字参数初始化模块对象
            module = self.module_class(config=config, dtype=dtype, **kwargs)
            # 调用父类的构造方法初始化对象
            super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

        def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
            # 初始化输入张量
            input_ids = jnp.zeros(input_shape, dtype="i4")
            # 确保初始化过程适用于 FlaxBlenderbotForSequenceClassificationModule
            input_ids = input_ids.at[(..., -1)].set(self.config.eos_token_id)
            attention_mask = jnp.ones_like(input_ids)
            decoder_input_ids = input_ids
            decoder_attention_mask = jnp.ones_like(input_ids)

            batch_size, sequence_length = input_ids.shape
            # 创建位置编码矩阵，形状与输入张量相同
            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))
            decoder_position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

            # 划分随机数生成器为参数和dropout使用的两部分
            params_rng, dropout_rng = jax.random.split(rng)
            rngs = {"params": params_rng, "dropout": dropout_rng}

            # 使用模块的初始化方法初始化模型参数
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
                # 如果提供了参数，则将随机生成的参数与已有参数进行融合
                random_params = flatten_dict(unfreeze(random_params))
                params = flatten_dict(unfreeze(params))
                for missing_key in self._missing_keys:
                    params[missing_key] = random_params[missing_key]
                self._missing_keys = set()
                return freeze(unflatten_dict(params))
            else:
                # 如果未提供参数，则返回随机生成的参数
                return random_params
    @add_start_docstrings(BLENDERBOT_ENCODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutput, config_class=BlenderbotConfig)
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
        """
        Encodes input sequences into hidden states using the Blenderbot model.

        Args:
            input_ids (`jnp.ndarray`):
                The input token IDs. Shape `(batch_size, sequence_length)`.
            attention_mask (`Optional[jnp.ndarray]`, optional):
                Mask to avoid performing attention on padding token indices. Shape `(batch_size, sequence_length)`.
            position_ids (`Optional[jnp.ndarray]`, optional):
                Indices of positions for each input token in the sequence. Shape `(batch_size, sequence_length)`.
            output_attentions (`Optional[bool]`, optional):
                Whether to output attentions weights. Default is `None`.
            output_hidden_states (`Optional[bool]`, optional):
                Whether to output hidden states of all layers. Default is `None`.
            return_dict (`Optional[bool]`, optional):
                Whether to return a dictionary instead of a tuple of outputs. Default is `None`.
            train (`bool`, optional):
                Whether the model is in training mode. Default is `False`.
            params (`dict`, optional):
                Optional parameters for model inference.
            dropout_rng (`PRNGKey`, optional):
                Dropout random number generator key for reproducibility.

        Returns:
            `FlaxBaseModelOutput`: A namedtuple with the following fields:
                - last_hidden_state (`jnp.ndarray`):
                    Sequence of hidden states at the output of the last layer of the model. Shape
                    `(batch_size, sequence_length, hidden_size)`.
                - hidden_states (`Optional[Tuple[jnp.ndarray]]`, optional):
                    Sequence of hidden states for all layers. Only returned when `output_hidden_states=True`.
                - attentions (`Optional[Tuple[jnp.ndarray]]`, optional):
                    Attention weights for each layer. Only returned when `output_attentions=True`.
        """
        # Combine the input arguments into a dictionary for model initialization
        input_args = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
            "train": train,
            "params": params,
            "dropout_rng": dropout_rng,
        }

        # Initialize the model using its `init` method with specified input arguments
        initialized_model = self.module.init(
            jax.random.PRNGKey(0),  # Initialize with a fixed PRNGKey for reproducibility
            **input_args,
        )

        # Return the initialized cache from the model's initialization results
        return unfreeze(initialized_model["cache"])
    ):
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, FlaxBlenderbotForConditionalGeneration

        >>> model = FlaxBlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")

        >>> text = "My friends are cool but they eat too many carbs."
        >>> inputs = tokenizer(text, max_length=1024, return_tensors="jax")
        >>> encoder_outputs = model.encode(**inputs)
        ```"""
        # 初始化输出注意力的设置，如果没有传入则使用模型配置中的默认设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 初始化输出隐藏状态的设置，如果没有传入则使用模型配置中的默认设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 初始化返回字典的设置，如果没有传入则使用模型配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 如果注意力掩码为None，则创建一个全1的掩码数组，形状与输入ids相同
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        # 如果位置ids为None，则根据输入ids的形状广播创建位置ids
        if position_ids is None:
            batch_size, sequence_length = input_ids.shape
            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # 处理任何需要的伪随机数生成器
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        # 定义_encoder_forward函数，用于对编码模块进行前向传播
        def _encoder_forward(module, input_ids, attention_mask, position_ids, **kwargs):
            encode_module = module._get_encoder_module()
            return encode_module(input_ids, attention_mask, position_ids, **kwargs)

        # 调用self.module.apply进行模型的前向传播
        return self.module.apply(
            {"params": params or self.params},  # 参数字典，使用传入的参数或者模型自身的参数
            input_ids=jnp.array(input_ids, dtype="i4"),  # 输入的ids数组，转换为JAX支持的整数数组
            attention_mask=jnp.array(attention_mask, dtype="i4"),  # 注意力掩码数组，转换为JAX支持的整数数组
            position_ids=jnp.array(position_ids, dtype="i4"),  # 位置ids数组，转换为JAX支持的整数数组
            output_attentions=output_attentions,  # 是否输出注意力权重
            output_hidden_states=output_hidden_states,  # 是否输出隐藏状态
            return_dict=return_dict,  # 是否返回字典格式的输出
            deterministic=not train,  # 是否确定性计算，如果非训练状态则确定性计算
            rngs=rngs,  # 伪随机数生成器字典
            method=_encoder_forward,  # 调用的方法，这里为_encoder_forward函数
        )

    @add_start_docstrings(BLENDERBOT_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=FlaxBaseModelOutputWithPastAndCrossAttentions, config_class=BlenderbotConfig
    )
    # 定义decode函数，用于解码
    def decode(
        self,
        decoder_input_ids,  # 解码器输入的ids
        encoder_outputs,  # 编码器的输出
        encoder_attention_mask: Optional[jnp.ndarray] = None,  # 编码器的注意力掩码
        decoder_attention_mask: Optional[jnp.ndarray] = None,  # 解码器的注意力掩码
        decoder_position_ids: Optional[jnp.ndarray] = None,  # 解码器的位置ids
        past_key_values: dict = None,  # 过去的键值对
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典格式的输出
        train: bool = False,  # 是否为训练模式
        params: dict = None,  # 参数字典
        dropout_rng: PRNGKey = None,  # 伪随机数生成器
    @add_start_docstrings_to_model_forward(BLENDERBOT_INPUTS_DOCSTRING)
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
        # 确定是否输出注意力权重，默认从模型配置获取
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 确定是否输出隐藏状态，默认从模型配置获取
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 确定是否返回字典格式，默认从模型配置获取
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 准备编码器的输入
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        if position_ids is None:
            batch_size, sequence_length = input_ids.shape
            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # 准备解码器的输入
        if decoder_input_ids is None:
            # 将输入右移一位，用于生成解码器的输入序列
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

        # 如果需要处理随机数生成器（PRNG）
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        # 应用模型的前向传播
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
            deterministic=not train,  # 确定是否使用确定性推理，根据训练标志
            rngs=rngs,
        )
# 给 FlaxBlenderbotModel 类添加文档字符串，描述其作为不带特定头部的裸 MBart 模型变换器的输出
@add_start_docstrings(
    "The bare MBart Model transformer outputting raw hidden-states without any specific head on top.",
    BLENDERBOT_START_DOCSTRING,
)
class FlaxBlenderbotModel(FlaxBlenderbotPreTrainedModel):
    # 引入 BlenderbotConfig 配置
    config: BlenderbotConfig
    # 计算中使用的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 指定模块类为 FlaxBlenderbotModule
    module_class = FlaxBlenderbotModule

# 向 FlaxBlenderbotModel 类附加示例调用文档字符串
append_call_sample_docstring(FlaxBlenderbotModel, _CHECKPOINT_FOR_DOC, FlaxSeq2SeqModelOutput, _CONFIG_FOR_DOC)


# 从 transformers.models.bart.modeling_flax_bart.FlaxBartForConditionalGenerationModule 复制代码，将 Bart 替换为 Blenderbot
class FlaxBlenderbotForConditionalGenerationModule(nn.Module):
    # 引入 BlenderbotConfig 配置
    config: BlenderbotConfig
    # 计算中使用的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 偏置初始化器，用于初始化偏置参数
    bias_init: Callable[..., jnp.ndarray] = jax.nn.initializers.zeros

    def setup(self):
        # 创建 Blenderbot 模块，并使用给定的配置和数据类型初始化
        self.model = FlaxBlenderbotModule(config=self.config, dtype=self.dtype)
        # 创建语言模型头部，使用全连接层，参数根据模型共享的词汇表大小初始化
        self.lm_head = nn.Dense(
            self.model.shared.num_embeddings,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        # 创建最终预测 logits 的偏置项，初始化为零向量
        self.final_logits_bias = self.param("final_logits_bias", self.bias_init, (1, self.model.shared.num_embeddings))

    # 获取编码器模块的方法
    def _get_encoder_module(self):
        return self.model.encoder

    # 获取解码器模块的方法
    def _get_decoder_module(self):
        return self.model.decoder

    # 定义对象调用方法，接收多个输入和控制参数，用于模型的正向传播
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
        # 后续还有更多参数...
        ):
            # 使用模型生成输出结果，传入输入张量及相关参数
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

            # 从模型输出中获取隐藏状态张量
            hidden_states = outputs[0]

            # 如果配置了词嵌入共享，使用共享的嵌入层进行计算
            if self.config.tie_word_embeddings:
                shared_embedding = self.model.variables["params"]["shared"]["embedding"]
                lm_logits = self.lm_head.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
            else:
                # 否则直接使用语言模型头部计算逻辑回归
                lm_logits = self.lm_head(hidden_states)

            # 添加最终逻辑偏置
            lm_logits += jax.lax.stop_gradient(self.final_logits_bias.astype(self.dtype))

            # 如果不要求返回字典形式的输出，则返回元组
            if not return_dict:
                output = (lm_logits,) + outputs[1:]
                return output

            # 返回 FlaxSeq2SeqLMOutput 类型的输出，包含逻辑回归、解码器隐藏状态和注意力权重等
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
    "The Blenderbot Model with a language modeling head. Can be used for summarization.", BLENDERBOT_START_DOCSTRING
)
class FlaxBlenderbotForConditionalGeneration(FlaxBlenderbotPreTrainedModel):
    module_class = FlaxBlenderbotForConditionalGenerationModule
    dtype: jnp.dtype = jnp.float32

    @add_start_docstrings(BLENDERBOT_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxCausalLMOutputWithCrossAttentions, config_class=BlenderbotConfig)
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
        Decode function for generation tasks.

        Args:
        - decoder_input_ids: Input tensor of decoder input token IDs.
        - encoder_outputs: Output from the encoder.
        - encoder_attention_mask: Optional tensor specifying which tokens in the encoder output should not be attended to.
        - decoder_attention_mask: Optional tensor specifying which tokens in the decoder input should not be attended to.
        - decoder_position_ids: Optional tensor specifying the position IDs for the decoder input tokens.
        - past_key_values: Dictionary containing cached key-value states from previous decoding steps.
        - output_attentions: Whether to output attention weights.
        - output_hidden_states: Whether to output hidden states.
        - return_dict: Whether to return a dictionary of outputs.
        - train: Whether the model is in training mode.
        - params: Additional parameters for decoding.
        - dropout_rng: Dropout random number generator key.

        Returns:
        - FlaxCausalLMOutputWithCrossAttentions: Output object containing generated logits and optional cross-attention weights.

        """
        # initializing the cache
        batch_size, seq_length = decoder_input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length, encoder_outputs)
        # Note that usually one would have to put 0's in the attention_mask for x > input_ids.shape[-1] and x < cache_length.
        # But since the decoder uses a causal mask, those positions are masked anyways.
        # Thus we can create a single static attention_mask here, which is more efficient for compilation
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
        Prepares inputs for generation by setting up attention masks and position IDs.

        Args:
        - decoder_input_ids: Tensor of input token IDs for the decoder.
        - max_length: Maximum length of the generated sequence.
        - attention_mask: Optional tensor specifying which tokens in the encoder output should not be attended to.
        - decoder_attention_mask: Optional tensor specifying which tokens in the decoder input should not be attended to.
        - encoder_outputs: Output from the encoder.
        - **kwargs: Additional keyword arguments.

        Returns:
        - dict: Dictionary containing prepared inputs for the generation process.
        """
        # initializing the cache
        batch_size, seq_length = decoder_input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length, encoder_outputs)
        # Note that usually one would have to put 0's in the attention_mask for x > input_ids.shape[-1] and x < cache_length.
        # But since the decoder uses a causal mask, those positions are masked anyways.
        # Thus we can create a single static attention_mask here, which is more efficient for compilation
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

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        """
        Updates inputs for the generation process by adjusting position IDs and past key values.

        Args:
        - model_outputs: Output from the model.
        - model_kwargs: Keyword arguments for the model.

        Returns:
        - dict: Updated model keyword arguments for generation.
        """
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["decoder_position_ids"] = model_kwargs["decoder_position_ids"][:, -1:] + 1
        return model_kwargs


FLAX_BLENDERBOT_CONDITIONAL_GENERATION_DOCSTRING = r"""
    Returns:

    Conversation example::

    ```py
    >>> from transformers import AutoTokenizer, FlaxBlenderbotForConditionalGeneration
    导入所需的库：从transformers库中导入AutoTokenizer和FlaxBlenderbotForConditionalGeneration类
    
    >>> model = FlaxBlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
    使用预训练模型"facebook/blenderbot-400M-distill"初始化生成模型对象model
    
    >>> tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
    使用预训练的tokenizer模型"facebook/blenderbot-400M-distill"初始化分词器对象tokenizer
    
    >>> UTTERANCE = "My friends are cool but they eat too many carbs."
    定义待生成回复的文本UTTERANCE
    
    >>> inputs = tokenizer([UTTERANCE], max_length=1024, return_tensors="np")
    使用tokenizer对UTTERANCE进行分词和编码，生成模型输入inputs，设置最大长度为1024，返回类型为numpy数组
    
    >>> # Generate Reply
    生成回复的注释标记
    
    >>> reply_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=5, early_stopping=True).sequences
    调用生成模型生成回复，使用输入的input_ids，设置生成束搜索数为4，最大生成长度为5，启用提前停止策略，并获取生成的序列reply_ids
    
    >>> print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in reply_ids])
    打印生成的回复，对每个生成的序列进行反向分词和解码，并去除特殊标记和清理分词空格
"""
给指定类 `FlaxBlenderbotForConditionalGeneration` 覆盖方法 `call` 的文档字符串，
合并 `BLENDERBOT_INPUTS_DOCSTRING` 和 `FLAX_BLENDERBOT_CONDITIONAL_GENERATION_DOCSTRING`。
"""
overwrite_call_docstring(
    FlaxBlenderbotForConditionalGeneration,
    BLENDERBOT_INPUTS_DOCSTRING + FLAX_BLENDERBOT_CONDITIONAL_GENERATION_DOCSTRING,
)

"""
给指定类 `FlaxBlenderbotForConditionalGeneration` 添加或替换方法 `append_replace_return_docstrings` 的文档字符串，
设置输出类型为 `FlaxSeq2SeqLMOutput`，配置类为 `_CONFIG_FOR_DOC`。
"""
append_replace_return_docstrings(
    FlaxBlenderbotForConditionalGeneration, output_type=FlaxSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC
)
```