# `.\models\pegasus\modeling_flax_pegasus.py`

```
# 导入必要的库和模块
import math  # 导入数学函数库
import random  # 导入随机数生成模块
from functools import partial  # 导入函数工具模块中的 partial 函数
from typing import Callable, Optional, Tuple  # 导入类型提示相关的模块

import flax.linen as nn  # 导入 Flax 的 Linen 模块，用于定义神经网络层
import jax  # 导入 JAX，用于自动求导和并行计算
import jax.numpy as jnp  # 导入 JAX 对应的 NumPy 函数库
import numpy as np  # 导入 NumPy 函数库
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze  # 导入冻结字典相关的功能
from flax.linen import combine_masks, make_causal_mask  # 导入组合掩码和创建因果掩码的函数
from flax.linen.attention import dot_product_attention_weights  # 导入点积注意力权重计算函数
from flax.traverse_util import flatten_dict, unflatten_dict  # 导入扁平化和反扁平化字典的工具函数
from jax import lax  # 导入 JAX 的低级 API，用于控制流程和并行计算
from jax.random import PRNGKey  # 导入 JAX 随机数生成器 PRNGKey

from ...modeling_flax_outputs import (  # 导入输出相关的 Flax 模块
    FlaxBaseModelOutput,
    FlaxBaseModelOutputWithPastAndCrossAttentions,
    FlaxCausalLMOutputWithCrossAttentions,
    FlaxSeq2SeqLMOutput,
    FlaxSeq2SeqModelOutput,
)
from ...modeling_flax_utils import (  # 导入 Flax 模型工具函数
    ACT2FN,
    FlaxPreTrainedModel,
    add_start_docstrings_to_model_forward,
    append_call_sample_docstring,
    append_replace_return_docstrings,
    overwrite_call_docstring,
)
from ...utils import add_start_docstrings, logging, replace_return_docstrings  # 导入工具函数和日志记录相关模块
from .configuration_pegasus import PegasusConfig  # 导入 Pegasus 模型的配置类

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

_CHECKPOINT_FOR_DOC = "google/pegasus-large"  # Pegasus 模型的预训练检查点名称
_CONFIG_FOR_DOC = "PegasusConfig"  # Pegasus 模型的配置名称

PEGASUS_START_DOCSTRING = r"""
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
    # 定义函数参数说明
    Parameters:
        config ([`PegasusConfig`]): Model configuration class with all the parameters of the model.
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

PEGASUS_INPUTS_DOCSTRING = r"""
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
        decoder_input_ids (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)
        decoder_attention_mask (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.

            If you want to change padding behavior, you should modify to your needs. See diagram 1 in [the
            paper](https://arxiv.org/abs/1910.13461) for more information on the default strategy.
        position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.
        decoder_position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the
            range `[0, config.max_position_embeddings - 1]`.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


PEGASUS_ENCODE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens to be encoded. These indices are obtained using a tokenizer, typically
            from a list of input strings. Each index corresponds to a token in the vocabulary.
        
        attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. This mask tensor has shape `(batch_size, 
            sequence_length)`, where each value is either 1 (token is not masked) or 0 (token is masked, typically 
            because it's padding).
        
        position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence token in the position embeddings matrix. These indices range 
            from 0 to `config.max_position_embeddings - 1`.
        
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. If `True`, the returned output 
            will include attention tensors from all layers of the model.
        
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. If `True`, the returned output will include 
            hidden states from all layers of the model.
        
        return_dict (`bool`, *optional*):
            Whether or not to return a `utils.ModelOutput` object instead of a plain tuple. If `True`, the output 
            will be encapsulated in a structured object that includes additional metadata.
"""
    Args:
        input_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
            # 输入序列标记的索引数组，形状为(batch_size, sequence_length)。默认情况下会忽略填充部分。
            # 可以使用AutoTokenizer获取这些索引。参见PreTrainedTokenizer.encode和PreTrainedTokenizer.__call__获取详细信息。
            # 输入IDs是什么？详见../glossary#input-ids

        attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            # 避免在填充的标记索引上执行注意力操作的掩码。掩码值选择在[0, 1]范围内：
            # - 1表示**未屏蔽**的标记，
            # - 0表示**屏蔽**的标记。
            # 注意掩码是什么？详见../glossary#attention-mask

        position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            # 输入序列每个标记在位置嵌入中的位置索引数组，形状为(batch_size, sequence_length)。
            # 索引值选在范围[0, config.max_position_embeddings - 1]内。

        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。更多详细信息参见返回的张量中的'attentions'部分。

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。更多详细信息参见返回的张量中的'hidden_states'部分。

        return_dict (`bool`, *optional*):
            # 是否返回一个utils.ModelOutput而不是普通的元组。
# PEGASUS_DECODE_INPUTS_DOCSTRING 是一个原始字符串（raw string），用于文档化 Pegasus 解码函数的输入参数及其含义。
PEGASUS_DECODE_INPUTS_DOCSTRING = r"""
    Args:
        decoder_input_ids (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`):
            解码器输入序列标记在词汇表中的索引。

            索引可以使用 [`AutoTokenizer`] 获得。有关详细信息，请参阅 [`PreTrainedTokenizer.encode`] 和
            [`PreTrainedTokenizer.__call__`]。

            [什么是解码器输入ID？](../glossary#decoder-input-ids)
        encoder_outputs (`tuple(tuple(jnp.ndarray)`):
            元组包括 (`last_hidden_state`, *可选*: `hidden_states`, *可选*: `attentions`)
            `last_hidden_state` 的形状为 `(batch_size, sequence_length, hidden_size)`，*可选*) 是编码器最后一层的隐藏状态序列。用于解码器的交叉注意力。
        encoder_attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *可选*):
            遮罩，避免在填充的标记索引上执行注意力。遮罩值选择在 `[0, 1]`:

            - 对于 **未遮罩** 的标记为 1,
            - 对于 **遮罩** 的标记为 0.

            [什么是注意力遮罩？](../glossary#attention-mask)
        decoder_attention_mask (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`, *可选*):
            默认行为: 生成一个张量，忽略 `decoder_input_ids` 中的填充标记。默认情况下也将使用因果遮罩。

            如果要更改填充行为，应根据需求进行修改。有关默认策略的更多信息，请参见 [论文中的图表 1](https://arxiv.org/abs/1910.13461)。
        decoder_position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *可选*):
            每个解码器输入序列标记在位置嵌入中的位置索引。选取范围为 `[0, config.max_position_embeddings - 1]`。
        past_key_values (`Dict[str, np.ndarray]`, *可选*, 由 `init_cache` 返回或传递先前的 `past_key_values`):
            预计算的隐藏状态的字典（在注意力块中的键和值）。用于快速自回归解码。预计算的键和值隐藏状态的形状为 *[batch_size, max_length]*。
        output_attentions (`bool`, *可选*):
            是否返回所有注意力层的注意力张量。有关返回张量的更多细节，请参见返回的张量下的 `attentions`。
        output_hidden_states (`bool`, *可选*):
            是否返回所有层的隐藏状态。有关返回张量的更多细节，请参见返回的张量下的 `hidden_states`。
        return_dict (`bool`, *可选*):
            是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""
# 将输入的 token ID 右移一个位置
def shift_tokens_right(input_ids: jnp.ndarray, pad_token_id: int, decoder_start_token_id: int) -> jnp.ndarray:
    """
    Shift input ids one token to the right.
    """
    # 创建与 input_ids 相同形状的全零数组
    shifted_input_ids = jnp.zeros_like(input_ids)
    # 将 input_ids 的每一行，从第二列开始到末尾的数据，复制到 shifted_input_ids 的对应位置
    shifted_input_ids = shifted_input_ids.at[:, 1:].set(input_ids[:, :-1])
    # 将每一行的第一列设为 decoder_start_token_id
    shifted_input_ids = shifted_input_ids.at[:, 0].set(decoder_start_token_id)

    # 将值为 -100 的位置（特殊标记），替换为 pad_token_id
    shifted_input_ids = jnp.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)
    return shifted_input_ids


# 从 transformers.models.marian.modeling_flax_marian.create_sinusoidal_positions 复制而来
# 创建一个正弦位置编码矩阵
def create_sinusoidal_positions(n_pos, dim):
    # 根据位置和维度生成一个正弦位置编码矩阵
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    sentinel = dim // 2 + dim % 2
    out = np.zeros_like(position_enc)
    # 将位置编码矩阵中偶数索引列的值设置为正弦值
    out[:, 0:sentinel] = np.sin(position_enc[:, 0::2])
    # 将位置编码矩阵中奇数索引列的值设置为余弦值
    out[:, sentinel:] = np.cos(position_enc[:, 1::2])

    return jnp.array(out)


# 从 transformers.models.bart.modeling_flax_bart.FlaxBartAttention 复制而来，将 Bart 替换为 Pegasus
# 定义 Pegasus 注意力机制的模块
class FlaxPegasusAttention(nn.Module):
    config: PegasusConfig
    embed_dim: int
    num_heads: int
    dropout: float = 0.0
    causal: bool = False
    bias: bool = True
    dtype: jnp.dtype = jnp.float32  # 计算时的数据类型

    def setup(self) -> None:
        # 计算每个头部的维度
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )

        # 定义用于计算的全连接层，初始化方式为正态分布
        dense = partial(
            nn.Dense,
            self.embed_dim,
            use_bias=self.bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )

        # 分别为查询、键、值和输出定义全连接层
        self.q_proj, self.k_proj, self.v_proj = dense(), dense(), dense()
        self.out_proj = dense()

        # 定义 dropout 层
        self.dropout_layer = nn.Dropout(rate=self.dropout)

        # 如果 causal 为 True，则创建一个因果遮罩
        if self.causal:
            self.causal_mask = make_causal_mask(
                jnp.ones((1, self.config.max_position_embeddings), dtype="bool"), dtype="bool"
            )

    # 将隐藏状态按头部数和头部维度进行分割
    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    # 将隐藏状态的头部合并回原始形状
    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slightly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # 检测是否初始化，通过检查缓存数据是否存在来判断
        is_initialized = self.has_variable("cache", "cached_key")
        # 获取或初始化缓存的键（key）和值（value）
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        # 获取或初始化缓存索引，用于追踪当前缓存的位置
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            # 获取当前缓存的形状信息，包括批次维度、最大长度、头数和每个头的深度
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # 使用新的一维空间切片更新键（key）和值（value）缓存
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            # 更新缓存索引，增加已更新的缓存向量数量
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # 为缓存的解码器自注意力生成因果遮罩：当前查询位置只能注意到已生成和缓存的键位置，而不是剩余的零元素。
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)
        
        # 返回更新后的键（key）、值（value）和注意力遮罩
        return key, value, attention_mask
# 从transformers.models.mbart.modeling_flax_mbart.FlaxMBartEncoderLayer复制而来，将MBart改为Pegasus
class FlaxPegasusEncoderLayer(nn.Module):
    # Pegasus模型的配置
    config: PegasusConfig
    # 计算中使用的数据类型，默认为32位浮点数
    dtype: jnp.dtype = jnp.float32

    # 设置方法，初始化编码器层的组件
    def setup(self) -> None:
        # 嵌入维度等于模型配置中的d_model参数
        self.embed_dim = self.config.d_model
        # Pegasus自注意力机制
        self.self_attn = FlaxPegasusAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.encoder_attention_heads,
            dropout=self.config.attention_dropout,
            dtype=self.dtype,
        )
        # 层归一化层，用于自注意力输出
        self.self_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        # 用于自注意力输出的Dropout层
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)
        # 激活函数，根据配置中的激活函数选择对应的激活函数
        self.activation_fn = ACT2FN[self.config.activation_function]
        # 激活函数后的Dropout层
        self.activation_dropout_layer = nn.Dropout(rate=self.config.activation_dropout)
        # 第一个全连接层，用于前馈神经网络
        self.fc1 = nn.Dense(
            self.config.encoder_ffn_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        # 第二个全连接层，用于前馈神经网络的输出
        self.fc2 = nn.Dense(
            self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std)
        )
        # 最终的层归一化层，用于前馈神经网络的输出
        self.final_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    # 调用方法，执行编码器层的前向计算
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        output_attentions: bool = True,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray]:
        # 保存残差连接的输入
        residual = hidden_states
        # 对输入进行自注意力输出的层归一化处理
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # 执行自注意力机制计算，并返回计算后的隐藏状态及注意力权重
        hidden_states, attn_weights = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask)
        # 应用Dropout层，以减少过拟合风险
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 添加残差连接
        hidden_states = residual + hidden_states

        # 保存残差连接的输入
        residual = hidden_states
        # 对输入进行最终输出的层归一化处理
        hidden_states = self.final_layer_norm(hidden_states)
        # 使用激活函数处理第一个全连接层的输出
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 应用激活函数后的Dropout层
        hidden_states = self.activation_dropout_layer(hidden_states, deterministic=deterministic)
        # 执行第二个全连接层的计算
        hidden_states = self.fc2(hidden_states)
        # 应用Dropout层，以减少过拟合风险
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 添加残差连接
        hidden_states = residual + hidden_states

        # 构建输出元组，包含最终的隐藏状态
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则添加到输出元组中
        if output_attentions:
            outputs += (attn_weights,)

        # 返回最终的输出元组
        return outputs


# 从transformers.models.bart.modeling_flax_bart.FlaxBartEncoderLayerCollection复制而来，将Bart改为Pegasus
class FlaxPegasusEncoderLayerCollection(nn.Module):
    # Pegasus模型的配置
    config: PegasusConfig
    # 计算中使用的数据类型，默认为32位浮点数
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    # 设置方法，初始化编码器层的集合
    def setup(self):
        # 创建编码器层的列表，每一层为FlaxPegasusEncoderLayer的实例
        self.layers = [
            FlaxPegasusEncoderLayer(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.encoder_layers)
        ]
        # 编码器层的Dropout概率，由配置文件中的encoder_layerdrop定义
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
        # 如果输出注意力权重，则初始化一个空元组用于存储所有注意力权重
        all_attentions = () if output_attentions else None
        # 如果输出隐藏状态，则初始化一个空元组用于存储所有隐藏状态
        all_hidden_states = () if output_hidden_states else None

        # 遍历每个编码器层
        for encoder_layer in self.layers:
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            # 使用 LayerDrop 方法来控制是否跳过当前层
            dropout_probability = random.uniform(0, 1)
            if not deterministic and (dropout_probability < self.layerdrop):  # 如果随机数小于 layerdrop，跳过当前层
                layer_outputs = (None, None)
            else:
                # 调用当前编码器层的前向传播方法
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

        # 如果需要输出隐藏状态，则将最终隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 构建模型输出对象，根据 return_dict 决定返回格式
        outputs = (hidden_states, all_hidden_states, all_attentions)

        # 如果 return_dict 为 False，则以元组形式返回 outputs 中非空的部分
        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        # 如果 return_dict 为 True，则构建 FlaxBaseModelOutput 对象返回
        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )
# 从transformers.models.mbart.modeling_flax_mbart.FlaxMBartDecoderLayer复制到Pegasus，定义了FlaxPegasusDecoderLayer类
class FlaxPegasusDecoderLayer(nn.Module):
    # 类变量：使用PegasusConfig配置
    config: PegasusConfig
    # 类变量：数据类型为jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 初始化方法，无返回值
    def setup(self) -> None:
        # 设置self.embed_dim为配置中的d_model值，即模型的维度大小
        self.embed_dim = self.config.d_model
        # 初始化self.self_attn为FlaxPegasusAttention对象，用于自注意力机制
        self.self_attn = FlaxPegasusAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.decoder_attention_heads,
            dropout=self.config.attention_dropout,
            causal=True,
            dtype=self.dtype,
        )
        # 初始化self.dropout_layer为Dropout层，用于随机失活
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)
        # 设置激活函数为配置中指定的激活函数，并初始化激活函数的随机失活层
        self.activation_fn = ACT2FN[self.config.activation_function]
        self.activation_dropout_layer = nn.Dropout(rate=self.config.activation_dropout)

        # 初始化self.self_attn_layer_norm为LayerNorm层，用于自注意力机制的归一化
        self.self_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        # 初始化self.encoder_attn为FlaxPegasusAttention对象，用于编码器注意力机制
        self.encoder_attn = FlaxPegasusAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.decoder_attention_heads,
            dropout=self.config.attention_dropout,
            dtype=self.dtype,
        )
        # 初始化self.encoder_attn_layer_norm为LayerNorm层，用于编码器注意力机制的归一化
        self.encoder_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        # 初始化self.fc1为全连接层，用于第一个前馈神经网络
        self.fc1 = nn.Dense(
            self.config.decoder_ffn_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        # 初始化self.fc2为全连接层，用于第二个前馈神经网络
        self.fc2 = nn.Dense(
            self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std)
        )
        # 初始化self.final_layer_norm为LayerNorm层，用于最终的归一化
        self.final_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    # 调用方法，定义了层的前向传播逻辑
    def __call__(
        self,
        hidden_states: jnp.ndarray,  # 输入的隐藏状态张量
        attention_mask: jnp.ndarray,  # 注意力掩码张量
        encoder_hidden_states: Optional[jnp.ndarray] = None,  # 编码器的隐藏状态张量（可选）
        encoder_attention_mask: Optional[jnp.ndarray] = None,  # 编码器的注意力掩码张量（可选）
        init_cache: bool = False,  # 是否初始化缓存（默认为False）
        output_attentions: bool = True,  # 是否输出注意力权重（默认为True）
        deterministic: bool = True,  # 是否使用确定性计算（默认为True）

        # 方法开始
        # 返回self.self_attn的前向传播结果，对输入的hidden_states进行自注意力计算
        # 返回值包括输出张量以及注意力权重（如果output_attentions为True）
        return self.self_attn(
            hidden_states,
            attention_mask,
            init_cache=init_cache,
            output_attentions=output_attentions,
            deterministic=deterministic,
        )
        ) -> Tuple[jnp.ndarray]:
        # 将输入的 hidden_states 复制给 residual，用于后续的残差连接
        residual = hidden_states
        # 对 hidden_states 进行 Layer Normalization
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        # 使用 self_attn 层处理 hidden_states，包括注意力计算和可能的缓存初始化
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, init_cache=init_cache
        )
        # 应用 dropout 层
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 添加残差连接
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        cross_attn_weights = None
        # 如果提供了 encoder_hidden_states，则执行以下操作
        if encoder_hidden_states is not None:
            # 将输入的 hidden_states 复制给 residual，用于后续的残差连接
            residual = hidden_states

            # 对 hidden_states 进行 Layer Normalization
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            # 使用 encoder_attn 层处理 hidden_states 和 encoder_hidden_states
            # 包括注意力计算和可能的 attention_mask 应用
            hidden_states, cross_attn_weights = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            # 应用 dropout 层
            hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
            # 添加残差连接
            hidden_states = residual + hidden_states

        # Fully Connected
        # 将输入的 hidden_states 复制给 residual，用于后续的残差连接
        residual = hidden_states
        # 对 hidden_states 进行 Layer Normalization
        hidden_states = self.final_layer_norm(hidden_states)
        # 应用激活函数 activation_fn
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 应用 activation_dropout_layer
        hidden_states = self.activation_dropout_layer(hidden_states, deterministic=deterministic)
        # 应用全连接层 fc2
        hidden_states = self.fc2(hidden_states)
        # 应用 dropout 层
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 添加残差连接
        hidden_states = residual + hidden_states

        # 准备输出，初始化为包含 hidden_states 的元组 outputs
        outputs = (hidden_states,)

        # 如果需要输出 attention weights，则将 self_attn_weights 和 cross_attn_weights 添加到 outputs 中
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        # 返回 outputs
        return outputs
# 从transformers.models.bart.modeling_flax_bart.FlaxBartDecoderLayerCollection复制的代码，将Bart更改为Pegasus
class FlaxPegasusDecoderLayerCollection(nn.Module):
    # Pegasus模型的配置
    config: PegasusConfig
    # 计算的数据类型
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型为浮点数（32位）

    def setup(self):
        # 创建Pegasus解码器层的集合
        self.layers = [
            FlaxPegasusDecoderLayer(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.decoder_layers)
        ]
        # 解码器层的LayerDrop概率
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
        # 存储所有隐藏状态（如果需要返回）
        all_hidden_states = () if output_hidden_states else None
        # 存储所有自注意力权重（如果需要返回）
        all_self_attns = () if output_attentions else None
        # 存储所有跨注意力权重（如果需要返回），仅在同时输出注意力并且存在编码器隐藏状态时才存储
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        # 对每个解码器层进行迭代
        for decoder_layer in self.layers:
            if output_hidden_states:
                # 如果需要输出隐藏状态，则将当前隐藏状态添加到存储中
                all_hidden_states += (hidden_states,)
                # 添加LayerDrop（参见https://arxiv.org/abs/1909.11556进行描述）

            # 随机采样一个Dropout概率
            dropout_probability = random.uniform(0, 1)
            # 如果是非确定性计算并且随机概率小于LayerDrop阈值，则将输出设为None
            if not deterministic and (dropout_probability < self.layerdrop):
                layer_outputs = (None, None, None)
            else:
                # 否则，调用当前解码器层进行前向传播计算
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
            # 如果需要输出注意力权重，则将当前解码器层的自注意力权重添加到存储中
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                # 如果存在编码器隐藏状态，则将当前解码器层的跨注意力权重添加到存储中
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # 如果需要输出最终隐藏状态，则将最终隐藏状态添加到存储中
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 将所有需要输出的结果存储在outputs列表中
        outputs = [hidden_states, all_hidden_states, all_self_attns, all_cross_attentions]

        # 如果不需要以字典形式返回结果，则返回输出列表中的非None元素
        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        # 以包含过去和跨注意力权重的形式返回FlaxBaseModelOutputWithPastAndCrossAttentions对象
        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class FlaxPegasusEncoder(nn.Module):
    # Pegasus模型的配置
    config: PegasusConfig
    # 嵌入令牌的层
    embed_tokens: nn.Embed
    # 计算的数据类型
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型为浮点数（32位）
    # 在类初始化方法中设置dropout层，根据配置中的dropout率创建实例
    def setup(self):
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)

        # 从配置中获取词嵌入的维度
        embed_dim = self.config.d_model
        # 从配置中获取填充标记的索引
        self.padding_idx = self.config.pad_token_id
        # 从配置中获取源序列的最大位置数
        self.max_source_positions = self.config.max_position_embeddings
        # 如果配置中设置了缩放词嵌入，则计算缩放因子为词嵌入维度的平方根，否则为1.0
        self.embed_scale = math.sqrt(embed_dim) if self.config.scale_embedding else 1.0

        # 创建正弦位置编码，并赋值给self.embed_positions
        self.embed_positions = create_sinusoidal_positions(self.config.max_position_embeddings, embed_dim)
        # 创建FlaxPegasusEncoderLayerCollection实例，用于后续编码器层的处理
        self.layers = FlaxPegasusEncoderLayerCollection(self.config, self.dtype)
        # 创建LayerNorm实例，用于对隐藏状态进行归一化处理
        self.layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    # 在实例调用时处理输入数据，执行编码器操作
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
        # 获取输入数据的形状信息
        input_shape = input_ids.shape
        # 将input_ids重新reshape为二维张量
        input_ids = input_ids.reshape(-1, input_shape[-1])

        # 使用嵌入词表将input_ids转换为嵌入向量，并乘以缩放因子
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # 嵌入位置信息
        embed_pos = jnp.take(self.embed_positions, position_ids, axis=0)
        # 显式地将位置信息embed_pos转换为与inputs_embeds相同的数据类型
        embed_pos = embed_pos.astype(inputs_embeds.dtype)

        # 将输入嵌入向量与位置嵌入向量相加形成隐藏状态
        hidden_states = inputs_embeds + embed_pos
        # 对隐藏状态应用dropout操作
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 将隐藏状态传递给编码器层进行处理
        outputs = self.layers(
            hidden_states,
            attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取编码器层输出中的最后一个隐藏状态
        last_hidden_state = outputs[0]
        # 对最后一个隐藏状态应用LayerNorm归一化
        last_hidden_state = self.layer_norm(last_hidden_state)

        # 如果需要输出所有隐藏状态，则更新outputs中的hidden_states
        hidden_states = None
        if output_hidden_states:
            hidden_states = outputs[1]
            hidden_states = hidden_states[:-1] + (last_hidden_state,)

        # 如果不返回字典，则根据需要重新组织输出的元组
        if not return_dict:
            outputs = (last_hidden_state, hidden_states) + (outputs[2:] if output_hidden_states else outputs[1:])
            return tuple(v for v in outputs if v is not None)

        # 返回FlaxBaseModelOutput对象，包括最后一个隐藏状态、所有隐藏状态和注意力权重（如果有）
        return FlaxBaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
        )
class FlaxPegasusDecoder(nn.Module):
    config: PegasusConfig  # Pegasus model configuration
    embed_tokens: nn.Embed  # Embedding tokens for input sequence
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)  # Dropout layer initialization

        embed_dim = self.config.d_model  # Dimension of the embedding
        self.padding_idx = self.config.pad_token_id  # Padding token index from configuration
        self.max_target_positions = self.config.max_position_embeddings  # Maximum target positions
        self.embed_scale = math.sqrt(self.config.d_model) if self.config.scale_embedding else 1.0  # Embedding scale factor

        self.embed_positions = create_sinusoidal_positions(self.config.max_position_embeddings, embed_dim)
        # Create sinusoidal positional embeddings

        self.layers = FlaxPegasusDecoderLayerCollection(self.config, self.dtype)
        # Layers of the Pegasus decoder
        self.layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        # Layer normalization initialization

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
        input_shape = input_ids.shape
        input_ids = input_ids.reshape(-1, input_shape[-1])
        # Reshape input_ids to flatten the sequence dimensions

        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        # Embed input tokens and scale embeddings

        # embed positions
        positions = jnp.take(self.embed_positions, position_ids, axis=0)
        # Retrieve positional embeddings based on position_ids
        positions = positions.astype(inputs_embeds.dtype)
        # Explicitly cast positions to match inputs_embeds dtype

        hidden_states = inputs_embeds + positions
        # Combine token embeddings with positional embeddings
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # Apply dropout to hidden states

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
        # Pass hidden states through decoder layers

        last_hidden_state = outputs[0]
        # Retrieve the last hidden state from the outputs
        last_hidden_state = self.layer_norm(last_hidden_state)
        # Apply layer normalization to the last hidden state

        hidden_states = None
        if output_hidden_states:
            hidden_states = outputs[1]
            hidden_states = hidden_states[:-1] + (last_hidden_state,)
            # Concatenate previous hidden states with the current one

        if not return_dict:
            outputs = (last_hidden_state, hidden_states) + (outputs[2:] if output_hidden_states else outputs[1:])
            return tuple(v for v in outputs if v is not None)
            # Return outputs as a tuple without the return_dict format

        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=last_hidden_state,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )
        # Return outputs as FlaxBaseModelOutputWithPastAndCrossAttentions object
# 从 transformers.models.bart.modeling_flax_bart.FlaxBartModule 复制并修改为 Pegasus
class FlaxPegasusModule(nn.Module):
    config: PegasusConfig  # Pegasus 模型的配置对象
    dtype: jnp.dtype = jnp.float32  # 计算时使用的数据类型

    def setup(self):
        # 创建共享的嵌入层，用于编码器和解码器
        self.shared = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.init_std),  # 使用正态分布初始化嵌入层
            dtype=self.dtype,
        )

        # 初始化编码器和解码器
        self.encoder = FlaxPegasusEncoder(self.config, dtype=self.dtype, embed_tokens=self.shared)
        self.decoder = FlaxPegasusDecoder(self.config, dtype=self.dtype, embed_tokens=self.shared)

    def _get_encoder_module(self):
        return self.encoder  # 返回编码器模块

    def _get_decoder_module(self):
        return self.decoder  # 返回解码器模块

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
        # 调用编码器得到编码器的输出
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        # 调用解码器得到解码器的输出，传入编码器的隐藏状态和注意力掩码
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
            return decoder_outputs + encoder_outputs  # 如果不返回字典，则返回所有输出

        # 返回序列到序列模型的输出
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
    config_class = PegasusConfig  # Pegasus 预训练模型的配置类
    base_model_prefix: str = "model"  # 基础模型的前缀名称
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
        # 使用配置和数据类型初始化模块对象
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 调用父类的构造函数，传入配置、模块对象、输入形状、种子、数据类型和初始化标志
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量
        input_ids = jnp.zeros(input_shape, dtype="i4")
        # 创建与 input_ids 形状相同的全 1 张量作为注意力掩码
        attention_mask = jnp.ones_like(input_ids)
        # 将 decoder_input_ids 初始化为与 input_ids 相同的张量
        decoder_input_ids = input_ids
        # 创建与 input_ids 形状相同的全 1 张量作为解码器的注意力掩码
        decoder_attention_mask = jnp.ones_like(input_ids)

        # 获取 batch_size 和 sequence_length
        batch_size, sequence_length = input_ids.shape
        # 创建位置编码，将序列长度广播到每个样本的每个位置
        position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))
        # 解码器的位置编码与编码器相同
        decoder_position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # 分割随机数生成器 rng 为 params_rng 和 dropout_rng
        params_rng, dropout_rng = jax.random.split(rng)
        # 创建随机数字典 rngs，包含 params_rng 和 dropout_rng
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 使用模块对象的初始化方法初始化模型参数
        random_params = self.module.init(
            rngs,
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            position_ids,
            decoder_position_ids,
        )["params"]

        # 如果提供了初始参数 params，则使用它替换随机初始化的部分参数
        if params is not None:
            # 展平并解冻参数字典
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            # 将缺失的参数键从随机参数复制到提供的参数中
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()  # 清空缺失键集合
            # 冻结并返回更新后的参数字典
            return freeze(unflatten_dict(params))
        else:
            # 如果未提供初始参数，则直接返回随机初始化的参数字典
            return random_params
    def init_cache(self, batch_size, max_length, encoder_outputs):
        r"""
        Args:
            batch_size (`int`):
                fast auto-regressive decoding 使用的 batch_size。定义了初始化缓存时的批处理大小。
            max_length (`int`):
                自动回归解码的最大可能长度。定义了初始化缓存时的序列长度。
            encoder_outputs (`Union[FlaxBaseModelOutput, tuple(tuple(jnp.ndarray)]`):
                `encoder_outputs` 包含 (`last_hidden_state`, *可选*: `hidden_states`, *可选*: `attentions`)。
                `last_hidden_state` 的形状为 `(batch_size, sequence_length, hidden_size)`，*可选*: 编码器最后一层输出的隐藏状态。
                在解码器的交叉注意力中使用。

        初始化缓存函数，用于预先设置解码器的缓存状态。

        """
        # 初始化解码器的输入标识，全部为1的数组，形状为 (batch_size, max_length)
        decoder_input_ids = jnp.ones((batch_size, max_length), dtype="i4")
        # 初始化解码器的注意力掩码，与 decoder_input_ids 形状相同的全1数组
        decoder_attention_mask = jnp.ones_like(decoder_input_ids)
        # 初始化解码器的位置标识，将一个广播数组设置为与 decoder_input_ids 形状相同的位置标识
        decoder_position_ids = jnp.broadcast_to(
            jnp.arange(jnp.atleast_2d(decoder_input_ids).shape[-1]), decoder_input_ids.shape
        )

        # 定义内部函数 _decoder_forward，用于调用解码器模块并返回结果
        def _decoder_forward(module, decoder_input_ids, decoder_attention_mask, decoder_position_ids, **kwargs):
            decoder_module = module._get_decoder_module()
            return decoder_module(
                decoder_input_ids,
                decoder_attention_mask,
                decoder_position_ids,
                **kwargs,
            )

        # 使用模型的初始化方法初始化变量，并设置解码器相关参数
        init_variables = self.module.init(
            jax.random.PRNGKey(0),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            encoder_hidden_states=encoder_outputs[0],
            init_cache=True,
            method=_decoder_forward,  # 仅需调用解码器以初始化缓存
        )
        # 返回解除冻结后的变量中的缓存部分
        return unfreeze(init_variables["cache"])
    ):
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, FlaxPegasusForConditionalGeneration

        >>> model = FlaxPegasusForConditionalGeneration.from_pretrained("google/pegasus-large")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/pegasus-large")

        >>> text = "My friends are cool but they eat too many carbs."
        >>> inputs = tokenizer(text, max_length=1024, return_tensors="np")
        >>> encoder_outputs = model.encode(**inputs)
        ```"""
        # 根据传入的参数设置输出注意力机制
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 根据传入的参数设置输出隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 根据传入的参数设置返回字典类型
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 如果 attention_mask 为 None，则创建一个全为1的掩码与 input_ids 形状相同
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        # 如果 position_ids 为 None，则根据 input_ids 形状创建对应的位置ID张量
        if position_ids is None:
            batch_size, sequence_length = input_ids.shape
            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # 如果 dropout_rng 不为 None，则将其作为 "dropout" 的随机数生成器
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        # 定义一个内部函数 _encoder_forward，用于调用编码器模块的前向方法
        def _encoder_forward(module, input_ids, attention_mask, position_ids, **kwargs):
            encode_module = module._get_encoder_module()
            return encode_module(input_ids, attention_mask, position_ids, **kwargs)

        # 调用 self.module.apply 方法执行模型的前向传播
        return self.module.apply(
            {"params": params or self.params},  # 使用传入的参数或者默认参数来执行前向传播
            input_ids=jnp.array(input_ids, dtype="i4"),  # 将 input_ids 转换为 JAX 数组
            attention_mask=jnp.array(attention_mask, dtype="i4"),  # 将 attention_mask 转换为 JAX 数组
            position_ids=jnp.array(position_ids, dtype="i4"),  # 将 position_ids 转换为 JAX 数组
            output_attentions=output_attentions,  # 控制是否输出注意力机制
            output_hidden_states=output_hidden_states,  # 控制是否输出隐藏状态
            return_dict=return_dict,  # 控制是否以字典形式返回结果
            deterministic=not train,  # 是否处于训练模式
            rngs=rngs,  # 随机数生成器的字典
            method=_encoder_forward,  # 指定执行的方法
        )

    @add_start_docstrings(PEGASUS_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutputWithPastAndCrossAttentions, config_class=PegasusConfig)
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
    @add_start_docstrings_to_model_forward(PEGASUS_INPUTS_DOCSTRING)
    # 定义一个 __call__ 方法，使对象可以像函数一样被调用，接收以下参数：
    #   - input_ids: 输入的编码序列，类型为 jnp.ndarray
    #   - attention_mask: 可选参数，注意力掩码，默认为 None
    #   - decoder_input_ids: 可选参数，解码器的输入编码序列，默认为 None
    #   - decoder_attention_mask: 可选参数，解码器的注意力掩码，默认为 None
    #   - position_ids: 可选参数，位置编码序列，默认为 None
    #   - decoder_position_ids: 可选参数，解码器的位置编码序列，默认为 None
    #   - output_attentions: 可选参数，是否输出注意力权重，默认为 None
    #   - output_hidden_states: 可选参数，是否输出隐藏状态，默认为 None
    #   - return_dict: 可选参数，是否返回字典格式的结果，默认为 None
    #   - train: 是否处于训练模式，默认为 False
    #   - params: 可选参数，模型的参数，默认为 None
    #   - dropout_rng: 可选参数，随机数生成器用于 dropout，默认为 None

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    # 如果 output_attentions 不为 None，则使用该值；否则使用 self.config.output_attentions 的值

    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    # 如果 output_hidden_states 不为 None，则使用该值；否则使用 self.config.output_hidden_states 的值

    return_dict = return_dict if return_dict is not None else self.config.return_dict
    # 如果 return_dict 不为 None，则使用该值；否则使用 self.config.return_dict 的值

    # 准备编码器的输入
    if attention_mask is None:
        attention_mask = jnp.ones_like(input_ids)
    # 如果 attention_mask 为 None，则创建一个与 input_ids 形状相同的全为 1 的注意力掩码

    if position_ids is None:
        batch_size, sequence_length = input_ids.shape
        position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))
    # 如果 position_ids 为 None，则根据 input_ids 的形状创建位置编码序列

    # 准备解码器的输入
    if decoder_input_ids is None:
        decoder_input_ids = shift_tokens_right(
            input_ids, self.config.pad_token_id, decoder_start_token_id=self.config.decoder_start_token_id
        )
    # 如果 decoder_input_ids 为 None，则将 input_ids 向右移动一个位置，并使用配置中的特殊标记进行填充

    if decoder_attention_mask is None:
        decoder_attention_mask = jnp.ones_like(decoder_input_ids)
    # 如果 decoder_attention_mask 为 None，则创建一个与 decoder_input_ids 形状相同的全为 1 的注意力掩码

    if decoder_position_ids is None:
        batch_size, sequence_length = decoder_input_ids.shape
        decoder_position_ids = jnp.broadcast_to(
            jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
        )
    # 如果 decoder_position_ids 为 None，则根据 decoder_input_ids 的形状创建位置编码序列

    # 处理可能需要的任何随机数生成器
    rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}
    # 如果 dropout_rng 不为 None，则创建一个包含 dropout_rng 的随机数生成器字典；否则创建一个空字典

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
    # 应用 self.module 中的函数：
    #   - 使用 params 或 self.params 中的参数
    #   - 将输入转换为 jnp.ndarray 格式并传递给相应参数
    #   - 设置是否输出注意力权重和隐藏状态
    #   - 设置是否以字典格式返回结果
    #   - 设置是否处于确定性计算模式
    #   - 传递随机数生成器字典 rngs
# 使用装饰器为 FlaxPegasusModel 类添加文档字符串，描述其作为 Pegasus 模型的基本转换器，输出原始隐藏状态而无顶部特定头部。
# PEGASUS_START_DOCSTRING 中包含 Pegasus 模型的起始文档字符串。
@add_start_docstrings(
    "The bare Pegasus Model transformer outputting raw hidden-states without any specific head on top.",
    PEGASUS_START_DOCSTRING,
)
# 定义 FlaxPegasusModel 类，继承自 FlaxPegasusPreTrainedModel，具有 PegasusConfig 类型的配置参数。
class FlaxPegasusModel(FlaxPegasusPreTrainedModel):
    config: PegasusConfig
    # 计算中使用的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 模型类别设置为 FlaxPegasusModule
    module_class = FlaxPegasusModule

# 调用函数，为 FlaxPegasusModel 类附加示例调用文档字符串，使用 _CHECKPOINT_FOR_DOC 和 FlaxSeq2SeqModelOutput。
append_call_sample_docstring(FlaxPegasusModel, _CHECKPOINT_FOR_DOC, FlaxSeq2SeqModelOutput, _CONFIG_FOR_DOC)


# 从 transformers.models.bart.modeling_flax_bart.FlaxBartForConditionalGenerationModule 复制代码，修改为 Pegasus 模型
class FlaxPegasusForConditionalGenerationModule(nn.Module):
    config: PegasusConfig
    dtype: jnp.dtype = jnp.float32
    # 偏置初始化器，使用 jax.nn.initializers.zeros 初始化
    bias_init: Callable[..., jnp.ndarray] = jax.nn.initializers.zeros

    # 模块设置方法
    def setup(self):
        # 创建 FlaxPegasusModule 模型对象，使用配置和数据类型作为参数
        self.model = FlaxPegasusModule(config=self.config, dtype=self.dtype)
        # 创建 lm_head 层，使用 nn.Dense 定义，输出维度为 self.model.shared.num_embeddings
        self.lm_head = nn.Dense(
            self.model.shared.num_embeddings,
            use_bias=False,
            dtype=self.dtype,
            # 使用正态分布初始化 kernel 参数，标准差为 self.config.init_std
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        # 定义 final_logits_bias 参数，形状为 (1, self.model.shared.num_embeddings)，使用 bias_init 初始化
        self.final_logits_bias = self.param("final_logits_bias", self.bias_init, (1, self.model.shared.num_embeddings))

    # 获取编码器模块的方法
    def _get_encoder_module(self):
        return self.model.encoder

    # 获取解码器模块的方法
    def _get_decoder_module(self):
        return self.model.decoder

    # 调用方法，定义模型的前向传播逻辑
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
        # 声明输入参数的类型和默认值
        **kwargs
    ):
    ):
        # 使用模型生成输出结果
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

        # 如果配置要求共享词嵌入，则使用共享的嵌入层进行计算
        if self.config.tie_word_embeddings:
            shared_embedding = self.model.variables["params"]["shared"]["embedding"]
            # 应用共享的嵌入层权重到隐藏状态，得到语言模型的logits
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
        else:
            # 否则直接使用语言模型头部计算logits
            lm_logits = self.lm_head(hidden_states)

        # 添加最终logits的偏置
        lm_logits += jax.lax.stop_gradient(self.final_logits_bias.astype(self.dtype))

        # 如果不要求返回字典形式的输出，则返回完整的输出元组
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return output

        # 否则，返回FlaxSeq2SeqLMOutput类型的对象，包含完整的输出信息
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
    "The PEGASUS Model with a language modeling head. Can be used for summarization.", PEGASUS_START_DOCSTRING
)
class FlaxPegasusForConditionalGeneration(FlaxPegasusPreTrainedModel):
    module_class = FlaxPegasusForConditionalGenerationModule
    dtype: jnp.dtype = jnp.float32

    @add_start_docstrings(PEGASUS_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxCausalLMOutputWithCrossAttentions, config_class=PegasusConfig)
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
    ):
        """
        Decode function for PEGASUS model, generating outputs based on decoder inputs and encoder outputs.

        Args:
            decoder_input_ids: Input IDs for the decoder.
            encoder_outputs: Outputs from the encoder.
            encoder_attention_mask: Optional attention mask for encoder outputs.
            decoder_attention_mask: Optional attention mask for decoder inputs.
            decoder_position_ids: Optional position IDs for the decoder inputs.
            past_key_values: Cached key values from previous decoding steps.
            output_attentions: Whether to output attention weights.
            output_hidden_states: Whether to output hidden states.
            return_dict: Whether to return outputs as a dictionary.
            deterministic: Whether to use deterministic behavior.
            params: Optional parameters for decoding.
            dropout_rng: Random number generator for dropout.

        Returns:
            Model outputs with cross attentions, conforming to PEGASUS configuration.
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
        Prepare inputs for generation based on decoder inputs and optional masks.

        Args:
            decoder_input_ids: Input IDs for the decoder.
            max_length: Maximum length of generated outputs.
            attention_mask: Optional attention mask for encoder outputs.
            decoder_attention_mask: Optional attention mask for decoder inputs.
            encoder_outputs: Optional outputs from the encoder.
            **kwargs: Additional keyword arguments.

        Returns:
            Dictionary of inputs formatted for generation.
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
        Update inputs for generation based on model outputs and current model arguments.

        Args:
            model_outputs: Outputs from the model.
            model_kwargs: Current model keyword arguments.

        Returns:
            Updated model keyword arguments for generation.
        """
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["decoder_position_ids"] = model_kwargs["decoder_position_ids"][:, -1:] + 1
        return model_kwargs
    >>> model = FlaxPegasusForConditionalGeneration.from_pretrained('google/pegasus-large')
    >>> tokenizer = AutoTokenizer.from_pretrained('google/pegasus-large')
    
    >>> ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
    >>> inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='np')
    
    >>> # 生成摘要
    >>> summary_ids = model.generate(inputs['input_ids']).sequences
    >>> print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))
    
    
    Mask filling example:
    
    
    >>> from transformers import AutoTokenizer, FlaxPegasusForConditionalGeneration
    
    >>> tokenizer = AutoTokenizer.from_pretrained("google/pegasus-large")
    >>> TXT = "My friends are <mask> but they eat too many carbs."
    
    >>> model = FlaxPegasusForConditionalGeneration.from_pretrained("google/pegasus-large")
    >>> input_ids = tokenizer([TXT], return_tensors="np")["input_ids"]
    >>> logits = model(input_ids).logits
    
    >>> # 获取掩码位置的索引
    >>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
    >>> # 对预测的概率进行 softmax 处理
    >>> probs = jax.nn.softmax(logits[0, masked_index], axis=0)
    >>> # 获取最高概率的值和预测的索引
    >>> values, predictions = jax.lax.top_k(probs)
    
    >>> # 解码预测的词语并以列表形式返回
    >>> tokenizer.decode(predictions).split()
"""
为FlaxPegasusForConditionalGeneration类的文档字符串添加内容
使用 PEGASUS_INPUTS_DOCSTRING 和 FLAX_PEGASUS_CONDITIONAL_GENERATION_DOCSTRING 进行覆盖
"""
overwrite_call_docstring(
    FlaxPegasusForConditionalGeneration, PEGASUS_INPUTS_DOCSTRING + FLAX_PEGASUS_CONDITIONAL_GENERATION_DOCSTRING
)

"""
为FlaxPegasusForConditionalGeneration类的返回文档字符串追加内容
使用 FlaxSeq2SeqLMOutput 作为输出类型，使用 _CONFIG_FOR_DOC 作为配置类
"""
append_replace_return_docstrings(
    FlaxPegasusForConditionalGeneration, output_type=FlaxSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC
)
```