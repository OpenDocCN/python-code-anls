# `.\transformers\models\marian\modeling_flax_marian.py`

```
# 设置文件编码为 utf-8
# 版权声明，版权归 The Marian Team 作者、The Google Flax Team 作者以及 HuggingFace Inc. 团队所有
# 使用 Apache License, Version 2.0 开源许可证，遵循许可证规定使用本文件
# 可以在以下链接找到开源许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，按“原样”分发软件，不附带任何明示或默示的担保或条件
# 查看相关语言的许可证以了解特定语言的规定和限制
""" Flax Marian 模型。

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
    该模型继承自 [`FlaxPreTrainedModel`]。查看父类文档以了解库实现的通用方法（如下载或保存、调整输入嵌入、修剪注意力头等）。

    该模型还是一个 Flax Linen [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) 子类。
    可以将其用作常规的 Flax 模块，并参考 Flax 文档了解与一般用法和行为相关的所有事宜。

    最后，该模型支持 JAX 的固有功能，如：

    - [即时（JIT）编译](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [自动微分](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [向量化](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [并行化](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

``` 
    Parameters:
        config ([`MarianConfig`]): Model configuration class with all the parameters of the model.
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
# 定义MARIAN模型的输入文档字符串
MARIAN_INPUTS_DOCSTRING = r"""
"""

# 定义MARIAN编码输入的文档字符串
MARIAN_ENCODE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
            输入序列的词汇表索引。默认情况下会忽略填充。

            索引可以使用 [`AutoTokenizer`] 获得。参见 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`] 了解详情。

            [什么是输入ID?](../glossary#input-ids)
        attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            用于避免对填充标记进行注意力计算的掩码。掩码值在 `[0, 1]` 中选择:

            - 1 表示 **未被遮蔽** 的令牌，
            - 0 表示 **被遮蔽** 的令牌。

            [什么是注意力掩码?](../glossary#attention-mask)
        position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            每个输入序列中的位置嵌入索引。从 `0` 到 `config.max_position_embeddings - 1` 中进行选择。
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。更多细节参见返回张量中的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。更多细节参见返回张量中的 `hidden_states`。
        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""

# 定义MARIAN解码输入的文档字符串
MARIAN_DECODE_INPUTS_DOCSTRING = r"""
"""

# 创建正弦位置编码
def create_sinusoidal_positions(n_pos, dim):
    # 计算位置编码矩阵
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    # 分割矩阵
    sentinel = dim // 2 + dim % 2
    out = np.zeros_like(position_enc)
    # 计算正弦和余弦部分
    out[:, 0:sentinel] = np.sin(position_enc[:, 0::2])
    out[:, sentinel:] = np.cos(position_enc[:, 1::2])
    # 转换为jnp.array格式
    return jnp.array(out)

# 向右移动输入ID一个token
def shift_tokens_right(input_ids: jnp.ndarray, pad_token_id: int, decoder_start_token_id: int) -> jnp.ndarray:
    """
    向右移动输入ID一个token。
    """
    # 创建一个全零的新数组
    shifted_input_ids = jnp.zeros_like(input_ids)
    # 将除第一个token外的所有token向右移动一位
    shifted_input_ids = shifted_input_ids.at[:, 1:].set(input_ids[:, :-1])
    # 将第一个token设置为decoder_start_token_id
    shifted_input_ids = shifted_input_ids.at[:, 0].set(decoder_start_token_id)
    # 将-100替换为pad_token_id
    shifted_input_ids = jnp.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)
    return shifted_input_ids

# 定义FlaxMarianAttention类
class FlaxMarianAttention(nn.Module):
    config: MarianConfig
    embed_dim: int
    num_heads: int
    dropout: float = 0.0
    causal: bool = False
    bias: bool = True
    # 定义变量 dtype，并指定其类型为 jnp.float32，表示计算过程中的数据类型
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    
    # 定义 setup 方法，用于初始化模型参数和设置相关属性
    def setup(self) -> None:
        # 计算每个头部的维度
        self.head_dim = self.embed_dim // self.num_heads
        # 检查 embed_dim 是否能被 num_heads 整除，若不能则抛出 ValueError 异常
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
    
        # 定义一个偏函数 dense，用于创建具有相同参数的全连接层
        dense = partial(
            nn.Dense,
            self.embed_dim,
            use_bias=self.bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
    
        # 初始化查询、键、值和输出投影层
        self.q_proj, self.k_proj, self.v_proj = dense(), dense(), dense()
        self.out_proj = dense()
    
        # 定义一个丢弃层，用于在计算过程中进行随机丢弃
        self.dropout_layer = nn.Dropout(rate=self.dropout)
    
        # 如果启用了因果注意力，创建一个因果注意力掩码
        if self.causal:
            self.causal_mask = make_causal_mask(
                jnp.ones((1, self.config.max_position_embeddings), dtype="bool"), dtype="bool"
            )
    
    # 将隐藏状态张量按头数分割成多个小张量
    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))
    
    # 将分割后的小张量合并成原始形状的隐藏状态张量
    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))
    
    # 使用 JAX 的 nn.compact 装饰器将下面的函数定义为一个紧凑形式的神经网络模块
    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slighly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # 检测是否通过缺少现有缓存数据进行初始化。
        is_initialized = self.has_variable("cache", "cached_key")

        # 如果已初始化，则获取缓存的键和值
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            # 获取缓存键的形状信息
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # 使用新的1D空间切片更新键和值缓存
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            # 更新缓存键和值
            cached_key.value = key
            cached_value.value = value
            # 更新缓存向量数
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # 对已缓存的解码器自注意力使用因果掩码: 我们的单个查询位置应该只与已生成和缓存的键位置发生关联，而不是剩余的零元素。
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)

        # 返回更新后的键、值和注意力掩码
        return key, value, attention_mask

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        key_value_states: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        deterministic: bool = True,
        # 此处省略了函数的其他逻辑
# 从transformers.models.bart.modeling_flax_bart.FlaxBartEncoderLayer复制并修改为Marian
class FlaxMarianEncoderLayer(nn.Module):
    # Marian配置
    config: MarianConfig
    # 计算中使用的数据类型，默认为32位浮点数
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        # 编码器的嵌入维度等于模型配置中的d_model
        self.embed_dim = self.config.d_model
        # 使用MarianAttention创建自注意力层
        self.self_attn = FlaxMarianAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.encoder_attention_heads,
            dropout=self.config.attention_dropout,
            dtype=self.dtype,
        )
        # 自注意力层的LayerNorm
        self.self_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        # 用于dropout的层
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)
        # 激活函数
        self.activation_fn = ACT2FN[self.config.activation_function]
        # 激活函数后的dropout
        self.activation_dropout_layer = nn.Dropout(rate=self.config.activation_dropout)
        # 全连接层1
        self.fc1 = nn.Dense(
            self.config.encoder_ffn_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        # 全连接层2
        self.fc2 = nn.Dense(
            self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std)
        )
        # 最终的LayerNorm
        self.final_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        output_attentions: bool = True,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray]:
        # 备份输入hidden_states
        residual = hidden_states
        # 计算自注意力层输出和注意力权重
        hidden_states, attn_weights = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask)

        # 应用dropout
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 添加残差连接
        hidden_states = residual + hidden_states
        # 应用自注意力层的LayerNorm
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 进行激活函数计算，然后应用全连接层1，并添加dropout
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.activation_dropout_layer(hidden_states, deterministic=deterministic)
        # 应用全连接层2
        hidden_states = self.fc2(hidden_states)
        # 添加dropout
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 添加残差连接
        hidden_states = residual + hidden_states
        # 最终的LayerNorm
        hidden_states = self.final_layer_norm(hidden_states)

        # 返回输出结果
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，添加到输出结果中
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

# 从transformers.models.bart.modeling_flax_bart.FlaxBartEncoderLayerCollection复制并修改为Marian
class FlaxMarianEncoderLayerCollection(nn.Module):
    # Marian配置
    config: MarianConfig
    # 计算中使用的数据类型，默认为32位浮点数
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        # 创建Marian编码器层的集合
        self.layers = [
            FlaxMarianEncoderLayer(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.encoder_layers)
        ]
        # 编码器层的随机丢弃率
        self.layerdrop = self.config.encoder_layerdrop
    # 定义一个调用方法，接受多个参数，包括隐藏状态、注意力掩码等
    def __call__(
        self,
        hidden_states,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 如果输出注意力矩阵，创建一个空元组来存储注意力信息
        all_attentions = () if output_attentions else None
        # 如果输出隐藏状态，创建一个空元组来存储隐藏状态信息
        all_hidden_states = () if output_hidden_states else None

        # 遍历所有的编码层
        for encoder_layer in self.layers:
            # 如果输出隐藏状态，则将当前隐藏状态加入隐藏状态元组
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            # 添加 LayerDrop（参考https://arxiv.org/abs/1909.11556）
            dropout_probability = random.uniform(0, 1)
            # 如果不是确定性的并且随机数小于层丢弃率，则跳过当前层
            if not deterministic and (dropout_probability < self.layerdrop):
                layer_outputs = (None, None)
            else:
                # 否则，调用当前编码层的方法，得到当前编码层的输出
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions,
                    deterministic,
                )
            # 更新隐藏状态为当前编码层的输出
            hidden_states = layer_outputs[0]
            # 如果输出注意力矩阵，将当前编码层的注意力信息加入注意力元组
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # 如果输出隐藏状态，将最终隐藏状态加入隐藏状态元组
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 将隐藏状态、隐藏状态元组、注意力元组组成元组作为输出
        outputs = (hidden_states, all_hidden_states, all_attentions)

        # 如果不返回字典格式的输出，则将结果展开并返回
        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        # 返回格式化的输出，使用FlaxBaseModelOutput包装
        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )
# 从 transformers.models.bart.modeling_flax_bart.FlaxBartDecoderLayer 复制并修改为使用 Marian 的解码器层
class FlaxMarianDecoderLayer(nn.Module):
    # 类型提示：Marian 的配置信息
    config: MarianConfig
    # 数据类型，默认为 32 位浮点数
    dtype: jnp.dtype = jnp.float32

    # 初始化函数
    def setup(self) -> None:
        # 解码器的嵌入维度等于配置中的 d_model
        self.embed_dim = self.config.d_model
        # 创建自注意力层，用于处理解码器内部的自注意力机制
        self.self_attn = FlaxMarianAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.decoder_attention_heads,
            dropout=self.config.attention_dropout,
            causal=True,  # 自注意力层是因果的，用于解码过程
            dtype=self.dtype,
        )
        # Dropout 层，用于自注意力层后的随机失活
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)
        # 激活函数，根据配置中的激活函数选择相应的激活函数
        self.activation_fn = ACT2FN[self.config.activation_function]
        # 激活函数后的随机失活层
        self.activation_dropout_layer = nn.Dropout(rate=self.config.activation_dropout)

        # 自注意力层后的 LayerNorm 层
        self.self_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        # 编码器注意力层，用于解码器与编码器之间的交互
        self.encoder_attn = FlaxMarianAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.decoder_attention_heads,
            dropout=self.config.attention_dropout,
            dtype=self.dtype,
        )
        # 编码器注意力层后的 LayerNorm 层
        self.encoder_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        # 第一个全连接层，用于解码器内部的前馈神经网络
        self.fc1 = nn.Dense(
            self.config.decoder_ffn_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),  # 使用正态分布初始化权重
        )
        # 第二个全连接层，用于将前馈神经网络的结果映射回嵌入维度
        self.fc2 = nn.Dense(
            self.embed_dim,  # 输出维度等于嵌入维度
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),  # 使用正态分布初始化权重
        )
        # 最终的 LayerNorm 层
        self.final_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    # 调用函数，用于执行解码器层的前向传播
    def __call__(
        self,
        hidden_states: jnp.ndarray,  # 输入的隐藏状态
        attention_mask: jnp.ndarray,  # 注意力掩码，用于指示哪些位置需要注意哪些位置不需要注意
        encoder_hidden_states: Optional[jnp.ndarray] = None,  # 编码器的隐藏状态，用于解码器与编码器的交互
        encoder_attention_mask: Optional[jnp.ndarray] = None,  # 编码器的注意力掩码
        init_cache: bool = False,  # 是否初始化缓存
        output_attentions: bool = True,  # 是否输出注意力权重
        deterministic: bool = True,  # 是否确定性地进行计算
    # 定义函数，返回类型为 Tuple[jnp.ndarray]
    ) -> Tuple[jnp.ndarray]:

        # 保存 residual（残差）
        residual = hidden_states

        # 自注意力机制
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, init_cache=init_cache
        )
        # 对 hidden_states 进行 dropout 处理
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 加上残差，并更新 hidden_states
        hidden_states = residual + hidden_states
        # 对 hidden_states 进行 layer normalization
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 交叉注意力块
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            hidden_states, cross_attn_weights = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            # 对 hidden_states 进行 dropout 处理
            hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
            # 加上残差，并更新 hidden_states
            hidden_states = residual + hidden_states
            # 对 hidden_states 进行 layer normalization
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # 全连接层
        residual = hidden_states
        # 使用激活函数激活全连接层 fc1
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 对激活后的 hidden_states 进行 dropout 处理
        hidden_states = self.activation_dropout_layer(hidden_states, deterministic=deterministic)
        # 使用全连接层 fc2
        hidden_states = self.fc2(hidden_states)
        # 对 hidden_states 进行 dropout 处理
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 加上残差，并更新 hidden_states
        hidden_states = residual + hidden_states
        # 对 hidden_states 进行 layer normalization
        hidden_states = self.final_layer_norm(hidden_states)

        # 输出为一个元组，包含 hidden_states
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则添加 self_attn_weights 和 cross_attn_weights 到输出中
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        # 返回输出
        return outputs
# 从transformers.models.bart.modeling_flax_bart.FlaxBartDecoderLayerCollection复制代码，并将Bart->Marian
class FlaxMarianDecoderLayerCollection(nn.Module):
    # 编码器配置，用于初始化模型
    config: MarianConfig
    # 计算时的数据类型
    dtype: jnp.dtype = jnp.float32

    # 模型初始化
    def setup(self):
        # 编码器层列表
        self.layers = [
            FlaxMarianDecoderLayer(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.decoder_layers)
        ]
        # 用于层级随机失活的参数
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
        # 编码器层的输出
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        # 遍历每个编码器层
        for decoder_layer in self.layers:
            if output_hidden_states:
                # 在输出hidden_states时，记录每层的hidden_states
                all_hidden_states += (hidden_states,)
                # 添加层级随机失活 (详细请参阅 https://arxiv.org/abs/1909.11556)
            dropout_probability = random.uniform(0, 1)
            if not deterministic and (dropout_probability < self.layerdrop):
                # 如果不是确定性执行且小于层级随机失活的概率，则忽略该层的输出
                layer_outputs = (None, None, None)
            else:
                # 执行编码器层
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    init_cache=init_cache,
                    output_attentions=output_attentions,
                    deterministic=deterministic,
                )

            # 更新hidden_states和attns
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # 添加最后一个编码器层的 hidden_states
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 返回输出
        outputs = [hidden_states, all_hidden_states, all_self_attns, all_cross_attentions]

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        # 返回具有过去和交叉注意力的输出
        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


# Marian编码器模型
class FlaxMarianEncoder(nn.Module):
    # 编码器配置，用于初始化模型
    config: MarianConfig
    # 计算时的数据类型
    dtype: jnp.dtype = jnp.float32
    # 在模型设置过程中初始化一些参数和层
    def setup(self):
        # 初始化一个丢弃层，使用给定的丢弃率
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)

        # 获取词嵌入的维度和最大源位置数
        embed_dim = self.config.d_model
        self.max_source_positions = self.config.max_position_embeddings
        # 如果配置中设置了缩放嵌入，则计算嵌入缩放因子
        self.embed_scale = math.sqrt(embed_dim) if self.config.scale_embedding else 1.0

        # 创建一个用于表示位置的正弦嵌入矩阵
        self.embed_positions = create_sinusoidal_positions(self.config.max_position_embeddings, embed_dim)
        # 初始化编码器层集合
        self.layers = FlaxMarianEncoderLayerCollection(self.config, self.dtype)

    # 模型的调用方法，接受输入和一些额外参数，并返回模型输出
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
        # 重新调整输入张量的形状，将其变为二维张量
        input_ids = input_ids.reshape(-1, input_shape[-1])

        # 通过词嵌入矩阵获取输入的词嵌入，并乘以嵌入缩放因子
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # 根据位置索引获取位置嵌入
        positions = jnp.take(self.embed_positions, position_ids, axis=0)
        # 明确地将位置嵌入转换为与输入嵌入相同的数据类型
        positions = positions.astype(inputs_embeds.dtype)

        # 将输入嵌入和位置嵌入相加得到隐藏状态
        hidden_states = inputs_embeds + positions
        # 对隐藏状态应用丢弃层
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)

        # 将隐藏状态传递给编码器层集合进行处理，并返回输出
        outputs = self.layers(
            hidden_states,
            attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果不返回字典，则直接返回输出
        if not return_dict:
            return outputs

        # 返回基础模型输出，包括最后的隐藏状态、所有隐藏状态和注意力权重
        return FlaxBaseModelOutput(
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 定义了一个名为FlaxMarianDecoder的类，继承自nn.Module
class FlaxMarianDecoder(nn.Module):
    # 类属性config为MarianConfig类型
    config: MarianConfig
    # 类属性embed_tokens为nn.Embed类型
    embed_tokens: nn.Embed
    # 类属性dtype为jnp.float32，默认为jnp.float32，用作计算的数据类型

    # 定义setup方法
    def setup(self):
        # 初始化self.dropout_layer为一个Dropout层，dropout率为self.config.dropout
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)

        # 获取self.config中的d_model作为embed_dim
        embed_dim = self.config.d_model
        # 获取self.config中的max_position_embeddings作为self.max_target_positions
        self.max_target_positions = self.config.max_position_embeddings
        # 如果self.config中的scale_embedding值为True，则self.embed_scale为math.sqrt(self.config.d_model)，否则为1.0
        self.embed_scale = math.sqrt(self.config.d_model) if self.config.scale_embedding else 1.0

        # 创建sinusoidal位置编码，保存在self.embed_positions中
        self.embed_positions = create_sinusoidal_positions(self.config.max_position_embeddings, embed_dim)
        # 初始化self.layers为一个FlaxMarianDecoderLayerCollection对象，参数为self.config和self.dtype

    # 定义__call__方法
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
        # 获取input_ids的形状
        input_shape = input_ids.shape
        # 重新整形input_ids成为(-1, input_shape[-1])
        input_ids = input_ids.reshape(-1, input_shape[-1])

        # 通过self.embed_tokens对input_ids进行embedding，乘以self.embed_scale
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # 根据position_ids获取位置编码positions
        positions = jnp.take(self.embed_positions, position_ids, axis=0)
        # 显式转换positions的数据类型为inputs_embeds的数据类型
        positions = positions.astype(inputs_embeds.dtype)

        # 将inputs_embeds与positions相加得到hidden_states
        hidden_states = inputs_embeds + positions

        # 对hidden_states进行dropout操作
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)

        # 调用self.layers处理hidden_states等参数，返回outputs
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

        # 如果return_dict为False，返回outputs
        if not return_dict:
            return outputs

        # 返回包含隐藏状态、注意力分布等信息的FlaxBaseModelOutputWithPastAndCrossAttentions对象
        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


# 定义了一个名为FlaxMarianModule的类，继承自nn.Module
class FlaxMarianModule(nn.Module):
    # 类属性config为MarianConfig类型
    config: MarianConfig
    # 类属性dtype为jnp.float32，默认为jnp.float32，用作计算的数据类型

    # 定义setup方法
    def setup(self):
        # 初始化self.shared为一个Embed层，词汇大小为self.config.vocab_size，维度为self.config.d_model，
        # embedding初始化使用正态分布，标准差为self.config.init_std
        self.shared = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.init_std),
        )

        # 初始化self.encoder为一个FlaxMarianEncoder对象，参数为self.config、dtype为self.dtype，embed_tokens为self.shared
        self.encoder = FlaxMarianEncoder(self.config, dtype=self.dtype, embed_tokens=self.shared)
        # 初始化self.decoder为一个FlaxMarianDecoder对象，参数为self.config、dtype为self.dtype，embed_tokens为self.shared

    # 定义_get_encoder_module方法，返回self.encoder模块
    def _get_encoder_module(self):
        return self.encoder
    # 获取解码器模块
    def _get_decoder_module(self):
        return self.decoder

    # 对模型进行调用，传入输入序列、注意力掩码、解码器输入序列、解码器注意力掩码等参数
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

        # 获取解码器的输出，传入编码器的隐藏状态等参数
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

        # 如果不返回字典形式的结果，则将解码器输出和编码器输出合并返回
        if not return_dict:
            return decoder_outputs + encoder_outputs

        # 返回字典形式的结果，包括解码器的最后隐藏状态、解码器的所有隐藏状态和注意力权重、编码器的最后隐藏状态、编码器的所有隐藏状态和注意力权重
        return FlaxSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
``` 
# 定义一个继承自FlaxPreTrainedModel的FlaxMarianPreTrainedModel类，用于Marian模型的预训练
class FlaxMarianPreTrainedModel(FlaxPreTrainedModel):
    # 指定配置类为MarianConfig
    config_class = MarianConfig
    # 基础模型前缀为"model"
    base_model_prefix: str = "model"
    # 模块类为空
    module_class: nn.Module = None

    # 初始化方法
    def __init__(
        self,
        config: MarianConfig,
        input_shape: Tuple[int] = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        # 创建模块对象，根据参数进行配置
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 调用父类的初始化方法
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    # 初始化权重方法
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量，全部填充为0，数据类型为整型
        input_ids = jnp.zeros(input_shape, dtype="i4")
        # 确保对FlaxMarianForSequenceClassificationModule的初始化传递能正常工作
        # 将最后一个位置的值设置为结束标记的标识符
        input_ids = input_ids.at[(..., -1)].set(self.config.eos_token_id)
        # 创建与input_ids形状相同的注意力掩码，全部填充为1
        attention_mask = jnp.ones_like(input_ids)
        # 将解码器的输入与输入相同
        decoder_input_ids = input_ids
        # 解码器的注意力掩码与输入相同
        decoder_attention_mask = jnp.ones_like(input_ids)

        # 获取批次大小和序列长度
        batch_size, sequence_length = input_ids.shape
        # 创建位置编码，形状与输入相同
        position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))
        # 解码器的位置编码与输入相同
        decoder_position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # 切割随机数生成器，用于参数和dropout的随机数
        params_rng, dropout_rng = jax.random.split(rng)
        # 将参数和dropout的随机数放入字典
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 使用模块的初始化方法初始化随机参数
        random_params = self.module.init(
            rngs,
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            position_ids,
            decoder_position_ids,
        )["params"]

        # 如果给定了参数，则用给定参数替换随机参数中的缺失键
        if params is not None:
            # 展开参数字典
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            # 将缺失的键从随机参数中复制到给定参数中
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()  # 重置缺失的键集合
            # 冻结并返回参数字典
            return freeze(unflatten_dict(params))
        else:
            # 返回随机参数
            return random_params
    # 初始化缓存，用于快速自回归解码，对解码器进行缓存初始化
    def init_cache(self, batch_size, max_length, encoder_outputs):
        r"""
        Args:
            batch_size (`int`):
                用于快速自回归解码的批量大小。定义了初始化缓存时的批处理大小。
            max_length (`int`):
                自回归解码的最大可能长度。定义了初始化缓存时的序列长度。
            encoder_outputs (`Union[FlaxBaseModelOutput, tuple(tuple(jnp.ndarray)]`):
                `encoder_outputs` 包括（`last_hidden_state`，*可选*：`hidden_states`，*可选*：`attentions`）。`last_hidden_state` 的形状为 `(batch_size, sequence_length, hidden_size)`，*可选* 是编码器最后一层的输出中包含的隐藏状态的序列。用于解码器的交叉注意力。
        """
        # 初始化用于检索缓存的输入变量
        decoder_input_ids = jnp.ones((batch_size, max_length), dtype="i4")
        decoder_attention_mask = jnp.ones_like(decoder_input_ids)
        decoder_position_ids = jnp.broadcast_to(
            jnp.arange(jnp.atleast_2d(decoder_input_ids).shape[-1]), decoder_input_ids.shape
        )

        def _decoder_forward(module, decoder_input_ids, decoder_attention_mask, decoder_position_ids, **kwargs):
            decoder_module = module._get_decoder_module()
            return decoder_module(decoder_input_ids, decoder_attention_mask, decoder_position_ids, **kwargs)

        # 初始化变量以检索缓存
        init_variables = self.module.init(
            jax.random.PRNGKey(0),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            encoder_hidden_states=encoder_outputs[0],
            init_cache=True,
            method=_decoder_forward,  # 我们只需要调用解码器来初始化缓存
        )
        # 返回解冻后的变量中的缓存
        return unfreeze(init_variables["cache"])

    @add_start_docstrings(MARIAN_ENCODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutput, config_class=MarianConfig)
    # 编码输入的方法
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

        """


"""python
        >>> from transformers import AutoTokenizer, FlaxMarianMTModel

        >>> tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
        >>> model = FlaxMarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-de")

        >>> text = "My friends are cool but they eat too many carbs."
        >>> inputs = tokenizer(text, max_length=64, return_tensors="jax")
        >>> encoder_outputs = model.encode(**inputs)
        """


        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        if position_ids is None:
            batch_size, sequence_length = input_ids.shape
            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        def _encoder_forward(module, input_ids, attention_mask, position_ids, **kwargs):
            encode_module = module._get_encoder_module()
            return encode_module(input_ids, attention_mask, position_ids, **kwargs)

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

    @add_start_docstrings(MARIAN_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutputWithPastAndCrossAttentions, config_class=MarianConfig)
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
    # 定义一个函数，用于调用模型
    def __call__(
        self,
        input_ids: jnp.ndarray,  # 输入的编码器输入 ID，类型为 JAX 数组
        attention_mask: Optional[jnp.ndarray] = None,  # 可选的编码器注意力掩码，默认为 None
        decoder_input_ids: Optional[jnp.ndarray] = None,  # 可选的解码器输入 ID，默认为 None
        decoder_attention_mask: Optional[jnp.ndarray] = None,  # 可选的解码器注意力掩码，默认为 None
        position_ids: Optional[jnp.ndarray] = None,  # 可选的位置 ID，默认为 None
        decoder_position_ids: Optional[jnp.ndarray] = None,  # 可选的解码器位置 ID，默认为 None
        output_attentions: Optional[bool] = None,  # 可选的是否输出注意力，默认为 None
        output_hidden_states: Optional[bool] = None,  # 可选的是否输出隐藏状态，默认为 None
        return_dict: Optional[bool] = None,  # 可选的是否返回字典，默认为 None
        train: bool = False,  # 是否处于训练模式，默认为 False
        params: dict = None,  # 参数字典，默认为 None
        dropout_rng: PRNGKey = None,  # 用于 dropout 的随机数生成器，默认为 None
    ):
        # 如果输出注意力未指定，则使用配置中的值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果输出隐藏状态未指定，则使用配置中的值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果返回字典未指定，则使用配置中的值
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 准备编码器输入
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)  # 如果未提供注意力掩码，则创建一个全为 1 的掩码
        if position_ids is None:
            batch_size, sequence_length = input_ids.shape
            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # 准备解码器输入
        if decoder_input_ids is None:
            # 将编码器输入向右移动一个位置，用 pad_token_id 填充开头，并在开头添加解码器的起始标记
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, decoder_start_token_id=self.config.decoder_start_token_id
            )
        if decoder_attention_mask is None:
            decoder_attention_mask = jnp.ones_like(decoder_input_ids)  # 创建一个全为 1 的解码器注意力掩码
        if decoder_position_ids is None:
            batch_size, sequence_length = decoder_input_ids.shape
            decoder_position_ids = jnp.broadcast_to(
                jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
            )  # 创建解码器位置 ID

        # 如果需要处理任何 PRNG
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        # 调用模型的 apply 方法进行前向传播
        return self.module.apply(
            {"params": params or self.params},  # 模型参数
            input_ids=jnp.array(input_ids, dtype="i4"),  # 编码器输入 ID
            attention_mask=jnp.array(attention_mask, dtype="i4"),  # 编码器注意力掩码
            position_ids=jnp.array(position_ids, dtype="i4"),  # 编码器位置 ID
            decoder_input_ids=jnp.array(decoder_input_ids, dtype="i4"),  # 解码器输入 ID
            decoder_attention_mask=jnp.array(decoder_attention_mask, dtype="i4"),  # 解码器注意力掩码
            decoder_position_ids=jnp.array(decoder_position_ids, dtype="i4"),  # 解码器位置 ID
            output_attentions=output_attentions,  # 是否输出注意力
            output_hidden_states=output_hidden_states,  # 是否输出隐藏状态
            return_dict=return_dict,  # 是否返回字典
            deterministic=not train,  # 是否处于确定性模式
            rngs=rngs,  # PRNG 生成器
        )
# 添加起始文档字符串，描述了这个 Marian 模型的输出
@add_start_docstrings(
    "The bare Marian Model transformer outputting raw hidden-states without any specific head on top.",
    MARIAN_START_DOCSTRING,
)
# 定义了 FlaxMarianModel 类，继承自 FlaxMarianPreTrainedModel
class FlaxMarianModel(FlaxMarianPreTrainedModel):
    # Marian 模型的配置
    config: MarianConfig
    # 计算时的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 模块类别是 FlaxMarianModule
    module_class = FlaxMarianModule

# 添加了调用示例文档字符串
append_call_sample_docstring(FlaxMarianModel, _CHECKPOINT_FOR_DOC, FlaxSeq2SeqModelOutput, _CONFIG_FOR_DOC)

# 定义了 FlaxMarianMTModule 类，继承自 nn.Module
class FlaxMarianMTModule(nn.Module):
    # Marian 模型的配置
    config: MarianConfig
    # 计算时的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 偏置初始化的函数
    bias_init: Callable[..., jnp.ndarray] = jax.nn.initializers.zeros

    def setup(self):
        # 定义 Marian 模型
        self.model = FlaxMarianModule(config=self.config, dtype=self.dtype)
        # 定义语言模型头
        self.lm_head = nn.Dense(
            self.model.shared.num_embeddings,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        # 定义最终输出的偏置
        self.final_logits_bias = self.param("final_logits_bias", self.bias_init, (1, self.model.shared.num_embeddings))

    # 获取编码器模块
    def _get_encoder_module(self):
        return self.model.encoder

    # 获取解码器模块
    def _get_decoder_module(self):
        return self.model.decoder

    # 定义了对象被调用时的行为
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
        # 调用 Marian 模型
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

        hidden_states = outputs[0]

        # 若 word embeddings 被绑定
        if self.config.tie_word_embeddings:
            shared_embedding = self.model.variables["params"]["shared"]["embedding"]
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)

        lm_logits += self.final_logits_bias.astype(self.dtype)

        # 如果不需要返回字典形式的输出
        if not return_dict:
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
# 该类是基于 MARIAN 模型的语言模型，可用于机器翻译任务
@add_start_docstrings(
    "The MARIAN Model with a language modeling head. Can be used for translation.", MARIAN_START_DOCSTRING
)
class FlaxMarianMTModel(FlaxMarianPreTrainedModel):
    # 指定模块类为 FlaxMarianMTModule
    module_class = FlaxMarianMTModule
    # 使用 float32 作为模型的默认数据类型
    dtype: jnp.dtype = jnp.float32

    # 解码输入文档字符串
    @add_start_docstrings(MARIAN_DECODE_INPUTS_DOCSTRING)
    # 替换返回类型文档字符串 
    @replace_return_docstrings(output_type=FlaxCausalLMOutputWithCrossAttentions, config_class=MarianConfig)
    # 定义解码函数
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
    # 定义一个适应于beam search的logits调整函数  
    def _adapt_logits_for_beam_search(self, logits):
        # 将logits中的padding token的值设为负无穷, 确保模型永远不会生成padding token
        logits = logits.at[:, :, self.config.pad_token_id].set(float("-inf"))
        return logits

    # 定义生成输入准备函数
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

        # 构建扩展后的注意力掩码
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if decoder_attention_mask is not None:
            position_ids = decoder_attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, decoder_attention_mask, (0, 0))
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        # 返回生成所需的输入
        return {
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "encoder_attention_mask": attention_mask,
            "decoder_attention_mask": extended_attention_mask,
            "decoder_position_ids": position_ids,
        }

    # 定义更新生成输入的函数
    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        # 更新 past_key_values 和 decoder_position_ids
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["decoder_position_ids"] = model_kwargs["decoder_position_ids"][:, -1:] + 1
        return model_kwargs
# 定义了 FLAX_MARIAN_MT_DOCSTRING，用于存储函数的返回值和示例
FLAX_MARIAN_MT_DOCSTRING = """
    Returns:

    Example:

    ```python
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

# 调用 overwrite_call_docstring 函数，给 FlaxMarianMTModel 添加文档字符串
overwrite_call_docstring(
    FlaxMarianMTModel,
    MARIAN_INPUTS_DOCSTRING + FLAX_MARIAN_MT_DOCSTRING,
)

# 调用 append_replace_return_docstrings 函数，给 FlaxMarianMTModel 添加函数输出类型和配置类的文档字符串
append_replace_return_docstrings(FlaxMarianMTModel, output_type=FlaxSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
```