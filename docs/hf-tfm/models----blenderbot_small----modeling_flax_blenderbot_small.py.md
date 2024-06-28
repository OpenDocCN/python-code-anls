# `.\models\blenderbot_small\modeling_flax_blenderbot_small.py`

```py
# 设置文件编码为 UTF-8
# 版权声明：2021 年 Facebook, Inc. 和 HuggingFace Inc. 团队保留所有权利
#
# 根据 Apache 许可证 2.0 版本授权使用本文件；
# 除非符合许可证要求，否则不得使用本文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件按"原样"分发，
# 没有任何形式的明示或暗示担保或条件。
# 有关详细信息，请参阅许可证。
""" Flax BlenderbotSmall 模型。"""

# 导入必要的库和模块
import math
import random
from functools import partial
from typing import Callable, Optional, Tuple

# 导入 Flax 相关模块和类
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from jax.random import PRNGKey

# 导入模型输出类和实用函数
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
from ...utils import add_start_docstrings, logging, replace_return_docstrings
from .configuration_blenderbot_small import BlenderbotSmallConfig

# 获取日志记录器实例
logger = logging.get_logger(__name__)

# 用于文档的模型检查点和配置
_CHECKPOINT_FOR_DOC = "facebook/blenderbot_small-90M"
_CONFIG_FOR_DOC = "BlenderbotSmallConfig"

# BlenderbotSmall 模型的起始文档字符串
BLENDERBOT_SMALL_START_DOCSTRING = r"""
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
    # config ([`BlenderbotSmallConfig`]): 模型配置类，包含模型的所有参数。
    #    使用配置文件初始化模型时，仅加载模型的配置，不加载与模型相关的权重。
    #    若要加载模型权重，请查看 [`~FlaxPreTrainedModel.from_pretrained`] 方法。
    # dtype (`jax.numpy.dtype`, *可选*, 默认为 `jax.numpy.float32`):
    #    计算时的数据类型。可选项包括 `jax.numpy.float32`、`jax.numpy.float16`（在GPU上）和
    #    `jax.numpy.bfloat16`（在TPU上）。
    #
    #    可用于在GPU或TPU上启用混合精度训练或半精度推断。如果指定了dtype，则所有计算将使用给定的 `dtype` 进行。
    #
    #    **注意这仅指定计算的数据类型，不影响模型参数的数据类型。**
    #
    #    如果要更改模型参数的数据类型，请参见 [`~FlaxPreTrainedModel.to_fp16`] 和 [`~FlaxPreTrainedModel.to_bf16`]。
"""
BLENDERBOT_SMALL_INPUTS_DOCSTRING = r"""
"""


BLENDERBOT_SMALL_ENCODE_INPUTS_DOCSTRING = r"""
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

BLENDERBOT_SMALL_DECODE_INPUTS_DOCSTRING = r"""
"""


# Copied from transformers.models.bart.modeling_flax_bart.shift_tokens_right
def shift_tokens_right(input_ids: jnp.ndarray, pad_token_id: int, decoder_start_token_id: int) -> jnp.ndarray:
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = jnp.zeros_like(input_ids)  # 创建一个与输入数组相同形状的全零数组
    shifted_input_ids = shifted_input_ids.at[:, 1:].set(input_ids[:, :-1])  # 将输入数组向右移动一个位置
    shifted_input_ids = shifted_input_ids.at[:, 0].set(decoder_start_token_id)  # 设置起始位置的标记
    
    shifted_input_ids = jnp.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)  # 替换特殊标记为pad_token_id
    return shifted_input_ids


# Copied from transformers.models.bart.modeling_flax_bart.FlaxBartAttention with Bart->BlenderbotSmall
class FlaxBlenderbotSmallAttention(nn.Module):
    config: BlenderbotSmallConfig  # 配置对象
    embed_dim: int  # 嵌入维度
    num_heads: int  # 头的数量
    dropout: float = 0.0  # dropout率，默认为0.0
    causal: bool = False  # 是否为因果（causal）注意力
    bias: bool = True  # 是否包含偏置项
    dtype: jnp.dtype = jnp.float32  # 计算时的数据类型，使用jnp.float32
    # 设置函数，用于初始化模型参数
    def setup(self) -> None:
        # 计算每个注意力头的维度
        self.head_dim = self.embed_dim // self.num_heads
        # 检查embed_dim是否能被num_heads整除，否则抛出数值错误
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )

        # 定义一个偏函数dense，用于创建带有预设参数的全连接层
        dense = partial(
            nn.Dense,
            self.embed_dim,
            use_bias=self.bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )

        # 创建查询、键、值投影层以及输出投影层
        self.q_proj, self.k_proj, self.v_proj = dense(), dense(), dense()
        self.out_proj = dense()

        # 创建一个dropout层，用于模型训练时的随机失活
        self.dropout_layer = nn.Dropout(rate=self.dropout)

        # 如果需要因果注意力机制，创建一个因果掩码
        if self.causal:
            self.causal_mask = make_causal_mask(
                jnp.ones((1, self.config.max_position_embeddings), dtype="bool"), dtype="bool"
            )

    # 将隐藏状态按照注意力头分割
    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    # 将分割后的注意力头重新合并
    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    # 使用JAX库的compact装饰器定义一个紧凑模型组件
    @nn.compact
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slightly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """

        # detect if we're initializing by absence of existing cache data.
        # 检测是否需要初始化，通过检查缓存数据是否存在来判断
        is_initialized = self.has_variable("cache", "cached_key")

        # initialize or retrieve cached key and value states with zeros of appropriate shape and type
        # 初始化或获取缓存的键和值状态，使用适当形状和类型的零值
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)

        # initialize or retrieve cache index, starting from 0
        # 初始化或获取缓存索引，起始为0
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            # extract batch dimensions and other relevant dimensions from cached key shape
            # 提取批量维度和其他相关维度，从缓存键的形状中
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape

            # update cached key and value with new 1d spatial slices based on current cache index
            # 使用当前缓存索引更新缓存键和值的新的一维空间切片
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)

            # update cached_key and cached_value variables with new values
            # 更新 cached_key 和 cached_value 变量的值
            cached_key.value = key
            cached_value.value = value

            # determine number of updated cache vectors from the current query shape
            # 确定从当前查询形状更新的缓存向量数量
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors

            # create a pad mask for causal attention to avoid attending to future elements
            # 创建一个用于因果注意力的填充掩码，以避免关注未来元素
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )

            # combine pad_mask with existing attention_mask if provided
            # 如果提供了 attention_mask，则与其结合
            attention_mask = combine_masks(pad_mask, attention_mask)

        # return updated key, value, and attention_mask
        # 返回更新后的 key、value 和 attention_mask
        return key, value, attention_mask
# Copied from transformers.models.bart.modeling_flax_bart.FlaxBartEncoderLayer with Bart->BlenderbotSmall
class FlaxBlenderbotSmallEncoderLayer(nn.Module):
    config: BlenderbotSmallConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.embed_dim = self.config.d_model  # 从配置中获取模型的嵌入维度
        self.self_attn = FlaxBlenderbotSmallAttention(  # 创建自注意力机制实例
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.encoder_attention_heads,
            dropout=self.config.attention_dropout,
            dtype=self.dtype,
        )
        self.self_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)  # 创建自注意力层规范化实例
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)  # 创建丢弃层实例
        self.activation_fn = ACT2FN[self.config.activation_function]  # 根据配置选择激活函数
        self.activation_dropout_layer = nn.Dropout(rate=self.config.activation_dropout)  # 创建激活函数丢弃层实例
        self.fc1 = nn.Dense(
            self.config.encoder_ffn_dim,  # 配置中编码器前馈网络的维度
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),  # 使用正态分布初始化权重
        )
        self.fc2 = nn.Dense(
            self.embed_dim,  # 嵌入维度
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),  # 使用正态分布初始化权重
        )
        self.final_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)  # 创建最终层规范化实例

    def __call__(
        self,
        hidden_states: jnp.ndarray,  # 隐藏状态张量
        attention_mask: jnp.ndarray,  # 注意力掩码张量
        output_attentions: bool = True,  # 是否输出注意力权重
        deterministic: bool = True,  # 是否使用确定性计算
    ) -> Tuple[jnp.ndarray]:
        residual = hidden_states  # 保存原始隐藏状态，用于残差连接

        # 自注意力计算
        hidden_states, attn_weights = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask)

        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)  # 应用丢弃层
        hidden_states = residual + hidden_states  # 残差连接
        hidden_states = self.self_attn_layer_norm(hidden_states)  # 自注意力层规范化

        residual = hidden_states  # 保存残差连接后的状态

        # 前馈网络计算
        hidden_states = self.activation_fn(self.fc1(hidden_states))  # 应用激活函数和第一个全连接层
        hidden_states = self.activation_dropout_layer(hidden_states, deterministic=deterministic)  # 应用激活函数的丢弃层
        hidden_states = self.fc2(hidden_states)  # 第二个全连接层
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)  # 应用丢弃层
        hidden_states = residual + hidden_states  # 残差连接
        hidden_states = self.final_layer_norm(hidden_states)  # 最终层规范化

        outputs = (hidden_states,)  # 输出隐藏状态作为元组的第一个元素

        if output_attentions:
            outputs += (attn_weights,)  # 如果需要输出注意力权重，则作为元组的第二个元素添加到输出中

        return outputs


# Copied from transformers.models.bart.modeling_flax_bart.FlaxBartEncoderLayerCollection with Bart->BlenderbotSmall
class FlaxBlenderbotSmallEncoderLayerCollection(nn.Module):
    config: BlenderbotSmallConfig
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型

    def setup(self):
        self.layers = [
            FlaxBlenderbotSmallEncoderLayer(self.config, name=str(i), dtype=self.dtype)  # 创建编码器层实例列表
            for i in range(self.config.encoder_layers)  # 根据配置中编码器层数创建
        ]
        self.layerdrop = self.config.encoder_layerdrop  # 设置编码器层的丢弃率
    # 定义一个调用方法，用于执行模型的前向传播
    def __call__(
        self,
        hidden_states,  # 输入的隐藏状态张量
        attention_mask,  # 注意力掩码，用于指示哪些位置需要注意
        deterministic: bool = True,  # 是否使用确定性推断
        output_attentions: bool = False,  # 是否输出注意力权重
        output_hidden_states: bool = False,  # 是否输出所有隐藏状态
        return_dict: bool = True,  # 是否返回字典形式的输出
    ):
        # 如果需要输出注意力权重，则初始化空元组用于存储所有注意力权重
        all_attentions = () if output_attentions else None
        # 如果需要输出所有隐藏状态，则初始化空元组用于存储所有隐藏状态
        all_hidden_states = () if output_hidden_states else None

        # 遍历所有编码器层
        for encoder_layer in self.layers:
            # 如果需要输出所有隐藏状态，则将当前隐藏状态添加到all_hidden_states中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            # 添加层丢弃（参见 https://arxiv.org/abs/1909.11556 进行描述）
            dropout_probability = random.uniform(0, 1)
            # 如果不是确定性推断且随机dropout概率小于层丢弃率，则跳过该层
            if not deterministic and (dropout_probability < self.layerdrop):
                layer_outputs = (None, None)
            else:
                # 否则，调用当前编码器层进行前向传播
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions,
                    deterministic,
                )
            # 更新隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]
            # 如果需要输出注意力权重，则将当前层的注意力权重添加到all_attentions中
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # 如果需要输出所有隐藏状态，则将最终的隐藏状态添加到all_hidden_states中
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 构建模型的输出结果，包括最终的隐藏状态、所有隐藏状态和所有注意力权重
        outputs = (hidden_states, all_hidden_states, all_attentions)

        # 如果不需要以字典形式返回结果，则返回一个元组，过滤掉None值
        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        # 否则，以FlaxBaseModelOutput的形式返回结果，包括最终的隐藏状态、所有隐藏状态和所有注意力权重
        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )
# 从 transformers.models.bart.modeling_flax_bart.FlaxBartDecoderLayer 复制并修改为使用 BlenderbotSmall
class FlaxBlenderbotSmallDecoderLayer(nn.Module):
    # 配置参数对象，指定为 BlenderbotSmallConfig 类型
    config: BlenderbotSmallConfig
    # 数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 初始化函数，设置层的属性
    def setup(self) -> None:
        # 设定嵌入维度为模型配置中的 d_model
        self.embed_dim = self.config.d_model
        # 使用 BlenderbotSmallAttention 定义自注意力机制
        self.self_attn = FlaxBlenderbotSmallAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.decoder_attention_heads,
            dropout=self.config.attention_dropout,
            causal=True,
            dtype=self.dtype,
        )
        # 定义 dropout 层，用于模型训练时的随机失活
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)
        # 激活函数，根据配置中的激活函数类型选择对应的函数
        self.activation_fn = ACT2FN[self.config.activation_function]
        # 激活函数的 dropout 层，用于激活函数的输出时的随机失活
        self.activation_dropout_layer = nn.Dropout(rate=self.config.activation_dropout)

        # 定义自注意力机制的 LayerNorm 层
        self.self_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        # 定义与编码器注意力相关的注意力机制
        self.encoder_attn = FlaxBlenderbotSmallAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.decoder_attention_heads,
            dropout=self.config.attention_dropout,
            dtype=self.dtype,
        )
        # 编码器注意力的 LayerNorm 层
        self.encoder_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        
        # 第一个全连接层，用于进行线性变换
        self.fc1 = nn.Dense(
            self.config.decoder_ffn_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        # 第二个全连接层，输出维度为嵌入维度，用于线性变换
        self.fc2 = nn.Dense(
            self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std)
        )
        # 最终的 LayerNorm 层，用于模型输出的标准化
        self.final_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    # 调用函数，定义层的前向传播逻辑
    def __call__(
        self,
        hidden_states: jnp.ndarray,  # 输入的隐藏状态
        attention_mask: jnp.ndarray,  # 注意力遮罩，掩盖无效位置
        encoder_hidden_states: Optional[jnp.ndarray] = None,  # 编码器隐藏状态（可选）
        encoder_attention_mask: Optional[jnp.ndarray] = None,  # 编码器注意力遮罩（可选）
        init_cache: bool = False,  # 是否初始化缓存（默认为 False）
        output_attentions: bool = True,  # 是否输出注意力权重（默认为 True）
        deterministic: bool = True,  # 是否确定性推断模式（默认为 True）
        # 函数定义未完，需继续编写
    ) -> Tuple[jnp.ndarray]:
        # 将输入的 hidden_states 保存为 residual，用于后续残差连接
        residual = hidden_states

        # 自注意力机制
        # 调用 self_attn 方法进行自注意力计算，得到更新后的 hidden_states 和 self_attn_weights
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, init_cache=init_cache
        )
        # 应用 dropout 层，根据 deterministic 参数确定是否使用确定性 dropout
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 添加残差连接
        hidden_states = residual + hidden_states
        # 应用自注意力层的 LayerNormalization
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 跨注意力块
        cross_attn_weights = None
        # 如果存在 encoder_hidden_states，则执行以下操作
        if encoder_hidden_states is not None:
            # 将当前的 hidden_states 保存为 residual
            residual = hidden_states

            # 执行 encoder_attn 方法进行跨注意力计算，得到更新后的 hidden_states 和 cross_attn_weights
            hidden_states, cross_attn_weights = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            # 应用 dropout 层，根据 deterministic 参数确定是否使用确定性 dropout
            hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
            # 添加残差连接
            hidden_states = residual + hidden_states
            # 应用跨注意力层的 LayerNormalization
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # 全连接层
        # 将当前的 hidden_states 保存为 residual
        residual = hidden_states
        # 应用激活函数 activation_fn 到 fc1 全连接层
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 应用 activation_dropout_layer，根据 deterministic 参数确定是否使用确定性 dropout
        hidden_states = self.activation_dropout_layer(hidden_states, deterministic=deterministic)
        # 应用 fc2 全连接层
        hidden_states = self.fc2(hidden_states)
        # 应用 dropout 层，根据 deterministic 参数确定是否使用确定性 dropout
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 添加残差连接
        hidden_states = residual + hidden_states
        # 应用最终的 LayerNormalization
        hidden_states = self.final_layer_norm(hidden_states)

        # 准备输出
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则将 self_attn_weights 和 cross_attn_weights 添加到 outputs 中
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        # 返回最终的 outputs
        return outputs
# 从transformers.models.bart.modeling_flax_bart.FlaxBartDecoderLayerCollection复制而来，修改为BlenderbotSmall模型
class FlaxBlenderbotSmallDecoderLayerCollection(nn.Module):
    # 使用BlenderbotSmallConfig配置
    config: BlenderbotSmallConfig
    # 计算过程中使用的数据类型
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        # 创建decoder层列表，根据配置中的decoder_layers数量
        self.layers = [
            FlaxBlenderbotSmallDecoderLayer(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.decoder_layers)
        ]
        # 设置layer drop参数
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
        # 如果需要输出隐藏状态，则初始化空元组
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力权重，则初始化空元组
        all_self_attns = () if output_attentions else None
        # 如果需要输出交叉注意力权重，并且encoder_hidden_states不为None，则初始化空元组
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        # 遍历每个decoder层
        for decoder_layer in self.layers:
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到all_hidden_states中
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                # 添加LayerDrop机制（参见https://arxiv.org/abs/1909.11556）
            # 生成0到1之间的随机数作为dropout概率
            dropout_probability = random.uniform(0, 1)
            # 如果不是确定性推断，并且dropout_probability小于layerdrop值，则将输出置为None
            if not deterministic and (dropout_probability < self.layerdrop):
                layer_outputs = (None, None, None)
            else:
                # 否则，调用当前decoder层进行前向传播计算
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    init_cache=init_cache,
                    output_attentions=output_attentions,
                    deterministic=deterministic,
                )

            # 更新隐藏状态为当前decoder层的输出的第一个元素
            hidden_states = layer_outputs[0]
            # 如果需要输出注意力权重，则将当前层的自注意力权重添加到all_self_attns中
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                # 如果encoder_hidden_states不为None，则将当前层的交叉注意力权重添加到all_cross_attentions中
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # 如果需要输出最终的隐藏状态，则将最终隐藏状态添加到all_hidden_states中
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 组装输出结果列表
        outputs = [hidden_states, all_hidden_states, all_self_attns, all_cross_attentions]

        # 如果return_dict为False，则返回元组形式的输出列表
        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        # 否则，返回带有过去和交叉注意力的FlaxBaseModelOutputWithPastAndCrossAttentions对象
        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class FlaxBlenderbotSmallEncoder(nn.Module):
    # 使用BlenderbotSmallConfig配置
    config: BlenderbotSmallConfig
    # 编码器token的嵌入层
    embed_tokens: nn.Embed
    # 定义默认数据类型为 jax 中的 float32，用于计算过程中的数据类型
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    # 初始化方法，设置模型中的 dropout 层和一些与 embedding 相关的属性
    def setup(self):
        # 根据配置参数初始化 dropout 层
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)

        # 获取配置中的 embedding 维度大小
        embed_dim = self.config.d_model
        # 获取配置中的填充索引
        self.padding_idx = self.config.pad_token_id
        # 获取配置中的最大位置编码长度
        self.max_source_positions = self.config.max_position_embeddings
        # 根据配置是否缩放 embedding 的初始化权重
        self.embed_scale = math.sqrt(embed_dim) if self.config.scale_embedding else 1.0

        # 初始化位置编码的嵌入层
        self.embed_positions = nn.Embed(
            self.config.max_position_embeddings,
            embed_dim,
            # 使用正态分布初始化权重，标准差为配置中的初始化标准差
            embedding_init=jax.nn.initializers.normal(self.config.init_std),
        )

        # 初始化多层编码器
        self.layers = FlaxBlenderbotSmallEncoderLayerCollection(self.config, self.dtype)
        
        # 初始化 embedding 的 LayerNorm 层
        self.layernorm_embedding = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    # 模型的调用方法，接收输入和各种标志位，执行模型的前向传播
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
        # 获取输入张量的形状信息
        input_shape = input_ids.shape
        # 将输入张量展平为二维张量，保留最后一个维度的形状
        input_ids = input_ids.reshape(-1, input_shape[-1])

        # 使用 token embedding 对输入 token 进行嵌入，并根据缩放因子缩放
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # 根据位置编码的位置 IDs 获取位置编码的嵌入
        embed_pos = self.embed_positions(position_ids)

        # 将 token embedding 和位置编码的嵌入相加得到最终的隐藏状态
        hidden_states = inputs_embeds + embed_pos
        # 对隐藏状态进行 LayerNorm 归一化处理
        hidden_states = self.layernorm_embedding(hidden_states)
        # 对归一化后的隐藏状态应用 dropout，根据 deterministic 标志位决定是否使用确定性 dropout
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)

        # 将隐藏状态传入多层编码器中进行编码
        outputs = self.layers(
            hidden_states,
            attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果 return_dict 为 False，则直接返回编码器的输出
        if not return_dict:
            return outputs

        # 否则，返回一个包含模型输出各部分的字典结构
        return FlaxBaseModelOutput(
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 定义了一个名为FlaxBlenderbotSmallDecoder的类，继承自nn.Module
class FlaxBlenderbotSmallDecoder(nn.Module):
    # 类变量config，类型为BlenderbotSmallConfig，用于存储模型配置信息
    config: BlenderbotSmallConfig
    # 类变量embed_tokens，类型为nn.Embed，用于存储嵌入层信息
    embed_tokens: nn.Embed
    # 类变量dtype，默认为jnp.float32，表示计算过程中的数据类型

    # 初始化方法setup，用于配置模型的各个组件
    def setup(self):
        # 初始化dropout_layer，用于实现随机失活
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)

        # 从config中获取嵌入维度
        embed_dim = self.config.d_model
        # 从config中获取填充token的索引
        self.padding_idx = self.config.pad_token_id
        # 从config中获取目标位置的最大值
        self.max_target_positions = self.config.max_position_embeddings
        # 初始化embed_scale，根据scale_embedding参数决定是否开启缩放
        self.embed_scale = math.sqrt(self.config.d_model) if self.config.scale_embedding else 1.0

        # 初始化embed_positions，用于嵌入位置信息
        self.embed_positions = nn.Embed(
            self.config.max_position_embeddings,  # 嵌入位置的最大数量
            embed_dim,                           # 嵌入的维度
            embedding_init=jax.nn.initializers.normal(self.config.init_std),  # 使用正态分布初始化嵌入矩阵
        )

        # 初始化layers，即解码器的层集合
        self.layers = FlaxBlenderbotSmallDecoderLayerCollection(self.config, self.dtype)
        # 初始化layernorm_embedding，用于对输入嵌入进行层归一化
        self.layernorm_embedding = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    # 实现调用方法，定义了模型的前向计算过程
    def __call__(
        self,
        input_ids,                               # 输入的token id
        attention_mask,                          # 注意力掩码
        position_ids,                            # 位置id
        encoder_hidden_states: Optional[jnp.ndarray] = None,  # 编码器隐藏状态，默认为None
        encoder_attention_mask: Optional[jnp.ndarray] = None,  # 编码器注意力掩码，默认为None
        init_cache: bool = False,                # 是否初始化缓存，默认为False
        output_attentions: bool = False,         # 是否输出注意力权重，默认为False
        output_hidden_states: bool = False,      # 是否输出隐藏状态，默认为False
        return_dict: bool = True,                # 是否返回字典格式的输出，默认为True
        deterministic: bool = True,             # 是否确定性计算，默认为True
    ):
        # 获取输入tensor的形状
        input_shape = input_ids.shape
        # 重塑input_ids的形状为(batch_size * seq_length, embed_dim)
        input_ids = input_ids.reshape(-1, input_shape[-1])

        # 根据input_ids获取对应的嵌入表示，并乘以embed_scale进行缩放
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # 嵌入位置信息
        positions = self.embed_positions(position_ids)

        # 对输入嵌入进行层归一化处理
        inputs_embeds = self.layernorm_embedding(inputs_embeds)
        # 将位置嵌入加到输入嵌入上形成最终的隐藏状态表示
        hidden_states = inputs_embeds + positions

        # 使用dropout_layer对隐藏状态进行随机失活处理
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)

        # 调用layers的前向计算方法，处理隐藏状态，返回相应的输出
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

        # 如果return_dict为True，则构造包含额外信息的输出对象并返回
        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


# 从transformers.models.bart.modeling_flax_bart.FlaxBartModule复制而来，修改Bart为BlenderbotSmall
class FlaxBlenderbotSmallModule(nn.Module):
    # 类变量config，类型为BlenderbotSmallConfig，用于存储模型配置信息
    config: BlenderbotSmallConfig
    # 类变量dtype，默认为jnp.float32，表示计算过程中的数据类型
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    # 初始化方法，设置共享的嵌入层，编码器和解码器模块
    def setup(self):
        self.shared = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.init_std),
            dtype=self.dtype,
        )

        # 初始化编码器模块，使用小型Blenderbot编码器
        self.encoder = FlaxBlenderbotSmallEncoder(self.config, dtype=self.dtype, embed_tokens=self.shared)
        # 初始化解码器模块，使用小型Blenderbot解码器，共享相同的嵌入层
        self.decoder = FlaxBlenderbotSmallDecoder(self.config, dtype=self.dtype, embed_tokens=self.shared)

    # 返回当前对象中的编码器模块
    def _get_encoder_module(self):
        return self.encoder

    # 返回当前对象中的解码器模块
    def _get_decoder_module(self):
        return self.decoder

    # 实现对象的调用接口，用于进行序列到序列的转换任务
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
        # 编码器模块处理输入序列，生成编码器输出
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        # 解码器模块处理解码器输入序列，使用编码器输出来辅助生成解码器输出
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            position_ids=decoder_position_ids,
            encoder_hidden_states=encoder_outputs[0],  # 使用编码器的隐藏状态作为解码器的输入
            encoder_attention_mask=attention_mask,      # 使用编码器的注意力掩码
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        # 如果不要求返回字典形式，则将编码器和解码器输出直接拼接返回
        if not return_dict:
            return decoder_outputs + encoder_outputs

        # 返回经过序列到序列模型包装的输出结果
        return FlaxSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
# 定义一个自定义的 Flax 模型类，继承自 FlaxPreTrainedModel
class FlaxBlenderbotSmallPreTrainedModel(FlaxPreTrainedModel):
    # 设置配置类为 BlenderbotSmallConfig
    config_class = BlenderbotSmallConfig
    # 基础模型前缀为 "model"
    base_model_prefix: str = "model"
    # 模块类初始化为 None，将在实例化时赋值

    def __init__(
        self,
        config: BlenderbotSmallConfig,
        input_shape: Tuple[int] = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        # 使用模块类创建模块实例
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 调用父类的初始化方法
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量 input_ids，数据类型为整型
        input_ids = jnp.zeros(input_shape, dtype="i4")
        # 确保初始化步骤适用于 FlaxBlenderbotSmallForSequenceClassificationModule
        input_ids = input_ids.at[(..., -1)].set(self.config.eos_token_id)
        # 初始化 attention_mask 为全 1 的张量，与 input_ids 形状相同
        attention_mask = jnp.ones_like(input_ids)
        # 将 decoder_input_ids 初始化为 input_ids
        decoder_input_ids = input_ids
        # 将 decoder_attention_mask 初始化为全 1 的张量，与 input_ids 形状相同
        decoder_attention_mask = jnp.ones_like(input_ids)

        # 获取 batch_size 和 sequence_length
        batch_size, sequence_length = input_ids.shape
        # 初始化 position_ids 为广播后的序列索引张量
        position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))
        # 初始化 decoder_position_ids 为广播后的序列索引张量
        decoder_position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # 分割随机数生成器
        params_rng, dropout_rng = jax.random.split(rng)
        # 创建随机数字典
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 使用模块的初始化方法生成随机参数
        random_params = self.module.init(
            rngs,
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            position_ids,
            decoder_position_ids,
        )["params"]

        # 如果传入了已有的参数，则将随机生成的参数与已有参数合并
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            # 否则，直接返回随机生成的参数
            return random_params
    # 初始化缓存用于快速自回归解码
    def init_cache(self, batch_size, max_length, encoder_outputs):
        r"""
        Args:
            batch_size (`int`):
                用于快速自回归解码的批处理大小。定义了初始化缓存的批处理大小。
            max_length (`int`):
                自回归解码的最大可能长度。定义了初始化缓存的序列长度。
            encoder_outputs (`Union[FlaxBaseModelOutput, tuple(tuple(jnp.ndarray)]`):
                `encoder_outputs` 包括 (`last_hidden_state`, *可选*: `hidden_states`, *可选*: `attentions`)。
                `last_hidden_state` 的形状为 `(batch_size, sequence_length, hidden_size)`，
                *可选* 是编码器最后一层输出的隐藏状态序列。用于解码器的交叉注意力。

        """
        # 初始化解码器的输入 ID，全部为1
        decoder_input_ids = jnp.ones((batch_size, max_length), dtype="i4")
        # 解码器的注意力掩码与输入 ID 相同，全部为1
        decoder_attention_mask = jnp.ones_like(decoder_input_ids)
        # 解码器的位置 ID，广播到与输入 ID 相同的形状
        decoder_position_ids = jnp.broadcast_to(
            jnp.arange(jnp.atleast_2d(decoder_input_ids).shape[-1]), decoder_input_ids.shape
        )

        # 定义内部函数 `_decoder_forward`，用于调用解码器模块
        def _decoder_forward(module, decoder_input_ids, decoder_attention_mask, decoder_position_ids, **kwargs):
            decoder_module = module._get_decoder_module()
            return decoder_module(
                decoder_input_ids,
                decoder_attention_mask,
                decoder_position_ids,
                **kwargs,
            )

        # 初始化模型的变量，用于初始化缓存
        init_variables = self.module.init(
            jax.random.PRNGKey(0),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            encoder_hidden_states=encoder_outputs[0],
            init_cache=True,
            method=_decoder_forward,  # 只需调用解码器来初始化缓存
        )
        # 解冻并返回初始化的缓存变量
        return unfreeze(init_variables["cache"])

    @add_start_docstrings(BLENDERBOT_SMALL_ENCODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutput, config_class=BlenderbotSmallConfig)
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
        >>> from transformers import AutoTokenizer, FlaxBlenderbotSmallForConditionalGeneration

        >>> model = FlaxBlenderbotSmallForConditionalGeneration.from_pretrained("facebook/blenderbot_small-90M")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot_small-90M")

        >>> text = "My friends are cool but they eat too many carbs."
        >>> inputs = tokenizer(text, max_length=1024, return_tensors="np")
        >>> encoder_outputs = model.encode(**inputs)
        ```"""
        # 初始化输出注意力的设置，如果未指定则使用模型配置的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 初始化输出隐藏状态的设置，如果未指定则使用模型配置的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 初始化返回字典的设置，如果未指定则使用模型配置的默认值
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 如果未提供注意力掩码，则创建一个全为1的注意力掩码，与输入张量形状相同
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        # 如果未提供位置编码，则使用输入张量的形状创建位置编码
        if position_ids is None:
            batch_size, sequence_length = input_ids.shape
            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # 如果需要处理任何伪随机数生成器，则创建一个空字典来存储这些伪随机数生成器
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        # 定义一个内部函数来执行编码器的前向传播
        def _encoder_forward(module, input_ids, attention_mask, position_ids, **kwargs):
            encode_module = module._get_encoder_module()
            return encode_module(input_ids, attention_mask, position_ids, **kwargs)

        # 调用模型的 apply 方法，执行编码器的前向传播
        return self.module.apply(
            {"params": params or self.params},  # 使用给定的参数或默认参数执行模型前向传播
            input_ids=jnp.array(input_ids, dtype="i4"),  # 将输入张量转换为 Flax 所需的数据类型和格式
            attention_mask=jnp.array(attention_mask, dtype="i4"),  # 将注意力掩码转换为 Flax 所需的数据类型和格式
            position_ids=jnp.array(position_ids, dtype="i4"),  # 将位置编码转换为 Flax 所需的数据类型和格式
            output_attentions=output_attentions,  # 指定是否输出注意力
            output_hidden_states=output_hidden_states,  # 指定是否输出隐藏状态
            return_dict=return_dict,  # 指定是否以字典形式返回结果
            deterministic=not train,  # 指定是否处于训练模式
            rngs=rngs,  # 提供任何伪随机数生成器
            method=_encoder_forward,  # 指定执行的方法
        )

    @add_start_docstrings(BLENDERBOT_SMALL_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=FlaxBaseModelOutputWithPastAndCrossAttentions, config_class=BlenderbotSmallConfig
    )
    # 定义解码方法，接受一系列输入参数，并可选地返回一个字典形式的输出
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
        # 设置输出注意力权重的选项，如果未指定则使用配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置输出隐藏状态的选项，如果未指定则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置返回字典的选项，如果未指定则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 准备编码器输入
        if attention_mask is None:
            # 如果未提供注意力遮罩，则创建一个全为1的遮罩，形状与input_ids相同
            attention_mask = jnp.ones_like(input_ids)
        if position_ids is None:
            # 如果未提供位置编码，则根据input_ids的形状创建位置编码
            batch_size, sequence_length = input_ids.shape
            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # 准备解码器输入
        if decoder_input_ids is None:
            # 如果未提供解码器输入的token ids，则通过向右移动input_ids创建解码器的输入
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, decoder_start_token_id=self.config.decoder_start_token_id
            )
        if decoder_attention_mask is None:
            # 如果未提供解码器的注意力遮罩，则创建一个全为1的遮罩，形状与decoder_input_ids相同
            decoder_attention_mask = jnp.ones_like(decoder_input_ids)
        if decoder_position_ids is None:
            # 如果未提供解码器的位置编码，则根据decoder_input_ids的形状创建位置编码
            batch_size, sequence_length = decoder_input_ids.shape
            decoder_position_ids = jnp.broadcast_to(
                jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
            )

        # 处理需要的任何随机数生成器（PRNG）
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        # 调用模块的apply方法，传递所需参数和设置
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
# 添加文档字符串到类定义，描述 BlenderbotSmall 模型的基本信息和功能
@add_start_docstrings(
    "The bare BlenderbotSmall Model transformer outputting raw hidden-states without any specific head on top.",
    BLENDERBOT_SMALL_START_DOCSTRING,
)
# 定义 FlaxBlenderbotSmallModel 类，继承自 FlaxBlenderbotSmallPreTrainedModel 类
class FlaxBlenderbotSmallModel(FlaxBlenderbotSmallPreTrainedModel):
    # 配置信息为 BlenderbotSmallConfig 类型的对象
    config: BlenderbotSmallConfig
    # 计算使用的数据类型为 jnp.float32
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    # 模块类为 FlaxBlenderbotSmallModule
    module_class = FlaxBlenderbotSmallModule

# 调用函数 append_call_sample_docstring，添加样例调用文档字符串到 FlaxBlenderbotSmallModel 类中
append_call_sample_docstring(FlaxBlenderbotSmallModel, _CHECKPOINT_FOR_DOC, FlaxSeq2SeqModelOutput, _CONFIG_FOR_DOC)


# 从 transformers.models.bart.modeling_flax_bart.FlaxBartForConditionalGenerationModule 复制而来，将 Bart 改为 BlenderbotSmall
# 定义 FlaxBlenderbotSmallForConditionalGenerationModule 类，继承自 nn.Module
class FlaxBlenderbotSmallForConditionalGenerationModule(nn.Module):
    # 配置信息为 BlenderbotSmallConfig 类型的对象
    config: BlenderbotSmallConfig
    # 计算使用的数据类型为 jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 偏置初始化函数为 jax.nn.initializers.zeros
    bias_init: Callable[..., jnp.ndarray] = jax.nn.initializers.zeros

    # 设置函数，初始化模型和 lm_head
    def setup(self):
        # 使用配置和数据类型初始化 FlaxBlenderbotSmallModule 模型
        self.model = FlaxBlenderbotSmallModule(config=self.config, dtype=self.dtype)
        # 初始化 lm_head，使用 Dense 层，无偏置，数据类型为 dtype，初始化方式为正态分布
        self.lm_head = nn.Dense(
            self.model.shared.num_embeddings,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        # 初始化 final_logits_bias，作为模型参数，维度为 (1, num_embeddings)，初始化方式为 bias_init
        self.final_logits_bias = self.param("final_logits_bias", self.bias_init, (1, self.model.shared.num_embeddings))

    # 获取编码器模块
    def _get_encoder_module(self):
        return self.model.encoder

    # 获取解码器模块
    def _get_decoder_module(self):
        return self.model.decoder

    # 定义 __call__ 方法，接受多个输入参数和标志位，执行条件生成任务
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
        # 使用模型进行推理，返回包含输出的字典
        outputs = self.model(
            input_ids=input_ids,  # 输入的token IDs
            attention_mask=attention_mask,  # 输入的注意力掩码
            decoder_input_ids=decoder_input_ids,  # 解码器的token IDs
            decoder_attention_mask=decoder_attention_mask,  # 解码器的注意力掩码
            position_ids=position_ids,  # 位置编码
            decoder_position_ids=decoder_position_ids,  # 解码器位置编码
            output_attentions=output_attentions,  # 是否输出注意力权重
            output_hidden_states=output_hidden_states,  # 是否输出隐藏状态
            return_dict=return_dict,  # 是否返回字典格式的输出
            deterministic=deterministic,  # 是否确定性推断
        )

        hidden_states = outputs[0]  # 提取模型输出的隐藏状态

        if self.config.tie_word_embeddings:
            # 如果配置了共享词嵌入，从模型变量中获取共享的嵌入层
            shared_embedding = self.model.variables["params"]["shared"]["embedding"]
            # 应用共享嵌入到隐藏状态上得到语言模型的logits
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
        else:
            # 否则直接使用语言模型头部处理隐藏状态得到logits
            lm_logits = self.lm_head(hidden_states)

        # 将最终logits加上偏置项，使用jax中的stop_gradient函数确保偏置项不参与梯度计算
        lm_logits += jax.lax.stop_gradient(self.final_logits_bias.astype(self.dtype))

        if not return_dict:
            # 如果不返回字典格式的输出，则将logits和其它输出作为元组返回
            output = (lm_logits,) + outputs[1:]
            return output

        # 返回FlaxSeq2SeqLMOutput格式的输出，包括logits和其它相关的隐藏状态和注意力权重
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
    "The BLENDERBOT_SMALL Model with a language modeling head. Can be used for summarization.",
    BLENDERBOT_SMALL_START_DOCSTRING,
)
class FlaxBlenderbotSmallForConditionalGeneration(FlaxBlenderbotSmallPreTrainedModel):
    module_class = FlaxBlenderbotSmallForConditionalGenerationModule
    dtype: jnp.dtype = jnp.float32

    @add_start_docstrings(BLENDERBOT_SMALL_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxCausalLMOutputWithCrossAttentions, config_class=BlenderbotSmallConfig)
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
        Decodes the input sequence using the model for conditional generation.

        Args:
            decoder_input_ids: Tensor of decoder input IDs.
            encoder_outputs: Output of the encoder model.
            encoder_attention_mask: Optional tensor indicating which positions in the encoder output should not be attended to.
            decoder_attention_mask: Optional tensor specifying which positions in the decoder input should not be attended to.
            decoder_position_ids: Optional tensor specifying positional IDs for the decoder input.
            past_key_values: Optional dictionary containing cached key-value pairs for fast decoding.
            output_attentions: Whether to output attentions.
            output_hidden_states: Whether to output hidden states.
            return_dict: Whether to return a dictionary.
            deterministic: Whether to apply deterministic computation.
            params: Optional parameters for the model.
            dropout_rng: Random number generator for dropout.

        Returns:
            FlaxCausalLMOutputWithCrossAttentions: Model outputs including logits, past key values, and optionally attentions and hidden states.
        """
        # Function body is implemented in the actual method, no further comment needed here.
        pass

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
        Prepares inputs for the generation process.

        Args:
            decoder_input_ids: Tensor of decoder input IDs.
            max_length: Maximum length of the generated sequence.
            attention_mask: Optional tensor indicating which positions should be attended to.
            decoder_attention_mask: Optional tensor specifying which positions in the decoder input should not be attended to.
            encoder_outputs: Optional outputs of the encoder model.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: Dictionary containing prepared inputs for the generation process.
                Includes past key values, encoder outputs, encoder attention mask, decoder attention mask, and decoder position IDs.
        """
        # initializing the cache
        batch_size, seq_length = decoder_input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length, encoder_outputs)

        # Note that usually one would have to put 0's in the attention_mask for x > input_ids.shape[-1] and x < cache_length.
        # But since the decoder uses a causal mask, those positions are masked anyways.
        # Thus we can create a single static attention_mask here, which is more efficient for compilation

        # Create an extended attention mask for the decoder
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if decoder_attention_mask is not None:
            # Calculate position IDs from decoder_attention_mask
            position_ids = decoder_attention_mask.cumsum(axis=-1) - 1
            # Update the extended_attention_mask with decoder_attention_mask values
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, decoder_attention_mask, (0, 0))
        else:
            # Broadcast positional IDs if decoder_attention_mask is not provided
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
        Updates model inputs for the generation process based on model outputs.

        Args:
            model_outputs: Outputs from the model.
            model_kwargs: Original input arguments for the model.

        Returns:
            dict: Updated model input arguments including past key values and adjusted decoder position IDs.
        """
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["decoder_position_ids"] = model_kwargs["decoder_position_ids"][:, -1:] + 1
        return model_kwargs
    # 导入所需的库和模型
    >>> from transformers import AutoTokenizer, FlaxBlenderbotSmallForConditionalGeneration
    
    # 使用预训练的 Blenderbot 模型初始化生成模型
    >>> model = FlaxBlenderbotSmallForConditionalGeneration.from_pretrained("facebook/blenderbot_small-90M")
    # 使用预训练的 tokenizer 初始化分词器
    >>> tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot_small-90M")
    
    # 待总结的文章内容
    >>> ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
    # 使用 tokenizer 处理文章，限定最大长度为 1024，并转换为 NumPy 数据结构
    >>> inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="np")
    
    # 生成摘要
    >>> summary_ids = model.generate(inputs["input_ids"]).sequences
    # 解码生成的摘要内容，去除特殊标记并保留原始分词方式
    >>> print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))
    
    # 掩码填充示例：
    
    >>> from transformers import AutoTokenizer, FlaxBlenderbotSmallForConditionalGeneration
    
    # 使用预训练的 tokenizer 初始化分词器
    >>> tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot_small-90M")
    # 待处理的文本带有掩码标记
    >>> TXT = "My friends are <mask> but they eat too many carbs."
    
    # 使用预训练的 Blenderbot 模型初始化生成模型
    >>> model = FlaxBlenderbotSmallForConditionalGeneration.from_pretrained("facebook/blenderbot_small-90M")
    # 将文本转换为输入的 token IDs，并转换为 NumPy 数据结构
    >>> input_ids = tokenizer([TXT], return_tensors="np")["input_ids"]
    # 获取模型的 logits
    >>> logits = model(input_ids).logits
    
    # 确定掩码位置的索引
    >>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
    # 对 logits 应用 softmax 函数，沿着指定的轴计算概率
    >>> probs = jax.nn.softmax(logits[0, masked_index], axis=0)
    # 获取概率最高的前 k 个预测结果和它们的值
    >>> values, predictions = jax.lax.top_k(probs)
    
    # 解码预测结果并按空格分割成单词列表
    >>> tokenizer.decode(predictions).split()
"""
给 FlaxBlenderbotSmallForConditionalGeneration 类的调用覆盖文档字符串，
使用 BLENDERBOT_SMALL_INPUTS_DOCSTRING 和 FLAX_BLENDERBOT_SMALL_CONDITIONAL_GENERATION_DOCSTRING 进行扩展。
"""
overwrite_call_docstring(
    FlaxBlenderbotSmallForConditionalGeneration,
    BLENDERBOT_SMALL_INPUTS_DOCSTRING + FLAX_BLENDERBOT_SMALL_CONDITIONAL_GENERATION_DOCSTRING,
)

"""
为 FlaxBlenderbotSmallForConditionalGeneration 类附加或替换返回文档字符串，
设置输出类型为 FlaxSeq2SeqLMOutput，配置类为 _CONFIG_FOR_DOC。
"""
append_replace_return_docstrings(
    FlaxBlenderbotSmallForConditionalGeneration, output_type=FlaxSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC
)
```