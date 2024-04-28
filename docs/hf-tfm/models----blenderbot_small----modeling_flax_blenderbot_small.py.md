# `.\transformers\models\blenderbot_small\modeling_flax_blenderbot_small.py`

```
# 使用 UTF-8 编码
# 版权声明，版权归 Facebook, Inc. 和 HuggingFace Inc. 团队所有
#
# 根据 Apache 许可证 2.0 版本（“许可证”）许可；
# 除非符合许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件，
# 不提供任何形式的担保或条件。
# 有关特定语言下的权限，请参阅许可证。
""" BlenderbotSmall 模型。"""

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

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的模型检查点和配置
_CHECKPOINT_FOR_DOC = "facebook/blenderbot_small-90M"
_CONFIG_FOR_DOC = "BlenderbotSmallConfig"

# BlenderbotSmall 的起始文档字符串
BLENDERBOT_SMALL_START_DOCSTRING = r"""
    此模型继承自 [`FlaxPreTrainedModel`]。检查超类文档以了解库实现的通用方法（例如下载或保存、调整输入嵌入、修剪头等）。

    此模型还是 Flax Linen
    [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) 子类。将其用作常规 Flax 模块，并参考 Flax 文档以获取与一般用法和行为相关的所有内容。

    最后，此模型支持固有的 JAX 功能，例如：

    - [即时编译（JIT）](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [自动微分](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [向量化](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [并行化](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)
``` 
    # 接受两个参数：config（BlenderbotSmallConfig 类型）和 dtype（可选，默认为 jax.numpy.float32）
    # config：模型配置类，包含模型的所有参数。使用配置文件初始化不会加载模型关联的权重，只加载配置。可使用 FlaxPreTrainedModel.from_pretrained 方法加载模型权重。
    # dtype：计算的数据类型。可以是 jax.numpy.float32、jax.numpy.float16（在 GPU 上）、jax.numpy.bfloat16（在 TPU 上）之一。
    # 这可用于在 GPU 或 TPU 上启用混合精度训练或半精度推断。如果指定了 dtype，则所有计算将使用给定的 dtype 执行。
    # **请注意，这只指定计算的数据类型，不影响模型参数的数据类型。**
    # 如果要更改模型参数的数据类型，请参阅 FlaxPreTrainedModel.to_fp16 和 FlaxPreTrainedModel.to_bf16。
# Blenderbot Small 模型的输入文档字符串，用于编码输入
BLENDERBOT_SMALL_INPUTS_DOCSTRING = r"""
"""

# Blenderbot Small 模型编码输入的文档字符串，包括输入的各种参数说明
BLENDERBOT_SMALL_ENCODE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
            输入序列标记在词汇表中的索引。默认情况下，将忽略填充。

            可以使用 [`AutoTokenizer`] 来获取索引。详细信息请参见 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`]。

            [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            避免在填充的标记索引上执行注意力的掩码。掩码值在 `[0, 1]` 之间：

            - 1 表示 **未掩码** 的标记，
            - 0 表示 **已掩码** 的标记。

            [什么是注意力掩码？](../glossary#attention-mask)
        position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            输入序列标记在位置嵌入中的位置索引。在范围 `[0, config.max_position_embeddings - 1]` 中选择。
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。有关详细信息，请参见返回张量下的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。有关详细信息，请参见返回张量下的 `hidden_states`。
        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`] 而不是简单的元组。
"""

# Blenderbot Small 模型解码输入的文档字符串，暂无内容
BLENDERBOT_SMALL_DECODE_INPUTS_DOCSTRING = r"""
"""

# 从 transformers.models.bart.modeling_flax_bart.shift_tokens_right 复制过来的函数，将输入的标记向右移动一个标记
def shift_tokens_right(input_ids: jnp.ndarray, pad_token_id: int, decoder_start_token_id: int) -> jnp.ndarray:
    """
    向右移动输入的标记一个标记位。
    """
    shifted_input_ids = jnp.zeros_like(input_ids)
    shifted_input_ids = shifted_input_ids.at[:, 1:].set(input_ids[:, :-1])
    shifted_input_ids = shifted_input_ids.at[:, 0].set(decoder_start_token_id)

    shifted_input_ids = jnp.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)
    return shifted_input_ids


# 从 transformers.models.bart.modeling_flax_bart.FlaxBartAttention 复制过来的类，用于 Blenderbot Small 模型的注意力机制
class FlaxBlenderbotSmallAttention(nn.Module):
    config: BlenderbotSmallConfig
    embed_dim: int
    num_heads: int
    dropout: float = 0.0
    causal: bool = False
    bias: bool = True
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型
    # 初始化函数，用于设置注意力头的维度和投影函数等
    def setup(self) -> None:
        # 计算每个注意力头的维度
        self.head_dim = self.embed_dim // self.num_heads
        # 检查是否能够整除，否则引发 ValueError 异常
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )

        # 创建部分函数，用于定义全连接层的参数
        dense = partial(
            nn.Dense,
            self.embed_dim,
            use_bias=self.bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )

        # 初始化查询、键、值的投影函数和输出的投影函数
        self.q_proj, self.k_proj, self.v_proj = dense(), dense(), dense()
        self.out_proj = dense()

        # 初始化 dropout 层
        self.dropout_layer = nn.Dropout(rate=self.dropout)

        # 如果是因果注意力，则创建因果掩码
        if self.causal:
            # 创建因果掩码
            self.causal_mask = make_causal_mask(
                jnp.ones((1, self.config.max_position_embeddings), dtype="bool"), dtype="bool"
            )

    # 将隐藏状态张量按注意力头分割
    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    # 将隐藏状态张量中的注意力头合并
    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    # 使用 JAX 的 nn.compact 装饰器，定义一个紧凑的模块
    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slightly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # 检测是否通过缺少现有缓存数据来进行初始化。
        is_initialized = self.has_variable("cache", "cached_key")
        # 获取或创建缓存的键（key）变量，如果不存在则创建一个与输入键相同形状和数据类型的全零数组。
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        # 获取或创建缓存的值（value）变量，如果不存在则创建一个与输入值相同形状和数据类型的全零数组。
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        # 获取或创建缓存索引变量，如果不存在则创建一个初始值为0的32位整数。
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        # 如果已经初始化了缓存
        if is_initialized:
            # 获取除了批处理维度以外的维度，以及缓存的键的最大长度、头数和每个头的深度。
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # 更新键（key）和值（value）缓存，使用新的1D空间切片。
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            # 更新缓存的键（key）和值（value）变量
            cached_key.value = key
            cached_value.value = value
            # 计算已更新的缓存向量数量，并更新缓存索引。
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # 用于缓存的解码器自注意力的因果遮罩：我们的单个查询位置应该只注意已生成并缓存的键位置，而不是剩余的零元素。
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            # 将因果遮罩和输入的注意力遮罩结合起来。
            attention_mask = combine_masks(pad_mask, attention_mask)
        # 返回更新后的键、值和注意力遮罩。
        return key, value, attention_mask

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        key_value_states: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        deterministic: bool = True,
# 从 transformers.models.bart.modeling_flax_bart.FlaxBartEncoderLayer 复制而来，将 Bart 模型改为 BlenderbotSmall 模型
class FlaxBlenderbotSmallEncoderLayer(nn.Module):
    # 定义模型配置，指定数据类型为 jnp.float32
    config: BlenderbotSmallConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        # 设置嵌入维度为模型配置中的 d_model
        self.embed_dim = self.config.d_model
        # 创建自注意力层对象，用于计算自注意力机制
        self.self_attn = FlaxBlenderbotSmallAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.encoder_attention_heads,
            dropout=self.config.attention_dropout,
            dtype=self.dtype,
        )
        # 创建自注意力层的 LayerNorm 层，用于归一化
        self.self_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        # 创建 Dropout 层，用于随机丢弃部分数据
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)
        # 根据配置选择激活函数，并创建激活函数对象
        self.activation_fn = ACT2FN[self.config.activation_function]
        # 创建激活函数的 Dropout 层，用于激活函数后的随机丢弃
        self.activation_dropout_layer = nn.Dropout(rate=self.config.activation_dropout)
        # 创建第一个全连接层，用于前馈神经网络的第一层计算
        self.fc1 = nn.Dense(
            self.config.encoder_ffn_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        # 创建第二个全连接层，用于前馈神经网络的第二层计算
        self.fc2 = nn.Dense(
            self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std)
        )
        # 创建最终的 LayerNorm 层，用于归一化输出
        self.final_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        output_attentions: bool = True,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray]:
        # 备份输入的隐藏状态，用于残差连接
        residual = hidden_states
        # 对输入的隐藏状态进行自注意力计算
        hidden_states, attn_weights = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask)

        # 对自注意力计算的结果应用 Dropout
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 残差连接
        hidden_states = residual + hidden_states
        # 对残差连接后的结果应用 LayerNorm
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 备份残差连接后的结果，用于后续残差连接
        residual = hidden_states
        # 对残差连接后的结果应用激活函数和第一个全连接层计算
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 对第一个全连接层计算结果应用 Dropout
        hidden_states = self.activation_dropout_layer(hidden_states, deterministic=deterministic)
        # 对第二个全连接层计算
        hidden_states = self.fc2(hidden_states)
        # 对第二个全连接层计算结果应用 Dropout
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 残差连接
        hidden_states = residual + hidden_states
        # 对残差连接后的结果应用 LayerNorm
        hidden_states = self.final_layer_norm(hidden_states)

        # 输出模型的最终隐藏状态
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则添加到输出中
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# 从 transformers.models.bart.modeling_flax_bart.FlaxBartEncoderLayerCollection 复制而来，将 Bart 模型改为 BlenderbotSmall 模型
class FlaxBlenderbotSmallEncoderLayerCollection(nn.Module):
    # 定义模型配置，指定数据类型为 jnp.float32
    config: BlenderbotSmallConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        # 创建一系列 BlenderbotSmallEncoderLayer 层组成的列表，根据配置中的 encoder_layers 参数
        self.layers = [
            FlaxBlenderbotSmallEncoderLayer(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.encoder_layers)
        ]
        # 设置层间的 Dropout 比率
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
# 从transformers.models.bart.modeling_flax_bart.FlaxBartDecoderLayer复制代码，将Bart->BlenderbotSmall
class FlaxBlenderbotSmallDecoderLayer(nn.Module):
    # BlenderbotSmallDecoderLayer的配置信息
    config: BlenderbotSmallConfig
    # 数据类型，默认为32位浮点数
    dtype: jnp.dtype = jnp.float32

    # 初始化函数，设置层的参数
    def setup(self) -> None:
        # 嵌入维度等于配置中的模型维度
        self.embed_dim = self.config.d_model
        # 创建自注意力层对象，用于处理自注意力机制
        self.self_attn = FlaxBlenderbotSmallAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.decoder_attention_heads,
            dropout=self.config.attention_dropout,
            causal=True,  # 自注意力层是否是有因果性的，即是否允许信息流向未来的位置
            dtype=self.dtype,
        )
        # 用于在自注意力层输出后应用dropout
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)
        # 激活函数，根据配置中的激活函数类型选择相应的函数
        self.activation_fn = ACT2FN[self.config.activation_function]
        # 用于在激活函数输出后应用dropout
        self.activation_dropout_layer = nn.Dropout(rate=self.config.activation_dropout)

        # 自注意力层输出后的LayerNorm层
        self.self_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        # 编码器-解码器注意力层，处理解码器对编码器隐藏状态的注意力
        self.encoder_attn = FlaxBlenderbotSmallAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.decoder_attention_heads,
            dropout=self.config.attention_dropout,
            dtype=self.dtype,
        )
        # 编码器-解码器注意力层输出后的LayerNorm层
        self.encoder_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        # 第一个全连接层，用于解码器中的前馈神经网络
        self.fc1 = nn.Dense(
            self.config.decoder_ffn_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),  # 使用正态分布初始化权重
        )
        # 第二个全连接层，用于解码器中的前馈神经网络
        self.fc2 = nn.Dense(
            self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std)
        )
        # 最终的LayerNorm层，对解码器层输出进行归一化处理
        self.final_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    # 调用函数，处理解码器层的前向传播
    def __call__(
        self,
        hidden_states: jnp.ndarray,  # 输入的隐藏状态
        attention_mask: jnp.ndarray,  # 注意力掩码
        encoder_hidden_states: Optional[jnp.ndarray] = None,  # 编码器的隐藏状态（可选）
        encoder_attention_mask: Optional[jnp.ndarray] = None,  # 编码器的注意力掩码（可选）
        init_cache: bool = False,  # 是否初始化缓存
        output_attentions: bool = True,  # 是否输出注意力权重
        deterministic: bool = True,  # 是否使用确定性操作
    ) -> Tuple[jnp.ndarray]:
        # 保存残差连接
        residual = hidden_states

        # 自注意力机制
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, init_cache=init_cache
        )
        # 使用 dropout 层进行正则化
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 添加残差连接
        hidden_states = residual + hidden_states
        # 使用层归一化
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 交叉注意力块
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            # 保存残差连接
            residual = hidden_states

            # 编码器注意力机制
            hidden_states, cross_attn_weights = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            # 使用 dropout 层进行正则化
            hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
            # 添加残差连接
            hidden_states = residual + hidden_states
            # 使用层归一化
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # 全连接层
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 使用激活函数的 dropout 层进行正则化
        hidden_states = self.activation_dropout_layer(hidden_states, deterministic=deterministic)
        hidden_states = self.fc2(hidden_states)
        # 使用 dropout 层进行正则化
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 添加残差连接
        hidden_states = residual + hidden_states
        # 使用层归一化
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs
# 从transformers.models.bart.modeling_flax_bart.FlaxBartDecoderLayerCollection复制代码，并将Bart->BlenderbotSmall
class FlaxBlenderbotSmallDecoderLayerCollection(nn.Module):
    # BlenderbotSmall配置
    config: BlenderbotSmallConfig
    # 计算的数据类型
    dtype: jnp.dtype = jnp.float32  

    def setup(self):
        # 创建BlenderbotSmallDecoderLayer对象列表
        self.layers = [
            FlaxBlenderbotSmallDecoderLayer(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.decoder_layers)
        ]
        # decoder layer的dropout概率
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
        # 如果需要输出交叉注意力权重且encoder_hidden_states不为空，则初始化空元组
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        # 遍历decoder layers
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                # 添加LayerDrop（参见https://arxiv.org/abs/1909.11556的描述）

            # 生成0到1之间的随机dropout概率
            dropout_probability = random.uniform(0, 1)
            # 如果非确定性且dropout概率小于layerdrop，则将layer_outputs设置为(None, None, None)
            if not deterministic and (dropout_probability < self.layerdrop):
                layer_outputs = (None, None, None)
            else:
                # 调用decoder_layer进行前向传播
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

        # 添加来自最后一个decoder layer的隐藏状态
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 输出结果
        outputs = [hidden_states, all_hidden_states, all_self_attns, all_cross_attentions]

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class FlaxBlenderbotSmallEncoder(nn.Module):
    # BlenderbotSmall配置
    config: BlenderbotSmallConfig
    # 嵌入的标记
    embed_tokens: nn.Embed
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    # 定义变量 dtype，表示计算时所采用的数据类型，默认为 jnp.float32

    def setup(self):
        # 设置模型的初始化操作
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)
        # 初始化 dropout 层，使用给定的丢弃率

        embed_dim = self.config.d_model
        # 获取模型配置中的 d_model 参数作为嵌入维度
        self.padding_idx = self.config.pad_token_id
        # 获取模型配置中的 pad_token_id 参数作为填充索引
        self.max_source_positions = self.config.max_position_embeddings
        # 获取模型配置中的 max_position_embeddings 参数作为最大位置编码数
        self.embed_scale = math.sqrt(embed_dim) if self.config.scale_embedding else 1.0
        # 计算嵌入缩放因子，如果配置了 scale_embedding 则使用 sqrt(embed_dim)，否则为 1.0

        self.embed_positions = nn.Embed(
            self.config.max_position_embeddings,
            embed_dim,
            embedding_init=jax.nn.initializers.normal(self.config.init_std),
        )
        # 初始化位置嵌入层，使用正态分布初始化位置嵌入矩阵
        self.layers = FlaxBlenderbotSmallEncoderLayerCollection(self.config, self.dtype)
        # 初始化编码器层集合，使用给定的模型配置和数据类型
        self.layernorm_embedding = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        # 初始化嵌入层的 LayerNorm 层，使用给定的数据类型和 epsilon 值

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
        # 定义模型的调用方法
        input_shape = input_ids.shape
        # 获取输入张量的形状
        input_ids = input_ids.reshape(-1, input_shape[-1])
        # 将输入张量变形为二维张量

        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        # 获取输入张量的嵌入表示，并缩放嵌入值

        embed_pos = self.embed_positions(position_ids)
        # 获取位置嵌入

        hidden_states = inputs_embeds + embed_pos
        # 将输入嵌入和位置嵌入相加得到隐藏状态
        hidden_states = self.layernorm_embedding(hidden_states)
        # 对隐藏状态进行 LayerNorm 归一化
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 对隐藏状态进行 dropout 操作，根据 deterministic 参数决定是否确定性

        outputs = self.layers(
            hidden_states,
            attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 对隐藏状态进行编码器层的前向传播

        if not return_dict:
            return outputs
        # 如果不返回字典形式的输出，则直接返回编码器层的输出

        return FlaxBaseModelOutput(
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # 返回 Flax 模型的输出，包括最后隐藏状态、隐藏状态序列和注意力权重
# 定义一个基于Flax的BlenderbotSmall解码器模块
class FlaxBlenderbotSmallDecoder(nn.Module):
    # BlenderbotSmall配置信息
    config: BlenderbotSmallConfig
    # 词嵌入矩阵
    embed_tokens: nn.Embed
    # 计算的数据类型，默认为32位浮点数
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    # 模块初始化函数
    def setup(self):
        # 初始化一个dropout层，根据配置的dropout率
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)

        # 获取词嵌入维度
        embed_dim = self.config.d_model
        # 获取padding标记的索引
        self.padding_idx = self.config.pad_token_id
        # 获取最大目标位置数
        self.max_target_positions = self.config.max_position_embeddings
        # 如果配置中开启了词嵌入缩放，计算词嵌入缩放因子
        self.embed_scale = math.sqrt(self.config.d_model) if self.config.scale_embedding else 1.0

        # 初始化位置嵌入层
        self.embed_positions = nn.Embed(
            self.config.max_position_embeddings,
            embed_dim,
            # 使用正态分布初始化位置嵌入
            embedding_init=jax.nn.initializers.normal(self.config.init_std),
        )

        # 初始化解码器层
        self.layers = FlaxBlenderbotSmallDecoderLayerCollection(self.config, self.dtype)
        # 初始化嵌入层的LayerNorm层
        self.layernorm_embedding = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    # 模块的调用函数
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
        # 将输入张量展平，除最后一个维度
        input_ids = input_ids.reshape(-1, input_shape[-1])

        # 获取输入的词嵌入
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # 嵌入位置信息
        positions = self.embed_positions(position_ids)

        # 在解码器中对输入的词嵌入进行LayerNorm
        inputs_embeds = self.layernorm_embedding(inputs_embeds)
        # 将词嵌入和位置嵌入相加得到最终的隐藏状态
        hidden_states = inputs_embeds + positions

        # 对隐藏状态应用dropout
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)

        # 调用解码器层进行解码
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

        # 如果不返回字典形式的结果，则直接返回outputs
        if not return_dict:
            return outputs

        # 返回带有过去注意力权重和交叉注意力权重的FlaxBaseModelOutputWithPastAndCrossAttentions对象
        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


# 从transformers.models.bart.modeling_flax_bart.FlaxBartModule中复制的代码，将Bart替换为BlenderbotSmall
class FlaxBlenderbotSmallModule(nn.Module):
    # BlenderbotSmall的配置信息
    config: BlenderbotSmallConfig
    # 计算的数据类型，默认为32位浮点数
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
``` 
    # 初始化模型参数
    def setup(self):
        # 创建共享的嵌入层，用于编码器和解码器
        self.shared = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.init_std),
            dtype=self.dtype,
        )

        # 创建编码器对象
        self.encoder = FlaxBlenderbotSmallEncoder(self.config, dtype=self.dtype, embed_tokens=self.shared)
        # 创建解码器对象
        self.decoder = FlaxBlenderbotSmallDecoder(self.config, dtype=self.dtype, embed_tokens=self.shared)

    # 获取编码器模块
    def _get_encoder_module(self):
        return self.encoder

    # 获取解码器模块
    def _get_decoder_module(self):
        return self.decoder

    # 模型调用函数
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
        # 编码器前向传播
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        # 解码器前向传播
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

        # 返回包含模型输出的字典
        return FlaxSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
class FlaxBlenderbotSmallPreTrainedModel(FlaxPreTrainedModel):
    # 使用 BlenderbotSmallConfig 配置类
    config_class = BlenderbotSmallConfig
    # 设置基础模型前缀
    base_model_prefix: str = "model"
    # 模块类，默认为空
    module_class: nn.Module = None

    def __init__(
        self,
        config: BlenderbotSmallConfig,
        input_shape: Tuple[int] = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        # 创建模块实例，根据传入参数配置
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 调用父类的初始化方法
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量，全零张量，数据类型为整数
        input_ids = jnp.zeros(input_shape, dtype="i4")
        # 确保初始化传递给 FlaxBlenderbotSmallForSequenceClassificationModule 的 token_id 为 EOS token_id
        input_ids = input_ids.at[(..., -1)].set(self.config.eos_token_id)
        # 初始化注意力掩码，全一张量，与输入张量形状相同
        attention_mask = jnp.ones_like(input_ids)
        # 初始化解码器输入张量为输入张量
        decoder_input_ids = input_ids
        # 初始化解码器注意力掩码为全一张量，与输入张量形状相同
        decoder_attention_mask = jnp.ones_like(input_ids)

        # 获取输入张量的形状，批次大小和序列长度
        batch_size, sequence_length = input_ids.shape
        # 初始化位置编码，将序列长度广播到与输入张量形状相同
        position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))
        # 初始化解码器位置编码，将序列长度广播到与输入张量形状相同
        decoder_position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # 使用随机数生成器拆分随机数源
        params_rng, dropout_rng = jax.random.split(rng)
        # 组合随机数源
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 使用模块初始化参数
        random_params = self.module.init(
            rngs,
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            position_ids,
            decoder_position_ids,
        )["params"]

        # 如果提供了参数，则使用提供的参数，否则返回随机初始化的参数
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params
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
                `last_hidden_state` 的形状为 `(batch_size, sequence_length, hidden_size)`，*可选*）是编码器最后一层的隐藏状态输出序列。
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
        return unfreeze(init_variables["cache"])

    # 编码输入文本
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
        返回：
            返回解码器的输出结果。

        示例：
        
        ```python
        >>> from transformers import AutoTokenizer, FlaxBlenderbotSmallForConditionalGeneration

        >>> model = FlaxBlenderbotSmallForConditionalGeneration.from_pretrained("facebook/blenderbot_small-90M")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot_small-90M")

        >>> text = "My friends are cool but they eat too many carbs."
        >>> inputs = tokenizer(text, max_length=1024, return_tensors="np")
        >>> encoder_outputs = model.encode(**inputs)
        ```"""
        # 如果未提供参数，则使用模型配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 如果未提供注意力掩码，则创建全为1的掩码
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        # 如果未提供位置编码，则创建一个与输入形状相同的位置编码
        if position_ids is None:
            batch_size, sequence_length = input_ids.shape
            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # 处理任何可能需要的 PRNG
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        def _encoder_forward(module, input_ids, attention_mask, position_ids, **kwargs):
            # 获取编码器模块
            encode_module = module._get_encoder_module()
            return encode_module(input_ids, attention_mask, position_ids, **kwargs)

        # 应用模型的正向传播
        return self.module.apply(
            {"params": params or self.params},  # 使用给定参数或者模型当前参数
            input_ids=jnp.array(input_ids, dtype="i4"),  # 输入的标记 ID
            attention_mask=jnp.array(attention_mask, dtype="i4"),  # 注意力掩码
            position_ids=jnp.array(position_ids, dtype="i4"),  # 位置 ID
            output_attentions=output_attentions,  # 是否返回注意力权重
            output_hidden_states=output_hidden_states,  # 是否返回隐藏状态
            return_dict=return_dict,  # 是否返回字典格式的输出
            deterministic=not train,  # 是否为确定性推理
            rngs=rngs,  # PRNG 键
            method=_encoder_forward,  # 使用编码器正向传播方法
        )

    @add_start_docstrings(BLENDERBOT_SMALL_DECODE_INPUTS_DOCSTRING)  # 添加解码器输入的文档字符串
    @replace_return_docstrings(
        output_type=FlaxBaseModelOutputWithPastAndCrossAttentions, config_class=BlenderbotSmallConfig
    )
    def decode(
        self,
        decoder_input_ids,  # 解码器的输入标记 ID
        encoder_outputs,  # 编码器的输出
        encoder_attention_mask: Optional[jnp.ndarray] = None,  # 编码器的注意力掩码
        decoder_attention_mask: Optional[jnp.ndarray] = None,  # 解码器的注意力掩码
        decoder_position_ids: Optional[jnp.ndarray] = None,  # 解码器的位置编码
        past_key_values: dict = None,  # 上下文的键值对
        output_attentions: Optional[bool] = None,  # 是否返回注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否返回隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典格式的输出
        train: bool = False,  # 是否为训练模式
        params: dict = None,  # 参数
        dropout_rng: PRNGKey = None,  # 随机数生成器
    # 定义一个方法，用于模型调用
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
        # 确定是否输出注意力权重
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 确定是否输出隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 确定是否返回字典格式的结果
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 准备编码器输入
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        if position_ids is None:
            batch_size, sequence_length = input_ids.shape
            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # 准备解码器输入
        if decoder_input_ids is None:
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

        # 处理需要的随机数生成器
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        # 调用模型的apply方法进行前向传播
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
# 导入必要的模块
@add_start_docstrings(
    "The bare BlenderbotSmall Model transformer outputting raw hidden-states without any specific head on top.",
    BLENDERBOT_SMALL_START_DOCSTRING,
)
# 定义 FlaxBlenderbotSmallModel 类，继承自 FlaxBlenderbotSmallPreTrainedModel 类
class FlaxBlenderbotSmallModel(FlaxBlenderbotSmallPreTrainedModel):
    # 模型配置信息
    config: BlenderbotSmallConfig
    # 计算时所使用的数据类型
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    # 模型类别
    module_class = FlaxBlenderbotSmallModule

# 添加调用示例的文档字符串
append_call_sample_docstring(FlaxBlenderbotSmallModel, _CHECKPOINT_FOR_DOC, FlaxSeq2SeqModelOutput, _CONFIG_FOR_DOC)

# 从 transformers.models.bart.modeling_flax_bart.FlaxBartForConditionalGenerationModule 复制而来，将 Bart 替换为 BlenderbotSmall
# 定义 FlaxBlenderbotSmallForConditionalGenerationModule 类，继承自 nn.Module 类
class FlaxBlenderbotSmallForConditionalGenerationModule(nn.Module):
    # 模型配置信息
    config: BlenderbotSmallConfig
    # 计算时所使用的数据类型
    dtype: jnp.dtype = jnp.float32
    # 偏置的初始化函数
    bias_init: Callable[..., jnp.ndarray] = jax.nn.initializers.zeros

    # 模型初始化设置
    def setup(self):
        # 初始化模型
        self.model = FlaxBlenderbotSmallModule(config=self.config, dtype=self.dtype)
        # 初始化语言模型头部
        self.lm_head = nn.Dense(
            self.model.shared.num_embeddings,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        # 初始化最终 logits 的偏置
        self.final_logits_bias = self.param("final_logits_bias", self.bias_init, (1, self.model.shared.num_embeddings))

    # 获取编码器模块
    def _get_encoder_module(self):
        return self.model.encoder

    # 获取解码器模块
    def _get_decoder_module(self):
        return self.model.decoder

    # 模型调用函数
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
        # 使用模型进行前向传播，获取模型输出
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

        # 获取模型输出中的隐藏状态
        hidden_states = outputs[0]

        # 如果配置了共享词嵌入，则使用共享的词嵌入参数计算语言模型的logits
        if self.config.tie_word_embeddings:
            # 获取共享的词嵌入参数
            shared_embedding = self.model.variables["params"]["shared"]["embedding"]
            # 使用共享的词嵌入参数计算语言模型的logits
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
        else:
            # 否则直接使用语言模型头计算logits
            lm_logits = self.lm_head(hidden_states)

        # 将偏置添加到语言模型的logits中
        lm_logits += jax.lax.stop_gradient(self.final_logits_bias.astype(self.dtype))

        # 如果不返回字典，则返回模型输出
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return output

        # 如果返回字典，则构建 FlaxSeq2SeqLMOutput 对象并返回
        return FlaxSeq2SeqLMOutput(
            logits=lm_logits,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
```  
# 添加文档字符串，描述了使用BLENDERBOT_SMALL模型进行文本摘要的功能
class FlaxBlenderbotSmallForConditionalGeneration(FlaxBlenderbotSmallPreTrainedModel):
    # 指定模型类为FlaxBlenderbotSmallForConditionalGenerationModule
    module_class = FlaxBlenderbotSmallForConditionalGenerationModule
    # 指定数据类型为jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 添加解码输入的文档字符串
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

        # 使用init_cache方法初始化past_key_values
        past_key_values = self.init_cache(batch_size, max_length, encoder_outputs)
        
        # 创建扩展的注意力掩码
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

    # 更新生成的输入
    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["decoder_position_ids"] = model_kwargs["decoder_position_ids"][:, -1:] + 1
        return model_kwargs
    # 导入所需的库
    from transformers import AutoTokenizer, FlaxBlenderbotSmallForConditionalGeneration

    # 从预训练模型中加载BlenderbotSmall模型
    model = FlaxBlenderbotSmallForConditionalGeneration.from_pretrained("facebook/blenderbot_small-90M")
    # 从预训练模型中加载分词器
    tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot_small-90M")

    # 待总结的文章内容
    ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
    # 使用分词器对文章进行编码，限制最大长度为1024，返回numpy数组
    inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="np")

    # 生成摘要
    summary_ids = model.generate(inputs["input_ids"]).sequences
    # 解码生成的摘要并打印
    print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))


Mask filling example:


    # 导入所需的库
    from transformers import AutoTokenizer, FlaxBlenderbotSmallForConditionalGeneration

    # 从预训练模型中加载分词器
    tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot_small-90M")
    # 待填充mask的文本
    TXT = "My friends are <mask> but they eat too many carbs."

    # 从预训练模型中加载BlenderbotSmall模型
    model = FlaxBlenderbotSmallForConditionalGeneration.from_pretrained("facebook/blenderbot_small-90M")
    # 使用分词器对文本进行编码，返回numpy数组中的input_ids
    input_ids = tokenizer([TXT], return_tensors="np")["input_ids"]
    # 获取模型的logits
    logits = model(input_ids).logits

    # 找到mask的位置
    masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
    # 对logits进行softmax处理
    probs = jax.nn.softmax(logits[0, masked_index], axis=0)
    # 获取最有可能的值和预测
    values, predictions = jax.lax.top_k(probs)

    # 解码预测的结果并以列表形式返回
    tokenizer.decode(predictions).split()
# 调用函数 overwrite_call_docstring，用于覆盖指定类的调用文档字符串
overwrite_call_docstring(
    FlaxBlenderbotSmallForConditionalGeneration,  # 被覆盖的类为 FlaxBlenderbotSmallForConditionalGeneration
    BLENDERBOT_SMALL_INPUTS_DOCSTRING + FLAX_BLENDERBOT_SMALL_CONDITIONAL_GENERATION_DOCSTRING,  # 调用文档字符串为 BLENDERBOT_SMALL_INPUTS_DOCSTRING 与 FLAX_BLENDERBOT_SMALL_CONDITIONAL_GENERATION_DOCSTRING 的组合
)

# 在指定类的返回文档字符串中添加或替换内容
append_replace_return_docstrings(
    FlaxBlenderbotSmallForConditionalGeneration,  # 操作的目标类为 FlaxBlenderbotSmallForConditionalGeneration
    output_type=FlaxSeq2SeqLMOutput,  # 输出类型为 FlaxSeq2SeqLMOutput
    config_class=_CONFIG_FOR_DOC  # 配置类为 _CONFIG_FOR_DOC
)
```