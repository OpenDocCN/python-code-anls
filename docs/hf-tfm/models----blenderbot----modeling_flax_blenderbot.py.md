# `.\transformers\models\blenderbot\modeling_flax_blenderbot.py`

```
# 设置文件编码为 utf-8
# 版权声明，版权归 Fairseq 作者、Google Flax 团队作者和 HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本，除非符合许可证，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”基础分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取特定语言的权限和限制
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

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的配置和检查点信息
_CONFIG_FOR_DOC = "BlenderbotConfig"
_CHECKPOINT_FOR_DOC = "facebook/blenderbot-400M-distill"

# Blenderbot 模型的起始文档字符串
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
``` 
    Parameters:
        config ([`BlenderbotConfig`]): Model configuration class with all the parameters of the model.
            初始化一个配置文件类，包含模型的所有参数。
            使用配置文件初始化模型时不会加载与模型关联的权重，仅加载配置信息。
            可以查看 [`~FlaxPreTrainedModel.from_pretrained`] 方法来加载模型权重。
"""

BLENDERBOT_INPUTS_DOCSTRING = r"""
"""


BLENDERBOT_ENCODE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
            输入序列标记在词汇表中的索引。默认情况下，将忽略填充。

            可以使用 [`AutoTokenizer`] 获得索引。有关详细信息，请参阅 [`PreTrainedTokenizer.encode`] 和
            [`PreTrainedTokenizer.__call__`]。

            [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            避免在填充标记索引上执行注意力计算的掩码。掩码的值选在 `[0, 1]` 之间：

            - 1 表示 **未掩码** 的标记，
            - 0 表示 **掩码** 的标记。

            [什么是注意力掩码？](../glossary#attention-mask)
        position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            每个输入序列标记在位置嵌入中的位置索引。选取范围在 `[0, config.max_position_embeddings - 1]` 之间。
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。有关更多详细信息，请参阅返回的张量中的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。有关更多详细信息，请参阅返回的张量中的 `hidden_states`。
        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""

BLENDERBOT_DECODE_INPUTS_DOCSTRING = r"""
"""


# 从 transformers.models.bart.modeling_flax_bart.shift_tokens_right 复制而来
def shift_tokens_right(input_ids: jnp.ndarray, pad_token_id: int, decoder_start_token_id: int) -> jnp.ndarray:
    """
    将输入 ID 向右移动一个标记位。
    """
    shifted_input_ids = jnp.zeros_like(input_ids)
    shifted_input_ids = shifted_input_ids.at[:, 1:].set(input_ids[:, :-1])
    shifted_input_ids = shifted_input_ids.at[:, 0].set(decoder_start_token_id)

    shifted_input_ids = jnp.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)
    return shifted_input_ids


# 从 transformers.models.bart.modeling_flax_bart.FlaxBartAttention 复制而来，将 Bart->Blenderbot
class FlaxBlenderbotAttention(nn.Module):
    config: BlenderbotConfig
    embed_dim: int
    num_heads: int
    dropout: float = 0.0
    causal: bool = False
    bias: bool = True
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型

``` 
    # 设置函数，初始化参数
    def setup(self) -> None:
        # 计算每个头的维度
        self.head_dim = self.embed_dim // self.num_heads
        # 检查 embed_dim 是否可以被 num_heads 整除
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )

        # 部分应用 nn.Dense 函数，创建用于计算 Q、K、V 的全连接层
        dense = partial(
            nn.Dense,
            self.embed_dim,
            use_bias=self.bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )

        # 创建 Q、K、V 的全连接层
        self.q_proj, self.k_proj, self.v_proj = dense(), dense(), dense()
        # 创建输出层的全连接层
        self.out_proj = dense()

        # 创建 Dropout 层
        self.dropout_layer = nn.Dropout(rate=self.dropout)

        # 如果是因果注意力机制，创建因果掩码
        if self.causal:
            self.causal_mask = make_causal_mask(
                jnp.ones((1, self.config.max_position_embeddings), dtype="bool"), dtype="bool"
            )

    # 将隐藏状态分割成多个头
    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    # 合并多个头的隐藏状态
    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    # 使用 nn.compact 装饰器
    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slightly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # 检测是否通过缺少现有缓存数据来初始化。
        is_initialized = self.has_variable("cache", "cached_key")
        # 获取缓存的键值，如果未初始化则创建全零数组
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        # 获取缓存的值，如果未初始化则创建全零数组
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        # 获取缓存索引，如果未初始化则创建值为0的整数
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        # 如果已初始化
        if is_initialized:
            # 提取批量维度及缓存键值状态的形状信息
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # 更新键、值缓存，将新的一维空间片段替换原有数据
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            # 更新缓存的键值
            cached_key.value = key
            cached_value.value = value
            # 计算更新的缓存向量数量，并更新缓存索引
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # 为缓存的解码器自注意力创建因果掩码：我们的单个查询位置应仅关注已生成并缓存的键位置，而不是剩余的零元素。
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            # 合并填充掩码和注意力掩码
            attention_mask = combine_masks(pad_mask, attention_mask)
        # 返回更新后的键、值、注意力掩码
        return key, value, attention_mask

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        key_value_states: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        deterministic: bool = True,
# 定义了一个 FlaxBlenderbotEncoderLayer 类，继承自 nn.Module 类，用于 Blenderbot 的编码器层
class FlaxBlenderbotEncoderLayer(nn.Module):
    # 类型注释，指定了 Blenderbot 的配置对象
    config: BlenderbotConfig
    # 指定了数据类型为 jnp.float32 的 dtype
    dtype: jnp.dtype = jnp.float32

    # 初始化方法
    def setup(self) -> None:
        # 获取配置中的编码器维度
        self.embed_dim = self.config.d_model
        # 创建自注意力层对象，使用 BlenderbotAttention 类
        self.self_attn = FlaxBlenderbotAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.encoder_attention_heads,
            dropout=self.config.attention_dropout,
            dtype=self.dtype,
        )
        # 创建自注意力层后的 LayerNorm 层
        self.self_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        # 创建 Dropout 层
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)
        # 获取激活函数
        self.activation_fn = ACT2FN[self.config.activation_function]
        # 创建激活函数后的 Dropout 层
        self.activation_dropout_layer = nn.Dropout(rate=self.config.activation_dropout)
        # 创建全连接层1，输入维度为编码器 FNN 维度，输出维度为编码器维度
        self.fc1 = nn.Dense(
            self.config.encoder_ffn_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        # 创建全连接层2，输入维度为编码器维度，输出维度为编码器维度
        self.fc2 = nn.Dense(
            self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std)
        )
        # 创建最终的 LayerNorm 层
        self.final_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    # 前向传播方法
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        output_attentions: bool = True,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray]:
        # 残差连接
        residual = hidden_states
        # LayerNorm 层
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # 自注意力机制
        hidden_states, attn_weights = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask)
        # Dropout
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 残差连接
        hidden_states = residual + hidden_states

        # 残差连接
        residual = hidden_states
        # 最终的 LayerNorm 层
        hidden_states = self.final_layer_norm(hidden_states)
        # 激活函数
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 激活函数后的 Dropout
        hidden_states = self.activation_dropout_layer(hidden_states, deterministic=deterministic)
        # 全连接层2
        hidden_states = self.fc2(hidden_states)
        # Dropout
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 残差连接
        hidden_states = residual + hidden_states

        # 输出结果
        outputs = (hidden_states,)

        # 如果需要输出注意力权重
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# 定义了一个 FlaxBlenderbotEncoderLayerCollection 类，继承自 nn.Module 类，用于 Blenderbot 的编码器层集合
class FlaxBlenderbotEncoderLayerCollection(nn.Module):
    # 类型注释，指定了 Blenderbot 的配置对象
    config: BlenderbotConfig
    # 指定了数据类型为 jnp.float32 的 dtype
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    # 初始化方法
    def setup(self):
        # 创建编码器层集合，包含多个编码器层
        self.layers = [
            FlaxBlenderbotEncoderLayer(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.encoder_layers)
        ]
        # 获取层之间的 dropout 率
        self.layerdrop = self.config.encoder_layerdrop
    # 定义 __call__ 方法，用于执行模型的前向传播
    def __call__(
        self,
        hidden_states,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 如果需要输出注意力矩阵，则初始化一个空元组以存储所有注意力矩阵
        all_attentions = () if output_attentions else None
        # 如果需要输出隐藏状态，则初始化一个空元组以存储所有隐藏状态
        all_hidden_states = () if output_hidden_states else None

        # 遍历每个编码器层
        for encoder_layer in self.layers:
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到所有隐藏状态元组中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            # 添加 LayerDrop（参见 https://arxiv.org/abs/1909.11556 进行描述）
            # 生成一个0到1之间的随机数，表示dropout的概率
            dropout_probability = random.uniform(0, 1)
            # 如果不是确定性的且随机数小于层丢弃率，则跳过这一层
            if not deterministic and (dropout_probability < self.layerdrop):  # skip the layer
                # 将层的输出置为空
                layer_outputs = (None, None)
            else:
                # 否则执行编码器层的前向传播
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions,
                    deterministic,
                )
            # 更新隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]
            # 如果需要输出注意力矩阵，则将当前层的注意力矩阵添加到所有注意力矩阵元组中
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # 如果需要输出隐藏状态，则将最后一个隐藏状态添加到所有隐藏状态元组中
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 将隐藏状态、所有隐藏状态和所有注意力矩阵组成一个元组作为模型输出
        outputs = (hidden_states, all_hidden_states, all_attentions)

        # 如果不需要返回字典，则将输出元组中的非空值组成一个元组返回
        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        # 否则返回一个 FlaxBaseModelOutput 对象，其中包含最后一个隐藏状态、所有隐藏状态和所有注意力矩阵
        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )
# 从transformers.models.mbart.modeling_flax_mbart.FlaxMBartDecoderLayer复制代码，并将MBart->Blenderbot
class FlaxBlenderbotDecoderLayer(nn.Module):
    # 定义类属性config为BlenderbotConfig，dtype为jnp.float32
    config: BlenderbotConfig
    dtype: jnp.dtype = jnp.float32

    # 初始化函数
    def setup(self) -> None:
        # 设置embed_dim为d_model
        self.embed_dim = self.config.d_model
        # 初始化self_attn为FlaxBlenderbotAttention对象，传入相关参数
        self.self_attn = FlaxBlenderbotAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.decoder_attention_heads,
            dropout=self.config.attention_dropout,
            causal=True,
            dtype=self.dtype,
        )
        # 初始化dropout_layer为Dropout对象，传入dropout率
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)
        # 设置activation_fn为激活函数
        self.activation_fn = ACT2FN[self.config.activation_function]
        # 初始化activation_dropout_layer为Dropout对象，传入激活函数的dropout率
        self.activation_dropout_layer = nn.Dropout(rate=self.config.activation_dropout)

        # 初始化self_attn_layer_norm为LayerNorm对象，设置数据类型和epsilon
        self.self_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        # 初始化encoder_attn为FlaxBlenderbotAttention对象，传入相关参数
        self.encoder_attn = FlaxBlenderbotAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.decoder_attention_heads,
            dropout=self.config.attention_dropout,
            dtype=self.dtype,
        )
        # 初始化encoder_attn_layer_norm为LayerNorm对象，设置数据类型和epsilon
        self.encoder_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        # 初始化fc1为Dense对象，设置输出维度、数据类型和初始化方式
        self.fc1 = nn.Dense(
            self.config.decoder_ffn_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        # 初始化fc2为Dense对象，设置输出维度、数据类型和初始化方式
        self.fc2 = nn.Dense(
            self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std)
        )
        # 初始化final_layer_norm为LayerNorm对象，设置数据类型和epsilon

    # 调用函数
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        output_attentions: bool = True,
        deterministic: bool = True,
    # 定义函数，并指定其返回值类型为包含单个 NumPy 数组的元组
    ) -> Tuple[jnp.ndarray]:
        # 将输入的隐藏状态存储到 residual 变量中，以备后续添加到处理后的隐藏状态中
        residual = hidden_states
        # 对隐藏状态进行自注意力机制的处理前，先对其进行 Layer Normalization
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        # 对隐藏状态进行自注意力机制的处理，返回处理后的隐藏状态和自注意力权重
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, init_cache=init_cache
        )
        # 对处理后的隐藏状态进行 Dropout 处理
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 将处理前的隐藏状态和处理后的隐藏状态相加，得到自注意力机制的结果
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        # 如果存在编码器的隐藏状态，则执行交叉注意力机制
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            # 将输入的隐藏状态存储到 residual 变量中，以备后续添加到处理后的隐藏状态中
            residual = hidden_states

            # 对隐藏状态进行编码器注意力机制前，先对其进行 Layer Normalization
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            # 对隐藏状态进行编码器注意力机制的处理，返回处理后的隐藏状态和交叉注意力权重
            hidden_states, cross_attn_weights = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            # 对处理后的隐藏状态进行 Dropout 处理
            hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
            # 将处理前的隐藏状态和处理后的隐藏状态相加，得到编码器注意力机制的结果
            hidden_states = residual + hidden_states

        # Fully Connected
        # 将输入的隐藏状态存储到 residual 变量中，以备后续添加到处理后的隐藏状态中
        residual = hidden_states
        # 对隐藏状态进行最终的 Layer Normalization
        hidden_states = self.final_layer_norm(hidden_states)
        # 对隐藏状态进行激活函数处理
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 对激活后的隐藏状态进行 Dropout 处理
        hidden_states = self.activation_dropout_layer(hidden_states, deterministic=deterministic)
        # 对隐藏状态进行线性变换
        hidden_states = self.fc2(hidden_states)
        # 对线性变换后的隐藏状态进行 Dropout 处理
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 将处理前的隐藏状态和处理后的隐藏状态相加，得到最终的输出
        hidden_states = residual + hidden_states

        # 构造输出元组
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则将自注意力权重和交叉注意力权重添加到输出元组中
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        # 返回处理后的隐藏状态和可能的注意力权重
        return outputs
# 从transformers.models.bart.modeling_flax_bart.FlaxBartDecoderLayerCollection复制代码，并将Bart->Blenderbot
class FlaxBlenderbotDecoderLayerCollection(nn.Module):
    # Blenderbot配置
    config: BlenderbotConfig
    # 计算的数据类型
    dtype: jnp.dtype = jnp.float32  

    def setup(self):
        # 创建Blenderbot解码器层集合
        self.layers = [
            FlaxBlenderbotDecoderLayer(self.config, name=str(i), dtype=self.dtype)
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
        # 如果需要输出隐藏状态，则初始化空元组
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力权重，则初始化空元组
        all_self_attns = () if output_attentions else None
        # 如果需要输出交叉注意力权重且编码器隐藏状态不为空，则初始化空元组
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        # 遍历解码器层
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                # 添加LayerDrop（参见https://arxiv.org/abs/1909.11556的描述）
            dropout_probability = random.uniform(0, 1)
            if not deterministic and (dropout_probability < self.layerdrop):
                layer_outputs = (None, None, None)
            else:
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

        # 添加最后一个解码器层的隐藏状态
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


class FlaxBlenderbotEncoder(nn.Module):
    # Blenderbot配置
    config: BlenderbotConfig
    # 嵌入标记
    embed_tokens: nn.Embed
    dtype: jnp.dtype = jnp.float32  # 定义变量dtype，表示计算的数据类型为jnp.float32

    def setup(self):
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)  # 初始化dropout层，使用配置中的dropout率

        embed_dim = self.config.d_model  # 获取配置中的d_model作为嵌入维度
        self.padding_idx = self.config.pad_token_id  # 获取配置中的pad_token_id作为填充索引
        self.max_source_positions = self.config.max_position_embeddings  # 获取配置中的max_position_embeddings作为最大位置编码
        self.embed_scale = math.sqrt(embed_dim) if self.config.scale_embedding else 1.0  # 计算嵌入缩放因子

        self.embed_positions = nn.Embed(
            self.config.max_position_embeddings,
            embed_dim,
            embedding_init=jax.nn.initializers.normal(self.config.init_std),
        )  # 初始化位置嵌入层

        self.layers = FlaxBlenderbotEncoderLayerCollection(self.config, self.dtype)  # 初始化编码器层集合
        self.layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)  # 初始化LayerNorm层

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
        input_shape = input_ids.shape  # 获取输入数据的形状
        input_ids = input_ids.reshape(-1, input_shape[-1])  # 重塑输入数据的形状

        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale  # 获取嵌入的输入数据并乘以嵌入缩放因子

        embed_pos = self.embed_positions(position_ids)  # 获取位置嵌入

        hidden_states = inputs_embeds + embed_pos  # 将嵌入的输入数据和位置嵌入相加
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)  # 应用dropout层

        outputs = self.layers(
            hidden_states,
            attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )  # 调用编码器层集合进行计算
        last_hidden_states = outputs[0]  # 获取输出中的最后一个隐藏状态
        last_hidden_states = self.layer_norm(last_hidden_states)  # 应用LayerNorm层

        # 在应用上述`layernorm`后更新`hidden_states`中的最后一个元素
        hidden_states = None
        if output_hidden_states:
            hidden_states = outputs[1]
            hidden_states = hidden_states[:-1] + (last_hidden_states,)  # 更新隐藏状��列表

        if not return_dict:
            outputs = (last_hidden_states, hidden_states) + (outputs[2:] if output_hidden_states else outputs[1:])  # 更新输出结果
            return tuple(v for v in outputs if v is not None)  # 返回更新后的输出结果

        return FlaxBaseModelOutput(
            last_hidden_state=last_hidden_states,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
        )  # 返回模型输出结果
# 定义一个继承自 nn.Module 的类 FlaxBlenderbotDecoder
class FlaxBlenderbotDecoder(nn.Module):
    # 定义类属性 config 为 BlenderbotConfig 类型
    config: BlenderbotConfig
    # 定义类属性 embed_tokens 为 nn.Embed 类型
    embed_tokens: nn.Embed
    # 定义类属性 dtype 为 jnp.float32 类型，表示计算的数据类型

    # 定义 setup 方法
    def setup(self):
        # 初始化 dropout_layer 层，设置丢弃率为 config.dropout
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)

        # 获取 embed_dim 为 config.d_model
        embed_dim = self.config.d_model
        # 获取 padding_idx 为 config.pad_token_id
        self.padding_idx = self.config.pad_token_id
        # 获取 max_target_positions 为 config.max_position_embeddings
        self.max_target_positions = self.config.max_position_embeddings
        # 设置 embed_scale 为 config.d_model 的平方根或者 1.0（根据 config.scale_embedding 的值）

        # 初始化 embed_positions 层，设置最大位置数量为 config.max_position_embeddings，嵌入维度为 embed_dim
        self.embed_positions = nn.Embed(
            self.config.max_position_embeddings,
            embed_dim,
            embedding_init=jax.nn.initializers.normal(self.config.init_std),
        )

        # 初始化 layers 为 FlaxBlenderbotDecoderLayerCollection 类型对象
        self.layers = FlaxBlenderbotDecoderLayerCollection(self.config, self.dtype)
        # 初始化 layer_norm 层，设置数据类型为 dtype，epsilon 为 1e-05

    # 定义 __call__ 方法
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
        # 获取 input_ids 的形状
        input_shape = input_ids.shape
        # 将 input_ids 重塑为二维数组
        input_ids = input_ids.reshape(-1, input_shape[-1])

        # 获取 inputs_embeds 为经过 embed_tokens 层处理后的结果乘以 embed_scale
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # embed positions
        # 获取 positions 为经过 embed_positions 层处理后的结果
        positions = self.embed_positions(position_ids)

        # 计算 hidden_states 为 inputs_embeds 和 positions 的和
        hidden_states = inputs_embeds + positions
        # 对 hidden_states 进行 dropout 处理
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)

        # 调用 layers 处理 hidden_states
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

        # 获取最后的隐藏状态 last_hidden_states
        last_hidden_states = outputs[0]
        # 对 last_hidden_states 进行 layer_norm 处理
        last_hidden_states = self.layer_norm(last_hidden_states)

        # 更新 hidden_states 中的最后一个元素，应用 layernorm 处理后
        hidden_states = None
        if output_hidden_states:
            hidden_states = outputs[1]
            hidden_states = hidden_states[:-1] + (last_hidden_states,)

        # 如果不返回字典形式的结果
        if not return_dict:
            # 重新组合 outputs
            outputs = (last_hidden_states, hidden_states) + (outputs[2:] if output_hidden_states else outputs[1:])
            return tuple(v for v in outputs if v is not None)

        # 返回 FlaxBaseModelOutputWithPastAndCrossAttentions 类型对象
        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=last_hidden_states,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


# 从 transformers.models.bart.modeling_flax_bart.FlaxBartModule 复制而来，将 Bart 替换为 Blenderbot
# 定义一个继承自 nn.Module 的类 FlaxBlenderbotModule
class FlaxBlenderbotModule(nn.Module):
    # 定义类属性 config 为 BlenderbotConfig 类型
    config: BlenderbotConfig
    # 定义类属性 dtype 为 jnp.float32 类型，默认值为 jnp.float32，用于计算的数据类型

    # 定义 setup 方法
    def setup(self):
        # 初始化 shared 属性为 nn.Embed 对象，包含词汇表大小、模型维度等参数
        self.shared = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.init_std),
            dtype=self.dtype,
        )

        # 初始化 encoder 属性为 FlaxBlenderbotEncoder 对象，传入配置参数和共享的 embed_tokens
        self.encoder = FlaxBlenderbotEncoder(self.config, dtype=self.dtype, embed_tokens=self.shared)
        # 初始化 decoder 属性为 FlaxBlenderbotDecoder 对象，传入配置参数和共享的 embed_tokens

    # 定义 _get_encoder_module 方法，返回 encoder 属性
    def _get_encoder_module(self):
        return self.encoder

    # 定义 _get_decoder_module 方法，返回 decoder 属性
    def _get_decoder_module(self):
        return self.decoder

    # 定义 __call__ 方法，接收多个参数，执行 encoder 和 decoder 的前向传播
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
        # 调用 encoder 进行前向传播
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        # 调用 decoder 进行前向传播
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

        # ��果 return_dict 为 False，则返回 decoder_outputs 和 encoder_outputs
        if not return_dict:
            return decoder_outputs + encoder_outputs

        # 返回 FlaxSeq2SeqModelOutput 对象，包含 decoder 和 encoder 的输出
        return FlaxSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


# 定义一个继承自 FlaxPreTrainedModel 的类 FlaxBlenderbotPreTrainedModel
class FlaxBlenderbotPreTrainedModel(FlaxPreTrainedModel):
    # 定义类属性 config_class 为 BlenderbotConfig 类型
    config_class = BlenderbotConfig
    # 定义类属性 base_model_prefix 为字符串 "model"
    base_model_prefix: str = "model"
    # 定义类属性 module_class 为 nn.Module ���型，默认值为 None

    # 定义初始化方法，接收配置参数、输入形状、种子、数据类型等参数
    def __init__(
        self,
        config: BlenderbotConfig,
        input_shape: Tuple[int] = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        # 使用配置和关键字参数实例化模块对象
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 调用父类的初始化方法
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量
        input_ids = jnp.zeros(input_shape, dtype="i4")
        # 确保初始化过程适用于FlaxBlenderbotForSequenceClassificationModule
        input_ids = input_ids.at[(..., -1)].set(self.config.eos_token_id)
        attention_mask = jnp.ones_like(input_ids)
        decoder_input_ids = input_ids
        decoder_attention_mask = jnp.ones_like(input_ids)

        batch_size, sequence_length = input_ids.shape
        # 生成位置编码
        position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))
        decoder_position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # 分割随机数生成器
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 初始化模块参数
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
            # 展开参数字典
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
                用于快速自回归解码的批大小。定义了初始化缓存的批大小。
            max_length (`int`):
                自回归解码的最大可能长度。定义了初始化缓存的序列长度。
            encoder_outputs (`Union[FlaxBaseModelOutput, tuple(tuple(jnp.ndarray)]`):
                `encoder_outputs` 包含 (`last_hidden_state`, *可选*: `hidden_states`, *可选*: `attentions`)。
                `last_hidden_state` 的形状为 `(batch_size, sequence_length, hidden_size)`，*可选* 是编码器最后一层的隐藏状态的序列。
                在解码器的交叉注意力中使用。
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

        # 使用解码器初始化缓存
        init_variables = self.module.init(
            jax.random.PRNGKey(0),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            encoder_hidden_states=encoder_outputs[0],
            init_cache=True,
            method=_decoder_forward,  # 只需要调用解码器来初始化缓存
        )
        return unfreeze(init_variables["cache"])

    @add_start_docstrings(BLENDERBOT_ENCODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutput, config_class=BlenderbotConfig)
    # 编码函数，用于将输入编码为隐藏状态
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
        返回:

        示例:

        ```python
        >>> from transformers import AutoTokenizer, FlaxBlenderbotForConditionalGeneration

        >>> model = FlaxBlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")

        >>> text = "My friends are cool but they eat too many carbs."
        >>> inputs = tokenizer(text, max_length=1024, return_tensors="jax")
        >>> encoder_outputs = model.encode(**inputs)
        ```"""
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

        # 处理可能的 PRNG
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

    @add_start_docstrings(BLENDERBOT_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=FlaxBaseModelOutputWithPastAndCrossAttentions, config_class=BlenderbotConfig
    )
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
    @add_start_docstrings_to_model_forward(BLENDERBOT_INPUTS_DOCSTRING)
    # 定义一个__call__方法，用于模型的调用
    def __call__(
        self,
        input_ids: jnp.ndarray,  # 输入的token IDs
        attention_mask: Optional[jnp.ndarray] = None,  # 输入的attention mask，默认为None
        decoder_input_ids: Optional[jnp.ndarray] = None,  # 解码器的输入token IDs，默认为None
        decoder_attention_mask: Optional[jnp.ndarray] = None,  # 解码器的attention mask，默认为None
        position_ids: Optional[jnp.ndarray] = None,  # 输入的位置编码，默认为None
        decoder_position_ids: Optional[jnp.ndarray] = None,  # 解码器的位置编码，默认为None
        output_attentions: Optional[bool] = None,  # 是否输出attention，默认为None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，默认为None
        return_dict: Optional[bool] = None,  # 是否返回字典，默认为None
        train: bool = False,  # 是否为训练模式，默认为False
        params: dict = None,  # 模型参数，默认为None
        dropout_rng: PRNGKey = None,  # 随机数生成器，默认为None
    ):
        # 如果未指定输出attention，则使用配置中的设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未指定输出隐藏状态，则使用配置中的设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定返回字典，则使用配置中的设置
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 准备编码器的输入
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)  # 如果没有提供attention mask，则创建一个全1的mask
        if position_ids is None:
            batch_size, sequence_length = input_ids.shape
            # 如果没有提供位置编码，则创建一个与输入shape相同的位置编码
            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # 准备解码器的输入
        if decoder_input_ids is None:
            # 如果没有提供解码器的输入token IDs，则右移输入token IDs并填充pad token
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, decoder_start_token_id=self.config.decoder_start_token_id
            )
        if decoder_attention_mask is None:
            decoder_attention_mask = jnp.ones_like(decoder_input_ids)  # 创建一个全1的attention mask
        if decoder_position_ids is None:
            batch_size, sequence_length = decoder_input_ids.shape
            # 如果没有提供解码器的位置编码，则创建一个与输入shape相同的位置编码
            decoder_position_ids = jnp.broadcast_to(
                jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
            )

        # 如果需要处理任何PRNG，则创建一个对应的随机数字典
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        # 调用模型的apply方法进行前向传播
        return self.module.apply(
            {"params": params or self.params},  # 参数为给定的参数或者已有的参数
            input_ids=jnp.array(input_ids, dtype="i4"),  # 转换输入token IDs为JAX数组
            attention_mask=jnp.array(attention_mask, dtype="i4"),  # 转换attention mask为JAX数组
            position_ids=jnp.array(position_ids, dtype="i4"),  # 转换位置编码为JAX数组
            decoder_input_ids=jnp.array(decoder_input_ids, dtype="i4"),  # 转换解码器输入token IDs为JAX数组
            decoder_attention_mask=jnp.array(decoder_attention_mask, dtype="i4"),  # 转换解码器attention mask为JAX数组
            decoder_position_ids=jnp.array(decoder_position_ids, dtype="i4"),  # 转换解码器位置编码为JAX数组
            output_attentions=output_attentions,  # 是否输出attention
            output_hidden_states=output_hidden_states,  # 是否输出隐藏状态
            return_dict=return_dict,  # 是否返回字典
            deterministic=not train,  # 是否为确定性推断
            rngs=rngs,  # 随机数生成器
        )
# 添加模型的文档字符串，说明该模型是一个不带特定头部的原始隐藏状态输出的 MBart 模型变换器
# 同时继承自 FlaxBlenderbotPreTrainedModel 类
@add_start_docstrings(
    "The bare MBart Model transformer outputting raw hidden-states without any specific head on top.",
    BLENDERBOT_START_DOCSTRING,
)
class FlaxBlenderbotModel(FlaxBlenderbotPreTrainedModel):
    # Blenderbot 模型的配置信息
    config: BlenderbotConfig
    # 计算时使用的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32  
    # 模型类别为 FlaxBlenderbotModule
    module_class = FlaxBlenderbotModule


# 添加模型的调用示例文档字符串
append_call_sample_docstring(FlaxBlenderbotModel, _CHECKPOINT_FOR_DOC, FlaxSeq2SeqModelOutput, _CONFIG_FOR_DOC)


# 从 transformers.models.bart.modeling_flax_bart.FlaxBartForConditionalGenerationModule 复制并修改为 Blenderbot
class FlaxBlenderbotForConditionalGenerationModule(nn.Module):
    # Blenderbot 模型的配置信息
    config: BlenderbotConfig
    # 计算时使用的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 偏置初始化器，使用零初始化
    bias_init: Callable[..., jnp.ndarray] = jax.nn.initializers.zeros

    def setup(self):
        # 创建 Blenderbot 模型对象，并指定配置信息和数据类型
        self.model = FlaxBlenderbotModule(config=self.config, dtype=self.dtype)
        # 创建语言模型头部，设置输出维度和初始化方式
        self.lm_head = nn.Dense(
            self.model.shared.num_embeddings,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        # 创建最终 logits 的偏置参数
        self.final_logits_bias = self.param("final_logits_bias", self.bias_init, (1, self.model.shared.num_embeddings))

    # 获取编码器模块
    def _get_encoder_module(self):
        return self.model.encoder

    # 获取解码器模块
    def _get_decoder_module(self):
        return self.model.decoder

    # 模型调用方法
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
        # 使用模型进行推理，传入输入和解码器相关参数
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

        # 如果配置了共享词嵌入，则使用共享的嵌入层
        if self.config.tie_word_embeddings:
            shared_embedding = self.model.variables["params"]["shared"]["embedding"]
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
        else:
            # 否则直接使用语言模型头部
            lm_logits = self.lm_head(hidden_states)

        # 添加最终的 logits 偏置
        lm_logits += jax.lax.stop_gradient(self.final_logits_bias.astype(self.dtype))

        # 如果不需要返回字典，则返回输出元组
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return output

        # 返回 FlaxSeq2SeqLMOutput 对象，包含各种输出信息
        return FlaxSeq2SeqLMOutput(
            logits=lm_logits,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
# 定义一个带有语言建模头的Blenderbot模型，可用于摘要生成
class FlaxBlenderbotForConditionalGeneration(FlaxBlenderbotPreTrainedModel):
    # 模型类为FlaxBlenderbotForConditionalGenerationModule
    module_class = FlaxBlenderbotForConditionalGenerationModule
    # 数据类型为32位浮点数
    dtype: jnp.dtype = jnp.float32

    # 解码方法，根据输入解码生成输出
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

        past_key_values = self.init_cache(batch_size, max_length, encoder_outputs)
        # 注意通常需要在attention_mask中放入0，以处理x > input_ids.shape[-1]和x < cache_length的情况
        # 但由于解码器使用因果mask，这些位置已经被掩盖了
        # 因此，我们可以在这里创建一个静态的attention_mask，这对于编译更有效
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


FLAX_BLENDERBOT_CONDITIONAL_GENERATION_DOCSTRING = r"""
    返回：

    对话示例::

    ```py
``` 
    # 从transformers库导入AutoTokenizer和FlaxBlenderbotForConditionalGeneration类
    from transformers import AutoTokenizer, FlaxBlenderbotForConditionalGeneration
    
    # 从预训练模型"facebook/blenderbot-400M-distill"中加载Blenderbot模型
    model = FlaxBlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
    # 从预训练模型"facebook/blenderbot-400M-distill"中加载分词器
    tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
    
    # 定义输入文本
    UTTERANCE = "My friends are cool but they eat too many carbs."
    # 使用分词器对输入文本进行编码，限制最大长度为1024，返回numpy格式的张量
    inputs = tokenizer([UTTERANCE], max_length=1024, return_tensors="np")
    
    # 生成回复
    # 调用模型的generate方法生成回复的token ids，使用4个beam搜索，最大长度为5，启用early stopping
    reply_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=5, early_stopping=True).sequences
    # 解码token ids为文本，跳过特殊token，不清理tokenization的空格，并打印回复
    print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in reply_ids])
# 重写 FlaxBlenderbotForConditionalGeneration 类的调用文档字符串，将其与 BLENDERBOT_INPUTS_DOCSTRING 和 FLAX_BLENDERBOT_CONDITIONAL_GENERATION_DOCSTRING 相结合
overwrite_call_docstring(
    FlaxBlenderbotForConditionalGeneration,
    BLENDERBOT_INPUTS_DOCSTRING + FLAX_BLENDERBOT_CONDITIONAL_GENERATION_DOCSTRING,
)
# 向 FlaxBlenderbotForConditionalGeneration 类追加、替换返回文档字符串，指定输出类型为 FlaxSeq2SeqLMOutput，并使用 _CONFIG_FOR_DOC 配置类
append_replace_return_docstrings(
    FlaxBlenderbotForConditionalGeneration, output_type=FlaxSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC
)
```  
```