# `.\transformers\models\bart\modeling_flax_bart.py`

```
# 设置编码格式为 UTF-8

# 引入必要的库
# 引入标准库中的 math 模块
import math
# 引入标准库中的 random 模块
import random
# 从 functools 库中引入 partial 函数
from functools import partial
# 从 typing 库中引入 Callable 和 Optional 类型
from typing import Callable, Optional, Tuple

# 引入 Flax 库中的模块
# 从 Flax 库中引入 linen 子模块，并用 nn 别名指代
import flax.linen as nn
# 引入 JAX 库
import jax
# 引入 JAX 库中的 numpy 模块，并用 jnp 别名指代
import jax.numpy as jnp
# 从 Flax 核心模块中引入 FrozenDict、freeze 和 unfreeze 函数
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
# 从 Flax 库的 linen 模块中引入 combine_masks 和 make_causal_mask 函数
from flax.linen import combine_masks, make_causal_mask
# 从 Flax 库的 linen 模块中引入 dot_product_attention_weights 函数
from flax.linen.attention import dot_product_attention_weights
# 从 Flax 库的 traverse_util 模块中引入 flatten_dict 和 unflatten_dict 函数
from flax.traverse_util import flatten_dict, unflatten_dict
# 从 JAX 库中的 lax 模块中引入 lax 模块
from jax import lax
# 从 JAX 库中的 random 模块中引入 PRNGKey 类
from jax.random import PRNGKey

# 引入模型输出相关的模块
# 从 transformers 库中的 modeling_flax_outputs 模块中引入各种模型输出类
from ...modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxBaseModelOutputWithPastAndCrossAttentions,
    FlaxCausalLMOutputWithCrossAttentions,
    FlaxSeq2SeqLMOutput,
    FlaxSeq2SeqModelOutput,
    FlaxSeq2SeqQuestionAnsweringModelOutput,
    FlaxSeq2SeqSequenceClassifierOutput,
)
# 引入模型工具相关的模块
# 从 transformers 库中的 modeling_flax_utils 模块中引入各种模型工具函数和类
from ...modeling_flax_utils import (
    ACT2FN,
    FlaxPreTrainedModel,
    append_call_sample_docstring,
    append_replace_return_docstring,
    overwrite_call_docstring,
)
# 引入通用工具相关的模块
# 从 transformers 库中的 utils 模块中引入各种通用工具函数
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
# 引入 Bart 配置类
# 从当前目录下的 configuration_bart 模块中引入 BartConfig 类
from .configuration_bart import BartConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置信息
_CHECKPOINT_FOR_DOC = "facebook/bart-base"
_CONFIG_FOR_DOC = "BartConfig"

# BART 模型的起始文档字符串
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
    # 定义函数参数
    Parameters:
        # 配置类参数，包含模型的所有参数
        config ([`BartConfig`]): Model configuration class with all the parameters of the model.
            # 使用配置文件初始化不加载模型权重，只加载配置信息
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.
        # 数据类型参数，默认为 `jax.numpy.float32`
        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
            # 计算时的数据类型，可选值有 `jax.numpy.float32`, `jax.numpy.float16` (在 GPU 上), `jax.numpy.bfloat16` (在 TPU 上)
            The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and
            `jax.numpy.bfloat16` (on TPUs).
            
            # 可用于启用在 GPU 或 TPU 上的混合精度训练或半精度推断。如果指定，所有计算将使用给定的 `dtype` 进行。
            This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
            specified all the computation will be performed with the given `dtype`.
            
            # 注意，这仅指定计算的 dtype，并不影响模型参数的 dtype。
            **Note that this only specifies the dtype of the computation and does not influence the dtype of model
            parameters.**
            
            # 如果想要改变模型参数的 dtype，请参阅 [`~FlaxPreTrainedModel.to_fp16`] 和 [`~FlaxPreTrainedModel.to_bf16`]。
            If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`] and
            [`~FlaxPreTrainedModel.to_bf16`].
"""
BART_INPUTS_DOCSTRING = r"""
"""


BART_ENCODE_INPUTS_DOCSTRING = r"""
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

BART_DECODE_INPUTS_DOCSTRING = r"""
"""


def shift_tokens_right(input_ids: jnp.ndarray, pad_token_id: int, decoder_start_token_id: int) -> jnp.ndarray:
    """
    Shift input ids one token to the right.
    """
    # 创建一个与 input_ids 形状相同的全零数组
    shifted_input_ids = jnp.zeros_like(input_ids)
    # 将 input_ids 中每个序列的第一个 token 放到 shifted_input_ids 的对应位置
    shifted_input_ids = shifted_input_ids.at[:, 1:].set(input_ids[:, :-1])
    # 将 decoder_start_token_id 放到每个序列的第一个位置
    shifted_input_ids = shifted_input_ids.at[:, 0].set(decoder_start_token_id)

    # 将值为 -100 的位置替换为 pad_token_id
    shifted_input_ids = jnp.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)
    return shifted_input_ids


class FlaxBartAttention(nn.Module):
    config: BartConfig
    embed_dim: int
    num_heads: int
    dropout: float = 0.0
    causal: bool = False
    bias: bool = True
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    # 设置函数，用于初始化模型参数
    def setup(self) -> None:
        # 计算每个头的维度
        self.head_dim = self.embed_dim // self.num_heads
        # 检查 embed_dim 是否能被 num_heads 整除
        if self.head_dim * self.num_heads != self.embed_dim:
            # 若不能整除，抛出数值错误异常
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )

        # 创建局部函数 dense，设置全连接层的参数
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

        # 若模型需要自回归（causal），创建自回归掩码
        if self.causal:
            self.causal_mask = make_causal_mask(
                jnp.ones((1, self.config.max_position_embeddings), dtype="bool"), dtype="bool"
            )

    # 将隐藏状态拆分成多个头
    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    # 将多个头的隐藏状态合并
    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    # 使用 JAX 的 @nn.compact 装饰器，声明了一个紧凑模块
    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slighly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # 检测是否通过缺少现有缓存数据来初始化。
        is_initialized = self.has_variable("cache", "cached_key")
        # 获取或创建缓存的键值对状态
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        # 获取或创建缓存索引，指示当前缓存的位置
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            # 获取批处理维度以及键值对状态的形状信息
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # 使用新的一维空间切片更新键、值缓存
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            # 更新缓存的键、值状态
            cached_key.value = key
            cached_value.value = value
            # 更新缓存索引，指示新缓存的位置
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # 用于缓存的因果掩码：我们的单个查询位置只应与已生成和缓存的键位置关联，而不是剩余的零元素。
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            # 合并填充掩码和注意力掩码
            attention_mask = combine_masks(pad_mask, attention_mask)
        return key, value, attention_mask

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        key_value_states: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        deterministic: bool = True,
class FlaxBartEncoderLayer(nn.Module):
    # 定义一个 FlaxBartEncoderLayer 类，继承自 nn.Module
    config: BartConfig
    # 定义 config 属性为 BartConfig 类型
    dtype: jnp.dtype = jnp.float32
    # 定义 dtype 属性为 jnp.float32 类型，默认值为 jnp.float32

    def setup(self) -> None:
        # 定义 setup 方法，无返回值
        self.embed_dim = self.config.d_model
        # 设置 embed_dim 属性为 config 的 d_model 属性值
        self.self_attn = FlaxBartAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.encoder_attention_heads,
            dropout=self.config.attention_dropout,
            dtype=self.dtype,
        )
        # 初始化 self_attn 属性为 FlaxBartAttention 类的实例
        self.self_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        # 初始化 self_attn_layer_norm 属性为 nn.LayerNorm 类的实例
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)
        # 初始化 dropout_layer 属性为 nn.Dropout 类的实例
        self.activation_fn = ACT2FN[self.config.activation_function]
        # 设置 activation_fn 属性为 ACT2FN 字典中 config.activation_function 对应的值
        self.activation_dropout_layer = nn.Dropout(rate=self.config.activation_dropout)
        # 初始化 activation_dropout_layer 属性为 nn.Dropout 类的实例
        self.fc1 = nn.Dense(
            self.config.encoder_ffn_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        # 初始化 fc1 属性为 nn.Dense 类的实例
        self.fc2 = nn.Dense(
            self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std)
        )
        # 初始化 fc2 属性为 nn.Dense 类的实例
        self.final_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        # 初始化 final_layer_norm 属性为 nn.LayerNorm 类的实例

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        output_attentions: bool = True,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray]:
        # 定义 __call__ 方法，返回类型为 Tuple[jnp.ndarray]
        residual = hidden_states
        # 将 hidden_states 赋值给 residual
        hidden_states, attn_weights = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask)
        # 调用 self_attn 方法，传入参数 hidden_states 和 attention_mask，将返回值分别赋给 hidden_states 和 attn_weights

        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 使用 dropout_layer 对 hidden_states 进行处理
        hidden_states = residual + hidden_states
        # 将 residual 与 hidden_states 相加
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # 使用 self_attn_layer_norm 对 hidden_states 进行处理

        residual = hidden_states
        # 将 hidden_states 赋值给 residual
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 使用 activation_fn 对 fc1 处理后的 hidden_states 进行处理
        hidden_states = self.activation_dropout_layer(hidden_states, deterministic=deterministic)
        # 使用 activation_dropout_layer 对 hidden_states 进行处理
        hidden_states = self.fc2(hidden_states)
        # 使用 fc2 对 hidden_states 进行处理
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 使用 dropout_layer 对 hidden_states 进行处理
        hidden_states = residual + hidden_states
        # 将 residual 与 hidden_states 相加
        hidden_states = self.final_layer_norm(hidden_states)
        # 使用 final_layer_norm 对 hidden_states 进行处理

        outputs = (hidden_states,)
        # 将 hidden_states 存入 outputs 元组中

        if output_attentions:
            outputs += (attn_weights,)
            # 如果 output_attentions 为真，则将 attn_weights 存入 outputs 元组中

        return outputs
        # 返回 outputs

class FlaxBartEncoderLayerCollection(nn.Module):
    # 定义一个 FlaxBartEncoderLayerCollection 类，继承自 nn.Module
    config: BartConfig
    # 定义 config 属性为 BartConfig 类型
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    # 定义 dtype 属性为 jnp.float32 类型，默认值为 jnp.float32

    def setup(self):
        # 定义 setup 方法
        self.layers = [
            FlaxBartEncoderLayer(self.config, name=str(i), dtype=self.dtype) for i in range(self.config.encoder_layers)
        ]
        # 初始化 layers 属性为包含多个 FlaxBartEncoderLayer 实例的列表
        self.layerdrop = self.config.encoder_layerdrop
        # 设置 layerdrop 属性为 config 的 encoder_layerdrop 属性值

    def __call__(
        self,
        hidden_states,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        # 定义 __call__ 方法的参数
        ):
        # 如果不输出注意力权重，则将 all_attentions 设置为 None
        all_attentions = () if output_attentions else None
        # 如果不输出隐藏状态，则将 all_hidden_states 设置为 None
        all_hidden_states = () if output_hidden_states else None

        # 遍历每个编码器层
        for encoder_layer in self.layers:
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            # 添加 LayerDrop（参考 https://arxiv.org/abs/1909.11556 进行描述）
            # 生成一个随机的丢弃概率
            dropout_probability = random.uniform(0, 1)
            # 如果不是确定性的且随机概率小于层丢弃率，则跳过该层
            if not deterministic and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                # 否则，调用编码器层的前向传播函数
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions,
                    deterministic,
                )
            # 更新隐藏状态为当前层的输出
            hidden_states = layer_outputs[0]
            # 如果需要输出注意力权重，则将当前层的注意力权重添加到 all_attentions 中
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # 如果需要输出隐藏状态，则将最终隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 将隐藏状态、所有隐藏状态和所有注意力权重组成输出
        outputs = (hidden_states, all_hidden_states, all_attentions)

        # 如果不返回字典，则返回非空元素的元组
        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        # 返回 FlaxBaseModelOutput 对象，包含最终隐藏状态、所有隐藏状态和所有注意力权重
        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )
class FlaxBartDecoderLayer(nn.Module):
    # 定义一个FlaxBartDecoderLayer类，继承自nn.Module
    config: BartConfig
    # 类变量config，类型为BartConfig

    dtype: jnp.dtype = jnp.float32
    # 类变量dtype，默认为jnp.float32数据类型

    def setup(self) -> None:
        # 定义setup方法，无返回值
        self.embed_dim = self.config.d_model
        # 设置embed_dim为config的d_model属性值

        self.self_attn = FlaxBartAttention(
            # 初始化self_attn，使用FlaxBartAttention类
            config=self.config,
            # 传入config参数
            embed_dim=self.embed_dim,
            # 传入embed_dim参数
            num_heads=self.config.decoder_attention_heads,
            # 传入decoder_attention_heads参数
            dropout=self.config.attention_dropout,
            # 传入attention_dropout参数
            causal=True,
            # 设定causal为True，表示自注意力机制为有向的
            dtype=self.dtype,
            # 传入dtype参数
        )

        self.dropout_layer = nn.Dropout(rate=self.config.dropout)
        # 初始化dropout_layer，使用nn.Dropout类，传入dropout率参数
        self.activation_fn = ACT2FN[self.config.activation_function]
        # 设置activation_fn为ACT2FN字典的config.activation_function对应的值
        self.activation_dropout_layer = nn.Dropout(rate=self.config.activation_dropout)
        # 初始化activation_dropout_layer，使用nn.Dropout类，传入activation_dropout率参数

        self.self_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        # 初始化self_attn_layer_norm，使用nn.LayerNorm类，传入dtype和epsilon参数
        self.encoder_attn = FlaxBartAttention(
            # 初始化encoder_attn，使用FlaxBartAttention类
            config=self.config,
            # 传入config参数
            embed_dim=self.embed_dim,
            # 传入embed_dim参数
            num_heads=self.config.decoder_attention_heads,
            # 传入decoder_attention_heads参数
            dropout=self.config.attention_dropout,
            # 传入attention_dropout参数
            dtype=self.dtype,
            # 传入dtype参数
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        # 初始化encoder_attn_layer_norm，使用nn.LayerNorm类，传入dtype和epsilon参数
        self.fc1 = nn.Dense(
            # 初始化fc1，使用nn.Dense类
            self.config.decoder_ffn_dim,
            # 传入decoder_ffn_dim参数
            dtype=self.dtype,
            # 传入dtype参数
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
            # 传入kernel_init参数，使用正态分布初始化权重，标准差为config.init_std
        )
        self.fc2 = nn.Dense(
            # 初始化fc2，使用nn.Dense类
            self.embed_dim, dtype=self.dtype,
            # 传入embed_dim和dtype参数
            kernel_init=jax.nn.initializers.normal(self.config.init_std)
            # 传入kernel_init参数，使用正态分布初始化权重，标准差为config.init_std
        )
        self.final_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        # 初始化final_layer_norm，使用nn.LayerNorm类，传入dtype和epsilon参数

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        # 输入参数hidden_states，数据类型为jnp.ndarray
        attention_mask: jnp.ndarray,
        # 输入参数attention_mask，数据类型为jnp.ndarray
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        # 输入参数encoder_hidden_states，数据类型为Optional[jnp.ndarray]，可选项，默认为None
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        # 输入参数encoder_attention_mask，数据类型为Optional[jnp.ndarray]，可选项，默认为None
        init_cache: bool = False,
        # 输入参数init_cache，数据类型为bool，默认为False
        output_attentions: bool = True,
        # 输入参数output_attentions，数据类型为bool，默认为True
        deterministic: bool = True,
        # 输入参数deterministic，数据类型为bool，默认为True
    # 定义函数，返回值为元组类型，包含一个 JAX 数组
    ) -> Tuple[jnp.ndarray]:
        # 保存残差连接
        residual = hidden_states

        # 自注意力机制
        # 调用 self_attn 方法进行自注意力计算，并返回计算结果以及自注意力权重
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, init_cache=init_cache
        )
        # 使用 dropout 对 hidden_states 进行处理
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 将残差连接加回来
        hidden_states = residual + hidden_states
        # 使用 LayerNorm 对 hidden_states 进行归一化
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 交叉注意力块
        cross_attn_weights = None
        # 如果有编码器的隐藏状态，则执行以下操作
        if encoder_hidden_states is not None:
            # 保存残差连接
            residual = hidden_states

            # 使用编码器注意力机制计算隐藏状态
            hidden_states, cross_attn_weights = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            # 使用 dropout 对 hidden_states 进行处理
            hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
            # 将残差连接加回来
            hidden_states = residual + hidden_states
            # 使用 LayerNorm 对 hidden_states 进行归一化
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # 全连接层
        # 保存残差连接
        residual = hidden_states
        # 使用激活函数对 hidden_states 进行处理
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 使用 dropout 对 hidden_states 进行处理
        hidden_states = self.activation_dropout_layer(hidden_states, deterministic=deterministic)
        # 使用全连接层 fc2 进行处理
        hidden_states = self.fc2(hidden_states)
        # 使用 dropout 对 hidden_states 进行处理
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 将残差连接加回来
        hidden_states = residual + hidden_states
        # 使用 LayerNorm 对 hidden_states 进行归一化
        hidden_states = self.final_layer_norm(hidden_states)

        # 返回隐藏状态的输出
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则将自注意力权重和交叉注意力权重添加到输出中
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs
class FlaxBartDecoderLayerCollection(nn.Module):
    # Bart 模型解码器层的集合
    config: BartConfig
    # 计算的数据类型
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        # 创建解码器层列表
        self.layers = [
            FlaxBartDecoderLayer(self.config, name=str(i), dtype=self.dtype) for i in range(self.config.decoder_layers)
        ]
        # 解码器层的丢弃率
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
        # 如果需要输出交叉注意力权重，并且存在编码器隐藏状态，则初始化空元组
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        # 循环遍历每个解码器层
        for decoder_layer in self.layers:
            # 如果需要输出隐藏状态，则添加当前隐藏状态
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                # 添加层丢弃（参考 https://arxiv.org/abs/1909.11556）
            dropout_probability = random.uniform(0, 1)
            if not deterministic and (dropout_probability < self.layerdrop):
                layer_outputs = (None, None, None)  # 若丢弃，则输出为 None
            else:
                # 否则进行解码器层的前向传播
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    init_cache=init_cache,
                    output_attentions=output_attentions,
                    deterministic=deterministic,
                )

            # 更新隐藏状态为当前层的输出
            hidden_states = layer_outputs[0]
            # 如果需要输出注意力权重，则添加当前层的自注意力权重
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                # 如果存在编码器隐藏状态，则添加当前层的交叉注意力权重
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # 如果需要输出隐藏状态，则添加最后一个解码器层的隐藏状态
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 将所有输出整合成一个列表
        outputs = [hidden_states, all_hidden_states, all_self_attns, all_cross_attentions]

        # 如果不以字典形式返回结果，则返回一个元组
        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        # 以字典形式返回结果
        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class FlaxBartClassificationHead(nn.Module):
    """句子级别分类任务的头部。"""

    config: BartConfig
    # 内部维度
    inner_dim: int
    # 类别数
    num_classes: int
    # 池化层丢弃率
    pooler_dropout: float
    # 数据类型
    dtype: jnp.dtype = jnp.float32
```  
    # 初始化模型参数
    def setup(self):
        # 创建一个全连接层，输入维度为 self.inner_dim，输出维度为 self.num_classes
        # 使用正态分布初始化权重矩阵，标准差为 self.config.init_std
        self.dense = nn.Dense(
            self.inner_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std)
        )
        # 创建一个用于随机失活的层，丢弃率为 self.pooler_dropout
        self.dropout = nn.Dropout(rate=self.pooler_dropout)
        # 创建一个全连接层，输入维度为 self.inner_dim，输出维度为 self.num_classes
        # 使用正态分布初始化权重矩阵，标准差为 self.config.init_std
        self.out_proj = nn.Dense(
            self.num_classes,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )

    # 模型的调用方法
    def __call__(self, hidden_states: jnp.ndarray, deterministic: bool):
        # 对输入的 hidden_states 应用随机失活，根据 deterministic 参数决定是否确定性操作
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 将隐层状态输入到全连接层中
        hidden_states = self.dense(hidden_states)
        # 对全连接层的输出应用 tanh 激活函数
        hidden_states = jnp.tanh(hidden_states)
        # 再次对隐层状态应用随机失活
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 将经过全连接层和激活函数后的结果输入到输出投影层中
        hidden_states = self.out_proj(hidden_states)
        # 返回模型输出的结果
        return hidden_states
# 定义一个FlaxBartEncoder类，继承自nn.Module
class FlaxBartEncoder(nn.Module):
    # BartConfig类型的config属性
    config: BartConfig
    # nn.Embed类型的embed_tokens属性
    embed_tokens: nn.Embed
    # jnp.float32类型的dtype属性，用于计算的数据类型

    # 初始化方法
    def setup(self):
        # 创建一个nn.Dropout对象，设置dropout率为config中的值
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)

        # 获取embed_dim为config中的d_model
        embed_dim = self.config.d_model
        # 获取padding_idx为config中的pad_token_id
        self.padding_idx = self.config.pad_token_id
        # 获取max_source_positions为config中的max_position_embeddings
        self.max_source_positions = self.config.max_position_embeddings
        # 如果config中的scale_embedding为True，则设置embed_scale为embed_dim的平方根，否则为1.0
        self.embed_scale = math.sqrt(embed_dim) if self.config.scale_embedding else 1.0

        # Bart设置了一个偏移量为2，用于处理padding_idx的情况
        self.offset = 2
        # 创建一个nn.Embed对象，用于处理位置编码
        self.embed_positions = nn.Embed(
            self.config.max_position_embeddings + self.offset,
            embed_dim,
            embedding_init=jax.nn.initializers.normal(self.config.init_std),
            dtype=self.dtype,
        )
        # 创建一个FlaxBartEncoderLayerCollection对象
        self.layers = FlaxBartEncoderLayerCollection(self.config, self.dtype)
        # 创建一个nn.LayerNorm对象，用于处理embedding的LayerNorm
        self.layernorm_embedding = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    # 调用方法
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
        # 获取input_ids的形状
        input_shape = input_ids.shape
        # 将input_ids重塑为二维数组
        input_ids = input_ids.reshape(-1, input_shape[-1])

        # 使用embed_tokens对input_ids进行embedding，并乘以embed_scale
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # 获取位置编码embed_pos
        embed_pos = self.embed_positions(position_ids + self.offset)

        # 将embedding和位置编码相加，并进行LayerNorm处理
        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        # 使用dropout_layer对hidden_states进行dropout处理
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)

        # 调用layers处理hidden_states
        outputs = self.layers(
            hidden_states,
            attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果return_dict为False，则返回outputs
        if not return_dict:
            return outputs

        # 返回FlaxBaseModelOutput对象，包括last_hidden_state、hidden_states和attentions
        return FlaxBaseModelOutput(
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# 定义一个FlaxBartDecoder类，继承自nn.Module
class FlaxBartDecoder(nn.Module):
    # BartConfig类型的config属性
    config: BartConfig
    # nn.Embed类型的embed_tokens属性
    embed_tokens: nn.Embed
    # jnp.float32类型的dtype属性，用于计算的数据类型
    # 初始化方法，设置模型的一些属性和参数
    def setup(self):
        # 初始化一个 dropout 层
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)

        # 获取模型的 embedding 维度
        embed_dim = self.config.d_model
        # 获取填充标记的索引
        self.padding_idx = self.config.pad_token_id
        # 获取最大目标位置
        self.max_target_positions = self.config.max_position_embeddings
        # 设置嵌入缩放因子
        self.embed_scale = math.sqrt(self.config.d_model) if self.config.scale_embedding else 1.0

        # 如果 padding_idx 被指定，则将嵌入 id 偏移 2，并相应调整 num_embeddings。其他模型没有这个 hack
        self.offset = 2
        # 初始化位置嵌入层
        self.embed_positions = nn.Embed(
            self.config.max_position_embeddings + self.offset,
            embed_dim,
            embedding_init=jax.nn.initializers.normal(self.config.init_std),
            dtype=self.dtype,
        )

        # 初始化解码器层集合
        self.layers = FlaxBartDecoderLayerCollection(self.config, self.dtype)
        # 初始化嵌入层的 LayerNorm
        self.layernorm_embedding = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    # 模型调用方法，处理输入并返回输出
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
        # 获取输入的形状
        input_shape = input_ids.shape
        # 重塑输入的形状
        input_ids = input_ids.reshape(-1, input_shape[-1])

        # 获取输入的嵌入表示并乘以嵌入缩放因子
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # 嵌入位置信息
        positions = self.embed_positions(position_ids + self.offset)

        # 将输入嵌入和位置嵌入相加
        hidden_states = inputs_embeds + positions
        # 对结果进行 LayerNorm
        hidden_states = self.layernorm_embedding(hidden_states)

        # 对结果进行 dropout
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)

        # 调用解码器层处理结果
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

        # 如果不返回字典，则直接返回输出
        if not return_dict:
            return outputs

        # 返回带有过去和交叉注意力的 FlaxBaseModelOutputWithPastAndCrossAttentions 对象
        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )
# 定义一个 FlaxBartModule 类，继承自 nn.Module 类
class FlaxBartModule(nn.Module):
    # BartConfig 类型的配置
    config: BartConfig
    # 计算的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    # 初始化方法
    def setup(self):
        # 创建一个共享的嵌入层，用于编码器和解码器
        self.shared = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.init_std),
            dtype=self.dtype,
        )

        # 创建编码器对象
        self.encoder = FlaxBartEncoder(self.config, dtype=self.dtype, embed_tokens=self.shared)
        # 创建解码器对象
        self.decoder = FlaxBartDecoder(self.config, dtype=self.dtype, embed_tokens=self.shared)

    # 获取编码器模块的方法
    def _get_encoder_module(self):
        return self.encoder

    # 获取解码器模块的方法
    def _get_decoder_module(self):
        return self.decoder

    # 定义对象调用时的行为
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
        # 调用编码器得到输出
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        # 调用解码器得到输出
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

        # 如果不返回字典，则将解码器输出和编码器输出连接并返回
        if not return_dict:
            return decoder_outputs + encoder_outputs

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


# 定义一个 FlaxBartPreTrainedModel 类，继承自 FlaxPreTrainedModel 类
class FlaxBartPreTrainedModel(FlaxPreTrainedModel):
    # 配置类为 BartConfig
    config_class = BartConfig
    # 基础模型前缀为 "model"
    base_model_prefix: str = "model"
    # 模块类为 nn.Module
    module_class: nn.Module = None

    # 初始化方法
    def __init__(
        self,
        config: BartConfig,
        input_shape: Tuple[int] = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        # 使用给定的配置和数据类型初始化模块对象
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 调用父类的初始化方法
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)
    # 初始化模型的权重
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量，全零张量
        input_ids = jnp.zeros(input_shape, dtype="i4")
        # 确保初始化过程适用于FlaxBartForSequenceClassificationModule
        # 将最后一个位置的值设置为结束标记的ID
        input_ids = input_ids.at[(..., -1)].set(self.config.eos_token_id)
        # 创建与输入张量相同形状的注意力遮罩，全一张量
        attention_mask = jnp.ones_like(input_ids)
        # 初始化解码器输入张量为输入张量
        decoder_input_ids = input_ids
        # 创建与输入张量相同形状的解码器注意力遮罩，全一张量
        decoder_attention_mask = jnp.ones_like(input_ids)

        # 获取输入张量的批量大小和序列长度
        batch_size, sequence_length = input_ids.shape
        # 生成位置ID张量，形状与输入张量相同，内容为序列长度的广播
        position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))
        # 生成解码器位置ID张量，形状与输入张量相同，内容为序列长度的广播
        decoder_position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # 使用随机数生成器分割随机数种子
        params_rng, dropout_rng = jax.random.split(rng)
        # 构建随机数种子字典
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 使用模型的初始化方法初始化参数
        random_params = self.module.init(
            rngs,
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            position_ids,
            decoder_position_ids,
        )["params"]

        # 如果提供了预定义参数，则将随机生成的参数与之对齐
        if params is not None:
            # 将随机参数展平
            random_params = flatten_dict(unfreeze(random_params))
            # 将提供的参数展平
            params = flatten_dict(unfreeze(params))
            # 处理缺失的参数键
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            # 清空缺失键集合
            self._missing_keys = set()
            # 返回对齐后的参数
            return freeze(unflatten_dict(params))
        else:
            # 返回随机初始化的参数
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
        # 初始化用于检索缓存的输入变量
        # 生成包含全1的形状为(batch_size, max_length)的解码器输入标识数组
        decoder_input_ids = jnp.ones((batch_size, max_length), dtype="i4")
        # 生成与decoder_input_ids形状相同的全1的注意力掩码数组
        decoder_attention_mask = jnp.ones_like(decoder_input_ids)
        # 生成与decoder_input_ids形状相同的位置标识数组，值为0到decoder_input_ids长度-1
        decoder_position_ids = jnp.broadcast_to(
            jnp.arange(jnp.atleast_2d(decoder_input_ids).shape[-1]), decoder_input_ids.shape
        )

        # 定义内部函数用于调用解码器以初始化缓存
        def _decoder_forward(module, decoder_input_ids, decoder_attention_mask, decoder_position_ids, **kwargs):
            decoder_module = module._get_decoder_module()
            return decoder_module(
                decoder_input_ids,
                decoder_attention_mask,
                decoder_position_ids,
                **kwargs,
            )

        # 使用给定的输入变量初始化模型参数
        init_variables = self.module.init(
            jax.random.PRNGKey(0),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            encoder_hidden_states=encoder_outputs[0],
            init_cache=True,
            method=_decoder_forward,  # 仅需要调用解码器以初始化缓存
        )
        # 解除初始化后的变量的冻结状态，并返回缓存
        return unfreeze(init_variables["cache"])

    @add_start_docstrings(BART_ENCODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutput, config_class=BartConfig)
    # 重写BART编码器的encode方法的文档字符串，并指定返回值和配置类
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
        >>> from transformers import AutoTokenizer, FlaxBartForConditionalGeneration

        >>> model = FlaxBartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

        >>> text = "My friends are cool but they eat too many carbs."
        >>> inputs = tokenizer(text, max_length=1024, return_tensors="jax")
        >>> encoder_outputs = model.encode(**inputs)
        ```"""
        # 如果输出注意力信息不为None，则使用参数中的值，否则使用模型配置中的值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果输出隐藏状态不为None，则使用参数中的值，否则使用模型配置中的值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果返回字典不为None，则使用参数中的值，否则使用模型配置中的值
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 如果没有提供注意力遮罩，则创建一个与输入ID相同形状的全1注意力遮罩
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        # 如果没有提供位置ID，则创建一个与输入ID相同形状的位置ID
        if position_ids is None:
            batch_size, sequence_length = input_ids.shape
            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # 处理任何需要的伪随机数生成器
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        # 定义编码器前向传播函数
        def _encoder_forward(module, input_ids, attention_mask, position_ids, **kwargs):
            encode_module = module._get_encoder_module()
            return encode_module(input_ids, attention_mask, position_ids, **kwargs)

        # 应用模型
        return self.module.apply(
            {"params": params or self.params},  # 使用参数或者模型的参数
            input_ids=jnp.array(input_ids, dtype="i4"),  # 转换输入ID为JAX数组
            attention_mask=jnp.array(attention_mask, dtype="i4"),  # 转换注意力遮罩为JAX数组
            position_ids=jnp.array(position_ids, dtype="i4"),  # 转换位置ID为JAX数组
            output_attentions=output_attentions,  # 输出注意力信息
            output_hidden_states=output_hidden_states,  # 输出隐藏状态
            return_dict=return_dict,  # 返回字典
            deterministic=not train,  # 是否确定性运行
            rngs=rngs,  # 伪随机数生成器
            method=_encoder_forward,  # 编码器前向传播方法
        )

    # 将BART的输入文档字符串添加到模型前向传播函数
    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
```  
    # 定义一个调用函数，接受多个参数
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
        # 如果未指定输出注意力，使用配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未指定输出隐藏状态，使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定返回字典，使用配置中的默认值
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

        # 处理任何 PRNG（伪随机数生成器）如果需要
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        # 调用模块的应用方法，传递参数和输入数据
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
# 导入必要的库
@add_start_docstrings(
    "The bare Bart Model transformer outputting raw hidden-states without any specific head on top.",  # 添加文档字符串
    BART_START_DOCSTRING,  # 引用 BART 模型的起始文档字符串
)
# 定义 FlaxBartModel 类，继承自 FlaxBartPreTrainedModel 类
class FlaxBartModel(FlaxBartPreTrainedModel):
    config: BartConfig  # 定义 config 属性，类型为 BartConfig
    dtype: jnp.dtype = jnp.float32  # 定义 dtype 属性，数据类型为 jnp.float32
    module_class = FlaxBartModule  # 指定 module_class 为 FlaxBartModule


# 添加调用样例的文档字符串
append_call_sample_docstring(FlaxBartModel, _CHECKPOINT_FOR_DOC, FlaxSeq2SeqModelOutput, _CONFIG_FOR_DOC)


# 定义 FlaxBartForConditionalGenerationModule 类，继承自 nn.Module
class FlaxBartForConditionalGenerationModule(nn.Module):
    config: BartConfig  # 定义 config 属性，类型为 BartConfig
    dtype: jnp.dtype = jnp.float32  # 定义 dtype 属性，数据类型为 jnp.float32
    bias_init: Callable[..., jnp.ndarray] = jax.nn.initializers.zeros  # 初始化 bias_init 属性为零矩阵生成函数

    # 设置模块
    def setup(self):
        # 初始化模型为 FlaxBartModule 类的实例
        self.model = FlaxBartModule(config=self.config, dtype=self.dtype)
        # 初始化 lm_head 属性为全连接层，参数包括词汇表大小、不使用偏置、数据类型、权重初始化方式
        self.lm_head = nn.Dense(
            self.model.shared.num_embeddings,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        # 初始化 final_logits_bias 属性为偏置项参数，形状为 (1, 词汇表大小)
        self.final_logits_bias = self.param("final_logits_bias", self.bias_init, (1, self.model.shared.num_embeddings))

    # 获取编码器模块
    def _get_encoder_module(self):
        return self.model.encoder

    # 获取解码器模块
    def _get_decoder_module(self):
        return self.model.decoder

    # 定义 __call__ 方法，用于模型调用
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
        # 使用模型进行前向传播，获取输出
        outputs = self.model(
            input_ids=input_ids,  # 输入的 token IDs
            attention_mask=attention_mask,  # 注意力遮罩
            decoder_input_ids=decoder_input_ids,  # 解码器的输入 token IDs
            decoder_attention_mask=decoder_attention_mask,  # 解码器的注意力遮罩
            position_ids=position_ids,  # 位置 IDs
            decoder_position_ids=decoder_position_ids,  # 解码器位置 IDs
            output_attentions=output_attentions,  # 是否输出注意力权重
            output_hidden_states=output_hidden_states,  # 是否输出隐藏状态
            return_dict=return_dict,  # 是否返回字典格式的输出
            deterministic=deterministic,  # 是否使用确定性计算
        )

        # 获取模型输出中的隐藏状态
        hidden_states = outputs[0]

        # 如果配置中指定了词嵌入共享
        if self.config.tie_word_embeddings:
            # 获取共享的嵌入矩阵
            shared_embedding = self.model.variables["params"]["shared"]["embedding"]
            # 使用共享的嵌入矩阵计算语言模型的输出 logits
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
        else:
            # 使用语言模型头计算输出 logits
            lm_logits = self.lm_head(hidden_states)

        # 添加最终 logits 的偏置项
        lm_logits += jax.lax.stop_gradient(self.final_logits_bias.astype(self.dtype))

        # 如果不需要返回字典格式的输出
        if not return_dict:
            # 构建输出元组
            output = (lm_logits,) + outputs[1:]
            # 返回输出元组
            return output

        # 返回字典格式的输出
        return FlaxSeq2SeqLMOutput(
            logits=lm_logits,  # 语言模型输出 logits
            decoder_hidden_states=outputs.decoder_hidden_states,  # 解码器隐藏状态
            decoder_attentions=outputs.decoder_attentions,  # 解码器注意力权重
            cross_attentions=outputs.cross_attentions,  # 交叉注意力权重
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,  # 编码器最后隐藏状态
            encoder_hidden_states=outputs.encoder_hidden_states,  # 编码器隐藏状态
            encoder_attentions=outputs.encoder_attentions,  # 编码器注意力权重
        )
# 添加起始文档字符串，说明该类是带有语言建模头的 BART 模型，可用于摘要生成
@add_start_docstrings(
    "The BART Model with a language modeling head. Can be used for summarization.", BART_START_DOCSTRING
)
class FlaxBartForConditionalGeneration(FlaxBartPreTrainedModel):
    # 模块类为 FlaxBartForConditionalGenerationModule
    module_class = FlaxBartForConditionalGenerationModule
    # 数据类型为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 添加解码输入文档字符串
    @add_start_docstrings(BART_DECODE_INPUTS_DOCSTRING)
    # 替换返回文档字符串
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
        # 准备生成的输入
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
            # 注意：通常需要在注意力掩码中放入 0，以使 x > input_ids.shape[-1] 和 x < cache_length 的位置。但由于解码器使用因果掩码，这些位置已经被掩盖。
            # 因此，我们可以在这里创建一个静态的注意力掩码，这对于编译更加高效。
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

FLAX_BART_CONDITIONAL_GENERATION_DOCSTRING = """
    Returns:

    Summarization example:

    ```python
    >>> from transformers import AutoTokenizer, FlaxBartForConditionalGeneration
    # 从预训练模型中加载 FlaxBartForConditionalGeneration 模型
    model = FlaxBartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    # 从预训练模型中加载 AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

    # 待总结的文章内容
    ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
    # 使用 tokenizer 对文章进行编码，限制最大长度为1024，返回 NumPy 数组
    inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="np")

    # 生成摘要
    summary_ids = model.generate(inputs["input_ids"]).sequences
    # 解码生成的摘要，跳过特殊标记并保留分词空格
    print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))


Mask filling example:


    # 导入必要的库
    import jax
    from transformers import AutoTokenizer, FlaxBartForConditionalGeneration

    # 从预训练模型中加载 FlaxBartForConditionalGeneration 模型
    model = FlaxBartForConditionalGeneration.from_pretrained("facebook/bart-large")
    # 从预训练模型中加载 AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

    # 待填充掩码的文本
    TXT = "My friends are <mask> but they eat too many carbs."
    # 使用 tokenizer 对文本进行编码，返回 JAX 数组
    input_ids = tokenizer([TXT], return_tensors="jax")["input_ids"]

    # 获取模型的输出 logits
    logits = model(input_ids).logits
    # 找到掩码位置的索引
    masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero()[0].item()
    # 对 logits 进行 softmax 处理
    probs = jax.nn.softmax(logits[0, masked_index], axis=0)
    # 获取最有可能的预测值和概率
    values, predictions = jax.lax.top_k(probs, k=1)

    # 解码预测结果并以空格分割
    tokenizer.decode(predictions).split()
# 覆盖调用文档字符串，将 FlaxBartForConditionalGeneration 类的文档字符串与 BART_INPUTS_DOCSTRING 和 FLAX_BART_CONDITIONAL_GENERATION_DOCSTRING 连接起来
overwrite_call_docstring(
    FlaxBartForConditionalGeneration, BART_INPUTS_DOCSTRING + FLAX_BART_CONDITIONAL_GENERATION_DOCSTRING
)
# 追加并替换返回文档字符串，将 FlaxBartForConditionalGeneration 类的文档字符串与输出类型为 FlaxSeq2SeqLMOutput、配置类为 _CONFIG_FOR_DOC 连接起来
append_replace_return_docstrings(
    FlaxBartForConditionalGeneration, output_type=FlaxSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC
)

# 定义 FlaxBartForSequenceClassificationModule 类
class FlaxBartForSequenceClassificationModule(nn.Module):
    # 类属性：配置，数据类型，默认为 float32，标签数量（可选）
    config: BartConfig
    dtype: jnp.dtype = jnp.float32
    num_labels: Optional[int] = None

    # 初始化方法
    def setup(self):
        # 创建 FlaxBartModule 类对象，并传入配置和数据类型
        self.model = FlaxBartModule(config=self.config, dtype=self.dtype)
        # 创建 FlaxBartClassificationHead 类对象，用于分类任务
        self.classification_head = FlaxBartClassificationHead(
            # 传入配置、内部维度、类别数量和池化器丢弃率
            config=self.config,
            inner_dim=self.config.d_model,
            num_classes=self.num_labels if self.num_labels is not None else self.config.num_labels,
            pooler_dropout=self.config.classifier_dropout,
        )

    # 获取编码器模块
    def _get_encoder_module(self):
        return self.model.encoder

    # 获取解码器模块
    def _get_decoder_module(self):
        return self.model.decoder

    # 调用方法
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
            # 调用模型进行前向传播，获取输出结果
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
            hidden_states = outputs[0]  # last hidden state

            # 创建一个 mask，用于标识输入中的 <eos> token
            eos_mask = jnp.where(input_ids == self.config.eos_token_id, 1, 0)

            # 处理由于 JAX 编译时出现的类型错误
            if type(eos_mask) != jax.interpreters.partial_eval.DynamicJaxprTracer:
                # 检查所有示例是否具有相同数量的 <eos> tokens
                if len(jnp.unique(eos_mask.sum(1))) > 1:
                    raise ValueError("All examples must have the same number of <eos> tokens.")

                # 检查是否存在缺失的 <eos> token
                if any(eos_mask.sum(1) == 0):
                    raise ValueError("There are missing <eos> tokens in input_ids")

                # 确保每个示例中仅保留最后一个 <eos> token
                eos_mask_noised = eos_mask + jnp.arange(eos_mask.shape[1]) * 1e-6
                eos_mask = jnp.where(eos_mask_noised == eos_mask_noised.max(1).reshape(-1, 1), 1, 0)

            # 根据 eos_mask 计算句子表示
            sentence_representation = jnp.einsum("ijk, ij -> ijk", hidden_states, eos_mask).sum(1)

            # 使用分类头部对句子表示进行分类
            logits = self.classification_head(sentence_representation, deterministic=deterministic)

            # 如果不返回字典，则返回分类结果和模型其他输出
            if not return_dict:
                output = (logits,) + outputs[1:]
                return output

            # 返回 Seq2Seq 分类器的输出字典
            return FlaxSeq2SeqSequenceClassifierOutput(
                logits=logits,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )
# 导入所需模块和函数
@add_start_docstrings(
    """
    在 Bart 模型的基础上添加了一个顶部的序列分类/头（在汇总输出之上的线性层），例如用于 GLUE 任务。
    """,
    BART_START_DOCSTRING,
)
# 创建 FlaxBartForSequenceClassification 类，继承自 FlaxBartPreTrainedModel
class FlaxBartForSequenceClassification(FlaxBartPreTrainedModel):
    # 定义模块类
    module_class = FlaxBartForSequenceClassificationModule
    # 定义数据类型
    dtype = jnp.float32

# 添加调用示例文档字符串
append_call_sample_docstring(
    FlaxBartForSequenceClassification,
    _CHECKPOINT_FOR_DOC,
    FlaxSeq2SeqSequenceClassifierOutput,
    _CONFIG_FOR_DOC,
)

# 创建 FlaxBartForQuestionAnsweringModule 类，继承自 nn.Module
class FlaxBartForQuestionAnsweringModule(nn.Module):
    # 定义配置
    config: BartConfig
    # 定义数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 定义标签数量为 2
    num_labels = 2

    # 初始化函数
    def setup(self):
        # 创建 Bart 模型
        self.model = FlaxBartModule(config=self.config, dtype=self.dtype)
        # 创建问答输出层
        self.qa_outputs = nn.Dense(
            self.num_labels, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std)
        )

    # 获取编码器模块
    def _get_encoder_module(self):
        return self.model.encoder

    # 获取解码器模块
    def _get_decoder_module(self):
        return self.model.decoder

    # 调用函数
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
        # 调用 Bart 模型
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

        # 获取序列输出
        sequence_output = outputs[0]

        # 通过问答输出层获得 logits
        logits = self.qa_outputs(sequence_output)
        # 将 logits 拆分为起始和结束 logits
        start_logits, end_logits = jnp.split(logits, logits.shape[-1], axis=-1)
        # 压缩维度
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # 如果不返回字典，则返回元组
        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return output

        # 返回字典形式的输出
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


@add_start_docstrings(
    """
    BART 模型在顶部添加了一个用于提取式问答任务（如 SQuAD）的跨度分类头部（在顶部的线性层）。
    """
    # 在隐藏状态输出之上的一层，用于计算“跨度开始标记”和“跨度结束标记”的逻辑。
    # `span start logits` 和 `span end logits` 的计算。
    """
    # BART 模型的文档字符串的起始部分
    BART_START_DOCSTRING,
# 导入语句结束符号
)

# 定义用于问答任务的 FlaxBartForQuestionAnswering 类，继承自 FlaxBartPreTrainedModel
class FlaxBartForQuestionAnswering(FlaxBartPreTrainedModel):
    # 指定模块类为 FlaxBartForQuestionAnsweringModule
    module_class = FlaxBartForQuestionAnsweringModule
    # 指定数据类型为 32 位浮点数
    dtype = jnp.float32

# 调用函数以添加调用示例的文档字符串
append_call_sample_docstring(
    FlaxBartForQuestionAnswering,  # 调用示例将添加到 FlaxBartForQuestionAnswering 类
    _CHECKPOINT_FOR_DOC,  # 用于检查点的参数
    FlaxSeq2SeqQuestionAnsweringModelOutput,  # 问答模型输出的类
    _CONFIG_FOR_DOC,  # 用于配置的参数
)

# 定义用于解码的预训练 BART 模型的类，继承自 FlaxPreTrainedModel
class FlaxBartDecoderPreTrainedModel(FlaxPreTrainedModel):
    # 配置类为 BartConfig
    config_class = BartConfig
    # 基础模型前缀为 "model"
    base_model_prefix: str = "model"
    # 模块类，默认为 None
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
        # 设置配置不是编码器解码器模式
        config.is_encoder_decoder = False
        # 根据模块类和配置创建模块
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 调用父类初始化方法
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)

        # 获取输入张量的形状
        batch_size, sequence_length = input_ids.shape
        # 创建位置编码
        position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # 分割随机数生成器
        params_rng, dropout_rng = jax.random.split(rng)
        # 创建随机数字典
        rngs = {"params": params_rng, "dropout": dropout_rng}
        # 初始化编码器隐藏状态和编码器注意力掩码
        encoder_hidden_states = jnp.zeros(input_shape + (self.config.d_model,))
        encoder_attention_mask = attention_mask
        # 初始化模块参数
        module_init_outputs = self.module.init(
            rngs,
            input_ids,
            attention_mask,
            position_ids,
            encoder_hidden_states,
            encoder_attention_mask,
            return_dict=False,
        )
        return module_init_outputs["params"]

    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
        """
        # 初始化检索缓存的输入变量
        input_ids = jnp.ones((batch_size, max_length), dtype="i4")
        attention_mask = jnp.ones_like(input_ids, dtype="i4")
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        # 初始化变量
        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        # 解冻并返回缓存
        return unfreeze(init_variables["cache"])

    @add_start_docstrings_to_model_forward(BART_DECODE_INPUTS_DOCSTRING)
    # 定义一个类的方法，用于模型的前向传播
    def __call__(
        self,
        # 输入的标识符数组，通常是输入序列的标识符编码
        input_ids: jnp.ndarray,
        # 注意力掩码，用于指示模型在哪些位置进行注意力计算
        attention_mask: Optional[jnp.ndarray] = None,
        # 位置标识符，用于指示输入序列中每个位置的绝对位置信息
        position_ids: Optional[jnp.ndarray] = None,
        # 编码器隐藏状态，用于传递编码器的隐藏状态给解码器
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        # 编码器注意力掩码，用于指示编码器哪些位置需要注意力计算
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        # 是否输出注意力权重
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典形式的结果
        return_dict: Optional[bool] = None,
        # 是否用于训练
        train: bool = False,
        # 模型参数
        params: dict = None,
        # 过去的键值对，用于处理有状态模型的过去状态
        past_key_values: dict = None,
        # 随机数生成器，用于 dropout 操作的随机数生成
        dropout_rng: PRNGKey = None,
```  
        # 如果 output_attentions 为 None，则使用配置中的 output_attentions 值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果 output_hidden_states 为 None，则使用配置中的 output_hidden_states 值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果 return_dict 为 None，则使用配置中的 return_dict 值
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 如果 encoder_hidden_states 不为 None 且 encoder_attention_mask 为 None，则创建全为 1 的注意力掩码
        if encoder_hidden_states is not None and encoder_attention_mask is None:
            batch_size, sequence_length = encoder_hidden_states.shape[:2]
            encoder_attention_mask = jnp.ones((batch_size, sequence_length))

        # 准备解码器输入
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        if position_ids is None:
            batch_size, sequence_length = input_ids.shape
            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # 处理任何 PRNG（伪随机数生成器）如果需要
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        inputs = {"params": params or self.params}

        # 如果传递了 past_key_values，则缓存已经初始化，必须传递一个私有标志 init_cache，以确保使用缓存。
        # 必须确保将缓存标记为可变，以便 FlaxBartAttention 模块可以更改它
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        # 应用模块，传递输入参数
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

        # 将更新后的缓存添加到模型输出
        if past_key_values is not None and return_dict:
            outputs, past_key_values = outputs
            outputs["past_key_values"] = unfreeze(past_key_values["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs, past_key_values = outputs
            outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

        return outputs
class FlaxBartDecoderWrapper(nn.Module):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when the causal language model is
    used in combination with the [`EncoderDecoderModel`] framework.
    """

    # 定义一个FlaxBartDecoderWrapper类，用于加载预训练检查点，当因果语言模型与EncoderDecoderModel框架结合使用时
    config: BartConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 设置embed_dim为config中的d_model
        embed_dim = self.config.d_model
        # 创建一个nn.Embed对象，用于嵌入tokens
        embed_tokens = nn.Embed(
            self.config.vocab_size,
            embed_dim,
            embedding_init=jax.nn.initializers.normal(self.config.init_std),
            dtype=self.dtype,
        )
        # 创建FlaxBartDecoder对象，传入config、embed_tokens和dtype
        self.decoder = FlaxBartDecoder(config=self.config, embed_tokens=embed_tokens, dtype=self.dtype)

    def __call__(self, *args, **kwargs):
        # 调用decoder对象的__call__方法
        return self.decoder(*args, **kwargs)


class FlaxBartForCausalLMModule(nn.Module):
    config: BartConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 创建FlaxBartDecoderWrapper对象，传入config和dtype
        self.model = FlaxBartDecoderWrapper(config=self.config, dtype=self.dtype)
        # 创建一个nn.Dense对象，用于LM头部
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
        # 调用model对象的__call__方法
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

        if self.config.tie_word_embeddings:
            # 如果配置中tie_word_embeddings为True，则共享嵌入层
            shared_embedding = self.model.variables["params"]["decoder"]["embed_tokens"]["embedding"]
            # 应用lm_head到hidden_states上
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
        else:
            # 否则直接应用lm_head到hidden_states上
            lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            # 如果不返回字典，则返回lm_logits和outputs[1:]
            return (lm_logits,) + outputs[1:]

        return FlaxCausalLMOutputWithCrossAttentions(
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


@add_start_docstrings(
    """
    Bart Decoder Model with a language modeling head on top (linear layer with weights tied to the input embeddings)
    e.g for autoregressive tasks.
    """,
    BART_START_DOCSTRING,
)
class FlaxBartForCausalLM(FlaxBartDecoderPreTrainedModel):
    # 将FlaxBartForCausalLMModule设置为module_class
    module_class = FlaxBartForCausalLMModule
    # 为生成准备输入数据，包括初始化缓存和构建注意力掩码
    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = None):
        # 初始化缓存，获取批次大小和序列长度
        batch_size, seq_length = input_ids.shape

        # 初始化过去的键值对，为每个样本准备缓存
        past_key_values = self.init_cache(batch_size, max_length)
        
        # 注意：通常情况下，需要在 attention_mask 中为 x > input_ids.shape[-1] 和 x < cache_length 的位置放置 0
        # 但由于解码器使用了因果型掩码，这些位置已经被掩码了
        # 因此，我们可以在这里创建一个静态的 attention_mask，这样更加高效
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            # 计算位置 ID，通过累加实现
            position_ids = attention_mask.cumsum(axis=-1) - 1
            # 动态更新掩码
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            # 如果没有给出注意力掩码，则创建一个形状与输入序列相同的默认掩码
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        # 返回生成所需的输入数据字典
        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    # 更新用于生成的输入数据，主要用于下一步的迭代生成
    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        # 更新模型关键参数，包括过去的键值对和位置 ID
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        # 返回更新后的参数字典
        return model_kwargs
# 调用函数append_call_sample_docstring，传入参数FlaxBartForCausalLM, _CHECKPOINT_FOR_DOC, FlaxCausalLMOutputWithCrossAttentions, _CONFIG_FOR_DOC
append_call_sample_docstring(
    FlaxBartForCausalLM,
    _CHECKPOINT_FOR_DOC,
    FlaxCausalLMOutputWithCrossAttentions,
    _CONFIG_FOR_DOC,
)
```