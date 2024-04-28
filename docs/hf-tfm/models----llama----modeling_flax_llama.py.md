# `.\transformers\models\llama\modeling_flax_llama.py`

```
# 设置文件编码为 utf-8
# 版权声明
# 基于 EleutherAI 的 GPT-NeoX 库以及该库中的 GPT-NeoX 和 OPT 实现的代码。已根据 Meta AI 团队训练模型时的一些架构差异进行了修改。
# 根据Apache许可证2.0版本授权。
# 除非符合许可证，否则不得使用此文件。
# 您可以获取许可证的副本。
# 在适用法律规定要求或书面同意的情况下，根据许可证分发的软件将基于“原样”基础分发，不附带任何类型的保证或条件。
# 请参阅许可证以获取有关特定语言授权权限和限制的信息。
# Flax LLaMA 模型。
from functools import partial                # 导入 functools 库的 partial 函数
from typing import Optional, Tuple           # 导入 typing 模块中的 Optional 和 Tuple 类型
import flax.linen as nn                     # 导入 flax.linen 库并且重命名为 nn
import jax                                  # 导入 jax 库
import jax.numpy as jnp                      # 导入 jax.numpy 并重命名为 jnp
import numpy as np                           # 导入 numpy 库并重命名为 np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze   # 从 flax.core.frozen_dict 库中导入 FrozenDict, freeze, unfreeze 函数
from flax.linen import combine_masks, make_causal_mask            # 导入 flax.linen 库中的 combine_masks 和 make_causal_mask 函数
from flax.linen.attention import dot_product_attention_weights    # 导入 flax.linen.attention 中的 dot_product_attention_weights 函数
from flax.traverse_util import flatten_dict, unflatten_dict        # 导入 flax.traverse_util 库的 flatten_dict, unflatten_dict 函数
from jax import lax                                               # 导入 jax 库中的 lax 模块
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput    # 导入 FlaxBaseModelOutput, FlaxCausalLMOutput 类
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring   # 导入 ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring 函数
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging    # 从 utils 模块中导入 add_start_docstrings, add_start_docstrings_to_model_forward, logging 函数
from .configuration_llama import LlamaConfig     # 从 configuration_llama 模块中导入 LlamaConfig 类

logger = logging.get_logger(__name__)    # 使用 logging 模块获取日志记录器，记录器名称为当前模块名称

_CONFIG_FOR_DOC = "LlamaConfig"    # 设置用于文档的配置信息为 "LlamaConfig"
_CHECKPOINT_FOR_DOC = "afmck/testing-llama-tiny"    # 设置用于文档的检查点为 "afmck/testing-llama-tiny"
_REAL_CHECKPOINT_FOR_DOC = "openlm-research/open_llama_3b_v2"    # 设置用于文档的真实检查点为 "openlm-research/open_llama_3b_v2"

LLAMA_START_DOCSTRING = r"""    # LLAMA_START_DOCSTRING 字符串开始标记

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


```    # LLAMA_START_DOCSTRING 字符串结束标记
    Parameters:
        # 参数说明:
        config ([`LlamaConfig`]): Model configuration class with all the parameters of the model.
            # 使用 `LlamaConfig` 类存储模型所有参数的配置。
            # 通过使用配置文件初始化该参数不会加载与模型相关的权重，只会加载配置。
            # 使用 [`~FlaxPreTrainedModel.from_pretrained`] 方法加载模型权重。
        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
            # 数据计算的数据类型。可选值为 `jax.numpy.float32`, `jax.numpy.float16`, 或 `jax.numpy.bfloat16`。
            # 可用于启用 GPUs 或 TPUs 上的混合精度训练或半精度推断。如果指定了，所有计算将使用给定的 `dtype` 进行。
            
            **Note that this only specifies the dtype of the computation and does not influence the dtype of model
            parameters.**
            # 注意：这只是指定计算的数据类型，并不影响模型参数的数据类型。

            If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`] and
            [`~FlaxPreTrainedModel.to_bf16`].
            # 如果想要更改模型参数的数据类型，请参阅 [`~FlaxPreTrainedModel.to_fp16`] 和 [`~FlaxPreTrainedModel.to_bf16`]。
"""

LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`numpy.ndarray` of shape `(batch_size, input_ids_length)`):
            模型输入的 token 索引数组，形状为 `(batch_size, input_ids_length)`。默认会忽略填充部分。

            可以使用 [`AutoTokenizer`] 获得索引。详见 [`PreTrainedTokenizer.encode`] 和
            [`PreTrainedTokenizer.__call__`] 了解详情。

            [什么是 input IDs?](../glossary#input-ids)
        attention_mask (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            避免对填充 token 进行注意力计算的掩码。掩码值取 `[0, 1]` 范围：

            - 1 表示**未掩码**的 token，
            - 0 表示**掩码**的 token。

            可以使用 [`AutoTokenizer`] 获得索引。详见 [`PreTrainedTokenizer.encode`] 和
            [`PreTrainedTokenizer.__call__`] 了解详情。

            如果使用 `past_key_values`，可能只需输入最后的 `decoder_input_ids`（参见 `past_key_values`）。

            如果需要更改填充行为，应阅读 [`modeling_opt._prepare_decoder_attention_mask`] 并根据需求修改。
            查看[论文](https://arxiv.org/abs/1910.13461)中的图1以了解默认策略的更多信息。

            - 1 表示头部**未掩码**，
            - 0 表示头部**掩码**。
        position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            每个输入 token 在位置嵌入中的位置索引数组，形状为 `(batch_size, sequence_length)`。取值范围为 `[0,
            config.n_positions - 1]`。

            [什么是 position IDs?](../glossary#position-ids)
        past_key_values (`Dict[str, np.ndarray]`, *optional*, 由 `init_cache` 返回或传递先前 `past_key_values`):
            预计算的隐藏状态键和值（在注意力块中）的字典，可用于快速自回归解码。预计算的键和值的隐藏状态为
            *[batch_size, max_length]* 的形状。
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。有关更多细节，请查看返回的张量中的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。有关更多细节，请查看返回的张量中的 `hidden_states`。
        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`] 而不是普通的元组。
"""


def create_sinusoidal_positions(num_pos, dim):
    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
    freqs = np.einsum("i , j -> i j", np.arange(num_pos), inv_freq).astype("float32")

    emb = np.concatenate((freqs, freqs), axis=-1)
    # 将正弦和余弦的结果连接起来，形成一个新的数组
    out = np.concatenate((np.sin(emb)[:, None, :], np.cos(emb)[:, None, :]), axis=-1)
    # 返回新数组中的前num_pos项
    return jnp.array(out[:, :, :num_pos])
def rotate_half(tensor):
    """Rotates half the hidden dims of the input."""
    # 将输入张量的一半隐藏维度进行旋转
    rotate_half_tensor = jnp.concatenate(
        (-tensor[..., tensor.shape[-1] // 2 :], tensor[..., : tensor.shape[-1] // 2]), axis=-1
    )
    return rotate_half_tensor


def apply_rotary_pos_emb(tensor, sin_pos, cos_pos):
    # 应用旋转位置编码到张量上
    return (tensor * cos_pos) + (rotate_half(tensor) * sin_pos)


class FlaxLlamaRMSNorm(nn.Module):
    config: LlamaConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 初始化 RMS 归一化层
        self.epsilon = self.config.rms_norm_eps
        self.weight = self.param("weight", lambda _, shape: jnp.ones(shape), self.config.hidden_size)

    def __call__(self, hidden_states):
        # 计算方差
        variance = jnp.asarray(hidden_states, dtype=jnp.float32)
        variance = jnp.power(variance, 2)
        variance = variance.mean(-1, keepdims=True)
        # 使用 `jax.numpy.sqrt`，因为 `jax.lax.rsqrt` 与 `torch.rsqrt` 不匹配
        hidden_states = hidden_states / jnp.sqrt(variance + self.epsilon)

        return self.weight * jnp.asarray(hidden_states, dtype=self.dtype)


class FlaxLlamaRotaryEmbedding(nn.Module):
    config: LlamaConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 设置 Llama 旋转嵌入层
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        self.sincos = create_sinusoidal_positions(self.config.max_position_embeddings, head_dim)

    def __call__(self, key, query, position_ids):
        # 获取位置编码
        sincos = self.sincos[position_ids]
        sin_pos, cos_pos = jnp.split(sincos, 2, axis=-1)

        # 应用旋转位置编码到关键键值和查询上
        key = apply_rotary_pos_emb(key, sin_pos, cos_pos)
        query = apply_rotary_pos_emb(query, sin_pos, cos_pos)

        key = jnp.asarray(key, dtype=self.dtype)
        query = jnp.asarray(query, dtype=self.dtype)

        return key, query


class FlaxLlamaAttention(nn.Module):
    config: LlamaConfig
    dtype: jnp.dtype = jnp.float32
    causal: bool = True
    is_cross_attention: bool = False

    def setup(self):
        # 设置 Llama 注意力层
        config = self.config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.attention_softmax_in_fp32 = self.dtype is not jnp.float32

        # 初始化 Dense 层
        dense = partial(
            nn.Dense,
            self.embed_dim,
            use_bias=config.attention_bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

        self.q_proj, self.k_proj, self.v_proj = dense(), dense(), dense()
        self.o_proj = dense()

        # 创建因果掩码
        self.causal_mask = make_causal_mask(jnp.ones((1, config.max_position_embeddings), dtype="bool"), dtype="bool")
        self.rotary_emb = FlaxLlamaRotaryEmbedding(config, dtype=self.dtype)

    def _split_heads(self, hidden_states):
        # 将隐藏状态拆分为头
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))
    # 定义一个私有方法用于合并头部，将隐藏状态张量重塑为（batch_size, seq_length, num_heads, head_dim）的形状
    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    @nn.compact
    # 从transformers库中的FlaxGPTNeoSelfAttention类中复制的方法，用于将单个输入标记的投影键、值状态连接到先前步骤的缓存状态
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slighly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # 检测是否通过缺少现有缓存数据来初始化
        is_initialized = self.has_variable("cache", "cached_key")
        # 初始化缓存键和缓存值的变量
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        # 初始化缓存索引的变量
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            # 获取批次维度和缓存长度等维度信息
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # 更新键、值缓存，将新的一维空间切片更新到缓存中
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            # 更新缓存索引，计算更新的缓存向量数量
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # 为缓存的解码器自注意力创建因果掩码：我们的单个查询位置只能关注已生成和缓存的键位置，而不是剩余的零元素
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)
        # 返回更新后的键、值和注意力掩码
        return key, value, attention_mask

    # 定义调用方法，用于执行自注意力机制
    def __call__(
        self,
        hidden_states,  # 隐藏状态张量
        attention_mask,  # 注意力掩码
        position_ids,  # 位置ID
        deterministic: bool = True,  # 是否确定性
        init_cache: bool = False,  # 是否初始化缓存
        output_attentions: bool = False,  # 是否输出注意力权重
        ):
        # 使用 self.q_proj 方法对 hidden_states 进行投影得到查询矩阵
        query = self.q_proj(hidden_states)
        # 使用 self.k_proj 方法对 hidden_states 进行投影得到键矩阵
        key = self.k_proj(hidden_states)
        # 使用 self.v_proj 方法对 hidden_states 进行投影得到数值矩阵
        value = self.v_proj(hidden_states)

        # 将查询矩阵进行头分割
        query = self._split_heads(query)
        # 将键矩阵进行头分割
        key = self._split_heads(key)
        # 将数值矩阵进行头分割
        value = self._split_heads(value)

        # 对键矩阵和查询矩阵进行轮换注意力机制操作并获取结果
        key, query = self.rotary_emb(key, query, position_ids)

        # 获取查询矩阵和键矩阵的长度
        query_length, key_length = query.shape[1], key.shape[1]

        # 如果存在缓存的键矩阵
        if self.has_variable("cache", "cached_key"):
            # 获取缓存的索引
            mask_shift = self.variables["cache"]["cache_index"]
            # 获取缓存的键矩阵的最大长度
            max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
            # 根据索引信息对 causal_mask 进行切片得到新的 causal_mask
            causal_mask = lax.dynamic_slice(
                self.causal_mask, (0, 0, mask_shift, 0), (1, 1, query_length, max_decoder_length)
            )
        else:
            # 获取正常的 causal_mask
            causal_mask = self.causal_mask[:, :, :query_length, :key_length]

        # 获取隐藏层状态的批次大小
        batch_size = hidden_states.shape[0]
        # 使用 jnp.broadcast_to 将 causal_mask 广播成和 hidden_states 相同的形状
        causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])

        # 使用 jnp.expand_dims 将 attention_mask 增加维度后，使用 jnp.broadcast_to 广播成和 causal_mask 相同的形状
        attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
        # 调用 combine_masks 方法将 attention_mask 和 causal_mask 合并
        attention_mask = combine_masks(attention_mask, causal_mask)

        dropout_rng = None
        # 如果非确定性模式下且注意力 dropout 大于 0，则需要创建 dropout_rng
        if not deterministic and self.config.attention_dropout > 0.0:
            dropout_rng = self.make_rng("dropout")

        # 在快速自回归解码期间，逐步对密钥和值进行缓存
        if self.has_variable("cache", "cached_key") or init_cache:
            key, value, attention_mask = self._concatenate_to_cache(key, value, query, attention_mask)

        # 将布尔类型的 attention_mask 转换为浮点型的 attention_bias
        attention_bias = lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
        )

        # 计算点乘注意力权重
        attention_dtype = jnp.float32 if self.attention_softmax_in_fp32 else self.dtype
        attn_weights = dot_product_attention_weights(
            query,
            key,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attention_dropout,
            deterministic=deterministic,
            dtype=attention_dtype,
        )

        # 如果 attention_softmax_in_fp32 为真，则将 attn_weights 转换为指定的数据类型
        if self.attention_softmax_in_fp32:
            attn_weights = attn_weights.astype(self.dtype)

        # 计算注意力输出
        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)
        # 对输出进行头合并
        attn_output = self._merge_heads(attn_output)
        # 使用 o_proj 对合并后的结果进行投影
        attn_output = self.o_proj(attn_output)

        # 如果需要输出注意力权重，则返回结果
        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs
# 定义一个 FlaxLlamaMLP 类，继承自 nn.Module 类
class FlaxLlamaMLP(nn.Module):
    # 设置类属性 config 为 LlamaConfig 类的实例，设置 dtype 为 jnp.float32 类型
    config: LlamaConfig
    dtype: jnp.dtype = jnp.float32

    # 设置类方法 setup
    def setup(self):
        # 获取隐藏层的嵌入维度
        embed_dim = self.config.hidden_size
        # 获取内部维度，如果未设置，则为 4 倍的隐藏层嵌入维度
        inner_dim = self.config.intermediate_size if self.config.intermediate_size is not None else 4 * embed_dim

        # 使用正态分布初始化器创建 kernel_init
        kernel_init = jax.nn.initializers.normal(self.config.initializer_range)
        # 根据 hidden_act 选择激活函数
        self.act = ACT2FN[self.config.hidden_act]

        # 创建全连接层 gate_proj，内部维度为 inner_dim，无偏置，数据类型为 dtype，权重初始化为 kernel_init
        self.gate_proj = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, kernel_init=kernel_init)
        # 创建全连接层 down_proj，内部维度为 embed_dim，无偏置，数据类型为 dtype，权重初始化为 kernel_init
        self.down_proj = nn.Dense(embed_dim, use_bias=False, dtype=self.dtype, kernel_init=kernel_init)
        # 创建全连接层 up_proj，内部维度为 inner_dim，无偏置，数据类型为 dtype，权重初始化为 kernel_init
        self.up_proj = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, kernel_init=kernel_init)

    # 定义类实例的调用方法
    def __call__(self, hidden_states):
        # 获取 up_proj 的输出
        up_proj_states = self.up_proj(hidden_states)
        # 获取 gate_proj 的输出，并经过激活函数处理
        gate_states = self.act(self.gate_proj(hidden_states))

        # 计算 hidden_states，使用 down_proj 对 up_proj_states 与 gate_states 的乘积进行处理
        hidden_states = self.down_proj(up_proj_states * gate_states)
        # 返回处理后的 hidden_states
        return hidden_states


# 定义一个 FlaxLlamaDecoderLayer 类，继承自 nn.Module 类
class FlaxLlamaDecoderLayer(nn.Module):
    # 设置类属性 config 为 LlamaConfig 类的实例，设置 dtype 为 jnp.float32 类型
    config: LlamaConfig
    dtype: jnp.dtype = jnp.float32

    # 设置类方法 setup
    def setup(self):
        # 初始化 input_layernorm 为 FlaxLlamaRMSNorm 类的实例，参数为 config 和 dtype
        self.input_layernorm = FlaxLlamaRMSNorm(self.config, dtype=self.dtype)
        # 初始化 self_attn 为 FlaxLlamaAttention 类的实例，参数为 config 和 dtype
        self.self_attn = FlaxLlamaAttention(self.config, dtype=self.dtype)
        # 初始化 post_attention_layernorm 为 FlaxLlamaRMSNorm 类的实例，参数为 config 和 dtype
        self.post_attention_layernorm = FlaxLlamaRMSNorm(self.config, dtype=self.dtype)
        # 初始化 mlp 为 FlaxLlamaMLP 类的实例，参数为 config 和 dtype
        self.mlp = FlaxLlamaMLP(self.config, dtype=self.dtype)

    # 定义类实例的调用方法
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):
        # 保存 hidden_states 作为 residual
        residual = hidden_states
        # 对 hidden_states 进行 input_layernorm 处理
        hidden_states = self.input_layernorm(hidden_states)
        # 使用 self_attn 处理 hidden_states，并返回 outputs
        outputs = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
        )
        # 添加 residual 连接
        attn_output = outputs[0]
        hidden_states = residual + attn_output

        # 保存 hidden_states 作为 residual
        residual = hidden_states
        # 对 hidden_states 进行 post_attention_layernorm 处理
        hidden_states = self.post_attention_layernorm(hidden_states)
        # 使用 mlp 处理 hidden_states
        hidden_states = self.mlp(hidden_states)
        # 添加 residual 连接
        hidden_states = residual + hidden_states

        # 返回包含 hidden_states 和 outputs[1:] 的元组
        return (hidden_states,) + outputs[1:]


# 从 transformers.models.gpt_neo.modeling_flax_gpt_neo.FlaxGPTNeoPreTrainedModel 复制，更新为处理 Llama 模型的 FlaxLlamaPreTrainedModel 类
class FlaxLlamaPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置 config_class 为 LlamaConfig，base_model_prefix 为 "model"，module_class 为 nn.Module 类的实例
    config_class = LlamaConfig
    base_model_prefix = "model"
    module_class: nn.Module = None
    # 初始化函数，设置模型的配置、输入形状、种子、数据类型等参数，并调用模块类的初始化函数
    def __init__(
        self,
        config: LlamaConfig,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        # 根据配置和数据类型初始化模块对象
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 调用父类的初始化函数
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    # 初始化权重函数，用于生成模型的随机参数
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        # 分割随机数种子
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 初始化模块参数
        random_params = self.module.init(rngs, input_ids, attention_mask, position_ids, return_dict=False)["params"]

        if params is not None:
            # 将随机参数和传入参数展开为字典，并对缺失的键进行处理
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    # 初始化缓存函数，用于生成自回归解码器的缓存
    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                用于快速自回归解码的批次大小。定义了初始化缓存的批次大小。
            max_length (`int`):
                自回归解码的最大可能长度。定义了初始化缓存的序列长度。
        """
        # 初始化用于检索缓存的输入变量
        input_ids = jnp.ones((batch_size, max_length))
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        # 初始化变量，并返回解冻的缓存
        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        return unfreeze(init_variables["cache"])

    # 调用函数，用于完成前向传播
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        params: dict = None,
        past_key_values: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 检查是否需要输出注意力权重
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    # 检查是否需要输出隐藏状态
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    # 检查是否需要返回字典形式的输出
    return_dict = return_dict if return_dict is not None else self.config.return_dict
    
    # 获取输入的batch_size和sequence_length
    batch_size, sequence_length = input_ids.shape
    
    # 如果未提供position_ids，则使用默认值，即序列的位置范围
    if position_ids is None:
        # 如果已传入past_key_values，则抛出异常
        if past_key_values is not None:
            raise ValueError("Make sure to provide `position_ids` when passing `past_key_values`.")
        # 使用默认的位置id：0到sequence_length-1
        position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))
    
    # 如果未提供attention_mask，则创建一个全为1的矩阵
    if attention_mask is None:
        attention_mask = jnp.ones((batch_size, sequence_length))
    
    # 处理丢弃随机数生成器（PRNG）（如果需要）
    rngs = {}
    if dropout_rng is not None:
        rngs["dropout"] = dropout_rng
    
    # 将参数封装为字典形式的输入
    inputs = {"params": params or self.params}
    
    # 如果已传入past_key_values，则将其作为cache传递给模型
    if past_key_values:
        inputs["cache"] = past_key_values
        # 将cache标记为可修改
        mutable = ["cache"]
    else:
        mutable = False
    
    # 调用模型的apply方法进行前向计算
    outputs = self.module.apply(
        inputs,
        jnp.array(input_ids, dtype="i4"),  # 输入的token ids编码
        jnp.array(attention_mask, dtype="i4"),  # 注意力掩码
        jnp.array(position_ids, dtype="i4"),  # 位置编码
        not train,  # 是否为训练模式
        False,
        output_attentions,  # 是否输出注意力权重
        output_hidden_states,  # 是否输出隐藏状态
        return_dict,  # 是否返回字典形式的输出
        rngs=rngs,  # 随机数生成器
        mutable=mutable,  # 可修改的输入
    )
    
    # 如果已传入past_key_values并且需要返回字典形式的输出，则将更新后的cache添加到模型输出中
    if past_key_values is not None and return_dict:
        outputs, past_key_values = outputs
        outputs["past_key_values"] = unfreeze(past_key_values["cache"])
        return outputs
    # 如果已传入past_key_values但不需要返回字典形式的输出，则将更新后的cache添加到模型输出的合适位置
    elif past_key_values is not None and not return_dict:
        outputs, past_key_values = outputs
        outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]
    
    # 返回模型输出
    return outputs
# 定义一个名为FlaxLlamaLayerCollection的类，继承自nn.Module
class FlaxLlamaLayerCollection(nn.Module):
    # 指定配置参数config为LlamaConfig类型
    config: LlamaConfig
    # 指定数据类型dtype为jnp.float32，默认为jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 初始化函数，用于设置层次结构
    def setup(self):
        # 创建一个由多个FlaxLlamaDecoderLayer对象组成的列表，列表长度为config中指定的num_hidden_layers
        self.blocks = [
            FlaxLlamaDecoderLayer(self.config, dtype=self.dtype, name=str(i))
            for i in range(self.config.num_hidden_layers)
        ]

    # 定义调用该类实例时的行为
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
    ):
        # 如果需要输出注意力值，则初始化一个空元组，用于存储所有注意力值，默认为None
        all_attentions = () if output_attentions else None
        # 如果需要输出隐藏状态，则初始化一个空元组，用于存储所有隐藏状态，默认为None
        all_hidden_states = () if output_hidden_states else None

        # 遍历self.blocks列表中的每个FlaxLlamaDecoderLayer对象
        for block in self.blocks:
            # 如果需要输出隐藏状态，则将当前隐藏状态加入到all_hidden_states元组中
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            # 调用当前FlaxLlamaDecoderLayer对象的__call__方法，计算当前层的输出
            layer_outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                deterministic=deterministic,
                init_cache=init_cache,
                output_attentions=output_attentions,
            )
            # 更新隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力值，则将当前层的注意力值加入到all_attentions元组中
            if output_attentions:
                all_attentions += (layer_outputs[1],)

        # outputs包含可能为None的值 - `FlaxLlamaModule`将过滤掉这些值
        # 将最终的输出结果组成一个元组，包括最终的隐藏状态、所有隐藏状态和所有注意力值
        outputs = (hidden_states, all_hidden_states, all_attentions)

        # 返回输出结果元组
        return outputs


# 定义一个名为FlaxLlamaModule的类，继承自nn.Module
class FlaxLlamaModule(nn.Module):
    # 指定配置参数config为LlamaConfig类型
    config: LlamaConfig
    # 指定数据类型dtype为jnp.float32，默认为jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 初始化函数，用于设置层次结构
    def setup(self):
        # 设置隐藏大小为config中指定的hidden_size
        self.hidden_size = self.config.hidden_size
        # 初始化嵌入层的参数，使用正态分布初始化，标准差为config中指定的initializer_range
        embedding_init = jax.nn.initializers.normal(stddev=self.config.initializer_range)
        # 创建嵌入层，用于将输入token转换为隐藏状态向量
        self.embed_tokens = nn.Embed(
            self.config.vocab_size,
            self.hidden_size,
            embedding_init=embedding_init,
            dtype=self.dtype,
        )
        # 创建FlaxLlamaLayerCollection对象，用于堆叠多个FlaxLlamaDecoderLayer对象
        self.layers = FlaxLlamaLayerCollection(self.config, dtype=self.dtype)
        # 创建FlaxLlamaRMSNorm对象，用于归一化输入数据
        self.norm = FlaxLlamaRMSNorm(self.config, dtype=self.dtype)

    # 定义调用该类实例时的行为
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        deterministic=True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    # 根据输入的参数进行模型推理
    def __call__(
        self,
        input_ids: jnp.ndarray,
        position_ids: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        init_cache: Optional[Dict[str, Optional[jnp.ndarray]]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 将输入的词向量嵌入为嵌入向量
        input_embeds = self.embed_tokens(input_ids.astype("i4"))
    
        # 使用嵌入向量进行模型的前向传播
        outputs = self.layers(
            input_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
        # 获取模型前向传播结果中的隐藏状态，并进行归一化处理
        hidden_states = outputs[0]
        hidden_states = self.norm(hidden_states)
    
        # 根据设置的参数判断是否需要返回所有隐藏状态
        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            outputs = (hidden_states,) + outputs[1:]
    
        # 根据设置的参数判断是否需要返回字典类型的结果
        if not return_dict:
            return tuple(v for v in outputs if v is not None)
    
        # 返回模型输出结果的字典形式
        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=outputs[1],
            attentions=outputs[-1],
        )
# 为 FlaxLlamaModel 添加起始注释字符串和 Llama 模型的基础字符串
@add_start_docstrings(
    "The bare Llama Model transformer outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
# 定义 FlaxLlamaModel 类，继承自 FlaxLlamaPreTrainedModel，module_class 为 FlaxLlamaModule
class FlaxLlamaModel(FlaxLlamaPreTrainedModel):
    module_class = FlaxLlamaModule

# 向 FlaxLlamaModel 添加调用示例的文档字符串
append_call_sample_docstring(
    FlaxLlamaModel,
    _CHECKPOINT_FOR_DOC,
    FlaxBaseModelOutput,
    _CONFIG_FOR_DOC,
    real_checkpoint=_REAL_CHECKPOINT_FOR_DOC,
)

# 定义 FlaxLlamaForCausalLMModule 类
class FlaxLlamaForCausalLMModule(nn.Module):
    config: LlamaConfig
    dtype: jnp.dtype = jnp.float32

    # 模块设置函数
    def setup(self):
        self.model = FlaxLlamaModule(self.config, dtype=self.dtype)
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )

    # 对象调用函数
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        outputs = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (lm_logits,) + outputs[1:]

        # 返回 FlaxCausalLMOutput 对象
        return FlaxCausalLMOutput(logits=lm_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

# 为 FlaxLlamaForCausalLM 添加起始注释字符串和 Llama 模型的基础字符串
@add_start_docstrings(
    """
    The Llama Model transformer with a language modeling head (linear layer) on top.
    """,
    LLAMA_START_DOCSTRING,
)
# 定义 FlaxLlamaForCausalLM 类，继承自 FlaxLlamaPreTrainedModel，module_class 为 FlaxLlamaForCausalLMModule
class FlaxLlamaForCausalLM(FlaxLlamaPreTrainedModel):
    module_class = FlaxLlamaForCausalLMModule
    # 定义生成过程中输入准备函数，接受输入的token IDs、最大生成长度以及可选的注意力掩码
    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = None):
        # 初始化缓存，获取输入的批量大小和序列长度
        batch_size, seq_length = input_ids.shape
    
        # 使用初始化函数初始化过去的键值对
        past_key_values = self.init_cache(batch_size, max_length)
    
        # 注意：通常需要将注意力掩码中大于输入序列长度和小于缓存长度的位置设为0。
        # 但由于Llama使用因果掩码，这些位置已经被掩盖了。
        # 因此，我们可以在这里创建一个静态的注意力掩码，这对于编译效率更高。
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        
        # 如果提供了注意力掩码，则根据累计位置计算位置 IDs
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            # 使用动态更新切片方法将注意力掩码应用到扩展的注意力掩码中
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            # 否则，生成位置 IDs 为 [0, 1, 2, ..., seq_length-1] 的数组，并广播到批量大小
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))
    
        # 返回准备好的输入字典，包括过去的键值对、注意力掩码和位置 IDs
        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }
    
    # 定义生成过程中输入更新函数，接受模型输出和模型参数字典
    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        # 更新模型参数字典中的过去键值对和位置 IDs
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        
        # 返回更新后的模型参数字典
        return model_kwargs
# 向函数中添加调用样例的文档字符串
append_call_sample_docstring(
    # 目标函数名
    FlaxLlamaForCausalLM,
    # 用于文档的检查点
    _CHECKPOINT_FOR_DOC,
    # 目标函数的输出类型
    FlaxCausalLMOutput,
    # 用于文档的配置
    _CONFIG_FOR_DOC,
    # 真实检查点的路径，用于文档
    real_checkpoint=_REAL_CHECKPOINT_FOR_DOC,
)
```