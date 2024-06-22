# `.\models\gpt_neo\modeling_flax_gpt_neo.py`

```py
# 导入所需的库
from functools import partial
from typing import Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
# 导入模型输出和工具类相关的库
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging

# 获取日志对象
logger = logging.get_logger(__name__)

# 为文档定义的配置和检查点信息
_CONFIG_FOR_DOC = "GPTNeoConfig"
_CHECKPOINT_FOR_DOC = "EleutherAI/gpt-neo-1.3B"

# GPT-NEO 模型的起始文档
GPT_NEO_START_DOCSTRING = r"""

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
    Parameters:
        config ([`GPTNeoConfig`]): Model configuration class with all the parameters of the model.
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


代码块解释：

- Parameters: 该函数的参数说明部分开始
- config ([`GPTNeoConfig`]): 输入参数 config，是 Model configuration 类的实例，包含模型的所有参数
- Initializing with a config file does not load the weights associated with the model, only the configuration.
    使用 config 文件进行初始化不会加载与模型关联的权重，只会加载配置信息
- Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.
    使用 [`~FlaxPreTrainedModel.from_pretrained`] 方法可以加载模型的权重
- dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
    dtype 参数，为 `jax.numpy.dtype` 类型，默认为 `jax.numpy.float32`
- The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and
    `jax.numpy.bfloat16` (on TPUs).
    计算的数据类型，可以是 `jax.numpy.float32`，`jax.numpy.float16`（在 GPU 上），`jax.numpy.bfloat16`（在 TPU 上）
- This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
    specified all the computation will be performed with the given `dtype`.
    此参数可用于在 GPU 或 TPU 上启用混合精度训练或半精度推理，如果指定，则所有计算将使用给定的 `dtype` 执行
- **Note that this only specifies the dtype of the computation and does not influence the dtype of model
    parameters.**
    注意，这只是指定计算的 dtype，并不影响模型参数的 dtype
- If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`] and
    [`~FlaxPreTrainedModel.to_bf16`].
    如果要更改模型参数的 dtype，请参阅 [`~FlaxPreTrainedModel.to_fp16`] 和 [`~FlaxPreTrainedModel.to_bf16`].
"""

GPT_NEO_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`numpy.ndarray` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length`. Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.
        past_key_values (`Dict[str, np.ndarray]`, *optional*, returned by `init_cache` or when passing previous `past_key_values`):
            Dictionary of pre-computed hidden-states (key and values in the attention blocks) that can be used for fast
            auto-regressive decoding. Pre-computed key and value hidden-states are of shape *[batch_size, max_length]*.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


# 定义 FlaxGPTNeoSelfAttention 类，继承自 nn.Module
class FlaxGPTNeoSelfAttention(nn.Module):
    # 定义类属性
    # config: GPTNeoConfig 指定 config 属性的类型为 GPTNeoConfig
    config: GPTNeoConfig
    # attention_type: str 指定 attention_type 属性的类型为 str
    attention_type: str
    # dtype: jnp.dtype = jnp.float32 指定 dtype 属性的类型为 jnp.dtype，默认值为 jnp.float32
    # 设置模型参数
    def setup(self):
        # 获取配置信息
        config = self.config
        # 设置嵌入维度
        self.embed_dim = config.hidden_size
        # 设置注意力头数
        self.num_heads = config.num_attention_heads
        # 计算每个注意力头的维度
        self.head_dim = self.embed_dim // self.num_heads
        # 如果 embed_dim 不能被 num_heads 整除，抛出数值错误
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and "
                f"`num_heads`: {self.num_heads})."
            )

        # 初始化注意力和残差的 dropout 层
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.resid_dropout)

        # 部分函数定义
        dense = partial(
            nn.Dense,
            self.embed_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

        # 初始化查询、键、值投影和输出投影层
        self.q_proj, self.k_proj, self.v_proj = dense(use_bias=False), dense(use_bias=False), dense(use_bias=False)
        self.out_proj = dense()

        # 创建因果掩码
        self.causal_mask = make_causal_mask(jnp.ones((1, config.max_position_embeddings), dtype="bool"), dtype="bool")
        # 如果注意力类型是 "local"，则更新因果掩码
        if self.attention_type == "local":
            self.causal_mask = self.causal_mask ^ jnp.tril(self.causal_mask, -config.window_size)

    # 将隐藏状态分割成多个注意力头
    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    # 将多个注意力头合并为原始隐藏状态
    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    # 定义一个紧凑层
    @nn.compact
    # 定义一个方法用于将来自单个输入标记的投影键、值状态与先前步骤的缓存状态连接起来。
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slighly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # 检测是否通过缺少现有缓存数据来初始化。
        is_initialized = self.has_variable("cache", "cached_key")
        # 如果未初始化，则使用全零数组初始化缓存的键
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        # 如果未初始化，则使用全零数组初始化缓存的值
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        # 如果未初始化，则使用标量值0初始化缓存索引
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        # 如果已经初始化
        if is_initialized:
            # 解压缓存的键的形状，以便获取其维度信息
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # 使用新的1D空间切片更新键、值缓存
            cur_index = cache_index.value
            # 计算新切片的索引
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            # 使用新的键更新缓存的键
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            # 使用新的值更新缓存的值
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            # 更新缓存的键和值
            cached_key.value = key
            cached_value.value = value
            # 更新缓存索引以反映新缓存向量的数量
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # 用于缓存的因果掩码：我们的单个查询位置只应该关注已生成和缓存的那些键位置，而不是剩余的零元素。
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            # 结合填充掩码和注意力掩码
            attention_mask = combine_masks(pad_mask, attention_mask)
        # 返回更新后的键、值和注意力掩码
        return key, value, attention_mask

    # 定义一个方法，允许该类的实例像函数一样被调用
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        # 使用查询矩阵乘以注意力投影矩阵，并乘以头尺寸的平方根，然后转换为指定数据类型
        query = self.q_proj(hidden_states) * jnp.sqrt(self.head_dim).astype(self.dtype)
        # 使用隐藏状态乘以键投影矩阵
        key = self.k_proj(hidden_states)
        # 使用隐藏状态乘以值投影矩阵
        value = self.v_proj(hidden_states)

        # 将查询、键、值进行头拆分
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        # 获取查询和键的长度
        query_length, key_length = query.shape[1], key.shape[1]

        # 如果存在缓存的键值对，则从缓存中获取相应的掩码
        if self.has_variable("cache", "cached_key"):
            mask_shift = self.variables["cache"]["cache_index"]
            max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
            # 使用动态切片获取因果掩码
            causal_mask = lax.dynamic_slice(
                self.causal_mask, (0, 0, mask_shift, 0), (1, 1, query_length, max_decoder_length)
            )
        else:
            # 否则直接使用预定义的因果掩码
            causal_mask = self.causal_mask[:, :, :query_length, :key_length]

        # 获取批量大小
        batch_size = hidden_states.shape[0]
        # 对因果掩码进行广播以适应批处理大小
        causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])

        # 将注意力掩码广播到与因果掩码相同的形状
        attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
        # 组合注意力掩码和因果掩码
        attention_mask = combine_masks(attention_mask, causal_mask)

        # 如果不是确定性的，并且注意力丢弃率大于0，则创建丢弃 RNG
        dropout_rng = None
        if not deterministic and self.config.attention_dropout > 0.0:
            dropout_rng = self.make_rng("dropout")

        # 在快速自回归解码期间，我们逐步地逐一提供一个位置，并逐步缓存键和值
        if self.has_variable("cache", "cached_key") or init_cache:
            # 将键、值、查询和注意力掩码连接到缓存中
            key, value, attention_mask = self._concatenate_to_cache(key, value, query, attention_mask)

        # 将布尔掩码转换为浮点数掩码
        attention_bias = lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
        )

        # 常规点积注意力计算
        attn_weights = dot_product_attention_weights(
            query,
            key,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attention_dropout,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=None,
        )

        # 将注意力权重乘以值，并使用 einsum 进行矩阵乘法
        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)
        # 将注意力输出进行头合并
        attn_output = self._merge_heads(attn_output)
        # 通过输出投影层处理注意力输出
        attn_output = self.out_proj(attn_output)
        # 使用残差丢弃层进行残差连接并进行丢弃
        attn_output = self.resid_dropout(attn_output, deterministic=deterministic)

        # 如果需要输出注意力权重，则将其包含在输出中
        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        # 返回输出
        return outputs
class FlaxGPTNeoAttention(nn.Module):
    # 定义一个自定义的注意力模块，继承自nn.Module
    config: GPTNeoConfig
    layer_id: int = 0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 设定注意力机制的类型
        attention_type = self.config.attention_layers[self.layer_id]
        # 创建自注意力对象
        self.attention = FlaxGPTNeoSelfAttention(self.config, attention_type, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):
        # 调用自注意力对象处理输入的hidden_states
        return self.attention(
            hidden_states,
            attention_mask=attention_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
        )


class FlaxGPTNeoMLP(nn.Module):
    # 定义一个多层感知器模块，继承自nn.Module
    config: GPTNeoConfig
    intermediate_size: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 获取隐藏层维度和初始化方法
        embed_dim = self.config.hidden_size
        kernel_init = jax.nn.initializers.normal(self.config.initializer_range)
        # 创建全连接层，并初始化
        self.c_fc = nn.Dense(self.intermediate_size, dtype=self.dtype, kernel_init=kernel_init)
        self.c_proj = nn.Dense(embed_dim, dtype=self.dtype, kernel_init=kernel_init)
        self.act = ACT2FN[self.config.activation_function]
        self.dropout = nn.Dropout(rate=self.config.resid_dropout)

    def __call__(self, hidden_states, deterministic: bool = True):
        # 处理隐藏状态数据，通过全连接层和激活函数，并应用dropout
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states


class FlaxGPTNeoBlock(nn.Module):
    # 定义一个GPTNeo模块，继承自nn.Module
    config: GPTNeoConfig
    layer_id: int = 0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 获取隐藏层大小和中间维度
        hidden_size = self.config.hidden_size
        inner_dim = self.config.intermediate_size if self.config.intermediate_size is not None else 4 * hidden_size
        # 创建层标准化模块和MLP模块
        self.ln_1 = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)
        self.attn = FlaxGPTNeoAttention(self.config, layer_id=self.layer_id, dtype=self.dtype)
        self.ln_2 = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)
        self.mlp = FlaxGPTNeoMLP(self.config, inner_dim, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        ):
            # 保存残差连接
            residual = hidden_states
            # Layer Normalization
            hidden_states = self.ln_1(hidden_states)
            # 通过self.attn进行自注意力机制计算
            outputs = self.attn(
                hidden_states,
                attention_mask=attention_mask,
                deterministic=deterministic,
                init_cache=init_cache,
                output_attentions=output_attentions,
            )
            # 残差连接
            attn_output = outputs[0]
            hidden_states = attn_output + residual

            # 保存残差连接
            residual = hidden_states
            # Layer Normalization
            hidden_states = self.ln_2(hidden_states)
            # 通过self.mlp进行Feed Forward计算
            feed_forward_hidden_states = self.mlp(hidden_states, deterministic=deterministic)
            # 残差连接
            hidden_states = residual + feed_forward_hidden_states

            # 返回结果
            return (hidden_states,) + outputs[1:]
class FlaxGPTNeoPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定配置类为 GPTNeoConfig
    config_class = GPTNeoConfig
    # 指定基础模型的前缀
    base_model_prefix = "transformer"
    # 模块类暂时未指定
    module_class: nn.Module = None

    def __init__(
        self,
        config: GPTNeoConfig,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        # 根据配置和其他参数初始化模块
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 调用父类的初始化方法
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        # 划分随机数生成器
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 初始化模块参数
        random_params = self.module.init(rngs, input_ids, attention_mask, position_ids, return_dict=False)["params"]

        # 如果提供了参数，则使用提供的参数替换缺失的参数
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
        """
        # 初始化用于检索缓存的输入变量
        input_ids = jnp.ones((batch_size, max_length))
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        # 初始化模块变量
        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        return unfreeze(init_variables["cache"])

    @add_start_docstrings_to_model_forward(GPT_NEO_INPUTS_DOCSTRING)
    # 定义一个调用方法，接受输入的参数，执行模型的前向传播
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
    ):
        # 设置输出注意力权重，默认为模型配置中的值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置输出隐藏状态，默认为模型配置中的值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置返回字典，默认为模型配置中的值
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 获取输入张量的形状
        batch_size, sequence_length = input_ids.shape

        # 如果位置编码为空
        if position_ids is None:
            # 如果过去的键值不为空，则抛出错误
            if past_key_values is not None:
                raise ValueError("Make sure to provide `position_ids` when passing `past_key_values`.")

            # 使用序列长度广播生成位置编码
            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # 如果注意力掩码为空，则初始化为全1
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))

        # 处理任何需要的 PRNG
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        # 构建输入字典
        inputs = {"params": params or self.params}

        # 如果传递了过去的键值，则初始化缓存，并确保缓存是可变的
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        # 执行模型的前向传播
        outputs = self.module.apply(
            inputs,
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            not train,
            False,
            output_attentions,
            output_hidden_states,
            return_dict,
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

        return outputs
class FlaxGPTNeoBlockCollection(nn.Module):
    # 定义一个类 FlaxGPTNeoBlockCollection，继承自 nn.Module
    config: GPTNeoConfig
    # 定义一个属性 config，类型为 GPTNeoConfig
    dtype: jnp.dtype = jnp.float32
    # 定义一个属性 dtype，类型为 jnp.float32，默认值为 jnp.float32

    def setup(self):
        # 定义一个方法 setup，用于初始化模块
        self.blocks = [
            # 创建一个列表 blocks
            FlaxGPTNeoBlock(self.config, layer_id=i, name=str(i), dtype=self.dtype)
            # 使用列表推导式创建 FlaxGPTNeoBlock 对象，并添加到 blocks 列表中
            for i in range(self.config.num_hidden_layers)
            # 遍历 num_hidden_layers 次
        ]

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 定义一个方法 __call__，用于执行模块的前向传播
        all_attentions = () if output_attentions else None
        # 如果 output_attentions 为真，则创建一个空元组，否则为 None
        all_hidden_states = () if output_hidden_states else None
        # 如果 output_hidden_states 为真，则创建一个空元组，否则为 None

        for block in self.blocks:
            # 遍历 blocks 列表中的每个元素
            if output_hidden_states:
                # 如果 output_hidden_states 为真
                all_hidden_states += (hidden_states,)
                # 将 hidden_states 添加到 all_hidden_states 元组中

            layer_outputs = block(
                hidden_states,
                attention_mask,
                deterministic=deterministic,
                init_cache=init_cache,
                output_attentions=output_attentions,
            )
            # 调用 block 对象进行前向传播，并获取输出
            hidden_states = layer_outputs[0]
            # 更新 hidden_states 为 block 输出的第一个元素

            if output_attentions:
                # 如果 output_attentions 为真
                all_attentions += (layer_outputs[1],)
                # 将 block 输出的第二个元素添加到 all_attentions 元组中

        # this contains possible `None` values - `FlaxGPTNeoModule` will filter them out
        # 返回包含可能为 None 值的元组，FlaxGPTNeoModule 将对其进行过滤
        outputs = (hidden_states, all_hidden_states, all_attentions)

        return outputs
        # 返回 outputs

class FlaxGPTNeoModule(nn.Module):
    # 定义一个类 FlaxGPTNeoModule，继承自 nn.Module
    config: GPTNeoConfig
    # 定义一个属性 config，类型为 GPTNeoConfig
    dtype: jnp.dtype = jnp.float32
    # 定义一个属性 dtype，类型为 jnp.float32，默认值为 jnp.float32

    def setup(self):
        # 定义一个方法 setup，用于初始化模块
        self.embed_dim = self.config.hidden_size
        # 设置 embed_dim 为 config 的 hidden_size
        embedding_init = jax.nn.initializers.normal(stddev=self.config.initializer_range)
        # 初始化 embedding_init 为正态分布，标准差为 config 的 initializer_range
        self.wte = nn.Embed(
            self.config.vocab_size,
            self.embed_dim,
            embedding_init=embedding_init,
        )
        # 创建一个嵌入层 wte，设置词汇大小和嵌入维度
        self.wpe = nn.Embed(
            self.config.max_position_embeddings,
            self.embed_dim,
            embedding_init=embedding_init,
        )
        # 创建一个嵌入层 wpe，设置最大位置嵌入和嵌入维度
        self.dropout = nn.Dropout(rate=self.config.embed_dropout)
        # 创建一个丢弃层 dropout，设置丢弃率为 config 的 embed_dropout
        self.h = FlaxGPTNeoBlockCollection(self.config, dtype=self.dtype)
        # 创建一个 FlaxGPTNeoBlockCollection 对象 h
        self.ln_f = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)
        # 创建一个 LayerNorm 层 ln_f，设置 epsilon 为 config 的 layer_norm_epsilon，dtype 为 self.dtype

    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        deterministic=True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        ):
        # 将输入的标识符转换为嵌入向量
        input_embeds = self.wte(input_ids.astype("i4"))
        # 将位置标识符转换为位置嵌入向量
        position_embeds = self.wpe(position_ids.astype("i4"))

        # 将输入嵌入向量和位置嵌入向量相加得到隐藏状态
        hidden_states = input_embeds + position_embeds
        # 对隐藏状态进行dropout处理
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)

        # 将隐藏状态传入Transformer模型的前向传播函数
        outputs = self.h(
            hidden_states,
            attention_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取模型输出中的隐藏状态
        hidden_states = outputs[0]
        # 对隐藏状态进行LayerNorm处理
        hidden_states = self.ln_f(hidden_states)

        # 再次获取模型输出中的隐藏状态
        hidden_states = outputs[0]
        # 对隐藏状态进行LayerNorm处理
        hidden_states = self.ln_f(hidden_states)

        # 如果需要输出所有隐藏状态
        if output_hidden_states:
            # 将所有隐藏状态存储在all_hidden_states中
            all_hidden_states = outputs[1] + (hidden_states,)
            # 更新模型输出
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            # 更新模型输出
            outputs = (hidden_states,) + outputs[1:]

        # 如果不需要返回字典形式的输出
        if not return_dict:
            # 返回模型输出中不为None的部分
            return tuple(v for v in outputs if v is not None)

        # 返回FlaxBaseModelOutput对象，包含最后的隐藏状态、所有隐藏状态和注意力权重
        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=outputs[1],
            attentions=outputs[-1],
        )
# 导入必要的库
@add_start_docstrings(
    "The bare GPTNeo Model transformer outputting raw hidden-states without any specific head on top.",
    GPT_NEO_START_DOCSTRING,
)
# 定义 FlaxGPTNeoModel 类，继承自 FlaxGPTNeoPreTrainedModel
class FlaxGPTNeoModel(FlaxGPTNeoPreTrainedModel):
    # 指定模块类为 FlaxGPTNeoModule
    module_class = FlaxGPTNeoModule

# 添加调用示例的文档字符串
append_call_sample_docstring(FlaxGPTNeoModel, _CHECKPOINT_FOR_DOC, FlaxBaseModelOutput, _CONFIG_FOR_DOC)

# 定义 FlaxGPTNeoForCausalLMModule 类，继承自 nn.Module
class FlaxGPTNeoForCausalLMModule(nn.Module):
    # 指定配置为 GPTNeoConfig，数据类型为 jnp.float32
    config: GPTNeoConfig
    dtype: jnp.dtype = jnp.float32

    # 初始化方法
    def setup(self):
        # 创建 GPTNeo 模块
        self.transformer = FlaxGPTNeoModule(self.config, dtype=self.dtype)
        # 创建语言模型头部
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )

    # 调用方法
    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 调用 transformer 方法
        outputs = self.transformer(
            input_ids,
            attention_mask,
            position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取隐藏状态
        hidden_states = outputs[0]

        # 如果词嵌入被绑定
        if self.config.tie_word_embeddings:
            # 获取共享的内核
            shared_kernel = self.transformer.variables["params"]["wte"]["embedding"].T
            # 应用 lm_head
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_kernel}}, hidden_states)
        else:
            # 应用 lm_head
            lm_logits = self.lm_head(hidden_states)

        # 如果不返回字典
        if not return_dict:
            return (lm_logits,) + outputs[1:]

        # 返回 FlaxCausalLMOutput 对象
        return FlaxCausalLMOutput(logits=lm_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

# 添加文档字符串
@add_start_docstrings(
    """
    The GPTNeo Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    GPT_NEO_START_DOCSTRING,
)
# 定义 FlaxGPTNeoForCausalLM 类，继承自 FlaxGPTNeoPreTrainedModel
class FlaxGPTNeoForCausalLM(FlaxGPTNeoPreTrainedModel):
    # 指定模块类为 FlaxGPTNeoForCausalLMModule
    module_class = FlaxGPTNeoForCausalLMModule
    # 为生成准备输入数据，初始化缓存
    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = None):
        # 获取输入数据的批量大小和序列长度
        batch_size, seq_length = input_ids.shape

        # 初始化缓存，返回过去的键值对
        past_key_values = self.init_cache(batch_size, max_length)
        
        # 由于 GPTNeo 使用因果掩码，不需要在 attention_mask 中填充 0，可以创建一个静态的 attention_mask
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            # 根据 attention_mask 计算位置 ID
            position_ids = attention_mask.cumsum(axis=-1) - 1
            # 更新 extended_attention_mask
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            # 如果没有提供 attention_mask，则使用默认的位置 ID
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        # 返回包含过去键值对、扩展的 attention_mask 和位置 ID 的字典
        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    # 更新生成的输入数据
    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        # 更新模型参数中的过去键值对和位置 ID
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs
# 调用函数append_call_sample_docstring，传入参数FlaxGPTNeoForCausalLM、_CHECKPOINT_FOR_DOC、FlaxCausalLMOutput、_CONFIG_FOR_DOC
append_call_sample_docstring(FlaxGPTNeoForCausalLM, _CHECKPOINT_FOR_DOC, FlaxCausalLMOutput, _CONFIG_FOR_DOC)
```