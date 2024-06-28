# `.\models\gpt_neo\modeling_flax_gpt_neo.py`

```
# 导入所需模块和库
from functools import partial
from typing import Optional, Tuple

# 导入 Flax 相关模块
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax

# 导入自定义的模型输出和工具函数
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging

# 导入 GPT-Neo 的配置类
from .configuration_gpt_neo import GPTNeoConfig

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 文档中使用的配置和检查点字符串常量
_CONFIG_FOR_DOC = "GPTNeoConfig"
_CHECKPOINT_FOR_DOC = "EleutherAI/gpt-neo-1.3B"

# GPT-Neo 模型的起始文档字符串
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
    # Parameters: 定义函数参数列表
    #     config ([`GPTNeoConfig`]): 使用 `GPTNeoConfig` 类配置模型的参数
    #         Initializing with a config file does not load the weights associated with the model, only the
    #         configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.
    #         用配置文件初始化不会加载与模型关联的权重，仅加载配置。查看 [`~FlaxPreTrainedModel.from_pretrained`] 方法以加载模型权重。
    #     dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
    #         计算的数据类型。可以是 `jax.numpy.float32`, `jax.numpy.float16`（在GPU上）和 `jax.numpy.bfloat16`（在TPU上）之一。
    #         可用于在GPU或TPU上启用混合精度训练或半精度推断。如果指定，则所有计算将使用给定的 `dtype` 进行。
    #         
    #         **注意，这仅指定计算的数据类型，不影响模型参数的数据类型。**
    #         
    #         如果希望更改模型参数的数据类型，请参阅 [`~FlaxPreTrainedModel.to_fp16`] 和 [`~FlaxPreTrainedModel.to_bf16`]。
"""
FlaxGPTNeoSelfAttention 类的文档字符串，描述了该类的输入参数和返回内容。

Args:
    input_ids (`numpy.ndarray` of shape `(batch_size, input_ids_length)`):
        输入序列标记的索引数组，形状为 `(batch_size, input_ids_length)`。
        可以使用 `AutoTokenizer` 获取这些索引。参见 `PreTrainedTokenizer.encode` 和 `PreTrainedTokenizer.__call__` 的详细说明。

        [什么是输入 ID？](../glossary#input-ids)
    attention_mask (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
        注意力遮罩，用于避免对填充标记索引进行注意力计算。遮罩值在 `[0, 1]` 范围内：

        - 对于不被遮罩的标记，值为 1，
        - 对于被遮罩的标记，值为 0。

        [什么是注意力遮罩？](../glossary#attention-mask)
    position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
        输入序列标记在位置嵌入中的位置索引数组。取值范围是 `[0, config.max_position_embeddings - 1]`。
    past_key_values (`Dict[str, np.ndarray]`, *optional*, 由 `init_cache` 返回或传入先前的 `past_key_values`):
        预先计算的隐藏状态字典（用于注意力模块中的键和值）。预计算的键和值的形状为 `[batch_size, max_length]`。
    output_attentions (`bool`, *optional*):
        是否返回所有注意力层的注意力张量。更多细节参见返回张量中的 `attentions` 字段。
    output_hidden_states (`bool`, *optional*):
        是否返回所有层的隐藏状态。更多细节参见返回张量中的 `hidden_states` 字段。
    return_dict (`bool`, *optional*):
        是否返回 `~utils.ModelOutput` 而不是普通的元组。
"""
class FlaxGPTNeoSelfAttention(nn.Module):
    # FlaxGPTNeoSelfAttention 类，继承自 nn.Module
    config: GPTNeoConfig
    # GPTNeoConfig 类型的 config 属性
    attention_type: str
    # 注意力类型字符串属性
    dtype: jnp.dtype = jnp.float32
    # 数据类型，默认为 jnp.float32
    def setup(self):
        # 从配置中获取参数
        config = self.config
        # 设置嵌入维度为隐藏大小
        self.embed_dim = config.hidden_size
        # 设置注意力头的数量
        self.num_heads = config.num_attention_heads
        # 计算每个注意力头的维度
        self.head_dim = self.embed_dim // self.num_heads
        # 检查 embed_dim 是否能被 num_heads 整除
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and "
                f"`num_heads`: {self.num_heads})."
            )

        # 初始化注意力和残差的 dropout 层
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.resid_dropout)

        # 部分应用带有特定初始化的 Dense 层函数
        dense = partial(
            nn.Dense,
            self.embed_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

        # 初始化 query、key、value 的投影层
        self.q_proj, self.k_proj, self.v_proj = dense(use_bias=False), dense(use_bias=False), dense(use_bias=False)
        # 初始化输出投影层
        self.out_proj = dense()

        # 创建因果遮蔽掩码
        self.causal_mask = make_causal_mask(jnp.ones((1, config.max_position_embeddings), dtype="bool"), dtype="bool")
        # 如果注意力类型为局部注意力，则修改因果遮蔽掩码
        if self.attention_type == "local":
            self.causal_mask = self.causal_mask ^ jnp.tril(self.causal_mask, -config.window_size)

    # 将隐藏状态按头分割
    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    # 将分割的头合并为隐藏状态
    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    # 使用 nn.compact 装饰器定义紧凑模块
    @nn.compact


这些注释描述了给定代码中每个方法和语句的功能和作用。
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slighly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # 检测是否正在初始化，通过检查缓存数据是否存在来判断
        is_initialized = self.has_variable("cache", "cached_key")
        # 获取或创建缓存的键（key）和值（value）变量，如果不存在则初始化为全零数组
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        # 获取或创建缓存的索引（index）变量，如果不存在则初始化为整数 0
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            # 获取缓存键（key）的形状，并提取批次维度、最大长度、头数、每头深度
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # 使用新的一维空间切片更新键（key）和值（value）的缓存
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            # 更新缓存中的键（key）和值（value）
            cached_key.value = key
            cached_value.value = value
            # 计算更新的缓存向量数，并更新缓存索引（index）
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # 用于缓存解码器自注意力的因果掩码：我们的单个查询位置应该仅注意到已生成并缓存的键位置，而不是剩余的零元素。
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            # 组合并更新注意力掩码（mask）
            attention_mask = combine_masks(pad_mask, attention_mask)
        
        # 返回更新后的键（key）、值（value）和注意力掩码（mask）
        return key, value, attention_mask
        ):
            # 计算查询向量，乘以 sqrt(head_dim)，并转换为指定数据类型
            query = self.q_proj(hidden_states) * jnp.sqrt(self.head_dim).astype(self.dtype)
            # 计算键向量
            key = self.k_proj(hidden_states)
            # 计算值向量
            value = self.v_proj(hidden_states)

            # 将查询向量拆分成多个头部
            query = self._split_heads(query)
            # 将键向量拆分成多个头部
            key = self._split_heads(key)
            # 将值向量拆分成多个头部
            value = self._split_heads(value)

            # 获取查询向量和键向量的长度
            query_length, key_length = query.shape[1], key.shape[1]

            # 如果存在缓存的键，则创建一个因果遮罩
            if self.has_variable("cache", "cached_key"):
                mask_shift = self.variables["cache"]["cache_index"]
                max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
                # 从因果遮罩中动态切片出部分用于当前查询和键的长度
                causal_mask = lax.dynamic_slice(
                    self.causal_mask, (0, 0, mask_shift, 0), (1, 1, query_length, max_decoder_length)
                )
            else:
                # 否则使用整个因果遮罩
                causal_mask = self.causal_mask[:, :, :query_length, :key_length]

            # 获取批次大小
            batch_size = hidden_states.shape[0]
            # 将因果遮罩广播到与注意力头部匹配的形状
            causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])

            # 将注意力遮罩广播到与因果遮罩相同的形状
            attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
            # 合并注意力遮罩和因果遮罩
            attention_mask = combine_masks(attention_mask, causal_mask)

            dropout_rng = None
            if not deterministic and self.config.attention_dropout > 0.0:
                # 如果不是确定性的且注意力 dropout 大于 0，则创建一个随机数生成器用于 dropout
                dropout_rng = self.make_rng("dropout")

            # 在快速自回归解码期间，我们逐步输入一个位置，并逐步缓存键和值
            if self.has_variable("cache", "cached_key") or init_cache:
                # 如果存在缓存的键或者需要初始化缓存，则将键、值和注意力遮罩连接到缓存中
                key, value, attention_mask = self._concatenate_to_cache(key, value, query, attention_mask)

            # 将布尔类型的注意力遮罩转换为浮点数类型的注意力偏置
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
            )

            # 常规的点积注意力计算
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

            # 使用 einsum 执行加权求和得到注意力输出
            attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)
            # 合并多头注意力的输出
            attn_output = self._merge_heads(attn_output)
            # 对输出应用输出投影
            attn_output = self.out_proj(attn_output)
            # 应用残差连接中的 dropout
            attn_output = self.resid_dropout(attn_output, deterministic=deterministic)

            # 根据需要返回注意力输出和注意力权重
            outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
            return outputs
class FlaxGPTNeoAttention(nn.Module):
    config: GPTNeoConfig  # 类变量，存储模型配置信息
    layer_id: int = 0  # 类变量，表示当前层的索引，默认为0
    dtype: jnp.dtype = jnp.float32  # 类变量，指定数据类型为32位浮点数

    def setup(self):
        attention_type = self.config.attention_layers[self.layer_id]
        self.attention = FlaxGPTNeoSelfAttention(self.config, attention_type, dtype=self.dtype)
        # 根据配置和注意力类型创建自注意力层对象

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):
        return self.attention(
            hidden_states,
            attention_mask=attention_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
        )
        # 调用自注意力层对象处理输入隐藏状态，并返回处理后的结果


class FlaxGPTNeoMLP(nn.Module):
    config: GPTNeoConfig  # 类变量，存储模型配置信息
    intermediate_size: int  # 类变量，表示中间隐藏层的大小
    dtype: jnp.dtype = jnp.float32  # 类变量，指定数据类型为32位浮点数

    def setup(self):
        embed_dim = self.config.hidden_size
        kernel_init = jax.nn.initializers.normal(self.config.initializer_range)
        self.c_fc = nn.Dense(self.intermediate_size, dtype=self.dtype, kernel_init=kernel_init)
        # 创建全连接层，用于变换隐藏状态的维度
        self.c_proj = nn.Dense(embed_dim, dtype=self.dtype, kernel_init=kernel_init)
        # 创建全连接层，将变换后的隐藏状态映射回原始维度
        self.act = ACT2FN[self.config.activation_function]
        # 根据配置选择激活函数
        self.dropout = nn.Dropout(rate=self.config.resid_dropout)
        # 创建Dropout层，用于随机置零输入张量的元素，以防止过拟合

    def __call__(self, hidden_states, deterministic: bool = True):
        hidden_states = self.c_fc(hidden_states)
        # 使用全连接层变换隐藏状态
        hidden_states = self.act(hidden_states)
        # 应用激活函数
        hidden_states = self.c_proj(hidden_states)
        # 使用全连接层将变换后的隐藏状态映射回原始维度
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 应用Dropout层
        return hidden_states
        # 返回处理后的隐藏状态


class FlaxGPTNeoBlock(nn.Module):
    config: GPTNeoConfig  # 类变量，存储模型配置信息
    layer_id: int = 0  # 类变量，表示当前层的索引，默认为0
    dtype: jnp.dtype = jnp.float32  # 类变量，指定数据类型为32位浮点数

    def setup(self):
        hidden_size = self.config.hidden_size
        inner_dim = self.config.intermediate_size if self.config.intermediate_size is not None else 4 * hidden_size
        # 根据配置确定内部维度大小

        self.ln_1 = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)
        # 创建LayerNorm层，用于归一化隐藏状态
        self.attn = FlaxGPTNeoAttention(self.config, layer_id=self.layer_id, dtype=self.dtype)
        # 创建注意力层对象
        self.ln_2 = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)
        # 创建LayerNorm层，用于归一化注意力输出
        self.mlp = FlaxGPTNeoMLP(self.config, inner_dim, dtype=self.dtype)
        # 创建MLP对象，用于处理注意力输出

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):
        # 通过注意力层和MLP层处理输入
        ):
            # 保存原始的隐藏状态，用于残差连接
            residual = hidden_states
            # LayerNormalization层，对隐藏状态进行归一化处理
            hidden_states = self.ln_1(hidden_states)
            # 注意力机制，处理隐藏状态并返回输出
            outputs = self.attn(
                hidden_states,
                attention_mask=attention_mask,
                deterministic=deterministic,
                init_cache=init_cache,
                output_attentions=output_attentions,
            )
            # 从注意力输出中获取注意力层的输出
            attn_output = outputs[0]
            # 执行残差连接，更新隐藏状态
            hidden_states = attn_output + residual

            # 保存当前隐藏状态作为残差
            residual = hidden_states
            # LayerNormalization层，再次对隐藏状态进行归一化处理
            hidden_states = self.ln_2(hidden_states)
            # 多层感知机（MLP），对隐藏状态进行前馈神经网络处理
            feed_forward_hidden_states = self.mlp(hidden_states, deterministic=deterministic)
            # 执行残差连接，更新隐藏状态
            hidden_states = residual + feed_forward_hidden_states

            # 返回更新后的隐藏状态以及可能的额外输出
            return (hidden_states,) + outputs[1:]
    # FlaxGPTNeoPreTrainedModel 类定义，继承自 FlaxPreTrainedModel，用于处理权重初始化以及预训练模型下载和加载的抽象类
    class FlaxGPTNeoPreTrainedModel(FlaxPreTrainedModel):
        """
        An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
        models.
        """

        # 指定配置类为 GPTNeoConfig
        config_class = GPTNeoConfig
        # 指定基础模型前缀为 "transformer"
        base_model_prefix = "transformer"
        # 模块类变量初始化为 None
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
            # 使用给定的配置和参数初始化模块对象
            module = self.module_class(config=config, dtype=dtype, **kwargs)
            # 调用父类构造函数初始化模型
            super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

        def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
            # 初始化输入张量
            input_ids = jnp.zeros(input_shape, dtype="i4")
            attention_mask = jnp.ones_like(input_ids)
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
            # 划分随机数生成器
            params_rng, dropout_rng = jax.random.split(rng)
            rngs = {"params": params_rng, "dropout": dropout_rng}

            # 使用初始化的参数生成随机参数
            random_params = self.module.init(rngs, input_ids, attention_mask, position_ids, return_dict=False)["params"]

            # 如果存在预定义的参数，将随机参数与预定义参数合并
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

            # 使用初始化的变量生成缓存
            init_variables = self.module.init(
                jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
            )
            return unfreeze(init_variables["cache"])

        # 添加模型输入文档字符串到模型前向方法
        @add_start_docstrings_to_model_forward(GPT_NEO_INPUTS_DOCSTRING)
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
        # 设置输出注意力权重的选项，如果未指定，则使用配置中的默认设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置输出隐藏状态的选项，如果未指定，则使用配置中的默认设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置返回字典的选项，如果未指定，则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 获取输入张量的批次大小和序列长度
        batch_size, sequence_length = input_ids.shape

        # 如果未提供位置编码，则根据序列长度和批次大小创建默认位置编码
        if position_ids is None:
            if past_key_values is not None:
                raise ValueError("Make sure to provide `position_ids` when passing `past_key_values`.")
            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # 如果未提供注意力掩码，则创建一个全为1的默认注意力掩码
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))

        # 处理任何需要的伪随机数生成器
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = {"params": params or self.params}

        # 如果传递了过去的键值，则初始化缓存，并确保缓存是可变的
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        # 调用模块的应用方法来处理输入，并传递必要的参数和选项
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

        # 如果传递了过去的键值并且需要返回字典，则将更新后的缓存添加到模型输出中
        if past_key_values is not None and return_dict:
            outputs, past_key_values = outputs
            outputs["past_key_values"] = unfreeze(past_key_values["cache"])
            return outputs
        # 如果传递了过去的键值但不需要返回字典，则更新缓存后将其添加到模型输出中
        elif past_key_values is not None and not return_dict:
            outputs, past_key_values = outputs
            outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

        # 返回最终的模型输出
        return outputs
class FlaxGPTNeoBlockCollection(nn.Module):
    config: GPTNeoConfig
    dtype: jnp.dtype = jnp.float32

    # 初始化方法，设置模块内的各个子块
    def setup(self):
        # 创建一个由多个 FlaxGPTNeoBlock 实例组成的列表，每个块对应模型的一个隐藏层
        self.blocks = [
            FlaxGPTNeoBlock(self.config, layer_id=i, name=str(i), dtype=self.dtype)
            for i in range(self.config.num_hidden_layers)
        ]

    # 调用方法，接受输入并依次经过每个块的处理
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
        # 如果需要输出注意力矩阵，则初始化空的元组用于存储每个块的注意力矩阵
        all_attentions = () if output_attentions else None
        # 如果需要输出隐藏状态，则初始化空的元组用于存储每个块的隐藏状态
        all_hidden_states = () if output_hidden_states else None

        # 对每个块进行迭代处理
        for block in self.blocks:
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # 调用当前块的处理方法，得到该块的输出
            layer_outputs = block(
                hidden_states,
                attention_mask,
                deterministic=deterministic,
                init_cache=init_cache,
                output_attentions=output_attentions,
            )
            # 更新当前隐藏状态为当前块的输出的第一个元素（通常是下一层的隐藏状态）
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力矩阵，则将当前块的注意力矩阵添加到 all_attentions 中
            if output_attentions:
                all_attentions += (layer_outputs[1],)

        # 组装最终输出元组，包含最终的隐藏状态、所有块的隐藏状态序列和所有块的注意力矩阵序列
        outputs = (hidden_states, all_hidden_states, all_attentions)

        return outputs


class FlaxGPTNeoModule(nn.Module):
    config: GPTNeoConfig
    dtype: jnp.dtype = jnp.float32

    # 初始化方法，设置模块内的各个子模块和变量
    def setup(self):
        # 设置词嵌入维度
        self.embed_dim = self.config.hidden_size
        # 使用正态分布初始化词嵌入矩阵
        embedding_init = jax.nn.initializers.normal(stddev=self.config.initializer_range)
        # 创建词嵌入层
        self.wte = nn.Embed(
            self.config.vocab_size,
            self.embed_dim,
            embedding_init=embedding_init,
        )
        # 创建位置嵌入层
        self.wpe = nn.Embed(
            self.config.max_position_embeddings,
            self.embed_dim,
            embedding_init=embedding_init,
        )
        # 创建 Dropout 层，用于随机断开输入的连接，防止过拟合
        self.dropout = nn.Dropout(rate=self.config.embed_dropout)
        # 创建 FlaxGPTNeoBlockCollection 实例，用于处理模型的多层块
        self.h = FlaxGPTNeoBlockCollection(self.config, dtype=self.dtype)
        # 创建 Layer Normalization 层，用于对最后的隐藏状态进行归一化处理
        self.ln_f = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)

    # 调用方法，接受输入并依次经过模型内的各个子模块处理
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
        ):
        # 将输入的词嵌入转换为特定数据类型的张量，并传递给词嵌入层处理
        input_embeds = self.wte(input_ids.astype("i4"))
        # 将位置编码转换为特定数据类型的张量，并传递给位置编码层处理
        position_embeds = self.wpe(position_ids.astype("i4"))

        # 将输入的词嵌入张量和位置编码张量相加得到隐藏状态张量
        hidden_states = input_embeds + position_embeds
        # 对隐藏状态张量应用dropout操作，用于模型训练中的随机失活
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)

        # 调用Transformer模型中的H层进行前向传播计算
        outputs = self.h(
            hidden_states,
            attention_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取H层输出的第一个张量，即隐藏状态张量
        hidden_states = outputs[0]
        # 对隐藏状态张量应用LN_F层，进行Layer Normalization处理
        hidden_states = self.ln_f(hidden_states)

        # 再次获取H层输出的第一个张量，即隐藏状态张量（重复的代码行）
        hidden_states = outputs[0]
        # 对隐藏状态张量应用LN_F层，进行Layer Normalization处理（重复的代码行）

        # 如果需要输出所有隐藏状态，则将当前隐藏状态张量添加到所有隐藏状态的列表中
        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)
            # 更新输出元组，将所有隐藏状态列表添加到输出元组中
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            # 更新输出元组，将当前隐藏状态张量添加到输出元组中
            outputs = (hidden_states,) + outputs[1:]

        # 如果不需要返回字典类型结果，则返回所有非空元素的元组
        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        # 返回模型输出结果的FlaxBaseModelOutput对象，包括最终隐藏状态、所有隐藏状态和注意力分数
        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=outputs[1],
            attentions=outputs[-1],
        )
# 添加起始文档字符串，描述该类是一个不带特定输出头部的原始隐藏状态的 GPTNeo 模型转换器。
@add_start_docstrings(
    "The bare GPTNeo Model transformer outputting raw hidden-states without any specific head on top.",
    GPT_NEO_START_DOCSTRING,
)
class FlaxGPTNeoModel(FlaxGPTNeoPreTrainedModel):
    # 模块类属性指定为 FlaxGPTNeoModule
    module_class = FlaxGPTNeoModule


# 添加调用样本文档字符串的方法，指定 FlaxGPTNeoModel 的一些文档化信息
append_call_sample_docstring(FlaxGPTNeoModel, _CHECKPOINT_FOR_DOC, FlaxBaseModelOutput, _CONFIG_FOR_DOC)


# 定义用于有因果语言建模的 GPTNeo 模型的模块
class FlaxGPTNeoForCausalLMModule(nn.Module):
    # 使用 GPTNeoConfig 作为配置，并指定默认数据类型为 jnp.float32
    config: GPTNeoConfig
    dtype: jnp.dtype = jnp.float32

    # 设置方法，在模块初始化时调用，初始化 transformer 和 lm_head
    def setup(self):
        self.transformer = FlaxGPTNeoModule(self.config, dtype=self.dtype)
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            dtype=self.dtype,
            # 使用正态分布初始化器初始化权重
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )

    # 调用方法，实现模块的前向传播
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
        # 使用 transformer 模型进行前向传播
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

        hidden_states = outputs[0]

        # 如果配置要求词嵌入权重共享，则共享 wte 参数的嵌入权重，并应用到 lm_head
        if self.config.tie_word_embeddings:
            shared_kernel = self.transformer.variables["params"]["wte"]["embedding"].T
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_kernel}}, hidden_states)
        else:
            # 否则直接将隐藏状态传递给 lm_head
            lm_logits = self.lm_head(hidden_states)

        # 如果不返回字典，则返回元组形式的结果
        if not return_dict:
            return (lm_logits,) + outputs[1:]

        # 返回具有自定义输出的 FlaxCausalLMOutput 对象
        return FlaxCausalLMOutput(logits=lm_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


# 添加起始文档字符串，描述带有语言建模头部的 GPTNeo 模型转换器
@add_start_docstrings(
    """
    The GPTNeo Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    GPT_NEO_START_DOCSTRING,
)
class FlaxGPTNeoForCausalLM(FlaxGPTNeoPreTrainedModel):
    module_class = FlaxGPTNeoForCausalLMModule
    # 为生成准备输入数据，接受输入的token ids、最大长度和可选的注意力掩码
    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = None):
        # 初始化缓存
        batch_size, seq_length = input_ids.shape

        # 使用输入的batch_size和max_length初始化缓存
        past_key_values = self.init_cache(batch_size, max_length)

        # 注意：通常需要在attention_mask的超出input_ids.shape[-1]和小于cache_length的位置放置0。
        # 但由于GPTNeo使用因果掩码，这些位置已经被掩盖了。
        # 因此，我们可以在这里创建一个静态的attention_mask，对编译更有效。
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")

        # 如果提供了attention_mask，则根据它计算position_ids
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            # 否则，使用默认的位置ids：从0到seq_length的广播
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        # 返回一个包含past_key_values、extended_attention_mask和position_ids的字典
        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    # 更新生成时的输入数据，接受模型输出和模型参数字典
    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        # 将模型输出中的past_key_values更新到模型参数字典中
        model_kwargs["past_key_values"] = model_outputs.past_key_values

        # 更新position_ids，只保留最后一个位置并加1
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1

        # 返回更新后的模型参数字典
        return model_kwargs
# 导入自动生成文档字符串所需的示例函数的模块，并添加到 FlaxGPTNeoForCausalLM 类的文档中
# _CHECKPOINT_FOR_DOC 是用于文档的检查点
# FlaxCausalLMOutput 是 FlaxGPTNeoForCausalLM 的输出类
# _CONFIG_FOR_DOC 是 FlaxGPTNeoForCausalLM 的配置类
append_call_sample_docstring(FlaxGPTNeoForCausalLM, _CHECKPOINT_FOR_DOC, FlaxCausalLMOutput, _CONFIG_FOR_DOC)
```