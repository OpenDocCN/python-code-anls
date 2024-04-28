# `.\models\gpt2\modeling_flax_gpt2.py`

```py
# 设置文件编码格式为utf-8
# 版权声明，指明代码的版权归属
# 根据 Apache 许可证 2.0 版本，可以在遵循许可协议的情况下使用此文件
# 可以在以下链接获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除依照适用法律要求或书面同意外，根据许可证分发的软件是在"如是"基础上分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关具体语言许可和限制的信息

# 导入所需的模块和类型提示
from typing import Any, Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax

# 从模型输出中导入需要的类
from ...modeling_flax_outputs import FlaxBaseModelOutputWithPastAndCrossAttentions, FlaxCausalLMOutputWithCrossAttentions
# 从工具模块中导入必要的类和函数
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring
# 从工具模块中导入日志记录器
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
# 导入 GPT2 配置
from .configuration_gpt2 import GPT2Config

# 获取 logger 实例
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置
_CHECKPOINT_FOR_DOC = "gpt2"
_CONFIG_FOR_DOC = "GPT2Config"

# GPT2 模型的起始文档字符串
GPT2_START_DOCSTRING = r"""

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
    # 参数：
    # config ([`GPT2Config`]): 包含模型所有参数的模型配置类。
    # 使用配置文件初始化不会加载与模型相关的权重，只会加载配置。参考 [`~FlaxPreTrainedModel.from_pretrained`] 方法加载模型权重。
    # dtype (`jax.numpy.dtype`, *optional*, 默认为 `jax.numpy.float32`):
    # 计算的数据类型。可以是 `jax.numpy.float32`、`jax.numpy.float16` (在 GPU 上) 和 `jax.numpy.bfloat16` (在 TPU 上) 中的一种。
    # 可用于启用 GPU 或 TPU 上的混合精度训练或半精度推理。如果指定，则所有计算将使用指定的 `dtype` 进行。
    # **注意这只指定了计算的数据类型，并不影响模型参数的数据类型。**
    # 如果要更改模型参数的数据类型，请参阅 [`~FlaxPreTrainedModel.to_fp16`] 和 [`~FlaxPreTrainedModel.to_bf16`]。
# GPT2 输入的文档字符串，用于说明输入参数的含义和格式
GPT2_INPUTS_DOCSTRING = r"""
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

# 创建一个 FlaxConv1D 类，它是 nn.Module 的子类
class FlaxConv1D(nn.Module):
    # 定义类属性 features，use_bias，dtype 和 precision
    features: int
    use_bias: bool = True
    dtype: Any = jnp.float32
    precision: Any = None

    # 对象的调用方法
    @nn.compact
    def __call__(self, inputs):
        # 将输入数据转换为 jnp 数组，使用类属性 dtype
        inputs = jnp.asarray(inputs, self.dtype)
        # 创建一个参数 kernel，使用 jax.nn.initializers.normal 初始化，shape 为 (features, inputs.shape[-1])
        kernel = self.param("kernel", jax.nn.initializers.normal(stddev=0.02), (self.features, inputs.shape[-1]))
        # 将 kernel 转置并转换为 jnp 数组
        kernel = jnp.asarray(kernel.transpose(), self.dtype)
        # 矩阵乘法运算，使用 lax.dot_general 函数
        y = lax.dot_general(inputs, kernel, (((inputs.ndim - 1,), (0,)), ((), ())), precision=self.precision)
        # 如果 use_bias 为 True，则添加偏置项
        if self.use_bias:
            # 创建一个参数 bias，使用 jax.nn.initializers.zeros 初始化，shape 为 (features,)
            bias = self.param("bias", jax.nn.initializers.zeros, (self.features,))
            # 将 bias 转换为 jnp 数组
            bias = jnp.asarray(bias, self.dtype)
            # 添加偏置项
            y = y + bias
        # 返回计算结果
        return y

# 创建一个 FlaxGPT2Attention 类，它是 nn.Module 的子类
class FlaxGPT2Attention(nn.Module):
    # 定义类属性 config、dtype、causal 和 is_cross_attention
    config: GPT2Config
    dtype: jnp.dtype = jnp.float32
    causal: bool = True
    is_cross_attention: bool = False
    def setup(self):
        config = self.config  # 将self.config赋值给config变量
        self.embed_dim = config.hidden_size  # 将config.hidden_size赋值给self.embed_dim
        self.num_heads = config.num_attention_heads  # 将config.num_attention_heads赋值给self.num_heads
        self.head_dim = self.embed_dim // self.num_heads  # 将self.embed_dim除以self.num_heads赋值给self.head_dim
    
        if self.is_cross_attention:  # 判断self.is_cross_attention是否为True
            self.c_attn = FlaxConv1D(2 * self.embed_dim, dtype=self.dtype)  # 创建一个FlaxConv1D对象赋值给self.c_attn变量，参数为2 * self.embed_dim和self.dtype
            self.q_attn = FlaxConv1D(self.embed_dim, dtype=self.dtype)  # 创建一个FlaxConv1D对象赋值给self.q_attn变量，参数为self.embed_dim和self.dtype
        else:
            self.c_attn = FlaxConv1D(3 * self.embed_dim, dtype=self.dtype)  # 创建一个FlaxConv1D对象赋值给self.c_attn变量，参数为3 * self.embed_dim和self.dtype
        self.c_proj = FlaxConv1D(self.embed_dim, dtype=self.dtype)  # 创建一个FlaxConv1D对象赋值给self.c_proj变量，参数为self.embed_dim和self.dtype
    
        self.resid_dropout = nn.Dropout(rate=config.resid_pdrop)  # 创建一个Dropout对象赋值给self.resid_dropout变量，参数为config.resid_pdrop
    
        if self.causal:  # 判断self.causal是否为True
            self.causal_mask = make_causal_mask(
                jnp.ones((1, config.max_position_embeddings), dtype="bool"), dtype="bool"
            )  # 调用make_causal_mask方法，并将返回值赋值给self.causal_mask，参数为一个由1组成的形状为(1, config.max_position_embeddings)的JAX数组和"dtype='bool'"
    
    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))
        # 对hidden_states进行重构，形状为hidden_states.shape[:2] + (self.num_heads, self.head_dim)
    
    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))
        # 对hidden_states进行重构，形状为hidden_states.shape[:2] + (self.embed_dim,)
    
    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slighly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # 检测是否初始化，如果缓存数据不存在则表示已初始化
        is_initialized = self.has_variable("cache", "cached_key")
        # 从缓存中获取已存在的key值，如果不存在则初始化为0数组
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        # 从缓存中获取已存在的value值，如果不存在则初始化为0数组
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        # 获取缓存索引，如果不存在则初始化为0
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            # 获取batch维度等相关维度信息
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # 用新的1d空间切片更新key、value缓存
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            # 更新缓存中key值和value值
            cached_key.value = key
            cached_value.value = value
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # 为缓存的解码器自注意力创建因果掩码：我们的单个查询位置应仅关注已生成并缓存的键位置，而不是剩余的零元素。
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)
        return key, value, attention_mask

    def __call__(
        self,
        hidden_states,
        key_value_states: Optional[jnp.ndarray] = None,
        attention_mask=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
# 定义一个名为FlaxGPT2MLP的类，继承自nn.Module
class FlaxGPT2MLP(nn.Module):
    # 声明类变量config，类型为GPT2Config
    config: GPT2Config
    # 声明类变量intermediate_size，类型为int
    intermediate_size: int
    # 声明类变量dtype，类型为jnp.dtype，默认值为jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 定义setup方法
    def setup(self):
        # 定义变量embed_dim，赋值为self.config.hidden_size
        embed_dim = self.config.hidden_size
        # 初始化c_fc为FlaxConv1D对象，传入参数intermediate_size和dtype
        self.c_fc = FlaxConv1D(self.intermediate_size, dtype=self.dtype)
        # 初始化c_proj为FlaxConv1D对象，传入参数embed_dim和dtype
        self.c_proj = FlaxConv1D(embed_dim, dtype=self.dtype)
        # 初始化act为ACT2FN字典中self.config.activation_function对应的值
        self.act = ACT2FN[self.config.activation_function]
        # 初始化dropout为nn.Dropout对象，传入参数rate为self.config.resid_pdrop
        self.dropout = nn.Dropout(rate=self.config.resid_pdrop)

    # 定义__call__方法
    def __call__(self, hidden_states, deterministic: bool = True):
        # 对hidden_states进行全连接操作，并赋值给hidden_states
        hidden_states = self.c_fc(hidden_states)
        # 对hidden_states进行激活操作，并赋值给hidden_states
        hidden_states = self.act(hidden_states)
        # 对hidden_states进行全连接操作，并赋值给hidden_states
        hidden_states = self.c_proj(hidden_states)
        # 对hidden_states进行dropout操作，并赋值给hidden_states
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 返回hidden_states
        return hidden_states


# 定义一个名为FlaxGPT2Block的类，继承自nn.Module
class FlaxGPT2Block(nn.Module):
    # 声明类变量config，类型为GPT2Config
    config: GPT2Config
    # 声明类变量dtype，类型为jnp.dtype，默认值为jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 定义setup方法
    def setup(self):
        # 初始化hidden_size为self.config.hidden_size
        hidden_size = self.config.hidden_size
        # 定义inner_dim，如果self.config.n_inner不为None则为self.config.n_inner，否则为4 * hidden_size
        inner_dim = self.config.n_inner if self.config.n_inner is not None else 4 * hidden_size

        # 初始化ln_1为nn.LayerNorm对象，传入参数epsilon为self.config.layer_norm_epsilon和dtype
        self.ln_1 = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)
        # 初始化attn为FlaxGPT2Attention对象，传入参数config和dtype
        self.attn = FlaxGPT2Attention(self.config, dtype=self.dtype)
        # 初始化ln_2为nn.LayerNorm对象，传入参数epsilon为self.config.layer_norm_epsilon和dtype
        self.ln_2 = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)

        # 如果self.config.add_cross_attention为True
        if self.config.add_cross_attention:
            # 初始化crossattention为FlaxGPT2Attention对象，传入参数config、dtype、causal为False、is_cross_attention为True
            self.crossattention = FlaxGPT2Attention(
                config=self.config, dtype=self.dtype, causal=False, is_cross_attention=True
            )
            # 初始化ln_cross_attn为nn.LayerNorm对象，传入参数epsilon为self.config.layer_norm_epsilon和dtype
            self.ln_cross_attn = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)

        # 初始化mlp为FlaxGPT2MLP对象，传入参数self.config、inner_dim和dtype
        self.mlp = FlaxGPT2MLP(self.config, inner_dim, dtype=self.dtype)

    # 定义__call__方法
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):
        # 保存残差连接的输入状态
        residual = hidden_states
        # LayerNormalization，对隐藏状态进行归一化处理
        hidden_states = self.ln_1(hidden_states)
        # 调用自注意力机制进行注意力计算
        attn_outputs = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
        )
        # 获取自注意力计算的输出
        attn_output = attn_outputs[0]  # output_attn: a, (attentions)
        # 提取额外的输出（如果需要）
        outputs = attn_outputs[1:]
        # 将自注意力计算的输出与残差连接的输入相加
        hidden_states = attn_output + residual

        # 交叉注意力块
        if encoder_hidden_states is not None:
            # 如果存在编码器隐藏状态，则添加交叉注意力块
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            # 保存残差连接的输入状态
            residual = hidden_states
            # LayerNormalization，对隐藏状态进行归一化处理
            hidden_states = self.ln_cross_attn(hidden_states)
            # 调用交叉注意力机制进行注意力计算
            cross_attn_outputs = self.crossattention(
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                deterministic=deterministic,
                output_attentions=output_attentions,
            )
            # 获取交叉注意力计算的输出
            attn_output = cross_attn_outputs[0]
            # 将残差连接的输入与交叉注意力计算的输出相加
            hidden_states = residual + attn_output
            # 如果需要，添加交叉注意力的输出
            outputs = outputs + cross_attn_outputs[1:]  # add cross attentions if we output attention weights

        # 保存残差连接的输入状态
        residual = hidden_states
        # LayerNormalization，对隐藏状态进行归一化处理
        hidden_states = self.ln_2(hidden_states)
        # 通过多层感知机进行前馈传播
        feed_forward_hidden_states = self.mlp(hidden_states, deterministic=deterministic)
        # 将残差连接的输入与前馈传播的输出相加
        hidden_states = residual + feed_forward_hidden_states

        # 将结果打包为元组
        outputs = (hidden_states,) + outputs

        # 返回结果
        return outputs
# 创建一个名为FlaxGPT2PreTrainedModel的类，继承自FlaxPreTrainedModel类
class FlaxGPT2PreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定config_class为GPT2Config
    config_class = GPT2Config
    # 指定base_model_prefix为"transformer"
    base_model_prefix = "transformer"
    # 设置module_class为None
    module_class: nn.Module = None

    def __init__(
        self,
        config: GPT2Config,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        # 使用module_class创建一个模块实例
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 调用父类的初始化方法
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    # 初始化参数权重
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 如果模型要求添加交叉注意力
        if self.config.add_cross_attention:
            encoder_hidden_states = jnp.zeros(input_shape + (self.config.n_embd,))
            encoder_attention_mask = attention_mask
            # 使用module的init方法初始化模块
            module_init_outputs = self.module.init(
                rngs,
                input_ids,
                attention_mask,
                position_ids,
                encoder_hidden_states,
                encoder_attention_mask,
                return_dict=False,
            )
        else:
            # 使用module的init方法初始化模块
            module_init_outputs = self.module.init(rngs, input_ids, attention_mask, position_ids, return_dict=False)

        random_params = module_init_outputs["params"]

        # 如果传入了params，则将其与随机初始化的参数进行融合
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
    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                用于快速自回归解码的批大小。定义了初始化缓存的批大小。
            max_length (`int`):
                自回归解码的最大可能长度。定义了初始化缓存的序列长度。
        """
        # 初始化用于检索缓存的输入变量
        input_ids = jnp.ones((batch_size, max_length))
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        # 使用输入变量初始化模型的变量，并指定初始化缓存的参数为True
        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        # 返回未冻结的缓存
        return unfreeze(init_variables["cache"])

    # 调用模型的正向传播函数，添加了GPT2输入说明的文档字符串
    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
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
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 检查是否返回字典格式结果
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 如果存在编码器的隐藏状态但不存在编码器的注意力掩码，则创建全为1的注意力掩码
        if encoder_hidden_states is not None and encoder_attention_mask is None:
            batch_size, sequence_length = encoder_hidden_states.shape[:2]
            encoder_attention_mask = jnp.ones((batch_size, sequence_length))

        # 获取输入数据的批量大小和序列长度
        batch_size, sequence_length = input_ids.shape

        # 如果未提供位置编码，则根据序列长度创建默认的位置编码
        if position_ids is None:
            if past_key_values is not None:
                raise ValueError("Make sure to provide `position_ids` when passing `past_key_values`.")
            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # 如果未提供注意力掩码，则创建全为1的注意力掩码
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))

        # 处理任何需要的伪随机数生成器
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        # 构建输入参数字典
        inputs = {"params": params or self.params}

        # 如果传递了过去的键值对，则传递一个私有标志init_cache以确保使用缓存。必须确保缓存标记为可变，以便FlaxGPT2Attention模块可以更改它
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        # 应用模型的正向传播
        outputs = self.module.apply(
            inputs,
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            encoder_hidden_states,
            encoder_attention_mask,
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

        # 返回模型输出
        return outputs
# 定义一个名为 "FlaxGPT2BlockCollection" 的类，继承自 nn.Module
class FlaxGPT2BlockCollection(nn.Module):
    # 声明一个名为 "config" 的属性，类型为 GPT2Config
    config: GPT2Config
    # 声明一个名为 "dtype" 的属性，类型为 jnp.dtype，默认值为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 定义一个名为 "setup" 的方法
    def setup(self):
        # 创建一个由多个 FlaxGPT2Block 对象组成的列表，列表长度为 self.config.num_hidden_layers
        self.blocks = [
            FlaxGPT2Block(self.config, name=str(i), dtype=self.dtype) for i in range(self.config.num_hidden_layers)
        ]

    # 定义一个名为 "__call__" 的方法，用于调用对象实例
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 根据 output_attentions 决定是否创建一个空元组，用于存储注意力矩阵
        all_attentions = () if output_attentions else None
        # 根据 output_hidden_states 决定是否创建一个空元组，用于存储隐藏状态
        all_hidden_states = () if output_hidden_states else None
        # 根据 output_attentions 和 encoder_hidden_states 决定是否创建一个空元组，用于存储交叉注意力矩阵
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        # 遍历 self.blocks 中的每一个 FlaxGPT2Block 对象
        for block in self.blocks:
            # 根据 output_hidden_states 决定是否将 hidden_states 添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # 调用 block 对象的方法，传入相应参数，并获取返回的结果
            layer_outputs = block(
                hidden_states,
                attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                deterministic=deterministic,
                init_cache=init_cache,
                output_attentions=output_attentions,
            )
            # 更新 hidden_states 为 block 方法返回的结果中的第一个元素
            hidden_states = layer_outputs[0]

            # 根据 output_attentions 决定是否将 layer_outputs[1] 添加到 all_attentions 中
            if output_attentions:
                all_attentions += (layer_outputs[1],)

                # 如果 encoder_hidden_states 不为 None，则将 layer_outputs[2] 添加到 all_cross_attentions 中
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # 创建一个元组，包含 hidden_states, all_hidden_states, all_attentions, all_cross_attentions，可能包含 None 值
        outputs = (hidden_states, all_hidden_states, all_attentions, all_cross_attentions)

        # 返回 outputs
        return outputs


# 定义一个名为 "FlaxGPT2Module" 的类，继承自 nn.Module
class FlaxGPT2Module(nn.Module):
    # 声明一个名为 "config" 的属性，类型为 GPT2Config
    config: GPT2Config
    # 声明一个名为 "dtype" 的属性，类型为 jnp.dtype，默认值为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 定义一个名为 "setup" 的方法
    def setup(self):
        # 声明一个名为 "embed_dim" 的属性，值为 self.config.hidden_size
        self.embed_dim = self.config.hidden_size

        # 创建一个 nn.Embed 对象，用于词嵌入，传入相应参数
        self.wte = nn.Embed(
            self.config.vocab_size,
            self.embed_dim,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        # 创建一个 nn.Embed 对象，用于位置嵌入，传入相应参数
        self.wpe = nn.Embed(
            self.config.max_position_embeddings,
            self.embed_dim,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        # 创建一个 nn.Dropout 对象，用于丢弃率为 self.config.embd_pdrop
        self.dropout = nn.Dropout(rate=self.config.embd_pdrop)
        # 创建一个 FlaxGPT2BlockCollection 对象，传入相应参数
        self.h = FlaxGPT2BlockCollection(self.config, dtype=self.dtype)
        # 创建一个 nn.LayerNorm 对象，用于 LayerNormalization，传入相应参数
        self.ln_f = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)
    # 在调用实例时执行以下操作
    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        deterministic=True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 对输入的 token IDs 进行词嵌入
        input_embeds = self.wte(input_ids.astype("i4"))
        # 对位置 IDs 进行位置嵌入
        position_embeds = self.wpe(position_ids.astype("i4"))

        # 将输入嵌入和位置嵌入相加得到隐藏状态
        hidden_states = input_embeds + position_embeds
        # 对隐藏状态进行丢弃正则化
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)

        # 使用 Transformer 头部处理隐藏状态
        outputs = self.h(
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

        # 获取处理后的隐藏状态
        hidden_states = outputs[0]
        # 对输出的隐藏状态进行 Layer Normalization
        hidden_states = self.ln_f(hidden_states)

        # 如果需要输出隐藏状态，则更新输出
        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            outputs = (hidden_states,) + outputs[1:]

        # 如果不需要返回字典，则返回非空项的元组
        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        # 返回带有过去及交叉注意力信息的基础模型输出
        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=outputs[1],
            attentions=outputs[2],
            cross_attentions=outputs[3],
        )
# 导入函数装饰器，用于添加初始文档字符串（docstring）
@add_start_docstrings(
    "The bare GPT2 Model transformer outputting raw hidden-states without any specific head on top.",  # 添加模型描述
    GPT2_START_DOCSTRING,  # 导入 GPT2 模型的初始文档字符串
)
# 定义 FlaxGPT2Model 类，继承自 FlaxGPT2PreTrainedModel 类
class FlaxGPT2Model(FlaxGPT2PreTrainedModel):
    # 设置模块类为 FlaxGPT2Module
    module_class = FlaxGPT2Module


# 添加调用示例的文档字符串
append_call_sample_docstring(
    FlaxGPT2Model,  # 添加文档字符串的目标类
    _CHECKPOINT_FOR_DOC,  # 检查点描述
    FlaxBaseModelOutputWithPastAndCrossAttentions,  # 输出格式
    _CONFIG_FOR_DOC,  # 配置描述
)


# 定义 FlaxGPT2LMHeadModule 类，继承自 nn.Module
class FlaxGPT2LMHeadModule(nn.Module):
    config: GPT2Config  # 定义配置属性
    dtype: jnp.dtype = jnp.float32  # 定义数据类型，默认为 jnp.float32

    # 模块设置函数
    def setup(self):
        # 创建 GPT2Transformer 实例
        self.transformer = FlaxGPT2Module(self.config, dtype=self.dtype)
        # 创建语言模型头部，即全连接层，输出大小为词汇表大小，不使用偏置，初始化方式为正态分布
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )

    # 模块调用函数
    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 调用 Transformer 模块处理输入数据
        outputs = self.transformer(
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

        # 获取隐藏状态
        hidden_states = outputs[0]

        # 如果词嵌入被绑定
        if self.config.tie_word_embeddings:
            # 获取共享的词嵌入矩阵
            shared_kernel = self.transformer.variables["params"]["wte"]["embedding"].T
            # 应用语言模型头部到隐藏状态上
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_kernel}}, hidden_states)
        else:
            # 否则直接应用语言模型头部到隐藏状态上
            lm_logits = self.lm_head(hidden_states)

        # 如果不需要返回字典
        if not return_dict:
            # 返回 logits 和其他输出
            return (lm_logits,) + outputs[1:]

        # 返回带有交叉注意力的因果语言模型输出
        return FlaxCausalLMOutputWithCrossAttentions(
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


# 添加文档字符串描述
@add_start_docstrings(
    """
    The GPT2 Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,  # 模型描述
    GPT2_START_DOCSTRING,  # GPT2 初始文档字符串
)
# 定义 FlaxGPT2LMHeadModel 类，继承自 FlaxGPT2PreTrainedModel 类
class FlaxGPT2LMHeadModel(FlaxGPT2PreTrainedModel):
    # 设置模块类为 FlaxGPT2LMHeadModule
    module_class = FlaxGPT2LMHeadModule
    # 为生成器准备输入数据，包括输入的词索引、生成序列的最大长度以及可选的注意力掩码
    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = None):
        # 初始化缓存，获取批量大小和序列长度
        batch_size, seq_length = input_ids.shape

        # 初始化缓存键值对，用于存储过去的注意力键值对
        past_key_values = self.init_cache(batch_size, max_length)
        
        # 注意：通常需要在注意力掩码中放入 0，对于 x > input_ids.shape[-1] 和 x < cache_length。
        # 但是由于 GPT2 使用因果遮罩，这些位置已经被遮蔽了。
        # 因此，我们可以在这里创建一个静态的注意力掩码，这样更有效率地编译
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        
        # 如果提供了注意力掩码，则更新位置索引和扩展注意力掩码
        if attention_mask is not None:
            # 计算位置索引
            position_ids = attention_mask.cumsum(axis=-1) - 1
            # 使用动态更新切片，将注意力掩码应用到扩展的注意力掩码中
            extended_attention_mask = lax.dynamic_update_slice(
                extended_attention_mask, attention_mask.astype("i4"), (0, 0)
            )
        else:
            # 如果没有提供注意力掩码，则创建一个广播到适当形状的位置索引
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        # 返回准备好的输入数据，包括过去的键值对、注意力掩码和位置索引
        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    # 更新生成器的输入数据
    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        # 更新过去的键值对和位置索引
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs
# 调用函数append_call_sample_docstring，传入参数FlaxGPT2LMHeadModel, _CHECKPOINT_FOR_DOC, FlaxCausalLMOutputWithCrossAttentions, _CONFIG_FOR_DOC
append_call_sample_docstring(
    FlaxGPT2LMHeadModel,  # 传入FlaxGPT2LMHeadModel参数
    _CHECKPOINT_FOR_DOC,   # 传入_CHECKPOINT_FOR_DOC参数
    FlaxCausalLMOutputWithCrossAttentions,   # 传入FlaxCausalLMOutputWithCrossAttentions参数
    _CONFIG_FOR_DOC,   # 传入_CONFIG_FOR_DOC参数
)
```