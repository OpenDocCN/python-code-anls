# `.\models\gpt2\modeling_flax_gpt2.py`

```py
# 导入必要的库和模块，包括类型提示
from typing import Any, Optional, Tuple

# 导入 Flax 和 JAX 的相关模块
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax

# 导入模型输出和工具函数
from ...modeling_flax_outputs import (
    FlaxBaseModelOutputWithPastAndCrossAttentions,
    FlaxCausalLMOutputWithCrossAttentions,
)
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging

# 导入 GPT2 配置
from .configuration_gpt2 import GPT2Config

# 获取日志记录器
logger = logging.get_logger(__name__)

# 模型使用的预训练模型检查点和配置名称
_CHECKPOINT_FOR_DOC = "openai-community/gpt2"
_CONFIG_FOR_DOC = "GPT2Config"

# GPT2 模型的开始文档字符串
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
    # Parameters参数描述：
    # config ([`GPT2Config`]): 模型配置类，包含模型的所有参数。
    #     通过配置文件初始化不会加载与模型相关的权重，只加载配置。
    #     可查看[`~FlaxPreTrainedModel.from_pretrained`]方法以加载模型权重。
    # dtype (`jax.numpy.dtype`, *optional*, 默认为 `jax.numpy.float32`):
    #     计算时的数据类型。可以是`jax.numpy.float32`、`jax.numpy.float16`（在GPU上）、`jax.numpy.bfloat16`（在TPU上）之一。
    #     可用于在GPU或TPU上启用混合精度训练或半精度推断。如果指定了dtype，则所有计算将使用给定的dtype。
    #
    #     **请注意，这仅指定计算的dtype，并不影响模型参数的dtype。**
    #
    #     如果想要更改模型参数的dtype，请参阅[`~FlaxPreTrainedModel.to_fp16`]和[`~FlaxPreTrainedModel.to_bf16`]。
"""
Implementing FlaxGPT2Attention module for GPT-2 model.

This module handles the attention mechanism used in GPT-2, including optional cross-attention.
"""

class FlaxGPT2Attention(nn.Module):
    # GPT-2模型的配置，包括注意事项和交叉注意事项的配置
    config: GPT2Config
    # 数据类型，默认为32位浮点数
    dtype: jnp.dtype = jnp.float32
    # 是否是因果注意力机制，默认为True
    causal: bool = True
    # 是否是交叉注意力机制，默认为False
    is_cross_attention: bool = False

    @nn.compact
    def __call__(self, inputs):
        # 将输入转换为指定的数据类型
        inputs = jnp.asarray(inputs, self.dtype)
        # 初始化注意力层的权重矩阵，形状为 (features, input_shape[-1])
        kernel = self.param("kernel", jax.nn.initializers.normal(stddev=0.02), (self.features, inputs.shape[-1]))
        kernel = jnp.asarray(kernel.transpose(), self.dtype)
        # 使用 dot_general 函数计算输入和权重的乘积
        y = lax.dot_general(inputs, kernel, (((inputs.ndim - 1,), (0,)), ((), ())), precision=self.precision)
        # 如果设置了使用偏置项
        if self.use_bias:
            # 初始化偏置项，形状为 (features,)
            bias = self.param("bias", jax.nn.initializers.zeros, (self.features,))
            bias = jnp.asarray(bias, self.dtype)
            # 在乘积结果上加上偏置项
            y = y + bias
        # 返回计算结果
        return y
    # 定义模型初始化设置方法
    def setup(self):
        # 从配置中获取参数
        config = self.config
        # 设置嵌入维度为隐藏层大小
        self.embed_dim = config.hidden_size
        # 设置注意力头数为配置中的头数
        self.num_heads = config.num_attention_heads
        # 计算每个头的维度
        self.head_dim = self.embed_dim // self.num_heads

        # 如果是跨注意力机制
        if self.is_cross_attention:
            # 使用两倍的嵌入维度创建跨注意力层
            self.c_attn = FlaxConv1D(2 * self.embed_dim, dtype=self.dtype)
            # 使用单个嵌入维度创建查询注意力层
            self.q_attn = FlaxConv1D(self.embed_dim, dtype=self.dtype)
        else:
            # 使用三倍的嵌入维度创建自注意力层
            self.c_attn = FlaxConv1D(3 * self.embed_dim, dtype=self.dtype)
        
        # 使用单个嵌入维度创建投影层
        self.c_proj = FlaxConv1D(self.embed_dim, dtype=self.dtype)

        # 定义残差连接的 Dropout 层，使用配置中的残差 Dropout 概率
        self.resid_dropout = nn.Dropout(rate=config.resid_pdrop)

        # 如果是因果注意力模型
        if self.causal:
            # 创建一个因果掩码，形状为 (1, 最大位置编码数)，类型为布尔型
            self.causal_mask = make_causal_mask(
                jnp.ones((1, config.max_position_embeddings), dtype="bool"), dtype="bool"
            )

    # 将隐藏状态按头分割方法
    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    # 将分割后的头合并为隐藏状态方法
    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    # 使用 nn.compact 装饰器定义紧凑的神经网络结构
    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slightly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # 检测是否通过检查"cache"中的"cached_key"变量来初始化
        is_initialized = self.has_variable("cache", "cached_key")
        # 如果未初始化，则将"cached_key"初始化为形状和类型与key相同的全零张量
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        # 如果未初始化，则将"cached_value"初始化为形状和类型与value相同的全零张量
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        # 如果未初始化，则将"cache_index"初始化为整数0
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            # 提取批次维度、序列最大长度、头数和每头深度，从现有的缓存中
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # 使用新的1维空间切片更新key和value缓存
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            # 更新缓存中的key和value
            cached_key.value = key
            cached_value.value = value
            # 更新缓存索引，增加已更新的缓存向量数量
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # 用于缓存解码器自注意力的因果掩码：我们的单个查询位置只能关注已生成并缓存的key位置，而不是剩余的零元素。
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            # 合并因果掩码和输入的注意力掩码
            attention_mask = combine_masks(pad_mask, attention_mask)
        # 返回更新后的key、value和注意力掩码
        return key, value, attention_mask
# 定义一个自定义的 FlaxGPT2MLP 类，继承自 nn.Module
class FlaxGPT2MLP(nn.Module):
    # 类型注解：配置信息为 GPT2Config 类型
    config: GPT2Config
    # 中间层的大小
    intermediate_size: int
    # 数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 初始化函数，设置网络结构
    def setup(self):
        # 嵌入维度等于配置中的隐藏大小
        embed_dim = self.config.hidden_size
        # 定义一个一维卷积层，输出大小为 intermediate_size
        self.c_fc = FlaxConv1D(self.intermediate_size, dtype=self.dtype)
        # 定义另一个一维卷积层，输出大小为 embed_dim
        self.c_proj = FlaxConv1D(embed_dim, dtype=self.dtype)
        # 激活函数，根据配置中的激活函数类型选择对应的函数
        self.act = ACT2FN[self.config.activation_function]
        # Dropout 层，根据配置中的 residual dropout 率初始化
        self.dropout = nn.Dropout(rate=self.config.resid_pdrop)

    # 前向传播函数，处理隐藏状态
    def __call__(self, hidden_states, deterministic: bool = True):
        # 先通过第一个卷积层处理隐藏状态
        hidden_states = self.c_fc(hidden_states)
        # 使用配置中指定的激活函数处理卷积层输出
        hidden_states = self.act(hidden_states)
        # 再通过第二个卷积层处理激活后的隐藏状态
        hidden_states = self.c_proj(hidden_states)
        # 使用 Dropout 层对卷积层输出进行随机失活处理
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 返回处理后的隐藏状态
        return hidden_states


# 定义一个自定义的 FlaxGPT2Block 类，继承自 nn.Module
class FlaxGPT2Block(nn.Module):
    # 类型注解：配置信息为 GPT2Config 类型
    config: GPT2Config
    # 数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 初始化函数，设置网络结构
    def setup(self):
        # 隐藏层大小等于配置中的隐藏大小
        hidden_size = self.config.hidden_size
        # 内部维度为配置中指定的内部层大小，如果未指定，则为 4 倍的隐藏层大小
        inner_dim = self.config.n_inner if self.config.n_inner is not None else 4 * hidden_size

        # 第一个 LayerNorm 层，使用配置中的层标准化 epsilon 值进行初始化
        self.ln_1 = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)
        # 自定义的 GPT2Attention 层，使用配置信息和数据类型进行初始化
        self.attn = FlaxGPT2Attention(self.config, dtype=self.dtype)
        # 第二个 LayerNorm 层，同样使用配置中的层标准化 epsilon 值进行初始化
        self.ln_2 = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)

        # 如果配置中包含跨注意力机制
        if self.config.add_cross_attention:
            # 初始化一个用于跨注意力的 GPT2Attention 层，关闭因果性，设为跨注意力
            self.crossattention = FlaxGPT2Attention(
                config=self.config, dtype=self.dtype, causal=False, is_cross_attention=True
            )
            # 第三个 LayerNorm 层，使用配置中的层标准化 epsilon 值进行初始化
            self.ln_cross_attn = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)

        # 自定义的 GPT2MLP 类，使用配置信息、内部维度和数据类型进行初始化
        self.mlp = FlaxGPT2MLP(self.config, inner_dim, dtype=self.dtype)

    # 前向传播函数，处理隐藏状态和注意力机制
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
            residual = hidden_states  # 保存当前隐藏状态作为残差连接的起点
            hidden_states = self.ln_1(hidden_states)  # LayerNormalization 1：对隐藏状态进行归一化

            attn_outputs = self.attn(
                hidden_states,
                attention_mask=attention_mask,
                deterministic=deterministic,
                init_cache=init_cache,
                output_attentions=output_attentions,
            )
            # 获取自注意力机制的输出
            attn_output = attn_outputs[0]  # attn_output: 自注意力机制的输出，第一个元素
            outputs = attn_outputs[1:]  # outputs: 自注意力机制的其他输出，如注意力权重等

            # 残差连接
            hidden_states = attn_output + residual

            # Cross-Attention Block
            if encoder_hidden_states is not None:
                # 添加交叉注意力块
                if not hasattr(self, "crossattention"):
                    raise ValueError(
                        f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                        "cross-attention layers by setting `config.add_cross_attention=True`"
                    )

                residual = hidden_states  # 保存当前隐藏状态作为残差连接的起点
                hidden_states = self.ln_cross_attn(hidden_states)  # LayerNormalization 2：对隐藏状态进行归一化

                cross_attn_outputs = self.crossattention(
                    hidden_states,
                    key_value_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    deterministic=deterministic,
                    output_attentions=output_attentions,
                )
                # 获取交叉注意力机制的输出
                attn_output = cross_attn_outputs[0]
                # 残差连接
                hidden_states = residual + attn_output
                # 添加交叉注意力的输出到总输出中（如果输出注意力权重的话）
                outputs = outputs + cross_attn_outputs[1:]

            residual = hidden_states  # 保存当前隐藏状态作为残差连接的起点
            hidden_states = self.ln_2(hidden_states)  # LayerNormalization 3：对隐藏状态进行归一化

            feed_forward_hidden_states = self.mlp(hidden_states, deterministic=deterministic)
            # 残差连接
            hidden_states = residual + feed_forward_hidden_states

            outputs = (hidden_states,) + outputs  # 将最终隐藏状态和所有输出组成元组作为模块的最终输出

            return outputs
# 定义一个继承自FlaxPreTrainedModel的抽象类，用于处理权重初始化、预训练模型的下载和加载接口
class FlaxGPT2PreTrainedModel(FlaxPreTrainedModel):
    # 配置类，指定为GPT2Config
    config_class = GPT2Config
    # 基础模型前缀，指定为"transformer"
    base_model_prefix = "transformer"
    # 模块类，暂未指定
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
        # 使用给定的配置和参数初始化模块
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 调用父类初始化方法，传入配置、模块、输入形状、种子、数据类型和是否初始化的标志
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量
        input_ids = jnp.zeros(input_shape, dtype="i4")
        # 创建与input_ids相同形状的全1张量作为注意力掩码
        attention_mask = jnp.ones_like(input_ids)
        # 根据input_ids的形状广播生成位置编码张量
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        # 划分随机数种子rng，用于参数和dropout
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        if self.config.add_cross_attention:
            # 如果配置需要添加交叉注意力，则初始化编码器隐藏状态为零张量，注意力掩码与attention_mask相同
            encoder_hidden_states = jnp.zeros(input_shape + (self.config.n_embd,))
            encoder_attention_mask = attention_mask
            # 使用模块的初始化方法初始化模型参数和其他输出
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
            # 否则，只使用input_ids、attention_mask和position_ids初始化模型参数
            module_init_outputs = self.module.init(rngs, input_ids, attention_mask, position_ids, return_dict=False)

        # 获取随机初始化的参数
        random_params = module_init_outputs["params"]

        if params is not None:
            # 如果给定了预训练参数params，则将随机初始化的参数与params进行扁平化和解冻后的比较，处理丢失的键
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            # 返回冻结和解扁平化的参数字典
            return freeze(unflatten_dict(params))
        else:
            # 否则，直接返回随机初始化的参数字典
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
        # 初始化输入变量以获取缓存
        # 创建一个 batch_size x max_length 大小的全为1的矩阵作为输入ID
        input_ids = jnp.ones((batch_size, max_length))
        # 根据 input_ids 创建与之形状相同的全为1的注意力掩码
        attention_mask = jnp.ones_like(input_ids)
        # 根据 input_ids 的形状广播创建位置ID，形状为 input_ids.shape
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        # 使用模型的初始化方法初始化变量，包括输入ID、注意力掩码、位置ID，并设置初始化缓存为True
        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        # 返回解除冻结后的缓存部分
        return unfreeze(init_variables["cache"])

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
        ):
            # 如果未显式提供 `output_attentions`，则使用配置中的设定
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            # 如果未显式提供 `output_hidden_states`，则使用配置中的设定
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            # 如果未显式提供 `return_dict`，则使用配置中的设定
            return_dict = return_dict if return_dict is not None else self.config.return_dict

            # 如果提供了 `encoder_hidden_states` 但未提供 `encoder_attention_mask`，则创建一个全为1的注意力掩码
            if encoder_hidden_states is not None and encoder_attention_mask is None:
                batch_size, sequence_length = encoder_hidden_states.shape[:2]
                encoder_attention_mask = jnp.ones((batch_size, sequence_length))

            # 获取输入张量的批量大小和序列长度
            batch_size, sequence_length = input_ids.shape

            # 如果未提供 `position_ids`
            if position_ids is None:
                # 如果提供了 `past_key_values` 但未提供 `position_ids`，则引发错误
                if past_key_values is not None:
                    raise ValueError("Make sure to provide `position_ids` when passing `past_key_values`.")
                # 使用序列长度创建广播后的位置编码
                position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

            # 如果未提供 `attention_mask`，则创建一个全为1的注意力掩码
            if attention_mask is None:
                attention_mask = jnp.ones((batch_size, sequence_length))

            # 处理任何需要的伪随机数生成器
            rngs = {}
            if dropout_rng is not None:
                rngs["dropout"] = dropout_rng

            # 准备输入参数字典
            inputs = {"params": params or self.params}

            # 如果提供了 `past_key_values`，则将其作为缓存传递给模块，同时确保缓存是可变的
            if past_key_values:
                inputs["cache"] = past_key_values
                mutable = ["cache"]
            else:
                mutable = False

            # 应用模块的前向传播函数，计算输出
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

            # 如果提供了 `past_key_values` 并且设置了 `return_dict`，则将更新后的缓存添加到模型输出中
            if past_key_values is not None and return_dict:
                outputs, past_key_values = outputs
                outputs["past_key_values"] = unfreeze(past_key_values["cache"])
                return outputs
            # 如果提供了 `past_key_values` 但未设置 `return_dict`，则更新模型输出
            elif past_key_values is not None and not return_dict:
                outputs, past_key_values = outputs
                outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

            # 返回模型输出
            return outputs
class FlaxGPT2BlockCollection(nn.Module):
    config: GPT2Config
    dtype: jnp.dtype = jnp.float32

    # 初始化模块，设置隐藏层块的集合
    def setup(self):
        # 创建一组 GPT2Block 对象，根据给定的层数配置
        self.blocks = [
            FlaxGPT2Block(self.config, name=str(i), dtype=self.dtype) for i in range(self.config.num_hidden_layers)
        ]

    # 调用模块的方法，处理输入的隐藏状态并返回输出
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
        # 初始化空的输出列表，根据需要存储注意力、隐藏状态及交叉注意力
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        # 遍历每个隐藏层块并处理隐藏状态
        for block in self.blocks:
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到列表中
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # 调用当前块的处理方法，获取块的输出
            layer_outputs = block(
                hidden_states,
                attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                deterministic=deterministic,
                init_cache=init_cache,
                output_attentions=output_attentions,
            )
            # 更新当前隐藏状态为当前块的输出的第一个元素（即隐藏状态）
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力，则将当前块的注意力添加到列表中
            if output_attentions:
                all_attentions += (layer_outputs[1],)

                # 如果存在编码器的隐藏状态，则将当前块的交叉注意力添加到列表中
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # 返回模块的输出，包括隐藏状态、所有隐藏状态、所有注意力和所有交叉注意力
        # 注意：这里可能包含 `None` 值，`FlaxGPT2Module` 将会过滤它们
        outputs = (hidden_states, all_hidden_states, all_attentions, all_cross_attentions)

        return outputs


class FlaxGPT2Module(nn.Module):
    config: GPT2Config
    dtype: jnp.dtype = jnp.float32

    # 初始化模块，设置词嵌入、位置嵌入、Dropout、隐藏层块集合和最终层归一化
    def setup(self):
        # 设置词嵌入维度为隐藏大小
        self.embed_dim = self.config.hidden_size

        # 初始化词嵌入和位置嵌入层
        self.wte = nn.Embed(
            self.config.vocab_size,
            self.embed_dim,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        self.wpe = nn.Embed(
            self.config.max_position_embeddings,
            self.embed_dim,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        # 初始化 Dropout 层
        self.dropout = nn.Dropout(rate=self.config.embd_pdrop)
        # 初始化隐藏层块集合
        self.h = FlaxGPT2BlockCollection(self.config, dtype=self.dtype)
        # 初始化最终层的归一化层
        self.ln_f = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)
    # 定义一个调用函数，接受多个参数：
    #   - input_ids: 输入的标识符序列
    #   - attention_mask: 注意力掩码
    #   - position_ids: 位置编码
    #   - encoder_hidden_states: 编码器的隐藏状态（可选）
    #   - encoder_attention_mask: 编码器的注意力掩码（可选）
    #   - deterministic: 是否确定性操作的标志，默认为True
    #   - init_cache: 是否初始化缓存的标志，默认为False
    #   - output_attentions: 是否输出注意力权重，默认为False
    #   - output_hidden_states: 是否输出隐藏状态，默认为False
    #   - return_dict: 是否以字典形式返回结果，默认为True
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
        # 通过输入标识符序列获取对应的词嵌入
        input_embeds = self.wte(input_ids.astype("i4"))
        # 根据位置编码获取对应的位置嵌入
        position_embeds = self.wpe(position_ids.astype("i4"))

        # 将词嵌入和位置嵌入相加得到初始的隐藏状态
        hidden_states = input_embeds + position_embeds
        # 对隐藏状态进行丢弃操作，以减少过拟合，deterministic参数控制是否确定性
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)

        # 调用模型的前向传播函数h，处理隐藏状态和相关参数，获取输出
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

        # 从模型输出中取出新的隐藏状态
        hidden_states = outputs[0]
        # 对新的隐藏状态进行 LayerNorm 归一化操作
        hidden_states = self.ln_f(hidden_states)

        # 如果需要输出所有隐藏状态，则将它们与最新的隐藏状态合并
        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            outputs = (hidden_states,) + outputs[1:]

        # 如果不需要以字典形式返回结果，则以元组形式返回所有非空结果
        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        # 以特定格式返回最终结果，包含最终隐藏状态、所有隐藏状态、注意力权重和交叉注意力权重
        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=outputs[1],
            attentions=outputs[2],
            cross_attentions=outputs[3],
        )
# 使用装饰器添加文档字符串，描述这是一个不带特定顶层头部的原始隐藏状态输出的 GPT2 模型转换器
@add_start_docstrings(
    "The bare GPT2 Model transformer outputting raw hidden-states without any specific head on top.",
    GPT2_START_DOCSTRING,
)
# 定义 FlaxGPT2Model 类，它继承自 FlaxGPT2PreTrainedModel
class FlaxGPT2Model(FlaxGPT2PreTrainedModel):
    # 模块类别设置为 FlaxGPT2Module
    module_class = FlaxGPT2Module


# 使用示例调用文档字符串添加函数，为 FlaxGPT2Model 类添加文档字符串
append_call_sample_docstring(
    FlaxGPT2Model,
    _CHECKPOINT_FOR_DOC,  # 使用的检查点文档
    FlaxBaseModelOutputWithPastAndCrossAttentions,  # 输出的模型输出类
    _CONFIG_FOR_DOC,  # 使用的配置文档
)


# 定义 FlaxGPT2LMHeadModule 类，继承自 nn.Module
class FlaxGPT2LMHeadModule(nn.Module):
    config: GPT2Config  # 类型为 GPT2Config 的配置对象
    dtype: jnp.dtype = jnp.float32  # 数据类型设置为 jnp.float32，默认为 float32

    def setup(self):
        # 初始化 transformer 层，使用 FlaxGPT2Module 类
        self.transformer = FlaxGPT2Module(self.config, dtype=self.dtype)
        # 初始化 lm_head 层，使用全连接层 Dense，设置参数包括词汇表大小、无偏置、dtype 和初始化范围
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )

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
        # 调用 transformer 层进行前向传播，获取输出结果
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

        hidden_states = outputs[0]  # 获取 transformer 输出的隐藏状态

        if self.config.tie_word_embeddings:
            # 如果配置要求共享词嵌入权重，则从 transformer 的参数中获取共享的词嵌入权重矩阵
            shared_kernel = self.transformer.variables["params"]["wte"]["embedding"].T
            # 应用共享的词嵌入权重进行 lm_head 的计算
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_kernel}}, hidden_states)
        else:
            # 否则直接使用 lm_head 层计算 lm_logits
            lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            # 如果不要求返回字典形式的输出，则返回元组形式的结果
            return (lm_logits,) + outputs[1:]

        # 返回 FlaxCausalLMOutputWithCrossAttentions 类的实例，包括 lm_logits、hidden_states、attentions 和 cross_attentions
        return FlaxCausalLMOutputWithCrossAttentions(
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


# 使用装饰器添加文档字符串，描述这是一个带有语言建模头部（线性层权重与输入嵌入绑定）的 GPT2 模型转换器
@add_start_docstrings(
    """
    The GPT2 Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    GPT2_START_DOCSTRING,
)
# 定义 FlaxGPT2LMHeadModel 类，继承自 FlaxGPT2PreTrainedModel
class FlaxGPT2LMHeadModel(FlaxGPT2PreTrainedModel):
    module_class = FlaxGPT2LMHeadModule  # 模块类别设置为 FlaxGPT2LMHeadModule
    # 准备生成过程的输入数据，包括初始化缓存和生成位置信息的处理
    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = None):
        # 初始化缓存，获取输入的批量大小和序列长度
        batch_size, seq_length = input_ids.shape

        # 使用初始化方法创建缓存的键值对
        past_key_values = self.init_cache(batch_size, max_length)

        # 创建一个扩展的注意力掩码，初始值为全1数组
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")

        # 如果有提供注意力掩码，则进行处理
        if attention_mask is not None:
            # 计算位置编码，累积求和减去1以确保位置正确
            position_ids = attention_mask.cumsum(axis=-1) - 1
            # 使用动态更新切片操作，将注意力掩码更新到扩展的注意力掩码中
            extended_attention_mask = lax.dynamic_update_slice(
                extended_attention_mask, attention_mask.astype("i4"), (0, 0)
            )
        else:
            # 如果未提供注意力掩码，则根据输入序列长度广播生成位置编码
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        # 返回准备好的输入字典，包括缓存的键值对、扩展的注意力掩码和位置编码
        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    # 更新生成过程的输入数据，主要是更新缓存和位置编码
    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        # 更新模型关键字参数中的缓存键值对和位置编码
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1

        # 返回更新后的模型关键字参数
        return model_kwargs
# 将样例文档字符串附加到函数或类中
append_call_sample_docstring(
    # 将 FlaxGPT2LMHeadModel 类作为目标对象
    FlaxGPT2LMHeadModel,
    # 使用 _CHECKPOINT_FOR_DOC 作为样例文档字符串的检查点
    _CHECKPOINT_FOR_DOC,
    # 将 FlaxCausalLMOutputWithCrossAttentions 类作为相关交叉注意力的因果语言模型输出
    FlaxCausalLMOutputWithCrossAttentions,
    # 使用 _CONFIG_FOR_DOC 作为样例文档字符串的配置
    _CONFIG_FOR_DOC,
)
```