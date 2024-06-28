# `.\models\gptj\modeling_flax_gptj.py`

```
from functools import partial
from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax

from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_gptj import GPTJConfig
    # 定义函数 `ultimate`，接收配置参数和数据类型参数
    Parameters:
        config ([`GPTJConfig`]): Model configuration class with all the parameters of the model.
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
"""

GPTJ_INPUTS_DOCSTRING = r"""
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


def create_sinusoidal_positions(num_pos, dim):
    # 计算频率因子
    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
    # 计算正弦和余弦位置编码
    sinusoid_inp = np.einsum("i , j -> i j", np.arange(num_pos), inv_freq).astype("float32")
    sin, cos = np.sin(sinusoid_inp), np.cos(sinusoid_inp)

    # 计算需要填充的数量
    sentinel = dim // 2 + dim % 2
    # 创建输出数组
    out = np.zeros((num_pos, dim))
    # 填充正弦和余弦值
    out[:, 0:sentinel] = sin
    out[:, sentinel:] = cos

    return jnp.array(out)


def rotate_every_two(tensor):
    # 旋转张量中的每两个元素
    rotate_half_tensor = jnp.stack((-tensor[:, :, :, 1::2], tensor[:, :, :, ::2]), axis=-1)
    rotate_half_tensor = rotate_half_tensor.reshape(rotate_half_tensor.shape[:-2] + (-1,))
    return rotate_half_tensor


def apply_rotary_pos_emb(tensor, sincos):
    sin_pos, cos_pos = sincos
    # 扩展正弦和余弦位置编码
    sin_pos = sin_pos[:, :, None, :].repeat(2, 3)
    cos_pos = cos_pos[:, :, None, :].repeat(2, 3)
    # 应用旋转位置编码
    return (tensor * cos_pos) + (rotate_every_two(tensor) * sin_pos)


class FlaxGPTJAttention(nn.Module):
    config: GPTJConfig
    dtype: jnp.dtype = jnp.float32
    # 定义一个布尔类型的实例变量 causal，默认为 True，表示是否使用因果注意力
    causal: bool = True
    # 定义一个布尔类型的实例变量 is_cross_attention，默认为 False，表示是否进行交叉注意力

    # 初始化方法，在创建类实例时被调用，用于配置模型参数和初始化一些变量
    def setup(self):
        # 获取配置信息
        config = self.config
        # 设置嵌入维度为隐藏大小
        self.embed_dim = config.hidden_size
        # 设置注意力头数为配置中指定的注意力头数
        self.num_heads = config.num_attention_heads
        # 计算每个注意力头的维度
        self.head_dim = self.embed_dim // self.num_heads

        # 设置旋转维度为配置中指定的旋转维度，如果未指定则使用嵌入维度
        self.rotary_dim = config.rotary_dim

        # 创建偏函数 dense，用于创建全连接层
        dense = partial(
            nn.Dense,
            self.embed_dim,
            use_bias=False,
            dtype=self.dtype,
            # 使用正态分布初始化权重，范围由配置中的 initializer_range 指定
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

        # 创建查询、键、值投影层
        self.q_proj, self.k_proj, self.v_proj = dense(), dense(), dense()
        # 创建输出投影层
        self.out_proj = dense()

        # 设置残差连接的 dropout 层，丢弃率由配置中的 resid_pdrop 指定
        self.resid_dropout = nn.Dropout(rate=config.resid_pdrop)

        # 创建因果注意力掩码，形状为 (1, max_position_embeddings)，用于屏蔽未来信息
        self.causal_mask = make_causal_mask(jnp.ones((1, config.max_position_embeddings), dtype="bool"), dtype="bool")

        # 计算位置编码维度，如果未指定旋转维度，则使用嵌入维度
        pos_embd_dim = self.rotary_dim or self.embed_dim
        # 创建正弦位置编码
        self.embed_positions = create_sinusoidal_positions(config.max_position_embeddings, pos_embd_dim)

    # 将隐藏状态按注意力头分割
    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    # 将分割的注意力头合并
    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    # 使用 JAX 提供的 @nn.compact 装饰器，表示这是一个使用了 JAX 的紧凑型层定义
    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slightly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # 检测是否通过缺少现有缓存数据来初始化。
        is_initialized = self.has_variable("cache", "cached_key")
        # 获取缓存的键，如果不存在则初始化为形状和类型与输入相同的零数组
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        # 获取缓存的值，如果不存在则初始化为形状和类型与输入相同的零数组
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        # 获取缓存索引，如果不存在则初始化为值为0的整数数组
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            # 获取批次维度、最大长度、注意力头数和每个头部深度的缓存键形状
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # 使用新的一维空间切片更新键和值缓存
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            # 更新缓存索引，增加已更新的缓存向量数量
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # 对于缓存的解码器自注意力，生成因果掩码：我们的单个查询位置只应注意到已生成和缓存的键位置，而不是剩余的零元素。
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            # 合并填充掩码和输入的注意力掩码
            attention_mask = combine_masks(pad_mask, attention_mask)
        # 返回更新后的键、值和注意力掩码
        return key, value, attention_mask
class FlaxGPTJMLP(nn.Module):
    # GPTJConfig 类型的配置对象
    config: GPTJConfig
    # 中间层大小
    intermediate_size: int
    # 数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 初始化模块
    def setup(self):
        # 嵌入维度为隐藏大小
        embed_dim = self.config.hidden_size
        # 使用正态分布初始化核
        kernel_init = jax.nn.initializers.normal(self.config.initializer_range)

        # 输入全连接层
        self.fc_in = nn.Dense(self.intermediate_size, dtype=self.dtype, kernel_init=kernel_init)
        # 输出全连接层
        self.fc_out = nn.Dense(embed_dim, dtype=self.dtype, kernel_init=kernel_init)

        # 激活函数选择
        self.act = ACT2FN[self.config.activation_function]
        # 丢弃率为 config 中的 resid_pdrop
        self.dropout = nn.Dropout(rate=self.config.resid_pdrop)

    # 调用模块
    def __call__(self, hidden_states, deterministic: bool = True):
        # 输入经过输入全连接层
        hidden_states = self.fc_in(hidden_states)
        # 应用激活函数
        hidden_states = self.act(hidden_states)
        # 输入经过输出全连接层
        hidden_states = self.fc_out(hidden_states)
        # 使用 dropout 进行正则化
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states


class FlaxGPTJBlock(nn.Module):
    # GPTJConfig 类型的配置对象
    config: GPTJConfig
    # 数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 初始化模块
    def setup(self):
        # 隐藏大小
        hidden_size = self.config.hidden_size
        # 内部维度为 n_inner 或者 4 * hidden_size
        inner_dim = self.config.n_inner if self.config.n_inner is not None else 4 * hidden_size

        # 层归一化
        self.ln_1 = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)
        # 自注意力机制
        self.attn = FlaxGPTJAttention(self.config, dtype=self.dtype)

        # 多层感知机
        self.mlp = FlaxGPTJMLP(self.config, inner_dim, dtype=self.dtype)

    # 调用模块
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):
        # 残差连接
        residual = hidden_states
        # 层归一化
        hidden_states = self.ln_1(hidden_states)
        # 自注意力机制输出
        attn_outputs = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]

        # 前馈网络的隐藏状态
        feed_forward_hidden_states = self.mlp(hidden_states, deterministic=deterministic)
        # 残差连接
        hidden_states = attn_output + feed_forward_hidden_states + residual

        return (hidden_states,) + attn_outputs[1:]


class FlaxGPTJPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # GPTJConfig 类型的配置类
    config_class = GPTJConfig
    # 基础模型前缀为 "transformer"
    base_model_prefix = "transformer"
    # 模块类
    module_class: nn.Module = None

    # 初始化方法
    def __init__(
        self,
        config: GPTJConfig,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        # 使用给定的配置和类型初始化模块
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量
        input_ids = jnp.zeros(input_shape, dtype="i4")
        # 创建与input_ids相同形状的全1注意力掩码
        attention_mask = jnp.ones_like(input_ids)
        # 创建位置编码，广播以匹配input_ids的形状
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        # 划分随机数生成器rng为参数rngs和dropout_rng
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        if self.config.add_cross_attention:
            # 如果配置中包含跨注意力机制，初始化编码器隐藏状态和注意力掩码
            encoder_hidden_states = jnp.zeros(input_shape + (self.config.n_embd,))
            encoder_attention_mask = attention_mask
            # 使用模型模块初始化
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
            # 否则仅使用输入初始化模型模块
            module_init_outputs = self.module.init(rngs, input_ids, attention_mask, position_ids, return_dict=False)

        # 获取随机初始化的模型参数
        random_params = module_init_outputs["params"]

        if params is not None:
            # 如果提供了预先定义的参数，将随机参数和预定义参数扁平化处理并合并
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            # 返回冻结后的合并参数
            return freeze(unflatten_dict(params))
        else:
            # 否则直接返回随机初始化的参数
            return random_params

    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                用于快速自回归解码的批量大小，定义了初始化缓存的批处理大小。
            max_length (`int`):
                自回归解码的最大可能长度，定义了初始化缓存的序列长度。
        """
        # 初始化用于检索缓存的输入变量
        input_ids = jnp.ones((batch_size, max_length))
        # 创建与input_ids相同形状的全1注意力掩码
        attention_mask = jnp.ones_like(input_ids)
        # 创建位置编码，广播以匹配input_ids的形状
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        # 使用模型模块初始化，设置init_cache标志以初始化缓存
        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        # 返回初始化的缓存
        return init_variables["cache"]

    @add_start_docstrings_to_model_forward(GPTJ_INPUTS_DOCSTRING)
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
            # 检查是否需要输出注意力权重，若未指定则使用配置中的默认设置
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            # 检查是否需要输出隐藏状态，若未指定则使用配置中的默认设置
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            # 检查是否需要返回字典形式的输出，若未指定则使用配置中的默认设置
            return_dict = return_dict if return_dict is not None else self.config.return_dict

            # 获取输入张量的批量大小和序列长度
            batch_size, sequence_length = input_ids.shape

            # 如果未提供位置编码（position_ids），且已传递过去的键值（past_key_values）不为空，则抛出错误
            if position_ids is None:
                if past_key_values is not None:
                    raise ValueError("Make sure to provide `position_ids` when passing `past_key_values`.")

                # 使用序列长度创建广播位置编码（position_ids）
                position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

            # 如果未提供注意力遮罩（attention_mask），则创建全为1的注意力遮罩
            if attention_mask is None:
                attention_mask = jnp.ones((batch_size, sequence_length))

            # 处理可能存在的伪随机数发生器
            rngs = {}
            if dropout_rng is not None:
                rngs["dropout"] = dropout_rng

            # 构建输入参数字典
            inputs = {"params": params or self.params}

            # 如果已传递过去的键值（past_key_values）不为空，则初始化缓存
            if past_key_values:
                inputs["cache"] = past_key_values
                mutable = ["cache"]
            else:
                mutable = False

            # 应用模型的前向传播
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

            # 如果已传递过去的键值（past_key_values）不为空且需要返回字典形式的输出（return_dict），则添加更新后的缓存到模型输出中
            if past_key_values is not None and return_dict:
                outputs, past_key_values = outputs
                outputs["past_key_values"] = unfreeze(past_key_values["cache"])
                return outputs
            # 如果已传递过去的键值（past_key_values）不为空且不需要返回字典形式的输出（return_dict），则更新输出元组中的缓存信息
            elif past_key_values is not None and not return_dict:
                outputs, past_key_values = outputs
                outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

            # 返回模型的输出
            return outputs
# 定义一个名为 FlaxGPTJBlockCollection 的类，继承自 nn.Module
class FlaxGPTJBlockCollection(nn.Module):
    # 类属性 config，类型为 GPTJConfig，dtype 默认为 jnp.float32
    config: GPTJConfig
    dtype: jnp.dtype = jnp.float32

    # 定义 setup 方法，用于初始化模块
    def setup(self):
        # 创建一个包含多个 FlaxGPTJBlock 实例的列表 self.blocks
        self.blocks = [
            FlaxGPTJBlock(self.config, name=str(i), dtype=self.dtype) for i in range(self.config.num_hidden_layers)
        ]

    # 定义 __call__ 方法，使对象可以像函数一样调用
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 如果输出注意力矩阵，则初始化一个空元组 all_attentions
        all_attentions = () if output_attentions else None
        # 如果输出隐藏状态，则初始化一个空元组 all_hidden_states
        all_hidden_states = () if output_hidden_states else None

        # 遍历 self.blocks 中的每个 FlaxGPTJBlock 实例
        for block in self.blocks:
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # 调用 block 对象处理当前的 hidden_states 等参数，返回 layer_outputs
            layer_outputs = block(
                hidden_states,
                attention_mask,
                position_ids=position_ids,
                deterministic=deterministic,
                init_cache=init_cache,
                output_attentions=output_attentions,
            )
            # 更新 hidden_states 为当前层处理后的输出
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力矩阵，则将当前层的注意力矩阵添加到 all_attentions
            if output_attentions:
                all_attentions += (layer_outputs[1],)

        # 构建输出元组 outputs，包含最终的 hidden_states、all_hidden_states 和 all_attentions
        # 其中 all_attentions 可能包含 None 值，会在 FlaxGPTJModule 中进行过滤处理
        outputs = (hidden_states, all_hidden_states, all_attentions)

        # 返回最终的输出结果
        return outputs


# 定义一个名为 FlaxGPTJModule 的类，继承自 nn.Module
class FlaxGPTJModule(nn.Module):
    # 类属性 config，类型为 GPTJConfig，dtype 默认为 jnp.float32
    config: GPTJConfig
    dtype: jnp.dtype = jnp.float32

    # 定义 setup 方法，用于初始化模块
    def setup(self):
        # 初始化 self.embed_dim 为 config.hidden_size
        self.embed_dim = self.config.hidden_size

        # 创建一个 nn.Embed 实例 self.wte，用于词嵌入
        self.wte = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            # 使用正态分布初始化词嵌入权重
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        # 创建一个 nn.Dropout 实例 self.dropout，用于词嵌入后的 dropout
        self.dropout = nn.Dropout(rate=self.config.embd_pdrop)
        # 创建一个 FlaxGPTJBlockCollection 实例 self.h，用于处理隐藏层
        self.h = FlaxGPTJBlockCollection(self.config, dtype=self.dtype)
        # 创建一个 nn.LayerNorm 实例 self.ln_f，用于最终的 Layer Normalization
        self.ln_f = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)

    # 定义 __call__ 方法，使对象可以像函数一样调用
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
        # 实际执行的内容在 FlaxGPTJBlockCollection 的 __call__ 方法中完成，此处不作进一步解释
        pass
        ):
        # 使用 self.wte 将输入的整数数组转换为嵌入向量，数据类型为 'i4'
        input_embeds = self.wte(input_ids.astype("i4"))

        # 对输入的嵌入向量应用 dropout，根据 deterministic 参数决定是否确定性地应用
        hidden_states = self.dropout(input_embeds, deterministic=deterministic)

        # 将处理后的隐藏状态传入 self.h 进行处理，接收多个命名参数
        outputs = self.h(
            hidden_states,
            attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从输出中获取第一个元素作为新的隐藏状态
        hidden_states = outputs[0]

        # 对新的隐藏状态应用 LayerNormalization，self.ln_f 是一个层标准化层
        hidden_states = self.ln_f(hidden_states)

        # 如果设置了 output_hidden_states 标志，将所有隐藏状态存储到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            outputs = (hidden_states,) + outputs[1:]

        # 如果 return_dict 是 False，则返回 outputs 中不为 None 的所有元素
        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        # 如果 return_dict 是 True，则返回 FlaxBaseModelOutput 对象，包含隐藏状态、所有隐藏状态和注意力
        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=outputs[1],
            attentions=outputs[-1],
        )
# 用于添加起始文档字符串的装饰器函数，描述了FlaxGPTJModel类的基本功能和输出信息
@add_start_docstrings(
    "The bare GPTJ Model transformer outputting raw hidden-states without any specific head on top.",
    GPTJ_START_DOCSTRING,
)
# 将示例调用文档字符串添加到FlaxGPTJModel类中，包含了模型检查点、输出配置信息和配置输出的样本
append_call_sample_docstring(
    FlaxGPTJModel,
    _CHECKPOINT_FOR_DOC,
    FlaxCausalLMOutput,
    _CONFIG_FOR_DOC,
)

# 定义一个用于语言建模的Flax模块类，依赖于GPTJConfig配置，并设置了数据类型为32位浮点数
class FlaxGPTJForCausalLMModule(nn.Module):
    config: GPTJConfig
    dtype: jnp.dtype = jnp.float32

    # 模块设置函数，初始化transformer和lm_head
    def setup(self):
        self.transformer = FlaxGPTJModule(self.config, dtype=self.dtype)
        # 使用配置中的词汇表大小初始化lm_head的全连接层
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            dtype=self.dtype,
            # 使用正态分布初始化全连接层的权重矩阵
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )

    # 模块的调用函数，接收多个参数和关键字参数，并返回语言建模的输出
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
        # 使用transformer处理输入数据，返回各种输出
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

        hidden_states = outputs[0]  # 提取transformer的隐藏状态作为下一步处理的输入

        if self.config.tie_word_embeddings:
            # 如果配置要求共享词嵌入矩阵，则从transformer的参数中提取共享的权重矩阵，并应用到lm_head
            shared_kernel = self.transformer.variables["params"]["wte"]["embedding"].T
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_kernel}}, hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)  # 否则直接使用lm_head进行预测

        if not return_dict:
            # 如果不返回字典形式的输出，则返回一个元组，包括lm_logits和outputs的其余部分
            return (lm_logits,) + outputs[1:]

        # 返回格式化的语言建模输出对象，包括logits、隐藏状态和注意力权重（如果有的话）
        return FlaxCausalLMOutput(logits=lm_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


# 为FlaxGPTJForCausalLM类添加起始文档字符串，描述其为一个带有语言建模头的GPTJ模型变体
@add_start_docstrings(
    """
    The GPTJ Model transformer with a language modeling head on top.
    """,
    GPTJ_START_DOCSTRING,
)
class FlaxGPTJForCausalLM(FlaxGPTJPreTrainedModel):
    module_class = FlaxGPTJForCausalLMModule  # 指定模块类为FlaxGPTJForCausalLMModule
    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = None):
        # initializing the cache
        # 获取输入张量的批大小和序列长度
        batch_size, seq_length = input_ids.shape

        # 使用初始化方法初始化过去键值（用于缓存）
        past_key_values = self.init_cache(batch_size, max_length)
        
        # 注意：通常需要在 attention_mask 中的位置（大于 input_ids.shape[-1] 和小于 cache_length）放置0。
        # 但由于 GPTJ 使用因果掩码，这些位置已经被掩盖了。
        # 因此，我们可以在这里创建一个静态的 attention_mask，这对编译来说更有效率。
        
        # 创建一个扩展的 attention_mask，全部初始化为1
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        
        # 如果传入了 attention_mask，则根据它的累积和位置更新 extended_attention_mask
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            # 否则，使用广播方法创建位置索引，范围是 [0, seq_length)
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        # 返回包含过去键值、扩展后的 attention_mask 和位置索引的字典
        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        # 更新用于生成的模型输入参数
        
        # 将模型输出的过去键值更新到模型参数中
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        
        # 更新位置索引，仅保留最后一个位置
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        
        # 返回更新后的模型参数
        return model_kwargs
```python`
# 调用函数append_call_sample_docstring，添加样例文档字符串到指定类和对象的关联属性中
append_call_sample_docstring(
    FlaxGPTJForCausalLM,  # 目标类：FlaxGPTJForCausalLM
    _CHECKPOINT_FOR_DOC,  # 样例文档字符串的检查点
    FlaxCausalLMOutput,   # 输出类：FlaxCausalLMOutput
    _CONFIG_FOR_DOC,      # 样例文档字符串的配置
)
```