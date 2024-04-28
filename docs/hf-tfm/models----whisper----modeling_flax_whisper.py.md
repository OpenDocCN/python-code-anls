# `.\transformers\models\whisper\modeling_flax_whisper.py`

```py
# 设置文件编码为utf-8
# 版权声明
#
# 根据Apache License，Version 2.0获得授权后，可以使用该文件
#
# 如果根据适用法律需要或同意，软件在遵守许可证的基础上分发
# 根据原样分发，不提供任何明示或暗示的担保或条件
# 请参阅许可证以获取特定语言的所有权利和限制
""" Flax whisper model."""
# 导入所需模块
import math
import random
from functools import partial
from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen import partitioning as nn_partitioning
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from jax.random import PRNGKey
# 导入其它文件中的一些类
from ...generation.flax_logits_process import FlaxWhisperTimeStampLogitsProcessor
from ...modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxBaseModelOutputWithPastAndCrossAttentions,
    FlaxCausalLMOutputWithCrossAttentions,
    FlaxSeq2SeqLMOutput,
    FlaxSeq2SeqModelOutput,
    FlaxSequenceClassifierOutput,
)
# 导入其它文件中的一些方法和变量
from ...modeling_flax_utils import (
    ACT2FN,
    FlaxPreTrainedModel,
    append_call_sample_docstring,
    append_replace_return_docstrings,
    overwrite_call_docstring,
)
# 导入其它文件中的一些方法和变量
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
# 从Whisper配置文件中导入WhisperConfig类
from .configuration_whisper import WhisperConfig

logger = logging.get_logger(__name__)

# 用于文档的一些配置
_CHECKPOINT_FOR_DOC = "openai/whisper-tiny"
_CONFIG_FOR_DOC = "WhisperConfig"

remat = nn_partitioning.remat

# 定义函数：生成正弦位置嵌入
def sinusoidal_embedding_init(key, shape, dtype=jnp.float_) -> jax.Array:
    """Returns sinusoids for positional embedding"""
    length, channels = shape
    if channels % 2 != 0:
        raise ValueError(
            f"Number of channels has to be divisible by 2 for sinusoidal positional embeddings, got {channels} channels."
        )
    log_timescale_increment = math.log(10000) / (channels // 2 - 1)
    inv_timescales = jnp.exp(-log_timescale_increment * jnp.arange(channels // 2))
    scaled_time = jnp.arange(length).reshape(-1, 1) * inv_timescales.reshape(1, -1)
    return jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=1).astype(dtype)

# 开始文档说明
WHISPER_START_DOCSTRING = r"""
    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its models (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.) This model is also a Flax Linen
    # [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a
    # regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.
    # Finally, this model supports inherent JAX features such as:
    # - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    # - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    # - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    # - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)
    
    Parameters:
        config ([`WhisperConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.
        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
            The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and
            `jax.numpy.bfloat16` (on TPUs). This can be used to enable mixed-precision training or half-precision
            inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
            **Note that this only specifies the dtype of the computation and does not influence the dtype of model
            parameters.** If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`]
            and [`~FlaxPreTrainedModel.to_bf16`].
"""

# WHISPER_INPUTS_DOCSTRING 为 Whisper 模型输入文档字符串定义

WHISPER_INPUTS_DOCSTRING = r"""
"""

# WHISPER_ENCODE_INPUTS_DOCSTRING 为 Whisper 模型编码器输入函数文档字符串定义
WHISPER_ENCODE_INPUTS_DOCSTRING = r"""
    Args:
        input_features (`numpy.ndarray` of shape `(batch_size, feature_size, sequence_length)`):
            Float values mel features extracted from the raw speech waveform. Raw speech waveform can be obtained by
            loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via
            the soundfile library (`pip install soundfile`). To prepare the array into `input_features`, the
            [`WhisperFeatureExtractor`] should be used for extracting the mel features, padding and conversion into a
            tensor of type `numpy.ndarray`. See [`~WhisperFeatureExtractor.__call__`].
        attention_mask (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Whisper does not support masking of the `input_features`, this argument is preserved for compatibility, but
            is not used. By default the silence in the input log mel spectrogram are ignored.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# WHISPER_DECODE_INPUTS_DOCSTRING 为 Whisper 模型解码器输入函数文档字符串定义
WHISPER_DECODE_INPUTS_DOCSTRING = r"""
    # 参数说明：decoder_input_ids是解码器的输入token索引，encoder_outputs是编码器的输出元组，包括最后一个隐藏状态、隐藏状态序列和注意力权重序列，encoder_attention_mask是编码器的注意力掩码，默认情况下会忽略输入日志梅尔频谱中的沉默部分
    # 参数说明：decoder_attention_mask是解码器的注意力掩码，默认行为是生成一个张量，忽略decoder_input_ids中的填充token，同时默认使用因果掩码。如果要更改填充行为，可以自行修改。更多信息请参考论文中的图表1
    # 参数说明：decoder_position_ids是解码器输入序列token在位置嵌入中的位置索引，返回的past_key_values是预计算的隐藏状态字典（注意力块中的键和值），可用于快速自回归解码
    # 参数说明：output_attentions和output_hidden_states表示是否返回所有层的注意力和隐藏状态张量，return_dict表示是否返回ModelOutput而不是普通的元组
# 定义一个名为FlaxWhisperAttention的类，继承自nn.Module
class FlaxWhisperAttention(nn.Module):
    # 声明类变量config，类型为WhisperConfig
    config: WhisperConfig
    # 声明类变量embed_dim，表示嵌入维度
    embed_dim: int
    # 声明类变量num_heads，表示注意力头的数量
    num_heads: int
    # 声明类变量dropout，表示dropout概率，默认为0.0
    dropout: float = 0.0
    # 声明类变量causal，表示是否是因果注意力，默认为False
    causal: bool = False
    # 声明类变量bias，表示是否使用偏置，默认为True
    bias: bool = True
    # 声明类变量dtype，表示数据类型，默认为jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 定义setup方法，用于初始化参数
    def setup(self) -> None:
        # 计算每个头的维度
        self.head_dim = self.embed_dim // self.num_heads
        # 如果embed_dim不能被num_heads整除，则抛出ValueError异常
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        # 使用偏置的全连接层，用于查询
        dense = partial(
            nn.Dense,
            self.embed_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        self.q_proj = dense(use_bias=self.bias)  # 查询投影
        self.k_proj = dense(use_bias=False)      # 键投影
        self.v_proj = dense(use_bias=self.bias)  # 值投影
        self.out_proj = dense(use_bias=self.bias) # 输出投影

        # 如果是因果注意力，创建一个因果mask
        if self.causal:
            self.causal_mask = make_causal_mask(
                jnp.ones((1, self.config.max_target_positions), dtype="bool"), dtype="bool"
            )

    # 定义__call__方法，用于实现类的调用
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        key_value_states: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        deterministic: bool = True,
    # 定义_split_heads方法，用于将隐藏状态拆分成多头注意力的格式
    def _split_heads(self, hidden_state) -> jnp.ndarray:
        return hidden_state.reshape(hidden_state.shape[:2] + (self.num_heads, self.head_dim))

    # 定义_merge_heads方法，用于将多头注意力合并成原始隐藏状态的格式
    def _merge_heads(self, hidden_state) -> jnp.ndarray:
        return hidden_state.reshape(hidden_state.shape[:2] + (self.embed_dim,))

    # 使用装饰器定义一个紧凑的函数
    @nn.compact
    # 将 key、value 和 attention_mask 拼接到缓存中，并返回更新后的 key、value 和 attention_mask
    def _concatenate_to_cache(self, key, value, query, attention_mask) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # 检测是否通过缺少现有缓存数据来初始化
        is_initialized = self.has_variable("cache", "cached_key")
        # 获取缓存的 key，如果不存在则初始化为与 key 相同形状和类型的零数组
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        # 获取缓存的 value，如果不存在则初始化为与 value 相同形状和类型的零数组
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        # 获取缓存的索引，如果不存在则初始化为 0
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            # 获取批量维度的长度，以及 cached_key 的剩余维度
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # 使用新的 1D 空间切片更新 key、value 缓存
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            # 更新缓存的 key 和 value
            cached_key.value = key
            cached_value.value = value
            # 更新 cache_index，增加已更新缓存向量的数量
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # 用于缓存的解码器自注意力的因果遮罩：我们的单个 query 位置只应关注已生成和缓存的 key 位置，而不是剩余的零元素
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)

        return key, value, attention_mask
# 从transformers.models.mbart.modeling_flax_mbart.FlaxMBartEncoderLayer复制代码并更改为Whisper
class FlaxWhisperEncoderLayer(nn.Module):
    # 定义类变量config，类型为WhisperConfig
    config: WhisperConfig
    # 定义dtype，表示计算中的数据类型，默认为jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 初始化函数，设置各个层的参数
    def setup(self) -> None:
        # 获取嵌入维度，即config中的d_model值
        self.embed_dim = self.config.d_model
        # 创建自注意力层对象，使用FlaxWhisperAttention
        self.self_attn = FlaxWhisperAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.encoder_attention_heads,
            dropout=self.config.attention_dropout,
            dtype=self.dtype,
        )
        # 创建自注意力层后的LayerNorm层
        self.self_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        # 创建dropout层
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)
        # 激活函数，根据config中的activation_function选择对应的函数
        self.activation_fn = ACT2FN[self.config.activation_function]
        # 激活函数后的dropout层
        self.activation_dropout_layer = nn.Dropout(rate=self.config.activation_dropout)
        # 第一个全连接层，输出维度为config中的encoder_ffn_dim
        self.fc1 = nn.Dense(
            self.config.encoder_ffn_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        # 第二个全连接层，输出维度为embed_dim
        self.fc2 = nn.Dense(
            self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std)
        )
        # 最终的LayerNorm层
        self.final_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    # 调用函数，接受隐藏状态和注意力掩码作为输入，返回隐藏状态和可选的注意力权重
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        output_attentions: bool = True,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray]:
        # 复制隐藏状态以备用
        residual = hidden_states
        # 对隐藏状态进行LayerNorm处理
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # 使用自注意力层处理隐藏状态
        hidden_states, attn_weights = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask)
        # 应用dropout层
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 添加残差连接
        hidden_states = residual + hidden_states

        # 复制更新后的隐藏状态以备用
        residual = hidden_states
        # 对更新后的隐藏状态进行LayerNorm处理
        hidden_states = self.final_layer_norm(hidden_states)
        # 使用激活函数处理第一个全连接层的输出
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 应用激活函数后的dropout层
        hidden_states = self.activation_dropout_layer(hidden_states, deterministic=deterministic)
        # 使用第二个全连接层
        hidden_states = self.fc2(hidden_states)
        # 应用dropout层
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 添加残差连接
        hidden_states = residual + hidden_states

        # 将隐藏状态作为输出
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则将注意力权重加入到输出中
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# 定义FlaxWhisperEncoderLayerCollection类，用于管理一组FlaxWhisperEncoderLayer
class FlaxWhisperEncoderLayerCollection(nn.Module):
    # 定义类变量config，类型为WhisperConfig
    config: WhisperConfig
    # 定义dtype，表示计算中的数据类型，默认为jnp.float32
    dtype: jnp.dtype = jnp.float32  
    # 梯度检查点标志，默认为False
    gradient_checkpointing: bool = False
    # 初始化操作，根据设置的梯度检查点来选择不同类型的编码层
    def setup(self):
        if self.gradient_checkpointing:
            # 使用 remat 函数将 FlaxWhisperEncoderLayer 转换为带有梯度检查点的层
            FlaxWhisperEncoderCheckpointLayer = remat(FlaxWhisperEncoderLayer, static_argnums=(2, 3))
            # 初始化编码层列表，根据配置创建对应个数的编码层，每个层的名称为对应的索引
            self.layers = [
                FlaxWhisperEncoderCheckpointLayer(self.config, name=str(i), dtype=self.dtype)
                for i in range(self.config.encoder_layers)
            ]
        else:
            # 初始化编码层列表，根据配置创建对应个数的编码层，每个层的名称为对应的索引
            self.layers = [
                FlaxWhisperEncoderLayer(self.config, name=str(i), dtype=self.dtype)
                for i in range(self.config.encoder_layers)
            ]
        # 设置层的丢失率为配置中设置的编码层丢失率
        self.layerdrop = self.config.encoder_layerdrop

    # 对编码层进行调用的方法
    def __call__(
        self,
        hidden_states,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 如果输出注意力，则创建一个空的元组来存放注意力
        all_attentions = () if output_attentions else None
        # 如果输出隐藏状态，则创建一个空的元组来存放隐藏状态
        all_hidden_states = () if output_hidden_states else None

        # 遍历每个编码层，并进行相应的操作
        for encoder_layer in self.layers:
            # 如果输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            # 为当前层添加 LayerDrop 操作，根据一定的概率决定是否跳过当前层
            dropout_probability = random.uniform(0, 1)
            if not deterministic and (dropout_probability < self.layerdrop):  # 如果不是确定性的且随机概率小于丢失率，则跳过当前层
                layer_outputs = (None, None)
            else:
                # 对当前层进行编码操作，得到编码后的隐藏状态和可能的注意力
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions,
                    deterministic,
                )
            # 更新隐藏状态为当前层的隐藏状态
            hidden_states = layer_outputs[0]
            # 如果输出注意力，则将当前注意力添加到 all_attentions 中
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # 如果输出隐藏状态，则将最后的隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 返回编码后的结果，包括最后的隐藏状态、所有的隐藏状态和所有的注意力
        outputs = (hidden_states, all_hidden_states, all_attentions)

        # 如果不需要返回字典，则将结果组成元组返回
        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        # 如果需要返回字典，则将结果包装成 FlaxBaseModelOutput 对象返回
        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )
# 从transformers.models.mbart.modeling_flax_mbart中复制FlaxMBartDecoderLayer，并将MBart改为Whisper
class FlaxWhisperDecoderLayer(nn.Module):
    # 定义WhisperConfig类型的config变量，dtype默认为jnp.float32
    config: WhisperConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        # 设置embed_dim为config中的d_model值
        self.embed_dim = self.config.d_model
        # 初始化self_attn为FlaxWhisperAttention对象
        self.self_attn = FlaxWhisperAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.decoder_attention_heads,
            dropout=self.config.attention_dropout,
            causal=True,
            dtype=self.dtype,
        )
        # 初始化dropout层
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)
        # 根据配置选择激活函数
        self.activation_fn = ACT2FN[self.config.activation_function]
        # 初始化激活函数的dropout层
        self.activation_dropout_layer = nn.Dropout(rate=self.config.activation_dropout)
        
        # 初始化self_attn_layer_norm为LayerNorm层
        self.self_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        # 初始化encoder_attn为FlaxWhisperAttention对象
        self.encoder_attn = FlaxWhisperAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.decoder_attention_heads,
            dropout=self.config.attention_dropout,
            dtype=self.dtype,
        )
        # 初始化encoder_attn_layer_norm为LayerNorm层
        self.encoder_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        # 初始化fc1为Dense层
        self.fc1 = nn.Dense(
            self.config.decoder_ffn_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        # 初始化fc2为Dense层
        self.fc2 = nn.Dense(
            self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std)
        )
        # 初始化final_layer_norm为LayerNorm层
        self.final_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        output_attentions: bool = True,
        deterministic: bool = True,
        # ...
    # 定义函数，接受隐藏状态作为输入，并返回一个元组，其中包含一个张量
    ) -> Tuple[jnp.ndarray]:
        # 将输入的隐藏状态作为残差连接的一部分保存下来
        residual = hidden_states
        # 对隐藏状态进行自注意力机制前的层归一化处理
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 自注意力机制
        # 调用 self_attn 方法对隐藏状态进行自注意力计算，并返回计算后的隐藏状态和注意力权重
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, init_cache=init_cache
        )
        # 对计算后的隐藏状态进行 dropout 处理
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 将残差与经过自注意力机制后的隐藏状态相加
        hidden_states = residual + hidden_states

        # 跨注意力机制块
        # 如果存在编码器的隐藏状态
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            # 保存隐藏状态的残差
            residual = hidden_states

            # 对隐藏状态进行编码器注意力层的归一化处理
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            # 调用 encoder_attn 方法进行编码器注意力计算，并返回计算后的隐藏状态和注意力权重
            hidden_states, cross_attn_weights = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            # 对计算后的隐藏状态进行 dropout 处理
            hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
            # 将残差与经过编码器注意力机制后的隐藏状态相加
            hidden_states = residual + hidden_states

        # 全连接层
        # 保存隐藏状态的残差
        residual = hidden_states
        # 对隐藏状态进行最终层的归一化处理
        hidden_states = self.final_layer_norm(hidden_states)
        # 使用激活函数对隐藏状态进行变换
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 对变换后的隐藏状态进行 dropout 处理
        hidden_states = self.activation_dropout_layer(hidden_states, deterministic=deterministic)
        # 使用第二个全连接层进行变换
        hidden_states = self.fc2(hidden_states)
        # 对变换后的隐藏状态进行 dropout 处理
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 将残差与经过全连接层处理后的隐藏状态相加
        hidden_states = residual + hidden_states

        # 将隐藏状态作为输出
        outputs = (hidden_states,)

        # 如果需要输出注意力权重
        if output_attentions:
            # 将自注意力机制和跨注意力机制的注意力权重添加到输出中
            outputs += (self_attn_weights, cross_attn_weights)

        # 返回输出
        return outputs
# 定义一个名为FlaxWhisperDecoderLayerCollection的类，继承自nn.Module
class FlaxWhisperDecoderLayerCollection(nn.Module):
    # 定义config属性，类型为WhisperConfig
    config: WhisperConfig
    # 定义dtype属性，类型为jnp.float32，默认值为jnp.float32，表示计算的数据类型
    dtype: jnp.dtype = jnp.float32  
    # 定义gradient_checkpointing属性，类型为bool，默认值为False，表示是否进行梯度检查点
    gradient_checkpointing: bool = False

    # 定义setup方法
    def setup(self):
        # 如果gradient_checkpointing为True
        if self.gradient_checkpointing:
            # 创建一个新的FlaxWhisperDecoderCheckpointLayer类，使用remat函数，并指定static_argnums参数
            FlaxWhisperDecoderCheckpointLayer = remat(FlaxWhisperDecoderLayer, static_argnums=(4, 5, 6))
            # 创建self.layers列表，包含多个FlaxWhisperDecoderCheckpointLayer实例
            self.layers = [
                FlaxWhisperDecoderCheckpointLayer(self.config, name=str(i), dtype=self.dtype)
                for i in range(self.config.decoder_layers)
            ]
        else:
            # 创建self.layers列表，包含多个FlaxWhisperDecoderLayer实例
            self.layers = [
                FlaxWhisperDecoderLayer(self.config, name=str(i), dtype=self.dtype)
                for i in range(self.config.decoder_layers)
            ]
        # 设置layerdrop属性为config.decoder_layerdrop的值
        self.layerdrop = self.config.decoder_layerdrop

    # 定义__call__方法，表示可以将对象作为函数调用
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
        # decoder layers
        # 如果输出隐藏状态，则初始化一个空元组，否则为 None
        all_hidden_states = () if output_hidden_states else None
        # 如果输出注意力，初始化一个空元组，否则为 None
        all_self_attns = () if output_attentions else None
        # 如果输出注意力并且编码器隐藏状态不为 None，初始化一个空元组，否则为 None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        # 循环遍历解码器层
        for decoder_layer in self.layers:
            # 如果输出隐藏状态，则添加当前隐藏状态到 all_hidden_states
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                # 添加 LayerDrop（详情请参考 https://arxiv.org/abs/1909.11556）
            # 生成一个 0 到 1 之间的随机浮点数
            dropout_probability = random.uniform(0, 1)
            # 如果不是确定性的，并且 dropout_probability 小于 self.layerdrop
            if not deterministic and (dropout_probability < self.layerdrop):
                # 将当前层的输出设置为 None
                layer_outputs = (None, None, None)
            else:
                # 调用解码器层的 forward 方法
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    init_cache,
                    output_attentions,
                    deterministic,
                )

            # 将当前层的隐藏状态更新为输出的隐藏状态
            hidden_states = layer_outputs[0]
            # 如果输出注意力，则添加当前注意力到 all_self_attns
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                # 如果编码器隐藏状态不为 None，则添加当前交叉注意力到 all_cross_attentions
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # 如果输出隐藏状态，则添加最后一个解码器层的隐藏状态到 all_hidden_states
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 将所有输出打包成一个列表
        outputs = [hidden_states, all_hidden_states, all_self_attns, all_cross_attentions]

        # 如果不返回字典，则返回包含所有非 None 值的元组
        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        # 返回 FlaxBaseModelOutputWithPastAndCrossAttentions 对象
        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )
# 定义一个名为FlaxWhisperEncoder的类，继承自nn.Module
class FlaxWhisperEncoder(nn.Module):
    # 声明类的属性config，类型为WhisperConfig
    config: WhisperConfig
    # 声明类的属性dtype，默认值为jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 声明类的属性gradient_checkpointing，默认值为False

    # 定义setup方法，不返回任何结果
    def setup(self) -> None:
        # 初始化第一个卷积层conv1
        self.conv1 = nn.Conv(
            self.config.d_model,
            kernel_size=(3,),
            padding=1,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
            dtype=self.dtype,
        )
        # 初始化第二个卷积层conv2
        self.conv2 = nn.Conv(
            self.config.d_model,
            kernel_size=(3,),
            strides=2,
            padding=1,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
            dtype=self.dtype,
        )

        # 初始化dropout_layer
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)

        # 初始化layers，一个FlaxWhisperEncoderLayerCollection对象
        self.layers = FlaxWhisperEncoderLayerCollection(
            self.config,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )

        # 初始化embed_positions，一个nn.Embed对象
        self.embed_positions = nn.Embed(
            self.config.max_source_positions,
            self.config.d_model,
            dtype=self.dtype,
            embedding_init=sinusoidal_embedding_init,
        )

        # 初始化layer_norm，一个nn.LayerNorm对象
        self.layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    # 定义__call__方法，接受多个参数
    def __call__(
        self,
        input_features: jnp.ndarray,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
    # 定义函数，接受输入特征并返回输出
    ) -> Tuple[jnp.ndarray]:
        # 检查输入特征的形状是否符合预期，如果不符合则引发 ValueError 异常
        if input_features.shape[1:] != (self.config.num_mel_bins, self.config.max_source_positions * 2):
            raise ValueError(
                "input_features.shape[1:], must be equal to (self.config.num_mel_bins,"
                f" self.config.max_source_positions * 2) (got {input_features.shape[1:]}, but should be"
                f" ({self.config.num_mel_bins}, {self.config.max_source_positions * 2}))"
            )

        # 将输入特征的维度重新排列，将时间维度放在前面
        input_features = input_features.transpose(0, 2, 1)
        # 对输入特征进行第一层卷积，并使用 GELU 激活函数
        hidden_states = jax.nn.gelu(self.conv1(input_features), approximate=False)
        # 对第一层卷积的输出进行第二层卷积，并使用 GELU 激活函数
        hidden_states = jax.nn.gelu(self.conv2(hidden_states), approximate=False)

        # 生成位置嵌入，通过对嵌入的梯度停止反向传播来冻结它们
        embed_positions = self.embed_positions(jnp.arange(self.config.max_source_positions))
        embed_positions = jax.lax.stop_gradient(embed_positions)
        # 将位置嵌入加到隐藏状态中
        hidden_states = hidden_states + embed_positions

        # 对隐藏状态进行 dropout
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)

        # 将隐藏状态传递给模型的层进行处理
        outputs = self.layers(
            hidden_states,
            attention_mask=None,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取模型输出中的最后一个隐藏状态
        last_hidden_states = outputs[0]
        # 对最后一个隐藏状态进行 LayerNormalization
        last_hidden_states = self.layer_norm(last_hidden_states)

        # 如果需要输出隐藏状态，则更新隐藏状态列表中的最后一个元素
        hidden_states = None
        if output_hidden_states:
            hidden_states = outputs[1]
            hidden_states = hidden_states[:-1] + (last_hidden_states,)

        # 如果不需要返回字典形式的输出，则将输出元组中不为 None 的值组成一个新的元组并返回
        if not return_dict:
            outputs = (last_hidden_states, hidden_states) + (outputs[2:] if output_hidden_states else outputs[1:])
            return tuple(v for v in outputs if v is not None)

        # 返回 FlaxBaseModelOutput 类型的输出，包括最后一个隐藏状态、隐藏状态列表和注意力矩阵
        return FlaxBaseModelOutput(
            last_hidden_state=last_hidden_states,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
        )
# 定义 FlaxWhisperDecoder 类，继承自 nn.Module
class FlaxWhisperDecoder(nn.Module):
    # 初始化类变量
    config: WhisperConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    # 模型的设置方法
    def setup(self) -> None:
        # 初始化词嵌入层
        self.embed_tokens = nn.Embed(self.config.vocab_size, self.config.d_model, dtype=self.dtype)
        # 初始化位置嵌入层
        self.embed_positions = nn.Embed(self.config.max_target_positions, self.config.d_model, dtype=self.dtype)

        # 初始化解码器层集合
        self.layers = FlaxWhisperDecoderLayerCollection(
            self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )

        # 初始化 Dropout 层
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)

        # 初始化 LayerNorm 层
        self.layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-5)

    # 定义模型的调用方法
    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: jnp.ndarray,
        position_ids: jnp.ndarray,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray]:
        # 获取输入词嵌入和位置嵌入
        input_embeds = self.embed_tokens(input_ids)
        position_embeds = self.embed_positions(position_ids)

        # 计算隐藏状态
        hidden_states = input_embeds + position_embeds
        # 应用 Dropout
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)

        # 调用解码器层
        outputs = self.layers(
            hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取最终隐藏状态并应用 LayerNorm
        last_hidden_states = outputs[0]
        last_hidden_states = self.layer_norm(last_hidden_states)

        # 更新 `hidden_states` 中的最后一个元素，在上面应用 `layernorm` 后
        hidden_states = None
        if output_hidden_states:
            hidden_states = outputs[1]
            hidden_states = hidden_states[:-1] + (last_hidden_states,)

        # 根据返回值设置输出结果
        if not return_dict:
            outputs = (last_hidden_states, hidden_states) + (outputs[2:] if output_hidden_states else outputs[1:])
            return tuple(v for v in outputs if v is not None)

        # 返回 FlaxBaseModelOutputWithPastAndCrossAttentions 对象
        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=last_hidden_states,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


# 定义 FlaxWhisperModule 类，继承自 nn.Module
class FlaxWhisperModule(nn.Module):
    # 初始化类变量
    config: WhisperConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False
    # 设置模型的初始化函数
    def setup(self) -> None:
        # 初始化编码器对象
        self.encoder = FlaxWhisperEncoder(
            self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )
        # 初始化解码器对象
        self.decoder = FlaxWhisperDecoder(
            self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )

    # 定义模型调用方法
    def __call__(
        self,
        input_features: jnp.ndarray,
        decoder_input_ids: jnp.ndarray,
        decoder_attention_mask: jnp.ndarray,
        decoder_position_ids: jnp.ndarray,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
    ):
        # 对输入特征进行编码
        encoder_outputs = self.encoder(
            input_features,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        # 对解码器进行解码
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            position_ids=decoder_position_ids,
            encoder_hidden_states=encoder_outputs[0],
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        # 如果不返回字典，则返回解码器和编码器的输出
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

    # 获取编码器模块
    def _get_encoder_module(self):
        return self.encoder

    # 获取解码器模块
    def _get_decoder_module(self):
        return self.decoder
# 这是一个预训练的 Whisper 模型的实现类
class FlaxWhisperPreTrainedModel(FlaxPreTrainedModel):
    # 指定配置类为 WhisperConfig
    config_class = WhisperConfig
    # 模型前缀为 "model"
    base_model_prefix: str = "model"
    # 主要输入名称为 "input_features"
    main_input_name = "input_features"
    # 模型类为 nn.Module 的子类
    module_class: nn.Module = None

    def __init__(
        self,
        config: WhisperConfig,
        input_shape: Tuple[int] = None,
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        gradient_checkpointing: bool = False,
        **kwargs,
    ):
        # 使用配置、数据类型和梯度检查点创建模块实例
        module = self.module_class(config=config, dtype=dtype, gradient_checkpointing=gradient_checkpointing, **kwargs)
        # 如果未指定输入形状，则设置默认值
        if input_shape is None:
            input_shape = (1, config.num_mel_bins, 2 * config.max_source_positions)
        # 调用父类的初始化方法
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    # 启用梯度检查点
    def enable_gradient_checkpointing(self):
        self._module = self.module_class(
            config=self.config,
            dtype=self.dtype,
            gradient_checkpointing=True,
        )

    # 初始化模型权重
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 创建输入特征张量
        input_features = jnp.zeros(input_shape, dtype="f4")
        input_features = input_features.at[(..., -1)].set(self.config.eos_token_id)

        # 创建解码器输入 ID 和注意力掩码
        decoder_input_ids = jnp.zeros((input_shape[0], 1), dtype="i4")
        decoder_attention_mask = jnp.ones_like(decoder_input_ids)

        # 创建解码器位置 ID
        batch_size, sequence_length = decoder_input_ids.shape
        decoder_position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # 将随机数生成器分成两部分
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 使用随机参数初始化模型
        random_params = self.module.init(
            rngs,
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
        )["params"]

        # 如果提供了参数，则合并缺失的键
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    # 复制自 transformers.models.bart.modeling_flax_bart.FlaxBartPreTrainedModel.init_cache，但将 Bart 替换为 Whisper
    # 初始化缓存，用于快速自回归解码的批处理大小
    def init_cache(self, batch_size, max_length, encoder_outputs):
        r"""
        Args:
            batch_size (`int`):
                用于快速自回归解码的批处理大小。定义了初始化缓存的批处理大小。
            max_length (`int`):
                自回归解码的最大可能长度。定义了初始化缓存的序列长度。
            encoder_outputs (`Union[FlaxBaseModelOutput, tuple(tuple(jnp.ndarray)]`):
                `encoder_outputs` 包含 (`last_hidden_state`, *可选*: `hidden_states`, *可选*: `attentions`)。
                `last_hidden_state` 的形状为 `(batch_size, sequence_length, hidden_size)`，*可选* 是编码器最后一层的输出的隐藏状态序列。用于解码器的交叉注意力。
        """
        # 初始化输入变量以检索缓存
        decoder_input_ids = jnp.ones((batch_size, max_length), dtype="i4")
        decoder_attention_mask = jnp.ones_like(decoder_input_ids)
        decoder_position_ids = jnp.broadcast_to(
            jnp.arange(jnp.atleast_2d(decoder_input_ids).shape[-1]), decoder_input_ids.shape
        )

        # 定义用于解码器前向传播的函数
        def _decoder_forward(module, decoder_input_ids, decoder_attention_mask, decoder_position_ids, **kwargs):
            decoder_module = module._get_decoder_module()
            return decoder_module(
                decoder_input_ids,
                decoder_attention_mask,
                decoder_position_ids,
                **kwargs,
            )

        # 初始化变量以获得缓存
        init_variables = self.module.init(
            jax.random.PRNGKey(0),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            encoder_hidden_states=encoder_outputs[0],
            init_cache=True,
            method=_decoder_forward,  # 我们只需要调用解码器来初始化缓存
        )
        # 返回解冻的缓存
        return unfreeze(init_variables["cache"])

    @add_start_docstrings(WHISPER_ENCODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutput, config_class=WhisperConfig)
    def encode(
        self,
        input_features: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
        **kwargs,
    ):

        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import WhisperProcessor, FlaxWhisperForConditionalGeneration
        >>> from datasets import load_dataset

        >>> processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = FlaxWhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en", from_pt=True)
        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="np")
        >>> input_features = inputs.input_features
        >>> encoder_outputs = model.encode(input_features=input_features)
        ```py"""

        # 设置输出注意力的开关
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置输出隐藏状态的开关
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置返回字典的开关
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 处理需要的 PRNG（伪随机数生成器）
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        # 定义编码器前向传播函数
        def _encoder_forward(module, input_features, **kwargs):
            encode_module = module._get_encoder_module()
            return encode_module(input_features, **kwargs)

        # 应用编码器模块的前向传播函数
        return self.module.apply(
            {"params": params or self.params},
            input_features=jnp.array(input_features, dtype="f4"),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=not train,
            rngs=rngs,
            method=_encoder_forward,
        )

    # 添加 WHISPER_DECODE_INPUTS_DOCSTRING 的注释文档
    @add_start_docstrings(WHISPER_DECODE_INPUTS_DOCSTRING)
    # 替换返回文档中的输出类型和配置类
    @replace_return_docstrings(output_type=FlaxBaseModelOutputWithPastAndCrossAttentions, config_class=WhisperConfig)
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
    # 添加 WHISPER_INPUTS_DOCSTRING 的注释文档
    @add_start_docstrings_to_model_forward(WHISPER_INPUTS_DOCSTRING)
    # 定义一个方法，用于执行模型的推理或训练
    def __call__(
        self,
        input_features: jnp.ndarray,  # 输入特征，使用 jax numpy 数组
        decoder_input_ids: jnp.ndarray,  # 解码器的输入标识，使用 jax numpy 数组
        attention_mask: Optional[jnp.ndarray] = None,  # 可选的注意力掩码，默认为 None
        decoder_attention_mask: Optional[jnp.ndarray] = None,  # 可选的解码器注意力掩码，默认为 None
        position_ids: Optional[jnp.ndarray] = None,  # 可选的位置标识，默认为 None
        decoder_position_ids: Optional[jnp.ndarray] = None,  # 可选的解码器位置标识，默认为 None
        output_attentions: Optional[bool] = None,  # 是否输出注意力矩阵，可选，默认为 None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选，默认为 None
        return_dict: Optional[bool] = None,  # 是否返回字典类型的结果，默认为 None
        train: bool = False,  # 是否处于训练状态，默认为 False
        params: dict = None,  # 参数字典，默认为 None
        dropout_rng: PRNGKey = None,  # dropout 的随机数生成器，默认为 None
    ):
        # 如果未指定输出注意力矩阵，则使用配置中的设定
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未指定输出隐藏状态，则使用配置中的设定
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定返回结果的形式，则使用配置中的设定
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 准备解码器的输入
        if decoder_position_ids is None:
            # 如果解码器位置标识为空
            if decoder_attention_mask is not None:
                # 如果解码器注意力掩码不为空
                # 计算解码器位置标识
                decoder_position_ids = (decoder_attention_mask.cumsum(-1) * decoder_attention_mask) - 1
            else:
                # 如果解码器注意力掩码为空
                # 获取解码器输入的批处理大小和序列长度
                batch_size, sequence_length = decoder_input_ids.shape
                # 生成解码器位置标识
                decoder_position_ids = jnp.broadcast_to(
                    jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
                )
        if decoder_attention_mask is None:
            # 如果解码器注意力掩码为空
            # 初始化解码器注意力掩码
            decoder_attention_mask = jnp.ones_like(decoder_input_ids)

        # 处理可能存在的随机数生成器
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        # 执行模型的应用，传入相关参数
        return self.module.apply(
            {"params": params or self.params},
            input_features=jnp.array(input_features, dtype="f4"),
            decoder_input_ids=jnp.array(decoder_input_ids, dtype="i4"),
            decoder_attention_mask=jnp.array(decoder_attention_mask, dtype="i4"),
            decoder_position_ids=jnp.array(decoder_position_ids, dtype="i4"),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=not train,  # 是否确定性的执行（非训练状态）
            rngs=rngs,  # 随机数生成器
        )
# 这是 FlaxWhisperModel 类的定义，它继承自 FlaxWhisperPreTrainedModel 类
@add_start_docstrings(
    "The bare Whisper Model transformer outputting raw hidden-states without any specific head on top.",
    WHISPER_START_DOCSTRING,
)
class FlaxWhisperModel(FlaxWhisperPreTrainedModel):
    # 配置参数
    config: WhisperConfig
    # 计算的数据类型，默认为 float32
    dtype: jnp.dtype = jnp.float32
    # FlaxWhisperModule 类
    module_class = FlaxWhisperModule

# 添加示例代码和配置文档
append_call_sample_docstring(FlaxWhisperModel, _CHECKPOINT_FOR_DOC, FlaxSeq2SeqModelOutput, _CONFIG_FOR_DOC)

# FlaxWhisperForConditionalGenerationModule 类的定义
class FlaxWhisperForConditionalGenerationModule(nn.Module):
    # 配置参数
    config: WhisperConfig
    # 计算的数据类型，默认为 float32
    dtype: jnp.dtype = jnp.float32
    # 是否使用梯度检查点
    gradient_checkpointing: bool = False

    def setup(self) -> None:
        # 创建 FlaxWhisperModule 实例
        self.model = FlaxWhisperModule(
            config=self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )
        # 创建线性层作为输出层
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )

    def _get_encoder_module(self):
        # 获取编码器模块
        return self.model.encoder

    def _get_decoder_module(self):
        # 获取解码器模块
        return self.model.decoder

    def __call__(
        self,
        input_features,
        decoder_input_ids,
        decoder_attention_mask: jnp.ndarray = None,
        decoder_position_ids: jnp.ndarray = None,
        position_ids: jnp.ndarray = None,
        attention_mask: jnp.ndarray = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
    ):
        # 调用 FlaxWhisperModule 的前向传播
        outputs = self.model(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        # 获取隐藏状态
        hidden_states = outputs[0]

        if self.config.tie_word_embeddings:
            # 如果词嵌入共享，则使用解码器的词嵌入
            shared_embedding = self.model.decoder.embed_tokens.variables["params"]["embedding"]
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
        else:
            # 否则使用线性层作为输出层
            lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            # 如果不返回字典，则返回元组
            output = (lm_logits,) + outputs[1:]
            return output

        # 返回 FlaxSeq2SeqLMOutput 对象
        return FlaxSeq2SeqLMOutput(
            logits=lm_logits,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
# 使用指定的文档字符串添加关于 Whisper 模型的说明，带有一个语言建模头部
class FlaxWhisperForConditionalGeneration(FlaxWhisperPreTrainedModel):
    # 模型类别设置为 FlaxWhisperForConditionalGenerationModule
    module_class = FlaxWhisperForConditionalGenerationModule
    # 数据类型设置为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 使用 WHISPER_DECODE_INPUTS_DOCSTRING 添加关于 decode 方法的说明
    @add_start_docstrings(WHISPER_DECODE_INPUTS_DOCSTRING)
    # 使用 FlaxCausalLMOutputWithCrossAttentions 类型替换输出文档字符串，使用 WhisperConfig 类配置
    @replace_return_docstrings(output_type=FlaxCausalLMOutputWithCrossAttentions, config_class=WhisperConfig)
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
    # 定义 generate 方法，生成文本
    def generate(
        self,
        input_features,
        generation_config=None,
        logits_processor=None,
        return_timestamps=None,
        task=None,
        language=None,
        is_multilingual=None,
        **kwargs,
):
    # 如果 generation_config 为 None，则使用 self.generation_config 的值
    if generation_config is None:
        generation_config = self.generation_config

    # 如果 return_timestamps 不为 None，则设置 generation_config.return_timestamps 为 return_timestamps 的值
    if return_timestamps is not None:
        generation_config.return_timestamps = return_timestamps

    # 如果 task 不为 None，则设置 generation_config.task 为 task 的值
    if task is not None:
        generation_config.task = task

    # 如果 is_multilingual 不为 None，则设置 generation_config.is_multilingual 为 is_multilingual 的值
    if is_multilingual is not None:
        generation_config.is_multilingual = is_multilingual

    # 如果 language 不为 None，则设置 generation_config.language 为 language 的值
    if language is not None:
        generation_config.language = language

    # 如果 kwargs 不为 None，并且 "decoder_input_ids" 在 kwargs 中，则设置 decoder_input_length 为 kwargs["decoder_input_ids"] 的长度，否则设置为 1
    if kwargs is not None and "decoder_input_ids" in kwargs:
        decoder_input_length = len(kwargs["decoder_input_ids"])
    else:
        decoder_input_length = 1

    # 初始化 forced_decoder_ids 为空列表
    forced_decoder_ids = []

    # 如果 generation_config.is_multilingual 为真，且 generation_config.language 存在，则将 (1, generation_config.lang_to_id[generation_config.language]) 添加到 forced_decoder_ids，否则将 (1, None) 添加到 forced_decoder_ids
    if hasattr(generation_config, "is_multilingual") and generation_config.is_multilingual:
        if hasattr(generation_config, "language"):
            forced_decoder_ids.append((1, generation_config.lang_to_id[generation_config.language]))
        else:
            forced_decoder_ids.append((1, None))

        # 如果 generation_config.task 存在，则将 (2, generation_config.task_to_id[generation_config.task]) 添加到 forced_decoder_ids，否则将 (2, generation_config.task_to_id["transcribe"]) 添加到 forced_decoder_ids
        if hasattr(generation_config, "task"):
            forced_decoder_ids.append((2, generation_config.task_to_id[generation_config.task]))
        else:
            forced_decoder_ids.append((2, generation_config.task_to_id["transcribe"]))

    # 如果 generation_config.return_timestamps 为真，或者 return_timestamps 存在，则创建一个包含 FlaxWhisperTimeStampLogitsProcessor 对象的列表 logits_processor
    if (hasattr(generation_config, "return_timestamps") and generation_config.return_timestamps) or return_timestamps:
        logits_processor = [
            FlaxWhisperTimeStampLogitsProcessor(generation_config, self.config, decoder_input_length)
        ]
    # 否则，如果 forced_decoder_ids 存在且最后一个元素的第一个值不等于 generation_config.no_timestamps_token_id，则将下一个可用的索引和 generation_config.no_timestamps_token_id 添加到 forced_decoder_ids
    else:
        if forced_decoder_ids and forced_decoder_ids[-1][0] != generation_config.no_timestamps_token_id:
            idx = forced_decoder_ids[-1][0] + 1 if forced_decoder_ids else 1
            forced_decoder_ids.append((idx, generation_config.no_timestamps_token_id))

    # 如果 forced_decoder_ids 的长度大于 0，则设置 generation_config.forced_decoder_ids 为 forced_decoder_ids
    if len(forced_decoder_ids) > 0:
        generation_config.forced_decoder_ids = forced_decoder_ids

    # 调用父类的 generate 方法，返回生成的结果
    return super().generate(
        input_features,
        generation_config,
        logits_processor=logits_processor,
        **kwargs,
    )

# 准备生成所需的输入
def prepare_inputs_for_generation(
    self,
    decoder_input_ids,
    max_length,
    attention_mask: Optional[jax.Array] = None,
    decoder_attention_mask: Optional[jax.Array] = None,
    encoder_outputs=None,
    **kwargs,
        # 初始化缓存
        batch_size, seq_length = decoder_input_ids.shape

        # 使用给定的参数初始化过去的键值
        past_key_values = self.init_cache(batch_size, max_length, encoder_outputs)
        # 注意：通常情况下，我们需要为超出 input_ids.shape[-1] 和小于 cache_length 的位置放入 0 的注意力掩码。
        # 但由于解码器使用的是因果注意力掩码，这些位置已经被掩盖了。
        # 因此，我们可以在这里创建一个单一静态的注意力掩码，这样更有效率
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if decoder_attention_mask is not None:
            # 计算位置编码
            position_ids = decoder_attention_mask.cumsum(-1) - 1
            # 将 decoder_attention_mask 更新到 extended_attention_mask 中对应的位置上
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, decoder_attention_mask, (0, 0))
        else:
            # 如果没有提供 decoder_attention_mask，则使用默认的位置编码
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        return {
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "encoder_attention_mask": attention_mask,
            "decoder_attention_mask": extended_attention_mask,
            "decoder_position_ids": position_ids,
        }

    # 用于生成时更新输入
    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        # 更新过去的键值
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        # 更新解码器的位置编码，增加 1
        model_kwargs["decoder_position_ids"] = model_kwargs["decoder_position_ids"][:, -1:] + 1
        return model_kwargs
# 定义了一个文档字符串常量，用于描述条件生成过程
FLAX_WHISPER_CONDITIONAL_GENERATION_DOCSTRING = r"""
    Returns:

    Transcription example:

    ```python
    >>> from transformers import WhisperProcessor, FlaxWhisperForConditionalGeneration
    >>> from datasets import load_dataset

    >>> processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    >>> model = FlaxWhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en", from_pt=True)
    >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="np")
    >>> input_features = inputs.input_features
    >>> generated_ids = model.generate(input_ids=input_features)
    >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    >>> transcription
    ' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
    ```py
"""

# 用新的文档字符串替换 FlaxWhisperForConditionalGeneration 类的输入和输出文档字符串
overwrite_call_docstring(
    FlaxWhisperForConditionalGeneration, WHISPER_INPUTS_DOCSTRING + FLAX_WHISPER_CONDITIONAL_GENERATION_DOCSTRING
)

# 为 FlaxWhisperForConditionalGeneration 类追加和替换返回值的文档字符串
append_replace_return_docstrings(
    FlaxWhisperForConditionalGeneration, output_type=FlaxSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC
)


# 创建 FlaxWhisperForAudioClassificationModule 类
class FlaxWhisperForAudioClassificationModule(nn.Module):
    # 定义类的属性
    config: WhisperConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    # 初始化函数，用于设置类的属性
    def setup(self) -> None:
        # 创建并初始化一个 FlaxWhisperEncoder 对象
        self.encoder = FlaxWhisperEncoder(
            config=self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )
        # 设置编码-解码模式为假
        self.config.is_encoder_decoder = False
        # 根据配置使用加权层求和的方式计算权重
        num_layers = self.config.num_hidden_layers + 1
        if self.config.use_weighted_layer_sum:
            self.layer_weights = jnp.repeat(1 / num_layers, num_layers)
        # 创建一个具有指定大小和数据类型的全连接层
        self.projector = nn.Dense(self.config.classifier_proj_size, dtype=self.dtype)
        # 创建一个具有指定大小和数据类型的全连接层
        self.classifier = nn.Dense(self.config.num_labels, dtype=self.dtype)

    # 调用函数，用于执行模型的前向传播
    def __call__(
        self,
        input_features,
        encoder_outputs=None,
        output_attentions=None,
        output_hidden_states: bool = True,
        return_dict: bool = True,
        ):
            # 如果输出注意力权重参数不为空，则使用输出的注意力权重参数，否则使用预设的模型配置中的输出注意力参数
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            # 如果输出隐藏状态参数不为空，则使用输出的隐藏状态参数，否则使用预设的模型配置中的输出隐藏状态参数
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            # 如果返回字典参数不为空，则使用返回字典参数，否则使用预设的模型配置中的返回字典参数
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            # 如果编码器输出为空，则调用编码器，传入输入特征和相关参数
            if encoder_outputs is None:
                encoder_outputs = self.encoder(
                    input_features,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

            # 如果使用加权层求和，则对编码器输出进行相应的操作
            if self.config.use_weighted_layer_sum:
                hidden_states = jnp.stack(encoder_outputs, axis=1)
                norm_weights = jax.nn.softmax(self.layer_weights, axis=-1)
                hidden_states = jnp.sum(hidden_states * jnp.reshape(norm_weights, [-1, 1, 1]), axis=1)
            else:
                hidden_states = encoder_outputs[0]

            # 对隐藏状态进行投影
            hidden_states = self.projector(hidden_states)
            # 对隐藏状态进行池化操作
            pooled_output = jnp.mean(hidden_states, axis=1)

            # 对池化后的结果进行分类
            logits = self.classifier(pooled_output)

            # 如果不需要返回字典，则返回分类结果和编码器输出的其他部分
            if not return_dict:
                return (logits,) + encoder_outputs[1:]

            # 如果需要返回字典，则构造相应的输出字典对象
            return FlaxSequenceClassifierOutput(
                logits=logits,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )
# 为音频分类头部的 Whisper 模型添加文档字符串
@add_start_docstrings("The Whisper Model with an audio classification head on top.", WHISPER_START_DOCSTRING)
class FlaxWhisperForAudioClassification(FlaxWhisperPreTrainedModel):
    module_class = FlaxWhisperForAudioClassificationModule
    dtype: jnp.dtype = jnp.float32

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量
        input_features = jnp.zeros(input_shape, dtype="f4")
        input_features = input_features.at[(..., -1)].set(self.config.eos_token_id)

        # 分割随机数生成器
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 使用初始化后的输入特征初始化参数
        random_params = self.module.init(
            rngs,
            input_features=input_features,
        )["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            # 冻结参数然后返回
            return freeze(unflatten_dict(params))
        else:
            # 否则返回随机参数
            return random_params

    @add_start_docstrings_to_model_forward(WHISPER_INPUTS_DOCSTRING)
    def __call__(
        self,
        input_features: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 处理任何 PRNG（伪随机数生成器）如果需要
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        # 使用模型的参数进行前向传播
        return self.module.apply(
            {"params": params or self.params},
            input_features=jnp.array(input_features, dtype="f4"),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            rngs=rngs,
        )


# 返回音频分类应用的 Whisper 预训练模型的说明文档
FLAX_WHISPER_AUDIO_CLASSIFICATION_DOCSTRING = r"""
    Returns:

    Transcription example:

    ```python
    >>> import jax.numpy as jnp
    >>> from transformers import AutoFeatureExtractor, FlaxWhisperForAudioClassification
    >>> from datasets import load_dataset

    >>> feature_extractor = AutoFeatureExtractor.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")
    >>> model = FlaxWhisperForAudioClassification.from_pretrained(
    # 加载数据集
    ds = load_dataset("google/fleurs", "all", split="validation", streaming=True)
    
    # 获取一个样本
    sample = next(iter(ds))
    
    # 使用特征提取器提取特征
    inputs = feature_extractor(
        sample["audio"]["array"], sampling_rate=sample["audio"]["sampling_rate"], return_tensors="np"
    )
    input_features = inputs.input_features
    
    # 使用模型进行推理
    logits = model(input_features).logits
    
    # 预测类别
    predicted_class_ids = jnp.argmax(logits).item()
    
    # 获取类别标签
    predicted_label = model.config.id2label[predicted_class_ids]
    predicted_label
# 调用 overwrite_call_docstring 方法，将 FlaxWhisperForAudioClassification 的文档字符串替换为 WHISPER_INPUTS_DOCSTRING 和 FLAX_WHISPER_AUDIO_CLASSIFICATION_DOCSTRING 的内容
overwrite_call_docstring(
    FlaxWhisperForAudioClassification, WHISPER_INPUTS_DOCSTRING + FLAX_WHISPER_AUDIO_CLASSIFICATION_DOCSTRING
)

# 调用 append_replace_return_docstrings 方法，将 FlaxWhisperForAudioClassification 的返回文档字符串替换为 FlaxSequenceClassifierOutput，并添加配置类 _CONFIG_FOR_DOC 的文档字符串
append_replace_return_docstrings(
    FlaxWhisperForAudioClassification, output_type=FlaxSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC
)
```