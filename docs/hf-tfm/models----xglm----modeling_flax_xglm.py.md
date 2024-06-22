# `.\transformers\models\xglm\modeling_flax_xglm.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 使用 Apache 许可版本 2.0 许可
# 如果没有遵守许可证，不得使用此文件
# 可以在以下网址获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或经书面同意，否则不得分发包含此文件的软件
# 软件基于“现状”分发，没有任何明示或默示的保证或条件
# 请阅读特定语言的许可证，了解权限和限制
""" Flax XGLM 模型。"""

# 导入模块
import math
import random
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
from jax.random import PRNGKey

from ...modeling_flax_outputs import (
    FlaxBaseModelOutputWithPastAndCrossAttentions,
    FlaxCausalLMOutputWithCrossAttentions,
)
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_xglm import XGLMConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 文档中需要的示例模型和配置
_CHECKPOINT_FOR_DOC = "facebook/xglm-564M"
_CONFIG_FOR_DOC = "XGLMConfig"

XGLM_START_DOCSTRING = r"""
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
    # 定义函数参数和类型提示
    # config：[`XGLMConfig`]，模型配置类，包含模型的所有参数
    #         使用配置文件进行初始化不会加载模型关联的权重，只会加载配置。查看[`~FlaxPreTrainedModel.from_pretrained`]方法以加载模型权重。
    # dtype：`jax.numpy.dtype`，*可选参数*，默认为`jax.numpy.float32`
    #        计算的数据类型。可以是`jax.numpy.float32`、`jax.numpy.float16`（在 GPU 上）、`jax.numpy.bfloat16`（在 TPU 上）之一。
    #        这可用于在 GPU 或 TPU 上启用混合精度训练或半精度推理。如果指定，所有计算将使用给定的`dtype`进行。
    #        **请注意，这仅指定计算的dtype，不影响模型参数的dtype。**
    #        如果您希望更改模型参数的dtype，请参阅[`~FlaxPreTrainedModel.to_fp16`]和[`~FlaxPreTrainedModel.to_bf16`]。
"""
# XGLM_INPUTS_DOCSTRING是一个文档字符串，用于描述函数create_sinusoidal_positions的输入参数的作用和格式
XGLM_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
            输入序列标记在词汇表中的索引。默认情况下将忽略填充。

            可以使用`AutoTokenizer`获取索引。有关详情，请参见`PreTrainedTokenizer.encode`和`PreTrainedTokenizer.__call__`。

            [什么是输入ID？](../glossary#input-ids)
        attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            避免对填充标记索引执行注意力的掩码。选择范围在`[0, 1]`：

            - 1表示**未屏蔽**的标记，
            - 0表示**屏蔽**的标记。

            [什么是注意力掩码？](../glossary#attention-mask)
        position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            输入序列标记在位置嵌入中的位置索引。选择范围在`[0, config.max_position_embeddings - 1]`。
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。有关更多细节，请参见返回张量中的`attentions`。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。有关更多细节，请参见返回张量中的`hidden_states`。
        return_dict (`bool`, *optional*):
            是否返回[`~utils.ModelOutput`]而不是普通元组。
"""

# 创建正弦位置编码，用于Transformer模型的位置编码
def create_sinusoidal_positions(n_pos, dim, padding_idx=1):
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = np.exp(np.arange(half_dim) * -emb)
    emb = np.expand_dims(np.arange(n_pos), 1) * np.expand_dims(emb, 0)
    emb = np.concatenate([np.sin(emb), np.cos(emb)], 1)
    emb = np.reshape(emb, (n_pos, dim))

    if padding_idx is not None:
        emb[padding_idx, :] = 0

    return jnp.array(emb)

# 定义FlaxXGLMAttention类，用于XGLM模型的注意力机制
class FlaxXGLMAttention(nn.Module):
    config: XGLMConfig
    embed_dim: int
    num_heads: int
    dropout: float = 0.0
    causal: bool = False
    bias: bool = True
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型
    # 在模型设置时进行的操作
    def setup(self) -> None:
        # 将每个头所需的维度计算出来
        self.head_dim = self.embed_dim // self.num_heads

        # 检查 embed_dim 能否被 num_heads 整除
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} "
                f"and `num_heads`: {self.num_heads})."
            )

        # 定义一个 nn.Dense 的偏函数 dense，为之后创建全连接层做准备
        dense = partial(
            nn.Dense,
            self.embed_dim,
            use_bias=self.bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )

        # 创建输入 q, k, v 的全连接层
        self.q_proj, self.k_proj, self.v_proj = dense(), dense(), dense()

        # 创建输出层全连接层
        self.out_proj = dense()

        # 创建一个丢弃层
        self.dropout_layer = nn.Dropout(rate=self.dropout)

        # 如果是因果的注意力机制，创建一个因果遮盖
        if self.causal:
            self.causal_mask = make_causal_mask(
                jnp.ones((1, self.config.max_position_embeddings), dtype="bool"), dtype="bool"
            )

    # 将 hidden_states 沿着最后两个维度拆分成 (batch_size, num_heads, seq_len, head_dim) 的形状
    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    # 将 hidden_states 沿着倒数第二个维度合并成 (batch_size, seq_len, embed_dim) 的形状
    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    # 使用 nn.compact 声明一个层的子类
    @nn.compact
    # 将新计算的 key 和 value 添加到缓存中
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        # 检查是否已初始化缓存
        is_initialized = self.has_variable("cache", "cached_key")
        # 获取缓存的 key 和 value
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        # 获取缓存索引
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))
    
        # 如果已初始化
        if is_initialized:
            # 获取缓存的最大长度和头数
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # 更新缓存的 key 和 value
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            # 更新缓存索引
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # 更新注意力掩码以考虑缓存
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)
    
        return key, value, attention_mask
    
    # 这是自注意力层的前向传播
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        key_value_states: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        deterministic: bool = True,
    ):
# 定义FlaxXGLMDecoderLayer类
class FlaxXGLMDecoderLayer(nn.Module):
    # XGLMConfig类型的配置参数
    config: XGLMConfig
    # 数据类型，默认为float32
    dtype: jnp.dtype = jnp.float32

    # 初始化函数
    def setup(self) -> None:
        # 嵌入维度等于配置参数的d_model
        self.embed_dim = self.config.d_model
        # 自注意力机制
        self.self_attn = FlaxXGLMAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.attention_heads,
            dropout=self.config.attention_dropout,
            causal=True,
            dtype=self.dtype,
        )
        # 对自注意力结果进行层归一化
        self.self_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        # dropout层
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)
        # 激活函数，默认为config.activation_function
        self.activation_fn = ACT2FN[self.config.activation_function]
        # 激活函数后的dropout层
        self.activation_dropout_layer = nn.Dropout(rate=self.config.activation_dropout)

        # 如果需要添加跨注意力机制
        if self.config.add_cross_attention:
            # 跨注意力机制
            self.encoder_attn = FlaxXGLMAttention(
                config=self.config,
                embed_dim=self.embed_dim,
                num_heads=self.config.decoder_attention_heads,
                dropout=self.config.attention_dropout,
                dtype=self.dtype,
            )
            # 对跨注意力机制结果进行层归一化
            self.encoder_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

        # 第一个全连接层
        self.fc1 = nn.Dense(
            self.config.ffn_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        
        # 第二个全连接层
        self.fc2 = nn.Dense(
            self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std)
        )
        # 全连接层输出结果进行层归一化
        self.final_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    # 调用函数
    # 从transformers库中的FlaxMBartDecoderLayer类进行修改
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        output_attentions: bool = True,
        deterministic: bool = True,
    # 定义函数，接收隐藏层状态作为输入，返回元组中包含一个 JAX 数组的输出
    ) -> Tuple[jnp.ndarray]:
        # 将隐藏层状态复制给残差变量
        residual = hidden_states
        # 对隐藏状态进行自注意力机制的规范化
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 自注意力机制
        # 调用 self_attn 方法处理隐藏状态，并得到自注意力权重
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, init_cache=init_cache
        )
        # 对隐藏状态进行 dropout 操作
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 将残差与新的隐藏状态相加
        hidden_states = residual + hidden_states

        # 交叉注意力块
        cross_attn_weights = None
        # 如果存在编码器的隐藏状态
        if encoder_hidden_states is not None:
            # 将隐藏状态复制给残差变量
            residual = hidden_states
            # 对隐藏状态进行编码器注意力机制的规范化
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            # 调用 encoder_attn 方法处理隐藏状态，并得到交叉注意力权重
            hidden_states, cross_attn_weights = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            # 对隐藏状态进行 dropout 操作
            hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
            # 将残差与新的隐藏状态相加
            hidden_states = residual + hidden_states

        # 全连接层
        # 将隐藏状态复制给残差变量
        residual = hidden_states
        # 对隐藏状态进行最终层规范化
        hidden_states = self.final_layer_norm(hidden_states)
        # 对隐藏状态进行激活函数处理
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 对隐藏状态进行激活函数的 dropout 操作
        hidden_states = self.activation_dropout_layer(hidden_states, deterministic=deterministic)
        # 经过第二个全连接层的处理
        hidden_states = self.fc2(hidden_states)
        # 对隐藏状态进行 dropout 操作
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 将残差与新的隐藏状态相加
        hidden_states = residual + hidden_states

        # 输出
        outputs = (hidden_states,)

        # 如果需要输出注意力权重
        if output_attentions:
            # 将自注意力权重和交叉注意力权重加入���输出中
            outputs += (self_attn_weights, cross_attn_weights)

        # 返回结果输出
        return outputs
# 定义一个名为FlaxXGLMDecoderLayerCollection的类，继承自nn.Module类
class FlaxXGLMDecoderLayerCollection(nn.Module):
    # 类属性：XGLMConfig类型的config变量
    config: XGLMConfig
    # 类属性：jnp.float32类型的dtype变量，默认为jnp.float32
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    # 初始化方法
    def setup(self):
        # 生成包含self.config.num_layers个FlaxXGLMDecoderLayer对象的列表，命名为self.layers
        self.layers = [
            FlaxXGLMDecoderLayer(self.config, name=str(i), dtype=self.dtype) for i in range(self.config.num_layers)
        ]
        # 设置类属性self.layerdrop为self.config.layerdrop
        self.layerdrop = self.config.layerdrop

    # 调用方法
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
        # 如果output_hidden_states为True，则初始化空的元组all_hidden_states；否则初始化为None
        all_hidden_states = () if output_hidden_states else None
        # 如果output_attentions为True，则初始化空的元组all_self_attns；否则初始化为None
        all_self_attns = () if output_attentions else None
        # 如果output_attentions为True且encoder_hidden_states不为None，则初始化空的元组all_cross_attentions；否则初始化为None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        # 遍历self.layers中的每个decoder_layer
        for decoder_layer in self.layers:
            # 如果output_hidden_states为True，则将hidden_states添加到all_hidden_states
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                # 添加LayerDrop（参考https://arxiv.org/abs/1909.11556）
            # 生成0到1之间的随机浮点数dropout_probability
            dropout_probability = random.uniform(0, 1)
            # 如果非确定性计算且dropout_probability小于self.layerdrop，则将layer_outputs设置为(None, None, None)
            if not deterministic and (dropout_probability < self.layerdrop):
                layer_outputs = (None, None, None)
            # 否则将layer_outputs设置为decoder_layer的输出
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

            # 更新hidden_states为layer_outputs的第一个元素
            hidden_states = layer_outputs[0]
            # 如果output_attentions为True，则将layer_outputs的第二个元素添加到all_self_attns
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                # 如果encoder_hidden_states不为None，则将layer_outputs的第三个元素添加到all_cross_attentions
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # 如果output_hidden_states为True，则将hidden_states添加到all_hidden_states
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 将结果存储在outputs中
        outputs = (hidden_states, all_hidden_states, all_self_attns, all_cross_attentions)

        # 如果return_dict为False，则返回outputs中的非None值组成的元组
        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        # 否则返回FlaxBaseModelOutputWithPastAndCrossAttentions类型的对象
        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


# 定义一个名为FlaxXGLMModule的类，继承自nn.Module类
class FlaxXGLMModule(nn.Module):
    # 类属性：XGLMConfig类型的config变量
    config: XGLMConfig
    # 类属性：jnp.float32类型的dtype变量，默认为jnp.float32
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
        # 设置 dropout 层，丢弃率为配置文件中指定的值
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)

        # 初始化词嵌入维度为配置文件中指定的模型维度
        embed_dim = self.config.d_model
        # 获取填充索引，用于填充序列
        self.padding_idx = self.config.pad_token_id
        # 获取最大目标位置，用于生成位置编码
        self.max_target_positions = self.config.max_position_embeddings
        # 计算词嵌入的缩放因子，如果配置文件中指定了缩放，则为模型维度的平方根，否则为1.0
        self.embed_scale = math.sqrt(self.config.d_model) if self.config.scale_embedding else 1.0

        # 初始化词嵌入层，词汇大小为配置文件中指定的值，词嵌入维度为模型维度，
        # 初始化方式为正态分布初始化，标准差为配置文件中指定的值
        self.embed_tokens = nn.Embed(
            self.config.vocab_size,
            embed_dim,
            embedding_init=jax.nn.initializers.normal(self.config.init_std),
        )

        # XGLM 模型的设定是如果指定了填充索引，则将嵌入id偏移2，并相应调整num_embeddings。
        # 其他模型没有这种处理方式
        self.offset = 2
        # 初始化位置编码，最大位置编码为配置文件中指定的最大位置编码加上偏移值，维度为词嵌入维度
        self.embed_positions = create_sinusoidal_positions(
            self.config.max_position_embeddings + self.offset, embed_dim
        )
        # 初始化 XGLM 解码器层集合，包括多个解码器层
        self.layers = FlaxXGLMDecoderLayerCollection(self.config, self.dtype)
        # 初始化层归一化，数据类型为配置文件中指定的数据类型，ε为1e-05
        self.layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
    # 获取输入的形状
        input_shape = input_ids.shape
        # 将输入的 ID 重塑为二维数组
        input_ids = input_ids.reshape(-1, input_shape[-1])

        # 对输入 ID 进行嵌入处理，并乘以嵌入比例
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # 嵌入位置信息
        position_ids = position_ids + self.offset
        positions = jnp.take(self.embed_positions, position_ids, axis=0)

        # 将嵌入的输入和位置信息相加，得到隐藏状态
        hidden_states = inputs_embeds + positions
        # 对隐藏状态进行丢弃层处理
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)

        # 通过层处理输入，得到输出
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

        # 获取最后的隐藏状态
        last_hidden_states = outputs[0]
        # 对最后的隐藏状态进行层归一化处理
        last_hidden_states = self.layer_norm(last_hidden_states)

        # 初始化隐藏状态为空
        hidden_states = None
        # 如果需要输出所有隐藏状态
        if output_hidden_states:
            # 获取所有隐藏状态
            hidden_states = outputs[1]
            # 将最后的隐藏状态添加到所有隐藏状态里面
            hidden_states = hidden_states[:-1] + (last_hidden_states,)

        # 如果不需要返回字典类型结果
        if not return_dict:
            # 构建输出结果
            outputs = (last_hidden_states, hidden_states) + (outputs[2:] if output_hidden_states else outputs[1:])
            return tuple(v for v in outputs if v is not None)

        # 返回带有过去和交叉注意力的 FlaxBaseModelOutputWithPastAndCrossAttentions 结果
        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=last_hidden_states,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )
``` 
class FlaxXGLMPreTrainedModel(FlaxPreTrainedModel):
    # 定义基于 XGLMConfig 的模型配置类
    config_class = XGLMConfig
    # 基本模型前缀
    base_model_prefix: str = "model"
    # 模块类别
    module_class: nn.Module = None

    def __init__(
        self,
        config: XGLMConfig,
        input_shape: Tuple[int] = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        # 创建模型模块
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        if self.config.add_cross_attention:
            encoder_hidden_states = jnp.zeros(input_shape + (self.config.n_embd,))
            encoder_attention_mask = attention_mask
            # 初始化模块并获取输出
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
            # 初始化模块并获取输出
            module_init_outputs = self.module.init(rngs, input_ids, attention_mask, position_ids, return_dict=False)

        random_params = module_init_outputs["params"]

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
        input_ids = jnp.ones((batch_size, max_length), dtype="i4")
        attention_mask = jnp.ones_like(input_ids, dtype="i4")
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        # 初始化模块并获取输出缓存
        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        return unfreeze(init_variables["cache"])
    # 给模型的前向传播函数添加文档字符串，文档字符串内容为 XGLM_INPUTS_DOCSTRING
    @add_start_docstrings_to_model_forward(XGLM_INPUTS_DOCSTRING)
    # 定义模型的前向传播函数
    def __call__(
        # 定义输入数据的变量类型和默认值
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        past_key_values: dict = None,
        dropout_rng: PRNGKey = None,
        # 检查是否输出注意力权重，如果未指定则使用配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 检查是否输出隐藏状态，如果未指定则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 检查是否返回字典型输出，如果未指定则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 如果传入了 encoder_hidden_states 但没有 encoder_attention_mask，则创建一个全 1 的注意力掩码
        if encoder_hidden_states is not None and encoder_attention_mask is None:
            batch_size, sequence_length = encoder_hidden_states.shape[:2]
            encoder_attention_mask = jnp.ones((batch_size, sequence_length))

        # 准备编码器的输入
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        if position_ids is None:
            batch_size, sequence_length = input_ids.shape
            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # 处理任何 PRNG（伪随机数生成器）需求
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        inputs = {"params": params or self.params}

        # 如果传入了 past_key_values，则 cache 已经初始化，需要传入 init_cache 标志以确保使用缓存
        # 必须确保将 cache 标记为可变，以便 FlaxXGLMAttention 模块可以修改它
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        # 应用模块
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

        # 将更新后的 cache 添加到模型输出中
        if past_key_values is not None and return_dict:
            outputs, past_key_values = outputs
            outputs["past_key_values"] = unfreeze(past_key_values["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs, past_key_values = outputs
            outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

        return outputs
# 导入必要的模块和函数
@add_start_docstrings(
    "The bare XGLM Model transformer outputting raw hidden-states without any specific head on top.",
    XGLM_START_DOCSTRING,
)
# 定义一个类，用于替换具有特定头部的原始隐藏状态的XGLM模型
class FlaxXGLMModel(FlaxXGLMPreTrainedModel):
    module_class = FlaxXGLMModule

# 在调用样例文档字符串中添加FlaxXGLMModel相关信息
append_call_sample_docstring(
    FlaxXGLMModel,
    _CHECKPOINT_FOR_DOC,
    FlaxBaseModelOutputWithPastAndCrossAttentions,
    _CONFIG_FOR_DOC,
)

# 定义一个FlaxXGLMForCausalLMModule类
class FlaxXGLMForCausalLMModule(nn.Module):
    config: XGLMConfig
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型

    def setup(self):
        self.model = FlaxXGLMModule(self.config, self.dtype)
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
        # 调用模型处理输入数据
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
            shared_embedding = self.model.variables["params"]["embed_tokens"]["embedding"]
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (lm_logits,) + outputs[1:]

        return FlaxCausalLMOutputWithCrossAttentions(
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

# 添加FlaxXGLMForCausalLM类的文档字符串
@add_start_docstrings(
    """
    The XGLM Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    XGLM_START_DOCSTRING,
)
# 定义FlaxXGLMForCausalLM类
class FlaxXGLMForCausalLM(FlaxXGLMPreTrainedModel):
    module_class = FlaxXGLMForCausalLMModule
    # 准备模型生成所需的输入数据
    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = None):
        # 初始化缓存
        batch_size, seq_length = input_ids.shape
        past_key_values = self.init_cache(batch_size, max_length)
        
        # 创建一个大小为 (batch_size, max_length) 的全 1 attention mask
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        
        # 如果提供了 attention_mask, 则更新 extended_attention_mask
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        # 如果没有提供 attention_mask, 则创建一个 position_ids 张量
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))
    
        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }
    
    # 更新模型生成所需的输入数据
    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        # 更新 past_key_values
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        # 更新 position_ids
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs
# 调用函数append_call_sample_docstring，传入参数为FlaxXGLMForCausalLM、_CHECKPOINT_FOR_DOC、FlaxCausalLMOutputWithCrossAttentions和_CONFIG_FOR_DOC
append_call_sample_docstring(
    FlaxXGLMForCausalLM,
    _CHECKPOINT_FOR_DOC,
    FlaxCausalLMOutputWithCrossAttentions,
    _CONFIG_FOR_DOC,
)
```