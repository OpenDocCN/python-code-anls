# `.\models\roformer\modeling_flax_roformer.py`

```py
# 导入必要的库和模块
from typing import Callable, Optional, Tuple  # 导入类型提示相关的模块

import flax.linen as nn  # 导入Flax的linen模块，用于定义模型结构
import jax  # 导入JAX，用于自动求导和并行计算
import jax.numpy as jnp  # 导入JAX的NumPy接口，用于多维数组操作
import numpy as np  # 导入NumPy，用于基本的数值计算
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze  # 导入Flax的FrozenDict等相关模块，用于管理不可变字典
from flax.linen.attention import dot_product_attention_weights  # 导入注意力机制相关模块
from flax.traverse_util import flatten_dict, unflatten_dict  # 导入工具函数，用于扁平化和反扁平化字典结构
from jax import lax  # 导入JAX的lax模块，用于定义低级的线性代数和信号处理原语

# 导入相关模型输出和工具函数
from ...modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxMaskedLMOutput,
    FlaxMultipleChoiceModelOutput,
    FlaxQuestionAnsweringModelOutput,
    FlaxSequenceClassifierOutput,
    FlaxTokenClassifierOutput,
)
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring, overwrite_call_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging

# 导入RoFormer配置文件
from .configuration_roformer import RoFormerConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义用于文档的检查点和配置信息
_CHECKPOINT_FOR_DOC = "junnyu/roformer_chinese_base"
_CONFIG_FOR_DOC = "RoFormerConfig"

# 预训练模型存档列表
FLAX_ROFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "junnyu/roformer_chinese_small",
    "junnyu/roformer_chinese_base",
    "junnyu/roformer_chinese_char_small",
    "junnyu/roformer_chinese_char_base",
    "junnyu/roformer_small_discriminator",
    "junnyu/roformer_small_generator",
    # 查看所有RoFormer模型 https://huggingface.co/models?filter=roformer
]

# RoFormer模型的起始文档字符串，包含模型介绍和Flax特性说明
ROFORMER_START_DOCSTRING = r"""

    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading, saving and converting weights from PyTorch models)

    This model is also a
    [flax.linen.Module](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html) subclass. Use it as
    a regular Flax linen Module and refer to the Flax documentation for all matter related to general usage and
    behavior.

    Finally, this model supports inherent JAX features such as:

    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)
    # Parameters:
    #     config ([`RoFormerConfig`]): 模型配置类，包含模型的所有参数。
    #         初始化时使用配置文件不会加载与模型关联的权重，只加载配置信息。
    #         可查看 [`~FlaxPreTrainedModel.from_pretrained`] 方法来加载模型权重。
    #     dtype (`jax.numpy.dtype`, *optional*, 默认为 `jax.numpy.float32`):
    #         计算的数据类型。可以是 `jax.numpy.float32`、`jax.numpy.float16`（在GPU上）、`jax.numpy.bfloat16`（在TPU上）之一。
    #         
    #         这可以用于在GPU或TPU上启用混合精度训练或半精度推断。如果指定了dtype，则所有计算将使用指定的数据类型。
    #         
    #         **注意，这只指定了计算的数据类型，并不影响模型参数的数据类型。**
    #         
    #         如果想要更改模型参数的数据类型，请参见 [`~FlaxPreTrainedModel.to_fp16`] 和 [`~FlaxPreTrainedModel.to_bf16`]。
"""

ROFORMER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`numpy.ndarray` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`numpy.ndarray` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`numpy.ndarray` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`numpy.ndarray` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.
        head_mask (`numpy.ndarray` of shape `({0})`, `optional):
            Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


# Copied from transformers.models.marian.modeling_flax_marian.create_sinusoidal_positions
def create_sinusoidal_positions(n_pos, dim):
    # 创建一个 sinusoidal 位置编码的函数
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    # 将维度除以2后加上余数，作为分割点
    sentinel = dim // 2 + dim % 2
    # 初始化一个和位置编码形状相同的零矩阵
    out = np.zeros_like(position_enc)
    # 计算 sin 和 cos 的值并填充到输出矩阵
    out[:, 0:sentinel] = np.sin(position_enc[:, 0::2])
    out[:, sentinel:] = np.cos(position_enc[:, 1::2])

    return jnp.array(out)


class FlaxRoFormerEmbeddings(nn.Module):
    """Construct the embeddings from word and token_type embeddings."""

    config: RoFormerConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    # 在对象的设置方法中初始化词嵌入层
    def setup(self):
        # 初始化词嵌入层，使用正态分布初始化，标准差为配置中的初始化范围
        self.word_embeddings = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        # 初始化标记类型嵌入层，使用正态分布初始化，标准差为配置中的初始化范围
        self.token_type_embeddings = nn.Embed(
            self.config.type_vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        # 初始化 Layer Normalization 层，设置 epsilon 为配置中的层归一化 epsilon 值，数据类型为对象的数据类型
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 初始化 Dropout 层，设置丢弃率为配置中的隐藏层丢弃率
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    # 在对象调用方法中执行模型的前向传播
    def __call__(self, input_ids, token_type_ids, attention_mask, deterministic: bool = True):
        # 将输入 ID 转换为整数类型，并进行词嵌入
        inputs_embeds = self.word_embeddings(input_ids.astype("i4"))
        # 将标记类型 ID 转换为整数类型，并进行标记类型嵌入
        token_type_embeddings = self.token_type_embeddings(token_type_ids.astype("i4"))

        # 将词嵌入和标记类型嵌入求和，得到隐藏状态
        hidden_states = inputs_embeds + token_type_embeddings

        # 对隐藏状态进行 Layer Normalization 处理
        hidden_states = self.LayerNorm(hidden_states)
        # 使用 Dropout 进行隐藏状态的随机丢弃，如果 deterministic=True，则使用确定性模式
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        
        # 返回处理后的隐藏状态作为模型的输出
        return hidden_states
class FlaxRoFormerSelfAttention(nn.Module):
    config: RoFormerConfig
    dtype: jnp.dtype = jnp.float32  # 计算时使用的数据类型

    def setup(self) -> None:
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError(
                "`config.hidden_size`: {self.config.hidden_size} has to be a multiple of `config.num_attention_heads` "
                "                   : {self.config.num_attention_heads}"
            )

        # 定义查询向量的全连接层
        self.query = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        # 定义键向量的全连接层
        self.key = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        # 定义值向量的全连接层
        self.value = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

        # 是否使用旋转注意力值
        self.rotary_value = self.config.rotary_value

    def __call__(
        self,
        hidden_states,
        attention_mask,
        sinusoidal_pos,
        layer_head_mask,
        deterministic=True,
        output_attentions: bool = False,
    ):
        # 将每个注意力头的维度设置为隐藏状态的维度除以注意力头的数量
        head_dim = self.config.hidden_size // self.config.num_attention_heads

        # 通过查询操作获取查询状态，并重新整形为 (batch_size, seq_length, num_attention_heads, head_dim)
        query_states = self.query(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )
        
        # 通过值操作获取值状态，并重新整形为 (batch_size, seq_length, num_attention_heads, head_dim)
        value_states = self.value(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )
        
        # 通过键操作获取键状态，并重新整形为 (batch_size, seq_length, num_attention_heads, head_dim)
        key_states = self.key(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )

        # 如果存在 sinusoidal_pos，则应用旋转位置嵌入到查询、键和值状态
        if sinusoidal_pos is not None:
            if self.rotary_value:
                # 如果启用了旋转值，应用旋转位置嵌入到查询、键和值状态
                query_states, key_states, value_states = self.apply_rotary_position_embeddings(
                    sinusoidal_pos, query_states, key_states, value_states
                )
            else:
                # 否则，仅应用旋转位置嵌入到查询和键状态
                query_states, key_states = self.apply_rotary_position_embeddings(
                    sinusoidal_pos, query_states, key_states
                )

        # 将布尔类型的注意力遮罩转换为注意力偏置
        if attention_mask is not None:
            # 在注意力遮罩的基础上扩展一个维度，形状变为 (batch_size, num_attention_heads, seq_length, seq_length)
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
            # 根据注意力遮罩的值生成注意力偏置
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
            )
        else:
            attention_bias = None

        # 初始化 dropout RNG
        dropout_rng = None
        if not deterministic and self.config.attention_probs_dropout_prob > 0.0:
            dropout_rng = self.make_rng("dropout")

        # 计算点积注意力权重
        attn_weights = dot_product_attention_weights(
            query_states,
            key_states,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attention_probs_dropout_prob,
            broadcast_dropout=True,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=None,
        )

        # 如果需要，通过层头掩码屏蔽注意力头
        if layer_head_mask is not None:
            attn_weights = jnp.einsum("...hqk,h->...hqk", attn_weights, layer_head_mask)

        # 计算注意力输出
        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)
        attn_output = attn_output.reshape(attn_output.shape[:2] + (-1,))

        # 根据需要返回注意力输出和注意力权重
        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs

    @staticmethod
    # 将输入的 sinusoidal_pos 张量按最后一个维度分割成两部分，分别表示 sin 和 cos 值
    sin, cos = sinusoidal_pos.split(2, axis=-1)
    # 根据分割后的 sin 值创建新的张量，维度与输入相同
    sin_pos = jnp.stack([sin, sin], axis=-1).reshape(sinusoidal_pos.shape)
    # 根据分割后的 cos 值创建新的张量，维度与输入相同
    cos_pos = jnp.stack([cos, cos], axis=-1).reshape(sinusoidal_pos.shape)

    # 定义函数，用于对输入的层进行旋转操作
    def rotate_layer(layer, sin_pos, cos_pos):
        # 将层按最后一个维度分割成两部分，取反交换顺序后再重组成原始形状的张量
        rotate_half_layer = jnp.stack([-layer[..., 1::2], layer[..., ::2]], axis=-1).reshape(layer.shape)
        # 使用输入的 cos_pos 张量对原始层进行加权叠加，生成旋转后的结果
        rotary_matrix_cos = jnp.einsum("bslh,...sh->bslh", layer, cos_pos)
        # 使用经过交换顺序的 rotate_half_layer 和 sin_pos 张量进行加权叠加，生成旋转后的结果
        rotary_matrix_sin = jnp.einsum("bslh,...sh->bslh", rotate_half_layer, sin_pos)
        # 将 cos 和 sin 叠加得到的结果相加，得到最终旋转后的层
        return rotary_matrix_cos + rotary_matrix_sin

    # 对 query_layer 和 key_layer 应用旋转函数
    query_layer = rotate_layer(query_layer, sin_pos, cos_pos)
    key_layer = rotate_layer(key_layer, sin_pos, cos_pos)
    # 如果提供了 value_layer，则也对其应用旋转函数，并返回旋转后的三个层
    if value_layer is not None:
        value_layer = rotate_layer(value_layer, sin_pos, cos_pos)
        return query_layer, key_layer, value_layer
    # 否则，只返回旋转后的 query_layer 和 key_layer
    return query_layer, key_layer
# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertSelfOutput with Bert->RoFormer
class FlaxRoFormerSelfOutput(nn.Module):
    config: RoFormerConfig  # 用于保存RoFormer模型配置的对象
    dtype: jnp.dtype = jnp.float32  # 计算过程中使用的数据类型，默认为32位浮点数

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,  # 创建具有指定大小的全连接层
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),  # 使用正态分布初始化全连接层的权重
            dtype=self.dtype,  # 指定层的数据类型
        )
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)  # 创建Layer Normalization层
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)  # 创建Dropout层，用于随机丢弃神经元

    def __call__(self, hidden_states, input_tensor, deterministic: bool = True):
        hidden_states = self.dense(hidden_states)  # 使用全连接层处理输入的隐藏状态
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)  # 对处理后的隐藏状态应用Dropout
        hidden_states = self.LayerNorm(hidden_states + input_tensor)  # 应用Layer Normalization到处理后的状态上
        return hidden_states


class FlaxRoFormerAttention(nn.Module):
    config: RoFormerConfig  # 用于保存RoFormer模型配置的对象
    dtype: jnp.dtype = jnp.float32  # 计算过程中使用的数据类型，默认为32位浮点数

    def setup(self):
        self.self = FlaxRoFormerSelfAttention(self.config, dtype=self.dtype)  # 创建RoFormer自注意力层对象
        self.output = FlaxRoFormerSelfOutput(self.config, dtype=self.dtype)  # 创建RoFormer自注意力输出层对象

    def __call__(
        self,
        hidden_states,
        attention_mask,
        sinusoidal_pos,
        layer_head_mask,
        deterministic=True,
        output_attentions: bool = False,
    ):
        # Attention mask comes in as attention_mask.shape == (*batch_sizes, kv_length)
        # FLAX expects: attention_mask.shape == (*batch_sizes, 1, 1, kv_length) such that it is broadcastable
        # with attn_weights.shape == (*batch_sizes, num_heads, q_length, kv_length)
        attn_outputs = self.self(
            hidden_states,
            attention_mask,
            sinusoidal_pos,
            layer_head_mask=layer_head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
        )  # 调用RoFormer自注意力层处理输入
        attn_output = attn_outputs[0]  # 获取自注意力层的输出
        hidden_states = self.output(attn_output, hidden_states, deterministic=deterministic)  # 将自注意力层的输出传递给RoFormer自注意力输出层处理

        outputs = (hidden_states,)  # 准备输出元组，包含处理后的隐藏状态

        if output_attentions:
            outputs += (attn_outputs[1],)  # 如果需要输出注意力权重，则添加到输出元组中

        return outputs


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertIntermediate with Bert->RoFormer
class FlaxRoFormerIntermediate(nn.Module):
    config: RoFormerConfig  # 用于保存RoFormer模型配置的对象
    dtype: jnp.dtype = jnp.float32  # 计算过程中使用的数据类型，默认为32位浮点数

    def setup(self):
        self.dense = nn.Dense(
            self.config.intermediate_size,  # 创建具有指定大小的全连接层
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),  # 使用正态分布初始化全连接层的权重
            dtype=self.dtype,  # 指定层的数据类型
        )
        self.activation = ACT2FN[self.config.hidden_act]  # 获取激活函数对象

    def __call__(self, hidden_states):
        hidden_states = self.dense(hidden_states)  # 使用全连接层处理输入的隐藏状态
        hidden_states = self.activation(hidden_states)  # 应用激活函数到处理后的隐藏状态上
        return hidden_states


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertOutput with Bert->RoFormer
# 定义一个名为 FlaxRoFormerOutput 的类，继承自 nn.Module
class FlaxRoFormerOutput(nn.Module):
    # 设置类变量 config，类型为 RoFormerConfig，表示 RoFormer 的配置
    config: RoFormerConfig
    # 设置类变量 dtype，默认为 jnp.float32，表示计算的数据类型
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型为浮点数

    # 初始化函数 setup，用于初始化模块的各个子模块
    def setup(self):
        # 初始化一个全连接层 Dense，用于隐藏状态到输出层的映射
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),  # 使用正态分布初始化权重
            dtype=self.dtype,  # 设置数据类型为类变量中定义的数据类型
        )
        # 初始化一个 Dropout 层，用于随机丢弃神经元，防止过拟合
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        # 初始化一个 LayerNorm 层，用于归一化隐藏状态，提升模型训练稳定性
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    # 实现类的调用方法，接受 hidden_states 和 attention_output 作为输入，返回处理后的 hidden_states
    def __call__(self, hidden_states, attention_output, deterministic: bool = True):
        # 将 hidden_states 通过全连接层 Dense 进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对变换后的 hidden_states 进行 Dropout 操作，根据 deterministic 参数决定是否确定性处理
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 将 Dropout 后的 hidden_states 与 attention_output 相加，并通过 LayerNorm 进行归一化处理
        hidden_states = self.LayerNorm(hidden_states + attention_output)
        # 返回处理后的 hidden_states
        return hidden_states


# 定义一个名为 FlaxRoFormerLayer 的类，继承自 nn.Module
class FlaxRoFormerLayer(nn.Module):
    # 设置类变量 config，类型为 RoFormerConfig，表示 RoFormer 的配置
    config: RoFormerConfig
    # 设置类变量 dtype，默认为 jnp.float32，表示计算的数据类型
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型为浮点数

    # 初始化函数 setup，用于初始化模块的各个子模块
    def setup(self):
        # 初始化一个 RoFormer 注意力机制模块 FlaxRoFormerAttention
        self.attention = FlaxRoFormerAttention(self.config, dtype=self.dtype)
        # 初始化一个 RoFormer 中间层模块 FlaxRoFormerIntermediate
        self.intermediate = FlaxRoFormerIntermediate(self.config, dtype=self.dtype)
        # 初始化一个 RoFormer 输出层模块 FlaxRoFormerOutput
        self.output = FlaxRoFormerOutput(self.config, dtype=self.dtype)

    # 实现类的调用方法，接受多个输入参数，返回处理后的输出
    def __call__(
        self,
        hidden_states,
        attention_mask,
        sinusiodal_pos,
        layer_head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
    ):
        # 使用注意力机制模块处理 hidden_states 和相关参数，得到 attention_outputs
        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            sinusiodal_pos,
            layer_head_mask=layer_head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
        )
        # 从 attention_outputs 中获取注意力输出结果 attention_output
        attention_output = attention_outputs[0]

        # 将 attention_output 通过中间层模块进行处理，得到 hidden_states
        hidden_states = self.intermediate(attention_output)
        # 将处理后的 hidden_states 通过输出层模块进行处理，得到最终的输出结果
        hidden_states = self.output(hidden_states, attention_output, deterministic=deterministic)

        # 将最终处理后的 hidden_states 存入 outputs 中
        outputs = (hidden_states,)

        # 如果需要输出注意力信息，将注意力信息加入到 outputs 中一并返回
        if output_attentions:
            outputs += (attention_outputs[1],)
        # 返回输出结果
        return outputs


# 定义一个名为 FlaxRoFormerLayerCollection 的类，继承自 nn.Module
class FlaxRoFormerLayerCollection(nn.Module):
    # 设置类变量 config，类型为 RoFormerConfig，表示 RoFormer 的配置
    config: RoFormerConfig
    # 设置类变量 dtype，默认为 jnp.float32，表示计算的数据类型
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型为浮点数

    # 初始化函数 setup，用于初始化模块的各个子模块
    def setup(self):
        # 初始化 RoFormer 的多层模块 layers，使用列表推导式创建多个 FlaxRoFormerLayer
        self.layers = [
            FlaxRoFormerLayer(self.config, name=str(i), dtype=self.dtype) for i in range(self.config.num_hidden_layers)
        ]

    # 实现类的调用方法，接受多个输入参数，返回处理后的输出
    def __call__(
        self,
        hidden_states,
        attention_mask,
        sinusoidal_pos,
        head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        ):
            # 如果不需要输出注意力，初始化一个空的元组
            all_attentions = () if output_attentions else None
            # 如果不需要输出隐藏状态，初始化一个空的元组
            all_hidden_states = () if output_hidden_states else None

            # 检查如果指定了 head_mask，则需要确保其层数正确
            if head_mask is not None:
                if head_mask.shape[0] != (len(self.layers)):
                    # 抛出数值错误，指出 head_mask 应该对应于 self.layers 的层数，但其对应于 head_mask.shape[0] 层。
                    raise ValueError(
                        f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.shape[0]}."
                    )

            # 遍历 self.layers 中的每一层
            for i, layer in enumerate(self.layers):
                # 如果需要输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                # 调用当前层的前向传播函数
                layer_outputs = layer(
                    hidden_states,
                    attention_mask,
                    sinusoidal_pos,
                    layer_head_mask=head_mask[i] if head_mask is not None else None,
                    deterministic=deterministic,
                    output_attentions=output_attentions,
                )

                # 更新隐藏状态为当前层的输出的第一个元素
                hidden_states = layer_outputs[0]

                # 如果需要输出注意力，则将当前层的注意力输出添加到 all_attentions 中
                if output_attentions:
                    all_attentions += (layer_outputs[1],)

            # 如果需要输出隐藏状态，则将最终的隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # 将最终的隐藏状态作为输出的第一个元素
            outputs = (hidden_states,)

            # 如果不需要返回字典形式的输出，则返回 outputs 中非 None 的值作为元组
            if not return_dict:
                return tuple(v for v in outputs if v is not None)

            # 返回 FlaxBaseModelOutput 类型的对象，包含最终的隐藏状态、所有隐藏状态和所有注意力输出
            return FlaxBaseModelOutput(
                last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
            )
# 定义一个 FlaxRoFormerEncoder 类，继承自 nn.Module
class FlaxRoFormerEncoder(nn.Module):
    config: RoFormerConfig  # RoFormer 的配置信息
    dtype: jnp.dtype = jnp.float32  # 计算时使用的数据类型

    # 模块设置方法，初始化嵌入位置和 RoFormer 层集合
    def setup(self):
        # 创建正弦位置编码
        self.embed_positions = create_sinusoidal_positions(
            self.config.max_position_embeddings, self.config.hidden_size // self.config.num_attention_heads
        )
        # 创建 RoFormer 层集合
        self.layer = FlaxRoFormerLayerCollection(self.config, dtype=self.dtype)

    # 调用方法，对输入进行编码处理
    def __call__(
        self,
        hidden_states,  # 输入的隐藏状态
        attention_mask,  # 注意力遮罩
        head_mask,  # 头部遮罩
        deterministic: bool = True,  # 是否确定性计算
        output_attentions: bool = False,  # 是否输出注意力权重
        output_hidden_states: bool = False,  # 是否输出隐藏状态
        return_dict: bool = True,  # 是否返回字典形式的结果
    ):
        # 获取正弦位置编码的一部分
        sinusoidal_pos = self.embed_positions[: hidden_states.shape[1], :]
        
        # 调用 RoFormer 层集合对输入进行处理并返回结果
        return self.layer(
            hidden_states,
            attention_mask,
            sinusoidal_pos,
            head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


# 从 transformers.models.bert.modeling_flax_bert.FlaxBertPredictionHeadTransform 复制而来，将 Bert 替换为 RoFormer
class FlaxRoFormerPredictionHeadTransform(nn.Module):
    config: RoFormerConfig  # RoFormer 的配置信息
    dtype: jnp.dtype = jnp.float32  # 计算时使用的数据类型

    # 模块设置方法，初始化密集层、激活函数和层归一化
    def setup(self):
        # 密集层，输出维度为配置中的隐藏大小
        self.dense = nn.Dense(self.config.hidden_size, dtype=self.dtype)
        # 激活函数，根据配置选择
        self.activation = ACT2FN[self.config.hidden_act]
        # 层归一化，使用配置中的参数
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    # 调用方法，对输入的隐藏状态进行变换处理
    def __call__(self, hidden_states):
        # 密集层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # 应用激活函数
        hidden_states = self.activation(hidden_states)
        # 应用层归一化
        return self.LayerNorm(hidden_states)


# 从 transformers.models.bert.modeling_flax_bert.FlaxBertLMPredictionHead 复制而来，将 Bert 替换为 RoFormer
class FlaxRoFormerLMPredictionHead(nn.Module):
    config: RoFormerConfig  # RoFormer 的配置信息
    dtype: jnp.dtype = jnp.float32  # 计算时使用的数据类型
    bias_init: Callable[..., np.ndarray] = jax.nn.initializers.zeros  # 偏置的初始化方法

    # 模块设置方法，初始化预测头变换和解码器层
    def setup(self):
        # 预测头变换，使用 RoFormerPredictionHeadTransform 进行初始化
        self.transform = FlaxRoFormerPredictionHeadTransform(self.config, dtype=self.dtype)
        # 解码器层，输出维度为配置中的词汇大小，不使用偏置
        self.decoder = nn.Dense(self.config.vocab_size, dtype=self.dtype, use_bias=False)
        # 偏置项，初始化为 0
        self.bias = self.param("bias", self.bias_init, (self.config.vocab_size,))

    # 调用方法，对输入的隐藏状态进行处理，生成预测结果
    def __call__(self, hidden_states, shared_embedding=None):
        # 使用预测头变换处理隐藏状态
        hidden_states = self.transform(hidden_states)

        # 如果有共享嵌入向量，则使用共享的权重矩阵进行解码
        if shared_embedding is not None:
            hidden_states = self.decoder.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
        else:
            # 否则直接使用解码器层进行解码
            hidden_states = self.decoder(hidden_states)

        # 添加偏置项
        bias = jnp.asarray(self.bias, self.dtype)
        hidden_states += bias
        return hidden_states


# 从 transformers.models.bert.modeling_flax_bert.FlaxBertOnlyMLMHead 复制而来，将 Bert 替换为 RoFormer
class FlaxRoFormerOnlyMLMHead(nn.Module):
    config: RoFormerConfig  # RoFormer 的配置信息
    dtype: jnp.dtype = jnp.float32  # 计算时使用的数据类型
    # 在对象初始化过程中设置预测头部，使用配置和指定数据类型
    def setup(self):
        self.predictions = FlaxRoFormerLMPredictionHead(self.config, dtype=self.dtype)

    # 调用实例时，将隐藏状态作为输入传递给预测头部模型，可选地传递共享的嵌入
    def __call__(self, hidden_states, shared_embedding=None):
        hidden_states = self.predictions(hidden_states, shared_embedding=shared_embedding)
        return hidden_states
class FlaxRoFormerClassificationHead(nn.Module):
    config: RoFormerConfig  # 定义一个属性 config，类型为 RoFormerConfig
    dtype: jnp.dtype = jnp.float32  # 定义一个属性 dtype，默认为 jnp.float32

    def setup(self):
        self.dense = nn.Dense(  # 初始化一个全连接层 dense
            self.config.hidden_size,  # 使用 config 中的 hidden_size
            dtype=self.dtype,  # 设置数据类型为 dtype
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),  # 使用正态分布初始化权重
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)  # 初始化一个 dropout 层
        self.out_proj = nn.Dense(  # 初始化一个全连接层 out_proj
            self.config.num_labels,  # 使用 config 中的 num_labels
            dtype=self.dtype,  # 设置数据类型为 dtype
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),  # 使用正态分布初始化权重
        )
        self.activation = ACT2FN[self.config.hidden_act]  # 设置激活函数为 config 中指定的 hidden_act 函数

    def __call__(self, hidden_states, deterministic=True):
        hidden_states = hidden_states[:, 0, :]  # 取隐藏状态的第一个 token（等同于 [CLS]）
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)  # 应用 dropout
        hidden_states = self.dense(hidden_states)  # 应用全连接层 dense
        hidden_states = self.activation(hidden_states)  # 应用激活函数
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)  # 再次应用 dropout
        hidden_states = self.out_proj(hidden_states)  # 应用全连接层 out_proj
        return hidden_states


class FlaxRoFormerPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RoFormerConfig  # 指定配置类为 RoFormerConfig
    base_model_prefix = "roformer"  # 基础模型前缀为 "roformer"
    module_class: nn.Module = None  # 模块类属性初始化为 None

    def __init__(
        self,
        config: RoFormerConfig,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)  # 初始化模块对象
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量
        input_ids = jnp.zeros(input_shape, dtype="i4")
        token_type_ids = jnp.zeros_like(input_ids)
        attention_mask = jnp.ones_like(input_ids)
        head_mask = jnp.ones((self.config.num_hidden_layers, self.config.num_attention_heads))

        params_rng, dropout_rng = jax.random.split(rng)  # 拆分随机数生成器
        rngs = {"params": params_rng, "dropout": dropout_rng}  # 构建随机数生成器字典

        random_params = self.module.init(  # 使用模块的初始化方法初始化参数
            rngs, input_ids, attention_mask, token_type_ids, head_mask, return_dict=False
        )["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))  # 展开随机参数
            params = flatten_dict(unfreeze(params))  # 展开给定参数
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]  # 将缺失的参数键添加到给定参数中
            self._missing_keys = set()  # 清空缺失键集合
            return freeze(unflatten_dict(params))  # 冻结和重建参数字典
        else:
            return random_params  # 返回随机初始化的参数
    # 添加模型前向传播的文档字符串，格式化插入批处理大小和序列长度信息
    @add_start_docstrings_to_model_forward(ROFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 定义对象调用方法，接受多个参数
    def __call__(
        self,
        input_ids,  # 输入的标识符张量
        attention_mask=None,  # 注意力掩码张量，默认为None
        token_type_ids=None,  # 标记类型张量，默认为None
        head_mask=None,  # 头掩码张量，默认为None
        params: dict = None,  # 参数字典，默认为None
        dropout_rng: jax.random.PRNGKey = None,  # 随机数生成器密钥，默认为None
        train: bool = False,  # 是否处于训练模式，默认为False
        output_attentions: Optional[bool] = None,  # 是否输出注意力，默认为None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，默认为None
        return_dict: Optional[bool] = None,  # 是否返回字典，默认为None
    ):
        # 如果未指定output_attentions，则使用配置中的output_attentions值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未指定output_hidden_states，则使用配置中的output_hidden_states值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定return_dict，则使用配置中的return_dict值
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 如果token_type_ids为None，则初始化为与input_ids相同形状的全零张量
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        # 如果attention_mask为None，则初始化为与input_ids相同形状的全一张量
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        # 如果head_mask为None，则初始化为形状为(num_hidden_layers, num_attention_heads)的全一张量
        if head_mask is None:
            head_mask = jnp.ones((self.config.num_hidden_layers, self.config.num_attention_heads))

        # 如果需要处理任何PRNG（伪随机数生成器），则将其存储在rngs字典中
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        # 调用模块的apply方法，进行模型前向传播
        return self.module.apply(
            {"params": params or self.params},  # 模型参数字典
            jnp.array(input_ids, dtype="i4"),  # 输入标识符张量，转换为32位有符号整数类型
            jnp.array(attention_mask, dtype="i4"),  # 注意力掩码张量，转换为32位有符号整数类型
            jnp.array(token_type_ids, dtype="i4"),  # 标记类型张量，转换为32位有符号整数类型
            jnp.array(head_mask, dtype="i4"),  # 头掩码张量，转换为32位有符号整数类型
            not train,  # 是否处于评估模式
            output_attentions,  # 是否输出注意力
            output_hidden_states,  # 是否输出隐藏状态
            return_dict,  # 是否返回字典
            rngs=rngs,  # 伪随机数生成器字典
        )
# 定义了一个 FlaxRoFormerModule 类，继承自 nn.Module
class FlaxRoFormerModule(nn.Module):
    # 类属性，指定了配置的类型为 RoFormerConfig
    config: RoFormerConfig
    # 类属性，指定了计算中使用的数据类型为 jnp.float32
    dtype: jnp.dtype = jnp.float32  # 计算中使用的数据类型

    # 设置方法，在实例化时被调用，用于初始化模块的各个组件
    def setup(self):
        # 初始化嵌入层，使用 FlaxRoFormerEmbeddings 类
        self.embeddings = FlaxRoFormerEmbeddings(self.config, dtype=self.dtype)
        # 初始化编码器，使用 FlaxRoFormerEncoder 类
        self.encoder = FlaxRoFormerEncoder(self.config, dtype=self.dtype)

    # 实现了调用运算符 ()，定义了模型的前向传播逻辑
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 获取输入经过嵌入层后的隐藏状态
        hidden_states = self.embeddings(input_ids, token_type_ids, attention_mask, deterministic=deterministic)
        # 将隐藏状态传入编码器，获取编码后的输出
        outputs = self.encoder(
            hidden_states,
            attention_mask,
            head_mask=head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 提取编码器输出的隐藏状态
        hidden_states = outputs[0]

        # 如果不要求返回字典形式的输出，则返回元组
        if not return_dict:
            return (hidden_states,) + outputs[1:]

        # 返回 FlaxBaseModelOutput 对象，包含最后的隐藏状态、所有隐藏状态和注意力机制输出
        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# 为 FlaxRoFormerModel 类添加文档字符串，描述其作为 RoFormer 模型的基本转换器的输出
@add_start_docstrings(
    "The bare RoFormer Model transformer outputting raw hidden-states without any specific head on top.",
    ROFORMER_START_DOCSTRING,
)
# FlaxRoFormerModel 类继承自 FlaxRoFormerPreTrainedModel 类
class FlaxRoFormerModel(FlaxRoFormerPreTrainedModel):
    # 指定模块类为 FlaxRoFormerModule
    module_class = FlaxRoFormerModule


# 添加调用示例文档字符串到 FlaxRoFormerModel 类
append_call_sample_docstring(FlaxRoFormerModel, _CHECKPOINT_FOR_DOC, FlaxBaseModelOutput, _CONFIG_FOR_DOC)


# 定义了一个 FlaxRoFormerForMaskedLMModule 类，继承自 nn.Module
class FlaxRoFormerForMaskedLMModule(nn.Module):
    # 类属性，指定了配置的类型为 RoFormerConfig
    config: RoFormerConfig
    # 类属性，指定了计算中使用的数据类型为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 设置方法，在实例化时被调用，用于初始化模块的各个组件
    def setup(self):
        # 初始化 RoFormer 模型，使用 FlaxRoFormerModule 类
        self.roformer = FlaxRoFormerModule(config=self.config, dtype=self.dtype)
        # 初始化仅包含 MLM 头部的类，使用 FlaxRoFormerOnlyMLMHead 类
        self.cls = FlaxRoFormerOnlyMLMHead(config=self.config, dtype=self.dtype)

    # 实现了调用运算符 ()，定义了模型的前向传播逻辑
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        # 调用模型进行前向传播，获取模型输出
        outputs = self.roformer(
            input_ids,
            attention_mask,
            token_type_ids,
            head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从模型输出中提取隐藏状态
        hidden_states = outputs[0]

        # 如果配置允许词嵌入共享，则获取共享的词嵌入
        if self.config.tie_word_embeddings:
            shared_embedding = self.roformer.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
        else:
            shared_embedding = None

        # 使用分类头部模型计算预测分数（logits）
        logits = self.cls(hidden_states, shared_embedding=shared_embedding)

        # 如果不需要返回字典形式的结果，则返回 logits 和额外的输出
        if not return_dict:
            return (logits,) + outputs[1:]

        # 返回 FlaxMaskedLMOutput 类的实例作为字典形式的输出结果
        return FlaxMaskedLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings("""RoFormer Model with a `language modeling` head on top.""", ROFORMER_START_DOCSTRING)
class FlaxRoFormerForMaskedLM(FlaxRoFormerPreTrainedModel):
    module_class = FlaxRoFormerForMaskedLMModule

# 添加用于RoFormer Masked Language Modeling模型的起始文档字符串，继承自FlaxRoFormerPreTrainedModel


append_call_sample_docstring(
    FlaxRoFormerForMaskedLM,
    _CHECKPOINT_FOR_DOC,
    FlaxMaskedLMOutput,
    _CONFIG_FOR_DOC,
    mask="<mask>",
)

# 添加调用示例的文档字符串，用于FlaxRoFormerForMaskedLM模型，包括检查点、输出、配置和掩码信息


class FlaxRoFormerForSequenceClassificationModule(nn.Module):
    config: RoFormerConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.roformer = FlaxRoFormerModule(config=self.config, dtype=self.dtype)
        self.classifier = FlaxRoFormerClassificationHead(config=self.config, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # Model
        outputs = self.roformer(
            input_ids,
            attention_mask,
            token_type_ids,
            head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output, deterministic=deterministic)

        if not return_dict:
            return (logits,) + outputs[1:]

        return FlaxSequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# RoFormer用于序列分类任务的模块定义，设置了RoFormer模型和分类器头部，支持返回字典或元组格式的输出


@add_start_docstrings(
    """
    RoFormer Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    ROFORMER_START_DOCSTRING,
)
class FlaxRoFormerForSequenceClassification(FlaxRoFormerPreTrainedModel):
    module_class = FlaxRoFormerForSequenceClassificationModule

# 添加RoFormer序列分类/回归模型的起始文档字符串，继承自FlaxRoFormerPreTrainedModel，支持GLUE任务等应用


append_call_sample_docstring(
    FlaxRoFormerForSequenceClassification,
    _CHECKPOINT_FOR_DOC,
    FlaxSequenceClassifierOutput,
    _CONFIG_FOR_DOC,
)

# 添加调用示例的文档字符串，用于FlaxRoFormerForSequenceClassification模型，包括检查点、输出和配置信息


class FlaxRoFormerForMultipleChoiceModule(nn.Module):
    config: RoFormerConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.roformer = FlaxRoFormerModule(config=self.config, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        self.classifier = nn.Dense(1, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        ):
        # 获取选择数量
        num_choices = input_ids.shape[1]
        # 将输入张量形状重新调整为二维数组，保留最后一个维度不变
        input_ids = input_ids.reshape(-1, input_ids.shape[-1])
        attention_mask = attention_mask.reshape(-1, attention_mask.shape[-1])
        token_type_ids = token_type_ids.reshape(-1, token_type_ids.shape[-1])

        # 调用模型
        outputs = self.roformer(
            input_ids,
            attention_mask,
            token_type_ids,
            head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 在 PyTorch 实现中，等同于调用 sequence_summary
        hidden_states = outputs[0]
        # 提取最后一个隐藏状态作为汇总输出
        pooled_output = hidden_states[:, -1]
        pooled_output = self.dropout(pooled_output, deterministic=deterministic)

        # 通过分类器获取 logits
        logits = self.classifier(pooled_output)

        # 将 logits 重新调整为与选择数量相关的形状
        reshaped_logits = logits.reshape(-1, num_choices)

        # 如果不返回字典，则返回元组，并包括额外的输出
        if not return_dict:
            return (reshaped_logits,) + outputs[2:]

        # 返回 FlaxMultipleChoiceModelOutput 对象，包括调用的相关输出
        return FlaxMultipleChoiceModelOutput(
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """
    RoFormer Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    ROFORMER_START_DOCSTRING,
)
class FlaxRoFormerForMultipleChoice(FlaxRoFormerPreTrainedModel):
    module_class = FlaxRoFormerForMultipleChoiceModule



overwrite_call_docstring(
    FlaxRoFormerForMultipleChoice, ROFORMER_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
)



append_call_sample_docstring(
    FlaxRoFormerForMultipleChoice,
    _CHECKPOINT_FOR_DOC,
    FlaxMultipleChoiceModelOutput,
    _CONFIG_FOR_DOC,
)



class FlaxRoFormerForTokenClassificationModule(nn.Module):
    config: RoFormerConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.roformer = FlaxRoFormerModule(config=self.config, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        self.classifier = nn.Dense(self.config.num_labels, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # Model
        outputs = self.roformer(
            input_ids,
            attention_mask,
            token_type_ids,
            head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        logits = self.classifier(hidden_states)

        if not return_dict:
            return (logits,) + outputs[1:]

        return FlaxTokenClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



@add_start_docstrings(
    """
    RoFormer Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    ROFORMER_START_DOCSTRING,
)
class FlaxRoFormerForTokenClassification(FlaxRoFormerPreTrainedModel):
    module_class = FlaxRoFormerForTokenClassificationModule



append_call_sample_docstring(
    FlaxRoFormerForTokenClassification,
    _CHECKPOINT_FOR_DOC,
    FlaxTokenClassifierOutput,
    _CONFIG_FOR_DOC,
)



class FlaxRoFormerForQuestionAnsweringModule(nn.Module):
    config: RoFormerConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.roformer = FlaxRoFormerModule(config=self.config, dtype=self.dtype)
        self.qa_outputs = nn.Dense(self.config.num_labels, dtype=self.dtype)
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 调用模型主函数，接受多个输入参数和可选的标志位参数
        # Model
        # 调用 RoFormer 模型进行前向传播
        outputs = self.roformer(
            input_ids,
            attention_mask,
            token_type_ids,
            head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从模型输出中提取隐藏状态
        hidden_states = outputs[0]

        # 使用全连接层计算问题回答的开始和结束位置的 logits
        logits = self.qa_outputs(hidden_states)

        # 根据标签数目将 logits 分割为开始和结束 logits
        start_logits, end_logits = logits.split(self.config.num_labels, axis=-1)

        # 压缩最后一个维度，确保 logits 张量维度的一致性
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # 如果 return_dict 为 False，则返回元组形式的结果
        if not return_dict:
            return (start_logits, end_logits) + outputs[1:]

        # 如果 return_dict 为 True，则返回 FlaxQuestionAnsweringModelOutput 对象
        return FlaxQuestionAnsweringModelOutput(
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """
    RoFormer Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    ROFORMER_START_DOCSTRING,
)



# 使用 @add_start_docstrings 装饰器为 FlaxRoFormerForQuestionAnswering 类添加文档字符串，
# 描述其为一种 RoFormer 模型，用于提取式问答任务（如 SQuAD），在隐藏状态输出之上有线性层，
# 用于计算 `span start logits` 和 `span end logits`。
# ROFORMER_START_DOCSTRING 变量用于提供 RoFormer 模型的开始文档字符串。
class FlaxRoFormerForQuestionAnswering(FlaxRoFormerPreTrainedModel):
    module_class = FlaxRoFormerForQuestionAnsweringModule



append_call_sample_docstring(
    FlaxRoFormerForQuestionAnswering,
    _CHECKPOINT_FOR_DOC,
    FlaxQuestionAnsweringModelOutput,
    _CONFIG_FOR_DOC,
)



# 调用 append_call_sample_docstring 函数，将样例调用的文档字符串添加到 FlaxRoFormerForQuestionAnswering 类中，
# 用于说明如何调用该类的样例，并引用了 _CHECKPOINT_FOR_DOC 和 _CONFIG_FOR_DOC 变量。
```