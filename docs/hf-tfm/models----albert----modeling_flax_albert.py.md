# `.\models\albert\modeling_flax_albert.py`

```
# coding=utf-8
# 上面是指定代码文件的字符编码格式

# 版权声明，指出代码的版权归属于 Google AI, Google Brain 和 HuggingFace Inc. 团队
# 根据 Apache 许可证 2.0 版本授权，只有符合许可证条件才能使用该文件
# 可以通过指定的链接获取许可证的副本
# 除非适用法律要求或书面同意，否则按“原样”分发此软件
# 没有任何明示或暗示的保证或条件
# 有关更多详细信息，请参阅许可证

# 引入必要的类型声明
from typing import Callable, Optional, Tuple

# 引入 Flax 框架及其组件
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

# 引入 Flax 相关模块
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax

# 引入模型输出类和实用函数
from ...modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxBaseModelOutputWithPooling,
    FlaxMaskedLMOutput,
    FlaxMultipleChoiceModelOutput,
    FlaxQuestionAnsweringModelOutput,
    FlaxSequenceClassifierOutput,
    FlaxTokenClassifierOutput,
)
from ...modeling_flax_utils import (
    ACT2FN,
    FlaxPreTrainedModel,
    append_call_sample_docstring,
    append_replace_return_docstrings,
    overwrite_call_docstring,
)
# 引入通用实用函数和配置
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_albert import AlbertConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 以下两行用于文档化，指定相关内容为文档中的关键检查点和配置信息
_CHECKPOINT_FOR_DOC = "albert/albert-base-v2"
_CONFIG_FOR_DOC = "AlbertConfig"

# 定义 FlaxAlbertForPreTrainingOutput 类，用于表示 FlaxAlbertForPreTraining 模型的输出类型
@flax.struct.dataclass
class FlaxAlbertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`FlaxAlbertForPreTraining`].
    用于 [`FlaxAlbertForPreTraining`] 的输出类型。
    """
    # 定义函数参数及其类型注解，用于描述模型预测的输出结果
    
    Args:
        prediction_logits (`jnp.ndarray` of shape `(batch_size, sequence_length, config.vocab_size)`):
            语言建模头部的预测分数（在 SoftMax 之前的每个词汇标记的分数）。
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        sop_logits (`jnp.ndarray` of shape `(batch_size, 2)`):
            下一个序列预测（分类）头部的预测分数（在 SoftMax 之前的 True/False 连续性的分数）。
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation before SoftMax).
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            模型隐藏状态的元组（jnp.ndarray），形状为 `(batch_size, sequence_length, hidden_size)`。
            
            每个层的输出和初始嵌入输出的隐藏状态。
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.
    
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            模型自注意力层的注意力权重的元组（jnp.ndarray），形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
            
            在注意力 SoftMax 之后的注意力权重，用于计算自注意力头部中的加权平均值。
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    
    prediction_logits: jnp.ndarray = None
    sop_logits: jnp.ndarray = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    attentions: Optional[Tuple[jnp.ndarray]] = None
# ALBERT_START_DOCSTRING 是一个包含模型文档字符串的原始字符串常量
ALBERT_START_DOCSTRING = r"""

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

    Parameters:
        config ([`AlbertConfig`]): Model configuration class with all the parameters of the model.
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

# ALBERT_INPUTS_DOCSTRING 是一个包含输入文档字符串的原始字符串常量，目前为空
ALBERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`numpy.ndarray` of shape `({0})`):
            # 输入序列中词汇表中的索引列表。

            # 使用 `AutoTokenizer` 可获取这些索引。参见 `PreTrainedTokenizer.encode` 和 `PreTrainedTokenizer.__call__`。

            # 更多信息请查看 [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`numpy.ndarray` of shape `({0})`, *optional*):
            # 遮罩，用于在填充的令牌索引上避免执行注意力操作。遮罩值在 `[0, 1]` 范围内：

            # - 对于 **未遮罩** 的令牌，值为 1，
            # - 对于 **已遮罩** 的令牌，值为 0。

            # 更多信息请查看 [什么是注意力遮罩？](../glossary#attention-mask)
        token_type_ids (`numpy.ndarray` of shape `({0})`, *optional*):
            # 分段令牌索引，用于指示输入的第一部分和第二部分。索引在 `[0, 1]` 范围内：

            # - 0 对应 *句子 A* 的令牌，
            # - 1 对应 *句子 B* 的令牌。

            # 更多信息请查看 [什么是令牌类型 ID？](../glossary#token-type-ids)
        position_ids (`numpy.ndarray` of shape `({0})`, *optional*):
            # 每个输入序列令牌在位置嵌入中的位置索引。选择范围为 `[0, config.max_position_embeddings - 1]`。
        return_dict (`bool`, *optional*):
            # 是否返回一个 [`~utils.ModelOutput`] 而不是普通的元组。
"""
class FlaxAlbertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    config: AlbertConfig  # 定义配置对象的类型
    dtype: jnp.dtype = jnp.float32  # 计算过程中使用的数据类型

    def setup(self):
        # 初始化词嵌入层，使用正态分布初始化器
        self.word_embeddings = nn.Embed(
            self.config.vocab_size,
            self.config.embedding_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        # 初始化位置嵌入层，使用正态分布初始化器
        self.position_embeddings = nn.Embed(
            self.config.max_position_embeddings,
            self.config.embedding_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        # 初始化标记类型嵌入层，使用正态分布初始化器
        self.token_type_embeddings = nn.Embed(
            self.config.type_vocab_size,
            self.config.embedding_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        # 初始化层归一化层，使用给定的 epsilon 和数据类型
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 初始化 dropout 层，使用给定的 dropout 概率
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, input_ids, token_type_ids, position_ids, deterministic: bool = True):
        # 嵌入输入 ID，转换为指定数据类型的张量
        inputs_embeds = self.word_embeddings(input_ids.astype("i4"))
        # 嵌入位置 ID，转换为指定数据类型的张量
        position_embeds = self.position_embeddings(position_ids.astype("i4"))
        # 嵌入标记类型 ID，转换为指定数据类型的张量
        token_type_embeddings = self.token_type_embeddings(token_type_ids.astype("i4"))

        # 将所有嵌入相加
        hidden_states = inputs_embeds + token_type_embeddings + position_embeds

        # 应用层归一化
        hidden_states = self.LayerNorm(hidden_states)
        # 应用 dropout
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states


class FlaxAlbertSelfAttention(nn.Module):
    config: AlbertConfig  # 定义配置对象的类型
    dtype: jnp.dtype = jnp.float32  # 计算过程中使用的数据类型
    # 在设置阶段验证隐藏层大小是否可以被注意力头数整除
    def setup(self):
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError(
                "`config.hidden_size`: {self.config.hidden_size} has to be a multiple of `config.num_attention_heads` "
                "                   : {self.config.num_attention_heads}"
            )

        # 创建用于查询的全连接层，输入大小为隐藏层大小，使用指定的数据类型和正态分布初始化
        self.query = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        # 创建用于键的全连接层，输入大小为隐藏层大小，使用指定的数据类型和正态分布初始化
        self.key = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        # 创建用于值的全连接层，输入大小为隐藏层大小，使用指定的数据类型和正态分布初始化
        self.value = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        # 创建用于最终输出的全连接层，输入大小为隐藏层大小，使用指定的正态分布初始化
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 创建层归一化对象，使用指定的 epsilon 值和数据类型
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 创建用于dropout的对象，指定丢弃率
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
    # 定义一个调用方法，接受隐藏状态、注意力掩码等参数，返回注意力层的输出
    def __call__(self, hidden_states, attention_mask, deterministic=True, output_attentions: bool = False):
        # 计算每个注意力头的维度
        head_dim = self.config.hidden_size // self.config.num_attention_heads

        # 使用 query 网络处理隐藏状态，然后重塑为 (batch_size, seq_length, num_attention_heads, head_dim)
        query_states = self.query(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )
        # 使用 value 网络处理隐藏状态，然后重塑为 (batch_size, seq_length, num_attention_heads, head_dim)
        value_states = self.value(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )
        # 使用 key 网络处理隐藏状态，然后重塑为 (batch_size, seq_length, num_attention_heads, head_dim)
        key_states = self.key(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )

        # 将布尔类型的注意力掩码转换为注意力偏置
        if attention_mask is not None:
            # 将注意力掩码扩展维度以匹配 query 和 key 张量的维度
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
            # 根据注意力掩码的值生成注意力偏置，使用 lax.select 来根据条件选择不同的值
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
            )
        else:
            attention_bias = None

        # 初始化 dropout RNG
        dropout_rng = None
        # 如果不是确定性计算并且设置了注意力概率的 dropout，则生成一个 dropout RNG
        if not deterministic and self.config.attention_probs_dropout_prob > 0.0:
            dropout_rng = self.make_rng("dropout")

        # 计算注意力权重，使用 dot_product_attention_weights 函数
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

        # 使用 einsum 函数计算注意力输出，将注意力权重应用到 value 状态上
        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)
        # 重塑注意力输出的形状为 (batch_size, seq_length, hidden_size)
        attn_output = attn_output.reshape(attn_output.shape[:2] + (-1,))

        # 将注意力输出投影到相同维度空间
        projected_attn_output = self.dense(attn_output)
        # 如果使用 dropout，则对投影后的注意力输出应用 dropout
        projected_attn_output = self.dropout(projected_attn_output, deterministic=deterministic)
        # 使用 LayerNorm 对投影后的注意力输出进行规范化，并与原始隐藏状态相加
        layernormed_attn_output = self.LayerNorm(projected_attn_output + hidden_states)
        # 根据需求决定是否输出注意力权重
        outputs = (layernormed_attn_output, attn_weights) if output_attentions else (layernormed_attn_output,)
        # 返回最终的输出
        return outputs
# 定义 FlaxAlbertLayer 类，继承自 nn.Module
class FlaxAlbertLayer(nn.Module):
    # 保存 AlbertConfig 类型的配置信息
    config: AlbertConfig
    # 指定计算中使用的数据类型为 jnp.float32
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    # 初始化方法，设置层的组件
    def setup(self):
        # 创建 self.attention 属性，使用 FlaxAlbertSelfAttention 类处理注意力机制
        self.attention = FlaxAlbertSelfAttention(self.config, dtype=self.dtype)
        # 创建 self.ffn 属性，使用 nn.Dense 层作为前馈神经网络
        self.ffn = nn.Dense(
            self.config.intermediate_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 根据配置选择激活函数
        self.activation = ACT2FN[self.config.hidden_act]
        # 创建 self.ffn_output 属性，使用 nn.Dense 层作为前馈神经网络输出层
        self.ffn_output = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 创建 self.full_layer_layer_norm 属性，使用 nn.LayerNorm 层进行层归一化
        self.full_layer_layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 创建 self.dropout 属性，使用 nn.Dropout 层进行随机失活
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    # 调用方法，定义前向传播逻辑
    def __call__(
        self,
        hidden_states,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
    ):
        # 使用 self.attention 处理注意力输出
        attention_outputs = self.attention(
            hidden_states, attention_mask, deterministic=deterministic, output_attentions=output_attentions
        )
        # 获取注意力输出的第一个元素作为 attention_output
        attention_output = attention_outputs[0]
        # 使用 self.ffn 处理 attention_output 得到前馈神经网络的输出
        ffn_output = self.ffn(attention_output)
        # 使用 self.activation 应用激活函数
        ffn_output = self.activation(ffn_output)
        # 使用 self.ffn_output 处理前馈神经网络输出得到最终输出
        ffn_output = self.ffn_output(ffn_output)
        # 使用 self.dropout 进行随机失活处理最终输出
        ffn_output = self.dropout(ffn_output, deterministic=deterministic)
        # 将前馈神经网络的输出与注意力输出相加，然后进行层归一化得到 hidden_states
        hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)

        # 将 hidden_states 存入 outputs 元组中
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则将 attention_outputs[1] 也加入 outputs 中
        if output_attentions:
            outputs += (attention_outputs[1],)

        # 返回 outputs 元组作为最终的输出结果
        return outputs


# 定义 FlaxAlbertLayerCollection 类，继承自 nn.Module
class FlaxAlbertLayerCollection(nn.Module):
    # 保存 AlbertConfig 类型的配置信息
    config: AlbertConfig
    # 指定计算中使用的数据类型为 jnp.float32
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    # 初始化方法，设置层的组件
    def setup(self):
        # 创建 self.layers 属性，包含多个 FlaxAlbertLayer 层组成的列表
        self.layers = [
            FlaxAlbertLayer(self.config, name=str(i), dtype=self.dtype) for i in range(self.config.inner_group_num)
        ]

    # 调用方法，定义前向传播逻辑
    def __call__(
        self,
        hidden_states,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        # 注意：此处代码未完整
        ):
            # 初始化空元组，用于存储各层的隐藏状态和注意力分布
            layer_hidden_states = ()
            layer_attentions = ()

            # 遍历模型的每一层
            for layer_index, albert_layer in enumerate(self.layers):
                # 调用当前层的前向传播方法，获取该层的输出
                layer_output = albert_layer(
                    hidden_states,
                    attention_mask,
                    deterministic=deterministic,
                    output_attentions=output_attentions,
                )
                # 更新隐藏状态为当前层的输出的第一个元素
                hidden_states = layer_output[0]

                # 如果需要输出注意力分布，将当前层的注意力分布添加到layer_attentions元组中
                if output_attentions:
                    layer_attentions = layer_attentions + (layer_output[1],)

                # 如果需要输出隐藏状态，将当前层的隐藏状态添加到layer_hidden_states元组中
                if output_hidden_states:
                    layer_hidden_states = layer_hidden_states + (hidden_states,)

            # 构建输出元组，包括最终的隐藏状态
            outputs = (hidden_states,)
            # 如果需要输出每层的隐藏状态，将其添加到输出元组中
            if output_hidden_states:
                outputs = outputs + (layer_hidden_states,)
            # 如果需要输出每层的注意力分布，将其添加到输出元组中
            if output_attentions:
                outputs = outputs + (layer_attentions,)
            # 返回模型的输出，包括最后一层的隐藏状态，可选的每层隐藏状态和每层注意力分布
            return outputs  # 最后一层的隐藏状态，(每层隐藏状态)，(每层注意力)
class FlaxAlbertLayerCollections(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = jnp.float32  # 计算所用的数据类型
    layer_index: Optional[str] = None

    def setup(self):
        self.albert_layers = FlaxAlbertLayerCollection(self.config, dtype=self.dtype)
        # 初始化 Albert 层集合

    def __call__(
        self,
        hidden_states,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        outputs = self.albert_layers(
            hidden_states,
            attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        return outputs
        # 调用 Albert 层集合并返回输出结果


class FlaxAlbertLayerGroups(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = jnp.float32  # 计算所用的数据类型

    def setup(self):
        self.layers = [
            FlaxAlbertLayerCollections(self.config, name=str(i), layer_index=str(i), dtype=self.dtype)
            for i in range(self.config.num_hidden_groups)
        ]
        # 初始化 Albert 层组

    def __call__(
        self,
        hidden_states,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = (hidden_states,) if output_hidden_states else None

        for i in range(self.config.num_hidden_layers):
            # 计算当前层所属的隐藏组索引
            group_idx = int(i / (self.config.num_hidden_layers / self.config.num_hidden_groups))
            layer_group_output = self.layers[group_idx](
                hidden_states,
                attention_mask,
                deterministic=deterministic,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            hidden_states = layer_group_output[0]

            if output_attentions:
                all_attentions = all_attentions + layer_group_output[-1]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )
        # 如果不返回字典，则返回相应的输出元组


class FlaxAlbertEncoder(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = jnp.float32  # 计算所用的数据类型

    def setup(self):
        self.embedding_hidden_mapping_in = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        self.albert_layer_groups = FlaxAlbertLayerGroups(self.config, dtype=self.dtype)
        # 初始化 Albert 编码器
    # 定义一个特殊方法 __call__，使得对象可以像函数一样被调用
    def __call__(
        self,
        hidden_states,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 将输入的隐藏状态通过 embedding_hidden_mapping_in 方法映射转换
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)
        # 调用 albert_layer_groups 方法处理映射后的隐藏状态和注意力掩码，
        # 可选参数包括 deterministic（是否确定性计算）、output_attentions（是否输出注意力权重）、
        # output_hidden_states（是否输出每层的隐藏状态），返回结果根据 return_dict 决定是否返回字典形式
        return self.albert_layer_groups(
            hidden_states,
            attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
# 定义一个名为 FlaxAlbertOnlyMLMHead 的类，继承自 nn.Module
class FlaxAlbertOnlyMLMHead(nn.Module):
    # 配置属性，指定为 AlbertConfig 类型
    config: AlbertConfig
    # 数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 偏置初始化函数，默认为零初始化
    bias_init: Callable[..., np.ndarray] = jax.nn.initializers.zeros

    # 初始化方法
    def setup(self):
        # 创建一个全连接层，输出维度为 config.embedding_size
        self.dense = nn.Dense(self.config.embedding_size, dtype=self.dtype)
        # 激活函数，根据配置选择 ACT2FN 中的激活函数
        self.activation = ACT2FN[self.config.hidden_act]
        # LayerNorm 层，使用 config.layer_norm_eps 作为 epsilon 参数
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 输出层，输出维度为 config.vocab_size，不使用偏置
        self.decoder = nn.Dense(self.config.vocab_size, dtype=self.dtype, use_bias=False)
        # 初始化偏置参数，维度为 (config.vocab_size,)
        self.bias = self.param("bias", self.bias_init, (self.config.vocab_size,))

    # 前向传播方法
    def __call__(self, hidden_states, shared_embedding=None):
        # 全连接层
        hidden_states = self.dense(hidden_states)
        # 激活函数
        hidden_states = self.activation(hidden_states)
        # LayerNorm 层
        hidden_states = self.LayerNorm(hidden_states)

        # 如果传入了 shared_embedding 参数，则使用 decoder 层进行解码
        if shared_embedding is not None:
            hidden_states = self.decoder.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
        else:
            # 否则直接使用 decoder 层
            hidden_states = self.decoder(hidden_states)

        # 加上偏置
        hidden_states += self.bias
        # 返回最终的隐藏状态
        return hidden_states


# 定义一个名为 FlaxAlbertSOPHead 的类，继承自 nn.Module
class FlaxAlbertSOPHead(nn.Module):
    # 配置属性，指定为 AlbertConfig 类型
    config: AlbertConfig
    # 数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 初始化方法
    def setup(self):
        # Dropout 层，使用配置中的 classifier_dropout_prob
        self.dropout = nn.Dropout(self.config.classifier_dropout_prob)
        # 全连接层，输出维度为 2
        self.classifier = nn.Dense(2, dtype=self.dtype)

    # 前向传播方法
    def __call__(self, pooled_output, deterministic=True):
        # 应用 Dropout
        pooled_output = self.dropout(pooled_output, deterministic=deterministic)
        # 分类器层，得到 logits
        logits = self.classifier(pooled_output)
        # 返回 logits
        return logits


# 定义一个名为 FlaxAlbertPreTrainedModel 的类，继承自 FlaxPreTrainedModel
class FlaxAlbertPreTrainedModel(FlaxPreTrainedModel):
    """
    一个处理权重初始化、预训练模型下载和加载的抽象类。
    """

    # 配置类，指定为 AlbertConfig
    config_class = AlbertConfig
    # 基础模型前缀名称为 "albert"
    base_model_prefix = "albert"
    # 模块类，默认为 None，需要在子类中指定具体的模块类
    module_class: nn.Module = None

    # 初始化方法
    def __init__(
        self,
        config: AlbertConfig,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        # 创建模块实例，传入配置和其他参数
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 调用父类的初始化方法
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)
    ```python`
        # 初始化权重函数，使用随机数种子和输入形状，返回参数字典
        def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
            # 初始化输入张量，创建一个全零张量，数据类型为整数4位
            input_ids = jnp.zeros(input_shape, dtype="i4")
            # 创建与 input_ids 相同形状的全零张量，作为 token 类型标识
            token_type_ids = jnp.zeros_like(input_ids)
            # 根据 input_ids 的最后一个维度生成位置 ID，使用广播到输入形状
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
            # 创建一个与 input_ids 相同形状的全一张量，作为注意力掩码
            attention_mask = jnp.ones_like(input_ids)
    
            # 分割随机数种子为参数随机种子和 dropout 随机种子
            params_rng, dropout_rng = jax.random.split(rng)
            rngs = {"params": params_rng, "dropout": dropout_rng}
    
            # 初始化模型参数，调用模块的 init 方法，返回随机参数
            random_params = self.module.init(
                rngs, input_ids, attention_mask, token_type_ids, position_ids, return_dict=False
            )["params"]
    
            # 如果传入了已有参数，合并随机参数和已有参数，并补全缺失的参数
            if params is not None:
                random_params = flatten_dict(unfreeze(random_params))
                params = flatten_dict(unfreeze(params))
                for missing_key in self._missing_keys:
                    params[missing_key] = random_params[missing_key]
                self._missing_keys = set()
                # 返回合并后的参数字典，进行冻结
                return freeze(unflatten_dict(params))
            else:
                # 返回随机初始化的参数
                return random_params
    
        # 添加文档字符串，定义模型前向传播方法的文档
        @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
        def __call__(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            params: dict = None,
            dropout_rng: jax.random.PRNGKey = None,
            train: bool = False,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ):
            # 根据配置，设置输出注意力和隐藏状态的默认值
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.return_dict
    
            # 如果未传入 token_type_ids，初始化为与 input_ids 相同的全零张量
            if token_type_ids is None:
                token_type_ids = jnp.zeros_like(input_ids)
    
            # 如果未传入 position_ids，初始化为根据 input_ids 的最后一个维度生成的广播张量
            if position_ids is None:
                position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)
    
            # 如果未传入 attention_mask，初始化为与 input_ids 相同的全一张量
            if attention_mask is None:
                attention_mask = jnp.ones_like(input_ids)
    
            # 初始化随机数种子字典
            rngs = {}
            if dropout_rng is not None:
                rngs["dropout"] = dropout_rng
    
            # 调用模型的 apply 方法，传入参数进行前向计算
            return self.module.apply(
                {"params": params or self.params},
                jnp.array(input_ids, dtype="i4"),
                jnp.array(attention_mask, dtype="i4"),
                jnp.array(token_type_ids, dtype="i4"),
                jnp.array(position_ids, dtype="i4"),
                not train,  # 训练模式为 False，推理模式为 True
                output_attentions,
                output_hidden_states,
                return_dict,
                rngs=rngs,
            )
# 定义一个继承自`nn.Module`的类，用于实现Flax版本的Albert模型
class FlaxAlbertModule(nn.Module):
    # 类型注解，指定`config`为AlbertConfig类型
    config: AlbertConfig
    # 指定`dtype`为jnp.float32，用于计算的数据类型
    dtype: jnp.dtype = jnp.float32  # 计算时的数据类型
    # 是否添加池化层的标志，默认为True
    add_pooling_layer: bool = True

    # 模块初始化函数
    def setup(self):
        # 初始化嵌入层`embeddings`，使用FlaxAlbertEmbeddings类
        self.embeddings = FlaxAlbertEmbeddings(self.config, dtype=self.dtype)
        # 初始化编码器`encoder`，使用FlaxAlbertEncoder类
        self.encoder = FlaxAlbertEncoder(self.config, dtype=self.dtype)
        # 如果设置添加池化层，则初始化`pooler`为全连接层，并指定激活函数为tanh
        if self.add_pooling_layer:
            self.pooler = nn.Dense(
                self.config.hidden_size,
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
                dtype=self.dtype,
                name="pooler",
            )
            self.pooler_activation = nn.tanh
        else:
            # 如果不添加池化层，则将`pooler`和`pooler_activation`设置为None
            self.pooler = None
            self.pooler_activation = None

    # 对象调用函数，实现模型的前向传播
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids: Optional[np.ndarray] = None,
        position_ids: Optional[np.ndarray] = None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 当未传入`token_type_ids`时，初始化为与`input_ids`相同形状的全零数组
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        # 当未传入`position_ids`时，初始化为广播形式的序列长度数组
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        # 使用嵌入层`embeddings`计算输入数据的隐状态表示
        hidden_states = self.embeddings(input_ids, token_type_ids, position_ids, deterministic=deterministic)

        # 将隐状态表示输入编码器`encoder`，获取模型输出
        outputs = self.encoder(
            hidden_states,
            attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # 获取编码器输出的隐状态表示
        hidden_states = outputs[0]

        # 如果设置添加池化层，则对第一个时间步的隐状态进行池化操作
        if self.add_pooling_layer:
            pooled = self.pooler(hidden_states[:, 0])
            pooled = self.pooler_activation(pooled)
        else:
            # 如果不添加池化层，则将`pooled`设置为None
            pooled = None

        # 如果不返回字典形式的输出，则根据`return_dict`的设置返回相应结果
        if not return_dict:
            if pooled is None:
                # 如果`pooled`为None，则不返回它
                return (hidden_states,) + outputs[1:]
            return (hidden_states, pooled) + outputs[1:]

        # 返回包含池化输出和其他模型输出的字典形式结果
        return FlaxBaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# 使用装饰器`add_start_docstrings`为`FlaxAlbertModel`类添加注释文档
@add_start_docstrings(
    "The bare Albert Model transformer outputting raw hidden-states without any specific head on top.",
    ALBERT_START_DOCSTRING,
)
# `FlaxAlbertModel`类继承自`FlaxAlbertPreTrainedModel`，指定使用的模块类为`FlaxAlbertModule`
class FlaxAlbertModel(FlaxAlbertPreTrainedModel):
    module_class = FlaxAlbertModule


# 调用`append_call_sample_docstring`函数，为`FlaxAlbertModel`类添加调用示例注释
append_call_sample_docstring(FlaxAlbertModel, _CHECKPOINT_FOR_DOC, FlaxBaseModelOutputWithPooling, _CONFIG_FOR_DOC)


# 定义一个继承自`nn.Module`的类，用于实现Albert预训练模型
class FlaxAlbertForPreTrainingModule(nn.Module):
    # 类型注解，指定`config`为AlbertConfig类型
    config: AlbertConfig
    # 定义默认的数据类型为 jnp.float32，使用了 jax.numpy 的数据类型
    dtype: jnp.dtype = jnp.float32

    # 初始化模型的方法，创建了 Albert 模型、MLM 头部和 SOP 分类器
    def setup(self):
        # 使用给定的配置和数据类型创建 Albert 模型
        self.albert = FlaxAlbertModule(config=self.config, dtype=self.dtype)
        # 使用给定的配置和数据类型创建只有 MLM 头部的模型
        self.predictions = FlaxAlbertOnlyMLMHead(config=self.config, dtype=self.dtype)
        # 使用给定的配置和数据类型创建 SOP 分类器
        self.sop_classifier = FlaxAlbertSOPHead(config=self.config, dtype=self.dtype)

    # 调用模型时的方法，接收多个输入参数和几个布尔型选项
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 调用 Albert 模型进行前向传播，获取输出
        outputs = self.albert(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果配置要求共享词嵌入，则获取共享的词嵌入
        if self.config.tie_word_embeddings:
            shared_embedding = self.albert.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
        else:
            shared_embedding = None

        # 从 Albert 模型的输出中提取隐藏状态和汇聚输出
        hidden_states = outputs[0]
        pooled_output = outputs[1]

        # 使用 MLM 头部对隐藏状态进行预测
        prediction_scores = self.predictions(hidden_states, shared_embedding=shared_embedding)
        # 使用 SOP 分类器对汇聚输出进行预测
        sop_scores = self.sop_classifier(pooled_output, deterministic=deterministic)

        # 如果不要求返回字典形式的结果，则返回元组形式的结果
        if not return_dict:
            return (prediction_scores, sop_scores) + outputs[2:]

        # 返回预训练 Albert 模型的输出结果，包括预测 logits、SOP logits、隐藏状态和注意力权重
        return FlaxAlbertForPreTrainingOutput(
            prediction_logits=prediction_scores,
            sop_logits=sop_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """
    Albert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a
    `sentence order prediction (classification)` head.
    """,
    ALBERT_START_DOCSTRING,
)
class FlaxAlbertForPreTraining(FlaxAlbertPreTrainedModel):
    module_class = FlaxAlbertForPreTrainingModule



FLAX_ALBERT_FOR_PRETRAINING_DOCSTRING = """
    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, FlaxAlbertForPreTraining

    >>> tokenizer = AutoTokenizer.from_pretrained("albert/albert-base-v2")
    >>> model = FlaxAlbertForPreTraining.from_pretrained("albert/albert-base-v2")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="np")
    >>> outputs = model(**inputs)

    >>> prediction_logits = outputs.prediction_logits
    >>> seq_relationship_logits = outputs.sop_logits
    ```
"""

# Overwrite the docstring of FlaxAlbertForPreTraining to include input docstring and predefined docstring
overwrite_call_docstring(
    FlaxAlbertForPreTraining,
    ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length") + FLAX_ALBERT_FOR_PRETRAINING_DOCSTRING,
)



# Append and replace return docstrings for FlaxAlbertForPreTraining
append_replace_return_docstrings(
    FlaxAlbertForPreTraining, output_type=FlaxAlbertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC
)



class FlaxAlbertForMaskedLMModule(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # Set up Albert model without pooling layer
        self.albert = FlaxAlbertModule(config=self.config, add_pooling_layer=False, dtype=self.dtype)
        # Set up Masked LM head for predictions
        self.predictions = FlaxAlbertOnlyMLMHead(config=self.config, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # Forward pass through Albert model
        outputs = self.albert(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Extract hidden states from model outputs
        hidden_states = outputs[0]

        # Determine if word embeddings are tied
        if self.config.tie_word_embeddings:
            shared_embedding = self.albert.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
        else:
            shared_embedding = None

        # Compute masked language modeling logits
        logits = self.predictions(hidden_states, shared_embedding=shared_embedding)

        # Return either a tuple or a named tuple depending on return_dict
        if not return_dict:
            return (logits,) + outputs[1:]

        return FlaxMaskedLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



@add_start_docstrings("""Albert Model with a `language modeling` head on top.""", ALBERT_START_DOCSTRING)
class FlaxAlbertForMaskedLM(FlaxAlbertPreTrainedModel):
    module_class = FlaxAlbertForMaskedLMModule



# Append call sample docstring for FlaxAlbertForMaskedLM
append_call_sample_docstring(
    # Import the specific classes and variables from the module for Flax-based Albert model for Masked Language Modeling
    FlaxAlbertForMaskedLM, _CHECKPOINT_FOR_DOC, FlaxMaskedLMOutput, _CONFIG_FOR_DOC, revision="refs/pr/11"
# 定义一个名为 FlaxAlbertForSequenceClassificationModule 的 PyTorch 模块，用于序列分类任务
class FlaxAlbertForSequenceClassificationModule(nn.Module):
    # 类属性，存储 Albert 模型的配置
    config: AlbertConfig
    # 类属性，默认数据类型为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 模块初始化方法
    def setup(self):
        # 根据配置创建一个 FlaxAlbertModule 实例
        self.albert = FlaxAlbertModule(config=self.config, dtype=self.dtype)
        # 根据配置中的 dropout 概率创建一个 dropout 层
        classifier_dropout = (
            self.config.classifier_dropout_prob
            if self.config.classifier_dropout_prob is not None
            else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(rate=classifier_dropout)
        # 创建一个全连接层作为分类器，输出维度为 config.num_labels
        self.classifier = nn.Dense(
            self.config.num_labels,
            dtype=self.dtype,
        )

    # 模块调用方法，用于模型推断
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 调用 Albert 模型进行前向传播
        outputs = self.albert(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从 Albert 模型的输出中获取池化后的输出
        pooled_output = outputs[1]
        # 应用 dropout 层到池化后的输出上
        pooled_output = self.dropout(pooled_output, deterministic=deterministic)
        # 将处理后的输出传入分类器得到最终的 logits
        logits = self.classifier(pooled_output)

        # 如果不要求返回字典，则返回 logits 和额外的隐藏状态
        if not return_dict:
            return (logits,) + outputs[2:]

        # 如果要求返回字典，则构建一个 FlaxSequenceClassifierOutput 对象并返回
        return FlaxSequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# 使用装饰器为 FlaxAlbertForSequenceClassification 类添加文档字符串
@add_start_docstrings(
    """
    Albert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    ALBERT_START_DOCSTRING,
)
# 继承自 FlaxAlbertPreTrainedModel 类的子类
class FlaxAlbertForSequenceClassification(FlaxAlbertPreTrainedModel):
    # 指定该模型使用的模块类为 FlaxAlbertForSequenceClassificationModule
    module_class = FlaxAlbertForSequenceClassificationModule


# 为 FlaxAlbertForSequenceClassification 类添加调用示例的文档字符串
append_call_sample_docstring(
    FlaxAlbertForSequenceClassification,
    _CHECKPOINT_FOR_DOC,
    FlaxSequenceClassifierOutput,
    _CONFIG_FOR_DOC,
)


# 定义一个名为 FlaxAlbertForMultipleChoiceModule 的 PyTorch 模块，用于多项选择任务
class FlaxAlbertForMultipleChoiceModule(nn.Module):
    # 类属性，存储 Albert 模型的配置
    config: AlbertConfig
    # 类属性，默认数据类型为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 模块初始化方法
    def setup(self):
        # 根据配置创建一个 FlaxAlbertModule 实例
        self.albert = FlaxAlbertModule(config=self.config, dtype=self.dtype)
        # 创建一个 dropout 层，dropout 率为 config.hidden_dropout_prob
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        # 创建一个全连接层作为分类器，输出维度为 1
        self.classifier = nn.Dense(1, dtype=self.dtype)

    # 模块调用方法，用于模型推断
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        ):
            # 获取输入张量的第二维大小，即选项的数量
            num_choices = input_ids.shape[1]
            # 如果输入张量不为空，则重塑为二维张量
            input_ids = input_ids.reshape(-1, input_ids.shape[-1]) if input_ids is not None else None
            attention_mask = attention_mask.reshape(-1, attention_mask.shape[-1]) if attention_mask is not None else None
            token_type_ids = token_type_ids.reshape(-1, token_type_ids.shape[-1]) if token_type_ids is not None else None
            position_ids = position_ids.reshape(-1, position_ids.shape[-1]) if position_ids is not None else None

            # 使用 ALBERT 模型进行前向推断
            outputs = self.albert(
                input_ids,
                attention_mask,
                token_type_ids,
                position_ids,
                deterministic=deterministic,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            # 获取池化后的输出
            pooled_output = outputs[1]
            # 使用 dropout 进行池化后输出的处理
            pooled_output = self.dropout(pooled_output, deterministic=deterministic)
            # 使用分类器进行分类预测
            logits = self.classifier(pooled_output)

            # 将 logits 重塑为二维张量，以便与选项数量对应
            reshaped_logits = logits.reshape(-1, num_choices)

            # 如果不返回字典形式的结果，则返回 logits 以及额外的输出
            if not return_dict:
                return (reshaped_logits,) + outputs[2:]

            # 返回多选题模型的输出结果，包括 logits、隐藏状态和注意力
            return FlaxMultipleChoiceModelOutput(
                logits=reshaped_logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
@add_start_docstrings(
    """
    Albert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    ALBERT_START_DOCSTRING,
)
class FlaxAlbertForMultipleChoice(FlaxAlbertPreTrainedModel):
    module_class = FlaxAlbertForMultipleChoiceModule


overwrite_call_docstring(
    FlaxAlbertForMultipleChoice, ALBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
)
append_call_sample_docstring(
    FlaxAlbertForMultipleChoice,
    _CHECKPOINT_FOR_DOC,
    FlaxMultipleChoiceModelOutput,
    _CONFIG_FOR_DOC,
)

这部分代码定义了一个基于Albert模型的多选题分类模型，包括一个线性层和softmax输出，适用于多选题任务，例如RocStories/SWAG任务。


class FlaxAlbertForTokenClassificationModule(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.albert = FlaxAlbertModule(config=self.config, dtype=self.dtype, add_pooling_layer=False)
        classifier_dropout = (
            self.config.classifier_dropout_prob
            if self.config.classifier_dropout_prob is not None
            else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(rate=classifier_dropout)
        self.classifier = nn.Dense(self.config.num_labels, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # Model
        outputs = self.albert(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
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

这段代码定义了一个基于Albert模型的Token分类模型，该模型在隐藏状态输出的基础上增加了线性层，适用于词元分类任务（例如命名实体识别）。其中包括了用于构建模型的初始化设置和__call__方法用于执行前向传播。


@add_start_docstrings(
    """
    Albert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    ALBERT_START_DOCSTRING,
)
class FlaxAlbertForTokenClassification(FlaxAlbertPreTrainedModel):
    module_class = FlaxAlbertForTokenClassificationModule


append_call_sample_docstring(
    FlaxAlbertForTokenClassification,
    _CHECKPOINT_FOR_DOC,
    FlaxTokenClassifierOutput,
    _CONFIG_FOR_DOC,
)

这部分代码定义了另一个基于Albert模型的Token分类模型，用于词元分类任务（如命名实体识别），包括一个线性层作为隐藏状态输出的顶部层。


class FlaxAlbertForQuestionAnsweringModule(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = jnp.float32

这段代码定义了一个用于问答任务的Albert模型模块，但是在这段代码中缺少进一步的实现和注释。
    # 初始化模型设置
    def setup(self):
        # 使用配置和数据类型创建一个 FlaxAlbertModule 实例，不添加池化层
        self.albert = FlaxAlbertModule(config=self.config, dtype=self.dtype, add_pooling_layer=False)
        # 创建一个全连接层 nn.Dense，输出维度为配置中指定的标签数
        self.qa_outputs = nn.Dense(self.config.num_labels, dtype=self.dtype)

    # 模型调用函数，接受多个输入和一些可选参数
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 调用 self.albert 模型进行前向传播
        outputs = self.albert(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从模型输出中获取隐藏状态
        hidden_states = outputs[0]

        # 使用 self.qa_outputs 对隐藏状态进行线性变换得到预测 logits
        logits = self.qa_outputs(hidden_states)

        # 将 logits 按最后一个维度分割成起始和结束 logits
        start_logits, end_logits = logits.split(self.config.num_labels, axis=-1)

        # 去除多余的维度，将 logits 压缩成一维张量
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # 如果 return_dict 为 False，则返回一个元组，包含 logits 和额外的模型输出
        if not return_dict:
            return (start_logits, end_logits) + outputs[1:]

        # 如果 return_dict 为 True，则封装输出成 FlaxQuestionAnsweringModelOutput 类型
        return FlaxQuestionAnsweringModelOutput(
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """
    Albert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    ALBERT_START_DOCSTRING,  # 添加起始文档字符串，描述了在Albert模型中加入用于抽取式问答任务的span分类头的结构和功能。
)
class FlaxAlbertForQuestionAnswering(FlaxAlbertPreTrainedModel):
    module_class = FlaxAlbertForQuestionAnsweringModule


append_call_sample_docstring(
    FlaxAlbertForQuestionAnswering,  # 将示例调用添加到文档字符串，展示如何使用FlaxAlbertForQuestionAnswering类的示例。
    _CHECKPOINT_FOR_DOC,
    FlaxQuestionAnsweringModelOutput,  # 调用样本文档字符串附上了模型输出的检查点。
    _CONFIG_FOR_DOC,
)
```