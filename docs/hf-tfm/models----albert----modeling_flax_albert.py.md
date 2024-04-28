# `.\transformers\models\albert\modeling_flax_albert.py`

```
# 设置文件编码为 UTF-8
# 版权声明，版权归 Google AI、Google Brain 和 HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本授权，除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制

# 导入所需的模块和类
from typing import Callable, Optional, Tuple
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax

# 导入模型输出相关的类
from ...modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxBaseModelOutputWithPooling,
    FlaxMaskedLMOutput,
    FlaxMultipleChoiceModelOutput,
    FlaxQuestionAnsweringModelOutput,
    FlaxSequenceClassifierOutput,
    FlaxTokenClassifierOutput,
)
# 导入模型相关的工具函数和类
from ...modeling_flax_utils import (
    ACT2FN,
    FlaxPreTrainedModel,
    append_call_sample_docstring,
    append_replace_return_docstrings,
    overwrite_call_docstring,
)
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_albert import AlbertConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置
_CHECKPOINT_FOR_DOC = "albert-base-v2"
_CONFIG_FOR_DOC = "AlbertConfig"

# FlaxAlbertForPreTrainingOutput 类型的输出
@flax.struct.dataclass
class FlaxAlbertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`FlaxAlbertForPreTraining`].
    Args:
        prediction_logits (`jnp.ndarray` of shape `(batch_size, sequence_length, config.vocab_size)`):
            预测语言建模头部的预测分数（SoftMax之前的每个词汇标记的分数）。
        sop_logits (`jnp.ndarray` of shape `(batch_size, 2)`):
            下一个序列预测（分类）头部的预测分数（SoftMax之前的True/False延续的分数）。
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            `jnp.ndarray`的元组（一个用于嵌入的输出 + 一个用于每个层的输出），形状为`(batch_size, sequence_length, hidden_size)`。

            模型在每个层的输出以及初始嵌入输出的隐藏状态。
        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            `jnp.ndarray`的元组（每个层一个），形状为`(batch_size, num_heads, sequence_length, sequence_length)`。

            在注意力 softmax 之后的注意力权重，用于计算自注意力头部中的加权平均值。
    """

    prediction_logits: jnp.ndarray = None
    sop_logits: jnp.ndarray = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    attentions: Optional[Tuple[jnp.ndarray]] = None
# ALBERT_START_DOCSTRING 是一个字符串变量，用于定义模型文档字符串的起始部分，包含了一些说明性文字和参数说明
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

# ALBERT_INPUTS_DOCSTRING 是一个字符串变量，用于定义模型输入文档字符串的部分，暂时为空，后续可能会填充
ALBERT_INPUTS_DOCSTRING = r"""

"""
    Args:
        input_ids (`numpy.ndarray` of shape `({0})`):
            # 输入序列标记在词汇表中的索引。
            # 可以使用 `AutoTokenizer` 来获取索引。详情请参见 `PreTrainedTokenizer.encode` 和 `PreTrainedTokenizer.__call__`。
            # [输入 ID 是什么？](../glossary#input-ids)
        attention_mask (`numpy.ndarray` of shape `({0})`, *optional*):
            # 遮罩，避免在填充的标记索引上执行注意力操作。
            # 遮罩值选择在 `[0, 1]`：

            # - 1 表示**未遮罩**的标记，
            # - 0 表示**已遮罩**的标记。

            # [注意力遮罩是什么？](../glossary#attention-mask)
        token_type_ids (`numpy.ndarray` of shape `({0})`, *optional*):
            # 段标记索引，指示输入的第一部分和第二部分。索引选择在 `[0, 1]`：

            # - 0 对应于*句子 A*的标记，
            # - 1 对应于*句子 B*的标记。

            # [什么是标记类型 ID？](../glossary#token-type-ids)
        position_ids (`numpy.ndarray` of shape `({0})`, *optional*):
            # 每个输入序列标记在位置嵌入中的位置索引。选择范围为 `[0, config.max_position_embeddings - 1]`。
        return_dict (`bool`, *optional*):
            # 是否返回一个[`~utils.ModelOutput`]而不是普通的元组。
```  
"""
# 定义 FlaxAlbertEmbeddings 类，用于构建词嵌入、位置嵌入和标记类型嵌入
class FlaxAlbertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    # 配置参数
    config: AlbertConfig
    # 计算时的数据类型
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    # 模块初始化方法
    def setup(self):
        # 初始化词嵌入层
        self.word_embeddings = nn.Embed(
            # 词汇表大小
            self.config.vocab_size,
            # 嵌入维度大小
            self.config.embedding_size,
            # 使用正态分布初始化
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        # 初始化位置嵌入层
        self.position_embeddings = nn.Embed(
            # 最大位置嵌入大小
            self.config.max_position_embeddings,
            # 嵌入维度大小
            self.config.embedding_size,
            # 使用正态分布初始化
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        # 初始化标记类型嵌入层
        self.token_type_embeddings = nn.Embed(
            # 标记类型嵌入大小
            self.config.type_vocab_size,
            # 嵌入维度大小
            self.config.embedding_size,
            # 使用正态分布初始化
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        # 初始化 LayerNorm 层
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 初始化 Dropout 层
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    # 实现 __call__ 方法
    def __call__(self, input_ids, token_type_ids, position_ids, deterministic: bool = True):
        # 嵌入输入词
        inputs_embeds = self.word_embeddings(input_ids.astype("i4"))
        # 嵌入位置信息
        position_embeds = self.position_embeddings(position_ids.astype("i4"))
        # 嵌入标记类型信息
        token_type_embeddings = self.token_type_embeddings(token_type_ids.astype("i4"))

        # 汇总所有嵌入
        hidden_states = inputs_embeds + token_type_embeddings + position_embeds

        # Layer Norm
        hidden_states = self.LayerNorm(hidden_states)
        # Dropout
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states


# 定义 FlaxAlbertSelfAttention 类
class FlaxAlbertSelfAttention(nn.Module):
    # 配置参数
    config: AlbertConfig
    # 计算时的数据类型
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    # 在模型初始化时进行设置
    def setup(self):
        # 检查隐藏层大小是否是注意力头数的倍数，如果不是则引发错误
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError(
                "`config.hidden_size`: {self.config.hidden_size} 必须是 `config.num_attention_heads` "
                "                   的倍数: {self.config.num_attention_heads}"
            )

        # 初始化查询层，用于生成查询向量
        self.query = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        # 初始化键层，用于生成键向量
        self.key = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        # 初始化值层，用于生成值向量
        self.value = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        # 初始化全连接层，用于处理注意力机制后的结果
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 初始化 LayerNorm 层，用于归一化每个样本的特征向量
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 初始化 Dropout 层，用于随机丢弃一部分神经元，防止过拟合
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
    # 定义一个方法，用于执行自注意力机制
    def __call__(self, hidden_states, attention_mask, deterministic=True, output_attentions: bool = False):
        # 计算每个头部的维度
        head_dim = self.config.hidden_size // self.config.num_attention_heads

        # 使用查询权重矩阵查询隐藏状态，并重塑结果以适应多头注意力机制的形状
        query_states = self.query(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )
        # 使用值权重矩阵查询隐藏状态，并重塑结果以适应多头注意力机制的形状
        value_states = self.value(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )
        # 使用键权重矩阵查询隐藏状态，并重塑结果以适应多头注意力机制的形状
        key_states = self.key(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )

        # 将布尔类型的注意力掩码转换为注意力偏置
        if attention_mask is not None:
            # 将注意力掩码扩展维度以适应多头注意力机制的形状
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
            # 根据注意力掩码生成注意力偏置
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
            )
        else:
            attention_bias = None

        dropout_rng = None
        # 如果不是确定性操作且存在注意力概率的丢弃，则创建丢弃随机数生成器
        if not deterministic and self.config.attention_probs_dropout_prob > 0.0:
            dropout_rng = self.make_rng("dropout")

        # 计算注意力权重
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

        # 使用注意力权重计算注意力输出
        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)
        # 重塑注意力输出以适应全连接层的输入形状
        attn_output = attn_output.reshape(attn_output.shape[:2] + (-1,))

        # 将注意力输出传递给全连接层
        projected_attn_output = self.dense(attn_output)
        # 对全连接层的输出应用丢弃操作
        projected_attn_output = self.dropout(projected_attn_output, deterministic=deterministic)
        # 将丢弃后的输出与原始隐藏状态相加，并应用层归一化
        layernormed_attn_output = self.LayerNorm(projected_attn_output + hidden_states)
        # 如果需要输出注意力权重，则将其包含在输出中
        outputs = (layernormed_attn_output, attn_weights) if output_attentions else (layernormed_attn_output,)
        # 返回模型输出
        return outputs
# FlaxAlbertLayer 类定义
class FlaxAlbertLayer(nn.Module):
    # AlbertConfig 类型的配置属性
    config: AlbertConfig
    # 计算的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    # 模块初始化方法
    def setup(self):
        # 初始化注意力机制模块
        self.attention = FlaxAlbertSelfAttention(self.config, dtype=self.dtype)
        # 初始化前馈神经网络模块
        self.ffn = nn.Dense(
            self.config.intermediate_size,
            # 使用正态分布初始化权重参数
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 激活函数
        self.activation = ACT2FN[self.config.hidden_act]
        # 初始化前馈神经网络输出模块
        self.ffn_output = nn.Dense(
            self.config.hidden_size,
            # 使用正态分布初始化权重参数
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 初始化全连接层的 LayerNorm
        self.full_layer_layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 初始化 Dropout 模块
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    # 模块的调用方法
    def __call__(
        self,
        hidden_states,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
    ):
        # 使用注意力机制模块处理隐藏状态
        attention_outputs = self.attention(
            hidden_states, attention_mask, deterministic=deterministic, output_attentions=output_attentions
        )
        attention_output = attention_outputs[0]
        # 前馈神经网络
        ffn_output = self.ffn(attention_output)
        # 激活函数
        ffn_output = self.activation(ffn_output)
        # 前馈神经网络输出
        ffn_output = self.ffn_output(ffn_output)
        # Dropout
        ffn_output = self.dropout(ffn_output, deterministic=deterministic)
        # LayerNorm
        hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)

        outputs = (hidden_states,)

        # 如果需要输出注意力矩阵，则添加到输出中
        if output_attentions:
            outputs += (attention_outputs[1],)
        return outputs


# FlaxAlbertLayerCollection 类定义
class FlaxAlbertLayerCollection(nn.Module):
    # AlbertConfig 类型的配置属性
    config: AlbertConfig
    # 计算的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    # 模块初始化方法
    def setup(self):
        # 初始化多层 AlbertLayer 模块组成的列表
        self.layers = [
            FlaxAlbertLayer(self.config, name=str(i), dtype=self.dtype) for i in range(self.config.inner_group_num)
        ]

    # 模块的调用方法
    def __call__(
        self,
        hidden_states,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        ):
        # 初始化空元组，用于存储每个层的隐藏状态
        layer_hidden_states = ()
        # 初始化空元组，用于存储每个层的注意力权重
        layer_attentions = ()

        # 遍历每个 ALBERT 模型层
        for layer_index, albert_layer in enumerate(self.layers):
            # 调用 ALBERT 模型层的前向传播方法，获取输出
            layer_output = albert_layer(
                hidden_states,
                attention_mask,
                deterministic=deterministic,
                output_attentions=output_attentions,
            )
            # 更新隐藏状态为当前层的输出
            hidden_states = layer_output[0]

            # 如果需要输出注意力权重
            if output_attentions:
                # 将当前层的注意力权重添加到元组中
                layer_attentions = layer_attentions + (layer_output[1],)

            # 如果需要输出隐藏状态
            if output_hidden_states:
                # 将当前层的隐藏状态添加到元组中
                layer_hidden_states = layer_hidden_states + (hidden_states,)

        # 初始化输出结果为包含最后一层隐藏状态的元组
        outputs = (hidden_states,)
        # 如果需要输出隐藏状态
        if output_hidden_states:
            # 将所有层的隐藏状态添加到输出结果中
            outputs = outputs + (layer_hidden_states,)
        # 如果需要输出注意力权重
        if output_attentions:
            # 将所有层的注意力权重添加到输出结果中
            outputs = outputs + (layer_attentions,)
        # 返回输出结果，包括最后一层隐藏状态、所有层的隐藏状态和注意力权重
        return outputs  # last-layer hidden state, (layer hidden states), (layer attentions)
class FlaxAlbertLayerCollections(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    layer_index: Optional[str] = None

    def setup(self):
        # 初始化 Albert 层集合
        self.albert_layers = FlaxAlbertLayerCollection(self.config, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        # 调用 Albert 层集合的前向传播方法
        outputs = self.albert_layers(
            hidden_states,
            attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        return outputs


class FlaxAlbertLayerGroups(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        # 初始化 Albert 层组
        self.layers = [
            FlaxAlbertLayerCollections(self.config, name=str(i), layer_index=str(i), dtype=self.dtype)
            for i in range(self.config.num_hidden_groups)
        ]

    def __call__(
        self,
        hidden_states,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 初始化注意力和隐藏状态
        all_attentions = () if output_attentions else None
        all_hidden_states = (hidden_states,) if output_hidden_states else None

        for i in range(self.config.num_hidden_layers):
            # 计算隐藏组的索引
            group_idx = int(i / (self.config.num_hidden_layers / self.config.num_hidden_groups))
            # 调用相应隐藏组的前向传播方法
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


class FlaxAlbertEncoder(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        # 初始化嵌入层映射
        self.embedding_hidden_mapping_in = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 初始化 Albert 编码器的层组
        self.albert_layer_groups = FlaxAlbertLayerGroups(self.config, dtype=self.dtype)
    # 定义 __call__ 方法，该方法用于将对象实例像函数一样调用
    def __call__(
        self,
        # 输入参数 hidden_states：模型的隐藏状态，即输入的特征表示
        hidden_states,
        # 输入参数 attention_mask：用于指示哪些位置需要被关注的掩码
        attention_mask,
        # 输入参数 deterministic：指示是否使用确定性模式进行推理，默认为 True
        deterministic: bool = True,
        # 输入参数 output_attentions：指示是否输出注意力权重，默认为 False
        output_attentions: bool = False,
        # 输入参数 output_hidden_states：指示是否输出每一层的隐藏状态，默认为 False
        output_hidden_states: bool = False,
        # 输入参数 return_dict：指示是否返回结果字典，默认为 True
        return_dict: bool = True,
    ):
        # 将输入的隐藏状态进行映射转换
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)
        # 调用 ALBERT 层组，对隐藏状态进行处理
        return self.albert_layer_groups(
            hidden_states,
            attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
class FlaxAlbertOnlyMLMHead(nn.Module):
    # 定义一个类，继承自 nn.Module
    config: AlbertConfig
    # 定义一个属性 config，类型为 AlbertConfig
    dtype: jnp.dtype = jnp.float32
    # 定义一个属性 dtype，默认值为 jnp.float32
    bias_init: Callable[..., np.ndarray] = jax.nn.initializers.zeros
    # 定义一个属性 bias_init，类型为 Callable[..., np.ndarray]，默认值为 jax.nn.initializers.zeros

    def setup(self):
        # 定义一个方法 setup
        self.dense = nn.Dense(self.config.embedding_size, dtype=self.dtype)
        # 初始化一个 nn.Dense 层，输入大小为 config.embedding_size，数据类型为 dtype
        self.activation = ACT2FN[self.config.hidden_act]
        # 获取激活函数，根据 config 中的 hidden_act
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 初始化一个 nn.LayerNorm 层，epsilon 为 config 中的 layer_norm_eps，数据类型为 dtype
        self.decoder = nn.Dense(self.config.vocab_size, dtype=self.dtype, use_bias=False)
        # 初始化一个 nn.Dense 层，输入大小为 config.vocab_size，数据类型为 dtype，不使用偏置
        self.bias = self.param("bias", self.bias_init, (self.config.vocab_size,))
        # 初始化一个参数 bias，使用 bias_init 函数，大小为 (config.vocab_size,)

    def __call__(self, hidden_states, shared_embedding=None):
        # 定义一个调用方法
        hidden_states = self.dense(hidden_states)
        # 使用 dense 层处理 hidden_states
        hidden_states = self.activation(hidden_states)
        # 使用激活函数处理 hidden_states
        hidden_states = self.LayerNorm(hidden_states)
        # 使用 LayerNorm 处理 hidden_states

        if shared_embedding is not None:
            # 如果 shared_embedding 不为 None
            hidden_states = self.decoder.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
            # 使用 decoder 层处理 hidden_states，传入参数为 shared_embedding 的转置
        else:
            hidden_states = self.decoder(hidden_states)
            # 使用 decoder 层处理 hidden_states

        hidden_states += self.bias
        # 将 bias 加到 hidden_states 上
        return hidden_states
        # 返回处理后的 hidden_states


class FlaxAlbertSOPHead(nn.Module):
    # 定义一个类，继承自 nn.Module
    config: AlbertConfig
    # 定义一个属性 config，类型为 AlbertConfig
    dtype: jnp.dtype = jnp.float32
    # 定义一个属性 dtype，默认值为 jnp.float32

    def setup(self):
        # 定义一个方法 setup
        self.dropout = nn.Dropout(self.config.classifier_dropout_prob)
        # 初始化一个 nn.Dropout 层，概率为 config 中的 classifier_dropout_prob
        self.classifier = nn.Dense(2, dtype=self.dtype)
        # 初始化一个 nn.Dense 层，输出大小为 2，数据类型为 dtype

    def __call__(self, pooled_output, deterministic=True):
        # 定义一个调用方法
        pooled_output = self.dropout(pooled_output, deterministic=deterministic)
        # 使用 dropout 处理 pooled_output
        logits = self.classifier(pooled_output)
        # 使用 classifier 处理 pooled_output
        return logits
        # 返回处理后的 logits


class FlaxAlbertPreTrainedModel(FlaxPreTrainedModel):
    # 定义一个类，继承自 FlaxPreTrainedModel
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = AlbertConfig
    # 定义一个属性 config_class，值为 AlbertConfig
    base_model_prefix = "albert"
    # 定义一个属性 base_model_prefix，值为 "albert"
    module_class: nn.Module = None
    # 定义一个属性 module_class，类型为 nn.Module，默认值为 None

    def __init__(
        self,
        config: AlbertConfig,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        # 定义一个初始化方法
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 使用 module_class 初始化一个模块
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)
        # 调用父类的初始化方法
    # 初始化模型的权重参数
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量
        input_ids = jnp.zeros(input_shape, dtype="i4")
        token_type_ids = jnp.zeros_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        attention_mask = jnp.ones_like(input_ids)

        # 分割随机数种子用于参数和dropout
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 初始化模型参数
        random_params = self.module.init(
            rngs, input_ids, attention_mask, token_type_ids, position_ids, return_dict=False
        )["params"]

        # 如果传入了参数，则使用传入参数，否则使用随机初始化的参数
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    # 模型调用函数，处理输入和返回结果
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
        # 设置输出注意力和隐藏状态的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 如果未传入token_type_ids，则初始化为全零
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        # 如果未传入position_ids，则根据input_ids的长度广播生成
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        # 如果未传入attention_mask，则初始化为全一
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        # 处理任何需要的PRNG
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        # 应用模型，传入参数和输入张量
        return self.module.apply(
            {"params": params or self.params},
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(token_type_ids, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            not train,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
        )
# 定义一个 FlaxAlbertModule 类，继承自 nn.Module
class FlaxAlbertModule(nn.Module):
    # Albert 模型的配置
    config: AlbertConfig
    # 计算的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    # 是否添加池化层，默认为 True
    add_pooling_layer: bool = True

    # 模型初始化方法
    def setup(self):
        # 初始化嵌入层
        self.embeddings = FlaxAlbertEmbeddings(self.config, dtype=self.dtype)
        # 初始化编码器
        self.encoder = FlaxAlbertEncoder(self.config, dtype=self.dtype)
        # 如果需要添加池化层
        if self.add_pooling_layer:
            # 初始化池化层
            self.pooler = nn.Dense(
                self.config.hidden_size,
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
                dtype=self.dtype,
                name="pooler",
            )
            # 池化层的激活函数为 tanh
            self.pooler_activation = nn.tanh
        else:
            self.pooler = None
            self.pooler_activation = None

    # 模型调用方法
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
        # 确保当未传入 token_type_ids 时，正确初始化为全零数组
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        # 确保当未传入 position_ids 时，正确初始化为广播后的数组
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        # 获取嵌入层的隐藏状态
        hidden_states = self.embeddings(input_ids, token_type_ids, position_ids, deterministic=deterministic)

        # 获取编码器的输出
        outputs = self.encoder(
            hidden_states,
            attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        # 如果需要添加池化层
        if self.add_pooling_layer:
            # 对隐藏状态进行池化
            pooled = self.pooler(hidden_states[:, 0])
            pooled = self.pooler_activation(pooled)
        else:
            pooled = None

        # 如果不返回字典
        if not return_dict:
            # 如果 pooled 为 None，则不返回它
            if pooled is None:
                return (hidden_states,) + outputs[1:]
            return (hidden_states, pooled) + outputs[1:]

        # 返回带有池化输出的 FlaxBaseModelOutputWithPooling 对象
        return FlaxBaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# 添加 Albert 模型的文档字符串
@add_start_docstrings(
    "The bare Albert Model transformer outputting raw hidden-states without any specific head on top.",
    ALBERT_START_DOCSTRING,
)
# 定义 FlaxAlbertModel 类，继承自 FlaxAlbertPreTrainedModel
class FlaxAlbertModel(FlaxAlbertPreTrainedModel):
    # 模型类为 FlaxAlbertModule
    module_class = FlaxAlbertModule


# 添加调用示例的文档字符串
append_call_sample_docstring(FlaxAlbertModel, _CHECKPOINT_FOR_DOC, FlaxBaseModelOutputWithPooling, _CONFIG_FOR_DOC)


# 定义一个 FlaxAlbertForPreTrainingModule 类，继承自 nn.Module
class FlaxAlbertForPreTrainingModule(nn.Module):
    # Albert 模型的配置
    config: AlbertConfig
    # 设置默认数据类型为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 初始化模型
    def setup(self):
        # 初始化 Albert 模型
        self.albert = FlaxAlbertModule(config=self.config, dtype=self.dtype)
        # 初始化仅包含 MLM 头部的模型
        self.predictions = FlaxAlbertOnlyMLMHead(config=self.config, dtype=self.dtype)
        # 初始化 SOP 头部模型
        self.sop_classifier = FlaxAlbertSOPHead(config=self.config, dtype=self.dtype)

    # 调用模型
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
        # 调用 Albert 模型
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

        # 如果配置中要求词嵌入共享，则获取共享的词嵌入
        if self.config.tie_word_embeddings:
            shared_embedding = self.albert.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
        else:
            shared_embedding = None

        # 获取隐藏状态和池化输出
        hidden_states = outputs[0]
        pooled_output = outputs[1]

        # 获取预测分数和SOP分数
        prediction_scores = self.predictions(hidden_states, shared_embedding=shared_embedding)
        sop_scores = self.sop_classifier(pooled_output, deterministic=deterministic)

        # 如果不返回字典，则返回元组
        if not return_dict:
            return (prediction_scores, sop_scores) + outputs[2:]

        # 返回预训练输出字典
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
# 定义了用于预训练的 Albert 模型，包含了一个“掩码语言建模”头和一个“句子顺序预测（分类）”头
class FlaxAlbertForPreTraining(FlaxAlbertPreTrainedModel):
    module_class = FlaxAlbertForPreTrainingModule


FLAX_ALBERT_FOR_PRETRAINING_DOCSTRING = """
    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, FlaxAlbertForPreTraining

    >>> tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
    >>> model = FlaxAlbertForPreTraining.from_pretrained("albert-base-v2")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="np")
    >>> outputs = model(**inputs)

    >>> prediction_logits = outputs.prediction_logits
    >>> seq_relationship_logits = outputs.sop_logits
    ```
"""
# 添加了预训练 Albert 模型的文档字符串
overwrite_call_docstring(
    FlaxAlbertForPreTraining,
    ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length") + FLAX_ALBERT_FOR_PRETRAINING_DOCSTRING,
)
# 追加替换了预训练 Albert 模型的返回值文档字符串
append_replace_return_docstrings(
    FlaxAlbertForPreTraining, output_type=FlaxAlbertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC
)


class FlaxAlbertForMaskedLMModule(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # Albert 模型，不包含池化层
        self.albert = FlaxAlbertModule(config=self.config, add_pooling_layer=False, dtype=self.dtype)
        # 只包含掩码语言建模头部
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
        # 模型前向传播
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
        if self.config.tie_word_embeddings:
            # 共享词嵌入矩阵
            shared_embedding = self.albert.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
        else:
            shared_embedding = None

        # 计算预测分数
        logits = self.predictions(hidden_states, shared_embedding=shared_embedding)

        if not return_dict:
            return (logits,) + outputs[1:]

        return FlaxMaskedLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings("""Albert Model with a `language modeling` head on top.""", ALBERT_START_DOCSTRING)
# 定义了带有语言建模头的 Albert 模型
class FlaxAlbertForMaskedLM(FlaxAlbertPreTrainedModel):
    module_class = FlaxAlbertForMaskedLMModule


# 追加了调用示例的文档字符串
append_call_sample_docstring(
    # 导入 FlaxAlbertForMaskedLM 模型、_CHECKPOINT_FOR_DOC 变量、FlaxMaskedLMOutput 类、_CONFIG_FOR_DOC 变量以及 revision 参数为 "refs/pr/11"
    from transformers import FlaxAlbertForMaskedLM, _CHECKPOINT_FOR_DOC, FlaxMaskedLMOutput, _CONFIG_FOR_DOC, revision="refs/pr/11"
# Flax 模块，用于序列分类任务
class FlaxAlbertForSequenceClassificationModule(nn.Module):
    # Albert 模型的配置
    config: AlbertConfig
    # 数据类型，默认为 32 位浮点数
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 初始化 Albert 模型
        self.albert = FlaxAlbertModule(config=self.config, dtype=self.dtype)
        # 分类器的丢弃率，如果未指定，则使用隐藏层丢弃率
        classifier_dropout = (
            self.config.classifier_dropout_prob
            if self.config.classifier_dropout_prob is not None
            else self.config.hidden_dropout_prob
        )
        # 初始化丢弃层
        self.dropout = nn.Dropout(rate=classifier_dropout)
        # 分类器层，输出大小为标签数
        self.classifier = nn.Dense(
            self.config.num_labels,
            dtype=self.dtype,
        )

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
        # 模型前向传播
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

        # 池化输出
        pooled_output = outputs[1]
        # 使用丢弃层
        pooled_output = self.dropout(pooled_output, deterministic=deterministic)
        # 使用分类器层得到 logits
        logits = self.classifier(pooled_output)

        # 如果不返回字典，则返回 logits 以及额外的隐藏状态
        if not return_dict:
            return (logits,) + outputs[2:]

        # 返回序列分类器输出
        return FlaxSequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# Albert 序列分类模型
@add_start_docstrings(
    """
    Albert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    ALBERT_START_DOCSTRING,
)
class FlaxAlbertForSequenceClassification(FlaxAlbertPreTrainedModel):
    module_class = FlaxAlbertForSequenceClassificationModule


# 添加调用示例文档字符串
append_call_sample_docstring(
    FlaxAlbertForSequenceClassification,
    _CHECKPOINT_FOR_DOC,
    FlaxSequenceClassifierOutput,
    _CONFIG_FOR_DOC,
)


# Flax 多项选择模块
class FlaxAlbertForMultipleChoiceModule(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 初始化 Albert 模型
        self.albert = FlaxAlbertModule(config=self.config, dtype=self.dtype)
        # 初始化丢弃层
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        # 多项选择分类器，输出大小为 1
        self.classifier = nn.Dense(1, dtype=self.dtype)

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
        # 获取每个选择项的数量
        num_choices = input_ids.shape[1]
        # 如果存在输入标识符，重新塑造其形状为二维数组，否则保持为 None
        input_ids = input_ids.reshape(-1, input_ids.shape[-1]) if input_ids is not None else None
        # 如果存在注意力遮罩，重新塑造其形状为二维数组，否则保持为 None
        attention_mask = attention_mask.reshape(-1, attention_mask.shape[-1]) if attention_mask is not None else None
        # 如果存在标记类型标识符，重新塑造其形状为二维数组，否则保持为 None
        token_type_ids = token_type_ids.reshape(-1, token_type_ids.shape[-1]) if token_type_ids is not None else None
        # 如果存在位置标识符，重新塑造其形状为二维数组，否则保持为 None
        position_ids = position_ids.reshape(-1, position_ids.shape[-1]) if position_ids is not None else None

        # 调用 ALBERT 模型
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

        # 提取汇聚输出
        pooled_output = outputs[1]
        # 使用丢弃层对汇聚输出进行处理，用于防止过拟合
        pooled_output = self.dropout(pooled_output, deterministic=deterministic)
        # 经过分类器生成分类结果
        logits = self.classifier(pooled_output)

        # 重新塑造分类结果的形状为二维数组
        reshaped_logits = logits.reshape(-1, num_choices)

        # 如果不返回字典，则返回重塑后的 logits 以及其他输出的元组
        if not return_dict:
            return (reshaped_logits,) + outputs[2:]

        # 如果返回字典，则构造 FlaxMultipleChoiceModelOutput 对象并返回
        return FlaxMultipleChoiceModelOutput(
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用多项选择分类头部的 Albert 模型（在汇总输出之上有一个线性层和 softmax），例如用于 RocStories/SWAG 任务
class FlaxAlbertForMultipleChoice(FlaxAlbertPreTrainedModel):
    module_class = FlaxAlbertForMultipleChoiceModule

# 覆盖调用文档字符串，描述输入参数
overwrite_call_docstring(
    FlaxAlbertForMultipleChoice, ALBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
)

# 添加调用示例文档字符串
append_call_sample_docstring(
    FlaxAlbertForMultipleChoice,
    _CHECKPOINT_FOR_DOC,
    FlaxMultipleChoiceModelOutput,
    _CONFIG_FOR_DOC,
)

# Albert 模型的令牌分类头部模块
class FlaxAlbertForTokenClassificationModule(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 使用 Albert 模块，不添加汇总层
        self.albert = FlaxAlbertModule(config=self.config, dtype=self.dtype, add_pooling_layer=False)
        # 分类器的 dropout 率
        classifier_dropout = (
            self.config.classifier_dropout_prob
            if self.config.classifier_dropout_prob is not None
            else self.config.hidden_dropout_prob
        )
        # dropout 层
        self.dropout = nn.Dropout(rate=classifier_dropout)
        # 分类器
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
        # 模型
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

        # 隐藏状态
        hidden_states = outputs[0]
        # 使用 dropout
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 获取 logits
        logits = self.classifier(hidden_states)

        if not return_dict:
            return (logits,) + outputs[1:]

        return FlaxTokenClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# 使用令牌分类头部的 Albert 模型（在隐藏状态输出之上有一个线性层），例如用于命名实体识别（NER）任务
@add_start_docstrings(
    """
    Albert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    ALBERT_START_DOCSTRING,
)
class FlaxAlbertForTokenClassification(FlaxAlbertPreTrainedModel):
    module_class = FlaxAlbertForTokenClassificationModule

# 添加调用示例文档字符串
append_call_sample_docstring(
    FlaxAlbertForTokenClassification,
    _CHECKPOINT_FOR_DOC,
    FlaxTokenClassifierOutput,
    _CONFIG_FOR_DOC,
)

# Albert 模型的问答头部模块
class FlaxAlbertForQuestionAnsweringModule(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = jnp.float32
    # 初始化模型
    def setup(self):
        # 初始化 Albert 模型，并关闭添加池化层选项
        self.albert = FlaxAlbertModule(config=self.config, dtype=self.dtype, add_pooling_layer=False)
        # 初始化全连接层，用于生成问题答案
        self.qa_outputs = nn.Dense(self.config.num_labels, dtype=self.dtype)

    # 定义模型的调用方法
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
        # 调用 Albert 模型，传入输入参数
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

        # 获取 Albert 模型输出的隐藏状态
        hidden_states = outputs[0]

        # 将隐藏状态输入全连接层，生成起始位置和结束位置的 logits
        logits = self.qa_outputs(hidden_states)
        # 将 logits 沿着指定维度分割，得到起始位置和结束位置的 logits
        start_logits, end_logits = logits.split(self.config.num_labels, axis=-1)
        # 压缩 logits 的最后一个维度
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # 如果不返回字典，则返回元组形式的输出
        if not return_dict:
            return (start_logits, end_logits) + outputs[1:]

        # 返回字典形式的输出
        return FlaxQuestionAnsweringModelOutput(
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 为提取式问答任务设计的 Albert 模型，包含一个用于分类的头部（在隐藏状态输出之上的线性层，用于计算“起始位置对数”和“结束位置对数”）。
# 继承自 FlaxAlbertPreTrainedModel 类
class FlaxAlbertForQuestionAnswering(FlaxAlbertPreTrainedModel):
    # 模型类为 FlaxAlbertForQuestionAnsweringModule

# 添加用于调用示例的文档字符串
append_call_sample_docstring(
    FlaxAlbertForQuestionAnswering,
    _CHECKPOINT_FOR_DOC,
    FlaxQuestionAnsweringModelOutput,
    _CONFIG_FOR_DOC,
)
```