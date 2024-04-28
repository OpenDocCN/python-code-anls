# `.\models\electra\modeling_flax_electra.py`

```py
# 指定文件编码为 utf-8
# 2021 The Google Flax Team Authors and The HuggingFace Inc. team 版权声明
# 根据 Apache 许可证 2.0 版本授权使用
# 详细许可条款请参考 http://www.apache.org/licenses/LICENSE-2.0
# 软件按"原样"分发，无论明示或暗示都不提供任何担保或条件
# 有关特定语言规定权限和限制的内容请参考许可证

# 引入所需模块和类型
from typing import Callable, Optional, Tuple
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen import partitioning as nn_partitioning
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxBaseModelOutputWithPastAndCrossAttentions,
    FlaxCausalLMOutputWithCrossAttentions,
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
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置
_CHECKPOINT_FOR_DOC = "google/electra-small-discriminator"
_CONFIG_FOR_DOC = "ElectraConfig"

# 定义 remat 函数和 FlaxElectraForPreTrainingOutput 类
remat = nn_partitioning.remat

@flax.struct.dataclass
class FlaxElectraForPreTrainingOutput(ModelOutput):
    """
    Output type of [`ElectraForPreTraining`].
    # 定义函数参数说明:
    # logits (`jnp.ndarray` of shape `(batch_size, sequence_length, config.vocab_size)`):
    #   预测语言建模头部的预测分数（SoftMax之前每个词汇标记的分数）。
    # hidden_states (`tuple(jnp.ndarray)`, *optional*, 当 `output_hidden_states=True` 传递或 `config.output_hidden_states=True` 时返回):
    #   形状为 `(batch_size, sequence_length, hidden_size)` 的 `jnp.ndarray` 元组（一个用于嵌入输出 + 一个用于每个层的输出）。
    #   模型在每一层输出的隐藏状态以及初始嵌入输出。
    # attentions (`tuple(jnp.ndarray)`, *optional*, 当 `output_attentions=True` 传递或 `config.output_attentions=True` 时返回):
    #   形状为 `(batch_size, num_heads, sequence_length, sequence_length)` 的 `jnp.ndarray` 元组（每个层一个）。
    #   自注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
    """
    
    # 定义函数参数
    logits: jnp.ndarray = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    attentions: Optional[Tuple[jnp.ndarray]] = None
ELECTRA_START_DOCSTRING = r"""

    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading, saving and converting weights from PyTorch models)

    This model is also a Flax Linen
    [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a
    regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.

    Finally, this model supports inherent JAX features such as:

    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

    Parameters:
        config ([`ElectraConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

ELECTRA_INPUTS_DOCSTRING = r"""



注释：
# 电气输入文件文档字符串。 This is a placeholder for documenting the inputs expected by the ELECTRA model. It's currently empty.
    Args:
        input_ids (`numpy.ndarray` of shape `({0})`):
            输入序列标记在词汇表中的索引。

            可以使用 [`AutoTokenizer`] 获取这些索引。详情请参阅 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`]。

            [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`numpy.ndarray` of shape `({0})`, *optional*):
            避免对填充的标记索引执行注意力操作的掩码。掩码值在 `[0, 1]` 之间：

            - 对于**未屏蔽**的标记，为 1，
            - 对于**屏蔽**的标记，为 0。

            [什么是注意力掩码？](../glossary#attention-mask)
        token_type_ids (`numpy.ndarray` of shape `({0})`, *optional*):
            段标记索引，用于指示输入的第一部分和第二部分。索引选择在 `[0, 1]` 范围内：

            - 0 对应于*句子 A*的标记，
            - 1 对应于*句子 B*的标记。

            [什么是标记类型 ID？](../glossary#token-type-ids)
        position_ids (`numpy.ndarray` of shape `({0})`, *optional*):
            每个输入序列标记在位置嵌入中的位置索引。选择在范围 `[0, config.max_position_embeddings - 1]` 内。

        head_mask (`numpy.ndarray` of shape `({0})`, `optional):
            用于屏蔽注意力模块中选定头部的掩码。掩码值在 `[0, 1]` 之间：

            - 1 表示头部**未被屏蔽**，
            - 0 表示头部**被屏蔽**。

        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""
用于构建从词嵌入、位置嵌入和标记类型嵌入中得到的嵌入。
"""

class FlaxElectraEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    config: ElectraConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        """
        配置词嵌入、位置嵌入和标记类型嵌入
        """
        self.word_embeddings = nn.Embed(
            self.config.vocab_size,
            self.config.embedding_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        self.position_embeddings = nn.Embed(
            self.config.max_position_embeddings,
            self.config.embedding_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        self.token_type_embeddings = nn.Embed(
            self.config.type_vocab_size,
            self.config.embedding_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    # Copied from transformers.models.bert.modeling_flax_bert.FlaxBertEmbeddings.__call__
    def __call__(self, input_ids, token_type_ids, position_ids, attention_mask, deterministic: bool = True):
        """
        执行嵌入操作
        """
        # Embed
        inputs_embeds = self.word_embeddings(input_ids.astype("i4"))
        position_embeds = self.position_embeddings(position_ids.astype("i4"))
        token_type_embeddings = self.token_type_embeddings(token_type_ids.astype("i4"))

        # Sum all embeddings
        hidden_states = inputs_embeds + token_type_embeddings + position_embeds

        # Layer Norm
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertSelfAttention with Bert->Electra
class FlaxElectraSelfAttention(nn.Module):
    config: ElectraConfig
    causal: bool = False
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation


注释：
    # 初始化方法，用于设置注意力头的维度等参数
    def setup(self):
        # 计算每个注意力头的维度
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        # 如果隐藏层大小不能被注意力头数整除，则引发 ValueError 异常
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError(
                # 异常信息，包含隐藏层大小和注意力头数
                "`config.hidden_size`: {self.config.hidden_size} has to be a multiple of `config.num_attention_heads` "
                "                   : {self.config.num_attention_heads}"
            )

        # 初始化查询矩阵的全连接层
        self.query = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        # 初始化键矩阵的全连接层
        self.key = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        # 初始化值矩阵的全连接层
        self.value = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

        # 如果是因果注意力，则创建因果掩码
        if self.causal:
            # 创建因果掩码，形状为 (1, 最大位置编码数)，数据类型为布尔型
            self.causal_mask = make_causal_mask(
                jnp.ones((1, self.config.max_position_embeddings), dtype="bool"), dtype="bool"
            )

    # 将隐藏状态分割成多个注意力头
    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.config.num_attention_heads, self.head_dim))

    # 将多个注意力头合并成隐藏状态
    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.config.hidden_size,))

    @nn.compact
    # 从 transformers.models.bart.modeling_flax_bart.FlaxBartAttention._concatenate_to_cache 复制过来的注释
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slightly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # 检测是否通过缺少现有缓存数据进行初始化。
        is_initialized = self.has_variable("cache", "cached_key")
        # 获取缓存的键（key）数据，如果不存在则初始化为全零数组
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        # 获取缓存的值（value）数据，如果不存在则初始化为全零数组
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        # 获取缓存索引值，如果不存在则初始化为0
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            # 获取缓存键（key）数据的形状
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # 更新键（key）和值（value）缓存，使用新的一维空间切片
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            # 使用新的键（key）更新缓存的键（key）数据
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            # 使用新的值（value）更新缓存的值（value）数据
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            # 更新缓存的键（key）和值（value）数据
            cached_key.value = key
            cached_value.value = value
            # 更新缓存索引值
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # 用于缓存的自注意力的因果掩码：我们的单个查询位置只应关注已生成和缓存的那些键位置，而不是剩余的零元素。
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            # 将缓存掩码与输入的注意力掩码进行合并
            attention_mask = combine_masks(pad_mask, attention_mask)
        # 返回更新后的键（key）、值（value）和注意力掩码
        return key, value, attention_mask

    def __call__(
        self,
        hidden_states,
        attention_mask,
        layer_head_mask,
        key_value_states: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        deterministic=True,
        output_attentions: bool = False,
# 从transformers.models.bert.modeling_flax_bert.FlaxBertSelfOutput复制代码，并将Bert->Electra
class FlaxElectraSelfOutput(nn.Module):
    config: ElectraConfig  # Electra模型的配置
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型

    def setup(self):
        self.dense = nn.Dense(  # 创建一个全连接层
            self.config.hidden_size,  # 输出大小为配置文件中的隐藏层大小
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),  # 使用正态分布初始化权重
            dtype=self.dtype,  # 数据类型为设定的类型
        )
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)  # 针对隐藏层进行 LayerNorm
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)  # 使用指定的丢弃率进行 Dropout

    def __call__(self, hidden_states, input_tensor, deterministic: bool = True):
        hidden_states = self.dense(hidden_states)  # 全连接层，将隐藏状态转换为指定大小
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)  # 使用指定的丢弃率进行 Dropout
        hidden_states = self.LayerNorm(hidden_states + input_tensor)  # 针对隐藏状态进行 LayerNorm，并与输入张量相加
        return hidden_states  # 返回处理后的隐藏状态


# 从transformers.models.bert.modeling_flax_bert.FlaxBertAttention复制代码，并将Bert->Electra
class FlaxElectraAttention(nn.Module):
    config: ElectraConfig  # Electra模型的配置
    causal: bool = False  # 是否因果的
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型

    def setup(self):
        self.self = FlaxElectraSelfAttention(self.config, causal=self.causal, dtype=self.dtype)  # 创建自注意力层
        self.output = FlaxElectraSelfOutput(self.config, dtype=self.dtype)  # 创建自注意力层的输出层

    def __call__(
        self,
        hidden_states,
        attention_mask,
        layer_head_mask,
        key_value_states=None,
        init_cache=False,
        deterministic=True,
        output_attentions: bool = False,
    ):
        # 注意力掩码的形状为(*batch_sizes, kv_length)
        # FLAX期望: attention_mask.shape == (*batch_sizes, 1, 1, kv_length)，以便广播
        attn_outputs = self.self(
            hidden_states,
            attention_mask,
            layer_head_mask=layer_head_mask,
            key_value_states=key_value_states,
            init_cache=init_cache,
            deterministic=deterministic,
            output_attentions=output_attentions,
        )  # 通过自注意力层进行计算
        attn_output = attn_outputs[0]  # 获取注意力计算的输出
        hidden_states = self.output(attn_output, hidden_states, deterministic=deterministic)  # 应用输出层

        outputs = (hidden_states,)  # 构建输出元组

        if output_attentions:  # 如果需要输出注意力
            outputs += (attn_outputs[1],)  # 将注意力信息添加到输出元组中

        return outputs  # 返回输出元组


# 从transformers.models.bert.modeling_flax_bert.FlaxBertIntermediate复制代码，并将Bert->Electra
class FlaxElectraIntermediate(nn.Module):
    config: ElectraConfig  # Electra模型的配置
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型

    def setup(self):
        self.dense = nn.Dense(  # 创建一个全连接层
            self.config.intermediate_size,  # 输出大小为配置文件中的中间层大小
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),  # 使用正态分布初始化权重
            dtype=self.dtype,  # 数据类型为设定的类型
        )
        self.activation = ACT2FN[self.config.hidden_act]  # 激活函数根据配置文件中的隐藏激活函数选择
    # 定义类的调用方法，传入参数 hidden_states
    def __call__(self, hidden_states):
        # 使用self.dense对隐藏状态进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的隐藏状态进行激活函数处理
        hidden_states = self.activation(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states
# 从 transformers.models.bert.modeling_flax_bert.FlaxBertOutput 复制代码，并将Bert->Electra
class FlaxElectraOutput(nn.Module):
    config: ElectraConfig
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型

    def setup(self):
        # 创建一个全连接层，输出大小为 config.hidden_size，使用正态分布初始化权重
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 创建一个dropout层，概率为config.hidden_dropout_prob
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        # 创建一个LayerNorm层，epsilon为config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    def __call__(self, hidden_states, attention_output, deterministic: bool = True):
        # 将隐藏状态传入全连接层
        hidden_states = self.dense(hidden_states)
        # 对全连接层的输出进行dropout
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 将dropout后的输出与注意力输出相加，并传入LayerNorm层
        hidden_states = self.LayerNorm(hidden_states + attention_output)
        return hidden_states


# 从 transformers.models.bert.modeling_flax_bert.FlaxBertLayer 复制代码，并将Bert->Electra
class FlaxElectraLayer(nn.Module):
    config: ElectraConfig
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型

    def setup(self):
        # 创建一个Electra注意力层
        self.attention = FlaxElectraAttention(self.config, causal=self.config.is_decoder, dtype=self.dtype)
        # 创建一个Electra中间层
        self.intermediate = FlaxElectraIntermediate(self.config, dtype=self.dtype)
        # 创建一个Electra输出层
        self.output = FlaxElectraOutput(self.config, dtype=self.dtype)
        # 如果需要添加跨注意力，创建一个Electra注意力层
        if self.config.add_cross_attention:
            self.crossattention = FlaxElectraAttention(self.config, causal=False, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        attention_mask,
        layer_head_mask,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        deterministic: bool = True,
        output_attentions: bool = False,
    ):
        # 自注意力机制
        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            layer_head_mask=layer_head_mask,
            init_cache=init_cache,
            deterministic=deterministic,
            output_attentions=output_attentions,
        )
        # 获取自注意力机制的输出
        attention_output = attention_outputs[0]

        # 交叉注意力块
        if encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask=encoder_attention_mask,
                layer_head_mask=layer_head_mask,
                key_value_states=encoder_hidden_states,
                deterministic=deterministic,
                output_attentions=output_attentions,
            )
            attention_output = cross_attention_outputs[0]

        # 使用中间层处理注意力输出
        hidden_states = self.intermediate(attention_output)
        # 使用输出层处理中间层输出和注意力输出
        hidden_states = self.output(hidden_states, attention_output, deterministic=deterministic)

        # 将结果放入元组中
        outputs = (hidden_states,)

        # 如果需要输出注意力信息
        if output_attentions:
            # 放入自注意力的注意力信息
            outputs += (attention_outputs[1],)
            # 如果有编码器隐藏状态，则放入交叉注意力的注意力信息
            if encoder_hidden_states is not None:
                outputs += (cross_attention_outputs[1],)
        # 返回结果元组
        return outputs
# 根据 transformers.models.bert.modeling_flax_bert.FlaxBertLayerCollection 复制代码，并将其中的 Bert 改为 Electra
class FlaxElectraLayerCollection(nn.Module):
    # Electra 配置
    config: ElectraConfig
    # 计算的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    # 是否使用梯度检查点
    gradient_checkpointing: bool = False

    def setup(self):
        # 如果使用梯度检查点
        if self.gradient_checkpointing:
            # 使用 remat 函数将 FlaxElectraLayer 转换为 FlaxElectraCheckpointLayer
            FlaxElectraCheckpointLayer = remat(FlaxElectraLayer, static_argnums=(5, 6, 7))
            # 创建 Electra 层集合，数量为配置中的隐藏层数量，每层使用梯度检查点
            self.layers = [
                FlaxElectraCheckpointLayer(self.config, name=str(i), dtype=self.dtype)
                for i in range(self.config.num_hidden_layers)
            ]
        else:
            # 创建 Electra 层集合，数量为配置中的隐藏层数量，每层不使用梯度检查点
            self.layers = [
                FlaxElectraLayer(self.config, name=str(i), dtype=self.dtype)
                for i in range(self.config.num_hidden_layers)
            ]

    def __call__(
        self,
        hidden_states,  # 输入的隐藏状态
        attention_mask,  # 注意力遮罩
        head_mask,  # 头部遮罩
        encoder_hidden_states: Optional[jnp.ndarray] = None,  # 编码器的隐藏状态，默认为空
        encoder_attention_mask: Optional[jnp.ndarray] = None,  # 编码器的注意力遮罩，默认为空
        init_cache: bool = False,  # 是否初始化缓存，默认为 False
        deterministic: bool = True,  # 是否确定性，默认为 True
        output_attentions: bool = False,  # 是否输出注意力，默认为 False
        output_hidden_states: bool = False,  # 是否输出隐藏状态，默认为 False
        return_dict: bool = True,  # 是否返回字典，默认为 True
        # 如果需要输出注意力权重，则初始化一个空元组，否则置为 None
        all_attentions = () if output_attentions else None
        # 如果需要输出隐藏状态，则初始化一个空元组，否则置为 None
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出交叉注意力权重且编码器隐藏状态不为空，则初始化一个空元组，否则置为 None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        # 如果需要使用头部遮罩，则检查头部遮罩的层数是否正确
        if head_mask is not None:
            # 如果头部遮罩的层数和当前模型的层数不匹配，则抛出 ValueError
            if head_mask.shape[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.shape[0]}."
                )

        # 遍历模型的每一层
        for i, layer in enumerate(self.layers):
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到全部隐藏状态中
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # 调用当前层的前向传播方法，获取当前层的输出
            layer_outputs = layer(
                hidden_states,
                attention_mask,
                head_mask[i] if head_mask is not None else None,
                encoder_hidden_states,
                encoder_attention_mask,
                init_cache,
                deterministic,
                output_attentions,
            )

            # 更新当前隐藏状态为当前层的输出
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力权重，则将当前层的注意力权重添加到全部注意力权重中
            if output_attentions:
                all_attentions += (layer_outputs[1],)

                # 如果编码器隐藏状态不为空且需要输出交叉注意力权重，则将当前层的交叉注意力权重添加到全部交叉注意力权重中
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # 如果需要输出隐藏状态，则将当前隐藏状态添加到全部隐藏状态中
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 将全部输出整合为 outputs 元组
        outputs = (hidden_states, all_hidden_states, all_attentions, all_cross_attentions)

        # 如果不需要返回字典形式的输出，则将 outputs 元组展平后返回
        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        # 否则，返回带过去和交叉注意力的 FlaxBaseModelOutputWithPastAndCrossAttentions 类的实例
        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )
# 从transformers.models.bert.modeling_flax_bert.FlaxBertEncoder复制代码，并将Bert->Electra
class FlaxElectraEncoder(nn.Module):
    # Electra的配置
    config: ElectraConfig
    # 计算的数据类型
    dtype: jnp.dtype = jnp.float32  
    # 梯度检查点
    gradient_checkpointing: bool = False

    # 初始化函数
    def setup(self):
        # 创建Electra层集合
        self.layer = FlaxElectraLayerCollection(
            self.config,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )

    # 调用函数
    def __call__(
        self,
        hidden_states,
        attention_mask,
        head_mask,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 调用Electra层集合
        return self.layer(
            hidden_states,
            attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            init_cache=init_cache,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class FlaxElectraGeneratorPredictions(nn.Module):
    # Electra的配置
    config: ElectraConfig
    # 计算的数据类型
    dtype: jnp.dtype = jnp.float32

    # 初始化函数
    def setup(self):
        # 创建LayerNorm层
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 创建全连接层
        self.dense = nn.Dense(self.config.embedding_size, dtype=self.dtype)

    # 调用函数
    def __call__(self, hidden_states):
        # 全连接层的计算
        hidden_states = self.dense(hidden_states)
        # 激活函数的应用
        hidden_states = ACT2FN[self.config.hidden_act](hidden_states)
        # LayerNorm的应用
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class FlaxElectraDiscriminatorPredictions(nn.Module):
    # 鉴别器的预测模块，由两个全连接层组成
    config: ElectraConfig
    # 计算的数据类型
    dtype: jnp.dtype = jnp.float32

    # 初始化函数
    def setup(self):
        # 创建全连接层
        self.dense = nn.Dense(self.config.hidden_size, dtype=self.dtype)
        # 创建预测全连接层
        self.dense_prediction = nn.Dense(1, dtype=self.dtype)

    # 调用函数
    def __call__(self, hidden_states):
        # 全连接层的计算
        hidden_states = self.dense(hidden_states)
        # 激活函数的应用
        hidden_states = ACT2FN[self.config.hidden_act](hidden_states)
        # 预测全连接层的计算并去掉最后一维
        hidden_states = self.dense_prediction(hidden_states).squeeze(-1)
        return hidden_states


class FlaxElectraPreTrainedModel(FlaxPreTrainedModel):
    """
    一个抽象类，处理权重初始化和一个简单的接口，用于下载和加载预训练模型。
    """

    # Electra的配置类
    config_class = ElectraConfig
    # 基础模型前缀
    base_model_prefix = "electra"
    # 模块类别，默认为空
    module_class: nn.Module = None
    # 初始化函数，用于初始化Electra模型
    def __init__(
        self,
        config: ElectraConfig,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        gradient_checkpointing: bool = False,
        **kwargs,
    ):
        # 根据给定的配置创建Electra模型的类对象
        module = self.module_class(config=config, dtype=dtype, gradient_checkpointing=gradient_checkpointing, **kwargs)
        # 调用父类的初始化函数，传入配置、模型类对象等参数
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    # 启用梯度检查点功能的函数，使得模型可以支持梯度检查点
    # 从transformers.models.bert.modeling_flax_bert.FlaxBertPreTrainedModel中复制而来
    def enable_gradient_checkpointing(self):
        # 根据当前模型的配置创建新的模型对象，启用梯度检查点功能
        self._module = self.module_class(
            config=self.config,
            dtype=self.dtype,
            gradient_checkpointing=True,
        )

    # 初始化权重的函数
    # 从transformers.models.bert.modeling_flax_bert.FlaxBertPreTrainedModel中复制而来
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量
        input_ids = jnp.zeros(input_shape, dtype="i4")
        token_type_ids = jnp.zeros_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        attention_mask = jnp.ones_like(input_ids)
        head_mask = jnp.ones((self.config.num_hidden_layers, self.config.num_attention_heads))

        # 分割随机数种子
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        if self.config.add_cross_attention:
            # 如果模型需要进行跨注意力机制的计算，则初始化额外的输入张量
            encoder_hidden_states = jnp.zeros(input_shape + (self.config.hidden_size,))
            encoder_attention_mask = attention_mask
            # 调用模型的init方法，初始化模型参数
            module_init_outputs = self.module.init(
                rngs,
                input_ids,
                attention_mask,
                token_type_ids,
                position_ids,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                return_dict=False,
            )
        else:
            # 否则，直接调用模型的init方法初始化模型参数
            module_init_outputs = self.module.init(
                rngs, input_ids, attention_mask, token_type_ids, position_ids, head_mask, return_dict=False
            )

        # 获取随机初始化的参数
        random_params = module_init_outputs["params"]

        if params is not None:
            # 如果传入了预训练模型的参数，则合并随机初始化的参数和预训练参数
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            # 否则，直接返回随机初始化的参数
            return random_params

    # 从transformers.models.bart.modeling_flax_bart.FlaxBartDecoderPreTrainedModel中复制而来
    # 初始化缓存的函数
    # 初始化缓存，用于快速自回归解码。batch_size 定义初始化缓存的批处理大小，max_length 定义初始化缓存的最大序列长度
    def init_cache(self, batch_size, max_length):
        """
        Args:
            batch_size (`int`):
                用于快速自回归解码的批处理大小。定义了初始化缓存的批处理大小
            max_length (`int`):
                自回归解码的最大可能长度。定义了初始化缓存的序列长度。
        """
        # 初始化用于检索缓存的输入变量
        input_ids = jnp.ones((batch_size, max_length), dtype="i4")
        attention_mask = jnp.ones_like(input_ids, dtype="i4")
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        # 初始化变量以检索缓存，并返回解冻（不受限制）的缓存
        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        return unfreeze(init_variables["cache"])

    # 调用模型的方法，添加模型前向传播的文档字符串
    @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        past_key_values: dict = None,
# 定义一个 FlaxElectraModule 类，继承自 nn.Module
class FlaxElectraModule(nn.Module):
    # ElectraConfig 类型的 config 属性
    config: ElectraConfig
    # jnp.float32 类型的 dtype 属性，作为计算的数据类型
    dtype: jnp.dtype = jnp.float32
    # 是否使用梯度检查点
    gradient_checkpointing: bool = False

    # 初始化方法
    def setup(self):
        # 创建 FlaxElectraEmbeddings 对象
        self.embeddings = FlaxElectraEmbeddings(self.config, dtype=self.dtype)
        # 如果嵌入大小不等于隐藏大小，则创建 nn.Dense 对象
        if self.config.embedding_size != self.config.hidden_size:
            self.embeddings_project = nn.Dense(self.config.hidden_size, dtype=self.dtype)
        # 创建 FlaxElectraEncoder 对象
        self.encoder = FlaxElectraEncoder(
            self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )

    # 调用方法
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask: Optional[np.ndarray] = None,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 调用 embeddings 计算嵌入
        embeddings = self.embeddings(
            input_ids, token_type_ids, position_ids, attention_mask, deterministic=deterministic
        )
        # 如果存在 embeddings_project，对嵌入进行处理
        if hasattr(self, "embeddings_project"):
            embeddings = self.embeddings_project(embeddings)

        # 返回 encoder 处理后的结果
        return self.encoder(
            embeddings,
            attention_mask,
            head_mask=head_mask,
            deterministic=deterministic,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


# 添加文档头信息到 FlaxElectraModel 类
@add_start_docstrings(
    "The bare Electra Model transformer outputting raw hidden-states without any specific head on top.",
    ELECTRA_START_DOCSTRING,
)
class FlaxElectraModel(FlaxElectraPreTrainedModel):
    # 指定 module_class 属性为 FlaxElectraModule 类
    module_class = FlaxElectraModule


# 添加调用示例文档字符串到 FlaxElectraModel 类
append_call_sample_docstring(FlaxElectraModel, _CHECKPOINT_FOR_DOC, FlaxBaseModelOutput, _CONFIG_FOR_DOC)


# 定义一个 FlaxElectraTiedDense 类，继承自 nn.Module
class FlaxElectraTiedDense(nn.Module):
    # 嵌入大小属性
    embedding_size: int
    # jnp.float32 类型的 dtype 属性
    dtype: jnp.dtype = jnp.float32
    # 精度属性
    precision = None
    # 偏置初始化函数
    bias_init: Callable[..., np.ndarray] = jax.nn.initializers.zeros

    # 初始化方法
    def setup(self):
        # 初始化偏置参数
        self.bias = self.param("bias", self.bias_init, (self.embedding_size,))

    # 调用方法
    def __call__(self, x, kernel):
        # 转换 x 和 kernel 为 jnp 数组
        x = jnp.asarray(x, self.dtype)
        kernel = jnp.asarray(kernel, self.dtype)
        # 执行 dot_general 运算
        y = lax.dot_general(
            x,
            kernel,
            (((x.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )
        # 转换偏置参数为 jnp 数组
        bias = jnp.asarray(self.bias, self.dtype)
        # 返回 y 加上偏置结果
        return y + bias


# 定义一个 FlaxElectraForMaskedLMModule 类，继承自 nn.Module
class FlaxElectraForMaskedLMModule(nn.Module):
    # ElectraConfig 类型的 config 属性
    config: ElectraConfig
    # jnp.float32 类型的 dtype 属性
    dtype: jnp.dtype = jnp.float32
    # 是否使用梯度检查点
    gradient_checkpointing: bool = False
    # 设置模型
    def setup(self):
        # 初始化 Electra 模块
        self.electra = FlaxElectraModule(
            config=self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )
        # 初始化 GeneratorPredictions
        self.generator_predictions = FlaxElectraGeneratorPredictions(config=self.config, dtype=self.dtype)
        if self.config.tie_word_embeddings:
            # 如果词嵌入需要打包，则使用 TiedDense 层
            self.generator_lm_head = FlaxElectraTiedDense(self.config.vocab_size, dtype=self.dtype)
        else:
            # 否则使用普通的 Dense 层
            self.generator_lm_head = nn.Dense(self.config.vocab_size, dtype=self.dtype)

    # 调用模块
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 调用 Electra 模块，获取输出
        outputs = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取隐藏层输出
        hidden_states = outputs[0]
        # 通过 GeneratorPredictions 获取预测分数
        prediction_scores = self.generator_predictions(hidden_states)

        if self.config.tie_word_embeddings:
            # 如果词嵌入需要打包，则共享 Embedding 层
            shared_embedding = self.electra.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
            prediction_scores = self.generator_lm_head(prediction_scores, shared_embedding.T)
        else:
            # 否则直接使用 Dense 层
            prediction_scores = self.generator_lm_head(prediction_scores)

        if not return_dict:
            # 如果不需要返回字典，则返回预测分数和其他输出
            return (prediction_scores,) + outputs[1:]

        # 返回 MaskedLMOutput 字典
        return FlaxMaskedLMOutput(
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 导入必要的模块和函数装饰器
@add_start_docstrings("""Electra Model with a `language modeling` head on top.""", ELECTRA_START_DOCSTRING)
class FlaxElectraForMaskedLM(FlaxElectraPreTrainedModel):
    module_class = FlaxElectraForMaskedLMModule


# 添加调用示例文档字符串
append_call_sample_docstring(FlaxElectraForMaskedLM, _CHECKPOINT_FOR_DOC, FlaxMaskedLMOutput, _CONFIG_FOR_DOC)


# 定义用于预训练的 Electra 模型模块
class FlaxElectraForPreTrainingModule(nn.Module):
    # 定义模型配置、数据类型和梯度检查点标志
    config: ElectraConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    # 模块设置方法
    def setup(self):
        # 初始化 Electra 模型和判别器预测头
        self.electra = FlaxElectraModule(
            config=self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )
        self.discriminator_predictions = FlaxElectraDiscriminatorPredictions(config=self.config, dtype=self.dtype)

    # 模块调用方法
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 使用 Electra 模型处理输入并获取输出
        outputs = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]  # 获取隐藏状态

        logits = self.discriminator_predictions(hidden_states)  # 获取判别器的预测结果

        # 如果不返回字典，则返回元组形式的输出
        if not return_dict:
            return (logits,) + outputs[1:]

        # 返回预训练模型的输出
        return FlaxElectraForPreTrainingOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# 添加预训练模型的文档字符串
@add_start_docstrings(
    """
    Electra model with a binary classification head on top as used during pretraining for identifying generated tokens.

    It is recommended to load the discriminator checkpoint into that model.
    """,
    ELECTRA_START_DOCSTRING,
)
class FlaxElectraForPreTraining(FlaxElectraPreTrainedModel):
    module_class = FlaxElectraForPreTrainingModule


# 预训练模型的文档字符串模板
FLAX_ELECTRA_FOR_PRETRAINING_DOCSTRING = """
    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, FlaxElectraForPreTraining

    >>> tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator")
    >>> model = FlaxElectraForPreTraining.from_pretrained("google/electra-small-discriminator")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="np")
    >>> outputs = model(**inputs)

    >>> prediction_logits = outputs.logits
    ```py
"""

# 重写预训练模型的调用文档字符串
overwrite_call_docstring(
    FlaxElectraForPreTraining,
    ELECTRA_INPUTS_DOCSTRING.format("batch_size, sequence_length") + FLAX_ELECTRA_FOR_PRETRAINING_DOCSTRING,
)
# 添加替换的返回文档字符串
append_replace_return_docstrings(
    FlaxElectraForPreTraining,  # 从模块中导入FlaxElectraForPreTraining类
    output_type=FlaxElectraForPreTrainingOutput,  # 设置output_type参数为FlaxElectraForPreTrainingOutput类
    config_class=_CONFIG_FOR_DOC  # 设置config_class为_CONFIG_FOR_DOC变量
# Flax 模块，用于为标记分类任务构建 Electra 模型
class FlaxElectraForTokenClassificationModule(nn.Module):
    # Electra 配置
    config: ElectraConfig
    # 数据类型，默认为 32 位浮点数
    dtype: jnp.dtype = jnp.float32
    # 是否使用梯度检查点，默认为 False
    gradient_checkpointing: bool = False

    # 初始化模块
    def setup(self):
        # 创建 Electra 模型
        self.electra = FlaxElectraModule(
            config=self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )
        # 分类器的丢弃率
        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )
        # 丢弃层
        self.dropout = nn.Dropout(classifier_dropout)
        # 分类器
        self.classifier = nn.Dense(self.config.num_labels, dtype=self.dtype)

    # 调用模块
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 模型前向传播
        outputs = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取隐藏状态
        hidden_states = outputs[0]

        # 应用丢弃层
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 分类
        logits = self.classifier(hidden_states)

        # 如果不返回字典，则返回元组
        if not return_dict:
            return (logits,) + outputs[1:]

        # 返回 Token 分类器的输出
        return FlaxTokenClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# 添加模型文档字符串
@add_start_docstrings(
    """
    带有标记分类头部的 Electra 模型。

    可以加载鉴别器和生成器到这个模型中。
    """,
    ELECTRA_START_DOCSTRING,
)
class FlaxElectraForTokenClassification(FlaxElectraPreTrainedModel):
    # 模块类别
    module_class = FlaxElectraForTokenClassificationModule


# 添加调用示例文档字符串
append_call_sample_docstring(
    FlaxElectraForTokenClassification,
    _CHECKPOINT_FOR_DOC,
    FlaxTokenClassifierOutput,
    _CONFIG_FOR_DOC,
)


# 定义身份函数
def identity(x, **kwargs):
    return x


# Flax 序列摘要模块，用于计算序列隐藏状态的单个向量摘要
class FlaxElectraSequenceSummary(nn.Module):
    r"""
    计算序列隐藏状态的单个向量摘要。
    # 定义一个参数为config的方法，用于设置模型使用的配置。模型配置类中的相关参数如下：
    # 
    # - **summary_use_proj** (`bool`) -- 在向量提取后是否添加投影。
    # - **summary_proj_to_labels** (`bool`) -- 如果为`True`，则投影输出到 `config.num_labels` 类别（否则输出到 `config.hidden_size`）。
    # - **summary_activation** (`Optional[str]`) -- 设置为`"tanh"`时，在输出中添加tanh激活函数，其他字符串或`None`则不添加激活函数。
    # - **summary_first_dropout** (`float`) -- 投影和激活之前的可选的丢弃概率。
    # - **summary_last_dropout** (`float`)-- 投影和激活之后的可选的丢弃概率。
    """
    # 定义一个ElectraConfig类型的属性变量config
    config: ElectraConfig
    # 定义一个jnp.dtype类型的属性变量dtype，默认值为jnp.float32

    def setup(self):
        # 将summary属性设置为身份函数
        self.summary = identity
        # 如果模型配置中有 "summary_use_proj" 属性并且设置为True
        if hasattr(self.config, "summary_use_proj") and self.config.summary_use_proj:
            # 如果模型配置中有 "summary_proj_to_labels" 属性并且设置为True，并且模型的标签数大于0
            if (
                hasattr(self.config, "summary_proj_to_labels")
                and self.config.summary_proj_to_labels
                and self.config.num_labels > 0
            ):
                # 设置num_classes为模型的标签数
                num_classes = self.config.num_labels
            else:
                # 否则设置num_classes为模型的隐藏单元数
                num_classes = self.config.hidden_size
                # 将summary属性设置为具有num_classes输出节点的全连接层
            self.summary = nn.Dense(num_classes, dtype=self.dtype)

        # 获取模型配置中的"summary_activation"属性值
        activation_string = getattr(self.config, "summary_activation", None)
        # 如果存在激活字符串，则将self.activation设置为相应的激活函数；否则设置为恒等函数
        self.activation = ACT2FN[activation_string] if activation_string else lambda x: x  # noqa F407

        # 将self.first_dropout设置为恒等函数
        self.first_dropout = identity
        # 如果模型配置中有 "summary_first_dropout" 属性并且大于0
        if hasattr(self.config, "summary_first_dropout") and self.config.summary_first_dropout > 0:
            # 将self.first_dropout设置为具有丢弃概率为"summary_first_dropout"的Dropout层
            self.first_dropout = nn.Dropout(self.config.summary_first_dropout)

        # 将self.last_dropout设置为恒等函数
        self.last_dropout = identity
        # 如果模型配置中有 "summary_last_dropout" 属性并且大于0
        if hasattr(self.config, "summary_last_dropout") and self.config.summary_last_dropout > 0:
            # 将self.last_dropout设置为具有丢弃概率为"summary_last_dropout"的Dropout层
            self.last_dropout = nn.Dropout(self.config.summary_last_dropout)
    def __call__(self, hidden_states, cls_index=None, deterministic: bool = True):
        """
        Compute a single vector summary of a sequence hidden states.

        Args:
            hidden_states (`jnp.ndarray` of shape `[batch_size, seq_len, hidden_size]`):
                The hidden states of the last layer.
            cls_index (`jnp.ndarray` of shape `[batch_size]` or `[batch_size, ...]` where ... are optional leading dimensions of `hidden_states`, *optional*):
                Used if `summary_type == "cls_index"` and takes the last token of the sequence as classification token.

        Returns:
            `jnp.ndarray`: The summary of the sequence hidden states.
        """
        # NOTE: this does "first" type summary always
        # 取隐藏状态的第一个token作为汇总的向量
        output = hidden_states[:, 0]
        # 对汇总的向量进行首次的dropout操作
        output = self.first_dropout(output, deterministic=deterministic)
        # 对汇总的向量进行汇总操作
        output = self.summary(output)
        # 对汇总后的向量进行激活函数处理
        output = self.activation(output)
        # 对最终的汇总向量进行最后一次的dropout操作
        output = self.last_dropout(output, deterministic=deterministic)
        # 返回最终的汇总向量
        return output
# 定义一个用于多选题的 ELECTRA 模型的模块类
class FlaxElectraForMultipleChoiceModule(nn.Module):
    # 保存 ELECTRA 模型的配置
    config: ElectraConfig
    # 默认数据类型为 32 位浮点数
    dtype: jnp.dtype = jnp.float32
    # 是否启用梯度检查点，默认为 False
    gradient_checkpointing: bool = False

    # 模块设置函数
    def setup(self):
        # 创建 ELECTRA 模型对象
        self.electra = FlaxElectraModule(
            config=self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )
        # 创建 ELECTRA 模型的序列摘要对象
        self.sequence_summary = FlaxElectraSequenceSummary(config=self.config, dtype=self.dtype)
        # 创建用于多选题分类的全连接层
        self.classifier = nn.Dense(1, dtype=self.dtype)

    # 模块调用函数
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 计算选项的数量
        num_choices = input_ids.shape[1]
        # 重塑输入以便适应 ELECTRA 模型的要求
        input_ids = input_ids.reshape(-1, input_ids.shape[-1]) if input_ids is not None else None
        attention_mask = attention_mask.reshape(-1, attention_mask.shape[-1]) if attention_mask is not None else None
        token_type_ids = token_type_ids.reshape(-1, token_type_ids.shape[-1]) if token_type_ids is not None else None
        position_ids = position_ids.reshape(-1, position_ids.shape[-1]) if position_ids is not None else None

        # 使用 ELECTRA 模型进行推理
        outputs = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取隐藏状态
        hidden_states = outputs[0]
        # 对隐藏状态进行序列摘要，得到汇总输出
        pooled_output = self.sequence_summary(hidden_states, deterministic=deterministic)
        # 将汇总输出传入分类器，得到预测的 logits
        logits = self.classifier(pooled_output)

        # 重塑 logits 以适应多选题的形式
        reshaped_logits = logits.reshape(-1, num_choices)

        # 如果不需要返回字典，则只返回 logits 和可能的额外输出
        if not return_dict:
            return (reshaped_logits,) + outputs[1:]

        # 返回经过多选题模型输出封装的结果字典
        return FlaxMultipleChoiceModelOutput(
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# 为 FlaxElectraForMultipleChoice 类添加文档字符串
@add_start_docstrings(
    """
    ELECTRA Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    ELECTRA_START_DOCSTRING,
)
class FlaxElectraForMultipleChoice(FlaxElectraPreTrainedModel):
    # 将模块类指定为 FlaxElectraForMultipleChoiceModule
    module_class = FlaxElectraForMultipleChoiceModule


# 为 FlaxElectraForMultipleChoice 类的调用函数添加示例文档字符串
overwrite_call_docstring(
    FlaxElectraForMultipleChoice, ELECTRA_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
)
append_call_sample_docstring(
    FlaxElectraForMultipleChoice,
    _CHECKPOINT_FOR_DOC,
    FlaxMultipleChoiceModelOutput,
    _CONFIG_FOR_DOC,
)


# 定义一个用于问答任务的 ELECTRA 模型的模块类
class FlaxElectraForQuestionAnsweringModule(nn.Module):
    # 保存 ELECTRA 模型的配置
    config: ElectraConfig
    # 默认数据类型为 32 位浮点数
    dtype: jnp.dtype = jnp.float32
    # 设置梯度检查点，默认为 False
    gradient_checkpointing: bool = False

    # 初始化方法
    def setup(self):
        # 创建一个 FlaxElectraModule 对象，传入配置、数据类型和梯度检查点参数
        self.electra = FlaxElectraModule(
            config=self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )
        # 创建一个具有特定数量标签的全连接层
        self.qa_outputs = nn.Dense(self.config.num_labels, dtype=self.dtype)

    # 调用方法
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 模型推理
        outputs = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取隐藏状态
        hidden_states = outputs[0]
        # 使用全连接层得到逻辑值
        logits = self.qa_outputs(hidden_states)
        # 将逻辑值按照标签数量划分成起始和结束逻辑值
        start_logits, end_logits = logits.split(self.config.num_labels, axis=-1)
        # 压缩结果维度
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # 如果不返回字典，则返回起始和结束逻辑值和输出的其他内容
        if not return_dict:
            return (start_logits, end_logits) + outputs[1:]

        # 返回 FlaxQuestionAnsweringModelOutput 对象
        return FlaxQuestionAnsweringModelOutput(
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用指定的文档字符串为 ELECTRA 模型添加起始文档字符串，用于针对提取式问答任务（如 SQuAD）的 span 分类头部
# 这里的模型在隐藏状态输出的基础上使用线性层来计算 `span start logits` 和 `span end logits`
@add_start_docstrings(
    """
    ELECTRA Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    ELECTRA_START_DOCSTRING,
)
class FlaxElectraForQuestionAnswering(FlaxElectraPreTrainedModel):
    module_class = FlaxElectraForQuestionAnsweringModule


# 附加调用示例的文档字符串，用于 FlaxElectraForQuestionAnswering 类
append_call_sample_docstring(
    FlaxElectraForQuestionAnswering,
    _CHECKPOINT_FOR_DOC,
    FlaxQuestionAnsweringModelOutput,
    _CONFIG_FOR_DOC,
)


class FlaxElectraClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    config: ElectraConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 初始化分类器的全连接层
        self.dense = nn.Dense(self.config.hidden_size, dtype=self.dtype)
        # 设置分类器的 dropout 操作，如果没有设置，则使用 hidden_dropout_prob
        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        # 初始化分类器的输出全连接层
        self.out_proj = nn.Dense(self.config.num_labels, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic: bool = True):
        # 取第一个特殊标记 <s>（等同于 [CLS]）对应的隐藏状态
        x = hidden_states[:, 0, :]
        # 对隐藏状态进行 dropout 操作
        x = self.dropout(x, deterministic=deterministic)
        # 通过全连接层进行线性变换
        x = self.dense(x)
        # 使用 GELU 激活函数，Electra 作者使用 GELU 而不是 BERT 中的 tanh
        x = ACT2FN["gelu"](x)
        # 再次进行 dropout 操作
        x = self.dropout(x, deterministic=deterministic)
        # 最后通过输出层进行分类
        x = self.out_proj(x)
        return x


class FlaxElectraForSequenceClassificationModule(nn.Module):
    config: ElectraConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        # 初始化 Electra 模型
        self.electra = FlaxElectraModule(
            config=self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )
        # 初始化序列分类任务的分类器
        self.classifier = FlaxElectraClassificationHead(config=self.config, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    # 定义模型的前向传播方法
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        deterministic=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # 使用 Electra 模型进行前向传播
        outputs = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取 Electra 模型输出的 hidden_states
        hidden_states = outputs[0]
        # 使用分类器对 hidden_states 进行分类预测
        logits = self.classifier(hidden_states, deterministic=deterministic)

        # 如果不返回 dict，则返回 logits 和 outputs 的其它部分
        if not return_dict:
            return (logits,) + outputs[1:]

        # 返回一个 FlaxSequenceClassifierOutput 对象，包含 logits, hidden_states 和 attentions
        return FlaxSequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 定义一个包含序列分类/回归头部的Electra模型转换器，顶部有一个线性层（放在池化输出之上），例如用于GLUE任务
@add_start_docstrings(
    """
    Electra Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    ELECTRA_START_DOCSTRING,
)
class FlaxElectraForSequenceClassification(FlaxElectraPreTrainedModel):
    module_class = FlaxElectraForSequenceClassificationModule

# 添加调用示例文档字符串
append_call_sample_docstring(
    FlaxElectraForSequenceClassification,
    _CHECKPOINT_FOR_DOC,
    FlaxSequenceClassifierOutput,
    _CONFIG_FOR_DOC,
)

# 定义用于CausalLM的FlaxElectra模块
class FlaxElectraForCausalLMModule(nn.Module):
    config: ElectraConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        # 创建Electra模块
        self.electra = FlaxElectraModule(
            config=self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )
        # 创建生成器预测模块
        self.generator_predictions = FlaxElectraGeneratorPredictions(config=self.config, dtype=self.dtype)
        # 如果配置了词嵌入共享，则创建电气LM头部
        if self.config.tie_word_embeddings:
            self.generator_lm_head = FlaxElectraTiedDense(self.config.vocab_size, dtype=self.dtype)
        else:
            self.generator_lm_head = nn.Dense(self.config.vocab_size, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask: Optional[jnp.ndarray] = None,
        token_type_ids: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        head_mask: Optional[jnp.ndarray] = None,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        ):
        # 使用 Electra 模型进行推理，传入输入标识符、注意力掩码、标记类型标识符、位置标识符、头部掩码，以及其他参数
        outputs = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            init_cache=init_cache,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取 Electra 模型输出的隐藏状态
        hidden_states = outputs[0]
        # 使用生成器来预测得分
        prediction_scores = self.generator_predictions(hidden_states)

        # 如果配置了词嵌入的共享属性，则使用共享的嵌入来进行预测
        if self.config.tie_word_embeddings:
            shared_embedding = self.electra.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
            prediction_scores = self.generator_lm_head(prediction_scores, shared_embedding.T)
        # 如果没有配置词嵌入的共享属性，则直接使用生成器进行预测
        else:
            prediction_scores = self.generator_lm_head(prediction_scores)

        # 如果不需要返回字典类型，则返回预测得分和其他输出
        if not return_dict:
            return (prediction_scores,) + outputs[1:]

        # 如果需要返回字典类型，则返回带有交叉注意力的带有因果语言模型输出的结果
        return FlaxCausalLMOutputWithCrossAttentions(
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )
# 添加描述文档到 Electra 模型，该模型在隐藏状态输出之上包括一个语言建模头部（一个线性层），用于自回归任务
# 这段代码从 transformers.models.bert.modeling_flax_bert.FlaxBertForCausalLM 复制并将 Bert 替换为 Electra
class FlaxElectraForCausalLM(FlaxElectraPreTrainedModel):
    module_class = FlaxElectraForCausalLMModule

    # 为生成准备输入，初始化缓存
    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = None):
        batch_size, seq_length = input_ids.shape
        past_key_values = self.init_cache(batch_size, max_length)
        # 注意，通常需要在 attention_mask 中填充 0，以处理 x > input_ids.shape[-1] 和 x < cache_length 的情况
        # 但由于解码器使用因果遮蔽，这些位置已经被遮蔽了
        # 因此，我们可以在这里创建一个静态的 attention_mask，对于编译来说更有效
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    # 更新生成的输入，将模型输出的过去键值和位置标识更新到模型关键字参数中
    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs

# 在 FlaxElectraForCausalLM 上附加调用示例文档字符串
append_call_sample_docstring(
    FlaxElectraForCausalLM,
    _CHECKPOINT_FOR_DOC,
    FlaxCausalLMOutputWithCrossAttentions,
    _CONFIG_FOR_DOC,
)
```