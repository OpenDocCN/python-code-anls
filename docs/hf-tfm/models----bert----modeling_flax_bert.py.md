# `.\transformers\models\bert\modeling_flax_bert.py`

```
# 设置文件编码为 UTF-8
# 版权声明，版权归 The Google Flax Team Authors 和 The HuggingFace Inc. team 所有
# 根据 Apache 许可证 2.0 版本授权，除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”基础分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制

# 导入所需的模块和类
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

# 导入模型输出相关的类
from ...modeling_flax_outputs import (
    FlaxBaseModelOutputWithPastAndCrossAttentions,
    FlaxBaseModelOutputWithPooling,
    FlaxBaseModelOutputWithPoolingAndCrossAttentions,
    FlaxCausalLMOutputWithCrossAttentions,
    FlaxMaskedLMOutput,
    FlaxMultipleChoiceModelOutput,
    FlaxNextSentencePredictorOutput,
    FlaxQuestionAnsweringModelOutput,
    FlaxSequenceClassifierOutput,
    FlaxTokenClassifierOutput,
)
# 导入模型相关的工具类和函数
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
_CHECKPOINT_FOR_DOC = "bert-base-uncased"
_CONFIG_FOR_DOC = "BertConfig"

# 定义 remat 函数
remat = nn_partitioning.remat

# 定义 FlaxBertForPreTrainingOutput 类，继承自 ModelOutput 类
@flax.struct.dataclass
class FlaxBertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`BertForPreTraining`].
    # 定义函数的参数，语言模型头的预测得分（SoftMax之前的每个词汇标记的分数）
    prediction_logits: jnp.ndarray = None
    # 定义函数的参数，下一个序列预测（分类）头的预测得分（SoftMax之前的True/False延续的分数）
    seq_relationship_logits: jnp.ndarray = None
    # 定义函数的参数，当传递output_hidden_states=True或config.output_hidden_states=True时返回，或者config.output_hidden_states=True时返回
    # 元组，包含jnp.ndarray（一个用于嵌入的输出 + 一个用于每层输出的输出）的形状为（batch_size，sequence_length，hidden_size）。
    # 模型在每一层的输出加上初始嵌入的输出。
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    # 定义函数的参数，当传递output_attentions=True或config.output_attentions=True时返回，或者config.output_attentions=True时返回
    # 元组，包含jnp.ndarray（每层一个）的形状为（batch_size，num_heads，sequence_length，sequence_length）。
    # 在注意力softmax之后的注意力权重，用于在自注意力头中计算加权平均值。
    attentions: Optional[Tuple[jnp.ndarray]] = None
# BERT 模型的文档字符串，包含了模型的继承关系、Flax 模块的使用说明以及支持的 JAX 特性
BERT_START_DOCSTRING = r"""

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
        config ([`BertConfig`]): Model configuration class with all the parameters of the model.
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

# BERT 模型输入的文档字符串
BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`numpy.ndarray` of shape `({0})`):
            # 输入序列标记在词汇表中的索引。
            # 可以使用 [`AutoTokenizer`] 获取索引。参见 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`] 获取详情。
            # [什么是输入 ID?](../glossary#input-ids)
        attention_mask (`numpy.ndarray` of shape `({0})`, *optional*):
            # 遮蔽填充标记索引，避免在这些标记上执行注意力操作。遮蔽值选在 `[0, 1]` 之间：
            # - 1 表示**未遮蔽**的标记，
            # - 0 表示**遮蔽**的标记。
            # [什么是注意力遮蔽?](../glossary#attention-mask)
        token_type_ids (`numpy.ndarray` of shape `({0})`, *optional*):
            # 段标记索引，指示输入的第一部分和第二部分。索引选在 `[0, 1]` 之间：
            # - 0 对应*句子 A* 标记，
            # - 1 对应*句子 B* 标记。
            # [什么是标记类型 ID?](../glossary#token-type-ids)
        position_ids (`numpy.ndarray` of shape `({0})`, *optional*):
            # 每个输入序列标记在位置嵌入中的位置索引。选在范围 `[0, config.max_position_embeddings - 1]`。
        head_mask (`numpy.ndarray` of shape `({0})`, `optional):
            # 用于将注意力模块中的特定头部置零的遮蔽。遮蔽值选在 `[0, 1]` 之间：
            # - 1 表示**未遮蔽**的头部，
            # - 0 表示**遮蔽**的头部。
        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""
定义了一个名为 FlaxBertEmbeddings 的类，用于构建 BERT 模型的嵌入层。

构造函数：初始化嵌入层所需的参数和模型配置。

setup 方法：初始化各个嵌入层，包括词嵌入、位置嵌入和标记类型嵌入，并设置 LayerNorm 和 Dropout。

__call__ 方法：接收输入的 input_ids、token_type_ids、position_ids 和 attention_mask，并对它们进行嵌入处理。
首先将 input_ids、position_ids 和 token_type_ids 分别传入对应的嵌入层中获取嵌入向量，然后将这些嵌入向量相加得到 hidden_states。
接着对 hidden_states 进行 LayerNorm 处理和 Dropout 处理，最后返回处理后的 hidden_states。

"""
class FlaxBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    config: BertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        # 初始化词嵌入层
        self.word_embeddings = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        # 初始化位置嵌入层
        self.position_embeddings = nn.Embed(
            self.config.max_position_embeddings,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        # 初始化标记类型嵌入层
        self.token_type_embeddings = nn.Embed(
            self.config.type_vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        # 初始化 LayerNorm
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 初始化 Dropout
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, input_ids, token_type_ids, position_ids, attention_mask, deterministic: bool = True):
        # Embed
        # 获取词嵌入向量
        inputs_embeds = self.word_embeddings(input_ids.astype("i4"))
        # 获取位置嵌入向量
        position_embeds = self.position_embeddings(position_ids.astype("i4"))
        # 获取标记类型嵌入向量
        token_type_embeddings = self.token_type_embeddings(token_type_ids.astype("i4"))

        # Sum all embeddings
        # 将词嵌入向量、位置嵌入向量和标记类型嵌入向量相加得到隐藏状态
        hidden_states = inputs_embeds + token_type_embeddings + position_embeds

        # Layer Norm
        # 对隐藏状态进行 LayerNorm 处理
        hidden_states = self.LayerNorm(hidden_states)
        # 对处理后的隐藏状态进行 Dropout 处理
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 返回处理后的隐藏状态
        return hidden_states


class FlaxBertSelfAttention(nn.Module):
    config: BertConfig
    causal: bool = False
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
"""
    # 在模型初始化设置阶段，计算每个注意力头的维度
    def setup(self):
        # 计算每个注意力头的维度
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        # 如果隐藏层大小不能被注意力头的数量整除，引发数值错误
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError(
                "`config.hidden_size`: {self.config.hidden_size} has to be a multiple of `config.num_attention_heads` "
                "                   : {self.config.num_attention_heads}"
            )

        # 初始化查询权重层
        self.query = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        # 初始化键权重层
        self.key = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        # 初始化值权重层
        self.value = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

        # 如果是因果（causal）注意力机制
        if self.causal:
            # 创建一个因果掩码
            self.causal_mask = make_causal_mask(
                jnp.ones((1, self.config.max_position_embeddings), dtype="bool"), dtype="bool"
            )

    # 将隐藏状态拆分成多个注意力头
    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.config.num_attention_heads, self.head_dim))

    # 合并多个注意力头为一个隐藏状态
    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.config.hidden_size,))

    @nn.compact
    # 从transformers.models.bart.modeling_flax_bart.FlaxBartAttention._concatenate_to_cache中复制过来的
```  
    # 定义一个方法，用于将来自单个输入标记的投影键、值状态与先前步骤中缓存的状态连接起来
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slightly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # 检测是否通过存在的缓存数据进行初始化
        is_initialized = self.has_variable("cache", "cached_key")
        # 获取缓存的键值，如果没有则初始化为零张量
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        # 获取缓存的索引，如果没有则初始化为0
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            # 获取缓存的键的形状
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # 使用新的一维空间片段更新键、值缓存
            cur_index = cache_index.value
            # 计算更新的索引位置
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            # 使用新的键更新缓存的键
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            # 使用新的值更新缓存的值
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            # 更新缓存索引，将已更新的缓存向量数量加到缓存索引上
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # 用于缓存的解码器自注意力的因果掩码：我们的单个查询位置只应该参与到已生成并缓存的键位置，而不是剩余的零元素
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            # 将填充掩码与输入的注意力掩码组合
            attention_mask = combine_masks(pad_mask, attention_mask)
        # 返回更新后的键、值以及注意力掩码
        return key, value, attention_mask

    # 定义__call__方法，用于调用这个类
    def __call__(
        self,
        hidden_states,
        attention_mask,
        layer_head_mask,
        key_value_states: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        deterministic=True,
        output_attentions: bool = False,
# 定义 FlaxBertSelfOutput 类，继承自 nn.Module
class FlaxBertSelfOutput(nn.Module):
    # BertConfig 类型的 config 属性
    config: BertConfig
    # jnp.float32 类型的 dtype 属性，用于计算的数据类型
    dtype: jnp.dtype = jnp.float32  

    # 初始化方法
    def setup(self):
        # 创建 nn.Dense 层，隐藏层大小为 config.hidden_size
        # 使用正态分布初始化权重，初始化范围为 config.initializer_range
        # 指定数据类型为 dtype
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 创建 nn.LayerNorm 层，epsilon 为 config.layer_norm_eps
        # 指定数据类型为 dtype
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 创建 nn.Dropout 层，丢弃率为 config.hidden_dropout_prob
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    # 调用方法
    def __call__(self, hidden_states, input_tensor, deterministic: bool = True):
        # 通过全连接层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # 对处理后的隐藏状态进行丢弃操作
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 对丢弃后的隐藏状态进行 LayerNorm 操作，并与输入张量相加
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的隐藏状态
        return hidden_states


# 定义 FlaxBertAttention 类，继承自 nn.Module
class FlaxBertAttention(nn.Module):
    # BertConfig 类型的 config 属性
    config: BertConfig
    # 是否为因果关系
    causal: bool = False
    # jnp.float32 类型的 dtype 属性，用于计算的数据类型
    dtype: jnp.dtype = jnp.float32

    # 初始化方法
    def setup(self):
        # 创建 FlaxBertSelfAttention 和 FlaxBertSelfOutput 实例
        self.self = FlaxBertSelfAttention(self.config, causal=self.causal, dtype=self.dtype)
        self.output = FlaxBertSelfOutput(self.config, dtype=self.dtype)

    # 调用方法
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
        # 注意力掩码的形状为 (*batch_sizes, kv_length)
        # FLAX 需要的形状为 (*batch_sizes, 1, 1, kv_length)，以便与 attn_weights 的形状广播
        attn_outputs = self.self(
            hidden_states,
            attention_mask,
            layer_head_mask=layer_head_mask,
            key_value_states=key_value_states,
            init_cache=init_cache,
            deterministic=deterministic,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]
        # 对注意力输出进行处理
        hidden_states = self.output(attn_output, hidden_states, deterministic=deterministic)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_outputs[1],)

        return outputs


# 定义 FlaxBertIntermediate 类，继承自 nn.Module
class FlaxBertIntermediate(nn.Module):
    # BertConfig 类型的 config 属性
    config: BertConfig
    # jnp.float32 类型的 dtype 属性，用于计算的数据类型
    dtype: jnp.dtype = jnp.float32  

    # 初始化方法
    def setup(self):
        # 创建 nn.Dense 层，隐藏层大小为 config.intermediate_size
        # 使用正态分布初始化权重，初始化范围为 config.initializer_range
        # 指定数据类型为 dtype
        self.dense = nn.Dense(
            self.config.intermediate_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 激活函数为 ACT2FN 中对应的函数
        self.activation = ACT2FN[self.config.hidden_act]

    # 调用方法
    def __call__(self, hidden_states):
        # 通过全连接层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # 使用激活函数处理隐藏状态
        hidden_states = self.activation(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states


# 定义 FlaxBertOutput 类，继承自 nn.Module
class FlaxBertOutput(nn.Module):
    # BertConfig 类型的 config 属性
    config: BertConfig
    # jnp.float32 类型的 dtype 属性，用于计算的数据类型
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    # 初始化神经网络的全连接层，设置隐藏层大小、初始化方式、数据类型
    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 初始化Dropout层，设置丢弃率
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        # 初始化LayerNorm层，设置epsilon值和数据类型
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    # 定义神经网络的前向传播过程
    def __call__(self, hidden_states, attention_output, deterministic: bool = True):
        # 全连接层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # Dropout层处理隐藏状态，根据deterministic参数确定是否使用确定性模式
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # LayerNorm层处理隐藏状态和注意力输出的加和结果
        hidden_states = self.LayerNorm(hidden_states + attention_output)
        # 返回处理后的隐藏状态
        return hidden_states
# 定义一个 FlaxBertLayer 类，继承自 nn.Module
class FlaxBertLayer(nn.Module):
    # 类属性 config，类型为 BertConfig
    config: BertConfig
    # 类属性 dtype，指定计算中的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    # 模块的初始化方法
    def setup(self):
        # 初始化 self.attention 属性为 FlaxBertAttention 实例
        self.attention = FlaxBertAttention(self.config, causal=self.config.is_decoder, dtype=self.dtype)
        # 初始化 self.intermediate 属性为 FlaxBertIntermediate 实例
        self.intermediate = FlaxBertIntermediate(self.config, dtype=self.dtype)
        # 初始化 self.output 属性为 FlaxBertOutput 实例
        self.output = FlaxBertOutput(self.config, dtype=self.dtype)
        # 如果配置中包含交叉注意力，则初始化 self.crossattention 属性为 FlaxBertAttention 实例
        if self.config.add_cross_attention:
            self.crossattention = FlaxBertAttention(self.config, causal=False, dtype=self.dtype)

    # 实现类的调用方法
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
        # 进行自注意力计算
        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            layer_head_mask=layer_head_mask,
            init_cache=init_cache,
            deterministic=deterministic,
            output_attentions=output_attentions,
        )
        # 获取自注意力输出
        attention_output = attention_outputs[0]

        # 如果存在编码器隐藏状态，则进行交叉注意力计算
        if encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask=encoder_attention_mask,
                layer_head_mask=layer_head_mask,
                key_value_states=encoder_hidden_states,
                deterministic=deterministic,
                output_attentions=output_attentions,
            )
            # 获取交叉注意力输出
            attention_output = cross_attention_outputs[0]

        # 经过中间层处理
        hidden_states = self.intermediate(attention_output)
        # 经过输出层处理
        hidden_states = self.output(hidden_states, attention_output, deterministic=deterministic)

        # 将隐藏状态作为输出
        outputs = (hidden_states,)

        # 如果需要输出注意力信息
        if output_attentions:
            outputs += (attention_outputs[1],)
            # 如果存在编码器隐藏状态，同时输出交叉注意力信息
            if encoder_hidden_states is not None:
                outputs += (cross_attention_outputs[1],)
        return outputs


# 定义一个 FlaxBertLayerCollection 类，继承自 nn.Module
class FlaxBertLayerCollection(nn.Module):
    # 类属性 config，类型为 BertConfig
    config: BertConfig
    # 类属性 dtype，指定计算中的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    # 类属性 gradient_checkpointing，用于指定是否使用梯度检查点，默认为 False
    gradient_checkpointing: bool = False

    # 模块的初始化方法
    def setup(self):
        # 如果使用梯度检查点，则将 FlaxBertLayer 包装成 FlaxBertCheckpointLayer
        if self.gradient_checkpointing:
            FlaxBertCheckpointLayer = remat(FlaxBertLayer, static_argnums=(5, 6, 7))
            # 初始化 self.layers 属性为 FlaxBertCheckpointLayer 的实例列表
            self.layers = [
                FlaxBertCheckpointLayer(self.config, name=str(i), dtype=self.dtype)
                for i in range(self.config.num_hidden_layers)
            ]
        else:
            # 否则，初始化 self.layers 属性为 FlaxBertLayer 的实例列表
            self.layers = [
                FlaxBertLayer(self.config, name=str(i), dtype=self.dtype) for i in range(self.config.num_hidden_layers)
            ]
    def __call__(
        self,
        hidden_states,  # 输入的隐藏状态
        attention_mask,  # 注意力掩码，用于控制模型关注哪些位置的信息
        head_mask,  # 头部掩码，用于控制每个注意力头的权重
        encoder_hidden_states: Optional[jnp.ndarray] = None,  # 编码器的隐藏状态，默认为None
        encoder_attention_mask: Optional[jnp.ndarray] = None,  # 编码器的注意力掩码，默认为None
        init_cache: bool = False,  # 是否初始化缓存，默认为False
        deterministic: bool = True,  # 是否使用确定性计算，默认为True
        output_attentions: bool = False,  # 是否输出注意力权重，默认为False
        output_hidden_states: bool = False,  # 是否输出所有隐藏状态，默认为False
        return_dict: bool = True,  # 是否返回字典格式的输出，默认为True
    ):
        # 如果输出注意力权重，初始化空的元组用于存储
        all_attentions = () if output_attentions else None
        # 如果输出所有隐藏状态，初始化空的元组用于存储
        all_hidden_states = () if output_hidden_states else None
        # 如果输出注意力权重和编码器隐藏状态不为空，初始化空的元组用于存储
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        # 检查头部掩码的层数是否正确
        if head_mask is not None:
            if head_mask.shape[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for                  "
                    f"       {head_mask.shape[0]}."
                )

        # 遍历每个层
        for i, layer in enumerate(self.layers):
            # 如果输出所有隐藏状态，将当前隐藏状态加入元组
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # 调用当前层的__call__方法，得到该层的输出
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

            # 更新隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]

            # 如果输出注意力权重，将当前层的注意力权重加入元组
            if output_attentions:
                all_attentions += (layer_outputs[1],)

                # 如果编码器隐藏状态不为空，将当前层的编码器-解码器注意力权重加入元组
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # 如果输出所有隐藏状态，将最终隐藏状态加入元组
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 构造输出元组
        outputs = (hidden_states, all_hidden_states, all_attentions, all_cross_attentions)

        # 如果不返回字典格式的输出，将元组展平
        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        # 返回字典格式的输出
        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )
# 定义一个用于BERT编码器的Flax模型类
class FlaxBertEncoder(nn.Module):
    # BertConfig类型的配置参数
    config: BertConfig
    # 计算的数据类型，默认为32位浮点数
    dtype: jnp.dtype = jnp.float32
    # 是否使用梯度检查点
    gradient_checkpointing: bool = False

    # 设置方法，初始化BERT编码器的层集合
    def setup(self):
        # 创建FlaxBertLayerCollection对象，传入配置参数、数据类型和梯度检查点设置
        self.layer = FlaxBertLayerCollection(
            self.config,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )

    # 调用方法，对输入进行编码
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
        # 调用FlaxBertLayerCollection对象对输入进行编码
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


# 定义一个用于BERT池化的Flax模型类
class FlaxBertPooler(nn.Module):
    # BertConfig类型的配置参数
    config: BertConfig
    # 计算的数据类型，默认为32位浮点数
    dtype: jnp.dtype = jnp.float32

    # 设置方法，初始化BERT池化层
    def setup(self):
        # 创建Dense层对象，用于池化
        self.dense = nn.Dense(
            self.config.hidden_size,
            # 使用正态分布初始化权重
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )

    # 调用方法，对隐藏状态进行池化
    def __call__(self, hidden_states):
        # 取出CLS标记对应的隐藏状态
        cls_hidden_state = hidden_states[:, 0]
        # 将CLS标记的隐藏状态传入全连接层进行池化
        cls_hidden_state = self.dense(cls_hidden_state)
        # 对池化后的结果应用tanh激活函数
        return nn.tanh(cls_hidden_state)


# 定义一个用于BERT预测头变换的Flax模型类
class FlaxBertPredictionHeadTransform(nn.Module):
    # BertConfig类型的配置参数
    config: BertConfig
    # 计算的数据类型，默认为32位浮点数
    dtype: jnp.dtype = jnp.float32

    # 设置方法，初始化预测头变换层
    def setup(self):
        # 创建全连接层对象，用于变换隐藏状态
        self.dense = nn.Dense(self.config.hidden_size, dtype=self.dtype)
        # 激活函数，根据配置参数选择对应的激活函数
        self.activation = ACT2FN[self.config.hidden_act]
        # LayerNorm归一化层
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    # 调用方法，对隐藏状态进行预测头变换
    def __call__(self, hidden_states):
        # 将隐藏状态传入全连接层进行变换
        hidden_states = self.dense(hidden_states)
        # 对变换后的隐藏状态应用激活函数
        hidden_states = self.activation(hidden_states)
        # 对激活后的隐藏状态应用LayerNorm归一化
        return self.LayerNorm(hidden_states)


# 定义一个用于BERT语言模型预测头的Flax模型类
class FlaxBertLMPredictionHead(nn.Module):
    # BertConfig类型的配置参数
    config: BertConfig
    # 计算的数据类型，默认为32位浮点数
    dtype: jnp.dtype = jnp.float32
    # 偏置初始化函数，默认为全零初始化
    bias_init: Callable[..., np.ndarray] = jax.nn.initializers.zeros

    # 设置方法，初始化BERT语言模型预测头
    def setup(self):
        # 创建预测头变换层对象
        self.transform = FlaxBertPredictionHeadTransform(self.config, dtype=self.dtype)
        # 全连接层，用于预测词汇表中的词
        self.decoder = nn.Dense(self.config.vocab_size, dtype=self.dtype, use_bias=False)
        # 偏置参数
        self.bias = self.param("bias", self.bias_init, (self.config.vocab_size,))
    # 定义 __call__ 方法，用于模型的调用
    def __call__(self, hidden_states, shared_embedding=None):
        # 对隐藏状态进行变换
        hidden_states = self.transform(hidden_states)

        # 如果提供了共享的嵌入矩阵
        if shared_embedding is not None:
            # 对隐藏状态应用共享嵌入矩阵的转置作为核心
            hidden_states = self.decoder.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
        else:
            # 否则，对隐藏状态应用解码器
            hidden_states = self.decoder(hidden_states)

        # 将偏置转换为 JAX 数组，并添加到隐藏状态中
        bias = jnp.asarray(self.bias, self.dtype)
        hidden_states += bias
        # 返回处理后的隐藏状态
        return hidden_states
class FlaxBertOnlyMLMHead(nn.Module):
    # 定义一个类 FlaxBertOnlyMLMHead，继承自 nn.Module
    config: BertConfig
    # 定义一个属性 config，类型为 BertConfig
    dtype: jnp.dtype = jnp.float32
    # 定义一个属性 dtype，默认值为 jnp.float32

    def setup(self):
        # 定义一个方法 setup，用于设置模型
        self.predictions = FlaxBertLMPredictionHead(self.config, dtype=self.dtype)
        # 初始化属性 predictions 为 FlaxBertLMPredictionHead 类的实例，传入参数为 config 和 dtype

    def __call__(self, hidden_states, shared_embedding=None):
        # 定义一个方法 __call__，用于调用模型
        hidden_states = self.predictions(hidden_states, shared_embedding=shared_embedding)
        # 调用 predictions 方法，传入参数 hidden_states 和 shared_embedding
        return hidden_states
        # 返回 hidden_states


class FlaxBertOnlyNSPHead(nn.Module):
    # 定义一个类 FlaxBertOnlyNSPHead，继承自 nn.Module
    dtype: jnp.dtype = jnp.float32
    # 定义一个属性 dtype，默认值为 jnp.float32

    def setup(self):
        # 定义一个方法 setup，用于设置模型
        self.seq_relationship = nn.Dense(2, dtype=self.dtype)
        # 初始化属性 seq_relationship 为 nn.Dense 类的实例，传入参数为 2 和 dtype

    def __call__(self, pooled_output):
        # 定义一个方法 __call__，用于调用模型
        return self.seq_relationship(pooled_output)
        # 返回 seq_relationship 方法，传入参数 pooled_output


class FlaxBertPreTrainingHeads(nn.Module):
    # 定义一个类 FlaxBertPreTrainingHeads，继承自 nn.Module
    config: BertConfig
    # 定义一个属性 config，类型为 BertConfig
    dtype: jnp.dtype = jnp.float32
    # 定义一个属性 dtype，默认值为 jnp.float32

    def setup(self):
        # 定义一个方法 setup，用于设置模型
        self.predictions = FlaxBertLMPredictionHead(self.config, dtype=self.dtype)
        # 初始化属性 predictions 为 FlaxBertLMPredictionHead 类的实例，传入参数为 config 和 dtype
        self.seq_relationship = nn.Dense(2, dtype=self.dtype)
        # 初始化属性 seq_relationship 为 nn.Dense 类的实例，传入参数为 2 和 dtype

    def __call__(self, hidden_states, pooled_output, shared_embedding=None):
        # 定义一个方法 __call__，用于调用模型
        prediction_scores = self.predictions(hidden_states, shared_embedding=shared_embedding)
        # 调用 predictions 方法，传入参数 hidden_states 和 shared_embedding，赋值给 prediction_scores
        seq_relationship_score = self.seq_relationship(pooled_output)
        # 调用 seq_relationship 方法，传入参数 pooled_output，赋值给 seq_relationship_score
        return prediction_scores, seq_relationship_score
        # 返回 prediction_scores 和 seq_relationship_score


class FlaxBertPreTrainedModel(FlaxPreTrainedModel):
    # 定义一个类 FlaxBertPreTrainedModel，继承自 FlaxPreTrainedModel
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    # 一个用于处理权重初始化和下载预训练模型的抽象类

    config_class = BertConfig
    # 定义一个属性 config_class，值为 BertConfig
    base_model_prefix = "bert"
    # 定义一个属性 base_model_prefix，值为 "bert"
    module_class: nn.Module = None
    # 定义一个属性 module_class，类型为 nn.Module，默认值为 None

    def __init__(
        self,
        config: BertConfig,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        gradient_checkpointing: bool = False,
        **kwargs,
    ):
        # 定义一个初始化方法
        module = self.module_class(
            config=config,
            dtype=dtype,
            gradient_checkpointing=gradient_checkpointing,
            **kwargs,
        )
        # 初始化 module 为 module_class 类的实例，传入参数 config、dtype、gradient_checkpointing 和 kwargs
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)
        # 调用父类的初始化方法，传入参数 config、module、input_shape、seed、dtype、_do_init

    def enable_gradient_checkpointing(self):
        # 定义一个方法 enable_gradient_checkpointing，用于启用梯度检查点
        self._module = self.module_class(
            config=self.config,
            dtype=self.dtype,
            gradient_checkpointing=True,
        )
        # 将 _module 初始化为 module_class 类的实例，传入参数 config、dtype 和 gradient_checkpointing 为 True
    # 初始化模型的权重，接受随机数种子 rng、输入形状 input_shape 和参数字典 params，默认为 None
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量
        input_ids = jnp.zeros(input_shape, dtype="i4")
        # 与 input_ids 相同形状的 token_type_ids 张量
        token_type_ids = jnp.zeros_like(input_ids)
        # 根据输入张量形状广播创建位置编码张量
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        # 初始化注意力掩码张量，形状与输入张量相同
        attention_mask = jnp.ones_like(input_ids)
        # 初始化头掩码张量，形状为 (self.config.num_hidden_layers, self.config.num_attention_heads)
        head_mask = jnp.ones((self.config.num_hidden_layers, self.config.num_attention_heads))

        # 拆分随机数种子 rng 为参数随机数种子和 dropout 随机数种子
        params_rng, dropout_rng = jax.random.split(rng)
        # 创建包含两个随机数种子的字典 rngs
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 如果模型配置中包含交叉注意力，初始化编码器隐藏状态和编码器注意力掩码
        if self.config.add_cross_attention:
            # 初始化编码器隐藏状态为形状为 input_shape + (self.config.hidden_size,) 的零张量
            encoder_hidden_states = jnp.zeros(input_shape + (self.config.hidden_size,))
            # 编码器注意力掩码与注意力掩码相同
            encoder_attention_mask = attention_mask
            # 使用初始化参数初始化模块，并返回初始化结果
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
            # 使用初始化参数初始化模块，并返回初始化结果
            module_init_outputs = self.module.init(
                rngs, input_ids, attention_mask, token_type_ids, position_ids, head_mask, return_dict=False
            )

        # 获取随机初始化的参数
        random_params = module_init_outputs["params"]

        # 如果传入了预定义参数，则替换模型中对应的随机参数
        if params is not None:
            # 将随机参数和预定义参数展开成一维字典
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            # 将预定义参数缺失的键对应的随机参数填入预定义参数中
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            # 清空缺失键集合
            self._missing_keys = set()
            # 冻结参数字典并返回
            return freeze(unflatten_dict(params))
        else:
            # 返回随机初始化的参数字典
            return random_params

    # 从 transformers.models.bart.modeling_flax_bart.FlaxBartDecoderPreTrainedModel.init_cache 复制而来
    # 初始化缓存，用于自回归解码
    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                用于快速自回归解码的批大小。定义了初始化缓存的批大小。
            max_length (`int`):
                自回归解码的最大可能长度。定义了初始化缓存的序列长度。
        """
        # 初始化用于检索缓存的输入变量
        input_ids = jnp.ones((batch_size, max_length), dtype="i4")
        # 初始化注意力掩码张量，形状与 input_ids 相同
        attention_mask = jnp.ones_like(input_ids, dtype="i4")
        # 广播创建位置编码张量
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        # 使用初始化参数初始化模块，并返回初始化结果
        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        # 解冻并返回初始化的缓存
        return unfreeze(init_variables["cache"])

    # 将文档字符串添加到模型前向传播函数
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 定义一个方法，用于模型调用，接收输入的各种参数
    def __call__(
        self,
        # 输入的 token IDs，表示输入的文本在词汇表中的索引序列
        input_ids,
        # 注意力掩码，用于指示模型在哪些位置需要进行注意力计算
        attention_mask=None,
        # 用于多句子输入时标识每个 token 属于哪个句子的 ID 序列
        token_type_ids=None,
        # 位置编码，用于指示每个 token 的位置信息
        position_ids=None,
        # 头部掩码，用于指定哪些注意力头需要被屏蔽
        head_mask=None,
        # 编码器的隐藏状态，用于在解码器中使用编码器的输出作为输入
        encoder_hidden_states=None,
        # 编码器的注意力掩码，用于指示编码器自注意力中哪些位置需要屏蔽
        encoder_attention_mask=None,
        # 用于传递额外参数的字典，可以包含模型的特定参数
        params: dict = None,
        # 随机数生成器，用于处理随机性的操作
        dropout_rng: jax.random.PRNGKey = None,
        # 是否处于训练模式
        train: bool = False,
        # 是否输出注意力分数
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典类型的输出结果
        return_dict: Optional[bool] = None,
        # 缓存的键值对，用于实现解码时的记忆功能
        past_key_values: dict = None,
# 定义一个 FlaxBertModule 类，继承自 nn.Module
class FlaxBertModule(nn.Module):
    # BertConfig 类型的 config 属性
    config: BertConfig
    # 计算的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 是否添加池化层，默认为 True
    add_pooling_layer: bool = True
    # 是否使用梯度检查点，默认为 False
    gradient_checkpointing: bool = False

    # 初始化方法
    def setup(self):
        # 初始化 embeddings 属性为 FlaxBertEmbeddings 类的实例
        self.embeddings = FlaxBertEmbeddings(self.config, dtype=self.dtype)
        # 初始化 encoder 属性为 FlaxBertEncoder 类的实例
        self.encoder = FlaxBertEncoder(
            self.config,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 初始化 pooler 属性为 FlaxBertPooler 类的实例
        self.pooler = FlaxBertPooler(self.config, dtype=self.dtype)

    # 调用方法
    def __call__(
        self,
        input_ids,
        attention_mask,
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
        # 如果 token_type_ids 未传入，则初始化为与 input_ids 相同形状的零数组
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        # 如果 position_ids 未传入，则初始化为广播形式的 input_ids 的最后一个维度的范围
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        # 获取 embeddings 层的输出
        hidden_states = self.embeddings(
            input_ids, token_type_ids, position_ids, attention_mask, deterministic=deterministic
        )
        # 获取 encoder 层的输出
        outputs = self.encoder(
            hidden_states,
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
        hidden_states = outputs[0]
        # 如果 add_pooling_layer 为 True，则获取池化层的输出
        pooled = self.pooler(hidden_states) if self.add_pooling_layer else None

        # 如果 return_dict 为 False
        if not return_dict:
            # 如果 pooled 为 None，则不返回它
            if pooled is None:
                return (hidden_states,) + outputs[1:]
            return (hidden_states, pooled) + outputs[1:]

        # 返回 FlaxBaseModelOutputWithPoolingAndCrossAttentions 类的实例
        return FlaxBaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

# 添加文档字符串，描述 FlaxBertModel 类
@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    BERT_START_DOCSTRING,
)
# 定义一个 FlaxBertModel 类，继承自 FlaxBertPreTrainedModel
class FlaxBertModel(FlaxBertPreTrainedModel):
    # module_class 属性为 FlaxBertModule 类
    module_class = FlaxBertModule
# 将示例文档字符串添加到 FlaxBertModel 类中
append_call_sample_docstring(FlaxBertModel, _CHECKPOINT_FOR_DOC, FlaxBaseModelOutputWithPooling, _CONFIG_FOR_DOC)

# 定义 FlaxBertForPreTrainingModule 类
class FlaxBertForPreTrainingModule(nn.Module):
    # 初始化类的属性
    config: BertConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    # 设置方法，初始化模型的组件
    def setup(self):
        # 创建 FlaxBertModule 实例
        self.bert = FlaxBertModule(
            config=self.config,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 创建 FlaxBertPreTrainingHeads 实例
        self.cls = FlaxBertPreTrainingHeads(config=self.config, dtype=self.dtype)

    # 调用方法，执行模型的前向传播
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 执行 BERT 模型的前向传播
        outputs = self.bert(
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

        # 根据配置决定是否共享词嵌入
        if self.config.tie_word_embeddings:
            shared_embedding = self.bert.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
        else:
            shared_embedding = None

        hidden_states = outputs[0]
        pooled_output = outputs[1]

        # 执行预训练头部的前向传播
        prediction_scores, seq_relationship_score = self.cls(
            hidden_states, pooled_output, shared_embedding=shared_embedding
        )

        # 根据 return_dict 决定返回结果的形式
        if not return_dict:
            return (prediction_scores, seq_relationship_score) + outputs[2:]

        # 返回 FlaxBertForPreTrainingOutput 实例
        return FlaxBertForPreTrainingOutput(
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# 添加文档字符串到 FlaxBertForPreTraining 类
@add_start_docstrings(
    """
    Bert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a `next
    sentence prediction (classification)` head.
    """,
    BERT_START_DOCSTRING,
)
class FlaxBertForPreTraining(FlaxBertPreTrainedModel):
    module_class = FlaxBertForPreTrainingModule

# 定义 FLAX_BERT_FOR_PRETRAINING_DOCSTRING 文档字符串
FLAX_BERT_FOR_PRETRAINING_DOCSTRING = """
    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, FlaxBertForPreTraining

    >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    >>> model = FlaxBertForPreTraining.from_pretrained("bert-base-uncased")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="np")
    >>> outputs = model(**inputs)

    >>> prediction_logits = outputs.prediction_logits
    >>> seq_relationship_logits = outputs.seq_relationship_logits
    ```
"""

# 覆盖 FlaxBertForPreTraining 类的调用文档字符串
overwrite_call_docstring(
    FlaxBertForPreTraining,
    # 使用 BERT_INPUTS_DOCSTRING 格式化字符串，插入 "batch_size, sequence_length" 参数说明，并与 FLAX_BERT_FOR_PRETRAINING_DOCSTRING 拼接
    BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length") + FLAX_BERT_FOR_PRETRAINING_DOCSTRING,
# 导入模块 append_replace_return_docstrings
append_replace_return_docstrings(
    FlaxBertForPreTraining, output_type=FlaxBertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC
)


# 定义类 FlaxBertForMaskedLMModule，继承自 nn.Module
class FlaxBertForMaskedLMModule(nn.Module):
    # 类的属性 config，表示 BERT 模型的配置信息
    config: BertConfig
    # 类的属性 dtype，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 类的属性 gradient_checkpointing，默认为 False，表示是否使用梯度检查点
    gradient_checkpointing: bool = False

    # 定义 setup 方法
    def setup(self):
        # 实例化 FlaxBertModule 类，作为 BERT 模型的主体部分
        self.bert = FlaxBertModule(
            config=self.config,
            add_pooling_layer=False,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 实例化 FlaxBertOnlyMLMHead 类，作为 BERT 模型的 Masked Language Modeling 头部
        self.cls = FlaxBertOnlyMLMHead(config=self.config, dtype=self.dtype)

    # 定义 __call__ 方法，用于模型的前向传播
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 调用 BERT 模型进行前向传播，获取输出结果
        outputs = self.bert(
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

        # 获取 BERT 模型的隐藏状态
        hidden_states = outputs[0]
        # 如果配置要求共享词嵌入，则获取共享的词嵌入
        if self.config.tie_word_embeddings:
            shared_embedding = self.bert.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
        else:
            shared_embedding = None

        # 计算预测分数
        logits = self.cls(hidden_states, shared_embedding=shared_embedding)

        # 如果不返回字典，则返回元组
        if not return_dict:
            return (logits,) + outputs[1:]

        # 返回 MaskedLM 的输出结果
        return FlaxMaskedLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# 添加模型说明文档
@add_start_docstrings("""Bert Model with a `language modeling` head on top.""", BERT_START_DOCSTRING)
class FlaxBertForMaskedLM(FlaxBertPreTrainedModel):
    # 指定模型类为 FlaxBertForMaskedLMModule
    module_class = FlaxBertForMaskedLMModule


# 添加调用示例的说明文档
append_call_sample_docstring(FlaxBertForMaskedLM, _CHECKPOINT_FOR_DOC, FlaxMaskedLMOutput, _CONFIG_FOR_DOC)


# 定义类 FlaxBertForNextSentencePredictionModule，继承自 nn.Module
class FlaxBertForNextSentencePredictionModule(nn.Module):
    # 类的属性 config，表示 BERT 模型的配置信息
    config: BertConfig
    # 类的属性 dtype，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 类的属性 gradient_checkpointing，默认为 False，表示是否使用梯度检查点
    gradient_checkpointing: bool = False

    # 定义 setup 方法
    def setup(self):
        # 实例化 FlaxBertModule 类，作为 BERT 模型的主体部分
        self.bert = FlaxBertModule(
            config=self.config,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 实例化 FlaxBertOnlyNSPHead 类，作为 BERT 模型的 Next Sentence Prediction 头部
        self.cls = FlaxBertOnlyNSPHead(dtype=self.dtype)

    # 定义 __call__ 方法，用于模型的前向传播
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 如果 return_dict 不为 None，则使用指定的 return_dict，否则使用配置中的 return_dict
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 使用 BERT 模型进行前向传播
        outputs = self.bert(
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

        # 提取 BERT 输出中的池化后的输出（pooled_output）
        pooled_output = outputs[1]

        # 使用分类层（self.cls）对池化后的输出进行下一句预测的分类
        seq_relationship_scores = self.cls(pooled_output)

        # 如果 return_dict 为 False，则返回元组，包括分类分数以及 BERT 输出中的隐藏状态
        if not return_dict:
            return (seq_relationship_scores,) + outputs[2:]

        # 否则，返回 FlaxNextSentencePredictorOutput 类型的对象，其中包括分类分数、隐藏状态和注意力权重
        return FlaxNextSentencePredictorOutput(
            logits=seq_relationship_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用装饰器添加文档字符串，说明此类是带有“下一句预测（分类）”头部的 Bert 模型
# 并引用了 BERT_START_DOCSTRING 中定义的文档字符串
@add_start_docstrings(
    """Bert Model with a `next sentence prediction (classification)` head on top.""",
    BERT_START_DOCSTRING,
)
# 定义 FlaxBertForNextSentencePrediction 类，继承自 FlaxBertPreTrainedModel
class FlaxBertForNextSentencePrediction(FlaxBertPreTrainedModel):
    # 将 module_class 属性指向 FlaxBertForNextSentencePredictionModule 类
    module_class = FlaxBertForNextSentencePredictionModule


# 定义 FLAX_BERT_FOR_NEXT_SENT_PRED_DOCSTRING 文档字符串
# 描述返回值和示例用法
FLAX_BERT_FOR_NEXT_SENT_PRED_DOCSTRING = """
    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, FlaxBertForNextSentencePrediction

    >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    >>> model = FlaxBertForNextSentencePrediction.from_pretrained("bert-base-uncased")

    >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
    >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
    >>> encoding = tokenizer(prompt, next_sentence, return_tensors="jax")

    >>> outputs = model(**encoding)
    >>> logits = outputs.logits
    >>> assert logits[0, 0] < logits[0, 1]  # next sentence was random
    ```
"""


# 使用 overwrite_call_docstring 函数替换 FlaxBertForNextSentencePrediction 类的调用文档字符串
# 描述输入和输出，并引用 FLAX_BERT_FOR_NEXT_SENT_PRED_DOCSTRING 中的文档字符串
overwrite_call_docstring(
    FlaxBertForNextSentencePrediction,
    BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length") + FLAX_BERT_FOR_NEXT_SENT_PRED_DOCSTRING,
)

# 使用 append_replace_return_docstrings 函数替换 FlaxBertForNextSentencePrediction 类的返回值文档字符串
# 将输出类型指定为 FlaxNextSentencePredictorOutput，并引用 _CONFIG_FOR_DOC
append_replace_return_docstrings(
    FlaxBertForNextSentencePrediction, output_type=FlaxNextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC
)


# 定义 FlaxBertForSequenceClassificationModule 类，继承自 nn.Module
class FlaxBertForSequenceClassificationModule(nn.Module):
    # 定义类属性 config，并指定 dtype 和 gradient_checkpointing 的默认值
    config: BertConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    # 初始化函数
    def setup(self):
        # 创建 FlaxBertModule 实例作为类属性 bert
        self.bert = FlaxBertModule(
            config=self.config,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 获取分类器的 dropout 比率，若未指定，则使用隐藏层的 dropout 比率
        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )
        # 创建 Dropout 层作为类属性 dropout
        self.dropout = nn.Dropout(rate=classifier_dropout)
        # 创建 Dense 层作为类属性 classifier，用于分类任务
        self.classifier = nn.Dense(
            self.config.num_labels,
            dtype=self.dtype,
        )

    # 调用函数
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        # Model
        # 调用 BERT 模型，传入输入的各种参数
        outputs = self.bert(
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

        # 获取 BERT 模型输出的池化后的结果
        pooled_output = outputs[1]
        # 对池化后的结果进行 dropout 处理
        pooled_output = self.dropout(pooled_output, deterministic=deterministic)
        # 将处理后的结果传入分类器得到 logits
        logits = self.classifier(pooled_output)

        # 如果不需要返回字典形式的结果
        if not return_dict:
            # 返回 logits 和 BERT 模型的隐藏状态
            return (logits,) + outputs[2:]

        # 返回字典形式的结果，包括 logits、隐藏状态和注意力权重
        return FlaxSequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用指定的文档字符串和 BERT 的起始文档字符串添加头部注释
@add_start_docstrings(
    """
    Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    BERT_START_DOCSTRING,
)
# 定义一个 FlaxBertForSequenceClassification 类，继承自 FlaxBertPreTrainedModel 类
class FlaxBertForSequenceClassification(FlaxBertPreTrainedModel):
    # 将模块类设置为 FlaxBertForSequenceClassificationModule
    module_class = FlaxBertForSequenceClassificationModule

# 添加样例调用的文档字符串
append_call_sample_docstring(
    FlaxBertForSequenceClassification,
    _CHECKPOINT_FOR_DOC,
    FlaxSequenceClassifierOutput,
    _CONFIG_FOR_DOC,
)

# 定义一个 FlaxBertForMultipleChoiceModule 类，继承自 nn.Module 类
class FlaxBertForMultipleChoiceModule(nn.Module):
    # 定义配置参数为 BertConfig，dtype 为 jnp.float32，默认情况下不使用梯度检查点
    config: BertConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    # 设置方法，初始化模块中的各个组件
    def setup(self):
        # 初始化 bert 层
        self.bert = FlaxBertModule(
            config=self.config,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 初始化 dropout 层
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        # 初始化分类器层，输出维度为 1
        self.classifier = nn.Dense(1, dtype=self.dtype)

    # 定义调用方法，接受多个输入参数
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 获取输入 ids 的数量
        num_choices = input_ids.shape[1]
        # 重塑输入，以便能够在多个选择之间进行运算
        input_ids = input_ids.reshape(-1, input_ids.shape[-1]) if input_ids is not None else None
        attention_mask = attention_mask.reshape(-1, attention_mask.shape[-1]) if attention_mask is not None else None
        token_type_ids = token_type_ids.reshape(-1, token_type_ids.shape[-1]) if token_type_ids is not None else None
        position_ids = position_ids.reshape(-1, position_ids.shape[-1]) if position_ids is not None else None

        # 使用 bert 模型进行前向传播
        outputs = self.bert(
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

        # 获取池化输出
        pooled_output = outputs[1]
        # 应用 dropout
        pooled_output = self.dropout(pooled_output, deterministic=deterministic)
        # 通过分类器进行分类
        logits = self.classifier(pooled_output)

        # 重新塑造 logits，以适应多选题的形式
        reshaped_logits = logits.reshape(-1, num_choices)

        # 如果不返回字典，则返回一个元组
        if not return_dict:
            return (reshaped_logits,) + outputs[2:]

        # 返回多选题模型输出
        return FlaxMultipleChoiceModelOutput(
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# 使用指定的文档字符串和 BERT 的起始文档字符串添加头部注释
@add_start_docstrings(
    """
    Bert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    BERT_START_DOCSTRING,
)
# 定义一个 FlaxBertForMultipleChoice 类，继承自 FlaxBertPreTrainedModel 类
class FlaxBertForMultipleChoice(FlaxBertPreTrainedModel):
    # 将模块类设置为 FlaxBertForMultipleChoiceModule
    module_class = FlaxBertForMultipleChoiceModule

# 覆盖调用方法的文档字符串
overwrite_call_docstring(
    # 导入 FlaxBertForMultipleChoice 类，用于多选题的 BERT 模型
    # 使用 BERT_INPUTS_DOCSTRING 格式化字符串，传入参数说明
    FlaxBertForMultipleChoice, BERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
# 在调用样例文档字符串后追加 FlaxBertForMultipleChoice 类的调用样例文档字符串
append_call_sample_docstring(
    FlaxBertForMultipleChoice, _CHECKPOINT_FOR_DOC, FlaxMultipleChoiceModelOutput, _CONFIG_FOR_DOC
)


class FlaxBertForTokenClassificationModule(nn.Module):
    # 定义 FlaxBertForTokenClassificationModule 类，继承自 nn.Module
    config: BertConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        # 初始化 self.bert 属性为一个 FlaxBertModule 实例，用于处理输入
        self.bert = FlaxBertModule(
            config=self.config,
            dtype=self.dtype,
            add_pooling_layer=False,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 根据配置初始化 dropout 层，用于防止过拟合
        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(rate=classifier_dropout)
        # 初始化分类器，用于执行标记分类任务
        self.classifier = nn.Dense(self.config.num_labels, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 执行模型前向传播
        outputs = self.bert(
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

        # 从模型输出中获取隐藏状态
        hidden_states = outputs[0]
        # 使用 dropout 层进行正则化处理
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 将隐藏状态输入分类器，得到分类结果
        logits = self.classifier(hidden_states)

        # 根据返回字典标志决定返回的内容
        if not return_dict:
            return (logits,) + outputs[1:]

        # 返回 Token 分类器的输出
        return FlaxTokenClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Bert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    BERT_START_DOCSTRING,
)
# 定义 FlaxBertForTokenClassification 类，继承自 FlaxBertPreTrainedModel
class FlaxBertForTokenClassification(FlaxBertPreTrainedModel):
    module_class = FlaxBertForTokenClassificationModule


# 在调用样例文档字符串后追加 FlaxBertForTokenClassification 类的调用样例文档字符串
append_call_sample_docstring(
    FlaxBertForTokenClassification, _CHECKPOINT_FOR_DOC, FlaxTokenClassifierOutput, _CONFIG_FOR_DOC
)


class FlaxBertForQuestionAnsweringModule(nn.Module):
    # 定义 FlaxBertForQuestionAnsweringModule 类，继承自 nn.Module
    config: BertConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        # 初始化 self.bert 属性为一个 FlaxBertModule 实例，用于处理输入
        self.bert = FlaxBertModule(
            config=self.config,
            dtype=self.dtype,
            add_pooling_layer=False,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 初始化 self.qa_outputs 属性为一个全连接层，用于执行问答任务
        self.qa_outputs = nn.Dense(self.config.num_labels, dtype=self.dtype)
```  
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 定义 __call__ 方法，接受输入参数
        # Model
        # 调用 BERT 模型，传入输入参数和其他设置
        outputs = self.bert(
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

        # 获取模型输出的隐藏状态
        hidden_states = outputs[0]

        # 将隐藏状态传入全连接层得到 logits
        logits = self.qa_outputs(hidden_states)
        # 将 logits 按照标签数量拆分为起始和结束 logits
        start_logits, end_logits = jnp.split(logits, self.config.num_labels, axis=-1)
        # 去除多余的维度
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # 如果不返回字典，则返回元组
        if not return_dict:
            return (start_logits, end_logits) + outputs[1:]

        # 返回 FlaxQuestionAnsweringModelOutput 对象
        return FlaxQuestionAnsweringModelOutput(
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 给 FlaxBertForQuestionAnswering 类添加文档字符串，描述其作为 Bert 模型在提取式问答任务（如 SQuAD）中的应用，包含一个线性层用于计算“起始位置对数”和“结束位置对数”的分类。
@add_start_docstrings(
    """
    Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    BERT_START_DOCSTRING,
)
class FlaxBertForQuestionAnswering(FlaxBertPreTrainedModel):
    # 指定模块类为 FlaxBertForQuestionAnsweringModule
    module_class = FlaxBertForQuestionAnsweringModule


# 添加调用示例文档字符串到 FlaxBertForQuestionAnswering 类
append_call_sample_docstring(
    FlaxBertForQuestionAnswering,
    _CHECKPOINT_FOR_DOC,
    FlaxQuestionAnsweringModelOutput,
    _CONFIG_FOR_DOC,
)


# 定义 FlaxBertForCausalLMModule 类
class FlaxBertForCausalLMModule(nn.Module):
    # 初始化方法
    config: BertConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        # 创建 BERT 模型，不包含池化层
        self.bert = FlaxBertModule(
            config=self.config,
            add_pooling_layer=False,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 创建仅包含 MLM 头部的模型
        self.cls = FlaxBertOnlyMLMHead(config=self.config, dtype=self.dtype)

    # 前向传播方法
    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        token_type_ids: Optional[jnp.ndarray] = None,
        head_mask: Optional[jnp.ndarray] = None,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 调用 BERT 模型前向传播
        outputs = self.bert(
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

        # 提取隐藏状态
        hidden_states = outputs[0]
        # 若配置了词嵌入共享，则获取共享的词嵌入层
        if self.config.tie_word_embeddings:
            shared_embedding = self.bert.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
        else:
            shared_embedding = None

        # 计算预测分数
        logits = self.cls(hidden_states, shared_embedding=shared_embedding)

        # 如果不返回字典，则返回元组
        if not return_dict:
            return (logits,) + outputs[1:]

        # 返回 CausalLMOutputWithCrossAttentions 对象
        return FlaxCausalLMOutputWithCrossAttentions(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


# 为 FlaxBertForCausalLM 类添加文档字符串，描述其作为 Bert 模型在语言建模任务中的应用，包含一个线性层用于计算隐藏状态输出的语言模型。
@add_start_docstrings(
    """
    Bert Model with a language modeling head on top (a linear layer on top of the hidden-states output) e.g for
    autoregressive tasks.
    """,
    BERT_START_DOCSTRING,
)
class FlaxBertForCausalLM(FlaxBertPreTrainedModel):
    # 定义模型类为FlaxBertForCausalLMModule
    module_class = FlaxBertForCausalLMModule
    
    # 为生成准备输入数据
    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = None):
        # 初始化缓存
        batch_size, seq_length = input_ids.shape
    
        # 初始化缓存
        past_key_values = self.init_cache(batch_size, max_length)
        
        # 注意：通常需要在attention_mask中为 x > input_ids.shape[-1] 和 x < cache_length 的位置放入0
        # 但由于解码器使用因果掩码，这些位置已经被掩盖了
        # 因此，我们可以在这里创建一个静态的attention_mask，这样更有效率
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
    
    # 更新生成的输入数据
    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs
# 调用函数 `append_call_sample_docstring`，用于向给定类的文档字符串中添加调用示例
# 第一个参数是目标类 `FlaxBertForCausalLM`，将在其文档字符串中添加示例
# 第二个参数是检查点文件的路径 `_CHECKPOINT_FOR_DOC`，用于示例中的模型加载
# 第三个参数是输出类 `FlaxCausalLMOutputWithCrossAttentions`，示例中的模型输出
# 第四个参数是配置文件的路径 `_CONFIG_FOR_DOC`，用于示例中的模型初始化配置
append_call_sample_docstring(
    FlaxBertForCausalLM,
    _CHECKPOINT_FOR_DOC,
    FlaxCausalLMOutputWithCrossAttentions,
    _CONFIG_FOR_DOC,
)
```