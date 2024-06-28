# `.\models\bert\modeling_flax_bert.py`

```py
# 导入所需的模块和类
from typing import Callable, Optional, Tuple  # 导入类型注解相关的类和方法

import flax  # 导入 Flax 深度学习框架
import flax.linen as nn  # 导入 Flax 提供的线性 API 模块
import jax  # 导入 JAX，用于定义和执行计算
import jax.numpy as jnp  # 导入 JAX 提供的 NumPy 兼容的数组处理工具
import numpy as np  # 导入 NumPy 数组处理库
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze  # 导入 Flax 提供的冻结字典相关方法
from flax.linen import combine_masks, make_causal_mask  # 导入 Flax 提供的掩码组合和因果掩码生成方法
from flax.linen import partitioning as nn_partitioning  # 导入 Flax 提供的分区模块
from flax.linen.attention import dot_product_attention_weights  # 导入 Flax 提供的点积注意力权重计算方法
from flax.traverse_util import flatten_dict, unflatten_dict  # 导入 Flax 提供的字典扁平化和反扁平化方法
from jax import lax  # 导入 JAX 提供的线性代数加速模块

# 从外部模块导入不同输出类型的模型结果类
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
# 从外部模块导入不同工具类和方法
from ...modeling_flax_utils import (
    ACT2FN,
    FlaxPreTrainedModel,
    append_call_sample_docstring,
    append_replace_return_docstrings,
    overwrite_call_docstring,
)
# 从外部模块导入模型输出类和配置相关的方法
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging

# 从本地模块导入 BERT 模型配置类
from .configuration_bert import BertConfig

# 获取日志记录器实例
logger = logging.get_logger(__name__)

# 定义用于文档的检查点名称和配置名称
_CHECKPOINT_FOR_DOC = "google-bert/bert-base-uncased"
_CONFIG_FOR_DOC = "BertConfig"

# 定义 Flax 提供的重要函数 remat
remat = nn_partitioning.remat

# 定义 FlaxBertForPreTrainingOutput 类，继承自 ModelOutput，用于表示预训练过程的输出类型
@flax.struct.dataclass
class FlaxBertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`BertForPreTraining`].
    """
    # 预测语言模型头部的预测得分（每个词汇标记的得分，未经过 SoftMax 处理）
    prediction_logits: jnp.ndarray = None

    # 下一序列预测（分类）头部的预测得分（True/False 继续的得分，未经过 SoftMax 处理）
    seq_relationship_logits: jnp.ndarray = None

    # 模型隐藏层的隐藏状态元组（当 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回）
    # 包含了每一层的输出（除了嵌入层外）的 `jnp.ndarray` 数组，形状为 `(batch_size, sequence_length, hidden_size)`
    hidden_states: Optional[Tuple[jnp.ndarray]] = None

    # 自注意力头部的注意力权重元组（当 `output_attentions=True` 或 `config.output_attentions=True` 时返回）
    # 包含了每一层的注意力权重 `jnp.ndarray` 数组，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`
    attentions: Optional[Tuple[jnp.ndarray]] = None
# BERT_START_DOCSTRING 是一个原始字符串文档，包含了有关模型继承关系和Flax模块的详细说明。
# 这个模型继承自 FlaxPreTrainedModel，并且可以作为 flax.linen.Module 使用。
# 支持 JAX 的 JIT 编译、自动微分、向量化和并行化特性。
# 
# Parameters:
#     config ([BertConfig]): 包含模型所有参数的配置类。
#         使用配置文件初始化时，不会加载与模型关联的权重，只加载配置。
#         可以使用 FlaxPreTrainedModel.from_pretrained 方法加载模型权重。
#     dtype (jax.numpy.dtype, 可选，默认为 jax.numpy.float32):
#         计算的数据类型。可以是 jax.numpy.float32、jax.numpy.float16（在GPU上）和 jax.numpy.bfloat16（在TPU上）。
#         可用于在GPU或TPU上启用混合精度训练或半精度推断。如果指定，则所有计算都将使用给定的 dtype。
#         
#         注意，这只指定计算的数据类型，不影响模型参数的数据类型。
#         
#         如果要更改模型参数的数据类型，请参阅 FlaxPreTrainedModel.to_fp16 和 FlaxPreTrainedModel.to_bf16。
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

# BERT_INPUTS_DOCSTRING 是一个原始字符串文档，目前为空，用于稍后添加BERT模型输入的说明文档。
BERT_INPUTS_DOCSTRING = r"""
    # Args 是一个 docstring（文档字符串）的一部分，用于描述函数的参数信息
    Args:
        input_ids (`numpy.ndarray` of shape `({0})`):
            # 输入序列中的标记索引，在词汇表中表示每个标记
            Indices of input sequence tokens in the vocabulary.
    
            # 可以使用 AutoTokenizer 来获取这些索引，详见 PreTrainedTokenizer.encode 和 PreTrainedTokenizer.__call__ 的详情
    
            [What are input IDs?](../glossary#input-ids)
            # 查看更多关于 input IDs 的信息，链接到 glossary 的相应条目
    
        attention_mask (`numpy.ndarray` of shape `({0})`, *optional*):
            # 用于避免在填充的标记索引上进行注意力计算的掩码
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
            - 1 表示**不被遮蔽**的标记，
            - 0 表示**被遮蔽**的标记。
    
            [What are attention masks?](../glossary#attention-mask)
            # 查看更多关于 attention masks 的信息，链接到 glossary 的相应条目
    
        token_type_ids (`numpy.ndarray` of shape `({0})`, *optional*):
            # 段标记索引，指示输入的第一部分和第二部分。索引在 `[0, 1]` 中选择：
    
            - 0 对应于*句子 A* 的标记，
            - 1 对应于*句子 B* 的标记。
    
            [What are token type IDs?](../glossary#token-type-ids)
            # 查看更多关于 token type IDs 的信息，链接到 glossary 的相应条目
    
        position_ids (`numpy.ndarray` of shape `({0})`, *optional*):
            # 每个输入序列标记在位置嵌入中的位置索引。选择在范围 `[0, config.max_position_embeddings - 1]` 中。
    
        head_mask (`numpy.ndarray` of shape `({0})`, `optional):
            # 用于使注意力模块中的特定 head 失效的掩码。掩码值选择在 `[0, 1]` 中：
    
            - 1 表示该 head **未被遮蔽**，
            - 0 表示该 head **被遮蔽**。
    
        return_dict (`bool`, *optional*):
            # 是否返回 `~utils.ModelOutput` 而不是普通元组。
    
    # 上述内容描述了函数的各个参数及其作用，帮助用户理解如何使用这些参数调用函数。
"""
class FlaxBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    config: BertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        # 初始化词嵌入层，输入词汇表大小和隐藏层大小，并使用正态分布初始化
        self.word_embeddings = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        # 初始化位置嵌入层，输入最大位置嵌入数量和隐藏层大小，并使用正态分布初始化
        self.position_embeddings = nn.Embed(
            self.config.max_position_embeddings,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        # 初始化类型嵌入层，输入类型词汇表大小和隐藏层大小，并使用正态分布初始化
        self.token_type_embeddings = nn.Embed(
            self.config.type_vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        # 初始化 Layer Normalization 层，设置 epsilon 为配置中的值
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 初始化 Dropout 层，设置丢弃率为配置中的值
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, input_ids, token_type_ids, position_ids, attention_mask, deterministic: bool = True):
        # 嵌入输入的词嵌入向量，将输入类型转换为整型
        inputs_embeds = self.word_embeddings(input_ids.astype("i4"))
        # 嵌入位置向量，将位置编码转换为整型
        position_embeds = self.position_embeddings(position_ids.astype("i4"))
        # 嵌入类型向量，将类型编码转换为整型
        token_type_embeddings = self.token_type_embeddings(token_type_ids.astype("i4"))

        # 汇总所有嵌入向量
        hidden_states = inputs_embeds + token_type_embeddings + position_embeds

        # 进行 Layer Normalization
        hidden_states = self.LayerNorm(hidden_states)
        # 应用 Dropout，根据 deterministic 参数决定是否使用确定性 Dropout
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states


class FlaxBertSelfAttention(nn.Module):
    config: BertConfig
    causal: bool = False
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
"""
    def setup(self):
        # 将隐藏层大小分成多个注意力头的维度
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        # 如果隐藏层大小不能被注意力头数整除，抛出数值错误
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError(
                "`config.hidden_size`: {self.config.hidden_size} has to be a multiple of `config.num_attention_heads` "
                "                   : {self.config.num_attention_heads}"
            )

        # 初始化查询权重的全连接层
        self.query = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        # 初始化键权重的全连接层
        self.key = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        # 初始化值权重的全连接层
        self.value = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

        # 如果启用因果注意力机制，创建因果掩码
        if self.causal:
            self.causal_mask = make_causal_mask(
                jnp.ones((1, self.config.max_position_embeddings), dtype="bool"), dtype="bool"
            )

    def _split_heads(self, hidden_states):
        # 将隐藏状态张量分割成多个注意力头
        return hidden_states.reshape(hidden_states.shape[:2] + (self.config.num_attention_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        # 将分割的注意力头合并回原始隐藏状态张量
        return hidden_states.reshape(hidden_states.shape[:2] + (self.config.hidden_size,))

    @nn.compact
    # 从 transformers.models.bart.modeling_flax_bart.FlaxBartAttention._concatenate_to_cache 复制而来
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slightly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # 检测是否通过缺失现有缓存数据进行初始化。
        is_initialized = self.has_variable("cache", "cached_key")
        # 获取或创建缓存的键值状态
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        # 获取或创建缓存索引
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # 使用新的一维空间片段更新键、值缓存
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            # 更新缓存索引以反映新加入的缓存向量数
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # 对于缓存的解码器自注意力，使用因果掩码：我们的单个查询位置只应与已生成和缓存的键位置相关联，而不是剩余的零元素。
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            # 合并因果掩码和给定的注意力掩码
            attention_mask = combine_masks(pad_mask, attention_mask)
        
        # 返回更新后的键、值和注意力掩码
        return key, value, attention_mask
# 定义 FlaxBertSelfOutput 类，继承自 nn.Module
class FlaxBertSelfOutput(nn.Module):
    config: BertConfig  # 类型注解，指定配置为 BertConfig 类型
    dtype: jnp.dtype = jnp.float32  # 计算中使用的数据类型

    # 设置方法，初始化网络层
    def setup(self):
        # 创建全连接层，输入大小为 hidden_size，使用正态分布初始化权重
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 创建 LayerNorm 层，使用配置中的 epsilon 参数
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 创建 Dropout 层，使用配置中的 hidden_dropout_prob 参数
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    # 调用方法，定义网络层间的传递逻辑
    def __call__(self, hidden_states, input_tensor, deterministic: bool = True):
        # 全连接层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # 使用 Dropout 层处理隐藏状态，根据 deterministic 参数确定是否采用确定性方式
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 使用 LayerNorm 层处理隐藏状态和输入张量的加和结果
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的隐藏状态
        return hidden_states


# 定义 FlaxBertAttention 类，继承自 nn.Module
class FlaxBertAttention(nn.Module):
    config: BertConfig  # 类型注解，指定配置为 BertConfig 类型
    causal: bool = False  # 是否使用因果注意力的标志，默认为 False
    dtype: jnp.dtype = jnp.float32  # 计算中使用的数据类型

    # 设置方法，初始化网络层
    def setup(self):
        # 创建自注意力层，使用配置和 causal 参数
        self.self = FlaxBertSelfAttention(self.config, causal=self.causal, dtype=self.dtype)
        # 创建自输出层，使用配置和数据类型参数
        self.output = FlaxBertSelfOutput(self.config, dtype=self.dtype)

    # 调用方法，定义网络层间的传递逻辑
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
        # FLAX 需要形状为 (*batch_sizes, 1, 1, kv_length)，以便与注意力权重形状匹配
        attn_outputs = self.self(
            hidden_states,
            attention_mask,
            layer_head_mask=layer_head_mask,
            key_value_states=key_value_states,
            init_cache=init_cache,
            deterministic=deterministic,
            output_attentions=output_attentions,
        )
        # 获取自注意力层的输出
        attn_output = attn_outputs[0]
        # 使用自输出层处理自注意力层的输出和隐藏状态
        hidden_states = self.output(attn_output, hidden_states, deterministic=deterministic)

        # 构建输出元组
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则添加到输出元组中
        if output_attentions:
            outputs += (attn_outputs[1],)

        # 返回最终输出
        return outputs


# 定义 FlaxBertIntermediate 类，继承自 nn.Module
class FlaxBertIntermediate(nn.Module):
    config: BertConfig  # 类型注解，指定配置为 BertConfig 类型
    dtype: jnp.dtype = jnp.float32  # 计算中使用的数据类型

    # 设置方法，初始化网络层
    def setup(self):
        # 创建全连接层，输入大小为 intermediate_size，使用正态分布初始化权重
        self.dense = nn.Dense(
            self.config.intermediate_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 使用配置中的隐藏激活函数名称，创建激活函数层
        self.activation = ACT2FN[self.config.hidden_act]

    # 调用方法，定义网络层间的传递逻辑
    def __call__(self, hidden_states):
        # 全连接层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # 使用激活函数处理全连接层的输出
        hidden_states = self.activation(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states


# 定义 FlaxBertOutput 类，继承自 nn.Module
class FlaxBertOutput(nn.Module):
    config: BertConfig  # 类型注解，指定配置为 BertConfig 类型
    dtype: jnp.dtype = jnp.float32  # 计算中使用的数据类型
    # 初始化模型中的层和参数
    def setup(self):
        # 创建一个全连接层，输出维度为 self.config.hidden_size
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 创建一个 Dropout 层，用于随机失活以防止过拟合
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        # 创建一个 LayerNorm 层，用于层标准化，epsilon 为 self.config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    # 定义模型的调用方法，实现前向传播
    def __call__(self, hidden_states, attention_output, deterministic: bool = True):
        # 使用全连接层进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对变换后的结果进行 Dropout 操作，以减少过拟合风险
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 对结果进行层标准化，并与 attention_output 相加作为最终输出
        hidden_states = self.LayerNorm(hidden_states + attention_output)
        # 返回最终的隐藏状态表示
        return hidden_states
class FlaxBertLayer(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型

    def setup(self):
        # 初始化注意力层、中间层和输出层
        self.attention = FlaxBertAttention(self.config, causal=self.config.is_decoder, dtype=self.dtype)
        self.intermediate = FlaxBertIntermediate(self.config, dtype=self.dtype)
        self.output = FlaxBertOutput(self.config, dtype=self.dtype)
        # 如果配置中包含交叉注意力，初始化交叉注意力层
        if self.config.add_cross_attention:
            self.crossattention = FlaxBertAttention(self.config, causal=False, dtype=self.dtype)

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

        hidden_states = self.intermediate(attention_output)
        hidden_states = self.output(hidden_states, attention_output, deterministic=deterministic)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attention_outputs[1],)
            if encoder_hidden_states is not None:
                outputs += (cross_attention_outputs[1],)
        return outputs


class FlaxBertLayerCollection(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型
    gradient_checkpointing: bool = False

    def setup(self):
        if self.gradient_checkpointing:
            # 如果梯度检查点开启，使用 remat 函数包装 FlaxBertLayer 并创建层集合
            FlaxBertCheckpointLayer = remat(FlaxBertLayer, static_argnums=(5, 6, 7))
            self.layers = [
                FlaxBertCheckpointLayer(self.config, name=str(i), dtype=self.dtype)
                for i in range(self.config.num_hidden_layers)
            ]
        else:
            # 否则，直接创建 FlaxBertLayer 的层集合
            self.layers = [
                FlaxBertLayer(self.config, name=str(i), dtype=self.dtype) for i in range(self.config.num_hidden_layers)
            ]
    # 定义 __call__ 方法，用于将对象实例作为可调用函数使用
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
        # 如果需要输出注意力权重，则初始化空元组，否则设为 None
        all_attentions = () if output_attentions else None
        # 如果需要输出隐藏状态，则初始化空元组，否则设为 None
        all_hidden_states = () if output_hidden_states else None
        # 如果同时需要输出注意力权重且有编码器隐藏状态，则初始化空元组，否则设为 None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        # 检查头部掩码的层数是否正确
        if head_mask is not None:
            if head_mask.shape[0] != (len(self.layers)):
                # 抛出异常，指出头部掩码应指定为与层数相同的层数
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for "
                    f"{head_mask.shape[0]}."
                )

        # 遍历每一层并执行前向传播
        for i, layer in enumerate(self.layers):
            # 如果需要输出隐藏状态，则记录当前层的隐藏状态
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # 调用当前层的前向传播方法
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

            # 提取当前层的输出隐藏状态
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力权重，则记录当前层的注意力权重
            if output_attentions:
                all_attentions += (layer_outputs[1],)

                # 如果有编码器隐藏状态，则记录当前层的交叉注意力权重
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # 如果需要输出隐藏状态，则记录最终的隐藏状态
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 整理所有输出，并根据 return_dict 决定输出格式
        outputs = (hidden_states, all_hidden_states, all_attentions, all_cross_attentions)

        if not return_dict:
            # 如果不需要以字典形式返回，则返回一个包含非 None 值的元组
            return tuple(v for v in outputs if v is not None)

        # 否则，以指定的输出格式返回结果
        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )
# 定义一个 FlaxBertEncoder 类，继承自 nn.Module，用于BERT编码器的实现
class FlaxBertEncoder(nn.Module):
    # 类属性：BERT 的配置信息
    config: BertConfig
    # 类属性：计算过程中使用的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 类属性：是否使用梯度检查点，默认为 False
    gradient_checkpointing: bool = False

    # 初始化方法，设置编码器的层集合
    def setup(self):
        self.layer = FlaxBertLayerCollection(
            self.config,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )

    # 调用方法，用于执行编码器的前向计算
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


# 定义一个 FlaxBertPooler 类，继承自 nn.Module，用于BERT的池化器
class FlaxBertPooler(nn.Module):
    # 类属性：BERT 的配置信息
    config: BertConfig
    # 类属性：计算过程中使用的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 初始化方法，设置池化器的全连接层
    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )

    # 调用方法，用于执行池化器的前向计算
    def __call__(self, hidden_states):
        # 取第一个位置的隐藏状态作为池化器输入
        cls_hidden_state = hidden_states[:, 0]
        # 经过全连接层变换
        cls_hidden_state = self.dense(cls_hidden_state)
        # 使用双曲正切函数进行激活
        return nn.tanh(cls_hidden_state)


# 定义一个 FlaxBertPredictionHeadTransform 类，继承自 nn.Module，用于BERT预测头的变换
class FlaxBertPredictionHeadTransform(nn.Module):
    # 类属性：BERT 的配置信息
    config: BertConfig
    # 类属性：计算过程中使用的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 初始化方法，设置预测头变换的全连接层、激活函数和 LayerNorm 层
    def setup(self):
        self.dense = nn.Dense(self.config.hidden_size, dtype=self.dtype)
        self.activation = ACT2FN[self.config.hidden_act]
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    # 调用方法，用于执行预测头变换的前向计算
    def __call__(self, hidden_states):
        # 经过全连接层变换
        hidden_states = self.dense(hidden_states)
        # 经过激活函数变换
        hidden_states = self.activation(hidden_states)
        # 经过 LayerNorm 层变换
        return self.LayerNorm(hidden_states)


# 定义一个 FlaxBertLMPredictionHead 类，继承自 nn.Module，用于BERT的语言模型预测头
class FlaxBertLMPredictionHead(nn.Module):
    # 类属性：BERT 的配置信息
    config: BertConfig
    # 类属性：计算过程中使用的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 类属性：偏置初始化函数，默认为全零
    bias_init: Callable[..., np.ndarray] = jax.nn.initializers.zeros

    # 初始化方法，设置预测头的变换和全连接层
    def setup(self):
        self.transform = FlaxBertPredictionHeadTransform(self.config, dtype=self.dtype)
        self.decoder = nn.Dense(self.config.vocab_size, dtype=self.dtype, use_bias=False)
        self.bias = self.param("bias", self.bias_init, (self.config.vocab_size,))
    # 定义一个特殊方法 __call__，用于将对象实例像函数一样调用
    def __call__(self, hidden_states, shared_embedding=None):
        # 调用 transform 方法，对输入的 hidden_states 进行转换处理
        hidden_states = self.transform(hidden_states)

        # 如果传入了共享的嵌入 shared_embedding，则使用 decoder 对象的 apply 方法
        if shared_embedding is not None:
            # 通过 decoder 对象的 apply 方法应用共享嵌入的转置 kernel 参数到 hidden_states
            hidden_states = self.decoder.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
        else:
            # 如果没有共享嵌入，则直接使用 decoder 对象处理 hidden_states
            hidden_states = self.decoder(hidden_states)

        # 将对象的 bias 属性转换为 JAX 的数组，并将其加到 hidden_states 上
        bias = jnp.asarray(self.bias, self.dtype)
        hidden_states += bias
        
        # 返回经过处理后的 hidden_states
        return hidden_states
class FlaxBertOnlyMLMHead(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32

    # 初始化模块，设置预测头部
    def setup(self):
        self.predictions = FlaxBertLMPredictionHead(self.config, dtype=self.dtype)

    # 调用模块，生成预测结果
    def __call__(self, hidden_states, shared_embedding=None):
        hidden_states = self.predictions(hidden_states, shared_embedding=shared_embedding)
        return hidden_states


class FlaxBertOnlyNSPHead(nn.Module):
    dtype: jnp.dtype = jnp.float32

    # 初始化模块，设置序列关系预测头部
    def setup(self):
        self.seq_relationship = nn.Dense(2, dtype=self.dtype)

    # 调用模块，生成序列关系预测结果
    def __call__(self, pooled_output):
        return self.seq_relationship(pooled_output)


class FlaxBertPreTrainingHeads(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32

    # 初始化模块，设置预测头部和序列关系预测头部
    def setup(self):
        self.predictions = FlaxBertLMPredictionHead(self.config, dtype=self.dtype)
        self.seq_relationship = nn.Dense(2, dtype=self.dtype)

    # 调用模块，生成预测头部和序列关系预测结果
    def __call__(self, hidden_states, pooled_output, shared_embedding=None):
        prediction_scores = self.predictions(hidden_states, shared_embedding=shared_embedding)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class FlaxBertPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BertConfig
    base_model_prefix = "bert"
    module_class: nn.Module = None

    # 初始化预训练模型，设置模块类和参数
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
        module = self.module_class(
            config=config,
            dtype=dtype,
            gradient_checkpointing=gradient_checkpointing,
            **kwargs,
        )
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    # 启用梯度检查点
    def enable_gradient_checkpointing(self):
        self._module = self.module_class(
            config=self.config,
            dtype=self.dtype,
            gradient_checkpointing=True,
        )
    # 初始化模型权重的函数，接受随机数生成器rng、输入形状input_shape和可选参数params，并返回参数字典
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量
        input_ids = jnp.zeros(input_shape, dtype="i4")  # 创建全零的输入张量
        token_type_ids = jnp.zeros_like(input_ids)  # 创建与input_ids相同形状的全零张量
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)  # 根据input_ids形状广播生成位置张量
        attention_mask = jnp.ones_like(input_ids)  # 创建与input_ids相同形状的全一张量作为注意力掩码
        head_mask = jnp.ones((self.config.num_hidden_layers, self.config.num_attention_heads))  # 创建全一头掩码张量

        params_rng, dropout_rng = jax.random.split(rng)  # 分割随机数生成器rng，用于参数和dropout

        rngs = {"params": params_rng, "dropout": dropout_rng}  # 创建包含params_rng和dropout_rng的随机数生成器字典

        if self.config.add_cross_attention:
            # 如果配置中包含跨注意力，则初始化编码器隐藏状态和注意力掩码
            encoder_hidden_states = jnp.zeros(input_shape + (self.config.hidden_size,))
            encoder_attention_mask = attention_mask
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
            )  # 调用模块的初始化函数，传入相应参数，返回初始化输出
        else:
            module_init_outputs = self.module.init(
                rngs, input_ids, attention_mask, token_type_ids, position_ids, head_mask, return_dict=False
            )  # 调用模块的初始化函数，传入相应参数，返回初始化输出

        random_params = module_init_outputs["params"]  # 从初始化输出中获取随机参数

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))  # 展开并解冻随机参数
            params = flatten_dict(unfreeze(params))  # 展开并解冻输入参数
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]  # 将缺失的键从随机参数复制到输入参数中
            self._missing_keys = set()  # 清空缺失键集合
            return freeze(unflatten_dict(params))  # 冻结和恢复输入参数字典结构并返回
        else:
            return random_params  # 如果没有输入参数，则直接返回随机参数

    # 从transformers.models.bart.modeling_flax_bart.FlaxBartDecoderPreTrainedModel.init_cache复制的函数
    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                fast auto-regressive decoding使用的批量大小，定义初始化缓存的批量大小。
            max_length (`int`):
                auto-regressive decoding的最大可能长度，定义初始化缓存的序列长度。
        """
        # 初始化用于检索缓存的输入变量
        input_ids = jnp.ones((batch_size, max_length), dtype="i4")  # 创建全一的输入张量
        attention_mask = jnp.ones_like(input_ids, dtype="i4")  # 创建与input_ids相同形状的全一张量作为注意力掩码
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)  # 根据input_ids形状广播生成位置张量

        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )  # 调用模块的初始化函数，传入相应参数并初始化缓存，返回初始化变量
        return unfreeze(init_variables["cache"])  # 解冻并返回初始化变量中的缓存

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 定义一个特殊方法 __call__，使得对象可以像函数一样被调用
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
# 定义一个名为FlaxBertModule的类，并继承自nn.Module
class FlaxBertModule(nn.Module):
    # 声明一个类型为BertConfig的config变量
    config: BertConfig
    # 声明一个名为dtype的变量，类型为jnp.dtype，默认值为jnp.float32，表示计算的数据类型
    dtype: jnp.dtype = jnp.float32
    # 声明一个名为add_pooling_layer的变量，类型为bool，默认值为True
    add_pooling_layer: bool = True
    # 声明一个名为gradient_checkpointing的变量，类型为bool，默认值为False
    gradient_checkpointing: bool = False

    # 定义一个setup方法
    def setup(self):
        # 初始化self.embeddings为FlaxBertEmbeddings类的实例对象
        self.embeddings = FlaxBertEmbeddings(self.config, dtype=self.dtype)
        # 初始化self.encoder为FlaxBertEncoder类的实例对象
        self.encoder = FlaxBertEncoder(
            self.config,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 初始化self.pooler为FlaxBertPooler类的实例对象
        self.pooler = FlaxBertPooler(self.config, dtype=self.dtype)

    # 定义一个__call__方法
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
        # 如果token_type_ids为None，则初始化为和input_ids形状相同的全0数组
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        # 如果position_ids为None，则初始化为将一维数组变成二维数组后进行广播扩展
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        # 将输入数据传递给self.embeddings进行处理，得到hidden_states
        hidden_states = self.embeddings(
            input_ids, token_type_ids, position_ids, attention_mask, deterministic=deterministic
        )
        # 将hidden_states传递给self.encoder进行处理，得到outputs
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
        # 从outputs中获取第一个元素赋值给hidden_states
        hidden_states = outputs[0]
        # 如果add_pooling_layer为True，则将hidden_states传递给self.pooler进行处理得到pooled，否则pooled为None
        pooled = self.pooler(hidden_states) if self.add_pooling_layer else None

        # 如果return_dict为False
        if not return_dict:
            # 如果pooled为None，不返回pooled
            if pooled is None:
                return (hidden_states,) + outputs[1:]
            # 返回包含hidden_states、pooled和outputs[1:]的元组
            return (hidden_states, pooled) + outputs[1:]

        # 返回FlaxBaseModelOutputWithPoolingAndCrossAttentions的实例对象，包含指定的属性和值
        return FlaxBaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

# 使用add_start_docstrings函数装饰FlaxBertModel类
@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    BERT_START_DOCSTRING,
)
class FlaxBertModel(FlaxBertPreTrainedModel):
    # 设置module_class属性为FlaxBertModule类
    module_class = FlaxBertModule
# 调用函数 `overwrite_call_docstring`，用于覆盖指定类的调用方法的文档字符串
overwrite_call_docstring(
    FlaxBertForPreTraining,



@add_start_docstrings(
    """
    Bert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a `next
    sentence prediction (classification)` head.
    """,
    BERT_START_DOCSTRING,
)



# 创建一个自定义的文档字符串注解，描述了 Bert 模型在预训练期间的结构，包括了 `masked language modeling` 和 `next sentence prediction (classification)` 两个头部任务
@add_start_docstrings(
    """
    Bert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a `next
    sentence prediction (classification)` head.
    """,
    BERT_START_DOCSTRING,
)



class FlaxBertForPreTraining(FlaxBertPreTrainedModel):
    # 将模型类设置为 FlaxBertForPreTrainingModule
    module_class = FlaxBertForPreTrainingModule



FLAX_BERT_FOR_PRETRAINING_DOCSTRING = """
    Returns:

    Example:

    ```
    >>> from transformers import AutoTokenizer, FlaxBertForPreTraining

    >>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    >>> model = FlaxBertForPreTraining.from_pretrained("google-bert/bert-base-uncased")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="np")
    >>> outputs = model(**inputs)

    >>> prediction_logits = outputs.prediction_logits
    >>> seq_relationship_logits = outputs.seq_relationship_logits
    ```



# 定义 FLAX_BERT_FOR_PRETRAINING_DOCSTRING，包含函数的返回值和使用示例
FLAX_BERT_FOR_PRETRAINING_DOCSTRING = """
    Returns:

    Example:

    ```
    >>> from transformers import AutoTokenizer, FlaxBertForPreTraining

    >>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    >>> model = FlaxBertForPreTraining.from_pretrained("google-bert/bert-base-uncased")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="np")
    >>> outputs = model(**inputs)

    >>> prediction_logits = outputs.prediction_logits
    >>> seq_relationship_logits = outputs.seq_relationship_logits
    ```
    # 使用字符串格式化函数 BERT_INPUTS_DOCSTRING 格式化输入的参数 "batch_size, sequence_length"，并加上 FLAX_BERT_FOR_PRETRAINING_DOCSTRING 的内容
    BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length") + FLAX_BERT_FOR_PRETRAINING_DOCSTRING,
# 导入所需模块和函数
append_replace_return_docstrings(
    FlaxBertForPreTraining, output_type=FlaxBertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC
)

# 定义一个自定义的 nn.Module 类 FlaxBertForMaskedLMModule，用于处理 BERT 模型的 masked language modeling 任务
class FlaxBertForMaskedLMModule(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    # 初始化函数，设置模型的各种参数和组件
    def setup(self):
        # 初始化一个 FlaxBertModule 实例，作为主要的 BERT 模型
        self.bert = FlaxBertModule(
            config=self.config,
            add_pooling_layer=False,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 初始化一个 FlaxBertOnlyMLMHead 实例，用于预测 masked token
        self.cls = FlaxBertOnlyMLMHead(config=self.config, dtype=self.dtype)

    # 调用函数，定义模型的前向传播过程
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
        # 调用 bert 模型进行前向传播，得到输出
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

        # 获取隐藏状态作为模型预测的输入
        hidden_states = outputs[0]
        
        # 根据配置判断是否共享词嵌入
        if self.config.tie_word_embeddings:
            shared_embedding = self.bert.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
        else:
            shared_embedding = None

        # 计算预测的 logits
        logits = self.cls(hidden_states, shared_embedding=shared_embedding)

        # 根据 return_dict 决定返回的格式
        if not return_dict:
            return (logits,) + outputs[1:]

        # 返回 FlaxMaskedLMOutput 类型的结果，包括 logits、隐藏状态和注意力分布
        return FlaxMaskedLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# 将自动生成的文档字符串添加到 FlaxBertForMaskedLM 类上，用于描述其语言建模头部的 BERT 模型
@add_start_docstrings("""Bert Model with a `language modeling` head on top.""", BERT_START_DOCSTRING)
class FlaxBertForMaskedLM(FlaxBertPreTrainedModel):
    module_class = FlaxBertForMaskedLMModule


# 添加示例调用文档字符串到 FlaxBertForMaskedLM 类上，用于指定检查点、输出类型和配置的示例调用
append_call_sample_docstring(FlaxBertForMaskedLM, _CHECKPOINT_FOR_DOC, FlaxMaskedLMOutput, _CONFIG_FOR_DOC)


# 定义一个自定义的 nn.Module 类 FlaxBertForNextSentencePredictionModule，用于处理 BERT 模型的下一句预测任务
class FlaxBertForNextSentencePredictionModule(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    # 初始化函数，设置模型的各种参数和组件
    def setup(self):
        # 初始化一个 FlaxBertModule 实例，作为主要的 BERT 模型
        self.bert = FlaxBertModule(
            config=self.config,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 初始化一个 FlaxBertOnlyNSPHead 实例，用于预测下一句
        self.cls = FlaxBertOnlyNSPHead(dtype=self.dtype)

    # 调用函数，定义模型的前向传播过程
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
        # 如果 return_dict 不为 None，则使用指定的 return_dict；否则使用类的默认配置值
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 调用 BERT 模型进行推断
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

        # 从 BERT 输出中获取池化后的特征
        pooled_output = outputs[1]
        
        # 将池化后的特征输入到分类层，得到句子关系的预测分数
        seq_relationship_scores = self.cls(pooled_output)

        # 如果 return_dict 为 False，则返回结果元组，包含预测分数和额外的隐藏状态列表
        if not return_dict:
            return (seq_relationship_scores,) + outputs[2:]

        # 如果 return_dict 为 True，则返回 FlaxNextSentencePredictorOutput 对象，包含预测 logits、隐藏状态和注意力
        return FlaxNextSentencePredictorOutput(
            logits=seq_relationship_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 给 FlaxBertForNextSentencePrediction 类添加文档字符串，描述其包含“下一句预测（分类）”头部的 BERT 模型
@add_start_docstrings(
    """Bert Model with a `next sentence prediction (classification)` head on top.""",
    BERT_START_DOCSTRING,
)
class FlaxBertForNextSentencePrediction(FlaxBertPreTrainedModel):
    # 模块类指向 FlaxBertForNextSentencePredictionModule
    module_class = FlaxBertForNextSentencePredictionModule


# FLAX_BERT_FOR_NEXT_SENT_PRED_DOCSTRING 包含详细的文档字符串，说明返回值和使用示例
FLAX_BERT_FOR_NEXT_SENT_PRED_DOCSTRING = """
    Returns:

    Example:

    ```
    >>> from transformers import AutoTokenizer, FlaxBertForNextSentencePrediction

    >>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    >>> model = FlaxBertForNextSentencePrediction.from_pretrained("google-bert/bert-base-uncased")

    >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
    >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
    >>> encoding = tokenizer(prompt, next_sentence, return_tensors="jax")

    >>> outputs = model(**encoding)
    >>> logits = outputs.logits
    >>> assert logits[0, 0] < logits[0, 1]  # next sentence was random
    ```
"""

# 将 BERT_INPUTS_DOCSTRING 格式化后的字符串和 FLAX_BERT_FOR_NEXT_SENT_PRED_DOCSTRING 添加到 FlaxBertForNextSentencePrediction 类的文档字符串中
overwrite_call_docstring(
    FlaxBertForNextSentencePrediction,
    BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length") + FLAX_BERT_FOR_NEXT_SENT_PRED_DOCSTRING,
)

# 附加和替换 FlaxBertForNextSentencePrediction 类的返回文档字符串，输出类型为 FlaxNextSentencePredictorOutput，配置类为 _CONFIG_FOR_DOC
append_replace_return_docstrings(
    FlaxBertForNextSentencePrediction, output_type=FlaxNextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC
)


class FlaxBertForSequenceClassificationModule(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        # 设置 BERT 模块，根据配置选择是否使用梯度检查点
        self.bert = FlaxBertModule(
            config=self.config,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 根据配置设置分类器的 dropout 率，若未指定，则使用隐藏层的 dropout 率
        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )
        # 设置 dropout 层
        self.dropout = nn.Dropout(rate=classifier_dropout)
        # 设置分类器，输出维度为配置中定义的标签数
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

        # 从 BERT 输出中获取池化后的特征表示
        pooled_output = outputs[1]
        # 对池化后的特征表示应用 dropout，以防止过拟合
        pooled_output = self.dropout(pooled_output, deterministic=deterministic)
        # 将池化后的特征表示输入分类器，得到 logits
        logits = self.classifier(pooled_output)

        # 如果不要求返回一个字典，则返回 logits 和额外的隐藏状态
        if not return_dict:
            return (logits,) + outputs[2:]

        # 如果要求返回一个字典，则封装输出为 FlaxSequenceClassifierOutput 类型
        return FlaxSequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """
    Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    BERT_START_DOCSTRING,
)
class FlaxBertForSequenceClassification(FlaxBertPreTrainedModel):
    module_class = FlaxBertForSequenceClassificationModule



append_call_sample_docstring(
    FlaxBertForSequenceClassification,
    _CHECKPOINT_FOR_DOC,
    FlaxSequenceClassifierOutput,
    _CONFIG_FOR_DOC,
)



class FlaxBertForMultipleChoiceModule(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        # 初始化 Bert 模型
        self.bert = FlaxBertModule(
            config=self.config,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # Dropout 层，用于随机丢弃输入特征
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        # 分类器，全连接层，用于多项选择任务
        self.classifier = nn.Dense(1, dtype=self.dtype)

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
        num_choices = input_ids.shape[1]
        # 重塑输入以适应 Bert 模型的期望形状
        input_ids = input_ids.reshape(-1, input_ids.shape[-1]) if input_ids is not None else None
        attention_mask = attention_mask.reshape(-1, attention_mask.shape[-1]) if attention_mask is not None else None
        token_type_ids = token_type_ids.reshape(-1, token_type_ids.shape[-1]) if token_type_ids is not None else None
        position_ids = position_ids.reshape(-1, position_ids.shape[-1]) if position_ids is not None else None

        # 调用 Bert 模型获取输出
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

        # 获取池化后的输出
        pooled_output = outputs[1]
        # 应用 Dropout 层
        pooled_output = self.dropout(pooled_output, deterministic=deterministic)
        # 应用分类器获取最终 logits
        logits = self.classifier(pooled_output)

        # 重塑 logits 以适应多项选择任务的形状
        reshaped_logits = logits.reshape(-1, num_choices)

        if not return_dict:
            # 如果不返回字典，则返回 logits 和额外的隐藏状态
            return (reshaped_logits,) + outputs[2:]

        # 如果返回字典，则构造输出对象
        return FlaxMultipleChoiceModelOutput(
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



@add_start_docstrings(
    """
    Bert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    BERT_START_DOCSTRING,
)
class FlaxBertForMultipleChoice(FlaxBertPreTrainedModel):
    module_class = FlaxBertForMultipleChoiceModule



overwrite_call_docstring(
    # 导入 FlaxBertForMultipleChoice 类
    FlaxBertForMultipleChoice, 
    # 使用 BERT_INPUTS_DOCSTRING 格式化字符串，描述输入参数的文档字符串
    BERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
# 调用append_call_sample_docstring函数，向FlaxBertForMultipleChoice类中添加示例文档字符串
append_call_sample_docstring(
    FlaxBertForMultipleChoice, _CHECKPOINT_FOR_DOC, FlaxMultipleChoiceModelOutput, _CONFIG_FOR_DOC
)


class FlaxBertForTokenClassificationModule(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        # 初始化self.bert作为FlaxBertModule实例，配置参数来自self.config，并设置相关参数
        self.bert = FlaxBertModule(
            config=self.config,
            dtype=self.dtype,
            add_pooling_layer=False,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 设置分类器的dropout率，若未指定则使用self.config.hidden_dropout_prob的值
        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )
        # 初始化self.dropout作为nn.Dropout实例，设定dropout率为classifier_dropout
        self.dropout = nn.Dropout(rate=classifier_dropout)
        # 初始化self.classifier作为nn.Dense实例，设定输出维度为self.config.num_labels
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
        # 调用self.bert，传入参数并返回输出
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

        # 获取BERT模型的隐藏状态
        hidden_states = outputs[0]
        # 对隐藏状态应用dropout操作，根据deterministic参数决定是否使用确定性dropout
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 对dropout后的隐藏状态进行分类预测，生成logits
        logits = self.classifier(hidden_states)

        # 如果return_dict为False，返回(logits,) + outputs[1:]
        if not return_dict:
            return (logits,) + outputs[1:]

        # 如果return_dict为True，返回FlaxTokenClassifierOutput对象，包括logits、隐藏状态和注意力机制
        return FlaxTokenClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Bert模型，在隐藏状态输出上增加了一个token分类头部（即隐藏状态输出之上的线性层），用于例如命名实体识别（NER）任务。
    """,
    BERT_START_DOCSTRING,
)
class FlaxBertForTokenClassification(FlaxBertPreTrainedModel):
    # 指定模块类为FlaxBertForTokenClassificationModule
    module_class = FlaxBertForTokenClassificationModule


# 调用append_call_sample_docstring函数，向FlaxBertForTokenClassification类中添加示例文档字符串
append_call_sample_docstring(
    FlaxBertForTokenClassification, _CHECKPOINT_FOR_DOC, FlaxTokenClassifierOutput, _CONFIG_FOR_DOC
)


class FlaxBertForQuestionAnsweringModule(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        # 初始化self.bert作为FlaxBertModule实例，配置参数来自self.config，并设置相关参数
        self.bert = FlaxBertModule(
            config=self.config,
            dtype=self.dtype,
            add_pooling_layer=False,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 初始化self.qa_outputs作为nn.Dense实例，设定输出维度为self.config.num_labels
        self.qa_outputs = nn.Dense(self.config.num_labels, dtype=self.dtype)
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
        """
        调用模型的方法，用于执行前向推断。

        Args:
            input_ids: 输入的token ID序列
            attention_mask: 注意力掩码，标识每个token的重要性
            token_type_ids: token类型ID，用于区分句子A和句子B等信息
            position_ids: 位置ID，指示每个token在输入序列中的位置
            head_mask: 头部掩码，控制每个注意力头的重要性
            deterministic: 是否以确定性方式运行（默认为True）
            output_attentions: 是否输出注意力权重（默认为False）
            output_hidden_states: 是否输出所有隐藏状态（默认为False）
            return_dict: 是否返回字典形式的输出（默认为True）

        Returns:
            FlaxQuestionAnsweringModelOutput 或 tuple:
                如果return_dict为True，则返回FlaxQuestionAnsweringModelOutput对象，
                包含起始和结束logits、隐藏状态和注意力权重等信息；
                如果return_dict为False，则返回元组，包含起始和结束logits以及额外的输出。
        """
        # 调用BERT模型的前向传播，获取模型输出
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

        # 从模型输出中提取隐藏状态
        hidden_states = outputs[0]

        # 将隐藏状态传入问答输出层，获取起始和结束logits
        logits = self.qa_outputs(hidden_states)
        
        # 根据问题答案的数量将logits进行分割
        start_logits, end_logits = jnp.split(logits, self.config.num_labels, axis=-1)
        
        # 去除最后一维的数据
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # 如果不需要以字典的形式返回结果，则返回元组
        if not return_dict:
            return (start_logits, end_logits) + outputs[1:]

        # 以FlaxQuestionAnsweringModelOutput对象的形式返回结果
        return FlaxQuestionAnsweringModelOutput(
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """
    Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    BERT_START_DOCSTRING,
)
class FlaxBertForQuestionAnswering(FlaxBertPreTrainedModel):
    # 将 Bert 模型与用于抽取式问答任务的跨度分类头部结合在一起，例如在 SQuAD 上操作（在隐藏状态输出之上的线性层，
    # 用于计算 `span start logits` 和 `span end logits`）。
    module_class = FlaxBertForQuestionAnsweringModule


append_call_sample_docstring(
    FlaxBertForQuestionAnswering,
    _CHECKPOINT_FOR_DOC,
    FlaxQuestionAnsweringModelOutput,
    _CONFIG_FOR_DOC,
)


class FlaxBertForCausalLMModule(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        # 初始化方法，设置模块中的组件
        self.bert = FlaxBertModule(
            config=self.config,
            add_pooling_layer=False,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        self.cls = FlaxBertOnlyMLMHead(config=self.config, dtype=self.dtype)

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
        # 模型调用方法
        # 调用内部的 Bert 模块进行前向传播
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

        hidden_states = outputs[0]
        if self.config.tie_word_embeddings:
            # 如果配置要求共享词嵌入，获取共享的嵌入参数
            shared_embedding = self.bert.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
        else:
            shared_embedding = None

        # 计算预测分数
        logits = self.cls(hidden_states, shared_embedding=shared_embedding)

        if not return_dict:
            return (logits,) + outputs[1:]

        # 返回带有交叉注意力的 FlaxCausalLMOutputWithCrossAttentions 对象
        return FlaxCausalLMOutputWithCrossAttentions(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


@add_start_docstrings(
    """
    Bert Model with a language modeling head on top (a linear layer on top of the hidden-states output) e.g for
    autoregressive tasks.
    """,
    BERT_START_DOCSTRING,
)
class FlaxBertForCausalLM(FlaxBertPreTrainedModel):
    # 将 Bert 模型与用于语言建模任务的头部结合在一起（在隐藏状态输出之上的线性层），例如自回归任务。
    # 设置模块类为 FlaxBertForCausalLMModule
    module_class = FlaxBertForCausalLMModule
    
    # 为生成器准备输入的函数定义，接受输入的token ids、最大长度、可选的注意力掩码
    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = None):
        # 初始化缓存
        batch_size, seq_length = input_ids.shape
    
        # 使用模型自定义方法初始化缓存，返回过去的键值对
        past_key_values = self.init_cache(batch_size, max_length)
    
        # 注意，在通常情况下需要在 attention_mask 的 x > input_ids.shape[-1] 和 x < cache_length 的位置放置 0。
        # 但由于解码器使用因果注意力掩码，这些位置已经被掩盖了。
        # 因此，我们可以在这里创建一个静态的注意力掩码，这样更有效地进行编译
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
    
        # 如果有传入注意力掩码，则根据它计算位置 ids，并动态更新静态的注意力掩码
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            # 如果没有传入注意力掩码，则生成位置 ids，用于模型输入
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))
    
        # 返回包含 past_key_values、extended_attention_mask 和 position_ids 的字典
        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }
    
    # 更新生成器输入的函数定义，接受模型输出和模型参数关键字作为输入
    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        # 更新模型参数关键字中的 past_key_values 和 position_ids
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs
# 调用函数 append_call_sample_docstring，添加示例文档字符串到 FlaxBertForCausalLM 类中
append_call_sample_docstring(
    FlaxBertForCausalLM,
    # 示例文档字符串的检查点
    _CHECKPOINT_FOR_DOC,
    # 使用交叉注意力的 FlaxCausalLMOutputWithCrossAttentions 类
    FlaxCausalLMOutputWithCrossAttentions,
    # 示例文档字符串的配置
    _CONFIG_FOR_DOC,
)
```