# `.\models\electra\modeling_flax_electra.py`

```py
# 引入必要的库和模块
from typing import Callable, Optional, Tuple  # 导入类型提示相关的模块

import flax  # 导入 Flax 深度学习库
import flax.linen as nn  # 导入 Flax 中的线性模块
import jax  # 导入 JAX，用于自动求导和并行计算
import jax.numpy as jnp  # 导入 JAX 的 NumPy 接口，用于数组操作
import numpy as np  # 导入 NumPy，用于常规的数学运算
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze  # 导入 Flax 中的冻结字典相关模块
from flax.linen import combine_masks, make_causal_mask  # 导入 Flax 中的掩码组合和因果掩码生成函数
from flax.linen import partitioning as nn_partitioning  # 导入 Flax 中的模块分割工具
from flax.linen.attention import dot_product_attention_weights  # 导入 Flax 中的点积注意力权重计算函数
from flax.traverse_util import flatten_dict, unflatten_dict  # 导入 Flax 中的字典扁平化和反扁平化工具
from jax import lax  # 导入 JAX 的低级别 API

# 导入输出类和实用函数
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
    ACT2FN,  # 导入激活函数映射
    FlaxPreTrainedModel,  # 导入 Flax 预训练模型基类
    append_call_sample_docstring,  # 导入用于追加调用示例文档字符串的函数
    append_replace_return_docstrings,  # 导入用于追加替换返回值文档字符串的函数
    overwrite_call_docstring,  # 导入用于覆盖调用文档字符串的函数
)
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging  # 导入模型输出、文档字符串添加和日志记录工具
from .configuration_electra import ElectraConfig  # 导入 Electra 配置类

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器实例

_CHECKPOINT_FOR_DOC = "google/electra-small-discriminator"  # 预训练模型的检查点路径
_CONFIG_FOR_DOC = "ElectraConfig"  # Electra 模型的配置类名称

remat = nn_partitioning.remat  # 设置 remat 变量为模块分割的重组矩阵操作

@flax.struct.dataclass
class FlaxElectraForPreTrainingOutput(ModelOutput):
    """
    [`ElectraForPreTraining`] 的输出类型。
    """
    # 此类用于定义 Electra 模型预训练的输出结构
    Args:
        logits (`jnp.ndarray` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        
        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # 定义 logits 变量，类型为 jnp.ndarray，形状为 (batch_size, sequence_length, config.vocab_size)
    logits: jnp.ndarray = None
    
    # 定义 hidden_states 变量，类型为 Optional[Tuple[jnp.ndarray]]，可选参数，当 `output_hidden_states=True` 时返回
    # 返回一个元组，包含 jnp.ndarray 类型的张量，形状为 (batch_size, sequence_length, hidden_size)
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    
    # 定义 attentions 变量，类型为 Optional[Tuple[jnp.ndarray]]，可选参数，当 `output_attentions=True` 时返回
    # 返回一个元组，包含 jnp.ndarray 类型的张量，形状为 (batch_size, num_heads, sequence_length, sequence_length)
    # 表示注意力权重经过 softmax 后的结果，用于计算自注意力头部中的加权平均值。
    attentions: Optional[Tuple[jnp.ndarray]] = None
# 定义模型的文档字符串，描述该模型从 FlaxPreTrainedModel 继承，并列出了库为所有模型实现的通用方法（如下载、保存和从 PyTorch 模型转换权重）
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

# 定义模型输入的文档字符串，目前为空白
ELECTRA_INPUTS_DOCSTRING = r"""
    # 将输入的各项参数打包成一个参数字典，用于传递给模型的前向推断函数
    Args:
        input_ids (`numpy.ndarray` of shape `({0})`):
            输入序列标记在词汇表中的索引。
    
            可以使用 [`AutoTokenizer`] 获取这些索引。详情见 [`PreTrainedTokenizer.encode`] 和
            [`PreTrainedTokenizer.__call__`]。
    
            [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`numpy.ndarray` of shape `({0})`, *optional*):
            避免对填充标记索引执行注意力的掩码。掩码值为 `[0, 1]`：
    
            - 1 表示**不屏蔽**的标记，
            - 0 表示**屏蔽**的标记。
    
            [什么是注意力掩码？](../glossary#attention-mask)
        token_type_ids (`numpy.ndarray` of shape `({0})`, *optional*):
            段标记索引，指示输入的第一部分和第二部分。索引值为 `[0, 1]`：
    
            - 0 对应*句子 A* 的标记，
            - 1 对应*句子 B* 的标记。
    
            [什么是标记类型 ID？](../glossary#token-type-ids)
        position_ids (`numpy.ndarray` of shape `({0})`, *optional*):
            每个输入序列标记在位置嵌入中的位置索引。选择范围为 `[0, config.max_position_embeddings - 1]`。
        head_mask (`numpy.ndarray` of shape `({0})`, `optional):
            选择性屏蔽注意力模块中的头部的掩码。掩码值为 `[0, 1]`：
    
            - 1 表示**不屏蔽**的头部，
            - 0 表示**屏蔽**的头部。
        return_dict (`bool`, *optional*):
            是否返回一个 [`~utils.ModelOutput`] 而不是普通的元组。
"""
定义一个名为 FlaxElectraEmbeddings 的 nn.Module 类，用于构建包括单词、位置和标记类型嵌入的 embeddings。

config: ElectraConfig
    # 保存了 Electra 模型的配置信息，如词汇大小、嵌入维度等

dtype: jnp.dtype = jnp.float32
    # 计算时使用的数据类型，默认为 jnp.float32

setup(self):
    # 初始化模型的各个组件

    self.word_embeddings = nn.Embed(
        self.config.vocab_size,
        self.config.embedding_size,
        embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
    )
    # 创建单词嵌入层，根据词汇大小和嵌入维度进行初始化

    self.position_embeddings = nn.Embed(
        self.config.max_position_embeddings,
        self.config.embedding_size,
        embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
    )
    # 创建位置嵌入层，根据最大位置嵌入数和嵌入维度进行初始化

    self.token_type_embeddings = nn.Embed(
        self.config.type_vocab_size,
        self.config.embedding_size,
        embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
    )
    # 创建标记类型嵌入层，根据标记类型的数量和嵌入维度进行初始化

    self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
    # 创建 Layer Normalization 层，使用给定的 epsilon 参数进行初始化

    self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
    # 创建 Dropout 层，使用给定的 dropout 概率进行初始化

__call__(self, input_ids, token_type_ids, position_ids, attention_mask, deterministic: bool = True):
    # 定义 __call__ 方法，实现模块的调用功能，接受输入参数并进行处理

    inputs_embeds = self.word_embeddings(input_ids.astype("i4"))
    # 将输入的词汇 ID 转换为单词嵌入

    position_embeds = self.position_embeddings(position_ids.astype("i4"))
    # 将位置 ID 转换为位置嵌入

    token_type_embeddings = self.token_type_embeddings(token_type_ids.astype("i4"))
    # 将标记类型 ID 转换为标记类型嵌入

    hidden_states = inputs_embeds + token_type_embeddings + position_embeds
    # 将单词、位置和标记类型嵌入求和，形成最终的隐藏状态表示

    hidden_states = self.LayerNorm(hidden_states)
    # 对隐藏状态进行 Layer Normalization 处理

    hidden_states = self.dropout(hidden_states, deterministic=deterministic)
    # 对处理后的隐藏状态进行 Dropout 操作

    return hidden_states
    # 返回处理后的最终隐藏状态
"""

# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertSelfAttention with Bert->Electra
class FlaxElectraSelfAttention(nn.Module):
    config: ElectraConfig
    causal: bool = False
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    # 设置函数用于初始化模型的配置
    def setup(self):
        # 计算每个注意力头的维度
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        # 如果隐藏层大小不能被注意力头数整除，抛出数值错误异常
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError(
                "`config.hidden_size`: {self.config.hidden_size} has to be a multiple of `config.num_attention_heads` "
                "                   : {self.config.num_attention_heads}"
            )

        # 初始化查询（query）的全连接层
        self.query = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        # 初始化键（key）的全连接层
        self.key = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        # 初始化值（value）的全连接层
        self.value = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

        # 如果是因果注意力模型，创建一个因果掩码
        if self.causal:
            self.causal_mask = make_causal_mask(
                jnp.ones((1, self.config.max_position_embeddings), dtype="bool"), dtype="bool"
            )

    # 函数用于将隐藏状态按照注意力头的数量进行分割
    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.config.num_attention_heads, self.head_dim))

    # 函数用于将按注意力头分割后的隐藏状态重新合并
    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.config.hidden_size,))

    @nn.compact
    # 从 transformers.models.bart.modeling_flax_bart.FlaxBartAttention._concatenate_to_cache 复制而来的函数装饰器
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slightly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # 检测是否通过缺少现有缓存数据来初始化。
        is_initialized = self.has_variable("cache", "cached_key")
        # 获取或创建缓存的键，若不存在则初始化为零张量，形状与传入的键相同
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        # 获取或创建缓存的值，若不存在则初始化为零张量，形状与传入的值相同
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        # 获取或创建缓存索引，若不存在则初始化为整数零
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            # 获取当前缓存的维度信息，即批次维度、最大长度、注意力头数和每个头的深度
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # 使用新的一维空间片段更新键、值缓存
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            # 更新缓存的键和值
            cached_key.value = key
            cached_value.value = value
            # 更新缓存索引，增加已更新的缓存向量数目
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # 生成因果掩码用于缓存的解码器自注意力：
            # 我们的单个查询位置只应该参与到已经生成和缓存的键位置，而不是剩余的零元素。
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            # 合并掩码，结合当前的注意力掩码
            attention_mask = combine_masks(pad_mask, attention_mask)
        # 返回更新后的键、值和注意力掩码
        return key, value, attention_mask
# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertSelfOutput with Bert->Electra
class FlaxElectraSelfOutput(nn.Module):
    config: ElectraConfig  # Electra模型的配置对象
    dtype: jnp.dtype = jnp.float32  # 计算时的数据类型

    def setup(self):
        # 创建一个全连接层，输出维度为配置中的hidden_size，权重初始化为正态分布
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 创建一个LayerNorm层，epsilon值为配置中的layer_norm_eps
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 创建一个Dropout层，dropout率为配置中的hidden_dropout_prob
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, hidden_states, input_tensor, deterministic: bool = True):
        # 输入经过全连接层
        hidden_states = self.dense(hidden_states)
        # 经过Dropout层，用于随机置零一部分神经元的输出
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 经过LayerNorm层，并加上输入张量，实现残差连接和层归一化
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertAttention with Bert->Electra
class FlaxElectraAttention(nn.Module):
    config: ElectraConfig  # Electra模型的配置对象
    causal: bool = False  # 是否是因果注意力
    dtype: jnp.dtype = jnp.float32  # 计算时的数据类型

    def setup(self):
        # 创建一个FlaxElectraSelfAttention对象，用于自注意力计算
        self.self = FlaxElectraSelfAttention(self.config, causal=self.causal, dtype=self.dtype)
        # 创建一个FlaxElectraSelfOutput对象，用于自注意力的输出处理

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
        # 注意力掩码的形状应为(*batch_sizes, kv_length)
        # FLAX期望的形状为(*batch_sizes, 1, 1, kv_length)，使其能广播
        # 注意力计算的输出包含在attn_outputs中
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
        # 使用self.output对自注意力的输出进行处理，实现残差连接和层归一化
        hidden_states = self.output(attn_output, hidden_states, deterministic=deterministic)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_outputs[1],)

        return outputs


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertIntermediate with Bert->Electra
class FlaxElectraIntermediate(nn.Module):
    config: ElectraConfig  # Electra模型的配置对象
    dtype: jnp.dtype = jnp.float32  # 计算时的数据类型

    def setup(self):
        # 创建一个全连接层，输出维度为配置中的intermediate_size，权重初始化为正态分布
        self.dense = nn.Dense(
            self.config.intermediate_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 选择一个激活函数，根据配置中的hidden_act确定
        self.activation = ACT2FN[self.config.hidden_act]
    # 定义一个特殊方法 __call__()，用于将对象实例像函数一样调用
    def __call__(self, hidden_states):
        # 使用 self.dense 对象处理 hidden_states，进行线性变换
        hidden_states = self.dense(hidden_states)
        # 使用 self.activation 对象处理 hidden_states，应用激活函数
        hidden_states = self.activation(hidden_states)
        # 返回处理后的 hidden_states
        return hidden_states
# Copied from transformers.models.electra.modeling_flax_electra.FlaxElectraOutput with Bert->Electra
class FlaxElectraOutput(nn.Module):
    config: ElectraConfig
    dtype: jnp.dtype = jnp.float32  # 计算时使用的数据类型

    def setup(self):
        # 密集连接层，用于将隐藏状态转换到指定大小的输出空间
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 随机失活层，以一定的概率丢弃隐藏状态中的部分数据，防止过拟合
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        # 层归一化，用于对输入数据进行归一化处理
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    def __call__(self, hidden_states, attention_output, deterministic: bool = True):
        # 密集连接层计算，将隐藏状态映射到指定大小的输出空间
        hidden_states = self.dense(hidden_states)
        # 随机失活操作，根据设定的概率丢弃部分数据，用于防止过拟合
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 层归一化操作，将处理后的数据进行归一化
        hidden_states = self.LayerNorm(hidden_states + attention_output)
        return hidden_states


# Copied from transformers.models.electra.modeling_flax_electra.FlaxElectraLayer with Bert->Electra
class FlaxElectraLayer(nn.Module):
    config: ElectraConfig
    dtype: jnp.dtype = jnp.float32  # 计算时使用的数据类型

    def setup(self):
        # Electra 自注意力层，根据配置初始化
        self.attention = FlaxElectraAttention(self.config, causal=self.config.is_decoder, dtype=self.dtype)
        # Electra 中间层，根据配置初始化
        self.intermediate = FlaxElectraIntermediate(self.config, dtype=self.dtype)
        # Electra 输出层，根据配置初始化
        self.output = FlaxElectraOutput(self.config, dtype=self.dtype)
        # 如果配置中包含跨注意力机制，初始化跨注意力层
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
        # Electra 自注意力计算，处理隐藏状态和注意力掩码
        attention_output = self.attention(
            hidden_states,
            attention_mask,
            layer_head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            init_cache=init_cache,
            deterministic=deterministic,
            output_attentions=output_attentions,
        )
        # Electra 中间层计算，处理注意力输出
        intermediate_output = self.intermediate(attention_output)
        # Electra 输出层计算，处理中间层输出和注意力输出
        layer_output = self.output(intermediate_output, attention_output, deterministic=deterministic)
        
        if self.config.add_cross_attention:
            # 如果配置中包含跨注意力机制，计算跨注意力
            attention_output = self.crossattention(
                layer_output,
                encoder_attention_mask,
                encoder_hidden_states,
                layer_head_mask=None,
                init_cache=init_cache,
                deterministic=deterministic,
                output_attentions=output_attentions,
            )
        
        return (layer_output, attention_output) if output_attentions else layer_output
        # Self Attention
        # 使用 self.attention 方法对输入的 hidden_states 进行自注意力计算
        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            layer_head_mask=layer_head_mask,
            init_cache=init_cache,
            deterministic=deterministic,
            output_attentions=output_attentions,
        )
        # 获取自注意力计算后的输出
        attention_output = attention_outputs[0]

        # Cross-Attention Block
        # 如果 encoder_hidden_states 不为空，则进行交叉注意力计算
        if encoder_hidden_states is not None:
            # 使用 self.crossattention 方法进行交叉注意力计算
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask=encoder_attention_mask,
                layer_head_mask=layer_head_mask,
                key_value_states=encoder_hidden_states,
                deterministic=deterministic,
                output_attentions=output_attentions,
            )
            # 获取交叉注意力计算后的输出
            attention_output = cross_attention_outputs[0]

        # 经过注意力计算后，通过 self.intermediate 方法处理中间隐藏层输出
        hidden_states = self.intermediate(attention_output)
        # 使用 self.output 方法根据注意力输出和隐藏层输出生成最终输出
        hidden_states = self.output(hidden_states, attention_output, deterministic=deterministic)

        # 将最终输出存入元组 outputs 中
        outputs = (hidden_states,)

        # 如果需要输出注意力权重信息
        if output_attentions:
            # 将自注意力的注意力权重信息添加到 outputs 中
            outputs += (attention_outputs[1],)
            # 如果存在 encoder_hidden_states，则将交叉注意力的注意力权重信息也添加到 outputs 中
            if encoder_hidden_states is not None:
                outputs += (cross_attention_outputs[1],)
        
        # 返回最终的输出元组
        return outputs
# 从 transformers.models.bert.modeling_flax_bert.FlaxBertLayerCollection 复制代码并将 Bert 替换为 Electra
class FlaxElectraLayerCollection(nn.Module):
    config: ElectraConfig  # 类属性，指定模型配置为 ElectraConfig 类型
    dtype: jnp.dtype = jnp.float32  # 计算过程中使用的数据类型，默认为 jnp.float32
    gradient_checkpointing: bool = False  # 是否开启梯度检查点，默认为 False

    def setup(self):
        # 如果开启了梯度检查点
        if self.gradient_checkpointing:
            # 使用 remat 函数包装 FlaxElectraLayer 类，指定静态参数索引为 (5, 6, 7)
            FlaxElectraCheckpointLayer = remat(FlaxElectraLayer, static_argnums=(5, 6, 7))
            # 创建 self.layers 列表，包含 num_hidden_layers 个 FlaxElectraCheckpointLayer 实例
            self.layers = [
                FlaxElectraCheckpointLayer(self.config, name=str(i), dtype=self.dtype)
                for i in range(self.config.num_hidden_layers)
            ]
        else:
            # 创建 self.layers 列表，包含 num_hidden_layers 个 FlaxElectraLayer 实例
            self.layers = [
                FlaxElectraLayer(self.config, name=str(i), dtype=self.dtype)
                for i in range(self.config.num_hidden_layers)
            ]

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
            # 如果不需要输出注意力权重信息，则初始化为空元组；否则置为 None
            all_attentions = () if output_attentions else None
            # 如果不需要输出隐藏状态信息，则初始化为空元组；否则置为 None
            all_hidden_states = () if output_hidden_states else None
            # 如果不需要输出交叉注意力权重信息，则初始化为空元组；否则置为 None
            all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

            # 检查头部掩码的层数是否正确
            if head_mask is not None:
                if head_mask.shape[0] != (len(self.layers)):
                    raise ValueError(
                        f"The head_mask should be specified for {len(self.layers)} layers, but it is for                  "
                        f"       {head_mask.shape[0]}."
                    )

            # 遍历每一层神经网络层
            for i, layer in enumerate(self.layers):
                # 如果需要输出隐藏状态信息，则记录当前层的隐藏状态
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

                # 更新隐藏状态为当前层的输出
                hidden_states = layer_outputs[0]

                # 如果需要输出注意力权重信息，则记录当前层的注意力权重
                if output_attentions:
                    all_attentions += (layer_outputs[1],)

                    # 如果有编码器隐藏状态，则记录当前层的交叉注意力权重
                    if encoder_hidden_states is not None:
                        all_cross_attentions += (layer_outputs[2],)

            # 如果需要输出隐藏状态信息，则记录最后一层的隐藏状态
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # 组装最终的输出元组
            outputs = (hidden_states, all_hidden_states, all_attentions, all_cross_attentions)

            # 如果不需要返回字典格式的输出，则返回元组中非空的部分
            if not return_dict:
                return tuple(v for v in outputs if v is not None)

            # 返回字典格式的输出，使用特定的输出类
            return FlaxBaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=all_attentions,
                cross_attentions=all_cross_attentions,
            )
# 从 transformers.models.bert.modeling_flax_bert.FlaxBertEncoder 复制并修改为使用 Electra 模型
class FlaxElectraEncoder(nn.Module):
    config: ElectraConfig
    dtype: jnp.dtype = jnp.float32  # 计算过程中使用的数据类型
    gradient_checkpointing: bool = False  # 是否使用梯度检查点技术

    def setup(self):
        # 初始化 Electra 编码器层集合
        self.layer = FlaxElectraLayerCollection(
            self.config,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )

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
        # 调用 Electra 编码器层集合来处理输入
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
    config: ElectraConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 初始化 Electra 生成器预测层的 LayerNorm 和 Dense 层
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.dense = nn.Dense(self.config.embedding_size, dtype=self.dtype)

    def __call__(self, hidden_states):
        # 执行 Electra 生成器预测过程：Dense -> 激活函数 -> LayerNorm
        hidden_states = self.dense(hidden_states)
        hidden_states = ACT2FN[self.config.hidden_act](hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class FlaxElectraDiscriminatorPredictions(nn.Module):
    """用于鉴别器的预测模块，由两个密集层组成。"""

    config: ElectraConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 初始化 Electra 鉴别器预测层的 Dense 层和 Dense 预测层
        self.dense = nn.Dense(self.config.hidden_size, dtype=self.dtype)
        self.dense_prediction = nn.Dense(1, dtype=self.dtype)

    def __call__(self, hidden_states):
        # 执行 Electra 鉴别器预测过程：Dense -> 激活函数 -> Dense 预测层
        hidden_states = self.dense(hidden_states)
        hidden_states = ACT2FN[self.config.hidden_act](hidden_states)
        hidden_states = self.dense_prediction(hidden_states).squeeze(-1)
        return hidden_states


class FlaxElectraPreTrainedModel(FlaxPreTrainedModel):
    """
    处理权重初始化和一个简单接口以下载和加载预训练模型的抽象类。
    """

    config_class = ElectraConfig
    base_model_prefix = "electra"
    module_class: nn.Module = None
    # 初始化方法，用于实例化一个新的对象
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
        # 使用传入的配置和参数初始化模块对象
        module = self.module_class(config=config, dtype=dtype, gradient_checkpointing=gradient_checkpointing, **kwargs)
        # 调用父类的初始化方法，传入配置、模块对象以及其他参数
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    # 从transformers库中复制而来，用于启用梯度检查点
    def enable_gradient_checkpointing(self):
        # 根据当前对象的配置和数据类型，启用模块的梯度检查点功能
        self._module = self.module_class(
            config=self.config,
            dtype=self.dtype,
            gradient_checkpointing=True,
        )

    # 从transformers库中复制而来，用于初始化模型的权重
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入的张量
        input_ids = jnp.zeros(input_shape, dtype="i4")
        token_type_ids = jnp.zeros_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        attention_mask = jnp.ones_like(input_ids)
        head_mask = jnp.ones((self.config.num_hidden_layers, self.config.num_attention_heads))

        # 拆分随机数生成器用于参数和dropout
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        if self.config.add_cross_attention:
            encoder_hidden_states = jnp.zeros(input_shape + (self.config.hidden_size,))
            encoder_attention_mask = attention_mask
            # 使用模块的初始化方法初始化模型的各种参数
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
            # 使用模块的初始化方法初始化模型的各种参数（无交叉注意力的情况）
            module_init_outputs = self.module.init(
                rngs, input_ids, attention_mask, token_type_ids, position_ids, head_mask, return_dict=False
            )

        # 获取随机初始化的参数
        random_params = module_init_outputs["params"]

        # 如果提供了预定义的参数，将缺失的参数补充进去
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params
    # 初始化缓存的方法，用于快速自回归解码
    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
        """
        # 初始化输入变量以检索缓存
        input_ids = jnp.ones((batch_size, max_length), dtype="i4")
        attention_mask = jnp.ones_like(input_ids, dtype="i4")
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        # 使用模型的初始化方法初始化变量，包括输入的 ID、注意力掩码、位置 ID，同时设置返回字典为 False 并初始化缓存
        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        # 返回解冻后的缓存部分
        return unfreeze(init_variables["cache"])

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
    # 配置属性，使用 ElectraConfig 类型
    config: ElectraConfig
    # 计算中使用的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 是否使用梯度检查点
    gradient_checkpointing: bool = False

    # 模块的初始化方法
    def setup(self):
        # 创建 FlaxElectraEmbeddings 实例，使用给定的配置和数据类型
        self.embeddings = FlaxElectraEmbeddings(self.config, dtype=self.dtype)
        # 如果嵌入维度不等于隐藏层维度，创建 Dense 层进行投影
        if self.config.embedding_size != self.config.hidden_size:
            self.embeddings_project = nn.Dense(self.config.hidden_size, dtype=self.dtype)
        # 创建 FlaxElectraEncoder 实例，使用给定的配置、数据类型和梯度检查点标志
        self.encoder = FlaxElectraEncoder(
            self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )

    # 实现调用模块时的行为
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
        # 调用 embeddings 方法生成嵌入向量
        embeddings = self.embeddings(
            input_ids, token_type_ids, position_ids, attention_mask, deterministic=deterministic
        )
        # 如果存在 embeddings_project 属性，对 embeddings 进行投影
        if hasattr(self, "embeddings_project"):
            embeddings = self.embeddings_project(embeddings)

        # 调用 encoder 方法对 embeddings 进行编码
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


# 添加文档字符串说明的装饰器，说明 FlaxElectraModel 是基于 FlaxElectraPreTrainedModel 的模型
@add_start_docstrings(
    "The bare Electra Model transformer outputting raw hidden-states without any specific head on top.",
    ELECTRA_START_DOCSTRING,
)
# 定义 FlaxElectraModel 类，继承自 FlaxElectraPreTrainedModel
class FlaxElectraModel(FlaxElectraPreTrainedModel):
    # 模块类设置为 FlaxElectraModule
    module_class = FlaxElectraModule


# 向 FlaxElectraModel 添加调用样本文档字符串的函数说明
append_call_sample_docstring(FlaxElectraModel, _CHECKPOINT_FOR_DOC, FlaxBaseModelOutput, _CONFIG_FOR_DOC)


# 定义 FlaxElectraTiedDense 类，继承自 nn.Module
class FlaxElectraTiedDense(nn.Module):
    # 嵌入大小属性
    embedding_size: int
    # 数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 精度设置，默认为 None
    precision = None
    # 偏置初始化函数，默认为全零初始化
    bias_init: Callable[..., np.ndarray] = jax.nn.initializers.zeros

    # 模块的初始化方法
    def setup(self):
        # 创建偏置参数，形状为 (embedding_size,)
        self.bias = self.param("bias", self.bias_init, (self.embedding_size,))

    # 实现调用模块时的行为
    def __call__(self, x, kernel):
        # 将输入 x 和 kernel 转换为指定数据类型的 jnp 数组
        x = jnp.asarray(x, self.dtype)
        kernel = jnp.asarray(kernel, self.dtype)
        # 使用 dot_general 函数进行矩阵乘法运算，加上偏置项
        y = lax.dot_general(
            x,
            kernel,
            (((x.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )
        # 将偏置转换为指定数据类型的 jnp 数组后，返回 y 加上 bias 的结果
        bias = jnp.asarray(self.bias, self.dtype)
        return y + bias


# 定义 FlaxElectraForMaskedLMModule 类，继承自 nn.Module
class FlaxElectraForMaskedLMModule(nn.Module):
    # 配置属性，使用 ElectraConfig 类型
    config: ElectraConfig
    # 数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 是否使用梯度检查点
    gradient_checkpointing: bool = False
    # 初始化模型设置，在对象的实例化过程中被调用
    def setup(self):
        # 初始化 Electra 模型模块，使用给定的配置、数据类型和梯度检查点设置
        self.electra = FlaxElectraModule(
            config=self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )
        # 初始化生成器预测模块，使用给定的配置和数据类型
        self.generator_predictions = FlaxElectraGeneratorPredictions(config=self.config, dtype=self.dtype)
        # 如果配置要求共享词嵌入
        if self.config.tie_word_embeddings:
            # 使用 Electra 模型的共享词嵌入初始化生成器 LM 头
            self.generator_lm_head = FlaxElectraTiedDense(self.config.vocab_size, dtype=self.dtype)
        else:
            # 否则，初始化一个普通的全连接层作为生成器 LM 头
            self.generator_lm_head = nn.Dense(self.config.vocab_size, dtype=self.dtype)

    # 在对象被调用时执行的方法，处理输入并生成预测结果
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
        # 将输入传递给 Electra 模型，获取模型输出
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
        # 从模型输出中提取隐藏状态
        hidden_states = outputs[0]
        # 使用生成器预测模块生成预测分数
        prediction_scores = self.generator_predictions(hidden_states)

        # 如果配置要求共享词嵌入
        if self.config.tie_word_embeddings:
            # 获取 Electra 模型的共享词嵌入
            shared_embedding = self.electra.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
            # 使用共享词嵌入调整生成器 LM 头的预测分数
            prediction_scores = self.generator_lm_head(prediction_scores, shared_embedding.T)
        else:
            # 否则，直接使用生成器 LM 头生成预测分数
            prediction_scores = self.generator_lm_head(prediction_scores)

        # 如果不需要返回字典
        if not return_dict:
            # 返回预测分数和其它输出
            return (prediction_scores,) + outputs[1:]

        # 返回封装了预测分数、隐藏状态和注意力的 MaskedLMOutput 对象
        return FlaxMaskedLMOutput(
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings("""Electra Model with a `language modeling` head on top.""", ELECTRA_START_DOCSTRING)
# 使用装饰器添加模型的文档字符串，指明这是一个在语言建模头部的Electra模型

class FlaxElectraForMaskedLM(FlaxElectraPreTrainedModel):
    module_class = FlaxElectraForMaskedLMModule

# 定义一个FlaxElectraForMaskedLM类，继承自FlaxElectraPreTrainedModel，并指定其模块类为FlaxElectraForMaskedLMModule

append_call_sample_docstring(FlaxElectraForMaskedLM, _CHECKPOINT_FOR_DOC, FlaxMaskedLMOutput, _CONFIG_FOR_DOC)

# 向FlaxElectraForMaskedLM类的__call__方法添加示例的文档字符串，展示了如何调用该模型的示例用法

class FlaxElectraForPreTrainingModule(nn.Module):
    config: ElectraConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        self.electra = FlaxElectraModule(
            config=self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )
        self.discriminator_predictions = FlaxElectraDiscriminatorPredictions(config=self.config, dtype=self.dtype)

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
        # Model
        # 调用self.electra模块进行模型计算
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
        hidden_states = outputs[0]

        # 使用self.discriminator_predictions预测生成的token
        logits = self.discriminator_predictions(hidden_states)

        if not return_dict:
            return (logits,) + outputs[1:]

        # 返回预训练输出对象FlaxElectraForPreTrainingOutput
        return FlaxElectraForPreTrainingOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Electra model with a binary classification head on top as used during pretraining for identifying generated tokens.

    It is recommended to load the discriminator checkpoint into that model.
    """,
    ELECTRA_START_DOCSTRING,
)
# 使用装饰器添加文档字符串，描述这是一个在预训练过程中用于识别生成token的Electra模型

class FlaxElectraForPreTraining(FlaxElectraPreTrainedModel):
    module_class = FlaxElectraForPreTrainingModule

# 定义一个FlaxElectraForPreTraining类，继承自FlaxElectraPreTrainedModel，并指定其模块类为FlaxElectraForPreTrainingModule

FLAX_ELECTRA_FOR_PRETRAINING_DOCSTRING = """
    Returns:

    Example:

    ```
    >>> from transformers import AutoTokenizer, FlaxElectraForPreTraining

    >>> tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator")
    >>> model = FlaxElectraForPreTraining.from_pretrained("google/electra-small-discriminator")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="np")
    >>> outputs = model(**inputs)

    >>> prediction_logits = outputs.logits
    ```
"""

overwrite_call_docstring(
    FlaxElectraForPreTraining,
    ELECTRA_INPUTS_DOCSTRING.format("batch_size, sequence_length") + FLAX_ELECTRA_FOR_PRETRAINING_DOCSTRING,
)
# 覆盖FlaxElectraForPreTraining类的__call__方法的文档字符串，展示模型的输入和输出示例用法
    # 导入FlaxElectraForPreTraining类和FlaxElectraForPreTrainingOutput类型
    # 使用_CONFIG_FOR_DOC指定的配置类
    FlaxElectraForPreTraining, output_type=FlaxElectraForPreTrainingOutput, config_class=_CONFIG_FOR_DOC
)


class FlaxElectraForTokenClassificationModule(nn.Module):
    config: ElectraConfig  # 类型注解，指定 config 属性的类型为 ElectraConfig
    dtype: jnp.dtype = jnp.float32  # 设置 dtype 属性，默认为 jnp.float32 类型
    gradient_checkpointing: bool = False  # 设置 gradient_checkpointing 属性，默认为 False

    def setup(self):
        self.electra = FlaxElectraModule(  # 初始化 electra 属性为 FlaxElectraModule 实例
            config=self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )
        classifier_dropout = (
            self.config.classifier_dropout  # 获取 config 对象的 classifier_dropout 属性
            if self.config.classifier_dropout is not None  # 如果其不为 None，则使用该值
            else self.config.hidden_dropout_prob  # 否则使用 hidden_dropout_prob 属性的值
        )
        self.dropout = nn.Dropout(classifier_dropout)  # 初始化 dropout 属性为 nn.Dropout 实例
        self.classifier = nn.Dense(self.config.num_labels, dtype=self.dtype)  # 初始化 classifier 属性为 nn.Dense 实例

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
        # Model
        outputs = self.electra(  # 调用 self.electra 进行模型计算
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
        hidden_states = outputs[0]  # 获取模型输出的第一个元素，即隐藏状态

        hidden_states = self.dropout(hidden_states, deterministic=deterministic)  # 对隐藏状态应用 dropout 操作
        logits = self.classifier(hidden_states)  # 将隐藏状态传递给分类器生成 logits

        if not return_dict:
            return (logits,) + outputs[1:]  # 如果 return_dict 为 False，则返回元组形式的结果

        return FlaxTokenClassifierOutput(  # 返回 FlaxTokenClassifierOutput 对象
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Electra model with a token classification head on top.

    Both the discriminator and generator may be loaded into this model.
    """,
    ELECTRA_START_DOCSTRING,  # 添加文档字符串，结合 ELECTRA_START_DOCSTRING 定义
)
class FlaxElectraForTokenClassification(FlaxElectraPreTrainedModel):
    module_class = FlaxElectraForTokenClassificationModule  # 设置模块类为 FlaxElectraForTokenClassificationModule


append_call_sample_docstring(
    FlaxElectraForTokenClassification,
    _CHECKPOINT_FOR_DOC,  # 添加函数调用示例的文档字符串，使用 _CHECKPOINT_FOR_DOC 参数
    FlaxTokenClassifierOutput,
    _CONFIG_FOR_DOC,  # 结合 _CONFIG_FOR_DOC 参数
)


def identity(x, **kwargs):
    return x  # 定义一个简单的函数 identity，返回其输入参数 x


class FlaxElectraSequenceSummary(nn.Module):
    r"""
    Compute a single vector summary of a sequence hidden states.
    """
    Args:
        config ([`PretrainedConfig`]):
            The config used by the model. Relevant arguments in the config class of the model are (refer to the actual
            config class of your model for the default values it uses):

            - **summary_use_proj** (`bool`) -- Add a projection after the vector extraction.
            - **summary_proj_to_labels** (`bool`) -- If `True`, the projection outputs to `config.num_labels` classes
              (otherwise to `config.hidden_size`).
            - **summary_activation** (`Optional[str]`) -- Set to `"tanh"` to add a tanh activation to the output,
              another string or `None` will add no activation.
            - **summary_first_dropout** (`float`) -- Optional dropout probability before the projection and activation.
            - **summary_last_dropout** (`float`)-- Optional dropout probability after the projection and activation.
    """
    # 定义一个类变量config，它是一个ElectraConfig对象
    config: ElectraConfig
    # 定义一个数据类型变量dtype，默认为jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 类的初始化方法
    def setup(self):
        # 设置summary初始值为identity函数
        self.summary = identity
        # 检查config对象是否有summary_use_proj属性，并且它为True
        if hasattr(self.config, "summary_use_proj") and self.config.summary_use_proj:
            # 检查config对象是否有summary_proj_to_labels属性，并且它为True，并且config.num_labels大于0
            if (
                hasattr(self.config, "summary_proj_to_labels")
                and self.config.summary_proj_to_labels
                and self.config.num_labels > 0
            ):
                # 设置num_classes为config.num_labels
                num_classes = self.config.num_labels
            else:
                # 否则设置num_classes为config.hidden_size
                num_classes = self.config.hidden_size
            # 将summary设置为一个全连接层nn.Dense，输出维度为num_classes，数据类型为self.dtype
            self.summary = nn.Dense(num_classes, dtype=self.dtype)

        # 获取summary_activation字符串属性值
        activation_string = getattr(self.config, "summary_activation", None)
        # 根据activation_string获取对应的激活函数，如果为None则使用恒等函数lambda x: x
        self.activation = ACT2FN[activation_string] if activation_string else lambda x: x  # noqa F407

        # 设置first_dropout初始值为identity函数
        self.first_dropout = identity
        # 检查config对象是否有summary_first_dropout属性，并且其值大于0
        if hasattr(self.config, "summary_first_dropout") and self.config.summary_first_dropout > 0:
            # 将first_dropout设置为一个Dropout层，丢弃概率为config.summary_first_dropout
            self.first_dropout = nn.Dropout(self.config.summary_first_dropout)

        # 设置last_dropout初始值为identity函数
        self.last_dropout = identity
        # 检查config对象是否有summary_last_dropout属性，并且其值大于0
        if hasattr(self.config, "summary_last_dropout") and self.config.summary_last_dropout > 0:
            # 将last_dropout设置为一个Dropout层，丢弃概率为config.summary_last_dropout
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
        # NOTE: This function computes a summary vector of the sequence hidden states.

        # Extract the first token's hidden state from each sequence in the batch
        output = hidden_states[:, 0]

        # Apply dropout to the extracted hidden state
        output = self.first_dropout(output, deterministic=deterministic)

        # Compute the summary vector using a predefined method
        output = self.summary(output)

        # Apply an activation function to the computed summary vector
        output = self.activation(output)

        # Apply dropout to the final output vector before returning
        output = self.last_dropout(output, deterministic=deterministic)

        # Return the final summary vector
        return output
# 定义一个基于 Flax 的 Electra 多选题模型的模块类
class FlaxElectraForMultipleChoiceModule(nn.Module):
    # 指定配置对象为 ElectraConfig
    config: ElectraConfig
    # 指定数据类型为 jnp.float32 的浮点数
    dtype: jnp.dtype = jnp.float32
    # 梯度检查点，默认为关闭状态
    gradient_checkpointing: bool = False

    # 模块初始化方法
    def setup(self):
        # 创建 Electra 模型对象
        self.electra = FlaxElectraModule(
            config=self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )
        # 创建序列摘要对象
        self.sequence_summary = FlaxElectraSequenceSummary(config=self.config, dtype=self.dtype)
        # 创建分类器对象，使用 Dense 层，输出维度为 1
        self.classifier = nn.Dense(1, dtype=self.dtype)

    # 对象调用方法，处理输入并返回输出
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
        # 获取选择题的数量
        num_choices = input_ids.shape[1]
        # 若输入不为 None，则重塑输入的形状以便处理
        input_ids = input_ids.reshape(-1, input_ids.shape[-1]) if input_ids is not None else None
        attention_mask = attention_mask.reshape(-1, attention_mask.shape[-1]) if attention_mask is not None else None
        token_type_ids = token_type_ids.reshape(-1, token_type_ids.shape[-1]) if token_type_ids is not None else None
        position_ids = position_ids.reshape(-1, position_ids.shape[-1]) if position_ids is not None else None

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
        # 提取隐藏状态
        hidden_states = outputs[0]
        # 对隐藏状态进行序列摘要
        pooled_output = self.sequence_summary(hidden_states, deterministic=deterministic)
        # 使用分类器进行分类，生成逻辑回归结果
        logits = self.classifier(pooled_output)

        # 重塑 logits 的形状以匹配输入的多选题数量
        reshaped_logits = logits.reshape(-1, num_choices)

        # 如果不返回字典，则返回元组形式的结果
        if not return_dict:
            return (reshaped_logits,) + outputs[1:]

        # 返回多选题模型的输出，包括重塑后的 logits，隐藏状态和注意力
        return FlaxMultipleChoiceModelOutput(
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# 为 FlaxElectraForMultipleChoice 类添加文档字符串，描述其功能和用途
@add_start_docstrings(
    """
    ELECTRA Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    ELECTRA_START_DOCSTRING,
)
class FlaxElectraForMultipleChoice(FlaxElectraPreTrainedModel):
    module_class = FlaxElectraForMultipleChoiceModule


# 为 FlaxElectraForMultipleChoice 类的调用方法添加文档字符串示例
overwrite_call_docstring(
    FlaxElectraForMultipleChoice, ELECTRA_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
)
# 为 FlaxElectraForMultipleChoice 类添加调用方法的样例文档字符串
append_call_sample_docstring(
    FlaxElectraForMultipleChoice,
    _CHECKPOINT_FOR_DOC,
    FlaxMultipleChoiceModelOutput,
    _CONFIG_FOR_DOC,
)


# 定义一个基于 Flax 的 Electra 问答模型的模块类
class FlaxElectraForQuestionAnsweringModule(nn.Module):
    # 指定配置对象为 ElectraConfig
    config: ElectraConfig
    # 指定数据类型为 jnp.float32 的浮点数
    dtype: jnp.dtype = jnp.float32
    # 设置类中的梯度检查点标志，默认为 False
    gradient_checkpointing: bool = False
    
    # 初始化模型设置
    def setup(self):
        # 使用给定的配置、数据类型和梯度检查点设置创建 FlaxElectraModule 实例
        self.electra = FlaxElectraModule(
            config=self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )
        # 创建输出层，用于问题回答任务，输出维度为 self.config.num_labels
        self.qa_outputs = nn.Dense(self.config.num_labels, dtype=self.dtype)
    
    # 定义对象的调用方法，处理输入并返回预测结果
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
        # 调用 Electra 模型进行前向传播
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
        # 使用输出层计算起始和结束位置的 logits
        logits = self.qa_outputs(hidden_states)
        # 按输出类别数将 logits 分割为起始位置和结束位置的 logits
        start_logits, end_logits = logits.split(self.config.num_labels, axis=-1)
        # 去除最后一个维度上的冗余维度
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
    
        # 如果不返回字典，则返回元组形式的结果
        if not return_dict:
            return (start_logits, end_logits) + outputs[1:]
    
        # 返回 FlaxQuestionAnsweringModelOutput 对象，包含起始和结束 logits、隐藏状态和注意力
        return FlaxQuestionAnsweringModelOutput(
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """
    ELECTRA Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    ELECTRA_START_DOCSTRING,
)
class FlaxElectraForQuestionAnswering(FlaxElectraPreTrainedModel):
    module_class = FlaxElectraForQuestionAnsweringModule

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
        # Initialize a fully connected layer with hidden_size neurons
        self.dense = nn.Dense(self.config.hidden_size, dtype=self.dtype)
        
        # Determine dropout rate based on config values
        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )
        # Apply dropout with computed rate
        self.dropout = nn.Dropout(classifier_dropout)
        
        # Final output layer with num_labels neurons
        self.out_proj = nn.Dense(self.config.num_labels, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic: bool = True):
        # Extract the representation of the first token (<s>) from hidden_states
        x = hidden_states[:, 0, :]
        
        # Apply dropout to the extracted token representation
        x = self.dropout(x, deterministic=deterministic)
        
        # Pass through the fully connected layer
        x = self.dense(x)
        
        # Apply GELU activation function (similar to BERT's tanh)
        x = ACT2FN["gelu"](x)
        
        # Apply dropout again
        x = self.dropout(x, deterministic=deterministic)
        
        # Pass through the output layer
        x = self.out_proj(x)
        
        # Return the logits for sequence classification
        return x


class FlaxElectraForSequenceClassificationModule(nn.Module):
    config: ElectraConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        # Initialize Electra module with specified configuration
        self.electra = FlaxElectraModule(
            config=self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )
        
        # Initialize classification head using the same configuration
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
        # 如果 `return_dict` 为 True，则返回一个命名元组对象 FlaxSequenceClassifierOutput
        # 包含 logits, hidden_states 和 attentions 这些字段
        if not return_dict:
            # 如果 `return_dict` 为 False，返回一个元组，包含 logits 和 outputs 的其余部分
            return (logits,) + outputs[1:]

        # 如果 `return_dict` 为 True，返回一个 FlaxSequenceClassifierOutput 对象
        return FlaxSequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """
    Electra Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    ELECTRA_START_DOCSTRING,
)



class FlaxElectraForSequenceClassification(FlaxElectraPreTrainedModel):
    module_class = FlaxElectraForSequenceClassificationModule



append_call_sample_docstring(
    FlaxElectraForSequenceClassification,
    _CHECKPOINT_FOR_DOC,
    FlaxSequenceClassifierOutput,
    _CONFIG_FOR_DOC,
)



class FlaxElectraForCausalLMModule(nn.Module):
    config: ElectraConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        self.electra = FlaxElectraModule(
            config=self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )
        self.generator_predictions = FlaxElectraGeneratorPredictions(config=self.config, dtype=self.dtype)
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


**注释：**


# 添加起始文档字符串，描述此模型是基于Electra模型的序列分类/回归头（线性层叠加在汇总输出之上），例如用于GLUE任务。
@add_start_docstrings(
    """
    Electra Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    ELECTRA_START_DOCSTRING,
)

# 定义用于序列分类的FlaxElectraForSequenceClassification类，继承自FlaxElectraPreTrainedModel类。
class FlaxElectraForSequenceClassification(FlaxElectraPreTrainedModel):
    module_class = FlaxElectraForSequenceClassificationModule

# 向FlaxElectraForSequenceClassification类添加调用示例文档字符串。
append_call_sample_docstring(
    FlaxElectraForSequenceClassification,
    _CHECKPOINT_FOR_DOC,
    FlaxSequenceClassifierOutput,
    _CONFIG_FOR_DOC,
)

# 定义用于因果语言模型的FlaxElectraForCausalLMModule类，继承自nn.Module。
class FlaxElectraForCausalLMModule(nn.Module):
    config: ElectraConfig  # 类型注解，指定config属性的类型为ElectraConfig。
    dtype: jnp.dtype = jnp.float32  # 类型注解，指定dtype属性的类型，默认为jnp.float32。
    gradient_checkpointing: bool = False  # 类型注解，指定gradient_checkpointing属性的类型，默认为False。

    # 模块的设置方法
    def setup(self):
        # 创建Electra模块并赋值给self.electra属性，根据配置、数据类型和梯度检查点设置。
        self.electra = FlaxElectraModule(
            config=self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )
        # 创建生成器预测模块并赋值给self.generator_predictions属性，根据配置和数据类型设置。
        self.generator_predictions = FlaxElectraGeneratorPredictions(config=self.config, dtype=self.dtype)
        # 如果配置要求共享词嵌入，则创建FlaxElectraTiedDense类型的生成器语言模型头部，否则创建普通的nn.Dense。
        if self.config.tie_word_embeddings:
            self.generator_lm_head = FlaxElectraTiedDense(self.config.vocab_size, dtype=self.dtype)
        else:
            self.generator_lm_head = nn.Dense(self.config.vocab_size, dtype=self.dtype)

    # 模块的调用方法，接收多个输入参数，执行因果语言模型的计算。
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
            # 调用 ELECTRA 模型进行推理，获取输出结果
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
            # 从模型输出中获取隐藏状态
            hidden_states = outputs[0]
            # 使用生成器生成预测分数
            prediction_scores = self.generator_predictions(hidden_states)

            # 如果配置指定词嵌入共享
            if self.config.tie_word_embeddings:
                # 获取 ELECTRA 模型中的共享词嵌入参数
                shared_embedding = self.electra.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
                # 使用共享词嵌入进行生成器的 LM 头部预测
                prediction_scores = self.generator_lm_head(prediction_scores, shared_embedding.T)
            else:
                # 否则，直接使用生成器的 LM 头部进行预测
                prediction_scores = self.generator_lm_head(prediction_scores)

            # 如果不返回字典形式的输出
            if not return_dict:
                # 返回包含预测分数和额外输出的元组
                return (prediction_scores,) + outputs[1:]

            # 返回 FlaxCausalLMOutputWithCrossAttentions 类的对象，其中包含预测分数、隐藏状态、注意力权重及交叉注意力权重
            return FlaxCausalLMOutputWithCrossAttentions(
                logits=prediction_scores,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                cross_attentions=outputs.cross_attentions,
            )
@add_start_docstrings(
    """
    Electra Model with a language modeling head on top (a linear layer on top of the hidden-states output) e.g for
    autoregressive tasks.
    """,
    ELECTRA_START_DOCSTRING,
)
# 基于 transformers.models.bert.modeling_flax_bert.FlaxBertForCausalLM 中的代码，将 Bert 替换为 Electra
class FlaxElectraForCausalLM(FlaxElectraPreTrainedModel):
    module_class = FlaxElectraForCausalLMModule

    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = None):
        # 初始化缓存
        batch_size, seq_length = input_ids.shape

        # 使用模型的初始化缓存方法创建过去键值对
        past_key_values = self.init_cache(batch_size, max_length)
        
        # 注意：通常情况下，需要在 attention_mask 中对超出 input_ids.shape[-1] 和小于 cache_length 的位置填充 0
        # 但由于解码器使用因果遮蔽，这些位置已经被遮蔽了
        # 因此，我们可以在这里创建一个静态的 attention_mask，这对于编译来说更有效
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            # 计算位置 ID，根据 attention_mask 累积和减去 1
            position_ids = attention_mask.cumsum(axis=-1) - 1
            # 更新 extended_attention_mask，使用 attention_mask 进行动态更新切片
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            # 如果没有提供 attention_mask，则广播生成位置 ID
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        # 更新生成过程中的模型参数
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs


# 将样例调用的文档字符串附加到类 FlaxElectraForCausalLM 上，用于文档化
append_call_sample_docstring(
    FlaxElectraForCausalLM,
    _CHECKPOINT_FOR_DOC,
    FlaxCausalLMOutputWithCrossAttentions,
    _CONFIG_FOR_DOC,
)
```