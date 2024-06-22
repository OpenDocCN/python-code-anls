# `.\transformers\models\xlm_roberta\modeling_flax_xlm_roberta.py`

```py
# 设置编码格式为 utf-8
# 版权声明
# 引入所需的库
# XLM-RoBERTa 模型
"""Flax XLM-RoBERTa model."""

# 引入所需的库
from typing import Callable, Optional, Tuple
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

# 引入其他所需的类和函数
from ...modeling_flax_outputs import (
    FlaxBaseModelOutputWithPastAndCrossAttentions,
    FlaxBaseModelOutputWithPooling,
    FlaxBaseModelOutputWithPoolingAndCrossAttentions,
    FlaxCausalLMOutputWithCrossAttentions,
    FlaxMaskedLMOutput,
    FlaxMultipleChoiceModelOutput,
    FlaxQuestionAnsweringModelOutput,
    FlaxSequenceClassifierOutput,
    FlaxTokenClassifierOutput,
)
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring, overwrite_call_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging

# 引入 XLM-RoBERTa 的配置
from .configuration_xlm_roberta import XLMRobertaConfig

# 使用 logging 模块
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置
_CHECKPOINT_FOR_DOC = "xlm-roberta-base"
_CONFIG_FOR_DOC = "XLMRobertaConfig"

# XLM-RoBERTa 预训练模型的存档列表
FLAX_XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "xlm-roberta-base",
    "xlm-roberta-large",
    # See all XLM-RoBERTa models at https://huggingface.co/models?filter=xlm-roberta
]

# 从 transformers.models.roberta.modeling_flax_roberta.create_position_ids_from_input_ids 复制的函数
def create_position_ids_from_input_ids(input_ids, padding_idx):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        input_ids: jnp.ndarray
        padding_idx: int

    Returns: jnp.ndarray
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = (input_ids != padding_idx).astype("i4")
    # 检查输入的掩码（mask）是否具有大于两个维度
    if mask.ndim > 2:
        # 如果掩码维度大于2，将掩码展平为二维（将除最后一个维度外的所有维度合并为一个）
        mask = mask.reshape((-1, mask.shape[-1]))
        # 计算累积和，沿着第二维度（axis=1）计算每一行的累积和，结果为整数类型（'i4'），然后与掩码元素相乘
        incremental_indices = jnp.cumsum(mask, axis=1).astype("i4") * mask
        # 将计算得到的累积索引重新形状为与输入 ids 相同的形状
        incremental_indices = incremental_indices.reshape(input_ids.shape)
    else:
        # 如果掩码维度不大于2，直接计算累积和，沿着第二维度（axis=1）计算每一行的累积和，结果为整数类型（'i4'），然后与掩码元素相乘
        incremental_indices = jnp.cumsum(mask, axis=1).astype("i4") * mask
    
    # 将得到的累积索引转换为整数类型（'i4'），并加上填充索引（padding_idx）
    return incremental_indices.astype("i4") + padding_idx
# 这个字符串包含 XLM-Roberta 模型的文档字符串，用于 Flax 预训练模型
XLM_ROBERTA_START_DOCSTRING = r"""

    # 该模型继承自 `FlaxPreTrainedModel` 类，库中的所有模型通用方法都在超类的文档中
    # 包括下载、保存以及从 PyTorch 模型转换权重等

    # 该模型也是 `flax.linen.Module` 的子类，作为常规的 Flax linen 模块使用
    # 详细内容请参考 Flax 文档，了解通用用法和行为

    # 该模型还支持 JAX 的固有特性，例如：

    - # 支持即时编译 (Just-In-Time, JIT)，可优化执行速度
    - # 支持自动微分，用于梯度计算和反向传播
    - # 支持向量化，允许同时处理多个输入
    - # 支持并行化，可以在多个设备上并行执行

    # 参数说明:
        # `XLMRobertaConfig` 是模型的配置类，包含模型的所有参数
        # 使用配置文件初始化时不会加载模型权重，只加载配置
        # 要加载模型权重，请参考 `~FlaxPreTrainedModel.from_pretrained` 方法
"""

# 用于 XLM-Roberta 模型输入文档字符串的字符串
XLM_ROBERTA_INPUTS_DOCSTRING = r"""
    # 这些是输入参数的注释，描述了每个参数的作用和形状
    Args:
        # input_ids 是输入序列中每个令牌在词汇表中的索引，形状为 (N,)
        input_ids (`numpy.ndarray` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.
    
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
    
            [What are input IDs?](../glossary#input-ids)
        # attention_mask 是一个掩码张量，用于避免在填充令牌上执行注意力计算，形状为 (N,)
        attention_mask (`numpy.ndarray` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
    
            [What are attention masks?](../glossary#attention-mask)
        # token_type_ids 是一个指示输入中第一部分和第二部分的段标记索引，形状为 (N,)  
        token_type_ids (`numpy.ndarray` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:
    
            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.
    
            [What are token type IDs?](../glossary#token-type-ids)
        # position_ids 是输入序列中每个令牌的位置索引，形状为 (N,)
        position_ids (`numpy.ndarray` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.
        # head_mask 是一个用于屏蔽注意力头的掩码张量，形状为 (N,)
        head_mask (`numpy.ndarray` of shape `({0})`, `optional):
            Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:
    
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
    
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
# 从transformers.models.bert.modeling_flax_bert.FlaxBertEmbeddings复制并修改为XLMRoberta
class FlaxXLMRobertaEmbeddings(nn.Module):
    """从单词、位置和标记类型嵌入中构建嵌入。"""

    config: XLMRobertaConfig
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型

    def setup(self):
        # 初始化单词嵌入
        self.word_embeddings = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        # 初始化位置嵌入
        self.position_embeddings = nn.Embed(
            self.config.max_position_embeddings,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        # 初始化标记类型嵌入
        self.token_type_embeddings = nn.Embed(
            self.config.type_vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        # 初始化Layer Norm
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 初始化Dropout
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, input_ids, token_type_ids, position_ids, attention_mask, deterministic: bool = True):
        # 嵌入
        inputs_embeds = self.word_embeddings(input_ids.astype("i4"))
        position_embeds = self.position_embeddings(position_ids.astype("i4"))
        token_type_embeddings = self.token_type_embeddings(token_type_ids.astype("i4"))

        # 汇总所有嵌入
        hidden_states = inputs_embeds + token_type_embeddings + position_embeds

        # 层规范化
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states


# 从transformers.models.bert.modeling_flax_bert.FlaxBertSelfAttention复制并修改为XLMRoberta
class FlaxXLMRobertaSelfAttention(nn.Module):
    config: XLMRobertaConfig
    causal: bool = False
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型
    # 在模型初始化设置阶段进行操作
    def setup(self):
        # 计算每个注意力头的维度
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        # 检查隐藏层大小是否是注意力头数量的倍数
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            # 若不是，则引发数值错误
            raise ValueError(
                "`config.hidden_size`: {self.config.hidden_size} has to be a multiple of `config.num_attention_heads` "
                "                   : {self.config.num_attention_heads}"
            )

        # 初始化查询、键、值层
        self.query = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.key = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.value = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

        # 如果是因果的，创建因果掩码
        if self.causal:
            self.causal_mask = make_causal_mask(
                jnp.ones((1, self.config.max_position_embeddings), dtype="bool"), dtype="bool"
            )

    # 将隐藏状态拆分成多个头
    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.config.num_attention_heads, self.head_dim))

    # 合并多个头为一个隐藏状态
    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.config.hidden_size,))

    @nn.compact
    # 从transformers.models.bart.modeling_flax_bart.FlaxBartAttention._concatenate_to_cache复制而来
    # 将来自单个输入标记的投影键和值状态与先前步骤的缓存状态进行连接
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slightly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # 检测是否通过缺少现有缓存数据进行初始化
        is_initialized = self.has_variable("cache", "cached_key")
        # 初始化缓存的键和值状态
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        # 初始化缓存索引
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            # 解构出批次维度、最大长度、头数、每个头的深度
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # 使用新的一维空间片段更新键和值缓存
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            # 更新缓存的键和值状态
            cached_key.value = key
            cached_value.value = value
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # 生成用于缓存的因果遮蔽：我们的单个查询位置应仅注意到已生成和缓存的关键位置，而不是剩余的零元素
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)
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
# 从transformers.models.bert.modeling_flax_bert中拷贝FlaxBertSelfOutput类，并将Bert->XLMRoberta
class FlaxXLMRobertaSelfOutput(nn.Module):
    # 模型配置
    config: XLMRobertaConfig
    # 计算的数据类型
    dtype: jnp.dtype = jnp.float32  

    def setup(self):
        # 全连接层，将隐藏层大小设置为配置中的隐藏层大小，使用正态分布初始化权重，设置数据类型
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # LayerNorm层，设置epsilon和数据类型
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # Dropout层，设置丢弃率
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, hidden_states, input_tensor, deterministic: bool = True):
        # 全连接层计算
        hidden_states = self.dense(hidden_states)
        # Dropout计算
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # LayerNorm计算
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# 从transformers.models.bert.modeling_flax_bert中拷贝FlaxBertAttention类，并将Bert->XLMRoberta
class FlaxXLMRobertaAttention(nn.Module):
    # 模型配置
    config: XLMRobertaConfig
    # 是否使用自回归
    causal: bool = False
    # 计算的数据类型
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 使用FlaxXLMRobertaSelfAttention和FlaxXLMRobertaSelfOutput初始化self和output
        self.self = FlaxXLMRobertaSelfAttention(self.config, causal=self.causal, dtype=self.dtype)
        self.output = FlaxXLMRobertaSelfOutput(self.config, dtype=self.dtype)

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
        # 对attention_mask进行格式变换
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
        # 对输出进行处理
        hidden_states = self.output(attn_output, hidden_states, deterministic=deterministic)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_outputs[1],)

        return outputs


# 从transformers.models.bert.modeling_flax_bert中拷贝FlaxBertIntermediate类，并将Bert->XLMRoberta
class FlaxXLMRobertaIntermediate(nn.Module):
    # 模型配置
    config: XLMRobertaConfig
    # 计算的数据类型
    dtype: jnp.dtype = jnp.float32 

    def setup(self):
        # 全连接层，将隐藏层大小设置为配置中的中间层大小，使用正态分布初始化权重，设置数据类型
        self.dense = nn.Dense(
            self.config.intermediate_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 激活函数，根据配置中的隐藏层激活类型选择对应的激活函数
        self.activation = ACT2FN[self.config.hidden_act]
    # 接收输入的隐藏状态，并通过全连接层进行线性变换
    def __call__(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        # 使用激活函数激活线性变换后的隐藏状态
        hidden_states = self.activation(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states
# 从transformers.models.bert.modeling_flax_bert.FlaxBertOutput复制并将Bert->XLMRoberta替换
class FlaxXLMRobertaOutput(nn.Module):
    config: XLMRobertaConfig
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,  # 隐藏层大小
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),  # 使用正态分布初始化密度矩阵
            dtype=self.dtype,  # 数据类型
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)  # 隐藏层的丢弃概率
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)  # 使用LayerNorm对隐藏层进行标准化处理

    def __call__(self, hidden_states, attention_output, deterministic: bool = True):
        hidden_states = self.dense(hidden_states)  # 隐藏层的线性变换
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)  # 隐藏层的dropout操作
        hidden_states = self.LayerNorm(hidden_states + attention_output)  # 使用LayerNorm对隐藏层进行标准化处理并加上注意力输出
        return hidden_states


# 从transformers.models.bert.modeling_flax_bert.FlaxBertLayer复制并将Bert->XLMRoberta替换
class FlaxXLMRobertaLayer(nn.Module):
    config: XLMRobertaConfig
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型

    def setup(self):
        self.attention = FlaxXLMRobertaAttention(self.config, causal=self.config.is_decoder, dtype=self.dtype)  # XLMRoberta的注意力层
        self.intermediate = FlaxXLMRobertaIntermediate(self.config, dtype=self.dtype)  # XLMRoberta的中间层
        self.output = FlaxXLMRobertaOutput(self.config, dtype=self.dtype)  #XLMRoberta的输出层
        if self.config.add_cross_attention:
            self.crossattention = FlaxXLMRobertaAttention(self.config, causal=False, dtype=self.dtype)  # XLMRoberta的交叉注意力层

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
        # Self Attention 自注意力机制
        # 使用 self.attention 方法计算自注意力输出
        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            layer_head_mask=layer_head_mask,
            init_cache=init_cache,
            deterministic=deterministic,
            output_attentions=output_attentions,
        )
        # 从自注意力输出中获取注意力值
        attention_output = attention_outputs[0]

        # Cross-Attention Block 交叉注意力块
        # 如果有编码器的隐藏状态，执行交叉注意力计算
        if encoder_hidden_states is not None:
            # 使用 self.crossattention 方法计算交叉注意力输出
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask=encoder_attention_mask,
                layer_head_mask=layer_head_mask,
                key_value_states=encoder_hidden_states,
                deterministic=deterministic,
                output_attentions=output_attentions,
            )
            # 从交叉注意力输出中获取注意力值
            attention_output = cross_attention_outputs[0]

        # 使用 intermediate 方法处理注意力输出
        hidden_states = self.intermediate(attention_output)
        # 使用 output 方法生成最终的输出隐藏状态
        hidden_states = self.output(hidden_states, attention_output, deterministic=deterministic)

        # 将隐藏状态打包成元组
        outputs = (hidden_states,)

        # 如果需要输出注意力值
        if output_attentions:
            # 将自注意力的注意力值添加到输出元组中
            outputs += (attention_outputs[1],)
            # 如果有编码器的隐藏状态，将交叉注意力的注意力值添加到输出元组中
            if encoder_hidden_states is not None:
                outputs += (cross_attention_outputs[1],)
        # 返回输出元组
        return outputs
# 定义FlaxXLMRobertaLayerCollection类

# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertLayerCollection with Bert->XLMRoberta
class FlaxXLMRobertaLayerCollection(nn.Module):


# 定义类的属性

config: XLMRobertaConfig  # XLMRoberta的配置
dtype: jnp.dtype = jnp.float32  # 计算的数据类型，默认为float32
gradient_checkpointing: bool = False  # 是否使用梯度检查点，默认为False

   
# 定义类的初始化方法

def setup(self):


## 梯度检查点为True的情况

    if self.gradient_checkpointing:
            FlaxXLMRobertaCheckpointLayer = remat(FlaxXLMRobertaLayer, static_argnums=(5, 6, 7))
            self.layers = [
                FlaxXLMRobertaCheckpointLayer(self.config, name=str(i), dtype=self.dtype)
                for i in range(self.config.num_hidden_layers)
            ]


### 创建FlaxXLMRobertaCheckpointLayer对象

FlaxXLMRobertaCheckpointLayer = remat(FlaxXLMRobertaLayer, static_argnums=(5, 6, 7))


### 创建多个FlaxXLMRobertaCheckpointLayer对象并赋值给self.layers列表

self.layers = [
    FlaxXLMRobertaCheckpointLayer(self.config, name=str(i), dtype=self.dtype)
    for i in range(self.config.num_hidden_layers)
]


## 梯度检查点为False的情况

        else:
            self.layers = [
                FlaxXLMRobertaLayer(self.config, name=str(i), dtype=self.dtype)
                for i in range(self.config.num_hidden_layers)
            ]


### 创建多个FlaxXLMRobertaLayer对象并赋值给self.layers列表

self.layers = [
    FlaxXLMRobertaLayer(self.config, name=str(i), dtype=self.dtype)
    for i in range(self.config.num_hidden_layers)
]


# 定义类的调用方法

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
            如果不输出注意力信息，则将 all_attentions 设置为一个空元组
            all_attentions = () if output_attentions else None
            如果不输出隐藏状态信息，则将 all_hidden_states 设置为一个空元组
            all_hidden_states = () if output_hidden_states else None
            如果不输出跨注意力信息，则将 all_cross_attentions 设置为一个空元组
            all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

            # 检查是否head_mask的层数设置正确（如果需要的话）
            如果有head_mask，则需要检查其层数是否与self.layers的长度相同
            如果 head_mask 存在且 head_mask.shape[0] 不等于 self.layers 的长度
            抛出 ValueError 异常提示 head_mask 的层数不正确
            if head_mask is not None:
                if head_mask.shape[0] != (len(self.layers)):
                    raise ValueError(
                        f"The head_mask should be specified for {len(self.layers)} layers, but it is for                  "
                        f"       {head_mask.shape[0]}."
                    )

            遍历每个层并执行操作
            for i, layer in enumerate(self.layers):
                如果需要输出隐藏状态信息，则将当前隐藏状态添加到 all_hidden_states 中
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                调用当前层的 __call__ 方法，并更新隐藏状态
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
                hidden_states = layer_outputs[0]

                如果需要输出注意力信息，则将当前层的注意力矩阵添加到 all_attentions 中
                if output_attentions:
                    all_attentions += (layer_outputs[1],)

                    如果 encoder_hidden_states 存在，则将当前层的跨注意力矩阵添加到 all_cross_attentions 中
                    if encoder_hidden_states is not None:
                        all_cross_attentions += (layer_outputs[2],)

            如果需要输出隐藏状态信息，则将最终隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            构建 outputs 元组，包括隐藏状态、所有隐藏状态、所有注意力矩阵和所有跨注意力矩阵
            outputs = (hidden_states, all_hidden_states, all_attentions, all_cross_attentions)

            如果不需要返回字典形式的结果，则返回非空元素的元组
            if not return_dict:
                return tuple(v for v in outputs if v is not None)

            否则，返回返回带过去和跨注意力信息的 FlaxBaseModelOutputWithPastAndCrossAttentions 对象
            return FlaxBaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=all_attentions,
                cross_attentions=all_cross_attentions,
            )
# 从transformers.models.bert.modeling_flax_bert.FlaxBertEncoder复制代码，将Bert更改为XLMRoberta
class FlaxXLMRobertaEncoder(nn.Module):
    # XLMRoberta的配置
    config: XLMRobertaConfig
    # 计算的数据类型
    dtype: jnp.dtype = jnp.float32
    # 是否使用梯度检查点
    gradient_checkpointing: bool = False

    def setup(self):
        # 初始化XLMRoberta的层集合
        self.layer = FlaxXLMRobertaLayerCollection(
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
        # 调用XLMRoberta层集合进行前向传播
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


# 从transformers.models.bert.modeling_flax_bert.FlaxBertPooler复制代码，将Bert更改为XLMRoberta
class FlaxXLMRobertaPooler(nn.Module):
    # XLMRoberta的配置
    config: XLMRobertaConfig
    # 计算的数据类型
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 初始化XLMRoberta的池化层
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )

    def __call__(self, hidden_states):
        # 提取[CLS]标记的隐藏状态
        cls_hidden_state = hidden_states[:, 0]
        # 通过全连接层处理隐藏状态
        cls_hidden_state = self.dense(cls_hidden_state)
        # 使用tanh函数处理得到的结果
        return nn.tanh(cls_hidden_state)


# 从transformers.models.roberta.modeling_flax_roberta.FlaxRobertaLMHead复制代码，将Roberta更改为XLMRoberta
class FlaxXLMRobertaLMHead(nn.Module):
    # XLMRoberta的配置
    config: XLMRobertaConfig
    # 计算的数据类型
    dtype: jnp.dtype = jnp.float32
    # 偏置初始化函数
    bias_init: Callable[..., np.ndarray] = jax.nn.initializers.zeros

    def setup(self):
        # 初始化XLMRoberta的语言模型头部
        self.dense = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        # 初始化LayerNorm
        self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 初始化解码层
        self.decoder = nn.Dense(
            self.config.vocab_size,
            dtype=self.dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        # 初始化偏置参数
        self.bias = self.param("bias", self.bias_init, (self.config.vocab_size,))

```  
    # 在自定义类的实例上调用该方法，用于处理隐藏状态
    def __call__(self, hidden_states, shared_embedding=None):
        # 使用全连接层对隐藏状态进行线性变换
        hidden_states = self.dense(hidden_states)
        # 使用GELU激活函数对隐藏状态进行非线性变换
        hidden_states = ACT2FN["gelu"](hidden_states)
        # 对处理后的隐藏状态进行层归一化
        hidden_states = self.layer_norm(hidden_states)

        # 如果共享的嵌入矩阵不为空
        if shared_embedding is not None:
            # 应用预训练的解码器，使用共享的嵌入矩阵进行解码
            hidden_states = self.decoder.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
        else:
            # 使用解码器对隐藏状态进行解码
            hidden_states = self.decoder(hidden_states)

        # 将偏置转换为JAX的数组，并添加到隐藏状态中
        bias = jnp.asarray(self.bias, self.dtype)
        hidden_states += bias
        # 返回处理后的隐藏状态
        return hidden_states
# 从 transformers.models.roberta.modeling_flax_roberta.FlaxRobertaClassificationHead 复制而来，将 Roberta 替换为 XLMRoberta
class FlaxXLMRobertaClassificationHead(nn.Module):
    config: XLMRobertaConfig  # 存储 XLMRoberta 的配置信息
    dtype: jnp.dtype = jnp.float32  # 设置默认数据类型为 jnp.float32

    def setup(self):
        # 创建一个全连接层，输出大小为配置中的 hidden_size，使用配置中的 initializer_range 进行初始化
        self.dense = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        # 判断是否指定了分类器的 dropout rate，如果没有则使用配置中的 hidden_dropout_prob
        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )
        # 创建一个 Dropout 层，用于在训练过程中随机断开连接以防止过拟合
        self.dropout = nn.Dropout(rate=classifier_dropout)
        # 创建一个全连接层，输出大小为标签的数量，使用配置中的 initializer_range 进行初始化
        self.out_proj = nn.Dense(
            self.config.num_labels,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

    def __call__(self, hidden_states, deterministic=True):
        hidden_states = hidden_states[:, 0, :]  # 仅取第一个 token（等同于 [CLS]）的隐藏状态
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)  # 应用 dropout
        hidden_states = self.dense(hidden_states)  # 应用全连接层
        hidden_states = nn.tanh(hidden_states)  # 使用双曲正切激活函数
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)  # 再次应用 dropout
        hidden_states = self.out_proj(hidden_states)  # 应用输出全连接层
        return hidden_states


# 从 transformers.models.roberta.modeling_flax_roberta.FlaxRobertaPreTrainedModel 复制而来，将 Roberta 替换为 XLMRoberta、roberta 替换为 xlm-roberta、ROBERTA 替换为 XLM_ROBERTA
class FlaxXLMRobertaPreTrainedModel(FlaxPreTrainedModel):
    """
    处理权重初始化、下载和加载预训练模型的抽象类。
    """

    config_class = XLMRobertaConfig  # 配置类为 XLMRobertaConfig
    base_model_prefix = "xlm-roberta"  # 基础模型前缀为 "xlm-roberta"

    module_class: nn.Module = None  # 模块类默认为 None

    def __init__(
        self,
        config: XLMRobertaConfig,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        gradient_checkpointing: bool = False,
        **kwargs,
    ):
        # 创建模块实例，初始化模型
        module = self.module_class(config=config, dtype=dtype, gradient_checkpointing=gradient_checkpointing, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    # 从 transformers.models.bert.modeling_flax_bert.FlaxBertPreTrainedModel 复制而来，启用梯度检查点
    def enable_gradient_checkpointing(self):
        self._module = self.module_class(
            config=self.config,
            dtype=self.dtype,
            gradient_checkpointing=True,
        )
    # 初始化模型参数
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量
        input_ids = jnp.zeros(input_shape, dtype="i4")  # 用零初始化输入张量
        token_type_ids = jnp.ones_like(input_ids)  # 用1初始化与输入形状相同的张量
        position_ids = create_position_ids_from_input_ids(input_ids, self.config.pad_token_id)  # 根据输入和填充标记 ID 创建位置 ID
        attention_mask = jnp.ones_like(input_ids)  # 用1初始化与输入形状相同的张量作为注意力掩码
        head_mask = jnp.ones((self.config.num_hidden_layers, self.config.num_attention_heads))  # 用1初始化头掩码矩阵

        params_rng, dropout_rng = jax.random.split(rng)  # 分离参数 RNG 和 dropout RNG
        rngs = {"params": params_rng, "dropout": dropout_rng}  # 创建 RNGs 字典包含参数 RNG 和 dropout RNG

        if self.config.add_cross_attention:  # 如果模型配置要求添加交叉注意力
            encoder_hidden_states = jnp.zeros(input_shape + (self.config.hidden_size,))  # 用零填充形状为输入形状加上隐藏大小的张量
            encoder_attention_mask = attention_mask  # 设置编码器注意力掩码为输入的注意力掩码
            module_init_outputs = self.module.init(  # 初始化模块输出
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
            module_init_outputs = self.module.init(  # 初始化模块输出
                rngs, input_ids, attention_mask, token_type_ids, position_ids, head_mask, return_dict=False
            )

        random_params = module_init_outputs["params"]  # 获取随机参数

        if params is not None:  # 如果参数不为空
            random_params = flatten_dict(unfreeze(random_params))  # 展平并解冻随机参数
            params = flatten_dict(unfreeze(params))  # 展平并解冻参数
            for missing_key in self._missing_keys:  # 遍历缺失的键
                params[missing_key] = random_params[missing_key]  # 将随机参数的缺失的键添加到参数中
            self._missing_keys = set()  # 清空缺失的键集合
            return freeze(unflatten_dict(params))  # 冻结并还原参数字典
        else:
            return random_params  # 返回随机参数

    # 从 transformers.models.bart.modeling_flax_bart.FlaxBartDecoderPreTrainedModel.init_cache 复制而来
    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                fast auto-regressive decoding 使用的批处理大小。定义了初始化缓存的批处理大小。
            max_length (`int`):
                自动回归解码的最大可能长度。定义了初始化缓存的序列长度。
        """
        # 初始化检索缓存的输入变量
        input_ids = jnp.ones((batch_size, max_length), dtype="i4")  # 用1初始化形状为(batch_size, max_length)的张量
        attention_mask = jnp.ones_like(input_ids, dtype="i4")  # 用1初始化与输入张量形状相同的注意力掩码张量
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)  # 广播创建位置 ID

        init_variables = self.module.init(  # 初始化模块变量
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        return unfreeze(init_variables["cache"])  # 解冻并返回初始化变量的缓存

    @add_start_docstrings_to_model_forward(XLM_ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 定义一个特殊方法，用于将模型实例像函数一样调用，处理输入并返回输出
    def __call__(
        # 输入的 token IDs，表示输入序列中每个 token 的索引
        self,
        input_ids,
        # 注意力掩码，用于指示模型注意哪些 token，哪些 token 需要被忽略
        attention_mask=None,
        # 分段 ID，用于区分不同句子的 token
        token_type_ids=None,
        # 位置 ID，用于表示 token 在序列中的位置
        position_ids=None,
        # 头部掩码，用于控制模型的 self-attention 中哪些头部被保留
        head_mask=None,
        # 编码器隐藏状态，用于传递给模型以供参考
        encoder_hidden_states=None,
        # 编码器注意力掩码，用于指示哪些编码器隐藏状态需要被忽略
        encoder_attention_mask=None,
        # 额外的参数，以字典形式传入
        params: dict = None,
        # 用于控制模型中的随机失活
        dropout_rng: jax.random.PRNGKey = None,
        # 指示是否处于训练模式
        train: bool = False,
        # 指示是否返回注意力权重
        output_attentions: Optional[bool] = None,
        # 指示是否返回隐藏状态
        output_hidden_states: Optional[bool] = None,
        # 指示是否返回字典形式的输出
        return_dict: Optional[bool] = None,
        # 过去的键值对，用于缓存模型的中间状态
        past_key_values: dict = None,
# 从transformers.models.bert.modeling_flax_bert.FlaxBertModule中拷贝代码，将Bert->XLMRoberta
class FlaxXLMRobertaModule(nn.Module):
    config: XLMRobertaConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    add_pooling_layer: bool = True
    gradient_checkpointing: bool = False

    def setup(self):
        self.embeddings = FlaxXLMRobertaEmbeddings(self.config, dtype=self.dtype)  # 初始化XLMRoberta的嵌入层
        self.encoder = FlaxXLMRobertaEncoder(
            self.config,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )  # 初始化XLMRoberta的编码器
        self.pooler = FlaxXLMRobertaPooler(self.config, dtype=self.dtype)  # 初始化XLMRoberta的池化层

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
        # 确保`token_type_ids`在未传入时正确初始化
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        # 确保`position_ids`在未传入时正确初始化
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        # 通过嵌入层处理输入，得到隐藏状态
        hidden_states = self.embeddings(
            input_ids, token_type_ids, position_ids, attention_mask, deterministic=deterministic
        )
        # 通过编码器得到输出
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
        # 如果需要加池化层，则对隐藏状态进行池化
        pooled = self.pooler(hidden_states) if self.add_pooling_layer else None

        # 如果不需要返回字典，则根据条件返回对应的值
        if not return_dict:
            # 如果没有池化结果，则不返回它
            if pooled is None:
                return (hidden_states,) + outputs[1:]
            return (hidden_states, pooled) + outputs[1:]

        # 返回带有池化和交叉注意力的输出字典
        return FlaxBaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )
    # XLM-ROBERTA 模型的起始文档字符串
    XLM_ROBERTA_START_DOCSTRING,
# 导入类
class FlaxXLMRobertaModel(FlaxXLMRobertaPreTrainedModel):
    # 模型类别为FlaxXLMRobertaModule
    module_class = FlaxXLMRobertaModule


# 为FlaxXLMRobertaModel添加调用示例的文档字符串
append_call_sample_docstring(FlaxXLMRobertaModel, _CHECKPOINT_FOR_DOC, FlaxBaseModelOutputWithPooling, _CONFIG_FOR_DOC)


# 从transformers.models.roberta.modeling_flax_roberta.FlaxRobertaForMaskedLMModule复制并将Roberta改为XLMRoberta
class FlaxXLMRobertaForMaskedLMModule(nn.Module):
    # XLMRoberta的配置
    config: XLMRobertaConfig
    # 数据类型为jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 是否进行梯度检查点
    gradient_checkpointing: bool = False

    def setup(self):
        # 创建FlaxXLMRobertaModule对象，作为语言模型的基础模型
        self.roberta = FlaxXLMRobertaModule(
            config=self.config,
            add_pooling_layer=False,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 创建FlaxXLMRobertaLMHead对象，作为语言模型的输出层
        self.lm_head = FlaxXLMRobertaLMHead(config=self.config, dtype=self.dtype)

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
        # 使用基础模型进行预测
        outputs = self.roberta(
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
        # 判断是否要共享词嵌入层
        if self.config.tie_word_embeddings:
            shared_embedding = self.roberta.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
        else:
            shared_embedding = None

        # 计算预测分数
        logits = self.lm_head(hidden_states, shared_embedding=shared_embedding)

        if not return_dict:
            return (logits,) + outputs[1:]

        # 返回FlaxMaskedLMOutput对象
        return FlaxMaskedLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# 为FlaxXLMRobertaForMaskedLM添加调用示例的文档字符串
append_call_sample_docstring(
    FlaxXLMRobertaForMaskedLM,
    _CHECKPOINT_FOR_DOC,
    FlaxBaseModelOutputWithPooling,
    _CONFIG_FOR_DOC,
    mask="<mask>",
)


# 从transformers.models.roberta.modeling_flax_roberta.FlaxRobertaForSequenceClassificationModule复制并将Roberta改为XLMRoberta
class FlaxXLMRobertaForSequenceClassificationModule(nn.Module):
    # XLMRoberta的配置
    config: XLMRobertaConfig
    # 数据类型为jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 是否进行梯度检查点
    gradient_checkpointing: bool = False
    # 初始化模型参数
    def setup(self):
        # 初始化 RoBERTa 模型
        self.roberta = FlaxXLMRobertaModule(
            config=self.config,  # 使用给定的配置初始化模型
            dtype=self.dtype,  # 设置数据类型
            add_pooling_layer=False,  # 不添加池化层
            gradient_checkpointing=self.gradient_checkpointing,  # 设置是否使用梯度检查点
        )
        # 初始化分类器
        self.classifier = FlaxXLMRobertaClassificationHead(config=self.config, dtype=self.dtype)

    # 模型调用方法
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
        # 模型前向传播
        outputs = self.roberta(
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

        # 获取序列输出
        sequence_output = outputs[0]
        # 通过分类器获取 logits
        logits = self.classifier(sequence_output, deterministic=deterministic)

        # 如果不返回字典，则返回 logits 和其他输出
        if not return_dict:
            return (logits,) + outputs[1:]

        # 返回包含 logits、隐藏状态和注意力权重的字典
        return FlaxSequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 为XLM Roberta Model添加一个序列分类/回归头部的transformer模型（在汇总输出之上是一个线性层），例如用于GLUE任务。
# 这是一个自定义的文档字符串
class FlaxXLMRobertaForSequenceClassification(FlaxXLMRobertaPreTrainedModel):
    module_class = FlaxXLMRobertaForSequenceClassificationModule

# 附加调用示例的文档字符串
append_call_sample_docstring(
    FlaxXLMRobertaForSequenceClassification,
    _CHECKPOINT_FOR_DOC,
    FlaxSequenceClassifierOutput,
    _CONFIG_FOR_DOC,
)

# 从transformers.models.bert.modeling_flax_bert.FlaxBertForMultipleChoiceModule复制，并将Bert->XLMRoberta，将self.bert->self.roberta
class FlaxXLMRobertaForMultipleChoiceModule(nn.Module):
    config: XLMRobertaConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        self.roberta = FlaxXLMRobertaModule(
            config=self.config,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
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
        input_ids = input_ids.reshape(-1, input_ids.shape[-1]) if input_ids is not None else None
        attention_mask = attention_mask.reshape(-1, attention_mask.shape[-1]) if attention_mask is not None else None
        token_type_ids = token_type_ids.reshape(-1, token_type_ids.shape[-1]) if token_type_ids is not None else None
        position_ids = position_ids.reshape(-1, position_ids.shape[-1]) if position_ids is not None else None

        # Model
        outputs = self.roberta(
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

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output, deterministic=deterministic)
        logits = self.classifier(pooled_output)

        reshaped_logits = logits.reshape(-1, num_choices)

        if not return_dict:
            return (reshaped_logits,) + outputs[2:]

        return FlaxMultipleChoiceModelOutput(
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# 为XLM Roberta Model添加一个多选分类输出头部的文档字符串
@add_start_docstrings(
    """
    XLM Roberta Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
    # 向模型添加额外的输出层（如softmax），例如用于RocStories/SWAG任务。
    """,
    # 添加XLM-RoBERTa模型的文档字符串开头信息
    XLM_ROBERTA_START_DOCSTRING,
# 导入必要的包
import jax.numpy as jnp
from flax import nn
from transformers.models.roberta.modeling_flax_roberta import FlaxRobertaForSequenceClassificationModule

# 为多选题任务定义 FlaxXLMRobertaForMultipleChoice 类
class FlaxXLMRobertaForMultipleChoice(FlaxXLMRobertaPreTrainedModel):
    # 模型类别为 FlaxXLMRobertaForMultipleChoiceModule
    module_class = FlaxXLMRobertaForMultipleChoiceModule

# 覆盖 FlaxXLMRobertaForMultipleChoice 类的文档字符串
overwrite_call_docstring(
    FlaxXLMRobertaForMultipleChoice, XLM_ROBERTA_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
)
# 为 FlaxXLMRobertaForMultipleChoice 类添加示例文档字符串
append_call_sample_docstring(
    FlaxXLMRobertaForMultipleChoice,
    _CHECKPOINT_FOR_DOC,
    FlaxMultipleChoiceModelOutput,
    _CONFIG_FOR_DOC,
)

# 定义一个用于标记分类任务的 FlaxXLMRobertaForTokenClassificationModule 类
class FlaxXLMRobertaForTokenClassificationModule(nn.Module):
    # 模型配置为 XLMRobertaConfig
    config: XLMRobertaConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        # 创建一个 FlaxXLMRobertaModule 实例
        self.roberta = FlaxXLMRobertaModule(
            config=self.config,
            dtype=self.dtype,
            add_pooling_layer=False,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 根据配置设置分类器的 dropout
        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )
        # 添加 dropout 层
        self.dropout = nn.Dropout(rate=classifier_dropout)
        # 添加分类器层
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
        # 调用模型，得到输出
        outputs = self.roberta(
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
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        logits = self.classifier(hidden_states)

        if not return_dict:
            return (logits,) + outputs[1:]

        return FlaxTokenClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# 为 FlaxXLMRobertaForTokenClassification 类添加示例文档字符串
@add_start_docstrings(
    """
    XLM Roberta Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    XLM_ROBERTA_START_DOCSTRING,
)
class FlaxXLMRobertaForTokenClassification(FlaxXLMRobertaPreTrainedModel):
    # 模型类别为 FlaxXLMRobertaForTokenClassificationModule
    module_class = FlaxXLMRobertaForTokenClassificationModule

# 为 FlaxXLMRobertaForTokenClassification 类添加示例文档字符串
append_call_sample_docstring(
    FlaxXLMRobertaForTokenClassification,
    _CHECKPOINT_FOR_DOC,
    FlaxTokenClassifierOutput,
    _CONFIG_FOR_DOC,
)
# 从 transformers.models.bert.modeling_flax_bert.FlaxBertForQuestionAnsweringModule 复制代码，并将其中的 Bert 替换为 XLMRoberta，self.bert 替换为 self.roberta
class FlaxXLMRobertaForQuestionAnsweringModule(nn.Module):
    # XLMRobertaForQuestionAnsweringModule 类的配置参数，类型为 XLMRobertaConfig
    config: XLMRobertaConfig
    # 数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 是否使用梯度检查点，默认为 False
    gradient_checkpointing: bool = False

    # 初始化函数
    def setup(self):
        # 创建 FlaxXLMRobertaModule 实例，并传入配置参数、数据类型、不添加池化层、梯度检查点参数
        self.roberta = FlaxXLMRobertaModule(
            config=self.config,
            dtype=self.dtype,
            add_pooling_layer=False,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 创建全连接层实例，输出维度为 num_labels，数据类型为 dtype
        self.qa_outputs = nn.Dense(self.config.num_labels, dtype=self.dtype)

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
    ):
        # 使用 XLMRobertaModule 处理输入数据，返回输出结果
        outputs = self.roberta(
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

        # 从输出结果中获取隐藏状态
        hidden_states = outputs[0]

        # 将隐藏状态经过全连接层得到 logits
        logits = self.qa_outputs(hidden_states)
        # 将 logits 沿着最后一个维度分割成 start_logits 和 end_logits
        start_logits, end_logits = jnp.split(logits, self.config.num_labels, axis=-1)
        # 压缩维度
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # 如果不返回字典，则返回 logits 和 outputs[1:]
        if not return_dict:
            return (start_logits, end_logits) + outputs[1:]

        # 返回 FlaxQuestionAnsweringModelOutput 类型的对象
        return FlaxQuestionAnsweringModelOutput(
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# 添加文档字符串，描述 XLM Roberta 模型在抽取式问答任务（如 SQuAD）上的应用
@add_start_docstrings(
    """
    XLM Roberta Model with a span classification head on top for extractive question-answering tasks like SQuAD (a
    linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    XLM_ROBERTA_START_DOCSTRING,
)
# XLMRoberta 问答模型类，继承自 FlaxXLMRobertaPreTrainedModel
class FlaxXLMRobertaForQuestionAnswering(FlaxXLMRobertaPreTrainedModel):
    # 模型类的实现
    module_class = FlaxXLMRobertaForQuestionAnsweringModule


# 添加函数调用示例的文档字符串
append_call_sample_docstring(
    FlaxXLMRobertaForQuestionAnswering,
    _CHECKPOINT_FOR_DOC,
    FlaxQuestionAnsweringModelOutput,
    _CONFIG_FOR_DOC,
)


# 从 transformers.models.roberta.modeling_flax_roberta.FlaxRobertaForCausalLMModule 复制代码，并将其中的 Roberta 替换为 XLMRoberta
class FlaxXLMRobertaForCausalLMModule(nn.Module):
    # XLMRobertaForCausalLMModule 类的配置参数，类型为 XLMRobertaConfig
    config: XLMRobertaConfig
    # 数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 是否使用梯度检查点，默认为 False
    gradient_checkpointing: bool = False
    # 设置模型参数
    def setup(self):
        # 初始化一个 FlaxXLMRobertaModule 对象，不添加池化层
        self.roberta = FlaxXLMRobertaModule(
            config=self.config,
            add_pooling_layer=False,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 初始化一个 FlaxXLMRobertaLMHead 对象
        self.lm_head = FlaxXLMRobertaLMHead(config=self.config, dtype=self.dtype)

    # __call__ 方法用于对模型进行调用
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
        # 对输入进行模型预测
        outputs = self.roberta(
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

        # 获取模型输出的隐藏状态
        hidden_states = outputs[0]
        # 如果需要共享词嵌入，则获取共享词嵌入
        if self.config.tie_word_embeddings:
            shared_embedding = self.roberta.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
        else:
            shared_embedding = None

        # 计算预测得分
        logits = self.lm_head(hidden_states, shared_embedding=shared_embedding)

        # 如果不需要返回字典形式的结果，则返回预测得分和其他输出
        if not return_dict:
            return (logits,) + outputs[1:]

        # 如果需要返回字典形式的结果，则返回带有交叉注意力的输出
        return FlaxCausalLMOutputWithCrossAttentions(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )
# 根据给定文档字符串创建XLM Roberta模型，增加了用于语言建模的头部（在隐藏状态输出的顶部）例如用于自回归任务
@add_start_docstrings(
    """
    XLM Roberta Model with a language modeling head on top (a linear layer on top of the hidden-states output) e.g for
    autoregressive tasks.
    """,
    XLM_ROBERTA_START_DOCSTRING,
)
# 复制自transformers.models.roberta.modeling_flax_roberta.FlaxRobertaForCausalLM，并将Roberta->XLMRoberta
class FlaxXLMRobertaForCausalLM(FlaxXLMRobertaPreTrainedModel):
    module_class = FlaxXLMRobertaForCausalLMModule

    # 准备用于生成的输入，初始化缓存
    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = None):
        batch_size, seq_length = input_ids.shape

        # 初始化缓存
        past_key_values = self.init_cache(batch_size, max_length)
        # 注意通常需要在attention_mask的x > input_ids.shape[-1]和x < cache_length位置放0
        # 但由于解码器使用因果掩码，这些位置已经被掩盖
        # 因此，我们可以在这里创建一个静态的attention_mask，这对编译更有效
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

    # 更新输入以进行生成
    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs

# 向FlaxXLMRobertaForCausalLM添加调用示例文档字符串
append_call_sample_docstring(
    FlaxXLMRobertaForCausalLM,
    _CHECKPOINT_FOR_DOC,
    FlaxCausalLMOutputWithCrossAttentions,
    _CONFIG_FOR_DOC,
)
```