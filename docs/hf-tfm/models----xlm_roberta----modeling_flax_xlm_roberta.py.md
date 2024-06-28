# `.\models\xlm_roberta\modeling_flax_xlm_roberta.py`

```
# coding=utf-8
# Copyright 2022 Facebook AI Research and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Flax XLM-RoBERTa model.
"""

from typing import Callable, Optional, Tuple

import flax.linen as nn                    # 导入 Flax 的 linen 模块，用于定义神经网络模型
import jax                                 # 导入 JAX，用于执行自动微分和数组操作
import jax.numpy as jnp                    # 导入 JAX 的 NumPy 接口，用作主要的数值计算库
import numpy as np                         # 导入 NumPy，用于处理数组和数值计算
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze  # 导入 Flax 的冻结字典相关函数
from flax.linen import combine_masks, make_causal_mask           # 导入 Flax linen 的函数，用于掩码操作
from flax.linen import partitioning as nn_partitioning           # 导入 Flax linen 的分区模块，用于模型分区
from flax.linen.attention import dot_product_attention_weights  # 导入 Flax linen 的注意力机制函数
from flax.traverse_util import flatten_dict, unflatten_dict     # 导入 Flax 的工具函数，用于字典扁平化和还原
from jax import lax                       # 导入 JAX 的 lax 模块，用于定义低级操作

from ...modeling_flax_outputs import (     # 导入 Flax 模型输出相关的类和函数
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
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring, overwrite_call_docstring  # 导入 Flax 模型工具函数和类
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging  # 导入相关的工具函数和日志记录模块
from .configuration_xlm_roberta import XLMRobertaConfig  # 导入当前模型配置文件

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

_CHECKPOINT_FOR_DOC = "FacebookAI/xlm-roberta-base"  # 模型文档中使用的检查点名称
_CONFIG_FOR_DOC = "XLMRobertaConfig"  # 模型文档中使用的配置文件名称

remat = nn_partitioning.remat  # 定义重映射函数

FLAX_XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = [  # 预训练模型存档列表
    "FacebookAI/xlm-roberta-base",
    "FacebookAI/xlm-roberta-large",
    # See all XLM-RoBERTa models at https://huggingface.co/models?filter=xlm-roberta
]


# Copied from transformers.models.roberta.modeling_flax_roberta.create_position_ids_from_input_ids
def create_position_ids_from_input_ids(input_ids, padding_idx):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        input_ids: jnp.ndarray  # 输入的 ID 数组
        padding_idx: int         # 填充符号的索引

    Returns: jnp.ndarray         # 返回一个新的位置 ID 数组
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = (input_ids != padding_idx).astype("i4")  # 创建一个掩码，标记非填充符号的位置为1，填充符号位置为0
    # 如果 mask 的维度大于2，则进行形状重塑，将其展平为二维数组
    if mask.ndim > 2:
        mask = mask.reshape((-1, mask.shape[-1]))
        # 计算累积和，结果为整数类型，乘以 mask，保留同样的形状
        incremental_indices = jnp.cumsum(mask, axis=1).astype("i4") * mask
        # 将累积和的结果重塑为与 input_ids 相同的形状
        incremental_indices = incremental_indices.reshape(input_ids.shape)
    else:
        # 如果 mask 的维度不大于2，则直接计算累积和，结果为整数类型，乘以 mask
        incremental_indices = jnp.cumsum(mask, axis=1).astype("i4") * mask

    # 将最终的累积和数组转换为整数类型，并加上 padding_idx
    return incremental_indices.astype("i4") + padding_idx
# XLM_ROBERTA_START_DOCSTRING 是一个包含多行字符串的文档字符串，描述了该模型的继承关系和基本特性，
# 以及它作为 Flax linen 模块的使用方式和支持的 JAX 特性。
XLM_ROBERTA_START_DOCSTRING = r"""

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
        config ([`XLMRobertaConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.
"""

# XLM_ROBERTA_INPUTS_DOCSTRING 是一个单行字符串的文档字符串，目前为空字符串。
XLM_ROBERTA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`numpy.ndarray` of shape `({0})`):
            # 输入序列的标记索引在词汇表中的位置。

            # 可以使用 [`AutoTokenizer`] 获取这些索引。参见 [`PreTrainedTokenizer.encode`] 和
            # [`PreTrainedTokenizer.__call__`] 获取详细信息。

            # [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`numpy.ndarray` of shape `({0})`, *optional*):
            # 避免在填充标记索引上执行注意力操作的掩码。掩码值为 `[0, 1]`：

            # - 1 表示**未被掩码**的标记，
            # - 0 表示**被掩码**的标记。

            # [什么是注意力掩码？](../glossary#attention-mask)
        token_type_ids (`numpy.ndarray` of shape `({0})`, *optional*):
            # 段标记索引，用于指示输入的第一部分和第二部分。索引值为 `[0, 1]`：

            # - 0 对应于*句子 A* 的标记，
            # - 1 对应于*句子 B* 的标记。

            # [什么是标记类型 ID？](../glossary#token-type-ids)
        position_ids (`numpy.ndarray` of shape `({0})`, *optional*):
            # 每个输入序列标记在位置嵌入中的位置索引。选择范围为 `[0, config.max_position_embeddings - 1]`。

        head_mask (`numpy.ndarray` of shape `({0})`, `optional):
            # 用于将注意力模块中选择的头部置零的掩码。掩码值为 `[0, 1]`：

            # - 1 表示**未被掩码**的头部，
            # - 0 表示**被掩码**的头部。

        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通元组。
# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertEmbeddings with Bert->XLMRoberta
class FlaxXLMRobertaEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    config: XLMRobertaConfig  # 类型提示：XLMRoberta 模型配置对象
    dtype: jnp.dtype = jnp.float32  # 计算使用的数据类型，默认为单精度浮点型

    def setup(self):
        # 初始化词嵌入层，用于将输入的词 ID 映射成对应的词向量
        self.word_embeddings = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        # 初始化位置嵌入层，用于表示词的位置信息
        self.position_embeddings = nn.Embed(
            self.config.max_position_embeddings,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        # 初始化类型嵌入层，用于区分不同类型的输入（如句子 A 和句子 B）
        self.token_type_embeddings = nn.Embed(
            self.config.type_vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        # 初始化 Layer Normalization 层，用于归一化隐藏状态
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 初始化 Dropout 层，用于在训练过程中随机丢弃部分隐藏状态，防止过拟合
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, input_ids, token_type_ids, position_ids, attention_mask, deterministic: bool = True):
        # Embed
        # 将输入的词 ID 转换为词嵌入向量
        inputs_embeds = self.word_embeddings(input_ids.astype("i4"))
        # 将位置 ID 转换为位置嵌入向量
        position_embeds = self.position_embeddings(position_ids.astype("i4"))
        # 将类型 ID 转换为类型嵌入向量
        token_type_embeddings = self.token_type_embeddings(token_type_ids.astype("i4"))

        # Sum all embeddings
        # 将词嵌入向量、位置嵌入向量和类型嵌入向量相加得到最终的隐藏状态
        hidden_states = inputs_embeds + token_type_embeddings + position_embeds

        # Layer Norm
        # 对隐藏状态进行 Layer Normalization 处理
        hidden_states = self.LayerNorm(hidden_states)
        # 对归一化后的隐藏状态进行 Dropout 操作，以防止过拟合
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertSelfAttention with Bert->XLMRoberta
class FlaxXLMRobertaSelfAttention(nn.Module):
    config: XLMRobertaConfig  # 类型提示：XLMRoberta 模型配置对象
    causal: bool = False  # 是否是因果注意力（自回归/自回归式），默认为否
    dtype: jnp.dtype = jnp.float32  # 计算使用的数据类型，默认为单精度浮点型
    # 在模型设置过程中调用，计算每个注意力头的维度
    def setup(self):
        # 将隐藏层大小除以注意力头的数量，以确定每个头的维度
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        # 如果隐藏层大小不能被注意力头的数量整除，抛出数值错误异常
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError(
                "`config.hidden_size`: {self.config.hidden_size} has to be a multiple of `config.num_attention_heads` "
                "                   : {self.config.num_attention_heads}"
            )

        # 初始化查询、键、值网络层，用于注意力机制
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

        # 如果启用因果注意力机制，则创建一个因果掩码
        if self.causal:
            self.causal_mask = make_causal_mask(
                jnp.ones((1, self.config.max_position_embeddings), dtype="bool"), dtype="bool"
            )

    # 将隐藏状态张量分割为多个注意力头
    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.config.num_attention_heads, self.head_dim))

    # 将多个注意力头的张量合并回隐藏状态张量
    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.config.hidden_size,))

    # 使用 nn.compact 修饰器，定义一个函数，此处功能与特定的函数一致
    @nn.compact
    # Copied from transformers.models.bart.modeling_flax_bart.FlaxBartAttention._concatenate_to_cache
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slightly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # 检测是否通过缺少现有缓存数据来初始化。
        is_initialized = self.has_variable("cache", "cached_key")
        # 获取或初始化缓存的键和值，使用零张量填充，维度和类型与输入的key和value相同。
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        # 获取或初始化缓存索引，初始化为整数0。
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            # 提取批处理维度、最大长度、头数和每头深度
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # 使用新的一维空间片段更新键和值缓存
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            # 更新缓存中的键和值
            cached_key.value = key
            cached_value.value = value
            # 更新缓存索引，增加更新的缓存向量数量
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # 生成用于缓存解码器自注意力的因果掩码：我们的单个查询位置只应关注已生成和缓存的键位置，而不是剩余的零元素。
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            # 将因果掩码与输入的注意力掩码结合起来
            attention_mask = combine_masks(pad_mask, attention_mask)
        # 返回更新后的键、值和注意力掩码
        return key, value, attention_mask
# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertSelfOutput with Bert->XLMRoberta
# 定义了一个用于 XLMRoberta 模型的自注意力输出层
class FlaxXLMRobertaSelfOutput(nn.Module):
    config: XLMRobertaConfig  # 类型注解，指定配置类 XLMRobertaConfig
    dtype: jnp.dtype = jnp.float32  # 计算中使用的数据类型

    def setup(self):
        # 初始化全连接层，输出维度为配置中指定的隐藏大小，使用正态分布初始化权重
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 初始化 LayerNorm 层，epsilon 参数由配置类 XLMRobertaConfig 提供
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 初始化 Dropout 层，dropout 率由配置类 XLMRobertaConfig 提供
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, hidden_states, input_tensor, deterministic: bool = True):
        # 前向传播函数，接收隐藏状态、输入张量和一个布尔值作为参数
        # 通过全连接层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # 使用 Dropout 层对处理后的隐藏状态进行随机失活
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 将处理后的隐藏状态与输入张量相加，并通过 LayerNorm 层处理
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertAttention with Bert->XLMRoberta
# 定义了一个用于 XLMRoberta 模型的注意力机制层
class FlaxXLMRobertaAttention(nn.Module):
    config: XLMRobertaConfig  # 类型注解，指定配置类 XLMRobertaConfig
    causal: bool = False  # 是否启用因果关系的布尔值，默认为 False
    dtype: jnp.dtype = jnp.float32  # 计算中使用的数据类型

    def setup(self):
        # 初始化自注意力层，使用 XLMRobertaSelfAttention 类处理
        self.self = FlaxXLMRobertaSelfAttention(self.config, causal=self.causal, dtype=self.dtype)
        # 初始化自注意力输出层，使用 FlaxXLMRobertaSelfOutput 类处理
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
        # 前向传播函数，接收多个参数用于处理注意力机制
        # 使用 self.self 处理自注意力计算，得到注意力输出
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
        # 使用 self.output 处理注意力输出，得到最终的隐藏状态
        hidden_states = self.output(attn_output, hidden_states, deterministic=deterministic)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_outputs[1],)

        return outputs


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertIntermediate with Bert->XLMRoberta
# 定义了一个用于 XLMRoberta 模型的中间层
class FlaxXLMRobertaIntermediate(nn.Module):
    config: XLMRobertaConfig  # 类型注解，指定配置类 XLMRobertaConfig
    dtype: jnp.dtype = jnp.float32  # 计算中使用的数据类型

    def setup(self):
        # 初始化全连接层，输出维度为配置中指定的中间大小，使用正态分布初始化权重
        self.dense = nn.Dense(
            self.config.intermediate_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 初始化激活函数，激活函数类型由配置类 XLMRobertaConfig 提供
        self.activation = ACT2FN[self.config.hidden_act]
    # 定义一个类中的特殊方法 __call__()，用于将对象实例像函数一样调用
    def __call__(self, hidden_states):
        # 将输入的隐藏状态数据通过全连接层进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对变换后的隐藏状态数据应用激活函数
        hidden_states = self.activation(hidden_states)
        # 返回经过线性变换和激活函数处理后的隐藏状态数据
        return hidden_states
# 从 transformers.models.bert.modeling_flax_bert.FlaxBertOutput 复制而来，将 Bert 替换为 XLMRoberta
class FlaxXLMRobertaOutput(nn.Module):
    config: XLMRobertaConfig  # XLMRoberta 模型的配置信息
    dtype: jnp.dtype = jnp.float32  # 计算时使用的数据类型

    def setup(self):
        # 初始化一个全连接层，输出大小为 config.hidden_size
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),  # 使用正态分布初始化权重
            dtype=self.dtype,
        )
        # 初始化一个 Dropout 层，丢弃率为 config.hidden_dropout_prob
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        # 初始化一个 LayerNorm 层，epsilon 为 config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    def __call__(self, hidden_states, attention_output, deterministic: bool = True):
        # 通过全连接层处理 hidden_states
        hidden_states = self.dense(hidden_states)
        # 应用 Dropout 处理 hidden_states
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 应用 LayerNorm 处理 hidden_states 和 attention_output 的和
        hidden_states = self.LayerNorm(hidden_states + attention_output)
        return hidden_states


# 从 transformers.models.bert.modeling_flax_bert.FlaxBertLayer 复制而来，将 Bert 替换为 XLMRoberta
class FlaxXLMRobertaLayer(nn.Module):
    config: XLMRobertaConfig  # XLMRoberta 模型的配置信息
    dtype: jnp.dtype = jnp.float32  # 计算时使用的数据类型

    def setup(self):
        # 初始化 self.attention 为 FlaxXLMRobertaAttention 实例
        self.attention = FlaxXLMRobertaAttention(self.config, causal=self.config.is_decoder, dtype=self.dtype)
        # 初始化 self.intermediate 为 FlaxXLMRobertaIntermediate 实例
        self.intermediate = FlaxXLMRobertaIntermediate(self.config, dtype=self.dtype)
        # 初始化 self.output 为 FlaxXLMRobertaOutput 实例
        self.output = FlaxXLMRobertaOutput(self.config, dtype=self.dtype)
        # 如果配置中包含交叉注意力，初始化 self.crossattention 为 FlaxXLMRobertaAttention 实例
        if self.config.add_cross_attention:
            self.crossattention = FlaxXLMRobertaAttention(self.config, causal=False, dtype=self.dtype)

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
        # 实现 FlaxXLMRobertaLayer 的调用功能，接收多个参数进行处理
        # （具体处理逻辑在实现该方法的类的调用实现中）
        pass  # 这里是函数体的结尾，没有实际的代码逻辑，因此不需要添加额外的注释
        # Self Attention
        # 使用 self.attention 方法进行自注意力计算，处理隐藏状态和注意力掩码
        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            layer_head_mask=layer_head_mask,
            init_cache=init_cache,
            deterministic=deterministic,
            output_attentions=output_attentions,
        )
        attention_output = attention_outputs[0]

        # Cross-Attention Block
        # 如果存在编码器的隐藏状态，则进行交叉注意力计算
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

        # 经过 self.intermediate 层的处理
        hidden_states = self.intermediate(attention_output)
        # 经过 self.output 层的处理，得到最终输出的隐藏状态
        hidden_states = self.output(hidden_states, attention_output, deterministic=deterministic)

        # 将隐藏状态打包成输出元组
        outputs = (hidden_states,)

        # 如果需要输出注意力信息
        if output_attentions:
            # 添加自注意力信息到输出元组
            outputs += (attention_outputs[1],)
            # 如果存在编码器的隐藏状态，则添加交叉注意力信息到输出元组
            if encoder_hidden_states is not None:
                outputs += (cross_attention_outputs[1],)

        # 返回最终的输出元组
        return outputs
# 从 transformers.models.bert.modeling_flax_bert.FlaxBertLayerCollection 复制并修改为 FlaxXLMRobertaLayerCollection
class FlaxXLMRobertaLayerCollection(nn.Module):
    config: XLMRobertaConfig  # 类型提示，指定配置对象为 XLMRobertaConfig 类型
    dtype: jnp.dtype = jnp.float32  # 计算中使用的数据类型，默认为 jnp.float32
    gradient_checkpointing: bool = False  # 是否使用梯度检查点，默认为 False

    def setup(self):
        if self.gradient_checkpointing:
            # 如果开启梯度检查点，使用 remat 函数对 FlaxXLMRobertaLayer 进行重建
            FlaxXLMRobertaCheckpointLayer = remat(FlaxXLMRobertaLayer, static_argnums=(5, 6, 7))
            # 创建一个包含检查点层的列表，每层的名称为索引号字符串
            self.layers = [
                FlaxXLMRobertaCheckpointLayer(self.config, name=str(i), dtype=self.dtype)
                for i in range(self.config.num_hidden_layers)
            ]
        else:
            # 如果未开启梯度检查点，创建一个 FlaxXLMRobertaLayer 的列表，每层的名称为索引号字符串
            self.layers = [
                FlaxXLMRobertaLayer(self.config, name=str(i), dtype=self.dtype)
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
        # 神经网络层的调用方法，接受多个输入参数和一些可选的布尔值参数
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        # 检查是否需要创建头部遮罩（head_mask），确保头部遮罩的层数与模型层数一致
        if head_mask is not None:
            if head_mask.shape[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.shape[0]}."
                )

        # 遍历模型的每一层并进行前向传播
        for i, layer in enumerate(self.layers):
            # 如果需要输出隐藏状态，则将当前层的隐藏状态加入到列表中
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # 调用当前层的前向传播函数，获取当前层的输出
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

            # 更新当前层的隐藏状态
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力权重，将当前层的注意力权重加入到列表中
            if output_attentions:
                all_attentions += (layer_outputs[1],)

                # 如果存在编码器的隐藏状态，将当前层的交叉注意力权重加入到列表中
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # 如果需要输出隐藏状态，将最后一层的隐藏状态加入到列表中
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 构建模型的输出
        outputs = (hidden_states, all_hidden_states, all_attentions, all_cross_attentions)

        # 如果不需要以字典形式返回结果，则返回元组形式的输出
        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        # 以带过去和交叉注意力的 Flax 模型输出格式返回结果
        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )
# 从transformers.models.bert.modeling_flax_bert.FlaxBertEncoder复制代码，并将Bert->XLMRoberta
class FlaxXLMRobertaEncoder(nn.Module):
    config: XLMRobertaConfig
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型
    gradient_checkpointing: bool = False  # 是否使用梯度检查点

    def setup(self):
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


# 从transformers.models.bert.modeling_flax_bert.FlaxBertPooler复制代码，并将Bert->XLMRoberta
class FlaxXLMRobertaPooler(nn.Module):
    config: XLMRobertaConfig
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )

    def __call__(self, hidden_states):
        cls_hidden_state = hidden_states[:, 0]  # 取第一个位置的CLS隐藏状态
        cls_hidden_state = self.dense(cls_hidden_state)  # 通过全连接层进行处理
        return nn.tanh(cls_hidden_state)  # 返回经过tanh激活的CLS隐藏状态


# 从transformers.models.roberta.modeling_flax_roberta.FlaxRobertaLMHead复制代码，并将Roberta->XLMRoberta
class FlaxXLMRobertaLMHead(nn.Module):
    config: XLMRobertaConfig
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型
    bias_init: Callable[..., np.ndarray] = jax.nn.initializers.zeros  # 偏置初始化函数

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)  # 层归一化
        self.decoder = nn.Dense(
            self.config.vocab_size,
            dtype=self.dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.bias = self.param("bias", self.bias_init, (self.config.vocab_size,))  # 偏置参数
    # 定义一个对象的调用方法，接受隐藏状态和共享嵌入作为参数
    def __call__(self, hidden_states, shared_embedding=None):
        # 将隐藏状态通过全连接层进行线性变换
        hidden_states = self.dense(hidden_states)
        # 使用 GELU 激活函数处理隐藏状态
        hidden_states = ACT2FN["gelu"](hidden_states)
        # 对处理后的隐藏状态进行 Layer Normalization
        hidden_states = self.layer_norm(hidden_states)

        # 如果提供了共享的嵌入向量，则将其作为参数应用到解码器中
        if shared_embedding is not None:
            hidden_states = self.decoder.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
        else:
            # 否则，直接使用解码器处理隐藏状态
            hidden_states = self.decoder(hidden_states)

        # 将偏置转换为 JAX 数组，并加到隐藏状态上
        bias = jnp.asarray(self.bias, self.dtype)
        hidden_states += bias
        # 返回处理后的隐藏状态
        return hidden_states
# 从 transformers.models.roberta.modeling_flax_roberta.FlaxRobertaClassificationHead 复制而来，将 Roberta 替换为 XLMRoberta
class FlaxXLMRobertaClassificationHead(nn.Module):
    config: XLMRobertaConfig  # 类的配置信息，使用 XLMRobertaConfig
    dtype: jnp.dtype = jnp.float32  # 数据类型设置为 jnp.float32

    def setup(self):
        # 初始化一个全连接层，输出大小为 config.hidden_size，使用正态分布初始化权重
        self.dense = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        # 设置分类器的 dropout 率为 config.classifier_dropout，如果为 None，则使用 config.hidden_dropout_prob
        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(rate=classifier_dropout)  # 设置 dropout 层
        # 初始化一个全连接层，输出大小为 config.num_labels，使用正态分布初始化权重
        self.out_proj = nn.Dense(
            self.config.num_labels,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

    def __call__(self, hidden_states, deterministic=True):
        hidden_states = hidden_states[:, 0, :]  # 取 <s> 标记对应的隐藏状态 (等同于 [CLS])
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)  # 应用 dropout
        hidden_states = self.dense(hidden_states)  # 应用全连接层
        hidden_states = nn.tanh(hidden_states)  # 应用 tanh 激活函数
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)  # 再次应用 dropout
        hidden_states = self.out_proj(hidden_states)  # 应用输出投影层
        return hidden_states


# 从 transformers.models.roberta.modeling_flax_roberta.FlaxRobertaPreTrainedModel 复制而来，将 Roberta 替换为 XLMRoberta，roberta 替换为 xlm-roberta，ROBERTA 替换为 XLM_ROBERTA
class FlaxXLMRobertaPreTrainedModel(FlaxPreTrainedModel):
    """
    处理权重初始化和简单接口以下载和加载预训练模型的抽象类。
    """

    config_class = XLMRobertaConfig  # 配置类为 XLMRobertaConfig
    base_model_prefix = "xlm-roberta"  # 基础模型前缀为 "xlm-roberta"

    module_class: nn.Module = None  # 模块类设置为 None

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
        # 初始化一个模块类对象，使用给定的配置、数据类型和参数
        module = self.module_class(config=config, dtype=dtype, gradient_checkpointing=gradient_checkpointing, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    # 从 transformers.models.bert.modeling_flax_bert.FlaxBertPreTrainedModel.enable_gradient_checkpointing 复制而来
    def enable_gradient_checkpointing(self):
        self._module = self.module_class(
            config=self.config,
            dtype=self.dtype,
            gradient_checkpointing=True,
        )
    # 初始化模型的权重
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量
        input_ids = jnp.zeros(input_shape, dtype="i4")  # 创建全零的输入张量
        token_type_ids = jnp.ones_like(input_ids)  # 创建与输入张量形状相同的全一张量
        position_ids = create_position_ids_from_input_ids(input_ids, self.config.pad_token_id)  # 根据输入张量创建位置编码
        attention_mask = jnp.ones_like(input_ids)  # 创建与输入张量形状相同的全一注意力掩码
        head_mask = jnp.ones((self.config.num_hidden_layers, self.config.num_attention_heads))  # 创建全一的头部掩码

        # 分割随机数生成器
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        if self.config.add_cross_attention:
            # 如果配置要求添加交叉注意力，初始化编码器隐藏状态和注意力掩码
            encoder_hidden_states = jnp.zeros(input_shape + (self.config.hidden_size,))
            encoder_attention_mask = attention_mask
            # 使用模型初始化，并返回模型初始化的输出
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
            # 否则，使用模型初始化，仅传入基本参数
            module_init_outputs = self.module.init(
                rngs, input_ids, attention_mask, token_type_ids, position_ids, head_mask, return_dict=False
            )

        # 从模型初始化的输出中获取随机参数
        random_params = module_init_outputs["params"]

        if params is not None:
            # 如果提供了预定义参数，则将随机参数展开并填充缺失的键
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))  # 冻结填充后的参数并返回
        else:
            return random_params  # 否则，返回随机初始化的参数

    # 从 transformers.models.bart.modeling_flax_bart.FlaxBartDecoderPreTrainedModel.init_cache 复制过来的方法
    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                用于快速自回归解码的批大小。定义初始化缓存时的批大小。
            max_length (`int`):
                自回归解码的最大可能长度。定义初始化缓存的序列长度。
        """
        # 初始化用于检索缓存的输入变量
        input_ids = jnp.ones((batch_size, max_length), dtype="i4")  # 创建全一的输入张量
        attention_mask = jnp.ones_like(input_ids, dtype="i4")  # 创建与输入张量形状相同的全一注意力掩码
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)  # 广播位置编码

        # 使用模型初始化，并返回初始化变量的缓存部分
        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        return unfreeze(init_variables["cache"])

    @add_start_docstrings_to_model_forward(XLM_ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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


        # 定义一个调用方法，接收多个输入参数，以下为详细参数解释

        # 必须的输入参数，表示模型的输入 token IDs
        input_ids,

        # 可选的输入参数，表示注意力遮罩，用于指示哪些标记是有效的
        attention_mask=None,

        # 可选的输入参数，表示标记类型的 IDs，通常在多段文本输入时使用
        token_type_ids=None,

        # 可选的输入参数，表示标记在序列中的位置 IDs
        position_ids=None,

        # 可选的输入参数，表示头部遮罩，用于指示哪些注意力头部是有效的
        head_mask=None,

        # 可选的输入参数，表示编码器的隐藏状态
        encoder_hidden_states=None,

        # 可选的输入参数，表示编码器注意力遮罩，用于指示哪些编码器隐藏状态是有效的
        encoder_attention_mask=None,

        # 可选的输入参数，表示额外的参数字典，用于模型配置
        params: dict = None,

        # 可选的输入参数，表示随机数生成器密钥，用于 dropout 操作
        dropout_rng: jax.random.PRNGKey = None,

        # 可选的输入参数，表示是否处于训练模式
        train: bool = False,

        # 可选的输入参数，表示是否输出注意力权重
        output_attentions: Optional[bool] = None,

        # 可选的输入参数，表示是否输出隐藏状态
        output_hidden_states: Optional[bool] = None,

        # 可选的输入参数，表示是否返回一个字典对象
        return_dict: Optional[bool] = None,

        # 可选的输入参数，表示过去的键值状态字典
        past_key_values: dict = None,
# 从transformers.models.bert.modeling_flax_bert.FlaxBertModule复制代码，并将Bert->XLMRoberta
class FlaxXLMRobertaModule(nn.Module):
    # 使用XLMRobertaConfig配置
    config: XLMRobertaConfig
    # 计算时的数据类型
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    # 是否添加池化层，默认为True
    add_pooling_layer: bool = True
    # 是否使用梯度检查点
    gradient_checkpointing: bool = False

    def setup(self):
        # 初始化嵌入层
        self.embeddings = FlaxXLMRobertaEmbeddings(self.config, dtype=self.dtype)
        # 初始化编码器
        self.encoder = FlaxXLMRobertaEncoder(
            self.config,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 初始化池化层
        self.pooler = FlaxXLMRobertaPooler(self.config, dtype=self.dtype)

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
        # 确保当token_type_ids未传入时被正确初始化为全零数组
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        # 确保当position_ids未传入时被正确初始化
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        # 通过嵌入层计算隐藏状态
        hidden_states = self.embeddings(
            input_ids, token_type_ids, position_ids, attention_mask, deterministic=deterministic
        )
        # 使用编码器计算输出
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
        # 获取编码器的隐藏状态
        hidden_states = outputs[0]
        # 如果需要添加池化层，则计算池化结果
        pooled = self.pooler(hidden_states) if self.add_pooling_layer else None

        # 如果不需要返回字典形式的结果
        if not return_dict:
            # 如果池化结果为None，则不返回它
            if pooled is None:
                return (hidden_states,) + outputs[1:]
            return (hidden_states, pooled) + outputs[1:]

        # 返回包含池化结果和交叉注意力的FlaxBaseModelOutputWithPoolingAndCrossAttentions对象
        return FlaxBaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )
    XLM_ROBERTA_START_DOCSTRING,



# 引用预定义的常量 XLM_ROBERTA_START_DOCSTRING
)
class FlaxXLMRobertaModel(FlaxXLMRobertaPreTrainedModel):
    module_class = FlaxXLMRobertaModule


append_call_sample_docstring(FlaxXLMRobertaModel, _CHECKPOINT_FOR_DOC, FlaxBaseModelOutputWithPooling, _CONFIG_FOR_DOC)


# 从 transformers.models.roberta.modeling_flax_roberta.FlaxRobertaForMaskedLMModule 复制并修改为 XLMRoberta
class FlaxXLMRobertaForMaskedLMModule(nn.Module):
    config: XLMRobertaConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        # 初始化 XLM-Roberta 模型，配置为不添加池化层，使用指定数据类型和梯度检查点
        self.roberta = FlaxXLMRobertaModule(
            config=self.config,
            add_pooling_layer=False,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 初始化 XLM-Roberta 语言模型头部
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
        # 调用 XLM-Roberta 模型
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
        # 如果配置指定共享词嵌入，则获取共享的词嵌入
        if self.config.tie_word_embeddings:
            shared_embedding = self.roberta.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
        else:
            shared_embedding = None

        # 计算预测分数
        logits = self.lm_head(hidden_states, shared_embedding=shared_embedding)

        if not return_dict:
            return (logits,) + outputs[1:]

        # 返回 XLM-Roberta 遮蔽语言建模的输出
        return FlaxMaskedLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings("""XLM RoBERTa Model with a `language modeling` head on top.""", XLM_ROBERTA_START_DOCSTRING)
class FlaxXLMRobertaForMaskedLM(FlaxXLMRobertaPreTrainedModel):
    module_class = FlaxXLMRobertaForMaskedLMModule


append_call_sample_docstring(
    FlaxXLMRobertaForMaskedLM,
    _CHECKPOINT_FOR_DOC,
    FlaxBaseModelOutputWithPooling,
    _CONFIG_FOR_DOC,
    mask="<mask>",
)


# 从 transformers.models.roberta.modeling_flax_roberta.FlaxRobertaForSequenceClassificationModule 复制并修改为 XLMRoberta
class FlaxXLMRobertaForSequenceClassificationModule(nn.Module):
    config: XLMRobertaConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False
    # 在对象初始化时设置模型结构
    def setup(self):
        self.roberta = FlaxXLMRobertaModule(
            config=self.config,                     # 使用给定配置初始化模型
            dtype=self.dtype,                       # 设定数据类型
            add_pooling_layer=False,                # 禁用池化层
            gradient_checkpointing=self.gradient_checkpointing,  # 设置梯度检查点
        )
        self.classifier = FlaxXLMRobertaClassificationHead(config=self.config, dtype=self.dtype)  # 初始化分类头部模块

    # 对象调用时执行的函数，用于模型推断
    def __call__(
        self,
        input_ids,                                  # 输入的token id序列
        attention_mask,                             # 注意力掩码
        token_type_ids,                             # token类型id
        position_ids,                               # 位置id
        head_mask,                                  # 头部掩码
        deterministic: bool = True,                 # 是否使用确定性计算
        output_attentions: bool = False,            # 是否输出注意力权重
        output_hidden_states: bool = False,         # 是否输出隐藏状态
        return_dict: bool = True,                   # 是否返回字典形式的结果
    ):
        # 执行RoBERTa模型的前向传播
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

        sequence_output = outputs[0]                # 提取序列输出
        logits = self.classifier(sequence_output, deterministic=deterministic)  # 使用分类头部预测logits

        if not return_dict:
            return (logits,) + outputs[1:]          # 返回tuple形式的输出

        return FlaxSequenceClassifierOutput(
            logits=logits,                          # 返回分类的logits
            hidden_states=outputs.hidden_states,    # 返回隐藏状态
            attentions=outputs.attentions,          # 返回注意力权重
        )
@add_start_docstrings(
    """
    XLM Roberta Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    XLM_ROBERTA_START_DOCSTRING,
)
"""
XLM Roberta模型转换器，顶部带有序列分类/回归头部（即池化输出的顶部线性层），例如用于GLUE任务。
"""

append_call_sample_docstring(
    FlaxXLMRobertaForSequenceClassification,
    _CHECKPOINT_FOR_DOC,
    FlaxSequenceClassifierOutput,
    _CONFIG_FOR_DOC,
)
"""
将示例调用的文档字符串附加到FlaxXLMRobertaForSequenceClassification类的文档中，
包括_CHECKPOINT_FOR_DOC、FlaxSequenceClassifierOutput和_CONFIG_FOR_DOC。
"""

# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertForMultipleChoiceModule with Bert->XLMRoberta, with self.bert->self.roberta
"""
从transformers.models.bert.modeling_flax_bert.FlaxBertForMultipleChoiceModule复制，
将Bert替换为XLMRoberta，将self.bert替换为self.roberta。
"""
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
"""
XLM Roberta模型，带有多选分类头部（即池化输出的顶部线性层和）。
"""

@add_start_docstrings(
    """
    XLM Roberta Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
    """,
    XLM_ROBERTA_START_DOCSTRING,
)
"""
XLM Roberta模型，带有多选分类头部（即池化输出的顶部线性层和）
"""
    a softmax) e.g. for RocStories/SWAG tasks.
    """
    XLM-RoBERTa 模型的起始文档字符串，用于生成模型文档说明。
    """
)
class FlaxXLMRobertaForMultipleChoice(FlaxXLMRobertaPreTrainedModel):
    module_class = FlaxXLMRobertaForMultipleChoiceModule


overwrite_call_docstring(
    FlaxXLMRobertaForMultipleChoice, XLM_ROBERTA_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
)
append_call_sample_docstring(
    FlaxXLMRobertaForMultipleChoice,
    _CHECKPOINT_FOR_DOC,
    FlaxMultipleChoiceModelOutput,
    _CONFIG_FOR_DOC,
)



# 从FlaxBertForTokenClassificationModule复制并修改为FlaxXLMRobertaForTokenClassificationModule，将self.bert->self.roberta
class FlaxXLMRobertaForTokenClassificationModule(nn.Module):
    config: XLMRobertaConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        # 初始化时创建FlaxXLMRobertaModule实例，并传入配置、数据类型、是否梯度检查点、不添加池化层
        self.roberta = FlaxXLMRobertaModule(
            config=self.config,
            dtype=self.dtype,
            add_pooling_layer=False,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 根据配置设置分类器的dropout率，若未指定则使用隐藏层的dropout率
        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )
        # 创建一个dropout层，用于隐藏状态
        self.dropout = nn.Dropout(rate=classifier_dropout)
        # 创建一个全连接层，输出维度为配置文件中指定的标签数
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
        # 调用self.roberta进行模型推断
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

        # 从输出中获取隐藏状态，并在推断时使用dropout层
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 使用分类器预测标签
        logits = self.classifier(hidden_states)

        # 若return_dict为False，则返回元组形式的输出
        if not return_dict:
            return (logits,) + outputs[1:]

        # 否则返回FlaxTokenClassifierOutput对象，包含logits、隐藏状态和注意力机制
        return FlaxTokenClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    XLM Roberta模型，顶部带有一个标记分类头（即隐藏状态输出的线性层），例如用于命名实体识别（NER）任务。
    """,
    XLM_ROBERTA_START_DOCSTRING,
)
class FlaxXLMRobertaForTokenClassification(FlaxXLMRobertaPreTrainedModel):
    module_class = FlaxXLMRobertaForTokenClassificationModule


append_call_sample_docstring(
    FlaxXLMRobertaForTokenClassification,
    _CHECKPOINT_FOR_DOC,
    FlaxTokenClassifierOutput,
    _CONFIG_FOR_DOC,
)
# 从 transformers.models.bert.modeling_flax_bert.FlaxBertForQuestionAnsweringModule 复制代码到这里，并将 Bert->XLMRoberta，self.bert->self.roberta
class FlaxXLMRobertaForQuestionAnsweringModule(nn.Module):
    # 使用 XLMRobertaConfig 配置类
    config: XLMRobertaConfig
    # 数据类型默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 是否使用梯度检查点，默认为 False
    gradient_checkpointing: bool = False

    def setup(self):
        # 初始化 self.roberta 为 FlaxXLMRobertaModule
        self.roberta = FlaxXLMRobertaModule(
            config=self.config,
            dtype=self.dtype,
            add_pooling_layer=False,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 初始化 self.qa_outputs 为 nn.Dense，输出大小为 self.config.num_labels
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
        # 调用 self.roberta 进行模型计算
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

        # 从模型输出中获取隐藏状态
        hidden_states = outputs[0]

        # 计算起始和结束位置的 logits
        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = jnp.split(logits, self.config.num_labels, axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # 如果 return_dict 为 False，则返回 tuple 类型
        if not return_dict:
            return (start_logits, end_logits) + outputs[1:]

        # 如果 return_dict 为 True，则返回 FlaxQuestionAnsweringModelOutput 类型
        return FlaxQuestionAnsweringModelOutput(
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    XLM Roberta Model with a span classification head on top for extractive question-answering tasks like SQuAD (a
    linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    XLM_ROBERTA_START_DOCSTRING,
)
# 继承自 FlaxXLMRobertaPreTrainedModel 的 XLMRoberta 问题回答模型类
class FlaxXLMRobertaForQuestionAnswering(FlaxXLMRobertaPreTrainedModel):
    # 指定模块类为 FlaxXLMRobertaForQuestionAnsweringModule
    module_class = FlaxXLMRobertaForQuestionAnsweringModule


# 从 transformers.models.roberta.modeling_flax_roberta.FlaxRobertaForCausalLMModule 复制代码到这里，并将 Roberta->XLMRoberta
class FlaxXLMRobertaForCausalLMModule(nn.Module):
    # 使用 XLMRobertaConfig 配置类
    config: XLMRobertaConfig
    # 数据类型默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 是否使用梯度检查点，默认为 False
    gradient_checkpointing: bool = False
    # 在模型设置方法中初始化 RoBERTa 模型和语言模型头部
    def setup(self):
        self.roberta = FlaxXLMRobertaModule(
            config=self.config,
            add_pooling_layer=False,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        self.lm_head = FlaxXLMRobertaLMHead(config=self.config, dtype=self.dtype)

    # 在调用方法中执行模型的前向传播
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
        # 调用 RoBERTa 模型的前向传播，并传入所有必要的参数
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

        # 获取模型的隐藏状态作为输入特征
        hidden_states = outputs[0]

        # 根据配置决定是否共享词嵌入层
        if self.config.tie_word_embeddings:
            shared_embedding = self.roberta.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
        else:
            shared_embedding = None

        # 计算语言模型头部的预测分数
        logits = self.lm_head(hidden_states, shared_embedding=shared_embedding)

        # 如果不要求返回字典形式的输出，则返回元组
        if not return_dict:
            return (logits,) + outputs[1:]

        # 返回带有交叉注意力的因果语言建模输出
        return FlaxCausalLMOutputWithCrossAttentions(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )
# 使用装饰器为类添加文档字符串，描述该类是在 XLM Roberta 模型基础上添加了语言建模头部的变体，用于自回归任务
@add_start_docstrings(
    """
    XLM Roberta Model with a language modeling head on top (a linear layer on top of the hidden-states output) e.g for
    autoregressive tasks.
    """,
    XLM_ROBERTA_START_DOCSTRING,
)
# 从 transformers.models.roberta.modeling_flax_roberta.FlaxRobertaForCausalLM 复制过来，并将 Roberta 改为 XLMRoberta
class FlaxXLMRobertaForCausalLM(FlaxXLMRobertaPreTrainedModel):
    # 使用 FlaxXLMRobertaForCausalLMModule 作为模块类
    module_class = FlaxXLMRobertaForCausalLMModule

    # 为生成准备输入的方法，接受输入的 token IDs，生成最大长度的序列
    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = None):
        # 初始化缓存
        batch_size, seq_length = input_ids.shape

        # 使用 self.init_cache 方法初始化过去的键值对
        past_key_values = self.init_cache(batch_size, max_length)
        
        # 注意：通常需要在 attention_mask 中为 x > input_ids.shape[-1] 和 x < cache_length 的位置放置 0
        # 但由于解码器使用因果蒙版，这些位置已经被蒙版了。因此，我们可以在这里创建一个静态的 attention_mask，这对编译更有效。
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        
        # 如果提供了 attention_mask，则根据其累积和更新 extended_attention_mask
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            # 否则，广播生成一个 position_ids，形状为 (batch_size, seq_length)
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    # 更新生成时的输入，将模型输出的 past_key_values 和 position_ids 更新到 model_kwargs 中
    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs

# 将样例调用文档字符串附加到 FlaxXLMRobertaForCausalLM 类上，描述如何调用该类以生成样本
append_call_sample_docstring(
    FlaxXLMRobertaForCausalLM,
    _CHECKPOINT_FOR_DOC,
    FlaxCausalLMOutputWithCrossAttentions,
    _CONFIG_FOR_DOC,
)
```