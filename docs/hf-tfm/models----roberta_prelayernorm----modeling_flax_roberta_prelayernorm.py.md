# `.\models\roberta_prelayernorm\modeling_flax_roberta_prelayernorm.py`

```
# coding=utf-8
# Copyright 2022 The Google Flax Team Authors and The HuggingFace Inc. team.
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
""" Flax RoBERTa-PreLayerNorm model."""
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
from .configuration_roberta_prelayernorm import RobertaPreLayerNormConfig

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "andreasmadsen/efficient_mlm_m0.40"
_CONFIG_FOR_DOC = "RobertaPreLayerNormConfig"

remat = nn_partitioning.remat


# Copied from transformers.models.roberta.modeling_flax_roberta.create_position_ids_from_input_ids
def create_position_ids_from_input_ids(input_ids, padding_idx):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        input_ids: jnp.ndarray
        padding_idx: int

    Returns: jnp.ndarray
    """
    # Create a mask where non-padding symbols are marked as 1 and padding symbols as 0
    mask = (input_ids != padding_idx).astype("i4")

    # Reshape mask if it has more than 2 dimensions
    if mask.ndim > 2:
        mask = mask.reshape((-1, mask.shape[-1]))

    # Calculate cumulative sum along the last dimension of the mask
    incremental_indices = jnp.cumsum(mask, axis=1).astype("i4") * mask

    # Reshape incremental indices to match the shape of input_ids
    incremental_indices = incremental_indices.reshape(input_ids.shape)

    # Add padding_idx to the incremental indices to get final position ids
    return incremental_indices.astype("i4") + padding_idx
# ROBERTA_PRELAYERNORM_START_DOCSTRING 字符串常量，包含关于 RobertaPreLayerNormModel 模型的详细文档字符串。
ROBERTA_PRELAYERNORM_START_DOCSTRING = r"""

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
        config ([`RobertaPreLayerNormConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.
"""

# ROBERTA_PRELAYERNORM_INPUTS_DOCSTRING 字符串常量，包含关于 RobertaPreLayerNormModel 模型输入的文档字符串，目前为空。
ROBERTA_PRELAYERNORM_INPUTS_DOCSTRING = r"""
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
# 从transformers.models.bert.modeling_flax_bert.FlaxBertEmbeddings复制并修改为RobertaPreLayerNorm
class FlaxRobertaPreLayerNormEmbeddings(nn.Module):
    """从单词、位置和标记类型嵌入构建嵌入。"""

    config: RobertaPreLayerNormConfig
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型

    def setup(self):
        # 初始化单词嵌入，使用正态分布初始化方法
        self.word_embeddings = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        # 初始化位置嵌入，使用正态分布初始化方法
        self.position_embeddings = nn.Embed(
            self.config.max_position_embeddings,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        # 初始化标记类型嵌入，使用正态分布初始化方法
        self.token_type_embeddings = nn.Embed(
            self.config.type_vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        # 初始化 LayerNorm 层
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 初始化 Dropout 层
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, input_ids, token_type_ids, position_ids, attention_mask, deterministic: bool = True):
        # 嵌入输入的 ids
        inputs_embeds = self.word_embeddings(input_ids.astype("i4"))
        # 嵌入位置 ids
        position_embeds = self.position_embeddings(position_ids.astype("i4"))
        # 嵌入标记类型 ids
        token_type_embeddings = self.token_type_embeddings(token_type_ids.astype("i4"))

        # 汇总所有嵌入
        hidden_states = inputs_embeds + token_type_embeddings + position_embeds

        # Layer Norm 层
        hidden_states = self.LayerNorm(hidden_states)
        # Dropout 层
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states


# 从transformers.models.bert.modeling_flax_bert.FlaxBertSelfAttention复制并修改为RobertaPreLayerNorm
class FlaxRobertaPreLayerNormSelfAttention(nn.Module):
    config: RobertaPreLayerNormConfig
    causal: bool = False
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型
    # 在对象初始化时设置方法
    def setup(self):
        # 计算每个注意力头的维度
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        # 检查隐藏层大小是否能被注意力头数整除
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError(
                "`config.hidden_size`: {self.config.hidden_size} has to be a multiple of `config.num_attention_heads` "
                "                   : {self.config.num_attention_heads}"
            )

        # 初始化查询、键、值的全连接层
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

        # 如果是因果注意力，则创建因果掩码
        if self.causal:
            self.causal_mask = make_causal_mask(
                jnp.ones((1, self.config.max_position_embeddings), dtype="bool"), dtype="bool"
            )

    # 将隐藏状态分割成多个注意力头
    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.config.num_attention_heads, self.head_dim))

    # 将分割后的注意力头合并成原始隐藏状态的形状
    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.config.hidden_size,))

    @nn.compact
    # 从transformers.models.bart.modeling_flax_bart.FlaxBartAttention._concatenate_to_cache复制而来的注释
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slighly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # 检测是否初始化缓存数据
        is_initialized = self.has_variable("cache", "cached_key")
        # 获取或创建缓存的键值和值的变量，如果不存在则创建新的变量
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        # 获取或创建缓存索引的变量，如果不存在则初始化为0
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            # 获取批次维度等信息，并获取当前缓存键的最大长度、头数和每个头的深度
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # 使用新的一维空间片段更新键值缓存
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            # 更新缓存索引以反映新增的缓存向量数目
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # 用于缓存解码器自注意力的因果掩码：我们的单个查询位置只应关注已生成并缓存的键位置，而不是剩余的零元素
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            # 合并填充掩码和注意力掩码
            attention_mask = combine_masks(pad_mask, attention_mask)
        # 返回更新后的键、值和注意力掩码
        return key, value, attention_mask
# 定义 FlaxRobertaPreLayerNormSelfOutput 类，继承自 nn.Module
class FlaxRobertaPreLayerNormSelfOutput(nn.Module):
    # 类型注解，指定 config 属性为 RobertaPreLayerNormConfig 类型
    config: RobertaPreLayerNormConfig
    # 定义 dtype 属性，默认为 jnp.float32，表示计算中使用的数据类型
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    # 初始化函数，在模块设置时调用
    def setup(self):
        # 初始化 dense 层，输出维度为 self.config.hidden_size
        # 使用正态分布初始化权重，范围为 self.config.initializer_range
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 初始化 dropout 层，丢弃率为 self.config.hidden_dropout_prob
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    # 调用函数，定义模块的前向传播逻辑
    def __call__(self, hidden_states, input_tensor, deterministic: bool = True):
        # 输入 hidden_states 经过 dense 层处理
        hidden_states = self.dense(hidden_states)
        # 对处理后的 hidden_states 进行 dropout 操作
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 将 dropout 后的 hidden_states 与 input_tensor 相加作为输出
        hidden_states = hidden_states + input_tensor
        return hidden_states


# 定义 FlaxRobertaPreLayerNormAttention 类，继承自 nn.Module
class FlaxRobertaPreLayerNormAttention(nn.Module):
    # 类型注解，指定 config 属性为 RobertaPreLayerNormConfig 类型
    config: RobertaPreLayerNormConfig
    # 定义 causal 属性，默认为 False，表示是否是因果关系的自注意力
    causal: bool = False
    # 定义 dtype 属性，默认为 jnp.float32，表示计算中使用的数据类型
    dtype: jnp.dtype = jnp.float32

    # 初始化函数，在模块设置时调用
    def setup(self):
        # 初始化 self 层为 FlaxRobertaPreLayerNormSelfAttention 类实例
        self.self = FlaxRobertaPreLayerNormSelfAttention(self.config, causal=self.causal, dtype=self.dtype)
        # 初始化 output 层为 FlaxRobertaPreLayerNormSelfOutput 类实例
        self.output = FlaxRobertaPreLayerNormSelfOutput(self.config, dtype=self.dtype)
        # 初始化 LayerNorm 层，用于层归一化，epsilon 为 self.config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    # 调用函数，定义模块的前向传播逻辑
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
        # 对 hidden_states 进行层归一化
        hidden_states_pre_layer_norm = self.LayerNorm(hidden_states)
        # 调用 self 层的前向传播函数，处理层归一化后的 hidden_states
        # 返回的 attn_outputs 包含注意力输出和可能的附加信息，如注意力权重
        attn_outputs = self.self(
            hidden_states_pre_layer_norm,
            attention_mask,
            layer_head_mask=layer_head_mask,
            key_value_states=key_value_states,
            init_cache=init_cache,
            deterministic=deterministic,
            output_attentions=output_attentions,
        )
        # 从 attn_outputs 中获取注意力输出
        attn_output = attn_outputs[0]
        # 调用 output 层的前向传播函数，处理注意力输出和原始 hidden_states
        hidden_states = self.output(attn_output, hidden_states, deterministic=deterministic)

        # 构建输出元组，包含更新后的 hidden_states
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则添加到输出元组中
        if output_attentions:
            outputs += (attn_outputs[1],)

        return outputs


# 定义 FlaxRobertaPreLayerNormIntermediate 类，继承自 nn.Module
class FlaxRobertaPreLayerNormIntermediate(nn.Module):
    # 类型注解，指定 config 属性为 RobertaPreLayerNormConfig 类型
    config: RobertaPreLayerNormConfig
    # 定义 dtype 属性，默认为 jnp.float32，表示计算中使用的数据类型
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    # 初始化函数，在模块设置时调用
    def setup(self):
        # 初始化 LayerNorm 层，用于层归一化，epsilon 为 self.config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 初始化 dense 层，输出维度为 self.config.intermediate_size
        # 使用正态分布初始化权重，范围为 self.config.initializer_range
        self.dense = nn.Dense(
            self.config.intermediate_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 激活函数 activation 为根据配置中的 hidden_act 从 ACT2FN 字典中选择的函数
        self.activation = ACT2FN[self.config.hidden_act]
    # 定义一个类方法，用于处理给定的隐藏状态数据
    def __call__(self, hidden_states):
        # 对隐藏状态进行层归一化操作
        hidden_states = self.LayerNorm(hidden_states)
        # 对归一化后的隐藏状态应用全连接层变换
        hidden_states = self.dense(hidden_states)
        # 对变换后的隐藏状态应用激活函数（如ReLU等）
        hidden_states = self.activation(hidden_states)
        # 返回处理后的隐藏状态作为结果
        return hidden_states
# 定义一个名为 FlaxRobertaPreLayerNormOutput 的新的神经网络模块（nn.Module）
class FlaxRobertaPreLayerNormOutput(nn.Module):
    # 配置对象，用于存储 RobertaPreLayerNormConfig 类的配置信息
    config: RobertaPreLayerNormConfig
    # 计算过程中所用的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    # 模块初始化方法
    def setup(self):
        # 定义一个全连接层，输出维度为 config.hidden_size，权重初始化为正态分布，范围为 config.initializer_range
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 定义一个 Dropout 层，丢弃率为 config.hidden_dropout_prob
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    # 模块调用方法，用于前向传播
    def __call__(self, hidden_states, attention_output, deterministic: bool = True):
        # 将输入的 hidden_states 通过全连接层 dense 进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的 hidden_states 应用 Dropout，用于随机丢弃部分神经元，防止过拟合
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 将 Dropout 后的 hidden_states 与 attention_output 相加，得到最终输出
        hidden_states = hidden_states + attention_output
        # 返回最终输出的 hidden_states
        return hidden_states


# 从 transformers.models.bert.modeling_flax_bert.FlaxBertLayer 复制的类，修改为使用 RobertaPreLayerNorm
class FlaxRobertaPreLayerNormLayer(nn.Module):
    # 配置对象，用于存储 RobertaPreLayerNormConfig 类的配置信息
    config: RobertaPreLayerNormConfig
    # 计算过程中所用的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    # 模块初始化方法
    def setup(self):
        # 定义自注意力层，使用 FlaxRobertaPreLayerNormAttention 模块，传入配置信息和数据类型
        self.attention = FlaxRobertaPreLayerNormAttention(self.config, causal=self.config.is_decoder, dtype=self.dtype)
        # 定义中间层，使用 FlaxRobertaPreLayerNormIntermediate 模块，传入配置信息和数据类型
        self.intermediate = FlaxRobertaPreLayerNormIntermediate(self.config, dtype=self.dtype)
        # 定义输出层，使用 FlaxRobertaPreLayerNormOutput 模块，传入配置信息和数据类型
        self.output = FlaxRobertaPreLayerNormOutput(self.config, dtype=self.dtype)
        # 如果配置中设置了添加交叉注意力，定义交叉注意力层，使用 FlaxRobertaPreLayerNormAttention 模块，传入配置信息和数据类型
        if self.config.add_cross_attention:
            self.crossattention = FlaxRobertaPreLayerNormAttention(self.config, causal=False, dtype=self.dtype)

    # 模块调用方法，用于前向传播
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
        # Self Attention
        # 调用 self.attention 方法进行自注意力计算
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
        # 如果提供了 encoder_hidden_states，则进行交叉注意力计算
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

        # 经过 intermediate 层的处理
        hidden_states = self.intermediate(attention_output)
        # 经过 output 层的处理
        hidden_states = self.output(hidden_states, attention_output, deterministic=deterministic)

        # 将 hidden_states 包装为 outputs 元组
        outputs = (hidden_states,)

        # 如果需要输出 attentions，则将 attention 输出包含在 outputs 中
        if output_attentions:
            outputs += (attention_outputs[1],)
            # 如果提供了 encoder_hidden_states，则将 cross-attention 输出也包含在 outputs 中
            if encoder_hidden_states is not None:
                outputs += (cross_attention_outputs[1],)
        return outputs
# 从transformers.models.bert.modeling_flax_bert.FlaxBertLayerCollection复制并修改为FlaxRobertaPreLayerNormLayerCollection类
class FlaxRobertaPreLayerNormLayerCollection(nn.Module):
    config: RobertaPreLayerNormConfig  # 类型提示，指定配置对象类型为RobertaPreLayerNormConfig
    dtype: jnp.dtype = jnp.float32  # 计算时所用的数据类型，默认为jnp.float32
    gradient_checkpointing: bool = False  # 是否使用梯度检查点

    def setup(self):
        if self.gradient_checkpointing:
            # 如果启用梯度检查点，则定义FlaxRobertaPreLayerNormCheckpointLayer，并传递静态参数索引
            FlaxRobertaPreLayerNormCheckpointLayer = remat(FlaxRobertaPreLayerNormLayer, static_argnums=(5, 6, 7))
            # 创建多个梯度检查点层对象，每个层对象对应一层神经网络层
            self.layers = [
                FlaxRobertaPreLayerNormCheckpointLayer(self.config, name=str(i), dtype=self.dtype)
                for i in range(self.config.num_hidden_layers)
            ]
        else:
            # 如果未启用梯度检查点，则创建多个Roberta预层归一化层对象，每个对象对应一层神经网络层
            self.layers = [
                FlaxRobertaPreLayerNormLayer(self.config, name=str(i), dtype=self.dtype)
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
            # 初始化空元组，根据需要输出注意力的设置确定是否包含
            all_attentions = () if output_attentions else None
            # 初始化空元组，根据需要输出隐藏状态的设置确定是否包含
            all_hidden_states = () if output_hidden_states else None
            # 初始化空元组，根据需要输出交叉注意力的设置和编码器隐藏状态是否存在确定是否包含
            all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

            # 检查头部掩码是否为每层指定了正确数量的层
            if head_mask is not None:
                if head_mask.shape[0] != (len(self.layers)):
                    raise ValueError(
                        f"The head_mask should be specified for {len(self.layers)} layers, but it is for                  "
                        f"       {head_mask.shape[0]}."
                    )

            # 遍历模型的每一层
            for i, layer in enumerate(self.layers):
                # 如果需要输出隐藏状态，则将当前隐藏状态添加到所有隐藏状态的元组中
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

                # 更新隐藏状态为当前层的输出的第一个元素
                hidden_states = layer_outputs[0]

                # 如果需要输出注意力权重，则将当前层的注意力权重添加到所有注意力的元组中
                if output_attentions:
                    all_attentions += (layer_outputs[1],)

                    # 如果存在编码器的隐藏状态，则将当前层的交叉注意力添加到所有交叉注意力的元组中
                    if encoder_hidden_states is not None:
                        all_cross_attentions += (layer_outputs[2],)

            # 如果需要输出隐藏状态，则将最终隐藏状态添加到所有隐藏状态的元组中
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # 组装最终的模型输出，包括隐藏状态、所有隐藏状态、所有注意力和所有交叉注意力
            outputs = (hidden_states, all_hidden_states, all_attentions, all_cross_attentions)

            # 如果不需要返回字典形式的输出，则返回非空元素的元组
            if not return_dict:
                return tuple(v for v in outputs if v is not None)

            # 返回具有过去和交叉注意力的 Flax 模型输出对象
            return FlaxBaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=all_attentions,
                cross_attentions=all_cross_attentions,
            )
# 从 transformers.models.bert.modeling_flax_bert.FlaxBertEncoder 复制并将 Bert 替换为 RobertaPreLayerNorm
class FlaxRobertaPreLayerNormEncoder(nn.Module):
    config: RobertaPreLayerNormConfig  # 使用 RobertaPreLayerNormConfig 配置
    dtype: jnp.dtype = jnp.float32  # 计算时使用的数据类型
    gradient_checkpointing: bool = False  # 梯度检查点标志，默认为 False

    def setup(self):
        # 初始化 FlaxRobertaPreLayerNormLayerCollection 层集合
        self.layer = FlaxRobertaPreLayerNormLayerCollection(
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
        # 调用 self.layer 进行编码器的前向传播
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


# 从 transformers.models.bert.modeling_flax_bert.FlaxBertPooler 复制并将 Bert 替换为 RobertaPreLayerNorm
class FlaxRobertaPreLayerNormPooler(nn.Module):
    config: RobertaPreLayerNormConfig  # 使用 RobertaPreLayerNormConfig 配置
    dtype: jnp.dtype = jnp.float32  # 计算时使用的数据类型

    def setup(self):
        # 初始化全连接层 dense
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )

    def __call__(self, hidden_states):
        # 取出每个序列的第一个隐藏状态作为 CLS 隐藏状态，然后经过全连接层 dense 和 tanh 函数处理后返回
        cls_hidden_state = hidden_states[:, 0]
        cls_hidden_state = self.dense(cls_hidden_state)
        return nn.tanh(cls_hidden_state)


# 从 transformers.models.roberta.modeling_flax_roberta.FlaxRobertaLMHead 复制并将 Roberta 替换为 RobertaPreLayerNorm
class FlaxRobertaPreLayerNormLMHead(nn.Module):
    config: RobertaPreLayerNormConfig  # 使用 RobertaPreLayerNormConfig 配置
    dtype: jnp.dtype = jnp.float32  # 计算时使用的数据类型
    bias_init: Callable[..., np.ndarray] = jax.nn.initializers.zeros  # 偏置初始化器为零初始化器

    def setup(self):
        # 初始化全连接层 dense、LayerNorm 层和解码器 dense
        self.dense = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.decoder = nn.Dense(
            self.config.vocab_size,
            dtype=self.dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.bias = self.param("bias", self.bias_init, (self.config.vocab_size,))
    # 定义一个特殊方法 __call__，用于将对象实例作为函数调用
    def __call__(self, hidden_states, shared_embedding=None):
        # 将隐藏状态通过全连接层 dense 进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对变换后的隐藏状态应用 GELU 激活函数
        hidden_states = ACT2FN["gelu"](hidden_states)
        # 对激活后的隐藏状态进行层归一化处理
        hidden_states = self.layer_norm(hidden_states)

        # 如果传入了共享的嵌入矩阵，则使用嵌入矩阵进行解码
        if shared_embedding is not None:
            # 使用 decoder 对象的 apply 方法应用共享的嵌入矩阵进行解码
            hidden_states = self.decoder.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
        else:
            # 否则，直接使用 decoder 对象进行解码
            hidden_states = self.decoder(hidden_states)

        # 将偏置转换为与当前数据类型匹配的 JAX 数组，并加到隐藏状态上
        bias = jnp.asarray(self.bias, self.dtype)
        hidden_states += bias
        # 返回处理后的隐藏状态作为结果
        return hidden_states
# Copied from transformers.models.roberta.modeling_flax_roberta.FlaxRobertaClassificationHead with Roberta->RobertaPreLayerNorm
class FlaxRobertaPreLayerNormClassificationHead(nn.Module):
    config: RobertaPreLayerNormConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 初始化一个全连接层，用于分类头部，输入维度是隐藏层大小
        self.dense = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        # 根据配置中的 dropout 概率设置 Dropout 层
        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(rate=classifier_dropout)
        # 初始化输出投影层，输出维度是标签数量
        self.out_proj = nn.Dense(
            self.config.num_labels,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

    def __call__(self, hidden_states, deterministic=True):
        # 取隐藏状态中的第一个 token 的表示，相当于 [CLS] 标志
        hidden_states = hidden_states[:, 0, :]
        # 应用 Dropout 层到隐藏状态，以减少过拟合
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 输入全连接层，对隐藏状态进行线性变换
        hidden_states = self.dense(hidden_states)
        # 应用 tanh 激活函数
        hidden_states = nn.tanh(hidden_states)
        # 再次应用 Dropout 层
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 输出投影层，生成最终的分类预测结果
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


# Copied from transformers.models.roberta.modeling_flax_roberta.FlaxRobertaPreTrainedModel with ROBERTA->ROBERTA_PRELAYERNORM,Roberta->RobertaPreLayerNorm,roberta->roberta_prelayernorm
class FlaxRobertaPreLayerNormPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 配置类为 RobertaPreLayerNormConfig
    config_class = RobertaPreLayerNormConfig
    # 基础模型前缀为 roberta_prelayernorm
    base_model_prefix = "roberta_prelayernorm"

    module_class: nn.Module = None

    def __init__(
        self,
        config: RobertaPreLayerNormConfig,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        gradient_checkpointing: bool = False,
        **kwargs,
    ):
        # 初始化模型实例，根据传入的配置和参数
        module = self.module_class(config=config, dtype=dtype, gradient_checkpointing=gradient_checkpointing, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    # Copied from transformers.models.bert.modeling_flax_bert.FlaxBertPreTrainedModel.enable_gradient_checkpointing
    def enable_gradient_checkpointing(self):
        # 启用梯度检查点，更新模型内部使用的模块实例
        self._module = self.module_class(
            config=self.config,
            dtype=self.dtype,
            gradient_checkpointing=True,
        )
    # 初始化权重函数，用于模型的参数初始化
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量
        input_ids = jnp.zeros(input_shape, dtype="i4")  # 创建全零张量，形状为input_shape，数据类型为32位整数
        token_type_ids = jnp.ones_like(input_ids)  # 创建与input_ids相同形状的全一张量
        position_ids = create_position_ids_from_input_ids(input_ids, self.config.pad_token_id)  # 根据input_ids创建位置编码张量
        attention_mask = jnp.ones_like(input_ids)  # 创建与input_ids相同形状的全一张量作为注意力掩码
        head_mask = jnp.ones((self.config.num_hidden_layers, self.config.num_attention_heads))  # 创建全一头掩码张量

        # 使用随机数生成器rng拆分参数rng和dropout_rng
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        if self.config.add_cross_attention:
            # 如果配置中包含跨注意力机制，则初始化编码器隐藏状态和注意力掩码
            encoder_hidden_states = jnp.zeros(input_shape + (self.config.hidden_size,))
            encoder_attention_mask = attention_mask
            # 调用模块的初始化函数，返回非字典形式的初始化输出
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
            # 否则，只使用基本输入初始化模块
            module_init_outputs = self.module.init(
                rngs, input_ids, attention_mask, token_type_ids, position_ids, head_mask, return_dict=False
            )

        random_params = module_init_outputs["params"]  # 获取模块初始化输出中的随机参数

        if params is not None:
            # 如果提供了预训练参数，则进行参数的扁平化和解冻操作
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))  # 返回冻结的参数字典
        else:
            return random_params  # 否则，返回随机初始化的参数

    # 从transformers库中复制的初始化缓存函数，用于自回归解码
    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                用于快速自回归解码的批大小。定义了初始化缓存的批大小。
            max_length (`int`):
                自回归解码的最大可能长度。定义了初始化缓存的序列长度。
        """
        # 初始化用于检索缓存的输入变量
        input_ids = jnp.ones((batch_size, max_length), dtype="i4")  # 创建全一张量作为输入ids，形状为(batch_size, max_length)
        attention_mask = jnp.ones_like(input_ids, dtype="i4")  # 创建与input_ids相同形状的全一张量作为注意力掩码
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)
        # 根据input_ids的维度广播位置ids张量

        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        return unfreeze(init_variables["cache"])  # 返回解冻的初始化缓存变量

    # 为模型前向传播添加文档字符串，来自ROBERTA_PRELAYERNORM_INPUTS_DOCSTRING格式化为模型前向传播
    # 定义一个特殊方法 __call__，用于将实例作为可调用对象使用
    self,
    # 输入参数 input_ids：用于表示输入序列的 token IDs
    input_ids,
    # attention_mask：用于指定哪些 token 应该被忽略（1 表示不被忽略，0 表示被忽略）
    attention_mask=None,
    # token_type_ids：用于区分不同句子的 token 类型
    token_type_ids=None,
    # position_ids：指定 token 的位置信息
    position_ids=None,
    # head_mask：用于控制多头注意力机制中每个注意力头的掩码
    head_mask=None,
    # encoder_hidden_states：编码器的隐藏状态
    encoder_hidden_states=None,
    # encoder_attention_mask：编码器的注意力掩码
    encoder_attention_mask=None,
    # params：额外的参数，应为字典类型
    params: dict = None,
    # dropout_rng：用于随机数生成的 PRNG 键
    dropout_rng: jax.random.PRNGKey = None,
    # train：是否处于训练模式
    train: bool = False,
    # output_attentions：是否输出注意力权重
    output_attentions: Optional[bool] = None,
    # output_hidden_states：是否输出隐藏状态
    output_hidden_states: Optional[bool] = None,
    # return_dict：是否返回字典形式的输出
    return_dict: Optional[bool] = None,
    # past_key_values：过去的键值对，应为字典类型
    past_key_values: dict = None,
class FlaxRobertaPreLayerNormModule(nn.Module):
    # 类型注解：配置对象为 RobertaPreLayerNormConfig 类型
    config: RobertaPreLayerNormConfig
    # 计算的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    # 是否添加池化层，默认为 True
    add_pooling_layer: bool = True
    # 是否使用梯度检查点，默认为 False
    gradient_checkpointing: bool = False

    # 模块初始化方法
    def setup(self):
        # 初始化嵌入层
        self.embeddings = FlaxRobertaPreLayerNormEmbeddings(self.config, dtype=self.dtype)
        # 初始化编码器
        self.encoder = FlaxRobertaPreLayerNormEncoder(
            self.config,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 初始化 LayerNorm 层
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 初始化池化层
        self.pooler = FlaxRobertaPreLayerNormPooler(self.config, dtype=self.dtype)

    # 对象调用方法
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
        # 确保当未传入 token_type_ids 时，其被正确初始化为全零数组
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        # 确保当未传入 position_ids 时，其被正确初始化为广播后的数组
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        # 调用嵌入层生成隐藏状态
        hidden_states = self.embeddings(
            input_ids, token_type_ids, position_ids, attention_mask, deterministic=deterministic
        )
        # 调用编码器处理隐藏状态
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
        # 对编码后的隐藏状态应用 LayerNorm
        hidden_states = outputs[0]
        hidden_states = self.LayerNorm(hidden_states)
        # 如果设置了添加池化层，则对处理后的隐藏状态进行池化
        pooled = self.pooler(hidden_states) if self.add_pooling_layer else None

        # 如果不返回字典，则按非字典格式返回结果
        if not return_dict:
            # 如果 pooled 为 None，则不返回它
            if pooled is None:
                return (hidden_states,) + outputs[1:]
            return (hidden_states, pooled) + outputs[1:]

        # 返回带有池化和交叉注意力的 FlaxBaseModelOutputWithPoolingAndCrossAttentions 对象
        return FlaxBaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


@add_start_docstrings(
    # "The bare RoBERTa-PreLayerNorm Model transformer outputting raw hidden-states without any specific head on top."
    # 上面的字符串描述了 RoBERTa-PreLayerNorm 模型的基本特征，它输出原始隐藏状态而没有特定的顶部头部。
    
    ROBERTA_PRELAYERNORM_START_DOCSTRING
    # 使用 ROBERTA_PRELAYERNORM_START_DOCSTRING 常量，可能是用于指定 RoBERTa-PreLayerNorm 模型的文档字符串的起始部分。
# 从 transformers.models.roberta.modeling_flax_roberta.FlaxRobertaModel 中复制代码，并将 Roberta->RobertaPreLayerNorm 进行了替换
class FlaxRobertaPreLayerNormModel(FlaxRobertaPreLayerNormPreTrainedModel):
    # 设定模块类为 FlaxRobertaPreLayerNormModule
    module_class = FlaxRobertaPreLayerNormModule


# 调用函数 append_call_sample_docstring，向 FlaxRobertaPreLayerNormModel 添加文档字符串示例
append_call_sample_docstring(
    FlaxRobertaPreLayerNormModel,
    _CHECKPOINT_FOR_DOC,
    FlaxBaseModelOutputWithPooling,
    _CONFIG_FOR_DOC,
)


# 从 transformers.models.roberta.modeling_flax_roberta.FlaxRobertaForMaskedLMModule 中复制代码，并将 Roberta->RobertaPreLayerNorm,roberta->roberta_prelayernorm 进行了替换
class FlaxRobertaPreLayerNormForMaskedLMModule(nn.Module):
    # 配置项为 RobertaPreLayerNormConfig，数据类型为 jnp.float32，默认情况下不使用梯度检查点
    config: RobertaPreLayerNormConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        # 初始化 self.roberta_prelayernorm 为 FlaxRobertaPreLayerNormModule 实例
        self.roberta_prelayernorm = FlaxRobertaPreLayerNormModule(
            config=self.config,
            add_pooling_layer=False,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 初始化 self.lm_head 为 FlaxRobertaPreLayerNormLMHead 实例
        self.lm_head = FlaxRobertaPreLayerNormLMHead(config=self.config, dtype=self.dtype)

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
        outputs = self.roberta_prelayernorm(
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
        if self.config.tie_word_embeddings:
            # 如果配置中指定共享词嵌入，则使用共享的词嵌入
            shared_embedding = self.roberta_prelayernorm.variables["params"]["embeddings"]["word_embeddings"][
                "embedding"
            ]
        else:
            shared_embedding = None

        # 计算预测得分
        logits = self.lm_head(hidden_states, shared_embedding=shared_embedding)

        if not return_dict:
            return (logits,) + outputs[1:]

        # 返回 FlaxMaskedLMOutput 对象，包含预测得分、隐藏状态、注意力权重
        return FlaxMaskedLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# 使用 add_start_docstrings 函数向 FlaxRobertaPreLayerNormForMaskedLM 添加文档字符串
@add_start_docstrings(
    """RoBERTa-PreLayerNorm 模型，顶部带有 `语言建模` 头.""", ROBERTA_PRELAYERNORM_START_DOCSTRING
)
# 从 transformers.models.roberta.modeling_flax_roberta.FlaxRobertaForMaskedLM 复制代码，并将 Roberta->RobertaPreLayerNorm 进行了替换
class FlaxRobertaPreLayerNormForMaskedLM(FlaxRobertaPreLayerNormPreTrainedModel):
    # 设定模块类为 FlaxRobertaPreLayerNormForMaskedLMModule
    module_class = FlaxRobertaPreLayerNormForMaskedLMModule


# 调用函数 append_call_sample_docstring，向 FlaxRobertaPreLayerNormForMaskedLM 添加文档字符串示例
append_call_sample_docstring(
    FlaxRobertaPreLayerNormForMaskedLM,
    _CHECKPOINT_FOR_DOC,
    FlaxBaseModelOutputWithPooling,
    _CONFIG_FOR_DOC,
    mask="<mask>",
)
# 从transformers.models.roberta.modeling_flax_roberta.FlaxRobertaForSequenceClassificationModule复制而来，修改了Roberta为RobertaPreLayerNorm，roberta为roberta_prelayernorm
class FlaxRobertaPreLayerNormForSequenceClassificationModule(nn.Module):
    # 配置信息为RobertaPreLayerNormConfig
    config: RobertaPreLayerNormConfig
    # 数据类型默认为jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 是否使用梯度检查点，默认为False
    gradient_checkpointing: bool = False

    # 模块初始化方法
    def setup(self):
        # 初始化self.roberta_prelayernorm，使用FlaxRobertaPreLayerNormModule
        self.roberta_prelayernorm = FlaxRobertaPreLayerNormModule(
            config=self.config,
            dtype=self.dtype,
            add_pooling_layer=False,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 初始化self.classifier，使用FlaxRobertaPreLayerNormClassificationHead
        self.classifier = FlaxRobertaPreLayerNormClassificationHead(config=self.config, dtype=self.dtype)

    # 模块调用方法
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
        # 模型处理过程
        outputs = self.roberta_prelayernorm(
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
        # 获取logits，使用self.classifier处理序列输出
        logits = self.classifier(sequence_output, deterministic=deterministic)

        # 如果return_dict为False，返回logits和其他输出状态
        if not return_dict:
            return (logits,) + outputs[1:]

        # 返回FlaxSequenceClassifierOutput对象，包括logits、隐藏状态和注意力权重
        return FlaxSequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# 添加起始文档字符串
@add_start_docstrings(
    """
    带有顶部序列分类/回归头部的RobertaPreLayerNorm模型变换器（在汇聚输出的顶部添加线性层），例如用于GLUE任务。
    """,
    ROBERTA_PRELAYERNORM_START_DOCSTRING,
)
# 从transformers.models.roberta.modeling_flax_roberta.FlaxRobertaForSequenceClassification复制而来，修改了Roberta为RobertaPreLayerNorm
class FlaxRobertaPreLayerNormForSequenceClassification(FlaxRobertaPreLayerNormPreTrainedModel):
    # 模块类为FlaxRobertaPreLayerNormForSequenceClassificationModule
    module_class = FlaxRobertaPreLayerNormForSequenceClassificationModule


# 添加调用示例文档字符串
append_call_sample_docstring(
    FlaxRobertaPreLayerNormForSequenceClassification,
    _CHECKPOINT_FOR_DOC,
    FlaxSequenceClassifierOutput,
    _CONFIG_FOR_DOC,
)


# 从transformers.models.bert.modeling_flax_bert.FlaxBertForMultipleChoiceModule复制而来，修改了Bert为RobertaPreLayerNorm，self.bert为self.roberta_prelayernorm
class FlaxRobertaPreLayerNormForMultipleChoiceModule(nn.Module):
    # 配置信息为RobertaPreLayerNormConfig
    config: RobertaPreLayerNormConfig
    # 数据类型默认为jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 是否使用梯度检查点，默认为False
    gradient_checkpointing: bool = False
    # 设置模型的初始状态，包括Roberta的预层归一化模块
    def setup(self):
        self.roberta_prelayernorm = FlaxRobertaPreLayerNormModule(
            config=self.config,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 设置dropout层，使用配置中指定的隐藏层dropout概率
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        # 设置分类器层，输出维度为1，使用指定的数据类型
        self.classifier = nn.Dense(1, dtype=self.dtype)

    # 定义类的调用方法，实现模型的前向传播
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
        # 计算选择题的数量
        num_choices = input_ids.shape[1]
        # 重塑输入张量，将其展平以便传递给模型
        input_ids = input_ids.reshape(-1, input_ids.shape[-1]) if input_ids is not None else None
        attention_mask = attention_mask.reshape(-1, attention_mask.shape[-1]) if attention_mask is not None else None
        token_type_ids = token_type_ids.reshape(-1, token_type_ids.shape[-1]) if token_type_ids is not None else None
        position_ids = position_ids.reshape(-1, position_ids.shape[-1]) if position_ids is not None else None

        # 调用Roberta的预层归一化模块进行模型计算
        outputs = self.roberta_prelayernorm(
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

        # 从模型输出中提取汇聚输出
        pooled_output = outputs[1]
        # 对汇聚输出应用dropout，以便在训练过程中随机丢弃一部分节点
        pooled_output = self.dropout(pooled_output, deterministic=deterministic)
        # 使用分类器层计算最终的logits
        logits = self.classifier(pooled_output)

        # 将logits张量重新形状为[num_choices, -1]，以便适应多选题的输出格式
        reshaped_logits = logits.reshape(-1, num_choices)

        # 如果不要求返回字典形式的输出，则返回包含logits和可能的额外输出的元组
        if not return_dict:
            return (reshaped_logits,) + outputs[2:]

        # 返回FlaxMultipleChoiceModelOutput类的实例，包含重塑后的logits、隐藏状态和注意力张量
        return FlaxMultipleChoiceModelOutput(
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    """
    RobertaPreLayerNorm Model with a multiple choice classification head on top (a linear layer on top of the pooled
    output and a softmax) e.g. for RocStories/SWAG tasks.
    """


    ROBERTA_PRELAYERNORM_START_DOCSTRING,


)


# Copied from transformers.models.roberta.modeling_flax_roberta.FlaxRobertaForMultipleChoice with Roberta->RobertaPreLayerNorm
class FlaxRobertaPreLayerNormForMultipleChoice(FlaxRobertaPreLayerNormPreTrainedModel):
    module_class = FlaxRobertaPreLayerNormForMultipleChoiceModule


overwrite_call_docstring(
    FlaxRobertaPreLayerNormForMultipleChoice,
    ROBERTA_PRELAYERNORM_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"),
)


append_call_sample_docstring(
    FlaxRobertaPreLayerNormForMultipleChoice,
    _CHECKPOINT_FOR_DOC,
    FlaxMultipleChoiceModelOutput,
    _CONFIG_FOR_DOC,
)


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertForTokenClassificationModule with Bert->RobertaPreLayerNorm, with self.bert->self.roberta_prelayernorm
class FlaxRobertaPreLayerNormForTokenClassificationModule(nn.Module):
    config: RobertaPreLayerNormConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        self.roberta_prelayernorm = FlaxRobertaPreLayerNormModule(
            config=self.config,
            dtype=self.dtype,
            add_pooling_layer=False,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
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
        head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # Model
        outputs = self.roberta_prelayernorm(
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


@add_start_docstrings(
    """
    RobertaPreLayerNorm Model with a token classification head on top (a linear layer on top of the hidden-states
    output) e.g. for Named-Entity-Recognition (NER) tasks.
    """
    ROBERTA_PRELAYERNORM_START_DOCSTRING,



    注释：
    RobertaPreLayerNorm 模型，其顶部有一个面向标记分类的头部（在隐藏状态输出之上的线性层），例如用于命名实体识别（NER）任务。
    """
    ROBERTA_PRELAYERNORM_START_DOCSTRING,
# 从 transformers.models.roberta.modeling_flax_roberta.FlaxRobertaForTokenClassification 复制代码，并将 Roberta 改为 RobertaPreLayerNorm
class FlaxRobertaPreLayerNormForTokenClassification(FlaxRobertaPreLayerNormPreTrainedModel):
    # 模块类设置为 FlaxRobertaPreLayerNormForTokenClassificationModule
    module_class = FlaxRobertaPreLayerNormForTokenClassificationModule

# 调用函数 append_call_sample_docstring，用于给定的类添加示例文档字符串，包括检查点、输出和配置信息
append_call_sample_docstring(
    FlaxRobertaPreLayerNormForTokenClassification,
    _CHECKPOINT_FOR_DOC,
    FlaxTokenClassifierOutput,
    _CONFIG_FOR_DOC,
)

# 从 transformers.models.bert.modeling_flax_bert.FlaxBertForQuestionAnsweringModule 复制代码，并将 Bert 改为 RobertaPreLayerNorm，self.bert 改为 self.roberta_prelayernorm
class FlaxRobertaPreLayerNormForQuestionAnsweringModule(nn.Module):
    # 配置为 RobertaPreLayerNormConfig 类型
    config: RobertaPreLayerNormConfig
    # 数据类型设置为 jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 梯度检查点设置为 False
    gradient_checkpointing: bool = False

    # 初始化函数
    def setup(self):
        # 创建 RobertaPreLayerNormModule 实例，包括配置、数据类型、不添加池化层、梯度检查点设置
        self.roberta_prelayernorm = FlaxRobertaPreLayerNormModule(
            config=self.config,
            dtype=self.dtype,
            add_pooling_layer=False,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 创建全连接层 nn.Dense，输出维度为 self.config.num_labels，数据类型为 self.dtype
        self.qa_outputs = nn.Dense(self.config.num_labels, dtype=self.dtype)

    # 调用函数，实现模型的前向传播
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
        # 调用 self.roberta_prelayernorm 进行模型前向传播，传入各种输入参数
        outputs = self.roberta_prelayernorm(
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

        # 计算问题回答的起始和结束 logits
        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = jnp.split(logits, self.config.num_labels, axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # 如果不返回字典，则返回元组和其他输出
        if not return_dict:
            return (start_logits, end_logits) + outputs[1:]

        # 返回 FlaxQuestionAnsweringModelOutput 类的实例，包括起始 logits、结束 logits、隐藏状态和注意力
        return FlaxQuestionAnsweringModelOutput(
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# 使用 add_start_docstrings 函数添加 RobertaPreLayerNorm 模型的文档字符串，适用于抽取式问答任务如 SQuAD
@add_start_docstrings(
    """
    RobertaPreLayerNorm Model with a span classification head on top for extractive question-answering tasks like SQuAD
    (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    ROBERTA_PRELAYERNORM_START_DOCSTRING,
)
# 从 transformers.models.roberta.modeling_flax_roberta.FlaxRobertaForQuestionAnswering 复制代码，并将 Roberta 改为 RobertaPreLayerNorm
class FlaxRobertaPreLayerNormForQuestionAnswering(FlaxRobertaPreLayerNormPreTrainedModel):
    # 定义变量 module_class 并赋值为 FlaxRobertaPreLayerNormForQuestionAnsweringModule 类
    module_class = FlaxRobertaPreLayerNormForQuestionAnsweringModule
# 向函数添加示例文档字符串，用于自动生成API文档
append_call_sample_docstring(
    FlaxRobertaPreLayerNormForQuestionAnswering,  # 要添加示例文档字符串的类
    _CHECKPOINT_FOR_DOC,  # 用于文档的检查点
    FlaxQuestionAnsweringModelOutput,  # 生成文档的模型输出
    _CONFIG_FOR_DOC,  # 用于文档的配置
)


# 从 transformers.models.roberta.modeling_flax_roberta.FlaxRobertaForCausalLMModule 复制的代码，将类名和部分参数修改为适应 Causal LM 的设置
class FlaxRobertaPreLayerNormForCausalLMModule(nn.Module):
    config: RobertaPreLayerNormConfig  # 模型配置信息
    dtype: jnp.dtype = jnp.float32  # 数据类型设置为 32 位浮点数
    gradient_checkpointing: bool = False  # 是否使用梯度检查点

    def setup(self):
        # 初始化 RoBERTa + LayerNorm 模块，不包括池化层
        self.roberta_prelayernorm = FlaxRobertaPreLayerNormModule(
            config=self.config,
            add_pooling_layer=False,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 初始化语言模型头部，与 RoBERTa + LayerNorm 共享配置和数据类型
        self.lm_head = FlaxRobertaPreLayerNormLMHead(config=self.config, dtype=self.dtype)

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
        # 执行模型前向传播
        outputs = self.roberta_prelayernorm(
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

        hidden_states = outputs[0]  # 提取隐藏状态作为输出的第一部分

        if self.config.tie_word_embeddings:
            # 如果配置要求共享词嵌入，获取共享的嵌入层参数
            shared_embedding = self.roberta_prelayernorm.variables["params"]["embeddings"]["word_embeddings"][
                "embedding"
            ]
        else:
            shared_embedding = None  # 否则不共享词嵌入

        # 计算预测分数
        logits = self.lm_head(hidden_states, shared_embedding=shared_embedding)

        if not return_dict:
            # 如果不需要返回字典形式的输出，则返回元组形式的结果
            return (logits,) + outputs[1:]

        # 返回带有交叉注意力的 Causal LM 输出
        return FlaxCausalLMOutputWithCrossAttentions(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


@add_start_docstrings(
    """
    在 RoBERTa + LayerNorm 模型上添加语言建模头部的预训练模型，
    例如用于自回归任务的隐藏状态输出之上的线性层。
    """,
    ROBERTA_PRELAYERNORM_START_DOCSTRING,  # 引用 RoBERTa + LayerNorm 的起始文档字符串
)
# 从 transformers.models.roberta.modeling_flax_roberta.FlaxRobertaForCausalLM 复制的代码，将类名和部分参数修改为适应 Causal LM 的设置
class FlaxRobertaPreLayerNormForCausalLM(FlaxRobertaPreLayerNormPreTrainedModel):
    # 指定内部模块的类为 FlaxRobertaPreLayerNormForCausalLMModule
    module_class = FlaxRobertaPreLayerNormForCausalLMModule

    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = None):
        # 初始化缓存
        batch_size, seq_length = input_ids.shape

        # 使用 self.init_cache 方法初始化 past_key_values
        past_key_values = self.init_cache(batch_size, max_length)
        
        # 由于解码器使用因果掩码，可以创建一个静态的扩展注意力掩码
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        
        # 如果存在 attention_mask，则根据其累积位置更新 extended_attention_mask
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            # 否则，根据序列长度广播位置 ids
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        # 返回准备好的生成输入
        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        # 更新生成输入的方法，将过去的键值和位置 ids 更新到 model_kwargs 中
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs


append_call_sample_docstring(
    FlaxRobertaPreLayerNormForCausalLM,
    _CHECKPOINT_FOR_DOC,
    FlaxCausalLMOutputWithCrossAttentions,
    _CONFIG_FOR_DOC,
)
```