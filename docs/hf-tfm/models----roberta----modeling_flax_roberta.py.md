# `.\transformers\models\roberta\modeling_flax_roberta.py`

```
# coding=utf-8
# 声明模块的编码格式为 UTF-8
# 版权声明
# Copyright 2021 The Google Flax Team Authors and The HuggingFace Inc. team.
# 版权声明，包括作者和团队信息
#
# 根据 Apache 许可证版本 2.0 进行许可 ("许可证");
# 除非符合许可证的要求，否则不得使用此文件。
# 您可以在以下网址获得许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按照 "原样" 的基础分发软件，
# 没有任何明示或暗示的担保或条件。
# 请查看许可证以获取有关特定语言的权限和限制

from typing import Callable, Optional, Tuple
# 导入所需的库和模块

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
# 导入必要的库和模块

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
from .configuration_roberta import RobertaConfig
# 从各个模块和类导入所需的内容

logger = logging.get_logger(__name__)
# 获取日志记录器

_CHECKPOINT_FOR_DOC = "roberta-base"
_CONFIG_FOR_DOC = "RobertaConfig"
# 检查点和配置的文档

remat = nn_partitioning.remat
# 从 nn_partitioning 模块导入 remat 方法

def create_position_ids_from_input_ids(input_ids, padding_idx):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        input_ids: jnp.ndarray
        padding_idx: int

    Returns: jnp.ndarray
    """
    # 根据输入的 input_ids 和 padding_idx 创建位置 id
    # 将非填充符号替换为其位置数字。位置编号从 padding_idx+1 开始。填充符号将被忽略
    # 此方法修改自 fairseq 的 `utils.make_positions`。

    # 创建一个布尔掩码，标记非填充符号为 1，填充符号为 0
    mask = (input_ids != padding_idx).astype("i4")

    if mask.ndim > 2:
        # 如果掩码的维度大于 2，则将掩码重塑为二维
        mask = mask.reshape((-1, mask.shape[-1]))
        # 计算非填充符号的累计索引，并乘以掩码
        incremental_indices = jnp.cumsum(mask, axis=1).astype("i4") * mask
        incremental_indices = incremental_indices.reshape(input_ids.shape)
    else:
        incremental_indices = jnp.cumsum(mask, axis=1).astype("i4") * mask

    return incremental_indices.astype("i4") + padding_idx
    # 将计算得到的位置 id 加上 padding_idx，返回结果

ROBERTA_START_DOCSTRING = r"""

    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the
# Roberta 模型的起始文档字符串
    # 该库为其所有模型实现了各种功能（例如从PyTorch模型下载、保存和转换权重）
    # 这个模型也是一个flax.linen.Module子类。将其用作常规的Flax linen Module，并参考Flax文档以了解所有与一般用法和行为相关的事项。
    # 最后，该模型支持固有的JAX功能，例如：
    # - 即时编译（JIT）: https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit
    # - 自动微分: https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation
    # - 向量化: https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap
    # - 并行化: https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap
    # 参数:
    #     config ([`RobertaConfig`]): 模型配置类，包含模型的所有参数。使用配置文件初始化不会加载与模型关联的权重，只加载配置。
    #     查看`~FlaxPreTrainedModel.from_pretrained`方法以加载模型权重。
# ROBERTA_INPUTS_DOCSTRING 是一个原始的文档字符串，用于描述 Roberta 模型的输入参数的含义和作用
# input_ids: 输入序列 token 在词汇表中的索引，可以通过 AutoTokenizer 得到
# attention_mask: 避免对填充标记的注意力进行计算的掩码，值为 0 或 1，1 表示未被遮蔽，0 表示被遮蔽
# token_type_ids: 指示输入的第一部分和第二部分的段落标记的 token 索引，0 代表第一个句子，1 代表第二个句子
# position_ids: 指示每个输入序列 token 在位置嵌入中的位置索引
# head_mask: 用于空掩去注意模块的选择头部的掩码，1 表示未被掩盖，0 表示已被掩盖
# return_dict: 是否返回 ModelOutput 而不是普通的元组

# 从 transformers 模块中的 FlaxBertEmbeddings 类复制并修改为 FlaxRobertaEmbeddings 类，用于构建 Roberta 模型的嵌入层
class FlaxRobertaEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    # 定义类属性 config 为 RobertaConfig 类型的对象
    config: RobertaConfig
    # 定义数据类型为 jnp.float32
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    # 初始化模型参数
    def setup(self):
        # 创建词嵌入层，输入维度为词表大小，输出维度为隐藏层大小，使用正态分布初始化
        self.word_embeddings = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        # 创建位置嵌入层，输入维度为最大位置编码，输出维度为隐藏层大小，使用正态分布初始化
        self.position_embeddings = nn.Embed(
            self.config.max_position_embeddings,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        # 创建标记类型嵌入层，输入维度为标记类型词表大小，输出维度为隐藏层大小，使用正态分布初始化
        self.token_type_embeddings = nn.Embed(
            self.config.type_vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        # 创建 LayerNorm 层，设置 epsilon 和数据类型
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 创建 Dropout 层，设置丢弃率
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, input_ids, token_type_ids, position_ids, attention_mask, deterministic: bool = True):
        # Embed
        # 将输入词索引转换为词嵌入表示
        inputs_embeds = self.word_embeddings(input_ids.astype("i4"))
        # 将输入位置索引转换为位置嵌入表示
        position_embeds = self.position_embeddings(position_ids.astype("i4"))
        # 将输入标记类型索引转换为标记类型嵌入表示
        token_type_embeddings = self.token_type_embeddings(token_type_ids.astype("i4"))

        # Sum all embeddings
        # 将词嵌入、位置嵌入和标记类型嵌入相加得到隐藏状态
        hidden_states = inputs_embeds + token_type_embeddings + position_embeds

        # Layer Norm
        # 应用 LayerNorm 层
        hidden_states = self.LayerNorm(hidden_states)
        # 应用 Dropout 层
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 返回隐藏状态
        return hidden_states
# 从transformers.models.bert.modeling_flax_bert.FlaxBertSelfAttention复制而来，将Bert->Roberta
class FlaxRobertaSelfAttention(nn.Module):
    config: RobertaConfig  # 类型注释：Roberta模型的配置信息
    causal: bool = False  # 是否使用因果注意力
    dtype: jnp.dtype = jnp.float32  # 计算过程中的数据类型

    def setup(self):
        # 计算每个注意力头的维度
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        # 检查隐藏层大小是否是注意力头数的整数倍
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError(
                "`config.hidden_size`: {self.config.hidden_size} has to be a multiple of `config.num_attention_heads` "
                "                   : {self.config.num_attention_heads}"
            )

        # 初始化查询、键、值权重矩阵
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

        # 如果使用因果注意力，创建一个因果掩码
        if self.causal:
            self.causal_mask = make_causal_mask(
                jnp.ones((1, self.config.max_position_embeddings), dtype="bool"), dtype="bool"
            )

    # 将隐藏状态分割为多个注意力头
    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.config.num_attention_heads, self.head_dim))

    # 合并多个注意力头为隐藏状态
    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.config.hidden_size,))

    @nn.compact
    # 从transformers.models.bart.modeling_flax_bart.FlaxBartAttention._concatenate_to_cache复制而来
```  
    # 该函数用于将当前单个输入标记的投影密钥和值状态连接到先前步骤缓存的状态。
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        # 检查是否初始化缓存数据
        is_initialized = self.has_variable("cache", "cached_key")
        # 定义缓存的密钥和值的变量
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        # 定义缓存索引的变量
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))
    
        # 如果已经初始化
        if is_initialized:
            # 获取缓存的密钥和值的大小
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # 更新密钥和值缓存
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            # 更新缓存索引
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # 构建因果掩码
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)
        return key, value, attention_mask
    
    # 该函数是模型的前向传播函数
    def __call__(
        self,
        hidden_states,
        attention_mask,
        layer_head_mask,
        key_value_states: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        deterministic=True,
        output_attentions: bool = False,
    ):
        # 该函数实现了模型的前向传播逻辑
        pass
# 根据 transformers.models.bert.modeling_flax_bert.FlaxBertSelfOutput 的定义创建名为 FlaxRobertaSelfOutput 的类
class FlaxRobertaSelfOutput(nn.Module):
    # 定义类属性 config，类型为 RobertaConfig
    config: RobertaConfig
    # 定义类属性 dtype，值为 jnp.float32，表示计算的数据类型

    # setup 方法为类的初始化方法，在创建类实例时执行
    def setup(self):
        # 使用 nn.Dense 创建一个全连接层对象 dense
        # 设置输出维度为 self.config.hidden_size
        # 使用 jax.nn.initializers.normal 进行权重初始化，标准差为 self.config.initializer_range
        # 设置数据类型为 self.dtype
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 使用 nn.LayerNorm 创建一个层归一化对象 LayerNorm
        # 设置 epsilon 为 self.config.layer_norm_eps
        # 设置数据类型为 self.dtype
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 使用 nn.Dropout 创建一个 dropout 层对象 dropout
        # 设置丢弃概率为 self.config.hidden_dropout_prob
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    # __call__ 方法为类的调用方法，用于对实例对象进行调用操作
    # 接受 hidden_states, input_tensor 和 deterministic 作为输入参数
    def __call__(self, hidden_states, input_tensor, deterministic: bool = True):
        # 使用全连接层对象 dense 对 hidden_states 进行线性变换
        hidden_states = self.dense(hidden_states)
        # 使用 dropout 层对象 dropout 对 hidden_states 进行随机丢弃一些元素
        # 丢弃的概率根据 deterministic 进行调整
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 将 dropout 的输出与 input_tensor 进行相加，然后通过层归一化对象 LayerNorm 进行归一化
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回归一化后的结果
        return hidden_states


# 根据 transformers.models.bert.modeling_flax_bert.FlaxBertAttention 的定义创建名为 FlaxRobertaAttention 的类
class FlaxRobertaAttention(nn.Module):
    # 定义类属性 config，类型为 RobertaConfig
    config: RobertaConfig
    # 定义类属性 causal，值为 False
    causal: bool = False
    # 定义类属性 dtype，值为 jnp.float32

    # setup 方法为类的初始化方法，在创建类实例时执行
    def setup(self):
        # 创建 FlaxRobertaSelfAttention 的实例对象 self.self
        # 将 self.config, self.causal 和 self.dtype 作为参数传入
        self.self = FlaxRobertaSelfAttention(self.config, causal=self.causal, dtype=self.dtype)
        # 创建 FlaxRobertaSelfOutput 的实例对象 self.output
        # 将 self.config 和 self.dtype 作为参数传入
        self.output = FlaxRobertaSelfOutput(self.config, dtype=self.dtype)

    # __call__ 方法为类的调用方法，用于对实例对象进行调用操作
    # 接受 hidden_states, attention_mask, layer_head_mask, key_value_states, init_cache, deterministic, output_attentions 作为输入参数
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
        # FlaxBertSelfAttention 类的 __call__ 方法的注释见下方

        # Attention mask 来源于 attention_mask.shape 的形状为 (*batch_sizes, kv_length)
        # FLAX 需要: attention_mask.shape 的形状为 (*batch_sizes, 1, 1, kv_length)，使其可广播
        # 对应位置的值应与 attn_weights.shape 的形状为(*batch_sizes, num_heads, q_length, kv_length)
        # 进行广播操作，表示特定位置的注意力权重
        attn_outputs = self.self(
            hidden_states,
            attention_mask,
            layer_head_mask=layer_head_mask,
            key_value_states=key_value_states,
            init_cache=init_cache,
            deterministic=deterministic,
            output_attentions=output_attentions,
        )
        # 从 attn_outputs 中获取第一个元素（attn_output）赋值给 attn_output 变量
        attn_output = attn_outputs[0]
        # 将 attn_output 和 hidden_states 作为参数传入 self.output，得到输出的 hidden_states
        hidden_states = self.output(attn_output, hidden_states, deterministic=deterministic)

        # 创建一个包含 hidden_states 的元组 outputs
        outputs = (hidden_states,)

        # 如果 output_attentions 为真
        if output_attentions:
            # 将 attn_outputs 的第二个元素添加到 outputs 中
            outputs += (attn_outputs[1],)

        # 返回 outputs
        return outputs


# 根据 transformers.models.bert.modeling_flax_bert.FlaxBertIntermediate 的定义创建名为 FlaxRobertaIntermediate 的类
class FlaxRobertaIntermediate(nn.Module):
    # 定义类属性 config，类型为 RobertaConfig
    config: RobertaConfig
    # 定义类属性 dtype，值为 jnp.float32

    # setup 方法为类的初始化方法，在创建类实例时执行
    def setup(self):
        # 使用 nn.Dense 创建一个全连接层对象 dense
        # 设置输出维度为 self.config.intermediate_size
        # 使用 jax.nn.initializers.normal 进行权重初始化，标准差为 self.config.initializer_range
        # 设置数据类型为 self.dtype
        self.dense = nn.Dense(
            self.config.intermediate_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 从全局变量 ACT2FN 中获取 self.config.hidden_act 对应的激活函数
        self.activation = ACT2FN[self.config.hidden_act]
    # 定义类中的方法，该方法用于处理隐藏状态
    def __call__(self, hidden_states):
        # 将隐藏状态输入到全连接层中进行线性变换
        hidden_states = self.dense(hidden_states)
        # 将线性变换的结果输入到激活函数中进行非线性变换
        hidden_states = self.activation(hidden_states)
        # 返回经过线性变换和非线性变换后的隐藏状态
        return hidden_states
# 从 transformers.models.bert.modeling_flax_bert.FlaxBertOutput 复制代码，并将 Bert 替换为 Roberta
class FlaxRobertaOutput(nn.Module):
    # 配置文件：RobertaConfig
    config: RobertaConfig
    # 计算时的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 初始化方法
    def setup(self):
        # 密集层，将输入转换为隐藏状态的大小，使用正态分布初始化
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 丢弃层，以一定概率丢弃输入
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        # 归一化层，对隐藏状态进行归一化处理
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    # 前向传播方法
    def __call__(self, hidden_states, attention_output, deterministic: bool = True):
        # 密集层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # 丢弃层处理隐藏状态
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 归一化层处理隐藏状态和注意力输出的相加结果
        hidden_states = self.LayerNorm(hidden_states + attention_output)
        # 返回处理后的隐藏状态
        return hidden_states


# 从 transformers.models.bert.modeling_flax_bert.FlaxBertLayer 复制代码，并将 Bert 替换为 Roberta
class FlaxRobertaLayer(nn.Module):
    # 配置文件：RobertaConfig
    config: RobertaConfig
    # 计算时的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 初始化方法
    def setup(self):
        # 自注意力层
        self.attention = FlaxRobertaAttention(self.config, causal=self.config.is_decoder, dtype=self.dtype)
        # 中间层
        self.intermediate = FlaxRobertaIntermediate(self.config, dtype=self.dtype)
        # 输出层
        self.output = FlaxRobertaOutput(self.config, dtype=self.dtype)
        # 如果配置中有跨注意力机制，则添加跨注意力层
        if self.config.add_cross_attention:
            self.crossattention = FlaxRobertaAttention(self.config, causal=False, dtype=self.dtype)

    # 前向传播方法
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
    # 执行 self-attention 机制，计算当前序列的 attention 输出
    attention_outputs = self.attention(
        hidden_states, # 当前序列的隐藏状态
        attention_mask, # 注意力掩码
        layer_head_mask=layer_head_mask, # 每一个头部的掩码
        init_cache=init_cache, # 初始化 cache
        deterministic=deterministic, # 确定性计算
        output_attentions=output_attentions # 是否输出 attention 权重
    )
    attention_output = attention_outputs[0] # 获取 attention 输出
    
    # 如果有 encoder 隐藏状态，则执行交叉注意力机制
    if encoder_hidden_states is not None:
        cross_attention_outputs = self.crossattention(
            attention_output, # 当前序列的 attention 输出
            attention_mask=encoder_attention_mask, # encoder 的注意力掩码
            layer_head_mask=layer_head_mask, # 每一个头部的掩码
            key_value_states=encoder_hidden_states, # encoder 的隐藏状态
            deterministic=deterministic, # 确定性计算
            output_attentions=output_attentions # 是否输出 attention 权重
        )
        attention_output = cross_attention_outputs[0] # 获取交叉注意力输出
    
    # 计算前馈神经网络输出
    hidden_states = self.intermediate(attention_output)
    hidden_states = self.output(hidden_states, attention_output, deterministic=deterministic)
    
    # 构造输出
    outputs = (hidden_states,)
    
    # 如果需要输出 attention 权重
    if output_attentions:
        outputs += (attention_outputs[1],)
        if encoder_hidden_states is not None:
            outputs += (cross_attention_outputs[1],)
    return outputs
# 定义 FlaxRobertaLayerCollection 类，继承自 nn.Module
class FlaxRobertaLayerCollection(nn.Module):
    # 接收 RobertaConfig 作为配置参数
    config: RobertaConfig
    # 指定计算时使用的数据类型为 float32
    dtype: jnp.dtype = jnp.float32
    # 是否启用梯度检查点机制
    gradient_checkpointing: bool = False

    # setup 方法在初始化时被调用
    def setup(self):
        # 如果启用梯度检查点机制
        if self.gradient_checkpointing:
            # 使用 remat 函数对 FlaxRobertaLayer 进行重新包装，使其支持梯度检查点
            FlaxRobertaCheckpointLayer = remat(FlaxRobertaLayer, static_argnums=(5, 6, 7))
            # 根据配置的 num_hidden_layers 参数，创建对应数量的 FlaxRobertaCheckpointLayer 实例，并存储在 self.layers 列表中
            self.layers = [
                FlaxRobertaCheckpointLayer(self.config, name=str(i), dtype=self.dtype)
                for i in range(self.config.num_hidden_layers)
            ]
        # 如果没有启用梯度检查点机制
        else:
            # 根据配置的 num_hidden_layers 参数，创建对应数量的 FlaxRobertaLayer 实例，并存储在 self.layers 列表中
            self.layers = [
                FlaxRobertaLayer(self.config, name=str(i), dtype=self.dtype)
                for i in range(self.config.num_hidden_layers)
            ]

    # 定义 __call__ 方法，用于处理输入数据
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
        # 在此处添加对输入参数的处理逻辑
        pass
    # 这是一个 Transformer 模型的前向传播函数
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        init_cache=None,
        deterministic=True,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # 如果需要输出注意力权重和隐藏状态，则初始化相应的元组
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
    
        # 检查 head_mask 的层数是否正确
        if head_mask is not None:
            if head_mask.shape[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.shape[0]}."
                )
    
        # 遍历每一个 Transformer 层
        for i, layer in enumerate(self.layers):
            # 如果需要输出隐藏状态，则记录当前隐藏状态
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
    
            # 调用当前层的前向传播函数
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
    
            # 更新隐藏状态
            hidden_states = layer_outputs[0]
    
            # 如果需要输出注意力权重，则记录当前层的注意力权重
            if output_attentions:
                all_attentions += (layer_outputs[1],)
    
                # 如果有 encoder 隐藏状态，则记录当前层的交叉注意力权重
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)
    
        # 如果需要输出隐藏状态，则记录最后一层的隐藏状态
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
    
        # 构建输出结果
        outputs = (hidden_states, all_hidden_states, all_attentions, all_cross_attentions)
    
        # 如果不需要返回字典，则返回元组
        if not return_dict:
            return tuple(v for v in outputs if v is not None)
    
        # 返回 Transformer 模型的输出
        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )
# 从transformers.models.bert.modeling_flax_bert.FlaxBertEncoder复制而来，将Bert->Roberta
class FlaxRobertaEncoder(nn.Module):
    config: RobertaConfig  # 类型为RobertaConfig的配置参数
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型为jnp.float32
    gradient_checkpointing: bool = False  # 梯度检查点是否开启，默认关闭

    def setup(self):
        self.layer = FlaxRobertaLayerCollection(  # 初始化FlaxRobertaLayerCollection实例
            self.config,  # 传入配置参数
            dtype=self.dtype,  # 传入数据类型参数
            gradient_checkpointing=self.gradient_checkpointing,  # 传入梯度检查点参数
        )

    def __call__(
        self,
        hidden_states,  # 隐藏状态
        attention_mask,  # 注意力掩码
        head_mask,  # 头部掩码
        encoder_hidden_states: Optional[jnp.ndarray] = None,  # 编码器隐藏状态，默认为None
        encoder_attention_mask: Optional[jnp.ndarray] = None,  # 编码器注意力掩码，默认为None
        init_cache: bool = False,  # 是否初始化缓存，默认为False
        deterministic: bool = True,  # 是否确定性，默认为True
        output_attentions: bool = False,  # 是否输出注意力，默认为False
        output_hidden_states: bool = False,  # 是否输出隐藏状态，默认为False
        return_dict: bool = True,  # 是否返回字典，默认为True
    ):
        return self.layer(  # 调用FlaxRobertaLayerCollection实例
            hidden_states,  # 传入隐藏状态
            attention_mask,  # 传入注意力掩码
            head_mask=head_mask,  # 传入头部掩码
            encoder_hidden_states=encoder_hidden_states,  # 传入编码器隐藏状态
            encoder_attention_mask=encoder_attention_mask,  # 传入编码器注意力掩码
            init_cache=init_cache,  # 传入是否初始化缓存参数
            deterministic=deterministic,  # 传入是否确定性参数
            output_attentions=output_attentions,  # 传入是否输出注意力参数
            output_hidden_states=output_hidden_states,  # 传入是否输出隐藏状态参数
            return_dict=return_dict,  # 传入是否返回字典参数
        )


# 从transformers.models.bert.modeling_flax_bert.FlaxBertPooler复制而来，将Bert->Roberta
class FlaxRobertaPooler(nn.Module):
    config: RobertaConfig  # 类型为RobertaConfig的配置参数
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型为jnp.float32

    def setup(self):
        self.dense = nn.Dense(  # 初始化nn.Dense实例
            self.config.hidden_size,  # 隐藏层大小
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),  # 初始化核参数
            dtype=self.dtype,  # 数据类型参数
        )

    def __call__(self, hidden_states):
        cls_hidden_state = hidden_states[:, 0]  # 获取隐藏状态的第一个位置
        cls_hidden_state = self.dense(cls_hidden_state)  # 将第一个位置的隐藏状态传入nn.Dense实例进行计算
        return nn.tanh(cls_hidden_state)  # 返回tanh激活后的结果


class FlaxRobertaLMHead(nn.Module):
    config: RobertaConfig  # 类型为RobertaConfig的配置参数
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型为jnp.float32
    bias_init: Callable[..., np.ndarray] = jax.nn.initializers.zeros  # 偏置的初始化函数

    def setup(self):
        self.dense = nn.Dense(  # 初始化nn.Dense实例
            self.config.hidden_size,  # 隐藏层大小
            dtype=self.dtype,  # 数据类型参数
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),  # 初始化核参数
        )
        self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)  # 初始化LayerNorm实例
        self.decoder = nn.Dense(  # 初始化nn.Dense实例
            self.config.vocab_size,  # 词汇表大小
            dtype=self.dtype,  # 数据类型参数
            use_bias=False,  # 不使用偏置
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),  # 初始化核参数
        )
        self.bias = self.param("bias", self.bias_init, (self.config.vocab_size,))  # 初始化偏置参数
    # 定义一个对象的调用方法，用于对隐藏状态进行处理，可选地使用共享的嵌入层
    def __call__(self, hidden_states, shared_embedding=None):
        # 将隐藏状态通过全连接层
        hidden_states = self.dense(hidden_states)
        # 将全连接层的输出应用 GELU 激活函数
        hidden_states = ACT2FN["gelu"](hidden_states)
        # 对输出进行层归一化处理
        hidden_states = self.layer_norm(hidden_states)

        # 如果提供了共享的嵌入层
        if shared_embedding is not None:
            # 将共享的嵌入层应用于解码器
            hidden_states = self.decoder.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
        else:
            # 否则直接将隐藏状态传递给解码器
            hidden_states = self.decoder(hidden_states)

        # 将偏置转换为 JAX 数组，并加到隐藏状态上
        bias = jnp.asarray(self.bias, self.dtype)
        hidden_states += bias
        # 返回处理后的隐藏状态
        return hidden_states
# 定义了一个继承自nn.Module的类FlaxRobertaClassificationHead
class FlaxRobertaClassificationHead(nn.Module):
    # 定义了一个config属性，类型为RobertaConfig
    config: RobertaConfig
    # 定义了一个dtype属性，默认值为jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 初始化方法
    def setup(self):
        # 创建一个全连接层dense，输入维度为config.hidden_size
        # 使用self.dtype作为数据类型，使用RobertaConfig的initializer_range作为权重初始化范围
        self.dense = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        # 判断config.classifier_dropout是否为None，如果是则使用config.hidden_dropout_prob，否则使用config.classifier_dropout
        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )
        # 创建一个Dropout层，rate为classifier_dropout
        self.dropout = nn.Dropout(rate=classifier_dropout)
        # 创建一个全连接层out_proj，输入维度为config.num_labels
        # 使用self.dtype作为数据类型，使用RobertaConfig的initializer_range作为权重初始化范围
        self.out_proj = nn.Dense(
            self.config.num_labels,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

    # 前向传播方法
    def __call__(self, hidden_states, deterministic=True):
        # 取hidden_states的第一个token的hidden_state（对应[CLS]）
        hidden_states = hidden_states[:, 0, :]
        # 使用self.dropout对hidden_states进行dropout操作
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 使用self.dense对hidden_states进行全连接操作
        hidden_states = self.dense(hidden_states)
        # 对hidden_states进行tanh激活函数操作
        hidden_states = nn.tanh(hidden_states)
        # 使用self.dropout对hidden_states进行dropout操作
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 使用self.out_proj对hidden_states进行全连接操作
        hidden_states = self.out_proj(hidden_states)
        # 返回输出结果hidden_states
        return hidden_states


# 定义了一个继承自FlaxPreTrainedModel的类FlaxRobertaPreTrainedModel
class FlaxRobertaPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 定义一个类属性config_class，值为RobertaConfig
    config_class = RobertaConfig
    # 定义一个类属性base_model_prefix，值为"roberta"
    base_model_prefix = "roberta"

    # 定义一个类属性module_class，值为None
    module_class: nn.Module = None

    # 初始化方法
    def __init__(
        self,
        config: RobertaConfig,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        gradient_checkpointing: bool = False,
        **kwargs,
    ):
        # 创建一个module对象，类型为self.module_class
        # 使用config作为参数，使用dtype作为数据类型，使用gradient_checkpointing作为是否开启梯度检查点的标志
        module = self.module_class(config=config, dtype=dtype, gradient_checkpointing=gradient_checkpointing, **kwargs)
        # 调用父类的初始化方法来初始化模型
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    # Copied from transformers.models.bert.modeling_flax_bert.FlaxBertPreTrainedModel.enable_gradient_checkpointing
    # 定义了一个方法enable_gradient_checkpointing
    def enable_gradient_checkpointing(self):
        # 将模型的_module属性设置为一个新的module_class对象，参数和初始化时一致
        self._module = self.module_class(
            config=self.config,
            dtype=self.dtype,
            gradient_checkpointing=True,
        )
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化权重函数，用于初始化模型参数
        # 初始化输入张量
        input_ids = jnp.zeros(input_shape, dtype="i4")  # 初始化输入 token IDs 张量为全零
        token_type_ids = jnp.ones_like(input_ids)  # 初始化 token 类型 IDs 张量为全一
        position_ids = create_position_ids_from_input_ids(input_ids, self.config.pad_token_id)  # 根据输入 token IDs 创建位置 IDs 张量
        attention_mask = jnp.ones_like(input_ids)  # 初始化注意力掩码张量为全一
        head_mask = jnp.ones((self.config.num_hidden_layers, self.config.num_attention_heads))  # 初始化头部掩码张量为全一

        params_rng, dropout_rng = jax.random.split(rng)  # 将随机数生成器切分为参数随机数生成器和 dropout 随机数生成器
        rngs = {"params": params_rng, "dropout": dropout_rng}  # 创建随机数生成器字典

        if self.config.add_cross_attention:
            # 如果配置中包含交叉注意力
            encoder_hidden_states = jnp.zeros(input_shape + (self.config.hidden_size,))  # 初始化编码器隐藏状态张量为全零
            encoder_attention_mask = attention_mask  # 使用输入的注意力掩码作为编码器的注意力掩码
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
            )  # 初始化模型，返回模型初始化的输出
        else:
            module_init_outputs = self.module.init(
                rngs, input_ids, attention_mask, token_type_ids, position_ids, head_mask, return_dict=False
            )  # 初始化模型，返回模型初始化的输出

        random_params = module_init_outputs["params"]  # 从初始化输出中获取随机参数

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))  # 将随机参数扁平化
            params = flatten_dict(unfreeze(params))  # 将输入的参数扁平化
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]  # 将缺失的参数从随机参数中填充
            self._missing_keys = set()  # 重置缺失参数集合
            return freeze(unflatten_dict(params))  # 返回填充后的参数冻结
        else:
            return random_params  # 返回随机参数

    # 从 transformers.models.bart.modeling_flax_bart.FlaxBartDecoderPreTrainedModel 中复制的 init_cache 函数
    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                用于快速自回归解码的批次大小。定义了初始化缓存的批次大小。
            max_length (`int`):
                自回归解码的最大可能长度。定义了初始化缓存的序列长度。
        """
        # 初始化输入变量以检索缓存
        input_ids = jnp.ones((batch_size, max_length), dtype="i4")  # 初始化输入 token IDs 张量为全一
        attention_mask = jnp.ones_like(input_ids, dtype="i4")  # 初始化注意力掩码张量为全一
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)  # 广播位置 IDs 张量
        # 初始化变量并获取缓存
        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        return unfreeze(init_variables["cache"])  # 返回解冻的缓存变量

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 这是一个用于执行前向传播的 callable 对象
    def __call__(
        self,
        # 输入 token ID 序列
        input_ids,
        # 对输入序列的注意力掩码，标识哪些部分需要关注
        attention_mask=None,
        # 输入序列的 token 类型 ID，区分不同类型的 token
        token_type_ids=None,
        # 输入序列中各 token 的位置 ID
        position_ids=None,
        # 编码器注意力层的掩码
        head_mask=None,
        # 编码器输出的隐藏状态
        encoder_hidden_states=None,
        # 编码器注意力层的掩码
        encoder_attention_mask=None,
        # 模型参数
        params: dict = None,
        # 随机数生成器的种子
        dropout_rng: jax.random.PRNGKey = None,
        # 是否为训练模式
        train: bool = False,
        # 是否输出注意力权重
        output_attentions: Optional[bool] = None,
        # 是否输出所有隐藏层输出
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典格式的输出
        return_dict: Optional[bool] = None,
        # 上次前向传播的中间结果
        past_key_values: dict = None,
    ):
# 从 transformers.models.bert.modeling_flax_bert.FlaxBertModule 复制代码，并将 Bert 改为 Roberta
class FlaxRobertaModule(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型
    add_pooling_layer: bool = True
    gradient_checkpointing: bool = False

    def setup(self):
        # 初始化 RoBERTa 的 embeddings 层
        self.embeddings = FlaxRobertaEmbeddings(self.config, dtype=self.dtype)
        # 初始化 RoBERTa 的 encoder 层
        self.encoder = FlaxRobertaEncoder(
            self.config,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 初始化 RoBERTa 的 pooler 层
        self.pooler = FlaxRobertaPooler(self.config, dtype=self.dtype)

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
        # 确保在未传入 token_type_ids 时被正确初始化
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        # 确保在未传入 position_ids 时被正确初始化
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        hidden_states = self.embeddings(
            input_ids, token_type_ids, position_ids, attention_mask, deterministic=deterministic
        )
        # 使用 encoder 层计算输出
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
        # 如果设置了 add_pooling_layer，使用 pooler 层
        pooled = self.pooler(hidden_states) if self.add_pooling_layer else None

        if not return_dict:
            # 如果 pooled 是 None，则不返回
            if pooled is None:
                return (hidden_states,) + outputs[1:]
            return (hidden_states, pooled) + outputs[1:]

        return FlaxBaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


@add_start_docstrings(
    "The bare RoBERTa Model transformer outputting raw hidden-states without any specific head on top.",
    ROBERTA_START_DOCSTRING,
)
class FlaxRobertaModel(FlaxRobertaPreTrainedModel):
    module_class = FlaxRobertaModule

append_call_sample_docstring(FlaxRobertaModel, _CHECKPOINT_FOR_DOC, FlaxBaseModelOutputWithPooling, _CONFIG_FOR_DOC)

class FlaxRobertaForMaskedLMModule(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        # 初始化 RoBERTa 模型
        self.roberta = FlaxRobertaModule(
            config=self.config,
            add_pooling_layer=False,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 初始化 RoBERTa 语言模型头部
        self.lm_head = FlaxRobertaLMHead(config=self.config, dtype=self.dtype)

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

        hidden_states = outputs[0]
        if self.config.tie_word_embeddings:
            shared_embedding = self.roberta.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
        else:
            shared_embedding = None

        # 计算预测分数
        logits = self.lm_head(hidden_states, shared_embedding=shared_embedding)

        if not return_dict:
            return (logits,) + outputs[1:]

        # 返回结果对象
        return FlaxMaskedLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings("""RoBERTa Model with a `language modeling` head on top.""", ROBERTA_START_DOCSTRING)
class FlaxRobertaForMaskedLM(FlaxRobertaPreTrainedModel):
    module_class = FlaxRobertaForMaskedLMModule

append_call_sample_docstring(
    FlaxRobertaForMaskedLM,
    _CHECKPOINT_FOR_DOC,
    FlaxBaseModelOutputWithPooling,
    _CONFIG_FOR_DOC,
    mask="<mask>",
)

class FlaxRobertaForSequenceClassificationModule(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        # 初始化 RoBERTa 模型
        self.roberta = FlaxRobertaModule(
            config=self.config,
            dtype=self.dtype,
            add_pooling_layer=False,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 初始化 RoBERTa 序列分类头部
        self.classifier = FlaxRobertaClassificationHead(config=self.config, dtype=self.dtype)
    # 这是一个 callable 对象，用于执行 BERT 模型的前向传播
    def __call__(
        self,
        input_ids,            # 输入序列的 token ID
        attention_mask,       # 输入序列的注意力掩码
        token_type_ids,       # 输入序列的 token 类型 ID
        position_ids,         # 输入序列的位置 ID
        head_mask,            # 注意力头的掩码
        deterministic: bool = True,    # 是否使用确定性行为
        output_attentions: bool = False,   # 是否输出注意力权重
        output_hidden_states: bool = False,    # 是否输出中间隐藏状态
        return_dict: bool = True,    # 是否返回字典格式的输出
    ):
        # 执行 BERT 模型的前向传播
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
        # 通过分类器计算预测logits
        logits = self.classifier(sequence_output, deterministic=deterministic)
    
        # 如果不返回字典格式，则返回 logits 和其他输出
        if not return_dict:
            return (logits,) + outputs[1:]
    
        # 返回预测 logits 以及可选的隐藏状态和注意力权重
        return FlaxSequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 这个类是 RoBERTa 模型的分类头，用于序列分类或回归任务（例如 GLUE 任务）
@add_start_docstrings(
    """
    Roberta Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    ROBERTA_START_DOCSTRING,
)
class FlaxRobertaForSequenceClassification(FlaxRobertaPreTrainedModel):
    module_class = FlaxRobertaForSequenceClassificationModule


# 为 FlaxRobertaForSequenceClassification 添加示例文档
append_call_sample_docstring(
    FlaxRobertaForSequenceClassification,
    _CHECKPOINT_FOR_DOC,
    FlaxSequenceClassifierOutput,
    _CONFIG_FOR_DOC,
)


# 这个类是 RoBERTa 模型的多项选择分类头，用于多项选择任务（例如 RocStories/SWAG 任务）
# 代码从 FlaxBertForMultipleChoiceModule 复制而来，并将 Bert 替换为 Roberta
class FlaxRobertaForMultipleChoiceModule(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        # 初始化 RoBERTa 模型
        self.roberta = FlaxRobertaModule(
            config=self.config,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 添加一个 dropout 层
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        # 添加一个线性层作为分类器
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
        # 计算 batch 大小和选择数量
        num_choices = input_ids.shape[1]
        # 将输入数据展平
        input_ids = input_ids.reshape(-1, input_ids.shape[-1]) if input_ids is not None else None
        attention_mask = attention_mask.reshape(-1, attention_mask.shape[-1]) if attention_mask is not None else None
        token_type_ids = token_type_ids.reshape(-1, token_type_ids.shape[-1]) if token_type_ids is not None else None
        position_ids = position_ids.reshape(-1, position_ids.shape[-1]) if position_ids is not None else None

        # 通过 RoBERTa 模型获得输出
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

        # 对输出进行处理
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output, deterministic=deterministic)
        logits = self.classifier(pooled_output)

        # 将 logits 调整为 (batch_size, num_choices) 的形状
        reshaped_logits = logits.reshape(-1, num_choices)

        if not return_dict:
            return (reshaped_logits,) + outputs[2:]

        return FlaxMultipleChoiceModelOutput(
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# 这个类是 RoBERTa 模型的多项选择分类头，用于多项选择任务（例如 RocStories/SWAG 任务）
@add_start_docstrings(
    """
    Roberta Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    # ROBERTA_START_DOCSTRING 常量，用于标识 Roberta 模型文档字符串的开始
    ROBERTA_START_DOCSTRING,
# 定义 FlaxRobertaForMultipleChoice 类，其父类为 FlaxRobertaPreTrainedModel
class FlaxRobertaForMultipleChoice(FlaxRobertaPreTrainedModel):
    # 设置模块类为 FlaxRobertaForMultipleChoiceModule
    module_class = FlaxRobertaForMultipleChoiceModule

# 覆盖 FlaxRobertaForMultipleChoice 的文档字符串
overwrite_call_docstring(
    FlaxRobertaForMultipleChoice, ROBERTA_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
)

# 添加调用示例的文档字符串
append_call_sample_docstring(
    FlaxRobertaForMultipleChoice,
    _CHECKPOINT_FOR_DOC,
    FlaxMultipleChoiceModelOutput,
    _CONFIG_FOR_DOC,
)

# 定义 FlaxRobertaForTokenClassificationModule 类，继承自 nn.Module
class FlaxRobertaForTokenClassificationModule(nn.Module):
    # 类型设置为 RobertaConfig
    config: RobertaConfig
    # 数据类型设置为 jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 梯度检查点设置为 False
    gradient_checkpointing: bool = False

    # 初始化方法
    def setup(self):
        # 创建 FlaxRobertaModule 对象，设置相关参数
        self.roberta = FlaxRobertaModule(
            config=self.config,
            dtype=self.dtype,
            add_pooling_layer=False,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 设置分类器的 dropout
        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )
        # 设置 dropout 层
        self.dropout = nn.Dropout(rate=classifier_dropout)
        # 设置分类器
        self.classifier = nn.Dense(self.config.num_labels, dtype=self.dtype)

    # 调用方法
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
        # 模型计算
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
        # 对隐藏状态进行 dropout
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 计算 logits
        logits = self.classifier(hidden_states)

        # 如果不返回字典，则返回元组
        if not return_dict:
            return (logits,) + outputs[1:]

        # 返回 FlaxTokenClassifierOutput 对象
        return FlaxTokenClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# 添加起始文档字符串
@add_start_docstrings(
    """
    Roberta Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    ROBERTA_START_DOCSTRING,
)
# 定义 FlaxRobertaForTokenClassification 类，其父类为 FlaxRobertaPreTrainedModel
class FlaxRobertaForTokenClassification(FlaxRobertaPreTrainedModel):
    # 设置模块类为 FlaxRobertaForTokenClassificationModule
    module_class = FlaxRobertaForTokenClassificationModule

# 添加调用示例的文档字符串
append_call_sample_docstring(
    FlaxRobertaForTokenClassification,
    _CHECKPOINT_FOR_DOC,
    FlaxTokenClassifierOutput,
    _CONFIG_FOR_DOC,
)
# 从transformers.models.bert.modeling_flax_bert.FlaxBertForQuestionAnsweringModule复制代码，将Bert->Roberta，将self.bert->self.roberta
class FlaxRobertaForQuestionAnsweringModule(nn.Module):
    # 定义config属性为RobertaConfig类型
    config: RobertaConfig
    # 定义dtype属性为jnp.float32类型，默认为jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 定义gradient_checkpointing属性为bool类型，默认为False
    gradient_checkpointing: bool = False

    def setup(self):
        # 使用FlaxRobertaModule初始化self.roberta，传入config、dtype、add_pooling_layer=False、gradient_checkpointing属性值
        self.roberta = FlaxRobertaModule(
            config=self.config,
            dtype=self.dtype,
            add_pooling_layer=False,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 初始化self.qa_outputs为一个全连接层，输出维度为self.config.num_labels，dtype为self.dtype
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
        # Model
        # 调用self.roberta模型，传入输入参数，并返回结果
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

        # 从输出中获取隐藏状态
        hidden_states = outputs[0]

        # 使用全连接层self.qa_outputs计算答案起始和结束的logits
        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = jnp.split(logits, self.config.num_labels, axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # 如果return_dict为False，则返回一个元组
        if not return_dict:
            return (start_logits, end_logits) + outputs[1:]

        # 如果return_dict为True，则返回FlaxQuestionAnsweringModelOutput对象
        return FlaxQuestionAnsweringModelOutput(
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    带有顶部span分类头的Roberta模型，用于抽取式问答任务，例如SQuAD（在隐藏状态输出的顶部添加线性层以计算`span start logits`和`span end logits`）。
    """,
    ROBERTA_START_DOCSTRING,
)
class FlaxRobertaForQuestionAnswering(FlaxRobertaPreTrainedModel):
    # 指定module_class为FlaxRobertaForQuestionAnsweringModule
    module_class = FlaxRobertaForQuestionAnsweringModule


# 将append_call_sample_docstring应用到FlaxRobertaForQuestionAnswering上
append_call_sample_docstring(
    FlaxRobertaForQuestionAnswering,
    _CHECKPOINT_FOR_DOC,
    FlaxQuestionAnsweringModelOutput,
    _CONFIG_FOR_DOC,
)


class FlaxRobertaForCausalLMModule(nn.Module):
    # 定义config属性为RobertaConfig类型
    config: RobertaConfig
    # 定义dtype属性为jnp.float32类型，默认为jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 定义gradient_checkpointing属性为bool类型，默认为False
    gradient_checkpointing: bool = False

    def setup(self):
        # 使用FlaxRobertaModule初始化self.roberta，传入config、add_pooling_layer=False、dtype、gradient_checkpointing属性值
        self.roberta = FlaxRobertaModule(
            config=self.config,
            add_pooling_layer=False,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 初始化self.lm_head为一个Roberta语言模型头
        self.lm_head = FlaxRobertaLMHead(config=self.config, dtype=self.dtype)
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
        # 调用模型对象的 __call__ 方法，传入参数，返回输出结果
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

        # 取出输出结果里的 hidden_states，赋值给变量 hidden_states
        hidden_states = outputs[0]
        # 判断是否需要共享词嵌入层的参数
        if self.config.tie_word_embeddings:
            # 如果需要共享，则获取共享词嵌入层的参数
            shared_embedding = self.roberta.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
        else:
            # 如果不需要共享，则将 shared_embedding 设为 None
            shared_embedding = None

        # 使用 hidden_states 和 shared_embedding 计算预测得分 logits
        logits = self.lm_head(hidden_states, shared_embedding=shared_embedding)

        # 判断是否需要返回一个 dict
        if not return_dict:
            # 如果不需要，则返回 (logits,) 加上 outputs 的其他元素
            return (logits,) + outputs[1:]

        # 返回一个 FlaxCausalLMOutputWithCrossAttentions 对象
        return FlaxCausalLMOutputWithCrossAttentions(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )
# 使用add_start_docstrings装饰器添加模型的描述文档
@add_start_docstrings(
    """
    Roberta Model with a language modeling head on top (a linear layer on top of the hidden-states output) e.g for
    autoregressive tasks.
    """,
    ROBERTA_START_DOCSTRING,
)
class FlaxRobertaForCausalLM(FlaxRobertaPreTrainedModel):
    module_class = FlaxRobertaForCausalLMModule

    # 为生成准备输入，初始化缓存
    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = None):
        # 初始化缓存
        batch_size, seq_length = input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length)
        # 注意：通常必须在attention_mask中为x>input_ids.shape[-1]和x<cache_length添加0，
        # 但由于解码器使用因果掩码，这些位置已经被掩盖了。
        # 因此，我们可以在这里创建一个单独的静态attention_mask，这对编译更有效
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

    # 更新生成输入，添加额外的缓存和位置id
    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs


# 添加调用示例的文档描述
append_call_sample_docstring(
    FlaxRobertaForCausalLM,
    _CHECKPOINT_FOR_DOC,
    FlaxCausalLMOutputWithCrossAttentions,
    _CONFIG_FOR_DOC,
)
```