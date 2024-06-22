# `.\transformers\models\roberta_prelayernorm\modeling_flax_roberta_prelayernorm.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 2022 年由 Google Flax 团队作者和 HuggingFace Inc. 团队撰写
#
# 根据 Apache 许可证 2.0 版（“许可证”）获得许可；
# 除非符合许可证，否则您不能使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件，
# 没有任何形式的明示或暗示保证或条件。
# 有关特定语言的权限，请参阅许可证。
""" Flax RoBERTa-PreLayerNorm 模型。"""
# 导入必要的库
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

# 导入其他模块
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
# 初始化日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置
_CHECKPOINT_FOR_DOC = "andreasmadsen/efficient_mlm_m0.40"
_CONFIG_FOR_DOC = "RobertaPreLayerNormConfig"

# 定义 remat 函数，用于重新材料化（Rematerialization）
remat = nn_partitioning.remat

# 从 transformers.models.roberta.modeling_flax_roberta.create_position_ids_from_input_ids 复制的函数
def create_position_ids_from_input_ids(input_ids, padding_idx):
    """
    将非填充符号替换为其位置编号。位置编号从 padding_idx+1 开始。忽略填充符号。这是从 fairseq 的 `utils.make_positions` 修改的。

    Args:
        input_ids: jnp.ndarray
        padding_idx: int

    Returns: jnp.ndarray
    """
    # 创建一个掩码，将填充符号替换为 0
    mask = (input_ids != padding_idx).astype("i4")

    # 如果掩码的维度大于 2，则重塑掩码的形状
    if mask.ndim > 2:
        mask = mask.reshape((-1, mask.shape[-1]))
        # 计算非填充符号的累积位置编号
        incremental_indices = jnp.cumsum(mask, axis=1).astype("i4") * mask
        incremental_indices = incremental_indices.reshape(input_ids.shape)
    else:
        incremental_indices = jnp.cumsum(mask, axis=1).astype("i4") * mask

    # 返回最终的位置编号
    return incremental_indices.astype("i4") + padding_idx
# ROBERTA_PRELAYERNORM_START_DOCSTRING 是一个原始文档字符串，用于描述一个继承自 FlaxPreTrainedModel 的模型类。
# 此类继承了通用方法，比如下载、保存和从 PyTorch 模型转换权重等。
# 此模型还是一个 flax.linen.Module 子类。可将其用作常规的 Flax linen Module，并参考 Flax 文档了解一般用法和行为。
# 最后，此模型支持 JAX 的内在特性，例如：
# - Just-In-Time (JIT) 编译
# - 自动微分
# - 向量化
# - 并行化
# 
# 参数:
#     config ([`RobertaPreLayerNormConfig`]): 包含模型所有参数的配置类。使用配置文件初始化不会加载与模型关联的权重，只加载配置。
#         可以查看 [`~FlaxPreTrainedModel.from_pretrained`] 方法来加载模型权重。

ROBERTA_PRELAYERNORM_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`numpy.ndarray` of shape `({0})`):
            输入序列标记在词汇表中的索引。

            可以使用 [`AutoTokenizer`] 获得索引。查看 [`PreTrainedTokenizer.encode`] 和
            [`PreTrainedTokenizer.__call__`] 了解详情。

            [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`numpy.ndarray` of shape `({0})`, *optional*):
            用于避免在填充标记索引上进行注意力计算的掩码。掩码的取值范围为 `[0, 1]`：

            - 对于**未被掩盖**的标记，取值为 1，
            - 对于**被掩盖**的标记，取值为 0。

            [什么是注意力掩码？](../glossary#attention-mask)
        token_type_ids (`numpy.ndarray` of shape `({0})`, *optional*):
            指示输入的第一部分和第二部分的段标记索引。索引的取值范围为 `[0, 1]`：

            - 0 对应*句子 A* 的标记，
            - 1 对应*句子 B* 的标记。

            [什么是标记类型 ID？](../glossary#token-type-ids)
        position_ids (`numpy.ndarray` of shape `({0})`, *optional*):
            指示每个输入序列标记在位置嵌入中的位置索引。索引的取值范围为 `[0, config.max_position_embeddings - 1]`。
        head_mask (`numpy.ndarray` of shape `({0})`, `optional):
            用于屏蔽注意力模块中的选定头部的掩码。掩码的取值范围为 `[0, 1]`：

            - 1 表示该头部**未被屏蔽**，
            - 0 表示该头部**被屏蔽**。

        return_dict (`bool`, *optional*):
            是否返回一个 [`~utils.ModelOutput`] 而不是一个普通的元组。
# 从transformers.models.bert.modeling_flax_bert.FlaxBertEmbeddings复制而来，修改为RobertaPreLayerNorm
class FlaxRobertaPreLayerNormEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    # 构造函数，初始化模型配置和数据类型
    config: RobertaPreLayerNormConfig
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型

    def setup(self):
        # 初始化词嵌入层，词汇大小为config.vocab_size，隐藏层大小为config.hidden_size
        # 使用正态分布初始化，标准差为config.initializer_range
        self.word_embeddings = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        # 初始化位置嵌入层，位置嵌入最大值为config.max_position_embeddings，隐藏层大小为config.hidden_size
        # 使用正态分布初始化，标准差为config.initializer_range
        self.position_embeddings = nn.Embed(
            self.config.max_position_embeddings,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        # 初始化标记类型嵌入层，类型词汇大小为config.type_vocab_size，隐藏层大小为config.hidden_size
        # 使用正态分布初始化，标准差为config.initializer_range
        self.token_type_embeddings = nn.Embed(
            self.config.type_vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        # 初始化Layer Norm层，epsilon为config.layer_norm_eps，数据类型为dtype
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 初始化Dropout层，丢弃率为config.hidden_dropout_prob
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    # 前向传播函数，接受输入id，标记类型id，位置id，注意力掩码，确定性标志
    def __call__(self, input_ids, token_type_ids, position_ids, attention_mask, deterministic: bool = True):
        # 嵌入
        inputs_embeds = self.word_embeddings(input_ids.astype("i4"))  # 输入词嵌入
        position_embeds = self.position_embeddings(position_ids.astype("i4"))  # 位置嵌入
        token_type_embeddings = self.token_type_embeddings(token_type_ids.astype("i4"))  # 标记类型嵌入

        # 合并所有嵌入
        hidden_states = inputs_embeds + token_type_embeddings + position_embeds

        # Layer Norm层
        hidden_states = self.LayerNorm(hidden_states)
        # Dropout层
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states


# 从transformers.models.bert.modeling_flax_bert.FlaxBertSelfAttention复制而来，修改为RobertaPreLayerNorm
class FlaxRobertaPreLayerNormSelfAttention(nn.Module):
    config: RobertaPreLayerNormConfig
    causal: bool = False
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型
    # 初始化函数，设置每个头的维度并检查隐藏层大小是否可以整除头数
    def setup(self):
        # 计算每个头的维度
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        # 检查隐藏层大小是否可以整除头数
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError(
                "`config.hidden_size`: {self.config.hidden_size} has to be a multiple of `config.num_attention_heads` "
                "                   : {self.config.num_attention_heads}"
            )

        # 初始化查询层
        self.query = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        # 初始化键层
        self.key = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        # 初始化值层
        self.value = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

        # 如果启用了因果关系，则创建因果掩码
        if self.causal:
            self.causal_mask = make_causal_mask(
                jnp.ones((1, self.config.max_position_embeddings), dtype="bool"), dtype="bool"
            )

    # 将隐藏状态分割为各个头
    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.config.num_attention_heads, self.head_dim))

    # 将各个头合并回原始隐藏状态
    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.config.hidden_size,))

    # 使用 nn.compact 装饰器，标记为 Flax 模型的组件
    @nn.compact
    # 从 transformers.models.bart.modeling_flax_bart.FlaxBartAttention._concatenate_to_cache 复制的代码
```  
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slightly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # 检测是否通过缺少现有缓存数据进行初始化。
        is_initialized = self.has_variable("cache", "cached_key")
        # 获取或创建缓存的键状态变量
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        # 获取或创建缓存的值状态变量
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        # 获取或创建缓存索引变量
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            # 获取批次维度、最大长度、头数以及每个头的深度
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # 使用新的一维空间切片更新键、值缓存
            cur_index = cache_index.value
            # 索引位置
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            # 动态更新键
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            # 动态更新值
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            # 更新缓存键值
            cached_key.value = key
            cached_value.value = value
            # 更新缓存向量数量
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # 为缓存的解码器自注意力创建因果掩码：我们的单个查询位置应仅注意已生成并缓存的键位置，而不是剩余的零元素。
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            # 合并掩码
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
# FlaxRobertaPreLayerNormSelfOutput 类
class FlaxRobertaPreLayerNormSelfOutput(nn.Module):
    # 类属性，为 RobertaPreLayerNormConfig 类型的 config 参数
    config: RobertaPreLayerNormConfig
    # 类属性，值为 jnp.float32 的 jnp.dtype
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    # 类方法
    def setup(self):
        # 创建一个 nn.Dense 对象，并使用 config 和 dtype 参数进行初始化
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 创建一个 nn.Dropout 对象，并使用 config 的 hidden_dropout_prob 参数进行初始化
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    # 类方法
    def __call__(self, hidden_states, input_tensor, deterministic: bool = True):
        # 将 hidden_states 传递给 self.dense 处理，并将结果赋值给 hidden_states
        hidden_states = self.dense(hidden_states)
        # 将 hidden_states 传递给 self.dropout 处理，并将结果赋值给 hidden_states
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 将 input_tensor 加上 hidden_states，并将结果返回
        hidden_states = hidden_states + input_tensor
        return hidden_states


# FlaxRobertaPreLayerNormAttention 类
class FlaxRobertaPreLayerNormAttention(nn.Module):
    # 类属性，为 RobertaPreLayerNormConfig 类型的 config 参数
    config: RobertaPreLayerNormConfig
    # 类属性，值为 False 的 bool 类型的变量
    causal: bool = False
    # 类属性，值为 jnp.float32 的 jnp.dtype
    dtype: jnp.dtype = jnp.float32

    # 类方法
    def setup(self):
        # 创建一个 FlaxRobertaPreLayerNormSelfAttention 对象，
        # 并使用 config、causal 和 dtype 参数进行初始化
        self.self = FlaxRobertaPreLayerNormSelfAttention(self.config, causal=self.causal, dtype=self.dtype)
        # 创建一个 FlaxRobertaPreLayerNormSelfOutput 对象，
        # 并使用 config 和 dtype 参数进行初始化
        self.output = FlaxRobertaPreLayerNormSelfOutput(self.config, dtype=self.dtype)
        # 创建一个 nn.LayerNorm 对象，并使用 config 的 layer_norm_eps 和 dtype 参数进行初始化
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    # 类方法
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
        # 将 hidden_states 传递给 self.LayerNorm 处理，并将结果赋值给 hidden_states_pre_layer_norm
        hidden_states_pre_layer_norm = self.LayerNorm(hidden_states)
        # 调用 self.self() 方法，将 hidden_states_pre_layer_norm 和其他参数传入，
        # 并将返回的结果赋值给 attn_outputs
        attn_outputs = self.self(
            hidden_states_pre_layer_norm,
            attention_mask,
            layer_head_mask=layer_head_mask,
            key_value_states=key_value_states,
            init_cache=init_cache,
            deterministic=deterministic,
            output_attentions=output_attentions,
        )
        # 取 attn_outputs 的第一个元素，并将结果赋值给 attn_output
        attn_output = attn_outputs[0]
        # 调用 self.output() 方法，将 attn_output、hidden_states 和其他参数传入，
        # 并将返回的结果赋值给 hidden_states
        hidden_states = self.output(attn_output, hidden_states, deterministic=deterministic)

        # 创建一个元组 outputs，包含 hidden_states
        outputs = (hidden_states,)

        # 如果 output_attentions 为 True，将 attn_outputs 的第二个元素添加到 outputs 中
        if output_attentions:
            outputs += (attn_outputs[1],)

        # 返回 outputs
        return outputs


# FlaxRobertaPreLayerNormIntermediate 类
class FlaxRobertaPreLayerNormIntermediate(nn.Module):
    # 类属性，为 RobertaPreLayerNormConfig 类型的 config 参数
    config: RobertaPreLayerNormConfig
    # 类属性，值为 jnp.float32 的 jnp.dtype
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    # 类方法
    def setup(self):
        # 创建一个 nn.LayerNorm 对象，并使用 config 的 layer_norm_eps 和 dtype 参数进行初始化
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 创建一个 nn.Dense 对象，并使用 config 的 intermediate_size、initializer_range 和 dtype 参数进行初始化
        self.dense = nn.Dense(
            self.config.intermediate_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 创建一个 ACT2FN 字典，并将 config 的 hidden_act 参数作为键，获得对应的激活函数，并将结果赋值给 activation
        self.activation = ACT2FN[self.config.hidden_act]
    # 定义一个类方法，接收隐藏状态作为参数
    def __call__(self, hidden_states):
        # 对隐藏状态进行 LayerNorm 处理
        hidden_states = self.LayerNorm(hidden_states)
        # 对处理后的隐藏状态进行全连接操作
        hidden_states = self.dense(hidden_states)
        # 对全连接操作后的结果进行激活函数处理
        hidden_states = self.activation(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states
# 定义一个 FlaxRobertaPreLayerNormOutput 类，继承自 nn.Module，用于输出预训练层归一化后的 RoBERTa 模型的结果
class FlaxRobertaPreLayerNormOutput(nn.Module):
    # 定义配置参数
    config: RobertaPreLayerNormConfig
    # 定义计算的数据类型，默认为 float32
    dtype: jnp.dtype = jnp.float32

    # 设置模型层
    def setup(self):
        # 添加一个全连接层，输入为 config.hidden_size，输出为 config.hidden_size，使用正态分布初始化权重
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 添加一个 Dropout 层，dropout 概率为 config.hidden_dropout_prob
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    # 定义前向传播过程
    def __call__(self, hidden_states, attention_output, deterministic: bool = True):
        # 通过全连接层得到隐藏状态
        hidden_states = self.dense(hidden_states)
        # 对隐藏状态应用 Dropout
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 将 attention_output 与 hidden_states 相加得到最终输出
        hidden_states = hidden_states + attention_output
        return hidden_states

# 定义一个 FlaxRobertaPreLayerNormLayer 类，继承自 nn.Module，用于实现 RoBERTa 预训练层归一化后的计算
class FlaxRobertaPreLayerNormLayer(nn.Module):
    # 定义配置参数
    config: RobertaPreLayerNormConfig
    # 定义计算的数据类型，默认为 float32
    dtype: jnp.dtype = jnp.float32

    # 设置模型层
    def setup(self):
        # 添加一个 FlaxRobertaPreLayerNormAttention 层
        self.attention = FlaxRobertaPreLayerNormAttention(self.config, causal=self.config.is_decoder, dtype=self.dtype)
        # 添加一个 FlaxRobertaPreLayerNormIntermediate 层
        self.intermediate = FlaxRobertaPreLayerNormIntermediate(self.config, dtype=self.dtype)
        # 添加一个 FlaxRobertaPreLayerNormOutput 层
        self.output = FlaxRobertaPreLayerNormOutput(self.config, dtype=self.dtype)
        # 如果需要交叉注意力，则添加一个 FlaxRobertaPreLayerNormAttention 层
        if self.config.add_cross_attention:
            self.crossattention = FlaxRobertaPreLayerNormAttention(self.config, causal=False, dtype=self.dtype)

    # 定义前向传播过程
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
        # 此处省略了后续的代码实现
    ):
        # Self Attention  self-attention层，用于对输入的hidden_states进行自注意力计算
        attention_outputs = self.attention(
            hidden_states,  # 输入的隐状态
            attention_mask,  # 注意力掩码
            layer_head_mask=layer_head_mask,  # 层和头的掩码
            init_cache=init_cache,  # 初始缓存
            deterministic=deterministic,  # 是否确定性计算
            output_attentions=output_attentions,  # 是否输出注意力值
        )
        attention_output = attention_outputs[0]  # 注意力层的输出

        # Cross-Attention Block 跨attention块
        if encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output,  # 输入的注意力输出
                attention_mask=encoder_attention_mask,  # 编码器注意力掩码
                layer_head_mask=layer_head_mask,  # 层和头的掩码
                key_value_states=encoder_hidden_states,  # 编码器的隐藏层状态
                deterministic=deterministic,  # 是否确定性计算
                output_attentions=output_attentions,  # 是否输出注意力值
            )
            attention_output = cross_attention_outputs[0]  # 跨attention块的输出

        hidden_states = self.intermediate(attention_output)  # 经过一层全连接层
        hidden_states = self.output(hidden_states, attention_output, deterministic=deterministic)  # 经过一层全连接层

        outputs = (hidden_states,)  # 将隐状态存储到元组中

        if output_attentions:  # 如果需要输出注意力值
            outputs += (attention_outputs[1],)  # 将自注意力值存储到元组中
            if encoder_hidden_states is not None:
                outputs += (cross_attention_outputs[1],)  # 将跨attention块的注意力值存储到元组中
        return outputs  # 返回输出结果元组
# 从transformers.models.bert.modeling_flax_bert.FlaxBertLayerCollection复制代码，并将Bert->RobertaPreLayerNorm
class FlaxRobertaPreLayerNormLayerCollection(nn.Module):
    config: RobertaPreLayerNormConfig
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型
    gradient_checkpointing: bool = False  # 梯度检查点

    def setup(self):
        # 如果使用梯度检查点，则使用remat函数对FlaxRobertaPreLayerNormLayer进行处理，静态参数为(5, 6, 7)
        if self.gradient_checkpointing:
            FlaxRobertaPreLayerNormCheckpointLayer = remat(FlaxRobertaPreLayerNormLayer, static_argnums=(5, 6, 7))
            self.layers = [
                FlaxRobertaPreLayerNormCheckpointLayer(self.config, name=str(i), dtype=self.dtype)
                for i in range(self.config.num_hidden_layers)
            ]
        # 否则，直接创建FlaxRobertaPreLayerNormLayer对象
        else:
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
    # 检查是否需要输出注意力和隐藏状态, 并根据需要初始化相关的元组
    all_attentions = () if output_attentions else None
    all_hidden_states = () if output_hidden_states else None
    all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
    
    # 检查 head_mask 的层数是否正确
    if head_mask is not None:
        if head_mask.shape[0] != (len(self.layers)):
            raise ValueError(
                f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.shape[0]}."
            )
    
    # 循环处理每一层
    for i, layer in enumerate(self.layers):
        # 如果需要输出隐藏状态, 则将当前隐藏状态添加到 all_hidden_states
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
    
        # 调用当前层的前向计算, 获得输出
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
    
        # 如果需要输出注意力, 则将注意力添加到 all_attentions
        if output_attentions:
            all_attentions += (layer_outputs[1],)
    
            # 如果存在 encoder_hidden_states, 则将跨注意力添加到 all_cross_attentions
            if encoder_hidden_states is not None:
                all_cross_attentions += (layer_outputs[2],)
    
    # 如果需要输出隐藏状态, 则将最终隐藏状态添加到 all_hidden_states
    if output_hidden_states:
        all_hidden_states += (hidden_states,)
    
    # 构建输出结果
    outputs = (hidden_states, all_hidden_states, all_attentions, all_cross_attentions)
    
    # 根据 return_dict 选择返回类型
    if not return_dict:
        return tuple(v for v in outputs if v is not None)
    
    return FlaxBaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        hidden_states=all_hidden_states,
        attentions=all_attentions,
        cross_attentions=all_cross_attentions,
    )
# 从transformers.models.bert.modeling_flax_bert.FlaxBertEncoder复制代码，并将Bert->RobertaPreLayerNorm
class FlaxRobertaPreLayerNormEncoder(nn.Module):
    # 定义config为RobertaPreLayerNormConfig类型的参数
    config: RobertaPreLayerNormConfig
    # 定义dtype为jnp.float32，默认为计算的数据类型
    dtype: jnp.dtype = jnp.float32  
    # 定义gradient_checkpointing为bool类型，默认为False
    gradient_checkpointing: bool = False

    def setup(self):
        # 创建FlaxRobertaPreLayerNormLayerCollection对象，并传入config、dtype和gradient_checkpointing参数
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
        # 调用self.layer对象并传入参数，返回结果
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


# 从transformers.models.bert.modeling_flax_bert.FlaxBertPooler复制代码，并将Bert->RobertaPreLayerNorm
class FlaxRobertaPreLayerNormPooler(nn.Module):
    # 定义config为RobertaPreLayerNormConfig类型的参数
    config: RobertaPreLayerNormConfig
    # 定义dtype为jnp.float32，默认为计算的数据类型
    dtype: jnp.dtype = jnp.float32  

    def setup(self):
        # 创建nn.Dense对象，并传入hidden_size、kernel_init和dtype参数
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )

    def __call__(self, hidden_states):
        # 获取hidden_states的第一个元素
        cls_hidden_state = hidden_states[:, 0]
        # 将cls_hidden_state传入self.dense对象，并返回结果
        cls_hidden_state = self.dense(cls_hidden_state)
        # 返回tanh函数作用后的cls_hidden_state
        return nn.tanh(cls_hidden_state)


# 从transformers.models.roberta.modeling_flax_roberta.FlaxRobertaLMHead复制代码，并将Roberta->RobertaPreLayerNorm
class FlaxRobertaPreLayerNormLMHead(nn.Module):
    # 定义config为RobertaPreLayerNormConfig类型的参数
    config: RobertaPreLayerNormConfig
    # 定义dtype为jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 定义bias_init为Callable类型，使用jax.nn.initializers.zeros函数初始化
    bias_init: Callable[..., np.ndarray] = jax.nn.initializers.zeros

    def setup(self):
        # 创建nn.Dense对象，并传入hidden_size、dtype和kernel_init参数
        self.dense = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        # 创建nn.LayerNorm对象，并指定epsilon和dtype参数
        self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 创建nn.Dense对象，并传入vocab_size、dtype、use_bias和kernel_init参数
        self.decoder = nn.Dense(
            self.config.vocab_size,
            dtype=self.dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        # 创建bias参数，并初始化为0，大小为vocab_size
        self.bias = self.param("bias", self.bias_init, (self.config.vocab_size,))
    #
# 定义一个继承自 nn.Module 的 FlaxRobertaPreLayerNormClassificationHead 类
# 该类用于对 RobertaPreLayerNorm 模型的输出进行分类任务
class FlaxRobertaPreLayerNormClassificationHead(nn.Module):
    # 获取 RobertaPreLayerNormConfig 配置
    config: RobertaPreLayerNormConfig
    # 设置数据类型为 float32
    dtype: jnp.dtype = jnp.float32

    # 执行模块的初始化
    def setup(self):
        # 定义一个全连接层，输入大小为 config.hidden_size，输出大小为 config.hidden_size
        # 权重初始化采用正态分布初始化方式，标准差为 config.initializer_range
        self.dense = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        # 根据配置确定分类器的dropout率，如果没有配置则使用隐藏层dropout率
        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )
        # 定义一个dropout层
        self.dropout = nn.Dropout(rate=classifier_dropout)
        # 定义一个全连接层，输入大小为 config.hidden_size，输出大小为 config.num_labels
        # 权重初始化采用正态分布初始化方式，标准差为 config.initializer_range
        self.out_proj = nn.Dense(
            self.config.num_labels,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

    # 定义前向传播过程
    def __call__(self, hidden_states, deterministic=True):
        # 取出序列的第一个token（即[CLS]token）
        hidden_states = hidden_states[:, 0, :]
        # 对token进行dropout
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 通过全连接层
        hidden_states = self.dense(hidden_states)
        # 使用tanh激活函数
        hidden_states = nn.tanh(hidden_states)
        # 再次进行dropout
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 通过最终的全连接层得到分类结果
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


# 定义一个继承自 FlaxPreTrainedModel 的 FlaxRobertaPreLayerNormPreTrainedModel 类
# 该类用于处理模型的初始化和加载
class FlaxRobertaPreLayerNormPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置配置类为 RobertaPreLayerNormConfig
    config_class = RobertaPreLayerNormConfig
    # 设置基础模型前缀为 "roberta_prelayernorm"
    base_model_prefix = "roberta_prelayernorm"

    # 设置模块类为 None
    module_class: nn.Module = None

    # 初始化模型
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
        # 根据配置创建模块
        module = self.module_class(config=config, dtype=dtype, gradient_checkpointing=gradient_checkpointing, **kwargs)
        # 调用父类的初始化方法
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    # 启用梯度检查点机制
    def enable_gradient_checkpointing(self):
        self._module = self.module_class(
            config=self.config,
            dtype=self.dtype,
            gradient_checkpointing=True,
        )
    # 初始化模型权重，根据给定的随机数种子、输入形状和参数（可选），返回初始化后的参数字典
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量
        input_ids = jnp.zeros(input_shape, dtype="i4")
        token_type_ids = jnp.ones_like(input_ids)
        position_ids = create_position_ids_from_input_ids(input_ids, self.config.pad_token_id)
        attention_mask = jnp.ones_like(input_ids)
        head_mask = jnp.ones((self.config.num_hidden_layers, self.config.num_attention_heads))

        # 分割随机数种子生成不同用途的随机数
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        if self.config.add_cross_attention:
            encoder_hidden_states = jnp.zeros(input_shape + (self.config.hidden_size,))
            encoder_attention_mask = attention_mask
            # 根据给定参数初始化模型
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
            # 根据给定参数初始化模型
            module_init_outputs = self.module.init(
                rngs, input_ids, attention_mask, token_type_ids, position_ids, head_mask, return_dict=False
            )

        random_params = module_init_outputs["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            # 对缺失的参数键进行处理
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    # 初始化缓存，用于快速自回归解码
    # Copied from transformers.models.bart.modeling_flax_bart.FlaxBartDecoderPreTrainedModel.init_cache
    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                用于快速自回归解码的批量大小。定义初始化缓存时的批量大小。
            max_length (`int`):
                自回归解码的最大可能长度。定义初始化缓存时的序列长度。
        """
        # 初始化用于获取缓存的输入变量
        input_ids = jnp.ones((batch_size, max_length), dtype="i4")
        attention_mask = jnp.ones_like(input_ids, dtype="i4")
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        # 初始化模型以获取缓存，根据给定的输入变量
        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        # 返回解封缓存
        return unfreeze(init_variables["cache"])

    # 将文档字符串添加到模型的前向传播中，描述 ROBERTA_PRELAYERNORM_INPUTS_DOCSTRING 模版格式
    @add_start_docstrings_to_model_forward(ROBERTA_PRELAYERNORM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 定义一个类的调用方法
    def __call__(
        self,
        # 输入的 token IDs
        input_ids,
        # 注意力掩码，用于指定哪些位置的 token 应该被注意到
        attention_mask=None,
        # token 类型 IDs，用于区分不同句子的 token
        token_type_ids=None,
        # 位置 IDs，用于标识 token 在句子中的位置
        position_ids=None,
        # 头部掩码，用于指定哪些注意力头部应该被屏蔽
        head_mask=None,
        # 编码器隐藏状态，用于传入预训练模型的编码器隐藏状态
        encoder_hidden_states=None,
        # 编码器注意力掩码，用于指定哪些编码器注意力应该被屏蔽
        encoder_attention_mask=None,
        # 参数字典，用于传入其他参数
        params: dict = None,
        # 随机数生成器，用于生成随机数
        dropout_rng: jax.random.PRNGKey = None,
        # 是否处于训练状态
        train: bool = False,
        # 是否输出注意力权重
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典
        return_dict: Optional[bool] = None,
        # 过去的键值对，用于传入预训练模型的过去键值对
        past_key_values: dict = None,
# 定义 FlaxRobertaPreLayerNormModule 类，继承自 nn.Module
class FlaxRobertaPreLayerNormModule(nn.Module):
    # 设置类属性 config，类型为 RobertaPreLayerNormConfig
    config: RobertaPreLayerNormConfig
    # 设置类属性 dtype，默认为 jnp.float32，用于计算的数据类型
    dtype: jnp.dtype = jnp.float32
    # 设置类属性 add_pooling_layer，默认为 True，表示是否添加池化层
    add_pooling_layer: bool = True
    # 设置类属性 gradient_checkpointing，默认为 False，表示是否使用梯度检查点

    def setup(self):
        # 初始化 self.embeddings，使用 FlaxRobertaPreLayerNormEmbeddings 类
        self.embeddings = FlaxRobertaPreLayerNormEmbeddings(self.config, dtype=self.dtype)
        # 初始化 self.encoder，使用 FlaxRobertaPreLayerNormEncoder 类
        self.encoder = FlaxRobertaPreLayerNormEncoder(
            self.config,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 初始化 self.LayerNorm，使用 nn.LayerNorm 类，设置 epsilon 值为 config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 初始化 self.pooler，使用 FlaxRobertaPreLayerNormPooler 类
        self.pooler = FlaxRobertaPreLayerNormPooler(self.config, dtype=self.dtype)

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
        # 如果 token_type_ids 未传入，则初始化为与 input_ids 形状相同的全零数组
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        # 如果 position_ids 未传入，则初始化为与 input_ids 形状相同的广播后的数组
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        # 获取输入的嵌入表示
        hidden_states = self.embeddings(
            input_ids, token_type_ids, position_ids, attention_mask, deterministic=deterministic
        )
        # 编码器计算
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
        # 取编码器输出的第一个元素作为隐藏状态
        hidden_states = outputs[0]
        # 对隐藏状态进行 LayerNorm 处理
        hidden_states = self.LayerNorm(hidden_states)
        # 如果设置了添加池化层，则进行池化
        pooled = self.pooler(hidden_states) if self.add_pooling_layer else None

        # 如果不返回字典格式，则按照元组格式返回结果
        if not return_dict:
            # 如果池化结果为 None，则不返回池化结果
            if pooled is None:
                return (hidden_states,) + outputs[1:]
            # 返回隐藏状态和池化结果以及其他输出
            return (hidden_states, pooled) + outputs[1:]

        # 返回包含隐藏状态、池化结果、隐藏状态序列、注意力分布、交叉注意力分布的字典格式结果
        return FlaxBaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


@add_start_docstrings(
    # "The bare RoBERTa-PreLayerNorm Model transformer outputting raw hidden-states without any specific head on top." 的说明文档起始
    ROBERTA_PRELAYERNORM_START_DOCSTRING,
)  # 无法确定这个括号的作用，可能是多余的错误字符

# 从transformers.models.roberta.modeling_flax_roberta.FlaxRobertaModel复制并更改为Roberta->RobertaPreLayerNorm
class FlaxRobertaPreLayerNormModel(FlaxRobertaPreLayerNormPreTrainedModel):
    module_class = FlaxRobertaPreLayerNormModule

# 向FlaxRobertaPreLayerNormModel类添加示例文档字符串
append_call_sample_docstring(
    FlaxRobertaPreLayerNormModel,
    _CHECKPOINT_FOR_DOC,
    FlaxBaseModelOutputWithPooling,
    _CONFIG_FOR_DOC,
)

# 从transformers.models.roberta.modeling_flax_roberta.FlaxRobertaForMaskedLMModule复制并更改为Roberta->RobertaPreLayerNorm,roberta->roberta_prelayernorm
class FlaxRobertaPreLayerNormForMaskedLMModule(nn.Module):
    config: RobertaPreLayerNormConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        self.roberta_prelayernorm = FlaxRobertaPreLayerNormModule(
            config=self.config,
            add_pooling_layer=False,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
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
        # 模型
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
            shared_embedding = self.roberta_prelayernorm.variables["params"]["embeddings"]["word_embeddings"][
                "embedding"
            ]
        else:
            shared_embedding = None

        # 计算预测分数
        logits = self.lm_head(hidden_states, shared_embedding=shared_embedding)

        if not return_dict:
            return (logits,) + outputs[1:]

        return FlaxMaskedLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

@add_start_docstrings(
    """RoBERTa-PreLayerNorm模型在顶部具有`语言建模`头。""", ROBERTA_PRELAYERNORM_START_DOCSTRING
)
# 从transformers.models.roberta.modeling_flax_roberta.FlaxRobertaForMaskedLM复制并更改为Roberta->RobertaPreLayerNorm
class FlaxRobertaPreLayerNormForMaskedLM(FlaxRobertaPreLayerNormPreTrainedModel):
    module_class = FlaxRobertaPreLayerNormForMaskedLMModule

# 向FlaxRobertaPreLayerNormForMaskedLM类添加示例文档字符串
append_call_sample_docstring(
    FlaxRobertaPreLayerNormForMaskedLM,
    _CHECKPOINT_FOR_DOC,
    FlaxBaseModelOutputWithPooling,
    _CONFIG_FOR_DOC,
    mask="<mask>",
)
# 从transformers.models.roberta.modeling_flax_roberta.FlaxRobertaForSequenceClassificationModule复制而来，将Roberta->RobertaPreLayerNorm，roberta->roberta_prelayernorm
class FlaxRobertaPreLayerNormForSequenceClassificationModule(nn.Module):
    # 类的配置信息
    config: RobertaPreLayerNormConfig
    # 数据类型，默认为jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 是否启用梯度检查点，默认为False
    gradient_checkpointing: bool = False

    # 设置函数
    def setup(self):
        # 使用配置信息和数据类型创建RobertaPreLayerNorm模块，不添加池化层，可选择是否启用梯度检查点
        self.roberta_prelayernorm = FlaxRobertaPreLayerNormModule(
            config=self.config,
            dtype=self.dtype,
            add_pooling_layer=False,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 创建RobertaPreLayerNorm分类头
        self.classifier = FlaxRobertaPreLayerNormClassificationHead(config=self.config, dtype=self.dtype)

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
        # 模型
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
        # 使用分类器获取logits
        logits = self.classifier(sequence_output, deterministic=deterministic)

        # 如果不返回字典，则返回logits和其他输出
        if not return_dict:
            return (logits,) + outputs[1:]

        # 返回FlaxSequenceClassifierOutput对象
        return FlaxSequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# 使用起始文档字符串添加说明，将RobertaPreLayerNorm Model转换为具有顶部序列分类/回归头的模型（池化输出顶部的线性层），例如GLUE任务
class FlaxRobertaPreLayerNormForSequenceClassification(FlaxRobertaPreLayerNormPreTrainedModel):
    module_class = FlaxRobertaPreLayerNormForSequenceClassificationModule


# 添加调用示例文档字符串
append_call_sample_docstring(
    FlaxRobertaPreLayerNormForSequenceClassification,
    _CHECKPOINT_FOR_DOC,
    FlaxSequenceClassifierOutput,
    _CONFIG_FOR_DOC,
)


# 从transformers.models.bert.modeling_flax_bert.FlaxBertForMultipleChoiceModule复制而来，将Bert->RobertaPreLayerNorm，self.bert->self.roberta_prelayernorm
class FlaxRobertaPreLayerNormForMultipleChoiceModule(nn.Module):
    # 类的配置信息
    config: RobertaPreLayerNormConfig
    # 数据类型，默认为jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 是否启用梯度检查点，默认为False
    gradient_checkpointing: bool = False
    # 定义初始化方法
    def setup(self):
        # 初始化 RoBERTa 模型的预层归一化模块
        self.roberta_prelayernorm = FlaxRobertaPreLayerNormModule(
            config=self.config,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 初始化 Dropout 层
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        # 初始化分类器
        self.classifier = nn.Dense(1, dtype=self.dtype)

    # 定义调用方法
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
        # 获取输入的选项个数
        num_choices = input_ids.shape[1]
        # 若输入不为空，则将其进行reshape处理
        input_ids = input_ids.reshape(-1, input_ids.shape[-1]) if input_ids is not None else None
        attention_mask = attention_mask.reshape(-1, attention_mask.shape[-1]) if attention_mask is not None else None
        token_type_ids = token_type_ids.reshape(-1, token_type_ids.shape[-1]) if token_type_ids is not None else None
        position_ids = position_ids.reshape(-1, position_ids.shape[-1]) if position_ids is not None else None

        # 模型计算
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

        # 获取池化输出
        pooled_output = outputs[1]
        # 对池化输出进行 dropout 处理
        pooled_output = self.dropout(pooled_output, deterministic=deterministic)
        # 通过分类器获取 logits
        logits = self.classifier(pooled_output)

        # 对 logits 进行reshape，以适应多个选项
        reshaped_logits = logits.reshape(-1, num_choices)

        # 若不返回字典，则返回结果的元组
        if not return_dict:
            return (reshaped_logits,) + outputs[2:]

        # 返回多选模型的输出
        return FlaxMultipleChoiceModelOutput(
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
自定义多选分类模型，基于RobertaPreLayerNorm模型（在池化输出之上加一个线性层和softmax），用于RocStories/SWAG任务。

参数：
- multiple_choice模型
- RobertaPreLayerNorm模型的输入文档字符串


覆盖调用的文档字符串为输入提供者中的文档字符串，格式为(batch_size, num_choices, sequence_length)。


追加调用示例的文档字符串，包括检查点和模型输出。


用于标记分类任务的RobertaPreLayerNorm模块的类，用于分词分类任务。


模块参数:
- RobertaPreLayerNorm配置
- 数据类型等于jnp.float32的损失函数
- 梯度检查点默认为False

设置：
- 创建RobertaPreLayerNorm模块
- 设置dropout层
- 设置分类器层


调用函数：
- 将输入传入RobertaPreLayerNorm模块，获得输出
- 使用dropout层对输出进行处理
- 使用分类器层对处理后的结果进行分类
- 返回分类结果，包括输出、隐藏层、注意力层
    # 这段代码是针对 RobertaPreLayerNorm 模型的描述。
    # RobertaPreLayerNorm 模型是在 RobertaModel 的基础上加上了一个 token 分类头(一个线性层位于隐藏状态输出之上)
    # 这种模型通常用于命名实体识别(NER)等任务。
    # 这段描述是 ROBERTA_PRELAYERNORM_START_DOCSTRING 的一部分。
        RobertaPreLayerNorm Model with a token classification head on top (a linear layer on top of the hidden-states
        output) e.g. for Named-Entity-Recognition (NER) tasks.
        """,
        ROBERTA_PRELAYERNORM_START_DOCSTRING,
# 这是 FlaxRobertaPreLayerNormForTokenClassification 类的定义
# 它继承自 FlaxRobertaPreLayerNormPreTrainedModel 类
# 它使用 FlaxRobertaPreLayerNormForTokenClassificationModule 作为其模块类
class FlaxRobertaPreLayerNormForTokenClassification(FlaxRobertaPreLayerNormPreTrainedModel):
    module_class = FlaxRobertaPreLayerNormForTokenClassificationModule

# 在类定义后添加了一个函数调用，用于为 FlaxRobertaPreLayerNormForTokenClassification 类添加文档样例
append_call_sample_docstring(
    FlaxRobertaPreLayerNormForTokenClassification,
    _CHECKPOINT_FOR_DOC,
    FlaxTokenClassifierOutput,
    _CONFIG_FOR_DOC,
)

# 这是 FlaxRobertaPreLayerNormForQuestionAnsweringModule 类的定义
# 它继承自 nn.Module 类
# 它包含了一个 FlaxRobertaPreLayerNormModule 实例和一个 Dense 层
class FlaxRobertaPreLayerNormForQuestionAnsweringModule(nn.Module):
    # 定义了一些配置参数和数据类型
    config: RobertaPreLayerNormConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        # 初始化 FlaxRobertaPreLayerNormModule 实例
        self.roberta_prelayernorm = FlaxRobertaPreLayerNormModule(
            config=self.config,
            dtype=self.dtype,
            add_pooling_layer=False,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 初始化 Dense 层
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
        # 调用 FlaxRobertaPreLayerNormModule 实例
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
        # 处理输出结果
        hidden_states = outputs[0]
        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = jnp.split(logits, self.config.num_labels, axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if not return_dict:
            return (start_logits, end_logits) + outputs[1:]

        return FlaxQuestionAnsweringModelOutput(
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# 这是 FlaxRobertaPreLayerNormForQuestionAnswering 类的定义
# 它继承自 FlaxRobertaPreLayerNormPreTrainedModel 类
# 它使用 FlaxRobertaPreLayerNormForQuestionAnsweringModule 作为其模块类
@add_start_docstrings(
    """
    RobertaPreLayerNorm Model with a span classification head on top for extractive question-answering tasks like SQuAD
    (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    ROBERTA_PRELAYERNORM_START_DOCSTRING,
)
class FlaxRobertaPreLayerNormForQuestionAnswering(FlaxRobertaPreLayerNormPreTrainedModel):
    module_class = FlaxRobertaPreLayerNormForQuestionAnsweringModule
    # 定义变量 module_class 为 FlaxRobertaPreLayerNormForQuestionAnsweringModule 类的引用
    module_class = FlaxRobertaPreLayerNormForQuestionAnsweringModule
# 在模型类中添加调用示例的文档字符串
append_call_sample_docstring(
    FlaxRobertaPreLayerNormForQuestionAnswering,  # 调用示例所在的模型类
    _CHECKPOINT_FOR_DOC,  # 文档中的检查点说明
    FlaxQuestionAnsweringModelOutput,  # 问题回答模型输出
    _CONFIG_FOR_DOC,  # 文档中的配置说明
)

# 从transformers.models.roberta.modeling_flax_roberta.FlaxRobertaForCausalLMModule复制，将Roberta->RobertaPreLayerNorm,roberta->roberta_prelayernorm
class FlaxRobertaPreLayerNormForCausalLMModule(nn.Module):
    config: RobertaPreLayerNormConfig  # 使用RobertaPreLayerNormConfig配置
    dtype: jnp.dtype = jnp.float32  # 数据类型为jnp.float32，默认为此
    gradient_checkpointing: bool = False  # 是否使用渐变检查点，默认为False

    def setup(self):
        # 初始化RobertaPreLayerNormModule和LMHead
        self.roberta_prelayernorm = FlaxRobertaPreLayerNormModule(
            config=self.config,  # 使用给定的配置初始化
            add_pooling_layer=False,  # 不添加池化层
            dtype=self.dtype,  # 使用指定的数据类型
            gradient_checkpointing=self.gradient_checkpointing,  # 使用梯度检查点
        )
        self.lm_head = FlaxRobertaPreLayerNormLMHead(config=self.config, dtype=self.dtype)  # 初始化LM头部

    def __call__(
        self,
        input_ids,  # 输入的标识符
        attention_mask,  # 注意力遮罩
        position_ids,  # 位置标识符
        token_type_ids: Optional[jnp.ndarray] = None,  # 令牌类型标识符，默认为None
        head_mask: Optional[jnp.ndarray] = None,  # 头部遮罩，默认为None
        encoder_hidden_states: Optional[jnp.ndarray] = None,  # 编码器隐藏状态，默认为None
        encoder_attention_mask: Optional[jnp.ndarray] = None,  # 编码器注意力遮罩，默认为None
        init_cache: bool = False,  # 是否初始化缓存，默认为False
        deterministic: bool = True,  # 是否确定性，默认为True
        output_attentions: bool = False,  # 是否输出注意力，默认为False
        output_hidden_states: bool = False,  # 是否输出隐藏状态，默认为False
        return_dict: bool = True,  # 是否返回字典，默认为True
    ):
        # 模型
        outputs = self.roberta_prelayernorm(
            input_ids,  # 输入标识符
            attention_mask,  # 注意力遮罩
            token_type_ids,  # 令牌类型标识符
            position_ids,  # 位置标识符
            head_mask,  # 头部遮罩
            encoder_hidden_states=encoder_hidden_states,  # 编码器隐藏状态
            encoder_attention_mask=encoder_attention_mask,  # 编码器注意力遮罩
            init_cache=init_cache,  # 是否初始化缓存
            deterministic=deterministic,  # 是否确定性
            output_attentions=output_attentions,  # 是否输出注意力
            output_hidden_states=output_hidden_states,  # 是否输出隐藏状态
            return_dict=return_dict,  # 是否返回字典
        )

        hidden_states = outputs[0]  # 获取隐藏状态
        if self.config.tie_word_embeddings:  # 如果需要绑定词嵌入
            shared_embedding = self.roberta_prelayernorm.variables["params"]["embeddings"]["word_embeddings"]["embedding"]  # 共享的嵌入
        else:
            shared_embedding = None  # 否则为None

        # 计算预测得分
        logits = self.lm_head(hidden_states, shared_embedding=shared_embedding)

        if not return_dict:  # 如果不返回字典
            return (logits,) + outputs[1:]  # 返回元组

        return FlaxCausalLMOutputWithCrossAttentions(  # 返回FlaxCausalLMOutputWithCrossAttentions
            logits=logits,  # 预测得分
            hidden_states=outputs.hidden_states,  # 隐藏状态
            attentions=outputs.attentions,  # 注意力
            cross_attentions=outputs.cross_attentions,  # 交叉注意力
        )


@add_start_docstrings(
    """
    带有语言建模头部的RobertaPreLayerNorm模型（在隐藏状态输出的顶部是线性层），用于自回归任务。
    """,
    ROBERTA_PRELAYERNORM_START_DOCSTRING,
)
# 从transformers.models.roberta.modeling_flax_roberta.FlaxRobertaForCausalLM中复制，将Roberta->RobertaPreLayerNorm
# 定义一个名为 FlaxRobertaPreLayerNormForCausalLM 的类，它继承自 FlaxRobertaPreLayerNormPreTrainedModel
class FlaxRobertaPreLayerNormForCausalLM(FlaxRobertaPreLayerNormPreTrainedModel):
    # 设置 module_class 属性为 FlaxRobertaPreLayerNormForCausalLMModule
    module_class = FlaxRobertaPreLayerNormForCausalLMModule

    # 定义一个 prepare_inputs_for_generation 方法，用于准备输入以进行生成
    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = None):
        # 初始化缓存
        batch_size, seq_length = input_ids.shape
        past_key_values = self.init_cache(batch_size, max_length)

        # 创建一个大小为 (batch_size, max_length) 的全 1 attention_mask
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")

        # 如果提供了 attention_mask，则将其复制到 extended_attention_mask 中
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        # 如果没有提供 attention_mask，则创建一个从 0 到 seq_length-1 的 position_ids
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        # 返回一个字典，包含 past_key_values、attention_mask 和 position_ids
        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    # 定义一个 update_inputs_for_generation 方法，用于更新输入以进行下一步生成
    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        # 更新 past_key_values 和 position_ids
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs


# 将一些额外的文档字符串附加到 FlaxRobertaPreLayerNormForCausalLM 类上
append_call_sample_docstring(
    FlaxRobertaPreLayerNormForCausalLM,
    _CHECKPOINT_FOR_DOC,
    FlaxCausalLMOutputWithCrossAttentions,
    _CONFIG_FOR_DOC,
)
```