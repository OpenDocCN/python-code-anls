# `.\transformers\models\roberta_prelayernorm\modeling_tf_roberta_prelayernorm.py`

```
# 设置文件编码为 utf-8
# 版权声明，包括 Google AI Language Team Authors 和 HuggingFace Inc. team 的版权声明以及 NVIDIA CORPORATION 的版权声明
# 根据 Apache 许可证 2.0 版本规定的权限使用本文件，在合规情况下使用，可获取许可证的副本
# 访问 http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，根据许可证分发的软件基于“原样”分发，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言授权权限和限制的详细信息
# TF 2.0 RoBERTa-PreLayerNorm 模型

from __future__ import annotations

import math
import warnings
from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf

# 导入相应的 TF 激活函数
from ...activations_tf import get_tf_activation
# 导入模型输出相关类
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithPastAndCrossAttentions,
    TFBaseModelOutputWithPoolingAndCrossAttentions,
    TFCausalLMOutputWithCrossAttentions,
    TFMaskedLMOutput,
    TFMultipleChoiceModelOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
# 导入模型工具类
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFMaskedLanguageModelingLoss,
    TFModelInputType,
    TFMultipleChoiceLoss,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFTokenClassificationLoss,
    get_initializer,
    keras_serializable,
    unpack_inputs,
)
# 导入工具类，检查嵌入是否在范围内
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
# 导入日志模块
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
# 导入 RoBERTaPreLayerNorm 配置类
from .configuration_roberta_prelayernorm import RobertaPreLayerNormConfig

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "andreasmadsen/efficient_mlm_m0.40"
_CONFIG_FOR_DOC = "RobertaPreLayerNormConfig"

# 预训练模型存档列表
TF_ROBERTA_PRELAYERNORM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "andreasmadsen/efficient_mlm_m0.15",
    "andreasmadsen/efficient_mlm_m0.20",
    "andreasmadsen/efficient_mlm_m0.30",
    "andreasmadsen/efficient_mlm_m0.40",
    "andreasmadsen/efficient_mlm_m0.50",
    "andreasmadsen/efficient_mlm_m0.60",
    "andreasmadsen/efficient_mlm_m0.70",
    "andreasmadsen/efficient_mlm_m0.80",
    # 查看所有 RoBERTaWithPreLayerNorm 模型，请访问 https://huggingface.co/models?filter=roberta_with_prelayernorm
]

# 从 transformers.models.roberta.modeling_tf_roberta 中复制 TFRobertaEmbeddings 类到 TFRobertaPreLayerNormEmbeddings 类
class TFRobertaPreLayerNormEmbeddings(tf.keras.layers.Layer):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    # 初始化函数，接受配置和其他关键字参数
    def __init__(self, config, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 设置填充索引
        self.padding_idx = 1
        # 保存配置信息
        self.config = config
        # 保存隐藏层大小
        self.hidden_size = config.hidden_size
        # 保存最大位置嵌入的长度
        self.max_position_embeddings = config.max_position_embeddings
        # 保存初始化范围
        self.initializer_range = config.initializer_range
        # 创建 LayerNormalization 层，用于规范化
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建 Dropout 层，用于随机丢弃
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    # 构建模型
    def build(self, input_shape=None):
        # 在 "word_embeddings" 命名空间下创建词嵌入权重
        with tf.name_scope("word_embeddings"):
            # 添加权重参数，用于词嵌入
            self.weight = self.add_weight(
                name="weight",
                shape=[self.config.vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        # 在 "token_type_embeddings" 命名空间下创建类型嵌入权重
        with tf.name_scope("token_type_embeddings"):
            # 添加权重参数，用于类型嵌入
            self.token_type_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.config.type_vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        # 在 "position_embeddings" 命名空间下创建位置嵌入权重
        with tf.name_scope("position_embeddings"):
            # 添加权重参数，用于位置嵌入
            self.position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_position_embeddings, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 LayerNormalization 层，则构建该层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])

    # 根据输入的 id 创建位置 id
    def create_position_ids_from_input_ids(self, input_ids, past_key_values_length=0):
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding
        symbols are ignored. This is modified from fairseq's `utils.make_positions`.

        Args:
            input_ids: tf.Tensor
        Returns: tf.Tensor
        """
        # 创建掩码，用于标记不是填充符号的位置
        mask = tf.cast(tf.math.not_equal(input_ids, self.padding_idx), dtype=input_ids.dtype)
        # 计算位置索引
        incremental_indices = (tf.math.cumsum(mask, axis=1) + past_key_values_length) * mask

        # 返回位置索引加上填充索引
        return incremental_indices + self.padding_idx

    # 调用函数，用于执行前向传播
    def call(
        self,
        input_ids=None,
        position_ids=None,
        token_type_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
        training=False,
        ):
        """
        应用基于输入张量的嵌入。

        返回:
            final_embeddings (`tf.Tensor`): 输出嵌入张量。
        """
        断言：输入id和输入embeds不能同时为空

        如果输入id不为空：
            检查输入id是否在范围内，与配置词汇表大小比较
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        获取输入embeds的形状，去掉最后一个维度

        如果token类型id为空：
            使用值0填充形状为input_shape的张量

        如果位置id为空：
            如果input id不为空：
                从输入的token id创建位置id，任何填充的token保持填充状态
                position_ids = self.create_position_ids_from_input_ids(
                    input_ids=input_ids, past_key_values_length=past_key_values_length
                )
            否则：
                在轴0上扩展范围从padding_idx+1到input_shape[-1]+padding_idx+1的张量

        获取位置嵌入，根据位置id采集
        获取token类型嵌入，根据token类型id采集
        final_embeddings = inputs_embeds + position_embeds + token_type_embeds
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)

        返回final_embeddings
# 从transformers.models.bert.modeling_tf_bert.TFBertPooler中复制代码，并将Bert->RobertaPreLayerNorm
class TFRobertaPreLayerNormPooler(tf.keras.layers.Layer):
    def __init__(self, config: RobertaPreLayerNormConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，用于池化模型，将隐藏状态转换为池化输出
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 我们通过简单地获取与第一个标记对应的隐藏状态来“池化”模型。
        first_token_tensor = hidden_states[:, 0]
        # 将第一个标记的隐藏状态输入全连接层
        pooled_output = self.dense(inputs=first_token_tensor)

        return pooled_output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经构建了该层，直接返回
        if getattr(self, "dense", None) is not None:
            # 使用该层的名称空间构建全连接层
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# 从transformers.models.bert.modeling_tf_bert.TFBertSelfAttention中复制代码，并将Bert->RobertaPreLayerNorm
class TFRobertaPreLayerNormSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config: RobertaPreLayerNormConfig, **kwargs):
        super().__init__(**kwargs)

        # 如果隐藏大小不是注意力头数的倍数，则抛出值错误
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_attention_heads})"
            )

        # 定义注意力头数、注意力头大小、所有头大小、注意力头大小的平方根
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

        # 定义查询、键和值的全连接层
        self.query = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        self.key = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key"
        )
        self.value = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )
        # 定义dropout层
        self.dropout = tf.keras.layers.Dropout(rate=config.attention_probs_dropout_prob)

        # 是否是解码器
        self.is_decoder = config.is_decoder
        self.config = config
    # 将输入张量转置为注意力权重计算所需的形状
    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        # 从 [batch_size, seq_length, all_head_size] 重塑为 [batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))
        
        # 将张量从 [batch_size, seq_length, num_attention_heads, attention_head_size] 转置为 [batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])
    
    # 执行注意力计算
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        encoder_hidden_states: tf.Tensor,
        encoder_attention_mask: tf.Tensor,
        past_key_value: Tuple[tf.Tensor],
        output_attentions: bool,
        training: bool = False,
    ):
        # 构建模型组件
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "query", None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.config.hidden_size])
        if getattr(self, "key", None) is not None:
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.config.hidden_size])
        if getattr(self, "value", None) is not None:
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.config.hidden_size])
class TFRobertaPreLayerNormSelfOutput(tf.keras.layers.Layer):
    # TFRobertaPreLayerNormSelfOutput 类，继承自 tf.keras.layers.Layer 类，用于实现自注意力输出层
    
    def __init__(self, config: RobertaPreLayerNormConfig, **kwargs):
        # 初始化函数，接受一个 config 参数和其他的关键字参数
        
        super().__init__(**kwargs)
        # 调用父类的初始化函数
        
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建一个 Dense 层，该层将输入数据进行线性变换，输出维度为 config.hidden_size
        
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        # 创建一个 Dropout 层，该层在训练过程中随机丢弃一部分神经元，防止过拟合
        
        self.config = config
        # 保存 config 参数
        
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 定义 call 方法，该方法用于实现层的前向传播
        
        hidden_states = self.dense(inputs=hidden_states)
        # 经过 Dense 层的线性变换，将 hidden_states 进行线性映射
        
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 在训练过程中随机丢弃一部分神经元
        
        hidden_states = hidden_states + input_tensor
        # 将 hidden_states 和 input_tensor 进行相加
        
        return hidden_states
        # 返回输出结果
        
    def build(self, input_shape=None):
        # 定义 build 方法，在第一次调用 call 方法时自动调用，用于构建层的内部变量
        
        if self.built:
            return
        # 如果已经构建过内部变量，则直接返回
        
        self.built = True
        # 将内部变量标记为已构建
        
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 构建 Dense 层的内部变量


class TFRobertaPreLayerNormAttention(tf.keras.layers.Layer):
    # TFRobertaPreLayerNormAttention 类，继承自 tf.keras.layers.Layer 类，用于实现自注意力层
    
    def __init__(self, config: RobertaPreLayerNormConfig, **kwargs):
        # 初始化函数，接受一个 config 参数和其他的关键字参数
        
        super().__init__(**kwargs)
        # 调用父类的初始化函数
        
        self.self_attention = TFRobertaPreLayerNormSelfAttention(config, name="self")
        # 创建 TFRobertaPreLayerNormSelfAttention 类的实例，用于实现自注意力机制
        
        self.dense_output = TFRobertaPreLayerNormSelfOutput(config, name="output")
        # 创建 TFRobertaPreLayerNormSelfOutput 类的实例，用于实现自注意力输出层
        
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建 LayerNorm 层，用于对输入进行层归一化
        
        self.config = config
        # 保存 config 参数
        
    # Copied from transformers.models.bert.modeling_tf_bert.TFBertAttention.prune_heads
    def prune_heads(self, heads):
        # 剪枝操作，删除指定的注意力头
        
        raise NotImplementedError
        # 抛出 NotImplementedError 异常
        
    def call(
        self,
        input_tensor: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        encoder_hidden_states: tf.Tensor,
        encoder_attention_mask: tf.Tensor,
        past_key_value: Tuple[tf.Tensor],
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        # 定义 call 方法，该方法用于实现层的前向传播
        
        hidden_states_pre_layer_norm = self.LayerNorm(inputs=input_tensor)
        # 对输入进行层归一化
        
        self_outputs = self.self_attention(
            hidden_states=hidden_states_pre_layer_norm,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            training=training,
        )
        # 调用 self_attention 的 call 方法，实现自注意力机制
        
        attention_output = self.dense_output(
            hidden_states=self_outputs[0], input_tensor=input_tensor, training=training
        )
        # 调用 dense_output 的 call 方法，实现自注意力输出层
        
        # add attentions (possibly with past_key_value) if we output them
        outputs = (attention_output,) + self_outputs[1:]
        # 构建输出结果
        
        return outputs
        # 返回输出结果
    # 定义神经网络层构建函数，如果已经构建过，则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 标记神经网络层已经构建
        self.built = True
        # 如果存在self_attention属性，则构建self attention层
        if getattr(self, "self_attention", None) is not None:
            with tf.name_scope(self.self_attention.name):
                self.self_attention.build(None)
        # 如果存在dense_output属性，则构建全连接层
        if getattr(self, "dense_output", None) is not None:
            with tf.name_scope(self.dense_output.name):
                self.dense_output.build(None)
        # 如果存在LayerNorm属性，则构建Layer Normalization层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])
class TFRobertaPreLayerNormIntermediate(tf.keras.layers.Layer):
    def __init__(self, config: RobertaPreLayerNormConfig, **kwargs):
        super().__init__(**kwargs)

        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dense = tf.keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.LayerNorm(inputs=hidden_states)  # 对输入的 hidden_states 执行 LayerNormalization
        hidden_states = self.dense(inputs=hidden_states)  # 使用密集层 dense 对 hidden_states 进行处理
        hidden_states = self.intermediate_act_fn(hidden_states)  # 使用激活函数对 hidden_states 进行处理

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])  # 利用 config.hidden_size 构建 LayerNorm
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])  # 利用 config.hidden_size 构建密集层


class TFRobertaPreLayerNormOutput(tf.keras.layers.Layer):
    def __init__(self, config: RobertaPreLayerNormConfig, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)  # 使用 dense 处理 hidden_states
        hidden_states = self.dropout(inputs=hidden_states, training=training)  # 在训练时使用dropout
        hidden_states = hidden_states + input_tensor  # hidden_states 与 input_tensor 相加

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])  # 利用 config.intermediate_size 构建 dense 层


# Copied from transformers.models.bert.modeling_tf_bert.TFBertLayer with Bert->RobertaPreLayerNorm
class TFRobertaPreLayerNormLayer(tf.keras.layers.Layer):
    # 初始化函数，创建一个 RobertaPreLayerNorm 模型的实例
    def __init__(self, config: RobertaPreLayerNormConfig, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 创建自注意力层，命名为"attention"
        self.attention = TFRobertaPreLayerNormAttention(config, name="attention")
        # 检查是否是解码器模型
        self.is_decoder = config.is_decoder
        # 检查是否添加了跨注意力
        self.add_cross_attention = config.add_cross_attention
        # 如果添加了跨注意力
        if self.add_cross_attention:
            # 如果不是解码器模型，则抛出错误
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 创建跨注意力层，命名为"crossattention"
            self.crossattention = TFRobertaPreLayerNormAttention(config, name="crossattention")
        # 创建中间层
        self.intermediate = TFRobertaPreLayerNormIntermediate(config, name="intermediate")
        # 创建输出层
        self.bert_output = TFRobertaPreLayerNormOutput(config, name="output")

    # 模型的调用函数，用于执行正向传播
    def call(
        self,
        hidden_states: tf.Tensor,  # 输入的隐藏状态张量
        attention_mask: tf.Tensor,  # 注意力掩码张量
        head_mask: tf.Tensor,  # 头部掩码张量
        encoder_hidden_states: tf.Tensor | None,  # 编码器的隐藏状态张量
        encoder_attention_mask: tf.Tensor | None,  # 编码器的注意力掩码张量
        past_key_value: Tuple[tf.Tensor] | None,  # 过去的键值张量元组
        output_attentions: bool,  # 是否输出注意力权重
        training: bool = False,  # 是否处于训练模式，默认为 False
```  
    ) -> Tuple[tf.Tensor]:
        # 定义函数的输入和输出类型，返回一个张量元组
        # decoder的单向自注意力缓存的键/值元组在位置1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 使用自注意力层处理隐藏状态，获取自注意力的输出
        self_attention_outputs = self.attention(
            input_tensor=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=self_attn_past_key_value,
            output_attentions=output_attentions,
            training=training,
        )
        attention_output = self_attention_outputs[0]

        # 如果是解码器，最后一个输出是自注意力缓存的元组
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # 如果输出注意力权重，添加自注意力
          
        cross_attn_present_key_value = None
        # 如果是解码器且有编码器的隐藏状态
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # 交叉注意力的缓存键/值元组在过去键/值元组中的位置3,4
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 使用交叉注意力层处理自注意力的输出，并获取交叉注意力的输出
            cross_attention_outputs = self.crossattention(
                input_tensor=attention_output,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
                training=training,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # 添加交叉注意力缓存到present_key_value元组的位置3,4
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 使用中间层处理注意力输出，得到最终层��出
        intermediate_output = self.intermediate(hidden_states=attention_output)
        layer_output = self.bert_output(
            hidden_states=intermediate_output, input_tensor=attention_output, training=training
        )
        outputs = (layer_output,) + outputs  # 添加注意力权重如果有输出的话

        # 如果是解码器，将注意力的键/值作为最后一个输出
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs
    # 构建函数，用于构建模型
    def build(self, input_shape=None):
        # 如果模型已构建，则直接返回
        if self.built:
            return
        # 标记模型已构建
        self.built = True
        # 如果存在注意力层，构建注意力层
        if getattr(self, "attention", None) is not None:
            # 使用注意力层的名称作为命名空间
            with tf.name_scope(self.attention.name):
                # 构建注意力层
                self.attention.build(None)
        # 如果存在中间层，构建中间层
        if getattr(self, "intermediate", None) is not None:
            # 使用中间层的名称作为命名空间
            with tf.name_scope(self.intermediate.name):
                # 构建中间层
                self.intermediate.build(None)
        # 如果存在 BERT 输出层，构建 BERT 输出层
        if getattr(self, "bert_output", None) is not None:
            # 使用 BERT 输出层的名称作为命名空间
            with tf.name_scope(self.bert_output.name):
                # 构建 BERT 输出层
                self.bert_output.build(None)
        # 如果存在交叉注意力层，构建交叉注意力层
        if getattr(self, "crossattention", None) is not None:
            # 使用交叉注意力层的名称作为命名空间
            with tf.name_scope(self.crossattention.name):
                # 构建交叉注意力层
                self.crossattention.build(None)
# 从 transformers.models.bert.modeling_tf_bert.TFBertEncoder 复制，替换 Bert 为 RobertaPreLayerNorm
class TFRobertaPreLayerNormEncoder(tf.keras.layers.Layer):
    # 初始化 TFRobertaPreLayerNormEncoder 类的实例
    def __init__(self, config: RobertaPreLayerNormConfig, **kwargs):
        # 调用基类的初始化方法
        super().__init__(**kwargs)
        # 保存配置对象
        self.config = config
        # 创建多个 TFRobertaPreLayerNormLayer 对象，数量由 config.num_hidden_layers 指定
        self.layer = [TFRobertaPreLayerNormLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]

    # 定义类的 call 方法，用于对输入的张量进行处理
    def call(
        self,
        hidden_states: tf.Tensor,  # 输入的隐藏状态张量
        attention_mask: tf.Tensor,  # 注意力掩码张量
        head_mask: tf.Tensor,  # 头部掩码张量
        encoder_hidden_states: tf.Tensor | None,  # 编码器的隐藏状态张量
        encoder_attention_mask: tf.Tensor | None,  # 编码器的注意力掩码张量
        past_key_values: Tuple[Tuple[tf.Tensor]] | None,  # 之前的键值对
        use_cache: Optional[bool],  # 是否使用缓存
        output_attentions: bool,  # 是否输出注意力
        output_hidden_states: bool,  # 是否输出隐藏状态
        return_dict: bool,  # 是否返回字典
        training: bool = False,  # 是否处于训练模式
    ) -> Union[TFBaseModelOutputWithPastAndCrossAttentions, Tuple[tf.Tensor]]:
        # 如果输出隐藏状态，则初始化空元组
        all_hidden_states = () if output_hidden_states else None
        # 如果输出注意力，则初始化空元组
        all_attentions = () if output_attentions else None
        # 如果输出交叉注意力，则初始化空元组
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 如果使用缓存，则初始化空元组
        next_decoder_cache = () if use_cache else None
        
        # 遍历每一层
        for i, layer_module in enumerate(self.layer):
            # 如果输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            # 获取当前层的过去键值对
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            # 使用 layer_module 对隐藏状态进行处理
            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                training=training,
            )
            # 更新隐藏状态为当前层的输出
            hidden_states = layer_outputs[0]
            
            # 如果使用缓存，则将当前层的输出添加到缓存中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            
            # 如果输出注意力，则将当前层的注意力输出添加到 all_attentions 中
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
                # 如果配置允许交叉注意力并且编码器隐藏状态不为空，则添加交叉注意力
                if self.config.add_cross_attention and encoder_hidden_states is not None:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
        
        # 添加最后一层
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # 如果不返回字典，则返回元组，去除为 None 的项
        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_attentions, all_cross_attentions] if v is not None
            )
        
        # 否则返回包含所有输出的字典
        return TFBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )
    # 构建神经网络模型
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 设置构建为已完成状态
        self.built = True
        # 检查是否存在层对象
        if getattr(self, "layer", None) is not None:
            # 遍历每个层对象
            for layer in self.layer:
                # 设置 TensorFlow 命名空间
                with tf.name_scope(layer.name):
                    # 构建每个层对象
                    layer.build(None)
# 使用 keras_serializable 装饰器将类标记为可序列化的
@keras_serializable
class TFRobertaPreLayerNormMainLayer(tf.keras.layers.Layer):
    # 设置配置类为 RobertaPreLayerNormConfig
    config_class = RobertaPreLayerNormConfig

    # 初始化方法，接受配置参数和是否添加池化层的标志
    def __init__(self, config, add_pooling_layer=True, **kwargs):
        # 调用父类初始化方法
        super().__init__(**kwargs)

        # 将配置参数保存在属性中
        self.config = config
        # 是否为解码器
        self.is_decoder = config.is_decoder
        # 隐藏层的数量
        self.num_hidden_layers = config.num_hidden_layers
        # 初始化范围
        self.initializer_range = config.initializer_range
        # 是否输出注意力
        self.output_attentions = config.output_attentions
        # 是否输出隐藏状态
        self.output_hidden_states = config.output_hidden_states
        # 是否返回字典
        self.return_dict = config.use_return_dict
        # 创建编码器对象
        self.encoder = TFRobertaPreLayerNormEncoder(config, name="encoder")
        # 创建 LayerNormalization 对象
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 如果需要添加池化层，则创建池化层对象
        self.pooler = TFRobertaPreLayerNormPooler(config, name="pooler") if add_pooling_layer else None
        # embeddings 必须放在最后声明，以遵循权重顺序
        self.embeddings = TFRobertaPreLayerNormEmbeddings(config, name="embeddings")

    # 获取输入嵌入层对象
    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        return self.embeddings

    # 设置输入嵌入层对象的值
    def set_input_embeddings(self, value: tf.Variable):
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    # 剪枝模型的头部
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    # 调用方法，接受多个输入参数，并根据需要进行处理
    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        encoder_hidden_states: np.ndarray | tf.Tensor | None = None,
        encoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        # 省略了部分输入参数，根据需要补充
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 设置构建标志为已构建
        self.built = True
        # 如果有编码器存在，构建编码器
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 如果有 LayerNorm 存在，构建 LayerNorm
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])
        # 如果有池化器存在，构建池化器
        if getattr(self, "pooler", None) is not None:
            with tf.name_scope(self.pooler.name):
                self.pooler.build(None)
        # 如果有嵌入层存在，构建嵌入层
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
# 从transformers.models.roberta.modeling_tf_roberta.TFRobertaPreTrainedModel复制代码，将Roberta->RobertaPreLayerNorm，roberta->roberta_prelayernorm
class TFRobertaPreLayerNormPreTrainedModel(TFPreTrainedModel):
    """
    用于处理权重初始化和一个简单接口以下载和加载预训练模型的抽象类。
    """

    # 指定配置类为RobertaPreLayerNormConfig
    config_class = RobertaPreLayerNormConfig
    # 指定基础模型前缀为"roberta_prelayernorm"


ROBERTA_PRELAYERNORM_START_DOCSTRING = r"""

    该模型继承自[`TFPreTrainedModel`]。检查超类文档，了解库实现的所有模型通用方法（例如下载或保存、调整输入嵌入、修剪头等）。

    该模型还是一个[tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model)子类。将其用作常规TF 2.0 Keras模型，并参考与一般用法和行为相关的TF 2.0文档。

    <Tip>

    `transformers`中的TensorFlow模型和层接受两种格式作为输入：

    - 将所有输入作为关键字参数（类似于PyTorch模型），或
    - 将所有输入作为列表、元组或字典放在第一个位置参数中。

    支持第二种格式的原因是Keras方法在将输入传递给模型和层时更喜欢此格式。由于此支持，当使用`model.fit()`等方法时，应该“只需使用”的方式-只需以`model.fit()`支持的任何格式传递您的输入和标签！但是，如果要在Keras方法之外使用第二种格式，例如在使用Keras“Functional”API创建自己的层或模型时，则可以使用三种可能性将所有输入张量收集到第一个位置参数中：

    - 仅具有`input_ids`的单个张量，而无其他内容：`model(input_ids)`
    - 包含一个或多个输入张量的长度不同的列表，并按照文档字符串中的顺序给出：`model([input_ids, attention_mask])`或`model([input_ids, attention_mask, token_type_ids])`
    - 具有一个或多个输入张量的字典，与文档字符串中给出的输入名称相关联：`model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    请注意，当使用子类创建模型和层时，您不必担心任何这些，因为您可以像对待其他Python函数一样传递输入！

    </Tip>

    Parameters:
        config ([`RobertaPreLayerNormConfig`]): 带有模型所有参数的模型配置类。使用配置文件初始化不会加载与模型关联的权重，只会加载配置。查看[`~PreTrainedModel.from_pretrained`]方法以加载模型权重。
"""

ROBERTA_PRELAYERNORM_INPUTS_DOCSTRING = r"""
"""
# 该代码定义了一个 TFRobertaPreLayerNormModel 类，它是一个 TFRobertaPreLayerNormPreTrainedModel 的子类。
# 该类用于生成 RoBERTa-PreLayerNorm 模型的原始隐藏状态输出，没有任何特定的头部。
@add_start_docstrings(
    "The bare RoBERTa-PreLayerNorm Model transformer outputting raw hidden-states without any specific head on top.",
    ROBERTA_PRELAYERNORM_START_DOCSTRING,
)
class TFRobertaPreLayerNormModel(TFRobertaPreLayerNormPreTrainedModel):
    # 该方法是构造函数，用于初始化 TFRobertaPreLayerNormModel 对象。
    # 它接受一个 config 对象作为输入，并将其传递给父类的构造函数。
    # 同时还创建了一个 TFRobertaPreLayerNormMainLayer 对象作为 roberta_prelayernorm 属性。
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.roberta_prelayernorm = TFRobertaPreLayerNormMainLayer(config, name="roberta_prelayernorm")

    # 该方法是模型的前向传播方法。
    # 它接受一系列输入参数，如 input_ids、attention_mask、token_type_ids 等。
    # 这些参数用于控制模型的行为,如是否使用缓存、是否输出注意力权重等。
    # 该方法最终返回一个 TFBaseModelOutputWithPoolingAndCrossAttentions 对象,包含模型的输出。
    @unpack_inputs
    @add_start_docstrings_to_model_forward(ROBERTA_PRELAYERNORM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        encoder_hidden_states: np.ndarray | tf.Tensor | None = None,
        encoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[Tuple, TFBaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

        past_key_values (`Tuple[Tuple[tf.Tensor]]` of length `config.n_layers`)
            contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*, defaults to `True`):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`). Set to `False` during training, `True` during generation
        """
        # 调用 RoBERTa 模型的前向传播方法
        outputs = self.roberta_prelayernorm(
            input_ids=input_ids,  # 输入的 token IDs
            attention_mask=attention_mask,  # 注意力掩码，指示哪些位置是真实的序列，哪些是填充的
            token_type_ids=token_type_ids,  # token 类型 IDs，用于区分不同句子或段落的位置
            position_ids=position_ids,  # 位置 IDs，指示每个 token 的绝对位置
            head_mask=head_mask,  # 头部掩码，用于屏蔽特定的注意力头部
            inputs_embeds=inputs_embeds,  # 嵌入输入
            encoder_hidden_states=encoder_hidden_states,  # 编码器隐藏状态，用于解码器的交叉注意力
            encoder_attention_mask=encoder_attention_mask,  # 编码器注意力掩码，用于解码器的交叉注意力
            past_key_values=past_key_values,  # 预计算的键和值的隐藏状态，用于加速解码
            use_cache=use_cache,  # 是否使用缓存以加速解码
            output_attentions=output_attentions,  # 是否输出注意力权重
            output_hidden_states=output_hidden_states,  # 是否输出隐藏状态
            return_dict=return_dict,  # 是否返回字典格式的输出
            training=training,  # 是否处于训练模式
        )

        # 返回模型输出
        return outputs

    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回
        if self.built:
            return
        # 设置模型为已构建状态
        self.built = True
        # 检查是否已经存在名为 "roberta_prelayernorm" 的属性
        if getattr(self, "roberta_prelayernorm", None) is not None:
            # 使用命名空间创建 RoBERTa 层
            with tf.name_scope(self.roberta_prelayernorm.name):
                # 构建 RoBERTa 层
                self.roberta_prelayernorm.build(None)
# 定义一个 RobertaPreLayerNorm 模型的语言建模头层
class TFRobertaPreLayerNormLMHead(tf.keras.layers.Layer):
    """RobertaPreLayerNorm Head for masked language modeling."""

    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)

        # 保存配置信息
        self.config = config
        self.hidden_size = config.hidden_size
        
        # 定义一个全连接层来进行线性变换
        self.dense = tf.keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        
        # 定义一个层归一化层
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        
        # 使用 GELU 激活函数
        self.act = get_tf_activation("gelu")

        # 使用输入的词嵌入层作为解码层
        self.decoder = input_embeddings

    def build(self, input_shape=None):
        # 定义一个可训练的偏置项
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="bias")

        if self.built:
            return
        self.built = True
        
        # 构建 dense 层和 layer_norm 层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.hidden_size])

    def get_output_embeddings(self):
        return self.decoder

    def set_output_embeddings(self, value):
        self.decoder.weight = value
        self.decoder.vocab_size = shape_list(value)[0]

    def get_bias(self):
        return {"bias": self.bias}

    def set_bias(self, value):
        self.bias = value["bias"]
        self.config.vocab_size = shape_list(value["bias"])[0]

    def call(self, hidden_states):
        # 首先进行线性变换和 GELU 激活
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act(hidden_states)
        
        # 然后进行层归一化
        hidden_states = self.layer_norm(hidden_states)

        # 将结果投影到词汇表大小，并加上偏置项
        seq_length = shape_list(tensor=hidden_states)[1]
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.hidden_size])
        hidden_states = tf.matmul(a=hidden_states, b=self.decoder.weight, transpose_b=True)
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)

        return hidden_states


# 定义一个 RobertaPreLayerNorm 模型的掩码语言模型任务
@add_start_docstrings(
    """RoBERTa-PreLayerNorm Model with a `language modeling` head on top.""", ROBERTA_PRELAYERNORM_START_DOCSTRING
)
class TFRobertaPreLayerNormForMaskedLM(TFRobertaPreLayerNormPreTrainedModel, TFMaskedLanguageModelingLoss):
    pass
    # 定义一个私有变量，用于指定在加载模型时要忽略的键列表
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"lm_head.decoder.weight"]
    
    # 从transformers.models.roberta.modeling_tf_roberta.TFRobertaForMaskedLM.__init__复制而来，将ROBERTA->ROBERTA_PRELAYERNORM，Roberta->RobertaPreLayerNorm，roberta->roberta_prelayernorm
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)
    
        # 创建 TFRobertaPreLayerNormMainLayer 层，用于处理 RoBERTa 模型的主体部分，不添加池化层，设置名称为"roberta_prelayernorm"
        self.roberta_prelayernorm = TFRobertaPreLayerNormMainLayer(
            config, add_pooling_layer=False, name="roberta_prelayernorm"
        )
        # 创建 TFRobertaPreLayerNormLMHead 层，用于 RoBERTa 模型的预测层，传入配置和 RoBERTa 模型的嵌入层，设置名称为"lm_head"
        self.lm_head = TFRobertaPreLayerNormLMHead(config, self.roberta_prelayernorm.embeddings, name="lm_head")
    
    # 返回预测头部层
    def get_lm_head(self):
        return self.lm_head
    
    # 获取前缀偏置名
    def get_prefix_bias_name(self):
        # 发出警告，提醒方法get_prefix_bias_name已被弃用，请改用`get_bias`代替
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        return self.name + "/" + self.lm_head.name
    
    # 在模型前向传播中应用装饰器，解压输入，添加模型前向传播的描述文档和代码示例的描述文档，修改中的ROBERTA->ROBERTA_PRELAYERNORM，Roberta->RobertaPreLayerNorm，roberta->roberta_prelayernorm
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    ) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional`):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        # 将输入传递给 self.roberta_prelayernorm 进行处理
        outputs = self.roberta_prelayernorm(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 提取输出的序列信息
        sequence_output = outputs[0]
        # 通过 lm_head 模块预测语言模型的结果
        prediction_scores = self.lm_head(sequence_output)

        # 如果存在 labels，则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, prediction_scores)

        # 如果 return_dict 为 False，则返回(prediction_scores,) 和其他输出
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则返回 TFMaskedLMOutput 对象，包含损失、logits、hidden_states 和 attentions
        return TFMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 构建自定义层
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果包含 roberta_prelayernorm 层，则构建该层
        if getattr(self, "roberta_prelayernorm", None) is not None:
            with tf.name_scope(self.roberta_prelayernorm.name):
                self.roberta_prelayernorm.build(None)
        # 如果包含 lm_head 层，则构建该层
        if getattr(self, "lm_head", None) is not None:
            with tf.name_scope(self.lm_head.name):
                self.lm_head.build(None)
# 从 transformers.models.roberta.modeling_tf_roberta.TFRobertaForCausalLM 复制代码修改为 TFRobertaPreLayerNormForCausalLM，修改 ROBERTA->ROBERTA_PRELAYERNORM,Roberta->RobertaPreLayerNorm,roberta->roberta_prelayernorm
class TFRobertaPreLayerNormForCausalLM(TFRobertaPreLayerNormPreTrainedModel, TFCausalLanguageModelingLoss):
    # 在加载 TF 模型时，忽略掉不匹配的层，例如"pooler", "lm_head.decoder.weight"
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"lm_head.decoder.weight"]

    # 初始化方法
    def __init__(self, config: RobertaPreLayerNormConfig, *inputs, **kwargs):
        # 调用父类初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 如果不是decoder，则警告
        if not config.is_decoder:
            logger.warning(
                "If you want to use `TFRobertaPreLayerNormLMHeadModel` as a standalone, add `is_decoder=True.`"
            )

        # 初始化 TFRobertaPreLayerNormMainLayer 和 TFRobertaPreLayerNormLMHead
        self.roberta_prelayernorm = TFRobertaPreLayerNormMainLayer(
            config, add_pooling_layer=False, name="roberta_prelayernorm"
        )
        self.lm_head = TFRobertaPreLayerNormLMHead(
            config, input_embeddings=self.roberta_prelayernorm.embeddings, name="lm_head"
        )

    # 获取 lm_head
    def get_lm_head(self):
        return self.lm_head

    # 获取前缀偏置名字，已过时
    def get_prefix_bias_name(self):
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        return self.name + "/" + self.lm_head.name

    # 从 transformers.models.bert.modeling_tf_bert.TFBertLMHeadModel.prepare_inputs_for_generation 复制代码
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        # 如果没有给定注意力掩码，则创建一个全为1的注意力掩码
        if attention_mask is None:
            attention_mask = tf.ones(input_shape)

        # 如果使用了过去的关键值，截断 decoder_input_ids
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # 返回输入参数字典
        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}

    # 增加输入参数解包
    @unpack_inputs
    # 添加模型前向推理的文档字符串
    @add_start_docstrings_to_model_forward(ROBERTA_PRELAYERNORM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 添加代码示例的文档字符串
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFCausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义模型调用方法，接受多个输入参数，返回模型输出
    def call(
        self,
        input_ids: TFModelInputType | None = None,  # 输入的 token IDs，类型为 TensorFlow 模型输入类型或者 None
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 注意力掩码，类型为 NumPy 数组、TensorFlow 张量或者 None
        token_type_ids: np.ndarray | tf.Tensor | None = None,  # token 类型 IDs，类型为 NumPy 数组、TensorFlow 张量或者 None
        position_ids: np.ndarray | tf.Tensor | None = None,  # 位置 IDs，类型为 NumPy 数组、TensorFlow 张量或者 None
        head_mask: np.ndarray | tf.Tensor | None = None,  # 头部掩码，类型为 NumPy 数组、TensorFlow 张量或者 None
        inputs_embeds: np.ndarray | tf.Tensor | None = None,  # 输入嵌入，类型为 NumPy 数组、TensorFlow 张量或者 None
        encoder_hidden_states: np.ndarray | tf.Tensor | None = None,  # 编码器隐藏状态，类型为 NumPy 数组、TensorFlow 张量或者 None
        encoder_attention_mask: np.ndarray | tf.Tensor | None = None,  # 编码器注意力掩码，类型为 NumPy 数组、TensorFlow 张量或者 None
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,  # 过去的键值对，可选参数，类型为元组的元组，元素为 NumPy 数组或 TensorFlow 张量，或者 None
        use_cache: Optional[bool] = None,  # 是否使用缓存，可选参数，类型为布尔值或者 None
        output_attentions: Optional[bool] = None,  # 是否输出注意力，可选参数，类型为布尔值或者 None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选参数，类型为布尔值或者 None
        return_dict: Optional[bool] = None,  # 是否返回字典格式的输出，可选参数，类型为布尔值或者 None
        labels: np.ndarray | tf.Tensor | None = None,  # 标签，类型为 NumPy 数组、TensorFlow 张量或者 None
        training: Optional[bool] = False,  # 是否处于训练模式，可选参数，默认为 False，类型为布尔值或者 None
    # 构建模型，设置模型属性
    def build(self, input_shape=None):
        # 如果模型已构建，则直接返回
        if self.built:
            return
        # 标记模型已构建
        self.built = True
        # 如果存在 roberta_prelayernorm 属性
        if getattr(self, "roberta_prelayernorm", None) is not None:
            # 在名为 roberta_prelayernorm 的命名空间下，构建 roberta_prelayernorm 属性
            with tf.name_scope(self.roberta_prelayernorm.name):
                self.roberta_prelayernorm.build(None)
        # 如果存在 lm_head 属性
        if getattr(self, "lm_head", None) is not None:
            # 在名为 lm_head 的命名空间下，构建 lm_head 属性
            with tf.name_scope(self.lm_head.name):
                self.lm_head.build(None)
# 定义一个 RoBERTa 预训练层归一化分类头的 Keras 层
class TFRobertaPreLayerNormClassificationHead(tf.keras.layers.Layer):
    """用于句子级分类任务的头部。"""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 构建一个全连接层，激活函数为 tanh，将输入转换为 config.hidden_size 大小的向量
        self.dense = tf.keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )
        # 根据配置选择分类器的dropout率
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 添加一个 Dropout 层
        self.dropout = tf.keras.layers.Dropout(classifier_dropout)
        # 构建一个全连接层，将上一层的输出转换为 config.num_labels 大小的向量
        self.out_proj = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="out_proj"
        )
        self.config = config

    def call(self, features, training=False):
        # 取出第一个token（也就是[CLS]token）的向量
        x = features[:, 0, :]
        # 对该向量进行 Dropout
        x = self.dropout(x, training=training)
        # 通过全连接层转换为 config.hidden_size 大小的向量
        x = self.dense(x)
        # 再次进行 Dropout
        x = self.dropout(x, training=training)
        # 通过最终的全连接层转换为 config.num_labels 大小的向量
        x = self.out_proj(x)
        return x

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build([None, None, self.config.hidden_size])


@add_start_docstrings(
    """
    RoBERTa-PreLayerNorm Model transformer with a sequence classification/regression head on top (a linear layer on top
    of the pooled output) e.g. for GLUE tasks.
    """,
    ROBERTA_PRELAYERNORM_START_DOCSTRING,
)
class TFRobertaPreLayerNormForSequenceClassification(
    TFRobertaPreLayerNormPreTrainedModel, TFSequenceClassificationLoss
):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"lm_head"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        # 创建一个 RoBERTa 预训练层归一化主层
        self.roberta_prelayernorm = TFRobertaPreLayerNormMainLayer(
            config, add_pooling_layer=False, name="roberta_prelayernorm"
        )
        # 创建一个分类头
        self.classifier = TFRobertaPreLayerNormClassificationHead(config, name="classifier")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ROBERTA_PRELAYERNORM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 从transformers.models.roberta.modeling_tf_roberta.TFRobertaForSequenceClassification.call复制而来，将roberta->roberta_prelayernorm
    def call(
        self,
        input_ids: TFModelInputType | None = None,  # 输入的token IDs
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 注意力掩码
        token_type_ids: np.ndarray | tf.Tensor | None = None,  # token类型IDs
        position_ids: np.ndarray | tf.Tensor | None = None,  # 位置IDs
        head_mask: np.ndarray | tf.Tensor | None = None,  # 头掩码
        inputs_embeds: np.ndarray | tf.Tensor | None = None,  # 嵌入的输入
        output_attentions: Optional[bool] = None,  # 是否输出注意力
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典形式的输出
        labels: np.ndarray | tf.Tensor | None = None,  # 标签
        training: Optional[bool] = False,  # 是否为训练模式
    ) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:  # 返回类型

        """
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        # 使用roberta_prelayernorm模型处理输入
        outputs = self.roberta_prelayernorm(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 获取序列输出
        sequence_output = outputs[0]
        # 使用分类器模型计算logits
        logits = self.classifier(sequence_output, training=training)

        # 如果有标签，计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果不返回字典形式的输出，组合输出
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回字典形式的输出
        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 构建模型
    def build(self, input_shape=None):
        # 如果已构建，直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果roberta_prelayernorm模型存在，构建之
        if getattr(self, "roberta_prelayernorm", None) is not None:
            with tf.name_scope(self.roberta_prelayernorm.name):
                self.roberta_prelayernorm.build(None)
        # 如果classifier模型存在，构建之
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build(None)
# 定义一个具有多选分类头部的RobertaPreLayerNorm模型（在汇总输出的基础上叠加一个线性层和softmax），例如用于RocStories/SWAG任务
@add_start_docstrings(
    """
    RobertaPreLayerNorm Model with a multiple choice classification head on top (a linear layer on top of the pooled
    output and a softmax) e.g. for RocStories/SWAG tasks.
    """,
    ROBERTA_PRELAYERNORM_START_DOCSTRING,
)
# 从transformers.models.roberta.modeling_tf_roberta.TFRobertaForMultipleChoice复制过来，并将ROBERTA->ROBERTA_PRELAYERNORM,Roberta->RobertaPreLayerNorm,roberta->roberta_prelayernorm
class TFRobertaPreLayerNormForMultipleChoice(TFRobertaPreLayerNormPreTrainedModel, TFMultipleChoiceLoss):
    # 带有'.'的名称表示在从PT模型加载TF模型时授权的意外缺失的层
    _keys_to_ignore_on_load_unexpected = [r"lm_head"]
    # 带有'.'的名称表示在从PT模型加载TF模型时授权的意外多余的层
    _keys_to_ignore_on_load_missing = [r"dropout"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 初始化RobertaPreLayerNorm主层
        self.roberta_prelayernorm = TFRobertaPreLayerNormMainLayer(config, name="roberta_prelayernorm")
        # 初始化Dropout层
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        # 初始化分类层
        self.classifier = tf.keras.layers.Dense(
            1, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        # 保存配置
        self.config = config

    # 模型调用方法，包括输入、输出和前向传播的详细文档
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        ROBERTA_PRELAYERNORM_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFMultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    def call(
        self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None, labels=None
    ) -> Union[TFMultipleChoiceModelOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
            where `num_choices` is the size of the second dimension of the input tensors. (See `input_ids` above)
        """
        
        # 如果存在 input_ids，则获取 num_choices 和 seq_length；否则，从 inputs_embeds 中获取
        if input_ids is not None:
            num_choices = shape_list(input_ids)[1]
            seq_length = shape_list(input_ids)[2]
        else:
            num_choices = shape_list(inputs_embeds)[1]
            seq_length = shape_list(inputs_embeds)[2]

        # 将输入张量扁平化，便于处理
        flat_input_ids = tf.reshape(input_ids, (-1, seq_length)) if input_ids is not None else None
        flat_attention_mask = tf.reshape(attention_mask, (-1, seq_length)) if attention_mask is not None else None
        flat_token_type_ids = tf.reshape(token_type_ids, (-1, seq_length)) if token_type_ids is not None else None
        flat_position_ids = tf.reshape(position_ids, (-1, seq_length)) if position_ids is not None else None
        
        # 使用 Roberta 的前处理层处理输入
        outputs = self.roberta_prelayernorm(
            flat_input_ids,
            flat_attention_mask,
            flat_token_type_ids,
            flat_position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 获取池化后的输出
        pooled_output = outputs[1]
        # 使用 dropout 进行正则化
        pooled_output = self.dropout(pooled_output, training=training)
        # 使用分类器进行分类，得到 logits
        logits = self.classifier(pooled_output)
        # 重塑 logits 的形状
        reshaped_logits = tf.reshape(logits, (-1, num_choices))

        # 如果存在 labels，则计算损失函数；否则，损失值为 None
        loss = None if labels is None else self.hf_compute_loss(labels, reshaped_logits)

        # 如果不要求返回字典，则直接返回结果
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回多选模型输出对象
        return TFMultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 Roberta 的前处理层，则构建之
        if getattr(self, "roberta_prelayernorm", None) is not None:
            with tf.name_scope(self.roberta_prelayernorm.name):
                self.roberta_prelayernorm.build(None)
        # 如果存在分类器，则构建之
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
# 添加文档字符串，描述了该模型的作用及其用途，特别是用于令牌分类任务（如命名实体识别）的线性层在隐藏状态输出之上的模型结构
@add_start_docstrings(
    """
    RoBERTa-PreLayerNorm Model with a token classification head on top (a linear layer on top of the hidden-states
    output) e.g. for Named-Entity-Recognition (NER) tasks.
    """,
    ROBERTA_PRELAYERNORM_START_DOCSTRING,
)
# 定义 TFRobertaPreLayerNormForTokenClassification 类，继承自 TFRobertaPreLayerNormPreTrainedModel 和 TFTokenClassificationLoss
class TFRobertaPreLayerNormForTokenClassification(TFRobertaPreLayerNormPreTrainedModel, TFTokenClassificationLoss):
    # 在从 PT 模型加载 TF 模型时，带有 '.' 的名称表示授权的意外/丢失的层
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"lm_head"]
    _keys_to_ignore_on_load_missing = [r"dropout"]

    # 初始化方法
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类初始化方法
        super().__init__(config, *inputs, **kwargs)
        # 获取标签数量
        self.num_labels = config.num_labels

        # 创建 RoBERTa 模型的主层，不添加池化层
        self.roberta_prelayernorm = TFRobertaPreLayerNormMainLayer(
            config, add_pooling_layer=False, name="roberta_prelayernorm"
        )
        # 获取分类器的 dropout 参数
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 创建 Dropout 层
        self.dropout = tf.keras.layers.Dropout(classifier_dropout)
        # 创建分类器层
        self.classifier = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        # 保存配置信息
        self.config = config

    # 模型调用方法
    @unpack_inputs
    @add_start_docstrings_to_model_forward(ROBERTA_PRELAYERNORM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 从 transformers.models.roberta.modeling_tf_roberta.TFRobertaForTokenClassification.call 复制代码，将 roberta->roberta_prelayernorm
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    # 该函数用于计算 token 分类任务的输出结果
    def call(
        self,
        input_ids: TFModelInputType,
        attention_mask: Optional[TFModelInputType] = None,
        token_type_ids: Optional[TFModelInputType] = None,
        position_ids: Optional[TFModelInputType] = None,
        head_mask: Optional[TFModelInputType] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
        # 如果提供了标签 labels，则计算损失函数
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 传入 RoBERTa 预训练层，获得输出序列
        outputs = self.roberta_prelayernorm(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        sequence_output = outputs[0]
    
        # 对输出序列进行 dropout 操作
        sequence_output = self.dropout(sequence_output, training=training)
        # 将序列输入分类器，获得分类结果 logits
        logits = self.classifier(sequence_output)
    
        # 如果提供了标签 labels，则计算损失函数
        loss = None if labels is None else self.hf_compute_loss(labels, logits)
    
        # 如果不返回字典，则返回 logits 和其他输出
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
    
        # 否则返回 TFTokenClassifierOutput 对象
        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    # 构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建过了，就直接返回
        if self.built:
            return
        self.built = True
        # 构建 RoBERTa 预训练层
        if getattr(self, "roberta_prelayernorm", None) is not None:
            with tf.name_scope(self.roberta_prelayernorm.name):
                self.roberta_prelayernorm.build(None)
        # 构建分类器层
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
# 定义一个 RoBERTa-PreLayerNorm 模型，该模型在提取性问答任务（如 SQuAD）上具有一个跨度分类头部（在隐藏状态输出的顶部有线性层，用于计算“跨度起始对数”和“跨度结尾对数”）
@add_start_docstrings(
    """
    RoBERTa-PreLayerNorm Model with a span classification head on top for extractive question-answering tasks like
    SQuAD (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    ROBERTA_PRELAYERNORM_START_DOCSTRING,
)
class TFRobertaPreLayerNormForQuestionAnswering(TFRobertaPreLayerNormPreTrainedModel, TFQuestionAnsweringLoss):
    # 在从 PT 模型加载 TF 模型时，包含'.'的名称表示授权的意外/缺失层
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"lm_head"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        # 使用 TFRobertaPreLayerNormMainLayer 创建 RoBERTa-PreLayerNorm 主层
        self.roberta_prelayernorm = TFRobertaPreLayerNormMainLayer(
            config, add_pooling_layer=False, name="roberta_prelayernorm"
        )
        # 创建一个全连接层 qa_outputs，用于输出答案
        self.qa_outputs = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )
        self.config = config

    # 为模型前向传播添加注释
    @unpack_inputs
    @add_start_docstrings_to_model_forward(ROBERTA_PRELAYERNORM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 从 transformers.models.roberta.modeling_tf_roberta.TFRobertaForQuestionAnswering.call 复制而来，将roberta->roberta_prelayernorm
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        start_positions: np.ndarray | tf.Tensor | None = None,
        end_positions: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    # 定义一个函数，该函数返回值为 TFQuestionAnsweringModelOutput 或者 Tuple[tf.Tensor]
    def call(
        self, 
        input_ids: tf.Tensor, 
        attention_mask = None, 
        token_type_ids = None, 
        position_ids = None, 
        head_mask = None, 
        inputs_embeds = None, 
        output_attentions = None, 
        output_hidden_states = None, 
        return_dict = None, 
        training = False,
        start_positions = None, 
        end_positions = None
    ) -> Union[TFQuestionAnsweringModelOutput, Tuple[tf.Tensor]]:
        
        # 使用预处理层处理输入数据
        outputs = self.roberta_prelayernorm(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        
        # 获取序列输出
        sequence_output = outputs[0]

        # 对序列输出进行分类得到 logits
        logits = self.qa_outputs(sequence_output)
        
        # 将 logits 分割成 start_logits 和 end_logits
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        
        # 去除多余的维度
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        # 初始化 loss 为 None
        loss = None
        
        # 如果 start_positions 和 end_positions 非空，则计算 loss
        if start_positions is not None and end_positions is not None:
            labels = {"start_position": start_positions}
            labels["end_position"] = end_positions
            loss = self.hf_compute_loss(labels, (start_logits, end_logits))

        # 如果 return_dict 为 False，则返回包含 loss 和输出的元组
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则返回 TFQuestionAnsweringModelOutput 对象
        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 构建模型
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        
        # 如果预处理层和分类层存在，则构建它们
        if getattr(self, "roberta_prelayernorm", None) is not None:
            with tf.name_scope(self.roberta_prelayernorm.name):
                self.roberta_prelayernorm.build(None)
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.config.hidden_size])
```