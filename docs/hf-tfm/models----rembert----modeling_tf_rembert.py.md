# `.\models\rembert\modeling_tf_rembert.py`

```
# 设置编码格式为 UTF-8
# 版权声明及许可信息
# 
# 根据 Apache 许可证 2.0 版本使用此文件
# 除非符合许可证的条款，否则不得使用此文件
# 您可以在以下网址获取许可证副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律要求或书面同意，否则本软件按"原样"分发，
# 不附带任何明示或暗示的担保或条件
# 请参阅许可证了解特定语言下的权限和限制

""" TF 2.0 RemBERT model."""

# 导入必要的库和模块
from __future__ import annotations  # 用于支持类型注释的反向兼容性

import math  # 导入数学库
from typing import Dict, Optional, Tuple, Union  # 导入类型定义

import numpy as np  # 导入 numpy 库
import tensorflow as tf  # 导入 tensorflow 库

# 导入模块中的各类输出定义
from ...activations_tf import get_tf_activation
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
# 导入模块中的各类实用函数和损失函数
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
    keras,
    keras_serializable,
    unpack_inputs,
)
# 导入模块中的各类实用函数
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
# 导入通用的实用函数
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
# 导入 RemBERT 的配置类
from .configuration_rembert import RemBertConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# RemBERT 的模型配置文档字符串
_CONFIG_FOR_DOC = "RemBertConfig"

# RemBERT 预训练模型的存档列表
TF_REMBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/rembert",
    # 查看所有 RemBERT 模型：https://huggingface.co/models?filter=rembert
]

# TFRemBertEmbeddings 类定义，用于构建来自单词、位置和标记类型嵌入的嵌入向量
class TFRemBertEmbeddings(keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    # 初始化函数，接受一个 RemBertConfig 对象作为参数
    def __init__(self, config: RemBertConfig, **kwargs):
        super().__init__(**kwargs)

        # 初始化配置
        self.config = config
        self.input_embedding_size = config.input_embedding_size  # 输入嵌入的大小
        self.max_position_embeddings = config.max_position_embeddings  # 最大位置嵌入数量
        self.initializer_range = config.initializer_range  # 初始化范围
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")  # LayerNorm 层
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)  # Dropout 层
    # 在构建函数中，用于构建模型层，初始化各种嵌入层的权重和偏置
    def build(self, input_shape=None):
        # 在 "word_embeddings" 命名空间下创建权重变量，用于词嵌入
        self.weight = self.add_weight(
            name="weight",
            shape=[self.config.vocab_size, self.input_embedding_size],
            initializer=get_initializer(self.initializer_range),
        )

        # 在 "token_type_embeddings" 命名空间下创建权重变量，用于类型嵌入
        self.token_type_embeddings = self.add_weight(
            name="embeddings",
            shape=[self.config.type_vocab_size, self.input_embedding_size],
            initializer=get_initializer(self.initializer_range),
        )

        # 在 "position_embeddings" 命名空间下创建权重变量，用于位置嵌入
        self.position_embeddings = self.add_weight(
            name="embeddings",
            shape=[self.max_position_embeddings, self.input_embedding_size],
            initializer=get_initializer(self.initializer_range),
        )

        # 如果已经构建过，直接返回
        if self.built:
            return

        # 标记该层已经构建
        self.built = True

        # 如果存在 LayerNorm 层，则构建该层，设置输入形状为 [None, None, self.config.input_embedding_size]
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.input_embedding_size])

    # 在调用函数中，根据输入张量进行嵌入操作
    def call(
        self,
        input_ids: tf.Tensor = None,
        position_ids: tf.Tensor = None,
        token_type_ids: tf.Tensor = None,
        inputs_embeds: tf.Tensor = None,
        past_key_values_length=0,
        training: bool = False,
    ) -> tf.Tensor:
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        # 断言输入张量 input_ids 或者 inputs_embeds 不为空
        assert not (input_ids is None and inputs_embeds is None)

        # 如果存在 input_ids，根据 input_ids 从权重中收集对应的嵌入向量
        if input_ids is not None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        # 获取输入嵌入的形状列表，去除最后一维（通常是 batch 维度）
        input_shape = shape_list(inputs_embeds)[:-1]

        # 如果 token_type_ids 为空，用零填充
        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)

        # 如果 position_ids 为空，根据 past_key_values_length 和输入形状的第二维度创建位置嵌入
        if position_ids is None:
            position_ids = tf.expand_dims(
                tf.range(start=past_key_values_length, limit=input_shape[1] + past_key_values_length), axis=0
            )

        # 根据 position_ids 从位置嵌入中获取对应的嵌入向量
        position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
        # 根据 token_type_ids 从类型嵌入中获取对应的嵌入向量
        token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)
        # 将输入嵌入、位置嵌入和类型嵌入相加得到最终的嵌入向量
        final_embeddings = inputs_embeds + position_embeds + token_type_embeds
        # 对最终嵌入向量进行 LayerNorm 归一化处理
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        # 根据训练模式进行 dropout 操作，避免过拟合
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)

        # 返回最终的嵌入向量
        return final_embeddings
# 从 transformers.models.bert.modeling_tf_bert.TFBertSelfAttention 复制代码并修改为使用 RemBert
class TFRemBertSelfAttention(keras.layers.Layer):
    def __init__(self, config: RemBertConfig, **kwargs):
        super().__init__(**kwargs)

        # 检查隐藏大小是否能被注意力头数整除
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_attention_heads})"
            )

        # 初始化注意力头数和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

        # 创建查询、键、值的全连接层，并初始化
        self.query = keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        self.key = keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key"
        )
        self.value = keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )
        # 设置注意力概率的dropout层
        self.dropout = keras.layers.Dropout(rate=config.attention_probs_dropout_prob)

        # 判断是否是解码器层，并保存配置
        self.is_decoder = config.is_decoder
        self.config = config

    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        # 将张量从 [batch_size, seq_length, all_head_size] 重塑为 [batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))

        # 将张量从 [batch_size, seq_length, num_attention_heads, attention_head_size] 转置为 [batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

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
        # 略，此处通常是进行自注意力计算的逻辑

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
# Copied from transformers.models.bert.modeling_tf_bert.TFBertSelfOutput with Bert->RemBert
class TFRemBertSelfOutput(keras.layers.Layer):
    def __init__(self, config: RemBertConfig, **kwargs):
        super().__init__(**kwargs)

        # 定义全连接层，用于映射隐藏状态到指定大小的向量空间
        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 定义 Layer Normalization 层，用于归一化输出的隐藏状态
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 定义 Dropout 层，用于在训练时随机断开一定比例的神经元，防止过拟合
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 通过全连接层进行线性变换
        hidden_states = self.dense(inputs=hidden_states)
        # 在训练时应用 Dropout
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 使用 Layer Normalization 并将残差连接加回来
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            # 构建全连接层
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        if getattr(self, "LayerNorm", None) is not None:
            # 构建 Layer Normalization 层
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


# Copied from transformers.models.bert.modeling_tf_bert.TFBertAttention with Bert->RemBert
class TFRemBertAttention(keras.layers.Layer):
    def __init__(self, config: RemBertConfig, **kwargs):
        super().__init__(**kwargs)

        # 初始化自注意力层
        self.self_attention = TFRemBertSelfAttention(config, name="self")
        # 初始化输出层
        self.dense_output = TFRemBertSelfOutput(config, name="output")

    def prune_heads(self, heads):
        # 精简自注意力头部，但此处未实现具体功能
        raise NotImplementedError

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
        # 使用自注意力层处理输入，得到自注意力输出
        self_outputs = self.self_attention(
            hidden_states=input_tensor,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            training=training,
        )
        # 使用输出层处理自注意力的输出，得到最终的注意力输出
        attention_output = self.dense_output(
            hidden_states=self_outputs[0], input_tensor=input_tensor, training=training
        )
        # 如果需要输出注意力，将其添加到输出中
        outputs = (attention_output,) + self_outputs[1:]

        return outputs
    # 构建方法，用于构造神经网络层，如果已经构建过则直接返回
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回，不进行重复构建
        if self.built:
            return
        # 将标志位设置为已构建
        self.built = True
        
        # 如果存在 self_attention 属性，则构建 self_attention
        if getattr(self, "self_attention", None) is not None:
            # 使用 self_attention 层的名称作为命名空间
            with tf.name_scope(self.self_attention.name):
                # 调用 self_attention 的 build 方法进行构建
                self.self_attention.build(None)
        
        # 如果存在 dense_output 属性，则构建 dense_output
        if getattr(self, "dense_output", None) is not None:
            # 使用 dense_output 层的名称作为命名空间
            with tf.name_scope(self.dense_output.name):
                # 调用 dense_output 的 build 方法进行构建
                self.dense_output.build(None)
# 从 transformers.models.bert.modeling_tf_bert.TFBertIntermediate 复制并将 Bert 替换为 RemBert
class TFRemBertIntermediate(keras.layers.Layer):
    def __init__(self, config: RemBertConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，输出单元数为 config.intermediate_size，使用指定初始化器初始化权重
        self.dense = keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        # 如果 hidden_act 是字符串类型，则根据字符串获取 TensorFlow 激活函数；否则直接使用 config.hidden_act
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        
        self.config = config

    # 定义层的前向传播逻辑，接受隐藏状态张量并返回转换后的张量
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 全连接层的前向传播，输入 hidden_states，输出转换后的 hidden_states
        hidden_states = self.dense(inputs=hidden_states)
        # 应用激活函数转换 hidden_states
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

    # 构建层，设置层的内部变量
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在 dense 层，则按照指定的形状构建它
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# 从 transformers.models.bert.modeling_tf_bert.TFBertOutput 复制并将 Bert 替换为 RemBert
class TFRemBertOutput(keras.layers.Layer):
    def __init__(self, config: RemBertConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，输出单元数为 config.hidden_size，使用指定初始化器初始化权重
        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建 LayerNormalization 层，epsilon 设置为 config.layer_norm_eps
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建 Dropout 层，dropout 率为 config.hidden_dropout_prob
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    # 定义层的前向传播逻辑，接受隐藏状态张量和输入张量，返回转换后的张量
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 全连接层的前向传播，输入 hidden_states，输出转换后的 hidden_states
        hidden_states = self.dense(inputs=hidden_states)
        # 使用 Dropout 对 hidden_states 进行处理，根据 training 参数决定是否使用
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 应用 LayerNormalization，加上输入张量 input_tensor
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states

    # 构建层，设置层的内部变量
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在 dense 层，则按照指定的形状构建它
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])
        # 如果存在 LayerNorm 层，则按照指定的形状构建它
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


# 从 transformers.models.bert.modeling_tf_bert.TFBertLayer 复制并将 Bert 替换为 RemBert
class TFRemBertLayer(keras.layers.Layer):
    # 使用给定的配置初始化 RemBert 模型
    def __init__(self, config: RemBertConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 创建 RemBertAttention 层，命名为 "attention"
        self.attention = TFRemBertAttention(config, name="attention")
        
        # 检查当前模型是否为解码器
        self.is_decoder = config.is_decoder
        
        # 检查是否添加了跨注意力机制
        self.add_cross_attention = config.add_cross_attention
        
        # 如果添加了跨注意力机制但当前模型不是解码器，则抛出错误
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            
            # 创建跨注意力机制的 RemBertAttention 层，命名为 "crossattention"
            self.crossattention = TFRemBertAttention(config, name="crossattention")
        
        # 创建 RemBertIntermediate 层，命名为 "intermediate"
        self.intermediate = TFRemBertIntermediate(config, name="intermediate")
        
        # 创建 RemBertOutput 层，命名为 "output"
        self.bert_output = TFRemBertOutput(config, name="output")
    ) -> Tuple[tf.Tensor]:
        # 定义函数的输入和输出类型，此函数返回一个元组，包含一个 TensorFlow 张量
        # decoder 单向自注意力的缓存键/值元组位于位置 1、2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 使用自注意力模块处理隐藏状态，生成自注意力输出
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
        # 获取自注意力输出的主要结果
        attention_output = self_attention_outputs[0]

        # 如果是解码器，最后一个输出是自注意力缓存的元组
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            # 如果不是解码器，添加自注意力结果（如果输出注意力权重的话）
            outputs = self_attention_outputs[1:]

        cross_attn_present_key_value = None
        # 如果是解码器且有编码器的隐藏状态作为输入
        if self.is_decoder and encoder_hidden_states is not None:
            # 如果没有设置交叉注意力层，则引发错误
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # 交叉注意力缓存的键/值元组位于过去键/值元组的位置 3、4
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 使用交叉注意力模块处理自注意力输出，生成交叉注意力输出
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
            # 获取交叉注意力输出的主要结果
            attention_output = cross_attention_outputs[0]
            # 添加交叉注意力结果（如果输出注意力权重的话）
            outputs = outputs + cross_attention_outputs[1:-1]

            # 将交叉注意力缓存添加到当前键/值元组的位置 3、4
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 使用中间层处理注意力输出的隐藏状态
        intermediate_output = self.intermediate(hidden_states=attention_output)
        # 使用 BERT 输出层处理中间层和输入的注意力输出，生成最终的层输出
        layer_output = self.bert_output(
            hidden_states=intermediate_output, input_tensor=attention_output, training=training
        )
        # 添加注意力（如果有输出的话）
        outputs = (layer_output,) + outputs

        # 如果是解码器，将注意力键/值作为最后一个输出返回
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        # 返回整个函数的输出
        return outputs
    # 构建模型的方法，用于设置模型的各个组件
    def build(self, input_shape=None):
        # 如果模型已经构建完成，则直接返回，不重复构建
        if self.built:
            return
        # 将模型标记为已构建状态
        self.built = True
        
        # 如果存在注意力层，则构建注意力层
        if getattr(self, "attention", None) is not None:
            # 使用注意力层的名称作为命名空间，构建注意力层
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        
        # 如果存在中间层，则构建中间层
        if getattr(self, "intermediate", None) is not None:
            # 使用中间层的名称作为命名空间，构建中间层
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        
        # 如果存在BERT输出层，则构建BERT输出层
        if getattr(self, "bert_output", None) is not None:
            # 使用BERT输出层的名称作为命名空间，构建BERT输出层
            with tf.name_scope(self.bert_output.name):
                self.bert_output.build(None)
        
        # 如果存在交叉注意力层，则构建交叉注意力层
        if getattr(self, "crossattention", None) is not None:
            # 使用交叉注意力层的名称作为命名空间，构建交叉注意力层
            with tf.name_scope(self.crossattention.name):
                self.crossattention.build(None)
class TFRemBertEncoder(keras.layers.Layer):
    # TFRemBertEncoder 类定义，继承自 keras.layers.Layer

    def __init__(self, config: RemBertConfig, **kwargs):
        # 初始化方法，接受一个 RemBertConfig 类型的 config 参数和额外的关键字参数

        super().__init__(**kwargs)
        # 调用父类的初始化方法

        self.config = config
        # 将传入的 config 参数保存为实例变量

        self.embedding_hidden_mapping_in = keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="embedding_hidden_mapping_in",
        )
        # 创建一个 Dense 层，用于映射输入到隐藏状态空间，使用了 config 中的 hidden_size 和 initializer_range 参数

        self.layer = [TFRemBertLayer(config, name="layer_._{}".format(i)) for i in range(config.num_hidden_layers)]
        # 创建 TFRemBertLayer 的列表，根据 num_hidden_layers 参数进行循环创建多个层

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        encoder_hidden_states: tf.Tensor,
        encoder_attention_mask: tf.Tensor,
        past_key_values: Tuple[Tuple[tf.Tensor]],
        use_cache: bool,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPastAndCrossAttentions, Tuple[tf.Tensor]]:
        # 对输入的隐藏状态进行嵌入映射，用于后续的处理
        hidden_states = self.embedding_hidden_mapping_in(inputs=hidden_states)
        # 如果需要输出所有隐藏状态，则初始化空元组
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力权重，则初始化空元组
        all_attentions = () if output_attentions else None
        # 如果需要输出交叉注意力权重且模型配置允许，则初始化空元组
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 如果需要使用缓存，则初始化空元组以存储下一层解码器的缓存
        next_decoder_cache = () if use_cache else None
        # 遍历每一层解码器层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到所有隐藏状态元组中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的过去键值对，用于解码器自注意力机制
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 执行当前层的解码器操作，包括自注意力和可能的交叉注意力
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

            # 如果使用缓存，将当前层的缓存信息添加到下一层解码器缓存中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            # 如果需要输出注意力权重，则将当前层的注意力权重添加到所有注意力元组中
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
                # 如果模型配置允许且存在编码器隐藏状态，则将当前层的交叉注意力权重添加到所有交叉注意力元组中
                if self.config.add_cross_attention and encoder_hidden_states is not None:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果需要输出最后一层的隐藏状态
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要返回字典形式的输出，则按顺序返回非空的结果元组
        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_attentions, all_cross_attentions] if v is not None
            )

        # 返回字典形式的 TFBaseModelOutputWithPastAndCrossAttentions 对象
        return TFBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )

    def build(self, input_shape=None):
        # 如果已经构建过模型，则直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果定义了嵌入隐藏映射函数，则构建该函数
        if getattr(self, "embedding_hidden_mapping_in", None) is not None:
            with tf.name_scope(self.embedding_hidden_mapping_in.name):
                self.embedding_hidden_mapping_in.build([None, None, self.config.input_embedding_size])
        # 如果定义了层序列，则逐层构建每一层
        if getattr(self, "layer", None) is not None:
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    layer.build(None)
# Copied from transformers.models.bert.modeling_tf_bert.TFBertPooler with Bert->RemBert
class TFRemBertPooler(keras.layers.Layer):
    def __init__(self, config: RemBertConfig, **kwargs):
        super().__init__(**kwargs)

        # Initialize a dense layer for pooling with specified hidden size, tanh activation, and name.
        self.dense = keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # Pooling operation by extracting the hidden state of the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(inputs=first_token_tensor)

        return pooled_output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # Build the dense layer with the configured hidden size.
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


class TFRemBertLMPredictionHead(keras.layers.Layer):
    def __init__(self, config: RemBertConfig, input_embeddings: keras.layers.Layer, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.initializer_range = config.initializer_range
        self.output_embedding_size = config.output_embedding_size
        # Dense layer for LM prediction with specified output embedding size and initializer.
        self.dense = keras.layers.Dense(
            config.output_embedding_size, kernel_initializer=get_initializer(self.initializer_range), name="dense"
        )
        # Activation function for the hidden layer.
        if isinstance(config.hidden_act, str):
            self.activation = get_tf_activation(config.hidden_act)
        else:
            self.activation = config.hidden_act
        # Layer normalization for the prediction head.
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")

    def build(self, input_shape=None):
        # Initialize weights for the LM decoder and bias.
        self.decoder = self.add_weight(
            name="decoder/weight",
            shape=[self.config.vocab_size, self.output_embedding_size],
            initializer=get_initializer(self.initializer_range),
        )
        self.decoder_bias = self.add_weight(
            shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="decoder/bias"
        )

        if self.built:
            return
        self.built = True
        # Build dense layer for the output embedding size.
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # Build layer normalization for output embedding size.
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, self.config.output_embedding_size])

    def get_output_embeddings(self) -> keras.layers.Layer:
        # Return the output embeddings layer.
        return self

    def set_output_embeddings(self, value):
        # Set the decoder weights for the LM head.
        self.decoder = value
        self.decoder.vocab_size = shape_list(value)[0]
    # 返回一个字典，包含解码器偏置的名称和变量
    def get_bias(self) -> Dict[str, tf.Variable]:
        return {"decoder_bias": self.decoder_bias}

    # 设置解码器的偏置值，并更新词汇表大小
    def set_bias(self, value: tf.Variable):
        self.decoder_bias = value["decoder_bias"]
        self.config.vocab_size = shape_list(value["decoder_bias"])[0]

    # 对隐藏状态进行一系列操作，用于解码器的推断过程
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 将隐藏状态通过全连接层
        hidden_states = self.dense(inputs=hidden_states)
        # 应用激活函数
        hidden_states = self.activation(hidden_states)
        # 获取序列长度
        seq_length = shape_list(tensor=hidden_states)[1]
        # 将隐藏状态重塑成指定形状
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.output_embedding_size])
        # 对隐藏状态进行层归一化
        hidden_states = self.LayerNorm(hidden_states)
        # 执行矩阵乘法，进行解码器的线性变换
        hidden_states = tf.matmul(a=hidden_states, b=self.decoder, transpose_b=True)
        # 将输出重塑为原始序列长度和词汇表大小的形状
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])
        # 添加解码器偏置到输出中
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.decoder_bias)
        # 返回最终的隐藏状态作为输出
        return hidden_states
# Copied from transformers.models.bert.modeling_tf_bert.TFBertMLMHead with Bert->RemBert
class TFRemBertMLMHead(keras.layers.Layer):
    def __init__(self, config: RemBertConfig, input_embeddings: keras.layers.Layer, **kwargs):
        super().__init__(**kwargs)

        # 创建 TFRemBertLMPredictionHead 实例作为预测头部
        self.predictions = TFRemBertLMPredictionHead(config, input_embeddings, name="predictions")

    def call(self, sequence_output: tf.Tensor) -> tf.Tensor:
        # 调用预测头部的前向传播，生成预测分数
        prediction_scores = self.predictions(hidden_states=sequence_output)

        return prediction_scores

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "predictions", None) is not None:
            with tf.name_scope(self.predictions.name):
                # 构建预测头部的内部层
                self.predictions.build(None)


@keras_serializable
class TFRemBertMainLayer(keras.layers.Layer):
    config_class = RemBertConfig

    def __init__(self, config: RemBertConfig, add_pooling_layer: bool = True, **kwargs):
        super().__init__(**kwargs)

        # 初始化 RemBert 主层，包括配置和是否为解码器
        self.config = config
        self.is_decoder = config.is_decoder

        # 创建 TFRemBertEmbeddings、TFRemBertEncoder 和 TFRemBertPooler（如果需要的话）
        self.embeddings = TFRemBertEmbeddings(config, name="embeddings")
        self.encoder = TFRemBertEncoder(config, name="encoder")
        self.pooler = TFRemBertPooler(config, name="pooler") if add_pooling_layer else None

    def get_input_embeddings(self) -> keras.layers.Layer:
        # 返回嵌入层
        return self.embeddings

    def set_input_embeddings(self, value: tf.Variable):
        # 设置输入的词嵌入权重和词汇大小
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    @unpack_inputs
    # 从 transformers.models.bert.modeling_tf_bert.TFBertMainLayer.call 复制而来
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
        ):
        # RemBert 主层的前向传播函数，接收多个输入参数，并返回相应的输出
        pass  # 实际代码中会有进一步的实现
    # 定义神经网络层的构建方法，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 将标志位设置为已构建
        self.built = True
        # 如果存在嵌入层，则构建嵌入层
        if getattr(self, "embeddings", None) is not None:
            # 在命名空间中构建嵌入层
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        # 如果存在编码器，则构建编码器
        if getattr(self, "encoder", None) is not None:
            # 在命名空间中构建编码器
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 如果存在池化层，则构建池化层
        if getattr(self, "pooler", None) is not None:
            # 在命名空间中构建池化层
            with tf.name_scope(self.pooler.name):
                self.pooler.build(None)
# 添加类 TFRemBertPreTrainedModel，继承自 TFPreTrainedModel，用于处理权重初始化、预训练模型下载和加载的抽象类
class TFRemBertPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定配置类为 RemBertConfig
    config_class = RemBertConfig
    # 模型基础名称前缀为 "rembert"
    base_model_prefix = "rembert"


# 定义字符串 REMBERT_START_DOCSTRING，用于提供 TFRemBertModel 的文档说明
REMBERT_START_DOCSTRING = r"""

    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a [keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
    behavior.

    <Tip>

    TensorFlow models and layers in `transformers` accept two formats as input:

    - having all inputs as keyword arguments (like PyTorch models), or
    - having all inputs as a list, tuple or dict in the first positional argument.

    The reason the second format is supported is that Keras methods prefer this format when passing inputs to models
    and layers. Because of this support, when using methods like `model.fit()` things should "just work" for you - just
    pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second
    format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with
    the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first
    positional argument:

    - a single Tensor with `input_ids` only and nothing else: `model(input_ids)`
    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
    `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`
    - a dictionary with one or several input Tensors associated to the input names given in the docstring:
    `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    Note that when creating models and layers with
    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don't need to worry
    about any of this, as you can just pass inputs like you would to any other Python function!

    </Tip>

    Args:
        config ([`RemBertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义字符串 REMBERT_INPUTS_DOCSTRING，用于描述 TFRemBertModel 的输入参数说明（在此处未提供具体内容）
REMBERT_INPUTS_DOCSTRING = r"""
"""

# 使用 add_start_docstrings 装饰器为 TFRemBertModel 添加文档字符串
@add_start_docstrings(
    "The bare RemBERT Model transformer outputing raw hidden-states without any specific head on top.",
    REMBERT_START_DOCSTRING,
)
# 定义 TFRemBertModel 类，继承自 TFRemBertPreTrainedModel
class TFRemBertModel(TFRemBertPreTrainedModel):
    pass  # 实际实现 TFRemBertModel 的代码未在此处提供，因此添加一个 pass 占位符
    # 初始化方法，接受一个 RemBertConfig 对象作为配置，以及其他可变数量的输入参数和关键字参数
    def __init__(self, config: RemBertConfig, *inputs, **kwargs):
        # 调用父类的初始化方法，将配置对象及其他参数传递给父类
        super().__init__(config, *inputs, **kwargs)

        # 创建 TFRemBertMainLayer 对象，命名为 "rembert"，使用传入的配置对象
        self.rembert = TFRemBertMainLayer(config, name="rembert")

    # 使用装饰器 unpack_inputs 包装
    # 使用装饰器 add_start_docstrings_to_model_forward 添加模型前向传播的起始文档字符串
    # 使用装饰器 add_code_sample_docstrings 添加代码示例的文档字符串
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
    ) -> Union[TFBaseModelOutputWithPoolingAndCrossAttentions, Tuple[tf.Tensor]]:
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
        outputs = self.rembert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 调用模型的主体部分 `rembert` 来处理输入和可选的缓存键值对
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        # 如果已经构建过，则直接返回，避免重复构建
        self.built = True
        # 如果模型已经存在 `rembert` 属性
        if getattr(self, "rembert", None) is not None:
            # 在名字作用域中构建 `rembert` 模型
            with tf.name_scope(self.rembert.name):
                # 使用 `rembert` 的构建方法来构建模型，参数为 `None`
                self.rembert.build(None)
@add_start_docstrings("""RemBERT Model with a `language modeling` head on top.""", REMBERT_START_DOCSTRING)
# 使用装饰器为类添加文档字符串，指出其作为带有语言建模头部的 RemBERT 模型
class TFRemBertForMaskedLM(TFRemBertPreTrainedModel, TFMaskedLanguageModelingLoss):
    def __init__(self, config: RemBertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        if config.is_decoder:
            logger.warning(
                "If you want to use `TFRemBertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # 初始化 RemBERT 主层，如果配置为解码器则发出警告
        self.rembert = TFRemBertMainLayer(config, name="rembert", add_pooling_layer=False)
        # 初始化 Masked LM 头部，使用 RemBERT 嵌入作为输入
        self.mlm = TFRemBertMLMHead(config, input_embeddings=self.rembert.embeddings, name="mlm___cls")

    def get_lm_head(self) -> keras.layers.Layer:
        # 返回 Masked LM 头部的预测层
        return self.mlm.predictions

    @unpack_inputs
    @add_start_docstrings_to_model_forward(REMBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="google/rembert",
        output_type=TFMaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义模型的前向传播函数，包括各种输入参数和返回值的文档说明
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
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        # 定义函数签名和返回类型，可以返回 TFMaskedLMOutput 或包含 tf.Tensor 的元组
        outputs = self.rembert(
            input_ids=input_ids,
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
        # 从模型获取输出的序列表示
        sequence_output = outputs[0]
        # 使用 MLM 层生成预测分数
        prediction_scores = self.mlm(sequence_output=sequence_output, training=training)
        # 如果提供了标签，计算损失；否则损失为 None
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=prediction_scores)

        # 如果不要求返回字典，则组装输出元组
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TFMaskedLMOutput 对象，包含损失、预测分数、隐藏状态和注意力权重
        return TFMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果已经构建过模型，则直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果存在 rembert 模型，则构建 rembert 模型
        if getattr(self, "rembert", None) is not None:
            with tf.name_scope(self.rembert.name):
                self.rembert.build(None)
        # 如果存在 mlm 模型，则构建 mlm 模型
        if getattr(self, "mlm", None) is not None:
            with tf.name_scope(self.mlm.name):
                self.mlm.build(None)
@add_start_docstrings(
    """RemBERT Model with a `language modeling` head on top for CLM fine-tuning.""", REMBERT_START_DOCSTRING
)
# 定义 TFRemBertForCausalLM 类，继承自 TFRemBertPreTrainedModel 和 TFCausalLanguageModelingLoss
class TFRemBertForCausalLM(TFRemBertPreTrainedModel, TFCausalLanguageModelingLoss):
    def __init__(self, config: RemBertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 如果配置文件中不是解码器，发出警告信息
        if not config.is_decoder:
            logger.warning("If you want to use `TFRemBertForCausalLM` as a standalone, add `is_decoder=True.`")

        # 初始化 RemBERT 主层，不添加池化层
        self.rembert = TFRemBertMainLayer(config, name="rembert", add_pooling_layer=False)
        # 初始化 RemBERT 的 MLM 头部
        self.mlm = TFRemBertMLMHead(config, input_embeddings=self.rembert.embeddings, name="mlm___cls")

    # 获取语言建模头部
    def get_lm_head(self) -> keras.layers.Layer:
        return self.mlm.predictions

    # 从 transformers.models.bert.modeling_tf_bert.TFBertLMHeadModel.prepare_inputs_for_generation 复制过来的方法
    # 准备生成的输入数据，包括输入的 ID，过去的键值，注意力掩码等
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        # 如果没有提供注意力掩码，则创建一个全为 1 的注意力掩码
        if attention_mask is None:
            attention_mask = tf.ones(input_shape)

        # 如果有过去的键值，只使用最后一个输入 ID
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}

    # 解包输入参数，并添加代码示例的文档字符串
    @unpack_inputs
    @add_code_sample_docstrings(
        checkpoint="google/rembert",
        output_type=TFCausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义调用函数，接受多个输入参数和一些可选的参数，返回一个输出
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
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    ):
        # 如果已经建立过模型，则直接返回
        if self.built:
            return
        # 标记模型已经建立
        self.built = True
        # 如果存在 RemBERT 主层，则在命名空间下建立主层
        if getattr(self, "rembert", None) is not None:
            with tf.name_scope(self.rembert.name):
                self.rembert.build(None)
        # 如果存在 MLM 头部，则在命名空间下建立 MLM 头部
        if getattr(self, "mlm", None) is not None:
            with tf.name_scope(self.mlm.name):
                self.mlm.build(None)

@add_start_docstrings(
    """
    RemBERT Model transformer with a sequence classification/regression head on top e.g., for GLUE tasks.
    """
    REMBERT_START_DOCSTRING,



# 定义了一个 RemBERT 模型转换器，带有顶部的序列分类/回归头，例如用于GLUE任务。
# 这部分代码用于文档字符串的开头标记 REMBERT_START_DOCSTRING。


这样的注释能够准确描述每行代码的功能和作用，而不会过多或者过少地概括其含义。
# 定义一个继承自 TFRemBertPreTrainedModel 和 TFSequenceClassificationLoss 的模型类 TFRemBertForSequenceClassification
class TFRemBertForSequenceClassification(TFRemBertPreTrainedModel, TFSequenceClassificationLoss):
    
    # 初始化方法，接受一个 RemBertConfig 对象和其他输入参数
    def __init__(self, config: RemBertConfig, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 设置模型的标签数量
        self.num_labels = config.num_labels

        # 创建 TFRemBertMainLayer 对象，用于主要的 RemBert 模型
        self.rembert = TFRemBertMainLayer(config, name="rembert")
        
        # 创建一个 Dropout 层，使用配置中的 dropout 比率
        self.dropout = keras.layers.Dropout(rate=config.classifier_dropout_prob)
        
        # 创建一个全连接层作为分类器，单元数为配置中的标签数量，初始化方式为配置中的初始化范围
        self.classifier = keras.layers.Dense(
            units=config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="classifier",
        )
        
        # 保存配置对象
        self.config = config

    # 调用方法，接受多种输入参数，并返回模型的输出
    @unpack_inputs
    @add_start_docstrings_to_model_forward(REMBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="google/rembert",
        output_type=TFSequenceClassifierOutput,
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
    ) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 调用 rembert 模型进行前向传播，获取模型输出
        outputs = self.rembert(
            input_ids=input_ids,
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
        # 从模型输出中获取池化后的特征表示
        pooled_output = outputs[1]
        # 在训练时对特征表示进行 dropout 处理
        pooled_output = self.dropout(inputs=pooled_output, training=training)
        # 使用分类器模型对池化后的特征表示进行分类预测
        logits = self.classifier(inputs=pooled_output)
        # 如果提供了标签，计算损失函数
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        # 如果不需要返回字典形式的结果，则组装输出元组
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典形式的结果，则创建 TFSequenceClassifierOutput 对象
        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果模型已经构建过，直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 构建 rembert 模型的网络结构
        if getattr(self, "rembert", None) is not None:
            with tf.name_scope(self.rembert.name):
                self.rembert.build(None)
        # 构建分类器模型的网络结构
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                # 指定分类器的输入形状
                self.classifier.build([None, None, self.config.hidden_size])
@add_start_docstrings(
    """
    RemBERT Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    REMBERT_START_DOCSTRING,
)
class TFRemBertForMultipleChoice(TFRemBertPreTrainedModel, TFMultipleChoiceLoss):
    def __init__(self, config: RemBertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 初始化 RemBERT 主层
        self.rembert = TFRemBertMainLayer(config, name="rembert")
        # 添加 dropout 层
        self.dropout = keras.layers.Dropout(rate=config.classifier_dropout_prob)
        # 添加分类器层，用于多选题目
        self.classifier = keras.layers.Dense(
            units=1, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(REMBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="google/rembert",
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
        ) -> Union[TFMultipleChoiceModelOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
            where `num_choices` is the size of the second dimension of the input tensors. (See `input_ids` above)
        """

        # 如果输入了 `input_ids`，则获取其第二维和第三维的大小作为 `num_choices` 和 `seq_length`
        if input_ids is not None:
            num_choices = shape_list(input_ids)[1]
            seq_length = shape_list(input_ids)[2]
        else:
            # 否则，使用 `inputs_embeds` 的第二维和第三维作为 `num_choices` 和 `seq_length`
            num_choices = shape_list(inputs_embeds)[1]
            seq_length = shape_list(inputs_embeds)[2]

        # 将 `input_ids` 展平为形状为 `(batch_size * num_choices, seq_length)` 的张量，如果 `input_ids` 不为 `None`
        flat_input_ids = tf.reshape(tensor=input_ids, shape=(-1, seq_length)) if input_ids is not None else None
        # 将 `attention_mask` 展平为形状为 `(batch_size * num_choices, seq_length)` 的张量，如果 `attention_mask` 不为 `None`
        flat_attention_mask = (
            tf.reshape(tensor=attention_mask, shape=(-1, seq_length)) if attention_mask is not None else None
        )
        # 将 `token_type_ids` 展平为形状为 `(batch_size * num_choices, seq_length)` 的张量，如果 `token_type_ids` 不为 `None`
        flat_token_type_ids = (
            tf.reshape(tensor=token_type_ids, shape=(-1, seq_length)) if token_type_ids is not None else None
        )
        # 将 `position_ids` 展平为形状为 `(batch_size * num_choices, seq_length)` 的张量，如果 `position_ids` 不为 `None`
        flat_position_ids = (
            tf.reshape(tensor=position_ids, shape=(-1, seq_length)) if position_ids is not None else None
        )
        # 将 `inputs_embeds` 展平为形状为 `(batch_size * num_choices, seq_length, embed_dim)` 的张量，如果 `inputs_embeds` 不为 `None`
        flat_inputs_embeds = (
            tf.reshape(tensor=inputs_embeds, shape=(-1, seq_length, shape_list(inputs_embeds)[3]))
            if inputs_embeds is not None
            else None
        )
        
        # 调用 `self.rembert` 方法进行模型推理，传入展平后的输入和其他参数
        outputs = self.rembert(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            token_type_ids=flat_token_type_ids,
            position_ids=flat_position_ids,
            head_mask=head_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        
        # 获取池化后的输出，通过 dropout 进行正则化
        pooled_output = outputs[1]
        pooled_output = self.dropout(inputs=pooled_output, training=training)
        # 使用分类器进行多选项分类任务的预测
        logits = self.classifier(inputs=pooled_output)
        # 将 logits 重塑为形状为 `(batch_size, num_choices)` 的张量
        reshaped_logits = tf.reshape(tensor=logits, shape=(-1, num_choices))
        # 如果提供了 `labels`，计算损失函数
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=reshaped_logits)

        # 如果 `return_dict` 为 False，返回非字典形式的输出
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        # 如果 `return_dict` 为 True，返回包含损失、logits、隐藏状态和注意力权重的字典输出
        return TFMultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    # 如果模型已经构建完成，则直接返回，不进行重复构建
    if self.built:
        return
    # 将模型标记为已构建状态
    self.built = True

    # 如果存在名为 "rembert" 的属性，并且不为 None，则构建其模型
    if getattr(self, "rembert", None) is not None:
        # 使用 TensorFlow 的命名空间为 "rembert" 构建模型
        with tf.name_scope(self.rembert.name):
            self.rembert.build(None)

    # 如果存在名为 "classifier" 的属性，并且不为 None，则构建其模型
    if getattr(self, "classifier", None) is not None:
        # 使用 TensorFlow 的命名空间为 "classifier" 构建模型，期望输入的形状为 [None, None, self.config.hidden_size]
        with tf.name_scope(self.classifier.name):
            self.classifier.build([None, None, self.config.hidden_size])
@add_start_docstrings(
    """
    RemBERT Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    REMBERT_START_DOCSTRING,
)
class TFRemBertForTokenClassification(TFRemBertPreTrainedModel, TFTokenClassificationLoss):
    def __init__(self, config: RemBertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 初始化分类任务的标签数量
        self.num_labels = config.num_labels

        # 初始化 RemBERT 主层，不包含池化层
        self.rembert = TFRemBertMainLayer(config, name="rembert", add_pooling_layer=False)
        
        # Dropout 层，根据配置中的隐藏层dropout概率
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        
        # 分类器，输出单元数为配置中的标签数量，使用指定的初始化器范围
        self.classifier = keras.layers.Dense(
            units=config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        
        # 保存配置对象
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(REMBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="google/rembert",
        output_type=TFTokenClassifierOutput,
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
    ) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 调用自定义模型 `rembert` 进行前向传播，获取输出
        outputs = self.rembert(
            input_ids=input_ids,
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
        # 从模型输出中获取序列输出（通常是最后一层的隐藏状态）
        sequence_output = outputs[0]
        # 根据训练状态进行 dropout 操作，用于防止过拟合
        sequence_output = self.dropout(inputs=sequence_output, training=training)
        # 将 dropout 后的输出送入分类器，得到预测的 logits
        logits = self.classifier(inputs=sequence_output)
        # 如果提供了标签，则计算损失；否则损失为 None
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        # 如果不要求返回字典，则组装输出并返回
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果要求返回字典形式的输出，则构建 TFTokenClassifierOutput 对象并返回
        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回
        if self.built:
            return
        # 设置模型为已构建状态
        self.built = True
        # 如果存在 `rembert` 模型，则构建其层次结构
        if getattr(self, "rembert", None) is not None:
            with tf.name_scope(self.rembert.name):
                self.rembert.build(None)
        # 如果存在分类器 `classifier`，则构建其层次结构，并指定输入形状
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
"""
RemBERT Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
"""
@add_start_docstrings(
    """
    RemBERT Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    REMBERT_START_DOCSTRING,
)
class TFRemBertForQuestionAnswering(TFRemBertPreTrainedModel, TFQuestionAnsweringLoss):
    def __init__(self, config: RemBertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels  # 从配置中获取标签的数量

        self.rembert = TFRemBertMainLayer(config, add_pooling_layer=False, name="rembert")  # 初始化 RemBERT 主层
        self.qa_outputs = keras.layers.Dense(
            units=config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )  # 初始化用于问答输出的全连接层，输出单元数为标签数量
        self.config = config  # 保存配置参数

    @unpack_inputs
    @add_start_docstrings_to_model_forward(REMBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="google/rembert",
        output_type=TFQuestionAnsweringModelOutput,
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
        start_positions: np.ndarray | tf.Tensor | None = None,
        end_positions: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
        """
        Forward pass of the model. This function handles various inputs for question-answering tasks
        and returns model outputs such as start and end logits.

        Args:
            input_ids: Indices of input tokens in the vocabulary.
            attention_mask: Mask to avoid performing attention on padding tokens.
            token_type_ids: Segment token indices to indicate first and second portions of the inputs.
            position_ids: Indices of positions of each input sequence token in the position embeddings.
            head_mask: Mask to nullify selected heads of the self-attention modules.
            inputs_embeds: Optionally provided embeddings instead of input_ids.
            output_attentions: Whether to return attentions weights (a.k.a. self-weights).
            output_hidden_states: Whether to return all hidden-states.
            return_dict: Whether to return a dictionary instead of a tuple.
            start_positions: Ground truth for the start position of the span in the input.
            end_positions: Ground truth for the end position of the span in the input.
            training: Whether the model is in training mode.

        Returns:
            Depending on `return_dict`, either a tuple or a dictionary containing model outputs such as
            start and end logits for span prediction.
        """
    ) -> Union[TFQuestionAnsweringModelOutput, Tuple[tf.Tensor]]:
        r"""
        start_positions (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        # 调用 RoBERTa 模型进行前向传播，获取输出
        outputs = self.rembert(
            input_ids=input_ids,
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
        # 从 RoBERTa 模型的输出中获取序列输出
        sequence_output = outputs[0]
        # 将序列输出传入问答模型的输出层，得到 logits
        logits = self.qa_outputs(inputs=sequence_output)
        # 将 logits 沿着最后一个维度分割为起始位置和结束位置的 logits
        start_logits, end_logits = tf.split(value=logits, num_or_size_splits=2, axis=-1)
        # 去除维度为 1 的维度，确保维度匹配
        start_logits = tf.squeeze(input=start_logits, axis=-1)
        end_logits = tf.squeeze(input=end_logits, axis=-1)
        loss = None

        # 如果提供了起始位置和结束位置的标签，则计算损失
        if start_positions is not None and end_positions is not None:
            labels = {"start_position": start_positions}
            labels["end_position"] = end_positions
            # 调用 HF 框架的损失计算函数，计算损失值
            loss = self.hf_compute_loss(labels=labels, logits=(start_logits, end_logits))

        # 如果不要求返回字典，则根据是否有损失值返回相应的输出
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TFQuestionAnsweringModelOutput 类型的对象，包含损失值、起始位置 logits、结束位置 logits、隐藏状态和注意力权重
        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果已经构建过模型，则直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果存在 RoBERTa 模型，使用其名称作为命名空间构建模型
        if getattr(self, "rembert", None) is not None:
            with tf.name_scope(self.rembert.name):
                self.rembert.build(None)
        # 如果存在问答输出层，使用配置的隐藏大小构建输出层
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.config.hidden_size])
```