# `.\models\bert\modeling_tf_bert.py`

```py
# coding=utf-8
# 版权声明：2018 年由 Google AI 语言团队和 HuggingFace Inc. 团队所有。
# 版权声明：2018 年，NVIDIA CORPORATION 版权所有。
#
# 根据 Apache 许可证 2.0 版本（"许可证"）获得许可；
# 除非符合许可证要求或书面同意，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按"原样"分发，
# 没有任何形式的明示或暗示的担保或条件。
# 有关详细信息，请参阅许可证。
""" TF 2.0 BERT 模型。"""


from __future__ import annotations

import math  # 导入数学函数库
import warnings  # 导入警告模块
from dataclasses import dataclass  # 导入 dataclass 用于定义数据类
from typing import Dict, Optional, Tuple, Union  # 导入类型提示工具

import numpy as np  # 导入 NumPy 库
import tensorflow as tf  # 导入 TensorFlow 库

from ...activations_tf import get_tf_activation  # 从本地包中导入 TensorFlow 激活函数
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithPastAndCrossAttentions,  # 导入 TFBaseModelOutputWithPastAndCrossAttentions 输出类
    TFBaseModelOutputWithPoolingAndCrossAttentions,  # 导入 TFBaseModelOutputWithPoolingAndCrossAttentions 输出类
    TFCausalLMOutputWithCrossAttentions,  # 导入 TFCausalLMOutputWithCrossAttentions 输出类
    TFMaskedLMOutput,  # 导入 TFMaskedLMOutput 输出类
    TFMultipleChoiceModelOutput,  # 导入 TFMultipleChoiceModelOutput 输出类
    TFNextSentencePredictorOutput,  # 导入 TFNextSentencePredictorOutput 输出类
    TFQuestionAnsweringModelOutput,  # 导入 TFQuestionAnsweringModelOutput 输出类
    TFSequenceClassifierOutput,  # 导入 TFSequenceClassifierOutput 输出类
    TFTokenClassifierOutput,  # 导入 TFTokenClassifierOutput 输出类
)
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,  # 导入 TFCausalLanguageModelingLoss 损失类
    TFMaskedLanguageModelingLoss,  # 导入 TFMaskedLanguageModelingLoss 损失类
    TFModelInputType,  # 导入 TFModelInputType 输入类型
    TFMultipleChoiceLoss,  # 导入 TFMultipleChoiceLoss 损失类
    TFNextSentencePredictionLoss,  # 导入 TFNextSentencePredictionLoss 损失类
    TFPreTrainedModel,  # 导入 TFPreTrainedModel 预训练模型类
    TFQuestionAnsweringLoss,  # 导入 TFQuestionAnsweringLoss 损失类
    TFSequenceClassificationLoss,  # 导入 TFSequenceClassificationLoss 损失类
    TFTokenClassificationLoss,  # 导入 TFTokenClassificationLoss 损失类
    get_initializer,  # 导入获取初始化器函数
    keras,  # 导入 Keras 库
    keras_serializable,  # 导入 Keras 序列化功能
    unpack_inputs,  # 导入解包输入函数
)
from ...tf_utils import (
    check_embeddings_within_bounds,  # 导入检查嵌入范围的函数
    shape_list,  # 导入获取张量形状的函数
    stable_softmax,  # 导入稳定 Softmax 函数
)
from ...utils import (
    ModelOutput,  # 导入模型输出类
    add_code_sample_docstrings,  # 导入添加代码示例文档字符串函数
    add_start_docstrings,  # 导入添加起始文档字符串函数
    add_start_docstrings_to_model_forward,  # 导入向前模型添加起始文档字符串函数
    logging,  # 导入日志模块
    replace_return_docstrings,  # 导入替换返回文档字符串函数
)
from .configuration_bert import BertConfig  # 从本地配置文件导入 BertConfig 类


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

_CHECKPOINT_FOR_DOC = "google-bert/bert-base-uncased"  # 预训练模型的文档检查点
_CONFIG_FOR_DOC = "BertConfig"  # BertConfig 的文档配置

# TokenClassification 文档字符串
_CHECKPOINT_FOR_TOKEN_CLASSIFICATION = "dbmdz/bert-large-cased-finetuned-conll03-english"  # 标记分类预训练模型检查点
_TOKEN_CLASS_EXPECTED_OUTPUT = (
    "['O', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'I-LOC', 'O', 'I-LOC', 'I-LOC'] "
)  # 标记分类预期输出
_TOKEN_CLASS_EXPECTED_LOSS = 0.01  # 标记分类预期损失

# QuestionAnswering 文档字符串
_CHECKPOINT_FOR_QA = "ydshieh/bert-base-cased-squad2"  # 问答预训练模型检查点
_QA_EXPECTED_OUTPUT = "'a nice puppet'"  # 问答预期输出
_QA_EXPECTED_LOSS = 7.41  # 问答预期损失
_QA_TARGET_START_INDEX = 14  # 问答目标起始索引
_QA_TARGET_END_INDEX = 15  # 问答目标结束索引

# SequenceClassification 文档字符串
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "ydshieh/bert-base-uncased-yelp-polarity"  # 序列分类预训练模型检查点
_SEQ_CLASS_EXPECTED_OUTPUT = "'LABEL_1'"  # 序列分类预期输出
_SEQ_CLASS_EXPECTED_LOSS = 0.01  # 序列分类预期损失

TF_BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google-bert/bert-base-uncased",  # 预训练模型存档列表
    "google-bert/bert-large-uncased",  # 预训练模型存档列表
    # 列出了多个预训练的BERT模型的名称，每个名称代表一个特定配置和语言的BERT模型
    [
        "google-bert/bert-base-cased",  # 谷歌的BERT基础模型，大小写敏感
        "google-bert/bert-large-cased",  # 谷歌的BERT大型模型，大小写敏感
        "google-bert/bert-base-multilingual-uncased",  # 谷歌的多语言BERT基础模型，大小写不敏感
        "google-bert/bert-base-multilingual-cased",  # 谷歌的多语言BERT基础模型，大小写敏感
        "google-bert/bert-base-chinese",  # 谷歌的中文BERT基础模型
        "google-bert/bert-base-german-cased",  # 谷歌的德语BERT基础模型，大小写敏感
        "google-bert/bert-large-uncased-whole-word-masking",  # 谷歌的大型BERT模型，全词遮盖，大小写不敏感
        "google-bert/bert-large-cased-whole-word-masking",  # 谷歌的大型BERT模型，全词遮盖，大小写敏感
        "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad",  # 谷歌的在SQuAD上微调的大型BERT模型，全词遮盖，大小写不敏感
        "google-bert/bert-large-cased-whole-word-masking-finetuned-squad",  # 谷歌的在SQuAD上微调的大型BERT模型，全词遮盖，大小写敏感
        "google-bert/bert-base-cased-finetuned-mrpc",  # 谷歌的在MRPC任务上微调的BERT基础模型，大小写敏感
        "cl-tohoku/bert-base-japanese",  # 东北大学的日语BERT基础模型
        "cl-tohoku/bert-base-japanese-whole-word-masking",  # 东北大学的日语BERT基础模型，全词遮盖
        "cl-tohoku/bert-base-japanese-char",  # 东北大学的日语BERT基础模型，字符级别
        "cl-tohoku/bert-base-japanese-char-whole-word-masking",  # 东北大学的日语BERT基础模型，字符级别，全词遮盖
        "TurkuNLP/bert-base-finnish-cased-v1",  # TurkuNLP的芬兰语BERT基础模型，大小写敏感
        "TurkuNLP/bert-base-finnish-uncased-v1",  # TurkuNLP的芬兰语BERT基础模型，大小写不敏感
        "wietsedv/bert-base-dutch-cased",  # Wietsedv的荷兰语BERT基础模型，大小写敏感
        # 查看所有BERT模型，请访问 https://huggingface.co/models?filter=bert
    ]
        super().__init__(**kwargs)

        # 初始化层参数，保存BERT配置
        self.config = config
        # 获取BERT模型隐藏层大小
        self.hidden_size = config.hidden_size
        # 获取最大位置嵌入数
        self.max_position_embeddings = config.max_position_embeddings
        # 获取初始化范围
        self.initializer_range = config.initializer_range
        # 创建LayerNorm层，并设置epsilon值
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建Dropout层，并设置丢弃率
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
    # 定义 build 方法，用于构建模型结构
    def build(self, input_shape=None):
        # 在 "word_embeddings" 命名空间下创建权重矩阵，用于词嵌入
        self.weight = self.add_weight(
            name="weight",
            shape=[self.config.vocab_size, self.hidden_size],
            initializer=get_initializer(self.initializer_range),
        )

        # 在 "token_type_embeddings" 命名空间下创建权重矩阵，用于标记类型嵌入
        self.token_type_embeddings = self.add_weight(
            name="embeddings",
            shape=[self.config.type_vocab_size, self.hidden_size],
            initializer=get_initializer(self.initializer_range),
        )

        # 在 "position_embeddings" 命名空间下创建权重矩阵，用于位置嵌入
        self.position_embeddings = self.add_weight(
            name="embeddings",
            shape=[self.max_position_embeddings, self.hidden_size],
            initializer=get_initializer(self.initializer_range),
        )

        # 如果模型已构建，则直接返回，避免重复构建
        if self.built:
            return
        self.built = True
        
        # 如果存在 LayerNorm 层，则构建 LayerNorm 层，输入形状为 [None, None, self.config.hidden_size]
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])

    # 定义 call 方法，用于执行模型前向传播
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
        # 如果没有提供 input_ids 或 inputs_embeds，则抛出 ValueError
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Need to provide either `input_ids` or `input_embeds`.")

        # 如果提供了 input_ids，则从权重矩阵中获取对应的嵌入向量
        if input_ids is not None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        input_shape = shape_list(inputs_embeds)[:-1]

        # 如果未提供 token_type_ids，则创建一个形状与 inputs_embeds 相同的全 0 张量
        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)

        # 如果未提供 position_ids，则创建一个序列张量，范围从 past_key_values_length 到 input_shape[1] + past_key_values_length
        if position_ids is None:
            position_ids = tf.expand_dims(
                tf.range(start=past_key_values_length, limit=input_shape[1] + past_key_values_length), axis=0
            )

        # 从 position_embeddings 中根据 position_ids 获取位置嵌入向量
        position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
        # 从 token_type_embeddings 中根据 token_type_ids 获取标记类型嵌入向量
        token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)
        # 计算最终的嵌入向量，包括输入嵌入、位置嵌入和标记类型嵌入
        final_embeddings = inputs_embeds + position_embeds + token_type_embeds
        # 对最终的嵌入向量应用 LayerNorm 层
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        # 对最终的嵌入向量应用 dropout，用于训练时防止过拟合
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)

        return final_embeddings
class TFBertSelfAttention(keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)

        # 检查隐藏大小是否是注意力头数的整数倍
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_attention_heads})"
            )

        # 初始化注意力头数和每个注意力头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

        # 创建用于计算查询、键和值的全连接层
        self.query = keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        self.key = keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key"
        )
        self.value = keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )
        # 配置注意力概率的丢弃层
        self.dropout = keras.layers.Dropout(rate=config.attention_probs_dropout_prob)

        # 是否作为解码器使用和配置信息
        self.is_decoder = config.is_decoder
        self.config = config

    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        # 重塑张量形状，从 [batch_size, seq_length, all_head_size] 到 [batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))

        # 转置张量，从 [batch_size, seq_length, num_attention_heads, attention_head_size] 到 [batch_size, num_attention_heads, seq_length, attention_head_size]
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
        # 本层的调用方法，将在实际使用时详细处理各种输入和输出逻辑
        pass

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建查询、键和值层，设置它们的输入形状
        if getattr(self, "query", None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.config.hidden_size])
        if getattr(self, "key", None) is not None:
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.config.hidden_size])
        if getattr(self, "value", None) is not None:
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.config.hidden_size])
    # 初始化函数，接受一个BertConfig对象和其他关键字参数
    def __init__(self, config: BertConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 创建一个全连接层，单元数为config.hidden_size，使用给定的初始化器初始化权重，命名为"dense"
        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        
        # 创建一个LayerNormalization层，使用给定的epsilon参数，命名为"LayerNorm"
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        
        # 创建一个Dropout层，使用给定的dropout率
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        
        # 保存传入的BertConfig对象
        self.config = config

    # call方法，接受hidden_states（隐藏状态）、input_tensor（输入张量）、training（是否在训练模式下）参数，
    # 返回处理后的隐藏状态张量
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 将隐藏状态传入全连接层进行线性变换
        hidden_states = self.dense(inputs=hidden_states)
        
        # 根据训练模式应用Dropout操作
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        
        # 将Dropout后的隐藏状态与输入张量相加，并通过LayerNormalization进行归一化处理
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        # 返回处理后的隐藏状态张量
        return hidden_states

    # build方法，用于构建层的权重（如果尚未构建）
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        
        # 标记为已构建
        self.built = True
        
        # 如果dense层已经定义，则使用dense层的名称作为作用域
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                # 构建dense层的权重，输入形状为[None, None, config.hidden_size]
                self.dense.build([None, None, self.config.hidden_size])
        
        # 如果LayerNorm层已经定义，则使用LayerNorm层的名称作为作用域
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                # 构建LayerNorm层的权重，输入形状为[None, None, config.hidden_size]
                self.LayerNorm.build([None, None, self.config.hidden_size])
# 定义一个基于 Keras 的自定义层 TFBertAttention，用于 BERT 模型的自注意力机制
class TFBertAttention(keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建自注意力层对象，使用给定的 BertConfig 进行配置
        self.self_attention = TFBertSelfAttention(config, name="self")
        # 创建自注意力输出层对象，使用给定的 BertConfig 进行配置
        self.dense_output = TFBertSelfOutput(config, name="output")

    # 未实现的方法，用于裁剪注意力机制中的某些头部
    def prune_heads(self, heads):
        raise NotImplementedError

    # 定义调用方法，实现自注意力机制的前向传播
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
        # 使用 self_attention 对输入张量进行自注意力计算
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
        # 使用 dense_output 对自注意力输出进行处理
        attention_output = self.dense_output(
            hidden_states=self_outputs[0], input_tensor=input_tensor, training=training
        )
        # 如果需要输出注意力权重，将其加入到输出元组中
        outputs = (attention_output,) + self_outputs[1:]

        return outputs

    # 构建层结构，在第一次调用时构建子层的图结构
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建 self_attention 子层的图结构
        if getattr(self, "self_attention", None) is not None:
            with tf.name_scope(self.self_attention.name):
                self.self_attention.build(None)
        # 构建 dense_output 子层的图结构
        if getattr(self, "dense_output", None) is not None:
            with tf.name_scope(self.dense_output.name):
                self.dense_output.build(None)


# 定义一个基于 Keras 的自定义层 TFBertIntermediate，用于 BERT 模型的中间层处理
class TFBertIntermediate(keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建全连接层对象，设置神经元数和初始化方式
        self.dense = keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        # 根据配置获取中间激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        self.config = config

    # 定义调用方法，实现中间层的前向传播
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 使用全连接层对输入张量进行线性变换
        hidden_states = self.dense(inputs=hidden_states)
        # 使用中间激活函数对线性变换结果进行非线性变换
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

    # 构建层结构，在第一次调用时构建图结构
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建 dense 子层的图结构
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


class TFBertOutput(keras.layers.Layer):
    # 这里继续补充 TFBertOutput 类的注释
    # 初始化函数，用于创建一个新的实例
    def __init__(self, config: BertConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 创建一个全连接层，用于线性变换
        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建一个层归一化层，用于归一化输入数据
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建一个丢弃层，用于在训练时随机丢弃部分数据，防止过拟合
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        # 存储配置对象，方便在调用中使用
        self.config = config

    # 调用函数，定义了实例的前向传播逻辑
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 将输入数据通过全连接层进行线性变换
        hidden_states = self.dense(inputs=hidden_states)
        # 在训练时，随机丢弃部分数据以防止过拟合
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 对线性变换后的数据进行层归一化，并与原始输入数据相加
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        # 返回处理后的数据作为输出
        return hidden_states

    # 构建函数，用于在首次调用时构建层的内部结构
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在全连接层，根据配置参数构建其内部结构
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])
        # 如果存在层归一化层，根据配置参数构建其内部结构
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])
# 定义一个自定义层 TFBertLayer，继承自 keras 的 Layer 类
class TFBertLayer(keras.layers.Layer):
    # 初始化方法，接受一个 BertConfig 类型的 config 参数和其他关键字参数
    def __init__(self, config: BertConfig, **kwargs):
        # 调用父类 Layer 的初始化方法
        super().__init__(**kwargs)

        # 创建一个 TFBertAttention 层实例，用给定的 config 参数和名称 "attention"
        self.attention = TFBertAttention(config, name="attention")
        
        # 根据 config 中的 is_decoder 属性设置当前层是否为解码器
        self.is_decoder = config.is_decoder
        
        # 根据 config 中的 add_cross_attention 属性设置是否添加跨注意力机制
        self.add_cross_attention = config.add_cross_attention
        
        # 如果要添加跨注意力机制，且当前层不是解码器，则抛出 ValueError 异常
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            
            # 创建一个 TFBertAttention 层实例，用给定的 config 参数和名称 "crossattention"
            self.crossattention = TFBertAttention(config, name="crossattention")
        
        # 创建一个 TFBertIntermediate 层实例，用给定的 config 参数和名称 "intermediate"
        self.intermediate = TFBertIntermediate(config, name="intermediate")
        
        # 创建一个 TFBertOutput 层实例，用给定的 config 参数和名称 "output"
        self.bert_output = TFBertOutput(config, name="output")

    # 定义层的调用方法，接受多个输入参数，包括隐藏状态、注意力掩码等
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        encoder_hidden_states: tf.Tensor | None,
        encoder_attention_mask: tf.Tensor | None,
        past_key_value: Tuple[tf.Tensor] | None,
        output_attentions: bool,
        training: bool = False,
        # 函数定义未完全展示，缺少返回类型注释
    ) -> Tuple[tf.Tensor]:
        # 如果存在过去的键/值缓存，取前两个元素作为自注意力的过去键/值
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 使用自注意力层处理隐藏状态，计算自注意力输出
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
        # 获取自注意力的输出
        attention_output = self_attention_outputs[0]

        # 如果模型为解码器，最后一个输出是自注意力缓存的元组
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            # 获取当前的键/值缓存
            present_key_value = self_attention_outputs[-1]
        else:
            # 否则，输出除了第一个元素外的所有元素（即自注意力权重）
            outputs = self_attention_outputs[1:]  # 如果需要输出注意力权重，添加自注意力权重

        cross_attn_present_key_value = None
        # 如果是解码器并且有编码器的隐藏状态输入
        if self.is_decoder and encoder_hidden_states is not None:
            # 如果未定义交叉注意力层，则引发错误
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # 如果存在过去的键/值缓存，取倒数第二个和最后一个元素作为交叉注意力的过去键/值
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 使用交叉注意力层处理自注意力输出，计算交叉注意力输出
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
            # 获取交叉注意力的输出
            attention_output = cross_attention_outputs[0]
            # 将交叉注意力权重添加到输出中
            outputs = outputs + cross_attention_outputs[1:-1]

            # 将交叉注意力缓存添加到当前的键/值缓存中的倒数第二个和最后一个位置
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 使用中间层处理注意力输出，得到中间层输出
        intermediate_output = self.intermediate(hidden_states=attention_output)
        # 使用BERT输出层处理中间层输出和注意力输出，得到层输出
        layer_output = self.bert_output(
            hidden_states=intermediate_output, input_tensor=attention_output, training=training
        )
        # 将层输出与注意力权重（如果存在）合并到输出中
        outputs = (layer_output,) + outputs

        # 如果是解码器，将注意力的键/值作为最后一个输出返回
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        # 返回最终的输出
        return outputs
    # 构建模型的方法，用于在给定输入形状的情况下构建模型的各个部分
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回，避免重复构建
        if self.built:
            return
        # 将模型标记为已构建状态
        self.built = True
        
        # 如果模型具有 attention 属性，则构建 attention 部分
        if getattr(self, "attention", None) is not None:
            # 使用 attention 的名称作为命名空间，构建 attention 层
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        
        # 如果模型具有 intermediate 属性，则构建 intermediate 部分
        if getattr(self, "intermediate", None) is not None:
            # 使用 intermediate 的名称作为命名空间，构建 intermediate 层
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        
        # 如果模型具有 bert_output 属性，则构建 bert_output 部分
        if getattr(self, "bert_output", None) is not None:
            # 使用 bert_output 的名称作为命名空间，构建 bert_output 层
            with tf.name_scope(self.bert_output.name):
                self.bert_output.build(None)
        
        # 如果模型具有 crossattention 属性，则构建 crossattention 部分
        if getattr(self, "crossattention", None) is not None:
            # 使用 crossattention 的名称作为命名空间，构建 crossattention 层
            with tf.name_scope(self.crossattention.name):
                self.crossattention.build(None)
# 定义一个基于Keras层的TFBertEncoder类，用于BERT模型的编码器部分
class TFBertEncoder(keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)
        # 初始化时保存BERT配置信息
        self.config = config
        # 创建多个TFBertLayer实例作为编码器的层，并命名每一层
        self.layer = [TFBertLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]

    # 定义调用方法，实现编码器的前向传播
    def call(
        self,
        hidden_states: tf.Tensor,                 # 输入的隐藏状态张量
        attention_mask: tf.Tensor,                # 自注意力机制的掩码张量
        head_mask: tf.Tensor,                     # 头部掩码张量，用于控制多头注意力中的哪些头参与计算
        encoder_hidden_states: tf.Tensor | None,  # 编码器的隐藏状态张量，如果存在的话
        encoder_attention_mask: tf.Tensor | None, # 编码器的注意力掩码张量，如果存在的话
        past_key_values: Tuple[Tuple[tf.Tensor]] | None,  # 过去的键值对，用于缓存
        use_cache: Optional[bool],                # 是否使用缓存
        output_attentions: bool,                  # 是否输出注意力权重
        output_hidden_states: bool,               # 是否输出所有隐藏状态
        return_dict: bool,                        # 是否返回字典形式的结果
        training: bool = False,                   # 是否处于训练模式
    ) -> Union[TFBaseModelOutputWithPastAndCrossAttentions, Tuple[tf.Tensor]]:
        # 初始化存储所有隐藏状态、注意力权重和交叉注意力权重的空元组
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 初始化下一个解码器缓存的空元组，如果使用缓存的话
        next_decoder_cache = () if use_cache else None

        # 遍历每一层编码器
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前层的隐藏状态添加到all_hidden_states中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的过去键值对，用于当前层的注意力机制
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 调用当前层的前向传播，得到当前层的输出
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
            # 更新当前隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]

            # 如果使用缓存，则将当前层的输出的最后一个元素添加到下一个解码器缓存中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            # 如果需要输出注意力权重，则将当前层的注意力权重添加到all_attentions中
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
                # 如果配置中添加了交叉注意力并且编码器隐藏状态不为空，则将当前层的交叉注意力添加到all_cross_attentions中
                if self.config.add_cross_attention and encoder_hidden_states is not None:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 添加最后一层编码器的隐藏状态，如果需要输出隐藏状态的话
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要返回字典形式的结果，则返回非None的所有元组元素
        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_attentions, all_cross_attentions] if v is not None
            )

        # 返回TFBaseModelOutputWithPastAndCrossAttentions类型的结果字典
        return TFBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )
    # 定义一个构建模型的方法，该方法可以接受输入形状作为参数
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回，不进行重复构建
        if self.built:
            return
        # 将标志位设置为已构建
        self.built = True
        # 检查是否存在self.layer属性，即模型是否包含层
        if getattr(self, "layer", None) is not None:
            # 遍历模型中的每一层
            for layer in self.layer:
                # 使用层的名称作为命名空间
                with tf.name_scope(layer.name):
                    # 调用每一层的build方法，传入None作为输入形状
                    layer.build(None)
class TFBertPooler(keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)

        # 定义一个全连接层，用于池化隐藏状态
        self.dense = keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 池化模型的输出，简单地选择第一个 token 对应的隐藏状态
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(inputs=first_token_tensor)

        return pooled_output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                # 构建全连接层
                self.dense.build([None, None, self.config.hidden_size])


class TFBertPredictionHeadTransform(keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)

        # 定义一个全连接层，用于转换隐藏状态
        self.dense = keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense",
        )

        # 根据配置选择激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.transform_act_fn = config.hidden_act

        # 使用 LayerNormalization 层对隐藏状态进行规范化
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 应用全连接层
        hidden_states = self.dense(inputs=hidden_states)
        # 应用激活函数
        hidden_states = self.transform_act_fn(hidden_states)
        # 应用 LayerNormalization
        hidden_states = self.LayerNorm(inputs=hidden_states)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                # 构建全连接层
                self.dense.build([None, None, self.config.hidden_size])
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                # 构建 LayerNormalization 层
                self.LayerNorm.build([None, None, self.config.hidden_size])


class TFBertLMPredictionHead(keras.layers.Layer):
    def __init__(self, config: BertConfig, input_embeddings: keras.layers.Layer, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.hidden_size = config.hidden_size

        # 创建一个预测头的转换层
        self.transform = TFBertPredictionHeadTransform(config, name="transform")

        # 输出权重与输入嵌入层相同，但每个 token 有一个仅输出的偏置
        self.input_embeddings = input_embeddings
    # 定义一个方法用于构建模型层，接受输入形状参数，默认为None
    def build(self, input_shape=None):
        # 初始化偏置项为零向量，形状与词汇表大小相同，可训练
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="bias")

        # 如果模型已构建，则直接返回，避免重复构建
        if self.built:
            return
        self.built = True  # 标记模型已构建

        # 如果有transform属性，使用其名字空间构建transform层
        if getattr(self, "transform", None) is not None:
            with tf.name_scope(self.transform.name):
                self.transform.build(None)

    # 返回输入嵌入层
    def get_output_embeddings(self) -> keras.layers.Layer:
        return self.input_embeddings

    # 设置输出嵌入层，更新权重和词汇表大小
    def set_output_embeddings(self, value: tf.Variable):
        self.input_embeddings.weight = value
        self.input_embeddings.vocab_size = shape_list(value)[0]

    # 返回偏置项作为字典
    def get_bias(self) -> Dict[str, tf.Variable]:
        return {"bias": self.bias}

    # 设置偏置项，更新偏置和词汇表大小
    def set_bias(self, value: tf.Variable):
        self.bias = value["bias"]
        self.config.vocab_size = shape_list(value["bias"])[0]

    # 模型调用函数，接受隐藏状态张量作为输入，返回处理后的张量
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 使用transform层处理隐藏状态
        hidden_states = self.transform(hidden_states=hidden_states)
        seq_length = shape_list(hidden_states)[1]  # 获取序列长度
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.hidden_size])  # 重塑张量形状
        hidden_states = tf.matmul(a=hidden_states, b=self.input_embeddings.weight, transpose_b=True)  # 执行矩阵乘法
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])  # 再次重塑张量形状
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)  # 添加偏置项到张量

        return hidden_states
class TFBertMLMHead(keras.layers.Layer):
    def __init__(self, config: BertConfig, input_embeddings: keras.layers.Layer, **kwargs):
        super().__init__(**kwargs)

        # 使用给定的配置和输入嵌入层创建预测头部对象
        self.predictions = TFBertLMPredictionHead(config, input_embeddings, name="predictions")

    def call(self, sequence_output: tf.Tensor) -> tf.Tensor:
        # 调用预测头部对象来计算序列输出的预测分数
        prediction_scores = self.predictions(hidden_states=sequence_output)

        return prediction_scores

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "predictions", None) is not None:
            with tf.name_scope(self.predictions.name):
                # 构建预测头部对象
                self.predictions.build(None)


class TFBertNSPHead(keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个密集层来处理序列关系分数的预测
        self.seq_relationship = keras.layers.Dense(
            units=2,
            kernel_initializer=get_initializer(config.initializer_range),
            name="seq_relationship",
        )
        self.config = config

    def call(self, pooled_output: tf.Tensor) -> tf.Tensor:
        # 使用密集层计算池化输出的序列关系分数
        seq_relationship_score = self.seq_relationship(inputs=pooled_output)

        return seq_relationship_score

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "seq_relationship", None) is not None:
            with tf.name_scope(self.seq_relationship.name):
                # 构建密集层，指定输入形状为 [None, None, 隐藏大小]
                self.seq_relationship.build([None, None, self.config.hidden_size])


@keras_serializable
class TFBertMainLayer(keras.layers.Layer):
    config_class = BertConfig

    def __init__(self, config: BertConfig, add_pooling_layer: bool = True, **kwargs):
        super().__init__(**kwargs)

        # 初始化BERT主层对象，配置及是否添加池化层
        self.config = config
        self.is_decoder = config.is_decoder

        # 创建BERT的嵌入层、编码器层和池化层（如果需要的话）
        self.embeddings = TFBertEmbeddings(config, name="embeddings")
        self.encoder = TFBertEncoder(config, name="encoder")
        self.pooler = TFBertPooler(config, name="pooler") if add_pooling_layer else None

    def get_input_embeddings(self) -> keras.layers.Layer:
        # 返回嵌入层对象
        return self.embeddings

    def set_input_embeddings(self, value: tf.Variable):
        # 设置嵌入层的权重和词汇大小
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    @unpack_inputs
    # 定义一个类方法，用于调用模型。接受多个输入参数，都有默认值为None或False。
    def call(
        self,
        input_ids: TFModelInputType | None = None,  # 输入的token IDs，类型为TFModelInputType或None
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 注意力遮罩，类型为numpy数组、Tensor或None
        token_type_ids: np.ndarray | tf.Tensor | None = None,  # token类型IDs，类型为numpy数组、Tensor或None
        position_ids: np.ndarray | tf.Tensor | None = None,  # 位置IDs，类型为numpy数组、Tensor或None
        head_mask: np.ndarray | tf.Tensor | None = None,  # 头部遮罩，类型为numpy数组、Tensor或None
        inputs_embeds: np.ndarray | tf.Tensor | None = None,  # 输入的嵌入，类型为numpy数组、Tensor或None
        encoder_hidden_states: np.ndarray | tf.Tensor | None = None,  # 编码器隐藏状态，类型为numpy数组、Tensor或None
        encoder_attention_mask: np.ndarray | tf.Tensor | None = None,  # 编码器注意力遮罩，类型为numpy数组、Tensor或None
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,  # 过去的键-值对，可选的类型为嵌套元组
        use_cache: Optional[bool] = None,  # 是否使用缓存，可选的布尔值
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可选的布尔值
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选的布尔值
        return_dict: Optional[bool] = None,  # 是否返回字典形式结果，可选的布尔值
        training: bool = False,  # 是否处于训练模式，默认为False


    # 构建模型的方法，用于建立模型的各个组件
    def build(self, input_shape=None):
        # 如果模型已经构建完毕，则直接返回
        if self.built:
            return
        # 标记模型为已构建状态
        self.built = True
        # 如果模型具有嵌入层（embeddings），则构建嵌入层
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):  # 使用嵌入层名称作为命名空间
                self.embeddings.build(None)  # 构建嵌入层，输入形状为None
        # 如果模型具有编码器（encoder），则构建编码器
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):  # 使用编码器名称作为命名空间
                self.encoder.build(None)  # 构建编码器，输入形状为None
        # 如果模型具有池化器（pooler），则构建池化器
        if getattr(self, "pooler", None) is not None:
            with tf.name_scope(self.pooler.name):  # 使用池化器名称作为命名空间
                self.pooler.build(None)  # 构建池化器，输入形状为None
class TFBertPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置配置类为BertConfig，用于模型配置
    config_class = BertConfig
    # 指定基础模型的前缀为"bert"
    base_model_prefix = "bert"


@dataclass
class TFBertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`TFBertForPreTraining`].

    Args:
        prediction_logits (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (`tf.Tensor` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # 定义输出类，包括预训练过程中的损失、预测logits、序列关系logits、隐藏状态和注意力
    loss: tf.Tensor | None = None
    prediction_logits: tf.Tensor = None
    seq_relationship_logits: tf.Tensor = None
    hidden_states: Optional[Union[Tuple[tf.Tensor], tf.Tensor]] = None
    attentions: Optional[Union[Tuple[tf.Tensor], tf.Tensor]] = None


BERT_START_DOCSTRING = r"""

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


```    
    Args:
        config ([`BertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""

BERT_INPUTS_DOCSTRING = r"""
"""


@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    BERT_START_DOCSTRING,
)
class TFBertModel(TFBertPreTrainedModel):
    def __init__(self, config: BertConfig, add_pooling_layer: bool = True, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 初始化 Bert 主模型层，并设置是否添加池化层
        self.bert = TFBertMainLayer(config, add_pooling_layer, name="bert")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        **kwargs
    ) -> Union[TFBaseModelOutputWithPoolingAndCrossAttentions, Tuple[tf.Tensor]]:
        """
        Perform the forward pass of the TFBertModel.

        This method overrides the call function in TFBertPreTrainedModel
        to allow for flexible input handling and model output specification.
        """
        # 以下代码为注释部分，解释了每个参数的作用和期望的输入输出类型
        # 参数解释和类型注释由 add_start_docstrings_to_model_forward 和 add_code_sample_docstrings 提供
        pass
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
        outputs = self.bert(
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
        return outputs
@add_start_docstrings(
    """
    Bert Model with two heads on top as done during the pretraining:
    a `masked language modeling` head and a `next sentence prediction (classification)` head.
    """,
    BERT_START_DOCSTRING,
)
class TFBertForPreTraining(TFBertPreTrainedModel, TFBertPreTrainingLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [
        r"position_ids",
        r"cls.predictions.decoder.weight",
        r"cls.predictions.decoder.bias",
    ]

    def __init__(self, config: BertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # Initialize the BERT main layer with the provided configuration
        self.bert = TFBertMainLayer(config, name="bert")
        
        # Initialize the Next Sentence Prediction (NSP) head with the provided configuration
        self.nsp = TFBertNSPHead(config, name="nsp___cls")
        
        # Initialize the Masked Language Modeling (MLM) head with the provided configuration,
        # using embeddings from the BERT main layer
        self.mlm = TFBertMLMHead(config, input_embeddings=self.bert.embeddings, name="mlm___cls")

    def get_lm_head(self) -> keras.layers.Layer:
        # Return the predictions layer from the MLM head
        return self.mlm.predictions

    def get_prefix_bias_name(self) -> str:
        # Deprecated method warning for obtaining the bias name
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        return self.name + "/" + self.mlm.name + "/" + self.mlm.predictions.name

    @unpack_inputs
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFBertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
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
        next_sentence_label: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    ):
        # Method defining the forward pass of the model
        ...

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "bert", None) is not None:
            # Build the BERT main layer within its name scope
            with tf.name_scope(self.bert.name):
                self.bert.build(None)
        if getattr(self, "nsp", None) is not None:
            # Build the NSP head within its name scope
            with tf.name_scope(self.nsp.name):
                self.nsp.build(None)
        if getattr(self, "mlm", None) is not None:
            # Build the MLM head within its name scope
            with tf.name_scope(self.mlm.name):
                self.mlm.build(None)


@add_start_docstrings("""Bert Model with a `language modeling` head on top.""", BERT_START_DOCSTRING)
class TFBertForMaskedLM(TFBertPreTrainedModel, TFMaskedLanguageModelingLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    ...
    # 在加载时忽略的键列表，这些键是在加载模型时不期望出现的
    _keys_to_ignore_on_load_unexpected = [
        r"pooler",  # 忽略名为"pooler"的键
        r"cls.seq_relationship",  # 忽略名为"cls.seq_relationship"的键
        r"cls.predictions.decoder.weight",  # 忽略名为"cls.predictions.decoder.weight"的键
        r"nsp___cls",  # 忽略名为"nsp___cls"的键
    ]

    # 初始化方法，接受一个BertConfig对象作为参数，以及其他可能的输入和关键字参数
    def __init__(self, config: BertConfig, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 如果配置指定为decoder，则发出警告
        if config.is_decoder:
            logger.warning(
                "If you want to use `TFBertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # 创建TFBertMainLayer实例，用给定的配置和名称"bert"，不添加池化层
        self.bert = TFBertMainLayer(config, add_pooling_layer=False, name="bert")
        # 创建TFBertMLMHead实例，用给定的配置和输入嵌入self.bert.embeddings，名称为"mlm___cls"
        self.mlm = TFBertMLMHead(config, input_embeddings=self.bert.embeddings, name="mlm___cls")

    # 返回语言模型头部（MLM头部）的Keras层对象
    def get_lm_head(self) -> keras.layers.Layer:
        return self.mlm.predictions

    # 获取前缀偏置名称，已弃用，发出警告
    def get_prefix_bias_name(self) -> str:
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        # 返回包含self.name、self.mlm.name和self.mlm.predictions.name的字符串
        return self.name + "/" + self.mlm.name + "/" + self.mlm.predictions.name

    # 调用方法，接收多个输入参数和关键字参数，包括输入ID、注意力掩码等
    @unpack_inputs
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFMaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'paris'",
        expected_loss=0.88,
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
    ) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        # 使用 TensorFlow 注解语法指定函数的返回类型，可以是 TFMaskedLMOutput 或包含 tf.Tensor 的元组
        outputs = self.bert(
            input_ids=input_ids,  # 输入的 token IDs
            attention_mask=attention_mask,  # 注意力遮罩，指示哪些 token 是真实的（1）和哪些是填充的（0）
            token_type_ids=token_type_ids,  # 用于区分两个句子的 token 类型 IDs
            position_ids=position_ids,  # 位置编码 IDs，用于指定 token 在序列中的位置
            head_mask=head_mask,  # 多头注意力机制中屏蔽的头部掩码
            inputs_embeds=inputs_embeds,  # 可选的输入嵌入，用于代替输入的 token IDs
            output_attentions=output_attentions,  # 是否返回注意力权重
            output_hidden_states=output_hidden_states,  # 是否返回隐藏状态
            return_dict=return_dict,  # 是否以字典形式返回输出
            training=training,  # 是否处于训练模式
        )
        # 获取 BERT 输出的序列输出
        sequence_output = outputs[0]
        # 将序列输出传递给 MLM 模型进行预测
        prediction_scores = self.mlm(sequence_output=sequence_output, training=training)
        # 如果提供了标签，则计算 MLM 损失
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=prediction_scores)

        # 如果不返回字典，则按照顺序构造输出元组
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]  # 包含预测分数和额外的输出状态
            return ((loss,) + output) if loss is not None else output

        # 返回 TFMaskedLMOutput 类型的对象，包括损失、预测分数、隐藏状态和注意力权重
        return TFMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经构建过，则直接返回
        if getattr(self, "bert", None) is not None:
            with tf.name_scope(self.bert.name):  # 在 TensorFlow 中使用指定的命名空间构建 BERT 模型
                self.bert.build(None)  # 构建 BERT 模型
        if getattr(self, "mlm", None) is not None:
            with tf.name_scope(self.mlm.name):  # 在 TensorFlow 中使用指定的命名空间构建 MLM 模型
                self.mlm.build(None)  # 构建 MLM 模型
# 继承自 TFBertPreTrainedModel 和 TFCausalLanguageModelingLoss，实现了 BERT 语言模型的头部部分
class TFBertLMHeadModel(TFBertPreTrainedModel, TFCausalLanguageModelingLoss):
    # 在从 PyTorch 模型加载到 TensorFlow 模型时，指定的可以忽略的层名称列表
    _keys_to_ignore_on_load_unexpected = [
        r"pooler",  # 忽略名为 "pooler" 的层
        r"cls.seq_relationship",  # 忽略名为 "cls.seq_relationship" 的层
        r"cls.predictions.decoder.weight",  # 忽略名为 "cls.predictions.decoder.weight" 的层
        r"nsp___cls",  # 忽略名为 "nsp___cls" 的层
    ]

    def __init__(self, config: BertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 如果配置不是解码器，则发出警告
        if not config.is_decoder:
            logger.warning("If you want to use `TFBertLMHeadModel` as a standalone, add `is_decoder=True.`")

        # 创建 BERT 主层，不添加池化层，命名为 "bert"
        self.bert = TFBertMainLayer(config, add_pooling_layer=False, name="bert")
        # 创建 BERT MLM 头部，使用 BERT embeddings 作为输入嵌入，命名为 "mlm___cls"
        self.mlm = TFBertMLMHead(config, input_embeddings=self.bert.embeddings, name="mlm___cls")

    # 返回 MLM 头部的预测层
    def get_lm_head(self) -> keras.layers.Layer:
        return self.mlm.predictions

    # 返回前缀偏置名称的字符串表示（已弃用）
    def get_prefix_bias_name(self) -> str:
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        return self.name + "/" + self.mlm.name + "/" + self.mlm.predictions.name

    # 准备生成时的输入，处理输入的形状和注意力掩码
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        # 如果没有提供注意力掩码，则创建全 1 的掩码
        if attention_mask is None:
            attention_mask = tf.ones(input_shape)

        # 如果有过去的键值对被使用，则截取输入的最后一个 token
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # 返回包含输入 ids、注意力掩码和过去键值对的字典
        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}

    # 调用模型时的装饰器，展开输入参数并添加代码示例的文档字符串
    @unpack_inputs
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFCausalLMOutputWithCrossAttentions,
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
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
        **kwargs,
    # 如果模型已经构建完成，则直接返回，不进行重复构建
    if self.built:
        return
    # 将标志位设置为已构建状态
    self.built = True
    
    # 如果存在属性self.bert且不为None，则构建self.bert模型
    if getattr(self, "bert", None) is not None:
        # 使用self.bert的名字作为命名空间，构建self.bert模型
        with tf.name_scope(self.bert.name):
            self.bert.build(None)
    
    # 如果存在属性self.mlm且不为None，则构建self.mlm模型
    if getattr(self, "mlm", None) is not None:
        # 使用self.mlm的名字作为命名空间，构建self.mlm模型
        with tf.name_scope(self.mlm.name):
            self.mlm.build(None)
# 使用装饰器添加模型文档字符串，描述带有顶部“下一个句子预测（分类）”头的Bert模型
@add_start_docstrings(
    """Bert Model with a `next sentence prediction (classification)` head on top.""",
    BERT_START_DOCSTRING,
)
class TFBertForNextSentencePrediction(TFBertPreTrainedModel, TFNextSentencePredictionLoss):
    # 在从PT模型加载TF模型时，指定要忽略的未预期/丢失的层名称列表
    _keys_to_ignore_on_load_unexpected = [r"mlm___cls", r"cls.predictions"]

    def __init__(self, config: BertConfig, *inputs, **kwargs):
        # 调用父类构造函数初始化模型配置
        super().__init__(config, *inputs, **kwargs)

        # 初始化BERT主层，并命名为“bert”
        self.bert = TFBertMainLayer(config, name="bert")
        # 初始化下一个句子预测头部，并命名为“nsp___cls”
        self.nsp = TFBertNSPHead(config, name="nsp___cls")

    # 使用装饰器解包输入和添加模型前向传播的文档字符串
    @unpack_inputs
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 替换返回值文档字符串，指定输出类型和配置类别
    @replace_return_docstrings(output_type=TFNextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC)
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
        next_sentence_label: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
        **kwargs
    ) -> Union[TFNextSentencePredictorOutput, Tuple[tf.Tensor]]:
        # 实现模型的前向传播逻辑，接受多个输入参数和训练标志
        pass  # 实际实现在此处被省略
    # 定义一个函数，用于进行下一句预测。函数返回类型为TFNextSentencePredictorOutput或Tuple[tf.Tensor]
    def __call__(
            self,
            input_ids: tf.Tensor,
            attention_mask: Optional[tf.Tensor] = None,
            token_type_ids: Optional[tf.Tensor] = None,
            position_ids: Optional[tf.Tensor] = None,
            head_mask: Optional[tf.Tensor] = None,
            inputs_embeds: Optional[tf.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = True,
            training: Optional[bool] = False,
            next_sentence_label: Optional[tf.Tensor] = None,
        ) -> Union[TFNextSentencePredictorOutput, Tuple[tf.Tensor]]:
            r"""
            返回函数的说明
            示例代码
            """
    
            # 使用BERT模型进行预测，输出为outputs
            outputs = self.bert(
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
            
            # 得到池化后的输出
            pooled_output = outputs[1]
            # 使用NSP模型对池化后的输出进行预测，输出为seq_relationship_scores
            seq_relationship_scores = self.nsp(pooled_output=pooled_output)
            # 如果有下一个句子的标签，计算下一个句子的损失
            next_sentence_loss = (
                None
                if next_sentence_label is None
                else self.hf_compute_loss(labels=next_sentence_label, logits=seq_relationship_scores)
            )
            
            # 如果不返回字典形式的结果，返回序列的得分和其他输出
            if not return_dict:
                output = (seq_relationship_scores,) + outputs[2:]
                return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output
            
            # 如果返回字典形式的结果，返回字典类型的输出
            return TFNextSentencePredictorOutput(
                loss=next_sentence_loss,
                logits=seq_relationship_scores,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
    
        # 构建模型
        def build(self, input_shape=None):
            if self.built:
                return
            self.built = True
            # 如果已经建立BERT模型，继续构建BERT模型
            if getattr(self, "bert", None) is not None:
                with tf.name_scope(self.bert.name):
                    self.bert.build(None)
            # 如果已经建立NSP模型，继续构建NSP模型
            if getattr(self, "nsp", None) is not None:
                with tf.name_scope(self.nsp.name):
                    self.nsp.build(None)
# 定义一个带有顶部序列分类/回归头的 BERT 模型转换器（在汇总输出之上有一个线性层），例如用于 GLUE 任务
@add_start_docstrings(
    """
    Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    BERT_START_DOCSTRING,
)
class TFBertForSequenceClassification(TFBertPreTrainedModel, TFSequenceClassificationLoss):
    # 当从 PT 模型加载 TF 模型时，带 '.' 的名称表示授权的意外/丢失的层
    _keys_to_ignore_on_load_unexpected = [r"mlm___cls", r"nsp___cls", r"cls.predictions", r"cls.seq_relationship"]
    # 当从 PT 模型加载 TF 模型时，缺失的层的名称
    _keys_to_ignore_on_load_missing = [r"dropout"]

    def __init__(self, config: BertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 设置模型标签数
        self.num_labels = config.num_labels

        # 使用 TF 的 Bert 主层初始化 BERT 模型
        self.bert = TFBertMainLayer(config, name="bert")

        # 设置分类器的丢弃率
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 添加丢弃层
        self.dropout = keras.layers.Dropout(rate=classifier_dropout)
        # 定义分类器层
        self.classifier = keras.layers.Dense(
            units=config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="classifier",
        )
        # 保存配置
        self.config = config

    # 调用模型的前向传播方法，用于处理输入并返回相应的输出和损失
    @unpack_inputs
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
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
        **kwargs,
    ):
        """
        BERT 模型的前向传播方法，处理输入并返回相应的输出和损失。

        Args:
            input_ids (TFModelInputType | None, optional): 输入的 token IDs. Defaults to None.
            attention_mask (np.ndarray | tf.Tensor | None, optional): 注意力遮罩. Defaults to None.
            token_type_ids (np.ndarray | tf.Tensor | None, optional): token 类型 IDs. Defaults to None.
            position_ids (np.ndarray | tf.Tensor | None, optional): 位置 IDs. Defaults to None.
            head_mask (np.ndarray | tf.Tensor | None, optional): 头部遮罩. Defaults to None.
            inputs_embeds (np.ndarray | tf.Tensor | None, optional): 输入的嵌入. Defaults to None.
            output_attentions (Optional[bool], optional): 是否输出注意力. Defaults to None.
            output_hidden_states (Optional[bool], optional): 是否输出隐藏状态. Defaults to None.
            return_dict (Optional[bool], optional): 是否返回字典. Defaults to None.
            labels (np.ndarray | tf.Tensor | None, optional): 标签. Defaults to None.
            training (Optional[bool], optional): 是否训练模式. Defaults to False.
            **kwargs: 其他关键字参数.

        Returns:
            TFSequenceClassifierOutput: 序列分类器输出对象.
        """
        # 处理输入以获取模型的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 BERT 模型的前向传播
        outputs = self.bert(
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
            **kwargs,
        )

        # 对 BERT 输出进行丢弃操作
        pooled_output = outputs[1]  # 汇总输出
        pooled_output = self.dropout(pooled_output, training=training)
        # 经过分类器层得到最终输出
        logits = self.classifier(pooled_output)

        # 准备模型的输出
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # 单标签分类任务
                loss_fn = tf.keras.losses.MeanSquaredError()
                loss = loss_fn(labels, logits)
            else:
                # 多标签分类任务
                loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                loss = loss_fn(labels, logits)

        if not return_dict:
            # 返回不同的输出对象
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回序列分类器输出对象
        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    ) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 调用BERT模型进行前向传播，获取输出
        outputs = self.bert(
            input_ids=input_ids,  # 输入的token IDs
            attention_mask=attention_mask,  # 注意力掩码，用于指示哪些token需要注意，哪些不需要
            token_type_ids=token_type_ids,  # token类型IDs，用于区分segment A和segment B
            position_ids=position_ids,  # token的位置IDs，指示每个token在序列中的位置
            head_mask=head_mask,  # 头部掩码，用于控制哪些attention头是有效的
            inputs_embeds=inputs_embeds,  # 输入的嵌入表示，代替输入的token IDs
            output_attentions=output_attentions,  # 是否输出attention权重
            output_hidden_states=output_hidden_states,  # 是否输出所有层的隐藏状态
            return_dict=return_dict,  # 返回类型，是否以字典形式返回输出
            training=training,  # 是否处于训练模式
        )
        pooled_output = outputs[1]  # 获取汇聚输出，通常是CLS token的表示
        pooled_output = self.dropout(inputs=pooled_output, training=training)  # 对汇聚输出进行dropout处理
        logits = self.classifier(inputs=pooled_output)  # 使用分类器对汇聚输出进行分类预测
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)  # 如果有标签，则计算损失

        if not return_dict:
            output = (logits,) + outputs[2:]  # 构造输出元组，包括logits和可能的额外输出
            return ((loss,) + output) if loss is not None else output  # 如果有损失，返回损失和输出，否则只返回输出

        # 如果return_dict为True，以TFSequenceClassifierOutput形式返回输出
        return TFSequenceClassifierOutput(
            loss=loss,  # 损失
            logits=logits,  # 预测的logits
            hidden_states=outputs.hidden_states,  # 所有层的隐藏状态
            attentions=outputs.attentions,  # 所有层的注意力权重
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "bert", None) is not None:
            with tf.name_scope(self.bert.name):
                self.bert.build(None)  # 构建BERT模型
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])  # 构建分类器模型
"""
Bert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.
"""
# 继承自TFBertPreTrainedModel和TFMultipleChoiceLoss，实现Bert模型添加多选分类头部
class TFBertForMultipleChoice(TFBertPreTrainedModel, TFMultipleChoiceLoss):

    # 当从PyTorch模型加载TF模型时，忽略的预期未知/丢失的层名称列表
    _keys_to_ignore_on_load_unexpected = [r"mlm___cls", r"nsp___cls", r"cls.predictions", r"cls.seq_relationship"]
    # 当从PyTorch模型加载TF模型时，忽略的缺失层名称列表
    _keys_to_ignore_on_load_missing = [r"dropout"]

    def __init__(self, config: BertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 初始化Bert主层，并命名为"bert"
        self.bert = TFBertMainLayer(config, name="bert")
        # 初始化Dropout层，使用配置中的隐藏层Dropout概率
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        # 初始化分类器Dense层，单元数为1，使用给定的初始化器范围初始化权重，并命名为"classifier"
        self.classifier = keras.layers.Dense(
            units=1, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFMultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义模型前向传播方法，接受一系列输入参数
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
        # 参数类型和默认值的注释
    ) -> Union[TFMultipleChoiceModelOutput, Tuple[tf.Tensor]]:
        r"""
        定义函数的返回类型，可以是 TFMultipleChoiceModelOutput 或包含 tf.Tensor 的元组
        """
        if input_ids is not None:
            # 获取 input_ids 的形状列表，并取第二个维度的大小作为 num_choices
            num_choices = shape_list(input_ids)[1]
            # 获取 input_ids 的第三个维度的大小作为 seq_length
            seq_length = shape_list(input_ids)[2]
        else:
            # 如果 input_ids 为 None，则使用 inputs_embeds 的形状列表中的第二个维度作为 num_choices
            num_choices = shape_list(inputs_embeds)[1]
            # 使用 inputs_embeds 的第三个维度的大小作为 seq_length
            seq_length = shape_list(inputs_embeds)[2]

        # 如果 input_ids 不为 None，则将其重塑为 (-1, seq_length) 的形状，否则为 None
        flat_input_ids = tf.reshape(tensor=input_ids, shape=(-1, seq_length)) if input_ids is not None else None
        # 如果 attention_mask 不为 None，则将其重塑为 (-1, seq_length) 的形状，否则为 None
        flat_attention_mask = tf.reshape(tensor=attention_mask, shape=(-1, seq_length)) if attention_mask is not None else None
        # 如果 token_type_ids 不为 None，则将其重塑为 (-1, seq_length) 的形状，否则为 None
        flat_token_type_ids = tf.reshape(tensor=token_type_ids, shape=(-1, seq_length)) if token_type_ids is not None else None
        # 如果 position_ids 不为 None，则将其重塑为 (-1, seq_length) 的形状，否则为 None
        flat_position_ids = tf.reshape(tensor=position_ids, shape=(-1, seq_length)) if position_ids is not None else None
        # 如果 inputs_embeds 不为 None，则将其重塑为 (-1, seq_length, inputs_embeds 的第四个维度大小) 的形状，否则为 None
        flat_inputs_embeds = tf.reshape(tensor=inputs_embeds, shape=(-1, seq_length, shape_list(inputs_embeds)[3])) if inputs_embeds is not None else None
        # 调用 BERT 模型，传递平铺后的输入及其他参数，并获取输出结果
        outputs = self.bert(
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
        # 从 BERT 输出中获取池化后的输出
        pooled_output = outputs[1]
        # 使用 dropout 方法对池化输出进行处理，根据 training 参数决定是否训练
        pooled_output = self.dropout(inputs=pooled_output, training=training)
        # 使用分类器对处理后的 pooled_output 进行分类预测
        logits = self.classifier(inputs=pooled_output)
        # 将 logits 重塑为 (-1, num_choices) 的形状
        reshaped_logits = tf.reshape(tensor=logits, shape=(-1, num_choices))
        # 如果 labels 不为 None，则计算损失，否则损失为 None
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=reshaped_logits)

        # 如果 return_dict 为 False，则返回格式化后的输出
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则返回 TFMultipleChoiceModelOutput 对象，包含损失、logits、隐藏状态和注意力分布
        return TFMultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    # 定义神经网络模型的构建方法，如果已经构建过，则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 标记模型为已构建状态
        self.built = True
        # 如果存在名为"bert"的属性，则构建BERT模型
        if getattr(self, "bert", None) is not None:
            # 在命名空间中构建BERT模型
            with tf.name_scope(self.bert.name):
                self.bert.build(None)
        # 如果存在名为"classifier"的属性，则构建分类器模型
        if getattr(self, "classifier", None) is not None:
            # 在命名空间中构建分类器模型，期望输入形状为[None, None, self.config.hidden_size]
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
# 使用装饰器为类添加文档字符串，描述其作为一个在 Bert 模型上加入了一个标记分类头部的模型，用于命名实体识别 (NER) 等任务
class TFBertForTokenClassification(TFBertPreTrainedModel, TFTokenClassificationLoss):
    # 当从 PyTorch 模型加载到 TF 模型时，忽略的意外/缺失的层的名称列表，包含不匹配的层名
    _keys_to_ignore_on_load_unexpected = [
        r"pooler",
        r"mlm___cls",
        r"nsp___cls",
        r"cls.predictions",
        r"cls.seq_relationship",
    ]
    # 当从 PyTorch 模型加载到 TF 模型时，忽略的缺失的层的名称列表，包含缺少的层名
    _keys_to_ignore_on_load_missing = [r"dropout"]

    def __init__(self, config: BertConfig, *inputs, **kwargs):
        # 调用父类构造函数，传递配置和其他输入参数
        super().__init__(config, *inputs, **kwargs)

        # 记录标签的数量
        self.num_labels = config.num_labels

        # 初始化 BERT 主层，禁用添加池化层，命名为 "bert"
        self.bert = TFBertMainLayer(config, add_pooling_layer=False, name="bert")

        # 设置分类器的丢弃率为配置中的分类器丢弃率或者隐藏层丢弃率，如果配置中未指定分类器丢弃率，则使用隐藏层丢弃率
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 添加一个丢弃层
        self.dropout = keras.layers.Dropout(rate=classifier_dropout)
        # 添加一个全连接层作为分类器，单元数为配置中的标签数量，初始化器使用配置中的初始化范围
        self.classifier = keras.layers.Dense(
            units=config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="classifier",
        )
        # 记录配置对象
        self.config = config

    # 使用装饰器为模型的前向传播方法添加文档字符串，描述其输入和输出，以及模型的用法示例和预期输出
    @unpack_inputs
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_TOKEN_CLASSIFICATION,
        output_type=TFTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_TOKEN_CLASS_EXPECTED_OUTPUT,
        expected_loss=_TOKEN_CLASS_EXPECTED_LOSS,
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
        定义函数签名和返回类型注解，此函数可以返回 TFTokenClassifierOutput 或包含 tf.Tensor 的元组。
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            用于计算标记分类损失的标签。索引应在 `[0, ..., config.num_labels - 1]` 范围内。
        """
        # 使用 BERT 模型处理输入数据，并获取输出结果
        outputs = self.bert(
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
        # 从 BERT 输出中获取序列输出
        sequence_output = outputs[0]
        # 根据训练状态应用 dropout 操作，用于防止过拟合
        sequence_output = self.dropout(inputs=sequence_output, training=training)
        # 使用分类器模型对序列输出进行分类，生成 logits
        logits = self.classifier(inputs=sequence_output)
        # 如果有标签，则计算损失，否则损失为 None
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        # 如果 return_dict 为 False，则按照非字典格式返回结果
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则以 TFTokenClassifierOutput 格式返回结果
        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 构建模型的方法
    def build(self, input_shape=None):
        # 如果模型已经构建完成，则直接返回
        if self.built:
            return
        # 将模型标记为已构建
        self.built = True
        # 如果存在名为 bert 的模型，则在 bert 命名空间下构建它
        if getattr(self, "bert", None) is not None:
            with tf.name_scope(self.bert.name):
                self.bert.build(None)
        # 如果存在名为 classifier 的模型，则在 classifier 命名空间下构建它
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
"""
Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
"""
# 引入函数装饰器，用于向模型添加文档字符串
@add_start_docstrings(
    """
    Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    BERT_START_DOCSTRING,
)
# 声明 TF 模型类 TFBertForQuestionAnswering，继承自 TFBertPreTrainedModel 和 TFQuestionAnsweringLoss
class TFBertForQuestionAnswering(TFBertPreTrainedModel, TFQuestionAnsweringLoss):
    # 在从 PT 模型加载 TF 模型时，指定忽略的层名称正则表达式列表
    _keys_to_ignore_on_load_unexpected = [
        r"pooler",
        r"mlm___cls",
        r"nsp___cls",
        r"cls.predictions",
        r"cls.seq_relationship",
    ]

    # 初始化方法，接受一个 BertConfig 对象和其他可选输入参数
    def __init__(self, config: BertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 将配置中的标签数量赋值给实例变量 num_labels
        self.num_labels = config.num_labels

        # 创建一个 TFBertMainLayer 实例，用于 BERT 的主要层，不包含池化层，命名为 "bert"
        self.bert = TFBertMainLayer(config, add_pooling_layer=False, name="bert")
        
        # 创建一个全连接层 Dense 实例，用于 QA 输出，指定单元数为配置中的标签数量，
        # 使用指定范围内的初始化器来初始化权重，命名为 "qa_outputs"
        self.qa_outputs = keras.layers.Dense(
            units=config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="qa_outputs",
        )
        
        # 将配置对象保存为实例变量
        self.config = config

    # 使用装饰器声明 call 方法，定义模型的前向传播逻辑
    @unpack_inputs
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_QA,
        output_type=TFQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        qa_target_start_index=_QA_TARGET_START_INDEX,
        qa_target_end_index=_QA_TARGET_END_INDEX,
        expected_output=_QA_EXPECTED_OUTPUT,
        expected_loss=_QA_EXPECTED_LOSS,
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
        outputs = self.bert(
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
        # 获取BERT模型的输出，包括序列输出
        sequence_output = outputs[0]
        # 将序列输出传递给QA输出层进行预测
        logits = self.qa_outputs(inputs=sequence_output)
        # 将预测的logits张量按照最后一个维度分割成起始和结束位置的预测
        start_logits, end_logits = tf.split(value=logits, num_or_size_splits=2, axis=-1)
        # 去除多余的维度，使得张量的维度降低为2
        start_logits = tf.squeeze(input=start_logits, axis=-1)
        end_logits = tf.squeeze(input=end_logits, axis=-1)
        # 初始化损失变量为None
        loss = None

        # 如果提供了起始位置和结束位置的标签，则计算损失
        if start_positions is not None and end_positions is not None:
            labels = {"start_position": start_positions}
            labels["end_position"] = end_positions
            # 使用Hugging Face的损失计算函数计算损失
            loss = self.hf_compute_loss(labels=labels, logits=(start_logits, end_logits))

        # 如果不要求返回字典格式的输出，则根据条件返回输出结果
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回TFQuestionAnsweringModelOutput类型的输出，包括损失、起始和结束位置的logits以及隐藏状态和注意力
        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果模型已经构建完毕，则直接返回
        if self.built:
            return
        # 将模型标记为已构建状态
        self.built = True
        # 如果BERT模型存在，则构建BERT模型
        if getattr(self, "bert", None) is not None:
            with tf.name_scope(self.bert.name):
                self.bert.build(None)
        # 如果QA输出层存在，则构建QA输出层
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.config.hidden_size])
```