# `.\transformers\models\bert\modeling_tf_bert.py`

```
# 指定编码格式为 UTF-8
# 版权声明
# 导入必要的模块和库
# 声明该文件和类的版本支持，使用 Python 的 __future__ 模块
import math  # 导入 math 模块
import warnings  # 导入 warnings 模块
from dataclasses import dataclass  # 从 dataclasses 模块导入 dataclass 装饰器
from typing import Dict, Optional, Tuple, Union  # 导入类型提示相关的模块和类

import numpy as np  # 导入 numpy 库并使用 np 别名
import tensorflow as tf  # 导入 TensorFlow 库并使用 tf 别名
# 从模块中导入指定函数和类
from ...activations_tf import get_tf_activation  
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithPastAndCrossAttentions,
    TFBaseModelOutputWithPoolingAndCrossAttentions,
    TFCausalLMOutputWithCrossAttentions,
    TFMaskedLMOutput,
    TFMultipleChoiceModelOutput,
    TFNextSentencePredictorOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFMaskedLanguageModelingLoss,
    TFModelInputType,
    TFMultipleChoiceLoss,
    TFNextSentencePredictionLoss,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFTokenClassificationLoss,
    get_initializer,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax  # 导入相关函数
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# 从配置文件中导入 BertConfig 类
from .configuration_bert import BertConfig  

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 以下是用于文档的变量和常量定义

# 预训练模型的检查点和配置文件
_CHECKPOINT_FOR_DOC = "bert-base-uncased"
_CONFIG_FOR_DOC = "BertConfig"

# TokenClassification 文档字符串相关定义
_CHECKPOINT_FOR_TOKEN_CLASSIFICATION = "dbmdz/bert-large-cased-finetuned-conll03-english"  # TokenClassification 的预训练模型检查点
_TOKEN_CLASS_EXPECTED_OUTPUT = (
    "['O', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'I-LOC', 'O', 'I-LOC', 'I-LOC'] "
)  # TokenClassification 的预期输出
_TOKEN_CLASS_EXPECTED_LOSS = 0.01  # TokenClassification 的预期损失

# QuestionAnswering 文档字符串相关定义
_CHECKPOINT_FOR_QA = "ydshieh/bert-base-cased-squad2"  # QuestionAnswering 的预训练模型检查点
_QA_EXPECTED_OUTPUT = "'a nice puppet'"  # QuestionAnswering 的预期输出
_QA_EXPECTED_LOSS = 7.41  # QuestionAnswering 的预期损失
_QA_TARGET_START_INDEX = 14  # QuestionAnswering 目标答案的起始索引
_QA_TARGET_END_INDEX = 15  # QuestionAnswering 目标答案的结束索引

# SequenceClassification 文档字符串相关定义
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "ydshieh/bert-base-uncased-yelp-polarity"  # SequenceClassification 的预训练模型检查点
_SEQ_CLASS_EXPECTED_OUTPUT = "'LABEL_1'"  # SequenceClassification 的预期输出
_SEQ_CLASS_EXPECTED_LOSS = 0.01  # SequenceClassification 的预期损失

# 预训练模型存档列表
TF_BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bert-base-uncased",
    "bert-large-uncased",
    "bert-base-cased",
    "bert-large-cased",
]
    # 定义了一系列预训练的BERT模型的名称字符串
    "bert-base-multilingual-uncased",
    "bert-base-multilingual-cased",
    "bert-base-chinese",
    "bert-base-german-cased",
    "bert-large-uncased-whole-word-masking",
    "bert-large-cased-whole-word-masking",
    "bert-large-uncased-whole-word-masking-finetuned-squad",
    "bert-large-cased-whole-word-masking-finetuned-squad",
    "bert-base-cased-finetuned-mrpc",
    "cl-tohoku/bert-base-japanese",
    "cl-tohoku/bert-base-japanese-whole-word-masking",
    "cl-tohoku/bert-base-japanese-char",
    "cl-tohoku/bert-base-japanese-char-whole-word-masking",
    "TurkuNLP/bert-base-finnish-cased-v1",
    "TurkuNLP/bert-base-finnish-uncased-v1",
    "wietsedv/bert-base-dutch-cased",
    # 查看所有BERT模型，请访问 https://huggingface.co/models?filter=bert
# 导入所需的库
]

# 定义一个类 TFBertPreTrainingLoss，用于BERT类似的预训练任务，结合 NSP + MLM
class TFBertPreTrainingLoss:
    """
    Loss function suitable for BERT-like pretraining, that is, the task of pretraining a language model by combining
    NSP + MLM. .. note:: Any label of -100 will be ignored (along with the corresponding logits) in the loss
    computation.
    """

    # 计算损失函数
    def hf_compute_loss(self, labels: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
        # 定义交叉熵损失函数
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )

        # 将负标签裁剪为零，以避免 NaN 和错误 - 这些位置稍后会被掩盖
        unmasked_lm_losses = loss_fn(y_true=tf.nn.relu(labels["labels"]), y_pred=logits[0])
        # 确保只有不等于 -100 的标签才会被考虑在损失计算中
        lm_loss_mask = tf.cast(labels["labels"] != -100, dtype=unmasked_lm_losses.dtype)
        masked_lm_losses = unmasked_lm_losses * lm_loss_mask
        reduced_masked_lm_loss = tf.reduce_sum(masked_lm_losses) / tf.reduce_sum(lm_loss_mask)

        # 将负标签裁剪为零，以避免 NaN 和错误 - 这些位置稍后会被掩盖
        unmasked_ns_loss = loss_fn(y_true=tf.nn.relu(labels["next_sentence_label"]), y_pred=logits[1])
        ns_loss_mask = tf.cast(labels["next_sentence_label"] != -100, dtype=unmasked_ns_loss.dtype)
        masked_ns_loss = unmasked_ns_loss * ns_loss_mask

        reduced_masked_ns_loss = tf.reduce_sum(masked_ns_loss) / tf.reduce_sum(ns_loss_mask)

        # 返回重塑后的损失值
        return tf.reshape(reduced_masked_lm_loss + reduced_masked_ns_loss, (1,))


# 定义一个类 TFBertEmbeddings，用于构建来自单词、位置和标记类型嵌入的嵌入
class TFBertEmbeddings(tf.keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)

        # 初始化嵌入层
        self.config = config
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings
        self.initializer_range = config.initializer_range
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
    # 在构建模型时定义嵌入层的参数
    def build(self, input_shape=None):
        # 嵌入层：单词的嵌入向量
        with tf.name_scope("word_embeddings"):
            # 添加单词嵌入矩阵的权重参数
            self.weight = self.add_weight(
                name="weight",
                shape=[self.config.vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        # 嵌入层：标记类型的嵌入向量
        with tf.name_scope("token_type_embeddings"):
            # 添加标记类型嵌入矩阵的权重参数
            self.token_type_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.config.type_vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        # 嵌入层：位置的嵌入向量
        with tf.name_scope("position_embeddings"):
            # 添加位置嵌入矩阵的权重参数
            self.position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_position_embeddings, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        # 如果模型已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 LayerNorm 层，则构建该层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])

    # 实现模型的前向传播
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
        # 如果既未提供 input_ids 也未提供 inputs_embeds，则抛出异常
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Need to provide either `input_ids` or `input_embeds`.")

        # 如果提供了 input_ids，则使用嵌入层参数获取对应的嵌入向量
        if input_ids is not None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        # 获取输入嵌入张量的形状
        input_shape = shape_list(inputs_embeds)[:-1]

        # 如果未提供 token_type_ids，则创建形状与输入嵌入张量相同的张量，并用0填充
        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)

        # 如果未提供 position_ids，则创建位置嵌入张量，并将其添加到输入张量的位置
        if position_ids is None:
            position_ids = tf.expand_dims(
                tf.range(start=past_key_values_length, limit=input_shape[1] + past_key_values_length), axis=0
            )

        # 根据位置索引获取位置嵌入向量
        position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
        # 根据标记类型索引获取标记类型嵌入向量
        token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)
        # 将输入嵌入向量、位置嵌入向量和标记类型嵌入向量相加，得到最终的嵌入张量
        final_embeddings = inputs_embeds + position_embeds + token_type_embeds
        # 对最终的嵌入张量进行 LayerNorm 归一化处理
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        # 对最终的嵌入张量进行 dropout 处理（在训练时）
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)

        # 返回最终的嵌入张量
        return final_embeddings
class TFBertSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)

        # 如果隐藏层大小不能被注意力头数整除，则引发 ValueError 异常
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_attention_heads})"
            )

        # 设置注意力头数和每个注意力头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

        # 创建用于计算注意力的全连接层
        self.query = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        self.key = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key"
        )
        self.value = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )
        self.dropout = tf.keras.layers.Dropout(rate=config.attention_probs_dropout_prob)

        # 是否为解码器
        self.is_decoder = config.is_decoder
        self.config = config

    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        # 将形状从 [batch_size, seq_length, all_head_size] 转换为 [batch_size, seq_length, num_attention_heads, attention_head_size]
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
        # 代码被截断，call 方法的定义被省略了
        pass

    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在查询、键和值的全连接层，则构建它们
        if getattr(self, "query", None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.config.hidden_size])
        if getattr(self, "key", None) is not None:
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.config.hidden_size])
        if getattr(self, "value", None) is not None:
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.config.hidden_size])


class TFBertSelfOutput(tf.keras.layers.Layer):
    # 该类的注释被省略，因为在提供的代码片段中它并没有实现任何方法或属性
    # 初始化方法，接受一个BertConfig对象和其他关键字参数
    def __init__(self, config: BertConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 创建一个全连接层，单元数为config.hidden_size，使用指定的初始化器，命名为"dense"
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建一个LayerNormalization层，epsilon为config.layer_norm_eps，命名为"LayerNorm"
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建一个Dropout层，丢弃率为config.hidden_dropout_prob
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        # 保存传入的BertConfig对象
        self.config = config

    # 前向传播方法，接受隐藏状态、输入张量和训练标志，返回处理后的隐藏状态张量
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 使用全连接层处理隐藏状态
        hidden_states = self.dense(inputs=hidden_states)
        # 使用Dropout层处理隐藏状态
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 使用LayerNormalization层处理隐藏状态和输入张量的和
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states

    # 构建方法，用于构建层的内部结构
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在全连接层dense
        if getattr(self, "dense", None) is not None:
            # 在dense的命名空间下构建全连接层
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 如果存在LayerNormalization层LayerNorm
        if getattr(self, "LayerNorm", None) is not None:
            # 在LayerNorm的命名空间下构建LayerNormalization层
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])
class TFBertAttention(tf.keras.layers.Layer):
    # 定义一个自定义的 BERT Attention 层
    def __init__(self, config: BertConfig, **kwargs):
        # 初始化函数，接受一个 BertConfig 对象和其他参数
        super().__init__(**kwargs)

        # 创建 self_attention 和 dense_output 层
        self.self_attention = TFBertSelfAttention(config, name="self")
        self.dense_output = TFBertSelfOutput(config, name="output")

    # 修剪头部的方法，抛出未实现的错误
    def prune_heads(self, heads):
        raise NotImplementedError

    # 调用函数，接受多个输入参数，返回一个元组
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
        # 使用 self_attention 层处理输入数据
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
        # 使用 dense_output 层处理 self_attention 的输出
        attention_output = self.dense_output(
            hidden_states=self_outputs[0], input_tensor=input_tensor, training=training
        )
        # 如果需要输出注意力，将注意力信息添加到输出中
        outputs = (attention_output,) + self_outputs[1:]

        return outputs

    # 构建函数，用于构建层的结构
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        self.built = True
        # 构建 self_attention 层
        if getattr(self, "self_attention", None) is not None:
            with tf.name_scope(self.self_attention.name):
                self.self_attention.build(None)
        # 构建 dense_output 层
        if getattr(self, "dense_output", None) is not None:
            with tf.name_scope(self.dense_output.name):
                self.dense_output.build(None)


class TFBertIntermediate(tf.keras.layers.Layer):
    # 定义一个自定义的 BERT Intermediate 层
    def __init__(self, config: BertConfig, **kwargs):
        # 初始化函数，��受一个 BertConfig 对象和其他参数
        super().__init__(**kwargs)

        # 创建一个全连接层 dense
        self.dense = tf.keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        # 根据 hidden_act 类型获取激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        self.config = config

    # 调用函数，接受一个输入参数，返回处理后的数据
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 使用全连接层 dense 处理输入数据
        hidden_states = self.dense(inputs=hidden_states)
        # 使用激活函数处理 dense 的输出
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

    # 构建函数，用于构建层的结构
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        self.built = True
        # 构建 dense 层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


class TFBertOutput(tf.keras.layers.Layer):
    # 这部分代码未提供，需要继续补充
    # 初始化方法，接受一个BertConfig对象和其他关键字参数
    def __init__(self, config: BertConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 创建一个全连接层，单元数为config.hidden_size，使用指定的初始化器，命名为"dense"
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建一个LayerNormalization层，epsilon为config.layer_norm_eps，命名为"LayerNorm"
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建一个Dropout层，丢弃率为config.hidden_dropout_prob
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        # 保存传入的BertConfig对象
        self.config = config

    # 前向传播方法，接受隐藏状态、输入张量和训练标志，返回处理后的隐藏状态张量
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 使用全连接层处理隐藏状态
        hidden_states = self.dense(inputs=hidden_states)
        # 使用Dropout层处理隐藏状态
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 使用LayerNormalization层处理隐藏状态和输入张量的和
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        # 返回处理后的隐藏状态张量
        return hidden_states

    # 构建方法，用于构建层的内部结构
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在全连接层dense
        if getattr(self, "dense", None) is not None:
            # 在dense的命名空间下构建全连接层
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])
        # 如果存在LayerNormalization层LayerNorm
        if getattr(self, "LayerNorm", None) is not None:
            # 在LayerNorm的命名空间下构建LayerNormalization层
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])
class TFBertLayer(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)

        # 初始化 BERT 层的注意力层
        self.attention = TFBertAttention(config, name="attention")
        # 判断是否为解码器
        self.is_decoder = config.is_decoder
        # 是否添加跨注意力
        self.add_cross_attention = config.add_cross_attention
        # 如果添加了跨注意力
        if self.add_cross_attention:
            # 如果不是解码器，则抛出错误
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 初始化跨注意力层
            self.crossattention = TFBertAttention(config, name="crossattention")
        # 初始化 BERT 层的中间层
        self.intermediate = TFBertIntermediate(config, name="intermediate")
        # 初始化 BERT 层的输出层
        self.bert_output = TFBertOutput(config, name="output")

    # 定义 BERT 层的前向传播函数
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
    ) -> Tuple[tf.Tensor]:  # 函数定义，接受一个参数，返回一个包含 Tensor 的元组
        # 如果过去的键/值缓存不为空，则解码器单向自注意力的缓存键/值元组位于位置 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 使用自注意力模块处理输入张量
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
        # 获取自注意力模块的输出
        attention_output = self_attention_outputs[0]

        # 如果是解码器，最后一个输出是自注意力缓存的元组
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]  # 获取除了最后一个元素外的所有元素
            present_key_value = self_attention_outputs[-1]  # 获取最后一个元素作为当前键/值
        else:
            outputs = self_attention_outputs[1:]  # 如果输出注意力权重，添加自注意力
                                                  # 注意：这里没有详细说明是添加自注意力的权重还是其他内容
                                                  # 可能需要根据上下文进一步理解
                                                  

        cross_attn_present_key_value = None
        # 如果是解码器并且存在编码器隐藏状态
        if self.is_decoder and encoder_hidden_states is not None:
            # 如果没有跨注意力层，抛出异常
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # 跨注意力缓存键/值元组位于过去键/值元组的位置 3,4
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 使用跨注意力模块处理输入张量
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
            # 获取跨注意力模块的输出
            attention_output = cross_attention_outputs[0]
            # 添加跨注意力的输出到输出元组中
            outputs = outputs + cross_attention_outputs[1:-1]

            # 添加跨注意力缓存到当前键/值元组
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 使用中间层模块处理注意力输出
        intermediate_output = self.intermediate(hidden_states=attention_output)
        # 使用 BERT 输出模块处理中间层输出
        layer_output = self.bert_output(
            hidden_states=intermediate_output, input_tensor=attention_output, training=training
        )
        outputs = (layer_output,) + outputs  # 如果输出注意力，添加到输出元组中

        # 如果是解码器，将注意力键/值作为最后一个输出返回
        if self.is_decoder:
            outputs = outputs + (present_key_value,)  # 添加当前键/值元组到输出中

        return outputs
```  
    # 构建模型，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 标记模型已经构建
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
        # 如果存在 BERT 输出层，则构建 BERT 输出层
        if getattr(self, "bert_output", None) is not None:
            # 使用 BERT 输出层的名称作为命名空间，构建 BERT 输出层
            with tf.name_scope(self.bert_output.name):
                self.bert_output.build(None)
        # 如果存在交叉注意力层，则构建交叉注意力层
        if getattr(self, "crossattention", None) is not None:
            # 使用交叉注意力层的名称作为命名空间，构建交叉注意力层
            with tf.name_scope(self.crossattention.name):
                self.crossattention.build(None)
class TFBertEncoder(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs):
        # 调用父类构造函数初始化实例属性
        super().__init__(**kwargs)
        # 存储 BERT 模型配置
        self.config = config
        # 创建 BERT 编码层列表，包含多个 TF-BERT 层
        self.layer = [TFBertLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        encoder_hidden_states: tf.Tensor | None,
        encoder_attention_mask: tf.Tensor | None,
        past_key_values: Tuple[Tuple[tf.Tensor]] | None,
        use_cache: Optional[bool],
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPastAndCrossAttentions, Tuple[tf.Tensor]]:
        # 如果需要输出隐藏状态，则初始化空元组
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力，则初始化空元组
        all_attentions = () if output_attentions else None
        # 如果需要输出交叉注意力且配置允许，则初始化空元组
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 如果需要使用缓存，则初始化空元组
        next_decoder_cache = () if use_cache else None
        # 遍历所有的 BERT 层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到列表中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取上一层的缓存键值对，若不存在则为 None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 调用当前层的前向传播函数
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

            # 如果需要使用缓存，则将当前层的缓存添加到列表中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            # 如果需要输出注意力，则将当前层的注意力添加到列表中
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
                # 如果配置允许添加交叉注意力且存在编码器隐藏状态，则将当前层的交叉注意力添加到列表中
                if self.config.add_cross_attention and encoder_hidden_states is not None:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果需要输出隐藏状态，则将最后一层的隐藏状态添加到列表中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要以字典形式返回结果，则返回非空元素的元组
        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_attentions, all_cross_attentions] if v is not None
            )

        # 以字典形式返回结果
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
        # 标记为已构建
        self.built = True
        # 检查是否存在子层
        if getattr(self, "layer", None) is not None:
            # 遍历每个子层
            for layer in self.layer:
                # 使用子层的名称创建命名空间
                with tf.name_scope(layer.name):
                    # 构建子层
                    layer.build(None)
# 定义一个自定义的 TensorFlow 层用于 BERT 模型的池化操作
class TFBertPooler(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)

        # 定义一个全连接层，用于将隐藏状态映射到指定大小的向量空间
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )
        # 保存配置信息
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 池化模型，只使用第一个 token 对应的隐藏状态
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(inputs=first_token_tensor)

        return pooled_output

    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        self.built = True
        # 如果全连接层已经定义，则构建全连接层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


class TFBertPredictionHeadTransform(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)

        # 定义一个全连接层，用于转换隐藏状态的维度
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense",
        )

        # 获取隐藏层激活函数并设置
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.transform_act_fn = config.hidden_act

        # 使用 LayerNormalization 进行归一化
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 保存配置信息
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 经过全连接层
        hidden_states = self.dense(inputs=hidden_states)
        # 使用激活函数
        hidden_states = self.transform_act_fn(hidden_states)
        # 进行 LayerNormalization
        hidden_states = self.LayerNorm(inputs=hidden_states)

        return hidden_states

    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        self.built = True
        # 如果全连接层已经定义，则构建全连接层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 如果 LayerNormalization 已经定义，则构建 LayerNormalization
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


class TFBertLMPredictionHead(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig, input_embeddings: tf.keras.layers.Layer, **kwargs):
        super().__init__(**kwargs)

        # 保存配置信息和隐藏状态的大小
        self.config = config
        self.hidden_size = config.hidden_size

        # 定义一个用于预测的头部变换层
        self.transform = TFBertPredictionHeadTransform(config, name="transform")

        # 输出权重与输入嵌入相同，但每个 token 都有一个输出偏置
        self.input_embeddings = input_embeddings
    # 构建方法，用于构建模型
    def build(self, input_shape=None):
        # 添加偏置项权重，形状为词汇表大小，初始化为零，可训练，命名为"bias"
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="bias")

        # 如果已经构建过，直接返回
        if self.built:
            return
        # 设置标记为已构建
        self.built = True
        # 如果存在transform属性
        if getattr(self, "transform", None) is not None:
            # 使用transform的名称空间
            with tf.name_scope(self.transform.name):
                # 构建transform
                self.transform.build(None)

    # 获取输出嵌入层
    def get_output_embeddings(self) -> tf.keras.layers.Layer:
        # 返回输入嵌入层
        return self.input_embeddings

    # 设置输出嵌入层
    def set_output_embeddings(self, value: tf.Variable):
        # 设置输入嵌入层权重为给定值
        self.input_embeddings.weight = value
        # 设置输入嵌入层词汇表大小为给定值的形状的第一个维度大小
        self.input_embeddings.vocab_size = shape_list(value)[0]

    # 获取偏置项
    def get_bias(self) -> Dict[str, tf.Variable]:
        # 返回包含偏置项的字典
        return {"bias": self.bias}

    # 设置偏置项
    def set_bias(self, value: tf.Variable):
        # 设置偏置项为给定值的偏置项
        self.bias = value["bias"]
        # 设置配置中的词汇表大小为给定值的形状的第一个维度大小
        self.config.vocab_size = shape_list(value["bias"])[0]

    # 模型调用方法
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 对隐藏状态进行transform
        hidden_states = self.transform(hidden_states=hidden_states)
        # 获取序列长度
        seq_length = shape_list(hidden_states)[1]
        # 将隐藏状态重塑为二维张量
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.hidden_size])
        # 计算嵌入后的隐藏状态
        hidden_states = tf.matmul(a=hidden_states, b=self.input_embeddings.weight, transpose_b=True)
        # 将隐藏状态重塑为三维张量
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])
        # 添加偏置项到隐藏状态
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)

        # 返回处理后的隐藏状态
        return hidden_states
class TFBertMLMHead(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig, input_embeddings: tf.keras.layers.Layer, **kwargs):
        super().__init__(**kwargs)

        # 初始化 Masked Language Model（MLM）的预测头部
        self.predictions = TFBertLMPredictionHead(config, input_embeddings, name="predictions")

    def call(self, sequence_output: tf.Tensor) -> tf.Tensor:
        # 使用预测头部生成序列的预测分数
        prediction_scores = self.predictions(hidden_states=sequence_output)

        return prediction_scores

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "predictions", None) is not None:
            with tf.name_scope(self.predictions.name):
                # 构建预测头部
                self.predictions.build(None)


class TFBertNSPHead(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)

        # 初始化 Next Sentence Prediction（NSP）的头部
        self.seq_relationship = tf.keras.layers.Dense(
            units=2,
            kernel_initializer=get_initializer(config.initializer_range),
            name="seq_relationship",
        )
        self.config = config

    def call(self, pooled_output: tf.Tensor) -> tf.Tensor:
        # 使用 NSP 头部生成序列对的关系分数
        seq_relationship_score = self.seq_relationship(inputs=pooled_output)

        return seq_relationship_score

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "seq_relationship", None) is not None:
            with tf.name_scope(self.seq_relationship.name):
                # 构建 NSP 头部
                self.seq_relationship.build([None, None, self.config.hidden_size])


@keras_serializable
class TFBertMainLayer(tf.keras.layers.Layer):
    config_class = BertConfig

    def __init__(self, config: BertConfig, add_pooling_layer: bool = True, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.is_decoder = config.is_decoder

        # 初始化 BERT 模型的嵌入层、编码器和池化层（如果需要）
        self.embeddings = TFBertEmbeddings(config, name="embeddings")
        self.encoder = TFBertEncoder(config, name="encoder")
        self.pooler = TFBertPooler(config, name="pooler") if add_pooling_layer else None

    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        # 返回嵌入层
        return self.embeddings

    def set_input_embeddings(self, value: tf.Variable):
        # 设置输入嵌入的权重并更新词汇表大小
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    @unpack_inputs
    # 定义一个方法，用于执行模型推断或训练过程中的前向传播
    def call(
        self,
        input_ids: TFModelInputType | None = None,  # 输入的 token IDs
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 注意力遮罩
        token_type_ids: np.ndarray | tf.Tensor | None = None,  # token 类型 IDs
        position_ids: np.ndarray | tf.Tensor | None = None,  # 位置 IDs
        head_mask: np.ndarray | tf.Tensor | None = None,  # 注意力头遮罩
        inputs_embeds: np.ndarray | tf.Tensor | None = None,  # 输入嵌入向量
        encoder_hidden_states: np.ndarray | tf.Tensor | None = None,  # 编码器隐藏状态
        encoder_attention_mask: np.ndarray | tf.Tensor | None = None,  # 编码器的注意力遮罩
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,  # 过去的键值对
        use_cache: Optional[bool] = None,  # 是否使用缓存
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典形式结果
        training: bool = False,  # 是否处于训练模式
    # 构建模型结构，包括嵌入层、编码器和池化层
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记已经构建
        self.built = True
        # 如果存在嵌入层，构建嵌入层
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        # 如果存在编码器，构建编码器
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 如果存在池化层，构建池化层
        if getattr(self, "pooler", None) is not None:
            with tf.name_scope(self.pooler.name):
                self.pooler.build(None)
```  
class TFBertPreTrainedModel(TFPreTrainedModel):
    """
    一个抽象类，用于处理权重初始化以及下载和加载预训练模型的简单接口。
    """

    # 设置配置类为BertConfig
    config_class = BertConfig
    # 基础模型前缀为"bert"
    base_model_prefix = "bert"


@dataclass
class TFBertForPreTrainingOutput(ModelOutput):
    """
    [`TFBertForPreTraining`]的输出类型。

    Args:
        prediction_logits (`tf.Tensor`，形状为`(batch_size, sequence_length, config.vocab_size)`):
            语言建模头的预测分数（SoftMax之前的每个词汇标记的分数）。
        seq_relationship_logits (`tf.Tensor`，形状为`(batch_size, 2)`):
            下一个序列预测（分类）头的预测分数（SoftMax之前的True/False继续的分数）。
        hidden_states (`tuple(tf.Tensor)`，*可选*，当传递`output_hidden_states=True`或`config.output_hidden_states=True`时返回):
            形状为`(batch_size, sequence_length, hidden_size)`的`tf.Tensor`元组
            模型在每一层输出时的隐藏状态加上初始嵌入输出。
        attentions (`tuple(tf.Tensor)`，*可选*，当传递`output_attentions=True`或`config.output_attentions=True`时返回):
            形状为`(batch_size, num_heads, sequence_length, sequence_length)`的`tf.Tensor`元组
            自注意力头中注意力softmax之后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    loss: tf.Tensor | None = None
    prediction_logits: tf.Tensor = None
    seq_relationship_logits: tf.Tensor = None
    hidden_states: Optional[Union[Tuple[tf.Tensor], tf.Tensor]] = None
    attentions: Optional[Union[Tuple[tf.Tensor], tf.Tensor]] = None


BERT_START_DOCSTRING = r"""

    该模型继承自[`TFPreTrainedModel`]。检查超类文档以获取库实现的所有模型通用方法（如下载或保存、调整输入嵌入、修剪头等）。

    该模型也是一个[tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model)子类。将其用作常规的TF 2.0 Keras模型，并参考TF 2.0文档以获取有关一般用法和行为的所有相关内容。

    <Tip>

    `transformers`中的TensorFlow模型和层接受两种格式的输入：

    - 将所有输入作为关键字参数（类似于PyTorch模型），或者
    - 将所有输入作为列表、元组或字典放在第一个位置参数中。

    支持第二种格式的原因是，当将输入传递给模型和层时，Keras方法更喜欢此格式。由于此支持，在使用`model.fit()`等方法时，应该可以正常工作 - 只需
    pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second
    format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with
    the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first
    positional argument:

# 在任何 `model.fit()` 支持的格式中传递输入和标签！然而，如果你想在 Keras 方法之外使用第二种格式，比如在使用 Keras `Functional` API 创建自己的层或模型时，有三种可能性可以在第一个位置参数中收集所有输入张量：

    - a single Tensor with `input_ids` only and nothing else: `model(input_ids)`
    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
    `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`
    - a dictionary with one or several input Tensors associated to the input names given in the docstring:
    `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

# - 仅具有 `input_ids` 的单个张量，没有其他内容：`model(input_ids)`
# - 变长列表，其中包含按照文档字符串中给定的顺序的一个或多个输入张量：`model([input_ids, attention_mask])` 或 `model([input_ids, attention_mask, token_type_ids])`
# - 字典，其中包含与文档字符串中给定的输入名称相关联的一个或多个输入张量：`model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    Note that when creating models and layers with
    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don't need to worry
    about any of this, as you can just pass inputs like you would to any other Python function!

# 注意，在使用 [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) 创建模型和层时，你不需要担心任何这些，因为你可以像对待任何其他 Python 函数一样传递输入！

    Args:
        config ([`BertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.

# 参数：
# - config ([`BertConfig`]): 包含模型所有参数的模型配置类。
#   使用配置文件初始化不会加载与模型关联的权重，只加载配置。查看 [`~TFPreTrainedModel.from_pretrained`] 方法以加载模型权重。
"""
BERT 输入的文档字符串
"""
"""

@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",  # 对不带特定头部的原始隐藏状态输出的 Bert 模型变压器进行解释
    BERT_START_DOCSTRING,  # 引用 BERT_START_DOCSTRING 中的文档字符串
)
class TFBertModel(TFBertPreTrainedModel):  # 定义 TFBertModel 类，继承自 TFBertPreTrainedModel
    def __init__(self, config: BertConfig, *inputs, **kwargs):  # 初始化方法，接收 BertConfig 和其他输入
        super().__init__(config, *inputs, **kwargs)  # 调用父类的初始化方法

        self.bert = TFBertMainLayer(config, name="bert")  # 实例化 TFBertMainLayer 并赋值给 self.bert

    @unpack_inputs  # 解包输入
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))  # 添加模型前向传播的文档字符串
    @add_code_sample_docstrings(  # 添加代码示例的文档字符串
        checkpoint=_CHECKPOINT_FOR_DOC,  # 使用的检查点
        output_type=TFBaseModelOutputWithPoolingAndCrossAttentions,  # 输出类型
        config_class=_CONFIG_FOR_DOC,  # 配置类
    )
    def call(  # 模型的前向传播方法
        self,
        input_ids: TFModelInputType | None = None,  # 输入的 token IDs
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 注意力遮罩
        token_type_ids: np.ndarray | tf.Tensor | None = None,  # token 类型 IDs
        position_ids: np.ndarray | tf.Tensor | None = None,  # 位置 IDs
        head_mask: np.ndarray | tf.Tensor | None = None,  # 头部遮罩
        inputs_embeds: np.ndarray | tf.Tensor | None = None,  # 嵌入式输入
        encoder_hidden_states: np.ndarray | tf.Tensor | None = None,  # 编码器隐藏状态
        encoder_attention_mask: np.ndarray | tf.Tensor | None = None,  # 编码器注意力遮罩
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,  # 过去的键值对
        use_cache: Optional[bool] = None,  # 是否使用缓存
        output_attentions: Optional[bool] = None,  # 是否输出注意力
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典
        training: Optional[bool] = False,  # 是否处于训练模式
    def call(self, input_ids: tf.Tensor, attention_mask: tf.Tensor, token_type_ids: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None, head_mask: Optional[tf.Tensor] = None, inputs_embeds: Optional[tf.Tensor] = None,
        encoder_hidden_states: Optional[tf.Tensor] = None, encoder_attention_mask: Optional[tf.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[tf.Tensor]]] = None, use_cache: Optional[bool] = True,
        output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = True,
        training: Optional[bool] = False) -> Union[TFBaseModelOutputWithPoolingAndCrossAttentions, Tuple[tf.Tensor]]:
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
        # 调用 BERT 模型，传入各种输入参数
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
        # 返回 BERT 模型的输出
        return outputs

    def build(self, input_shape=None):
        # 如果已经构建过模型，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 BERT 模型，则构建 BERT 模型
        if getattr(self, "bert", None) is not None:
            with tf.name_scope(self.bert.name):
                self.bert.build(None)
# 使用预定义的文档字符串注释 Bert 模型，该模型包含两个预训练头部：
# 一个是“masked language modeling（MLM）”头部，另一个是“next sentence prediction（NSP）”头部。
# 基于 BERT_START_DOCSTRING 和额外的说明构建开始文档字符串。
class TFBertForPreTraining(TFBertPreTrainedModel, TFBertPreTrainingLoss):
    # 在加载 TF 模型时，需要忽略的预定义不匹配的层的名称列表
    _keys_to_ignore_on_load_unexpected = [
        r"position_ids",
        r"cls.predictions.decoder.weight",
        r"cls.predictions.decoder.bias",
    ]

    # 初始化方法，接受 BertConfig 对象和其他输入参数
    def __init__(self, config: BertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 创建 Bert 主层，使用指定的名称命名为 "bert"
        self.bert = TFBertMainLayer(config, name="bert")
        # 创建 NSP 头部，使用指定的名称命名为 "nsp___cls"
        self.nsp = TFBertNSPHead(config, name="nsp___cls")
        # 创建 MLM 头部，使用指定的名称命名为 "mlm___cls"，并将输入嵌入层设置为 bert.embeddings
        self.mlm = TFBertMLMHead(config, input_embeddings=self.bert.embeddings, name="mlm___cls")

    # 获取 MLM 头部
    def get_lm_head(self) -> tf.keras.layers.Layer:
        return self.mlm.predictions

    # 获取前缀偏置的名称，已弃用，发出警告
    def get_prefix_bias_name(self) -> str:
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        return self.name + "/" + self.mlm.name + "/" + self.mlm.predictions.name

    # 模型调用方法，接受各种输入参数和配置，并返回预训练输出
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
        # 此处为模型调用的具体实现，详细逻辑不在注释中
        pass

    # 构建模型，根据输入形状构建子层
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "bert", None) is not None:
            with tf.name_scope(self.bert.name):
                self.bert.build(None)
        if getattr(self, "nsp", None) is not None:
            with tf.name_scope(self.nsp.name):
                self.nsp.build(None)
        if getattr(self, "mlm", None) is not None:
            with tf.name_scope(self.mlm.name):
                self.mlm.build(None)


# 使用预定义的文档字符串注释 Bert 模型，该模型具有顶部的“language modeling（LM）”头部
# 继承自 TFBertPreTrainedModel 和 TFMaskedLanguageModelingLoss
@add_start_docstrings("""Bert Model with a `language modeling` head on top.""", BERT_START_DOCSTRING)
class TFBertForMaskedLM(TFBertPreTrainedModel, TFMaskedLanguageModelingLoss):
    # 在加载 TF 模型时，需要忽略的预定义不匹配的层的名称列表
    # 初始化一个列表，包含需要在加载时忽略的键名
    _keys_to_ignore_on_load_unexpected = [
        r"pooler",
        r"cls.seq_relationship",
        r"cls.predictions.decoder.weight",
        r"nsp___cls",
    ]

    # 初始化方法，接受一个BertConfig对象和其他可变参数
    def __init__(self, config: BertConfig, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 如果配置为decoder，则发出警告，说明应该将`config.is_decoder`设为False以使用双向自注意力
        if config.is_decoder:
            logger.warning(
                "If you want to use `TFBertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # 初始化BERT主层，不包含池化层，命名为"bert"
        self.bert = TFBertMainLayer(config, add_pooling_layer=False, name="bert")
        # 初始化BERT的MLM头部，传入输入嵌入层，命名为"mlm___cls"
        self.mlm = TFBertMLMHead(config, input_embeddings=self.bert.embeddings, name="mlm___cls")

    # 获取语言模型头部
    def get_lm_head(self) -> tf.keras.layers.Layer:
        return self.mlm.predictions

    # 获取前缀偏置名称
    def get_prefix_bias_name(self) -> str:
        # 发出警告，说明该方法已被弃用，请使用`get_bias`代替
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        # 返回名称组成的字符串
        return self.name + "/" + self.mlm.name + "/" + self.mlm.predictions.name

    # 调用方法，接受多个输入参数，包括输入的ID、注意力掩码、token类型ID、位置ID、头掩码、嵌入层输入等
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
        # 调用BERT模型进行前向传播，得到模型输出
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
        # 从BERT模型的输出中提取序列输出
        sequence_output = outputs[0]
        # 将序列输出传递给MLM头，得到预测的token分数
        prediction_scores = self.mlm(sequence_output=sequence_output, training=training)
        # 如果提供了标签，则计算MLM损失
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=prediction_scores)

        # 如果return_dict为False，则返回未打包的输出元组
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果return_dict为True，则返回包含损失、预测分数、隐藏状态和注意力的TFMaskedLMOutput对象
        return TFMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回
        if self.built:
            return
        # 将模型标记为已构建
        self.built = True
        # 构建BERT模型
        if getattr(self, "bert", None) is not None:
            with tf.name_scope(self.bert.name):
                self.bert.build(None)
        # 构建MLM头
        if getattr(self, "mlm", None) is not None:
            with tf.name_scope(self.mlm.name):
                self.mlm.build(None)
class TFBertLMHeadModel(TFBertPreTrainedModel, TFCausalLanguageModelingLoss):
    # 定义一个类 TFBertLMHeadModel，继承自 TFBertPreTrainedModel 和 TFCausalLanguageModelingLoss

    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    # 定义一个列表，包含在从 PT 模型加载 TF 模型时授权的意外/缺失层
    _keys_to_ignore_on_load_unexpected = [
        r"pooler",
        r"cls.seq_relationship",
        r"cls.predictions.decoder.weight",
        r"nsp___cls",
    ]

    def __init__(self, config: BertConfig, *inputs, **kwargs):
        # 初始化方法，接受 BertConfig 类型的 config 参数和其他可变数量的位置参数和关键字参数
        super().__init__(config, *inputs, **kwargs)

        if not config.is_decoder:
            # 如果 config 不是 decoder，则发出警告
            logger.warning("If you want to use `TFBertLMHeadModel` as a standalone, add `is_decoder=True.`")

        # 创建一个 TFBertMainLayer 对象，不添加 pooling 层，命名为 "bert"
        self.bert = TFBertMainLayer(config, add_pooling_layer=False, name="bert")
        # 创建一个 TFBertMLMHead 对象，传入 config 和 self.bert.embeddings 作为 input_embeddings，命名为 "mlm___cls"
        self.mlm = TFBertMLMHead(config, input_embeddings=self.bert.embeddings, name="mlm___cls")

    def get_lm_head(self) -> tf.keras.layers.Layer:
        # 返回 mlm.predictions 层
        return self.mlm.predictions

    def get_prefix_bias_name(self) -> str:
        # 发出警告，方法已弃用，请使用 `get_bias` 替代
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        # 返回包含名称的字符串
        return self.name + "/" + self.mlm.name + "/" + self.mlm.predictions.name

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        # 准备生成的输入，根据传入的参数返回相应的输入字典
        input_shape = input_ids.shape
        # 如果 attention_mask 为 None，则创建一个全为 1 的 mask
        if attention_mask is None:
            attention_mask = tf.ones(input_shape)

        # 如果 past_key_values 不为 None，则截取 decoder_input_ids
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # 返回包含输入的字典
        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}

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
    # 构建模型，如果已经构建过则直接返回
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在属性"bert"，则构建"bert"模型
        if getattr(self, "bert", None) is not None:
            # 在命名空间下构建"bert"模型
            with tf.name_scope(self.bert.name):
                self.bert.build(None)
        # 如果存在属性"mlm"，则构建"mlm"模型
        if getattr(self, "mlm", None) is not None:
            # 在命名空间下构建"mlm"模型
            with tf.name_scope(self.mlm.name):
                self.mlm.build(None)
# 使用指定的文档字符串添加一个分类头部的 BERT 模型，用于下一句预测任务
# 这个类继承自 TFBertPreTrainedModel 类和 TFNextSentencePredictionLoss 类
class TFBertForNextSentencePrediction(TFBertPreTrainedModel, TFNextSentencePredictionLoss):
    # 在从 PT 模型加载 TF 模型时，以下名字表示授权的意外/缺失的层
    _keys_to_ignore_on_load_unexpected = [r"mlm___cls", r"cls.predictions"]

    def __init__(self, config: BertConfig, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 创建一个 BERT 主层，指定名称为 "bert"
        self.bert = TFBertMainLayer(config, name="bert")
        # 创建一个 BERT 下一句预测头部，指定名称为 "nsp___cls"
        self.nsp = TFBertNSPHead(config, name="nsp___cls")

    # 用于模型调用的方法，包括输入参数的解包和文档字符串的添加
    @unpack_inputs
    # 添加输入的文档字符串，格式化输入参数的批量大小和序列长度
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 替换返回值的文档字符串类型为 TFNextSentencePredictorOutput，使用指定的配置类
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
    ) -> Union[TFNextSentencePredictorOutput, Tuple[tf.Tensor]]:
        r"""
        返回预测下一句的输出。

        Examples:

        ```python
        >>> import tensorflow as tf
        >>> from transformers import AutoTokenizer, TFBertForNextSentencePrediction

        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> model = TFBertForNextSentencePrediction.from_pretrained("bert-base-uncased")

        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
        >>> encoding = tokenizer(prompt, next_sentence, return_tensors="tf")

        >>> logits = model(encoding["input_ids"], token_type_ids=encoding["token_type_ids"])[0]
        >>> assert logits[0][0] < logits[0][1]  # the next sentence was random
        ```"""
        获取BERT模型的输出。

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
        获取BERT模型的输出。

        pooled_output = outputs[1]
        从BERT模型的输出中获取汇总输出。

        seq_relationship_scores = self.nsp(pooled_output=pooled_output)
        使用NSP模型预测下一句的相关性得分。

        next_sentence_loss = (
            None
            if next_sentence_label is None
            else self.hf_compute_loss(labels=next_sentence_label, logits=seq_relationship_scores)
        )
        计算下一句预测的损失。

        if not return_dict:
            output = (seq_relationship_scores,) + outputs[2:]
            如果不返回字典，则返回输出元组。

            return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output
            如果存在下一句损失，则将其添加到输出中，否则只返回输出。

        return TFNextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        返回TFNextSentencePredictorOutput对象，包含损失、预测的下一句相关性得分以及隐藏状态和注意力。

    def build(self, input_shape=None):
        如果模型已经构建，则返回。

        if self.built:
            return
        设置模型已构建标志。

        self.built = True
        如果存在BERT模型，则构建BERT模型。

        if getattr(self, "bert", None) is not None:
            with tf.name_scope(self.bert.name):
                self.bert.build(None)
        如果存在NSP模型，则构建NSP模型。

        if getattr(self, "nsp", None) is not None:
            with tf.name_scope(self.nsp.name):
                self.nsp.build(None)
# 导入所需模块
@add_start_docstrings(
    """
    Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    BERT_START_DOCSTRING,
)
# 声明一个用于序列分类/回归任务的 BERT 模型转换器，顶部带有一个线性层（放在汇聚输出之上），例如用于 GLUE 任务
class TFBertForSequenceClassification(TFBertPreTrainedModel, TFSequenceClassificationLoss):
    # 在从 PT 模型加载 TF 模型时，带有 '.' 的名称表示授权的未预期/丢失的层
    _keys_to_ignore_on_load_unexpected = [r"mlm___cls", r"nsp___cls", r"cls.predictions", r"cls.seq_relationship"]
    # 在从 PT 模型加载 TF 模型时，授权的丢失层
    _keys_to_ignore_on_load_missing = [r"dropout"]

    # 初始化方法
    def __init__(self, config: BertConfig, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 获取标签的数量
        self.num_labels = config.num_labels

        # 实例化一个 BERT 主层
        self.bert = TFBertMainLayer(config, name="bert")
        # 如果分类器的丢弃率不为空，则使用其值；否则使用隐藏层丢弃率作为分类器丢弃率
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 实例化一个丢弃层
        self.dropout = tf.keras.layers.Dropout(rate=classifier_dropout)
        # 实例化一个全连接层作为分类器
        self.classifier = tf.keras.layers.Dense(
            units=config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="classifier",
        )
        # 存储配置信息
        self.config = config

    # 定义 call 方法，实现模型的前向传播
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
    ) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 调用 BERT 模型进行序列分类/回归预测，返回包含输出的对象
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
        # 获取汇总的输出，通常是[CLS]标记对应的输出
        pooled_output = outputs[1]
        # 对汇总的输出应用 dropout，用于防止过拟合
        pooled_output = self.dropout(inputs=pooled_output, training=training)
        # 使用分类器层生成最终的分类/回归预测
        logits = self.classifier(inputs=pooled_output)
        # 计算损失，如果提供了标签
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        # 如果不要求以字典形式返回输出
        if not return_dict:
            # 组装输出
            output = (logits,) + outputs[2:]
            # 返回输出，包括损失和其他输出
            return ((loss,) + output) if loss is not None else output

        # 以 TFSequenceClassifierOutput 对象形式返回输出，包括损失、预测结果、隐藏状态和注意力权重
        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回
        if self.built:
            return
        # 设置模型已构建的标志
        self.built = True
        # 如果存在 BERT 模型，则构建 BERT 模型
        if getattr(self, "bert", None) is not None:
            with tf.name_scope(self.bert.name):
                self.bert.build(None)
        # 如果存在分类器层，则构建分类器层
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                # 构建分类器，指定输入形状为 [None, None, self.config.hidden_size]
                self.classifier.build([None, None, self.config.hidden_size])
# 定义一个带有多选分类头部的 Bert 模型（在池化输出之上有一个线性层和一个 softmax），例如用于 RocStories/SWAG 任务
@add_start_docstrings(
    """
    Bert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    BERT_START_DOCSTRING,
)
class TFBertForMultipleChoice(TFBertPreTrainedModel, TFMultipleChoiceLoss):
    # 在从 PT 模型加载 TF 模型时，带有 '.' 的名称表示授权的意外/缺失层
    _keys_to_ignore_on_load_unexpected = [r"mlm___cls", r"nsp___cls", r"cls.predictions", r"cls.seq_relationship"]
    _keys_to_ignore_on_load_missing = [r"dropout"]

    def __init__(self, config: BertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 初始化 Bert 主层
        self.bert = TFBertMainLayer(config, name="bert")
        # 初始化 dropout 层
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        # 初始化分类器
        self.classifier = tf.keras.layers.Dense(
            units=1, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        self.config = config

    # 定义模型的前向传播方法
    @unpack_inputs
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
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
    ) -> Union[TFMultipleChoiceModelOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
            where `num_choices` is the size of the second dimension of the input tensors. (See `input_ids` above)
        """
        # 检查输入的参数，返回类型为 TFMultipleChoiceModelOutput 或 Tuple[tf.Tensor]
        # 如果存在 labels 参数，表示计算多选分类损失，labels 的形状应为 (batch_size,)，可选参数
        # labels 应当是在 `[0, ..., num_choices]` 范围内的索引，其中 `num_choices` 是输入张量的第二个维度的大小（参见上面的 `input_ids`）
        if input_ids is not None:
            # 如果 input_ids 不为 None，则获取 num_choices 和 seq_length
            num_choices = shape_list(input_ids)[1]
            seq_length = shape_list(input_ids)[2]
        else:
            # 如果 input_ids 为 None，则获取 num_choices 和 seq_length
            num_choices = shape_list(inputs_embeds)[1]
            seq_length = shape_list(inputs_embeds)[2]

        # 将输入张量展平为二维张量，如果输入为 None，则相应的输出也为 None
        flat_input_ids = tf.reshape(tensor=input_ids, shape=(-1, seq_length)) if input_ids is not None else None
        flat_attention_mask = (
            tf.reshape(tensor=attention_mask, shape=(-1, seq_length)) if attention_mask is not None else None
        )
        flat_token_type_ids = (
            tf.reshape(tensor=token_type_ids, shape=(-1, seq_length)) if token_type_ids is not None else None
        )
        flat_position_ids = (
            tf.reshape(tensor=position_ids, shape=(-1, seq_length)) if position_ids is not None else None
        )
        flat_inputs_embeds = (
            tf.reshape(tensor=inputs_embeds, shape=(-1, seq_length, shape_list(inputs_embeds)[3]))
            if inputs_embeds is not None
            else None
        )
        # 调用 BERT 模型来处理平展后的输入张量和其他参数
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
        # 获取汇聚后的输出
        pooled_output = outputs[1]
        # 使用 dropout 对汇聚后的输出进行处理，以防止过拟合
        pooled_output = self.dropout(inputs=pooled_output, training=training)
        # 将处理后的输出输入分类器，得到 logits
        logits = self.classifier(inputs=pooled_output)
        # 重新整形 logits，以适应多选分类的形式
        reshaped_logits = tf.reshape(tensor=logits, shape=(-1, num_choices))
        # 计算损失，如果 labels 为 None，则损失为 None
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=reshaped_logits)

        # 如果不返回字典，则将结果组织为元组形式输出
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TFMultipleChoiceModelOutput 类型的对象，包括损失、logits、隐藏状态和注意力权重
        return TFMultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    # 检查模型是否已经构建，如果已构建，则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 标记模型已构建
        self.built = True
        # 如果存在 BERT 模型，则构建 BERT 模型
        if getattr(self, "bert", None) is not None:
            # 使用 TensorFlow 的命名空间为 BERT 模型命名，并构建它
            with tf.name_scope(self.bert.name):
                self.bert.build(None)
        # 如果存在分类器，则构建分类器
        if getattr(self, "classifier", None) is not None:
            # 使用 TensorFlow 的命名空间为分类器命名，并构建它
            with tf.name_scope(self.classifier.name):
                # 构建分类器，指定输入形状为 [None, None, self.config.hidden_size]
                self.classifier.build([None, None, self.config.hidden_size])
@add_start_docstrings(
    """
    Bert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    BERT_START_DOCSTRING,
)
class TFBertForTokenClassification(TFBertPreTrainedModel, TFTokenClassificationLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    # 定义在从 PyTorch 模型加载 TF 模型时授权的未预期/丢失的层名称列表
    _keys_to_ignore_on_load_unexpected = [
        r"pooler",
        r"mlm___cls",
        r"nsp___cls",
        r"cls.predictions",
        r"cls.seq_relationship",
    ]
    # 定义在从 PyTorch 模型加载 TF 模型时授权的缺失的层名称列表
    _keys_to_ignore_on_load_missing = [r"dropout"]

    def __init__(self, config: BertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels

        # 初始化 Bert 主层，不添加池化层，命名为 "bert"
        self.bert = TFBertMainLayer(config, add_pooling_layer=False, name="bert")
        # 确定分类器的 dropout 率，如果未指定，则使用配置中的隐藏层 dropout 率
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 添加 dropout 层
        self.dropout = tf.keras.layers.Dropout(rate=classifier_dropout)
        # 添加分类器层，输出单元数为标签数量，使用配置中的初始化范围初始化权重，命名为 "classifier"
        self.classifier = tf.keras.layers.Dense(
            units=config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="classifier",
        )
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_TOKEN_CLASSIFICATION,
        output_type=TFTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_TOKEN_CLASS_EXPECTED_OUTPUT,
        expected_loss=_TOKEN_CLASS_EXPECTED_LOSS,
    )
    # 定义模型前向传播函数，接受各种输入参数并返回模型输出
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
        # 将输入传递给 BERT 模型进行处理，并获取输出
        outputs = self.bert(
            input_ids=input_ids,                    # 输入的 token IDs
            attention_mask=attention_mask,          # 注意力掩码，用于控制模型关注哪些位置
            token_type_ids=token_type_ids,          # token 类型 IDs，用于区分不同句子的位置
            position_ids=position_ids,              # 位置 IDs，指示 token 的绝对位置
            head_mask=head_mask,                    # 头部掩码，用于控制注意力的分配
            inputs_embeds=inputs_embeds,            # 输入的嵌入向量
            output_attentions=output_attentions,    # 是否输出注意力权重
            output_hidden_states=output_hidden_states,  # 是否输出隐藏状态
            return_dict=return_dict,                # 是否以字典形式返回结果
            training=training,                      # 是否处于训练模式
        )
        # 从 BERT 输出中获取序列输出
        sequence_output = outputs[0]
        # 对序列输出进行 dropout 处理，用于防止过拟合
        sequence_output = self.dropout(inputs=sequence_output, training=training)
        # 将 dropout 后的输出传递给分类器，得到分类器的 logits
        logits = self.classifier(inputs=sequence_output)
        # 如果提供了标签，则计算损失；否则损失为 None
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        # 如果不要求以字典形式返回结果，则返回 logits 和其他可能的输出
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 以字典形式返回结果，包括损失、logits、隐藏状态和注意力权重
        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 构建模型
    def build(self, input_shape=None):
        # 如果模型已构建，则直接返回
        if self.built:
            return
        # 设置模型为已构建状态
        self.built = True
        # 如果存在 BERT 模型，则构建它
        if getattr(self, "bert", None) is not None:
            with tf.name_scope(self.bert.name):
                self.bert.build(None)
        # 如果存在分类器，则构建它
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
@add_start_docstrings(
    """
    Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    BERT_START_DOCSTRING,
)
class TFBertForQuestionAnswering(TFBertPreTrainedModel, TFQuestionAnsweringLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [
        r"pooler",
        r"mlm___cls",
        r"nsp___cls",
        r"cls.predictions",
        r"cls.seq_relationship",
    ]

    def __init__(self, config: BertConfig, *inputs, **kwargs):
        # 调用父类构造函数，传入配置和其他参数
        super().__init__(config, *inputs, **kwargs)

        # 保存标签的数量
        self.num_labels = config.num_labels

        # 创建 BERT 主层，设置不添加池化层，命名为 "bert"
        self.bert = TFBertMainLayer(config, add_pooling_layer=False, name="bert")
        # 创建用于回答问题的输出层，包含单位数量为标签数量的全连接层
        self.qa_outputs = tf.keras.layers.Dense(
            units=config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="qa_outputs",
        )
        # 保存配置
        self.config = config

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
    ):
        """
        BERT 模型的调用函数，接收多个输入参数并返回输出结果。

        Args:
            input_ids (TFModelInputType, optional): 输入序列的标识符. Defaults to None.
            attention_mask (np.ndarray or tf.Tensor, optional): 注意力掩码，掩盖不需要参与计算的位置. Defaults to None.
            token_type_ids (np.ndarray or tf.Tensor, optional): 标识不同句子的标签. Defaults to None.
            position_ids (np.ndarray or tf.Tensor, optional): 标识输入位置的标签. Defaults to None.
            head_mask (np.ndarray or tf.Tensor, optional): 头部掩码，控制哪些层的注意力应该被屏蔽. Defaults to None.
            inputs_embeds (np.ndarray or tf.Tensor, optional): 直接传入嵌入表示而不是输入 IDs. Defaults to None.
            output_attentions (bool, optional): 是否返回注意力权重. Defaults to None.
            output_hidden_states (bool, optional): 是否返回所有隐藏状态. Defaults to None.
            return_dict (bool, optional): 是否返回字典形式的输出结果. Defaults to None.
            start_positions (np.ndarray or tf.Tensor, optional): 开始位置的标签. Defaults to None.
            end_positions (np.ndarray or tf.Tensor, optional): 结束位置的标签. Defaults to None.
            training (bool, optional): 是否处于训练模式. Defaults to False.

        Returns:
            TFQuestionAnsweringModelOutput or Tuple[tf.Tensor], optional: 输出结果，可能是字典形式或元组形式的张量.
        """
        # 调用 BERT 主层，传入参数
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
        )

        # 取出 BERT 输出的隐藏状态
        sequence_output = outputs[0]

        # 传入隐藏状态到回答问题的输出层
        logits = self.qa_outputs(sequence_output)

        # 如果需要返回字典形式的输出结果
        if return_dict:
            # 构造输出字典
            return TFQuestionAnsweringModelOutput(
                logits=logits,
                hidden_states=outputs.hidden_states if output_hidden_states else None,
                attentions=outputs.attentions if output_attentions else None,
            )
        else:
            # 返回元组形式的输出张量
            return (logits,)
```  
    def call(self, input_ids: tf.Tensor, attention_mask: Optional[tf.Tensor] = None, token_type_ids: Optional[tf.Tensor] = None,
             position_ids: Optional[tf.Tensor] = None, head_mask: Optional[tf.Tensor] = None, inputs_embeds: Optional[tf.Tensor] = None,
             output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None,
             training: Optional[bool] = None, start_positions: Optional[tf.Tensor] = None, end_positions: Optional[tf.Tensor] = None
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
        # 使用BERT模型处理输入
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
        # 从BERT模型输出中提取序列输出
        sequence_output = outputs[0]
        # 将序列输出传递给QA输出层得到logits
        logits = self.qa_outputs(inputs=sequence_output)
        # 将logits分割成开始和结束位置的logits
        start_logits, end_logits = tf.split(value=logits, num_or_size_splits=2, axis=-1)
        # 去除多余的维度
        start_logits = tf.squeeze(input=start_logits, axis=-1)
        end_logits = tf.squeeze(input=end_logits, axis=-1)
        # 初始化损失值
        loss = None

        # 如果提供了开始和结束位置的标签，则计算损失
        if start_positions is not None and end_positions is not None:
            labels = {"start_position": start_positions}
            labels["end_position"] = end_positions
            loss = self.hf_compute_loss(labels=labels, logits=(start_logits, end_logits))

        # 如果不需要返回字典形式的输出，则构造输出并返回
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回TFQuestionAnsweringModelOutput对象
        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 构建BERT模型
        if getattr(self, "bert", None) is not None:
            with tf.name_scope(self.bert.name):
                self.bert.build(None)
        # 构建QA输出层
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.config.hidden_size])
```  
```