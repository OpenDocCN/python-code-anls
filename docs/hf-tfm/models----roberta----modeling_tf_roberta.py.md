# `.\transformers\models\roberta\modeling_tf_roberta.py`

```py
# coding=utf-8
# 设置脚本编码为 UTF-8

# 引入模块
# numpy - 数值计算库
# tensorflow - 机器学习库
# typing - 类型提示工具
# logging - 日志记录工具
import math
import warnings
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
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
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)

# 设置模块内日志记录器
logger = logging.get_logger(__name__)

# 设置示例模型的检查点名称、配置文件名
_CHECKPOINT_FOR_DOC = "roberta-base"
_CONFIG_FOR_DOC = "RobertaConfig"

# 设定可获取的包含模型的压缩包列表
TF_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "roberta-base",
    "roberta-large",
    "roberta-large-mnli",
    "distilroberta-base",
    # See all RoBERTa models at https://huggingface.co/models?filter=roberta
]


class TFRobertaEmbeddings(tf.keras.layers.Layer):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    # 继承自 Keras 层（Layer）

    def __init__(self, config, **kwargs):
        # 构造函数，初始化对象

        # 调用父类构造函数
        super().__init__(**kwargs)

        # 设置属性
        # 填充标记索引
        self.padding_idx = 1
        # 配置对象
        self.config = config
        # 隐藏尺寸
        self.hidden_size = config.hidden_size
        # 最大位置编码数
        self.max_position_embeddings = config.max_position_embeddings
        # 初始化范围
        self.initializer_range = config.initializer_range
        # 层归一化层
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 丢弃层
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
    # 构建模型，设置嵌入层参数
    def build(self, input_shape=None):
        # 设置词嵌入层参数
        with tf.name_scope("word_embeddings"):
            self.weight = self.add_weight(
                name="weight",
                shape=[self.config.vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        # 设置 token 类型嵌入层参数
        with tf.name_scope("token_type_embeddings"):
            self.token_type_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.config.type_vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        # 设置位置嵌入层参数
        with tf.name_scope("position_embeddings"):
            self.position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_position_embeddings, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        # 如果模型已构建，则直接返回
        if self.built:
            return
        self.built = True
        # 如果 LayerNorm 存在，则构建 LayerNorm 层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])

    # 根据输入的 tokens ID，生成对应的位置 ID
    def create_position_ids_from_input_ids(self, input_ids, past_key_values_length=0):
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding
        symbols are ignored. This is modified from fairseq's `utils.make_positions`.

        Args:
            input_ids: tf.Tensor
        Returns: tf.Tensor
        """
        # 创建一个掩码矩阵，标记非填充符号
        mask = tf.cast(tf.math.not_equal(input_ids, self.padding_idx), dtype=input_ids.dtype)
        # 计算累积位置索引
        incremental_indices = (tf.math.cumsum(mask, axis=1) + past_key_values_length) * mask

        return incremental_indices + self.padding_idx

    # 模型调用函数，实现前向传播过程
    def call(
        self,
        input_ids=None,
        position_ids=None,
        token_type_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
        training=False,
    # 应用基于输入张量的嵌入
    def apply_embedding(
        self,
        input_ids=None,
        inputs_embeds=None,
        token_type_ids=None,
        position_ids=None,
        past_key_values_length=0,
        training=False,
    ):
        """
        Applies embedding based on inputs tensor.
    
        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        # 确保输入了 input_ids 或 inputs_embeds 之一
        assert not (input_ids is None and inputs_embeds is None)
    
        # 如果输入了 input_ids，则根据 input_ids 获取嵌入
        if input_ids is not None:
            # 检查 input_ids 是否在词汇表大小范围内
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)
    
        # 获取输入嵌入的形状
        input_shape = shape_list(inputs_embeds)[:-1]
    
        # 如果没有输入 token_type_ids，则填充为 0
        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)
    
        # 如果没有输入 position_ids，则根据 input_ids 或者默认值创建
        if position_ids is None:
            if input_ids is not None:
                # 根据 input_ids 创建 position_ids，填充 padding 部分
                position_ids = self.create_position_ids_from_input_ids(
                    input_ids=input_ids, past_key_values_length=past_key_values_length
                )
            else:
                # 创建默认的 position_ids
                position_ids = tf.expand_dims(
                    tf.range(start=self.padding_idx + 1, limit=input_shape[-1] + self.padding_idx + 1), axis=0
                )
    
        # 获取位置嵌入和 token 类型嵌入
        position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
        token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)
    
        # 将输入嵌入、位置嵌入和 token 类型嵌入相加得到最终嵌入
        final_embeddings = inputs_embeds + position_embeds + token_type_embeds
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)
    
        return final_embeddings
# 从transformers.models.bert.modeling_tf_bert.TFBertPooler复制代码，并将Bert->Roberta
class TFRobertaPooler(tf.keras.layers.Layer):
    def __init__(self, config: RobertaConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，用于池化操作，输出大小为config.hidden_size
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 通过选择第一个token对应的hidden state来"池化"模型
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(inputs=first_token_tensor)

        return pooled_output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# 从transformers.models.bert.modeling_tf_bert.TFBertSelfAttention复制代码，并将Bert->Roberta
class TFRobertaSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config: RobertaConfig, **kwargs):
        super().__init__(**kwargs)

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

        # 创建用于查询、键和值的全连接层，并设定初始化方式
        self.query = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        self.key = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key"
        )
        self.value = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )
        # 基于配置中的dropout比例创建dropout层
        self.dropout = tf.keras.layers.Dropout(rate=config.attention_probs_dropout_prob)

        self.is_decoder = config.is_decoder
        self.config = config
    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        # 将张量从[batch_size, seq_length, all_head_size]重塑为[batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))

        # 将张量从[batch_size, seq_length, num_attention_heads, attention_head_size]转置为[batch_size, num_attention_heads, seq_length, attention_head_size]
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
        # 模型调用函数
        if self.built:
            return
        # 设置已构建标志
        self.built = True
        # 如果存在查询键值权重，则构建查询权重
        if getattr(self, "query", None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.config.hidden_size])
        # 如果存在查询键值权重，则构建键权重
        if getattr(self, "key", None) is not None:
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.config.hidden_size])
        # 如果存在查询键值权重，则构建值权重
        if getattr(self, "value", None) is not None:
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.config.hidden_size])
```  
# 创建一个名为TFRobertaSelfOutput的自定义层，继承自tf.keras.layers.Layer类
class TFRobertaSelfOutput(tf.keras.layers.Layer):
    # 初始化函数，接收一个RobertaConfig类型的参数config
    def __init__(self, config: RobertaConfig, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 创建一个全连接层，units为config.hidden_size，使用config.initializer_range作为权重初始化方法，名称为"dense"
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建一个LayerNormalization层，epsilon为config.layer_norm_eps，名称为"LayerNorm"
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建一个Dropout层，rate为config.hidden_dropout_prob，用于添加随机失活，丢弃部分神经元
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        # 设置一个属性config，值为传入的config参数
        self.config = config

    # 调用函数，接收两个张量hidden_states和input_tensor，还有一个布尔类型的参数training，返回一个张量
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 将hidden_states通过全连接层进行线性变换
        hidden_states = self.dense(inputs=hidden_states)
        # 在训练模式下，对hidden_states进行dropout操作
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 将hidden_states和input_tensor相加，并通过LayerNormalization层进行归一化
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states

    # 构建函数，用于构建该层相关的参数
    def build(self, input_shape=None):
        # 如果已经构建完成，则直接返回
        if self.built:
            return
        # 设置已构建标志为True
        self.built = True
        # 对dense层进行构建
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 对LayerNorm层进行构建
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


# 创建一个名为TFRobertaAttention的自定义层，继承自tf.keras.layers.Layer类
class TFRobertaAttention(tf.keras.layers.Layer):
    # 初始化函数，接收一个RobertaConfig类型的参数config
    def __init__(self, config: RobertaConfig, **kwargs):
        # 调用父类的初始化函��
        super().__init__(**kwargs)

        # 创建一个TFRobertaSelfAttention层，使用config作为参数，名称为"self"
        self.self_attention = TFRobertaSelfAttention(config, name="self")
        # 创建一个TFRobertaSelfOutput层，使用config作为参数，名称为"output"
        self.dense_output = TFRobertaSelfOutput(config, name="output")

    # 剪枝函数，用于剪枝不需要的head
    def prune_heads(self, heads):
        raise NotImplementedError

    # 调用函数，接收多个张量参数和布尔类型的参数training，返回一个元组
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
        # 调用self_attention层进行attention计算
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
        # 经过dense_output层处理得到attention_output
        attention_output = self.dense_output(
            hidden_states=self_outputs[0], input_tensor=input_tensor, training=training
        )
        # 如果需要输出attention，将attention_output与self_outputs[1:]合并成一个元组并返回，否则只返回attention_output
        outputs = (attention_output,) + self_outputs[1:]

        return outputs
    # 构建函数，用于建立模型层
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在 self_attention 属性，则构建自注意力层
        if getattr(self, "self_attention", None) is not None:
            # 在 TensorFlow 中使用 name_scope 为变量作用域命名
            with tf.name_scope(self.self_attention.name):
                # 构建自注意力层
                self.self_attention.build(None)
        # 如果存在 dense_output 属性，则构建密集输出层
        if getattr(self, "dense_output", None) is not None:
            # 在 TensorFlow 中使用 name_scope 为变量作用域命名
            with tf.name_scope(self.dense_output.name):
                # 构建密集输出层
                self.dense_output.build(None)
# 从 transformers.models.bert.modeling_tf_bert.TFBertIntermediate 复制并替换 Bert 为 Roberta
class TFRobertaIntermediate(tf.keras.layers.Layer):
    def __init__(self, config: RobertaConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，units 参数为中间层的大小，kernel_initializer 根据配置获取初始化器，命名为 dense
        self.dense = tf.keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        # 根据配置中的隐藏激活函数创建激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        self.config = config

    # 前向传播函数，输入为隐藏状态张量，输出为经过全连接层和激活函数处理后的隐藏状态张量
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

    # 构建函数，如果已构建则跳过，通过全连接层创建模型
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# 从 transformers.models.bert.modeling_tf_bert.TFBertOutput 复制并替换 Bert 为 Roberta
class TFRobertaOutput(tf.keras.layers.Layer):
    def __init__(self, config: RobertaConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，units 参数为隐藏状态大小，kernel_initializer 根据配置获取初始化器，命名为 dense
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建一个层归一化层，epsilon 参数为层标准化的 eps 值，命名为 LayerNorm
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建一个 dropout 层，rate 参数为隐藏层的 dropout 比例
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    # 前向传播函数，输入为隐藏状态张量、输入张量和训练模式标志，输出为经过全连接、dropout 和层归一化处理后的隐藏状态张量
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states

    # 构建函数，如果已构建则跳过，通过全连接、层归一化层创建模型
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


# 从 transformers.models.bert.modeling_tf_bert.TFBertLayer 复制并替换 Bert 为 Roberta
class TFRobertaLayer(tf.keras.layers.Layer):
    def __init__(self, config: RobertaConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建 TFRobertaAttention 实例，用于处理自注意力机制
        self.attention = TFRobertaAttention(config, name="attention")

        # 配置是否为解码器
        self.is_decoder = config.is_decoder

        # 配置是否添加交叉注意力
        self.add_cross_attention = config.add_cross_attention

        # 如果添加了交叉注意力，且当前模型不是解码器，则抛出异常
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")

            # 创建 TFRobertaAttention 实例，用于处理交叉注意力机制
            self.crossattention = TFRobertaAttention(config, name="crossattention")

        # 创建 TFRobertaIntermediate 实例，用于处理中间层的前向传播
        self.intermediate = TFRobertaIntermediate(config, name="intermediate")

        # 创建 TFRobertaOutput 实例，用于处理 BERT 模型的输出
        self.bert_output = TFRobertaOutput(config, name="output")

    # 模型的前向传播过程
    def call(
        self,
        hidden_states: tf.Tensor,  # 输入的隐藏状态张量
        attention_mask: tf.Tensor,  # 注意力掩码张量
        head_mask: tf.Tensor,  # 头部掩码张量
        encoder_hidden_states: tf.Tensor | None,  # 编码器的隐藏状态张量
        encoder_attention_mask: tf.Tensor | None,  # 编码器的注意力掩码张量
        past_key_value: Tuple[tf.Tensor] | None,  # 过去的键-值张量元组
        output_attentions: bool,  # 是否输出注意力权重张量
        training: bool = False,  # 是否处于训练模式，默认为False
    # 定义函数的输入参数和返回值类型
    ) -> Tuple[tf.Tensor]:
        # 如果过去的键/值非空，则self-attention缓存键/值元组位于位置1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 进行self-attention计算
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

        # 如果是解码器，最后一个输出是self-attn缓存的元组
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # 如果我们要输出注意力权重，则添加self注意力权重

        cross_attn_present_key_value = None
        # 如果是解码器并且传入了encoder隐藏状态
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn的缓存键/值元组位于过去键/值元组的位置3,4
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 进行cross-attention计算
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
            # 如果我们要输出注意力权重，则添加cross attentions
            outputs = outputs + cross_attention_outputs[1:-1]

            # 将cross-attn缓存添加到present_key_value元组的位置3,4
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        intermediate_output = self.intermediate(hidden_states=attention_output)
        layer_output = self.bert_output(
            hidden_states=intermediate_output, input_tensor=attention_output, training=training
        )
        outputs = (layer_output,) + outputs  # 如果我们要输出注意力，则将其添加

        # 如果是解码器，则将注意力键/值作为最后一个输出返回
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs
    # 构建当前层
    def build(self, input_shape=None):
        # 如果层已经构建好，直接返回
        if self.built:
            return
        # 标记当前层已经构建好
        self.built = True
        # 如果存在注意力层，构建注意力层
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        # 如果存在中间层，构建中间层
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        # 如果存在输出层，构建输出层
        if getattr(self, "bert_output", None) is not None:
            with tf.name_scope(self.bert_output.name):
                self.bert_output.build(None)
        # 如果存在交叉注意力层，构建交叉注意力层
        if getattr(self, "crossattention", None) is not None:
            with tf.name_scope(self.crossattention.name):
                self.crossattention.build(None)
# 从 transformers.models.bert.modeling_tf_bert.TFBertEncoder 复制而来，将 Bert 替换为 Roberta
class TFRobertaEncoder(tf.keras.layers.Layer):
    def __init__(self, config: RobertaConfig, **kwargs):
        super().__init__(**kwargs)
        # 初始化 RobretaEncoder 层，传入配置
        self.config = config
        # 创建多个 RobertaLayer 实例组成的列表
        self.layer = [TFRobertaLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]

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
        # 如果需要输出注意力权重，则初始化空元组
        all_attentions = () if output_attentions else None
        # 如果需要输出跨层注意力权重且配置允许，则初始化空元组
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        # 如果使用缓存，则初始化空元组
        next_decoder_cache = () if use_cache else None
        
        # 遍历每个层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 如果存在过去的键值，则获取当前层的过去键值
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 调用当前层的 call 方法，计算当前层的输出
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

            # 如果使用缓存，则将当前层的输出添加到下一个解码器缓存中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            # 如果需要输出注意力权重，则将当前层的注意力权重添加到 all_attentions 中
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
                # 如果需要输出跨层注意力权重且有编码器隐藏状态，则将当前层的跨层注意力权重添加到 all_cross_attentions 中
                if self.config.add_cross_attention and encoder_hidden_states is not None:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 添加最后一层的隐藏状态到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典，则返回非空元素的元组
        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_attentions, all_cross_attentions] if v is not None
            )

        # 返回字典形式的模型输出
        return TFBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )
```py  
    # 构建模型
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记已经构建完成
        self.built = True
        # 如果存在子层
        if getattr(self, "layer", None) is not None:
            # 遍历每个子层
            for layer in self.layer:
                # 使用子层的名称创建命名空间
                with tf.name_scope(layer.name):
                    # 构建子层
                    layer.build(None)
# 使用 keras_serializable 装饰器标记该类，使其可以被序列化
@keras_serializable
class TFRobertaMainLayer(tf.keras.layers.Layer):
    # 将一个类属性指向 RobertaConfig 类
    config_class = RobertaConfig

    # 初始化方法，接受配置参数和是否添加 pooling 层的标志
    def __init__(self, config, add_pooling_layer=True, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 将传入的配置参数赋值给 self.config
        self.config = config
        # 判断是否为解码器
        self.is_decoder = config.is_decoder

        # 读取配置参数中的一些属性值
        self.num_hidden_layers = config.num_hidden_layers
        self.initializer_range = config.initializer_range
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.return_dict = config.use_return_dict
        # 创建 TFRobertaEncoder 对象，名为 encoder
        self.encoder = TFRobertaEncoder(config, name="encoder")
        # 如果需要添加 pooling 层，则创建 TFRobertaPooler 对象，名为 pooler
        self.pooler = TFRobertaPooler(config, name="pooler") if add_pooling_layer else None
        # 创建 TFRobertaEmbeddings 对象，名为 embeddings，必须在最后声明以保持权重顺序
        self.embeddings = TFRobertaEmbeddings(config, name="embeddings")

    # 获取输入嵌入层
    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        return self.embeddings

    # 设置输入嵌入层的权重和词汇大小
    def set_input_embeddings(self, value: tf.Variable):
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    # 剪枝模型的某些头
    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError

    # 调用模型，接受多种输入参数
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
    # 如果模型已经构建完成，直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 设置模型已构建完成的标志
        self.built = True
        # 检查是否有 encoder 层，如果有则构建 encoder 层
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 检查是否有 pooler 层，如果有则构建 pooler 层  
        if getattr(self, "pooler", None) is not None:
            with tf.name_scope(self.pooler.name):
                self.pooler.build(None)
        # 检查是否有 embeddings 层，如果有则构建 embeddings 层
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
# 定义一个继承自TFPreTrainedModel的抽象类，用于处理权重初始化和简单的接口用于下载和加载预训练模型
class TFRobertaPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 配置类为RobertaConfig
    config_class = RobertaConfig
    # 基础模型前缀为"roberta"


# ROBERTA_START_DOCSTRING是用于生成文档的字符串
ROBERTA_START_DOCSTRING = r"""

    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
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

    Parameters:
        config ([`RobertaConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# ROBERTA_INPUTS_DOCSTRING是用于生成文档的字符串
ROBERTA_INPUTS_DOCSTRING = r"""
"""


# 添加注释到TFRobertaModel类
@add_start_docstrings(
    "The bare RoBERTa Model transformer outputting raw hidden-states without any specific head on top.",
    ROBERTA_START_DOCSTRING,
)
class TFRobertaModel(TFRobertaPreTrainedModel):
    # 定义一个类，继承自父类
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)
        # 创建一个TFRobertaMainLayer对象，命名为"roberta"
        self.roberta = TFRobertaMainLayer(config, name="roberta")
    
    # 标记unpack_inputs装饰器开始，并将以下的函数称为unpack_inputs的控制函数
    # 标记add_start_docstrings_to_model_forward装饰器开始，并将以下的函数称为add_start_docstrings_to_model_forward的控制函数
    # 标记add_code_sample_docstrings装饰器开始，并将以下的函数称为add_code_sample_docstrings的控制函数
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
    )：
    # 定义一个函数，输入参数包括input_ids、attention_mask、token_type_ids、position_ids、head_mask、inputs_embeds、encoder_hidden_states、encoder_attention_mask、past_key_values、use_cache、output_attentions、output_hidden_states、return_dict和training等参数，返回类型为Tuple或TFBaseModelOutputWithPoolingAndCrossAttentions
    def call(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                 inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_values=None,
                 use_cache=True, output_attentions=False, output_hidden_states=False, return_dict=True, training=False) -> Union[Tuple, TFBaseModelOutputWithPoolingAndCrossAttentions]:
    
            # 调用RoBERTa模型的函数，传入各种输入参数
            outputs = self.roberta(
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
    
            # 返回RoBERTa模型的输出
            return outputs
    
        # 构建模型
        def build(self, input_shape=None):
            # 如果已经构建过，则直接返回
            if self.built:
                return
            # 将标记设置为已构建
            self.built = True
            # 如果RoBERTa模型已存在
            if getattr(self, "roberta", None) is not None:
                # 在命名范围内构建RoBERTa模型
                with tf.name_scope(self.roberta.name):
                    self.roberta.build(None)
class TFRobertaLMHead(tf.keras.layers.Layer):
    """Roberta Head for masked language modeling."""  # 用于掩码语言建模的RoBERTa头部

    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.hidden_size = config.hidden_size
        # 密集层，将输入映射到与隐藏层大小相同的空间
        self.dense = tf.keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 层归一化，对密集层输出进行归一化处理
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        # GELU激活函数
        self.act = get_tf_activation("gelu")

        # 输出权重与输入嵌入相同，但每个标记都有一个仅输出的偏置
        self.decoder = input_embeddings

    def build(self, input_shape=None):
        # 创建一个形状为(config.vocab_size,)的偏置张量
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="bias")

        if self.built:
            return
    # 初始化方法，接受一个配置和多个输入，并调用父类的初始化方法
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 初始化 RoBERTa 模型的主层
        self.roberta = TFRobertaMainLayer(config, add_pooling_layer=False, name="roberta")
        # 初始化 RoBERTa 模型的语言模型头部
        self.lm_head = TFRobertaLMHead(config, self.roberta.embeddings, name="lm_head")

    # 获取语言模型头部
    def get_lm_head(self):
        return self.lm_head

    # 获取前缀偏置名称
    def get_prefix_bias_name(self):
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        return self.name + "/" + self.lm_head.name

    # 调用方法，用于模型的前向推断
    @unpack_inputs
    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFMaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="<mask>",
        expected_output="' Paris'",
        expected_loss=0.1,
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
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        # 通过 RoBERTa 模型获取输出
        outputs = self.roberta(
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
        # 使用语言模型头部生成预测分数
        prediction_scores = self.lm_head(sequence_output)

        # 如果存在标签，则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, prediction_scores)

        # 如果不返回字典，则返回输出结果
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果返回字典，则返回 TFMaskedLMOutput 对象
        return TFMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    # 构建模型
    def build(self, input_shape=None):
        # 如果模型已经被构建过了，直接返回
        if self.built:
            return
        # 将 built 标志设置为 True，表示模型已经被构建
        self.built = True
        # 如果模型中有 roberta 属性，则构建 roberta 子模型
        if getattr(self, "roberta", None) is not None:
            # 使用 roberta 的名称创建一个 TensorFlow 命名空间
            with tf.name_scope(self.roberta.name):
                self.roberta.build(None)
        # 如果模型中有 lm_head 属性，则构建 lm_head 子模型
        if getattr(self, "lm_head", None) is not None:
            # 使用 lm_head 的名称创建一个 TensorFlow 命名空间
            with tf.name_scope(self.lm_head.name):
                self.lm_head.build(None)
class TFRobertaForCausalLM(TFRobertaPreTrainedModel, TFCausalLanguageModelingLoss):
    # 定义具有因果语言模型损失的 TFRobertaForCausalLM 类，继承自 TFRobertaPreTrainedModel 和 TFCausalLanguageModelingLoss
    ### names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"lm_head.decoder.weight"]
    # 设置加载 TF 模型时允许出现的意外/丢失的层的名称列表

    def __init__(self, config: RobertaConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 初始化函数，接受 RobertaConfig 类型的 config 参数以及其他参数

        if not config.is_decoder:
            logger.warning("If you want to use `TFRobertaLMHeadModel` as a standalone, add `is_decoder=True.`")
        # 如果 config 不是解码器，发出警告信息

        self.roberta = TFRobertaMainLayer(config, add_pooling_layer=False, name="roberta")
        # 初始化 TFRobertaMainLayer 对象，作为 roberta 属性
        self.lm_head = TFRobertaLMHead(config, input_embeddings=self.roberta.embeddings, name="lm_head")
        # 初始化 TFRobertaLMHead 对象，作为 lm_head 属性

    def get_lm_head(self):
        return self.lm_head
    # 返回 lm_head 属性

    def get_prefix_bias_name(self):
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        return self.name + "/" + self.lm_head.name
    # 返回 bias 名称的前缀

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        # 获取输入 input_ids 的形状
        if attention_mask is None:
            attention_mask = tf.ones(input_shape)
        # 如果注意力掩码为空，则设置为形状为 input_shape 的全 1 掩码

        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        # 如果 past_key_values 不为空，则截取 input_ids 的最后一项

        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}
        # 返回字典，包含 input_ids、attention_mask 和 past_key_values

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
    # 构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回，避免重复构建
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果模型包含名为 "roberta" 的属性且不为 None，则构建 "roberta" 子模型
        if getattr(self, "roberta", None) is not None:
            # 使用命名空间来指定模型结构
            with tf.name_scope(self.roberta.name):
                # 构建 "roberta" 子模型，输入形状为 None 表示输入形状未知
                self.roberta.build(None)
        # 如果模型包含名为 "lm_head" 的属性且不为 None，则构建 "lm_head" 子模型
        if getattr(self, "lm_head", None) is not None:
            # 使用命名空间来指定模型结构
            with tf.name_scope(self.lm_head.name):
                # 构建 "lm_head" 子模型，输入形状为 None 表示输入形状未知
                self.lm_head.build(None)
class TFRobertaClassificationHead(tf.keras.layers.Layer):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 创建一个全连接层，用于处理隐藏层的输出
        self.dense = tf.keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )
        # 根据配置参数决定是否使用分类器的 dropout 操作
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = tf.keras.layers.Dropout(classifier_dropout)  # 添加 dropout
        # 创建输出层，用于处理分类任务
        self.out_proj = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="out_proj"
        )
        self.config = config

    # 定义模型的前向传播逻辑
    def call(self, features, training=False):
        x = features[:, 0, :]  # 提取第一个位置的特征（等同于 [CLS] 标记）
        x = self.dropout(x, training=training)  # 应用 dropout
        x = self.dense(x)  # 全连接层处理
        x = self.dropout(x, training=training)  # 再次应用 dropout
        x = self.out_proj(x)  # 输出层处理
        return x

    # 构建模型，处理模型的输入形状
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在全连接层，则构建该层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 如果存在输出层，则构建该层
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build([None, None, self.config.hidden_size])


@add_start_docstrings(
    """
    RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    ROBERTA_START_DOCSTRING,
)
class TFRobertaForSequenceClassification(TFRobertaPreTrainedModel, TFSequenceClassificationLoss):
    # 加载 TF 模型时忽略的层的命名
    # 带有 '.' 的名称表示在加载 PT 模型时被授权的未预期/丢失的层
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"lm_head"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        # 使用 RoBERTa 的主层，不添加��化层
        self.roberta = TFRobertaMainLayer(config, add_pooling_layer=False, name="roberta")
        # 定义分类头部，用于执行分类任务
        self.classifier = TFRobertaClassificationHead(config, name="classifier")

    # 模型前向传播逻辑的注解
    @unpack_inputs
    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="cardiffnlp/twitter-roberta-base-emotion",
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'optimism'",
        expected_loss=0.08,
    )
    # 定义一个方法，用于调用模型
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
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`.
            If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 调用预训练的RoBERTa模型，获取输出
        outputs = self.roberta(
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
        # 获取模型输出序列
        sequence_output = outputs[0]
        # 对序列输出进行分类
        logits = self.classifier(sequence_output, training=training)
        
        # 如果有标签，则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果不需要返回字典形式的结果，返回损失和输出
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        # 返回字典形式的输出结果
        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 构建模型
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        self.built = True
        # 构建 RoBERTa 模型
        if getattr(self, "roberta", None) is not None:
            with tf.name_scope(self.roberta.name):
                self.roberta.build(None)
        # 构建分类器
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build(None)
# 定义了一个带有多选题分类头的 Roberta 模型，该头部包括一个线性层和 softmax 激活函数，用于处理类似于 RocStories/SWAG 任务
# 这里用到了 ROBERTA_START_DOCSTRING 中的文档字符串
class TFRobertaForMultipleChoice(TFRobertaPreTrainedModel, TFMultipleChoiceLoss):
    # 在加载 TF 模型时，'.' 开头的名称表示在 PT 模型中未授权的异常/缺失层
    _keys_to_ignore_on_load_unexpected = [r"lm_head"]
    _keys_to_ignore_on_load_missing = [r"dropout"]

    # 初始化方法，接受 config 参数并传递给父类进行初始化
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 创建 Roberta 主层，使用 config 初始化，并命名为 "roberta"
        self.roberta = TFRobertaMainLayer(config, name="roberta")
        # 创建 Dropout 层，使用 config 中的隐藏层 dropout 概率
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        # 创建 Dense 层，用于分类任务，输出维度为 1，使用指定 initializer 初始化，命名为 "classifier"
        self.classifier = tf.keras.layers.Dense(
            1, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        # 保存 config 参数
        self.config = config

    # 使用装饰器指定输入参数和输出类型的文档注释
    # 使用装饰器添加代码示例文档注释
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
    # 定义一个用于多项选择任务的损失函数和输出计算的函数
    def call(
        self,
        input_ids: Optional[tf.Tensor] = None,
        attention_mask: Optional[tf.Tensor] = None,
        token_type_ids: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        head_mask: Optional[tf.Tensor] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[tf.Tensor] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFMultipleChoiceModelOutput, Tuple[tf.Tensor]]:
        # 根据输入的 input_ids 或 inputs_embeds 计算输入序列的长度和选择数量
        if input_ids is not None:
            num_choices = shape_list(input_ids)[1]
            seq_length = shape_list(input_ids)[2]
        else:
            num_choices = shape_list(inputs_embeds)[1]
            seq_length = shape_list(inputs_embeds)[2]
    
        # 将输入拉平为二维tensor
        flat_input_ids = tf.reshape(input_ids, (-1, seq_length)) if input_ids is not None else None
        flat_attention_mask = tf.reshape(attention_mask, (-1, seq_length)) if attention_mask is not None else None
        flat_token_type_ids = tf.reshape(token_type_ids, (-1, seq_length)) if token_type_ids is not None else None
        flat_position_ids = tf.reshape(position_ids, (-1, seq_length)) if position_ids is not None else None
    
        # 通过 RoBERTa 模型获得输出
        outputs = self.roberta(
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
    
        # 对输出进行处理并计算分类逻辑输出
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output, training=training)
        logits = self.classifier(pooled_output)
        reshaped_logits = tf.reshape(logits, (-1, num_choices))
    
        # 计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, reshaped_logits)
    
        # 根据 return_dict 返回不同格式的输出
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
    
        return TFMultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    # 构建模型
    def build(self, input_shape=None):
        # 如果模型已构建则直接返回
        if self.built:
            return
        self.built = True
    
        # 构建 RoBERTa 和分类器部分
        if getattr(self, "roberta", None) is not None:
            with tf.name_scope(self.roberta.name):
                self.roberta.build(None)
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
# 使用指定的文档字符串初始化一个 RoBERTa 模型，该模型在隐藏状态的输出之上有一个标记分类头部（线性层），例如用于命名实体识别（NER）任务
class TFRobertaForTokenClassification(TFRobertaPreTrainedModel, TFTokenClassificationLoss):
    # 当从 PT 模型加载 TF 模型时，带有 '.' 的名称表示授权的意外/丢失的层
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"lm_head"]
    # 当从 PT 模型加载 TF 模型时，缺少的层的名称列表
    _keys_to_ignore_on_load_missing = [r"dropout"]

    # 初始化函数，接受配置对象以及任意数量的输入和关键字参数
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化函数
        super().__init__(config, *inputs, **kwargs)
        # 标签数量等于配置对象中的标签数量
        self.num_labels = config.num_labels

        # 初始化 RoBERTa 主层，不添加池化层，设置名称为 "roberta"
        self.roberta = TFRobertaMainLayer(config, add_pooling_layer=False, name="roberta")
        # 分类器的 dropout 等于配置对象中的分类器 dropout，如果为空则使用配置对象中的隐藏层 dropout
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 添加一个 dropout 层
        self.dropout = tf.keras.layers.Dropout(classifier_dropout)
        # 添加一个全连接层作为分类器
        self.classifier = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        # 保存配置对象
        self.config = config

    # 用于模型正向传播的函数，接受多种输入和关键字参数
    @unpack_inputs
    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="ydshieh/roberta-large-ner-english",
        output_type=TFTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="['O', 'ORG', 'ORG', 'O', 'O', 'O', 'O', 'O', 'LOC', 'O', 'LOC', 'LOC']",
        expected_loss=0.01,
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
    def call(self, input_ids, attention_mask=None, token_type_ids=None,
             position_ids=None, head_mask=None, inputs_embeds=None,
             labels=None, output_attentions=None, output_hidden_states=None,
             return_dict=None, training=False) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 使用输入参数调用 RoBERTa 模型得到输出
        outputs = self.roberta(
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
        # 获取 RoBERTa 模型的输出序列
        sequence_output = outputs[0]

        # 对输出序列进行 dropout 操作
        sequence_output = self.dropout(sequence_output, training=training)
        # 经过分类器得到预测的 logits
        logits = self.classifier(sequence_output)

        # 如果有标签，则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果不要返回字典形式的输出，则返回 logits 和额外信息
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果要返回字典形式的输出，包含 loss、logits、隐藏层和注意力
        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        # 设置标志为已经构建
        self.built = True
        # 如果存在 RoBERTa 模型，则构建 RoBERTa 模型
        if getattr(self, "roberta", None) is not None:
            with tf.name_scope(self.roberta.name):
                self.roberta.build(None)
        # 如果存在分类器模型，则构建分类器模型
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
# 为在提取问答任务（如SQuAD）上具有一个以span分类头部为主的RoBERTa模型添加文档字符串，文档字符串描述了模型结构和任务用途
class TFRobertaForQuestionAnswering(TFRobertaPreTrainedModel, TFQuestionAnsweringLoss):
    # 加载时忽略的层的键名列表，用于从PT模型加载TF模型时遇到授权的意外/缺失层
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"lm_head"]

    # 初始化函数定义
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化函数
        super().__init__(config, *inputs, **kwargs)
        # 从配置中获取标签数量
        self.num_labels = config.num_labels

        # 创建RoBERTa主体层，不添加汇聚层，用于提取 具名任务处理
        self.roberta = TFRobertaMainLayer(config, add_pooling_layer=False, name="roberta")
        # 创建一个Dense输出层，用于预测答案的起点和终点位置
        self.qa_outputs = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )
        # 保存配置
        self.config = config

    # 模型调用函数，使用装饰器添加文档字符串，描述模型输入和输出格式，以及示例代码
    @unpack_inputs
    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="ydshieh/roberta-base-squad2",
        output_type=TFQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="' puppet'",
        expected_loss=0.86,
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
        start_positions (`tf.Tensor` of shape `(batch_size,)`, *optional`):
            Labels for the position (index) of the start of the labeled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Positions outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`tf.Tensor` of shape `(batch_size,)`, *optional`):
            Labels for the position (index) of the end of the labeled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Positions outside of the sequence
            are not taken into account for computing the loss.
        """


        # 用输入参数调用 Transformer model，返回一个包含各种输出的对象
        outputs = self.roberta(
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
        # 从输出对象中取出序列输出
        sequence_output = outputs[0]

        # 用序列输出计算答案的 logit
        logits = self.qa_outputs(sequence_output)
        # 将 logit 分成开始和结束两部分
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        # 降维，去除多余的维度
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        loss = None
        # 如果给定了开始和结束位置的标签，则计算损失
        if start_positions is not None and end_positions is not None:
            labels = {"start_position": start_positions}
            labels["end_position"] = end_positions
            loss = self.hf_compute_loss(labels, (start_logits, end_logits))

        # 如果不需要返回一个字典，则直接返回输出
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 构建 TFQuestionAnsweringModelOutput 对象返回
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
        if getattr(self, "roberta", None) is not None:
            with tf.name_scope(self.roberta.name):
                self.roberta.build(None)
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.config.hidden_size])
```