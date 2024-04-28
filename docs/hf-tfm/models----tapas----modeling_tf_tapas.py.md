# `.\transformers\models\tapas\modeling_tf_tapas.py`

```py
# 设置文件编码格式为 utf-8
# 版权声明
# 2021年版权属于Google Research和HuggingFace Inc.团队
# 根据Apache License 2.0许可
# 除非符合许可，否则不得使用此文件
# 您可以在以下网址获取许可的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按"原样"基础分布的软件
# 没有任何形式的保证或条件，无论是明示还是暗示
# 查看许可协议以获取具体语言的权限和限制
"""TF 2.0 TAPAS 模型。"""


from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithPastAndCrossAttentions,
    TFBaseModelOutputWithPooling,
    TFMaskedLMOutput,
    TFSequenceClassifierOutput,
)
from ...modeling_tf_utils import (
    TFMaskedLanguageModelingLoss,
    TFModelInputType,
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    get_initializer,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_tensorflow_probability_available,
    logging,
    replace_return_docstrings,
    requires_backends,
)
from .configuration_tapas import TapasConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# soft dependency
# 检查依赖关系: tensorflow_probability是否可用
if is_tensorflow_probability_available():
    try:
        import tensorflow_probability as tfp
        # 在第一次调用时检查是否安装了兼容的版本的TensorFlow
        # TensorFlow Probability 依赖于最新稳定版本的TensorFlow
        n = tfp.distributions.Normal(loc=0.0, scale=1.0)
    except ImportError:
        # 引发错误提示，指出无法加载tensorflow_probability，提供重新安装的指南链接
        logger.error(
            "TAPAS models are not usable since `tensorflow_probability` can't be loaded. "
            "It seems you have `tensorflow_probability` installed with the wrong tensorflow version. "
            "Please try to reinstall it following the instructions here: https://github.com/tensorflow/probability."
        )

# 文档中显示的配置和检查点
_CONFIG_FOR_DOC = "TapasConfig"
_CHECKPOINT_FOR_DOC = "google/tapas-base"

# TF_TAPAS_PRETRAINED_MODEL_ARCHIVE_LIST列表
# 大型模型
# 基于google/tapas-large的预训练模型
TF_TAPAS_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/tapas-large",
    "google/tapas-large-finetuned-sqa",
    "google/tapas-large-finetuned-wtq",
    "google/tapas-large-finetuned-wikisql-supervised",
    "google/tapas-large-finetuned-tabfact",
    # 基础模型
    "google/tapas-base",
    "google/tapas-base-finetuned-sqa",
    "google/tapas-base-finetuned-wtq",
    "google/tapas-base-finetuned-wikisql-supervised",
    "google/tapas-base-finetuned-tabfact",
    # 小型模型
    "google/tapas-small",
    # 添加模型名称到 Hugging Face 模型库
    "google/tapas-small-finetuned-sqa",
    "google/tapas-small-finetuned-wtq",
    "google/tapas-small-finetuned-wikisql-supervised",
    "google/tapas-small-finetuned-tabfact",
    # 小型模型
    "google/tapas-mini",
    "google/tapas-mini-finetuned-sqa",
    "google/tapas-mini-finetuned-wtq",
    "google/tapas-mini-finetuned-wikisql-supervised",
    "google/tapas-mini-finetuned-tabfact",
    # 超小型模型
    "google/tapas-tiny",
    "google/tapas-tiny-finetuned-sqa",
    "google/tapas-tiny-finetuned-wtq",
    "google/tapas-tiny-finetuned-wikisql-supervised",
    "google/tapas-tiny-finetuned-tabfact",
    # 查看所有 TAPAS 模型，请访问 https://huggingface.co/models?filter=tapas
# 误删除的右括号，需添加回去，保证语法正确性
]

# Zero division保护的常数
EPSILON_ZERO_DIVISION = 1e-10
# 接近 log 零的常数，用于处理极小的概率值
CLOSE_ENOUGH_TO_LOG_ZERO = -10000.0

# TFTableQuestionAnsweringOutput 类用于 TAPAS 问答模型的输出
@dataclass
class TFTableQuestionAnsweringOutput(ModelOutput):
    """
    Output type of [`TFTapasForQuestionAnswering`].

    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `labels` (and possibly `answer`, `aggregation_labels`, `numeric_values` and `numeric_values_scale` are provided)):
            Total loss as the sum of the hierarchical cell selection log-likelihood loss and (optionally) the
            semi-supervised regression loss and (optionally) supervised loss for aggregations.
        logits (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            Prediction scores of the cell selection head, for every token.
        logits_aggregation (`tf.Tensor`, *optional*, of shape `(batch_size, num_aggregation_labels)`):
            Prediction scores of the aggregation head, for every aggregation operator.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer plus
            the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """
    # 初始化属性
    loss: tf.Tensor | None = None
    logits: tf.Tensor = None
    logits_aggregation: tf.Tensor | None = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None

# TFTapasEmbeddings 类用于构建 TAPAS 模型的嵌入层
class TFTapasEmbeddings(tf.keras.layers.Layer):
    """
    Construct the embeddings from word, position and token_type embeddings. Same as BertEmbeddings but with a number of
    additional token type embeddings to encode tabular structure.
    """
    # 初始化方法
    def __init__(self, config: TapasConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 获取配置信息
        self.config = config
        # 附加的 token type embeddings 的数量
        self.number_of_token_type_embeddings = len(config.type_vocab_sizes)
        # 是否为每个单元格重置位置索引
        self.reset_position_index_per_cell = config.reset_position_index_per_cell
        # 隐藏层大小
        self.hidden_size = config.hidden_size
        # 最大位置嵌入
        self.max_position_embeddings = config.max_position_embeddings
        # 初始化范围
        self.initializer_range = config.initializer_range
        # LayerNormalization 层
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # Dropout 层
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
    # 构建方法，初始化词嵌入权重和位置嵌入
    def build(self, input_shape=None):
        # 在名字域"word_embeddings"下创建权重Tensor
        with tf.name_scope("word_embeddings"):
            self.weight = self.add_weight(
                name="weight",
                shape=[self.config.vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        # 在名字域"position_embeddings"下创建位置嵌入Tensor
        with tf.name_scope("position_embeddings"):
            self.position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_position_embeddings, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )
        
        # 针对每种类型的token，创建相应的类型嵌入Tensor
        for i, type_vocab_size in enumerate(self.config.type_vocab_sizes):
            with tf.name_scope(f"token_type_embeddings_{i}"):
                setattr(
                    self,
                    f"token_type_embeddings_{i}",
                    self.add_weight(
                        name="embeddings",
                        shape=[type_vocab_size, self.hidden_size],
                        initializer=get_initializer(self.initializer_range),
                    ),
                )

        # 如果已经构建完成，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在LayerNorm，构建LayerNorm层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])

    # 调用方法，处理输入数据
    def call(
        self,
        input_ids: tf.Tensor = None,
        position_ids: tf.Tensor = None,
        token_type_ids: tf.Tensor = None,
        inputs_embeds: tf.Tensor = None,
        training: bool = False,
    # 这个函数是用来应用基于输入 tensor 的嵌入
    def apply_embedding(
        self,
        input_ids=None,
        inputs_embeds=None,
        token_type_ids=None,
        position_ids=None,
        training=False,
    ) -> tf.Tensor:
        """
        Applies embedding based on inputs tensor.
    
        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        # 断言输入必须是输入 ID 或 embeds 之一
        assert not (input_ids is None and inputs_embeds is None)
        # 如果有输入 ID，获取输入形状
        if input_ids is not None:
            input_shape = shape_list(input_ids)
        # 否则根据 inputs_embeds 计算输入形状
        else:
            input_shape = shape_list(inputs_embeds)[:-1]
    
        # 计算序列长度
        seq_length = input_shape[1]
    
        # 如果没有 token_type_ids，创建一个全为 0 的 tensor
        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape + [self.number_of_token_type_embeddings], value=0)
    
        # 如果没有 position_ids，创建绝对位置嵌入
        if position_ids is None:
            position_ids = tf.expand_dims(tf.range(start=0, limit=seq_length), axis=0)
            position_ids = tf.broadcast_to(position_ids, shape=input_shape)
            # 如果需要重置每个单元格的位置索引
            if self.reset_position_index_per_cell:
                # 计算每个单元格的首个位置索引
                col_index = IndexMap(token_type_ids[:, :, 1], self.config.type_vocab_sizes[1], batch_dims=1)
                row_index = IndexMap(token_type_ids[:, :, 2], self.config.type_vocab_sizes[2], batch_dims=1)
                full_index = ProductIndexMap(col_index, row_index)
                first_position_per_segment = reduce_min(position_ids, full_index)[0]
                first_position = gather(first_position_per_segment, full_index)
                position = tf.expand_dims(tf.range(start=0, limit=seq_length), axis=0)
                position_ids = tf.math.minimum(self.max_position_embeddings - 1, position - first_position)
    
        # 如果有输入 ID，检查是否在词汇表范围内，并获取相应的嵌入
        if input_ids is not None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)
    
        # 获取位置嵌入
        position_embeddings = tf.gather(self.position_embeddings, indices=position_ids)
    
        # 将输入嵌入和位置嵌入相加
        final_embeddings = inputs_embeds + position_embeddings
    
        # 添加 token 类型嵌入
        for i in range(self.number_of_token_type_embeddings):
            name = f"token_type_embeddings_{i}"
            final_embeddings += tf.gather(params=getattr(self, name), indices=token_type_ids[:, :, i])
    
        # 进行层归一化和 dropout
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)
    
        # 返回最终的嵌入 tensor
        return final_embeddings
# 从transformers.models.bert.modeling_tf_bert.TFBertSelfAttention复制代码，将Bert->Tapas
class TFTapasSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs):
        super().__init__(**kwargs)

        # 如果hidden_size不能被num_attention_heads整除，则抛出异常
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

        # 创建Dense层，用于计算查询、键、值
        self.query = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        self.key = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key"
        )
        self.value = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )
        # 创建Dropout层，用于注意力概率的dropout
        self.dropout = tf.keras.layers.Dropout(rate=config.attention_probs_dropout_prob)

        self.is_decoder = config.is_decoder
        self.config = config

    # 将输入的tensor重塑为scores计算所需的格式
    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        # 重塑tensor的形状，从[batch_size, seq_length, all_head_size]到[batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size)

        # 转置tensor的形状，从[batch_size, seq_length, num_attention_heads, attention_head_size]到[batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    # 定义层的前向传播逻辑
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
# 从transformers.models.bert.modeling_tf_bert.TFBertSelfOutput复制代码，将Bert->Tapas
class TFTapasSelfOutput(tf.keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个密集层，用于处理隐藏状态
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建一个LayerNormalization层，用于规范化输入数据
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建一个Dropout层，用于随机失活
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        # 保存配置
        self.config = config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 通过密集层处理隐藏状态
        hidden_states = self.dense(inputs=hidden_states)
        # 使用Dropout层进行随机失活
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 对处理后的数据进行LayerNormalization，然后与输入数据相加
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states

    def build(self, input_shape=None):
        # 如果已建立则返回
        if self.built:
            return
        self.built = True
        # 如果存在密集层，则构建密集层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 如果存在LayerNormalization层，则构建LayerNormalization层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


# 从transformers.models.bert.modeling_tf_bert.TFBertAttention复制代码，将Bert->Tapas
class TFTapasAttention(tf.keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建TFTapasSelfAttention实例，用于组合自注意力层
        self.self_attention = TFTapasSelfAttention(config, name="self")
        # 创建TFTapasSelfOutput实例，用于组合自输出层
        self.dense_output = TFTapasSelfOutput(config, name="output")

    def prune_heads(self, heads):
        # 抛出未实现的错误
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
        # 通过self_attention处理输入tensor
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
        # 通过dense_output处理self_attention的输出
        attention_output = self.dense_output(
            hidden_states=self_outputs[0], input_tensor=input_tensor, training=training
        )
        # 如果有需要输出attentions，则将attentions返回
        outputs = (attention_output,) + self_outputs[1:]

        return outputs
    # 构建模型的方法，用于初始化模型参数
    def build(self, input_shape=None):
        # 如果模型已经构建完成，则直接返回，不再重复构建
        if self.built:
            return
        # 将模型标记为已构建状态
        self.built = True
        # 检查是否存在 self_attention 属性，并且不为 None
        if getattr(self, "self_attention", None) is not None:
            # 在命名空间下构建 self_attention 层
            with tf.name_scope(self.self_attention.name):
                # 构建 self_attention 层，传入 None 作为输入形状
                self.self_attention.build(None)
        # 检查是否存在 dense_output 属性，并且不为 None
        if getattr(self, "dense_output", None) is not None:
            # 在命名空间下构建 dense_output 层
            with tf.name_scope(self.dense_output.name):
                # 构建 dense_output 层，传入 None 作为输入形状
                self.dense_output.build(None)
# Copied from transformers.models.bert.modeling_tf_bert.TFBertIntermediate with Bert->Tapas
从transformers.models.bert.modeling_tf_bert.TFBertIntermediate复制，将Bert改为Tapas

class TFTapasIntermediate(tf.keras.layers.Layer):
    定义一个名为TFTapasIntermediate的类，继承自tf.keras.layers.Layer

    def __init__(self, config: TapasConfig, **kwargs):
        初始化TFTapasIntermediate类的实例对象

        super().__init__(**kwargs)
        调用父类的构造函数

        self.dense = tf.keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        定义一个全连接层(Dense)，输出大小为config.intermediate_size，权重的初始化方法为config.initializer_range所规定的方法，名称为"dense"

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        判断config.hidden_act是否为字符串类型，如果是，则调用get_tf_activation函数得到激活函数；否则，直接使用config.hidden_act作为激活函数

        self.config = config
        将输入参数config赋值给成员变量self.config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        定义call方法，用于前向传播计算

        hidden_states = self.dense(inputs=hidden_states)
        将输入hidden_states作为输入，通过全连接层self.dense进行线性变换计算得到hidden_states

        hidden_states = self.intermediate_act_fn(hidden_states)
        将hidden_states输入到激活函数self.intermediate_act_fn中进行激活得到hidden_states

        return hidden_states
        返回hidden_states作为输出结果

    def build(self, input_shape=None):
        定义build方法，用于构建Layer的可训练部分的参数

        if self.built:
            return
        判断是否已经构建过，如果已经构建过则直接返回

        self.built = True
        设置self.built为True，表示已经构建过

        if getattr(self, "dense", None) is not None:
            判断self.dense是否存在

            with tf.name_scope(self.dense.name):
                使用tf.name_scope设置作用域名称

                self.dense.build([None, None, self.config.hidden_size])
                构建全连接层self.dense的参数，设置输入的shape为[None, None, self.config.hidden_size]


# Copied from transformers.models.bert.modeling_tf_bert.TFBertOutput with Bert->Tapas
从transformers.models.bert.modeling_tf_bert.TFBertOutput复制，将Bert改为Tapas

class TFTapasOutput(tf.keras.layers.Layer):
    定义一个名为TFTapasOutput的类，继承自tf.keras.layers.Layer

    def __init__(self, config: TapasConfig, **kwargs):
        初始化TFTapasOutput类的实例对象

        super().__init__(**kwargs)
        调用父类的构造函数

        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        定义一个全连接层(Dense)，输出大小为config.hidden_size，权重的初始化方法为config.initializer_range所规定的方法，名称为"dense"

        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        定义一个LayerNormalization层，参数epsilon为config.layer_norm_eps，名称为"LayerNorm"

        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        定义一个Dropout层，参数rate为config.hidden_dropout_prob

        self.config = config
        将输入参数config赋值给成员变量self.config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        定义call方法，用于前向传播计算

        hidden_states = self.dense(inputs=hidden_states)
        将输入hidden_states作为输入，通过全连接层self.dense进行线性变换计算得到hidden_states

        hidden_states = self.dropout(inputs=hidden_states, training=training)
        将hidden_states输入到Dropout层self.dropout中进行随机失活操作得到hidden_states

        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)
        将hidden_states与输入input_tensor相加后，输入到LayerNormalization层self.LayerNorm中进行归一化得到hidden_states

        return hidden_states
        返回hidden_states作为输出结果

    def build(self, input_shape=None):
        定义build方法，用于构建Layer的可训练部分的参数

        if self.built:
            return
        判断是否已经构建过，如果已经构建过则直接返回

        self.built = True
        设置self.built为True，表示已经构建过

        if getattr(self, "dense", None) is not None:
            判断self.dense是否存在

            with tf.name_scope(self.dense.name):
                使用tf.name_scope设置作用域名称

                self.dense.build([None, None, self.config.intermediate_size])
                构建全连接层self.dense的参数，设置输入的shape为[None, None, self.config.intermediate_size]

        if getattr(self, "LayerNorm", None) is not None:
            判断self.LayerNorm是否存在

            with tf.name_scope(self.LayerNorm.name):
                使用tf.name_scope设置作用域名称

                self.LayerNorm.build([None, None, self.config.hidden_size])
                构建LayerNormalization层self.LayerNorm的参数，设置输入的shape为[None, None, self.config.hidden_size]


# Copied from transformers.models.bert.modeling_tf_bert.TFBertLayer with Bert->Tapas
从transformers.models.bert.modeling_tf_bert.TFBertLayer复制，将Bert改为Tapas

class TFTapasLayer(tf.keras.layers.Layer):
    定义一个名为TFTapasLayer的类，继承自tf.keras.layers.Layer

    ... （省略部分代码）
    # 初始化函数，接受config和其他关键字参数
    def __init__(self, config: TapasConfig, **kwargs):
        # 调用父类初始化函数
        super().__init__(**kwargs)
    
        # 创建自注意力层对象
        self.attention = TFTapasAttention(config, name="attention")
        # 判断是否为解码器模型
        self.is_decoder = config.is_decoder
        # 判断是否添加交叉注意力
        self.add_cross_attention = config.add_cross_attention
        # 如果添加了交叉注意力
        if self.add_cross_attention:
            # 如果不是解码器模型抛出异常
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 创建交叉注意力层对象
            self.crossattention = TFTapasAttention(config, name="crossattention")
        # 创建中间层对象
        self.intermediate = TFTapasIntermediate(config, name="intermediate")
        # 创建bert输出层对象
        self.bert_output = TFTapasOutput(config, name="output")
    
    # 调用函数，传入一系列参数
    def call(
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        encoder_hidden_states: tf.Tensor | None,
        encoder_attention_mask: tf.Tensor | None,
        past_key_value: Tuple[tf.Tensor] | None,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        # 将解码器的单向自注意力缓存的键/值元组位于位置1、2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 进行自注意力计算
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
        # 获取自注意力计算的输出
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        # 如果是解码器，则最后一个输出是自注意力缓存的元组
        if self.is_decoder:
            # 将输出中的自注意力缓存去除，得到最终输出
            outputs = self_attention_outputs[1:-1]
            # 获取当前位置的键/值对象
            present_key_value = self_attention_outputs[-1]
        else:
            # 如果是编码器，则将输出中的自注意力信息去除后作为结果
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            # 如果是解码器并且有编码器的隐藏状态，则进行交叉注意力计算
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            # 交叉注意力缓存的键/值元组位于 past_key_value 元组的第3、4位置
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 进行交叉注意力计算
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
            # 获取交叉注意力计算的输出
            attention_output = cross_attention_outputs[0]
            # 将交叉注意力的输出添加到结果中
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            # 将交叉注意力缓存添加到当前位置的键/值元组的第3、4位置
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 进行中间输出计算
        intermediate_output = self.intermediate(hidden_states=attention_output)
        # 进行BERT输出计算
        layer_output = self.bert_output(
            hidden_states=intermediate_output, input_tensor=attention_output, training=training
        )
        # 将计算结果添加到输出中
        outputs = (layer_output,) + outputs  # add attentions if we output them

        # if decoder, return the attn key/values as the last output
        # 如果是解码器，则将注意力的键/值作为最后一个输出返回
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs
    def build(self, input_shape=None):
        # 检查模型是否已经被建立，如果是则直接返回，不做任何操作
        if self.built:
            return
        # 设置模型已经被建立的标志为True
        self.built = True
        # 如果模型中存在"attention"属性
        if getattr(self, "attention", None) is not None:
            # 在attention层的名字空间中，建立attention层
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        # 如果模型中存在"intermediate"属性
        if getattr(self, "intermediate", None) is not None:
            # 在intermediate层的名字空间中，建立intermediate层
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        # 如果模型中存在"bert_output"属性
        if getattr(self, "bert_output", None) is not None:
            # 在bert_output层的名字空间中，建立bert_output层
            with tf.name_scope(self.bert_output.name):
                self.bert_output.build(None)
        # 如果模型中存在"crossattention"属性
        if getattr(self, "crossattention", None) is not None:
            # 在crossattention层的名字空间中，建立crossattention层
            with tf.name_scope(self.crossattention.name):
                self.crossattention.build(None)
# 从 transformers.models.bert.modeling_tf_bert.TFBertEncoder 复制代码，并将 Bert 替换为 Tapas
class TFTapasEncoder(tf.keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs):
        super().__init__(**kwargs)
        # 初始化 TapasEncoder 层，根据配置中的层数创建多个 TapasLayer
        self.config = config
        self.layer = [TFTapasLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]

    # 调用 TapasEncoder，处理输入并输出结果
    def call(
        self,
        hidden_states: tf.Tensor,  # 输入的隐藏状态张量
        attention_mask: tf.Tensor,  # 注意力掩码张量
        head_mask: tf.Tensor,  # 头部掩码张量
        encoder_hidden_states: tf.Tensor | None,  # 编码器隐藏状态张量或 None
        encoder_attention_mask: tf.Tensor | None,  # 编码器注意力掩码张量或 None
        past_key_values: Tuple[Tuple[tf.Tensor]] | None,  # 上下文关键值元组或 None
        use_cache: Optional[bool],  # 是否使用缓存
        output_attentions: bool,  # 是否输出注意力张量
        output_hidden_states: bool,  # 是否输出隐藏状态
        return_dict: bool,  # 是否返回字典形式的输出
        training: bool = False,  # 是否处于训练模式，默认为 False
    ) -> Union[TFBaseModelOutputWithPastAndCrossAttentions, Tuple[tf.Tensor]]:
        # 初始化保存所有隐藏状态的元组
        all_hidden_states = () if output_hidden_states else None
        # 初始化保存所有注意力张量的元组
        all_attentions = () if output_attentions else None
        # 初始化保存所有交叉注意力张量的元组（如果输出注意力张量且配置中有交叉注意力）
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        # 初始化保存下一个解码器缓存的元组
        next_decoder_cache = () if use_cache else None
        
        # 遍历每个 TapasLayer
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到保存所有隐藏状态的元组中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前 TapasLayer 所需的上下文关键值
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 调用当前 TapasLayer 处理输入，得到输出
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
            # 更新隐藏状态为当前 TapasLayer 的输出隐藏状态
            hidden_states = layer_outputs[0]

            # 如果需要使用缓存，则将当前 TapasLayer 的缓存添加到下一个解码器缓存中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            # 如果需要输出注意力张量，则将当前 TapasLayer 的注意力张量添加到保存所有注意力张量的元组中
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
                # 如果配置中有交叉注意力且存在编码器隐藏状态，则将当前 TapasLayer 的交叉注意力张量添加到保存所有交叉注意力张量的元组中
                if self.config.add_cross_attention and encoder_hidden_states is not None:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果需要输出隐藏状态，则将最后一个 TapasLayer 的隐藏状态添加到保存所有隐藏状态的元组中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要返回字典形式的输出，则将所有非 None 的结果组成元组返回
        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_attentions, all_cross_attentions] if v is not None
            )

        # 返回字典形式的输出
        return TFBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )
    # 构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回，不进行重复构建
        if self.built:
            return
        # 标记模型为已构建状态
        self.built = True
        # 检查模型是否包含子层，如果有则对每个子层进行构建
        if getattr(self, "layer", None) is not None:
            for layer in self.layer:
                # 使用子层的名称创建 TensorFlow 命名空间
                with tf.name_scope(layer.name):
                    # 构建子层，传入 None 作为输入形状
                    layer.build(None)
# 从transformers.models.bert.modeling_tf_bert.TFBertPooler复制代码，将Bert->Tapas
class TFTapasPooler(tf.keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 通过简单地取对应于第一个标记的隐藏状态来"池化"模型。
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


# 从transformers.models.bert.modeling_tf_bert.TFBertPredictionHeadTransform复制代码，将Bert->Tapas
class TFTapasPredictionHeadTransform(tf.keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense",
        )

        # 根据配置使用指定的激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.transform_act_fn = config.hidden_act

        # 使用层标准化
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 通过全连接层处理隐藏状态
        hidden_states = self.dense(inputs=hidden_states)
        # 应用激活函数
        hidden_states = self.transform_act_fn(hidden_states)
        # 应用层标准化
        hidden_states = self.LayerNorm(inputs=hidden_states)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


# 从transformers.models.bert.modeling_tf_bert.TFBertLMPredictionHead复制代码，将Bert->Tapas
class TFTapasLMPredictionHead(tf.keras.layers.Layer):
    # 初始化函数，接收配置信息和输入的嵌入层对象作为参数
    def __init__(self, config: TapasConfig, input_embeddings: tf.keras.layers.Layer, **kwargs):
        super().__init__(**kwargs)

        # 保存配置信息和隐藏层尺寸
        self.config = config
        self.hidden_size = config.hidden_size

        # 创建一个用于预测结果的转换层对象
        self.transform = TFTapasPredictionHeadTransform(config, name="transform")

        # 设置输入的嵌入层对象
        self.input_embeddings = input_embeddings

    # 构建层
    def build(self, input_shape=None):
        # 添加输出偏置参数
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="bias")

        # 如果已经构建则直接返回
        if self.built:
            return
        self.built = True
        # 如果转换层存在，构建转换层
        if getattr(self, "transform", None) is not None:
            with tf.name_scope(self.transform.name):
                self.transform.build(None)

    # 返回输出的嵌入层对象
    def get_output_embeddings(self) -> tf.keras.layers.Layer:
        return self.input_embeddings

    # 设置输出的嵌入层对象
    def set_output_embeddings(self, value: tf.Variable):
        self.input_embeddings.weight = value
        self.input_embeddings.vocab_size = shape_list(value)[0]

    # 返回偏置参数
    def get_bias(self) -> Dict[str, tf.Variable]:
        return {"bias": self.bias}

    # 设置偏置参数
    def set_bias(self, value: tf.Variable):
        self.bias = value["bias"]
        self.config.vocab_size = shape_list(value["bias"])[0]

    # 前向传播函数
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 对输入数据进行转换
        hidden_states = self.transform(hidden_states=hidden_states)
        # 获取序列长度
        seq_length = shape_list(hidden_states)[1]
        # 展平隐藏状态
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.hidden_size])
        # 矩阵相乘获取隐藏状态和嵌入层权重的乘积
        hidden_states = tf.matmul(a=hidden_states, b=self.input_embeddings.weight, transpose_b=True)
        # 将结果重新调整成序列形状
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])
        # 添加偏置
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)

        # 返回结果
        return hidden_states
# 定义一个自定义层，继承自 tf.keras.layers.Layer 类，用于 Tapas 模型的 MLM 头部
class TFTapasMLMHead(tf.keras.layers.Layer):
    def __init__(self, config: TapasConfig, input_embeddings: tf.keras.layers.Layer, **kwargs):
        super().__init__(**kwargs)

        # 创建 MLM 预测头部对象
        self.predictions = TFTapasLMPredictionHead(config, input_embeddings, name="predictions")

    # 定义 call 方法，用于调用预测头部对象进行预测
    def call(self, sequence_output: tf.Tensor) -> tf.Tensor:
        prediction_scores = self.predictions(hidden_states=sequence_output)

        return prediction_scores

    # 定义 build 方法，用于搭建模型
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "predictions", None) is not None:
            with tf.name_scope(self.predictions.name):
                self.predictions.build(None)

# 定义主要的 Tapas 模型层，继承自 tf.keras.layers.Layer 类
@keras_serializable
class TFTapasMainLayer(tf.keras.layers.Layer):
    # 设定配置类为 TapasConfig
    config_class = TapasConfig

    def __init__(self, config: TapasConfig, add_pooling_layer: bool = True, **kwargs):
        requires_backends(self, "tensorflow_probability")
        super().__init__(**kwargs)

        self.config = config

        # 创建嵌入层对象
        self.embeddings = TFTapasEmbeddings(config, name="embeddings")
        # 创建编码器对象
        self.encoder = TFTapasEncoder(config, name="encoder")
        # 如果需要添加池化层，则创建池化层对象
        self.pooler = TFTapasPooler(config, name="pooler") if add_pooling_layer else None

    # 获取输入嵌入层对象
    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        return self.embeddings

    # 设置输入嵌入层对象
    def set_input_embeddings(self, value: tf.Variable):
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    # 剪枝模型头部
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    # 定义 model.call 方法，用于模型调用
    @unpack_inputs
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
        training: bool = False,
    # 定义 build 方法，用于搭建模型
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在嵌入层对象，则构建嵌入层
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        # 如果存在编码器对象，则构建编码器
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 如果存在池化层对象，则构建池化层
        if getattr(self, "pooler", None) is not None:
            with tf.name_scope(self.pooler.name):
                self.pooler.build(None)
# 定义了一个抽象类，用于处理权重初始化和一个简单的接口来下载和加载预训练模型
class TFTapasPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定配置类
    config_class = TapasConfig
    # 指定基础模型前缀
    base_model_prefix = "tapas"

    # 定义输入签名，规定输入的 Tensor 格式和名称
    @property
    def input_signature(self):
        return {
            "input_ids": tf.TensorSpec((None, None), tf.int32, name="input_ids"),
            "attention_mask": tf.TensorSpec((None, None), tf.float32, name="attention_mask"),
            "token_type_ids": tf.TensorSpec((None, None, 7), tf.int32, name="token_type_ids"),
        }


# 开始文档字符串
TAPAS_START_DOCSTRING = r"""

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
    参数:
        config ([`TapasConfig`]): 模型配置类，包含模型的所有参数。
            使用配置文件初始化不会加载与模型相关的权重，只会加载配置。
            查看 [`~PreTrainedModel.from_pretrained`] 方法加载模型权重。
"""

TAPAS_INPUTS_DOCSTRING = r"""
"""


@add_start_docstrings(
    "The bare Tapas Model transformer outputting raw hidden-states without any specific head on top.",
    TAPAS_START_DOCSTRING,
)
class TFTapasModel(TFTapasPreTrainedModel):
    def __init__(self, config: TapasConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 初始化 Tapas 主层
        self.tapas = TFTapasMainLayer(config, name="tapas")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(TAPAS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFBaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
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
        training: Optional[bool] = False,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, TapasModel
        >>> import pandas as pd

        >>> tokenizer = AutoTokenizer.from_pretrained("google/tapas-base")
        >>> model = TapasModel.from_pretrained("google/tapas-base")

        >>> data = {
        ...     "Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
        ...     "Age": ["56", "45", "59"],
        ...     "Number of movies": ["87", "53", "69"],
        ... }
        >>> table = pd.DataFrame.from_dict(data)
        >>> queries = ["How many movies has George Clooney played in?", "How old is Brad Pitt?"]

        >>> inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="tf")
        >>> outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
        ```py"""
        
        # 调用 Tapas 主层处理输入
        outputs = self.tapas(
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

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "tapas", None) is not None:
            with tf.name_scope(self.tapas.name):
                self.tapas.build(None)


@add_start_docstrings("""Tapas Model with a `language modeling` head on top.""", TAPAS_START_DOCSTRING)
class TFTapasForMaskedLM(TFTapasPreTrainedModel, TFMaskedLanguageModelingLoss):
    def __init__(self, config: TapasConfig, *inputs, **kwargs):
        # 调用父类的初始化函数，传入配置和其他参数
        super().__init__(config, *inputs, **kwargs)

        # 如果配置的是decoder，则发出警告
        if config.is_decoder:
            logger.warning(
                "If you want to use `TFTapasForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # 创建 Tapas 主层对象，不添加池化层，命名为"tapas"
        self.tapas = TFTapasMainLayer(config, add_pooling_layer=False, name="tapas")
        # 创建 Tapas MLM 头部对象，传入配置和输入嵌入对象，命名为"cls"
        self.lm_head = TFTapasMLMHead(config, input_embeddings=self.tapas.embeddings, name="cls")

    # 获取 MLM 头部
    def get_lm_head(self) -> tf.keras.layers.Layer:
        return self.lm_head.predictions

    # 定义call方法，接受多种输入参数，包括输入ids、注意力掩码、token类型ids等
    @unpack_inputs
    @add_start_docstrings_to_model_forward(TAPAS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFMaskedLMOutput, config_class=_CONFIG_FOR_DOC)
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
    # 定义函数，输入为self，input_ids，attention_mask，token_type_ids，position_ids，head_mask，inputs_embeds，output_attentions，output_hidden_states，return_dict，training以及labels（可选），输出为TFMaskedLMOutput或者包含tf.Tensor的元组
    def __call__(
        self,
        input_ids: Optional[tf.Tensor] = None,
        attention_mask: Optional[tf.Tensor] = None,
        token_type_ids: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        head_mask: Optional[tf.Tensor] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        labels: Optional[Union[tf.Tensor, np.ndarray]] = None,
    ) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional`):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
    
        Returns:
    
        Examples:
    
        ```python
        >>> from transformers import AutoTokenizer, TapasForMaskedLM
        >>> import pandas as pd
    
        >>> tokenizer = AutoTokenizer.from_pretrained("google/tapas-base")
        >>> model = TapasForMaskedLM.from_pretrained("google/tapas-base")
    
        >>> data = {
        ...     "Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
        ...     "Age": ["56", "45", "59"],
        ...     "Number of movies": ["87", "53", "69"],
        ... }
        >>> table = pd.DataFrame.from_dict(data)
    
        >>> inputs = tokenizer(
        ...     table=table, queries="How many [MASK] has George [MASK] played in?", return_tensors="tf"
        ... )
        >>> labels = tokenizer(
        ...     table=table, queries="How many movies has George Clooney played in?", return_tensors="tf"
        ... )["input_ids"]
    
        >>> outputs = model(**inputs, labels=labels)
        >>> logits = outputs.logits
        ```py"""
        # 调用Tapas模型生成结果
        outputs = self.tapas(
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
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)
        # 如果labels不为None，则使用hf_compute_loss计算损失，否则为None
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=prediction_scores)
    
        # 如果return_dict为False，返回包含prediction_scores和outputs[2:]的元组，如果loss不为None，则包含loss
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
    
        # 如果return_dict为True，则返回TFMaskedLMOutput对象
        return TFMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    # 定义函数，构建模型
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果tapas存在，则构建tapas模型
        if getattr(self, "tapas", None) is not None:
            with tf.name_scope(self.tapas.name):
                self.tapas.build(None)
        # 如果lm_head存在，则构建lm_head模型
        if getattr(self, "lm_head", None) is not None:
            with tf.name_scope(self.lm_head.name):
                self.lm_head.build(None)
# 定义一个 TensorFlow 自定义层，用于计算每个 token 的 logits
class TFTapasComputeTokenLogits(tf.keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs):
        super().__init__(**kwargs)

        # 设置温度参数
        self.temperature = config.temperature
        # 定义输出的权重和偏置
        with tf.name_scope("output"):
            self.output_weights = self.add_weight(
                name="output_weights",
                shape=(config.hidden_size,),
                dtype=tf.float32,
                trainable=True,
                initializer=tf.zeros_initializer()
                if config.init_cell_selection_weights_to_zero
                else tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range),
            )
            self.output_bias = self.add_weight(
                name="output_bias", shape=(), trainable=True, initializer=tf.zeros_initializer()
            )

    def call(self, sequence_output: tf.Tensor) -> tf.Tensor:
        """
        Computes logits per token

        Args:
            sequence_output (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Also known as last_hidden_state. Sequence of hidden-states at the output of the last layer of the
                model.

        Returns:
            logits (`tf.Tensor` of shape `(batch_size, sequence_length)`): Logits per token.
        """
        # 计算 logits
        logits = (tf.einsum("bsj,j->bs", sequence_output, self.output_weights) + self.output_bias) / self.temperature
        return logits


# 定义一个 TensorFlow 自定义层，用于计算每列的 logits
class TFTapasComputeColumnLogits(tf.keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs):
        super().__init__(**kwargs)

        # 定义输出列的权重和偏置
        with tf.name_scope("column_output"):
            self.column_output_weights = self.add_weight(
                name="column_output_weights",
                shape=[config.hidden_size],
                dtype=tf.float32,
                trainable=True,
                initializer=tf.zeros_initializer()
                if config.init_cell_selection_weights_to_zero
                else tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range),
            )
            self.column_output_bias = self.add_weight(
                name="column_output_bias", shape=(), trainable=True, initializer=tf.zeros_initializer()
            )
    def call(self, sequence_output, cell_index, cell_mask, allow_empty_column_selection) -> tf.Tensor:
        """
        Computes the column logits.

        Args:
            sequence_output (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Also known as last_hidden_state. Sequence of hidden-states at the output of the last layer of the
                model.
            cell_index (`ProductIndexMap`):
                Index that groups tokens into cells.
            cell_mask (`tf.Tensor` of shape `(batch_size, max_num_rows * max_num_cols)`):
                Mask for cells that exist in the table (i.e. that are not padding).
            allow_empty_column_selection (`bool`):
                Whether to allow not to select any column

        Returns:
            column_logits (`tf.Tensor`of shape `(batch_size, max_num_cols)`): Tensor containing the column logits for
            every example in the batch.
        """

        # First, compute the token logits (batch_size, seq_len) - without temperature
        token_logits = tf.einsum("bsj,j->bs", sequence_output, self.column_output_weights) + self.column_output_bias

        # Next, average the logits per cell (batch_size, max_num_cols*max_num_rows)
        cell_logits, cell_logits_index = reduce_mean(token_logits, cell_index)

        # Finally, average the logits per column (batch_size, max_num_cols)
        column_index = cell_index.project_inner(cell_logits_index)
        column_logits, out_index = reduce_sum(cell_logits * cell_mask, column_index)

        # Compute the total non-padding cell count per column
        cell_count, _ = reduce_sum(cell_mask, column_index)
        # Avoid division by zero
        column_logits /= cell_count + EPSILON_ZERO_DIVISION

        # Mask columns that do not appear in the example.
        is_padding = tf.logical_and(cell_count < 0.5, tf.not_equal(out_index.indices, 0))
        column_logits += CLOSE_ENOUGH_TO_LOG_ZERO * tf.cast(is_padding, tf.float32)

        # If not allowed to select empty column, mask the zero index
        if not allow_empty_column_selection:
            column_logits += CLOSE_ENOUGH_TO_LOG_ZERO * tf.cast(tf.equal(out_index.indices, 0), tf.float32)

        return column_logits
# 定义了一个 Tapas 问答模型类，包括单元格选择头部和可选的聚合头部
class TFTapasForQuestionAnswering(TFTapasPreTrainedModel):
    
    # 初始化函数
    def __init__(self, config: TapasConfig, *inputs, **kwargs):
        # 调用父类初始化函数
        super().__init__(config, *inputs, **kwargs)

        # 基础模型
        # 创建 Tapas 主层模型对象
        self.tapas = TFTapasMainLayer(config, name="tapas")

        # dropout
        # 创建 dropout 层，用于模型训练过程中的随机失活
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

        # 计算 token 的 logits 层
        self.compute_token_logits = TFTapasComputeTokenLogits(config, name="compute_token_logits")

        # 计算 column 的 logits 层
        self.compute_column_logits = TFTapasComputeColumnLogits(config, name="compute_column_logits")

        # 如果聚合标签的数量大于0，则创建一个全连接层用于聚合分类
        if config.num_aggregation_labels > 0:
            self.aggregation_classifier = tf.keras.layers.Dense(
                config.num_aggregation_labels,
                kernel_initializer=get_initializer(config.initializer_range),
                name="aggregation_classifier",
            )
        self.config = config

    # 在 call 方法上添加装饰器，用于接受输入参数并调用前向传播函数
    @unpack_inputs
    @add_start_docstrings_to_model_forward(TAPAS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFTableQuestionAnsweringOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        table_mask: np.ndarray | tf.Tensor | None = None,
        aggregation_labels: np.ndarray | tf.Tensor | None = None,
        float_answer: np.ndarray | tf.Tensor | None = None,
        numeric_values: np.ndarray | tf.Tensor | None = None,
        numeric_values_scale: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    # 根据输入形状构建模型，如果已经构建过，则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 设置标记为已构建
        self.built = True
        # 如果存在 tapas 属性，使用 tapas 的名称创建作用域，并构建 tapas
        if getattr(self, "tapas", None) is not None:
            with tf.name_scope(self.tapas.name):
                self.tapas.build(None)
        # 如果存在 compute_token_logits 属性，使用 compute_token_logits 的名称创建作用域，并构建 compute_token_logits
        if getattr(self, "compute_token_logits", None) is not None:
            with tf.name_scope(self.compute_token_logits.name):
                self.compute_token_logits.build(None)
        # 如果存在 compute_column_logits 属性，使用 compute_column_logits 的名称创建作用域，并构建 compute_column_logits
        if getattr(self, "compute_column_logits", None) is not None:
            with tf.name_scope(self.compute_column_logits.name):
                self.compute_column_logits.build(None)
        # 如果存在 aggregation_classifier 属性，使用 aggregation_classifier 的名称创建作用域，并构建 aggregation_classifier
        if getattr(self, "aggregation_classifier", None) is not None:
            with tf.name_scope(self.aggregation_classifier.name):
                self.aggregation_classifier.build([None, None, self.config.hidden_size])
# 定义一个 TFTapasForSequenceClassification 类，它继承自 TFTapasPreTrainedModel 和 TFSequenceClassificationLoss
@add_start_docstrings(
    """
    Tapas Model with a sequence classification head on top (a linear layer on top of the pooled output), e.g. for table
    entailment tasks, such as TabFact (Chen et al., 2020).
    """,
    TAPAS_START_DOCSTRING,
)
class TFTapasForSequenceClassification(TFTapasPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config: TapasConfig, *inputs, **kwargs):
        # 调用父类的构造方法
        super().__init__(config, *inputs, **kwargs)
        # 设置分类标签的数量
        self.num_labels = config.num_labels

        # 创建 TFTapasMainLayer 实例
        self.tapas = TFTapasMainLayer(config, name="tapas")
        # 创建 Dropout 层
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob, name="dropout")
        # 创建 Dense 层作为分类器
        self.classifier = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        self.config = config

    # 解包输入参数
    @unpack_inputs
    # 添加模型前向传播的文档字符串
    @add_start_docstrings_to_model_forward(TAPAS_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    # 替换返回值的文档字符串
    @replace_return_docstrings(output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        # 输入 ID
        input_ids: TFModelInputType | None = None,
        # 注意力掩码
        attention_mask: np.ndarray | tf.Tensor | None = None,
        # 标记类型 ID
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        # 位置 ID
        position_ids: np.ndarray | tf.Tensor | None = None,
        # 头部掩码
        head_mask: np.ndarray | tf.Tensor | None = None,
        # 输入嵌入
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        # 是否输出注意力权重
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典格式的输出
        return_dict: Optional[bool] = None,
        # 标签
        labels: np.ndarray | tf.Tensor | None = None,
        # 是否为训练模式
        training: Optional[bool] = False,
    def sequence_classifier(
        self,
        input_ids: tf.Tensor,
        attention_mask: tf.Tensor,
        token_type_ids: tf.Tensor = None,
        position_ids: tf.Tensor = None,
        head_mask: tf.Tensor = None,
        inputs_embeds: tf.Tensor = None,
        return_dict: bool = False,
        training: bool = False,
        labels: Optional[tf.Tensor] = None,
    ) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy). Note: this is called
            "classification_class_index" in the original implementation.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, TapasForSequenceClassification
        >>> import tensorflow as tf
        >>> import pandas as pd

        >>> tokenizer = AutoTokenizer.from_pretrained("google/tapas-base-finetuned-tabfact")
        >>> model = TapasForSequenceClassification.from_pretrained("google/tapas-base-finetuned-tabfact")

        >>> data = {
        ...     "Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
        ...     "Age": ["56", "45", "59"],
        ...     "Number of movies": ["87", "53", "69"],
        ... }
        >>> table = pd.DataFrame.from_dict(data)
        >>> queries = [
        ...     "There is only one actor who is 45 years old",
        ...     "There are 3 actors which played in more than 60 movies",
        ... ]

        >>> inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="tf")
        >>> labels = tf.convert_to_tensor([1, 0])  # 1 means entailed, 0 means refuted

        >>> outputs = model(**inputs, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
        ```py"""

        # 使用tapas模型对输入进行序列分类
        outputs = self.tapas(
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
        # 获取池化后的输出
        pooled_output = outputs[1]
        # 使用dropout进行训练过程中的神经元随机失活
        pooled_output = self.dropout(inputs=pooled_output, training=training)
        # 将池化后的输出输入到分类器中得到logits
        logits = self.classifier(inputs=pooled_output)
        # 如果存在标签则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        # 如果不返回字典格式的结果
        if not return_dict:
            # 返回不同元素的组合
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回值为TF模型的序列分类输出对象
        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    # 在建立模型时调用此方法，如果已经建立过则直接返回
    def build(self, input_shape=None):
        # 如果已经建立过则直接返回
        if self.built:
            return
        # 标记为已经建立
        self.built = True
        # 如果存在 tapas 属性，则建立 tapas 模型
        if getattr(self, "tapas", None) is not None:
            with tf.name_scope(self.tapas.name):
                self.tapas.build(None)
        # 如果存在 dropout 属性，则建立 dropout 模型
        if getattr(self, "dropout", None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)
        # 如果存在 classifier 属性，则建立 classifier 模型
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                # 建立 classifier 模型，设置输入形状为 [None, None, self.config.hidden_size]
                self.classifier.build([None, None, self.config.hidden_size])
""" TAPAS utilities."""


class AverageApproximationFunction(str, enum.Enum):
    RATIO = "ratio"  # 定义枚举类，表示平均逼近函数的类型为比率
    FIRST_ORDER = "first_order"  # 第一阶
    SECOND_ORDER = "second_order"  # 第二阶


# Beginning of everything related to segmented tensors


class IndexMap(object):
    """Index grouping entries within a tensor."""
    
    def __init__(self, indices, num_segments, batch_dims=0):
        """
        Creates an index.

        Args:
          indices: <int32> Tensor of indices, same shape as `values`. 存储索引的张量，与 `values` 相同形状。
          num_segments: <int32> Scalar tensor, the number of segments. All elements
            in a batched segmented tensor must have the same number of segments (although many segments can be empty).
            标量张量，段的数量。批量分段张量中的所有元素必须具有相同数量的段（尽管许多段可以为空）。
          batch_dims: Python integer, the number of batch dimensions. The first
            `batch_dims` dimensions of a SegmentedTensor are treated as batch dimensions. Segments in different batch
            elements are always distinct even if they have the same index.
            Python 整数，批量维度的数量。SegmentedTensor 的前 `batch_dims` 维被视为批量维度。即使具有相同索引，不同批次元素中的段始终是不同的。
        """
        self.indices = tf.convert_to_tensor(indices)  # 将索引转换为张量
        self.num_segments = tf.convert_to_tensor(num_segments)  # 将段的数量转换为张量
        self.batch_dims = batch_dims  # 批量维度数

    def batch_shape(self):
        return tf.shape(self.indices)[: self.batch_dims]  # 返回批量形状


class ProductIndexMap(IndexMap):
    """The product of two indices."""
    
    def __init__(self, outer_index, inner_index):
        """
        Combines indices i and j into pairs (i, j). The result is an index where each segment (i, j) is the
        intersection of segments i and j. For example if the inputs represent table cells indexed by respectively rows
        and columns the output will be a table indexed by (row, column) pairs, i.e. by cell. The implementation
        combines indices {0, .., n - 1} and {0, .., m - 1} into {0, .., nm - 1}. The output has `num_segments` equal to
        `outer_index.num_segements` * `inner_index.num_segments`.

        Args:
          outer_index: IndexMap. 外部索引。
          inner_index: IndexMap, must have the same shape as `outer_index`. 内部索引，必须与 `outer_index` 具有相同的形状。
        """
        if outer_index.batch_dims != inner_index.batch_dims:
            raise ValueError("outer_index.batch_dims and inner_index.batch_dims must be the same.")
        
        # 调用父类构造函数
        super(ProductIndexMap, self).__init__(
            indices=(
                inner_index.indices
                + outer_index.indices * tf.cast(inner_index.num_segments, inner_index.indices.dtype)
            ),
            num_segments=inner_index.num_segments * outer_index.num_segments,
            batch_dims=inner_index.batch_dims,
        )
        self.outer_index = outer_index  # 外部索引
        self.inner_index = inner_index  # 内部索引

    def project_outer(self, index):
        """Projects an index with the same index set onto the outer components."""
        return IndexMap(
            indices=tf.math.floordiv(index.indices, self.inner_index.num_segments),  # 对外部分量进行投影
            num_segments=self.outer_index.num_segments,  # 段的数量
            batch_dims=index.batch_dims,  # 批量维度数
        )
    # 定义一个方法，用于将具有相同索引集的索引投影到内部组件上
    def project_inner(self, index):
        """Projects an index with the same index set onto the inner components."""
        # 使用 TensorFlow 的数学函数 floormod 对索引的 indices 进行取模操作，
        # 以使其在内部索引的段数范围内
        return IndexMap(
            indices=tf.math.floormod(index.indices, self.inner_index.num_segments),
            # 设置投影后的索引的段数与内部索引的段数相同
            num_segments=self.inner_index.num_segments,
            # 将索引映射对象的批次维度设置为原始索引的批次维度
            batch_dims=index.batch_dims,
        )
# 从`values`中使用索引映射`index`进行聚合。对于索引映射的域中的每个元素，此操作查找`values`中该索引的值。同一段中的两个元素始终被赋予相同的值。
def gather(values, index, name="segmented_gather"):
    return tf.gather(values, index.indices, batch_dims=index.batch_dims, name=name)

# 将批处理的索引映射展平为一维索引映射。此操作重新标记段，以保持批处理元素不同。第k个批处理元素的索引会偏移 `num_segments` * (k - 1)。结果是一个张量，其尺寸为 `num_segments` 乘以批处理元素的数量。
def flatten(index, name="segmented_flatten"):
    batch_size = tf.reduce_prod(index.batch_shape())
    offset = tf.range(batch_size) * index.num_segments
    offset = tf.reshape(offset, index.batch_shape())
    for _ in range(index.batch_dims, index.indices.shape.rank):
        offset = tf.expand_dims(offset, -1)
    indices = tf.cast(offset, index.indices.dtype) + index.indices
    return IndexMap(indices=tf.reshape(indices, [-1]), num_segments=index.num_segments * batch_size, batch_dims=0)

# 构建等于 range(num_segments) 的索引映射
def range_index_map(batch_shape, num_segments, name="range_index_map"):
    batch_shape = tf.convert_to_tensor(batch_shape)
    batch_shape.shape.assert_has_rank(1)
    num_segments = tf.convert_to_tensor(num_segments)
    num_segments.shape.assert_has_rank(0)
    indices = tf.range(num_segments)
    shape = tf.concat([tf.ones_like(batch_shape, dtype=tf.int32), tf.expand_dims(num_segments, axis=0)], axis=0)
    indices = tf.reshape(indices, shape)
    multiples = tf.concat([batch_shape, [1]], axis=0)
    indices = tf.tile(indices, multiples)
    return IndexMap(indices=indices, num_segments=num_segments, batch_dims=batch_shape.shape.as_list()[0])

# 对`values`应用段归约，逐段处理。
def _segment_reduce(values, index, segment_reduce_fn, name):
    # 这里需要补充代码
    Args:
        values (`tf.Tensor`):
            Tensor with segment values.  # 接受一个包含段值的张量
        index (`IndexMap`):
            IndexMap.  # 索引映射
        segment_reduce_fn (`str`):
            Name for the reduce operation. One of "sum", "mean", "max" or "min".  # 减少操作的名称，可以是“sum”、“mean”、“max”或“min”之一
        name (`str`):
            Name for the operation. Currently not used  # 操作的名称，目前未使用

    Returns:
        (`IndexMap`): IndexMap of shape batch_shape with elements equal to range(num_segments).
    """
    # Flatten the batch dimensions, as segments ops do not support batching.
    # However if `values` has extra dimensions to the right keep them
    # unflattened. Segmented ops support vector-valued operations.
    # 压平批次维度，因为分段操作不支持批处理
    # 然而，如果“values”的右侧有额外的维度，则保持未压平。分段操作支持矢量值操作。
    flat_index = flatten(index)
    vector_shape = tf.shape(values)[index.indices.shape.rank :]
    flattened_shape = tf.concat([[-1], vector_shape], axis=0)
    flat_values = tf.reshape(values, flattened_shape)
    segment_means = segment_reduce_fn(
        data=flat_values, segment_ids=flat_index.indices, num_segments=flat_index.num_segments
    )

    # Unflatten the values.
    new_shape = tf.concat([index.batch_shape(), [index.num_segments], vector_shape], axis=0)
    output_values = tf.reshape(segment_means, new_shape)
    output_index = range_index_map(index.batch_shape(), index.num_segments)
    # 返回处理后的值和索引
    return output_values, output_index
``` 
# 对张量在其各个段上求平均值。对于空段，输出为0。此操作计算段的平均值，支持以下功能：
#   - 使用第一维度 [B1, B2, ..., Bn] 进行批处理。批处理中的每个元素可以具有不同的索引。
#   - 使用最后一维度 [V1, V2, ...] 进行矢量化。如果存在，则输出将是矢量的平均值，而不是标量。
#   仅通过操作来减少中间维度 [I1, ..., Ik]。
def reduce_mean(values, index, name="segmented_reduce_mean"):
    return _segment_reduce(values, index, tf.math.unsorted_segment_mean, name)


# 对张量在其各个段上求和。对于空段，输出为0。此操作计算段的和，支持以下功能：
#   - 使用第一维度 [B1, B2, ..., Bn] 进行批处理。批处理中的每个元素可以具有不同的索引。
#   - 使用最后一维度 [V1, V2, ...] 进行矢量化。如果存在，则输出将是矢量的总和，而不是标量。
#   仅通过操作来减少中间维度 [I1, ..., Ik]。
def reduce_sum(values, index, name="segmented_reduce_sum"):
    return _segment_reduce(values, index, tf.math.unsorted_segment_sum, name)


# 计算各个段的最大值。此操作计算段的最大值，支持以下功能：
#   - 使用第一维度 [B1, B2, ..., Bn] 进行批处理。批处理中的每个元素可以具有不同的索引。
#   - 使用最后一维度 [V1, V2, ...] 进行矢量化。如果存在，则输出将是矢量的逐元素最大值，而不是标量。
#   仅通过操作来减少中间维度 [I1, ..., Ik]。
def reduce_max(values, index, name="segmented_reduce_max"):
    返回:
      一个包含输出值和输出索引的元组，其中`output_values`是一个形状为[B1, B2, ..., Bn, num_segments, V1, V2, ..]的张量，`index`是一个形状为[B1, B2, ..., Bn, num_segments]的索引映射。
    """
    # 调用_segment_reduce函数，使用tf.math.unsorted_segment_max函数进行降维操作，返回结果
    return _segment_reduce(values, index, tf.math.unsorted_segment_max, name)
def reduce_min(values, index, name="segmented_reduce_min"):
    """计算各个段的最小值。"""
    return _segment_reduce(values, index, tf.math.unsorted_segment_min, name)


def _single_column_cell_selection_loss(token_logits, column_logits, labels, cell_index, col_index, cell_mask):
    """
    计算受单个列约束的单元选择损失。损失是一个分层的对数似然。模型首先预测一个列，然后在该列中选择单元（在列的条件下）。
    永远不会选择所选列之外的单元。

    Args:
        token_logits (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            每个标记的逻辑值的张量。
        column_logits (`tf.Tensor` of shape `(batch_size, max_num_cols)`):
            每个列的逻辑值的张量。
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            每个标记的标签。
        cell_index (`ProductIndexMap`):
            将标记分组成单元的索引。
        col_index (`IndexMap`):
            将标记分组到列的索引。
        cell_mask (`tf.Tensor` of shape `(batch_size, max_num_rows * max_num_cols)`):
            表中存在的单元的掩码（即不是填充的单元）。

    Returns:
        selection_loss_per_example (`tf.Tensor` of shape `(batch_size,)`): 每个示例的损失。
        logits (`tf.Tensor` of shape `(batch_size, sequence_length)`): 新的逻辑值，仅允许选择单个列中的单元。
        根据 *column_logits* 最可能的列外的逻辑将被设置为非常低的值（使概率为 0）。
    """
    # 先找到应该选择的列。我们使用具有最大选定单元数量的列。
    labels_per_column, _ = reduce_sum(tf.cast(labels, tf.float32), col_index)
    column_label = tf.argmax(labels_per_column, axis=-1, output_type=tf.int32)
    # 检查列中是否没有选定的单元。在这种情况下，模型应该预测特殊列 id 0，表示“不选择任何东西”。
    no_cell_selected = tf.equal(tf.reduce_max(labels_per_column, axis=-1), 0)
    column_label = tf.where(no_cell_selected, tf.zeros_like(column_label), column_label)

    column_dist = tfp.distributions.Categorical(logits=column_logits)
    column_loss_per_example = -column_dist.log_prob(column_label)

    # 将标签和逻辑从每个标记减少到每个单元。
    logits_per_cell, _ = reduce_mean(token_logits, cell_index)
    labels_per_cell, labels_index = reduce_max(tf.cast(labels, tf.int32), cell_index)

    # 用于选择的列的掩码。
    column_id_for_cells = cell_index.project_inner(labels_index).indices
    column_mask = tf.cast(tf.equal(column_id_for_cells, tf.expand_dims(column_label, axis=1)), tf.float32)

    # 计算单元的对数似然，但仅对所选列计算。
    # 创建一个伯努利分布对象，用于计算每个细胞的概率
    cell_dist = tfp.distributions.Bernoulli(logits=logits_per_cell)
    # 计算每个细胞的对数概率
    cell_log_prob = cell_dist.log_prob(labels_per_cell)
    # 计算每个细胞的损失，乘以列掩码和细胞掩码，然后沿着维度1求和
    cell_loss = -tf.reduce_sum(cell_log_prob * column_mask * cell_mask, axis=1)
    # 我们需要通过列中的细胞数量对损失进行归一化
    cell_loss /= tf.reduce_sum(column_mask * cell_mask, axis=1) + EPSILON_ZERO_DIVISION

    # 每个样本的选择损失等于每个列的损失
    selection_loss_per_example = column_loss_per_example
    # 如果没有选择任何细胞，则添加细胞损失到选择损失中
    selection_loss_per_example += tf.where(no_cell_selected, tf.zeros_like(selection_loss_per_example), cell_loss)

    # 将模型选择的列的概率设为0，以确保与选择多列细胞的模型的向后兼容性
    # 选择的列ID是每个细胞预测的列ID中最大值
    selected_column_id = tf.argmax(column_logits, axis=-1, output_type=tf.int32)
    # 创建一个布尔掩码，用于标识是否是模型选择的列
    selected_column_mask = tf.cast(
        tf.equal(column_id_for_cells, tf.expand_dims(selected_column_id, axis=-1)), tf.float32
    )
    # 永远不要选择特殊列ID为0的细胞
    selected_column_mask = tf.where(
        tf.equal(column_id_for_cells, 0), tf.zeros_like(selected_column_mask), selected_column_mask
    )
    # 将logits_per_cell中被选择列的细胞的对数概率设置为接近log(0)的值
    logits_per_cell += CLOSE_ENOUGH_TO_LOG_ZERO * (1.0 - cell_mask * selected_column_mask)
    # 使用cell_index从logits_per_cell中聚合logits
    logits = gather(logits_per_cell, cell_index)

    # 返回每个样本的选择损失和logits
    return selection_loss_per_example, logits
# 计算聚合掩码
def _calculate_aggregate_mask(answer, pooled_output, cell_selection_preference, labels, aggregation_classifier):
    # 找到应该选择没有聚合的单元的示例
    # 返回一个掩码，确定哪些示例应该直接从表格中选择答案，而无需任何聚合函数
    # 如果答案是一段文本，则该情况是明确的，因为聚合函数只适用于数字。如果答案是一个数字但不出现在表格中，那么我们必须使用一些聚合情况。模棱两可的情况是答案是一个在表格中也出现的数字。在这种情况下，我们使用模型预测的聚合函数概率来决定选择还是聚合。这个阈值是一个超参数cell_selection_preference
    # 答案 (形状为`(batch_size, )`的`tf.Tensor`)：每个示例的答案。如果没有标量答案，则为Nan
    # 池化输出 (形状为`(batch_size, hidden_size)`的`tf.Tensor`)：编码器层之上的pooler(BertPooler)的输出
    # cell_selection_preference (`float`)：模糊情况下单元选择的偏好
    # 标签 (形状为`(batch_size, sequence_length)`的`tf.Tensor`)：每个标记的标签。聚合分类器(`torch.nn.Linear`): 聚合头
    # 返回一个聚合掩码，形状为`(batch_size,)`，用于表明哪些示例应该使用聚合函数
    # `tf.Tensor(batch_size,)`
    aggregate_mask_init = tf.cast(tf.logical_not(tf.math.is_nan(answer)), tf.float32)
    logits_aggregation = aggregation_classifier(pooled_output)
    dist_aggregation = tfp.distributions.Categorical(logits=logits_aggregation)
    # 索引0对应于“没有聚合”。
    aggregation_ops_total_mass = tf.reduce_sum(dist_aggregation.probs_parameter()[:, 1:], axis=1)
    # 当前模型的单元选择示例
    is_pred_cell_selection = aggregation_ops_total_mass <= cell_selection_preference
    # 具有非空单元选择监督的示例。
    is_cell_supervision_available = tf.reduce_sum(labels, axis=1) > 0
    aggregate_mask = tf.where(
        tf.logical_and(is_pred_cell_selection, is_cell_supervision_available),
        tf.zeros_like(aggregate_mask_init, dtype=tf.float32),
        aggregate_mask_init,
    )
    aggregate_mask = tf.stop_gradient(aggregate_mask)
    return aggregate_mask
# 当已知其类型时，计算聚合丢失
def _calculate_aggregation_loss_known(
    logits_aggregation, aggregate_mask, aggregation_labels, use_answer_as_supervision, num_aggregation_labels
):
    # 计算训练时聚合类型已知的聚合损失
    # 在弱监督设置中，仅已知信息是对于单元选择示例，为“无聚合”
    # 这段代码是用于计算聚合操作的损失函数，根据设置来判断是否累积损失
    # logits_aggregation: 每个聚合操作的logits，形状为(batch_size, num_aggregation_labels)
    # aggregate_mask: 一个形状为(batch_size, )的掩码，对于需要使用聚合函数的示例设置为1
    # aggregation_labels: 批次中每个示例的聚合函数id，形状为(batch_size, )
    # use_answer_as_supervision: 是否仅使用答案作为聚合示例的监督（可选）
    # num_aggregation_labels: 预测的聚合操作数（可选，默认为0）
    
    def calculate_aggregation_loss(logits_aggregation, aggregate_mask, aggregation_labels,
                                   use_answer_as_supervision=False, num_aggregation_labels=0):
        if use_answer_as_supervision:
            # 为cell selection示例准备“无聚合”目标
            target_aggregation = tf.zeros_like(aggregate_mask, dtype=tf.int32)
        else:
            # 使用聚合监督作为目标
            target_aggregation = aggregation_labels
    
        # 将目标进行独热编码
        one_hot_labels = tf.one_hot(target_aggregation, depth=num_aggregation_labels, dtype=tf.float32)
        # 对logits进行log_softmax
        log_probs = tf.nn.log_softmax(logits_aggregation, axis=-1)
    
        # 计算每个示例的聚合中间损失
        per_example_aggregation_intermediate = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        if use_answer_as_supervision:
            # 仅为需要进行单元选择的示例（无聚合）累积损失
            return per_example_aggregation_intermediate * (1 - aggregate_mask)
        else:
            return per_example_aggregation_intermediate
# 计算在有答案监督的情况下的聚合损失
def _calculate_aggregation_loss_unknown(logits_aggregation, aggregate_mask):
    """
    Calculates aggregation loss in the case of answer supervision.

    Args:
        logits_aggregation (`tf.Tensor` of shape `(batch_size, num_aggregation_labels)`):
            Logits per aggregation operation.
        aggregate_mask (`tf.Tensor` of shape `(batch_size, )`):
            A mask set to 1 for examples that should use aggregation functions

    Returns:
        aggregation_loss_unknown (`tf.Tensor` of shape `(batch_size,)`): Aggregation loss (in case of answer
        supervision) per example.
    """
    # 根据聚合操作的逻辑回归计算分布
    dist_aggregation = tfp.distributions.Categorical(logits=logits_aggregation)
    # 索引0对应“不使用聚合”。
    # 计算聚合操作总概率质量，排除"不使用聚合"的概率
    aggregation_ops_total_mass = tf.reduce_sum(dist_aggregation.probs_parameter()[:, 1:], axis=1)
    # 预测一些聚合操作以应对需要聚合的答案。
    # 这会增加所有聚合函数的概率，类似于最大边际似然估计（MML），但不考虑该函数是否给出正确答案。
    return -tf.math.log(aggregation_ops_total_mass) * aggregate_mask


# 计算每个示例的聚合损失
def _calculate_aggregation_loss(
    logits_aggregation,
    aggregate_mask,
    aggregation_labels,
    use_answer_as_supervision,
    num_aggregation_labels,
    aggregation_loss_weight,
):
    """
    Calculates the aggregation loss per example.

    Args:
        logits_aggregation (`tf.Tensor` of shape `(batch_size, num_aggregation_labels)`):
            Logits per aggregation operation.
        aggregate_mask (`tf.Tensor` of shape `(batch_size, )`):
            A mask set to 1 for examples that should use aggregation functions.
        aggregation_labels (`tf.Tensor` of shape `(batch_size, )`):
            Aggregation function id for every example in the batch.
        use_answer_as_supervision (`bool`, *optional*):
            Whether to use the answer as the only supervision for aggregation examples.
        num_aggregation_labels (`int`, *optional*, defaults to 0):
            The number of aggregation operators to predict.
        aggregation_loss_weight (`float`, *optional*, defaults to 1.0):
            Importance weight for the aggregation loss.

    Returns:
        aggregation_loss (`tf.Tensor` of shape `(batch_size,)`): Aggregation loss per example.
    """
    # 计算已知的聚合损失
    per_example_aggregation_loss = _calculate_aggregation_loss_known(
        logits_aggregation, aggregate_mask, aggregation_labels, use_answer_as_supervision, num_aggregation_labels
    )

    if use_answer_as_supervision:
        # 为需要聚合的数字答案添加聚合损失
        per_example_aggregation_loss += _calculate_aggregation_loss_unknown(logits_aggregation, aggregate_mask)
    return aggregation_loss_weight * per_example_aggregation_loss


def _calculate_expected_result(
    dist_per_cell, numeric_values, numeric_values_scale, input_mask_float, logits_aggregation, config
):
    """
    # 从分布中计算期望结果
    # 根据单元格和聚合概率计算期望结果
    
    # dist_per_cell: 每个单元格的选择分布
    # numeric_values: 每个标记的数值。对于非数值标记为 NaN
    # numeric_values_scale: 每个标记的数值的比例
    # input_mask_float: 表的遮罩，不包括问题标记和表头
    # logits_aggregation: 每种聚合操作的 logits
    # config: 包含模型所有超参数的模型配置类
    
    # 返回期望结果：每个示例的预期结果
    
    if config.use_gumbel_for_cells:
        # 创建Gumbel松弛伯努利分布
        gumbel_dist = tfp.distributions.RelaxedBernoulli(
            # 标记的logits已经通过温度进行除法，并用于计算单元格选择错误，因此需要在此处再次乘以
            config.temperature,
            logits=dist_per_cell.logits_parameter() * config.temperature,
        )
        # 从分布中抽取缩放的概率
        scaled_probability_per_cell = gumbel_dist.sample()
    else:
        # 获取单元格的概率
        scaled_probability_per_cell = dist_per_cell.probs_parameter()
    
    # 将概率根据数值的比例和表的遮罩进行缩放
    scaled_probability_per_cell = (scaled_probability_per_cell / numeric_values_scale) * input_mask_float
    
    # 计算每个示例中的概率和
    count_result = tf.reduce_sum(scaled_probability_per_cell, axis=1)
    
    # 将非数值表值的标记为零
    numeric_values_masked = tf.where(
        tf.math.is_nan(numeric_values), tf.zeros_like(numeric_values), numeric_values
    )
    
    # 计算每个示例中数值表值和的总和
    sum_result = tf.reduce_sum(scaled_probability_per_cell * numeric_values_masked, axis=1)
    
    # 获取平均近似函数的类型
    avg_approximation = config.average_approximation_function
    if avg_approximation == AverageApproximationFunction.RATIO:
        # 计算比率平均值
        average_result = sum_result / (count_result + EPSILON_ZERO_DIVISION)
    elif avg_approximation == AverageApproximationFunction.FIRST_ORDER:
        # 计算第一阶段的平均值
        ex = tf.reduce_sum(scaled_probability_per_cell, axis=1, keepdims=True) - scaled_probability_per_cell + 1
        average_result = tf.reduce_sum(numeric_values_masked * scaled_probability_per_cell / ex, axis=1)
    elif avg_approximation == AverageApproximationFunction.SECOND_ORDER:
        # 如果均值逼近方法是二阶的话
        # 求解除了当前单元格对应的所有概率之和，保持维度，再减去当前单元格的概率加一
        ex = tf.reduce_sum(scaled_probability_per_cell, axis=1, keepdims=True) - scaled_probability_per_cell + 1
        # 逐点乘积计算每个单元格的方差，即概率乘一减概率
        pointwise_var = scaled_probability_per_cell * (1 - scaled_probability_per_cell)
        # 按维度求解方差之和并减去方差
        var = tf.reduce_sum(pointwise_var, axis=1, keepdims=True) - pointwise_var
        # 计算乘数，即（方差除以ex的平方加一）除以ex
        multiplier = (var / tf.math.square(ex) + 1) / ex
        # 按维度求解数值掩码与概率乘乘数的乘积之和
        average_result = tf.reduce_sum(numeric_values_masked * scaled_probability_per_cell * multiplier, axis=1)
    else:
        # 均值逼近方法不合法则抛出异常
        raise ValueError("Invalid average_approximation_function: %s", config.average_approximation_function)

    if config.use_gumbel_for_aggregation:
        # 如果使用 Gumbel 分布进行聚合
        # 创建一个 Gumbel 分布对象，传入聚合温度和剔除了第一个标签的逻辑值
        gumbel_dist = tfp.distributions.RelaxedOneHotCategorical(
            config.aggregation_temperature, logits=logits_aggregation[:, 1:]
        )
        # <float32>[batch_size, num_aggregation_labels - 1]
        # 从 Gumbel 分布中采样得到聚合操作的概率
        aggregation_op_only_probs = gumbel_dist.sample()
    else:
        # 如果不使用 Gumbel 分布进行聚合
        # <float32>[batch_size, num_aggregation_labels - 1]
        # 根据配置使用稳定的 softmax 计算聚合操作的概率
        aggregation_op_only_probs = stable_softmax(logits_aggregation[:, 1:] / config.aggregation_temperature, axis=-1)
    # 将总和结果、均值结果、计数结果沿指定轴进行连接
    all_results = tf.concat(
        [
            tf.expand_dims(sum_result, axis=1),
            tf.expand_dims(average_result, axis=1),
            tf.expand_dims(count_result, axis=1),
        ],
        axis=1,
    )
    # 按维度求解所有结果乘以聚合操作概率之和
    expected_result = tf.reduce_sum(all_results * aggregation_op_only_probs, axis=1)
    # 返回期望结果
    return expected_result
def _calculate_regression_loss(
    answer,  # 答案对于每个样本的张量
    aggregate_mask,  # 对于应该使用聚合函数的样本设置为1的掩码
    dist_per_cell,  # 每个单元格的单元格选择分布
    numeric_values,  # 每个令牌的数值值
    numeric_values_scale,  # 每个令牌的数值值的比例
    input_mask_float,  # 表的掩码，不包括问题令牌和表头
    logits_aggregation,  # 每个聚合操作的logits
    config,  # 包含模型所有参数的模型配置类

    """
    Calculates the regression loss per example.

    Args:
        answer (`tf.Tensor` of shape `(batch_size,)`):
            Answer for every example in the batch. Nan if there is no scalar answer.
        aggregate_mask (`tf.Tensor` of shape `(batch_size,)`):
            A mask set to 1 for examples that should use aggregation functions.
        dist_per_cell (`torch.distributions.Bernoulli`):
            Cell selection distribution for each cell.
        numeric_values (`tf.Tensor` of shape `(batch_size, seq_length)`):
            Numeric values of every token. Nan for tokens which are not numeric values.
        numeric_values_scale (`tf.Tensor` of shape `(batch_size, seq_length)`):
            Scale of the numeric values of every token.
        input_mask_float (`tf.Tensor` of shape `(batch_size, seq_length)`):
            Mask for the table, without question tokens and table headers.
        logits_aggregation (`tf.Tensor` of shape `(batch_size, num_aggregation_labels)`):
            Logits per aggregation operation.
        config ([`TapasConfig`]):
            Model configuration class with all the parameters of the model

    Returns:
        per_example_answer_loss_scaled (`tf.Tensor` of shape `(batch_size,)`): Scales answer loss for each example in
        the batch. large_answer_loss_mask (`tf.Tensor` of shape `(batch_size,)`): A mask which is 1 for examples for
        which their answer loss is larger than the answer_loss_cutoff.
    """
    
    # 计算期望结果
    expected_result = _calculate_expected_result(
        dist_per_cell, numeric_values, numeric_values_scale, input_mask_float, logits_aggregation, config
    )

    # 将含有NaN的答案为0
    answer_masked = tf.where(tf.math.is_nan(answer), tf.zeros_like(answer), answer)

    # 如果使用标准化的答案损失
    if config.use_normalized_answer_loss:
        # 计算标准化的答案损失
        normalizer = tf.stop_gradient(
            tf.math.maximum(tf.math.abs(expected_result), tf.math.abs(answer_masked)) + EPSILON_ZERO_DIVISION
        )
        normalized_answer_masked = answer_masked / normalizer
        normalized_expected_result = expected_result / normalizer
        per_example_answer_loss = tf.compat.v1.losses.huber_loss(
            normalized_answer_masked * aggregate_mask,
            normalized_expected_result * aggregate_mask,
            delta=tf.cast(1.0, tf.float32),
            reduction=tf.losses.Reduction.NONE,
        )
    else:
        # 计算标准的答案损失
        per_example_answer_loss = tf.compat.v1.losses.huber_loss(
            answer_masked * aggregate_mask,
            expected_result * aggregate_mask,
            delta=tf.cast(config.huber_loss_delta, tf.float32),
            reduction=tf.losses.Reduction.NONE,
        )
    # 如果答案损失阈值为None，则创建一个与per_example_answer_loss相同形状的全1张量
    if config.answer_loss_cutoff is None:
        large_answer_loss_mask = tf.ones_like(per_example_answer_loss, dtype=tf.float32)
    # 如果答案损失阈值不为None，则根据条件设置大于阈值的位置为0，小于等于阈值的位置为1
    else:
        large_answer_loss_mask = tf.where(
            per_example_answer_loss > config.answer_loss_cutoff,
            tf.zeros_like(per_example_answer_loss, dtype=tf.float32),
            tf.ones_like(per_example_answer_loss, dtype=tf.float32),
        )
    # 对每个样本的答案损失进行缩放，并乘以聚合掩码
    per_example_answer_loss_scaled = config.answer_loss_importance * (per_example_answer_loss * aggregate_mask)
    # 返回缩放后的答案损失和大于阈值的掩码
    return per_example_answer_loss_scaled, large_answer_loss_mask
```