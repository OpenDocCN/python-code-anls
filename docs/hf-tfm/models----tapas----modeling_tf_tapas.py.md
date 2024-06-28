# `.\models\tapas\modeling_tf_tapas.py`

```
# 设置文件编码为 UTF-8
# 版权声明，指出版权归属于 Google Research 和 HuggingFace Inc. 团队
# 根据 Apache 许可证 2.0 版本授权，除非符合许可证条件，否则不得使用此文件
# 获取许可证的副本，请访问 http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则依据 "AS IS" 原则分发软件，无论是明示还是默示的任何保证或条件都不包括在内
# 请参阅许可证，获取特定语言的详细信息和限制

"""TF 2.0 TAPAS 模型。"""

# 引入必要的库和模块
from __future__ import annotations

import enum  # 引入枚举类型
import math  # 引入数学库函数
from dataclasses import dataclass  # 引入数据类装饰器
from typing import Dict, Optional, Tuple, Union  # 引入类型提示

import numpy as np  # 引入 NumPy 库
import tensorflow as tf  # 引入 TensorFlow 库

# 引入相关的自定义库和模块
from ...activations_tf import get_tf_activation  # 从指定路径引入 TensorFlow 激活函数
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithPastAndCrossAttentions,
    TFBaseModelOutputWithPooling,
    TFMaskedLMOutput,
    TFSequenceClassifierOutput,
)  # 从指定路径引入 TensorFlow 模型输出类
from ...modeling_tf_utils import (
    TFMaskedLanguageModelingLoss,
    TFModelInputType,
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    get_initializer,
    keras,
    keras_serializable,
    unpack_inputs,
)  # 从指定路径引入 TensorFlow 模型相关工具函数
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax  # 从指定路径引入 TensorFlow 相关工具函数
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_tensorflow_probability_available,
    logging,
    replace_return_docstrings,
    requires_backends,
)  # 从指定路径引入通用工具函数和类
from .configuration_tapas import TapasConfig  # 从指定路径引入 Tapas 配置类

# 获取 logger 对象，用于日志记录
logger = logging.get_logger(__name__)

# 软依赖项
# 检查是否导入了 TensorFlow Probability 库
if is_tensorflow_probability_available():
    try:
        import tensorflow_probability as tfp  # 导入 TensorFlow Probability 库
        # 在第一次调用时，检查安装的 TensorFlow 版本是否兼容
        # TensorFlow Probability 依赖于最新的稳定版本的 TensorFlow
        n = tfp.distributions.Normal(loc=0.0, scale=1.0)
    except ImportError:
        # 如果导入失败，则记录错误信息
        logger.error(
            "TAPAS 模型无法使用，因为无法加载 `tensorflow_probability`。"
            "看起来您安装了与 TensorFlow 版本不匹配的 `tensorflow_probability`。"
            "请尝试按照以下说明重新安装：https://github.com/tensorflow/probability。"
        )

# 用于文档的配置和检查点的字符串常量
_CONFIG_FOR_DOC = "TapasConfig"
_CHECKPOINT_FOR_DOC = "google/tapas-base"

# TF TAPAS 预训练模型的存档列表
TF_TAPAS_PRETRAINED_MODEL_ARCHIVE_LIST = [
    # 大模型
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
    # 小模型
    "google/tapas-small",
]
    # 定义一个列表，包含了各种 TAPAS 模型的名称字符串
    models = [
        "google/tapas-small-finetuned-sqa",  # Google TAPAS 小模型，针对 SQA 数据集进行了微调
        "google/tapas-small-finetuned-wtq",  # Google TAPAS 小模型，针对 WTQ 数据集进行了微调
        "google/tapas-small-finetuned-wikisql-supervised",  # Google TAPAS 小模型，针对 Wikisql 数据集进行了监督学习微调
        "google/tapas-small-finetuned-tabfact",  # Google TAPAS 小模型，针对 TabFact 数据集进行了微调
        # 迷你模型
        "google/tapas-mini",  # Google TAPAS 迷你模型
        "google/tapas-mini-finetuned-sqa",  # Google TAPAS 迷你模型，针对 SQA 数据集进行了微调
        "google/tapas-mini-finetuned-wtq",  # Google TAPAS 迷你模型，针对 WTQ 数据集进行了微调
        "google/tapas-mini-finetuned-wikisql-supervised",  # Google TAPAS 迷你模型，针对 Wikisql 数据集进行了监督学习微调
        "google/tapas-mini-finetuned-tabfact",  # Google TAPAS 迷你模型，针对 TabFact 数据集进行了微调
        # 超迷你模型
        "google/tapas-tiny",  # Google TAPAS 超迷你模型
        "google/tapas-tiny-finetuned-sqa",  # Google TAPAS 超迷你模型，针对 SQA 数据集进行了微调
        "google/tapas-tiny-finetuned-wtq",  # Google TAPAS 超迷你模型，针对 WTQ 数据集进行了微调
        "google/tapas-tiny-finetuned-wikisql-supervised",  # Google TAPAS 超迷你模型，针对 Wikisql 数据集进行了监督学习微调
        "google/tapas-tiny-finetuned-tabfact",  # Google TAPAS 超迷你模型，针对 TabFact 数据集进行了微调
        # 查看所有 TAPAS 模型，请访问 https://huggingface.co/models?filter=tapas
    ]
]

# 定义一个全局常量，用于避免零除错误的微小值
EPSILON_ZERO_DIVISION = 1e-10
# 定义一个常量，用于表示接近于负无穷大的值，通常用于表示对数概率为零的情况
CLOSE_ENOUGH_TO_LOG_ZERO = -10000.0


@dataclass
class TFTableQuestionAnsweringOutput(ModelOutput):
    """
    [`TFTapasForQuestionAnswering`]的输出类型。

    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `labels` (and possibly `answer`, `aggregation_labels`, `numeric_values` and `numeric_values_scale` are provided)):
            如果提供了 `labels`（可能还有 `answer`, `aggregation_labels`, `numeric_values` 和 `numeric_values_scale`），则返回总损失，
            包括分层单元选择的对数似然损失，以及（可选的）半监督回归损失和聚合的监督损失。
        logits (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            每个标记的单元选择头的预测分数。
        logits_aggregation (`tf.Tensor`, *optional*, of shape `(batch_size, num_aggregation_labels)`):
            每个聚合操作符的聚合头的预测分数。
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            形状为 `(batch_size, sequence_length, hidden_size)` 的 `tf.Tensor` 元组。
            模型在每个层的输出隐藏状态以及初始嵌入输出。
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            形状为 `(batch_size, num_heads, sequence_length, sequence_length)` 的 `tf.Tensor` 元组。
            在注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。

    """

    loss: tf.Tensor | None = None
    logits: tf.Tensor = None
    logits_aggregation: tf.Tensor | None = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None


class TFTapasEmbeddings(keras.layers.Layer):
    """
    根据词嵌入、位置嵌入和标记类型嵌入构建嵌入。与 BertEmbeddings 相同，但包含多个用于编码表格结构的标记类型嵌入。
    """

    def __init__(self, config: TapasConfig, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.number_of_token_type_embeddings = len(config.type_vocab_sizes)
        self.reset_position_index_per_cell = config.reset_position_index_per_cell
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings
        self.initializer_range = config.initializer_range
        # 创建一个 LayerNormalization 层，用于规范化输入数据
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建一个 Dropout 层，用于随机失活输入单元，防止过拟合
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
    # 在构建模型时，定义输入形状，并添加词嵌入层的权重参数
    def build(self, input_shape=None):
        # 在 "word_embeddings" 命名空间下，创建词嵌入层的权重参数
        with tf.name_scope("word_embeddings"):
            self.weight = self.add_weight(
                name="weight",
                shape=[self.config.vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        # 在 "position_embeddings" 命名空间下，创建位置嵌入层的权重参数
        with tf.name_scope("position_embeddings"):
            self.position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_position_embeddings, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        # 对于每个类型的词汇表大小，分别创建对应的类型嵌入层的权重参数
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

        # 如果模型已经构建，则直接返回，避免重复构建
        if self.built:
            return

        # 标记模型已经构建
        self.built = True

        # 如果存在 LayerNorm 层，则在其命名空间下构建
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                # 根据配置的隐藏大小构建 LayerNorm 层
                self.LayerNorm.build([None, None, self.config.hidden_size])

    # 模型调用方法，接受输入张量并进行处理
    def call(
        self,
        input_ids: tf.Tensor = None,
        position_ids: tf.Tensor = None,
        token_type_ids: tf.Tensor = None,
        inputs_embeds: tf.Tensor = None,
        training: bool = False,
    ) -> tf.Tensor:
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        # Ensure either `input_ids` or `inputs_embeds` is provided
        assert not (input_ids is None and inputs_embeds is None)

        if input_ids is not None:
            # Get the shape of `input_ids`
            input_shape = shape_list(input_ids)
        else:
            # Get the shape of `inputs_embeds` excluding the last dimension
            input_shape = shape_list(inputs_embeds)[:-1]

        # Determine the sequence length from `input_shape`
        seq_length = input_shape[1]

        if token_type_ids is None:
            # If `token_type_ids` is not provided, fill with zeros
            token_type_ids = tf.fill(dims=input_shape + [self.number_of_token_type_embeddings], value=0)

        if position_ids is None:
            # Create absolute position embeddings
            position_ids = tf.expand_dims(tf.range(start=0, limit=seq_length), axis=0)
            position_ids = tf.broadcast_to(position_ids, shape=input_shape)

            # Conditionally create relative position embeddings when `reset_position_index_per_cell` is True
            if self.reset_position_index_per_cell:
                # Calculate column and row indices based on `token_type_ids`
                col_index = IndexMap(token_type_ids[:, :, 1], self.config.type_vocab_sizes[1], batch_dims=1)
                row_index = IndexMap(token_type_ids[:, :, 2], self.config.type_vocab_sizes[2], batch_dims=1)

                # Combine column and row indices to create full index
                full_index = ProductIndexMap(col_index, row_index)

                # Determine the first absolute position for every segment
                first_position_per_segment = reduce_min(position_ids, full_index)[0]

                # Calculate the first absolute position of the cell for every token
                first_position = gather(first_position_per_segment, full_index)

                # Calculate relative positions within the cell, ensuring within bounds
                position = tf.expand_dims(tf.range(start=0, limit=seq_length), axis=0)
                position_ids = tf.math.minimum(self.max_position_embeddings - 1, position - first_position)

        if input_ids is not None:
            # Validate `input_ids` are within bounds of vocabulary size
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            # Gather embeddings based on `input_ids`
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        # Gather position embeddings based on `position_ids`
        position_embeddings = tf.gather(self.position_embeddings, indices=position_ids)

        # Combine input embeddings with position embeddings
        final_embeddings = inputs_embeds + position_embeddings

        # Add token type embeddings for each token type
        for i in range(self.number_of_token_type_embeddings):
            name = f"token_type_embeddings_{i}"
            final_embeddings += tf.gather(params=getattr(self, name), indices=token_type_ids[:, :, i])

        # Apply layer normalization
        final_embeddings = self.LayerNorm(inputs=final_embeddings)

        # Apply dropout during training
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)

        # Return the final embedding tensor
        return final_embeddings
# 从 transformers.models.bert.modeling_tf_bert.TFBertSelfAttention 复制并修改为 Tapas
class TFTapasSelfAttention(keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs):
        super().__init__(**kwargs)

        # 检查隐藏层大小是否能被注意力头数整除
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_attention_heads})"
            )

        # 初始化参数
        self.num_attention_heads = config.num_attention_heads  # 注意力头的数量
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)  # 每个注意力头的大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 所有头部的总大小
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)  # 注意力头大小的平方根

        # 创建查询、键和值的全连接层
        self.query = keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        self.key = keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key"
        )
        self.value = keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )
        self.dropout = keras.layers.Dropout(rate=config.attention_probs_dropout_prob)  # 注意力概率的dropout

        self.is_decoder = config.is_decoder  # 是否为解码器
        self.config = config  # 配置信息

    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        # 将张量重塑从 [batch_size, seq_length, all_head_size] 到 [batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))

        # 转置张量从 [batch_size, seq_length, num_attention_heads, attention_head_size] 到 [batch_size, num_attention_heads, seq_length, attention_head_size]
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
        # 该函数定义层的正向传播逻辑，包括自注意力机制和可选的输出注意力权重
        # （此处省略具体实现细节，不在注释内展开）

    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True

        # 构建查询、键和值的全连接层
        if getattr(self, "query", None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.config.hidden_size])
        if getattr(self, "key", None) is not None:
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.config.hidden_size])
        if getattr(self, "value", None) is not None:
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.config.hidden_size])
# Copied from transformers.models.bert.modeling_tf_bert.TFBertSelfOutput with Bert->Tapas
class TFTapasSelfOutput(keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，用于变换隐藏状态的维度
        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建一个 LayerNormalization 层，用于归一化隐藏状态
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建一个 Dropout 层，用于在训练时随机失活部分神经元
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 将隐藏状态传入全连接层进行维度变换
        hidden_states = self.dense(inputs=hidden_states)
        # 在训练时对输出使用 Dropout
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 使用 LayerNormalization 层归一化隐藏状态并与输入张量相加
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建全连接层，设置输入维度为 config.hidden_size
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 构建 LayerNormalization 层，设置输入维度为 config.hidden_size
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


# Copied from transformers.models.bert.modeling_tf_bert.TFBertAttention with Bert->Tapas
class TFTapasAttention(keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建自注意力层对象
        self.self_attention = TFTapasSelfAttention(config, name="self")
        # 创建输出层对象
        self.dense_output = TFTapasSelfOutput(config, name="output")

    def prune_heads(self, heads):
        # 暂未实现剪枝功能
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
        # 调用自注意力层处理输入
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
        # 将自注意力层的输出传递给输出层处理
        attention_output = self.dense_output(
            hidden_states=self_outputs[0], input_tensor=input_tensor, training=training
        )
        # 如果需要输出注意力权重或过去键值对，则添加到输出元组中
        outputs = (attention_output,) + self_outputs[1:]

        return outputs
    # 定义神经网络层的构建方法，当输入形状为None时表示使用默认形状
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回，避免重复构建
        if self.built:
            return
        # 标记该层已经构建
        self.built = True
        
        # 如果存在self_attention属性，执行以下操作
        if getattr(self, "self_attention", None) is not None:
            # 使用self_attention的名字作为命名空间，开始构建self_attention
            with tf.name_scope(self.self_attention.name):
                self.self_attention.build(None)
        
        # 如果存在dense_output属性，执行以下操作
        if getattr(self, "dense_output", None) is not None:
            # 使用dense_output的名字作为命名空间，开始构建dense_output
            with tf.name_scope(self.dense_output.name):
                self.dense_output.build(None)
# Copied from transformers.models.bert.modeling_tf_bert.TFBertIntermediate with Bert->Tapas
class TFTapasIntermediate(keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs):
        super().__init__(**kwargs)

        # 定义一个全连接层，输出大小为 config.intermediate_size，使用指定的初始化器初始化权重
        self.dense = keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        # 根据配置中的 hidden_act 参数，获取中间激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 将输入的 hidden_states 输入到全连接层 dense 中
        hidden_states = self.dense(inputs=hidden_states)
        # 使用中间激活函数处理全连接层的输出
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                # 构建全连接层 dense，输入大小为 [None, None, self.config.hidden_size]
                self.dense.build([None, None, self.config.hidden_size])


# Copied from transformers.models.bert.modeling_tf_bert.TFBertOutput with Bert->Tapas
class TFTapasOutput(keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs):
        super().__init__(**kwargs)

        # 定义一个全连接层，输出大小为 config.hidden_size，使用指定的初始化器初始化权重
        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 定义 LayerNormalization 层，epsilon 参数为 config.layer_norm_eps
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 定义 Dropout 层，丢弃率为 config.hidden_dropout_prob
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 将输入的 hidden_states 输入到全连接层 dense 中
        hidden_states = self.dense(inputs=hidden_states)
        # 在训练时应用 dropout
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # LayerNormalization 处理全连接层的输出并加上输入 tensor input_tensor
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                # 构建全连接层 dense，输入大小为 [None, None, self.config.intermediate_size]
                self.dense.build([None, None, self.config.intermediate_size])
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                # 构建 LayerNormalization 层，输入大小为 [None, None, self.config.hidden_size]
                self.LayerNorm.build([None, None, self.config.hidden_size])


# Copied from transformers.models.bert.modeling_tf_bert.TFBertLayer with Bert->Tapas
class TFTapasLayer(keras.layers.Layer):
    # 留待实现
    pass
    # 初始化方法，用于创建一个 Tapas 模型对象
    def __init__(self, config: TapasConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 创建一个 TapasAttention 层对象，并命名为 "attention"
        self.attention = TFTapasAttention(config, name="attention")
        
        # 从配置中获取是否为解码器模型的标志
        self.is_decoder = config.is_decoder
        
        # 从配置中获取是否添加跨注意力的标志
        self.add_cross_attention = config.add_cross_attention
        
        # 如果要添加跨注意力，并且当前模型不是解码器模型，则抛出错误
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            
            # 创建一个 TapasAttention 层对象用于跨注意力，并命名为 "crossattention"
            self.crossattention = TFTapasAttention(config, name="crossattention")
        
        # 创建一个 TapasIntermediate 层对象，并命名为 "intermediate"
        self.intermediate = TFTapasIntermediate(config, name="intermediate")
        
        # 创建一个 TapasOutput 层对象，并命名为 "output"
        self.bert_output = TFTapasOutput(config, name="output")

    # 模型调用方法，用于执行模型的前向传播
    def call(
        self,
        hidden_states: tf.Tensor,  # 输入的隐藏状态张量
        attention_mask: tf.Tensor,  # 注意力掩码张量
        head_mask: tf.Tensor,  # 头部掩码张量
        encoder_hidden_states: tf.Tensor | None,  # 编码器的隐藏状态张量（可选）
        encoder_attention_mask: tf.Tensor | None,  # 编码器的注意力掩码张量（可选）
        past_key_value: Tuple[tf.Tensor] | None,  # 过去的键值元组（可选）
        output_attentions: bool,  # 是否输出注意力权重
        training: bool = False,  # 是否处于训练模式，默认为 False
    ) -> Tuple[tf.Tensor]:
        # 定义函数的返回类型为包含单个 TensorFlow 张量的元组
        # 如果有过去的键/值信息，则仅保留解码器自注意力部分的前两个位置
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 使用自注意力层处理隐藏状态，生成自注意力的输出
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
        # 提取自注意力输出中的第一个元素作为注意力输出
        attention_output = self_attention_outputs[0]

        # 如果是解码器模式，则最后一个输出为自注意力缓存的元组
        if self.is_decoder:
            # 输出中排除最后一个元素（自注意力缓存），其余部分为网络层输出
            outputs = self_attention_outputs[1:-1]
            # 提取当前的键/值信息作为解码器的最新键/值信息
            present_key_value = self_attention_outputs[-1]
        else:
            # 输出中排除第一个元素（自注意力输出），保留其余部分（可能包含注意力权重）
            outputs = self_attention_outputs[1:]
        
        cross_attn_present_key_value = None
        # 如果是解码器且有编码器的隐藏状态作为输入
        if self.is_decoder and encoder_hidden_states is not None:
            # 如果模型未包含交叉注意力层，抛出错误
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )
            
            # 从过去的键/值信息中提取交叉注意力层的键/值信息
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 使用交叉注意力层处理自注意力输出，生成交叉注意力的输出
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
            # 提取交叉注意力输出中的第一个元素作为注意力输出
            attention_output = cross_attention_outputs[0]
            # 将交叉注意力输出中的第二个到倒数第二个元素添加到输出中（可能包含注意力权重）
            outputs = outputs + cross_attention_outputs[1:-1]

            # 将交叉注意力输出中的最后一个元素添加到当前键/值信息中
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value
        
        # 使用中间层处理注意力输出，生成中间层输出
        intermediate_output = self.intermediate(hidden_states=attention_output)
        # 使用BERT输出层处理中间层输出和注意力输出，生成网络层输出
        layer_output = self.bert_output(
            hidden_states=intermediate_output, input_tensor=attention_output, training=training
        )
        # 将网络层输出添加到输出元组中
        outputs = (layer_output,) + outputs
        
        # 如果是解码器模式，将当前的键/值信息作为最后一个输出添加到输出元组中
        if self.is_decoder:
            outputs = outputs + (present_key_value,)
        
        # 返回最终的输出元组
        return outputs
    # 构建方法，用于构造模型
    def build(self, input_shape=None):
        # 如果模型已经构建完成，则直接返回
        if self.built:
            return
        # 将模型标记为已构建状态
        self.built = True
        
        # 如果存在 self.attention 属性，则构建注意力层
        if getattr(self, "attention", None) is not None:
            # 使用注意力层的名称作为命名空间，构建注意力层
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        
        # 如果存在 self.intermediate 属性，则构建中间层
        if getattr(self, "intermediate", None) is not None:
            # 使用中间层的名称作为命名空间，构建中间层
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        
        # 如果存在 self.bert_output 属性，则构建 BERT 输出层
        if getattr(self, "bert_output", None) is not None:
            # 使用 BERT 输出层的名称作为命名空间，构建 BERT 输出层
            with tf.name_scope(self.bert_output.name):
                self.bert_output.build(None)
        
        # 如果存在 self.crossattention 属性，则构建交叉注意力层
        if getattr(self, "crossattention", None) is not None:
            # 使用交叉注意力层的名称作为命名空间，构建交叉注意力层
            with tf.name_scope(self.crossattention.name):
                self.crossattention.build(None)
# 从 transformers.models.bert.modeling_tf_bert.TFBertEncoder 复制并修改为 Tapas 模型的编码器类 TFTapasEncoder
class TFTapasEncoder(keras.layers.Layer):
    # 初始化方法，接受 TapasConfig 类型的配置参数 config
    def __init__(self, config: TapasConfig, **kwargs):
        super().__init__(**kwargs)
        # 保存传入的配置参数
        self.config = config
        # 创建多个 Tapas 层组成的列表，每一层命名为 "layer_._{i}"
        self.layer = [TFTapasLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]

    # 调用方法，定义了模型的前向传播逻辑
    def call(
        self,
        hidden_states: tf.Tensor,                          # 输入的隐藏状态张量
        attention_mask: tf.Tensor,                         # 注意力掩码张量
        head_mask: tf.Tensor,                              # 头部掩码张量
        encoder_hidden_states: tf.Tensor | None,           # 编码器的隐藏状态张量或空值
        encoder_attention_mask: tf.Tensor | None,           # 编码器的注意力掩码张量或空值
        past_key_values: Tuple[Tuple[tf.Tensor]] | None,    # 历史键值对的元组或空值
        use_cache: Optional[bool],                         # 是否使用缓存的布尔值，可选
        output_attentions: bool,                           # 是否输出注意力张量的布尔值
        output_hidden_states: bool,                        # 是否输出隐藏状态的布尔值
        return_dict: bool,                                 # 是否返回字典类型的布尔值
        training: bool = False,                            # 是否处于训练模式的布尔值，默认为 False
    ) -> Union[TFBaseModelOutputWithPastAndCrossAttentions, Tuple[tf.Tensor]]:
        # 初始化存储所有隐藏状态、注意力张量和跨层注意力张量的元组
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 如果使用缓存，则初始化下一个解码器缓存的元组
        next_decoder_cache = () if use_cache else None

        # 遍历每一层 Tapas 层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到所有隐藏状态的元组中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的历史键值对
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 调用当前 Tapas 层的前向传播方法，获取该层的输出
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
            # 更新隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]

            # 如果使用缓存，将当前层的输出的最后一个元素添加到下一个解码器缓存的元组中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            # 如果输出注意力张量，则将当前层的输出的第二个元素添加到所有注意力张量的元组中
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
                # 如果配置要求添加跨层注意力，并且编码器的隐藏状态不为空，则将当前层的输出的第三个元素添加到所有跨层注意力张量的元组中
                if self.config.add_cross_attention and encoder_hidden_states is not None:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果需要输出隐藏状态，则将最后一层的隐藏状态添加到所有隐藏状态的元组中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典类型的结果，则返回所有非空的张量组成的元组
        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_attentions, all_cross_attentions] if v is not None
            )

        # 返回 TFBaseModelOutputWithPastAndCrossAttentions 类型的字典结果
        return TFBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )
    # 定义神经网络模型的构建方法，接受输入形状参数，默认为None
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回，不再重复构建
        if self.built:
            return
        # 将模型标记为已构建状态
        self.built = True
        # 检查是否存在名为"layer"的属性
        if getattr(self, "layer", None) is not None:
            # 遍历模型中的每一层
            for layer in self.layer:
                # 使用层的名称为当前层创建一个命名空间
                with tf.name_scope(layer.name):
                    # 调用每一层的build方法，传入输入形状参数为None，表示根据需要自动确定输入形状
                    layer.build(None)
# 从transformers.models.bert.modeling_tf_bert.TFBertPooler复制并修改为Tapas
class TFTapasPooler(keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，用于池化模型的隐藏状态，输出维度为config.hidden_size
        self.dense = keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 通过仅仅取第一个标记的隐藏状态来“池化”模型
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(inputs=first_token_tensor)

        return pooled_output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                # 构建dense层，输入维度为[None, None, self.config.hidden_size]
                self.dense.build([None, None, self.config.hidden_size])


# 从transformers.models.bert.modeling_tf_bert.TFBertPredictionHeadTransform复制并修改为Tapas
class TFTapasPredictionHeadTransform(keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，输出维度为config.hidden_size
        self.dense = keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense",
        )

        # 根据config.hidden_act初始化激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.transform_act_fn = config.hidden_act

        # 创建LayerNormalization层，epsilon值为config.layer_norm_eps
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 通过dense层处理hidden_states
        hidden_states = self.dense(inputs=hidden_states)
        # 应用激活函数transform_act_fn
        hidden_states = self.transform_act_fn(hidden_states)
        # 应用LayerNormalization层
        hidden_states = self.LayerNorm(inputs=hidden_states)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                # 构建dense层，输入维度为[None, None, self.config.hidden_size]
                self.dense.build([None, None, self.config.hidden_size])
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                # 构建LayerNorm层，输入维度为[None, None, self.config.hidden_size]
                self.LayerNorm.build([None, None, self.config.hidden_size])


# 从transformers.models.bert.modeling_tf_bert.TFBertLMPredictionHead复制并修改为Tapas
class TFTapasLMPredictionHead(keras.layers.Layer):
    # 使用 TapasConfig 和输入的嵌入层初始化模型
    def __init__(self, config: TapasConfig, input_embeddings: keras.layers.Layer, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 保存配置对象和隐藏层大小
        self.config = config
        self.hidden_size = config.hidden_size

        # 创建预测头转换层，用于处理模型的输出
        self.transform = TFTapasPredictionHeadTransform(config, name="transform")

        # 输入嵌入层是模型的输入
        self.input_embeddings = input_embeddings

    # 构建模型
    def build(self, input_shape=None):
        # 添加偏置项，初始化为全零向量
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="bias")

        # 如果已经构建过模型，直接返回
        if self.built:
            return
        self.built = True
        
        # 如果存在转换层，构建转换层
        if getattr(self, "transform", None) is not None:
            with tf.name_scope(self.transform.name):
                self.transform.build(None)

    # 获取输出嵌入层
    def get_output_embeddings(self) -> keras.layers.Layer:
        return self.input_embeddings

    # 设置输出嵌入层的权重和词汇表大小
    def set_output_embeddings(self, value: tf.Variable):
        self.input_embeddings.weight = value
        self.input_embeddings.vocab_size = shape_list(value)[0]

    # 获取偏置项
    def get_bias(self) -> Dict[str, tf.Variable]:
        return {"bias": self.bias}

    # 设置偏置项
    def set_bias(self, value: tf.Variable):
        self.bias = value["bias"]
        # 更新配置对象的词汇表大小
        self.config.vocab_size = shape_list(value["bias"])[0]

    # 模型的调用方法，进行前向传播
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 使用转换层处理隐藏状态
        hidden_states = self.transform(hidden_states=hidden_states)
        
        # 获取序列长度
        seq_length = shape_list(hidden_states)[1]
        
        # 将隐藏状态重新形状为二维张量
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.hidden_size])
        
        # 矩阵乘法计算输出
        hidden_states = tf.matmul(a=hidden_states, b=self.input_embeddings.weight, transpose_b=True)
        
        # 将输出重新形状为三维张量
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])
        
        # 添加偏置项到输出
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)

        # 返回处理后的隐藏状态
        return hidden_states
# 从transformers.models.bert.modeling_tf_bert.TFBertMLMHead复制而来，将Bert替换为Tapas
class TFTapasMLMHead(keras.layers.Layer):
    def __init__(self, config: TapasConfig, input_embeddings: keras.layers.Layer, **kwargs):
        super().__init__(**kwargs)

        # 使用TapasLMPredictionHead类创建predictions对象
        self.predictions = TFTapasLMPredictionHead(config, input_embeddings, name="predictions")

    # 调用函数，根据输入的sequence_output计算预测分数prediction_scores
    def call(self, sequence_output: tf.Tensor) -> tf.Tensor:
        prediction_scores = self.predictions(hidden_states=sequence_output)

        return prediction_scores

    # 构建函数，在第一次调用时建立层的内部结构
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在predictions对象，则在其命名空间下建立结构
        if getattr(self, "predictions", None) is not None:
            with tf.name_scope(self.predictions.name):
                self.predictions.build(None)


# keras_serializable装饰器用于声明TFTapasMainLayer类是可序列化的
@keras_serializable
class TFTapasMainLayer(keras.layers.Layer):
    config_class = TapasConfig

    def __init__(self, config: TapasConfig, add_pooling_layer: bool = True, **kwargs):
        # 调用父类的初始化函数，并添加tensorflow_probability作为后端库的要求
        requires_backends(self, "tensorflow_probability")
        super().__init__(**kwargs)

        self.config = config

        # 创建TFTapasEmbeddings对象作为embeddings
        self.embeddings = TFTapasEmbeddings(config, name="embeddings")
        # 创建TFTapasEncoder对象作为encoder
        self.encoder = TFTapasEncoder(config, name="encoder")
        # 如果add_pooling_layer为True，则创建TFTapasPooler对象作为pooler
        self.pooler = TFTapasPooler(config, name="pooler") if add_pooling_layer else None

    # 返回embeddings对象，用于获取输入的嵌入层
    def get_input_embeddings(self) -> keras.layers.Layer:
        return self.embeddings

    # 设置输入的嵌入层的权重值为value，并更新vocab_size属性
    def set_input_embeddings(self, value: tf.Variable):
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    # _prune_heads函数用于修剪模型中的注意力头
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    # call函数，接收多个输入参数，并根据配置调用embeddings、encoder和pooler的对应方法
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
    ):
        # 这里是call函数的具体实现，根据输入参数调用对应的功能

    # 构建函数，在第一次调用时建立层的内部结构
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在embeddings对象，则在其命名空间下建立结构
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        # 如果存在encoder对象，则在其命名空间下建立结构
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 如果存在pooler对象，则在其命名空间下建立结构
        if getattr(self, "pooler", None) is not None:
            with tf.name_scope(self.pooler.name):
                self.pooler.build(None)
"""
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
"""
    Parameters:
        config ([`TapasConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
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

        # 初始化Tapas主层，使用给定的配置参数
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
        前向传播函数，接受多种输入参数，并返回模型的输出。

        Args:
            input_ids: 输入的token IDs。
            attention_mask: 注意力掩码，指示哪些位置是padding的。
            token_type_ids: token类型IDs，用于区分segment。
            position_ids: 位置IDs，用于指定每个token在文本中的位置。
            head_mask: 头部掩码，用于指定哪些注意力头部被屏蔽。
            inputs_embeds: 嵌入的输入张量。
            output_attentions: 是否输出注意力权重。
            output_hidden_states: 是否输出所有隐藏状态。
            return_dict: 是否返回字典格式的输出。
            training: 是否为训练模式。

        Returns:
            模型的输出，可以是包含池化的基础模型输出或者张量的元组。

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
        ```
        """
        # 调用Tapas主层处理输入，返回处理结果
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
# TapasForMaskedLM 类继承自 TFTapasPreTrainedModel 和 TFMaskedLanguageModelingLoss，用于处理 Tapas 模型的 Masked Language Modeling 任务
class TFTapasForMaskedLM(TFTapasPreTrainedModel, TFMaskedLanguageModelingLoss):
    
    # 初始化方法，接受一个 TapasConfig 对象和额外的输入参数
    def __init__(self, config: TapasConfig, *inputs, **kwargs):
        # 调用父类的初始化方法，传入配置和其他输入参数
        super().__init__(config, *inputs, **kwargs)
        
        # 如果配置指定为 decoder，发出警告，建议将 `config.is_decoder` 设为 False，以支持双向自注意力
        if config.is_decoder:
            logger.warning(
                "If you want to use `TFTapasForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )
        
        # 创建 Tapas 主层对象，关闭添加池化层选项，命名为 "tapas"
        self.tapas = TFTapasMainLayer(config, add_pooling_layer=False, name="tapas")
        
        # 创建 Tapas MLM 头部对象，传入输入嵌入层为 tapas 的嵌入层，命名为 "cls"
        self.lm_head = TFTapasMLMHead(config, input_embeddings=self.tapas.embeddings, name="cls")
    
    # 获取 MLM 头部的方法，返回 lm_head 的 predictions 属性
    def get_lm_head(self) -> keras.layers.Layer:
        return self.lm_head.predictions
    
    # call 方法，处理模型的前向传播
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
        # 带有注释的参数列表，定义输入和控制模型行为的选项
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
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
        # 获取模型的序列输出
        sequence_output = outputs[0]
        # 将序列输出传递给语言模型头部以预测分数
        prediction_scores = self.lm_head(sequence_output)
        # 如果提供了标签，则计算损失；否则损失为 None
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=prediction_scores)

        # 如果 return_dict 为 False，则组织输出格式
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 构建 TFMaskedLMOutput 对象，封装损失、预测分数、隐藏状态和注意力权重
        return TFMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果存在 Tapas 模型，则构建 Tapas 模型
        if getattr(self, "tapas", None) is not None:
            with tf.name_scope(self.tapas.name):
                self.tapas.build(None)
        # 如果存在语言模型头部，则构建语言模型头部
        if getattr(self, "lm_head", None) is not None:
            with tf.name_scope(self.lm_head.name):
                self.lm_head.build(None)
# 定义一个 TensorFlow 自定义层 TFTapasComputeTokenLogits，用于计算每个标记的逻辑回归结果
class TFTapasComputeTokenLogits(keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs):
        super().__init__(**kwargs)

        # 从配置中获取温度参数，用于调节逻辑回归的温度
        self.temperature = config.temperature

        # 定义输出层权重和偏置，这些权重用于计算逻辑回归
        with tf.name_scope("output"):
            self.output_weights = self.add_weight(
                name="output_weights",
                shape=(config.hidden_size,),
                dtype=tf.float32,
                trainable=True,
                # 根据配置选择初始化输出权重为零或截断正态分布
                initializer=tf.zeros_initializer()
                if config.init_cell_selection_weights_to_zero
                else keras.initializers.TruncatedNormal(stddev=config.initializer_range),
            )
            self.output_bias = self.add_weight(
                name="output_bias", shape=(), trainable=True, initializer=tf.zeros_initializer()
            )

    # 定义调用函数，输入是序列输出张量，输出是每个标记的逻辑回归结果张量
    def call(self, sequence_output: tf.Tensor) -> tf.Tensor:
        """
        计算每个标记的逻辑回归结果

        Args:
            sequence_output (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                也称为 last_hidden_state。模型最后一层的隐藏状态序列输出。

        Returns:
            logits (`tf.Tensor` of shape `(batch_size, sequence_length)`): 每个标记的逻辑回归结果。
        """
        # 计算逻辑回归结果，通过张量乘法和偏置，然后除以温度参数
        logits = (tf.einsum("bsj,j->bs", sequence_output, self.output_weights) + self.output_bias) / self.temperature
        return logits


# 定义另一个 TensorFlow 自定义层 TFTapasComputeColumnLogits，用于计算每列的逻辑回归结果
class TFTapasComputeColumnLogits(keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs):
        super().__init__(**kwargs)

        # 定义列输出层的权重和偏置，用于计算列的逻辑回归
        with tf.name_scope("column_output"):
            self.column_output_weights = self.add_weight(
                name="column_output_weights",
                shape=[config.hidden_size],
                dtype=tf.float32,
                trainable=True,
                # 根据配置选择初始化输出权重为零或截断正态分布
                initializer=tf.zeros_initializer()
                if config.init_cell_selection_weights_to_zero
                else keras.initializers.TruncatedNormal(stddev=config.initializer_range),
            )
            self.column_output_bias = self.add_weight(
                name="column_output_bias", shape=(), trainable=True, initializer=tf.zeros_initializer()
            )
    # 计算列的逻辑回归结果

    # 首先，计算没有温度调节的令牌逻辑回归结果 (batch_size, seq_len)
    token_logits = tf.einsum("bsj,j->bs", sequence_output, self.column_output_weights) + self.column_output_bias

    # 接下来，对每个单元格平均逻辑回归结果 (batch_size, max_num_cols*max_num_rows)
    cell_logits, cell_logits_index = reduce_mean(token_logits, cell_index)

    # 最后，对每列平均逻辑回归结果 (batch_size, max_num_cols)
    column_index = cell_index.project_inner(cell_logits_index)
    column_logits, out_index = reduce_sum(cell_logits * cell_mask, column_index)

    # 计算每列的单元格数目，避免零除错误
    cell_count, _ = reduce_sum(cell_mask, column_index)
    column_logits /= cell_count + EPSILON_ZERO_DIVISION

    # 掩盖不出现在示例中的列
    is_padding = tf.logical_and(cell_count < 0.5, tf.not_equal(out_index.indices, 0))
    column_logits += CLOSE_ENOUGH_TO_LOG_ZERO * tf.cast(is_padding, tf.float32)

    # 如果不允许选择空列，进一步掩盖选择了空列的情况
    if not allow_empty_column_selection:
        column_logits += CLOSE_ENOUGH_TO_LOG_ZERO * tf.cast(tf.equal(out_index.indices, 0), tf.float32)

    return column_logits
@add_start_docstrings(
    """
    Tapas Model with a cell selection head and optional aggregation head on top for question-answering tasks on tables
    (linear layers on top of the hidden-states output to compute `logits` and optional `logits_aggregation`), e.g. for
    SQA, WTQ or WikiSQL-supervised tasks.
    """,
    TAPAS_START_DOCSTRING,
)
class TFTapasForQuestionAnswering(TFTapasPreTrainedModel):
    def __init__(self, config: TapasConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # base model
        self.tapas = TFTapasMainLayer(config, name="tapas")

        # dropout
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)

        # initialize layer to compute token-level logits
        self.compute_token_logits = TFTapasComputeTokenLogits(config, name="compute_token_logits")

        # initialize layer to compute column-level logits
        self.compute_column_logits = TFTapasComputeColumnLogits(config, name="compute_column_logits")

        # optional aggregation classifier if specified in the configuration
        if config.num_aggregation_labels > 0:
            self.aggregation_classifier = keras.layers.Dense(
                config.num_aggregation_labels,
                kernel_initializer=get_initializer(config.initializer_range),
                name="aggregation_classifier",
            )
        self.config = config

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
    # 如果已经构建过网络结构，则直接返回，避免重复构建
    if self.built:
        return
    # 设置标志表示网络结构已经构建
    self.built = True

    # 如果存在名为 "tapas" 的属性，则构建其对应的子模型
    if getattr(self, "tapas", None) is not None:
        # 使用 "tapas" 的名称作为命名空间，构建子模型
        with tf.name_scope(self.tapas.name):
            self.tapas.build(None)

    # 如果存在名为 "compute_token_logits" 的属性，则构建其对应的子模型
    if getattr(self, "compute_token_logits", None) is not None:
        # 使用 "compute_token_logits" 的名称作为命名空间，构建子模型
        with tf.name_scope(self.compute_token_logits.name):
            self.compute_token_logits.build(None)

    # 如果存在名为 "compute_column_logits" 的属性，则构建其对应的子模型
    if getattr(self, "compute_column_logits", None) is not None:
        # 使用 "compute_column_logits" 的名称作为命名空间，构建子模型
        with tf.name_scope(self.compute_column_logits.name):
            self.compute_column_logits.build(None)

    # 如果存在名为 "aggregation_classifier" 的属性，则构建其对应的子模型
    if getattr(self, "aggregation_classifier", None) is not None:
        # 使用 "aggregation_classifier" 的名称作为命名空间，构建子模型
        with tf.name_scope(self.aggregation_classifier.name):
            # 构建 "aggregation_classifier" 子模型，输入维度为 [None, None, self.config.hidden_size]
            self.aggregation_classifier.build([None, None, self.config.hidden_size])
# 使用自定义的装饰器添加文档字符串，描述 Tapas 模型用于序列分类任务的结构和功能
@add_start_docstrings(
    """
    Tapas Model with a sequence classification head on top (a linear layer on top of the pooled output), e.g. for table
    entailment tasks, such as TabFact (Chen et al., 2020).
    """,
    TAPAS_START_DOCSTRING,
)
# 定义 TFTapasForSequenceClassification 类，继承自 TFTapasPreTrainedModel 和 TFSequenceClassificationLoss
class TFTapasForSequenceClassification(TFTapasPreTrainedModel, TFSequenceClassificationLoss):
    
    # 初始化方法，接受 TapasConfig 对象和其他输入参数
    def __init__(self, config: TapasConfig, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)
        # 设置分类任务的标签数量
        self.num_labels = config.num_labels

        # 创建 Tapas 主层对象，命名为 "tapas"
        self.tapas = TFTapasMainLayer(config, name="tapas")
        # 创建 Dropout 层，使用配置中的隐藏层 Dropout 概率，命名为 "dropout"
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob, name="dropout")
        # 创建 Dense 层作为分类器，输出大小为标签数量，使用指定的初始化器范围初始化，命名为 "classifier"
        self.classifier = keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        # 将配置对象保存在实例中
        self.config = config

    # 使用装饰器将函数添加到模型前向传播路径中，并添加相关文档字符串描述输入格式
    @unpack_inputs
    @add_start_docstrings_to_model_forward(TAPAS_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @replace_return_docstrings(output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    # 定义模型的前向传播方法，接受多种输入参数
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
        # 调用模型的tapas方法进行前向传播，获取模型输出
        pooled_output = outputs[1]
        # 从模型输出中获取池化后的特征表示
        pooled_output = self.dropout(inputs=pooled_output, training=training)
        # 对池化后的特征表示进行dropout处理
        logits = self.classifier(inputs=pooled_output)
        # 使用分类器对特征表示进行分类得到logits
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)
        # 如果提供了标签，计算损失函数，否则损失为None

        if not return_dict:
            # 如果不要求返回字典格式的输出
            output = (logits,) + outputs[2:]
            # 组装输出元组，包括logits和可能的其他输出
            return ((loss,) + output) if loss is not None else output
            # 返回包含损失和输出元组的结果，如果损失为None则只返回输出元组

        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # 返回TFSequenceClassifierOutput对象，包含损失、logits、隐藏状态和注意力信息
    # 构建方法，用于构建模型的各个组件的计算图
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回，避免重复构建
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        
        # 如果存在 tapas 属性，则构建 tapas 组件的计算图
        if getattr(self, "tapas", None) is not None:
            # 使用 tapas 组件的名称作为命名空间
            with tf.name_scope(self.tapas.name):
                # 调用 tapas 组件的 build 方法，传入 None 作为输入形状
                self.tapas.build(None)
        
        # 如果存在 dropout 属性，则构建 dropout 组件的计算图
        if getattr(self, "dropout", None) is not None:
            # 使用 dropout 组件的名称作为命名空间
            with tf.name_scope(self.dropout.name):
                # 调用 dropout 组件的 build 方法，传入 None 作为输入形状
                self.dropout.build(None)
        
        # 如果存在 classifier 属性，则构建 classifier 组件的计算图
        if getattr(self, "classifier", None) is not None:
            # 使用 classifier 组件的名称作为命名空间
            with tf.name_scope(self.classifier.name):
                # 调用 classifier 组件的 build 方法，传入 [None, None, self.config.hidden_size] 作为输入形状
                self.classifier.build([None, None, self.config.hidden_size])
""" TAPAS utilities."""

# 定义一个枚举类，表示平均近似函数的不同类型
class AverageApproximationFunction(str, enum.Enum):
    RATIO = "ratio"         # 比率
    FIRST_ORDER = "first_order"   # 一阶
    SECOND_ORDER = "second_order"   # 二阶


# 与分段张量相关的所有内容的起点


class IndexMap(object):
    """Index grouping entries within a tensor."""

    def __init__(self, indices, num_segments, batch_dims=0):
        """
        Creates an index.

        Args:
          indices: <int32> Tensor of indices, same shape as `values`.
                  索引的张量，类型为<int32>，形状与`values`相同。
          num_segments: <int32> Scalar tensor, the number of segments. All elements
                        in a batched segmented tensor must have the same number of segments (although many segments can be empty).
                        分段张量的段数，作为一个标量张量。批处理的分段张量中的所有元素必须具有相同数量的段（尽管许多段可以为空）。
          batch_dims: Python integer, the number of batch dimensions. The first
                      `batch_dims` dimensions of a SegmentedTensor are treated as batch dimensions. Segments in different batch
                      elements are always distinct even if they have the same index.
                      批处理维度的数量，作为一个整数。分段张量的前`batch_dims`个维度被视为批处理维度。不同批处理元素中的段始终是不同的，即使它们具有相同的索引。
        """
        self.indices = tf.convert_to_tensor(indices)
        self.num_segments = tf.convert_to_tensor(num_segments)
        self.batch_dims = batch_dims

    def batch_shape(self):
        return tf.shape(self.indices)[: self.batch_dims]


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
          outer_index: IndexMap.
                      外部索引，类型为IndexMap。
          inner_index: IndexMap, must have the same shape as `outer_index`.
                      内部索引，类型为IndexMap，必须与`outer_index`具有相同的形状。
        """
        if outer_index.batch_dims != inner_index.batch_dims:
            raise ValueError("outer_index.batch_dims and inner_index.batch_dims must be the same.")

        super(ProductIndexMap, self).__init__(
            indices=(
                inner_index.indices
                + outer_index.indices * tf.cast(inner_index.num_segments, inner_index.indices.dtype)
            ),
            num_segments=inner_index.num_segments * outer_index.num_segments,
            batch_dims=inner_index.batch_dims,
        )
        self.outer_index = outer_index
        self.inner_index = inner_index

    def project_outer(self, index):
        """Projects an index with the same index set onto the outer components."""
        return IndexMap(
            indices=tf.math.floordiv(index.indices, self.inner_index.num_segments),
            num_segments=self.outer_index.num_segments,
            batch_dims=index.batch_dims,
        )
    # 定义一个方法 `project_inner`，用于对传入的索引对象进行投影操作
    def project_inner(self, index):
        """Projects an index with the same index set onto the inner components."""
        # 使用 TensorFlow 的数学函数 `floormod` 对索引对象的 indices 属性进行取模运算
        # 以确保索引值不超过内部索引的段数，从而实现投影操作
        return IndexMap(
            indices=tf.math.floormod(index.indices, self.inner_index.num_segments),
            # 设置投影后的索引段数为内部索引的段数
            num_segments=self.inner_index.num_segments,
            # 将索引对象的批次维度（batch_dims）直接传递到新的 IndexMap 对象中
            batch_dims=index.batch_dims,
        )
# 使用 TensorFlow 提供的 tf.gather 函数，根据给定的索引从 values 中收集数据，index.indices 是索引的列表，
# batch_dims 指定批次维度的数量，name 是操作的名称
def gather(values, index, name="segmented_gather"):
    return tf.gather(values, index.indices, batch_dims=index.batch_dims, name=name)


# 将批处理的索引映射压平成一维索引映射。这个操作重新标记段，以保持批处理元素的不同性。
# 第 k 个批处理元素的索引会偏移 `num_segments` * (k - 1)。结果是一个张量，其大小是 `num_segments` 乘以批处理元素的数量。
def flatten(index, name="segmented_flatten"):
    batch_size = tf.reduce_prod(index.batch_shape())
    offset = tf.range(batch_size) * index.num_segments
    offset = tf.reshape(offset, index.batch_shape())
    for _ in range(index.batch_dims, index.indices.shape.rank):
        offset = tf.expand_dims(offset, -1)

    indices = tf.cast(offset, index.indices.dtype) + index.indices
    return IndexMap(indices=tf.reshape(indices, [-1]), num_segments=index.num_segments * batch_size, batch_dims=0)


# 构造一个索引映射，其值等于 range(num_segments)。
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


# 应用段内的分段减少功能。
# 此函数尚未完全定义，将在后续代码中继续定义。
def _segment_reduce(values, index, segment_reduce_fn, name):
    """
    Args:
        values (`tf.Tensor`):
            Tensor with segment values.  # 输入参数，包含分段数值的张量
        index (`IndexMap`):
            IndexMap.  # 输入参数，索引映射对象
        segment_reduce_fn (`str`):
            Name for the reduce operation. One of "sum", "mean", "max" or "min".  # 输入参数，指定分段操作的类型，可以是"sum"、"mean"、"max"或"min"
        name (`str`):
            Name for the operation. Currently not used  # 输入参数，操作的名称，目前未使用

    Returns:
        (`IndexMap`): IndexMap of shape batch_shape with elements equal to range(num_segments).
        # 返回值，返回形状为 batch_shape 的 IndexMap 对象，其元素等于范围内的 num_segments。

    """
    # Flatten the batch dimensions, as segments ops do not support batching.
    # However if `values` has extra dimensions to the right keep them
    # unflattened. Segmented ops support vector-valued operations.
    # 将批处理维度展平，因为分段操作不支持批处理。
    # 如果 `values` 右侧有额外的维度，则保持它们不展平。分段操作支持矢量值操作。
    flat_index = flatten(index)
    vector_shape = tf.shape(values)[index.indices.shape.rank :]
    flattened_shape = tf.concat([[-1], vector_shape], axis=0)
    flat_values = tf.reshape(values, flattened_shape)
    segment_means = segment_reduce_fn(
        data=flat_values, segment_ids=flat_index.indices, num_segments=flat_index.num_segments
    )

    # Unflatten the values.
    # 将值重新展开。
    new_shape = tf.concat([index.batch_shape(), [index.num_segments], vector_shape], axis=0)
    output_values = tf.reshape(segment_means, new_shape)
    output_index = range_index_map(index.batch_shape(), index.num_segments)
    return output_values, output_index
def reduce_mean(values, index, name="segmented_reduce_mean"):
    """
    Averages a tensor over its segments. Outputs 0 for empty segments. This operations computes the mean over segments,
    with support for:

      - Batching using the first dimensions [B1, B2, ..., Bn]. Each element in a batch can have different indices.
      - Vectorization using the last dimension [V1, V2, ...]. If they are present the output will be a mean of vectors
        rather than scalars.
    Only the middle dimensions [I1, ..., Ik] are reduced by the operation.

    Args:
      values: [B1, B2, ..., Bn, I1, .., Ik, V1, V2, ..] tensor of values to be
        averaged.
      index: IndexMap [B1, B2, ..., Bn, I1, .., Ik] index defining the segments.
      name: Name for the TensorFlow ops.

    Returns:
      A pair (output_values, output_index) where `output_values` is a tensor of shape [B1, B2, ..., Bn, num_segments,
      V1, V2, ..] and `index` is an IndexMap with shape [B1, B2, ..., Bn, num_segments].
    """
    return _segment_reduce(values, index, tf.math.unsorted_segment_mean, name)



def reduce_sum(values, index, name="segmented_reduce_sum"):
    """
    Sums a tensor over its segments. Outputs 0 for empty segments. This operations computes the sum over segments, with
    support for:

      - Batching using the first dimensions [B1, B2, ..., Bn]. Each element in a batch can have different indices.
      - Vectorization using the last dimension [V1, V2, ...]. If they are present the output will be a sum of vectors
        rather than scalars.
    Only the middle dimensions [I1, ..., Ik] are reduced by the operation.

    Args:
      values: [B1, B2, ..., Bn, I1, .., Ik, V1, V2, ..] tensor of values to be
        averaged.
      index: IndexMap [B1, B2, ..., Bn, I1, .., Ik] index defining the segments.
      name: Name for the TensorFlow ops.

    Returns:
      A pair (output_values, output_index) where `output_values` is a tensor of shape [B1, B2, ..., Bn, num_segments,
      V1, V2, ..] and `index` is an IndexMap with shape [B1, B2, ..., Bn, num_segments].
    """
    return _segment_reduce(values, index, tf.math.unsorted_segment_sum, name)



def reduce_max(values, index, name="segmented_reduce_max"):
    """
    Computes the maximum over segments. This operations computes the maximum over segments, with support for:

      - Batching using the first dimensions [B1, B2, ..., Bn]. Each element in a batch can have different indices.
      - Vectorization using the last dimension [V1, V2, ...]. If they are present the output will be an element-wise
        maximum of vectors rather than scalars.
    Only the middle dimensions [I1, ..., Ik] are reduced by the operation.

    Args:
      values: [B1, B2, ..., Bn, I1, .., Ik, V1, V2, ..] tensor of values to be
        averaged.
      index: IndexMap [B1, B2, ..., Bn, I1, .., Ik] index defining the segments.
      name: Name for the TensorFlow ops.
    """
    # 使用 TensorFlow 的 unsorted_segment_max 函数对给定的 values 和 index 进行分段最大值计算
    return _segment_reduce(values, index, tf.math.unsorted_segment_max, name)
    # 调用私有函数 _segment_reduce，执行分段归约操作，使用 tf.math.unsorted_segment_max 函数进行归约
    # 函数返回一个元组 (output_values, output_index)，其中 output_values 是形状为 [B1, B2, ..., Bn, num_segments, V1, V2, ..] 的张量
    # output_index 是形状为 [B1, B2, ..., Bn, num_segments] 的索引映射对象 IndexMap
    return _segment_reduce(values, index, tf.math.unsorted_segment_max, name)
def reduce_min(values, index, name="segmented_reduce_min"):
    """Computes the minimum over segments."""
    # 调用内部函数 _segment_reduce 来实现分段最小值计算，使用 tf.math.unsorted_segment_min 方法
    return _segment_reduce(values, index, tf.math.unsorted_segment_min, name)


def _single_column_cell_selection_loss(token_logits, column_logits, labels, cell_index, col_index, cell_mask):
    """
    Computes the loss for cell selection constrained to a single column. The loss is a hierarchical log-likelihood. The
    model first predicts a column and then selects cells within that column (conditioned on the column). Cells outside
    the selected column are never selected.

    Args:
        token_logits (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            Tensor containing the logits per token.
        column_logits (`tf.Tensor` of shape `(batch_size, max_num_cols)`):
            Tensor containing the logits per column.
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            Labels per token.
        cell_index (`ProductIndexMap`):
            Index that groups tokens into cells.
        col_index (`IndexMap`):
            Index that groups tokens into columns.
        cell_mask (`tf.Tensor` of shape `(batch_size, max_num_rows * max_num_cols)`):
            Mask for cells that exist in the table (i.e. that are not padding).

    Returns:
        selection_loss_per_example (`tf.Tensor` of shape `(batch_size,)`): Loss for each example. logits (`tf.Tensor`
        of shape `(batch_size, sequence_length)`): New logits which are only allowed to select cells in a single
        column. Logits outside of the most likely column according to *column_logits* will be set to a very low value
        (such that the probabilities are 0).
    """
    # First find the column we should select. We use the column with maximum
    # number of selected cells.
    labels_per_column, _ = reduce_sum(tf.cast(labels, tf.float32), col_index)
    column_label = tf.argmax(labels_per_column, axis=-1, output_type=tf.int32)
    # Check if there are no selected cells in the column. In that case the model
    # should predict the special column id 0, which means "select nothing".
    no_cell_selected = tf.equal(tf.reduce_max(labels_per_column, axis=-1), 0)
    column_label = tf.where(no_cell_selected, tf.zeros_like(column_label), column_label)

    # Create a categorical distribution based on column logits for loss computation
    column_dist = tfp.distributions.Categorical(logits=column_logits)
    column_loss_per_example = -column_dist.log_prob(column_label)

    # Reduce the labels and logits to per-cell from per-token.
    logits_per_cell, _ = reduce_mean(token_logits, cell_index)
    labels_per_cell, labels_index = reduce_max(tf.cast(labels, tf.int32), cell_index)

    # Mask for the selected column.
    column_id_for_cells = cell_index.project_inner(labels_index).indices
    column_mask = tf.cast(tf.equal(column_id_for_cells, tf.expand_dims(column_label, axis=1)), tf.float32)

    # Compute the log-likelihood for cells, but only for the selected column.
    # 创建一个伯努利分布对象，使用给定的 logits 参数
    cell_dist = tfp.distributions.Bernoulli(logits=logits_per_cell)
    # 计算每个细胞的对数概率，根据标签值
    cell_log_prob = cell_dist.log_prob(labels_per_cell)
    # 计算每个细胞的损失，考虑列掩码和细胞掩码
    cell_loss = -tf.reduce_sum(cell_log_prob * column_mask * cell_mask, axis=1)
    # 将损失标准化为每列中的细胞数量，避免零除错误
    cell_loss /= tf.reduce_sum(column_mask * cell_mask, axis=1) + EPSILON_ZERO_DIVISION

    # 每个样本的选择损失等于每个列的损失
    selection_loss_per_example = column_loss_per_example
    # 添加细胞损失，仅在模型选择了细胞时
    selection_loss_per_example += tf.where(no_cell_selected, tf.zeros_like(selection_loss_per_example), cell_loss)

    # 根据模型选择的列，将选定列以外的概率设置为零
    selected_column_id = tf.argmax(column_logits, axis=-1, output_type=tf.int32)
    selected_column_mask = tf.cast(
        tf.equal(column_id_for_cells, tf.expand_dims(selected_column_id, axis=-1)), tf.float32
    )
    # 永远不要选择具有特殊列标识符 0 的细胞
    selected_column_mask = tf.where(
        tf.equal(column_id_for_cells, 0), tf.zeros_like(selected_column_mask), selected_column_mask
    )
    # 调整细胞的 logits，确保在选择的列之外的细胞概率为零
    logits_per_cell += CLOSE_ENOUGH_TO_LOG_ZERO * (1.0 - cell_mask * selected_column_mask)
    # 从 logits_per_cell 中收集指定的 logits
    logits = gather(logits_per_cell, cell_index)

    # 返回每个示例的选择损失和 logits
    return selection_loss_per_example, logits
# 计算聚合掩码，以确定模型是否应选择表中的单元格而非聚合
def _calculate_aggregate_mask(answer, pooled_output, cell_selection_preference, labels, aggregation_classifier):
    """
    Finds examples where the model should select cells with no aggregation.

    Returns a mask that determines for which examples should the model select answers directly from the table, without
    any aggregation function. If the answer is a piece of text the case is unambiguous as aggregation functions only
    apply to numbers. If the answer is a number but does not appear in the table then we must use some aggregation
    case. The ambiguous case is when the answer is a number that also appears in the table. In this case we use the
    aggregation function probabilities predicted by the model to decide whether to select or aggregate. The threshold
    for this is a hyperparameter *cell_selection_preference*

    Args:
        answer (`tf.Tensor` of shape `(batch_size, )`):
            Answer for every example in the batch. Nan if there is no scalar answer.
        pooled_output (`tf.Tensor` of shape `(batch_size, hidden_size)`):
            Output of the pooler (BertPooler) on top of the encoder layer.
        cell_selection_preference (`float`):
            Preference for cell selection in ambiguous cases.
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            Labels per token.
        aggregation_classifier (`torch.nn.Linear`): Aggregation head

    Returns:
        aggregate_mask (`tf.Tensor` of shape `(batch_size,)`): A mask set to 1 for examples that should use aggregation
        functions.
    """
    # 初始化聚合掩码，判断答案是否为数字而非NaN
    aggregate_mask_init = tf.cast(tf.logical_not(tf.math.is_nan(answer)), tf.float32)
    
    # 计算聚合分类器的逻辑回归结果
    logits_aggregation = aggregation_classifier(pooled_output)
    
    # 创建分类分布对象，用于计算聚合函数的概率分布
    dist_aggregation = tfp.distributions.Categorical(logits=logits_aggregation)
    
    # 计算除“无聚合”外其他聚合操作的总质量
    aggregation_ops_total_mass = tf.reduce_sum(dist_aggregation.probs_parameter()[:, 1:], axis=1)
    
    # 根据当前模型判断是否选择单元格
    is_pred_cell_selection = aggregation_ops_total_mass <= cell_selection_preference
    
    # 判断是否存在单元格选择监督的例子
    is_cell_supervision_available = tf.reduce_sum(labels, axis=1) > 0
    
    # 根据判断结果设置聚合掩码
    aggregate_mask = tf.where(
        tf.logical_and(is_pred_cell_selection, is_cell_supervision_available),
        tf.zeros_like(aggregate_mask_init, dtype=tf.float32),
        aggregate_mask_init,
    )
    
    # 停止梯度在聚合掩码上的传播
    aggregate_mask = tf.stop_gradient(aggregate_mask)
    
    return aggregate_mask


def _calculate_aggregation_loss_known(
    logits_aggregation, aggregate_mask, aggregation_labels, use_answer_as_supervision, num_aggregation_labels
):
    """
    Calculates aggregation loss when its type is known during training.

    In the weakly supervised setting, the only known information is that for cell selection examples, "no aggregation"
    """
    # 计算已知类型聚合损失，用于训练中已知聚合类型的情况
    # 仅在弱监督设置中，已知信息是对于单元格选择的示例，“无聚合”
    """
    Calculate aggregation loss based on logits and supervision signals.

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

    Returns:
        aggregation_loss_known (`tf.Tensor` of shape `(batch_size,)`): Aggregation loss (when its type is known during
        training) per example.
    """
    if use_answer_as_supervision:
        # Prepare "no aggregation" targets for cell selection examples.
        target_aggregation = tf.zeros_like(aggregate_mask, dtype=tf.int32)
    else:
        # Use aggregation supervision as the target.
        target_aggregation = aggregation_labels

    # Convert aggregation labels to one-hot encoding.
    one_hot_labels = tf.one_hot(target_aggregation, depth=num_aggregation_labels, dtype=tf.float32)

    # Compute log probabilities of the logits.
    log_probs = tf.nn.log_softmax(logits_aggregation, axis=-1)

    # Calculate cross entropy loss per example.
    per_example_aggregation_intermediate = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)

    if use_answer_as_supervision:
        # Accumulate loss only for examples requiring cell selection
        # (no aggregation).
        return per_example_aggregation_intermediate * (1 - aggregate_mask)
    else:
        # Return aggregation loss for all examples.
        return per_example_aggregation_intermediate
# 计算每个细胞的期望结果，考虑数值分布、数值、缩放因子、输入掩码、聚合逻辑和配置参数
def _calculate_expected_result(
    dist_per_cell, numeric_values, numeric_values_scale, input_mask_float, logits_aggregation, config
):
    Calculates the expected result given cell and aggregation probabilities.

    Args:
        dist_per_cell (`tfp.distributions.Bernoulli`):
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
            Model configuration class with all the hyperparameters of the model

    Returns:
        expected_result (`tf.Tensor` of shape `(batch_size,)`): The expected result per example.
    """
    if config.use_gumbel_for_cells:
        # 使用 Gumbel 分布进行采样，用于模拟以伯努利分布为基础的元胞选择
        gumbel_dist = tfp.distributions.RelaxedBernoulli(
            config.temperature,
            logits=dist_per_cell.logits_parameter() * config.temperature,
        )
        scaled_probability_per_cell = gumbel_dist.sample()  # 从 Gumbel 分布中采样元胞选择的概率
    else:
        scaled_probability_per_cell = dist_per_cell.probs_parameter()  # 直接使用伯努利分布的概率参数

    # 对每个元胞选择的概率进行缩放，同时应用数字值的比例和表的掩码
    scaled_probability_per_cell = (scaled_probability_per_cell / numeric_values_scale) * input_mask_float

    # 计算每个示例中选中元胞的数量总和
    count_result = tf.reduce_sum(scaled_probability_per_cell, axis=1)

    # 将非数字表格值的数值设为零，用于遮蔽那些非数值的标记
    numeric_values_masked = tf.where(
        tf.math.is_nan(numeric_values), tf.zeros_like(numeric_values), numeric_values
    )

    # 计算加权平均的结果总和
    sum_result = tf.reduce_sum(scaled_probability_per_cell * numeric_values_masked, axis=1)

    # 根据配置中的平均逼近方法选择相应的方法计算平均结果
    avg_approximation = config.average_approximation_function
    if avg_approximation == AverageApproximationFunction.RATIO:
        # 使用比率逼近方法计算平均结果
        average_result = sum_result / (count_result + EPSILON_ZERO_DIVISION)
    elif avg_approximation == AverageApproximationFunction.FIRST_ORDER:
        # 使用一阶逼近方法计算平均结果，考虑到其他元胞的概率
        ex = tf.reduce_sum(scaled_probability_per_cell, axis=1, keepdims=True) - scaled_probability_per_cell + 1
        average_result = tf.reduce_sum(numeric_values_masked * scaled_probability_per_cell / ex, axis=1)
    elif avg_approximation == AverageApproximationFunction.SECOND_ORDER:
        # 如果平均逼近方法为二阶，执行以下操作
        # 计算每个单元格的调整概率总和，除了当前单元格对应的概率，加上常数1
        ex = tf.reduce_sum(scaled_probability_per_cell, axis=1, keepdims=True) - scaled_probability_per_cell + 1
        # 计算每个单元格的点态方差
        pointwise_var = scaled_probability_per_cell * (1 - scaled_probability_per_cell)
        # 计算总体方差，排除当前单元格的贡献
        var = tf.reduce_sum(pointwise_var, axis=1, keepdims=True) - pointwise_var
        # 计算乘子，用于调整结果
        multiplier = (var / tf.math.square(ex) + 1) / ex
        # 计算加权平均结果
        average_result = tf.reduce_sum(numeric_values_masked * scaled_probability_per_cell * multiplier, axis=1)
    else:
        # 如果平均逼近方法不是二阶，则抛出错误
        raise ValueError("Invalid average_approximation_function: %s", config.average_approximation_function)

    if config.use_gumbel_for_aggregation:
        # 如果配置使用 Gumbel 分布进行聚合操作
        gumbel_dist = tfp.distributions.RelaxedOneHotCategorical(
            config.aggregation_temperature, logits=logits_aggregation[:, 1:]
        )
        # <float32>[batch_size, num_aggregation_labels - 1]
        # 从 Gumbel 分布中抽样，得到聚合操作的概率
        aggregation_op_only_probs = gumbel_dist.sample()
    else:
        # 如果不使用 Gumbel 分布进行聚合操作
        # <float32>[batch_size, num_aggregation_labels - 1]
        # 使用稳定的 softmax 函数计算聚合操作的概率
        aggregation_op_only_probs = stable_softmax(logits_aggregation[:, 1:] / config.aggregation_temperature, axis=-1)
    
    # 将所有结果按行拼接成一个张量
    all_results = tf.concat(
        [
            tf.expand_dims(sum_result, axis=1),
            tf.expand_dims(average_result, axis=1),
            tf.expand_dims(count_result, axis=1),
        ],
        axis=1,
    )
    # 计算期望结果，即所有结果与聚合操作概率的加权和
    expected_result = tf.reduce_sum(all_results * aggregation_op_only_probs, axis=1)
    # 返回期望结果张量
    return expected_result
    # 计算期望结果，根据每个单元格的分布、数值、数值规模、输入掩码、聚合操作的逻辑
    expected_result = _calculate_expected_result(
        dist_per_cell, numeric_values, numeric_values_scale, input_mask_float, logits_aggregation, config
    )

    # 将答案中的 NaN 替换为 0
    answer_masked = tf.where(tf.math.is_nan(answer), tf.zeros_like(answer), answer)

    # 如果配置启用了标准化答案损失
    if config.use_normalized_answer_loss:
        # 计算损失的标准化因子
        normalizer = tf.stop_gradient(
            tf.math.maximum(tf.math.abs(expected_result), tf.math.abs(answer_masked)) + EPSILON_ZERO_DIVISION
        )
        # 标准化答案和期望结果
        normalized_answer_masked = answer_masked / normalizer
        normalized_expected_result = expected_result / normalizer
        # 使用 Huber 损失函数计算每个示例的答案损失
        per_example_answer_loss = tf.compat.v1.losses.huber_loss(
            normalized_answer_masked * aggregate_mask,
            normalized_expected_result * aggregate_mask,
            delta=tf.cast(1.0, tf.float32),
            reduction=tf.losses.Reduction.NONE,
        )
    else:
        # 使用 Huber 损失函数计算每个示例的答案损失，未标准化的情况
        per_example_answer_loss = tf.compat.v1.losses.huber_loss(
            answer_masked * aggregate_mask,
            expected_result * aggregate_mask,
            delta=tf.cast(config.huber_loss_delta, tf.float32),
            reduction=tf.losses.Reduction.NONE,
        )
    # 如果配置中的答案损失截断值为 None，则创建一个全为 1 的张量作为大答案损失掩码
    if config.answer_loss_cutoff is None:
        large_answer_loss_mask = tf.ones_like(per_example_answer_loss, dtype=tf.float32)
    # 否则，根据答案损失是否大于答案损失截断值，生成大答案损失掩码
    else:
        large_answer_loss_mask = tf.where(
            per_example_answer_loss > config.answer_loss_cutoff,
            tf.zeros_like(per_example_answer_loss, dtype=tf.float32),
            tf.ones_like(per_example_answer_loss, dtype=tf.float32),
        )
    # 计算每个示例的答案损失加权，乘以聚合掩码
    per_example_answer_loss_scaled = config.answer_loss_importance * (per_example_answer_loss * aggregate_mask)
    # 返回加权后的每个示例的答案损失以及大答案损失掩码
    return per_example_answer_loss_scaled, large_answer_loss_mask
```