# `.\models\xlm_roberta\modeling_tf_xlm_roberta.py`

```py
# 编码声明，指定文件编码为UTF-8
# 版权声明，版权归Facebook AI Research和HuggingFace Inc.团队所有
# 版权声明，版权归NVIDIA CORPORATION所有，保留所有权利
#
# 根据Apache许可证2.0版授权，除非符合许可证要求，否则不得使用此文件
# 您可以在以下链接获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则依据“原样”分发本软件，
# 不提供任何形式的明示或暗示担保或条件。
# 请查阅许可证了解详细信息
""" TF 2.0 XLM-RoBERTa 模型。"""

from __future__ import annotations  # 用于支持后续版本的类型注释

import math  # 导入数学模块
import warnings  # 导入警告模块
from typing import Optional, Tuple, Union  # 导入类型注释相关模块

import numpy as np  # 导入NumPy库
import tensorflow as tf  # 导入TensorFlow库

from ...activations_tf import get_tf_activation  # 导入自定义TensorFlow激活函数
from ...modeling_tf_outputs import (  # 导入TensorFlow模型输出相关模块
    TFBaseModelOutputWithPastAndCrossAttentions,
    TFBaseModelOutputWithPoolingAndCrossAttentions,
    TFCausalLMOutputWithCrossAttentions,
    TFMaskedLMOutput,
    TFMultipleChoiceModelOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from ...modeling_tf_utils import (  # 导入TensorFlow模型工具函数
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
from ...tf_utils import (  # 导入TensorFlow工具函数
    check_embeddings_within_bounds,
    shape_list,
    stable_softmax,
)
from ...utils import (  # 导入通用工具函数
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from .configuration_xlm_roberta import XLMRobertaConfig  # 导入XLM-RoBERTa配置

logger = logging.get_logger(__name__)  # 获取模块专用的日志记录器

logger = logging.get_logger(__name__)  # 获取模块专用的日志记录器

_CHECKPOINT_FOR_DOC = "FacebookAI/xlm-roberta-base"  # 预训练模型的检查点名称，用于文档
_CONFIG_FOR_DOC = "XLMRobertaConfig"  # XLM-RoBERTa配置的名称，用于文档

# XLM-RoBERTa预训练模型的存档列表
TF_XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "FacebookAI/xlm-roberta-base",
    "FacebookAI/xlm-roberta-large",
    "joeddav/xlm-roberta-large-xnli",
    "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    # 更多的模型存档可以在https://huggingface.co/models?filter=xlm-roberta查看
]

# XLM-RoBERTa模型文档的起始描述
XLM_ROBERTA_START_DOCSTRING = r"""

    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a [keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
    behavior.

    <Tip>

    TensorFlow models and layers in `transformers` accept two formats as input:
"""
    # having all inputs as keyword arguments (like PyTorch models), or
    # having all inputs as a list, tuple or dict in the first positional argument.
    
    # The reason the second format is supported is that Keras methods prefer this format when passing inputs to models
    # and layers. Because of this support, when using methods like `model.fit()` things should "just work" for you - just
    # pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second
    # format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with
    # the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first
    # positional argument:
    
    # a single Tensor with `input_ids` only and nothing else: `model(input_ids)`
    # a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
    # `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`
    # a dictionary with one or several input Tensors associated to the input names given in the docstring:
    # `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`
    
    # Note that when creating models and layers with
    # [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don't need to worry
    # about any of this, as you can just pass inputs like you would to any other Python function!
"""

XLM_ROBERTA_INPUTS_DOCSTRING = r"""
"""


# 从 transformers.models.roberta.modeling_tf_roberta.TFRobertaEmbeddings 复制并修改为 XLMRobertaEmbeddings
class TFXLMRobertaEmbeddings(keras.layers.Layer):
    """
    和 BertEmbeddings 相同，但稍作调整以适应位置嵌入的索引。
    """

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        # 设定填充符索引为1
        self.padding_idx = 1
        self.config = config
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings
        self.initializer_range = config.initializer_range
        # 使用配置的 epsilon 值创建 LayerNormalization 层
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 使用配置的 dropout 概率创建 Dropout 层
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def build(self, input_shape=None):
        with tf.name_scope("word_embeddings"):
            # 创建词嵌入权重矩阵，形状为 [vocab_size, hidden_size]
            self.weight = self.add_weight(
                name="weight",
                shape=[self.config.vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        with tf.name_scope("token_type_embeddings"):
            # 创建类型嵌入权重矩阵，形状为 [type_vocab_size, hidden_size]
            self.token_type_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.config.type_vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        with tf.name_scope("position_embeddings"):
            # 创建位置嵌入权重矩阵，形状为 [max_position_embeddings, hidden_size]
            self.position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_position_embeddings, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        if self.built:
            return
        self.built = True
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                # 构建 LayerNormalization 层，输入形状为 [None, None, hidden_size]
                self.LayerNorm.build([None, None, self.config.hidden_size])

    def create_position_ids_from_input_ids(self, input_ids, past_key_values_length=0):
        """
        将非填充符号替换为它们的位置编号。位置编号从 padding_idx+1 开始，忽略填充符号。
        这是从 fairseq 的 `utils.make_positions` 修改而来。

        Args:
            input_ids: tf.Tensor
        Returns: tf.Tensor
        """
        # 创建一个掩码，将非填充符号转换为 1，其余为 0
        mask = tf.cast(tf.math.not_equal(input_ids, self.padding_idx), dtype=input_ids.dtype)
        # 计算累积位置索引，跳过填充符号
        incremental_indices = (tf.math.cumsum(mask, axis=1) + past_key_values_length) * mask

        return incremental_indices + self.padding_idx

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
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        assert not (input_ids is None and inputs_embeds is None)  # 断言确保 input_ids 和 inputs_embeds 至少有一个不为空

        if input_ids is not None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)  # 检查 input_ids 是否在有效范围内
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)  # 根据 input_ids 从权重矩阵中获取对应的嵌入向量

        input_shape = shape_list(inputs_embeds)[:-1]  # 获取输入嵌入张量的形状，去掉最后一个维度（通常是嵌入维度）

        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)  # 如果未提供 token_type_ids，则创建一个全零的张量，形状与输入嵌入张量一致

        if position_ids is None:
            if input_ids is not None:
                # 如果存在 input_ids，则基于它创建 position_ids，确保任何填充的标记仍然是填充的
                position_ids = self.create_position_ids_from_input_ids(
                    input_ids=input_ids, past_key_values_length=past_key_values_length
                )
            else:
                # 否则，创建默认的 position_ids，从 self.padding_idx 开始，长度为 input_shape[-1]
                position_ids = tf.expand_dims(
                    tf.range(start=self.padding_idx + 1, limit=input_shape[-1] + self.padding_idx + 1), axis=0
                )

        position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)  # 根据 position_ids 从位置嵌入矩阵中获取位置嵌入向量
        token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)  # 根据 token_type_ids 从 token type 嵌入矩阵中获取 token type 嵌入向量
        final_embeddings = inputs_embeds + position_embeds + token_type_embeds  # 将输入嵌入、位置嵌入和 token type 嵌入相加，形成最终的嵌入向量
        final_embeddings = self.LayerNorm(inputs=final_embeddings)  # 对最终的嵌入向量进行 Layer Normalization 处理
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)  # 应用 dropout 处理最终的嵌入向量，用于训练中的随机失活

        return final_embeddings  # 返回处理后的最终嵌入向量作为输出
# Copied from transformers.models.bert.modeling_tf_bert.TFBertPooler with Bert->XLMRoberta
class TFXLMRobertaPooler(keras.layers.Layer):
    def __init__(self, config: XLMRobertaConfig, **kwargs):
        super().__init__(**kwargs)

        # 初始化一个全连接层，用于池化隐藏状态
        self.dense = keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 池化模型的输出，通过取第一个标记对应的隐藏状态
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(inputs=first_token_tensor)

        return pooled_output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            # 构建密集层
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# Copied from transformers.models.bert.modeling_tf_bert.TFBertSelfAttention with Bert->XLMRoberta
class TFXLMRobertaSelfAttention(keras.layers.Layer):
    def __init__(self, config: XLMRobertaConfig, **kwargs):
        super().__init__(**kwargs)

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_attention_heads})"
            )

        # 定义自注意力层，确保隐藏大小是注意力头数的倍数
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

        # 定义查询、键、值的全连接层
        self.query = keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        self.key = keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key"
        )
        self.value = keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )
        # 定义dropout层，用于注意力概率的随机丢弃
        self.dropout = keras.layers.Dropout(rate=config.attention_probs_dropout_prob)

        self.is_decoder = config.is_decoder
        self.config = config
    # 将输入的张量重塑为 [batch_size, seq_length, num_attention_heads, attention_head_size] 的形状
    tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))

    # 将张量进行转置，从 [batch_size, seq_length, num_attention_heads, attention_head_size] 变为 [batch_size, num_attention_heads, seq_length, attention_head_size]
    return tf.transpose(tensor, perm=[0, 2, 1, 3])

    # 检查是否已经构建过网络层，如果是则直接返回
    if self.built:
        return

    # 标记网络层已构建
    self.built = True

    # 如果存在查询（query）张量，则按指定形状构建
    if getattr(self, "query", None) is not None:
        with tf.name_scope(self.query.name):
            self.query.build([None, None, self.config.hidden_size])

    # 如果存在键（key）张量，则按指定形状构建
    if getattr(self, "key", None) is not None:
        with tf.name_scope(self.key.name):
            self.key.build([None, None, self.config.hidden_size])

    # 如果存在值（value）张量，则按指定形状构建
    if getattr(self, "value", None) is not None:
        with tf.name_scope(self.value.name):
            self.value.build([None, None, self.config.hidden_size])
# Copied from transformers.models.bert.modeling_tf_bert.TFBertSelfOutput with Bert->XLMRoberta
class TFXLMRobertaSelfOutput(keras.layers.Layer):
    def __init__(self, config: XLMRobertaConfig, **kwargs):
        super().__init__(**kwargs)

        # 初始化一个全连接层，用于变换隐藏状态的维度
        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 初始化 LayerNormalization 层，用于归一化隐藏状态
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 初始化 Dropout 层，用于在训练时随机失活部分神经元，防止过拟合
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 通过全连接层变换隐藏状态的维度
        hidden_states = self.dense(inputs=hidden_states)
        # 在训练时应用 Dropout 操作，随机失活部分神经元
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 对变换后的隐藏状态进行 LayerNormalization，加上输入张量，构成残差连接
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经构建，直接返回；否则按照指定的维度构建全连接层和 LayerNormalization 层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


# Copied from transformers.models.bert.modeling_tf_bert.TFBertAttention with Bert->XLMRoberta
class TFXLMRobertaAttention(keras.layers.Layer):
    def __init__(self, config: XLMRobertaConfig, **kwargs):
        super().__init__(**kwargs)

        # 初始化 XLMRoberta 的自注意力层和输出层
        self.self_attention = TFXLMRobertaSelfAttention(config, name="self")
        self.dense_output = TFXLMRobertaSelfOutput(config, name="output")

    def prune_heads(self, heads):
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
        # 使用自注意力层处理输入张量，得到自注意力层的输出
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
        # 将自注意力层的输出传入输出层，得到注意力输出
        attention_output = self.dense_output(
            hidden_states=self_outputs[0], input_tensor=input_tensor, training=training
        )
        # 如果需要输出注意力信息，将其添加到输出元组中
        outputs = (attention_output,) + self_outputs[1:]

        return outputs
    # 定义一个方法用于建立模型，在输入形状为None时
    def build(self, input_shape=None):
        # 如果模型已经建立过，则直接返回，不再重复建立
        if self.built:
            return
        # 设置模型已经建立的标志为True
        self.built = True
        # 如果存在self_attention属性，则构建self_attention模块
        if getattr(self, "self_attention", None) is not None:
            # 在TensorFlow的命名作用域下，建立self_attention模块
            with tf.name_scope(self.self_attention.name):
                # 调用self_attention的build方法，传入None作为输入形状
                self.self_attention.build(None)
        # 如果存在dense_output属性，则构建dense_output模块
        if getattr(self, "dense_output", None) is not None:
            # 在TensorFlow的命名作用域下，建立dense_output模块
            with tf.name_scope(self.dense_output.name):
                # 调用dense_output的build方法，传入None作为输入形状
                self.dense_output.build(None)
# Copied from transformers.models.bert.modeling_tf_bert.TFBertIntermediate with Bert->XLMRoberta
class TFXLMRobertaIntermediate(keras.layers.Layer):
    def __init__(self, config: XLMRobertaConfig, **kwargs):
        super().__init__(**kwargs)

        # 定义一个全连接层，用于中间层的转换，输出单元数为config.intermediate_size
        self.dense = keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        # 根据配置选择中间激活函数，若配置为字符串则使用相应的 TensorFlow 激活函数，否则直接使用给定的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 将输入的隐藏状态通过全连接层进行线性转换
        hidden_states = self.dense(inputs=hidden_states)
        # 应用中间激活函数对转换后的隐藏状态进行非线性变换
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        self.built = True
        # 如果定义了全连接层dense，则按照给定的形状构建该层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# Copied from transformers.models.bert.modeling_tf_bert.TFBertOutput with Bert->XLMRoberta
class TFXLMRobertaOutput(keras.layers.Layer):
    def __init__(self, config: XLMRobertaConfig, **kwargs):
        super().__init__(**kwargs)

        # 定义一个全连接层，用于输出层的转换，输出单元数为config.hidden_size
        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # LayerNormalization层，用于归一化输出
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # Dropout层，用于随机失活以防止过拟合
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 将输入的隐藏状态通过全连接层进行线性转换
        hidden_states = self.dense(inputs=hidden_states)
        # 应用Dropout层进行随机失活
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 将输出与输入张量进行残差连接，并通过LayerNormalization层进行归一化
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states

    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        self.built = True
        # 如果定义了全连接层dense，则按照给定的形状构建该层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])
        # 如果定义了LayerNorm层，则按照给定的形状构建该层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


# Copied from transformers.models.bert.modeling_tf_bert.TFBertLayer with Bert->XLMRoberta
class TFXLMRobertaLayer(keras.layers.Layer):
    # 这里将代码片段留空，因为它未完全给出
    pass
    # 初始化函数，用于初始化一个 XLMRobertaModel 对象
    def __init__(self, config: XLMRobertaConfig, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 创建自注意力层对象，使用给定的配置和名称
        self.attention = TFXLMRobertaAttention(config, name="attention")
        
        # 设置是否作为解码器模型的标志
        self.is_decoder = config.is_decoder
        
        # 设置是否添加交叉注意力的标志，并进行相应的检查
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                # 如果未设置为解码器模型但添加了交叉注意力，则抛出异常
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            
            # 创建交叉注意力层对象，使用给定的配置和名称
            self.crossattention = TFXLMRobertaAttention(config, name="crossattention")
        
        # 创建中间层对象，使用给定的配置和名称
        self.intermediate = TFXLMRobertaIntermediate(config, name="intermediate")
        
        # 创建BERT输出层对象，使用给定的配置和名称
        self.bert_output = TFXLMRobertaOutput(config, name="output")
    
    # 调用函数，用于执行模型的前向传播
    def call(
        self,
        hidden_states: tf.Tensor,  # 输入的隐藏状态张量
        attention_mask: tf.Tensor,  # 注意力掩码张量
        head_mask: tf.Tensor,  # 头部掩码张量
        encoder_hidden_states: tf.Tensor | None,  # 编码器隐藏状态张量或None
        encoder_attention_mask: tf.Tensor | None,  # 编码器注意力掩码张量或None
        past_key_value: Tuple[tf.Tensor] | None,  # 过去键值元组或None
        output_attentions: bool,  # 是否输出注意力权重
        training: bool = False,  # 是否在训练模式下
    ) -> Tuple[tf.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # Perform self-attention on the input hidden states
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

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]  # slice to exclude self-attention cache tuple
            present_key_value = self_attention_outputs[-1]  # last element is present key/value
        else:
            outputs = self_attention_outputs[1:]  # include self attentions if outputting attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # Perform cross-attention using the self-attention output and encoder hidden states
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
            # append cross-attention outputs to existing outputs
            outputs = outputs + cross_attention_outputs[1:-1]

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # Compute intermediate output using the attention output
        intermediate_output = self.intermediate(hidden_states=attention_output)
        # Compute final layer output using intermediate output and attention output
        layer_output = self.bert_output(
            hidden_states=intermediate_output, input_tensor=attention_output, training=training
        )
        outputs = (layer_output,) + outputs  # append attentions if outputting them

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs
    # 构建方法用于构建模型层，接受输入形状作为参数，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 将标记置为已构建
        self.built = True
        
        # 如果存在注意力层，则按名称作用域构建并调用其 build 方法
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        
        # 如果存在中间层，则按名称作用域构建并调用其 build 方法
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        
        # 如果存在 BERT 输出层，则按名称作用域构建并调用其 build 方法
        if getattr(self, "bert_output", None) is not None:
            with tf.name_scope(self.bert_output.name):
                self.bert_output.build(None)
        
        # 如果存在交叉注意力层，则按名称作用域构建并调用其 build 方法
        if getattr(self, "crossattention", None) is not None:
            with tf.name_scope(self.crossattention.name):
                self.crossattention.build(None)
# 从 transformers.models.bert.modeling_tf_bert.TFBertEncoder 复制并修改为 XLMRobertaEncoder 类
class TFXLMRobertaEncoder(keras.layers.Layer):
    def __init__(self, config: XLMRobertaConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        # 初始化层列表，每一层使用 TFXLMRobertaLayer 类，命名为 layer_._{i}
        self.layer = [TFXLMRobertaLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]

    def call(
        self,
        hidden_states: tf.Tensor,                   # 输入的隐藏状态张量
        attention_mask: tf.Tensor,                  # 注意力掩码张量
        head_mask: tf.Tensor,                       # 头部掩码张量
        encoder_hidden_states: tf.Tensor | None,    # 编码器的隐藏状态张量或 None
        encoder_attention_mask: tf.Tensor | None,   # 编码器的注意力掩码张量或 None
        past_key_values: Tuple[Tuple[tf.Tensor]] | None,  # 过去的键值对或 None
        use_cache: Optional[bool],                  # 是否使用缓存的可选布尔值
        output_attentions: bool,                    # 是否输出注意力张量的布尔值
        output_hidden_states: bool,                 # 是否输出隐藏状态的布尔值
        return_dict: bool,                          # 是否返回字典格式的布尔值
        training: bool = False,                     # 是否在训练模式下的布尔值，默认为 False
    ) -> Union[TFBaseModelOutputWithPastAndCrossAttentions, Tuple[tf.Tensor]]:
        # 如果需要输出隐藏状态，则初始化 all_hidden_states 为空元组，否则为 None
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力，则初始化 all_attentions 为空元组，否则为 None
        all_attentions = () if output_attentions else None
        # 如果需要输出交叉注意力且配置允许，则初始化 all_cross_attentions 为空元组，否则为 None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 如果需要使用缓存，则初始化 next_decoder_cache 为空元组，否则为 None
        next_decoder_cache = () if use_cache else None
        # 遍历每一层进行处理
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 如果存在过去的键值对，则获取当前层的过去键值对，否则为 None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 调用当前层的处理函数，获取当前层的输出
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
            # 更新隐藏状态为当前层输出的第一个元素
            hidden_states = layer_outputs[0]

            # 如果需要使用缓存，则将当前层输出的最后一个元素添加到 next_decoder_cache 中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            # 如果需要输出注意力，则将当前层输出的第二个元素添加到 all_attentions 中
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
                # 如果配置允许且存在编码器的隐藏状态，则将当前层输出的第三个元素添加到 all_cross_attentions 中
                if self.config.add_cross_attention and encoder_hidden_states is not None:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果需要输出隐藏状态，则将最后一层的隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要返回字典格式，则返回非空结果的元组
        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_attentions, all_cross_attentions] if v is not None
            )

        # 返回 TFBaseModelOutputWithPastAndCrossAttentions 类型的字典格式结果
        return TFBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )
    # 定义神经网络模型的构建方法，用于建立模型的层次结构
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回，避免重复构建
        if self.built:
            return
        # 将模型标记为已构建状态
        self.built = True
        # 检查是否存在层属性，如果存在，则遍历每一层并构建
        if getattr(self, "layer", None) is not None:
            for layer in self.layer:
                # 使用 TensorFlow 的命名作用域，为每一层设置命名空间
                with tf.name_scope(layer.name):
                    # 调用每一层的 build 方法，传入 None 作为输入形状，具体参数由输入数据形状决定
                    layer.build(None)
@keras_serializable
# 使用 keras_serializable 装饰器，指示此类可以在 Keras 中序列化

# 从 transformers.models.roberta.modeling_tf_roberta.TFRobertaMainLayer 复制而来，将 Roberta 替换为 XLMRoberta
class TFXLMRobertaMainLayer(keras.layers.Layer):
    # 使用 XLMRobertaConfig 作为配置类
    config_class = XLMRobertaConfig

    def __init__(self, config, add_pooling_layer=True, **kwargs):
        super().__init__(**kwargs)

        # 保存配置对象
        self.config = config
        # 是否为解码器
        self.is_decoder = config.is_decoder

        # 从配置中获取一些参数
        self.num_hidden_layers = config.num_hidden_layers
        self.initializer_range = config.initializer_range
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.return_dict = config.use_return_dict

        # 创建编码器（TFXLMRobertaEncoder），名称为 "encoder"
        self.encoder = TFXLMRobertaEncoder(config, name="encoder")

        # 如果指定了要添加池化层，则创建池化层（TFXLMRobertaPooler），名称为 "pooler"
        self.pooler = TFXLMRobertaPooler(config, name="pooler") if add_pooling_layer else None

        # 创建嵌入层（TFXLMRobertaEmbeddings），名称为 "embeddings"
        # embeddings 必须是最后声明的以保持权重顺序
        self.embeddings = TFXLMRobertaEmbeddings(config, name="embeddings")

    # 从 transformers.models.bert.modeling_tf_bert.TFBertMainLayer.get_input_embeddings 复制而来
    # 返回嵌入层对象
    def get_input_embeddings(self) -> keras.layers.Layer:
        return self.embeddings

    # 从 transformers.models.bert.modeling_tf_bert.TFBertMainLayer.set_input_embeddings 复制而来
    # 设置嵌入层的权重和词汇表大小
    def set_input_embeddings(self, value: tf.Variable):
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    # 从 transformers.models.bert.modeling_tf_bert.TFBertMainLayer._prune_heads 复制而来
    # 用于剪枝模型中的注意力头部
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    @unpack_inputs
    # 从 transformers.models.bert.modeling_tf_bert.TFBertMainLayer.call 复制而来
    # 模型的前向传播函数，处理输入并返回输出
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
    # 定义神经网络模型的 build 方法，用于构建模型的网络结构
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回，避免重复构建
        if self.built:
            return
        # 标记模型已构建
        self.built = True
        
        # 如果模型中存在编码器（encoder）属性，则构建编码器
        if getattr(self, "encoder", None) is not None:
            # 在 TensorFlow 中，使用 name_scope 可以为模型的不同部分指定命名空间
            with tf.name_scope(self.encoder.name):
                # 调用编码器对象的 build 方法来构建编码器的网络结构
                self.encoder.build(None)
        
        # 如果模型中存在池化器（pooler）属性，则构建池化器
        if getattr(self, "pooler", None) is not None:
            with tf.name_scope(self.pooler.name):
                # 调用池化器对象的 build 方法来构建池化器的网络结构
                self.pooler.build(None)
        
        # 如果模型中存在嵌入层（embeddings）属性，则构建嵌入层
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                # 调用嵌入层对象的 build 方法来构建嵌入层的网络结构
                self.embeddings.build(None)
# Copied from transformers.models.roberta.modeling_tf_roberta.TFRobertaPreTrainedModel with Roberta->XLMRoberta
class TFXLMRobertaPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定使用的配置类为 XLMRobertaConfig
    config_class = XLMRobertaConfig
    # 基础模型的前缀名为 "roberta"
    base_model_prefix = "roberta"


@add_start_docstrings(
    "The bare XLM RoBERTa Model transformer outputting raw hidden-states without any specific head on top.",
    XLM_ROBERTA_START_DOCSTRING,
)
# Copied from transformers.models.roberta.modeling_tf_roberta.TFRobertaModel with Roberta->XLMRoberta, ROBERTA->XLM_ROBERTA
class TFXLMRobertaModel(TFXLMRobertaPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 初始化 XLM-RoBERTa 主层，命名为 "roberta"
        self.roberta = TFXLMRobertaMainLayer(config, name="roberta")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLM_ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        # 调用 `self.roberta` 模型的前向传播方法，传入各种参数进行计算
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

        # 返回 `self.roberta` 的计算结果
        return outputs

    def build(self, input_shape=None):
        # 如果已经构建过网络结构，则直接返回，避免重复构建
        if self.built:
            return
        # 将网络标记为已构建状态
        self.built = True
        # 如果 `self.roberta` 存在，则在命名作用域内构建 `self.roberta` 模型
        if getattr(self, "roberta", None) is not None:
            with tf.name_scope(self.roberta.name):
                # 调用 `self.roberta` 模型的 build 方法，传入 None 作为输入形状
                self.roberta.build(None)
# 从transformers.models.roberta.modeling_tf_roberta.TFRobertaLMHead复制而来，将Roberta替换为XLMRoberta
class TFXLMRobertaLMHead(keras.layers.Layer):
    """XLMRoberta Head for masked language modeling."""

    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)

        self.config = config  # 设置配置参数
        self.hidden_size = config.hidden_size  # 获取隐藏层大小
        # 创建一个全连接层，用于将隐藏状态映射到与词汇表大小相同的向量空间
        self.dense = keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 应用LayerNormalization来规范化隐藏状态
        self.layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        # 使用GELU激活函数
        self.act = get_tf_activation("gelu")

        # 输出权重与输入嵌入相同，但每个token有一个输出偏置
        self.decoder = input_embeddings  # 解码器等于输入嵌入

    def build(self, input_shape=None):
        # 创建一个形状为(config.vocab_size,)的偏置向量，初始化为0，用于输出层
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="bias")

        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                # 构建全连接层
                self.dense.build([None, None, self.config.hidden_size])
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                # 构建LayerNormalization层
                self.layer_norm.build([None, None, self.config.hidden_size])

    def get_output_embeddings(self):
        return self.decoder  # 获取输出嵌入

    def set_output_embeddings(self, value):
        self.decoder.weight = value  # 设置输出嵌入的权重
        self.decoder.vocab_size = shape_list(value)[0]  # 设置输出嵌入的词汇表大小

    def get_bias(self):
        return {"bias": self.bias}  # 获取偏置向量

    def set_bias(self, value):
        self.bias = value["bias"]  # 设置偏置向量
        self.config.vocab_size = shape_list(value["bias"])[0]  # 更新配置中的词汇表大小

    def call(self, hidden_states):
        hidden_states = self.dense(hidden_states)  # 全连接层
        hidden_states = self.act(hidden_states)  # GELU激活函数
        hidden_states = self.layer_norm(hidden_states)  # LayerNormalization规范化

        # 使用偏置向量将隐藏状态投影回词汇表大小
        seq_length = shape_list(tensor=hidden_states)[1]  # 获取序列长度
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.hidden_size])  # 重塑张量形状
        hidden_states = tf.matmul(a=hidden_states, b=self.decoder.weight, transpose_b=True)  # 矩阵乘法
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])  # 重塑张量形状
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)  # 添加偏置向量

        return hidden_states  # 返回隐藏状态


@add_start_docstrings("""XLM RoBERTa Model with a `language modeling` head on top.""", XLM_ROBERTA_START_DOCSTRING)
# 从transformers.models.roberta.modeling_tf_roberta.TFRobertaForMaskedLM复制而来，将Roberta替换为XLMRoberta，ROBERTA替换为XLM_ROBERTA
class TFXLMRobertaForMaskedLM(TFXLMRobertaPreTrainedModel, TFMaskedLanguageModelingLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    # 初始化时，指定一些不希望加载的键名列表
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"lm_head.decoder.weight"]

    # 初始化方法，继承自父类并传入配置信息及其他可变参数
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 使用TFXLMRobertaMainLayer类构建self.roberta对象，关闭添加池化层选项，命名为"roberta"
        self.roberta = TFXLMRobertaMainLayer(config, add_pooling_layer=False, name="roberta")
        # 使用TFXLMRobertaLMHead类构建self.lm_head对象，传入self.roberta.embeddings作为参数，命名为"lm_head"
        self.lm_head = TFXLMRobertaLMHead(config, self.roberta.embeddings, name="lm_head")

    # 获取self.lm_head对象的方法
    def get_lm_head(self):
        return self.lm_head

    # 获取前缀偏置名称的方法，已被标记为不推荐使用，发出FutureWarning警告
    def get_prefix_bias_name(self):
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        # 返回self.name与self.lm_head.name拼接而成的字符串作为结果
        return self.name + "/" + self.lm_head.name

    # 调用装饰器unpack_inputs、add_start_docstrings_to_model_forward和add_code_sample_docstrings的call方法
    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLM_ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,  # 模型的检查点
        output_type=TFMaskedLMOutput,  # 输出类型为TFMaskedLMOutput
        config_class=_CONFIG_FOR_DOC,  # 配置类
        mask="<mask>",  # 掩码标识
        expected_output="' Paris'",  # 预期输出
        expected_loss=0.1,  # 预期损失
    )
    # 模型的前向传播方法，接受多个输入参数，包括输入ID、注意力掩码、标记类型ID等等
    def call(
        self,
        input_ids: TFModelInputType | None = None,  # 输入ID，可能为空
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 注意力掩码，可能为NumPy数组或Tensor，也可能为空
        token_type_ids: np.ndarray | tf.Tensor | None = None,  # 标记类型ID，可能为NumPy数组或Tensor，也可能为空
        position_ids: np.ndarray | tf.Tensor | None = None,  # 位置ID，可能为NumPy数组或Tensor，也可能为空
        head_mask: np.ndarray | tf.Tensor | None = None,  # 头部掩码，可能为NumPy数组或Tensor，也可能为空
        inputs_embeds: np.ndarray | tf.Tensor | None = None,  # 输入嵌入，可能为NumPy数组或Tensor，也可能为空
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，默认为None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，默认为None
        return_dict: Optional[bool] = None,  # 是否返回字典，默认为None
        labels: np.ndarray | tf.Tensor | None = None,  # 标签，可能为NumPy数组或Tensor，也可能为空
        training: Optional[bool] = False,  # 是否为训练模式，默认为False

        # 函数体内容在下一段继续
        pass
    ) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
        r"""
        定义方法的返回类型注解，可以返回 TFMaskedLMOutput 或包含 tf.Tensor 的元组
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            用于计算掩码语言建模损失的标签。索引应在 `[-100, 0, ..., config.vocab_size]` 范围内（参见 `input_ids` 文档）。索引设置为 `-100` 的标记将被忽略（掩码），损失仅计算具有标签 `[0, ..., config.vocab_size]` 的标记。
        """
        # 使用 Roberta 模型处理输入
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
        # 通过语言建模头部生成预测分数
        prediction_scores = self.lm_head(sequence_output)

        # 如果没有标签，则损失为 None；否则计算预测分数和标签之间的损失
        loss = None if labels is None else self.hf_compute_loss(labels, prediction_scores)

        # 如果不返回字典，则按顺序输出损失和其他输出
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TFMaskedLMOutput 对象，包括损失、预测分数、隐藏状态和注意力权重
        return TFMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记已经构建
        self.built = True
        # 如果存在 self.roberta，则在相应命名空间下构建
        if getattr(self, "roberta", None) is not None:
            with tf.name_scope(self.roberta.name):
                self.roberta.build(None)
        # 如果存在 self.lm_head，则在相应命名空间下构建
        if getattr(self, "lm_head", None) is not None:
            with tf.name_scope(self.lm_head.name):
                self.lm_head.build(None)
@add_start_docstrings(
    "XLM-RoBERTa Model with a `language modeling` head on top for CLM fine-tuning.",
    XLM_ROBERTA_START_DOCSTRING,
)
# 从 transformers.models.roberta.modeling_tf_roberta.TFRobertaForCausalLM 复制而来，将所有的 "Roberta" 替换为 "XLMRoberta"，"ROBERTA" 替换为 "XLM_ROBERTA"
class TFXLMRobertaForCausalLM(TFXLMRobertaPreTrainedModel, TFCausalLanguageModelingLoss):
    # 在加载 TF 模型时忽略以下名称的层，这些层可能是意外的或缺失的，当从 PT 模型加载 TF 模型时
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"lm_head.decoder.weight"]

    def __init__(self, config: XLMRobertaConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        if not config.is_decoder:
            logger.warning("If you want to use `TFXLMRobertaLMHeadModel` as a standalone, add `is_decoder=True.`")

        # 初始化 XLM-RoBERTa 主层，不添加池化层，命名为 "roberta"
        self.roberta = TFXLMRobertaMainLayer(config, add_pooling_layer=False, name="roberta")
        # 初始化 XLM-RoBERTa 语言建模头部，使用 roberta 的嵌入层作为输入嵌入，命名为 "lm_head"
        self.lm_head = TFXLMRobertaLMHead(config, input_embeddings=self.roberta.embeddings, name="lm_head")

    def get_lm_head(self):
        return self.lm_head

    def get_prefix_bias_name(self):
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        # 返回 lm_head 的完整名称，包括所属的模型名称前缀
        return self.name + "/" + self.lm_head.name

    # 从 transformers.models.bert.modeling_tf_bert.TFBertLMHeadModel.prepare_inputs_for_generation 复制而来
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        # 如果没有给定注意力掩码，则创建一个形状与 input_ids 相同的全 1 的注意力掩码
        if attention_mask is None:
            attention_mask = tf.ones(input_shape)

        # 如果使用了过去的 key values，截取最后一个输入 token
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # 返回用于生成的输入参数字典，包括输入 token IDs、注意力掩码和过去的 key values
        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}

    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLM_ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFCausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义一个方法 `build`，用于构建模型结构
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回，避免重复构建
        if self.built:
            return
        # 将标志位 `built` 设置为 True，表示模型已经构建
        self.built = True
        # 如果模型中有名为 `roberta` 的子模型，则构建 `roberta` 子模型
        if getattr(self, "roberta", None) is not None:
            # 在 TensorFlow 中使用命名作用域 `roberta.name` 来构建 `roberta`
            with tf.name_scope(self.roberta.name):
                self.roberta.build(None)
        # 如果模型中有名为 `lm_head` 的子模型，则构建 `lm_head` 子模型
        if getattr(self, "lm_head", None) is not None:
            # 在 TensorFlow 中使用命名作用域 `lm_head.name` 来构建 `lm_head`
            with tf.name_scope(self.lm_head.name):
                self.lm_head.build(None)
# 从 transformers.models.roberta.modeling_tf_roberta.TFRobertaClassificationHead 复制并修改为 XLMRoberta
class TFXLMRobertaClassificationHead(keras.layers.Layer):
    """用于句子级分类任务的头部。"""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 创建一个全连接层，用于分类任务
        self.dense = keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )
        # 根据配置添加分类器的 dropout 层
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = keras.layers.Dropout(classifier_dropout)
        # 输出投影层，将全连接层的输出映射到类别数量
        self.out_proj = keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="out_proj"
        )
        self.config = config

    def call(self, features, training=False):
        # 取特征的第一个 token 作为输入（相当于 [CLS]）
        x = features[:, 0, :]
        x = self.dropout(x, training=training)
        x = self.dense(x)
        x = self.dropout(x, training=training)
        x = self.out_proj(x)
        return x

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果 dense 层已定义，则建立 dense 层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 如果 out_proj 层已定义，则建立 out_proj 层
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build([None, None, self.config.hidden_size])


@add_start_docstrings(
    """
    XLM RoBERTa 模型的变压器，顶部带有序列分类/回归头部（池化输出上的线性层），例如 GLUE 任务。
    """,
    XLM_ROBERTA_START_DOCSTRING,
)
# 从 transformers.models.roberta.modeling_tf_roberta.TFRobertaForSequenceClassification 复制并修改为 XLMRoberta，ROBERTA->XLM_ROBERTA
class TFXLMRobertaForSequenceClassification(TFXLMRobertaPreTrainedModel, TFSequenceClassificationLoss):
    # 带有 '.' 的名称表示加载 PT 模型时，预期未授权的/丢失的层
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"lm_head"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 类别数量
        self.num_labels = config.num_labels

        # XLMRoberta 主层，不添加池化层
        self.roberta = TFXLMRobertaMainLayer(config, add_pooling_layer=False, name="roberta")
        # 分类头部
        self.classifier = TFXLMRobertaClassificationHead(config, name="classifier")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLM_ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 使用装饰器添加代码示例的文档字符串，说明模型预训练检查点和输出类型
    @add_code_sample_docstrings(
        checkpoint="cardiffnlp/twitter-roberta-base-emotion",
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'optimism'",
        expected_loss=0.08,
    )
    # 定义模型的调用方法，接受多个输入参数和一个可选的标签参数，返回序列分类器的输出或元组
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
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            用于计算序列分类/回归损失的标签。索引应在 `[0, ..., config.num_labels - 1]` 范围内。
            如果 `config.num_labels == 1`，则计算回归损失（均方损失）；如果 `config.num_labels > 1`，则计算分类损失（交叉熵损失）。
        """
        # 使用 Roberta 模型处理输入数据，根据参数设置返回不同的数据结构
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
        # 从模型输出中获取序列输出
        sequence_output = outputs[0]
        # 使用分类器对序列输出进行分类，根据训练状态进行不同的处理
        logits = self.classifier(sequence_output, training=training)
    
        # 如果未提供标签，则损失为 None；否则计算预测损失
        loss = None if labels is None else self.hf_compute_loss(labels, logits)
    
        # 如果不要求返回字典形式的输出，则返回 logits 和可能的其他输出
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
    
        # 返回 TFSequenceClassifierOutput 类的对象，包含损失、logits、隐藏状态和注意力权重
        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    # 定义模型的构建方法，用于建立模型的层次结构，确保仅在未构建时执行
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在 self.roberta 属性，则在其命名范围内构建 Roberta 模型
        if getattr(self, "roberta", None) is not None:
            with tf.name_scope(self.roberta.name):
                self.roberta.build(None)
        # 如果存在 self.classifier 属性，则在其命名范围内构建分类器模型
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build(None)
@add_start_docstrings(
    """
    XLM Roberta Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
    a softmax) e.g. for RocStories/SWAG tasks.
    """,
    XLM_ROBERTA_START_DOCSTRING,
)
# 从 transformers.models.roberta.modeling_tf_roberta.TFRobertaForMultipleChoice 复制过来，将 Roberta 改为 XLMRoberta，ROBERTA 改为 XLM_ROBERTA
class TFXLMRobertaForMultipleChoice(TFXLMRobertaPreTrainedModel, TFMultipleChoiceLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    # 在加载 TF 模型时，忽略掉这些意外/缺失的层
    _keys_to_ignore_on_load_unexpected = [r"lm_head"]
    _keys_to_ignore_on_load_missing = [r"dropout"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 初始化 XLM-Roberta 主层
        self.roberta = TFXLMRobertaMainLayer(config, name="roberta")
        # 添加 dropout 层
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)
        # 分类器层，用于多选分类任务
        self.classifier = keras.layers.Dense(
            1, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        XLM_ROBERTA_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFMultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 模型的前向传播函数，接受多个输入和参数
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
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
            where `num_choices` is the size of the second dimension of the input tensors. (See `input_ids` above)
        """

        # 如果输入了 input_ids，则获取 num_choices 和 seq_length
        if input_ids is not None:
            num_choices = shape_list(input_ids)[1]  # 获取选择个数，即第二个维度的大小
            seq_length = shape_list(input_ids)[2]   # 获取序列长度，即第三个维度的大小
        else:
            num_choices = shape_list(inputs_embeds)[1]  # 否则从 inputs_embeds 中获取选择个数
            seq_length = shape_list(inputs_embeds)[2]   # 从 inputs_embeds 中获取序列长度

        # 将输入张量展平，如果存在的话
        flat_input_ids = tf.reshape(input_ids, (-1, seq_length)) if input_ids is not None else None
        flat_attention_mask = tf.reshape(attention_mask, (-1, seq_length)) if attention_mask is not None else None
        flat_token_type_ids = tf.reshape(token_type_ids, (-1, seq_length)) if token_type_ids is not None else None
        flat_position_ids = tf.reshape(position_ids, (-1, seq_length)) if position_ids is not None else None
        
        # 调用 RoBERTa 模型
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
        
        # 获取池化输出并应用 dropout
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output, training=training)
        
        # 使用分类器进行分类
        logits = self.classifier(pooled_output)
        reshaped_logits = tf.reshape(logits, (-1, num_choices))  # 重新整形 logits
        
        # 计算损失（如果提供了 labels）
        loss = None if labels is None else self.hf_compute_loss(labels, reshaped_logits)
        
        # 如果不返回字典，则构建输出元组
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        # 返回 TF 模型输出对象，包括损失、logits、隐藏状态和注意力权重
        return TFMultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "roberta", None) is not None:
            with tf.name_scope(self.roberta.name):
                self.roberta.build(None)  # 构建 RoBERTa 层
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])  # 构建分类器层
@add_start_docstrings(
    """
    XLM RoBERTa Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    XLM_ROBERTA_START_DOCSTRING,
)
# 从 transformers.models.roberta.modeling_tf_roberta.TFRobertaForTokenClassification 复制而来，将 Roberta 改为 XLMRoberta，ROBERTA 改为 XLM_ROBERTA
class TFXLMRobertaForTokenClassification(TFXLMRobertaPreTrainedModel, TFTokenClassificationLoss):
    # 当从 PT 模型加载 TF 模型时，以下带 '.' 的名称表示在加载时可以忽略的未预期/缺失的层
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"lm_head"]
    # 当从 PT 模型加载 TF 模型时，以下名称表示可以忽略的缺失层
    _keys_to_ignore_on_load_missing = [r"dropout"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        # 使用 XLMRobertaMainLayer 初始化 self.roberta，不添加 pooling 层，命名为 "roberta"
        self.roberta = TFXLMRobertaMainLayer(config, add_pooling_layer=False, name="roberta")
        # 根据 config 中的 classifier_dropout 或 hidden_dropout_prob 初始化 Dropout 层
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = keras.layers.Dropout(classifier_dropout)
        # 使用 config 中的 initializer_range 初始化 Dense 层，输出维度为 config.num_labels，命名为 "classifier"
        self.classifier = keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLM_ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
    ) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 调用 RoBERTa 模型进行前向传播，并返回输出结果
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
        # 从 RoBERTa 输出的结果中取出序列输出（通常是最后一层的隐藏状态）
        sequence_output = outputs[0]

        # 对序列输出应用 dropout，用于防止过拟合
        sequence_output = self.dropout(sequence_output, training=training)
        # 将 dropout 后的输出传递给分类器，得到预测的 logits
        logits = self.classifier(sequence_output)

        # 如果提供了标签，计算损失值；否则损失值设为 None
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果 return_dict=False，则按照非字典格式组织输出
        if not return_dict:
            output = (logits,) + outputs[2:]  # 输出包括 logits 和 RoBERTa 的其他返回值
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict=True，则按 TFTokenClassifierOutput 格式组织输出
        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果已经构建过模型，直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果模型中包含 RoBERTa，构建 RoBERTa 层
        if getattr(self, "roberta", None) is not None:
            with tf.name_scope(self.roberta.name):
                self.roberta.build(None)
        # 如果模型中包含分类器，构建分类器并指定输入形状
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
@add_start_docstrings(
    """
    XLM RoBERTa Model with a span classification head on top for extractive question-answering tasks like SQuAD (a
    linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    XLM_ROBERTA_START_DOCSTRING,
)
# 基于 XLM RoBERTa 模型，在顶部增加了用于抽取式问答任务（如 SQuAD）的跨度分类头部（在隐藏状态输出的线性层上计算 `span start logits` 和 `span end logits`）。

class TFXLMRobertaForQuestionAnswering(TFXLMRobertaPreTrainedModel, TFQuestionAnsweringLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    # 当从 PyTorch 模型加载到 TF 模型时，带有 '.' 的名称代表授权的意外/丢失层
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"lm_head"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        # Initialize the XLM-RoBERTa main layer without adding a pooling layer
        # 初始化 XLM-RoBERTa 主层，不添加汇聚层
        self.roberta = TFXLMRobertaMainLayer(config, add_pooling_layer=False, name="roberta")
        
        # Dense layer for question answering output, initialized with specified initializer range
        # 用于问答输出的全连接层，使用指定的初始化范围初始化
        self.qa_outputs = keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLM_ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="ydshieh/roberta-base-squad2",
        output_type=TFQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="' puppet'",
        expected_loss=0.86,
    )
    # Forward method for the model, with specific docstrings added for model input details and examples
    # 模型的前向方法，添加了特定的文档字符串，描述了模型输入的细节和示例
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
        **kwargs
    ):
        # Method signature follows the TFModelInputType and accepts various optional inputs for model processing
        # 方法签名遵循 TFModelInputType，并接受各种可选输入进行模型处理
        pass  # Placeholder for the actual implementation of the call method
    ) -> Union[TFQuestionAnsweringModelOutput, Tuple[tf.Tensor]]:
        r"""
        start_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
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
        # 从 RoBERTa 模型中获取输出的序列表示
        sequence_output = outputs[0]

        # 将序列表示传入 QA 输出层得到 logits
        logits = self.qa_outputs(sequence_output)
        
        # 将 logits 按最后一个维度分割为 start_logits 和 end_logits
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        
        # 去除多余的维度，使得 start_logits 和 end_logits 的形状变为 (batch_size, sequence_length)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        # 计算损失（如果给定了起始位置和结束位置）
        loss = None
        if start_positions is not None and end_positions is not None:
            labels = {"start_position": start_positions}
            labels["end_position"] = end_positions
            # 使用 Hugging Face 的损失计算函数计算损失
            loss = self.hf_compute_loss(labels, (start_logits, end_logits))

        # 根据 return_dict 决定返回格式
        if not return_dict:
            # 如果不返回字典，则将 loss 和 output 打包成 tuple 返回
            output = (start_logits, end_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果返回字典，则构建 TFQuestionAnsweringModelOutput 对象并返回
        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经构建过，则直接返回

        # 如果 self.roberta 已定义，则构建它
        if getattr(self, "roberta", None) is not None:
            with tf.name_scope(self.roberta.name):
                self.roberta.build(None)

        # 如果 self.qa_outputs 已定义，则构建它
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                # 构建 QA 输出层，输入形状为 [None, None, self.config.hidden_size]
                self.qa_outputs.build([None, None, self.config.hidden_size])
```