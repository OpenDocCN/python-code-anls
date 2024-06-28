# `.\models\albert\modeling_tf_albert.py`

```py
# 设置编码格式为 UTF-8，确保代码中可以正确处理各种字符
# 版权声明，版权归 OpenAI Team Authors 和 HuggingFace Inc. team 所有
# 版权声明，版权归 NVIDIA CORPORATION 所有，保留所有权利
#
# 根据 Apache 许可证 2.0 版本使用本文件
# 除非符合许可证的相关法律要求或书面同意，否则不得使用本文件
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按“原样”分发，不附带任何明示或暗示的担保或条件。
# 有关详细信息，请参阅许可证。

""" TF 2.0 ALBERT model."""

from __future__ import annotations  # 允许在类型注释中使用未定义的类型名称

import math  # 导入数学库，用于执行数学运算
from dataclasses import dataclass  # 导入 dataclass 用于创建不可变对象
from typing import Dict, Optional, Tuple, Union  # 导入类型注释支持的类型

import numpy as np  # 导入 NumPy 库，用于数值计算
import tensorflow as tf  # 导入 TensorFlow 库，用于构建和训练深度学习模型

from ...activations_tf import get_tf_activation  # 导入自定义函数，用于获取 TensorFlow 激活函数
from ...modeling_tf_outputs import (  # 导入模型输出相关类
    TFBaseModelOutput,
    TFBaseModelOutputWithPooling,
    TFMaskedLMOutput,
    TFMultipleChoiceModelOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from ...modeling_tf_utils import (  # 导入模型工具函数
    TFMaskedLanguageModelingLoss,
    TFModelInputType,
    TFMultipleChoiceLoss,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFTokenClassificationLoss,
    get_initializer,
    keras,  # TensorFlow 的高级 API
    keras_serializable,  # 可序列化的 Keras
    unpack_inputs,  # 解包输入数据的函数
)
from ...tf_utils import (  # 导入 TensorFlow 实用函数
    check_embeddings_within_bounds,  # 检查嵌入向量是否在边界范围内
    shape_list,  # 获取张量的形状列表
    stable_softmax,  # 稳定的 softmax 函数
)
from ...utils import (  # 导入实用工具函数
    ModelOutput,  # 模型输出类
    add_code_sample_docstrings,  # 添加代码示例的文档字符串
    add_start_docstrings,  # 添加函数的起始文档字符串
    add_start_docstrings_to_model_forward,  # 添加模型前向传播的起始文档字符串
    logging,  # 日志记录工具
    replace_return_docstrings,  # 替换返回值的文档字符串
)
from .configuration_albert import AlbertConfig  # 导入 ALBERT 模型配置类

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

_CHECKPOINT_FOR_DOC = "albert/albert-base-v2"  # 文档中使用的模型检查点名称
_CONFIG_FOR_DOC = "AlbertConfig"  # 文档中使用的配置名称

TF_ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [  # ALBERT 预训练模型的列表
    "albert/albert-base-v1",
    "albert/albert-large-v1",
    "albert/albert-xlarge-v1",
    "albert/albert-xxlarge-v1",
    "albert/albert-base-v2",
    "albert/albert-large-v2",
    "albert/albert-xlarge-v2",
    "albert/albert-xxlarge-v2",
    # 查看所有 ALBERT 模型：https://huggingface.co/models?filter=albert
]


class TFAlbertPreTrainingLoss:
    """
    适用于 ALBERT 预训练的损失函数，即通过结合 SOP + MLM 的语言模型预训练任务。
    .. 注意:: 在损失计算中，任何标签为 -100 的样本将被忽略（以及对应的 logits）。
    """
    def hf_compute_loss(self, labels: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
        # 定义损失函数为稀疏分类交叉熵，从 logits 计算，不进行降维
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=keras.losses.Reduction.NONE)
        
        if self.config.tf_legacy_loss:
            # 确保只有标签不等于 -100 的位置会计算损失
            masked_lm_active_loss = tf.not_equal(tf.reshape(tensor=labels["labels"], shape=(-1,)), -100)
            # 使用布尔掩码从 logits 中筛选出有效位置的预测值
            masked_lm_reduced_logits = tf.boolean_mask(
                tensor=tf.reshape(tensor=logits[0], shape=(-1, shape_list(logits[0])[2])),
                mask=masked_lm_active_loss,
            )
            # 使用布尔掩码从标签中筛选出有效位置的真实值
            masked_lm_labels = tf.boolean_mask(
                tensor=tf.reshape(tensor=labels["labels"], shape=(-1,)), mask=masked_lm_active_loss
            )
            # 确保只有标签不等于 -100 的位置会计算损失
            sentence_order_active_loss = tf.not_equal(
                tf.reshape(tensor=labels["sentence_order_label"], shape=(-1,)), -100
            )
            # 使用布尔掩码从 logits 中筛选出有效位置的预测值
            sentence_order_reduced_logits = tf.boolean_mask(
                tensor=tf.reshape(tensor=logits[1], shape=(-1, 2)), mask=sentence_order_active_loss
            )
            # 使用布尔掩码从标签中筛选出有效位置的真实值
            sentence_order_label = tf.boolean_mask(
                tensor=tf.reshape(tensor=labels["sentence_order_label"], shape=(-1,)), mask=sentence_order_active_loss
            )
            # 计算掩码语言模型的损失
            masked_lm_loss = loss_fn(y_true=masked_lm_labels, y_pred=masked_lm_reduced_logits)
            # 计算序列顺序预测的损失
            sentence_order_loss = loss_fn(y_true=sentence_order_label, y_pred=sentence_order_reduced_logits)
            # 将掩码语言模型损失按照序列顺序预测的数量均匀化
            masked_lm_loss = tf.reshape(tensor=masked_lm_loss, shape=(-1, shape_list(sentence_order_loss)[0]))
            masked_lm_loss = tf.reduce_mean(input_tensor=masked_lm_loss, axis=0)

            return masked_lm_loss + sentence_order_loss

        # 将负标签裁剪为零，避免 NaN 和错误，这些位置后续将被掩盖
        unmasked_lm_losses = loss_fn(y_true=tf.nn.relu(labels["labels"]), y_pred=logits[0])
        # 确保只有标签不等于 -100 的位置会计算损失
        lm_loss_mask = tf.cast(labels["labels"] != -100, dtype=unmasked_lm_losses.dtype)
        masked_lm_losses = unmasked_lm_losses * lm_loss_mask
        reduced_masked_lm_loss = tf.reduce_sum(masked_lm_losses) / tf.reduce_sum(lm_loss_mask)

        sop_logits = tf.reshape(logits[1], (-1, 2))
        # 将负标签裁剪为零，避免 NaN 和错误，这些位置后续将被掩盖
        unmasked_sop_loss = loss_fn(y_true=tf.nn.relu(labels["sentence_order_label"]), y_pred=sop_logits)
        sop_loss_mask = tf.cast(labels["sentence_order_label"] != -100, dtype=unmasked_sop_loss.dtype)

        masked_sop_loss = unmasked_sop_loss * sop_loss_mask
        reduced_masked_sop_loss = tf.reduce_sum(masked_sop_loss) / tf.reduce_sum(sop_loss_mask)

        return tf.reshape(reduced_masked_lm_loss + reduced_masked_sop_loss, (1,))
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config: AlbertConfig, **kwargs):
        super().__init__(**kwargs)

        # 初始化层的配置和参数
        self.config = config
        self.embedding_size = config.embedding_size  # 嵌入向量的维度大小
        self.max_position_embeddings = config.max_position_embeddings  # 最大位置嵌入数量
        self.initializer_range = config.initializer_range  # 初始化范围
        # 层归一化操作，使用配置中的 epsilon 参数
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # Dropout 操作，使用配置中的 dropout 比率
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def build(self, input_shape=None):
        with tf.name_scope("word_embeddings"):
            # 添加词嵌入权重矩阵，形状为 [词汇表大小, 嵌入维度大小]
            self.weight = self.add_weight(
                name="weight",
                shape=[self.config.vocab_size, self.embedding_size],
                initializer=get_initializer(self.initializer_range),
            )

        with tf.name_scope("token_type_embeddings"):
            # 添加 token_type 嵌入权重矩阵，形状为 [token_type 数量, 嵌入维度大小]
            self.token_type_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.config.type_vocab_size, self.embedding_size],
                initializer=get_initializer(self.initializer_range),
            )

        with tf.name_scope("position_embeddings"):
            # 添加位置嵌入权重矩阵，形状为 [最大位置嵌入数量, 嵌入维度大小]
            self.position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_position_embeddings, self.embedding_size],
                initializer=get_initializer(self.initializer_range),
            )

        if self.built:
            return
        self.built = True
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                # 构建层归一化的结构，输入形状为 [None, None, 嵌入维度大小]
                self.LayerNorm.build([None, None, self.config.embedding_size])

    # Copied from transformers.models.bert.modeling_tf_bert.TFBertEmbeddings.call
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
        # 检查输入参数，确保至少提供了 `input_ids` 或 `inputs_embeds`
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Need to provide either `input_ids` or `input_embeds`.")

        # 如果提供了 `input_ids`，从权重矩阵中根据索引收集对应的嵌入向量
        if input_ids is not None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        # 获取输入嵌入张量的形状
        input_shape = shape_list(inputs_embeds)[:-1]

        # 如果未提供 `token_type_ids`，则创建一个形状与输入嵌入张量相同的张量，并用0填充
        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)

        # 如果未提供 `position_ids`，则根据序列长度和历史键值长度生成位置索引张量
        if position_ids is None:
            position_ids = tf.expand_dims(
                tf.range(start=past_key_values_length, limit=input_shape[1] + past_key_values_length), axis=0
            )

        # 根据位置索引从位置嵌入矩阵中收集位置嵌入向量
        position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
        # 根据 token_type_ids 从 token_type_embeddings 中收集 token 类型嵌入向量
        token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)
        # 将输入嵌入向量、位置嵌入向量和 token 类型嵌入向量相加，得到最终的嵌入向量
        final_embeddings = inputs_embeds + position_embeds + token_type_embeds
        # 对最终的嵌入向量进行 LayerNormalization 处理
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        # 在训练模式下对最终嵌入向量进行 dropout 处理
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)

        # 返回最终的嵌入向量作为输出
        return final_embeddings
    """Contains the complete attention sublayer, including both dropouts and layer norm."""

    def __init__(self, config: AlbertConfig, **kwargs):
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
        self.output_attentions = config.output_attentions

        # Initialize Dense layers for query, key, value, and dense transformations
        self.query = keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        self.key = keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key"
        )
        self.value = keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )
        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        # Layer normalization for post-attention processing
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")

        # Dropout layers with specified dropout rates
        self.attention_dropout = keras.layers.Dropout(rate=config.attention_probs_dropout_prob)
        self.output_dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)

        self.config = config

    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        # Reshape from [batch_size, seq_length, all_head_size] to [batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))

        # Transpose the tensor from [batch_size, seq_length, num_attention_heads, attention_head_size] to [batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(
        self,
        input_tensor: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        # 获取输入张量的批量大小
        batch_size = shape_list(input_tensor)[0]
        # 调用 self.query 方法，生成混合查询层张量
        mixed_query_layer = self.query(inputs=input_tensor)
        # 调用 self.key 方法，生成混合键层张量
        mixed_key_layer = self.key(inputs=input_tensor)
        # 调用 self.value 方法，生成混合值层张量
        mixed_value_layer = self.value(inputs=input_tensor)
        # 将混合查询层张量转置以适应注意力分数计算的形状
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        # 将混合键层张量转置以适应注意力分数计算的形状
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        # 将混合值层张量转置以适应注意力分数计算的形状
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        # 计算查询和键之间的点积，得到原始的注意力分数
        # 形状为 (batch size, num_heads, seq_len_q, seq_len_k)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        # 计算缩放因子 dk，并将注意力分数除以 dk
        dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(attention_scores, dk)

        if attention_mask is not None:
            # 如果存在注意力掩码，应用注意力掩码
            attention_scores = tf.add(attention_scores, attention_mask)

        # 将注意力分数归一化为概率值
        attention_probs = stable_softmax(logits=attention_scores, axis=-1)

        # 对注意力概率值进行 dropout 处理
        attention_probs = self.attention_dropout(inputs=attention_probs, training=training)

        # 如果存在头部掩码，将注意力概率值与头部掩码相乘
        if head_mask is not None:
            attention_probs = tf.multiply(attention_probs, head_mask)

        # 计算上下文张量，将注意力概率值与值层张量相乘
        context_layer = tf.matmul(attention_probs, value_layer)
        # 对上下文张量进行转置操作，调整其形状
        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])

        # 调整上下文张量的形状，以适应下一层网络的输入要求
        # 形状为 (batch_size, seq_len_q, all_head_size)
        context_layer = tf.reshape(tensor=context_layer, shape=(batch_size, -1, self.all_head_size))
        # 将上下文张量作为 self_outputs 的第一个元素
        self_outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        # 获取 self_outputs 的第一个元素作为隐藏状态张量
        hidden_states = self_outputs[0]
        # 将隐藏状态张量传递给全连接层进行线性变换
        hidden_states = self.dense(inputs=hidden_states)
        # 对线性变换后的隐藏状态进行 dropout 处理
        hidden_states = self.output_dropout(inputs=hidden_states, training=training)
        # 将 dropout 后的隐藏状态与输入张量相加，并应用 LayerNorm
        attention_output = self.LayerNorm(inputs=hidden_states + input_tensor)

        # 如果需要输出注意力分数，则将注意力分数添加到输出中
        outputs = (attention_output,) + self_outputs[1:]

        # 返回最终的输出
        return outputs
    # 如果已经构建过网络结构，则直接返回，不再重复构建
    if self.built:
        return
    # 将标记置为已构建
    self.built = True
    
    # 如果存在查询（query）模块，根据其名称创建作用域，并构建其形状
    if getattr(self, "query", None) is not None:
        with tf.name_scope(self.query.name):
            self.query.build([None, None, self.config.hidden_size])
    
    # 如果存在键（key）模块，根据其名称创建作用域，并构建其形状
    if getattr(self, "key", None) is not None:
        with tf.name_scope(self.key.name):
            self.key.build([None, None, self.config.hidden_size])
    
    # 如果存在值（value）模块，根据其名称创建作用域，并构建其形状
    if getattr(self, "value", None) is not None:
        with tf.name_scope(self.value.name):
            self.value.build([None, None, self.config.hidden_size])
    
    # 如果存在密集层（dense），根据其名称创建作用域，并构建其形状
    if getattr(self, "dense", None) is not None:
        with tf.name_scope(self.dense.name):
            self.dense.build([None, None, self.config.hidden_size])
    
    # 如果存在层归一化（LayerNorm），根据其名称创建作用域，并构建其形状
    if getattr(self, "LayerNorm", None) is not None:
        with tf.name_scope(self.LayerNorm.name):
            self.LayerNorm.build([None, None, self.config.hidden_size])
class TFAlbertLayer(keras.layers.Layer):
    def __init__(self, config: AlbertConfig, **kwargs):
        super().__init__(**kwargs)

        # 初始化注意力层，使用给定的配置
        self.attention = TFAlbertAttention(config, name="attention")
        
        # 初始化前馈神经网络层，使用给定的中间大小和初始化器范围
        self.ffn = keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="ffn"
        )

        # 根据配置获取激活函数，或者使用默认的激活函数
        if isinstance(config.hidden_act, str):
            self.activation = get_tf_activation(config.hidden_act)
        else:
            self.activation = config.hidden_act

        # 初始化前馈神经网络输出层，使用给定的隐藏大小和初始化器范围
        self.ffn_output = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="ffn_output"
        )
        
        # 初始化全层标准化层，使用给定的 epsilon 参数
        self.full_layer_layer_norm = keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="full_layer_layer_norm"
        )
        
        # 初始化 dropout 层，使用给定的隐藏 dropout 概率
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        # 调用注意力层，获取注意力输出
        attention_outputs = self.attention(
            input_tensor=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            training=training,
        )
        
        # 前馈神经网络计算过程
        ffn_output = self.ffn(inputs=attention_outputs[0])  # 使用注意力输出作为输入
        ffn_output = self.activation(ffn_output)  # 应用激活函数
        ffn_output = self.ffn_output(inputs=ffn_output)  # 再次使用前馈神经网络输出层
        
        # 应用 dropout 操作
        ffn_output = self.dropout(inputs=ffn_output, training=training)
        
        # 添加全层标准化层，结合注意力输出和前馈神经网络输出
        hidden_states = self.full_layer_layer_norm(inputs=ffn_output + attention_outputs[0])

        # 如果需要输出注意力，则将注意力输出包含在结果中
        outputs = (hidden_states,) + attention_outputs[1:]

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        
        # 构建注意力层
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        
        # 构建前馈神经网络层
        if getattr(self, "ffn", None) is not None:
            with tf.name_scope(self.ffn.name):
                self.ffn.build([None, None, self.config.hidden_size])
        
        # 构建前馈神经网络输出层
        if getattr(self, "ffn_output", None) is not None:
            with tf.name_scope(self.ffn_output.name):
                self.ffn_output.build([None, None, self.config.intermediate_size])
        
        # 构建全层标准化层
        if getattr(self, "full_layer_layer_norm", None) is not None:
            with tf.name_scope(self.full_layer_layer_norm.name):
                self.full_layer_layer_norm.build([None, None, self.config.hidden_size])
    # 使用传入的 AlbertConfig 对象初始化模型，调用父类的初始化方法
    def __init__(self, config: AlbertConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建多个 AlbertLayer 层组成的列表，每个层有一个唯一的名称
        self.albert_layers = [
            TFAlbertLayer(config, name=f"albert_layers_._{i}") for i in range(config.inner_group_num)
        ]

    # 模型的调用方法，接收隐藏状态、注意力掩码、头部掩码等输入，输出模型的隐藏状态、层的隐藏状态和注意力分数（如果有的话）
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        output_hidden_states: bool,
        training: bool = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        # 如果需要输出隐藏状态，则创建一个空元组用于存储每个层的隐藏状态
        layer_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力分数，则创建一个空元组用于存储每个层的注意力分数
        layer_attentions = () if output_attentions else None

        # 遍历所有 AlbertLayer 层
        for layer_index, albert_layer in enumerate(self.albert_layers):
            # 如果需要输出隐藏状态，则将当前层的隐藏状态添加到存储中
            if output_hidden_states:
                layer_hidden_states = layer_hidden_states + (hidden_states,)

            # 调用当前 AlbertLayer 层的处理方法，更新隐藏状态
            layer_output = albert_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[layer_index],
                output_attentions=output_attentions,
                training=training,
            )
            # 更新主要的隐藏状态为当前层的输出隐藏状态
            hidden_states = layer_output[0]

            # 如果需要输出注意力分数，则将当前层的注意力分数添加到存储中
            if output_attentions:
                layer_attentions = layer_attentions + (layer_output[1],)

        # 添加最后一层的隐藏状态到存储中（如果需要输出隐藏状态）
        if output_hidden_states:
            layer_hidden_states = layer_hidden_states + (hidden_states,)

        # 返回隐藏状态、层的隐藏状态和注意力分数的元组，去除其中为 None 的部分
        return tuple(v for v in [hidden_states, layer_hidden_states, layer_attentions] if v is not None)

    # 构建模型，在第一次调用前进行模型的构建
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经创建了 AlbertLayer 层，则依次构建每一层
        if getattr(self, "albert_layers", None) is not None:
            for layer in self.albert_layers:
                # 使用每个层的名称创建一个命名空间，并调用层的构建方法
                with tf.name_scope(layer.name):
                    layer.build(None)
# 定义一个名为TFAlbertTransformer的类，继承自keras的Layer类
class TFAlbertTransformer(keras.layers.Layer):
    # 初始化方法，接受config和其他参数
    def __init__(self, config: AlbertConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 从config中获取隐藏层的数量和组数
        self.num_hidden_layers = config.num_hidden_layers
        self.num_hidden_groups = config.num_hidden_groups
        # 计算每个隐藏组中的层的数量
        self.layers_per_group = int(config.num_hidden_layers / config.num_hidden_groups)
        # 创建一个Dense层来映射嵌入的隐藏状态
        self.embedding_hidden_mapping_in = keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="embedding_hidden_mapping_in",
        )
        # 创建多个ALBERT层组，数量等于隐藏组的数量
        self.albert_layer_groups = [
            TFAlbertLayerGroup(config, name=f"albert_layer_groups_._{i}") for i in range(config.num_hidden_groups)
        ]
        # 保存config
        self.config = config

    # 定义处理输入数据的方法
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        # 使用Dense层处理输入的隐藏状态
        hidden_states = self.embedding_hidden_mapping_in(inputs=hidden_states)
        # 初始化存储注意力权重和隐藏状态的变量
        all_attentions = () if output_attentions else None
        all_hidden_states = (hidden_states,) if output_hidden_states else None

        # 循环遍历每个隐藏层
        for i in range(self.num_hidden_layers):
            # 计算当前层所在的隐藏组的索引
            group_idx = int(i / (self.num_hidden_layers / self.num_hidden_groups))
            # 调用对应的ALBERT层组，处理隐藏状态
            layer_group_output = self.albert_layer_groups[group_idx](
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[group_idx * self.layers_per_group : (group_idx + 1) * self.layers_per_group],
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                training=training,
            )
            # 更新隐藏状态
            hidden_states = layer_group_output[0]

            # 如果需要输出注意力权重，则将当前层的注意力权重添加到存储变量中
            if output_attentions:
                all_attentions = all_attentions + layer_group_output[-1]

            # 如果需要输出隐藏状态，则将当前层的隐藏状态添加到存储变量中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要返回字典形式的结果，则将所有结果组合成一个元组返回
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

        # 如果需要返回字典形式的结果，则创建一个TFBaseModelOutput对象并返回
        return TFBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )
    # 定义 build 方法，用于构建模型的结构
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回，避免重复构建
        if self.built:
            return
        # 将模型标记为已构建状态
        self.built = True
        
        # 如果存在 embedding_hidden_mapping_in 属性
        if getattr(self, "embedding_hidden_mapping_in", None) is not None:
            # 使用 tf.name_scope 来限定命名空间，命名为 embedding_hidden_mapping_in 的名称
            with tf.name_scope(self.embedding_hidden_mapping_in.name):
                # 使用 embedding_hidden_mapping_in 属性构建层，输入形状为 [None, None, self.config.embedding_size]
                self.embedding_hidden_mapping_in.build([None, None, self.config.embedding_size])
        
        # 如果存在 albert_layer_groups 属性
        if getattr(self, "albert_layer_groups", None) is not None:
            # 遍历 albert_layer_groups 中的每个层
            for layer in self.albert_layer_groups:
                # 使用 tf.name_scope 来限定命名空间，命名为 layer 的名称
                with tf.name_scope(layer.name):
                    # 构建当前层，输入形状为 None（未指定特定输入形状）
                    layer.build(None)
    """
    处理权重初始化、预训练模型下载和加载的抽象类。
    """
    
    # 配置类为 AlbertConfig
    config_class = AlbertConfig
    # 基础模型前缀为 "albert"
    base_model_prefix = "albert"

class TFAlbertMLMHead(keras.layers.Layer):
    def __init__(self, config: AlbertConfig, input_embeddings: keras.layers.Layer, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.embedding_size = config.embedding_size
        
        # 创建一个全连接层，用于预测下一个词的特征
        self.dense = keras.layers.Dense(
            config.embedding_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        
        # 根据配置中的激活函数类型，获取激活函数
        if isinstance(config.hidden_act, str):
            self.activation = get_tf_activation(config.hidden_act)
        else:
            self.activation = config.hidden_act
        
        # LayerNormalization 层，用于归一化输入的词嵌入
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")

        # 输入词嵌入层，用于解码器的输出权重
        self.decoder = input_embeddings

    def build(self, input_shape=None):
        # 增加偏置项，用于每个词汇的输出偏置
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="bias")
        
        # 解码器的偏置项，用于每个词汇的解码偏置
        self.decoder_bias = self.add_weight(
            shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="decoder/bias"
        )

        # 如果已经构建，则直接返回
        if self.built:
            return
        self.built = True
        
        # 构建全连接层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        
        # 构建 LayerNormalization 层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.embedding_size])

    def get_output_embeddings(self) -> keras.layers.Layer:
        # 返回解码器的词嵌入层
        return self.decoder

    def set_output_embeddings(self, value: tf.Variable):
        # 设置解码器的权重
        self.decoder.weight = value
        self.decoder.vocab_size = shape_list(value)[0]

    def get_bias(self) -> Dict[str, tf.Variable]:
        # 返回偏置项字典
        return {"bias": self.bias, "decoder_bias": self.decoder_bias}

    def set_bias(self, value: tf.Variable):
        # 设置偏置项的值
        self.bias = value["bias"]
        self.decoder_bias = value["decoder_bias"]
        self.config.vocab_size = shape_list(value["bias"])[0]
    # 使用全连接层对隐藏状态进行线性变换
    hidden_states = self.dense(inputs=hidden_states)
    # 应用激活函数对线性变换后的隐藏状态进行非线性变换
    hidden_states = self.activation(hidden_states)
    # 应用层归一化操作对隐藏状态进行归一化处理
    hidden_states = self.LayerNorm(inputs=hidden_states)
    # 获取隐藏状态张量的第二个维度，即序列长度
    seq_length = shape_list(tensor=hidden_states)[1]
    # 对隐藏状态张量进行形状重塑，将其转换为二维张量
    hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.embedding_size])
    # 对重塑后的隐藏状态张量与解码器权重矩阵进行矩阵乘法运算
    hidden_states = tf.matmul(a=hidden_states, b=self.decoder.weight, transpose_b=True)
    # 将矩阵乘法结果的张量形状重塑为三维张量，恢复为序列长度相关的形状
    hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])
    # 对矩阵乘法结果张量添加偏置项
    hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.decoder_bias)

    # 返回经过线性变换、激活函数、归一化、矩阵乘法、偏置项处理后的隐藏状态张量
    return hidden_states
# 使用 keras_serializable 装饰器将类 TFAlbertMainLayer 序列化为 Keras 层
@keras_serializable
class TFAlbertMainLayer(keras.layers.Layer):
    # 设置配置类为 AlbertConfig
    config_class = AlbertConfig

    # 初始化函数，接受 AlbertConfig 类型的 config 和一个布尔值 add_pooling_layer
    def __init__(self, config: AlbertConfig, add_pooling_layer: bool = True, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 保存传入的配置对象
        self.config = config

        # 创建 TFAlbertEmbeddings 层，并命名为 "embeddings"
        self.embeddings = TFAlbertEmbeddings(config, name="embeddings")

        # 创建 TFAlbertTransformer 层，并命名为 "encoder"
        self.encoder = TFAlbertTransformer(config, name="encoder")

        # 如果 add_pooling_layer 为 True，则创建一个 Dense 层作为池化层，否则为 None
        self.pooler = (
            keras.layers.Dense(
                units=config.hidden_size,
                kernel_initializer=get_initializer(config.initializer_range),
                activation="tanh",
                name="pooler",
            )
            if add_pooling_layer
            else None
        )

    # 返回输入嵌入层 embeddings
    def get_input_embeddings(self) -> keras.layers.Layer:
        return self.embeddings

    # 设置输入嵌入层的权重值和词汇大小
    def set_input_embeddings(self, value: tf.Variable):
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    # 未实现的方法，用于剪枝模型中的头部
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    # 使用 unpack_inputs 装饰器，处理输入的各种参数
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
        # 函数内容未提供

    # 构建层，如果已经构建则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        
        # 如果存在嵌入层 embeddings，则构建其内部结构
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        
        # 如果存在编码器 encoder，则构建其内部结构
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        
        # 如果存在池化层 pooler，则构建其内部结构，输入形状为 [None, None, self.config.hidden_size]
        if getattr(self, "pooler", None) is not None:
            with tf.name_scope(self.pooler.name):
                self.pooler.build([None, None, self.config.hidden_size])


# 使用 dataclass 装饰器创建 TFAlbertForPreTrainingOutput 类，继承自 ModelOutput
@dataclass
class TFAlbertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`TFAlbertForPreTraining`].
    """
    loss: tf.Tensor = None
    # 损失值，初始化为 None

    prediction_logits: tf.Tensor = None
    # 语言建模头部的预测分数张量，形状为 `(batch_size, sequence_length, config.vocab_size)`，在 SoftMax 之前的分数。

    sop_logits: tf.Tensor = None
    # 下一个序列预测（分类）头部的预测分数张量，形状为 `(batch_size, 2)`，在 SoftMax 之前的分数，表示 True/False 的延续。

    hidden_states: Tuple[tf.Tensor] | None = None
    # 当 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回的隐藏状态元组，
    # 包含每个层的输出张量和初始嵌入输出，形状为 `(batch_size, sequence_length, hidden_size)`。

    attentions: Tuple[tf.Tensor] | None = None
    # 当 `output_attentions=True` 或 `config.output_attentions=True` 时返回的注意力张量元组，
    # 包含每个层的注意力权重张量，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
"""
    这个模型继承自 `TFPreTrainedModel`。查看超类文档以获取库实现的通用方法，比如下载或保存模型、调整输入嵌入大小、修剪头等。

    这个模型也是 [keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) 的子类。将其用作常规的 TF 2.0 Keras 模型，并参考 TF 2.0 文档，以获取所有与一般使用和行为相关的信息。

    <Tip>

    `transformers` 中的 TensorFlow 模型和层接受两种输入格式：

    - 将所有输入作为关键字参数（类似于 PyTorch 模型），或者
    - 将所有输入作为列表、元组或字典传递给第一个位置参数。

    支持第二种格式的原因在于，Keras 方法在将输入传递给模型和层时更喜欢这种格式。由于这种支持，当使用诸如 `model.fit()` 这样的方法时，只需传递模型支持的任何格式的输入和标签即可！然而，如果您想在 Keras 方法之外使用第二种格式，比如在使用 Keras `Functional` API 创建自己的层或模型时，可以使用三种可能性来收集第一个位置参数中的所有输入张量：

    - 只有 `input_ids` 的单个张量：`model(input_ids)`
    - 长度可变的列表，按照文档字符串中给定的顺序包含一个或多个输入张量：`model([input_ids, attention_mask])` 或 `model([input_ids, attention_mask, token_type_ids])`
    - 一个字典，将一个或多个输入张量与文档字符串中给定的输入名称相关联：`model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    请注意，当使用 [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) 创建模型和层时，您无需担心这些问题，因为可以像将输入传递给任何其他 Python 函数一样传递输入！

    </Tip>

    Args:
        config ([`AlbertConfig`]): 包含模型所有参数的模型配置类。
            使用配置文件初始化不会加载与模型关联的权重，仅加载配置。查看 [`~PreTrainedModel.from_pretrained`] 方法以加载模型权重。
"""

@add_start_docstrings(
    "不带任何特定头部的裸 Albert 模型变压器输出原始隐藏状态。",
    ALBERT_START_DOCSTRING,
)
class TFAlbertModel(TFAlbertPreTrainedModel):
    def __init__(self, config: AlbertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.albert = TFAlbertMainLayer(config, name="albert")

    @unpack_inputs
    # 使用装饰器添加模型前向传播的文档字符串，指定ALBERT模型输入的批次大小和序列长度
    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 使用装饰器添加代码示例的文档字符串，包括检查点、输出类型、配置类等信息
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义模型的前向传播方法，接收多个可能为None的输入参数，并返回模型输出
    def call(
        self,
        input_ids: TFModelInputType | None = None,  # 输入的token ID序列，可以为None
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 注意力遮罩，可以为None
        token_type_ids: np.ndarray | tf.Tensor | None = None,  # token类型ID，可以为None
        position_ids: np.ndarray | tf.Tensor | None = None,  # 位置ID，可以为None
        head_mask: np.ndarray | tf.Tensor | None = None,  # 头部遮罩，可以为None
        inputs_embeds: np.ndarray | tf.Tensor | None = None,  # 输入的嵌入表示，可以为None
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，默认为None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，默认为None
        return_dict: Optional[bool] = None,  # 是否以字典形式返回输出，默认为None
        training: Optional[bool] = False,  # 是否处于训练模式，默认为False
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        # 调用ALBERT模型的前向传播方法，传递所有参数，并接收输出
        outputs = self.albert(
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
        # 返回ALBERT模型的输出
        return outputs

    # 构建方法，用于构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建完成，直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果模型的albert属性存在
        if getattr(self, "albert", None) is not None:
            # 在albert的命名空间内构建albert模型
            with tf.name_scope(self.albert.name):
                # 构建albert模型，不需要输入形状参数
                self.albert.build(None)
"""
Albert Model with two heads on top for pretraining: a `masked language modeling` head and a `sentence order
prediction` (classification) head.
"""
# 继承 TFAlbertPreTrainedModel 和 TFAlbertPreTrainingLoss 类，实现预训练模型
@add_start_docstrings(
    """
    Albert Model with two heads on top for pretraining: a `masked language modeling` head and a `sentence order
    prediction` (classification) head.
    """,
    ALBERT_START_DOCSTRING,  # 添加 Albert 模型的起始文档字符串
)
class TFAlbertForPreTraining(TFAlbertPreTrainedModel, TFAlbertPreTrainingLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    # 在从 PyTorch 模型加载 TF 模型时，带 '.' 的名称表示被授权的意外/缺失的层
    _keys_to_ignore_on_load_unexpected = [r"predictions.decoder.weight"]

    def __init__(self, config: AlbertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 初始化 Albert 预训练模型，设定标签数
        self.num_labels = config.num_labels

        # Albert 主层，使用 TFAlbertMainLayer 初始化，命名为 "albert"
        self.albert = TFAlbertMainLayer(config, name="albert")

        # Albert MLM 头部，使用 TFAlbertMLMHead 初始化，输入嵌入使用 self.albert.embeddings，命名为 "predictions"
        self.predictions = TFAlbertMLMHead(config, input_embeddings=self.albert.embeddings, name="predictions")

        # Albert SOP 分类头部，使用 TFAlbertSOPHead 初始化，命名为 "sop_classifier"
        self.sop_classifier = TFAlbertSOPHead(config, name="sop_classifier")

    # 返回 MLM 头部
    def get_lm_head(self) -> keras.layers.Layer:
        return self.predictions

    # 模型的前向传播函数，接受一系列输入，参照 ALBERT_INPUTS_DOCSTRING 添加起始文档字符串
    @unpack_inputs
    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFAlbertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
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
        sentence_order_label: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
        **kwargs,
    ) -> Union[TFAlbertForPreTrainingOutput, Tuple[tf.Tensor]]:
        # 省略的部分对于模型前向传播的具体实现，输出和配置信息
        ) -> Union[TFAlbertForPreTrainingOutput, Tuple[tf.Tensor]]:
        r"""
        Return:

        Example:

        ```
        >>> import tensorflow as tf
        >>> from transformers import AutoTokenizer, TFAlbertForPreTraining

        >>> tokenizer = AutoTokenizer.from_pretrained("albert/albert-base-v2")
        >>> model = TFAlbertForPreTraining.from_pretrained("albert/albert-base-v2")

        >>> input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True))[None, :]
        >>> # Batch size 1
        >>> outputs = model(input_ids)

        >>> prediction_logits = outputs.prediction_logits
        >>> sop_logits = outputs.sop_logits
        ```"""

        # 调用 self.albert 模型进行预测
        outputs = self.albert(
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
        # 提取模型输出的序列输出和池化输出
        sequence_output, pooled_output = outputs[:2]
        # 使用 predictions 层生成预测分数
        prediction_scores = self.predictions(hidden_states=sequence_output)
        # 使用 sop_classifier 层生成 SOP 分类分数
        sop_scores = self.sop_classifier(pooled_output=pooled_output, training=training)
        # 初始化总损失
        total_loss = None

        # 如果有标签和句子顺序标签，则计算损失
        if labels is not None and sentence_order_label is not None:
            # 构建标签字典
            d_labels = {"labels": labels}
            d_labels["sentence_order_label"] = sentence_order_label
            # 使用 hf_compute_loss 计算总损失
            total_loss = self.hf_compute_loss(labels=d_labels, logits=(prediction_scores, sop_scores))

        # 如果 return_dict 为 False，则返回扁平化的输出元组
        if not return_dict:
            output = (prediction_scores, sop_scores) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        # 如果 return_dict 为 True，则返回 TFAlbertForPreTrainingOutput 对象
        return TFAlbertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            sop_logits=sop_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 设置构建标志为 True
        self.built = True
        # 如果存在 self.albert 属性，则构建 self.albert 模型
        if getattr(self, "albert", None) is not None:
            with tf.name_scope(self.albert.name):
                self.albert.build(None)
        # 如果存在 self.predictions 属性，则构建 self.predictions 层
        if getattr(self, "predictions", None) is not None:
            with tf.name_scope(self.predictions.name):
                self.predictions.build(None)
        # 如果存在 self.sop_classifier 属性，则构建 self.sop_classifier 层
        if getattr(self, "sop_classifier", None) is not None:
            with tf.name_scope(self.sop_classifier.name):
                self.sop_classifier.build(None)
# 定义 TFAlbertSOPHead 类，继承自 keras 的 Layer 类
class TFAlbertSOPHead(keras.layers.Layer):
    
    # 初始化方法，接受 AlbertConfig 类型的 config 参数和其他关键字参数
    def __init__(self, config: AlbertConfig, **kwargs):
        super().__init__(**kwargs)
        
        # 使用 config 的 classifier_dropout_prob 属性创建一个 Dropout 层
        self.dropout = keras.layers.Dropout(rate=config.classifier_dropout_prob)
        
        # 使用 config 的 num_labels 属性和 initializer_range 属性创建一个全连接 Dense 层
        self.classifier = keras.layers.Dense(
            units=config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="classifier",
        )
        
        # 将 config 参数存储在实例变量中
        self.config = config

    # call 方法用于定义层的前向传播逻辑
    def call(self, pooled_output: tf.Tensor, training: bool) -> tf.Tensor:
        # 对输入的 pooled_output 应用 Dropout 操作
        dropout_pooled_output = self.dropout(inputs=pooled_output, training=training)
        
        # 将 Dropout 后的输出传递给全连接 Dense 层，得到 logits
        logits = self.classifier(inputs=dropout_pooled_output)

        # 返回 logits
        return logits

    # build 方法用于构建层，在此方法中创建层的变量
    def build(self, input_shape=None):
        # 如果层已经构建过，直接返回
        if self.built:
            return
        
        # 将层标记为已构建
        self.built = True
        
        # 如果 self.classifier 存在，则在名为 self.classifier 的命名作用域下构建全连接层
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])


# 使用装饰器 @add_start_docstrings 添加文档字符串描述 Albert Model 的语言建模头部
@add_start_docstrings("""Albert Model with a `language modeling` head on top.""", ALBERT_START_DOCSTRING)
class TFAlbertForMaskedLM(TFAlbertPreTrainedModel, TFMaskedLanguageModelingLoss):
    
    # 定义 _keys_to_ignore_on_load_unexpected 属性，用于在加载 TF 模型时忽略指定的层
    # 名称中带有 '.' 表示的是从 PT 模型加载 TF 模型时可能会出现的未预期的或丢失的层
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"predictions.decoder.weight"]

    # 初始化方法，接受 AlbertConfig 类型的 config 参数和其他位置和关键字参数
    def __init__(self, config: AlbertConfig, *inputs, **kwargs):
        # 调用父类 TFAlbertPreTrainedModel 的初始化方法
        super().__init__(config, *inputs, **kwargs)
        
        # 创建 TFAlbertMainLayer 类的实例 albert，设置 add_pooling_layer=False，命名为 "albert"
        self.albert = TFAlbertMainLayer(config, add_pooling_layer=False, name="albert")
        
        # 创建 TFAlbertMLMHead 类的实例 predictions，设置 input_embeddings 为 self.albert.embeddings，命名为 "predictions"
        self.predictions = TFAlbertMLMHead(config, input_embeddings=self.albert.embeddings, name="predictions")

    # 返回预测头部的方法，返回 self.predictions
    def get_lm_head(self) -> keras.layers.Layer:
        return self.predictions

    # 使用装饰器 @unpack_inputs、@add_start_docstrings_to_model_forward 和 @replace_return_docstrings
    # 添加文档字符串描述 call 方法的输入和输出
    @unpack_inputs
    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        outputs = self.albert(
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
        # 从 ALBERT 模型中获取输出结果，包括序列输出和其他选项
        sequence_output = outputs[0]
        # 使用序列输出计算预测得分
        prediction_scores = self.predictions(hidden_states=sequence_output, training=training)
        # 如果提供了标签，计算损失；否则损失为 None
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=prediction_scores)

        # 如果不要求返回字典形式的结果，按顺序返回预测得分和其他输出
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 否则，返回 TFMaskedLMOutput 对象，包括损失、预测得分、隐藏状态和注意力权重
        return TFMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    # 定义 build 方法，用于构建模型
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回，避免重复构建
        if self.built:
            return
        # 设置标记表示已经构建
        self.built = True
        
        # 如果模型中存在名为 albert 的属性，开始构建 albert 部分
        if getattr(self, "albert", None) is not None:
            # 使用 albert 的名称作为命名空间，开始构建 albert
            with tf.name_scope(self.albert.name):
                self.albert.build(None)
        
        # 如果模型中存在名为 predictions 的属性，开始构建 predictions 部分
        if getattr(self, "predictions", None) is not None:
            # 使用 predictions 的名称作为命名空间，开始构建 predictions
            with tf.name_scope(self.predictions.name):
                self.predictions.build(None)
# 使用装饰器添加文档字符串，描述了这个类的用途和结构
@add_start_docstrings(
    """
    Albert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    ALBERT_START_DOCSTRING,
)
# 定义 TFAlbertForSequenceClassification 类，继承自 TFAlbertPreTrainedModel 和 TFSequenceClassificationLoss
class TFAlbertForSequenceClassification(TFAlbertPreTrainedModel, TFSequenceClassificationLoss):
    # 在加载过程中忽略的不期望/缺失的层名称列表
    _keys_to_ignore_on_load_unexpected = [r"predictions"]
    # 在加载过程中忽略的缺失的层名称列表
    _keys_to_ignore_on_load_missing = [r"dropout"]

    # 构造方法，初始化类的实例
    def __init__(self, config: AlbertConfig, *inputs, **kwargs):
        # 调用父类的构造方法
        super().__init__(config, *inputs, **kwargs)

        # 设置类别数目
        self.num_labels = config.num_labels

        # 创建 Albert 主层，使用给定的配置和名称
        self.albert = TFAlbertMainLayer(config, name="albert")
        # 添加 Dropout 层，使用给定的分类器 dropout 概率
        self.dropout = keras.layers.Dropout(rate=config.classifier_dropout_prob)
        # 添加 Dense 层作为分类器，设置输出单元数为类别数目，使用给定的初始化器范围和名称
        self.classifier = keras.layers.Dense(
            units=config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        # 保存配置对象
        self.config = config

    # 使用装饰器添加文档字符串，描述了 call 方法的输入和输出
    @unpack_inputs
    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="vumichien/albert-base-v2-imdb",
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'LABEL_1'",
        expected_loss=0.12,
    )
    # 定义 call 方法，实现模型的前向传播
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
        # 可选的输入参数，用于自动解包输入
    ) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 调用 ALBERT 模型进行前向传播，获取模型输出
        outputs = self.albert(
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
        # 从 ALBERT 模型输出中获取汇聚输出
        pooled_output = outputs[1]
        # 对汇聚输出应用 dropout，以防止过拟合
        pooled_output = self.dropout(inputs=pooled_output, training=training)
        # 使用分类器模型对汇聚输出进行分类，得到预测 logits
        logits = self.classifier(inputs=pooled_output)
        # 如果提供了标签，则计算损失值
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        # 如果不要求返回字典，则按顺序返回 logits 和可能的额外输出
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TFSequenceClassifierOutput 对象，包括损失、logits、隐藏状态和注意力权重
        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果模型已经构建过，则直接返回
        if getattr(self, "albert", None) is not None:
            # 使用 ALBERT 模型的名字空间构建模型
            with tf.name_scope(self.albert.name):
                self.albert.build(None)
        # 如果存在分类器模型，则使用分类器的名字空间构建模型
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
@add_start_docstrings(
    """
    Albert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    ALBERT_START_DOCSTRING,
)
class TFAlbertForTokenClassification(TFAlbertPreTrainedModel, TFTokenClassificationLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    # 定义在从 PyTorch 模型加载 TF 模型时，可以忽略的意外/缺失层的名称列表
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"predictions"]
    # 定义在从 PyTorch 模型加载 TF 模型时，可以忽略的缺失层的名称列表
    _keys_to_ignore_on_load_missing = [r"dropout"]

    def __init__(self, config: AlbertConfig, *inputs, **kwargs):
        # 调用父类构造函数初始化模型
        super().__init__(config, *inputs, **kwargs)

        # 从配置中获取标签数量
        self.num_labels = config.num_labels

        # 创建 Albert 主层对象，不添加池化层，并命名为 "albert"
        self.albert = TFAlbertMainLayer(config, add_pooling_layer=False, name="albert")

        # 根据配置中的分类器丢弃率或者隐藏层丢弃率，创建 Dropout 层
        classifier_dropout_prob = (
            config.classifier_dropout_prob
            if config.classifier_dropout_prob is not None
            else config.hidden_dropout_prob
        )
        self.dropout = keras.layers.Dropout(rate=classifier_dropout_prob)

        # 创建分类器 Dense 层，用于标签分类，初始化方式使用配置中的范围初始化
        self.classifier = keras.layers.Dense(
            units=config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )

        # 将配置对象保存到模型中
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
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
        **kwargs,
    ):
        """
        Performs forward pass of the model.
        """
        # 调用父类的 `call` 方法，执行模型的前向传播
        return super().call(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
            training=training,
            **kwargs,
        )
    ) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 调用 ALBERT 模型，获取模型的输出结果
        outputs = self.albert(
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
        # 从 ALBERT 模型的输出中获取序列输出
        sequence_output = outputs[0]
        # 对序列输出应用 dropout 操作，用于防止过拟合
        sequence_output = self.dropout(inputs=sequence_output, training=training)
        # 将 dropout 后的序列输出输入分类器，得到 logits（预测结果）
        logits = self.classifier(inputs=sequence_output)
        # 如果提供了标签，则计算损失值
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        # 如果不需要返回字典形式的输出
        if not return_dict:
            # 组装输出元组
            output = (logits,) + outputs[2:]
            # 如果有损失值，则将损失值作为输出的第一个元素
            return ((loss,) + output) if loss is not None else output

        # 返回 TFTokenClassifierOutput 对象，包含损失值、logits、隐藏状态和注意力权重
        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果模型已经构建完成，则直接返回
        if self.built:
            return
        # 将模型标记为已构建
        self.built = True
        # 如果存在 ALBERT 模型，则构建 ALBERT 模型
        if getattr(self, "albert", None) is not None:
            with tf.name_scope(self.albert.name):
                self.albert.build(None)
        # 如果存在分类器模型，则构建分类器模型
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                # 构建分类器模型，输入形状为 [None, None, self.config.hidden_size]
                self.classifier.build([None, None, self.config.hidden_size])
"""
Albert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
"""
# 使用 Albert 模型，添加一个用于抽取式问答任务（如 SQuAD）的跨度分类头部（在隐藏状态输出顶部的线性层，用于计算“跨度起始对数”和“跨度终止对数”）。

# 导入 ALBERT_START_DOCSTRING 作为注释的一部分
@add_start_docstrings(ALBERT_START_DOCSTRING)

# 定义 TFAlbertForQuestionAnswering 类，继承自 TFAlbertPreTrainedModel 和 TFQuestionAnsweringLoss
class TFAlbertForQuestionAnswering(TFAlbertPreTrainedModel, TFQuestionAnsweringLoss):

    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    # _keys_to_ignore_on_load_unexpected 是在从 PT 模型加载 TF 模型时允许的未预期/丢失的层的名称列表
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"predictions"]

    # 初始化方法，接收一个 AlbertConfig 类型的 config 对象和其他位置参数
    def __init__(self, config: AlbertConfig, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 设置类属性 num_labels 等于 config 中的 num_labels
        self.num_labels = config.num_labels

        # 初始化 Albert 主层对象，设置不添加池化层，并命名为 "albert"
        self.albert = TFAlbertMainLayer(config, add_pooling_layer=False, name="albert")

        # 初始化 QA 输出层，使用 Dense 层，单元数为 config 中的 num_labels，使用指定的初始化器范围初始化权重，命名为 "qa_outputs"
        self.qa_outputs = keras.layers.Dense(
            units=config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )

        # 将 config 对象保存为类属性
        self.config = config

    # 调用方法，用装饰器添加了多个文档字符串，说明了输入和输出的详细信息，以及模型的用法示例
    @unpack_inputs
    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="vumichien/albert-base-v2-squad2",
        output_type=TFQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        qa_target_start_index=12,
        qa_target_end_index=13,
        expected_output="'a nice puppet'",
        expected_loss=7.36,
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
        start_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        outputs = self.albert(
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
        # 获取 ALBERT 模型的输出，包括序列输出、注意力权重等

        sequence_output = outputs[0]
        # 从 ALBERT 输出中提取序列输出

        logits = self.qa_outputs(inputs=sequence_output)
        # 将序列输出传递给 QA 输出层，得到预测的开始和结束位置的 logits
        
        start_logits, end_logits = tf.split(value=logits, num_or_size_splits=2, axis=-1)
        # 将 logits 沿最后一个维度分割为开始和结束 logits
        
        start_logits = tf.squeeze(input=start_logits, axis=-1)
        end_logits = tf.squeeze(input=end_logits, axis=-1)
        # 移除 logits 的单维度，以匹配预期的形状

        loss = None

        if start_positions is not None and end_positions is not None:
            labels = {"start_position": start_positions}
            labels["end_position"] = end_positions
            # 构建标签字典，包含开始和结束位置的真实标签

            loss = self.hf_compute_loss(labels=labels, logits=(start_logits, end_logits))
            # 使用损失计算函数计算开始和结束位置的损失

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            # 如果不返回字典形式的结果，构建输出元组

            return ((loss,) + output) if loss is not None else output
            # 如果有损失，则在输出元组前添加损失；否则只返回输出元组

        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # 返回 TFQuestionAnsweringModelOutput 对象，包含损失、开始和结束 logits、隐藏状态和注意力权重

    def build(self, input_shape=None):
        if self.built:
            return
        # 如果模型已经建立过，则直接返回

        self.built = True
        # 将模型标记为已建立

        if getattr(self, "albert", None) is not None:
            with tf.name_scope(self.albert.name):
                self.albert.build(None)
        # 如果存在 ALBERT 模型，使用其名称作为作用域，构建 ALBERT 模型

        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.config.hidden_size])
        # 如果存在 QA 输出层，使用其名称作为作用域，构建 QA 输出层
"""
Albert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.
"""
@add_start_docstrings(
    """
    Albert 模型，顶部带有一个多选分类头部（在汇总输出的基础上添加一个线性层和 softmax），例如 RocStories/SWAG 任务。
    """,
    ALBERT_START_DOCSTRING,
)
class TFAlbertForMultipleChoice(TFAlbertPreTrainedModel, TFMultipleChoiceLoss):
    """
    names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    """
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"predictions"]
    # List of keys ignored when certain layers are missing during TF model loading from PT model
    _keys_to_ignore_on_load_missing = [r"dropout"]

    def __init__(self, config: AlbertConfig, *inputs, **kwargs):
        """
        Initialize TFAlbertForMultipleChoice model
        """
        super().__init__(config, *inputs, **kwargs)

        # Initialize Albert main layer with provided configuration
        self.albert = TFAlbertMainLayer(config, name="albert")
        # Dropout layer with dropout rate set from configuration
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        # Classifier dense layer initialized with specific initializer range from configuration
        self.classifier = keras.layers.Dense(
            units=1, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        # Store the configuration object for reference
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
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
        ):
        """
        Perform forward pass of TFAlbertForMultipleChoice model.
        """
    ) -> Union[TFMultipleChoiceModelOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
            where `num_choices` is the size of the second dimension of the input tensors. (See `input_ids` above)
        """

        # 如果提供了 input_ids，则获取 num_choices 和 seq_length
        if input_ids is not None:
            num_choices = shape_list(input_ids)[1]
            seq_length = shape_list(input_ids)[2]
        # 否则，从 inputs_embeds 获取 num_choices 和 seq_length
        else:
            num_choices = shape_list(inputs_embeds)[1]
            seq_length = shape_list(inputs_embeds)[2]

        # 将 input_ids 展平成二维张量，如果 input_ids 不为 None
        flat_input_ids = tf.reshape(input_ids, (-1, seq_length)) if input_ids is not None else None
        # 将 attention_mask 展平成二维张量，如果 attention_mask 不为 None
        flat_attention_mask = (
            tf.reshape(tensor=attention_mask, shape=(-1, seq_length)) if attention_mask is not None else None
        )
        # 将 token_type_ids 展平成二维张量，如果 token_type_ids 不为 None
        flat_token_type_ids = (
            tf.reshape(tensor=token_type_ids, shape=(-1, seq_length)) if token_type_ids is not None else None
        )
        # 将 position_ids 展平成二维张量，如果 position_ids 不为 None
        flat_position_ids = (
            tf.reshape(tensor=position_ids, shape=(-1, seq_length)) if position_ids is not None else None
        )
        # 将 inputs_embeds 展平成三维张量，如果 inputs_embeds 不为 None
        flat_inputs_embeds = (
            tf.reshape(tensor=inputs_embeds, shape=(-1, seq_length, shape_list(inputs_embeds)[3]))
            if inputs_embeds is not None
            else None
        )
        # 使用 ALBERT 模型进行推断
        outputs = self.albert(
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
        # 获取池化后的输出
        pooled_output = outputs[1]
        # 应用 dropout
        pooled_output = self.dropout(inputs=pooled_output, training=training)
        # 使用分类器进行预测
        logits = self.classifier(inputs=pooled_output)
        # 将 logits 重新形状为二维张量
        reshaped_logits = tf.reshape(tensor=logits, shape=(-1, num_choices))
        # 如果提供了 labels，则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=reshaped_logits)

        # 如果不需要返回字典格式的输出，则返回相应的结果元组
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典格式的输出，则构造 TFMultipleChoiceModelOutput 对象并返回
        return TFMultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    # 如果模型已经构建，直接返回，避免重复构建
    if self.built:
        return
    # 标记模型已经构建
    self.built = True
    
    # 如果存在属性 self.albert，则构建 self.albert 模型
    if getattr(self, "albert", None) is not None:
        # 使用 self.albert 的名称作为命名空间，并构建该模型
        with tf.name_scope(self.albert.name):
            self.albert.build(None)
    
    # 如果存在属性 self.classifier，则构建 self.classifier 模型
    if getattr(self, "classifier", None) is not None:
        # 使用 self.classifier 的名称作为命名空间，并构建该模型
        with tf.name_scope(self.classifier.name):
            # 构建 classifier 模型，传入输入形状 [None, None, self.config.hidden_size]
            self.classifier.build([None, None, self.config.hidden_size])
```