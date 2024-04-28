# `.\models\deberta\modeling_tf_deberta.py`

```py
# 设置脚本编码为 UTF-8
# 版权声明，2021年由 Microsoft 和 HuggingFace Inc. 团队保留所有权利
#
# 根据 Apache 许可证 2.0 版本（“许可证”）授权;
# 除非符合许可证规定，否则不得使用此文件。
# 您可以获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件将根据“按原样”
# 分发，没有任何形式的明示或暗示保修或条件。
# 有关具体语言的许可证以及限制，请查看许可证
""" TF 2.0 DeBERTa model."""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFMaskedLMOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from ...modeling_tf_utils import (
    TFMaskedLanguageModelingLoss,
    TFModelInputType,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFTokenClassificationLoss,
    get_initializer,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_deberta import DebertaConfig

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 用于文档的配置和检查点信息
_CONFIG_FOR_DOC = "DebertaConfig"
_CHECKPOINT_FOR_DOC = "kamalkraj/deberta-base"

# DeBERTa 预训练模型的归档列表
TF_DEBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "kamalkraj/deberta-base",
    # 查看所有 DeBERTa 模型，请访问 https://huggingface.co/models?filter=DeBERTa
]

# 定义 TFDebertaContextPooler 类
class TFDebertaContextPooler(tf.keras.layers.Layer):

    def __init__(self, config: DebertaConfig, **kwargs):
        super().__init__(**kwargs)
        # 全连接层，用于池化隐藏状态
        self.dense = tf.keras.layers.Dense(config.pooler_hidden_size, name="dense")
        # dropout 层
        self.dropout = TFDebertaStableDropout(config.pooler_dropout, name="dropout")
        self.config = config

    def call(self, hidden_states, training: bool = False):
        # 我们通过简单地获取与第一个令牌对应的隐藏状态来"池化"模型。
        context_token = hidden_states[:, 0]
        context_token = self.dropout(context_token, training=training)
        pooled_output = self.dense(context_token)
        # 使用激活函数激活池化输出
        pooled_output = get_tf_activation(self.config.pooler_hidden_act)(pooled_output)
        return pooled_output

    @property
    def output_dim(self) -> int:
        return self.config.hidden_size
```  
    # 构建模型
    def build(self, input_shape=None):
        # 检查模型是否已经构建，若已经构建则直接返回
        if self.built:
            return
        # 将模型设为已构建状态
        self.built = True
        
        # 检查是否存在 dense 属性，若存在则构建 dense 层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):  # 为 dense 层创建一个命名作用域
                self.dense.build([None, None, self.config.pooler_hidden_size])  # 构建 dense 层，指定输入形状
        
        # 检查是否存在 dropout 属性，若存在则构建 dropout 层
        if getattr(self, "dropout", None) is not None:
            with tf.name_scope(self.dropout.name):  # 为 dropout 层创建一个命名作用域
                self.dropout.build(None)  # 构建 dropout 层，不指定输入形状
class TFDebertaXSoftmax(tf.keras.layers.Layer):
    """
    Masked Softmax which is optimized for saving memory

    Args:
        input (`tf.Tensor`): The input tensor that will apply softmax.
        mask (`tf.Tensor`): The mask matrix where 0 indicate that element will be ignored in the softmax calculation.
        dim (int): The dimension that will apply softmax
    """

    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs: tf.Tensor, mask: tf.Tensor):
        # Invert the mask to represent elements to ignore
        rmask = tf.logical_not(tf.cast(mask, tf.bool))
        # Apply -inf to elements to ignore
        output = tf.where(rmask, float("-inf"), inputs)
        # Apply stable softmax
        output = stable_softmax(output, self.axis)
        # Reapply ignored elements as 0
        output = tf.where(rmask, 0.0, output)
        return output


class TFDebertaStableDropout(tf.keras.layers.Layer):
    """
    Optimized dropout module for stabilizing the training

    Args:
        drop_prob (float): the dropout probabilities
    """

    def __init__(self, drop_prob, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    @tf.custom_gradient
    def xdropout(self, inputs):
        """
        Applies dropout to the inputs, as vanilla dropout, but also scales the remaining elements up by 1/drop_prob.
        """
        # Generate dropout mask
        mask = tf.cast(
            1
            - tf.compat.v1.distributions.Bernoulli(probs=1.0 - self.drop_prob).sample(sample_shape=shape_list(inputs)),
            tf.bool,
        )
        # Calculate scaling factor
        scale = tf.convert_to_tensor(1.0 / (1 - self.drop_prob), dtype=tf.float32)
        if self.drop_prob > 0:
            # Apply dropout and scaling
            inputs = tf.where(mask, 0.0, inputs) * scale

        def grad(upstream):
            if self.drop_prob > 0:
                # Gradient calculation for dropout
                return tf.where(mask, 0.0, upstream) * scale
            else:
                return upstream

        return inputs, grad

    def call(self, inputs: tf.Tensor, training: tf.Tensor = False):
        # Apply dropout during training
        if training:
            return self.xdropout(inputs)
        return inputs


class TFDebertaLayerNorm(tf.keras.layers.Layer):
    """LayerNorm module in the TF style (epsilon inside the square root)."""

    def __init__(self, size, eps=1e-12, **kwargs):
        super().__init__(**kwargs)
        self.size = size
        self.eps = eps

    def build(self, input_shape):
        # Initialize weights for gamma and beta
        self.gamma = self.add_weight(shape=[self.size], initializer=tf.ones_initializer(), name="weight")
        self.beta = self.add_weight(shape=[self.size], initializer=tf.zeros_initializer(), name="bias")
        return super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # Compute mean, variance, and standard deviation
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
        std = tf.math.sqrt(variance + self.eps)
        # Normalize using gamma and beta
        return self.gamma * (x - mean) / std + self.beta


class TFDebertaSelfOutput(tf.keras.layers.Layer):
    # 初始化方法，接受DebertaConfig类型的config参数和任意数量的关键字参数
    def __init__(self, config: DebertaConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 创建一个全连接层，输出维度为config.hidden_size，命名为"dense"
        self.dense = tf.keras.layers.Dense(config.hidden_size, name="dense")
        # 创建一个LayerNormalization层，使用config.layer_norm_eps作为epsilon，命名为"LayerNorm"
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建一个TFDebertaStableDropout层，使用config.hidden_dropout_prob作为dropout概率，命名为"dropout"
        self.dropout = TFDebertaStableDropout(config.hidden_dropout_prob, name="dropout")
        # 记录config参数
        self.config = config

    # 前向传播方法，接受hidden_states、input_tensor和training参数
    def call(self, hidden_states, input_tensor, training: bool = False):
        # 将hidden_states输入到全连接层中进行计算
        hidden_states = self.dense(hidden_states)
        # 对计算结果进行dropout处理
        hidden_states = self.dropout(hidden_states, training=training)
        # 对生成的结果进行LayerNormalization处理并与input_tensor相加
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的结果
        return hidden_states

    # 模型构建方法，接受输入形状参数input_shape
    def build(self, input_shape=None):
        # 如果模型已经构建好了，直接返回
        if self.built:
            return
        # 将模型标记为已构建
        self.built = True
        # 如果存在"self.dense"属性
        if getattr(self, "dense", None) is not None:
            # 在tf的name_scope下，使用self.dense.name为名字，对全连接层进行构建，输入形状为[None, None, self.config.hidden_size]
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 如果存在"self.LayerNorm"属性
        if getattr(self, "LayerNorm", None) is not None:
            # 在tf的name_scope下，使用self.LayerNorm.name为名字，对LayerNormalization层进行构建，输入形状为[None, None, self.config.hidden_size]
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])
        # 如果存在"self.dropout"属性
        if getattr(self, "dropout", None) is not None:
            # 在tf的name_scope下，使用self.dropout.name为名字，对dropout层进行构建，输入形状为None
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)
```py python
# 创建一个名为TFDebertaAttention的类，该类继承自tf.keras.layers.Layer
class TFDebertaAttention(tf.keras.layers.Layer):
    # 定义初始化函数，其中config是DebertaConfig的实例
    def __init__(self, config: DebertaConfig, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)
        # 创建一个名为self的TFDebertaDisentangledSelfAttention对象，并将其赋值给self.self属性
        self.self = TFDebertaDisentangledSelfAttention(config, name="self")
        # 创建一个名为dense_output的TFDebertaSelfOutput对象，并将其赋值给self.dense_output属性
        self.dense_output = TFDebertaSelfOutput(config, name="output")
        # 将传入的config赋值给self.config属性
        self.config = config

    # 定义call方法，用于调用该层(layer)的操作
    def call(
        self,
        input_tensor: tf.Tensor,
        attention_mask: tf.Tensor,
        query_states: tf.Tensor = None,
        relative_pos: tf.Tensor = None,
        rel_embeddings: tf.Tensor = None,
        output_attentions: bool = False,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        # 调用self.self的call方法，传入相应参数，并将返回值赋值给self_outputs
        self_outputs = self.self(
            hidden_states=input_tensor,
            attention_mask=attention_mask,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
            output_attentions=output_attentions,
            training=training,
        )
        # 如果query_states为None，则将input_tensor赋值给query_states
        if query_states is None:
            query_states = input_tensor
        # 调用self.dense_output的call方法，传入相应参数，并将返回值赋值给attention_output
        attention_output = self.dense_output(
            hidden_states=self_outputs[0], input_tensor=query_states, training=training
        )

        # 创建一个元组output，包含attention_output和self_outputs[1:]的元素
        output = (attention_output,) + self_outputs[1:]

        # 返回output
        return output

    # 定义build方法，用于构建该层(layer)
    def build(self, input_shape=None):
        # 如果该层已经构建，则直接返回
        if self.built:
            return
        self.built = True
        # 如果self存在，则构建self
        if getattr(self, "self", None) is not None:
            with tf.name_scope(self.self.name):
                self.self.build(None)
        # 如果dense_output存在，则构建dense_output
        if getattr(self, "dense_output", None) is not None:
            with tf.name_scope(self.dense_output.name):
                self.dense_output.build(None)
                
                
# 创建��个名为TFDebertaIntermediate的类，该类继承自tf.keras.layers.Layer
class TFDebertaIntermediate(tf.keras.layers.Layer):
    # 定义初始化函数，其中config是DebertaConfig的实例
    def __init__(self, config: DebertaConfig, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 创建一个全连接层，units为config.intermediate_size，kernel_initializer为指定的初始化器，名称为dense，并将其赋值给self.dense属性
        self.dense = tf.keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        # 判断config.hidden_act的类型，如果是str类型，使用get_tf_activation函数获取激活函数，否则直接使用config.hidden_act作为self.intermediate_act_fn属性
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        # 将传入的config赋值给self.config属性
        self.config = config

    # 定义call方法，用于调用该层(layer)的操作
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 使用全连接层self.dense处理hidden_states，并将处理后的结果赋值给hidden_states
        hidden_states = self.dense(inputs=hidden_states)
        # 使用self
    def __init__(self, config: DebertaConfig, **kwargs):
        # 继承父类的初始化方法
        super().__init__(**kwargs)

        # 创建一个全连接层，输出大小为config.hidden_size，初始化方式为config.initializer_range
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建一个LayerNorm层，epsilon为config.layer_norm_eps
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建一个dropout层，丢弃概率为config.hidden_dropout_prob
        self.dropout = TFDebertaStableDropout(config.hidden_dropout_prob, name="dropout")
        # 保存config
        self.config = config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 使用全连接层处理hidden_states
        hidden_states = self.dense(inputs=hidden_states)
        # 使用dropout层处理hidden_states
        hidden_states = self.dropout(hidden_states, training=training)
        # 使用LayerNorm层处理hidden_states与input_tensor的和
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

    def build(self, input_shape=None):
        # 如果已经build过了，直接返回
        if self.built:
            return
        # 设置self.built为True，表示已经build过了
        self.built = True
        # 如果self.dense不为空
        if getattr(self, "dense", None) is not None:
            # 在名字域中对self.dense内的操作进行命名，并build
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])
        # 如果self.LayerNorm不为空
        if getattr(self, "LayerNorm", None) is not None:
            # 在名字域中对self.LayerNorm内的操作进行命名，并build
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])
        # 如果self.dropout不为空
        if getattr(self, "dropout", None) is not None:
            # 在名字域中对self.dropout内的操作进行命名，并build
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)
# TFDebertaLayer 是一个 Keras 层类，实现了 DeBERTa 模型的一个隐藏层
class TFDebertaLayer(tf.keras.layers.Layer):
    def __init__(self, config: DebertaConfig, **kwargs):
        # 调用父类的构造函数
        super().__init__(**kwargs)

        # 初始化 DeBERTa 注意力层
        self.attention = TFDebertaAttention(config, name="attention")
        # 初始化 DeBERTa 中间层
        self.intermediate = TFDebertaIntermediate(config, name="intermediate")
        # 初始化 DeBERTa 输出层
        self.bert_output = TFDebertaOutput(config, name="output")

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        query_states: tf.Tensor = None,
        relative_pos: tf.Tensor = None,
        rel_embeddings: tf.Tensor = None,
        output_attentions: bool = False,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        # 计算注意力输出
        attention_outputs = self.attention(
            input_tensor=hidden_states,
            attention_mask=attention_mask,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
            output_attentions=output_attentions,
            training=training,
        )
        attention_output = attention_outputs[0]
        # 计算中间层输出
        intermediate_output = self.intermediate(hidden_states=attention_output)
        # 计算输出层输出
        layer_output = self.bert_output(
            hidden_states=intermediate_output, input_tensor=attention_output, training=training
        )
        # 将注意力输出也包含在最终输出中
        outputs = (layer_output,) + attention_outputs[1:]

        return outputs

    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回
        if self.built:
            return
        self.built = True
        # 构建注意力层
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        # 构建中间层
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        # 构建输出层
        if getattr(self, "bert_output", None) is not None:
            with tf.name_scope(self.bert_output.name):
                self.bert_output.build(None)


# TFDebertaEncoder 是一个 Keras 层类，实现了 DeBERTa 模型的编码器
class TFDebertaEncoder(tf.keras.layers.Layer):
    def __init__(self, config: DebertaConfig, **kwargs):
        # 调用父类的构造函数
        super().__init__(**kwargs)

        # 创建多个 TFDebertaLayer 层，组成 DeBERTa 的编码器
        self.layer = [TFDebertaLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]
        self.relative_attention = getattr(config, "relative_attention", False)
        self.config = config
        # 如果使用相对位置编码，则需要计算最大相对位置
        if self.relative_attention:
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
    # 构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建完毕，则直接返回
        if self.built:
            return
        self.built = True
        # 如果 relative_attention 为 True，则添加相对位置编码的权重
        if self.relative_attention:
            self.rel_embeddings = self.add_weight(
                name="rel_embeddings.weight",
                shape=[self.max_relative_positions * 2, self.config.hidden_size],
                initializer=get_initializer(self.config.initializer_range),
            )
        # 如果模型的层已经存在，则遍历每个层，对每个层建立模型
        if getattr(self, "layer", None) is not None:
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    layer.build(None)

    # 获取相对位置编码的权重
    def get_rel_embedding(self):
        # 如果 relative_attention 为 True，则返回相对位置编码的权重
        # 否则返回 None
        rel_embeddings = self.rel_embeddings if self.relative_attention else None
        return rel_embeddings

    # 获取注意力的遮罩
    def get_attention_mask(self, attention_mask):
        # 如果注意力遮罩的尺寸小于等于2，则添加两个尺寸为1的维度扩展
        if len(shape_list(attention_mask)) <= 2:
            extended_attention_mask = tf.expand_dims(tf.expand_dims(attention_mask, 1), 2)
            # 将扩展后的注意力遮罩与其维度为1的压缩后的注意力遮罩相乘
            attention_mask = extended_attention_mask * tf.expand_dims(tf.squeeze(extended_attention_mask, -2), -1)
            # 将注意力遮罩的数据类型转换为 tf.uint8
            attention_mask = tf.cast(attention_mask, tf.uint8)
        # 如果注意力遮罩的尺寸等于3，则添加一个尺寸为1的维度扩展
        elif len(shape_list(attention_mask)) == 3:
            attention_mask = tf.expand_dims(attention_mask, 1)

        return attention_mask

    # 获取相对位置编码
    def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
        # 如果 relative_attention 为 True，且相对位置编码为 None，则通过 query_states 和 hidden_states 的维度构建相对位置编码
        if self.relative_attention and relative_pos is None:
            q = shape_list(query_states)[-2] if query_states is not None else shape_list(hidden_states)[-2]
            relative_pos = build_relative_position(q, shape_list(hidden_states)[-2])
        return relative_pos

    # 前向传播
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        query_states: tf.Tensor = None,
        relative_pos: tf.Tensor = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        training: bool = False,
        ```

    注释：
        ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        # 如果不需要输出隐藏层状态，则初始化空元组
        all_hidden_states = () if output_hidden_states else None
        # 如果不需要输出注意力权重，则初始化空元组
        all_attentions = () if output_attentions else None

        # 获取注意力遮罩
        attention_mask = self.get_attention_mask(attention_mask)
        # 根据隐藏状态和查询状态获取相对位置
        relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)

        # 如果隐藏状态是序列类型，则取第一个隐藏状态作为下一个键值对（key-value pair）
        if isinstance(hidden_states, Sequence):
            next_kv = hidden_states[0]
        else:
            next_kv = hidden_states

        # 获取相对位置嵌入
        rel_embeddings = self.get_rel_embedding()

        # 遍历每个层模块
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏层状态，则将当前隐藏状态添加到all_hidden_states中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 对当前层进行前向传播
            layer_outputs = layer_module(
                hidden_states=next_kv,
                attention_mask=attention_mask,
                query_states=query_states,
                relative_pos=relative_pos,
                rel_embeddings=rel_embeddings,
                output_attentions=output_attentions,
                training=training,
            )
            # 更新隐藏状态为当前层的输出
            hidden_states = layer_outputs[0]

            # 如果有查询状态，则更新查询状态为当前隐藏状态
            if query_states is not None:
                query_states = hidden_states
                # 如果隐藏状态是序列类型，则更新下一个键值对（key-value pair）为下一个隐藏状态
                if isinstance(hidden_states, Sequence):
                    next_kv = hidden_states[i + 1] if i + 1 < len(self.layer) else None
            else:
                next_kv = hidden_states

            # 如果需要输出注意力权重，则将当前层的注意力权重添加到all_attentions中
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # 添加最后一层的隐藏状态到all_hidden_states中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要返回字典形式的结果，则将非空部分组成元组返回
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

        # 返回TFBaseModelOutput类对象
        return TFBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )
def build_relative_position(query_size, key_size):
    """
    根据查询和键构建相对位置关系

    我们假设查询的绝对位置 \\(P_q\\) 范围在 (0, query_size)，键的绝对位置 \\(P_k\\) 范围在 (0, key_size)，
    查询到键的相对位置为 \\(R_{q \\rightarrow k} = P_q - P_k\\)

    Args:
        query_size (int): 查询的长度
        key_size (int): 键的长度

    Return:
        `tf.Tensor`: 一个形状为 [1, query_size, key_size] 的张量

    """
    q_ids = tf.range(query_size, dtype=tf.int32)
    k_ids = tf.range(key_size, dtype=tf.int32)
    rel_pos_ids = q_ids[:, None] - tf.tile(tf.reshape(k_ids, [1, -1]), [query_size, 1])
    rel_pos_ids = rel_pos_ids[:query_size, :]
    rel_pos_ids = tf.expand_dims(rel_pos_ids, axis=0)
    return tf.cast(rel_pos_ids, tf.int64)


def c2p_dynamic_expand(c2p_pos, query_layer, relative_pos):
    shapes = [
        shape_list(query_layer)[0],
        shape_list(query_layer)[1],
        shape_list(query_layer)[2],
        shape_list(relative_pos)[-1],
    ]
    return tf.broadcast_to(c2p_pos, shapes)


def p2c_dynamic_expand(c2p_pos, query_layer, key_layer):
    shapes = [
        shape_list(query_layer)[0],
        shape_list(query_layer)[1],
        shape_list(key_layer)[-2],
        shape_list(key_layer)[-2],
    ]
    return tf.broadcast_to(c2p_pos, shapes)


def pos_dynamic_expand(pos_index, p2c_att, key_layer):
    shapes = shape_list(p2c_att)[:2] + [shape_list(pos_index)[-2], shape_list(key_layer)[-2]]
    return tf.broadcast_to(pos_index, shapes)


def torch_gather(x, indices, gather_axis):
    if gather_axis < 0:
        gather_axis = tf.rank(x) + gather_axis

    if gather_axis != tf.rank(x) - 1:
        pre_roll = tf.rank(x) - 1 - gather_axis
        permutation = tf.roll(tf.range(tf.rank(x)), pre_roll, axis=0)
        x = tf.transpose(x, perm=permutation)
        indices = tf.transpose(indices, perm=permutation)
    else:
        pre_roll = 0

    flat_x = tf.reshape(x, (-1, tf.shape(x)[-1]))
    flat_indices = tf.reshape(indices, (-1, tf.shape(indices)[-1]))
    gathered = tf.gather(flat_x, flat_indices, batch_dims=1)
    gathered = tf.reshape(gathered, tf.shape(indices))

    if pre_roll != 0:
        permutation = tf.roll(tf.range(tf.rank(x)), -pre_roll, axis=0)
        gathered = tf.transpose(gathered, perm=permutation)

    return gathered


class TFDebertaDisentangledSelfAttention(tf.keras.layers.Layer):
    """
    分解自注意力模块

    Parameters:
        config (`str`):
            一个包含构建新模型的配置的模型配置类实例。模式类似于 *BertConfig*，详情请参阅[`DebertaConfig`]

    """
    # 初始化方法，接受一个DebertaConfig类型的config参数和其他关键字参数
    def __init__(self, config: DebertaConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 检查隐藏层大小是否能被注意力头的数量整除
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        # 初始化注意力头的数量和注意力头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # 使用tf.keras.layers.Dense创建一个线性层，用于输入项目
        self.in_proj = tf.keras.layers.Dense(
            self.all_head_size * 3,
            kernel_initializer=get_initializer(config.initializer_range),
            name="in_proj",
            use_bias=False,
        )
        # 初始化位置注意力类型
        self.pos_att_type = config.pos_att_type if config.pos_att_type is not None else []

        # 初始化相对注意力机制和talking head机制
        self.relative_attention = getattr(config, "relative_attention", False)
        self.talking_head = getattr(config, "talking_head", False)

        if self.talking_head:
            # 如果使用talking head机制，创建头部logits投影和头部权重投影的线性层
            self.head_logits_proj = tf.keras.layers.Dense(
                self.num_attention_heads,
                kernel_initializer=get_initializer(config.initializer_range),
                name="head_logits_proj",
                use_bias=False,
            )
            self.head_weights_proj = tf.keras.layers.Dense(
                self.num_attention_heads,
                kernel_initializer=get_initializer(config.initializer_range),
                name="head_weights_proj",
                use_bias=False,
            )

        # 使用TFDebertaXSoftmax创建一个softmax层
        self.softmax = TFDebertaXSoftmax(axis=-1)

        if self.relative_attention:
            # 如果使用相对注意力机制，初始化最大相对位置以及位置dropout
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.pos_dropout = TFDebertaStableDropout(config.hidden_dropout_prob, name="pos_dropout")
            if "c2p" in self.pos_att_type:
                # 如果位置注意力类型中包含"c2p"，创建一个位置投影的线性层
                self.pos_proj = tf.keras.layers.Dense(
                    self.all_head_size,
                    kernel_initializer=get_initializer(config.initializer_range),
                    name="pos_proj",
                    use_bias=False,
                )
            if "p2c" in self.pos_att_type:
                # 如果位置注意力类型中包含"p2c"，创建一个位置Q投影的线性层
                self.pos_q_proj = tf.keras.layers.Dense(
                    self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="pos_q_proj"
                )

        # 使用TFDebertaStableDropout创建一个dropout层
        self.dropout = TFDebertaStableDropout(config.attention_probs_dropout_prob, name="dropout")
        # 将config参数保存在对象中
        self.config = config
    def build(self, input_shape=None):
        # 如果已经构建过就直接返回
        if self.built:
            return
        # 设置为已构建状态
        self.built = True
        # 添加权重参数，初始化为全零
        self.q_bias = self.add_weight(
            name="q_bias", shape=(self.all_head_size), initializer=tf.keras.initializers.Zeros()
        )
        # 添加权重参数，初始化为全零
        self.v_bias = self.add_weight(
            name="v_bias", shape=(self.all_head_size), initializer=tf.keras.initializers.Zeros()
        )
        # 如果存在 in_proj 属性，则构建 in_proj
        if getattr(self, "in_proj", None) is not None:
            with tf.name_scope(self.in_proj.name):
                self.in_proj.build([None, None, self.config.hidden_size])
        # 如果存在 dropout 属性，则构建 dropout
        if getattr(self, "dropout", None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)
        # 如果存在 head_logits_proj 属性，则构建 head_logits_proj
        if getattr(self, "head_logits_proj", None) is not None:
            with tf.name_scope(self.head_logits_proj.name):
                self.head_logits_proj.build(None)
        # 如果存在 head_weights_proj 属性，则构建 head_weights_proj
        if getattr(self, "head_weights_proj", None) is not None:
            with tf.name_scope(self.head_weights_proj.name):
                self.head_weights_proj.build(None)
        # 如果存在 pos_dropout 属性，则构建 pos_dropout
        if getattr(self, "pos_dropout", None) is not None:
            with tf.name_scope(self.pos_dropout.name):
                self.pos_dropout.build(None)
        # 如果存在 pos_proj 属性，则构建 pos_proj
        if getattr(self, "pos_proj", None) is not None:
            with tf.name_scope(self.pos_proj.name):
                self.pos_proj.build([self.config.hidden_size])
        # 如果存在 pos_q_proj 属性，则构建 pos_q_proj
        if getattr(self, "pos_q_proj", None) is not None:
            with tf.name_scope(self.pos_q_proj.name):
                self.pos_q_proj.build([self.config.hidden_size])

    def transpose_for_scores(self, tensor: tf.Tensor) -> tf.Tensor:
        shape = shape_list(tensor)[:-1] + [self.num_attention_heads, -1]
        # 重塑张量形状，转换维度
        tensor = tf.reshape(tensor=tensor, shape=shape)

        # 将张量转置
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        query_states: tf.Tensor = None,
        relative_pos: tf.Tensor = None,
        rel_embeddings: tf.Tensor = None,
        output_attentions: bool = False,
        training: bool = False,
    # 计算位置注意力偏置
    def disentangled_att_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
        # 如果没有相对位置信息，则根据查询层和键层的形状构建相对位置
        if relative_pos is None:
            q = shape_list(query_layer)[-2]
            relative_pos = build_relative_position(q, shape_list(key_layer)[-2])
        shape_list_pos = shape_list(relative_pos)
        # 扩展相对位置张量的维度
        if len(shape_list_pos) == 2:
            relative_pos = tf.expand_dims(tf.expand_dims(relative_pos, 0), 0)
        elif len(shape_list_pos) == 3:
            relative_pos = tf.expand_dims(relative_pos, 1)
        # 异常情况处理，相对位置维度必须为2或3或4
        elif len(shape_list_pos) != 4:
            raise ValueError(f"Relative position ids must be of dim 2 or 3 or 4. {len(shape_list_pos)}")
        
        # 计算注意力范围
        att_span = tf.cast(
            tf.minimum(
                tf.maximum(shape_list(query_layer)[-2], shape_list(key_layer)[-2]), self.max_relative_positions
            ),
            tf.int64,
        )
        # 扩展相对位置嵌入
        rel_embeddings = tf.expand_dims(
            rel_embeddings[self.max_relative_positions - att_span : self.max_relative_positions + att_span, :], 0
        )

        score = 0

        # 内容到位置的注意力
        if "c2p" in self.pos_att_type:
            pos_key_layer = self.pos_proj(rel_embeddings)
            pos_key_layer = self.transpose_for_scores(pos_key_layer)
            c2p_att = tf.matmul(query_layer, tf.transpose(pos_key_layer, [0, 1, 3, 2]))
            c2p_pos = tf.clip_by_value(relative_pos + att_span, 0, att_span * 2 - 1)
            c2p_att = torch_gather(c2p_att, c2p_dynamic_expand(c2p_pos, query_layer, relative_pos), -1)
            score += c2p_att

        # 位置到内容的注意力
        if "p2c" in self.pos_att_type:
            pos_query_layer = self.pos_q_proj(rel_embeddings)
            pos_query_layer = self.transpose_for_scores(pos_query_layer)
            pos_query_layer /= tf.math.sqrt(tf.cast(shape_list(pos_query_layer)[-1] * scale_factor, dtype=tf.float32))
            if shape_list(query_layer)[-2] != shape_list(key_layer)[-2]:
                r_pos = build_relative_position(shape_list(key_layer)[-2], shape_list(key_layer)[-2])
            else:
                r_pos = relative_pos
            p2c_pos = tf.clip_by_value(-r_pos + att_span, 0, att_span * 2 - 1)
            p2c_att = tf.matmul(key_layer, tf.transpose(pos_query_layer, [0, 1, 3, 2]))
            p2c_att = tf.transpose(
                torch_gather(p2c_att, p2c_dynamic_expand(p2c_pos, query_layer, key_layer), -1), [0, 1, 3, 2]
            )
            if shape_list(query_layer)[-2] != shape_list(key_layer)[-2]:
                pos_index = tf.expand_dims(relative_pos[:, :, :, 0], -1)
                p2c_att = torch_gather(p2c_att, pos_dynamic_expand(pos_index, p2c_att, key_layer), -2)
            score += p2c_att

        return score
class TFDebertaEmbeddings(tf.keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        # 初始化对象的配置信息和其他参数
        self.config = config
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings
        self.position_biased_input = getattr(config, "position_biased_input", True)
        self.initializer_range = config.initializer_range
        # 如果嵌入大小不等于隐藏大小，则创建嵌入投影层
        if self.embedding_size != config.hidden_size:
            self.embed_proj = tf.keras.layers.Dense(
                config.hidden_size,
                kernel_initializer=get_initializer(config.initializer_range),
                name="embed_proj",
                use_bias=False,
            )
        # 创建 LayerNormalization 层和稳定的 Dropout 层
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = TFDebertaStableDropout(config.hidden_dropout_prob, name="dropout")

    def build(self, input_shape=None):
        with tf.name_scope("word_embeddings"):
            # 创建词嵌入权重矩阵
            self.weight = self.add_weight(
                name="weight",
                shape=[self.config.vocab_size, self.embedding_size],
                initializer=get_initializer(self.initializer_range),
            )

        with tf.name_scope("token_type_embeddings"):
            # 如果存在 token_type_embeddings，则创建对应的嵌入权重矩阵
            if self.config.type_vocab_size > 0:
                self.token_type_embeddings = self.add_weight(
                    name="embeddings",
                    shape=[self.config.type_vocab_size, self.embedding_size],
                    initializer=get_initializer(self.initializer_range),
                )
            else:
                self.token_type_embeddings = None

        with tf.name_scope("position_embeddings"):
            # 如果使用位置偏置输入，则创建位置嵌入权重矩阵
            if self.position_biased_input:
                self.position_embeddings = self.add_weight(
                    name="embeddings",
                    shape=[self.max_position_embeddings, self.hidden_size],
                    initializer=get_initializer(self.initializer_range),
                )
            else:
                self.position_embeddings = None

        # 如果已经构建完成，则直接返回
        if self.built:
            return
        # 标记已经构建完成
        self.built = True
        # 如果存在 LayerNorm 层，则构建该层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])
        # 如果存在 dropout 层，则构建该层
        if getattr(self, "dropout", None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)
        # 如果存在嵌入投影层，则构建该层
        if getattr(self, "embed_proj", None) is not None:
            with tf.name_scope(self.embed_proj.name):
                self.embed_proj.build([None, None, self.embedding_size])
    # 定义一个call方法，用于将输入应用于嵌入操作
    def call(
        self,
        input_ids: tf.Tensor = None,  # 输入的ID张量，默认为None
        position_ids: tf.Tensor = None,  # 位置ID张量，默认为None
        token_type_ids: tf.Tensor = None,  # 标记类型ID张量，默认为None
        inputs_embeds: tf.Tensor = None,  # 输入的嵌入张量，默认为None
        mask: tf.Tensor = None,  # 掩码张量，默认为None
        training: bool = False,  # 训练模式，默认为False
    ) -> tf.Tensor:  # 返回值为张量类型的结果
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.  # 返回最终的嵌入张量
        """
        if input_ids is None and inputs_embeds is None:  # 如果input_ids和inputs_embeds都为None，抛出数值错误
            raise ValueError("Need to provide either `input_ids` or `input_embeds`.")

        if input_ids is not None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)  # 检查输入ID张量是否在词汇大小范围内
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)  # 用索引ID从权重中获取嵌入参数

        input_shape = shape_list(inputs_embeds)[:-1]  # 获取输入嵌入的形状

        if token_type_ids is None:  # 如果标记类型ID为空
            token_type_ids = tf.fill(dims=input_shape, value=0)  # 用0填充标记类型ID形状的张量

        if position_ids is None:  # 如果位置ID为空
            position_ids = tf.expand_dims(tf.range(start=0, limit=input_shape[-1]), axis=0)  # 用范围内的值扩展位置ID张量

        final_embeddings = inputs_embeds  # 最终嵌入张量等于输入嵌入张量
        if self.position_biased_input:  # 如果存在位置偏置输入
            position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)  # 从位置嵌入中获取位置ID张量的嵌入参数
            final_embeddings += position_embeds  # 最终嵌入张量加上位置嵌入
        if self.config.type_vocab_size > 0:  # 如果类型词汇大小大于0
            token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)  # 从标记类型嵌入中获取标记类型ID张量的嵌入参数
            final_embeddings += token_type_embeds  # 最终嵌入张量加上标记类型嵌入

        if self.embedding_size != self.hidden_size:  # 如果嵌入大小不等于隐藏大小
            final_embeddings = self.embed_proj(final_embeddings)  # 使用embed_proj对最终嵌入进行处理

        final_embeddings = self.LayerNorm(final_embeddings)  # 对最终嵌入进行LayerNorm处理

        if mask is not None:  # 如果掩码不为空
            if len(shape_list(mask)) != len(shape_list(final_embeddings)):  # 如果掩码的形状列表长度不等于最终嵌入的形状列表长度
                if len(shape_list(mask)) == 4:  # 如果掩码的形状列表长度为4
                    mask = tf.squeeze(tf.squeeze(mask, axis=1), axis=1)  # 对掩码进行两次挤压操作
                mask = tf.cast(tf.expand_dims(mask, axis=2), tf.float32)  # 将掩码转换为浮点类型，并在第2个维度上扩展

            final_embeddings = final_embeddings * mask  # 最终嵌入等于最终嵌入与掩码的乘积

        final_embeddings = self.dropout(final_embeddings, training=training)  # 使用dropout对最终嵌入进行处理，训练模式为training

        return final_embeddings  # 返回最终嵌入
class TFDebertaPredictionHeadTransform(tf.keras.layers.Layer):
    # 初始化函数，接受 DebertaConfig 配置对象和其他参数
    def __init__(self, config: DebertaConfig, **kwargs):
        super().__init__(**kwargs)

        # 获取嵌入大小，如果配置中没有指定则为隐藏大小
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)

        # 创建全连接层，单元数为嵌入大小，使用配置中的初始化方法，命名为 dense
        self.dense = tf.keras.layers.Dense(
            units=self.embedding_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense",
        )

        # 根据配置中的隐藏激活函数，获取相应的激活函数或者使用配置中的激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.transform_act_fn = config.hidden_act
        
        # 创建 LayerNormalization 层，使用配置中的 epsilon，命名为 LayerNorm
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.config = config

    # 前向传播函数，接收隐藏状态并返回处理后的隐藏状态
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 通过全连接层处理隐藏状态
        hidden_states = self.dense(inputs=hidden_states)
        # 通过激活函数处理处理后的隐藏状态
        hidden_states = self.transform_act_fn(hidden_states)
        # 通过 LayerNormalization 处理处理后的隐藏状态
        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states

    # 构建函数，用于构建层的变量
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 设置标记为已构建
        self.built = True
        # 如果已经创建了全连接层，则构建全连接层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 如果已经创建了 LayerNormalization 层，则构建 LayerNormalization 层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.embedding_size])


class TFDebertaLMPredictionHead(tf.keras.layers.Layer):
    # 初始化函数，接受 DebertaConfig 配置对象、输入嵌入层和其他参数
    def __init__(self, config: DebertaConfig, input_embeddings: tf.keras.layers.Layer, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        # 获取嵌入大小，如果配置中没有指定则为隐藏大小
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)

        # 创建 TFDebertaPredictionHeadTransform 对象，命名为 transform
        self.transform = TFDebertaPredictionHeadTransform(config, name="transform")

        # 输出权重与输入嵌入相同，但每个标记有一个仅输出的偏置
        self.input_embeddings = input_embeddings

    # 构建函数，用于构建层的变量
    def build(self, input_shape=None):
        # 添加一个形状为（词汇表大小）的可训练的偏置
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="bias")

        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 设置标记为已构建
        self.built = True
        # 如果已经创建了 TFDebertaPredictionHeadTransform 对象，则构建它
        if getattr(self, "transform", None) is not None:
            with tf.name_scope(self.transform.name):
                self.transform.build(None)

    # 获取输出嵌入层
    def get_output_embeddings(self) -> tf.keras.layers.Layer:
        return self.input_embeddings

    # 设置输出嵌入层
    def set_output_embeddings(self, value: tf.Variable):
        self.input_embeddings.weight = value
        self.input_embeddings.vocab_size = shape_list(value)[0]

    # 获取偏置
    def get_bias(self) -> Dict[str, tf.Variable]:
        return {"bias": self.bias}

    # 设置偏置
    def set_bias(self, value: tf.Variable):
        self.bias = value["bias"]
        self.config.vocab_size = shape_list(value["bias"])[0]
    # 定义一个函数，参数为隐藏状态张量，返回值为张量
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 通过transform函数处理隐藏状态
        hidden_states = self.transform(hidden_states=hidden_states)
        # 获取隐藏状态的序列长度
        seq_length = shape_list(hidden_states)[1]
        # 将隐藏状态重塑为二维张量
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.embedding_size])
        # 矩阵相乘，计算隐藏状态与输入嵌入权重的乘积
        hidden_states = tf.matmul(a=hidden_states, b=self.input_embeddings.weight, transpose_b=True)
        # 将结果重新重塑为三维张量
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])
        # 添加偏置
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)
        # 返回处理后的隐藏状态
        return hidden_states
# 定义 TFDebertaOnlyMLMHead 类，继承自 tf.keras.layers.Layer
class TFDebertaOnlyMLMHead(tf.keras.layers.Layer):
    # 初始化方法，接受 DebertaConfig 对象和 input_embeddings 参数
    def __init__(self, config: DebertaConfig, input_embeddings: tf.keras.layers.Layer, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 创建 TFDebertaLMPredictionHead 实例，命名为 predictions
        self.predictions = TFDebertaLMPredictionHead(config, input_embeddings, name="predictions")

    # 调用方法，接受 sequence_output 参数，返回 prediction_scores 结果
    def call(self, sequence_output: tf.Tensor) -> tf.Tensor:
        # 调用 predictions 实例的方法，传入 sequence_output 参数，获取预测得分
        prediction_scores = self.predictions(hidden_states=sequence_output)
        # 返回预测得分
        return prediction_scores

    # 构建方法，构建 predictions
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 predictions 实例，使用 tf.name_scope 构建 predictions
        if getattr(self, "predictions", None) is not None:
            with tf.name_scope(self.predictions.name):
                self.predictions.build(None)

# 定义 TFDebertaMainLayer 类，继承自 tf.keras.layers.Layer
class TFDebertaMainLayer(tf.keras.layers.Layer):
    # 设置 config_class 属性为 DebertaConfig
    config_class = DebertaConfig
    # 初始化方法，接受 config 参数
    def __init__(self, config: DebertaConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 设置 self.config 为传入的 config
        self.config = config
        # 创建 TFDebertaEmbeddings 实例，命名为 embeddings
        self.embeddings = TFDebertaEmbeddings(config, name="embeddings")
        # 创建 TFDebertaEncoder 实例，命名为 encoder
        self.encoder = TFDebertaEncoder(config, name="encoder")

    # 获取输入 embeddings
    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        return self.embeddings

    # 设置输入 embeddings
    def set_input_embeddings(self, value: tf.Variable):
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    # 精简模型头部
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    # 调用方法
    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        # 该函数定义了返回值的类型注解

        if input_ids is not None and inputs_embeds is not None:
            # 如果同时指定了input_ids和inputs_embeds，则抛出数值错误
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            # 如果指定了input_ids，则获取其形状
            input_shape = shape_list(input_ids)
        elif inputs_embeds is not None:
            # 如果指定了inputs_embeds，则获取其形状直到倒数第二个元素
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            # 如果既没有指定input_ids，也没有指定inputs_embeds，则抛出数值错误
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            # 如果未指定attention_mask，则创建一个形状和input_shape相同，值全为1的tensor
            attention_mask = tf.fill(dims=input_shape, value=1)

        if token_type_ids is None:
            # 如果未指定token_type_ids，则创建一个形状和input_shape相同，值全为0的tensor
            token_type_ids = tf.fill(dims=input_shape, value=0)

        # 使用embeddings方法生成embedding_output
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            mask=attention_mask,
            training=training,
        )

        # 使用encoder方法生成encoder_outputs
        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 提取encoder_outputs的第一个元素作为sequence_output
        sequence_output = encoder_outputs[0]

        if not return_dict:
            # 如果不是return_dict，则返回一个元组，包含sequence_output和encoder_outputs的其他元素
            return (sequence_output,) + encoder_outputs[1:]

        # 如果是return_dict，则返回一个TFBaseModelOutput对象
        return TFBaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            # 如果已经构建过，则直接返回
            return
        self.built = True
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                # 使用embeddings的name空间来构建embeddings
                self.embeddings.build(None)
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                # 使用encoder的name空间来构建encoder
                self.encoder.build(None)
class TFDebertaPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 使用DebertaConfig作为配置类
    config_class = DebertaConfig
    # base_model_prefix指定为"deberta"
    base_model_prefix = "deberta"


DEBERTA_START_DOCSTRING = r"""
    The DeBERTa model was proposed in [DeBERTa: Decoding-enhanced BERT with Disentangled
    Attention](https://arxiv.org/abs/2006.03654) by Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen. It's build
    on top of BERT/RoBERTa with two improvements, i.e. disentangled attention and enhanced mask decoder. With those two
    improvements, it out perform BERT/RoBERTa on a majority of tasks with 80GB pretraining data.

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
        config ([`DebertaConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

DEBERTA_INPUTS_DOCSTRING = r"""
    """
    # 输入 ID 是指输入序列中每个token在词汇表中的索引
    Args:
        input_ids (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` ``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `({0})`):
            Indices of input sequence tokens in the vocabulary.
    
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
    
            [What are input IDs?](../glossary#input-ids)
    # 注意力掩码用于避免对填充token进行注意力计算
        attention_mask (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
    
            [What are attention masks?](../glossary#attention-mask)
    # Token type ID用于区分句子A和句子B
        token_type_ids (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:
    
            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.
    
            [What are token type IDs?](../glossary#token-type-ids)
    # Position ID用于指示每个token在序列中的位置
        position_ids (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.
    
            [What are position IDs?](../glossary#position-ids)
    # 输入嵌入允许直接传入嵌入向量，而不是输入ID
        inputs_embeds (`np.ndarray` or `tf.Tensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert *input_ids* indices into associated vectors than the
            model's internal embedding lookup matrix.
    # 输出注意力权重
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
    # 输出所有隐藏层状态
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
    # 是否返回ModelOutput对象
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput``] instead of a plain tuple.
# 添加起始文档字符串，描述了该模型的基本信息
@add_start_docstrings(
    "The bare DeBERTa Model transformer outputting raw hidden-states without any specific head on top.",
    DEBERTA_START_DOCSTRING,
)
# 定义 TFDebertaModel 类，继承自 TFDebertaPreTrainedModel
class TFDebertaModel(TFDebertaPreTrainedModel):
    # 初始化方法，接受配置参数和其他参数
    def __init__(self, config: DebertaConfig, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 创建一个 TFDebertaMainLayer 对象，命名为 "deberta"
        self.deberta = TFDebertaMainLayer(config, name="deberta")

    # 使用装饰器 unpack_inputs 对 call 方法进行装饰，解包输入参数
    # 添加模型前向传播的起始文档字符串
    # 添加代码示例文档字符串
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        # 调用 self.deberta 的前向传播方法，得到输出
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 返回输出结果
        return outputs

    # 定义 build 方法，用于构建模型
    def build(self, input_shape=None):
        # 如果已经构建过模型，则直接返回
        if self.built:
            return
        # 设置模型为已构建状态
        self.built = True
        # 如果 self.deberta 存在，构建 self.deberta
        if getattr(self, "deberta", None) is not None:
            with tf.name_scope(self.deberta.name):
                self.deberta.build(None)


# 添加起始文档字符串，描述了该模型的基本信息
@add_start_docstrings("""DeBERTa Model with a `language modeling` head on top.""", DEBERTA_START_DOCSTRING)
# 定义 TFDebertaForMaskedLM 类，继承自 TFDebertaPreTrainedModel 和 TFMaskedLanguageModelingLoss
class TFDebertaForMaskedLM(TFDebertaPreTrainedModel, TFMaskedLanguageModelingLoss):
    # 初始化方法，接受配置参数和其他参数
    def __init__(self, config: DebertaConfig, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        if config.is_decoder:
            logger.warning(
                "If you want to use `TFDebertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # 创建一个 TFDebertaMainLayer 对象，命名为 "deberta"
        self.deberta = TFDebertaMainLayer(config, name="deberta")
        # 创建一个 TFDebertaOnlyMLMHead 对象，命名为 "cls"
        self.mlm = TFDebertaOnlyMLMHead(config, input_embeddings=self.deberta.embeddings, name="cls")

    # 获取语言模型头部
    def get_lm_head(self) -> tf.keras.layers.Layer:
        return self.mlm.predictions

    # 使用装饰器 unpack_inputs 对 call 方法进行装饰，解包输入参数
    # 添加模型前向传播的起始文档字符串
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,  # 添加代码示例的文档字符串，用于指定文档字符串的检查点
        output_type=TFMaskedLMOutput,  # 输出类型为 TFMaskedLMOutput 类型
        config_class=_CONFIG_FOR_DOC,  # 使用_CONFIG_FOR_DOC配置类
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,  # 输入参数 input_ids 的类型为 TFModelInputType 或 None 类型，默认为 None
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 输入参数 attention_mask 的类型为 np.ndarray 或 tf.Tensor 或 None，默认为 None
        token_type_ids: np.ndarray | tf.Tensor | None = None,  # 输入参数 token_type_ids 的类型为 np.ndarray 或 tf.Tensor 或 None，默认为 None
        position_ids: np.ndarray | tf.Tensor | None = None,  # 输入参数 position_ids 的类型为 np.ndarray 或 tf.Tensor 或 None，默认为 None
        inputs_embeds: np.ndarray | tf.Tensor | None = None,  # 输入参数 inputs_embeds 的类型为 np.ndarray 或 tf.Tensor 或 None，默认为 None
        output_attentions: Optional[bool] = None,  # 输入参数 output_attentions 的类型为 Optional[bool] 类型或 None，默认为 None
        output_hidden_states: Optional[bool] = None,  # 输入参数 output_hidden_states 的类型为 Optional[bool] 类型或 None，默认为 None
        return_dict: Optional[bool] = None,  # 输入参数 return_dict 的类型为 Optional[bool] 类型或 None，默认为 None
        labels: np.ndarray | tf.Tensor | None = None,  # 输入参数 labels 的类型为 np.ndarray 或 tf.Tensor 或 None，默认为 None
        training: Optional[bool] = False,  # 输入参数 training 的类型为 Optional[bool] 类型或 False，默认为 False
    ) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:  # 函数返回类型为 TFMaskedLMOutput 类型或 Tuple[tf.Tensor] 类型的元组
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        # 调用 self.deberta 和 self.mlm 进行相应的操作，并返回结果
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        sequence_output = outputs[0]  # 获取 outputs 的第一个元素
        prediction_scores = self.mlm(sequence_output=sequence_output, training=training)  # 使用 sequence_output 和 training 参数调用 self.mlm 函数
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=prediction_scores)  # 如果 labels 为 None，则 loss 为 None，否则调用 self.hf_compute_loss() 函数

        # 根据 return_dict 的值返回不同的结果
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]  # 如果 return_dict 为 False，则返回 (prediction_scores,) 和 outputs[2:] 组成的元组
            return ((loss,) + output) if loss is not None else output  # 如果 loss 不为 None，则返回 (loss, output) 组成的元组，否则只返回 output

        return TFMaskedLMOutput(  # 返回 TFMaskedLMOutput 类型的对象
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):  # 定义 build 方法，输入参数为 input_shape，默认为 None
        if self.built:  # 如果已经构建过则直接返回
            return
        self.built = True  # 设置标记为已构建
        if getattr(self, "deberta", None) is not None:  # 如果 self 中包含 deberta 属性
            with tf.name_scope(self.deberta.name):  # 使用 deberta 的名称创建命名空间
                self.deberta.build(None)  # 调用 deberta 的 build 方法
        if getattr(self, "mlm", None) is not None:  # 如果 self 中包含 mlm 属性
            with tf.name_scope(self.mlm.name):  # 使用 mlm 的名称创建命名空间
                self.mlm.build(None)  # 调用 mlm 的 build 方法
# 使用给定的起始文档字符串初始化 TFDebertaForSequenceClassification 类
@add_start_docstrings(
    """
    DeBERTa Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    DEBERTA_START_DOCSTRING,
)
class TFDebertaForSequenceClassification(TFDebertaPreTrainedModel, TFSequenceClassificationLoss):
    # 初始化 TFDebertaForSequenceClassification 类
    def __init__(self, config: DebertaConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 获取标签数
        self.num_labels = config.num_labels

        # 创建 DeBERTa 主层和上下文池化层
        self.deberta = TFDebertaMainLayer(config, name="deberta")
        self.pooler = TFDebertaContextPooler(config, name="pooler")

        # 获取分类器的随机失活率
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        # 创建稳定的随机失活层
        self.dropout = TFDebertaStableDropout(drop_out, name="cls_dropout")
        # 创建分类器层
        self.classifier = tf.keras.layers.Dense(
            units=config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="classifier",
        )
        # 获取池化层的输出维度
        self.output_dim = self.pooler.output_dim

    # 对前向传播函数进行解包并添加文档字符串
    @unpack_inputs
    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 调用函数，接受输入参数并执行前向传播
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
        # ...
        ) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 使用输入的参数调用deberta模型，获取输出结果
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 从输出结果中获取序列输出
        sequence_output = outputs[0]
        # 使用pooler对序列输出进行池化处理
        pooled_output = self.pooler(sequence_output, training=training)
        # 对池化后的结果进行dropout处理
        pooled_output = self.dropout(pooled_output, training=training)
        # 使用分类器对处理后的结果进行分类
        logits = self.classifier(pooled_output)
        # 如果有labels，则计算损失，否则损失为None
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        # 如果return_dict为False，则输出元组
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果return_dict为True，则输出TFSequenceClassifierOutput类型的对象
        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果已经构建过则直接返回
        if self.built:
            return
        self.built = True
        # 构建deberta模型
        if getattr(self, "deberta", None) is not None:
            with tf.name_scope(self.deberta.name):
                self.deberta.build(None)
        # 构建pooler
        if getattr(self, "pooler", None) is not None:
            with tf.name_scope(self.pooler.name):
                self.pooler.build(None)
        # 构建dropout
        if getattr(self, "dropout", None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)
        # 构建分类器
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.output_dim])
# 定义一个基于 DeBERTa 模型的标记分类器，用于识别命名实体等任务
@add_start_docstrings(
    """
    DeBERTa Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    DEBERTA_START_DOCSTRING,
)
class TFDebertaForTokenClassification(TFDebertaPreTrainedModel, TFTokenClassificationLoss):
    def __init__(self, config: DebertaConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 记录标签的数量
        self.num_labels = config.num_labels

        # DeBERTa 主体模型
        self.deberta = TFDebertaMainLayer(config, name="deberta")
        # Dropout 层，用于减少过拟合
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        # 全连接层，用于分类
        self.classifier = tf.keras.layers.Dense(
            units=config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        # 保存模型配置信息
        self.config = config

    # 前向传播函数
    @unpack_inputs
    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
        # 输入参数包括：input_ids、attention_mask、token_type_ids、position_ids、inputs_embeds、output_attentions、
        # output_hidden_states、return_dict、labels 和 training
    # 定义了一个函数，接受输入参数并返回 TFTokenClassifierOutput 或 Tuple[tf.Tensor] 类型的值
    def call(self, 
             input_ids: tf.Tensor, 
             attention_mask: tf.Tensor, 
             token_type_ids: tf.Tensor = None, 
             position_ids: tf.Tensor = None, 
             inputs_embeds: tf.Tensor = None, 
             output_attentions: bool = False, 
             output_hidden_states: bool = False, 
             return_dict: bool = True, 
             training: bool = False, 
             labels: Union[tf.Tensor, np.ndarray] = None) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 调用deberta模型进行计算
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 获取模型输出的序列
        sequence_output = outputs[0]
        # 采用dropout层对序列进行处理
        sequence_output = self.dropout(sequence_output, training=training)
        # 将经过处理的序列输入到分类器中进行分类
        logits = self.classifier(inputs=sequence_output)
        # 如果存在标签，计算损失
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        # 如果不需要返回字典，则返回输出结果和损失
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TFTokenClassifierOutput 类型的对象，包括损失、logits、hidden_states 和 attentions
        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 定义了一个build函数，用于构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回
        if self.built:
            return
        self.built = True
        # 如果deberta模型存在，则构建模型
        if getattr(self, "deberta", None) is not None:
            with tf.name_scope(self.deberta.name):
                self.deberta.build(None)
        # 如果classifier模型存在，则构建模型
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
@add_start_docstrings(
    """
    DeBERTa Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    DEBERTA_START_DOCSTRING,
)
class TFDebertaForQuestionAnswering(TFDebertaPreTrainedModel, TFQuestionAnsweringLoss):
    def __init__(self, config: DebertaConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 获取标签数量
        self.num_labels = config.num_labels

        # 创建Deberta主层
        self.deberta = TFDebertaMainLayer(config, name="deberta")

        # 创建QA输出层，做线性变换
        self.qa_outputs = tf.keras.layers.Dense(
            units=config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        start_positions: np.ndarray | tf.Tensor | None = None,
        end_positions: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,


注释：
    def call(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, inputs_embeds=None,
             output_attentions=None, output_hidden_states=None, return_dict=None, training=None, start_positions=None,
             end_positions=None) -> Union[TFQuestionAnsweringModelOutput, Tuple[tf.Tensor]]:
        r"""
        start_positions (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        # 调用模型的主要方法，传入各种参数，返回结果
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 从模型输出获取序列输出
        sequence_output = outputs[0]
        # 通过序列输出获取问题回答的 logits
        logits = self.qa_outputs(inputs=sequence_output)
        # 将 logits 拆分为开始位置的 logits 和结束位置的 logits
        start_logits, end_logits = tf.split(value=logits, num_or_size_splits=2, axis=-1)
        # 移除多余的维度
        start_logits = tf.squeeze(input=start_logits, axis=-1)
        end_logits = tf.squeeze(input=end_logits, axis=-1)
        loss = None

        # 如果给定了起始位置和结束位置则计算损失
        if start_positions is not None and end_positions is not None:
            labels = {"start_position": start_positions}
            labels["end_position"] = end_positions
            loss = self.hf_compute_loss(labels=labels, logits=(start_logits, end_logits))

        # 如果不需要返回字典则直接返回结果
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典则构建 TFQuestionAnsweringModelOutput 对象并返回
        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 deeberta 则构建 deeberta
        if getattr(self, "deberta", None) is not None:
            with tf.name_scope(self.deberta.name):
                self.deberta.build(None)
        # 如果存在 qa_outputs 则构建 qa_outputs
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.config.hidden_size])
```