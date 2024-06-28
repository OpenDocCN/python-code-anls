# `.\models\deberta_v2\modeling_tf_deberta_v2.py`

```py
"""
TF 2.0 DeBERTa-v2 model.

"""

# 导入所需的模块和库
from __future__ import annotations
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFMaskedLMOutput,
    TFMultipleChoiceModelOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from ...modeling_tf_utils import (
    TFMaskedLanguageModelingLoss,
    TFModelInputType,
    TFMultipleChoiceLoss,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFTokenClassificationLoss,
    get_initializer,
    keras,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_deberta_v2 import DebertaV2Config

# 获取日志记录器
logger = logging.get_logger(__name__)

# 模型配置文档信息
_CONFIG_FOR_DOC = "DebertaV2Config"
_CHECKPOINT_FOR_DOC = "kamalkraj/deberta-v2-xlarge"

# 预训练模型存档列表
TF_DEBERTA_V2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "kamalkraj/deberta-v2-xlarge",
    # See all DeBERTa models at https://huggingface.co/models?filter=deberta-v2
]

# 自定义的上下文池化层，继承自Keras层
# 从transformers.models.deberta.modeling_tf_deberta.TFDebertaContextPooler中复制并修改为TFDebertaV2ContextPooler
class TFDebertaV2ContextPooler(keras.layers.Layer):
    def __init__(self, config: DebertaV2Config, **kwargs):
        super().__init__(**kwargs)
        # 创建全连接层dense和稳定Dropout层dropout
        self.dense = keras.layers.Dense(config.pooler_hidden_size, name="dense")
        self.dropout = TFDebertaV2StableDropout(config.pooler_dropout, name="dropout")
        self.config = config

    def call(self, hidden_states, training: bool = False):
        # 通过取第一个token对应的隐藏状态来“池化”模型
        context_token = hidden_states[:, 0]
        context_token = self.dropout(context_token, training=training)
        pooled_output = self.dense(context_token)
        # 应用激活函数到池化的输出
        pooled_output = get_tf_activation(self.config.pooler_hidden_act)(pooled_output)
        return pooled_output

    @property
    def output_dim(self) -> int:
        return self.config.hidden_size
    # 定义 build 方法，用于构建模型
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回，避免重复构建
        if self.built:
            return
        # 设置标志位表明已经构建过
        self.built = True
        # 如果存在名为 dense 的属性
        if getattr(self, "dense", None) is not None:
            # 使用 tf.name_scope 为 dense 层设置命名空间
            with tf.name_scope(self.dense.name):
                # 调用 dense 层的 build 方法，设置输入形状为 [None, None, self.config.pooler_hidden_size]
                self.dense.build([None, None, self.config.pooler_hidden_size])
        # 如果存在名为 dropout 的属性
        if getattr(self, "dropout", None) is not None:
            # 使用 tf.name_scope 为 dropout 层设置命名空间
            with tf.name_scope(self.dropout.name):
                # 调用 dropout 层的 build 方法，不设置具体的输入形状
                self.dropout.build(None)
# 从 transformers.models.deberta.modeling_tf_deberta.TFDebertaXSoftmax 复制的 TFDebertaV2XSoftmax 类，用于 Deberta 到 DebertaV2 的转换
class TFDebertaV2XSoftmax(keras.layers.Layer):
    """
    优化内存的掩码 Softmax 层

    Args:
        input (`tf.Tensor`): 需要应用 softmax 的输入张量。
        mask (`tf.Tensor`): 掩码矩阵，其中 0 表示在 softmax 计算中忽略该元素。
        dim (int): 应用 softmax 的维度
    """

    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs: tf.Tensor, mask: tf.Tensor):
        # 创建反掩码，将 mask 转换为布尔型的反向
        rmask = tf.logical_not(tf.cast(mask, tf.bool))
        # 在需要忽略的位置设置为负无穷大
        output = tf.where(rmask, float("-inf"), inputs)
        # 应用稳定 softmax
        output = stable_softmax(output, self.axis)
        # 将需要忽略的位置设置为 0
        output = tf.where(rmask, 0.0, output)
        return output


# 从 transformers.models.deberta.modeling_tf_deberta.TFDebertaStableDropout 复制的 TFDebertaV2StableDropout 类，用于 Deberta 到 DebertaV2 的转换
class TFDebertaV2StableDropout(keras.layers.Layer):
    """
    优化训练稳定性的 Dropout 模块

    Args:
        drop_prob (float): dropout 概率
    """

    def __init__(self, drop_prob, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    @tf.custom_gradient
    def xdropout(self, inputs):
        """
        对输入应用 dropout，类似于普通的 dropout，但同时将剩余元素缩放为 1/drop_prob 倍。
        """
        # 创建 dropout 掩码，按照指定的概率丢弃
        mask = tf.cast(
            1
            - tf.compat.v1.distributions.Bernoulli(probs=1.0 - self.drop_prob).sample(sample_shape=shape_list(inputs)),
            tf.bool,
        )
        scale = tf.convert_to_tensor(1.0 / (1 - self.drop_prob), dtype=tf.float32)
        if self.drop_prob > 0:
            # 如果 dropout 概率大于 0，则应用 dropout 并缩放剩余元素
            inputs = tf.where(mask, 0.0, inputs) * scale

        def grad(upstream):
            if self.drop_prob > 0:
                return tf.where(mask, 0.0, upstream) * scale
            else:
                return upstream

        return inputs, grad

    def call(self, inputs: tf.Tensor, training: tf.Tensor = False):
        if training:
            return self.xdropout(inputs)
        return inputs


# 从 transformers.models.deberta.modeling_tf_deberta.TFDebertaSelfOutput 复制的 TFDebertaV2SelfOutput 类，用于 Deberta 到 DebertaV2 的转换
class TFDebertaV2SelfOutput(keras.layers.Layer):
    def __init__(self, config: DebertaV2Config, **kwargs):
        super().__init__(**kwargs)
        # 创建全连接层，隐藏层大小为 config.hidden_size
        self.dense = keras.layers.Dense(config.hidden_size, name="dense")
        # 创建 LayerNormalization 层，使用 config.layer_norm_eps 作为 epsilon
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建 TFDebertaV2StableDropout 层，使用 config.hidden_dropout_prob 作为 dropout 概率
        self.dropout = TFDebertaV2StableDropout(config.hidden_dropout_prob, name="dropout")
        self.config = config
    # 对输入的隐藏状态进行全连接层操作，映射到新的表示空间
    hidden_states = self.dense(hidden_states)
    # 根据训练模式进行 dropout 操作，以防止过拟合
    hidden_states = self.dropout(hidden_states, training=training)
    # 将经过全连接层和 dropout 后的隐藏状态与输入张量相加，再进行 Layer Normalization
    hidden_states = self.LayerNorm(hidden_states + input_tensor)
    # 返回经过全连接层、dropout 和 Layer Normalization 处理后的隐藏状态
    return hidden_states

    # 构建模型的方法，用于在第一次调用时创建层次结构
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果存在 dense 层，则根据输入形状构建 dense 层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 如果存在 LayerNorm 层，则根据输入形状构建 LayerNorm 层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])
        # 如果存在 dropout 层，则构建 dropout 层
        if getattr(self, "dropout", None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)
# Copied from transformers.models.deberta.modeling_tf_deberta.TFDebertaAttention with Deberta->DebertaV2
class TFDebertaV2Attention(keras.layers.Layer):
    def __init__(self, config: DebertaV2Config, **kwargs):
        super().__init__(**kwargs)
        # 初始化自注意力层，使用DebertaV2DisentangledSelfAttention定义的层，并命名为"self"
        self.self = TFDebertaV2DisentangledSelfAttention(config, name="self")
        # 初始化自注意力层输出层，使用TFDebertaV2SelfOutput定义的层，并命名为"output"
        self.dense_output = TFDebertaV2SelfOutput(config, name="output")
        self.config = config

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
        # 调用自注意力层，传递输入张量及其他参数，获取自注意力层的输出
        self_outputs = self.self(
            hidden_states=input_tensor,
            attention_mask=attention_mask,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
            output_attentions=output_attentions,
            training=training,
        )
        if query_states is None:
            query_states = input_tensor
        # 将自注意力层的输出作为输入，传递给自注意力层输出层，获取注意力输出
        attention_output = self.dense_output(
            hidden_states=self_outputs[0], input_tensor=query_states, training=training
        )

        # 组装并返回输出元组，包含注意力输出和可能的额外输出
        output = (attention_output,) + self_outputs[1:]

        return output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已构建，则直接返回；否则按名称作用域构建自注意力层和输出层
        if getattr(self, "self", None) is not None:
            with tf.name_scope(self.self.name):
                self.self.build(None)
        if getattr(self, "dense_output", None) is not None:
            with tf.name_scope(self.dense_output.name):
                self.dense_output.build(None)


# Copied from transformers.models.deberta.modeling_tf_deberta.TFDebertaIntermediate with Deberta->DebertaV2
class TFDebertaV2Intermediate(keras.layers.Layer):
    def __init__(self, config: DebertaV2Config, **kwargs):
        super().__init__(**kwargs)

        # 初始化全连接层，使用给定的中间大小和初始化器，并命名为"dense"
        self.dense = keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        # 根据配置中的激活函数类型或函数本身，设置中间激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 对输入的隐藏状态应用全连接层
        hidden_states = self.dense(inputs=hidden_states)
        # 对全连接层输出应用中间激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已构建，则直接返回；否则按名称作用域构建全连接层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
# 从 transformers.models.deberta.modeling_tf_deberta.TFDebertaOutput 复制而来，将 Deberta 修改为 DebertaV2
class TFDebertaV2Output(keras.layers.Layer):
    def __init__(self, config: DebertaV2Config, **kwargs):
        super().__init__(**kwargs)

        # 定义一个全连接层，用于变换隐藏状态的维度
        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 定义 LayerNormalization 层，用于规范化隐藏状态
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 定义一个稳定的 Dropout 层，用于在训练时随机丢弃部分隐藏状态
        self.dropout = TFDebertaV2StableDropout(config.hidden_dropout_prob, name="dropout")
        # 保存配置信息
        self.config = config

    # 定义层的前向传播逻辑
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 全连接变换隐藏状态的维度
        hidden_states = self.dense(inputs=hidden_states)
        # 使用 Dropout 随机丢弃部分隐藏状态
        hidden_states = self.dropout(hidden_states, training=training)
        # 对变换后的隐藏状态进行 LayerNormalization，并加上输入张量
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

    # 构建层，确保所有子层被正确构建
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果定义了全连接层 dense，则构建该层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])
        # 如果定义了 LayerNormalization 层 LayerNorm，则构建该层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])
        # 如果定义了 Dropout 层 dropout，则构建该层
        if getattr(self, "dropout", None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)


# 从 transformers.models.deberta.modeling_tf_deberta.TFDebertaLayer 复制而来，将 Deberta 修改为 DebertaV2
class TFDebertaV2Layer(keras.layers.Layer):
    def __init__(self, config: DebertaV2Config, **kwargs):
        super().__init__(**kwargs)

        # 定义自注意力层
        self.attention = TFDebertaV2Attention(config, name="attention")
        # 定义中间层
        self.intermediate = TFDebertaV2Intermediate(config, name="intermediate")
        # 定义输出层
        self.bert_output = TFDebertaV2Output(config, name="output")

    # 定义层的前向传播逻辑
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
        # 调用 self.attention 方法，执行注意力计算，返回注意力输出元组
        attention_outputs = self.attention(
            input_tensor=hidden_states,  # 使用 hidden_states 作为输入张量
            attention_mask=attention_mask,  # 注意力掩码
            query_states=query_states,  # 查询状态
            relative_pos=relative_pos,  # 相对位置
            rel_embeddings=rel_embeddings,  # 相关嵌入
            output_attentions=output_attentions,  # 是否输出注意力信息
            training=training,  # 训练模式标志
        )
        attention_output = attention_outputs[0]  # 获取注意力输出张量
        intermediate_output = self.intermediate(hidden_states=attention_output)
        # 将注意力输出张量输入到 self.intermediate 方法中进行中间层处理
        layer_output = self.bert_output(
            hidden_states=intermediate_output,  # 使用中间输出作为隐藏状态输入
            input_tensor=attention_output,  # 注意力输出也作为输入之一
            training=training  # 训练模式标志传递给 bert_output 方法
        )
        outputs = (layer_output,) + attention_outputs[1:]  # 构建输出元组，包括层输出和可能的注意力信息

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)  # 构建 self.attention 层
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)  # 构建 self.intermediate 层
        if getattr(self, "bert_output", None) is not None:
            with tf.name_scope(self.bert_output.name):
                self.bert_output.build(None)  # 构建 self.bert_output 层
# 定义 TFDebertaV2ConvLayer 类，继承自 keras.layers.Layer
class TFDebertaV2ConvLayer(keras.layers.Layer):
    # 初始化方法，接受 DebertaV2Config 对象和其他关键字参数
    def __init__(self, config: DebertaV2Config, **kwargs):
        super().__init__(**kwargs)

        # 设置卷积核大小为 config.conv_kernel_size，默认为 3
        self.kernel_size = getattr(config, "conv_kernel_size", 3)
        # 获取激活函数并转换为 TensorFlow 激活函数对象
        self.conv_act = get_tf_activation(getattr(config, "conv_act", "tanh"))
        # 根据卷积核大小计算填充数
        self.padding = (self.kernel_size - 1) // 2
        # 创建 LayerNormalization 层，使用给定的 epsilon
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建 TFDebertaV2StableDropout 实例，使用隐藏层 dropout 概率
        self.dropout = TFDebertaV2StableDropout(config.hidden_dropout_prob, name="dropout")
        # 存储配置对象
        self.config = config

    # 构建层的方法，用于定义层的权重
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 在 "conv" 命名空间下创建卷积核权重
        with tf.name_scope("conv"):
            self.conv_kernel = self.add_weight(
                name="kernel",
                shape=[self.kernel_size, self.config.hidden_size, self.config.hidden_size],
                initializer=get_initializer(self.config.initializer_range),
            )
            # 创建卷积层的偏置项
            self.conv_bias = self.add_weight(
                name="bias", shape=[self.config.hidden_size], initializer=tf.zeros_initializer()
            )
        # 如果存在 LayerNorm 层，则构建其权重
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])
        # 如果存在 dropout 层，则构建其权重
        if getattr(self, "dropout", None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)

    # 定义调用方法，用于执行层的前向传播逻辑
    def call(
        self, hidden_states: tf.Tensor, residual_states: tf.Tensor, input_mask: tf.Tensor, training: bool = False
    ) -> tf.Tensor:
        # 执行二维卷积操作，输入是 hidden_states 的扩展维度和卷积核的扩展维度
        out = tf.nn.conv2d(
            tf.expand_dims(hidden_states, 1),
            tf.expand_dims(self.conv_kernel, 0),
            strides=1,
            padding=[[0, 0], [0, 0], [self.padding, self.padding], [0, 0]],
        )
        # 添加卷积偏置项并去除添加的维度
        out = tf.squeeze(tf.nn.bias_add(out, self.conv_bias), 1)
        # 计算输入 mask 的逆 mask，并将 out 中不需要的部分置为 0
        rmask = tf.cast(1 - input_mask, tf.bool)
        out = tf.where(tf.broadcast_to(tf.expand_dims(rmask, -1), shape_list(out)), 0.0, out)
        # 对 out 应用 dropout
        out = self.dropout(out, training=training)
        # 对 out 应用激活函数 conv_act
        out = self.conv_act(out)

        # 计算 Layer Normalization 的输入
        layer_norm_input = residual_states + out
        # 对 layer_norm_input 应用 LayerNormalization
        output = self.LayerNorm(layer_norm_input)

        # 如果 input_mask 为 None，则直接使用 output 作为输出
        if input_mask is None:
            output_states = output
        else:
            # 如果 input_mask 和 layer_norm_input 的维度不匹配，则进行相应的维度调整
            if len(shape_list(input_mask)) != len(shape_list(layer_norm_input)):
                if len(shape_list(input_mask)) == 4:
                    input_mask = tf.squeeze(tf.squeeze(input_mask, axis=1), axis=1)
                input_mask = tf.cast(tf.expand_dims(input_mask, axis=2), tf.float32)

            # 使用 input_mask 对 output 进行加权处理
            output_states = output * input_mask

        # 返回最终的输出状态
        return output_states
    # 初始化函数，接受一个 DebertaV2Config 类型的配置对象和其他关键字参数
    def __init__(self, config: DebertaV2Config, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 创建 self.layer 列表，包含 config.num_hidden_layers 个 TFDebertaV2Layer 对象
        self.layer = [TFDebertaV2Layer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]
        
        # 检查是否启用相对注意力机制
        self.relative_attention = getattr(config, "relative_attention", False)
        self.config = config

        # 如果启用了相对注意力机制
        if self.relative_attention:
            # 获取最大相对位置偏移量，默认为 config.max_position_embeddings
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings

            # 获取位置桶数，默认为 -1
            self.position_buckets = getattr(config, "position_buckets", -1)
            self.pos_ebd_size = self.max_relative_positions * 2

            # 如果设置了位置桶数，则重新计算位置嵌入大小
            if self.position_buckets > 0:
                self.pos_ebd_size = self.position_buckets * 2

        # 从配置中获取并解析 norm_rel_ebd 属性，以列表形式存储在 self.norm_rel_ebd 中
        self.norm_rel_ebd = [x.strip() for x in getattr(config, "norm_rel_ebd", "none").lower().split("|")]

        # 如果 norm_rel_ebd 中包含 'layer_norm'，则创建 LayerNormalization 层对象
        if "layer_norm" in self.norm_rel_ebd:
            self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")

        # 如果配置中的 conv_kernel_size 大于 0，则创建 TFDebertaV2ConvLayer 对象
        self.conv = TFDebertaV2ConvLayer(config, name="conv") if getattr(config, "conv_kernel_size", 0) > 0 else None

    # 构建函数，用于构建模型的层次结构
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True

        # 如果启用了相对注意力机制，创建相对位置嵌入权重 rel_embeddings
        if self.relative_attention:
            self.rel_embeddings = self.add_weight(
                name="rel_embeddings.weight",
                shape=[self.pos_ebd_size, self.config.hidden_size],
                initializer=get_initializer(self.config.initializer_range),
            )

        # 如果存在卷积层对象 self.conv，则调用其 build 方法构建卷积层
        if getattr(self, "conv", None) is not None:
            with tf.name_scope(self.conv.name):
                self.conv.build(None)

        # 如果存在 LayerNormalization 层对象 self.LayerNorm，则调用其 build 方法构建该层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, self.config.hidden_size])

        # 遍历 self.layer 列表中的每个 TFDebertaV2Layer 对象，调用其 build 方法构建各层
        if getattr(self, "layer", None) is not None:
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    layer.build(None)

    # 获取相对位置嵌入向量
    def get_rel_embedding(self):
        rel_embeddings = self.rel_embeddings if self.relative_attention else None
        
        # 如果相对位置嵌入存在且需要进行 LayerNormalization，则应用 LayerNorm 层
        if rel_embeddings is not None and ("layer_norm" in self.norm_rel_ebd):
            rel_embeddings = self.LayerNorm(rel_embeddings)
        
        return rel_embeddings

    # 获取注意力掩码
    def get_attention_mask(self, attention_mask):
        # 如果 attention_mask 的维度小于等于 2，则扩展维度以适应模型需求
        if len(shape_list(attention_mask)) <= 2:
            extended_attention_mask = tf.expand_dims(tf.expand_dims(attention_mask, 1), 2)
            attention_mask = extended_attention_mask * tf.expand_dims(tf.squeeze(extended_attention_mask, -2), -1)
            attention_mask = tf.cast(attention_mask, tf.uint8)
        # 如果 attention_mask 的维度为 3，则添加额外的维度以适应模型需求
        elif len(shape_list(attention_mask)) == 3:
            attention_mask = tf.expand_dims(attention_mask, 1)

        return attention_mask
    # 如果启用相对注意力且未提供相对位置参数，则根据查询状态或隐藏状态的形状获取相对位置
    if self.relative_attention and relative_pos is None:
        q = shape_list(query_states)[-2] if query_states is not None else shape_list(hidden_states)[-2]
        relative_pos = build_relative_position(
            q,
            shape_list(hidden_states)[-2],
            bucket_size=self.position_buckets,
            max_position=self.max_relative_positions,
        )
    
    # 返回相对位置参数
    return relative_pos


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
) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
    # 如果注意力掩码的维度小于等于2，则直接使用注意力掩码作为输入掩码
    if len(shape_list(attention_mask)) <= 2:
        input_mask = attention_mask
    else:
        # 将多维度的注意力掩码按最后第二维求和，并转换成 uint8 类型的掩码
        input_mask = tf.cast(tf.math.reduce_sum(attention_mask, axis=-2) > 0, dtype=tf.uint8)

    # 如果设置输出隐藏状态，则初始化空元组以存储所有隐藏状态
    all_hidden_states = () if output_hidden_states else None
    # 如果设置输出注意力权重，则初始化空元组以存储所有注意力权重
    all_attentions = () if output_attentions else None

    # 获取注意力掩码，确保其为正确的形式
    attention_mask = self.get_attention_mask(attention_mask)
    # 获取相对位置编码
    relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)

    # 初始化下一层键值对，即当前隐藏状态
    next_kv = hidden_states

    # 获取相对位置嵌入
    rel_embeddings = self.get_rel_embedding()
    # 初始化输出状态为当前隐藏状态
    output_states = next_kv

    # 遍历所有层进行前向传播
    for i, layer_module in enumerate(self.layer):
        # 如果需要输出隐藏状态，则将当前输出状态加入所有隐藏状态元组中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (output_states,)

        # 调用当前层的前向传播
        layer_outputs = layer_module(
            hidden_states=next_kv,
            attention_mask=attention_mask,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
            output_attentions=output_attentions,
            training=training,
        )
        # 更新输出状态为当前层的输出
        output_states = layer_outputs[0]

        # 如果是第一层且有卷积操作，则将当前隐藏状态与输出状态应用卷积
        if i == 0 and self.conv is not None:
            output_states = self.conv(hidden_states, output_states, input_mask)

        # 更新下一层键值对为当前输出状态
        next_kv = output_states

        # 如果需要输出注意力权重，则将当前层的注意力权重加入所有注意力元组中
        if output_attentions:
            all_attentions = all_attentions + (layer_outputs[1],)

    # 如果需要输出隐藏状态，则将最后一层的输出状态加入所有隐藏状态元组中
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (output_states,)

    # 如果不需要返回字典形式的输出，则按顺序返回相应的结果元组
    if not return_dict:
        return tuple(v for v in [output_states, all_hidden_states, all_attentions] if v is not None)

    # 返回 TFBaseModelOutput 类型的输出，包括最后的隐藏状态、所有隐藏状态和所有注意力权重
    return TFBaseModelOutput(
        last_hidden_state=output_states, hidden_states=all_hidden_states, attentions=all_attentions
    )
# 根据相对位置、桶大小和最大位置生成日志桶位置
def make_log_bucket_position(relative_pos, bucket_size, max_position):
    # 确定相对位置的符号
    sign = tf.math.sign(relative_pos)
    # 计算相对位置的绝对值
    mid = bucket_size // 2
    abs_pos = tf.where((relative_pos < mid) & (relative_pos > -mid), mid - 1, tf.math.abs(relative_pos))
    # 计算对数位置
    log_pos = (
        tf.math.ceil(
            tf.cast(tf.math.log(abs_pos / mid), tf.float32) / tf.math.log((max_position - 1) / mid) * (mid - 1)
        )
        + mid
    )
    # 根据绝对位置是否小于等于桶大小的一半来确定最终桶位置
    bucket_pos = tf.cast(
        tf.where(abs_pos <= mid, tf.cast(relative_pos, tf.float32), log_pos * tf.cast(sign, tf.float32)), tf.int32
    )
    return bucket_pos


# 构建相对位置张量
def build_relative_position(query_size, key_size, bucket_size=-1, max_position=-1):
    """
    Build relative position according to the query and key

    We assume the absolute position of query \\(P_q\\) is range from (0, query_size) and the absolute position of key
    \\(P_k\\) is range from (0, key_size), The relative positions from query to key is \\(R_{q \\rightarrow k} = P_q -
    P_k\\)

    Args:
        query_size (int): the length of query
        key_size (int): the length of key
        bucket_size (int): the size of position bucket
        max_position (int): the maximum allowed absolute position

    Return:
        `tf.Tensor`: A tensor with shape [1, query_size, key_size]

    """
    # 生成查询和键的索引
    q_ids = tf.range(query_size, dtype=tf.int32)
    k_ids = tf.range(key_size, dtype=tf.int32)
    # 计算相对位置
    rel_pos_ids = q_ids[:, None] - tf.tile(tf.expand_dims(k_ids, axis=0), [shape_list(q_ids)[0], 1])
    # 如果指定了桶大小和最大位置，则使用日志桶位置函数计算相对位置
    if bucket_size > 0 and max_position > 0:
        rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position)
    # 裁剪并扩展相对位置张量的维度
    rel_pos_ids = rel_pos_ids[:query_size, :]
    rel_pos_ids = tf.expand_dims(rel_pos_ids, axis=0)
    return tf.cast(rel_pos_ids, tf.int64)


# 扩展相对位置张量以匹配查询层
def c2p_dynamic_expand(c2p_pos, query_layer, relative_pos):
    shapes = [
        shape_list(query_layer)[0],
        shape_list(query_layer)[1],
        shape_list(query_layer)[2],
        shape_list(relative_pos)[-1],
    ]
    return tf.broadcast_to(c2p_pos, shapes)


# 扩展相对位置张量以匹配键层
def p2c_dynamic_expand(c2p_pos, query_layer, key_layer):
    shapes = [
        shape_list(query_layer)[0],
        shape_list(query_layer)[1],
        shape_list(key_layer)[-2],
        shape_list(key_layer)[-2],
    ]
    return tf.broadcast_to(c2p_pos, shapes)


# 扩展位置索引以匹配关键层
def pos_dynamic_expand(pos_index, p2c_att, key_layer):
    shapes = shape_list(p2c_att)[:2] + [shape_list(pos_index)[-2], shape_list(key_layer)[-2]]
    return tf.broadcast_to(pos_index, shapes)


# 沿着轴取出张量的元素
def take_along_axis(x, indices):
    # 当 gather 轴为 -1 时才是有效的 np.take_along_axis 的端口

    # TPU 和 gather 操作在一起可能存在问题 -- 参考 https://github.com/huggingface/transformers/issues/18239
    pass  # 这个函数目前没有实际代码实现，暂时只是占位
    # 检查当前的分布策略是否为 TPUStrategy
    if isinstance(tf.distribute.get_strategy(), tf.distribute.TPUStrategy):
        # 对输入的索引进行独热编码，扩展最后一个维度的深度为 x 张量的最后一个维度的大小
        one_hot_indices = tf.one_hot(indices, depth=x.shape[-1], dtype=x.dtype)

        # 使用 Einstein Summation (einsum) 实现矩阵乘法，将独热编码的张量和 x 张量相乘，忽略前两个维度，得到形状为 [B, S, P] 的结果
        # 这里滥用符号表示：[B, S, P, D] . [B, S, D] = [B, S, P]
        gathered = tf.einsum("ijkl,ijl->ijk", one_hot_indices, x)

    else:
        # 在 GPU 上，通常使用 gather 操作代替大规模的独热编码和矩阵乘法
        gathered = tf.gather(x, indices, batch_dims=2)

    # 返回最终的 gathered 张量
    return gathered
class TFDebertaV2DisentangledSelfAttention(keras.layers.Layer):
    """
    Disentangled self-attention module

    Parameters:
        config (`DebertaV2Config`):
            A model config class instance with the configuration to build a new model. The schema is similar to
            *BertConfig*, for more details, please refer [`DebertaV2Config`]

    """

    def transpose_for_scores(self, tensor: tf.Tensor, attention_heads: int) -> tf.Tensor:
        # 获取张量的形状列表
        tensor_shape = shape_list(tensor)
        # 在图模式下，如果第一个维度（批处理大小）为None，则无法将最终维度为-1的形状进行重塑
        shape = tensor_shape[:-1] + [attention_heads, tensor_shape[-1] // attention_heads]
        # 从[batch_size, seq_length, all_head_size]重塑为[batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(tensor=tensor, shape=shape)
        # 转置张量的维度顺序
        tensor = tf.transpose(tensor, perm=[0, 2, 1, 3])
        x_shape = shape_list(tensor)
        # 再次重塑张量的形状
        tensor = tf.reshape(tensor, shape=[-1, x_shape[-2], x_shape[-1]])
        return tensor

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        query_states: tf.Tensor = None,
        relative_pos: tf.Tensor = None,
        rel_embeddings: tf.Tensor = None,
        output_attentions: bool = False,
        training: bool = False,
    ):
        # 该方法定义了层的正向传播逻辑
        # 省略了具体的实现细节

    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 设置为已构建状态
        self.built = True
        # 如果存在查询投影层，则构建查询投影层
        if getattr(self, "query_proj", None) is not None:
            with tf.name_scope(self.query_proj.name):
                self.query_proj.build([None, None, self.config.hidden_size])
        # 如果存在键投影层，则构建键投影层
        if getattr(self, "key_proj", None) is not None:
            with tf.name_scope(self.key_proj.name):
                self.key_proj.build([None, None, self.config.hidden_size])
        # 如果存在值投影层，则构建值投影层
        if getattr(self, "value_proj", None) is not None:
            with tf.name_scope(self.value_proj.name):
                self.value_proj.build([None, None, self.config.hidden_size])
        # 如果存在dropout层，则构建dropout层
        if getattr(self, "dropout", None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)
        # 如果存在位置dropout层，则构建位置dropout层
        if getattr(self, "pos_dropout", None) is not None:
            with tf.name_scope(self.pos_dropout.name):
                self.pos_dropout.build(None)
        # 如果存在位置键投影层，则构建位置键投影层
        if getattr(self, "pos_key_proj", None) is not None:
            with tf.name_scope(self.pos_key_proj.name):
                self.pos_key_proj.build([None, None, self.config.hidden_size])
        # 如果存在位置查询投影层，则构建位置查询投影层
        if getattr(self, "pos_query_proj", None) is not None:
            with tf.name_scope(self.pos_query_proj.name):
                self.pos_query_proj.build([None, None, self.config.hidden_size])

# 从transformers.models.deberta.modeling_tf_deberta.TFDebertaEmbeddings复制而来 Deberta->DebertaV2
class TFDebertaV2Embeddings(keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""
    # 初始化方法，接收配置对象和额外的关键字参数
    def __init__(self, config, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 将配置对象保存到实例变量中
        self.config = config
        # 获取嵌入向量的大小，默认为隐藏层大小
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)
        # 保存隐藏层大小到实例变量
        self.hidden_size = config.hidden_size
        # 保存最大位置嵌入长度到实例变量
        self.max_position_embeddings = config.max_position_embeddings
        # 根据配置设置是否使用位置偏置输入，默认为True
        self.position_biased_input = getattr(config, "position_biased_input", True)
        # 保存初始化范围到实例变量
        self.initializer_range = config.initializer_range
        
        # 如果嵌入向量大小不等于隐藏层大小，则创建一个全连接层作为投影层
        if self.embedding_size != config.hidden_size:
            self.embed_proj = keras.layers.Dense(
                config.hidden_size,
                kernel_initializer=get_initializer(config.initializer_range),
                name="embed_proj",
                use_bias=False,
            )
        
        # 创建LayerNormalization层，并设置epsilon参数
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建稳定Dropout层，并设置隐藏层dropout概率
        self.dropout = TFDebertaV2StableDropout(config.hidden_dropout_prob, name="dropout")

    # 构建模型的方法，用于创建模型的各种层和权重
    def build(self, input_shape=None):
        # 创建词嵌入层的权重矩阵，形状为[vocab_size, embedding_size]
        with tf.name_scope("word_embeddings"):
            self.weight = self.add_weight(
                name="weight",
                shape=[self.config.vocab_size, self.embedding_size],
                initializer=get_initializer(self.initializer_range),
            )

        # 创建token type嵌入层的权重矩阵，形状为[type_vocab_size, embedding_size]
        with tf.name_scope("token_type_embeddings"):
            if self.config.type_vocab_size > 0:
                self.token_type_embeddings = self.add_weight(
                    name="embeddings",
                    shape=[self.config.type_vocab_size, self.embedding_size],
                    initializer=get_initializer(self.initializer_range),
                )
            else:
                self.token_type_embeddings = None

        # 创建位置嵌入层的权重矩阵，形状为[max_position_embeddings, hidden_size]
        with tf.name_scope("position_embeddings"):
            if self.position_biased_input:
                self.position_embeddings = self.add_weight(
                    name="embeddings",
                    shape=[self.max_position_embeddings, self.hidden_size],
                    initializer=get_initializer(self.initializer_range),
                )
            else:
                self.position_embeddings = None

        # 如果模型已经构建，则直接返回
        if self.built:
            return
        self.built = True
        
        # 如果存在LayerNorm层，则构建LayerNorm层的结构
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])
        
        # 如果存在dropout层，则构建dropout层的结构
        if getattr(self, "dropout", None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)
        
        # 如果存在embed_proj投影层，则构建embed_proj层的结构
        if getattr(self, "embed_proj", None) is not None:
            with tf.name_scope(self.embed_proj.name):
                self.embed_proj.build([None, None, self.embedding_size])

    # 模型调用方法，定义了模型的前向传播逻辑
    def call(
        self,
        input_ids: tf.Tensor = None,
        position_ids: tf.Tensor = None,
        token_type_ids: tf.Tensor = None,
        inputs_embeds: tf.Tensor = None,
        mask: tf.Tensor = None,
        training: bool = False,
        # 继续定义其他输入参数
    def apply_embeddings(
        self,
        input_ids: Optional[tf.Tensor] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        token_type_ids: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        mask: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> tf.Tensor:
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        # 如果既没有提供 input_ids 也没有提供 inputs_embeds，则抛出数值错误
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Need to provide either `input_ids` or `inputs_embeds`.")

        # 如果提供了 input_ids，则检查其是否在合法范围内
        if input_ids is not None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            # 使用 self.weight 中的参数和 input_ids 进行 gather 操作，得到 inputs_embeds
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        # 获取 inputs_embeds 的形状，去除最后一个维度
        input_shape = shape_list(inputs_embeds)[:-1]

        # 如果 token_type_ids 未提供，则使用全零张量进行填充
        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)

        # 如果 position_ids 未提供，则创建一个范围从 0 到 input_shape[-1] 的张量，并扩展维度
        if position_ids is None:
            position_ids = tf.expand_dims(tf.range(start=0, limit=input_shape[-1]), axis=0)

        # 初始的 final_embeddings 设为 inputs_embeds
        final_embeddings = inputs_embeds

        # 如果设置了 self.position_biased_input，则添加 position_embeddings 到 final_embeddings
        if self.position_biased_input:
            position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
            final_embeddings += position_embeds

        # 如果配置中的 type_vocab_size 大于 0，则添加 token_type_embeddings 到 final_embeddings
        if self.config.type_vocab_size > 0:
            token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)
            final_embeddings += token_type_embeds

        # 如果 embedding_size 不等于 hidden_size，则对 final_embeddings 应用 embed_proj 函数
        if self.embedding_size != self.hidden_size:
            final_embeddings = self.embed_proj(final_embeddings)

        # 对 final_embeddings 应用 LayerNorm 函数
        final_embeddings = self.LayerNorm(final_embeddings)

        # 如果提供了 mask，则根据其形状调整 final_embeddings
        if mask is not None:
            if len(shape_list(mask)) != len(shape_list(final_embeddings)):
                if len(shape_list(mask)) == 4:
                    mask = tf.squeeze(tf.squeeze(mask, axis=1), axis=1)
                mask = tf.cast(tf.expand_dims(mask, axis=2), tf.float32)

            final_embeddings = final_embeddings * mask

        # 对 final_embeddings 应用 dropout 函数，如果处于训练状态
        final_embeddings = self.dropout(final_embeddings, training=training)

        # 返回最终的嵌入张量 final_embeddings
        return final_embeddings
# 从 transformers.models.deberta.modeling_tf_deberta.TFDebertaPredictionHeadTransform 复制而来，将 Deberta->DebertaV2
class TFDebertaV2PredictionHeadTransform(keras.layers.Layer):
    def __init__(self, config: DebertaV2Config, **kwargs):
        super().__init__(**kwargs)

        # 从配置中获取嵌入大小，默认为隐藏大小
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)

        # 创建一个全连接层，用于转换隐藏状态到嵌入大小
        self.dense = keras.layers.Dense(
            units=self.embedding_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense",
        )

        # 根据配置获取隐藏层激活函数，如果是字符串则获取相应的 TensorFlow 激活函数，否则直接使用配置中的函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.transform_act_fn = config.hidden_act
        
        # 创建一个 LayerNormalization 层，用于归一化隐藏状态
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 使用全连接层进行隐藏状态的转换
        hidden_states = self.dense(inputs=hidden_states)
        # 应用激活函数
        hidden_states = self.transform_act_fn(hidden_states)
        # 应用 LayerNormalization
        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states

    def build(self, input_shape=None):
        # 如果已经构建，则直接返回
        if self.built:
            return
        self.built = True
        # 如果 dense 层已存在，则构建 dense 层，指定输入形状为 [None, None, self.config.hidden_size]
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 如果 LayerNorm 层已存在，则构建 LayerNorm 层，指定输入形状为 [None, None, self.embedding_size]
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.embedding_size])


# 从 transformers.models.deberta.modeling_tf_deberta.TFDebertaLMPredictionHead 复制而来，将 Deberta->DebertaV2
class TFDebertaV2LMPredictionHead(keras.layers.Layer):
    def __init__(self, config: DebertaV2Config, input_embeddings: keras.layers.Layer, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        # 从配置中获取嵌入大小，默认为隐藏大小
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)

        # 使用 TFDebertaV2PredictionHeadTransform 进行隐藏状态到嵌入大小的转换
        self.transform = TFDebertaV2PredictionHeadTransform(config, name="transform")

        # 输出权重与输入嵌入相同，但每个标记仅有一个输出偏置
        self.input_embeddings = input_embeddings

    def build(self, input_shape=None):
        # 添加一个与词汇表大小相同的偏置，初始化为零，可训练，命名为 "bias"
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="bias")

        # 如果已经构建，则直接返回
        if self.built:
            return
        self.built = True
        # 如果 transform 层已存在，则构建 transform 层
        if getattr(self, "transform", None) is not None:
            with tf.name_scope(self.transform.name):
                self.transform.build(None)

    def get_output_embeddings(self) -> keras.layers.Layer:
        # 返回输入嵌入层
        return self.input_embeddings

    def set_output_embeddings(self, value: tf.Variable):
        # 设置输出嵌入权重
        self.input_embeddings.weight = value
        # 设置输出嵌入词汇表大小为 value 的第一个维度长度
        self.input_embeddings.vocab_size = shape_list(value)[0]
    # 返回包含偏置项的字典，字典中键为"bias"，值为 self.bias 变量
    def get_bias(self) -> Dict[str, tf.Variable]:
        return {"bias": self.bias}

    # 设置偏置项，从给定的 value 字典中取出"bias"键对应的值，并赋给 self.bias
    # 同时更新 self.config.vocab_size，使用 shape_list 函数获取 value["bias"] 的形状，并取其第一个元素作为 vocab_size
    def set_bias(self, value: tf.Variable):
        self.bias = value["bias"]
        self.config.vocab_size = shape_list(value["bias"])[0]

    # 对隐藏状态进行变换，调用 self.transform 方法
    # 获取隐藏状态的序列长度，并保存在 seq_length 中
    # 将隐藏状态进行形状重塑，变成二维张量，第一维度为-1，第二维度为 self.embedding_size
    # 使用矩阵乘法计算 hidden_states 和 self.input_embeddings.weight 的转置的乘积
    # 再次对 hidden_states 进行形状重塑，变成三维张量，形状为 [-1, seq_length, self.config.vocab_size]
    # 使用偏置项 self.bias 对 hidden_states 进行偏置添加操作
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.transform(hidden_states=hidden_states)
        seq_length = shape_list(hidden_states)[1]
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.embedding_size])
        hidden_states = tf.matmul(a=hidden_states, b=self.input_embeddings.weight, transpose_b=True)
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)

        return hidden_states
# 从 transformers.models.deberta.modeling_tf_deberta.TFDebertaOnlyMLMHead 复制而来，将 Deberta 替换为 DebertaV2
class TFDebertaV2OnlyMLMHead(keras.layers.Layer):
    def __init__(self, config: DebertaV2Config, input_embeddings: keras.layers.Layer, **kwargs):
        super().__init__(**kwargs)
        # 使用给定的配置和输入嵌入层创建预测头部对象
        self.predictions = TFDebertaV2LMPredictionHead(config, input_embeddings, name="predictions")

    def call(self, sequence_output: tf.Tensor) -> tf.Tensor:
        # 调用预测头部对象以生成预测分数
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


# 从 transformers.models.deberta.modeling_tf_deberta.TFDebertaMainLayer 复制而来，将 Deberta 替换为 DebertaV2
class TFDebertaV2MainLayer(keras.layers.Layer):
    config_class = DebertaV2Config

    def __init__(self, config: DebertaV2Config, **kwargs):
        super().__init__(**kwargs)

        self.config = config

        # 使用给定的配置创建嵌入层和编码器
        self.embeddings = TFDebertaV2Embeddings(config, name="embeddings")
        self.encoder = TFDebertaV2Encoder(config, name="encoder")

    def get_input_embeddings(self) -> keras.layers.Layer:
        return self.embeddings

    def set_input_embeddings(self, value: tf.Variable):
        # 设置输入嵌入层的权重和词汇大小
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 剪枝模型的注意力头部，具体实现未提供
        raise NotImplementedError

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
        ):
        # 调用模型的主要层，处理输入并返回相应的输出
        # 这里的 unpack_inputs 装饰器用于解包输入参数
        ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        # 如果同时指定了 input_ids 和 inputs_embeds，则抛出数值错误异常
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        # 如果只指定了 input_ids，则获取其形状信息
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        # 如果只指定了 inputs_embeds，则获取其形状信息，去掉最后一个维度
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        # 如果既没有指定 input_ids 也没有指定 inputs_embeds，则抛出数值错误异常
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # 如果未提供 attention_mask，则用维度为 input_shape 的全 1 张量来填充
        if attention_mask is None:
            attention_mask = tf.fill(dims=input_shape, value=1)

        # 如果未提供 token_type_ids，则用维度为 input_shape 的全 0 张量来填充
        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)

        # 使用 embeddings 层处理输入，得到嵌入输出
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            mask=attention_mask,
            training=training,
        )

        # 使用 encoder 层处理嵌入输出，得到编码器的输出
        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 从编码器的输出中取出序列输出（第一个元素）
        sequence_output = encoder_outputs[0]

        # 如果不要求返回字典格式的输出，则返回元组形式的编码器输出
        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        # 否则，返回 TFBaseModelOutput 对象，包括最后隐藏状态、隐藏状态列表和注意力列表
        return TFBaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果已经构建过网络结构，则直接返回
        if self.built:
            return
        # 将标志设置为已构建
        self.built = True
        # 如果存在 embeddings 层，则构建 embeddings 层
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        # 如果存在 encoder 层，则构建 encoder 层
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
# 从 transformers.models.deberta.modeling_tf_deberta.TFDebertaPreTrainedModel 复制的代码，将 Deberta->DebertaV2
class TFDebertaV2PreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定配置类为 DebertaV2Config
    config_class = DebertaV2Config
    # 设置基础模型前缀为 "deberta"
    base_model_prefix = "deberta"


# DEBERTA_START_DOCSTRING 的原始文档字符串
DEBERTA_START_DOCSTRING = r"""
    The DeBERTa model was proposed in [DeBERTa: Decoding-enhanced BERT with Disentangled
    Attention](https://arxiv.org/abs/2006.03654) by Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen. It's build
    on top of BERT/RoBERTa with two improvements, i.e. disentangled attention and enhanced mask decoder. With those two
    improvements, it out perform BERT/RoBERTa on a majority of tasks with 80GB pretraining data.

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
        config ([`DebertaV2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""
This block defines the docstring for inputs expected by the DeBERTaV2Model class.
It specifies the arguments and their types that can be passed to the model.

@add_start_docstrings is a decorator that adds a specific docstring template to the class.
It describes the bare DeBERTa Model transformer outputting raw hidden-states without specific head.

The class TFDebertaV2Model inherits from TFDebertaV2PreTrainedModel and represents the DeBERTa V2 model.
It initializes with a configuration object and optional inputs, and initializes a TFDebertaV2MainLayer named "deberta".

unpack_inputs is a decorator that likely unpacks and prepares inputs before feeding them into the model.
"""
class TFDebertaV2Model(TFDebertaV2PreTrainedModel):
    def __init__(self, config: DebertaV2Config, *inputs, **kwargs):
        # Initialize the superclass with the provided configuration and optional inputs
        super().__init__(config, *inputs, **kwargs)

        # Create an instance of TFDebertaV2MainLayer to serve as the core DeBERTa V2 transformer
        self.deberta = TFDebertaV2MainLayer(config, name="deberta")

    # Decorator function that manages the unpacking of inputs
    @unpack_inputs
    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 添加模型前向传播方法的文档字符串，包含 DEBERTA_INPUTS_DOCSTRING 的格式化参数
    def call(
        self,
        input_ids: TFModelInputType | None = None,  # 输入的 token IDs，可以是 None
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 注意力遮罩，可以是 numpy 数组或 TensorFlow 张量，也可以是 None
        token_type_ids: np.ndarray | tf.Tensor | None = None,  # token 类型 IDs，可以是 numpy 数组或 TensorFlow 张量，也可以是 None
        position_ids: np.ndarray | tf.Tensor | None = None,  # 位置 IDs，可以是 numpy 数组或 TensorFlow 张量，也可以是 None
        inputs_embeds: np.ndarray | tf.Tensor | None = None,  # 输入的嵌入向量，可以是 numpy 数组或 TensorFlow 张量，也可以是 None
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可选布尔值，默认为 None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选布尔值，默认为 None
        return_dict: Optional[bool] = None,  # 是否返回字典格式的输出，可选布尔值，默认为 None
        training: Optional[bool] = False,  # 是否处于训练模式，可选布尔值，默认为 False
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        # 调用 DEBERTA 模型的前向传播方法，并传递相应的参数
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

        # 返回 DEBERTA 模型前向传播方法的输出
        return outputs

    def build(self, input_shape=None):
        # 如果已经构建过模型，则直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果 self.deberta 存在，则在 TensorFlow 的命名空间下构建它
        if getattr(self, "deberta", None) is not None:
            with tf.name_scope(self.deberta.name):
                self.deberta.build(None)
# 给 TFDebertaV2ForMaskedLM 类添加文档字符串，描述其作为 DeBERTa 模型的一个扩展，包含语言建模头部
@add_start_docstrings("""DeBERTa Model with a `language modeling` head on top.""", DEBERTA_START_DOCSTRING)
# 从 transformers.models.deberta.modeling_tf_deberta.TFDebertaForMaskedLM 复制代码，并将 Deberta 更改为 DebertaV2
class TFDebertaV2ForMaskedLM(TFDebertaV2PreTrainedModel, TFMaskedLanguageModelingLoss):
    def __init__(self, config: DebertaV2Config, *inputs, **kwargs):
        # 调用父类构造函数初始化模型
        super().__init__(config, *inputs, **kwargs)

        # 如果配置指定为解码器，发出警告信息，建议设置 config.is_decoder=False 来使用双向自注意力
        if config.is_decoder:
            logger.warning(
                "If you want to use `TFDebertaV2ForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # 初始化 DeBERTa V2 主层
        self.deberta = TFDebertaV2MainLayer(config, name="deberta")
        # 初始化仅包含 MLM 头部的 DeBERTa V2 模型头部
        self.mlm = TFDebertaV2OnlyMLMHead(config, input_embeddings=self.deberta.embeddings, name="cls")

    # 获取语言建模头部的方法
    def get_lm_head(self) -> keras.layers.Layer:
        return self.mlm.predictions

    # 实现模型调用的方法，支持多种输入和输出
    @unpack_inputs
    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFMaskedLMOutput,
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
        # 以下参数用于详细描述输入要求和预期的输出
    ) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        # 调用 DeBERTa 模型进行前向传播，获取模型的输出
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
        # 从模型输出中获取序列输出（通常是模型最后一层的输出）
        sequence_output = outputs[0]
        # 将序列输出输入到 MLM 层，生成预测的分数
        prediction_scores = self.mlm(sequence_output=sequence_output, training=training)
        # 如果有提供标签，则计算 MLM 损失
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=prediction_scores)

        # 如果不要求返回字典形式的输出，则按照元组形式构建输出结果
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果要求返回字典形式的输出，则构建 TFMaskedLMOutput 对象并返回
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
        # 设置模型为已构建状态
        self.built = True
        # 如果模型包含 DeBERTa 层，则构建 DeBERTa 层
        if getattr(self, "deberta", None) is not None:
            with tf.name_scope(self.deberta.name):
                self.deberta.build(None)
        # 如果模型包含 MLM 层，则构建 MLM 层
        if getattr(self, "mlm", None) is not None:
            with tf.name_scope(self.mlm.name):
                self.mlm.build(None)
"""
DeBERTa Model transformer with a sequence classification/regression head on top (a linear layer on top of the
pooled output) e.g. for GLUE tasks.
"""

# 从transformers.models.deberta.modeling_tf_deberta.TFDebertaForSequenceClassification复制而来，将Deberta改为DebertaV2
class TFDebertaV2ForSequenceClassification(TFDebertaV2PreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config: DebertaV2Config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels

        # 初始化DeBERTa V2主层和上下文池化层
        self.deberta = TFDebertaV2MainLayer(config, name="deberta")
        self.pooler = TFDebertaV2ContextPooler(config, name="pooler")

        # 设置分类器的dropout率，如果未指定则使用config中的默认值
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = TFDebertaV2StableDropout(drop_out, name="cls_dropout")

        # 定义分类器，用于进行具体的分类任务
        self.classifier = keras.layers.Dense(
            units=config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="classifier",
        )

        # 输出维度为池化层的输出维度
        self.output_dim = self.pooler.output_dim

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 模型的前向传播函数，处理输入并返回分类结果
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
        **kwargs
    ):
        """
        接收输入参数，执行DeBERTa V2模型的前向传播，返回分类任务的输出结果。
        """
    ) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 调用DeBERTa模型进行前向传播计算
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
        # 取模型输出的第一个元素作为序列输出
        sequence_output = outputs[0]
        # 通过池化层计算汇聚输出
        pooled_output = self.pooler(sequence_output, training=training)
        # 使用dropout进行汇聚输出的随机失活
        pooled_output = self.dropout(pooled_output, training=training)
        # 通过分类器获取最终的逻辑回归输出
        logits = self.classifier(pooled_output)
        # 如果提供了标签，则计算损失值
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        # 如果不要求返回字典，则将输出打包成元组
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 否则，返回TFSequenceClassifierOutput对象，包含损失、逻辑回归输出、隐藏状态和注意力分布
        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回
        if self.built:
            return
        # 标记模型为已构建状态
        self.built = True
        # 如果模型包含DeBERTa组件，则构建DeBERTa模型
        if getattr(self, "deberta", None) is not None:
            with tf.name_scope(self.deberta.name):
                self.deberta.build(None)
        # 如果模型包含池化层组件，则构建池化层
        if getattr(self, "pooler", None) is not None:
            with tf.name_scope(self.pooler.name):
                self.pooler.build(None)
        # 如果模型包含dropout组件，则构建dropout层
        if getattr(self, "dropout", None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)
        # 如果模型包含分类器组件，则构建分类器
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.output_dim])
@add_start_docstrings(
    """
    DeBERTa Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    DEBERTA_START_DOCSTRING,
)
# 基于 transformers.models.deberta.modeling_tf_deberta.TFDebertaForTokenClassification 的修改版本，用于 DeBERTaV2
class TFDebertaV2ForTokenClassification(TFDebertaV2PreTrainedModel, TFTokenClassificationLoss):
    def __init__(self, config: DebertaV2Config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 设置分类的标签数量
        self.num_labels = config.num_labels

        # 初始化 DeBERTaV2 主层
        self.deberta = TFDebertaV2MainLayer(config, name="deberta")
        # 设置 dropout 层
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        # 分类器，用于输出最终的标签预测
        self.classifier = keras.layers.Dense(
            units=config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        # 保存配置信息
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 模型调用函数，用于推断和训练
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
    ) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        outputs = self.deberta(
            input_ids=input_ids,                      # 输入的 token IDs
            attention_mask=attention_mask,            # 注意力掩码
            token_type_ids=token_type_ids,            # token 类型 IDs
            position_ids=position_ids,                # 位置 IDs
            inputs_embeds=inputs_embeds,              # 嵌入的输入
            output_attentions=output_attentions,      # 是否输出注意力权重
            output_hidden_states=output_hidden_states,# 是否输出隐藏状态
            return_dict=return_dict,                  # 是否返回字典格式结果
            training=training,                        # 是否在训练模式下
        )
        sequence_output = outputs[0]                  # 取出模型输出的序列输出
        sequence_output = self.dropout(sequence_output, training=training)  # 对序列输出应用 dropout
        logits = self.classifier(inputs=sequence_output)  # 将序列输出输入分类器得到 logits
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)  # 如果有标签，则计算损失

        if not return_dict:
            output = (logits,) + outputs[1:]         # 如果不返回字典，则组合输出
            return ((loss,) + output) if loss is not None else output  # 如果有损失，则包含在输出中

        return TFTokenClassifierOutput(
            loss=loss,                               # 返回 TFTokenClassifierOutput 对象，包含损失
            logits=logits,                           # logits
            hidden_states=outputs.hidden_states,     # 隐藏状态
            attentions=outputs.attentions,           # 注意力权重
        )

    def build(self, input_shape=None):
        if self.built:                              # 如果已经建立则直接返回
            return
        self.built = True                           # 标记为已建立

        if getattr(self, "deberta", None) is not None:  # 如果存在 deberta 模型
            with tf.name_scope(self.deberta.name):   # 在 tf 中使用 deberta 模型的名称作为命名空间
                self.deberta.build(None)             # 建立 deberta 模型

        if getattr(self, "classifier", None) is not None:  # 如果存在分类器
            with tf.name_scope(self.classifier.name):  # 在 tf 中使用分类器的名称作为命名空间
                self.classifier.build([None, None, self.config.hidden_size])  # 建立分类器，输入形状为 [None, None, hidden_size]
@add_start_docstrings(
    """
    DeBERTa Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    DEBERTA_START_DOCSTRING,
)
# 从 transformers.models.deberta.modeling_tf_deberta.TFDebertaForQuestionAnswering 复制并修改为 Deberta->DebertaV2
class TFDebertaV2ForQuestionAnswering(TFDebertaV2PreTrainedModel, TFQuestionAnsweringLoss):
    def __init__(self, config: DebertaV2Config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 设置分类标签数量
        self.num_labels = config.num_labels

        # 初始化 DeBERTa 主层
        self.deberta = TFDebertaV2MainLayer(config, name="deberta")
        
        # 定义输出层，用于计算起始和结束位置的 logit
        self.qa_outputs = keras.layers.Dense(
            units=config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )
        
        # 存储配置信息
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
        **kwargs,
    ):
        """
        执行 DeBERTaV2ForQuestionAnswering 的前向传播。
        
        Args:
            input_ids: 输入的 token IDs
            attention_mask: 输入的注意力掩码
            token_type_ids: 输入的 token 类型 IDs
            position_ids: 输入的位置 IDs
            inputs_embeds: 替代输入的嵌入表示
            output_attentions: 是否输出注意力权重
            output_hidden_states: 是否输出隐藏状态
            return_dict: 是否返回字典格式的输出
            start_positions: 起始位置的标签
            end_positions: 结束位置的标签
            training: 是否为训练模式
        """
        # 调用 DeBERTa 主层进行前向传播
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        
        # 如果训练模式，则计算起始和结束位置的 logit
        if training:
            start_logits, end_logits = self.qa_outputs(outputs.last_hidden_state)
            return start_logits, end_logits
        
        # 否则返回模型的标准输出
        return TFQuestionAnsweringModelOutput(
            start_logits=outputs.start_logits,
            end_logits=outputs.end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
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
        # 调用 DeBERTa 模型进行推理，获取模型的输出
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
        # 获取模型输出的序列输出
        sequence_output = outputs[0]
        # 使用 QA 输出层对序列输出进行处理，得到问题回答的 logits
        logits = self.qa_outputs(inputs=sequence_output)
        # 将 logits 分割为开始位置和结束位置的 logits
        start_logits, end_logits = tf.split(value=logits, num_or_size_splits=2, axis=-1)
        # 去除 logits 张量的一个维度，使其维度减少到 [-1]
        start_logits = tf.squeeze(input=start_logits, axis=-1)
        end_logits = tf.squeeze(input=end_logits, axis=-1)
        # 初始化 loss 为 None
        loss = None

        # 如果给定了起始位置和结束位置的标签，则计算损失
        if start_positions is not None and end_positions is not None:
            # 将起始和结束位置标签存储在字典中
            labels = {"start_position": start_positions}
            labels["end_position"] = end_positions
            # 使用损失计算函数计算损失
            loss = self.hf_compute_loss(labels=labels, logits=(start_logits, end_logits))

        # 如果不需要返回字典形式的输出，则返回 logits 和可能的附加输出
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典形式的输出，则创建 TFQuestionAnsweringModelOutput 对象
        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果 DeBERTa 模型存在，则构建其结构
        if getattr(self, "deberta", None) is not None:
            with tf.name_scope(self.deberta.name):
                self.deberta.build(None)
        # 如果 QA 输出层存在，则构建其结构
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.config.hidden_size])
@add_start_docstrings(
    """
    DeBERTa Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    DEBERTA_START_DOCSTRING,
)
class TFDebertaV2ForMultipleChoice(TFDebertaV2PreTrainedModel, TFMultipleChoiceLoss):
    """
    DeBERTa V2 model for multiple choice tasks. Extends TFDebertaV2PreTrainedModel and TFMultipleChoiceLoss.

    This class defines a model architecture with a DeBERTa V2 main layer, dropout, context pooler, and a dense
    classifier layer for multiple choice classification tasks.
    """

    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    # _keys_to_ignore_on_load_unexpected = [r"mlm___cls", r"nsp___cls", r"cls.predictions", r"cls.seq_relationship"]
    # _keys_to_ignore_on_load_missing = [r"dropout"]

    def __init__(self, config: DebertaV2Config, *inputs, **kwargs):
        """
        Initializes TFDebertaV2ForMultipleChoice.

        Args:
            config (DebertaV2Config): The model configuration class specifying the model architecture and hyperparameters.
            *inputs: Variable length argument list for passing inputs to parent classes.
            **kwargs: Additional keyword arguments passed to parent classes.
        """
        super().__init__(config, *inputs, **kwargs)

        # Initialize DeBERTa V2 main layer
        self.deberta = TFDebertaV2MainLayer(config, name="deberta")
        # Dropout layer with dropout rate specified in config
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        # Context pooler layer for pooling contextual embeddings
        self.pooler = TFDebertaV2ContextPooler(config, name="pooler")
        # Dense classifier layer for multiple choice classification
        self.classifier = keras.layers.Dense(
            units=1, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        # Output dimensionality from the pooler layer
        self.output_dim = self.pooler.output_dim

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
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
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
        **kwargs
    ):
        """
        Perform the forward pass of the TFDebertaV2ForMultipleChoice model.

        Args:
            input_ids (TFModelInputType, optional): Input ids of shape (batch_size, num_choices, sequence_length).
            attention_mask (np.ndarray or tf.Tensor, optional): Attention mask of shape (batch_size, num_choices, sequence_length).
            token_type_ids (np.ndarray or tf.Tensor, optional): Token type ids of shape (batch_size, num_choices, sequence_length).
            position_ids (np.ndarray or tf.Tensor, optional): Position ids of shape (batch_size, num_choices, sequence_length).
            inputs_embeds (np.ndarray or tf.Tensor, optional): Embedded inputs of shape (batch_size, num_choices, sequence_length, hidden_size).
            output_attentions (bool, optional): Whether to output attentions.
            output_hidden_states (bool, optional): Whether to output hidden states.
            return_dict (bool, optional): Whether to return a dictionary.
            labels (np.ndarray or tf.Tensor, optional): Labels for multiple choice task.
            training (bool, optional): Whether the model is in training mode.
            **kwargs: Additional keyword arguments for future extension.

        Returns:
            TFMultipleChoiceModelOutput: Output class with scores and optionally other relevant outputs.
        """
        # Implementation of the model forward pass
        # Details depend on the specific implementation of the DeBERTa model
        pass
    ) -> Union[TFMultipleChoiceModelOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
            where `num_choices` is the size of the second dimension of the input tensors. (See `input_ids` above)
        """
        # 如果输入 `input_ids` 不为 None，则确定选择数量和序列长度
        if input_ids is not None:
            num_choices = shape_list(input_ids)[1]  # 获取选择数量
            seq_length = shape_list(input_ids)[2]   # 获取序列长度
        else:
            num_choices = shape_list(inputs_embeds)[1]  # 获取选择数量（从 `inputs_embeds` 中获取）
            seq_length = shape_list(inputs_embeds)[2]   # 获取序列长度（从 `inputs_embeds` 中获取）

        # 将输入张量展平为二维张量，如果对应输入不为 None
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
        
        # 调用 `deberta` 模型进行前向传播
        outputs = self.deberta(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            token_type_ids=flat_token_type_ids,
            position_ids=flat_position_ids,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        
        # 获取模型输出的序列输出
        sequence_output = outputs[0]
        
        # 使用池化器获取池化输出
        pooled_output = self.pooler(sequence_output, training=training)
        
        # 应用 dropout 处理池化输出
        pooled_output = self.dropout(pooled_output, training=training)
        
        # 使用分类器获取最终的 logits
        logits = self.classifier(pooled_output)
        
        # 将 logits 重新整形为二维张量
        reshaped_logits = tf.reshape(tensor=logits, shape=(-1, num_choices))
        
        # 如果提供了标签，计算损失
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=reshaped_logits)

        # 如果 `return_dict` 为 False，则返回包含 logits 和其它输出的元组
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        # 如果 `return_dict` 为 True，则返回 `TFMultipleChoiceModelOutput` 类的对象
        return TFMultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    # 构建函数用于构建模型的输入形状
    def build(self, input_shape=None):
        # 如果模型已经构建完成，则直接返回，不做任何操作
        if self.built:
            return
        # 标记模型为已构建状态
        self.built = True
        
        # 如果模型中存在名为 "deberta" 的属性且不为 None，则构建 "deberta" 层
        if getattr(self, "deberta", None) is not None:
            # 使用 "deberta" 层的名称作为命名空间，在该命名空间下构建 "deberta" 层
            with tf.name_scope(self.deberta.name):
                self.deberta.build(None)
        
        # 如果模型中存在名为 "pooler" 的属性且不为 None，则构建 "pooler" 层
        if getattr(self, "pooler", None) is not None:
            # 使用 "pooler" 层的名称作为命名空间，在该命名空间下构建 "pooler" 层
            with tf.name_scope(self.pooler.name):
                self.pooler.build(None)
        
        # 如果模型中存在名为 "classifier" 的属性且不为 None，则构建 "classifier" 层
        if getattr(self, "classifier", None) is not None:
            # 使用 "classifier" 层的名称作为命名空间，在该命名空间下构建 "classifier" 层
            self.classifier.build([None, None, self.output_dim])
```