# `.\models\deberta_v2\modeling_tf_deberta_v2.py`

```
# 设置文件编码格式为 UTF-8
# 版权声明
# 版权信息，告知使用者遵守 Apache License 2.0
# 获取许可证的链接
# 根据适用法律或书面同意，依据"原样"的基础分发软件，不提供任何明示或暗示的保证。详见许可证，规定语言的权限和限制
""" TF 2.0 DeBERTa-v2 model."""

# 从__future__模块中导入annotations
from __future__ import annotations

# 导入相关的类型提示
from typing import Dict, Optional, Tuple, Union

# 导入依赖的库
import numpy as np
import tensorflow as tf

# 导入本地依赖的模块和类
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
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_deberta_v2 import DebertaV2Config

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的配置信息和检查点
_CONFIG_FOR_DOC = "DebertaV2Config"
_CHECKPOINT_FOR_DOC = "kamalkraj/deberta-v2-xlarge"

# 预训练模型的存档列表
TF_DEBERTA_V2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "kamalkraj/deberta-v2-xlarge",
    # 查看所有 DeBERTa 模型：https://huggingface.co/models?filter=deberta-v2
]

# 从 transformers.models.deberta.modeling_tf_deberta.TFDebertaContextPooler 复制并修改为 DebertaV2
class TFDebertaV2ContextPooler(tf.keras.layers.Layer):
    def __init__(self, config: DebertaV2Config, **kwargs):
        super().__init__(**kwargs)
        # 定义稠密连接层和稳定 Dropout 层
        self.dense = tf.keras.layers.Dense(config.pooler_hidden_size, name="dense")
        self.dropout = TFDebertaV2StableDropout(config.pooler_dropout, name="dropout")
        self.config = config

    def call(self, hidden_states, training: bool = False):
        # 通过取第一个令牌的隐藏状态来"池化"模型
        context_token = hidden_states[:, 0]
        context_token = self.dropout(context_token, training=training)
        pooled_output = self.dense(context_token)
        pooled_output = get_tf_activation(self.config.pooler_hidden_act)(pooled_output)
        return pooled_output

    @property
    def output_dim(self) -> int:
        return self.config.hidden_size
    # 构建模型，如果已经构建过了则直接返回
    def build(self, input_shape=None):
        # 如果已经构建过了，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在名为dense的属性，则构建dense层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.pooler_hidden_size])
        # 如果存在名为dropout的属性，则构建dropout层
        if getattr(self, "dropout", None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)
# Defining a custom layer TFDebertaV2XSoftmax by inheriting from tf.keras.layers.Layer.
# This layer applies masked softmax on the input tensor, ignoring elements indicated by the mask matrix.
class TFDebertaV2XSoftmax(tf.keras.layers.Layer):
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

    # Overriding the default call method of the layer
    def call(self, inputs: tf.Tensor, mask: tf.Tensor):
        # Create a reverse mask by performing element-wise negation and type conversion to bool
        rmask = tf.logical_not(tf.cast(mask, tf.bool))
        # Replace elements corresponding to the reverse mask with negative infinity
        output = tf.where(rmask, float("-inf"), inputs)
        # Apply stable softmax on the modified input tensor
        output = stable_softmax(output, self.axis)
        # Replace elements corresponding to the reverse mask with 0.0 in the output softmax tensor
        output = tf.where(rmask, 0.0, output)
        return output


# Defining a custom layer TFDebertaV2StableDropout by inheriting from tf.keras.layers.Layer.
# This layer implements an optimized dropout module for stabilizing the training.
class TFDebertaV2StableDropout(tf.keras.layers.Layer):
    """
    Optimized dropout module for stabilizing the training

    Args:
        drop_prob (float): the dropout probabilities
    """

    def __init__(self, drop_prob, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    # Using the tf.custom_gradient decorator to specify custom gradient for the dropout operation
    @tf.custom_gradient
    def xdropout(self, inputs):
        """
        Applies dropout to the inputs, as vanilla dropout, but also scales the remaining elements up by 1/drop_prob.
        """
        # Generate a dropout mask using a Bernoulli distribution
        mask = tf.cast(
            1
            - tf.compat.v1.distributions.Bernoulli(probs=1.0 - self.drop_prob).sample(sample_shape=shape_list(inputs)),
            tf.bool,
        )
        # Calculate the scale factor for the remaining elements after dropout
        scale = tf.convert_to_tensor(1.0 / (1 - self.drop_prob), dtype=tf.float32)
        # If the dropout probability is greater than 0, apply dropout by setting masked elements to 0 and then scale the remaining elements
        if self.drop_prob > 0:
            inputs = tf.where(mask, 0.0, inputs) * scale

        # Define the gradient function for the dropout operation
        def grad(upstream):
            # If the dropout probability is greater than 0, propagate the upstream gradient only through the unmasked elements
            if self.drop_prob > 0:
                return tf.where(mask, 0.0, upstream) * scale
            else:
                return upstream

        # Return the modified input tensor and the gradient function
        return inputs, grad

    # Overriding the default call method of the layer
    def call(self, inputs: tf.Tensor, training: tf.Tensor = False):
        # If in training mode, apply the custom xdropout function to the inputs
        if training:
            return self.xdropout(inputs)
        return inputs


# Defining a custom layer TFDebertaV2SelfOutput by inheriting from tf.keras.layers.Layer.
class TFDebertaV2SelfOutput(tf.keras.layers.Layer):
    def __init__(self, config: DebertaV2Config, **kwargs):
        super().__init__(**kwargs)
        # Dense layer with hidden_size neurons and "dense" name
        self.dense = tf.keras.layers.Dense(config.hidden_size, name="dense")
        # Layer normalization with epsilon=config.layer_norm_eps and "LayerNorm" name
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # Dropout layer using the TFDebertaV2StableDropout with hidden_dropout_prob as drop_prob and "dropout" name
        self.dropout = TFDebertaV2StableDropout(config.hidden_dropout_prob, name="dropout")
        self.config = config
    # 定义一个call方法，接收hidden_states、input_tensor和training参数，返回更新后的hidden_states
    def call(self, hidden_states, input_tensor, training: bool = False):
        # 对hidden_states进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对hidden_states进行dropout操作
        hidden_states = self.dropout(hidden_states, training=training)
        # 对hidden_states进行残差连接和LayerNorm操作
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回更新后的hidden_states
        return hidden_states

    # 构建方法，用于构建网络层
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 将标志位设为已构建
        self.built = True
        # 如果存在dense层，则构建dense层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 如果存在LayerNorm层，则构建LayerNorm层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])
        # 如果存在dropout层，则构建dropout层
        if getattr(self, "dropout", None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)
# 从transformers.models.deberta.modeling_tf_deberta.TFDebertaAttention中复制代码，并将Deberta更改为DebertaV2
class TFDebertaV2Attention(tf.keras.layers.Layer):
    # 初始化函数，接收DebertaV2Config配置以及其他参数
    def __init__(self, config: DebertaV2Config, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)
        # 创建TFDebertaV2DisentangledSelfAttention对象，并命名为"self"
        self.self = TFDebertaV2DisentangledSelfAttention(config, name="self")
        # 创建TFDebertaV2SelfOutput对象，并命名为"output"
        self.dense_output = TFDebertaV2SelfOutput(config, name="output")
        # 存储配置信息
        self.config = config

    # 调用函数，接收输入张量以及其他参数，返回注意力层的输出
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
        # 调用self.self中的call函数，得到注意力层的输出
        self_outputs = self.self(
            hidden_states=input_tensor,
            attention_mask=attention_mask,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
            output_attentions=output_attentions,
            training=training,
        )
        # 如果query_states为None，则将query_states设置为input_tensor
        if query_states is None:
            query_states = input_tensor
        # 调用self.dense_output中的call函数，计算注意力层的输出
        attention_output = self.dense_output(
            hidden_states=self_outputs[0], input_tensor=query_states, training=training
        )

        # 组合并返回输出结果
        output = (attention_output,) + self_outputs[1:]

        return output

    # 构建函数，用于初始化各层的权重
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在self，则构建self
        if getattr(self, "self", None) is not None:
            with tf.name_scope(self.self.name):
                self.self.build(None)
        # 如果存在dense_output，则构建dense_output
        if getattr(self, "dense_output", None) is not None:
            with tf.name_scope(self.dense_output.name):
                self.dense_output.build(None)


# 从transformers.models.deberta.modeling_tf_deberta.TFDebertaIntermediate中复制代码，并将Deberta更改为DebertaV2
class TFDebertaV2Intermediate(tf.keras.layers.Layer):
    # 初始化函数，接��DebertaV2Config配置以及其他参数
    def __init__(self, config: DebertaV2Config, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 创建Dense层，units为配置中的intermediate_size，kernel_initializer为配置中的initializer_range，命名为"dense"
        self.dense = tf.keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        # 根据配置中的hidden_act类型，选择激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        # 存储配置信息
        self.config = config

    # 调用函数，接收隐藏状态张量作为输入，返回处理后的隐藏状态张量
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 将隐藏状态通过dense层进行线性转换
        hidden_states = self.dense(inputs=hidden_states)
        # 将转换后的隐藏状态通过激活函数进行非线性转换
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

    # 构建函数，用于初始化各层的权
# 定义 TFDebertaV2Output 类，继承自 tf.keras.layers.Layer 类，用于 Deberta 到 DebertaV2 的转换
class TFDebertaV2Output(tf.keras.layers.Layer):
    # 初始化函数，接受 DebertaV2Config 类型的配置参数
    def __init__(self, config: DebertaV2Config, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，输出维度为 hidden_size，初始化方式为配置参数中的初始化方式
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建 LayerNormalization 层，epsilon 为配置参数中的 layer_norm_eps，用于归一化
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建 TFDebertaV2StableDropout 层，概率为配置参数中的 hidden_dropout_prob，用于实现 dropout
        self.dropout = TFDebertaV2StableDropout(config.hidden_dropout_prob, name="dropout")
        # 存储配置参数
        self.config = config

    # 调用函数，接受输入 hidden_states, input_tensor 和 training 标志位，返回处理后的 hidden_states
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 全连接层处理 hidden_states
        hidden_states = self.dense(inputs=hidden_states)
        # dropout 处理 hidden_states
        hidden_states = self.dropout(hidden_states, training=training)
        # LayerNormalization 处理 hidden_states 和 input_tensor 的残差连接
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

    # 构建函数，用于构建层的权重和变量
    def build(self, input_shape=None):
        if self.built:  # 如果已经构建过，直接返回
            return
        self.built = True
        # 构建 dense 层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])
        # 构建 LayerNorm 层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])
        # 构建 dropout 层
        if getattr(self, "dropout", None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)


# 定义 TFDebertaV2Layer 类，继承自 tf.keras.layers.Layer 类，用于 Deberta 到 DebertaV2 的转换
class TFDebertaV2Layer(tf.keras.layers.Layer):
    # 初始化函数，接受 DebertaV2Config 类型的配置参数
    def __init__(self, config: DebertaV2Config, **kwargs):
        super().__init__(**kwargs)

        # 创建 TFDebertaV2Attention 层
        self.attention = TFDebertaV2Attention(config, name="attention")
        # 创建 TFDebertaV2Intermediate 层
        self.intermediate = TFDebertaV2Intermediate(config, name="intermediate")
        # 创建 TFDebertaV2Output 层
        self.bert_output = TFDebertaV2Output(config, name="output")

    # 调用函数，接受输入 hidden_states, attention_mask 和其他参数，返回处理后的 hidden_states
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        query_states: tf.Tensor = None,
        relative_pos: tf.Tensor = None,
        rel_embeddings: tf.Tensor = None,
        output_attentions: bool = False,
        training: bool = False,
    # 定义一个函数，参数为hidden_states, attention_mask, query_states, relative_pos, rel_embeddings, output_attentions, training
    # 调用self.attention函数计算 attention_outputs，包括注意力权重、上下文向量等
    attention_outputs = self.attention(
        input_tensor=hidden_states,
        attention_mask=attention_mask,
        query_states=query_states,
        relative_pos=relative_pos,
        rel_embeddings=rel_embeddings,
        output_attentions=output_attentions,
        training=training,
    )
    # 取出attention_outputs中的第一个元素作为attention_output
    attention_output = attention_outputs[0]
    # 调用self.intermediate函数处理attention_output，得到intermediate_output
    intermediate_output = self.intermediate(hidden_states=attention_output)
    # 调用self.bert_output函数处理intermediate_output和attention_output并得到layer_output
    layer_output = self.bert_output(
        hidden_states=intermediate_output, input_tensor=attention_output, training=training
    )
    # 将layer_output和attention_outputs[1:]组成outputs
    outputs = (layer_output,) + attention_outputs[1:]  # 如果需要输出注意力权重，则添加到outputs中

    # 返回outputs
    return outputs

    # 定义build函数，当未构建时才构建网络
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在self.attention，则构建self.attention
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        # 如果存在self.intermediate，则构建self.intermediate
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        # 如果存在self.bert_output，则构建self.bert_output
        if getattr(self, "bert_output", None) is not None:
            with tf.name_scope(self.bert_output.name):
                self.bert_output.build(None)
class TFDebertaV2ConvLayer(tf.keras.layers.Layer):
    # TFDebertaV2ConvLayer 类的构造函数，接受一个 DebertaV2Config 对象和其他关键字参数
    def __init__(self, config: DebertaV2Config, **kwargs):
        # 调用父类的构造函数
        super().__init__(**kwargs)

        # 从 config 中获取卷积核大小，默认为 3
        self.kernel_size = getattr(config, "conv_kernel_size", 3)
        # 从 config 中获取卷积激活函数，默认为 "tanh"，并将其转换为 TensorFlow 激活函数
        self.conv_act = get_tf_activation(getattr(config, "conv_act", "tanh"))
        # 计算卷积核的填充大小
        self.padding = (self.kernel_size - 1) // 2
        # 创建 LayerNormalization 层，用于归一化输出
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建 Dropout 层，用于防止过拟合
        self.dropout = TFDebertaV2StableDropout(config.hidden_dropout_prob, name="dropout")
        # 将配置对象保存到实例中
        self.config = config

    # 构建层，在第一次调用 call() 方法时执行
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        self.built = True
        # 在 "conv" 命名空间下构建卷积核和偏置项
        with tf.name_scope("conv"):
            # 创建卷积核的权重
            self.conv_kernel = self.add_weight(
                name="kernel",
                shape=[self.kernel_size, self.config.hidden_size, self.config.hidden_size],
                initializer=get_initializer(self.config.initializer_range),
            )
            # 创建卷积层的偏置项
            self.conv_bias = self.add_weight(
                name="bias", shape=[self.config.hidden_size], initializer=tf.zeros_initializer()
            )
        # 如果存在 LayerNorm 层，构建 LayerNorm 层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])
        # 如果存在 dropout 层，构建 dropout 层
        if getattr(self, "dropout", None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)

    # 调用层，执行前向传播
    def call(
        self, hidden_states: tf.Tensor, residual_states: tf.Tensor, input_mask: tf.Tensor, training: bool = False
    ) -> tf.Tensor:
        # 对输入的 hidden_states 进行二维卷积操作
        out = tf.nn.conv2d(
            tf.expand_dims(hidden_states, 1),
            tf.expand_dims(self.conv_kernel, 0),
            strides=1,
            padding=[[0, 0], [0, 0], [self.padding, self.padding], [0, 0]],
        )
        # 添加卷积层的偏置项并去除多余的维度
        out = tf.squeeze(tf.nn.bias_add(out, self.conv_bias), 1)
        # 根据输入掩码屏蔽输出
        rmask = tf.cast(1 - input_mask, tf.bool)
        out = tf.where(tf.broadcast_to(tf.expand_dims(rmask, -1), shape_list(out)), 0.0, out)
        # 应用 dropout 层
        out = self.dropout(out, training=training)
        # 应用激活函数
        out = self.conv_act(out)

        # 计算残差连接
        layer_norm_input = residual_states + out
        # 对残差连接的结果进行 LayerNormalization
        output = self.LayerNorm(layer_norm_input)

        # 如果输入掩码为空，直接返回 output
        if input_mask is None:
            output_states = output
        else:
            # 如果输入掩码的维度不匹配，进行维度调整
            if len(shape_list(input_mask)) != len(shape_list(layer_norm_input)):
                if len(shape_list(input_mask)) == 4:
                    input_mask = tf.squeeze(tf.squeeze(input_mask, axis=1), axis=1)
                input_mask = tf.cast(tf.expand_dims(input_mask, axis=2), tf.float32)

            # 将输出与输入掩码相乘
            output_states = output * input_mask

        # 返回输出
        return output_states


class TFDebertaV2Encoder(tf.keras.layers.Layer):
    # 定义一个类，继承自父类
    def __init__(self, config: DebertaV2Config, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
    
        # 根据隐藏层数创建一个由TFDebertaV2Layer对象组成的列表
        self.layer = [TFDebertaV2Layer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]
        
        # 判断是否使用相对注意力机制
        self.relative_attention = getattr(config, "relative_attention", False)
        
        # 保存传入的配置对象
        self.config = config
        
        # 如果使用相对注意力机制
        if self.relative_attention:
            # 获取相对位置的最大数目，并确保其大于0
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            
            # 获取位置单元的桶数目
            self.position_buckets = getattr(config, "position_buckets", -1)
            self.pos_ebd_size = self.max_relative_positions * 2
    
            # 如果桶数目大于0，则更新位置单元的维度
            if self.position_buckets > 0:
                self.pos_ebd_size = self.position_buckets * 2
    
        # 获取相对位置嵌入的归一化方式列表，并去除空格且全部转为小写
        self.norm_rel_ebd = [x.strip() for x in getattr(config, "norm_rel_ebd", "none").lower().split("|")]
    
        # 如果列表中包含"layer_norm"字符串，则创建一个LayerNormalization对象
        if "layer_norm" in self.norm_rel_ebd:
            self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        
        # 如果conv_kernel_size大于0，则创建一个TFDebertaV2ConvLayer对象
        self.conv = TFDebertaV2ConvLayer(config, name="conv") if getattr(config, "conv_kernel_size", 0) > 0 else None
    
    # 构建网络
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 将构建状态设置为True
        self.built = True
        
        # 如果使用了相对注意力机制，则创建相对位置嵌入
        if self.relative_attention:
            self.rel_embeddings = self.add_weight(
                name="rel_embeddings.weight",
                shape=[self.pos_ebd_size, self.config.hidden_size],
                initializer=get_initializer(self.config.initializer_range),
            )
        
        # 如果存在卷积层，则构建卷积层
        if getattr(self, "conv", None) is not None:
            with tf.name_scope(self.conv.name):
                self.conv.build(None)
        
        # 如果存在LayerNorm，则构建LayerNorm
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, self.config.hidden_size])
        
        # 如果存���层列表，则依次构建每一层
        if getattr(self, "layer", None) is not None:
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    layer.build(None)
    
    # 获取相对位置嵌入
    def get_rel_embedding(self):
        # 如果使用相对注意力机制，则返回相对位置嵌入
        rel_embeddings = self.rel_embeddings if self.relative_attention else None
        
        # 如果相对位置嵌入存在且归一化方式为"layer_norm"，则对相对位置嵌入进行LayerNorm
        if rel_embeddings is not None and ("layer_norm" in self.norm_rel_ebd):
            rel_embeddings = self.LayerNorm(rel_embeddings)
        
        return rel_embeddings
    
    # 获取注意力掩码
    def get_attention_mask(self, attention_mask):
        # 如果attention_mask的维度小于等于2，则在维度1和2上分别添加1，形成维度为[batch_size, 1, 1, seq_length, seq_length]的张量
        if len(shape_list(attention_mask)) <= 2:
            extended_attention_mask = tf.expand_dims(tf.expand_dims(attention_mask, 1), 2)
            attention_mask =
    # 获取相对位置信息，用于相对注意力机制
    def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
        # 如果需要使用相对注意力机制，且没有提供相对位置信息
        if self.relative_attention and relative_pos is None:
            # 获取查询状态的长度，如果提供了查询状态，否则使用隐藏状态的长度
            q = shape_list(query_states)[-2] if query_states is not None else shape_list(hidden_states)[-2]
            # 构建相对位置信息
            relative_pos = build_relative_position(
                q,
                shape_list(hidden_states)[-2],
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions,
            )
        # 返回相对位置信息
        return relative_pos
    
    # 多头注意力层的前向传播
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
        # 如果注意力掩码的维度小于等于2，直接使用注意力掩码
        if len(shape_list(attention_mask)) <= 2:
            input_mask = attention_mask
        # 否则，根据注意力掩码计算输入掩码
        else:
            input_mask = tf.cast(tf.math.reduce_sum(attention_mask, axis=-2) > 0, dtype=tf.uint8)
    
        # 初始化输出隐藏状态和注意力张量
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
    
        # 获取注意力掩码
        attention_mask = self.get_attention_mask(attention_mask)
        # 获取相对位置信息
        relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)
    
        # 更新下一个键值对
        next_kv = hidden_states
    
        # 获取相对位置嵌入
        rel_embeddings = self.get_rel_embedding()
        # 初始化输出状态
        output_states = next_kv
        # 遍历注意力层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，添加当前状态
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (output_states,)
    
            # 计算当前层的注意力输出
            layer_outputs = layer_module(
                hidden_states=next_kv,
                attention_mask=attention_mask,
                query_states=query_states,
                relative_pos=relative_pos,
                rel_embeddings=rel_embeddings,
                output_attentions=output_attentions,
                training=training,
            )
            # 更新输出状态
            output_states = layer_outputs[0]
    
            # 如果是第一层，且存在卷积层，应用卷积操作
            if i == 0 and self.conv is not None:
                output_states = self.conv(hidden_states, output_states, input_mask)
    
            # 更新下一个键值对
            next_kv = output_states
    
            # 如果需要输出注意力，添加当前注意力
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
    
        # 添加最后一层
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (output_states,)
    
        # 根据返回方式返回输出
        if not return_dict:
            return tuple(v for v in [output_states, all_hidden_states, all_attentions] if v is not None)
    
        return TFBaseModelOutput(
            last_hidden_state=output_states, hidden_states=all_hidden_states, attentions=all_attentions
        )
# 根据相对位置、桶的大小和最大位置计算位置索引
def make_log_bucket_position(relative_pos, bucket_size, max_position):
    # 计算相对位置的符号
    sign = tf.math.sign(relative_pos)
    # 计算绝对位置
    mid = bucket_size // 2
    abs_pos = tf.where((relative_pos < mid) & (relative_pos > -mid), mid - 1, tf.math.abs(relative_pos))
    # 计算对数位置
    log_pos = (
        tf.math.ceil(
            tf.cast(tf.math.log(abs_pos / mid), tf.float32) / tf.math.log((max_position - 1) / mid) * (mid - 1)
        )
        + mid
    )
    # 计算桶位置
    bucket_pos = tf.cast(
        tf.where(abs_pos <= mid, tf.cast(relative_pos, tf.float32), log_pos * tf.cast(sign, tf.float32)), tf.int32
    )
    return bucket_pos


# 构建与查询和键相关的相对位置
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
    # 计算相对位置索引
    rel_pos_ids = q_ids[:, None] - tf.tile(tf.expand_dims(k_ids, axis=0), [shape_list(q_ids)[0], 1])
    if bucket_size > 0 and max_position > 0:
        rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position)
    rel_pos_ids = rel_pos_ids[:query_size, :]
    rel_pos_ids = tf.expand_dims(rel_pos_ids, axis=0)
    return tf.cast(rel_pos_ids, tf.int64)


# 动态扩展C2P的位置输入
def c2p_dynamic_expand(c2p_pos, query_layer, relative_pos):
    shapes = [
        shape_list(query_layer)[0],
        shape_list(query_layer)[1],
        shape_list(query_layer)[2],
        shape_list(relative_pos)[-1],
    ]
    return tf.broadcast_to(c2p_pos, shapes)


# 动态扩展P2C的位置输入
def p2c_dynamic_expand(c2p_pos, query_layer, key_layer):
    shapes = [
        shape_list(query_layer)[0],
        shape_list(query_layer)[1],
        shape_list(key_layer)[-2],
        shape_list(key_layer)[-2],
    ]
    return tf.broadcast_to(c2p_pos, shapes)


# 动态扩展位置索引
def pos_dynamic_expand(pos_index, p2c_att, key_layer):
    shapes = shape_list(p2c_att)[:2] + [shape_list(pos_index)[-2], shape_list(key_layer)[-2]]
    return tf.broadcast_to(pos_index, shapes)


# 沿着轴进行索引选取
def take_along_axis(x, indices):
    # 只有在gather轴为-1时才有效
    # TPU + gather和reshape不兼容 -- 参考 https://github.com/huggingface/transformers/issues/18239
    # 检查当前环境是否是 TPUStrategy，如果是则执行下面的代码块
    if isinstance(tf.distribute.get_strategy(), tf.distribute.TPUStrategy):
        # 将输入的索引转换成 one hot 编码，维度为 [B, S, P, D]
        one_hot_indices = tf.one_hot(indices, depth=x.shape[-1], dtype=x.dtype)

        # 如果忽略前两个维度，这相当于将一个矩阵（one hot 编码）和一个向量（x）相乘
        # 滥用符号表示：[B, S, P, D] . [B, S, D] = [B, S, P]
        # 使用 einsum 函数进行张量乘法
        gathered = tf.einsum("ijkl,ijl->ijk", one_hot_indices, x)

    # 如果不是 TPUStrategy 则表示是 GPU，更倾向于使用 gather 而不是大规模的 one-hot+matmul
    else:
        # 使用 gather 函数根据索引收集张量 x，指定 batch_dims=2，表示在第三个维度上进行收集
        gathered = tf.gather(x, indices, batch_dims=2)

    # 返回收集结果
    return gathered
class TFDebertaV2DisentangledSelfAttention(tf.keras.layers.Layer):
    """
    Disentangled self-attention module

    Parameters:
        config (`DebertaV2Config`):
            A model config class instance with the configuration to build a new model. The schema is similar to
            *BertConfig*, for more details, please refer [`DebertaV2Config`]

    """

    def transpose_for_scores(self, tensor: tf.Tensor, attention_heads: int) -> tf.Tensor:
        # 获取 tensor 的形状
        tensor_shape = shape_list(tensor)
        # 如果第一个维度（batch 大小）为 None，无法使用 -1 来进行 reshape
        shape = tensor_shape[:-1] + [attention_heads, tensor_shape[-1] // attention_heads]
        # 把形状从 [batch_size, seq_length, all_head_size] 转换为 [batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(tensor=tensor, shape=shape)
        # 转置 tensor 使得在 [batch_size, num_attention_heads, seq_length, attention_head_size] 的形状上进行操作
        tensor = tf.transpose(tensor, perm=[0, 2, 1, 3])
        # 重新 reshape tensor 为 [batch_size * num_attention_heads, seq_length, attention_head_size]
        x_shape = shape_list(tensor)
        tensor = tf.reshape(tensor, shape=[-1, x_shape[-2], x_shape[-1]])
        # 返回结果 tensor
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
    def build(self, input_shape=None):
        # 如果已经构建过 layer，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 query_proj 属性，就进行构建
        if getattr(self, "query_proj", None) is not None:
            with tf.name_scope(self.query_proj.name):
                self.query_proj.build([None, None, self.config.hidden_size])
        # 如果存在 key_proj 属性，就进行构建
        if getattr(self, "key_proj", None) is not None:
            with tf.name_scope(self.key_proj.name):
                self.key_proj.build([None, None, self.config.hidden_size])
        # 如果存在 value_proj 属性，就进行构建
        if getattr(self, "value_proj", None) is not None:
            with tf.name_scope(self.value_proj.name):
                self.value_proj.build([None, None, self.config.hidden_size])
        # 如果存在 dropout 属性，就进行构建
        if getattr(self, "dropout", None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)
        # 如果存在 pos_dropout 属性，就进行构建
        if getattr(self, "pos_dropout", None) is not None:
            with tf.name_scope(self.pos_dropout.name):
                self.pos_dropout.build(None)
        # 如果存在 pos_key_proj 属性，就进行构建
        if getattr(self, "pos_key_proj", None) is not None:
            with tf.name_scope(self.pos_key_proj.name):
                self.pos_key_proj.build([None, None, self.config.hidden_size])
        # 如果存在 pos_query_proj 属性，就进行构建
        if getattr(self, "pos_query_proj", None) is not None:
            with tf.name_scope(self.pos_query_proj.name):
                self.pos_query_proj.build([None, None, self.config.hidden_size])


# 从 transformers.models.deberta.modeling_tf_deberta.TFDebertaEmbeddings 复制过来的 Deberta->DebertaV2 嵌入层
class TFDebertaV2Embeddings(tf.keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""
    # 自定义类的初始化方法，接收配置和可变参数
    def __init__(self, config, **kwargs):
        # 调用父类的初始化方法，传递可变参数
        super().__init__(**kwargs)
    
        # 保存配置对象
        self.config = config
        # 获取嵌入大小，默认为隐藏层大小
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)
        # 获取隐藏层大小
        self.hidden_size = config.hidden_size
        # 获取最大位置嵌入数
        self.max_position_embeddings = config.max_position_embeddings
        # 是否启用位置偏置输入
        self.position_biased_input = getattr(config, "position_biased_input", True)
        # 初始化范围
        self.initializer_range = config.initializer_range
        
        # 如果嵌入大小与隐藏层大小不同，定义投影层
        if self.embedding_size != config.hidden_size:
            self.embed_proj = tf.keras.layers.Dense(
                config.hidden_size,
                kernel_initializer=get_initializer(config.initializer_range),
                name="embed_proj",
                use_bias=False,
            )
        # 定义层归一化对象
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 定义丢弃层
        self.dropout = TFDebertaV2StableDropout(config.hidden_dropout_prob, name="dropout")
    
    # 构建模型的方法，接收输入形状
    def build(self, input_shape=None):
        # 为词嵌入定义变量
        with tf.name_scope("word_embeddings"):
            self.weight = self.add_weight(
                name="weight",
                shape=[self.config.vocab_size, self.embedding_size],
                initializer=get_initializer(self.initializer_range),
            )
    
        # 为标记类型嵌入定义变量
        with tf.name_scope("token_type_embeddings"):
            if self.config.type_vocab_size > 0:
                self.token_type_embeddings = self.add_weight(
                    name="embeddings",
                    shape=[self.config.type_vocab_size, self.embedding_size],
                    initializer=get_initializer(self.initializer_range),
                )
            else:
                self.token_type_embeddings = None
    
        # 为位置嵌入定义变量
        with tf.name_scope("position_embeddings"):
            if self.position_biased_input:
                self.position_embeddings = self.add_weight(
                    name="embeddings",
                    shape=[self.max_position_embeddings, self.hidden_size],
                    initializer=get_initializer(self.initializer_range),
                )
            else:
                self.position_embeddings = None
    
        # 确保只构建一次
        if self.built:
            return
        self.built = True
    
        # 如果定义了层归一化，调用其构建方法
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])
        
        # 如果定义了丢弃层，调用其构建方法
        if getattr(self, "dropout", None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)
        
        # 如果定义了嵌入投影，调用其构建方法
        if getattr(self, "embed_proj", None) is not None:
            with tf.name_scope(self.embed_proj.name):
                self.embed_proj.build([None, None, self.embedding_size])
    
    # 模型的前向传递方法，接受各种输入参数
    def call(
        self,
        input_ids: tf.Tensor = None,
        position_ids: tf.Tensor = None,
        token_type_ids: tf.Tensor = None,
        inputs_embeds: tf.Tensor = None,
        mask: tf.Tensor = None,
        training: bool = False,
    # 定义函数apply_embeddings，输入为input_ids（输入的标记ID列表）、inputs_embeds（输入的嵌入张量）、
    # token_type_ids（标记类型ID列表）、position_ids（位置ID列表）、mask（遮盖张量）和training（表明是否在训练模式下）
    # 输出为嵌入张量final_embeddings
    
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
            # 如果既未提供input_ids，也未提供inputs_embeds，则抛出ValueError异常
            if input_ids is None and inputs_embeds is None:
                raise ValueError("Need to provide either `input_ids` or `input_embeds`.")
    
            # 如果提供了input_ids，则检查输入的嵌入ID是否在合法范围内，如果不合法则抛出异常
            if input_ids is not None:
                check_embeddings_within_bounds(input_ids, self.config.vocab_size)
                # 从self.weight中根据输入的ID抽取对应的嵌入向量，并将结果赋值给inputs_embeds
                inputs_embeds = tf.gather(params=self.weight, indices=input_ids)
    
            # 获取inputs_embeds的形状
            input_shape = shape_list(inputs_embeds)[:-1]
    
            # 如果token_type_ids未提供，则创建一个维度与input_shape相同的张量，其中所有元素的值为0
            if token_type_ids is None:
                token_type_ids = tf.fill(dims=input_shape, value=0)
    
            # 如果position_ids未提供，则根据input_shape的最后一个维度创建一个张量，
            # 张量的元素逐一递增，最小值为0，最大值为input_shape的最后一个维度值
            if position_ids is None:
                position_ids = tf.expand_dims(tf.range(start=0, limit=input_shape[-1]), axis=0)
    
            # 将inputs_embeds赋值给final_embeddings
            final_embeddings = inputs_embeds
            # 如果位置偏置输入（position_biased_input）为真
            if self.position_biased_input:
                # 从self.position_embeddings中根据position_ids抽取对应的位置嵌入向量，并将结果赋值给position_embeds
                position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
                # 将final_embeddings与position_embeds相加，得到新的final_embeddings
                final_embeddings += position_embeds
            # 如果配置的类型词汇量（type_vocab_size）大于0
            if self.config.type_vocab_size > 0:
                # 从self.token_type_embeddings中根据token_type_ids抽取对应的类型嵌入向量，并将结果赋值给token_type_embeds
                token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)
                # 将final_embeddings与token_type_embeds相加，得到新的final_embeddings
                final_embeddings += token_type_embeds
    
            # 如果embedding_size不等于hidden_size，则使用embed_proj对final_embeddings进行投影
            if self.embedding_size != self.hidden_size:
                final_embeddings = self.embed_proj(final_embeddings)
    
            # 对final_embeddings进行LayerNorm
            final_embeddings = self.LayerNorm(final_embeddings)
    
            # 如果mask不为None
            if mask is not None:
                # 如果mask的形状的维度数与final_embeddings的形状的维度数不同
                if len(shape_list(mask)) != len(shape_list(final_embeddings)):
                    # 如果mask的维度数为4，则先去除两个多余的维度
                    if len(shape_list(mask)) == 4:
                        mask = tf.squeeze(tf.squeeze(mask, axis=1), axis=1)
                    # 将mask扩展一维，并将数据类型转换为float32
                    mask = tf.cast(tf.expand_dims(mask, axis=2), tf.float32)
    
                # 将final_embeddings与mask相乘，得到新的final_embeddings
                final_embeddings = final_embeddings * mask
    
            # 根据训练模式，使用dropout函数对final_embeddings进行丢弃操作
            final_embeddings = self.dropout(final_embeddings, training=training)
    
            # 返回最终的嵌入张量final_embeddings
            return final_embeddings
# TFDebertaV2PredictionHeadTransform 类实现了 DebertaV2 模型的预测头变换层
# 该层用于将隐藏状态转换为预测的词汇表分布
class TFDebertaV2PredictionHeadTransform(tf.keras.layers.Layer):
    def __init__(self, config: DebertaV2Config, **kwargs):
        super().__init__(**kwargs)

        # 获取嵌入大小，默认为隐藏状态的大小
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)

        # 创建一个全连接层，将隐藏状态映射到嵌入大小
        self.dense = tf.keras.layers.Dense(
            units=self.embedding_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense",
        )

        # 获取激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.transform_act_fn = config.hidden_act

        # 添加一个层归一化层
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 首先将隐藏状态通过全连接层
        hidden_states = self.dense(inputs=hidden_states)
        # 然后应用激活函数
        hidden_states = self.transform_act_fn(hidden_states)
        # 最后进行层归一化
        hidden_states = self.LayerNorm(hidden_states)

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
                self.LayerNorm.build([None, None, self.embedding_size])

# TFDebertaV2LMPredictionHead 类实现了 DebertaV2 模型的语言模型预测头
class TFDebertaV2LMPredictionHead(tf.keras.layers.Layer):
    def __init__(self, config: DebertaV2Config, input_embeddings: tf.keras.layers.Layer, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)

        # 创建一个 TFDebertaV2PredictionHeadTransform 层
        self.transform = TFDebertaV2PredictionHeadTransform(config, name="transform")

        # 使用输入词嵌入层作为输出词嵌入层，并添加一个偏置项
        self.input_embeddings = input_embeddings

    def build(self, input_shape=None):
        # 添加一个可训练的偏置项
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="bias")

        if self.built:
            return
        self.built = True
        if getattr(self, "transform", None) is not None:
            with tf.name_scope(self.transform.name):
                self.transform.build(None)

    def get_output_embeddings(self) -> tf.keras.layers.Layer:
        return self.input_embeddings

    def set_output_embeddings(self, value: tf.Variable):
        self.input_embeddings.weight = value
        self.input_embeddings.vocab_size = shape_list(value)[0]
    def get_bias(self) -> Dict[str, tf.Variable]:
        返回包含bias的字典
        return {"bias": self.bias}

    def set_bias(self, value: tf.Variable):
        设置bias的值为参数中的value字典中的"bias"对应的值
        self.bias = value["bias"]
        设置config中的vocab_size为bias值的shape的第一个维度大小
        self.config.vocab_size = shape_list(value["bias"])[0]

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        使用transform方法对hidden_states进行处理
        hidden_states = self.transform(hidden_states=hidden_states)
        获取hidden_states的第二个维度的大小
        seq_length = shape_list(hidden_states)[1]
        将hidden_states reshape为[-1, self.embedding_size]的形状
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.embedding_size])
        计算hidden_states与self.input_embeddings.weight的矩阵相乘(转置b矩阵)
        hidden_states = tf.matmul(a=hidden_states, b=self.input_embeddings.weight, transpose_b=True)
        将结果reshape为[-1, seq_length, self.config.vocab_size]的形状
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])
        将bias添加到hidden_states中
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)

        返回处理后的hidden_states
        return hidden_states
# 从 transformers.models.deberta.modeling_tf_deberta.TFDebertaOnlyMLMHead 复制而来，修改 Deberta 为 DebertaV2
class TFDebertaV2OnlyMLMHead(tf.keras.layers.Layer):
    def __init__(self, config: DebertaV2Config, input_embeddings: tf.keras.layers.Layer, **kwargs):
        super().__init__(**kwargs)
        # 初始化 MLM 头部预测层
        self.predictions = TFDebertaV2LMPredictionHead(config, input_embeddings, name="predictions")

    def call(self, sequence_output: tf.Tensor) -> tf.Tensor:
        # 通过预测层计算预测分数
        prediction_scores = self.predictions(hidden_states=sequence_output)

        return prediction_scores

    def build(self, input_shape=None):
        # 如果已经构建，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在预测层，构建预测层
        if getattr(self, "predictions", None) is not None:
            with tf.name_scope(self.predictions.name):
                self.predictions.build(None)


# 从 transformers.models.deberta.modeling_tf_deberta.TFDebertaMainLayer 复制而来，修改 Deberta 为 DebertaV2
class TFDebertaV2MainLayer(tf.keras.layers.Layer):
    config_class = DebertaV2Config

    def __init__(self, config: DebertaV2Config, **kwargs):
        super().__init__(**kwargs)

        self.config = config

        # 初始化 DebertaV2 的嵌入层
        self.embeddings = TFDebertaV2Embeddings(config, name="embeddings")
        # 初始化 DebertaV2 的编码器
        self.encoder = TFDebertaV2Encoder(config, name="encoder")

    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        return self.embeddings

    def set_input_embeddings(self, value: tf.Variable):
        # 设置输入嵌入权重
        self.embeddings.weight = value
        # 更新嵌入层的词汇大小
        self.embeddings.vocab_size = shape_list(value)[0]

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
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
    # 定义一个方法，它接受输入参数并返回TFBaseModelOutput或tf.Tensor的元组
    def call(
        self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, inputs_embeds=None,
        output_attentions=None, output_hidden_states=None, return_dict=None, training=None,
    ):
        # 如果input_ids和inputs_embeds都不为空，则触发异常
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        # 如果input_ids不为空，则获取shape并赋值给input_shape
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        # 如果inputs_embeds不为空，则获取shape的除最后一个元素之外的所有元素，并赋值给input_shape
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        # 否则触发异常，要求必须指定input_ids或inputs_embeds
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # 如果attention_mask为空，则使用填充值1填充为input_shape的形状
        if attention_mask is None:
            attention_mask = tf.fill(dims=input_shape, value=1)

        # 如果token_type_ids为空，则使用填充值0填充为input_shape的形状
        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)

        # 使用embeddings方法构建嵌入输出
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            mask=attention_mask,
            training=training,
        )

        # 使用encoder方法构建编码器输出
        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 获取编码器输出的第一个元素作为序列输出
        sequence_output = encoder_outputs[0]

        # 如果不返回字典，则返回序列输出和编码器输出的剩余部分
        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        # 返回TFBaseModelOutput对象，包括最后的隐藏状态、隐藏状态和注意力
        return TFBaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    # 构建方法
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记构建已完成
        self.built = True
        # 如果embeddings存在，则构建embeddings
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        # 如果encoder存在，则构建encoder
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
# 从transformers.models.deberta.modeling_tf_deberta.TFDebertaPreTrainedModel复制而来，将Deberta->DebertaV2
class TFDebertaV2PreTrainedModel(TFPreTrainedModel):
    """
    一个用于处理权重初始化以及下载和加载预训练模型的抽象类。
    """
    # 配置类为DebertaV2Config
    config_class = DebertaV2Config
    # 基础模型前缀为"deberta"


# DeBERTa模型在[DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654)中提出
# 由Pengcheng He，Xiaodong Liu，Jianfeng Gao，Weizhu Chen提出。它在BERT/RoBERTa基础上进行了两项改进，即解耦的注意力和改进的遮罩解码器。通过这两项改进，在80GB的预训练数据上，它超越了BERT/RoBERTa在大多数任务上的表现。

# 该模型也是一个[tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model)子类。使用它就像使用常规的TF 2.0 Keras Model，并参考TF 2.0文档，了解与通用用法和行为相关的所有事项。

# <Tip>

# `transformers`中的TensorFlow模型和层接受两种输入格式：
# - 将所有输入作为关键字参数（与PyTorch模型类似），或者
# - 将所有输入作为列表、元组或字典的第一个位置参数。

# 支持第二种格式的原因是，Keras方法在将输入传递给模型和层时更喜欢这种格式。由于这种支持，在使用`model.fit()`等方法时，只需要以`model.fit()`支持的任何格式传递输入和标签即可“正常工作”！但是，如果您希望在Keras方法之外使用第二种格式，例如在使用Keras`Functional` API创建自己的图层或模型时，您可以使用三种可能性来收集第一个位置参数中的所有输入张量:
# - 只有一个仅包含`input_ids`的张量：`model(input_ids)`
# - 长度不等的列表，其中包含在文档字符串中给出的顺序中的一个或多个输入张量：
# `model([input_ids, attention_mask])`或`model([input_ids, attention_mask, token_type_ids])`
# - 包含与文档字符串中给出的输入名称相关联的一个或多个输入张量的字典:
# `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

# 请注意，当使用[subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/)创建模型和层时，您无需担心任何这些，因为您可以像对待任何其他Python函数一样传递输入！

# </Tip>
    # 参数说明：
    # config ([`DebertaV2Config`]): 使用包含模型所有参数的模型配置类。
    # 用配置文件初始化不会加载与模型关联的权重，只会加载配置。查看 [`~PreTrainedModel.from_pretrained`] 方法来加载模型权重。
"""
文档字符串，用于解释DeBERTa模型的输入参数及其含义
DEBERTA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` ``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `({0})`):
            Indices of input sequence tokens in the vocabulary.
            # 输入序列token在词汇表中的索引
            # 可以使用AutoTokenizer获取。参见PreTrainedTokenizer.encode和PreTrainedTokenizer.__call__的详细说明
            # 输入ID是什么？
        attention_mask (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            # 避免在padding token索引上执行注意力的掩码。选择的掩码值在[0, 1]范围内
            # 1表示未被掩码的token，0表示被掩码的token
        token_type_ids (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
            # 分段令牌索引，表示输入的第一部分和第二部分。索引在[0, 1]中选择
            # 0对应于*sentence A* 令牌，1对应于*sentence B* 令牌
        position_ids (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
            # 输入序列token在位置嵌入中的位置索引。在范围[0, config.max_position_embeddings - 1]中选择
            # 什么是位置ID？
        inputs_embeds (`np.ndarray` or `tf.Tensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert *input_ids* indices into associated vectors than the
            model's internal embedding lookup matrix.
            # 可选的，可以直接传递嵌入表示而不是传递`input_ids`。如果您想对如何将*input_ids*索引转换为关联向量有更多控制权，这很有用
            # 输出注意返回每个注意力层的注意力张量
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
            # 是否返回所有注意层的注意力张量
            # 输出注意层（returned tensors）中有关更多详细信息
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
            # 是否返回所有层的隐藏状态
            # 输出隐藏状态（returned tensors）中有关更多详细信息
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput``] instead of a plain tuple.
            # 是否返回`~utils.ModelOutput``而不是普通元组
"""


@add_start_docstrings(
    "The bare DeBERTa Model transformer outputting raw hidden-states without any specific head on top.",
    DEBERTA_START_DOCSTRING,
)
# 通过将Deberta->DebertaV2，创建了TFDebertaV2Model类，继承自TFDebertaV2PreTrainedModel
class TFDebertaV2Model(TFDebertaV2PreTrainedModel):
    def __init__(self, config: DebertaV2Config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 初始化TFDebertaV2Model类的对象，其中包含Deberta的主要层
        self.deberta = TFDebert
    # 给模型前向调用方法添加文档字符串和示例代码
    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义模型前向调用方法，包括输入参数和输出类型
    def call(
        self,
        input_ids: TFModelInputType | None = None,  # 输入的词 ID 列表
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 注意力掩码，用于遮挡某些位置
        token_type_ids: np.ndarray | tf.Tensor | None = None,  # 分隔标记，用于区分句子
        position_ids: np.ndarray | tf.Tensor | None = None,  # 位置 ID，用于位置编码
        inputs_embeds: np.ndarray | tf.Tensor | None = None,  # 输入嵌入，如果有提供
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 返回字典还是元组
        training: Optional[bool] = False,  # 指定训练模式还是推理模式
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:  # 输出结果的类型
        # 使用参数调用 DeBERTa 模型的前向传播
        outputs = self.deberta(
            input_ids=input_ids,  # 输入的词 ID 列表
            attention_mask=attention_mask,  # 注意力掩码
            token_type_ids=token_type_ids,  # 分隔标记
            position_ids=position_ids,  # 位置 ID
            inputs_embeds=inputs_embeds,  # 输入嵌入
            output_attentions=output_attentions,  # 是否输出注意力
            output_hidden_states=output_hidden_states,  # 是否输出隐藏状态
            return_dict=return_dict,  # 返回结果的格式
            training=training,  # 是否处于训练模式
        )
    
        # 返回 DeBERTa 模型的输出
        return outputs
    
    # 构建模型的辅助方法
    def build(self, input_shape=None):
        # 如果模型已构建，则返回
        if self.built:
            return
        # 标记模型已构建
        self.built = True
        # 如果 DeBERTa 模型属性存在
        if getattr(self, "deberta", None) is not None:
            # 使用 DeBERTa 模型的名字创建命名空间
            with tf.name_scope(self.deberta.name):
                # 调用 DeBERTa 模型的构建方法
                self.deberta.build(None)
# 给 TFDebertaV2ForMaskedLM 类添加文档字符串，说明该类是在 DeBERTa 模型基础上增加了语言建模头部
# 从 transformers.models.deberta.modeling_tf_deberta.TFDebertaForMaskedLM 复制而来，将 Deberta->DebertaV2
class TFDebertaV2ForMaskedLM(TFDebertaV2PreTrainedModel, TFMaskedLanguageModelingLoss):
    def __init__(self, config: DebertaV2Config, *inputs, **kwargs):
        # 调用父类的初始化函数
        super().__init__(config, *inputs, **kwargs)

        # 如果模型是解码器，发出警告信息
        if config.is_decoder:
            logger.warning(
                "If you want to use `TFDebertaV2ForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # 创建 Deberta 主层对象
        self.deberta = TFDebertaV2MainLayer(config, name="deberta")
        # 创建仅包含 MLM 头部的 Deberta V2 模型
        self.mlm = TFDebertaV2OnlyMLMHead(config, input_embeddings=self.deberta.embeddings, name="cls")

    # 获取语言模型头部
    def get_lm_head(self) -> tf.keras.layers.Layer:
        return self.mlm.predictions

    # 封装输入数据，添加模型前向传播的文档字符串，添加代码示例的文档字符串
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
    ) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional`):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        # 调用self.deberta()方法进行模型计算
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
        # 获取模型计算后的输出的第一个元素
        sequence_output = outputs[0]
        # 调用self.mlm()方法进行预测分数计算
        prediction_scores = self.mlm(sequence_output=sequence_output, training=training)
        # 如果labels为None，则损失为None，否则调用hf_compute_loss()计算损失
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=prediction_scores)

        # 如果return_dict为False，则返回输出元组
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

    def build(self, input_shape=None):
        # 如果已经构建，则直接返回
        if self.built:
            return
        self.built = True
        # 如果self.deberta为None，则构建self.deberta模型
        if getattr(self, "deberta", None) is not None:
            with tf.name_scope(self.deberta.name):
                self.deberta.build(None)
        # 如果self.mlm为None，则构建self.mlm模型
        if getattr(self, "mlm", None) is not None:
            with tf.name_scope(self.mlm.name):
                self.mlm.build(None)
# 导入必要的模块，并从 "transformers.models.deberta.modeling_tf_deberta" 中导入相关类和函数
@add_start_docstrings(
    """
    DeBERTa Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    DEBERTA_START_DOCSTRING,
)
# 使用 DebertaV2PreTrainedModel 和 TFSequenceClassificationLoss 类创建 TFDebertaV2ForSequenceClassification 类
# 这个类用于序列分类或回归任务，例如 GLUE 任务
class TFDebertaV2ForSequenceClassification(TFDebertaV2PreTrainedModel, TFSequenceClassificationLoss):
    # 初始化方法，接受一个 DebertaV2Config 类的实例作为参数
    def __init__(self, config: DebertaV2Config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 获取标签的数量
        self.num_labels = config.num_labels

        # 创建 Deberta 主层对象，用于处理输入序列
        self.deberta = TFDebertaV2MainLayer(config, name="deberta")
        # 创建上下文池化层对象，用于提取序列的池化表示
        self.pooler = TFDebertaV2ContextPooler(config, name="pooler")

        # 获取配置中的分类器的 dropout 参数，如果没有指定，则使用隐藏层的 dropout 参数
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        # 创建稳定的 dropout 层对象，用于在分类器中应用 dropout
        self.dropout = TFDebertaV2StableDropout(drop_out, name="cls_dropout")
        # 创建全连接层，用于进行分类
        self.classifier = tf.keras.layers.Dense(
            units=config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="classifier",
        )
        # 获取池化层的输出维度
        self.output_dim = self.pooler.output_dim

    # call 方法，用于模型的前向传播
    @unpack_inputs
    # 将文档字符串添加到模型的前向传播方法中，说明输入参数的含义
    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 添加代码示例的文档字符串，说明该方法的调用方式和输出类型
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义方法参数，包括输入序列、注意力掩码、token 类型编码、位置编码、输入嵌入、是否输出注意力权重、是否输出隐藏状态、是否返回字典格式结果、标签、训练模式等
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
    ) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If`
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 调用 DeBERTa 模型进行推理
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
        # 使用池化层处理序列输出
        pooled_output = self.pooler(sequence_output, training=training)
        # 对处理后的输出进行 dropout 操作
        pooled_output = self.dropout(pooled_output, training=training)
        # 通过分类器获取最终的 logits
        logits = self.classifier(pooled_output)
        # 如果存在标签，则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        if not return_dict:
            # 如果不返回 dict，则将 logits 和其它输出合并返回
            output = (logits,) + outputs[1:]

            return ((loss,) + output) if loss is not None else output

        # 返回 Sequence 分类器的输出对象
        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 构建模型
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "deberta", None) is not None:
            with tf.name_scope(self.deberta.name):
                # 构建 DeBERTa 模型
                self.deberta.build(None)
        if getattr(self, "pooler", None) is not None:
            with tf.name_scope(self.pooler.name):
                # 构建池化层
                self.pooler.build(None)
        if getattr(self, "dropout", None) is not None:
            with tf.name_scope(self.dropout.name):
                # 构建 dropout 层
                self.dropout.build(None)
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                # 构建分类器层
                self.classifier.build([None, None, self.output_dim])
# 为 DeBERTa 模型添加一个在隐藏状态之上的标记分类头部（一个线性层），用于命名实体识别（NER）等任务
@add_start_docstrings(
    """
    DeBERTa Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    DEBERTA_START_DOCSTRING,
)
# 从transformers.models.deberta.modeling_tf_deberta.TFDebertaForTokenClassification 复制而来，将 Deberta 改为 DebertaV2
class TFDebertaV2ForTokenClassification(TFDebertaV2PreTrainedModel, TFTokenClassificationLoss):
    def __init__(self, config: DebertaV2Config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 初始化模型的标签数量
        self.num_labels = config.num_labels

        # 初始化 DeBERTa 主层
        self.deberta = TFDebertaV2MainLayer(config, name="deberta")
        # 添加 dropout 层
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        # 添加分类器层
        self.classifier = tf.keras.layers.Dense(
            units=config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        self.config = config

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
    def call(
        self,
        input_ids: tf.Tensor,
        attention_mask: tf.Tensor = None,
        token_type_ids: tf.Tensor = None,
        position_ids: tf.Tensor = None,
        inputs_embeds: tf.Tensor = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = None,
        labels: Optional[tf.Tensor] = None
    ) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the token classification loss. Indices should be in `[0,...,config.num_labels - 1]`.
        """
    
        # 使用Deberta模型进行前向传播
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
    
        # 获取模型输出中的序列输出
        sequence_output = outputs[0]
    
        # 使用dropout进行序列输出的正则化
        sequence_output = self.dropout(sequence_output, training=training)
    
        # 将序列输出输入到分类器中，得到分类结果的logits
        logits = self.classifier(inputs=sequence_output)
    
        # 如果提供了标签，计算分类损失
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)
    
        # 如果return_dict为False，则返回元组(logit, outputs[1:])
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
    
        # 如果return_dict为True，则返回TFTokenClassifierOutput对象
        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
    
        # 构建Deberta模型
        if getattr(self, "deberta", None) is not None:
            with tf.name_scope(self.deberta.name):
                self.deberta.build(None)
    
        # 构建分类器模型
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
# 为 DeBERTa 模型添加一个在顶部用于抽取式问答任务（如 SQuAD）的跨度分类头部
# 这个头部包括一些线性层，用于计算 `span start logits` 和 `span end logits`
@add_start_docstrings(
    """
    DeBERTa Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    DEBERTA_START_DOCSTRING,
)
# 从 transformers.models.deberta.modeling_tf_deberta.TFDebertaForQuestionAnswering 复制而来，将 Deberta->DebertaV2
class TFDebertaV2ForQuestionAnswering(TFDebertaV2PreTrainedModel, TFQuestionAnsweringLoss):
    # 初始化函数，接受一个 DebertaV2Config 对象和其他输入参数
    def __init__(self, config: DebertaV2Config, *inputs, **kwargs):
        # 调用父类的初始化函数
        super().__init__(config, *inputs, **kwargs)

        # 将类别数目设置为配置中的类别数
        self.num_labels = config.num_labels

        # 初始化 Deberta 主层
        self.deberta = TFDebertaV2MainLayer(config, name="deberta")
        # 初始化问题-答案输出层
        self.qa_outputs = tf.keras.layers.Dense(
            units=config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )
        # 保存配置
        self.config = config

    # 定义前向传播函数
    @unpack_inputs
    # 将输入的文档字符串添加到模型前向传播函数的文档字符串中
    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 将代码示例的文档字符串添加到模型前向传播函数的文档字符串中
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
        ) -> Union[TFQuestionAnsweringModelOutput, Tuple[tf.Tensor]]:
        r"""
        start_positions (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        # 调用deberta模型，输入参数包括input_ids等，返回output结果
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
        # 从output结果中取出sequence_output
        sequence_output = outputs[0]
        # 使用sequence_output作为输入获取模型的logits
        logits = self.qa_outputs(inputs=sequence_output)
        # 对logits进行分割，得到start_logits和end_logits
        start_logits, end_logits = tf.split(value=logits, num_or_size_splits=2, axis=-1)
        # 对start_logits和end_logits进行压缩，去除维度为1的维度
        start_logits = tf.squeeze(input=start_logits, axis=-1)
        end_logits = tf.squeeze(input=end_logits, axis=-1)
        loss = None

        # 如果给定了start_positions和end_positions，计算loss
        if start_positions is not None and end_positions is not None:
            labels = {"start_position": start_positions}
            labels["end_position"] = end_positions
            loss = self.hf_compute_loss(labels=labels, logits=(start_logits, end_logits))

        # 如果return_dict为False，返回output结果
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        # 如果return_dict为True，返回TFQuestionAnsweringModelOutput类型的结果
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
        # 如果模型已经构建，则直接返回
        if getattr(self, "deberta", None) is not None:
            with tf.name_scope(self.deberta.name):
                # 构建deberta模型
                self.deberta.build(None)
        # 如果模型已经构建，则直接返回
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                # 构建qa_outputs模型
                self.qa_outputs.build([None, None, self.config.hidden_size])
# 添加文档字符串，描述了该模型是在DeBERTa模型的基础上添加了多选分类头部的模型
@add_start_docstrings(
    """
    DeBERTa Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    DEBERTA_START_DOCSTRING,
)
# 定义TFDebertaV2ForMultipleChoice类，继承自TFDebertaV2PreTrainedModel和TFMultipleChoiceLoss
class TFDebertaV2ForMultipleChoice(TFDebertaV2PreTrainedModel, TFMultipleChoiceLoss):
    
    # 初始化方法
    def __init__(self, config: DebertaV2Config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 创建deberta层，使用TFDebertaV2MainLayer类进行初始化
        self.deberta = TFDebertaV2MainLayer(config, name="deberta")
        # 创建dropout层，使用tf.keras.layers.Dropout类进行初始化
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        # 创建pooler层，使用TFDebertaV2ContextPooler类进行初始化
        self.pooler = TFDebertaV2ContextPooler(config, name="pooler")
        # 创建分类器层，使用tf.keras.layers.Dense类进行初始化
        self.classifier = tf.keras.layers.Dense(
            units=1, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        # 设置output_dim属性为pooler层的输出维度
        self.output_dim = self.pooler.output_dim

    # call方法用于模型前向传播
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
    ) -> Union[TFMultipleChoiceModelOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
            where `num_choices` is the size of the second dimension of the input tensors. (See `input_ids` above)
        """
        声明函数，它的返回类型可以是 `TFMultipleChoiceModelOutput` 或者元组类型 `Tuple[tf.Tensor]`
        定义了一个可选的参数 `labels`，它的类型可以是 `tf.Tensor` 或者 `np.ndarray`，它的形状为 `(batch_size,)`
        这个参数用于计算多项选择分类的损失。索引应该在 `[0, ..., num_choices]` 的范围内，
        其中 `num_choices` 是输入张量的第二个维度的大小。 (参见上面的 `input_ids`)
        """
        if input_ids is not None:
            如果 `input_ids` 不为空
            获取 `input_ids` 张量的形状，并将其第二个维度的大小赋值给变量 `num_choices`
            获取 `input_ids` 张量的形状，并将其第三个维度的大小赋值给变量 `seq_length`
        else:
            否则（`input_ids` 为空）
            获取 `inputs_embeds` 张量的形状，并将其第二个维度的大小赋值给变量 `num_choices`
            获取 `inputs_embeds` 张量的形状，并将其第三个维度的大小赋值给变量 `seq_length`

        如果 `input_ids` 不为空，则将 `input_ids` 重塑成形状为 `(-1, seq_length)` 的张量，否则赋值为 None
        如果 `attention_mask` 不为空，则将 `attention_mask` 重塑成形状为 `(-1, seq_length)` 的张量，否则赋值为 None
        如果 `token_type_ids` 不为空，则将 `token_type_ids` 重塑成形状为 `(-1, seq_length)` 的张量，否则赋值为 None
        如果 `position_ids` 不为空，则将 `position_ids` 重塑成形状为 `(-1, seq_length)` 的张量，否则赋值为 None
        如果 `inputs_embeds` 不为空，则将 `inputs_embeds` 重塑成形状为 `(-1, seq_length, shape_list(inputs_embeds)[3])` 的张量，否则赋值为 None
        调用 `deberta` 方法，传入重塑后的参数，并将结果赋值给变量 `outputs`
        获取 `outputs` 的第一个元素作为 `sequence_output`
        调用 `pooler` 方法，传入 `sequence_output` 和 `training` 参数，并将结果赋值给变量 `pooled_output`
        调用 `dropout` 方法，传入 `pooled_output` 和 `training` 参数，并将结果赋值给变量 `pooled_output`
        调用 `classifier` 方法，传入 `pooled_output` 参数，并将结果赋值给变量 `logits`
        将 `logits` 重塑成形状为 `(-1, num_choices)` 的张量，赋值给变量 `reshaped_logits`
        如果 `labels` 为空，则将变量 `loss` 赋值为 None，否则调用 `hf_compute_loss` 方法，传入 `labels` 和 `reshaped_logits` 参数，并将结果赋值给 `loss`

        如果 `return_dict` 为 False，则将 `reshaped_logits` 和 `outputs` 的第三个元素之后的所有元素作为输出，并返回这个元组，
        如果 `loss` 不为空，则在元组前添加 `loss`，否则只返回这个元组
        如果 `return_dict` 为 True，则返回类型为 `TFMultipleChoiceModelOutput` 的对象，
        其中 `loss` 属性为 `loss`，`logits` 属性为 `reshaped_logits`，`hidden_states` 属性为 `outputs` 的 `hidden_states`，
        `attentions` 属性为 `outputs` 的 `attentions`
    # 构建模型，如果已经构建过，则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果存在 DeBERTa 模型，则构建之
        if getattr(self, "deberta", None) is not None:
            with tf.name_scope(self.deberta.name):
                self.deberta.build(None)
        # 如果存在池化层，则构建之
        if getattr(self, "pooler", None) is not None:
            with tf.name_scope(self.pooler.name):
                self.pooler.build(None)
        # 如果存在分类器，则构建之
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                # 构建分类器，输入形状为 [None, None, self.output_dim]
                self.classifier.build([None, None, self.output_dim])
```