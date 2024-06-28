# `.\models\mpnet\modeling_tf_mpnet.py`

```
# 指定编码格式为 UTF-8
# 版权声明和许可信息，遵循 Apache License 2.0
# 导入必要的库和模块
# 定义 MPNet 模型的 Tensorflow 实现

from __future__ import annotations

import math  # 导入数学函数库
import warnings  # 导入警告模块，用于警告处理
from typing import Optional, Tuple, Union  # 导入类型提示相关的类和函数

import numpy as np  # 导入 NumPy 库
import tensorflow as tf  # 导入 TensorFlow 库

# 导入自定义的 Tensorflow 激活函数
from ...activations_tf import get_tf_activation
# 导入 Tensorflow 版本的模型输出相关类
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFBaseModelOutputWithPooling,
    TFMaskedLMOutput,
    TFMultipleChoiceModelOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
# 导入 Tensorflow 工具函数
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
    keras_serializable,
    unpack_inputs,
)
# 导入 Tensorflow 工具函数，用于检查嵌入是否在界限内
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
# 导入通用的工具函数和模块
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
# 导入 MPNet 的配置类
from .configuration_mpnet import MPNetConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置信息
_CHECKPOINT_FOR_DOC = "microsoft/mpnet-base"
_CONFIG_FOR_DOC = "MPNetConfig"

# 预训练模型存档列表
TF_MPNET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/mpnet-base",
]

class TFMPNetPreTrainedModel(TFPreTrainedModel):
    """
    一个处理权重初始化、下载和加载预训练模型的抽象类。
    """

    # 配置类为 MPNetConfig
    config_class = MPNetConfig
    # 基础模型前缀为 "mpnet"
    base_model_prefix = "mpnet"


class TFMPNetEmbeddings(keras.layers.Layer):
    """从单词和位置嵌入构建嵌入层。"""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        # 填充索引为 1
        self.padding_idx = 1
        # 配置对象
        self.config = config
        # 隐藏层大小
        self.hidden_size = config.hidden_size
        # 最大位置嵌入数
        self.max_position_embeddings = config.max_position_embeddings
        # 初始化范围
        self.initializer_range = config.initializer_range
        # 层归一化模块
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # Dropout 层
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
    # 在建立模型时调用的方法，用于构建模型的输入形状
    def build(self, input_shape=None):
        # 在 "word_embeddings" 命名空间下创建权重张量
        self.weight = self.add_weight(
            name="weight",
            shape=[self.config.vocab_size, self.hidden_size],
            initializer=get_initializer(initializer_range=self.initializer_range),
        )

        # 在 "position_embeddings" 命名空间下创建位置嵌入张量
        self.position_embeddings = self.add_weight(
            name="embeddings",
            shape=[self.max_position_embeddings, self.hidden_size],
            initializer=get_initializer(initializer_range=self.initializer_range),
        )

        # 如果模型已经建立，则直接返回，避免重复建立
        if self.built:
            return
        self.built = True

        # 如果存在 LayerNorm 属性，则构建 LayerNorm 层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])

    def create_position_ids_from_input_ids(self, input_ids):
        """
        根据输入的 token ids 创建位置 ids。非填充符号用它们的位置数字替换。
        位置编号从 padding_idx+1 开始，填充符号被忽略。这是从 fairseq 的 `utils.make_positions` 修改而来。

        Args:
            input_ids: tf.Tensor，输入的 token ids
        Returns: tf.Tensor，位置 ids
        """
        # 创建一个掩码，标记非填充符号的位置
        mask = tf.cast(tf.math.not_equal(input_ids, self.padding_idx), dtype=input_ids.dtype)
        # 计算累积的索引，乘以掩码确保只处理非填充符号
        incremental_indices = tf.math.cumsum(mask, axis=1) * mask

        return incremental_indices + self.padding_idx

    def call(self, input_ids=None, position_ids=None, inputs_embeds=None, training=False):
        """
        根据输入张量应用嵌入。

        Returns:
            final_embeddings (`tf.Tensor`): 输出的嵌入张量。
        """
        # 断言输入的 input_ids 或 inputs_embeds 至少有一个不为空
        assert not (input_ids is None and inputs_embeds is None)

        if input_ids is not None:
            # 检查 input_ids 是否在词汇表大小范围内
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            # 使用 weight 张量从参数中获取对应 input_ids 的嵌入向量
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        input_shape = shape_list(inputs_embeds)[:-1]

        if position_ids is None:
            if input_ids is not None:
                # 从输入的 token ids 创建位置 ids，保持填充的仍然填充
                position_ids = self.create_position_ids_from_input_ids(input_ids=input_ids)
            else:
                # 如果没有输入 token ids，则创建一个从 padding_idx+1 开始的位置 ids
                position_ids = tf.expand_dims(
                    tf.range(start=self.padding_idx + 1, limit=input_shape[-1] + self.padding_idx + 1), axis=0
                )

        # 从 position_embeddings 中获取对应 position_ids 的位置嵌入
        position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
        # 最终的嵌入向量是输入嵌入与位置嵌入的和
        final_embeddings = inputs_embeds + position_embeds
        # 对最终的嵌入向量进行 LayerNorm 处理
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        # 在训练时使用 dropout 处理最终的嵌入向量
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)

        return final_embeddings
# 从 transformers.models.bert.modeling_tf_bert.TFBertPooler 复制并修改为 MPNet
class TFMPNetPooler(keras.layers.Layer):
    def __init__(self, config: MPNetConfig, **kwargs):
        super().__init__(**kwargs)

        # 使用 Dense 层定义一个全连接层，units 参数为配置文件中的 hidden_size
        # 使用指定的初始化器初始化权重
        # 激活函数为 tanh
        self.dense = keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 对模型进行池化操作，简单地选择对应于第一个标记的隐藏状态
        first_token_tensor = hidden_states[:, 0]
        # 将第一个标记的隐藏状态输入到 Dense 层中
        pooled_output = self.dense(inputs=first_token_tensor)

        return pooled_output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经构建过，直接返回
        # 否则，在 Dense 层上下文中创建并构建 Dense 层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


class TFMPNetSelfAttention(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        # 确保隐藏大小是注意力头数的整数倍
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads}"
            )

        # 计算每个注意力头的大小和总头大小
        self.num_attention_heads = config.num_attention_heads
        assert config.hidden_size % config.num_attention_heads == 0
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 定义 q、k、v、o 四个 Dense 层，用于注意力计算和输出
        self.q = keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="q"
        )
        self.k = keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="k"
        )
        self.v = keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="v"
        )
        self.o = keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="o"
        )
        # 定义 Dropout 层，用于注意力概率的 dropout
        self.dropout = keras.layers.Dropout(config.attention_probs_dropout_prob)
        self.config = config

    def transpose_for_scores(self, x, batch_size):
        # 重塑张量形状，从 [batch_size, seq_length, all_head_size] 到 [batch_size, seq_length, num_attention_heads, attention_head_size]
        x = tf.reshape(x, (batch_size, -1, self.num_attention_heads, self.attention_head_size))

        return tf.transpose(x, perm=[0, 2, 1, 3])
    # 定义一个方法，用于执行自注意力机制
    def call(self, hidden_states, attention_mask, head_mask, output_attentions, position_bias=None, training=False):
        # 获取批量大小
        batch_size = shape_list(hidden_states)[0]

        # 计算查询（q）、键（k）、值（v）的线性变换结果
        q = self.q(hidden_states)
        k = self.k(hidden_states)
        v = self.v(hidden_states)

        # 将查询（q）、键（k）、值（v）转换为适合计算注意力分数的形状
        q = self.transpose_for_scores(q, batch_size)
        k = self.transpose_for_scores(k, batch_size)
        v = self.transpose_for_scores(v, batch_size)

        # 计算注意力分数
        attention_scores = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(shape_list(k)[-1], attention_scores.dtype)
        attention_scores = attention_scores / tf.math.sqrt(dk)

        # 如果提供了位置偏置（在 MPNetEncoder 中预先计算），则将其应用于注意力分数
        if position_bias is not None:
            attention_scores += position_bias

        # 如果存在注意力遮罩，则将其应用于注意力分数
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # 计算注意力权重
        attention_probs = stable_softmax(attention_scores, axis=-1)

        # 在训练时应用 dropout
        attention_probs = self.dropout(attention_probs, training=training)

        # 如果存在头部遮罩，则将其应用于注意力权重
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算加权后的值向量
        c = tf.matmul(attention_probs, v)
        c = tf.transpose(c, perm=[0, 2, 1, 3])
        c = tf.reshape(c, (batch_size, -1, self.all_head_size))

        # 将加权后的值向量传递给输出层
        o = self.o(c)

        # 返回输出结果，包括加权后的值向量和可能的注意力权重（如果需要输出注意力权重）
        outputs = (o, attention_probs) if output_attentions else (o,)
        return outputs

    # 构建层结构，初始化并构建 q、k、v、o 等变量
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return

        # 标记为已构建
        self.built = True

        # 如果存在 q 变量，初始化并构建其结构
        if getattr(self, "q", None) is not None:
            with tf.name_scope(self.q.name):
                self.q.build([None, None, self.config.hidden_size])

        # 如果存在 k 变量，初始化并构建其结构
        if getattr(self, "k", None) is not None:
            with tf.name_scope(self.k.name):
                self.k.build([None, None, self.config.hidden_size])

        # 如果存在 v 变量，初始化并构建其结构
        if getattr(self, "v", None) is not None:
            with tf.name_scope(self.v.name):
                self.v.build([None, None, self.config.hidden_size])

        # 如果存在 o 变量，初始化并构建其结构
        if getattr(self, "o", None) is not None:
            with tf.name_scope(self.o.name):
                self.o.build([None, None, self.config.hidden_size])
# 定义自定义的注意力层类 TFMPNetAttention，继承自 keras.layers.Layer
class TFMPNetAttention(keras.layers.Layer):
    # 初始化函数，接受一个 config 对象和其他关键字参数
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        # 创建 TFMPNetSelfAttention 层实例，命名为 "attn"
        self.attn = TFMPNetSelfAttention(config, name="attn")
        # 创建 LayerNormalization 层实例，使用给定的 epsilon 值
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建 Dropout 层实例，使用给定的 dropout 概率
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)
        # 存储传入的 config 对象
        self.config = config

    # 未实现的方法，用于裁剪注意力头部
    def prune_heads(self, heads):
        raise NotImplementedError

    # 调用函数，处理输入张量并执行自注意力操作
    def call(self, input_tensor, attention_mask, head_mask, output_attentions, position_bias=None, training=False):
        # 使用 self.attn 处理输入张量，得到自注意力层的输出
        self_outputs = self.attn(
            input_tensor, attention_mask, head_mask, output_attentions, position_bias=position_bias, training=training
        )
        # 对自注意力层的输出进行 LayerNormalization 和 Dropout 处理，并与输入张量相加
        attention_output = self.LayerNorm(self.dropout(self_outputs[0]) + input_tensor)
        # 构建输出元组，包含处理后的注意力输出及可能的额外输出
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

    # 构建函数，用于构建层结构
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果 self.attn 存在，使用其名称空间构建 self.attn
        if getattr(self, "attn", None) is not None:
            with tf.name_scope(self.attn.name):
                self.attn.build(None)
        # 如果 self.LayerNorm 存在，使用其名称空间构建 self.LayerNorm
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


# 从 transformers.models.bert.modeling_tf_bert.TFBertIntermediate 复制并修改为 MPNet
# 定义 MPNet 的中间层类 TFMPNetIntermediate，继承自 keras.layers.Layer
class TFMPNetIntermediate(keras.layers.Layer):
    # 初始化函数，接受一个 MPNetConfig 对象和其他关键字参数
    def __init__(self, config: MPNetConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建 Dense 层实例，使用给定的单元数和初始化器
        self.dense = keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        # 根据 hidden_act 类型确定激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        # 存储传入的 config 对象
        self.config = config

    # 调用函数，处理隐藏状态张量并执行 Dense 层和激活函数操作
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 经过 Dense 层处理隐藏状态张量
        hidden_states = self.dense(inputs=hidden_states)
        # 使用 intermediate_act_fn 执行激活函数操作
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

    # 构建函数，用于构建层结构
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果 self.dense 存在，使用其名称空间构建 self.dense
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# 从 transformers.models.bert.modeling_tf_bert.TFBertOutput 复制并修改为 MPNet
# 定义 MPNet 的输出层类 TFMPNetOutput，继承自 keras.layers.Layer
class TFMPNetOutput(keras.layers.Layer):
    # 初始化方法，接受一个配置对象和其他关键字参数
    def __init__(self, config: MPNetConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 创建一个全连接层，使用给定的隐藏单元数和初始化器范围
        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        
        # 创建一个层归一化层，使用给定的 epsilon 参数
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        
        # 创建一个 dropout 层，使用给定的 dropout 比例
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        
        # 存储配置对象，以便后续使用
        self.config = config

    # 模型调用方法，接受隐藏状态张量、输入张量和训练标志，并返回变换后的隐藏状态张量
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 使用全连接层变换隐藏状态张量
        hidden_states = self.dense(inputs=hidden_states)
        
        # 使用 dropout 层对变换后的隐藏状态张量进行 dropout 处理
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        
        # 使用层归一化层对 dropout 处理后的张量和输入张量进行残差连接和归一化处理
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        # 返回变换后的隐藏状态张量
        return hidden_states

    # 构建方法，用于构建层对象，根据输入形状构建层的内部结构
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 设置标记，表示已经构建过
        self.built = True
        
        # 如果存在全连接层，使用给定的名称作为命名空间，构建全连接层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])
        
        # 如果存在层归一化层，使用给定的名称作为命名空间，构建层归一化层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])
# 定义 TFMPNetLayer 类，继承自 keras 的 Layer 类
class TFMPNetLayer(keras.layers.Layer):
    # 初始化方法，接受 config 和其他关键字参数
    def __init__(self, config, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 创建 TFMPNetAttention 实例，命名为 "attention"
        self.attention = TFMPNetAttention(config, name="attention")
        # 创建 TFMPNetIntermediate 实例，命名为 "intermediate"
        self.intermediate = TFMPNetIntermediate(config, name="intermediate")
        # 创建 TFMPNetOutput 实例，命名为 "output"
        self.out = TFMPNetOutput(config, name="output")

    # call 方法定义了层的前向传播逻辑
    def call(self, hidden_states, attention_mask, head_mask, output_attentions, position_bias=None, training=False):
        # 调用 self.attention 的 call 方法，进行自注意力计算
        self_attention_outputs = self.attention(
            hidden_states, attention_mask, head_mask, output_attentions, position_bias=position_bias, training=training
        )
        # 获取自注意力计算的输出结果
        attention_output = self_attention_outputs[0]
        # 如果需要输出注意力权重，将其添加到 outputs 中
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # 将注意力输出传递给 self.intermediate 层进行处理
        intermediate_output = self.intermediate(attention_output)
        # 将 intermediate_output 和 attention_output 传递给 self.out 层进行处理
        layer_output = self.out(intermediate_output, attention_output, training=training)
        # 将 layer_output 添加到 outputs 中
        outputs = (layer_output,) + outputs  # add attentions if we output them

        # 返回最终的输出结果
        return outputs

    # build 方法用于构建层，处理层的内部状态
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        self.built = True

        # 如果 self.attention 存在，则构建 self.attention 层
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)

        # 如果 self.intermediate 存在，则构建 self.intermediate 层
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)

        # 如果 self.out 存在，则构建 self.out 层
        if getattr(self, "out", None) is not None:
            with tf.name_scope(self.out.name):
                self.out.build(None)


# 定义 TFMPNetEncoder 类，继承自 keras 的 Layer 类
class TFMPNetEncoder(keras.layers.Layer):
    # 初始化方法，接受 config 和其他关键字参数
    def __init__(self, config, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 保存 config 中的参数
        self.config = config
        self.n_heads = config.num_attention_heads
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.initializer_range = config.initializer_range

        # 创建 TFMPNetLayer 的列表，命名为 "layer_._{i}"，共 config.num_hidden_layers 个
        self.layer = [TFMPNetLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]
        # 保存相对注意力的桶数
        self.relative_attention_num_buckets = config.relative_attention_num_buckets

    # build 方法用于构建层，处理层的内部状态
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        self.built = True

        # 使用 tf.name_scope 构建 "relative_attention_bias" 命名空间
        with tf.name_scope("relative_attention_bias"):
            # 创建相对注意力的偏置权重 self.relative_attention_bias
            self.relative_attention_bias = self.add_weight(
                name="embeddings",
                shape=[self.relative_attention_num_buckets, self.n_heads],
                initializer=get_initializer(self.initializer_range),
            )

        # 如果 self.layer 列表存在，则依次构建其中的每个 TFMPNetLayer 层
        if getattr(self, "layer", None) is not None:
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    layer.build(None)
    # 定义一个方法，用于执行模型的前向推断过程，生成输出结果
    def call(
        self,
        hidden_states,                # 输入的隐藏状态张量
        attention_mask,               # 注意力掩码张量，用于指定哪些位置需要进行注意力计算
        head_mask,                    # 多头注意力的掩码，控制每个注意力头的屏蔽情况
        output_attentions,            # 是否输出注意力权重
        output_hidden_states,         # 是否输出所有层的隐藏状态
        return_dict,                  # 是否以字典形式返回结果
        training=False,               # 是否在训练模式下运行，默认为 False
    ):
        # 计算位置偏置，用于处理位置编码的影响
        position_bias = self.compute_position_bias(hidden_states)
        # 如果需要输出所有层的隐藏状态，则初始化空元组
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出所有层的注意力权重，则初始化空元组
        all_attentions = () if output_attentions else None

        # 遍历每一层Transformer层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出所有层的隐藏状态，则将当前层的隐藏状态加入到元组中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 调用Transformer层的前向传播方法
            layer_outputs = layer_module(
                hidden_states,         # 输入的隐藏状态张量
                attention_mask,        # 注意力掩码张量
                head_mask[i],          # 当前层的多头注意力掩码
                output_attentions,     # 是否输出注意力权重
                position_bias=position_bias,  # 位置偏置张量
                training=training,     # 是否在训练模式下运行
            )
            # 更新隐藏状态为当前层的输出隐藏状态
            hidden_states = layer_outputs[0]

            # 如果需要输出所有层的注意力权重，则将当前层的注意力权重加入到元组中
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # 添加最后一层的隐藏状态到元组中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要以字典形式返回结果，则返回非空元组中的元素
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

        # 如果需要以字典形式返回结果，则创建TFBaseModelOutput对象返回
        return TFBaseModelOutput(
            last_hidden_state=hidden_states,  # 最后一层的隐藏状态
            hidden_states=all_hidden_states,  # 所有层的隐藏状态元组
            attentions=all_attentions         # 所有层的注意力权重元组
        )

    @staticmethod
    # 定义一个静态方法，用于计算相对位置的桶索引
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += tf.cast(tf.math.less(n, 0), dtype=relative_position.dtype) * num_buckets
        n = tf.math.abs(n)

        # 现在 n 的范围是 [0, inf)
        max_exact = num_buckets // 2
        is_small = tf.math.less(n, max_exact)

        # 如果 n 较小，则直接使用 n 作为桶索引
        val_if_large = max_exact + tf.cast(
            tf.math.log(n / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact),
            dtype=relative_position.dtype,
        )

        # 限制桶索引的最大值为 num_buckets - 1
        val_if_large = tf.math.minimum(val_if_large, num_buckets - 1)
        # 根据 n 的大小选择最终的桶索引
        ret += tf.where(is_small, n, val_if_large)
        return ret
    # 定义一个方法用于计算位置偏置，基于输入的相对位置进行分桶操作
    def compute_position_bias(self, x, position_ids=None):
        """Compute binned relative position bias"""
        # 获取输入张量的形状信息
        input_shape = shape_list(x)
        # 获取输入张量的长度信息，假设是相等的，qlen 表示查询长度，klen 表示键长度
        qlen, klen = input_shape[1], input_shape[1]

        # 如果给定了位置 ID，则使用这些 ID；否则创建一个从 0 到 qlen-1 的序列作为位置 ID
        if position_ids is not None:
            # 获取上下文位置，形状为 (batch_size, qlen, 1)
            context_position = position_ids[:, :, None]
            # 获取记忆位置，形状为 (batch_size, 1, klen)
            memory_position = position_ids[:, None, :]
        else:
            # 创建一个从 0 到 qlen-1 的序列，表示上下文位置
            context_position = tf.range(qlen)[:, None]
            # 创建一个从 0 到 klen-1 的序列，表示记忆位置
            memory_position = tf.range(klen)[None, :]

        # 计算相对位置，形状为 (qlen, klen)，表示每个查询位置和每个键位置的相对偏移量
        relative_position = memory_position - context_position

        # 对相对位置进行分桶，将相对位置映射到预定义数量的桶中
        rp_bucket = self._relative_position_bucket(
            relative_position,
            num_buckets=self.relative_attention_num_buckets,
        )
        
        # 从预先计算的相对注意力偏置中获取对应的值，形状为 (qlen, klen, num_heads)
        values = tf.gather(self.relative_attention_bias, rp_bucket)

        # 调整维度顺序，将注意力偏置按照 (num_heads, qlen, klen) 的顺序排列，并添加一个维度作为批处理维度
        values = tf.expand_dims(tf.transpose(values, [2, 0, 1]), axis=0)

        # 返回计算后的位置偏置张量
        return values
# 将类 TFMPNetMainLayer 标记为 Keras 序列化类
@keras_serializable
class TFMPNetMainLayer(keras.layers.Layer):
    # 指定配置类为 MPNetConfig
    config_class = MPNetConfig

    # 初始化方法，接受配置参数 config
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        # 将传入的配置参数赋给实例变量
        self.config = config
        self.num_hidden_layers = config.num_hidden_layers  # 设置隐藏层的数量
        self.initializer_range = config.initializer_range  # 设置初始化范围
        self.output_attentions = config.output_attentions  # 是否输出注意力权重
        self.output_hidden_states = config.output_hidden_states  # 是否输出隐藏状态
        self.return_dict = config.use_return_dict  # 是否返回字典格式结果
        # 创建 TFMPNetEncoder 实例，用于编码器
        self.encoder = TFMPNetEncoder(config, name="encoder")
        # 创建 TFMPNetPooler 实例，用于池化器
        self.pooler = TFMPNetPooler(config, name="pooler")
        # 创建 TFMPNetEmbeddings 实例，用于嵌入层，必须在最后声明以保持权重顺序
        self.embeddings = TFMPNetEmbeddings(config, name="embeddings")

    # 从 transformers.models.bert.modeling_tf_bert.TFBertMainLayer 中复制的方法，获取输入嵌入层
    def get_input_embeddings(self) -> keras.layers.Layer:
        return self.embeddings

    # 从 transformers.models.bert.modeling_tf_bert.TFBertMainLayer 中复制的方法，设置输入嵌入层
    def set_input_embeddings(self, value: tf.Variable):
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    # 从 transformers.models.bert.modeling_tf_bert.TFBertMainLayer 中复制的方法，修剪模型中的注意力头
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    # 从装饰器 @unpack_inputs 中复制的方法，定义层的调用方式及其参数
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
    ):
        ...

    # 在构建层时调用的方法，用于构建内部组件的 TensorFlow 图
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在编码器，则构建编码器
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 如果存在池化器，则构建池化器
        if getattr(self, "pooler", None) is not None:
            with tf.name_scope(self.pooler.name):
                self.pooler.build(None)
        # 如果存在嵌入层，则构建嵌入层
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)


# MPNetStartDocstring 是一个原始字符串常量，用于描述 MPNet 模型的详细信息和用法
MPNET_START_DOCSTRING = r"""

    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a [keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
    behavior.

    <Tip>
"""
    # TensorFlow 模型和层在 `transformers` 中支持两种输入格式：
    
    # 1. 将所有输入作为关键字参数（类似于 PyTorch 模型）；
    # 2. 将所有输入作为列表、元组或字典的第一个位置参数。
    
    # 第二种格式的支持是因为 Keras 方法在将输入传递给模型和层时偏向于使用这种格式。
    # 因此，当使用诸如 `model.fit()` 这样的方法时，您只需传递模型所支持的任何格式的输入和标签即可！
    
    # 然而，如果您希望在 Keras 方法之外使用第二种格式，例如在使用 Keras 的 `Functional` API 创建自己的层或模型时，
    # 可以使用三种方式将所有输入张量聚合到第一个位置参数中：
    
    # - 只包含 `input_ids` 的单个张量：`model(input_ids)`
    # - 长度可变的张量列表，按照文档字符串中给出的顺序：`model([input_ids, attention_mask])`
    #   或 `model([input_ids, attention_mask, token_type_ids])`
    # - 包含一个或多个输入张量，并与文档字符串中给出的输入名称关联的字典：
    #   `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`
    
    # 请注意，如果使用[子类化](https://keras.io/guides/making_new_layers_and_models_via_subclassing/)创建模型和层，
    # 那么您无需担心这些问题，可以像传递输入给任何其他 Python 函数一样简单！
    
    # Args:
    # config ([`MPNetConfig`]): 模型配置类，包含模型的所有参数。
    # 初始化配置文件并不会加载与模型相关的权重，仅加载配置。
    # 可以查看 [`~PreTrainedModel.from_pretrained`] 方法来加载模型权重。
MPNET_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`Numpy array` or `tf.Tensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.__call__`] and
            [`PreTrainedTokenizer.encode`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`Numpy array` or `tf.Tensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`Numpy array` or `tf.Tensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`Numpy array` or `tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`tf.Tensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
            config will be used instead.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
            used instead.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in
            eager mode, in graph mode the value will always be set to True.
        training (`bool`, *optional*, defaults to `False`):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""


@add_start_docstrings(
    "The bare MPNet Model transformer outputting raw hidden-states without any specific head on top.",
    MPNET_INPUTS_DOCSTRING.format("batch_size, sequence_length")
)
    MPNET_START_DOCSTRING,


注释：


# 使用 MPNET_START_DOCSTRING 指示符，可能用于开始一个多行字符串文档的标记
)

class TFMPNetModel(TFMPNetPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.mpnet = TFMPNetMainLayer(config, name="mpnet")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(MPNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,  # 定义输入参数 input_ids，可以为 None
        attention_mask: Optional[Union[np.array, tf.Tensor]] = None,  # 定义输入参数 attention_mask，可以为 None
        position_ids: Optional[Union[np.array, tf.Tensor]] = None,  # 定义输入参数 position_ids，可以为 None
        head_mask: Optional[Union[np.array, tf.Tensor]] = None,  # 定义输入参数 head_mask，可以为 None
        inputs_embeds: tf.Tensor | None = None,  # 定义输入参数 inputs_embeds，可以为 None
        output_attentions: Optional[bool] = None,  # 定义输入参数 output_attentions，可以为 None
        output_hidden_states: Optional[bool] = None,  # 定义输入参数 output_hidden_states，可以为 None
        return_dict: Optional[bool] = None,  # 定义输入参数 return_dict，可以为 None
        training: bool = False,  # 定义输入参数 training，默认为 False
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:  # 指定函数返回类型为 TFBaseModelOutput 或 Tuple[tf.Tensor]
        outputs = self.mpnet(
            input_ids=input_ids,
            attention_mask=attention_mask,
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
            return  # 如果模型已经建立，则直接返回
        self.built = True  # 标记模型已经建立
        if getattr(self, "mpnet", None) is not None:
            with tf.name_scope(self.mpnet.name):  # 使用 mpnet 层的名字作为命名空间
                self.mpnet.build(None)  # 构建 mpnet 层

class TFMPNetLMHead(keras.layers.Layer):
    """MPNet head for masked and permuted language modeling"""

    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)

        self.config = config  # 存储配置信息
        self.hidden_size = config.hidden_size  # 存储隐藏层大小
        self.dense = keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )  # 创建全连接层，使用指定的初始化方法和名字
        self.layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")  # 创建 LayerNormalization 层，使用指定的 epsilon 和名字
        self.act = get_tf_activation("gelu")  # 获取 GELU 激活函数

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = input_embeddings  # 存储输入的嵌入层权重
    # 在神经网络模型中构建操作，根据输入形状初始化偏置项
    def build(self, input_shape=None):
        # 添加一个形状为 (vocab_size,) 的可训练偏置项，初始化为零向量
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="bias")

        # 如果已经构建过网络，直接返回，避免重复构建
        if self.built:
            return
        # 标记模型已经构建
        self.built = True

        # 如果存在 dense 层，则构建该层并指定输入形状
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])

        # 如果存在 layer_norm 层，则构建该层并指定输入形状
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.hidden_size])

    # 返回解码器对象，用于输出嵌入
    def get_output_embeddings(self):
        return self.decoder

    # 设置输出嵌入的值
    def set_output_embeddings(self, value):
        # 更新解码器的权重
        self.decoder.weight = value
        # 更新解码器的词汇表大小
        self.decoder.vocab_size = shape_list(value)[0]

    # 返回偏置项字典
    def get_bias(self):
        return {"bias": self.bias}

    # 设置偏置项的值
    def set_bias(self, value):
        # 更新偏置项的值
        self.bias = value["bias"]
        # 更新配置中的词汇表大小
        self.config.vocab_size = shape_list(value["bias"])[0]

    # 实现模型的前向传播
    def call(self, hidden_states):
        # 线性变换：全连接层
        hidden_states = self.dense(hidden_states)
        # 激活函数处理
        hidden_states = self.act(hidden_states)
        # 层归一化处理
        hidden_states = self.layer_norm(hidden_states)

        # 将隐藏状态投影回词汇表大小，同时加上偏置项
        seq_length = shape_list(tensor=hidden_states)[1]
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.hidden_size])
        hidden_states = tf.matmul(a=hidden_states, b=self.decoder.weight, transpose_b=True)
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)

        return hidden_states
@add_start_docstrings("""MPNet Model with a `language modeling` head on top.""", MPNET_START_DOCSTRING)
class TFMPNetForMaskedLM(TFMPNetPreTrainedModel, TFMaskedLanguageModelingLoss):
    _keys_to_ignore_on_load_missing = [r"pooler"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 初始化 MPNet 主层，使用给定的配置和名称
        self.mpnet = TFMPNetMainLayer(config, name="mpnet")
        # 初始化语言模型头部，使用给定的配置和嵌入层
        self.lm_head = TFMPNetLMHead(config, self.mpnet.embeddings, name="lm_head")

    def get_lm_head(self):
        # 返回语言模型头部
        return self.lm_head

    def get_prefix_bias_name(self):
        # 发出警告，告知该方法已经废弃，请使用 get_bias 方法替代
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        # 返回包含名称和语言模型头部名称的字符串
        return self.name + "/" + self.lm_head.name

    @unpack_inputs
    @add_start_docstrings_to_model_forward(MPNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFMaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: tf.Tensor | None = None,
        training: bool = False,
    ) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        # 调用 MPNet 主层，传递所有输入参数并获取输出
        outputs = self.mpnet(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 从 MPNet 主层输出中提取序列输出
        sequence_output = outputs[0]
        # 使用语言模型头部生成预测分数
        prediction_scores = self.lm_head(sequence_output)

        # 如果未提供标签，则损失设为 None；否则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, prediction_scores)

        # 如果不要求返回字典形式的输出，则组装输出结果
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TFMaskedLMOutput 对象，包含损失、预测日志、隐藏状态和注意力权重
        return TFMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    # 定义神经网络模型的 build 方法，用于构建模型结构
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回，避免重复构建
        if self.built:
            return
        # 将模型标记为已构建状态
        self.built = True
        # 如果模型中存在名为 mpnet 的属性，并且不为 None，则构建 mpnet
        if getattr(self, "mpnet", None) is not None:
            # 使用 tf.name_scope 将 mpnet 的构建过程放入命名空间 self.mpnet.name 中
            with tf.name_scope(self.mpnet.name):
                # 调用 mpnet 对象的 build 方法进行模型构建
                self.mpnet.build(None)
        # 如果模型中存在名为 lm_head 的属性，并且不为 None，则构建 lm_head
        if getattr(self, "lm_head", None) is not None:
            # 使用 tf.name_scope 将 lm_head 的构建过程放入命名空间 self.lm_head.name 中
            with tf.name_scope(self.lm_head.name):
                # 调用 lm_head 对象的 build 方法进行模型构建
                self.lm_head.build(None)
# 定义一个自定义的 Keras 层，用于处理文本序列级别的分类任务
class TFMPNetClassificationHead(keras.layers.Layer):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 创建一个全连接层，输出维度为 config.hidden_size，激活函数为 tanh
        self.dense = keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )
        # 添加一个 Dropout 层，以减少过拟合，丢弃率为 config.hidden_dropout_prob
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)
        # 创建一个全连接层，输出维度为 config.num_labels，用于最终的分类预测
        self.out_proj = keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="out_proj"
        )
        # 保存传入的配置对象
        self.config = config

    def call(self, features, training=False):
        # 获取 features 的第一个 token 的表示，通常是 <s>（等同于 [CLS]）
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        # 对输入进行 Dropout 处理，用于训练阶段避免过拟合
        x = self.dropout(x, training=training)
        # 经过全连接层进行特征转换
        x = self.dense(x)
        # 再次进行 Dropout 处理
        x = self.dropout(x, training=training)
        # 经过最终的全连接层，得到分类预测结果
        x = self.out_proj(x)
        return x

    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        self.built = True
        # 如果 dense 层已定义，则根据输入形状构建它
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 如果 out_proj 层已定义，则根据输入形状构建它
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build([None, None, self.config.hidden_size])


# 使用装饰器为 TFMPNetForSequenceClassification 类添加文档字符串
@add_start_docstrings(
    """
    MPNet Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    MPNET_START_DOCSTRING,
)
# TFMPNetForSequenceClassification 类继承自 TFMPNetPreTrainedModel 和 TFSequenceClassificationLoss
class TFMPNetForSequenceClassification(TFMPNetPreTrainedModel, TFSequenceClassificationLoss):
    # 加载模型时忽略的键名列表，避免缺失的 "pooler" 键名导致的加载错误
    _keys_to_ignore_on_load_missing = [r"pooler"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 保存分类数目
        self.num_labels = config.num_labels

        # 创建 MPNet 主层，并命名为 "mpnet"
        self.mpnet = TFMPNetMainLayer(config, name="mpnet")
        # 创建分类器头部，用于进行分类任务，命名为 "classifier"
        self.classifier = TFMPNetClassificationHead(config, name="classifier")

    # 使用装饰器解包输入并添加模型前向传播的文档字符串
    @unpack_inputs
    @add_start_docstrings_to_model_forward(MPNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: Optional[Union[np.array, tf.Tensor]] = None,
        position_ids: Optional[Union[np.array, tf.Tensor]] = None,
        head_mask: Optional[Union[np.array, tf.Tensor]] = None,
        inputs_embeds: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: tf.Tensor | None = None,
        training: bool = False,
        # 声明函数的输入参数，这些参数用于模型的前向传播
        ) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 调用 MPNet 模型进行推理，返回的是一个包含各种输出的命名元组或元组
        outputs = self.mpnet(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 从 MPNet 的输出中获取序列输出
        sequence_output = outputs[0]
        # 将序列输出传递给分类器模型，并指定是否处于训练状态
        logits = self.classifier(sequence_output, training=training)

        # 如果没有提供标签，则损失为 None；否则，使用标签和预测值计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果不要求返回字典形式的输出，则构建一个元组作为结果返回
        if not return_dict:
            output = (logits,) + outputs[2:]  # 排除了第一个元素以外的所有元素作为输出
            return ((loss,) + output) if loss is not None else output

        # 如果要求返回字典形式的输出，则创建 TFSequenceClassifierOutput 对象并返回
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
        self.built = True
        
        # 构建 MPNet 模型（如果存在）
        if getattr(self, "mpnet", None) is not None:
            with tf.name_scope(self.mpnet.name):
                self.mpnet.build(None)
        
        # 构建分类器模型（如果存在）
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build(None)
"""
MPNet Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.
"""
@add_start_docstrings(
    """
    MPNet Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    MPNET_START_DOCSTRING,
)
class TFMPNetForMultipleChoice(TFMPNetPreTrainedModel, TFMultipleChoiceLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 初始化 MPNet 主层，命名为 'mpnet'
        self.mpnet = TFMPNetMainLayer(config, name="mpnet")
        # 添加 dropout 层，使用给定的隐藏层 dropout 概率
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)
        # 定义分类器，输出维度为 1，使用给定初始化范围的初始化器
        self.classifier = keras.layers.Dense(
            1, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        # 保存配置
        self.config = config

    # 定义前向传播函数，接受一组输入参数
    @unpack_inputs
    @add_start_docstrings_to_model_forward(MPNET_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFMultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: tf.Tensor | None = None,
        training: bool = False,

        # MPNet 的前向传播函数，接受多种输入参数，包括输入的 token IDs、注意力掩码、位置 IDs、头部掩码、
        # 嵌入向量等，还可以控制是否返回字典形式的输出，是否在训练模式下
        # 返回的结果类型为 TFMultipleChoiceModelOutput
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: tf.Tensor | None = None,
        training: bool = False,
    ) -> Union[TFMultipleChoiceModelOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
            where `num_choices` is the size of the second dimension of the input tensors. (See `input_ids` above)
        """
        # 如果传入了 `input_ids` 参数，则确定 `num_choices` 和 `seq_length` 的值
        if input_ids is not None:
            num_choices = shape_list(input_ids)[1]  # 获取选择项的数量
            seq_length = shape_list(input_ids)[2]   # 获取序列长度
        else:
            num_choices = shape_list(inputs_embeds)[1]  # 否则，使用 `inputs_embeds` 确定 `num_choices`
            seq_length = shape_list(inputs_embeds)[2]   # 并且使用 `inputs_embeds` 确定 `seq_length`

        # 将输入张量展平以便适应模型输入要求
        flat_input_ids = tf.reshape(input_ids, (-1, seq_length)) if input_ids is not None else None
        flat_attention_mask = tf.reshape(attention_mask, (-1, seq_length)) if attention_mask is not None else None
        flat_position_ids = tf.reshape(position_ids, (-1, seq_length)) if position_ids is not None else None
        flat_inputs_embeds = (
            tf.reshape(inputs_embeds, (-1, seq_length, shape_list(inputs_embeds)[3]))
            if inputs_embeds is not None
            else None
        )
        # 调用模型的主体部分 `mpnet` 进行推断
        outputs = self.mpnet(
            flat_input_ids,
            flat_attention_mask,
            flat_position_ids,
            head_mask,
            flat_inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 提取汇聚的输出特征，并应用 dropout
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output, training=training)
        # 应用分类器获取最终的 logits
        logits = self.classifier(pooled_output)
        # 重新整形 logits 以匹配预期的形状
        reshaped_logits = tf.reshape(logits, (-1, num_choices))
        # 如果提供了标签，则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, reshaped_logits)

        # 根据 return_dict 参数决定返回值的格式
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果设置了 return_dict，则返回一个包含多选模型输出的对象
        return TFMultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果模型已经构建完毕，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 `mpnet` 属性，则构建 `mpnet`
        if getattr(self, "mpnet", None) is not None:
            with tf.name_scope(self.mpnet.name):
                self.mpnet.build(None)
        # 如果存在 `classifier` 属性，则构建 `classifier`
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
# 定义一个 TFMPNetForTokenClassification 类，继承自 TFMPNetPreTrainedModel 和 TFTokenClassificationLoss，用于在 MPNet 模型的隐藏状态输出上添加一个标记分类头（即一个线性层），例如用于命名实体识别（NER）任务。
@add_start_docstrings(
    """
       MPNet Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
       Named-Entity-Recognition (NER) tasks.
       """,
    MPNET_START_DOCSTRING,  # 添加开始文档字符串，引用了 MPNet 的起始文档字符串
)
class TFMPNetForTokenClassification(TFMPNetPreTrainedModel, TFTokenClassificationLoss):
    _keys_to_ignore_on_load_missing = [r"pooler"]  # 在加载模型时要忽略的键列表，这里忽略了名为 "pooler" 的项

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)  # 调用父类的初始化方法

        self.num_labels = config.num_labels  # 标签的数量，从配置中获取
        self.mpnet = TFMPNetMainLayer(config, name="mpnet")  # MPNet 的主层，使用给定的配置创建
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)  # Dropout 层，使用给定的隐藏层dropout概率
        self.classifier = keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )  # 分类器层，用于输出标签数量的分类结果，使用给定的初始化器范围进行初始化
        self.config = config  # 存储配置对象

    @unpack_inputs  # 解包输入参数装饰器
    @add_start_docstrings_to_model_forward(MPNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))  # 添加开始文档字符串到模型前向传播
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,  # 参考的检查点信息
        output_type=TFTokenClassifierOutput,  # 输出类型为 TFTokenClassifierOutput
        config_class=_CONFIG_FOR_DOC,  # 参考的配置类信息
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,  # 输入的 token IDs，类型为 TFModelInputType 或 None
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 注意力掩码，类型为 numpy 数组、Tensor 或 None
        position_ids: np.ndarray | tf.Tensor | None = None,  # 位置 IDs，类型为 numpy 数组、Tensor 或 None
        head_mask: np.ndarray | tf.Tensor | None = None,  # 头部掩码，类型为 numpy 数组、Tensor 或 None
        inputs_embeds: tf.Tensor | None = None,  # 嵌入输入，类型为 Tensor 或 None
        output_attentions: Optional[bool] = None,  # 是否输出注意力
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典形式的输出
        labels: tf.Tensor | None = None,  # 标签数据，类型为 Tensor 或 None
        training: bool = False,  # 是否为训练模式
    ) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 调用 MPNet 主层进行前向传播
        outputs = self.mpnet(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        sequence_output = outputs[0]  # 获取模型输出的序列输出

        sequence_output = self.dropout(sequence_output, training=training)  # 应用 dropout 操作到序列输出上
        logits = self.classifier(sequence_output)  # 使用分类器层得到分类 logits

        loss = None if labels is None else self.hf_compute_loss(labels, logits)  # 计算 token 分类损失，如果没有标签则为 None

        if not return_dict:
            output = (logits,) + outputs[1:]  # 如果不返回字典形式的输出，构建输出元组
            return ((loss,) + output) if loss is not None else output  # 返回损失和输出，如果损失不为 None

        return TFTokenClassifierOutput(  # 返回 TFTokenClassifierOutput 类型的输出
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    # 构建模型的方法，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        
        # 标记模型已经构建
        self.built = True
        
        # 如果存在名为 "mpnet" 的属性并且不为 None，则构建 mpnet
        if getattr(self, "mpnet", None) is not None:
            # 在命名空间下构建 mpnet
            with tf.name_scope(self.mpnet.name):
                self.mpnet.build(None)
        
        # 如果存在名为 "classifier" 的属性并且不为 None，则构建 classifier
        if getattr(self, "classifier", None) is not None:
            # 在命名空间下构建 classifier
            with tf.name_scope(self.classifier.name):
                # 构建 classifier，期望输入形状为 [None, None, self.config.hidden_size]
                self.classifier.build([None, None, self.config.hidden_size])
# 使用装饰器添加文档字符串，描述了该类是基于 MPNet 模型，用于抽取式问答任务（如 SQuAD），在隐藏状态的基础上增加一个分类头部来计算“span start logits”和“span end logits”。
@add_start_docstrings(
    """
    MPNet Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    MPNET_START_DOCSTRING,
)
class TFMPNetForQuestionAnswering(TFMPNetPreTrainedModel, TFQuestionAnsweringLoss):
    # 在加载过程中需要忽略的键列表
    _keys_to_ignore_on_load_missing = [r"pooler"]

    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)
        # 设置类别数目
        self.num_labels = config.num_labels

        # 初始化 MPNet 主层
        self.mpnet = TFMPNetMainLayer(config, name="mpnet")
        # 初始化用于问答输出的 Dense 层
        self.qa_outputs = keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )
        # 设置配置信息
        self.config = config

    # 使用装饰器定义 call 方法的输入和输出文档字符串
    @unpack_inputs
    @add_start_docstrings_to_model_forward(MPNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: Optional[Union[np.array, tf.Tensor]] = None,
        position_ids: Optional[Union[np.array, tf.Tensor]] = None,
        head_mask: Optional[Union[np.array, tf.Tensor]] = None,
        inputs_embeds: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        start_positions: tf.Tensor | None = None,
        end_positions: tf.Tensor | None = None,
        training: bool = False,
        **kwargs,
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
        # 调用 MPNet 模型进行前向传播，获取输出结果
        outputs = self.mpnet(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 从模型输出中提取序列输出
        sequence_output = outputs[0]

        # 使用 QA 输出层计算 logits
        logits = self.qa_outputs(sequence_output)
        # 将 logits 按最后一个维度分割为起始和结束 logits
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        # 去除多余的维度，确保与标签维度一致
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)
        loss = None

        # 如果提供了起始和结束位置的标签，则计算损失
        if start_positions is not None and end_positions is not None:
            labels = {"start_position": start_positions, "end_position": end_positions}
            loss = self.hf_compute_loss(labels, (start_logits, end_logits))

        # 如果 return_dict=False，则组织输出结果
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict=True，则构建 TFQuestionAnsweringModelOutput 对象返回
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
        # 如果存在 MPNet 模型，则构建 MPNet 模型
        if getattr(self, "mpnet", None) is not None:
            with tf.name_scope(self.mpnet.name):
                self.mpnet.build(None)
        # 如果存在 QA 输出层，则构建 QA 输出层
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.config.hidden_size])
```