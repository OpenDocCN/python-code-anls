# `.\models\convbert\modeling_tf_convbert.py`

```
# 设置编码格式为 UTF-8
# 版权声明及许可证信息
from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np  # 导入 NumPy 库，并使用 np 别名
import tensorflow as tf  # 导入 TensorFlow 库，并使用 tf 别名

# 导入必要的模块和类
from ...activations_tf import get_tf_activation  # 从 activations_tf 模块导入 get_tf_activation 函数
from ...modeling_tf_outputs import (  # 从 modeling_tf_outputs 模块导入多个模型输出类
    TFBaseModelOutput,
    TFMaskedLMOutput,
    TFMultipleChoiceModelOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from ...modeling_tf_utils import (  # 从 modeling_tf_utils 模块导入多个工具类和函数
    TFMaskedLanguageModelingLoss,
    TFModelInputType,
    TFMultipleChoiceLoss,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFSequenceSummary,
    TFTokenClassificationLoss,
    get_initializer,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import (  # 从 tf_utils 模块导入多个工具函数
    check_embeddings_within_bounds,
    shape_list,
    stable_softmax,
)
from ...utils import (  # 从 utils 模块导入多个工具函数
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from .configuration_convbert import ConvBertConfig  # 从 configuration_convbert 模块导入 ConvBertConfig 类


logger = logging.get_logger(__name__)  # 获取模块的日志记录器对象


_CHECKPOINT_FOR_DOC = "YituTech/conv-bert-base"  # 预训练模型的路径用于文档生成
_CONFIG_FOR_DOC = "ConvBertConfig"  # 模型配置类名用于文档生成

# 支持的预训练模型列表
TF_CONVBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "YituTech/conv-bert-base",
    "YituTech/conv-bert-medium-small",
    "YituTech/conv-bert-small",
    # See all ConvBERT models at https://huggingface.co/models?filter=convbert
]


# 以下类用于构建 ConvBERT 模型的嵌入层
# 从 transformers.models.albert.modeling_tf_albert.TFAlbertEmbeddings 复制并修改了部分代码
class TFConvBertEmbeddings(tf.keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config: ConvBertConfig, **kwargs):  # 初始化函数
        super().__init__(**kwargs)  # 调用父类的初始化函数

        self.config = config  # 保存配置对象到实例属性
        self.embedding_size = config.embedding_size  # 保存嵌入维度到实例属性
        self.max_position_embeddings = config.max_position_embeddings  # 保存最大位置嵌入数到实例属性
        self.initializer_range = config.initializer_range  # 保存初始化范围到实例属性
        # 创建 LayerNormalization 层，并保存到实例属性
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)  # 创建 Dropout 层并保存到实例属性
    # 定义 build 方法，在此方法中构建模型的嵌入层
    def build(self, input_shape=None):
        # 使用 tf.name_scope 定义 "word_embeddings" 的命名域
        with tf.name_scope("word_embeddings"):
            # 添加权重参数，用于嵌入词向量
            self.weight = self.add_weight(
                name="weight",
                shape=[self.config.vocab_size, self.embedding_size],
                initializer=get_initializer(self.initializer_range),
            )

        # 使用 tf.name_scope 定义 "token_type_embeddings" 的命名域
        with tf.name_scope("token_type_embeddings"):
            # 添加权重参数，用于嵌入类型向量
            self.token_type_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.config.type_vocab_size, self.embedding_size],
                initializer=get_initializer(self.initializer_range),
            )

        # 使用 tf.name_scope 定义 "position_embeddings" 的命名域
        with tf.name_scope("position_embeddings"):
            # 添加权重参数，用于嵌入位置向量
            self.position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_position_embeddings, self.embedding_size],
                initializer=get_initializer(self.initializer_range),
            )

        # 如果已经构建过模型，则直接返回，不再执行后续操作
        if self.built:
            return
        self.built = True
        
        # 如果存在 LayerNorm 层，则在命名域内构建 LayerNorm 层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.embedding_size])

    # 定义 call 方法，用于对输入进行嵌入
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
        # 如果没有给定 input_ids 或 inputs_embeds，则抛出 ValueError
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Need to provide either `input_ids` or `input_embeds`.")

        # 如果给定 input_ids，则对其进行范围检查
        if input_ids is not None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            # 使用 input_ids 从权重参数中取出对应的词向量
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        input_shape = shape_list(inputs_embeds)[:-1]

        # 如果没有给定 token_type_ids，则默认为全零向量
        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)

        # 如果没有给定 position_ids，则根据输入形状创建位置向量
        if position_ids is None:
            position_ids = tf.expand_dims(
                tf.range(start=past_key_values_length, limit=input_shape[1] + past_key_values_length), axis=0
            )

        # 从位置嵌入中取出对应的位置向量
        position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
        # 从类型嵌入中取出对应的类型向量
        token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)
        # 最终嵌入结果为词向量、位置向量和类型向量的相加
        final_embeddings = inputs_embeds + position_embeds + token_type_embeds
        # 对最终嵌入结果进行 LayerNorm 处理
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        # 对最终嵌入结果进行 dropout 处理
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)

        # 返回最终嵌入结果
        return final_embeddings
class TFConvBertSelfAttention(tf.keras.layers.Layer):
    # 定义 TFConvBertSelfAttention 类，继承自 tf.keras.layers.Layer
    def __init__(self, config, **kwargs):
        # 初始化函数，接受 config 参数和其他关键字参数
        super().__init__(**kwargs)
        # 调用父类的初始化函数

        if config.hidden_size % config.num_attention_heads != 0:
            # 如果 hidden_size 不能整除 num_attention_heads，抛出数值错误
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        new_num_attention_heads = int(config.num_attention_heads / config.head_ratio)
        if new_num_attention_heads < 1:
            # 如果新的 attention_heads 小于1，则更新 head_ratio 和 num_attention_heads
            self.head_ratio = config.num_attention_heads
            num_attention_heads = 1
        else:
            num_attention_heads = new_num_attention_heads
            self.head_ratio = config.head_ratio

        self.num_attention_heads = num_attention_heads
        # 设置 num_attention_heads
        self.conv_kernel_size = config.conv_kernel_size
        # 设置 conv_kernel_size

        if config.hidden_size % self.num_attention_heads != 0:
            # 如果 hidden_size 不能整除 num_attention_heads，抛出数值错误
            raise ValueError("hidden_size should be divisible by num_attention_heads")

        self.attention_head_size = config.hidden_size // config.num_attention_heads
        # 计算 attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # 计算 all_head_size
        self.query = tf.keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        # 创建 Dense 层，用于查询

        self.key = tf.keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key"
        )
        # 创建 Dense 层，用于键

        self.value = tf.keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )
        # 创建 Dense 层，用于值

        self.key_conv_attn_layer = tf.keras.layers.SeparableConv1D(
            self.all_head_size,
            self.conv_kernel_size,
            padding="same",
            activation=None,
            depthwise_initializer=get_initializer(1 / self.conv_kernel_size),
            pointwise_initializer=get_initializer(config.initializer_range),
            name="key_conv_attn_layer",
        )
        # 创建可分离卷积层，用��处理键的注意力层

        self.conv_kernel_layer = tf.keras.layers.Dense(
            self.num_attention_heads * self.conv_kernel_size,
            activation=None,
            name="conv_kernel_layer",
            kernel_initializer=get_initializer(config.initializer_range),
        )
        # 创建 Dense 层，用于卷积核层

        self.conv_out_layer = tf.keras.layers.Dense(
            self.all_head_size,
            activation=None,
            name="conv_out_layer",
            kernel_initializer=get_initializer(config.initializer_range),
        )
        # 创建 Dense 层，用于卷积输出层

        self.dropout = tf.keras.layers.Dropout(config.attention_probs_dropout_prob)
        # 创建 Dropout 层，用于注意力概率的丢弃
        self.config = config
        # 保存配置信息

    def transpose_for_scores(self, x, batch_size):
        # 定义转置函数，将输入 x 从 [batch_size, seq_length, all_head_size] 转换到 [batch_size, seq_length, num_attention_heads, attention_head_size]
        x = tf.reshape(x, (batch_size, -1, self.num_attention_heads, self.attention_head_size))
        # 重塑张量形状
        return tf.transpose(x, perm=[0, 2, 1, 3])
        # 返回转置后的张量
    # 构建神经网络模型
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回，不重复构建
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在查询参数，则构建查询参数
        if getattr(self, "query", None) is not None:
            # 设置查询参数的命名空间，并构建查询参数
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.config.hidden_size])
        # 如果存在键参数，则构建键参数
        if getattr(self, "key", None) is not None:
            # 设置键参数的命名空间，并构建键参数
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.config.hidden_size])
        # 如果存在值参数，则构建值参数
        if getattr(self, "value", None) is not None:
            # 设置值参数的命名空间，并构建值参数
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.config.hidden_size])
        # 如果存在键卷积注意力层，则构建键卷积注意力层
        if getattr(self, "key_conv_attn_layer", None) is not None:
            # 设置键卷积注意力层的命名空间，并构建键卷积注意力层
            with tf.name_scope(self.key_conv_attn_layer.name):
                self.key_conv_attn_layer.build([None, None, self.config.hidden_size])
        # 如果存在卷积核层，则构建卷积核层
        if getattr(self, "conv_kernel_layer", None) is not None:
            # 设置卷积核层的命名空间，并构建卷积核层
            with tf.name_scope(self.conv_kernel_layer.name):
                self.conv_kernel_layer.build([None, None, self.all_head_size])
        # 如果存在卷积输出层，则构建卷积输出层
        if getattr(self, "conv_out_layer", None) is not None:
            # 设置卷积输出层的命名空间，并构建卷积输出层
            with tf.name_scope(self.conv_out_layer.name):
                self.conv_out_layer.build([None, None, self.config.hidden_size])
# 定义 TFConvBertSelfOutput 类，继承自 tf.keras.layers.Layer
class TFConvBertSelfOutput(tf.keras.layers.Layer):
    # 初始化函数
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，用于映射隐藏状态的维度
        self.dense = tf.keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建一个 LayerNormalization 层，用于对数据进行归一化处理
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建一个丢弃层，用于在训练时随机丢弃一部分神经元，以减少过拟合
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        # 记录 config
        self.config = config
    
    # call() 方法定义了层的正向传播计算过程
    def call(self, hidden_states, input_tensor, training=False):
        # 先通过全连接层映射隐藏状态的维度
        hidden_states = self.dense(hidden_states)
        # 在训练阶段以一定的概率随机丢弃一部分神经元
        hidden_states = self.dropout(hidden_states, training=training)
        # 将丢弃后的结果与输入张量相加，并通过 LayerNorm 层进行归一化处理
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
    
    # 在 build() 方法中定义了自定义层中的所有变量
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                # 构建全连接层
                self.dense.build([None, None, self.config.hidden_size])
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                # 构建归一化层
                self.LayerNorm.build([None, None, self.config.hidden_size])


# 定义 TFConvBertAttention 类，继承自 tf.keras.layers.Layer
class TFConvBertAttention(tf.keras.layers.Layer):
    # 初始化函数
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        # 创建一个 TFConvBertSelfAttention 实例，用于实现自注意力机制
        self.self_attention = TFConvBertSelfAttention(config, name="self")
        # 创建一个 TFConvBertSelfOutput 实例，用于实现自注意力机制的输出层
        self.dense_output = TFConvBertSelfOutput(config, name="output")
    
    # prune_heads() 方法用于修剪注意力头
    def prune_heads(self, heads):
        raise NotImplementedError
    
    # call() 方法定义了层的正向传播计算过程
    def call(self, input_tensor, attention_mask, head_mask, output_attentions, training=False):
        # 调用自注意力层的 call() 方法实现自注意力机制的计算
        self_outputs = self.self_attention(
            input_tensor, attention_mask, head_mask, output_attentions, training=training
        )
        # 调用输出层的 call() 方法实现输出结果的计算
        attention_output = self.dense_output(self_outputs[0], input_tensor, training=training)
        # 将输出结果整合为一个元组并返回（如果需要输出 attention 矩阵，则将其加入到输出结果中）
        outputs = (attention_output,) + self_outputs[1:]

        return outputs
    
    # 在 build() 方法中定义了自定义层中的所有变量
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "self_attention", None) is not None:
            with tf.name_scope(self.self_attention.name):
                # 构建自注意力层
                self.self_attention.build(None)
        if getattr(self, "dense_output", None) is not None:
            with tf.name_scope(self.dense_output.name):
                # 构建输出层
                self.dense_output
    def build(self, input_shape=None):
        # 添加权重参数，用于存储卷积核的权重，初始化为指定形状的张量
        self.kernel = self.add_weight(
            "kernel",
            shape=[self.group_out_dim, self.group_in_dim, self.num_groups],
            initializer=self.kernel_initializer,
            trainable=True,
        )

        # 添加偏置参数，用于存储偏置值，初始化为指定形状的张量
        self.bias = self.add_weight(
            "bias", shape=[self.output_size], initializer=self.kernel_initializer, dtype=self.dtype, trainable=True
        )
        # 调用父类的 build 方法
        super().build(input_shape)

    def call(self, hidden_states):
        # 获取隐藏状态的批量大小
        batch_size = shape_list(hidden_states)[0]
        # 将隐藏状态重新排列成三维张量
        x = tf.transpose(tf.reshape(hidden_states, [-1, self.num_groups, self.group_in_dim]), [1, 0, 2])
        # 执行矩阵乘法操作
        x = tf.matmul(x, tf.transpose(self.kernel, [2, 1, 0]))
        # 对张量进行转置操作
        x = tf.transpose(x, [1, 0, 2])
        # 将张量重新reshape成指定形状
        x = tf.reshape(x, [batch_size, -1, self.output_size])
        # 将偏置加到张量上
        x = tf.nn.bias_add(value=x, bias=self.bias)
        # 返回处理后的张量
        return x
# 创建 TFConvBertIntermediate 类，继承自 tf.keras.layers.Layer
class TFConvBertIntermediate(tf.keras.layers.Layer):
    # 初始化方法，接受 config 和其他参数
    def __init__(self, config, **kwargs):
        # 调用父类初始化方法
        super().__init__(**kwargs)
        # 如果 config.num_groups 等于 1
        if config.num_groups == 1:
            # 创建一个全连接层，设置输出维度为 config.intermediate_size，初始化方式为指定的初始化器
            self.dense = tf.keras.layers.Dense(
                config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
            )
        # 否则
        else:
            # 创建一个分组线性层，设置输入维度为 config.hidden_size，输出维度为 config.intermediate_size，分组数为 config.num_groups
            # 初始化方式为指定的初始化器
            self.dense = GroupedLinearLayer(
                config.hidden_size,
                config.intermediate_size,
                num_groups=config.num_groups,
                kernel_initializer=get_initializer(config.initializer_range),
                name="dense",
            )

        # 如果 config.hidden_act 是字符串
        if isinstance(config.hidden_act, str):
            # 将 config.hidden_act 转换为对应的激活函数
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        # 否则
        else:
            # 直接使用 config.hidden_act 作为激活函数
            self.intermediate_act_fn = config.hidden_act
        # 保存 config 参数
        self.config = config

    # 调用方法，接受隐藏状态 hidden_states 作为输入
    def call(self, hidden_states):
        # 使用全连接层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # 使用激活函数处理全连接层输出
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

    # 构建方法，接受输入形状 input_shape
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 设置为已构建状态
        self.built = True
        # 如果有 self.dense 属性
        if getattr(self, "dense", None) is not None:
            # 在命名空间内构建全连接层，输入形状为 [None, None, self.config.hidden_size]
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# 创建 TFConvBertOutput 类，继承自 tf.keras.layers.Layer
class TFConvBertOutput(tf.keras.layers.Layer):
    # 初始化方法，接受 config 和其他参数
    def __init__(self, config, **kwargs):
        # 调用父类初始化方法
        super().__init__(**kwargs)

        # 如果 config.num_groups 等于 1
        if config.num_groups == 1:
            # 创建一个全连接层，设置输出维度为 config.hidden_size，初始化方式为指定的初始化器
            self.dense = tf.keras.layers.Dense(
                config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
            )
        # 否则
        else:
            # 创建一个分组线性层，设置输入维度为 config.intermediate_size，输出维度为 config.hidden_size，分组数为 config.num_groups
            # 初始化方式为指定的初始化器
            self.dense = GroupedLinearLayer(
                config.intermediate_size,
                config.hidden_size,
                num_groups=config.num_groups,
                kernel_initializer=get_initializer(config.initializer_range),
                name="dense",
            )
        # 创建一个 LayerNormalization 层，设置 epsilon 为 config.layer_norm_eps
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建一个 Dropout 层，设置 dropout 概率为 config.hidden_dropout_prob
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        # 保存 config 参数
        self.config = config

    # 调用方法，接受隐藏状态 hidden_states 和输入张量 input_tensor 以及是否训练状态 training 作为输入
    def call(self, hidden_states, input_tensor, training=False):
        # 使用全连接层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # 使用 Dropout 层处理隐藏状态，根据训练状态进行处理
        hidden_states = self.dropout(hidden_states, training=training)
        # 使用 LayerNormalization 层处理隐藏状态和输入张量的残差连接
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

    # 构建方法，接受输入形状 input_shape
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 设置为已构建状态
        self.built = True
        # 如果有 self.LayerNorm 属性
        if getattr(self, "LayerNorm", None) is not None:
            # 在命名空间内构建 LayerNormalization 层，输入形状为 [None, None, self.config.hidden_size]
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])
        # 如果有 self.dense 属性
        if getattr(self, "dense", None) is not None:
            # 在命名空间内构建全连接层，输入形状为 [None, None, self.config.intermediate_size]
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])


class TFConvBertLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        # 调用父类的初始化方法，传入额外的关键字参数
        super().__init__(**kwargs)

        # 创建 TFConvBertAttention 对象，并命名为 "attention"
        self.attention = TFConvBertAttention(config, name="attention")
        # 创建 TFConvBertIntermediate 对象，并命名为 "intermediate"
        self.intermediate = TFConvBertIntermediate(config, name="intermediate")
        # 创建 TFConvBertOutput 对象，并命名为 "output"
        self.bert_output = TFConvBertOutput(config, name="output")

    def call(self, hidden_states, attention_mask, head_mask, output_attentions, training=False):
        # 调用 attention 对象的 call 方法，并传入相应参数
        attention_outputs = self.attention(
            hidden_states, attention_mask, head_mask, output_attentions, training=training
        )
        # 获取 attention_outputs 的第一个元素作为 attention_output
        attention_output = attention_outputs[0]
        # 将 attention_output 传入 intermediate 对象
        intermediate_output = self.intermediate(attention_output)
        # 将 intermediate_output 和 attention_output 传入 bert_output 中
        layer_output = self.bert_output(intermediate_output, attention_output, training=training)
        # 如果需要输出 attentions，则在 outputs 中添加 attentions
        outputs = (layer_output,) + attention_outputs[1:]

        return outputs

    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 attention 对象，则构建 attention
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        # 如果存在 intermediate 对象，则构建 intermediate
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        # 如果存在 bert_output 对象，则构建 bert_output
        if getattr(self, "bert_output", None) is not None:
            with tf.name_scope(self.bert_output.name):
                self.bert_output.build(None)
# 定义一个名为TFConvBertEncoder的类，继承自tf.keras.layers.Layer
class TFConvBertEncoder(tf.keras.layers.Layer):
    # 构造方法，初始化函数
    def __init__(self, config, **kwargs):
        # 调用父类的构造方法
        super().__init__(**kwargs)

        # 创建一个名为layer的列表，包含config.num_hidden_layers个TFConvBertLayer对象
        self.layer = [TFConvBertLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]

    # 定义调用函数
    def call(
        self,
        hidden_states,
        attention_mask,
        head_mask,
        output_attentions,
        output_hidden_states,
        return_dict,
        training=False,
    ):
        # 如果output_hidden_states为True，则初始化all_hidden_states为空元组，否则为None
        all_hidden_states = () if output_hidden_states else None
        # 如果output_attentions为True，则初始化all_attentions为空元组，否则为None
        all_attentions = () if output_attentions else None

        # 遍历self.layer中的每个layer_module
        for i, layer_module in enumerate(self.layer):
            # 如果output_hidden_states为True，则将当前hidden_states添加到all_hidden_states中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 调用layer_module的call函数，得到layer_outputs
            layer_outputs = layer_module(
                hidden_states, attention_mask, head_mask[i], output_attentions, training=training
            )
            # 更新hidden_states为layer_outputs的第一个元素
            hidden_states = layer_outputs[0]

            # 如果output_attentions为True，则将layer_outputs的第二个元素添加到all_attentions中
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # 如果output_hidden_states为True，则将最终的hidden_states添加到all_hidden_states中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果return_dict为False，返回非None的v元素组成的元组
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

        # 返回TFBaseModelOutput对象
        return TFBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )

    # 构建函数
    def build(self, input_shape=None):
        # 如果已经构建，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 遍历self.layer中的每个layer
        if getattr(self, "layer", None) is not None:
            for layer in self.layer:
                # 为每个layer创建命名空间和构建该layer
                with tf.name_scope(layer.name):
                    layer.build(None)


# 定义一个名为TFConvBertPredictionHeadTransform的类，继承自tf.keras.layers.Layer
class TFConvBertPredictionHeadTransform(tf.keras.layers.Layer):
    # 构造方法，初始化函数
    def __init__(self, config, **kwargs):
        # 调用父类的构造方法
        super().__init__(**kwargs)

        # 创建一个全连接层dense
        self.dense = tf.keras.layers.Dense(
            config.embedding_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        # 根据config中的hidden_act初始化transform_act_fn
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.transform_act_fn = config.hidden_act

        # 创建一个LayerNormalization层LayerNorm
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 保存config配置
        self.config = config

    # 定义调用函数
    def call(self, hidden_states):
        # 通过全连接层dense处理hidden_states
        hidden_states = self.dense(hidden_states)
        # 通过transform_act_fn激活函数对hidden_states进行激活
        hidden_states = self.transform_act_fn(hidden_states)
        # 通过LayerNorm进行层归一化处理
        hidden_states = self.LayerNorm(hidden_states)

        # 返回处理后的hidden_states
        return hidden_states
    # 构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回
        if self.built:
            return
        # 将模型置为已构建状态
        self.built = True
        # 如果存在 dense 层
        if getattr(self, "dense", None) is not None:
            # 在 dense 层的命名空间中构建 dense 层
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 如果存在 LayerNorm 层
        if getattr(self, "LayerNorm", None) is not None:
            # 在 LayerNorm 层的命名空间中构建 LayerNorm 层
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])
# 基于 Keras 的可序列化类装饰器，用于对 TFConvBertMainLayer 类进行序列化
@keras_serializable
# 定义 TFConvBertMainLayer 类，继承自 tf.keras.layers.Layer 类
class TFConvBertMainLayer(tf.keras.layers.Layer):
    # 类属性，将 ConvBertConfig 类赋值给 config_class
    config_class = ConvBertConfig

    # 初始化方法
    def __init__(self, config, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 创建 TFConvBertEmbeddings 类对象，命名为 embeddings
        self.embeddings = TFConvBertEmbeddings(config, name="embeddings")

        # 如果 embedding_size 不等于 hidden_size
        if config.embedding_size != config.hidden_size:
            # 创建 Dense 层对象，命名为 embeddings_project
            self.embeddings_project = tf.keras.layers.Dense(config.hidden_size, name="embeddings_project")

        # 创建 TFConvBertEncoder 类对象，命名为 encoder
        self.encoder = TFConvBertEncoder(config, name="encoder")
        # 设置类属性 config 等于传入的 config
        self.config = config

    # 获取输入嵌入层的方法
    def get_input_embeddings(self):
        return self.embeddings

    # 设置输入嵌入层的方法
    def set_input_embeddings(self, value):
        self.embeddings.weight = value
        self.embeddings.vocab_size = value.shape[0]

    # 对模型的 heads 进行修剪的私有方法
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 抛出未实现异常
        raise NotImplementedError

    # 获取扩展的注意力掩码的方法
    def get_extended_attention_mask(self, attention_mask, input_shape, dtype):
        # 如果注意力掩码为 None，则使用 1 填充
        if attention_mask is None:
            attention_mask = tf.fill(input_shape, 1)

        # 创建一个 3D 的注意力掩码
        extended_attention_mask = tf.reshape(attention_mask, (input_shape[0], 1, 1, input_shape[1]))

        # 将注意力掩码转换成指定类型的张量
        extended_attention_mask = tf.cast(extended_attention_mask, dtype)
        # 将注意力掩码中的位置为 0 的地方替换为 -10000.0，用于从原始得分中移除这些位置
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask

    # 获取头掩码的方法
    def get_head_mask(self, head_mask):
        # 如果头掩码不为 None，则抛出未实现异常
        if head_mask is not None:
            raise NotImplementedError
        # 否则创建一个长度等于隐藏层数的 None 列表
        else:
            head_mask = [None] * self.config.num_hidden_layers

        return head_mask

    # 调用方法
    @unpack_inputs
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
        ):
        # 如果同时指定了 input_ids 和 inputs_embeds，则抛出数值错误
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        # 如果只指定了 input_ids，则获取其形状
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        # 如果只指定了 inputs_embeds，则获取其形状并去除最后一维
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        # 如果既没有指定 input_ids 也没有指定 inputs_embeds，则抛出数值错误
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # 如果没有指定 attention_mask，则创建形状与 input_shape 相同的填充为1的张量
        if attention_mask is None:
            attention_mask = tf.fill(input_shape, 1)

        # 如果没有指定 token_type_ids，则创建形状与 input_shape 相同的填充为0的张量
        if token_type_ids is None:
            token_type_ids = tf.fill(input_shape, 0)

        # 使用模型的 embeddings 方法得到隐藏状态
        hidden_states = self.embeddings(input_ids, position_ids, token_type_ids, inputs_embeds, training=training)
        # 获取扩展的注意力掩码
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, hidden_states.dtype)
        # 获取头部掩码
        head_mask = self.get_head_mask(head_mask)

        # 如果模型有 embeddings_project 属性，则对隐藏状态进行处理
        if hasattr(self, "embeddings_project"):
            hidden_states = self.embeddings_project(hidden_states, training=training)

        # 使用 encoder 处理隐藏状态
        hidden_states = self.encoder(
            hidden_states,
            extended_attention_mask,
            head_mask,
            output_attentions,
            output_hidden_states,
            return_dict,
            training=training,
        )

        # 返回隐藏状态

    def build(self, input_shape=None):
        # 如果已经构建过模型，则直接返回
        if self.built:
            return
        # 将模型标记为已构建
        self.built = True
        # 如果 embeddings 属性存在，则构建 embeddings
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        # 如果 encoder 属性存在，则构建 encoder
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 如果 embeddings_project 属性存在，则构建 embeddings_project
        if getattr(self, "embeddings_project", None) is not None:
            with tf.name_scope(self.embeddings_project.name):
                self.embeddings_project.build([None, None, self.config.embedding_size])
# 定义 TFConvBertPreTrainedModel 类，用于初始化权重和处理预训练模型的下载和加载
class TFConvBertPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置配置类为 ConvBertConfig
    config_class = ConvBertConfig
    # 设置基础模型前缀为 "convbert"
    base_model_prefix = "convbert"


# 起始的文档字符串，提供有关模型的信息，继承自 TFPreTrainedModel
CONVBERT_START_DOCSTRING = r"""

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

    Args:
        config ([`ConvBertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 输入文档字符串
CONVBERT_INPUTS_DOCSTRING = r"""
"""


@add_start_docstrings(
    "The bare ConvBERT Model transformer outputting raw hidden-states without any specific head on top.",
    CONVBERT_START_DOCSTRING,
)
# 定义 TFConvBertModel 类，输出不带特定头部的原始隐藏状态的 ConvBERT 模型变压器
class TFConvBertModel(TFConvBertPreTrainedModel):
    # 初始化方法，接收配置和其他输入参数
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法，传递配置和其他输入参数
        super().__init__(config, *inputs, **kwargs)

        # 创建一个 ConvBert 主层，并命名为"convbert"
        self.convbert = TFConvBertMainLayer(config, name="convbert")

    # 定义 call 方法，处理模型的前向传播
    @unpack_inputs
    # 添加模型前向传播的文档字符串
    @add_start_docstrings_to_model_forward(CONVBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 添加代码示例的文档字符串
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: Optional[Union[np.array, tf.Tensor]] = None,
        token_type_ids: Optional[Union[np.array, tf.Tensor]] = None,
        position_ids: Optional[Union[np.array, tf.Tensor]] = None,
        head_mask: Optional[Union[np.array, tf.Tensor]] = None,
        inputs_embeds: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        # 调用 ConvBert 主层的 call 方法，进行前向传播
        outputs = self.convbert(
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

        # 返回模型输出结果
        return outputs

    # 构建模型方法
    def build(self, input_shape=None):
        # 如果已经构建过模型，直接返回
        if self.built:
            return
        # 标记模型为已构建
        self.built = True
        # 如果存在 ConvBert 层
        if getattr(self, "convbert", None) is not None:
            # 在命名空间下构建 ConvBert 层
            with tf.name_scope(self.convbert.name):
                # 构建 ConvBert 层，传入输入形状为 None
                self.convbert.build(None)
# 定义一个 TFConvBertMaskedLMHead 类，继承自 tf.keras 的 Layer 类
class TFConvBertMaskedLMHead(tf.keras.layers.Layer):
    # 初始化方法，接受 config 和 input_embeddings 参数
    def __init__(self, config, input_embeddings, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        
        # 将参数赋值给对象属性
        self.config = config
        self.embedding_size = config.embedding_size
        self.input_embeddings = input_embeddings

    # build 方法，构建层
    def build(self, input_shape):
        # 添加偏置项作为层的可训练参数
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="bias")

        # 调用父类的 build 方法
        super().build(input_shape)

    # 获取输出的嵌入层对象
    def get_output_embeddings(self):
        return self.input_embeddings

    # 设置输出的嵌入层对象
    def set_output_embeddings(self, value):
        self.input_embeddings.weight = value
        self.input_embeddings.vocab_size = shape_list(value)[0]

    # 获取偏置项
    def get_bias(self):
        return {"bias": self.bias}

    # 设置偏置项
    def set_bias(self, value):
        self.bias = value["bias"]
        self.config.vocab_size = shape_list(value["bias"])[0]

    # call 方法，定义层的前向传播逻辑
    def call(self, hidden_states):
        seq_length = shape_list(tensor=hidden_states)[1]
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.embedding_size])
        hidden_states = tf.matmul(a=hidden_states, b=self.input_embeddings.weight, transpose_b=True)
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)

        # 返回处理后的 hidden_states
        return hidden_states


# 定义一个 TFConvBertGeneratorPredictions 类，继承自 tf.keras 的 Layer 类
class TFConvBertGeneratorPredictions(tf.keras.layers.Layer):
    # 初始化方法，接受 config 参数
    def __init__(self, config, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        
        # 创建 LayerNormalization 层和 Dense 层，将其赋给对象属性
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dense = tf.keras.layers.Dense(config.embedding_size, name="dense")
        self.config = config

    # call 方法，定义层的前向传播逻辑
    def call(self, generator_hidden_states, training=False):
        hidden_states = self.dense(generator_hidden_states)
        hidden_states = get_tf_activation("gelu")(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)

        # 返回处理后的 hidden_states
        return hidden_states

    # build 方法，构建层
    def build(self, input_shape=None):
        # 判断是否已经构建，若已构建则直接返回
        if self.built:
            return
        self.built = True
        # 构建 LayerNormalization 层和 Dense 层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.embedding_size])
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size)


# 为 TFConvBertForMaskedLM 类添加文档字符串，并继承自 TFConvBertPreTrainedModel 和 TFMaskedLanguageModelingLoss
@add_start_docstrings("""ConvBERT Model with a `language modeling` head on top.""", CONVBERT_START_DOCSTRING)
class TFConvBertForMaskedLM(TFConvBertPreTrainedModel, TFMaskedLanguageModelingLoss):
    # 省略了类的具体实现
    # 构造函数，初始化模型
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的构造函数进行初始化
        super().__init__(config, **kwargs)

        # 初始化模型配置
        self.config = config
        # 创建一个 TFConvBertMainLayer 对象，传入配置和名称
        self.convbert = TFConvBertMainLayer(config, name="convbert")
        # 创建一个 TFConvBertGeneratorPredictions 对象，传入配置和名称
        self.generator_predictions = TFConvBertGeneratorPredictions(config, name="generator_predictions")

        # 如果隐藏激活函数是字符串，则获取对应的激活函数
        if isinstance(config.hidden_act, str):
            self.activation = get_tf_activation(config.hidden_act)
        # 否则直接使用配置中的激活函数
        else:
            self.activation = config.hidden_act

        # 创建一个 TFConvBertMaskedLMHead 对象，传入配置、词嵌入层和名称
        self.generator_lm_head = TFConvBertMaskedLMHead(config, self.convbert.embeddings, name="generator_lm_head")

    # 获取语言模型头部
    def get_lm_head(self):
        return self.generator_lm_head

    # 获取前缀偏置的名称
    def get_prefix_bias_name(self):
        return self.name + "/" + self.generator_lm_head.name

    # 使用装饰器对 call 方法进行配置和注解
    @unpack_inputs
    @add_start_docstrings_to_model_forward(CONVBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFMaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # call 方法定义，接受多个输入参数
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: tf.Tensor | None = None,
        training: Optional[bool] = False,
    def compute_loss(self, input_ids: tf.Tensor,
                     labels: tf.Tensor = None,
                     ) -> Union[Tuple, TFMaskedLMOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            用于计算掩码语言建模损失的标签。索引应为 `[-100, 0, ..., config.vocab_size]`（参见`input_ids`文档）。索引设置为`-100`的标记被忽略（掩码），仅针对具有`[0, ..., config.vocab_size]`标签的标记计算损失
        """
        # 使用ConvBERT模型处理输入并生成隐藏状态
        generator_hidden_states = self.convbert(
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
        # 提取生成器的序列输出
        generator_sequence_output = generator_hidden_states[0]
        # 生成器预测的分数
        prediction_scores = self.generator_predictions(generator_sequence_output, training=training)
        # 生成器语言模型的头部处理预测分数
        prediction_scores = self.generator_lm_head(prediction_scores, training=training)
        # 计算损失，如果没有标签则为None
        loss = None if labels is None else self.hf_compute_loss(labels, prediction_scores)

        # 如果不返回字典，返回(prediction_scores, generator_hidden_states中除去生成器的其他隐藏状态)
        if not return_dict:
            output = (prediction_scores,) + generator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回带有损失、逻辑、隐藏状态和注意力的TFMaskedLMOutput对象
        return TFMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=generator_hidden_states.hidden_states,
            attentions=generator_hidden_states.attentions,
        )

    # 构建方法，用于构建模型的层
    def build(self, input_shape=None):
        # 如果已经构建过了，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在convbert模型，构建convbert模型
        if getattr(self, "convbert", None) is not None:
            with tf.name_scope(self.convbert.name):
                self.convbert.build(None)
        # 如果存在generator_predictions模型，构建generator_predictions模型
        if getattr(self, "generator_predictions", None) is not None:
            with tf.name_scope(self.generator_predictions.name):
                self.generator_predictions.build(None)
        # 如果存在generator_lm_head模型，构建generator_lm_head模型
        if getattr(self, "generator_lm_head", None) is not None:
            with tf.name_scope(self.generator_lm_head.name):
                self.generator_lm_head.build(None)
class TFConvBertClassificationHead(tf.keras.layers.Layer):
    """定义一个用于句子级分类任务的头部。"""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，输出维度为config.hidden_size，使用初始化器初始化权重，命名为"dense"
        self.dense = tf.keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 获取分类器的dropout值，如果config.classifier_dropout不为空，则使用该值，否则使用config.hidden_dropout_prob
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = tf.keras.layers.Dropout(classifier_dropout)  # 创建一个dropout层
        # 创建一个全连接层，输出维度为config.num_labels，使用初始化器初始化权重，命名为"out_proj"
        self.out_proj = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="out_proj"
        )

        self.config = config  # 保存配置信息

    def call(self, hidden_states, **kwargs):
        x = hidden_states[:, 0, :]  # 取出项为< s >的标记（等同于[CLS]）
        x = self.dropout(x)  # 对x进行dropout处理
        x = self.dense(x)  # 将x输入全连接层
        x = get_tf_activation(self.config.hidden_act)(x)  # 对x应用激活函数
        x = self.dropout(x)  # 再次对x进行dropout处理
        x = self.out_proj(x)  # 将x输入输出层

        return x  # 返回处理后的输出

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):  # 使用全连接层的名字作为命名空间
                self.dense.build([None, None, self.config.hidden_size])  # 构建全连接层
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):  # 使用输出层的名字作为命名空间
                self.out_proj.build([None, None, self.config.hidden_size])  # 构建输出层


@add_start_docstrings(
    """
    ConvBERT Model transformer with a sequence classification/regression head on top e.g., for GLUE tasks.
    """,
    CONVBERT_START_DOCSTRING,  # 添加Transformer模型的起始文档和ConvBERT模型特有的起始文档
)
class TFConvBertForSequenceClassification(TFConvBertPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels  # 保存标签数量
        self.convbert = TFConvBertMainLayer(config, name="convbert")  # 创建ConvBert主层
        self.classifier = TFConvBertClassificationHead(config, name="classifier")  # 创建分类头部

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CONVBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))  # 添加模型前向传播的文档字符串
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,  # 添加代码示例的文档字符串
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,  # 输入的token IDs，可以为空
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 注意力掩码，可以为空
        token_type_ids: np.ndarray | tf.Tensor | None = None,  # token类型IDs，可以为空
        position_ids: np.ndarray | tf.Tensor | None = None,  # 位置IDs，可以为空
        head_mask: np.ndarray | tf.Tensor | None = None,  # 头部掩码，可以为空
        inputs_embeds: tf.Tensor | None = None,  # 输入的嵌入向量，可以为空
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否以字典形式返回结果
        labels: tf.Tensor | None = None,  # 用于计算损失的标签，可以为空
        training: Optional[bool] = False,  # 是否处于训练模式，默认为False
    ) -> Union[Tuple, TFSequenceClassifierOutput]:  # 返回值可以是元组或TFSequenceClassifierOutput对象
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 使用ConvBERT模型进行前向传播
        outputs = self.convbert(
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
        # 使用分类器对输出进行分类
        logits = self.classifier(outputs[0], training=training)
        # 如果存在标签，则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果不以字典形式返回结果
        if not return_dict:
            # 构建输出元组
            output = (logits,) + outputs[1:]
            # 返回损失和输出，如果损失不为None，则将损失添加到输出中
            return ((loss,) + output) if loss is not None else output

        # 以字典形式返回结果
        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):  # 构建模型
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 设置模型为已构建状态
        self.built = True
        # 构建ConvBERT模型
        if getattr(self, "convbert", None) is not None:
            with tf.name_scope(self.convbert.name):
                self.convbert.build(None)
        # 构建分类器
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build(None)
# 使用指定的文档字符串初始化 TFConvBertForMultipleChoice 类
# 这个类是在 ConvBERT 模型的基础上添加了一个多项选择分类头部的模型，分类头部包括一个线性层和 softmax 函数
# 用于 RocStories/SWAG 等任务
class TFConvBertForMultipleChoice(TFConvBertPreTrainedModel, TFMultipleChoiceLoss):
    # 初始化函数，接受配置参数 config 和其他输入
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 创建 ConvBERT 主层对象，并命名为 convbert
        self.convbert = TFConvBertMainLayer(config, name="convbert")
        # 创建序列摘要对象，用于生成序列摘要特征
        self.sequence_summary = TFSequenceSummary(
            config, initializer_range=config.initializer_range, name="sequence_summary"
        )
        # 创建分类器，一个全连接层，用于多项选择任务的分类
        self.classifier = tf.keras.layers.Dense(
            1, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        # 保存配置对象
        self.config = config

    # 调用方法，接受输入并返回模型的输出
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        # 添加输入文档字符串，说明输入参数的含义
        CONVBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    @add_code_sample_docstrings(
        # 添加代码示例的文档字符串
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
        inputs_embeds: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: tf.Tensor | None = None,
        training: Optional[bool] = False,
```   
    ) -> Union[Tuple, TFMultipleChoiceModelOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
            where `num_choices` is the size of the second dimension of the input tensors. (See `input_ids` above)
        """
        # 如果输入的 input_ids 不为空，则获取 num_choices 和 seq_length
        if input_ids is not None:
            num_choices = shape_list(input_ids)[1]
            seq_length = shape_list(input_ids)[2]
        else:
            # 如果输入的 input_ids 为空，则获取 num_choices 和 seq_length
            num_choices = shape_list(inputs_embeds)[1]
            seq_length = shape_list(inputs_embeds)[2]

        # 将输入的 input_ids 重塑成二维张量，形状为 (-1, seq_length)，如果 input_ids 为空则为 None
        flat_input_ids = tf.reshape(input_ids, (-1, seq_length)) if input_ids is not None else None
        # 将输入的 attention_mask 重塑成二维张量，形状为 (-1, seq_length)，如果 attention_mask 为空则为 None
        flat_attention_mask = tf.reshape(attention_mask, (-1, seq_length)) if attention_mask is not None else None
        # 将输入的 token_type_ids 重塑成二维张量，形状为 (-1, seq_length)，如果 token_type_ids 为空则为 None
        flat_token_type_ids = tf.reshape(token_type_ids, (-1, seq_length)) if token_type_ids is not None else None
        # 将输入的 position_ids 重塑成二维张量，形状为 (-1, seq_length)，如果 position_ids 为空则为 None
        flat_position_ids = tf.reshape(position_ids, (-1, seq_length)) if position_ids is not None else None
        # 将输入的 inputs_embeds 重塑成三维张量，形状为 (-1, seq_length, hidden_size)，如果 inputs_embeds 为空则为 None
        flat_inputs_embeds = (
            tf.reshape(inputs_embeds, (-1, seq_length, shape_list(inputs_embeds)[3]))
            if inputs_embeds is not None
            else None
        )
        # 使用 ConvBert 模型处理输入数据
        outputs = self.convbert(
            flat_input_ids,
            flat_attention_mask,
            flat_token_type_ids,
            flat_position_ids,
            head_mask,
            flat_inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 使用序列摘要模块处理模型输出，得到 logits
        logits = self.sequence_summary(outputs[0], training=training)
        # 使用分类器对 logits 进行分类
        logits = self.classifier(logits)
        # 将 logits 重塑成二维张量，形状为 (-1, num_choices)
        reshaped_logits = tf.reshape(logits, (-1, num_choices))
        # 如果 labels 不为空，则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, reshaped_logits)

        # 如果不返回字典，则按原始输出格式返回结果
        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]

            return ((loss,) + output) if loss is not None else output

        # 返回字典格式的输出
        return TFMultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回
        if self.built:
            return
        # 设置模型为已构建状态
        self.built = True
        # 如果 ConvBert 模型存在，则构建 ConvBert 模型
        if getattr(self, "convbert", None) is not None:
            with tf.name_scope(self.convbert.name):
                self.convbert.build(None)
        # 如果序列摘要模块存在，则构建序列摘要模块
        if getattr(self, "sequence_summary", None) is not None:
            with tf.name_scope(self.sequence_summary.name):
                self.sequence_summary.build(None)
        # 如果分类器存在，则构建分类器
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
# 在 ConvBERT 模型的基础上加上一个面向标记分类任务的头部（即在隐藏状态输出之上的线性层），例如用于命名实体识别（NER）任务。
# 在 TFConvBertPreTrainedModel 和 TFTokenClassificationLoss 的基础上创建 ConvBERT 模型
class TFConvBertForTokenClassification(TFConvBertPreTrainedModel, TFTokenClassificationLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 初始化标签数量
        self.num_labels = config.num_labels
        # 创建 ConvBERT 主层
        self.convbert = TFConvBertMainLayer(config, name="convbert")
        # 设置分类器的 dropout
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 创建 dropout 层
        self.dropout = tf.keras.layers.Dropout(classifier_dropout)
        # 创建分类器
        self.classifier = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        # 保存配置
        self.config = config

    # 将输入参数展开，并添加模型详情的文档字符串
    @unpack_inputs
    @add_start_docstrings_to_model_forward(CONVBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 模型调用方法
    def call(
        self,
        input_ids: TFModelInputType | None = None,  # 输入的词元 ID
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 注意力遮盖
        token_type_ids: np.ndarray | tf.Tensor | None = None,  # 词元类型 ID
        position_ids: np.ndarray | tf.Tensor | None = None,  # 位置 ID
        head_mask: np.ndarray | tf.Tensor | None = None,  # 头遮盖
        inputs_embeds: tf.Tensor | None = None,  # 输入嵌入
        output_attentions: Optional[bool] = None,  # 输出注意力
        output_hidden_states: Optional[bool] = None,  # 输出隐藏状态
        return_dict: Optional[bool] = None,  # 返回结果字典
        labels: tf.Tensor | None = None,  # 标签
        training: Optional[bool] = False,  # 是否训练
    ) -> Union[Tuple, TFTokenClassifierOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 调用 ConvBert 模型进行前向传播，得到输出
        outputs = self.convbert(
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
        # 从输出中提取序列输出
        sequence_output = outputs[0]
        # 使用 dropout 对序列输出进行处理，防止过拟合
        sequence_output = self.dropout(sequence_output, training=training)
        # 将处理后的序列输出传递给分类器，得到 logits
        logits = self.classifier(sequence_output)
        # 如果存在标签，计算损失；否则损失为 None
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果不返回字典，则将结果按顺序组成元组返回
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TFTokenClassifierOutput 对象，其中包含损失、logits、隐藏状态和注意力权重
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
        # 标记为已构建
        self.built = True
        # 构建 ConvBert 模型
        if getattr(self, "convbert", None) is not None:
            with tf.name_scope(self.convbert.name):
                self.convbert.build(None)
        # 构建分类器
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
```  
# 使用 add_start_docstrings 装饰器添加模型描述文档字符串，描述了 ConvBERT 模型与其适用的任务
# 继承自 TFConvBertPreTrainedModel 和 TFQuestionAnsweringLoss 类
class TFConvBertForQuestionAnswering(TFConvBertPreTrainedModel, TFQuestionAnsweringLoss):
    # 初始化方法
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 从配置中获取标签数
        self.num_labels = config.num_labels
        # 创建 ConvBERT 主层对象
        self.convbert = TFConvBertMainLayer(config, name="convbert")
        # 创建用于输出答案的全连接层
        self.qa_outputs = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )
        # 将配置保存在对象中
        self.config = config

    # 使用 unpack_inputs 装饰器
    # 添加模型前向传播的文档字符串
    # 添加代码示例的文档字符串，包括检查点、输出类型和配置类
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        start_positions: tf.Tensor | None = None,
        end_positions: tf.Tensor | None = None,
        training: Optional[bool] = False,
    ) -> Union[Tuple, TFQuestionAnsweringModelOutput]:
        r"""
        start_positions (`tf.Tensor` of shape `(batch_size,)`, *optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`tf.Tensor` of shape `(batch_size,)`, *optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        outputs = self.convbert(
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
        # 获取模型的输出序列
        sequence_output = outputs[0]
        # 使用qa_outputs处理sequence_output得到logits
        logits = self.qa_outputs(sequence_output)
        # 将logits在最后一个维度上均分为两份，得到start_logits和end_logits
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        # 去掉start_logits和end_logits的单维度
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)
        loss = None

        if start_positions is not None and end_positions is not None:
            labels = {"start_position": start_positions}
            labels["end_position"] = end_positions
            # 使用hf_compute_loss计算loss
            loss = self.hf_compute_loss(labels, (start_logits, end_logits))

        # 如果return_dict为False，则输出(loss, start_logits, end_logits)和其他输出
        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果return_dict为True，则返回TFQuestionAnsweringModelOutput格式的结果
        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 构建模型，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果convbert存在，则构建convbert模型
        if getattr(self, "convbert", None) is not None:
            with tf.name_scope(self.convbert.name):
                self.convbert.build(None)
        # 如果qa_outputs存在，则构建qa_outputs模型
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.config.hidden_size])
```