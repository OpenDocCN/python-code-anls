# `.\models\efficientformer\modeling_tf_efficientformer.py`

```
# 设置文件编码为 UTF-8
# 版权声明，版权归 Snapchat Research 和 The HuggingFace Inc. 团队所有
#
# 根据 Apache License, Version 2.0 许可，除非遵守许可规定，否则不得使用此文件
# 可在以下网址获取许可副本：http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或事先书面同意，根据许可分发的软件基于“原样”提供，
# 没有任何形式的担保或条件，明示或暗示
# 有关特定语言的具体许可内容，请查看许可内容并查看许可规定的限制
""" TensorFlow EfficientFormer model."""

# 导入所需的模块
import itertools
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import tensorflow as tf

from ...activations_tf import ACT2FN
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFBaseModelOutputWithPooling,
    TFImageClassifierOutput,
)
from ...modeling_tf_utils import (
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    get_initializer,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from .configuration_efficientformer import EfficientFormerConfig

# 获取 logger 对象用于记录日志
logger = logging.get_logger(__name__)

# General docstring
# 文档字符串
_CONFIG_FOR_DOC = "EfficientFormerConfig"

# Base docstring
# 基本文档字符串
_CHECKPOINT_FOR_DOC = "snap-research/efficientformer-l1-300"
_EXPECTED_OUTPUT_SHAPE = [1, 49, 448]

# Image classification docstring
# 图像分类文档字符串
_IMAGE_CLASS_CHECKPOINT = "snap-research/efficientformer-l1-300"
_IMAGE_CLASS_EXPECTED_OUTPUT = "LABEL_281"

# 预训练模型的存档列表
TF_EFFICIENTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "snap-research/efficientformer-l1-300",
    # See all EfficientFormer models at https://huggingface.co/models?filter=efficientformer
]

# 定义一个 TensorFlow 层，用于在两个阶段之间进行下采样
class TFEfficientFormerPatchEmbeddings(tf.keras.layers.Layer):
    """
    This class performs downsampling between two stages. For the input tensor with the shape [batch_size, num_channels,
    height, width] it produces output tensor with the shape [batch_size, num_channels, height/stride, width/stride]
    """

    def __init__(
        self, config: EfficientFormerConfig, num_channels: int, embed_dim: int, apply_norm: bool = True, **kwargs
```  
    # 构造函数，初始化方法
    def __init__(self, num_channels: int, embed_dim: int, apply_norm: bool = True, **kwargs) -> None:
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 设置通道数
        self.num_channels = num_channels

        # 使用零填充2D层，padding为config.downsample_pad
        self.padding = tf.keras.layers.ZeroPadding2D(padding=config.downsample_pad)
        # 创建卷积层，用于将输入数据投影到embed_dim维度
        self.projection = tf.keras.layers.Conv2D(
            filters=embed_dim,
            kernel_size=config.downsample_patch_size,
            strides=config.downsample_stride,
            padding="valid",
            name="projection",
        )
        # 使用与PyTorch等价的BatchNormalization，默认的momentum和epsilon
        self.norm = (
            tf.keras.layers.BatchNormalization(axis=-1, epsilon=config.batch_norm_eps, momentum=0.9, name="norm")
            if apply_norm
            else tf.identity
        )
        # 设置嵌入维度
        self.embed_dim = embed_dim

    # 前向传播方法
    def call(self, pixel_values: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 断言输入张量的形状是否正确
        tf.debugging.assert_shapes(
            [(pixel_values, (..., None, None, self.num_channels))],
            message="Make sure that the channel dimension of the pixel values match with the one set in the configuration.",
        )
        # 计算投影后的嵌入
        embeddings = self.projection(self.padding(pixel_values))
        # 对嵌入进行归一化处理
        embeddings = self.norm(embeddings, training=training)
        # 返回处理后的嵌入
        return embeddings

    # 构建方法
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 将标记设置为已构建
        self.built = True
        # 如果存在投影层，则构建投影层
        if getattr(self, "projection", None) is not None:
            with tf.name_scope(self.projection.name):
                self.projection.build([None, None, None, self.num_channels])
        # 如果存在归一化层，则构建归一化层
        if getattr(self, "norm", None) is not None:
            if hasattr(self.norm, "name"):
                with tf.name_scope(self.norm.name):
                    self.norm.build([None, None, None, self.embed_dim])
# 定义自注意力层类，用于在EfficientFormer模型中实现自注意力机制
class TFEfficientFormerSelfAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        dim: int,  # 输入维度
        key_dim: int,  # 关键向量的维度
        num_heads: int,  # 头数
        attention_ratio: int,  # 注意力扩展比率
        resolution: int,  # 分辨率
        config: EfficientFormerConfig,  # 模型配置
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_heads = num_heads  # 初始化头数
        self.key_dim = key_dim  # 初始化关键向量维度
        self.attention_ratio = attention_ratio  # 初始化注意力扩展比率
        self.scale = key_dim**-0.5  # 缩放系数
        self.total_key_dim = key_dim * num_heads  # 总关键向量维度
        self.expanded_key_dim = int(attention_ratio * key_dim)  # 扩展后的关键向量维度
        self.total_expanded_key_dim = int(self.expanded_key_dim * num_heads)  # 总扩展后的关键向量维度
        hidden_size = self.total_expanded_key_dim + self.total_key_dim * 2  # 隐藏层大小

        # 创建用于计算查询、键和值的全连接层
        self.qkv = tf.keras.layers.Dense(
            units=hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="qkv"
        )
        # 创建投影层
        self.projection = tf.keras.layers.Dense(
            units=dim, kernel_initializer=get_initializer(config.initializer_range), name="projection"
        )
        self.resolution = resolution  # 初始化分辨率
        self.dim = dim  # 初始化输入维度

    def build(self, input_shape: tf.TensorShape) -> None:
        # 生成点的坐标组合列表
        points = list(itertools.product(range(self.resolution), range(self.resolution)))
        num_points = len(points)  # 点的数量
        attention_offsets = {}  # 初始化注意力偏置字典

        idxs = []  # 初始化索引列表

        # 遍历所有点的组合，计算它们之间的偏移，并存储偏移对应的索引
        for point_1 in points:
            for point_2 in points:
                offset = (abs(point_1[0] - point_2[0]), abs(point_1[1] - point_2[1]))  # 计算偏移
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)  # 若偏移不在字典中，则添加
                idxs.append(attention_offsets[offset])  # 添加索引

        # 创建注意力偏置权重
        self.attention_biases = self.add_weight(
            shape=(self.num_heads, len(attention_offsets)),
            initializer=tf.keras.initializers.zeros(),
            trainable=True,
            name="attention_biases",
        )
        # 创建注意力偏置索引
        self.attention_bias_idxs = self.add_weight(
            shape=(num_points, num_points),
            trainable=False,
            dtype=tf.int32,
            name="attention_bias_idxs",
        )

        # 将索引列表转换为张量，并赋给注意力偏置索引
        self.attention_bias_idxs.assign(tf.reshape(tf.cast(idxs, dtype=tf.int32), (num_points, num_points)))

        # 如果已经构建，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在查询、键和值的全连接层，则构建它们
        if getattr(self, "qkv", None) is not None:
            with tf.name_scope(self.qkv.name):
                self.qkv.build([None, None, self.dim])
        # 如果存在投影层，则构建它
        if getattr(self, "projection", None) is not None:
            with tf.name_scope(self.projection.name):
                self.projection.build([None, None, self.total_expanded_key_dim])

    def call(
        self, hidden_states: tf.Tensor, output_attentions: bool = False, training: bool = False
        # 实现自注意力计算
    # 定义一个函数，参数包括 self 和 hidden_states，返回类型为一个元组，包含一个 TensorFlow 张量
    def call(self, hidden_states: tf.Tensor, output_attentions: bool = False) -> Tuple[tf.Tensor]:
        # 获取 hidden_states 的形状
        batch_size, sequence_length, *_ = shape_list(hidden_states)
        # 使用 self.qkv 对象处理 hidden_states，并将结果保存在 qkv 中
        qkv = self.qkv(inputs=hidden_states)

        # 将 qkv 分割成 query_layer, key_layer, value_layer，并对 key_layer 和 value_layer 进行维度变换
        query_layer, key_layer, value_layer = tf.split(
            tf.reshape(tensor=qkv, shape=(batch_size, sequence_length, self.num_heads, -1)),
            num_or_size_splits=[self.key_dim, self.key_dim, self.expanded_key_dim],
            axis=3,
        )

        # 对 query_layer, key_layer, value_layer 进行维度变换
        query_layer = tf.transpose(query_layer, perm=[0, 2, 1, 3])
        key_layer = tf.transpose(key_layer, perm=[0, 2, 1, 3])
        value_layer = tf.transpose(value_layer, perm=[0, 2, 1, 3])

        # 计算注意力得分
        attention_probs = tf.matmul(query_layer, tf.transpose(key_layer, perm=[0, 1, 3, 2]))
        # 对注意力得分进行缩放
        scale = tf.cast(self.scale, dtype=attention_probs.dtype)
        attention_probs = tf.multiply(attention_probs, scale)

        # 获取注意力偏置并用于调整注意力得分
        attention_biases = tf.gather(params=self.attention_biases, indices=self.attention_bias_idxs, axis=1)
        attention_probs = attention_probs + attention_biases
        # 对注意力得分进行稳定的 softmax 处理
        attention_probs = stable_softmax(logits=attention_probs, axis=-1)

        # 使用注意力得分计算上下文向量
        context_layer = tf.matmul(attention_probs, value_layer)
        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])

        # 对上下文向量进行形状变换并使用投影层处理
        context_layer = tf.reshape(
            tensor=context_layer, shape=(batch_size, sequence_length, self.total_expanded_key_dim)
        )
        context_layer = self.projection(context_layer)

        # 根据条件返回不同的输出
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        
        return outputs
class TFEfficientFormerConvStem(tf.keras.layers.Layer):
    # 初始化方法，用于创建卷积层的 stem（卷积干部）部分
    def __init__(self, config: EfficientFormerConfig, out_channels: int, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 添加一层零填充
        self.padding = tf.keras.layers.ZeroPadding2D(padding=1)
        # 添加第一个卷积层
        self.convolution1 = tf.keras.layers.Conv2D(
            filters=out_channels // 2, kernel_size=3, strides=2, padding="valid", name="convolution1"
        )
        # 使用与 PyTorch 等效的 BatchNormalization 的默认动量和 epsilon
        self.batchnorm_before = tf.keras.layers.BatchNormalization(
            axis=-1, epsilon=config.batch_norm_eps, momentum=0.9, name="batchnorm_before"
        )

        # 添加第二个卷积层
        self.convolution2 = tf.keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=3,
            strides=2,
            padding="valid",
            name="convolution2",
        )
        # 使用与 PyTorch 等效的 BatchNormalization 的默认动量和 epsilon
        self.batchnorm_after = tf.keras.layers.BatchNormalization(
            axis=-1, epsilon=config.batch_norm_eps, momentum=0.9, name="batchnorm_after"
        )

        # 添加激活函数层
        self.activation = tf.keras.layers.Activation(activation=tf.keras.activations.relu, name="activation")
        self.out_channels = out_channels
        self.config = config

    # 前向传播方法
    def call(self, pixel_values: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 在输入数据上进行零填充，并通过第一个卷积层和批标准化层
        features = self.batchnorm_before(self.convolution1(self.padding(pixel_values)), training=training)
        features = self.activation(features)
        # 通过第二个卷积层和批标准化层
        features = self.batchnorm_after(self.convolution2(self.padding(features)), training=training)
        features = self.activation(features)
        return features

    # 构建方法，用于构建层的权重
    def build(self, input_shape=None):
        # 如果已经构建，则直接返回
        if self.built:
            return
        # 设置构建标志为 True
        self.built = True
        # 构建第一个卷积层
        if getattr(self, "convolution1", None) is not None:
            with tf.name_scope(self.convolution1.name):
                self.convolution1.build([None, None, None, self.config.num_channels])
        # 构建第一个批标准化层
        if getattr(self, "batchnorm_before", None) is not None:
            with tf.name_scope(self.batchnorm_before.name):
                self.batchnorm_before.build([None, None, None, self.out_channels // 2])
        # 构建第二个卷积层
        if getattr(self, "convolution2", None) is not None:
            with tf.name_scope(self.convolution2.name):
                self.convolution2.build([None, None, None, self.out_channels // 2])
        # 构建第二个批标准化层
        if getattr(self, "batchnorm_after", None) is not None:
            with tf.name_scope(self.batchnorm_after.name):
                self.batchnorm_after.build([None, None, None, self.out_channels])
        # 构建激活函数层
        if getattr(self, "activation", None) is not None:
            with tf.name_scope(self.activation.name):
                self.activation.build(None)


class TFEfficientFormerPooling(tf.keras.layers.Layer):
    # 初始化函数，传入参数池大小和其他关键字参数
    def __init__(self, pool_size: int, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)
        # 创建一个平均池化层，并设定池大小、步幅和填充方式
        self.pool = tf.keras.layers.AveragePooling2D(pool_size=pool_size, strides=1, padding="same")
    
    # 调用函数，传入隐藏状态张量，返回变换后的张量
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 使用平均池化层对隐藏状态进行池化操作
        output = self.pool(hidden_states)
        # 对池化后的结果减去原始隐藏状态，得到输出张量
        output = output - hidden_states
        # 返回变换后的张量
        return output
# 定义一个 TensorFlow Keras 层类，用于实现 EfficientFormer 中的 Dense MLP（多层感知机）
class TFEfficientFormerDenseMlp(tf.keras.layers.Layer):
    # 构造函数，接受配置、输入特征数、可选的隐藏特征数和输出特征数等参数
    def __init__(
        self,
        config: EfficientFormerConfig,  # EfficientFormer 的配置对象
        in_features: int,  # 输入特征的数量
        hidden_features: Optional[int] = None,  # 隐藏层特征数量（可选）
        out_features: Optional[int] = None,  # 输出特征数量（可选）
        **kwargs,  # 其他额外的参数
    ):
        # 调用父类的构造函数
        super().__init__(**kwargs)
        # 如果未提供输出特征数，则使用输入特征数
        out_features = out_features or in_features
        # 如果未提供隐藏特征数，则使用输入特征数
        hidden_features = hidden_features or in_features

        # 定义第一个 Dense 层，用于转换输入特征到隐藏特征
        self.linear_in = tf.keras.layers.Dense(
            units=hidden_features,  # 隐藏特征的数量
            kernel_initializer=get_initializer(config.initializer_range),  # 权重初始化
            name="linear_in"  # 层的名称
        )
        # 定义激活函数，根据配置选择
        self.activation = ACT2FN[config.hidden_act]
        # 定义 Dropout 层，防止过拟合
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

        # 定义第二个 Dense 层，用于转换隐藏特征到输出特征
        self.linear_out = tf.keras.layers.Dense(
            units=out_features,  # 输出特征的数量
            kernel_initializer=get_initializer(config.initializer_range),  # 权重初始化
            name="linear_out"  # 层的名称
        )
        # 保存隐藏特征和输入特征的数量
        self.hidden_features = hidden_features
        self.in_features = in_features

    # 定义前向传播函数，接受输入的隐藏状态和训练标记
    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 使用第一个 Dense 层进行特征转换
        hidden_states = self.linear_in(inputs=hidden_states)
        # 应用激活函数
        hidden_states = self.activation(hidden_states)
        # 应用 Dropout，如果在训练模式下
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 使用第二个 Dense 层进行特征转换
        hidden_states = self.linear_out(inputs=hidden_states)
        # 再次应用 Dropout，如果在训练模式下
        hidden_states = self.dropout(inputs=hidden_states, training=training)

        # 返回转换后的隐藏状态
        return hidden_states

    # 构建函数，用于在第一次使用该层时初始化
    def build(self, input_shape=None):
        # 如果已经构建过，则不再继续
        if self.built:
            return
        # 设置已构建标记
        self.built = True
        # 如果已定义 `linear_in`，则构建它
        if getattr(self, "linear_in", None) is not None:
            with tf.name_scope(self.linear_in.name):
                self.linear_in.build([None, None, self.in_features])
        # 如果已定义 `linear_out`，则构建它
        if getattr(self, "linear_out", None) is not None:
            with tf.name_scope(self.linear_out.name):
                self.linear_out.build([None, None, self.hidden_features])


# 定义一个 TensorFlow Keras 层类，用于实现 EfficientFormer 中的卷积 MLP
class TFEfficientFormerConvMlp(tf.keras.layers.Layer):
    # 构造函数，接受配置、输入特征数、可选的隐藏特征数和输出特征数、Dropout 等参数
    def __init__(
        config: EfficientFormerConfig,  # EfficientFormer 的配置对象
        in_features: int,  # 输入特征的数量
        hidden_features: Optional[int] = None,  # 隐藏层特征数量（可选）
        out_features: Optional[int] = None,  # 输出特征数量（可选）
        drop: float = 0.0,  # Dropout 比例
        **kwargs,  # 其他额外的参数
    # 定义一个继承自 tf.keras.layers.Layer 的新类，用于实现一个卷积块
    ):
        # 调用父类的构造函数，传入kwargs参数
        super().__init__(**kwargs)
        # 如果没有指定输出特征数，则默认为输入特征数
        out_features = out_features or in_features
        # 如果没有指定隐藏层特征数，则默认为输入特征数
        hidden_features = hidden_features or in_features

        # 创建第一个卷积层，使用隐藏特征数作为过滤器数量，1x1的卷积核大小，valid填充方式
        self.convolution1 = tf.keras.layers.Conv2D(
            filters=hidden_features,
            kernel_size=1,
            name="convolution1",
            padding="valid",
        )

        # 使用给定的激活函数名称从全局字典ACT2FN中获取对应的激活函数
        self.activation = ACT2FN[config.hidden_act]

        # 创建第二个卷积层，使用输出特征数作为过滤器数量，1x1的卷积核大小，valid填充方式
        self.convolution2 = tf.keras.layers.Conv2D(
            filters=out_features,
            kernel_size=1,
            name="convolution2",
            padding="valid",
        )

        # 创建一个Dropout层，用于在训练时进行随机失活，参数为指定的丢弃率
        self.dropout = tf.keras.layers.Dropout(rate=drop)

        # 创建BatchNormalization层，用于标准化前一个卷积层的输出
        # 使用与PyTorch相当的默认动量和epsilon参数
        self.batchnorm_before = tf.keras.layers.BatchNormalization(
            axis=-1, epsilon=config.batch_norm_eps, momentum=0.9, name="batchnorm_before"
        )
        # 创建BatchNormalization层，用于标准化后一个卷积层的输出
        # 使用与PyTorch相当的默认动量和epsilon参数
        self.batchnorm_after = tf.keras.layers.BatchNormalization(
            axis=-1, epsilon=config.batch_norm_eps, momentum=0.9, name="batchnorm_after"
        )
        # 记录隐藏层特征数、输入特征数和输出特征数
        self.hidden_features = hidden_features
        self.in_features = in_features
        self.out_features = out_features

    # 定义调用该卷积块时的操作
    def call(self, hidden_state: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 进行第一次卷积操作
        hidden_state = self.convolution1(hidden_state)
        # 对卷积输出进行BatchNormalization
        hidden_state = self.batchnorm_before(hidden_state, training=training)
        # 对卷积输出进行激活函数处理
        hidden_state = self.activation(hidden_state)
        # 对卷积输出进行Dropout操作
        hidden_state = self.dropout(hidden_state, training=training)
        # 进行第二次卷积操作
        hidden_state = self.convolution2(hidden_state)
        # 对卷积输出进行BatchNormalization
        hidden_state = self.batchnorm_after(hidden_state, training=training)
        # 对卷积输出进行Dropout操作
        hidden_state = self.dropout(hidden_state, training=training)
        # 返回处理后的结果
        return hidden_state

    # 构建网络层，指定输入形状并创建内部网络层
    def build(self, input_shape=None):
        # 如果已经构建过网络层，则直接返回
        if self.built:
            return
        # 标记网络层已经构建
        self.built = True
        # 如果存在第一个卷积层，则构建第一个卷积层
        if getattr(self, "convolution1", None) is not None:
            with tf.name_scope(self.convolution1.name):
                self.convolution1.build([None, None, None, self.in_features])
        # 如果存在第二个卷积层，则构建第二个卷积层
        if getattr(self, "convolution2", None) is not None:
            with tf.name_scope(self.convolution2.name):
                self.convolution2.build([None, None, None, self.hidden_features])
        # 如果存在前BatchNormalization层，则构建前BatchNormalization层
        if getattr(self, "batchnorm_before", None) is not None:
            with tf.name_scope(self.batchnorm_before.name):
                self.batchnorm_before.build([None, None, None, self.hidden_features])
        # 如果存在后BatchNormalization层，则构建后BatchNormalization层
        if getattr(self, "batchnorm_after", None) is not None:
            with tf.name_scope(self.batchnorm_after.name):
                self.batchnorm_after.build([None, None, None, self.out_features])
# 定义一个名为TFEfficientFormerDropPath的类，继承自tf.keras.layers.Layer类，
# 用于在每个样本中应用主路径中的随机深度（Stochastic Depth）（当应用于残差块的主路径时）。
# 引用自github.com:rwightman/pytorch-image-models
class TFEfficientFormerDropPath(tf.keras.layers.Layer):

    def __init__(self, drop_path: float, **kwargs):
        super().__init__(**kwargs)
        self.drop_path = drop_path  # 初始化drop_path概率

    def call(self, x: tf.Tensor, training=None):
        if training:
            keep_prob = 1 - self.drop_path  # 计算保留概率
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)  # 创建形状信息的元组
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)  # 生成随机张量
            random_tensor = tf.floor(random_tensor)  # 对随机张量取整
            return (x / keep_prob) * random_tensor  # 返回随机深度之后的张量
        return x  # 若非训练状态，直接返回输入张量


class TFEfficientFormerFlat(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, hidden_states: tf.Tensor) -> Tuple[tf.Tensor]:
        batch_size, _, _, in_channels = shape_list(hidden_states)  # 获取形状信息
        hidden_states = tf.reshape(hidden_states, shape=[batch_size, -1, in_channels])  # 重塑张量形状
        return hidden_states  # 返回重塑后的张量


class TFEfficientFormerMeta3D(tf.keras.layers.Layer):
    def __init__(self, config: EfficientFormerConfig, dim: int, drop_path: float = 0.0, **kwargs):
        super().__init__(**kwargs)

        self.token_mixer = TFEfficientFormerSelfAttention(
            dim=config.dim,
            key_dim=config.key_dim,
            num_heads=config.num_attention_heads,
            attention_ratio=config.attention_ratio,
            resolution=config.resolution,
            name="token_mixer",
            config=config,
        )
        self.dim = dim  # 维度信息
        self.config = config  # 配置信息

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm1")  # 初始化LayerNormalization层
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm2")  # 初始化LayerNormalization层
        mlp_hidden_dim = int(dim * config.mlp_expansion_ratio)  # 计算MLP隐藏层维度
        self.mlp = TFEfficientFormerDenseMlp(config, in_features=dim, hidden_features=mlp_hidden_dim, name="mlp")  # 初始化TFEfficientFormerDenseMlp层

        # 使用`layers.Activation`替代`tf.identity`以更好地控制`training`的行为。
        self.drop_path = (
            TFEfficientFormerDropPath(drop_path)  # 若drop_path大于0.0��则初始化TFEfficientFormerDropPath层
            if drop_path > 0.0
            else tf.keras.layers.Activation("linear", name="drop_path")  # 否则初始化Activation层
        )
        self.config = config  # 保存配置信息
    def build(self, input_shape=None):
        # 初始化 layer_scale_1 和 layer_scale_2 为 None
        self.layer_scale_1 = None
        self.layer_scale_2 = None

        # 如果配置中开启了 layer_scale
        if self.config.use_layer_scale:
            # 添加 layer_scale_1 的权重，shape 为 (self.dim,)
            # 使用指定的常量初始化权重的值
            # 可以进行训练，命名为 "layer_scale_1"
            self.layer_scale_1 = self.add_weight(
                shape=(self.dim,),
                initializer=tf.keras.initializers.Constant(value=self.config.layer_scale_init_value),
                trainable=True,
                name="layer_scale_1",
            )
            # 添加 layer_scale_2 的权重，shape 为 (self.dim,)
            # 使用指定的常量初始化权重的值
            # 可以进行训练，命名为 "layer_scale_2"
            self.layer_scale_2 = self.add_weight(
                shape=(self.dim,),
                initializer=tf.keras.initializers.Constant(value=self.config.layer_scale_init_value),
                trainable=True,
                name="layer_scale_2",
            )

        # 如果已经构建，则直接返回
        if self.built:
            return
        # 设置已构建标志为 True
        self.built = True
        
        # 如果 token_mixer 存在，则构建 token_mixer 层
        if getattr(self, "token_mixer", None) is not None:
            with tf.name_scope(self.token_mixer.name):
                self.token_mixer.build(None)
        
        # 如果 layernorm1 存在，则构建 layernorm1 层
        if getattr(self, "layernorm1", None) is not None:
            with tf.name_scope(self.layernorm1.name):
                self.layernorm1.build([None, None, self.dim])
        
        # 如果 layernorm2 存在，则构建 layernorm2 层
        if getattr(self, "layernorm2", None) is not None:
            with tf.name_scope(self.layernorm2.name):
                self.layernorm2.build([None, None, self.dim])
        
        # 如果 mlp 存在，则构建 mlp 层
        if getattr(self, "mlp", None) is not None:
            with tf.name_scope(self.mlp.name):
                self.mlp.build(None)
        
        # 如果 drop_path 存在，则构建 drop_path 层
        if getattr(self, "drop_path", None) is not None:
            with tf.name_scope(self.drop_path.name):
                self.drop_path.build(None)

    def call(
        self, hidden_states: tf.Tensor, output_attentions: bool = False, training: bool = False
    ):
    # 定义 Transformer 层的调用方法，输入隐藏状态张量，返回元组（tf.Tensor）类型
    def call(self, inputs: tf.Tensor, training: bool = False, output_attentions: bool = False) -> Tuple[tf.Tensor]:
        # 使用 TokenMixer 处理输入隐藏状态张量，获取自注意力层输出
        self_attention_outputs = self.token_mixer(
            hidden_states=self.layernorm1(hidden_states, training=training),  # 使用层归一化处理隐藏状态
            output_attentions=output_attentions,  # 是否输出注意力权重
            training=training,  # 训练模式标志
        )

        # 获取自注意力层输出
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # 如果需要输出注意力权重，则将其加入输出中

        # 如果使用层尺度（layer scale）
        if self.config.use_layer_scale:
            # 对当前层输出进行层归一化和自注意力权重的叠加
            layer_output = hidden_states + self.drop_path(
                tf.expand_dims(tf.expand_dims(self.layer_scale_1, 0), 0) * attention_output,
                training=training,
            )
            # 对层输出进行层归一化和 MLP 处理，并将自注意力层输出与 MLP 输出相加
            layer_output = layer_output + self.drop_path(
                tf.expand_dims(tf.expand_dims(self.layer_scale_2, 0), 0)
                * self.mlp(hidden_states=self.layernorm2(inputs=layer_output, training=training), training=training),
                training=training,
            )
        else:
            # 对当前层输出进行自注意力权重的叠加
            layer_output = hidden_states + self.drop_path(attention_output, training=training)
            # 对层输出进行层归一化和 MLP 处理，并将 MLP 输出与当前层输出相加
            layer_output = layer_output + self.drop_path(
                self.mlp(hidden_states=self.layernorm2(inputs=layer_output, training=training), training=training),
                training=training,
            )

        # 将当前层输出加入输出元组中
        outputs = (layer_output,) + outputs

        # 返回输出元组
        return outputs
# 定义 TF Efficient Former 模型的 3D 元层（meta-layer）的类
class TFEfficientFormerMeta3DLayers(tf.keras.layers.Layer):
    def __init__(self, config: EfficientFormerConfig, **kwargs):
        super().__init__(**kwargs)
        # 计算每个块的丢弃路径（drop path）率
        drop_paths = [
            config.drop_path_rate * (block_idx + sum(config.depths[:-1]))
            for block_idx in range(config.num_meta3d_blocks)
        ]
        # 创建多个 3D 元层块
        self.blocks = [
            TFEfficientFormerMeta3D(config, config.hidden_sizes[-1], drop_path=drop_path, name=f"blocks.{i}")
            for i, drop_path in enumerate(drop_paths)
        ]

    # 模型的调用方法
    def call(
        self, hidden_states: tf.Tensor, output_attentions: bool = False, training: bool = False
    ) -> Tuple[tf.Tensor]:
        # 如果需要输出注意力信息，则初始化一个空元组
        all_attention_outputs = () if output_attentions else None

        # 遍历所有的 3D 元层块
        for i, layer_module in enumerate(self.blocks):
            # 如果隐藏状态是元组，则只选择第一个元素
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

            # 调用当前 3D 元层块，并更新隐藏状态
            hidden_states = layer_module(
                hidden_states=hidden_states, output_attentions=output_attentions, training=training
            )
            # 如果需要输出注意力信息，则记录当前层的注意力输出
            if output_attentions:
                all_attention_outputs = all_attention_outputs + (hidden_states[1],)

        # 如果需要输出注意力信息，则返回隐藏状态及所有注意力输出
        if output_attentions:
            outputs = (hidden_states[0],) + all_attention_outputs
            return outputs

        # 否则，只返回隐藏状态
        return hidden_states

    # 构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 blocks 属性，则构建每个块
        if getattr(self, "blocks", None) is not None:
            for layer in self.blocks:
                with tf.name_scope(layer.name):
                    layer.build(None)


# 定义 TF Efficient Former 模型的 4D 层
class TFEfficientFormerMeta4D(tf.keras.layers.Layer):
    def __init__(self, config: EfficientFormerConfig, dim: int, drop_path: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        # 获取池化大小，如果未指定，则默认为 3
        pool_size = config.pool_size if config.pool_size is not None else 3
        # 创建 token 混合层
        self.token_mixer = TFEfficientFormerPooling(pool_size=pool_size, name="token_mixer")
        # 设置维度
        self.dim = dim
        # 计算 MLP 隐藏层维度
        mlp_hidden_dim = int(dim * config.mlp_expansion_ratio)
        # 创建 MLP 层
        self.mlp = TFEfficientFormerConvMlp(
            config=config, in_features=dim, hidden_features=mlp_hidden_dim, drop=config.hidden_dropout_prob, name="mlp"
        )
        # 创建丢弃路径层
        self.drop_path = (
            TFEfficientFormerDropPath(drop_path, name="drop_path")
            if drop_path > 0.0
            else tf.keras.layers.Activation("linear", name="drop_path")
        )
        self.config = config
    # 构建方法，用于初始化层的参数和状态
    def build(self, input_shape=None):
        # 初始化层的缩放参数
        self.layer_scale_1 = None
        self.layer_scale_2 = None

        # 如果配置中使用了层缩放
        if self.config.use_layer_scale:
            # 添加第一个层缩放参数，并设置为可训练
            self.layer_scale_1 = self.add_weight(
                shape=(self.dim),
                initializer=tf.keras.initializers.Constant(value=self.config.layer_scale_init_value),
                trainable=True,
                name="layer_scale_1",
            )
            # 添加第二个层缩放参数，并设置为可训练
            self.layer_scale_2 = self.add_weight(
                shape=(self.dim),
                initializer=tf.keras.initializers.Constant(value=self.config.layer_scale_init_value),
                trainable=True,
                name="layer_scale_2",
            )

        # 如果层已构建完成，则直接返回
        if self.built:
            return
        # 将层标记为已构建
        self.built = True
        # 如果存在令牌混合器，则构建令牌混合器
        if getattr(self, "token_mixer", None) is not None:
            with tf.name_scope(self.token_mixer.name):
                self.token_mixer.build(None)
        # 如果存在 MLP，则构建 MLP
        if getattr(self, "mlp", None) is not None:
            with tf.name_scope(self.mlp.name):
                self.mlp.build(None)
        # 如果存在 drop_path，则构建 drop_path
        if getattr(self, "drop_path", None) is not None:
            with tf.name_scope(self.drop_path.name):
                self.drop_path.build(None)

    # 调用方法，用于执行层的前向传播逻辑
    def call(self, hidden_states: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor]:
        # 使用令牌混合器处理隐藏状态
        outputs = self.token_mixer(hidden_states)

        # 如果配置中使用了层缩放
        if self.config.use_layer_scale:
            # 计算第一个层输出，应用层缩放和 drop_path
            layer_output = hidden_states + self.drop_path(
                tf.expand_dims(tf.expand_dims(self.layer_scale_1, 0), 0) * outputs,
                training=training,
            )

            # 计算第二个层输出，应用层缩放、MLP 和 drop_path
            layer_output = layer_output + self.drop_path(
                tf.expand_dims(tf.expand_dims(self.layer_scale_2, 0), 0)
                * self.mlp(hidden_state=layer_output, training=training),
                training=training,
            )

        # 如果不使用层缩放
        else:
            # 计算层输出，应用 drop_path
            layer_output = hidden_states + self.drop_path(outputs, training=training)
            # 计算层输出，应用 MLP 和 drop_path
            layer_output = layer_output + self.drop_path(
                self.mlp(hidden_state=layer_output, training=training), training=training
            )

        # 返回层的输出
        return layer_output
# 定义一个自定义的 Keras 层，用于 EfficientFormer 模型的中间层
class TFEfficientFormerMeta4DLayers(tf.keras.layers.Layer):
    def __init__(self, config: EfficientFormerConfig, stage_idx: int, **kwargs):
        super().__init__(**kwargs)
        # 根据给定的配置和阶段索引计算要创建的层的数量
        num_layers = (
            config.depths[stage_idx] if stage_idx != -1 else config.depths[stage_idx] - config.num_meta3d_blocks
        )
        # 计算每个块的丢弃路径概率，根据层索引和总深度计算
        drop_paths = [
            config.drop_path_rate * (block_idx + sum(config.depths[:stage_idx])) for block_idx in range(num_layers)
        ]
        # 创建一系列 Meta4D 层组成的列表
        self.blocks = [
            TFEfficientFormerMeta4D(
                config=config, dim=config.hidden_sizes[stage_idx], drop_path=drop_paths[i], name=f"blocks.{i}"
            )
            for i in range(len(drop_paths))
        ]

    def call(self, hidden_states: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor]:
        # 逐层传递隐藏状态，应用每个 Meta4D 层
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states=hidden_states, training=training)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经构建了模型，直接返回
        if getattr(self, "blocks", None) is not None:
            # 对于每个 Meta4D 层，构建层
            for layer in self.blocks:
                with tf.name_scope(layer.name):
                    layer.build(None)


# 定义 EfficientFormer 模型的中间阶段的自定义 Keras 层
class TFEfficientFormerIntermediateStage(tf.keras.layers.Layer):
    def __init__(self, config: EfficientFormerConfig, index: int, **kwargs):
        super().__init__(**kwargs)
        # 创建中间阶段的 Meta4D 层
        self.meta4D_layers = TFEfficientFormerMeta4DLayers(config=config, stage_idx=index, name="meta4D_layers")

    def call(self, hidden_states: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor]:
        # 将隐藏状态传递给中间阶段的 Meta4D 层
        hidden_states = self.meta4D_layers(hidden_states=hidden_states, training=training)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经构建了模型，直接返回
        if getattr(self, "meta4D_layers", None) is not None:
            # 构建中间阶段的 Meta4D 层
            with tf.name_scope(self.meta4D_layers.name):
                self.meta4D_layers.build(None)


# 定义 EfficientFormer 模型的最后阶段的自定义 Keras 层
class TFEfficientFormerLastStage(tf.keras.layers.Layer):
    def __init__(self, config: EfficientFormerConfig, **kwargs):
        super().__init__(**kwargs)
        # 创建最后阶段的 Meta4D 层
        self.meta4D_layers = TFEfficientFormerMeta4DLayers(config=config, stage_idx=-1, name="meta4D_layers")
        # 创建扁平化层
        self.flat = TFEfficientFormerFlat(name="flat")
        # 创建 Meta3D 层
        self.meta3D_layers = TFEfficientFormerMeta3DLayers(config, name="meta3D_layers")

    def call(
        self, hidden_states: tf.Tensor, output_attentions: bool = False, training: bool = False
    ) -> Tuple[tf.Tensor]:
        # 将隐藏状态传递给最后阶段的 Meta4D 层
        hidden_states = self.meta4D_layers(hidden_states=hidden_states, training=training)
        # 执行扁平化操作
        hidden_states = self.flat(hidden_states=hidden_states)
        # 将扁平化后的隐藏状态传递给 Meta3D 层
        hidden_states = self.meta3D_layers(
            hidden_states=hidden_states, output_attentions=output_attentions, training=training
        )

        return hidden_states
    # 定义神经网络模型的构建方法，输入形状为可选参数
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果存在 meta4D_layers 属性，则在指定的命名空间下构建 meta4D_layers
        if getattr(self, "meta4D_layers", None) is not None:
            with tf.name_scope(self.meta4D_layers.name):
                self.meta4D_layers.build(None)
        # 如果存在 flat 属性，则在指定的命名空间下构建 flat 层
        if getattr(self, "flat", None) is not None:
            with tf.name_scope(self.flat.name):
                self.flat.build(None)
        # 如果存在 meta3D_layers 属性，则在指定的命名空间下构建 meta3D_layers
        if getattr(self, "meta3D_layers", None) is not None:
            with tf.name_scope(self.meta3D_layers.name):
                self.meta3D_layers.build(None)
# 定义一个自定义的 TensorFlow 层，用于 EfficientFormer 模型的编码器部分
class TFEfficientFormerEncoder(tf.keras.layers.Layer):
    # 初始化方法
    def __init__(self, config: EfficientFormerConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 存储模型配置信息
        self.config = config
        # 计算中间阶段的数量
        num_intermediate_stages = len(config.depths) - 1
        # 判断每个中间阶段是否需要下采样
        downsamples = [
            config.downsamples[i] or config.hidden_sizes[i] != config.hidden_sizes[i + 1]
            for i in range(num_intermediate_stages)
        ]

        # 存储中间阶段的层
        intermediate_stages = []
        layer_count = -1
        # 遍历中间阶段
        for i in range(num_intermediate_stages):
            layer_count += 1
            # 创建中间阶段的层并添加到列表中
            intermediate_stages.append(
                TFEfficientFormerIntermediateStage(config, i, name=f"intermediate_stages.{layer_count}")
            )
            # 如果需要下采样，则添加一个 Patch Embedding 层
            if downsamples[i]:
                layer_count += 1
                intermediate_stages.append(
                    TFEfficientFormerPatchEmbeddings(
                        config,
                        config.hidden_sizes[i],
                        config.hidden_sizes[i + 1],
                        name=f"intermediate_stages.{layer_count}",
                    )
                )
        # 存储中间阶段的层列表
        self.intermediate_stages = intermediate_stages
        # 创建最后一个阶段的层
        self.last_stage = TFEfficientFormerLastStage(config, name="last_stage")

    # 前向传播方法
    def call(
        self,
        hidden_states: tf.Tensor,
        output_hidden_states: bool,
        output_attentions: bool,
        return_dict: bool,
        training: bool = False,
    ) -> TFBaseModelOutput:
        # 如果需要输出隐藏状态，则创建一个空元组来存储所有隐藏状态
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力矩阵，则创建一个空元组来存储所有注意力矩阵
        all_self_attentions = () if output_attentions else None

        # 如果需要输出隐藏状态，则将初始隐藏状态加入到隐藏状态元组中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 遍历中间阶段的层，并进行前向传播
        for layer_module in self.intermediate_stages:
            hidden_states = layer_module(hidden_states, training=training)

            # 如果需要输出隐藏状态，则将当前阶段的隐藏状态加入到隐藏状态元组中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        # 对最后一个阶段进行前向传播
        layer_output = self.last_stage(hidden_states, output_attentions=output_attentions, training=training)

        # 如果需要输出注意力矩阵，则将注意力矩阵加入到注意力矩阵元组中
        if output_attentions:
            all_self_attentions = all_self_attentions + layer_output[1:]

        # 如果需要输出隐藏状态，则将最后一个阶段的隐藏状态加入到隐藏状态元组中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (layer_output[0],)

        # 如果不需要返回字典，则返回一个元组
        if not return_dict:
            return tuple(v for v in [layer_output[0], all_hidden_states, all_self_attentions] if v is not None)

        # 如果需要返回字典，则返回一个 TFBaseModelOutput 对象
        return TFBaseModelOutput(
            last_hidden_state=layer_output[0],
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    # 构建方法
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在最后一个阶段的层，则构建最后一个阶段的层
        if getattr(self, "last_stage", None) is not None:
            with tf.name_scope(self.last_stage.name):
                self.last_stage.build(None)
        # 遍历中间阶段的层，并构建每个中间阶段的层
        for layer in self.intermediate_stages:
            with tf.name_scope(layer.name):
                layer.build(None)


# 标记这个类是可序列化的
@keras_serializable
# 定义 TF EfficientFormer 主层，继承自 tf.keras.layers.Layer 类
class TFEfficientFormerMainLayer(tf.keras.layers.Layer):
    # 配置类为 EfficientFormerConfig
    config_class = EfficientFormerConfig

    # 初始化方法，接受 EfficientFormerConfig 类型的配置对象和其他关键字参数
    def __init__(self, config: EfficientFormerConfig, **kwargs) -> None:
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 将传入的配置对象赋值给实例的 config 属性
        self.config = config

        # 创建 patch_embed 层，使用 TFEfficientFormerConvStem 类
        self.patch_embed = TFEfficientFormerConvStem(config, config.hidden_sizes[0], name="patch_embed")
        # 创建 encoder 层，使用 TFEfficientFormerEncoder 类
        self.encoder = TFEfficientFormerEncoder(config, name="encoder")
        # 创建 layernorm 层，使用 LayerNormalization 类，设置 epsilon 参数为配置对象的 layer_norm_eps 属性值
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm")

    # 定义 call 方法，处理输入并返回输出
    @unpack_inputs
    def call(
        self,
        pixel_values: Optional[tf.Tensor] = None,
        output_attentions: Optional[tf.Tensor] = None,
        output_hidden_states: Optional[tf.Tensor] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    # 定义一个方法，用于推理或训练过程中对输入进行编码并生成输出
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor, ...]]:
        # 如果未指定输出注意力的设置，则采用模型配置中的设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        # 如果未指定输出隐藏状态的设置，则采用模型配置中的设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定返回字典的设置，则采用模型配置中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果未提供像素值，则抛出数值错误
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 当在 CPU 上运行时，tf.keras.layers.Conv2D 和 tf.keras.layers.AveragePool2D 不支持通道优先的 NCHW 格式。
        # 许多块同时包含两种格式。因此，将输入格式从 (batch_size, num_channels, height, width) 更改为
        # (batch_size, height, width, num_channels)。
        # shape = (batch_size, in_height, in_width, in_channels=num_channels)
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))
        # 使用 patch_embed 方法对像素值进行编码
        embedding_output = self.patch_embed(pixel_values, training=training)

        # 将编码后的像素值传递给编码器进行进一步处理
        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 获取编码器的序列输出
        sequence_output = encoder_outputs[0]
        # 对序列输出进行 layer normalization
        sequence_output = self.layernorm(sequence_output, training=training)

        # 如果需要输出隐藏状态，则将隐藏状态格式从 (batch_size, height, width, num_channels) 更改为
        # (batch_size, num_channels, height, width)
        if output_hidden_states:
            hidden_states = tuple([tf.transpose(h, perm=(0, 3, 1, 2)) for h in encoder_outputs[1][:-1]]) + (
                encoder_outputs[1][-1],
            )

        # 如果不需要返回字典，则返回编码器的序列输出和其他隐藏状态
        if not return_dict:
            head_outputs = (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        # 否则，返回 TFBaseModelOutput 对象，其中包含序列输出、隐藏状态和注意力权重
        return TFBaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=hidden_states if output_hidden_states else encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    # 构建模型
    def build(self, input_shape=None):
        # 如果已经构建，则直接返回
        if self.built:
            return
        # 将构建标志置为 True
        self.built = True
        # 如果存在 patch_embed 属性，则构建 patch_embed
        if getattr(self, "patch_embed", None) is not None:
            with tf.name_scope(self.patch_embed.name):
                self.patch_embed.build(None)
        # 如果存在 encoder 属性，则构建 encoder
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 如果存在 layernorm 属性，则构建 layernorm
        if getattr(self, "layernorm", None) is not None:
            with tf.name_scope(self.layernorm.name):
                # 构建 layernorm，输入形状为 [None, None, 最后一个隐藏层的大小]
                self.layernorm.build([None, None, self.config.hidden_sizes[-1]])
class TFEfficientFormerPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 定义一个配置类，用于处理模型的参数配置
    config_class = EfficientFormerConfig
    # 模型名称前缀
    base_model_prefix = "efficientformer"
    # 主输入名称
    main_input_name = "pixel_values"


EFFICIENTFORMER_START_DOCSTRING = r"""
    This model is a TensorFlow
    [tf.keras.layers.Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer). Use it as a regular
    TensorFlow Module and refer to the TensorFlow documentation for all matter related to general usage and behavior.


    Parameters:
        config ([`EfficientFormerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

EFFICIENTFORMER_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values ((`tf.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`EfficientFormerImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare EfficientFormer Model transformer outputting raw hidden-states without any specific head on top.",
    EFFICIENTFORMER_START_DOCSTRING,
)
# 定义TFEfficientFormerModel类，继承自TFEfficientFormerPreTrainedModel
class TFEfficientFormerModel(TFEfficientFormerPreTrainedModel):
    # 初始化方法
    def __init__(self, config: EfficientFormerConfig, **kwargs) -> None:
        # 调用父类的初始化方法
        super().__init__(config, **kwargs)
        # 定义efficientformer层
        self.efficientformer = TFEfficientFormerMainLayer(config, name="efficientformer")

    # call方法，处理输入并返回输出
    @unpack_inputs
    @add_start_docstrings_to_model_forward(EFFICIENTFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def call(
        self,
        pixel_values: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[Tuple, TFBaseModelOutput]:
        # 调用 efficientformer 模型进行推理，返回模型输出结果
        outputs = self.efficientformer(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        return outputs

    def build(self, input_shape=None):
        # 如果模型已经构建完成，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 efficientformer 模型，则构建 efficientformer 模型
        if getattr(self, "efficientformer", None) is not None:
            # 在 Tensorflow 图中创建一个名称范围，使用 efficientformer 模型的名称
            with tf.name_scope(self.efficientformer.name):
                # 构建 efficientformer 模型
                self.efficientformer.build(None)
@add_start_docstrings(
    """
    EfficientFormer Model transformer with an image classification head on top of pooled last hidden state, e.g. for
    ImageNet.
    """,
    EFFICIENTFORMER_START_DOCSTRING,
)
class TFEfficientFormerForImageClassification(TFEfficientFormerPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config: EfficientFormerConfig):
        super().__init__(config)

        self.num_labels = config.num_labels  # 存储模型配置文件中的标签数量
        self.efficientformer = TFEfficientFormerMainLayer(config, name="efficientformer")  # 创建主模型层

        # Classifier head
        self.classifier = (
            tf.keras.layers.Dense(config.num_labels, name="classifier")  # 如果标签数量大于零，则创建具有指定标签数量的密集层
            if config.num_labels > 0
            else tf.keras.layers.Activation("linear", name="classifier")  # 否则创建线性激活层
        )
        self.config = config  # 存储模型配置

    @unpack_inputs
    @add_start_docstrings_to_model_forward(EFFICIENTFORMER_INPUTS_DOCSTRING)  # 添加模型前向传播的文档字符串
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=TFImageClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )  # 添加代码示例文档字符串
    def call(
        self,
        pixel_values: Optional[tf.Tensor] = None,  # 输入像素值，可选
        labels: Optional[tf.Tensor] = None,  # 标签，可选
        output_attentions: Optional[bool] = None,  # 是否输出注意力值，可选
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选
        return_dict: Optional[bool] = None,  # 是否返回字典格式的输出，可选
        training: bool = False,  # 是否为训练模式
    ) -> Union[tf.Tensor, TFImageClassifierOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).  # 用于计算图像分类/回归损失的标签。索引应在 `[0, ..., config.num_labels - 1]`之间。如果`config.num_labels == 1`，则计算回归损失（均方损失）；如果`config.num_labels > 1`，则计算分类损失（交叉熵损失）。

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # 如果返回字典为空则使用模型配置中的默认值

        outputs = self.efficientformer(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )  # 调用主模型层得到输出

        sequence_output = outputs[0]  # 获取序列输出

        logits = self.classifier(tf.reduce_mean(sequence_output, axis=-2))  # 通过对序列输出进行池化并通过分类器得到logits

        loss = None if labels is None else self.hf_compute_loss(labels, logits)  # 如果没有标签，则损失为空，否则通过标签和logits计算损失

        if not return_dict:
            output = (logits,) + outputs[1:]  # 如果不返回字典，则将logits和其他输出组合成tuple
            return ((loss,) + output) if loss is not None else output  # 如果有损失，则将损失和output组合成tuple返回，否则返回output

        return TFImageClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )  # 返回TFImageClassifierOutput对象
    # 构建模型结构，如果已经构建过，则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 设置标志表示已经构建过
        self.built = True
        # 如果存在 efficientformer，则构建 efficientformer
        if getattr(self, "efficientformer", None) is not None:
            # 设置 efficientformer 的名字作为命名空间
            with tf.name_scope(self.efficientformer.name):
                # 构建 efficientformer
                self.efficientformer.build(None)
        # 如果存在 classifier，则构建 classifier
        if getattr(self, "classifier", None) is not None:
            # 如果 classifier 有名字属性，则使用其名字作为命名空间
            if hasattr(self.classifier, "name"):
                with tf.name_scope(self.classifier.name):
                    # 构建 classifier，输入 shape 为 [None, None, 隐藏层尺寸]
                    self.classifier.build([None, None, self.config.hidden_sizes[-1]])
# 使用 dataclass 装饰器定义类 TFEfficientFormerForImageClassificationWithTeacherOutput，该类是 ModelOutput 的子类
@dataclass
class TFEfficientFormerForImageClassificationWithTeacherOutput(ModelOutput):
    """
    Args:
    Output type of [`EfficientFormerForImageClassificationWithTeacher`].
        logits (`tf.Tensor` of shape `(batch_size, config.num_labels)`):
            Prediction scores as the average of the cls_logits and distillation logits.
        cls_logits (`tf.Tensor` of shape `(batch_size, config.num_labels)`):
            Prediction scores of the classification head (i.e. the linear layer on top of the final hidden state of the
            class token).
        distillation_logits (`tf.Tensor` of shape `(batch_size, config.num_labels)`):
            Prediction scores of the distillation head (i.e. the linear layer on top of the final hidden state of the
            distillation token).
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when
        `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer plus
            the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when
        `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    # 定义属性 logits、cls_logits、distillation_logits、hidden_states、attentions
    logits: tf.Tensor = None
    cls_logits: tf.Tensor = None
    distillation_logits: tf.Tensor = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None


# 通过 add_start_docstrings 给类 TFEfficientFormerForImageClassificationWithTeacher 添加相关文档
@add_start_docstrings(
    """
    EfficientFormer Model transformer with image classification heads on top (a linear layer on top of the final hidden
    state and a linear layer on top of the final hidden state of the distillation token) e.g. for ImageNet.

    .. warning::
            This model supports inference-only. Fine-tuning with distillation (i.e. with a teacher) is not yet
            supported.
    """,
    EFFICIENTFORMER_START_DOCSTRING,
)
# 定义类 TFEfficientFormerForImageClassificationWithTeacher，该类是 TFEfficientFormerPreTrainedModel 的子类
class TFEfficientFormerForImageClassificationWithTeacher(TFEfficientFormerPreTrainedModel):
    # 初始化方法，接受配置参数，并调用父类的初始化方法
    def __init__(self, config: EfficientFormerConfig) -> None:
        super().__init__(config)

        # 从配置参数中获取标签数量
        self.num_labels = config.num_labels
        # 创建 EfficientFormer 主层
        self.efficientformer = TFEfficientFormerMainLayer(config, name="efficientformer")

        # 分类器头部
        # 根据标签数量是否大于零创建分类器
        self.classifier = (
            tf.keras.layers.Dense(config.num_labels, name="classifier")
            if config.num_labels > 0
            else tf.keras.layers.Activation("linear", name="classifier")
        )
        # 根据标签数量是否大于零创建蒸馏分类器
        self.distillation_classifier = (
            tf.keras.layers.Dense(config.num_labels, name="distillation_classifier")
            if config.num_labels > 0
            else tf.keras.layers.Activation("linear", name="distillation_classifier")
        )

    # 装饰器，用于处理输入并添加文档字符串信息
    @unpack_inputs
    @add_start_docstrings_to_model_forward(EFFICIENTFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=TFEfficientFormerForImageClassificationWithTeacherOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    # 模型调用方法
    def call(
        self,
        pixel_values: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[tuple, TFEfficientFormerForImageClassificationWithTeacherOutput]:
        # 如果 return_dict 为 None，则使用配置参数中的 use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if training:
            # 如果处于训练模式，抛出异常，因为该模型仅支持推理
            raise Exception(
                "This model supports inference-only. Fine-tuning with distillation (i.e. with a teacher) is not yet supported."
            )

        # 使用 EfficientFormer 处理输入数据
        outputs = self.efficientformer(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        sequence_output = outputs[0]

        # 使用分类器预测类别
        cls_logits = self.classifier(tf.reduce_mean(sequence_output, axis=-2))
        # 使用蒸馏分类器预测类别
        distillation_logits = self.distillation_classifier(tf.reduce_mean(sequence_output, axis=-2))
        # 将分类和蒸馏分类的结果取平均作为最终预测结果
        logits = (cls_logits + distillation_logits) / 2

        if not return_dict:
            # 如果不需要返回字典形式的输出，则以元组方式返回
            output = (logits, cls_logits, distillation_logits) + outputs[1:]
            return output

        # 返回带有教师输出的 EfficientFormer 模型的输出
        return TFEfficientFormerForImageClassificationWithTeacherOutput(
            logits=logits,
            cls_logits=cls_logits,
            distillation_logits=distillation_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    # 建立模型，如果已经建立则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 设置已建立标志为 True
        self.built = True
        # 如果存在 efficientformer 属性
        if getattr(self, "efficientformer", None) is not None:
            # 使用 efficientformer 的名字为当前操作创建一个命名空间
            with tf.name_scope(self.efficientformer.name):
                # 构建 efficientformer 模型
                self.efficientformer.build(None)
        # 如果存在 classifier 属性
        if getattr(self, "classifier", None) is not None:
            # 如果 classifier 具有名字属性
            if hasattr(self.classifier, "name"):
                # 使用 classifier 的名字为当前操作创建一个命名空间
                with tf.name_scope(self.classifier.name):
                    # 构建 classifier 模型
                    self.classifier.build([None, None, self.config.hidden_sizes[-1]])
        # 如果存在 distillation_classifier 属性
        if getattr(self, "distillation_classifier", None) is not None:
            # 如果 distillation_classifier 具有名字属性
            if hasattr(self.distillation_classifier, "name"):
                # 使用 distillation_classifier 的名字为当前操作创建一个命名空间
                with tf.name_scope(self.distillation_classifier.name):
                    # 构建 distillation_classifier 模型
                    self.distillation_classifier.build([None, None, self.config.hidden_sizes[-1]])
```