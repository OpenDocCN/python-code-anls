# `.\models\efficientformer\modeling_tf_efficientformer.py`

```py
# coding=utf-8
# 版权所有 2023 Snapchat Research 和 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）许可；
# 除非符合许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按“原样”分发，
# 没有任何明示或暗示的保证或条件。
# 有关详细信息，请参阅许可证。
""" TensorFlow EfficientFormer 模型。"""

import itertools
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import tensorflow as tf

# 导入自定义模块
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
    keras,
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

# 获取 logger 实例
logger = logging.get_logger(__name__)

# 用于文档的配置信息
_CONFIG_FOR_DOC = "EfficientFormerConfig"

# 用于文档的检查点信息
_CHECKPOINT_FOR_DOC = "snap-research/efficientformer-l1-300"
_EXPECTED_OUTPUT_SHAPE = [1, 49, 448]

# 用于图像分类的检查点和预期输出
_IMAGE_CLASS_CHECKPOINT = "snap-research/efficientformer-l1-300"
_IMAGE_CLASS_EXPECTED_OUTPUT = "LABEL_281"

# EfficientFormer 模型的预训练模型存档列表
TF_EFFICIENTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "snap-research/efficientformer-l1-300",
    # 查看所有 EfficientFormer 模型：https://huggingface.co/models?filter=efficientformer
]

# 自定义层：TFEfficientFormerPatchEmbeddings
class TFEfficientFormerPatchEmbeddings(keras.layers.Layer):
    """
    此类在两个阶段之间执行下采样。
    对于形状为 [batch_size, num_channels, height, width] 的输入张量，
    它产生形状为 [batch_size, num_channels, height/stride, width/stride] 的输出张量。
    """

    def __init__(
        self, config: EfficientFormerConfig, num_channels: int, embed_dim: int, apply_norm: bool = True, **kwargs
    ):
        super().__init__(**kwargs)
    ) -> None:
        # 调用父类初始化方法，并传递额外的关键字参数
        super().__init__(**kwargs)
        # 设置网络的通道数属性
        self.num_channels = num_channels

        # 创建用于填充的 ZeroPadding2D 层，使用配置文件中的填充大小
        self.padding = keras.layers.ZeroPadding2D(padding=config.downsample_pad)
        
        # 创建投影层，使用指定的滤波器数目、卷积核大小、步长和填充方式
        self.projection = keras.layers.Conv2D(
            filters=embed_dim,
            kernel_size=config.downsample_patch_size,
            strides=config.downsample_stride,
            padding="valid",
            name="projection",
        )
        
        # 如果应用归一化，则创建批量归一化层，使用配置文件中的动量和 epsilon，模仿 PyTorch 中的默认设置
        self.norm = (
            keras.layers.BatchNormalization(axis=-1, epsilon=config.batch_norm_eps, momentum=0.9, name="norm")
            if apply_norm
            else tf.identity
        )
        
        # 设置嵌入维度属性
        self.embed_dim = embed_dim

    def call(self, pixel_values: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 断言输入张量的形状是否正确，确保通道维度与配置中设置的一致
        tf.debugging.assert_shapes(
            [(pixel_values, (..., None, None, self.num_channels))],
            message="Make sure that the channel dimension of the pixel values match with the one set in the configuration.",
        )
        
        # 对输入像素值进行填充和投影操作，生成嵌入表示
        embeddings = self.projection(self.padding(pixel_values))
        
        # 对嵌入表示进行归一化处理，根据训练模式决定是否使用训练模式
        embeddings = self.norm(embeddings, training=training)
        
        # 返回处理后的嵌入表示张量
        return embeddings

    def build(self, input_shape=None):
        # 如果已经构建过网络，则直接返回
        if self.built:
            return
        
        # 标记网络已经构建
        self.built = True
        
        # 如果存在投影层，则构建投影层
        if getattr(self, "projection", None) is not None:
            with tf.name_scope(self.projection.name):
                self.projection.build([None, None, None, self.num_channels])
        
        # 如果存在归一化层，则根据嵌入维度构建归一化层
        if getattr(self, "norm", None) is not None:
            if hasattr(self.norm, "name"):
                with tf.name_scope(self.norm.name):
                    self.norm.build([None, None, None, self.embed_dim])
    # 自定义的 TensorFlow/Keras 层，实现 EfficientFormer 中的自注意力机制
    class TFEfficientFormerSelfAttention(keras.layers.Layer):
        def __init__(
            self,
            dim: int,
            key_dim: int,
            num_heads: int,
            attention_ratio: int,
            resolution: int,
            config: EfficientFormerConfig,
            **kwargs,
        ):
            super().__init__(**kwargs)

            # 初始化层的参数
            self.num_heads = num_heads  # 自注意力头的数量
            self.key_dim = key_dim  # 键向量的维度
            self.attention_ratio = attention_ratio  # 注意力扩展比率
            self.scale = key_dim**-0.5  # 缩放因子，用于缩放注意力分数
            self.total_key_dim = key_dim * num_heads  # 总的键向量维度
            self.expanded_key_dim = int(attention_ratio * key_dim)  # 扩展后的键向量维度
            self.total_expanded_key_dim = int(self.expanded_key_dim * num_heads)  # 总的扩展键向量维度
            hidden_size = self.total_expanded_key_dim + self.total_key_dim * 2  # 隐藏层的大小，用于 QKV 矩阵

            # 创建 Dense 层，用于计算 QKV 矩阵
            self.qkv = keras.layers.Dense(
                units=hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="qkv"
            )
            # 创建 Dense 层，用于最终投影到输出维度
            self.projection = keras.layers.Dense(
                units=dim, kernel_initializer=get_initializer(config.initializer_range), name="projection"
            )
            self.resolution = resolution  # 分辨率
            self.dim = dim  # 输出维度

        def build(self, input_shape: tf.TensorShape) -> None:
            # 生成所有可能的注意力偏移量
            points = list(itertools.product(range(self.resolution), range(self.resolution)))
            num_points = len(points)
            attention_offsets = {}

            idxs = []

            # 遍历所有点对，计算它们之间的注意力偏移量
            for point_1 in points:
                for point_2 in points:
                    offset = (abs(point_1[0] - point_2[0]), abs(point_1[1] - point_2[1]))
                    if offset not in attention_offsets:
                        attention_offsets[offset] = len(attention_offsets)
                    idxs.append(attention_offsets[offset])

            # 创建注意力偏置权重，用于注意力计算
            self.attention_biases = self.add_weight(
                shape=(self.num_heads, len(attention_offsets)),
                initializer=keras.initializers.zeros(),
                trainable=True,
                name="attention_biases",
            )
            # 创建索引权重，用于指示每对点之间的偏置
            self.attention_bias_idxs = self.add_weight(
                shape=(num_points, num_points),
                trainable=False,
                dtype=tf.int32,
                name="attention_bias_idxs",
            )

            # 将偏置索引转换并分配给注意力偏置索引权重
            self.attention_bias_idxs.assign(tf.reshape(tf.cast(idxs, dtype=tf.int32), (num_points, num_points)))

            if self.built:
                return
            self.built = True
            if getattr(self, "qkv", None) is not None:
                with tf.name_scope(self.qkv.name):
                    self.qkv.build([None, None, self.dim])
            if getattr(self, "projection", None) is not None:
                with tf.name_scope(self.projection.name):
                    self.projection.build([None, None, self.total_expanded_key_dim])

        def call(
            self, hidden_states: tf.Tensor, output_attentions: bool = False, training: bool = False
        ):
            # 留空，因为这里只注释定义和构建部分的代码
    # 定义函数，接收隐藏状态张量并返回元组，包含注意力机制的输出
    ) -> Tuple[tf.Tensor]:
        # 获取隐藏状态张量的形状信息
        batch_size, sequence_length, *_ = shape_list(hidden_states)
        # 调用 self.qkv 方法处理隐藏状态张量，得到 qkv 张量
        qkv = self.qkv(inputs=hidden_states)

        # 将 qkv 张量按照指定大小拆分为查询、键、值张量
        query_layer, key_layer, value_layer = tf.split(
            tf.reshape(tensor=qkv, shape=(batch_size, sequence_length, self.num_heads, -1)),
            num_or_size_splits=[self.key_dim, self.key_dim, self.expanded_key_dim],
            axis=3,
        )

        # 转置查询、键、值张量的维度顺序，以便后续计算注意力矩阵
        query_layer = tf.transpose(query_layer, perm=[0, 2, 1, 3])
        key_layer = tf.transpose(key_layer, perm=[0, 2, 1, 3])
        value_layer = tf.transpose(value_layer, perm=[0, 2, 1, 3])

        # 计算注意力矩阵的原始分数
        attention_probs = tf.matmul(query_layer, tf.transpose(key_layer, perm=[0, 1, 3, 2]))
        # 缩放注意力矩阵
        scale = tf.cast(self.scale, dtype=attention_probs.dtype)
        attention_probs = tf.multiply(attention_probs, scale)

        # 获取注意力偏置项并添加到注意力矩阵中
        attention_biases = tf.gather(params=self.attention_biases, indices=self.attention_bias_idxs, axis=1)
        attention_probs = attention_probs + attention_biases
        # 对注意力矩阵进行稳定的 softmax 归一化
        attention_probs = stable_softmax(logits=attention_probs, axis=-1)

        # 计算上下文张量，即加权值张量乘以值张量
        context_layer = tf.matmul(attention_probs, value_layer)
        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])

        # 重新调整上下文张量的形状，以便进行最终的投影操作
        context_layer = tf.reshape(
            tensor=context_layer, shape=(batch_size, sequence_length, self.total_expanded_key_dim)
        )
        # 应用投影层处理上下文张量，生成最终输出
        context_layer = self.projection(context_layer)

        # 根据输出设置是否返回注意力矩阵，构造最终的输出元组
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        # 返回函数的最终输出
        return outputs
# 定义一个自定义层 TFEfficientFormerConvStem，继承自 keras.layers.Layer
class TFEfficientFormerConvStem(keras.layers.Layer):
    # 初始化方法，接受配置参数 config、输出通道数 out_channels 和其他关键字参数 kwargs
    def __init__(self, config: EfficientFormerConfig, out_channels: int, **kwargs):
        super().__init__(**kwargs)

        # 添加一个 ZeroPadding2D 层，在输入数据周围填充1个像素的零填充
        self.padding = keras.layers.ZeroPadding2D(padding=1)
        
        # 添加第一个卷积层 Conv2D，用于特征提取，输出通道数为 out_channels 的一半，使用 3x3 的卷积核，步幅为2，valid padding
        self.convolution1 = keras.layers.Conv2D(
            filters=out_channels // 2, kernel_size=3, strides=2, padding="valid", name="convolution1"
        )
        
        # 添加一个 BatchNormalization 层，在卷积层前进行批量归一化，axis=-1 表示归一化沿着通道维度，使用指定的 epsilon 和 momentum 参数
        self.batchnorm_before = keras.layers.BatchNormalization(
            axis=-1, epsilon=config.batch_norm_eps, momentum=0.9, name="batchnorm_before"
        )

        # 添加第二个卷积层 Conv2D，用于进一步特征提取，输出通道数为 out_channels，使用 3x3 的卷积核，步幅为2，valid padding
        self.convolution2 = keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=3,
            strides=2,
            padding="valid",
            name="convolution2",
        )
        
        # 添加另一个 BatchNormalization 层，在第二个卷积层后进行批量归一化，axis=-1 表示归一化沿着通道维度，使用指定的 epsilon 和 momentum 参数
        self.batchnorm_after = keras.layers.BatchNormalization(
            axis=-1, epsilon=config.batch_norm_eps, momentum=0.9, name="batchnorm_after"
        )

        # 添加激活函数层，使用 ReLU 激活函数
        self.activation = keras.layers.Activation(activation=keras.activations.relu, name="activation")
        
        # 记录输出通道数和配置参数，以备后用
        self.out_channels = out_channels
        self.config = config

    # 定义调用方法，接受输入像素值张量 pixel_values 和训练标志 training，返回特征张量
    def call(self, pixel_values: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 对输入像素进行填充、卷积、批量归一化和激活操作，得到特征张量
        features = self.batchnorm_before(self.convolution1(self.padding(pixel_values)), training=training)
        features = self.activation(features)
        
        # 对第一次卷积后得到的特征进行填充、第二次卷积、批量归一化和激活操作，得到最终特征张量
        features = self.batchnorm_after(self.convolution2(self.padding(features)), training=training)
        features = self.activation(features)
        
        # 返回最终的特征张量作为输出
        return features

    # 定义构建方法，用于动态构建层的参数
    def build(self, input_shape=None):
        if self.built:
            return
        
        # 标记该层为已构建
        self.built = True
        
        # 如果存在卷积层 convolution1，则动态构建该卷积层
        if getattr(self, "convolution1", None) is not None:
            with tf.name_scope(self.convolution1.name):
                self.convolution1.build([None, None, None, self.config.num_channels])
        
        # 如果存在批量归一化层 batchnorm_before，则动态构建该批量归一化层
        if getattr(self, "batchnorm_before", None) is not None:
            with tf.name_scope(self.batchnorm_before.name):
                self.batchnorm_before.build([None, None, None, self.out_channels // 2])
        
        # 如果存在卷积层 convolution2，则动态构建该卷积层
        if getattr(self, "convolution2", None) is not None:
            with tf.name_scope(self.convolution2.name):
                self.convolution2.build([None, None, None, self.out_channels // 2])
        
        # 如果存在批量归一化层 batchnorm_after，则动态构建该批量归一化层
        if getattr(self, "batchnorm_after", None) is not None:
            with tf.name_scope(self.batchnorm_after.name):
                self.batchnorm_after.build([None, None, None, self.out_channels])
        
        # 如果存在激活函数层 activation，则动态构建该激活函数层
        if getattr(self, "activation", None) is not None:
            with tf.name_scope(self.activation.name):
                self.activation.build(None)
    # 初始化函数，用于设置对象的初始状态
    def __init__(self, pool_size: int, **kwargs):
        # 调用父类的初始化方法，传入其他关键字参数
        super().__init__(**kwargs)
        # 创建一个平均池化层对象，设定池化窗口大小、步幅为1、填充方式为"same"
        self.pool = keras.layers.AveragePooling2D(pool_size=pool_size, strides=1, padding="same")

    # 实现调用方法，处理隐藏状态张量并返回处理后的张量
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 使用平均池化层对隐藏状态进行池化操作
        output = self.pool(hidden_states)
        # 计算输出与原始隐藏状态的差值
        output = output - hidden_states
        # 返回处理后的张量作为最终输出
        return output
# 定义一个名为 TFEfficientFormerDenseMlp 的自定义 Keras 层
class TFEfficientFormerDenseMlp(keras.layers.Layer):
    
    # 初始化方法，接收配置对象 config 和输入、隐藏、输出特征的参数
    def __init__(
        self,
        config: EfficientFormerConfig,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # 如果未提供 out_features，默认与 in_features 相同
        out_features = out_features or in_features
        # 如果未提供 hidden_features，默认与 in_features 相同
        hidden_features = hidden_features or in_features
        
        # 创建一个 Dense 层，用于输入特征的线性变换
        self.linear_in = keras.layers.Dense(
            units=hidden_features, kernel_initializer=get_initializer(config.initializer_range), name="linear_in"
        )
        # 根据配置中的激活函数选择对应的激活函数
        self.activation = ACT2FN[config.hidden_act]
        # 创建一个 Dropout 层，用于随机失活
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)

        # 创建一个 Dense 层，用于输出特征的线性变换
        self.linear_out = keras.layers.Dense(
            units=out_features, kernel_initializer=get_initializer(config.initializer_range), name="linear_out"
        )
        # 记录隐藏特征的维度
        self.hidden_features = hidden_features
        # 记录输入特征的维度
        self.in_features = in_features

    # 调用方法，实现层的正向传播逻辑
    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 输入特征经过线性变换
        hidden_states = self.linear_in(inputs=hidden_states)
        # 应用激活函数
        hidden_states = self.activation(hidden_states)
        # 根据训练模式应用 Dropout
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 经过输出层的线性变换
        hidden_states = self.linear_out(inputs=hidden_states)
        # 再次根据训练模式应用 Dropout
        hidden_states = self.dropout(inputs=hidden_states, training=training)

        return hidden_states

    # 构建方法，用于构建层的内部结构
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 标记该层已经构建
        self.built = True
        
        # 如果存在 linear_in 层，设置其构建方式和输入维度
        if getattr(self, "linear_in", None) is not None:
            with tf.name_scope(self.linear_in.name):
                self.linear_in.build([None, None, self.in_features])
        
        # 如果存在 linear_out 层，设置其构建方式和输入维度
        if getattr(self, "linear_out", None) is not None:
            with tf.name_scope(self.linear_out.name):
                self.linear_out.build([None, None, self.hidden_features])


# 定义一个名为 TFEfficientFormerConvMlp 的自定义 Keras 层
class TFEfficientFormerConvMlp(keras.layers.Layer):
    
    # 初始化方法，接收配置对象 config 和输入、隐藏、输出特征的参数以及 dropout 率
    def __init__(
        self,
        config: EfficientFormerConfig,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs),
        
        # 如果未提供 out_features，默认与 in_features 相同
        out_features = out_features or in_features
        # 如果未提供 hidden_features，默认与 in_features 相同
        hidden_features = hidden_features or in_features
        # 记录 dropout 率
        self.drop = drop
        
        # 待补充...
    ):
        # 调用父类的初始化方法，传递所有关键字参数
        super().__init__(**kwargs)
        # 如果未提供输出特征数，则默认与输入特征数相同
        out_features = out_features or in_features
        # 如果未提供隐藏层特征数，则默认与输入特征数相同
        hidden_features = hidden_features or in_features

        # 创建第一个卷积层，使用隐藏特征数作为滤波器数，1x1 的卷积核
        self.convolution1 = keras.layers.Conv2D(
            filters=hidden_features,
            kernel_size=1,
            name="convolution1",
            padding="valid",
        )

        # 根据配置中的隐藏激活函数选择激活层
        self.activation = ACT2FN[config.hidden_act]

        # 创建第二个卷积层，使用输出特征数作为滤波器数，1x1 的卷积核
        self.convolution2 = keras.layers.Conv2D(
            filters=out_features,
            kernel_size=1,
            name="convolution2",
            padding="valid",
        )

        # 创建一个 Dropout 层，使用给定的丢弃率
        self.dropout = keras.layers.Dropout(rate=drop)

        # 使用与 PyTorch BatchNormalization 相同的默认动量和 epsilon 参数创建 BatchNormalization 层
        self.batchnorm_before = keras.layers.BatchNormalization(
            axis=-1, epsilon=config.batch_norm_eps, momentum=0.9, name="batchnorm_before"
        )
        # 使用与 PyTorch BatchNormalization 相同的默认动量和 epsilon 参数创建 BatchNormalization 层
        self.batchnorm_after = keras.layers.BatchNormalization(
            axis=-1, epsilon=config.batch_norm_eps, momentum=0.9, name="batchnorm_after"
        )

        # 保存隐藏特征数、输入特征数和输出特征数
        self.hidden_features = hidden_features
        self.in_features = in_features
        self.out_features = out_features

    # 定义调用方法，接受隐藏状态张量和训练标志，返回张量
    def call(self, hidden_state: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 应用第一个卷积层到隐藏状态张量
        hidden_state = self.convolution1(hidden_state)
        # 应用 BatchNormalization 层到卷积结果
        hidden_state = self.batchnorm_before(hidden_state, training=training)
        # 应用激活函数到归一化后的张量
        hidden_state = self.activation(hidden_state)
        # 应用 Dropout 层到激活后的张量
        hidden_state = self.dropout(hidden_state, training=training)
        # 应用第二个卷积层到 Dropout 后的张量
        hidden_state = self.convolution2(hidden_state)
        # 应用 BatchNormalization 层到第二个卷积结果
        hidden_state = self.batchnorm_after(hidden_state, training=training)
        # 再次应用 Dropout 层到归一化后的张量
        hidden_state = self.dropout(hidden_state, training=training)
        # 返回处理后的张量作为最终结果
        return hidden_state

    # 定义构建方法，用于根据输入形状建立网络结构
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记网络已经构建
        self.built = True
        # 如果存在第一个卷积层，则根据输入特征数构建其形状
        if getattr(self, "convolution1", None) is not None:
            with tf.name_scope(self.convolution1.name):
                self.convolution1.build([None, None, None, self.in_features])
        # 如果存在第二个卷积层，则根据隐藏特征数构建其形状
        if getattr(self, "convolution2", None) is not None:
            with tf.name_scope(self.convolution2.name):
                self.convolution2.build([None, None, None, self.hidden_features])
        # 如果存在 BatchNormalization 层，则根据隐藏特征数构建其形状
        if getattr(self, "batchnorm_before", None) is not None:
            with tf.name_scope(self.batchnorm_before.name):
                self.batchnorm_before.build([None, None, None, self.hidden_features])
        # 如果存在第二个 BatchNormalization 层，则根据输出特征数构建其形状
        if getattr(self, "batchnorm_after", None) is not None:
            with tf.name_scope(self.batchnorm_after.name):
                self.batchnorm_after.build([None, None, None, self.out_features])
# Copied from transformers.models.convnext.modeling_tf_convnext.TFConvNextDropPath with ConvNext->EfficientFormer
class TFEfficientFormerDropPath(keras.layers.Layer):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    References:
        (1) github.com:rwightman/pytorch-image-models
    """

    def __init__(self, drop_path: float, **kwargs):
        super().__init__(**kwargs)
        self.drop_path = drop_path  # 初始化时设置 drop_path 参数

    def call(self, x: tf.Tensor, training=None):
        if training:
            keep_prob = 1 - self.drop_path
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            # 应用随机深度丢弃路径技术
            return (x / keep_prob) * random_tensor
        return x


class TFEfficientFormerFlat(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, hidden_states: tf.Tensor) -> Tuple[tf.Tensor]:
        batch_size, _, _, in_channels = shape_list(hidden_states)
        # 对输入的隐藏状态进行扁平化处理
        hidden_states = tf.reshape(hidden_states, shape=[batch_size, -1, in_channels])
        return hidden_states


class TFEfficientFormerMeta3D(keras.layers.Layer):
    def __init__(self, config: EfficientFormerConfig, dim: int, drop_path: float = 0.0, **kwargs):
        super().__init__(**kwargs)

        # 创建自注意力层 `token_mixer`，用于处理 token 之间的交互
        self.token_mixer = TFEfficientFormerSelfAttention(
            dim=config.dim,
            key_dim=config.key_dim,
            num_heads=config.num_attention_heads,
            attention_ratio=config.attention_ratio,
            resolution=config.resolution,
            name="token_mixer",
            config=config,
        )
        self.dim = dim
        self.config = config

        # 第一个 LayerNormalization 层
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm1")
        # 第二个 LayerNormalization 层
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm2")
        # 多层感知机（MLP）的隐藏层维度计算
        mlp_hidden_dim = int(dim * config.mlp_expansion_ratio)
        # 创建 MLP 层
        self.mlp = TFEfficientFormerDenseMlp(config, in_features=dim, hidden_features=mlp_hidden_dim, name="mlp")

        # 使用 `layers.Activation` 代替 `tf.identity` 控制 `training` 行为更精确
        # 创建丢弃路径层或者一个线性激活层，取决于 `drop_path` 的值
        self.drop_path = (
            TFEfficientFormerDropPath(drop_path)
            if drop_path > 0.0
            else keras.layers.Activation("linear", name="drop_path")
        )
        self.config = config
    # 在模型建立时初始化层缩放的权重，这里初始化为None
    def build(self, input_shape=None):
        self.layer_scale_1 = None  # 初始化第一个层缩放权重为None
        self.layer_scale_2 = None  # 初始化第二个层缩放权重为None

        # 如果配置中指定使用层缩放
        if self.config.use_layer_scale:
            # 添加第一个层缩放权重，形状为(self.dim,)
            self.layer_scale_1 = self.add_weight(
                shape=(self.dim,),
                initializer=keras.initializers.Constant(value=self.config.layer_scale_init_value),
                trainable=True,
                name="layer_scale_1",
            )
            # 添加第二个层缩放权重，形状为(self.dim,)
            self.layer_scale_2 = self.add_weight(
                shape=(self.dim,),
                initializer=keras.initializers.Constant(value=self.config.layer_scale_init_value),
                trainable=True,
                name="layer_scale_2",
            )

        # 如果模型已经建立，直接返回
        if self.built:
            return
        # 标记模型已经建立
        self.built = True

        # 如果存在token_mixer层，对其进行建立
        if getattr(self, "token_mixer", None) is not None:
            with tf.name_scope(self.token_mixer.name):
                self.token_mixer.build(None)

        # 如果存在layernorm1层，对其进行建立，输入维度为[None, None, self.dim]
        if getattr(self, "layernorm1", None) is not None:
            with tf.name_scope(self.layernorm1.name):
                self.layernorm1.build([None, None, self.dim])

        # 如果存在layernorm2层，对其进行建立，输入维度为[None, None, self.dim]
        if getattr(self, "layernorm2", None) is not None:
            with tf.name_scope(self.layernorm2.name):
                self.layernorm2.build([None, None, self.dim])

        # 如果存在mlp层，对其进行建立
        if getattr(self, "mlp", None) is not None:
            with tf.name_scope(self.mlp.name):
                self.mlp.build(None)

        # 如果存在drop_path层，对其进行建立
        if getattr(self, "drop_path", None) is not None:
            with tf.name_scope(self.drop_path.name):
                self.drop_path.build(None)

    # 模型调用方法，接收隐藏状态、是否输出注意力权重以及训练状态，并返回元组包含的张量
    def call(
        self, hidden_states: tf.Tensor, output_attentions: bool = False, training: bool = False
    ) -> Tuple[tf.Tensor]:
        # 使用token_mixer层处理layernorm1层处理后的隐藏状态，输出注意力权重
        self_attention_outputs = self.token_mixer(
            hidden_states=self.layernorm1(hidden_states, training=training),
            output_attentions=output_attentions,
            training=training,
        )

        # 取自注意力输出的第一个张量作为attention_output
        attention_output = self_attention_outputs[0]
        # 如果要输出注意力权重，则将其它张量也加入outputs中
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # 如果配置中指定使用层缩放
        if self.config.use_layer_scale:
            # 计算第一层输出，加上drop_path层作用于self.layer_scale_1与attention_output的乘积
            layer_output = hidden_states + self.drop_path(
                tf.expand_dims(tf.expand_dims(self.layer_scale_1, 0), 0) * attention_output,
                training=training,
            )
            # 计算第二层输出，加上drop_path层作用于self.layer_scale_2与mlp层作用于layernorm2层处理后的layer_output的乘积
            layer_output = layer_output + self.drop_path(
                tf.expand_dims(tf.expand_dims(self.layer_scale_2, 0), 0)
                * self.mlp(hidden_states=self.layernorm2(inputs=layer_output, training=training), training=training),
                training=training,
            )
        else:
            # 否则，计算第一层输出，加上drop_path层作用于attention_output
            layer_output = hidden_states + self.drop_path(attention_output, training=training)
            # 计算第二层输出，加上drop_path层作用于mlp层作用于layernorm2层处理后的layer_output
            layer_output = layer_output + self.drop_path(
                self.mlp(hidden_states=self.layernorm2(inputs=layer_output, training=training), training=training),
                training=training,
            )

        # 将layer_output加入到输出张量中
        outputs = (layer_output,) + outputs

        # 返回所有输出张量的元组
        return outputs
# 定义一个名为 TFEfficientFormerMeta3DLayers 的自定义层，继承自 keras.layers.Layer 类
class TFEfficientFormerMeta3DLayers(keras.layers.Layer):
    
    # 初始化方法，接受 EfficientFormerConfig 类型的 config 参数和其他关键字参数
    def __init__(self, config: EfficientFormerConfig, **kwargs):
        super().__init__(**kwargs)
        
        # 根据 config 中的参数计算每个块的 drop path 值，存储在 drop_paths 列表中
        drop_paths = [
            config.drop_path_rate * (block_idx + sum(config.depths[:-1]))
            for block_idx in range(config.num_meta3d_blocks)
        ]
        
        # 创建一个由 TFEfficientFormerMeta3D 层组成的列表 self.blocks
        self.blocks = [
            TFEfficientFormerMeta3D(config, config.hidden_sizes[-1], drop_path=drop_path, name=f"blocks.{i}")
            for i, drop_path in enumerate(drop_paths)
        ]

    # call 方法，用于定义层的前向传播逻辑
    def call(
        self, hidden_states: tf.Tensor, output_attentions: bool = False, training: bool = False
    ) -> Tuple[tf.Tensor]:
        # 如果需要输出注意力机制的信息，则初始化 all_attention_outputs 为一个空元组，否则置为 None
        all_attention_outputs = () if output_attentions else None
        
        # 遍历 self.blocks 中的每个层模块
        for i, layer_module in enumerate(self.blocks):
            # 如果 hidden_states 是一个元组，则取其第一个元素
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
            
            # 调用当前层模块的前向传播方法，更新 hidden_states
            hidden_states = layer_module(
                hidden_states=hidden_states, output_attentions=output_attentions, training=training
            )
            
            # 如果需要输出注意力机制的信息，则更新 all_attention_outputs
            if output_attentions:
                all_attention_outputs = all_attention_outputs + (hidden_states[1],)
        
        # 如果需要输出注意力机制的信息，则返回包含 hidden_states 和 all_attention_outputs 的元组
        if output_attentions:
            outputs = (hidden_states[0],) + all_attention_outputs
            return outputs
        
        # 否则，返回更新后的 hidden_states
        return hidden_states

    # build 方法，用于构建层，确保在调用前未构建过
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        
        # 如果 self.blocks 存在，则遍历每个层并调用其 build 方法
        if getattr(self, "blocks", None) is not None:
            for layer in self.blocks:
                with tf.name_scope(layer.name):
                    layer.build(None)


# 定义一个名为 TFEfficientFormerMeta4D 的自定义层，继承自 keras.layers.Layer 类
class TFEfficientFormerMeta4D(keras.layers.Layer):
    
    # 初始化方法，接受 EfficientFormerConfig 类型的 config 参数、维度 dim 和其他关键字参数
    def __init__(self, config: EfficientFormerConfig, dim: int, drop_path: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        
        # 根据 config 中的参数设置池化大小 pool_size，默认为 3
        pool_size = config.pool_size if config.pool_size is not None else 3
        
        # 创建名为 token_mixer 的 TFEfficientFormerPooling 层，用于混合 token
        self.token_mixer = TFEfficientFormerPooling(pool_size=pool_size, name="token_mixer")
        
        # 存储维度信息到 self.dim
        self.dim = dim
        
        # 计算 MLP 隐藏层维度 mlp_hidden_dim
        mlp_hidden_dim = int(dim * config.mlp_expansion_ratio)
        
        # 创建名为 mlp 的 TFEfficientFormerConvMlp 层，用于处理卷积 MLP
        self.mlp = TFEfficientFormerConvMlp(
            config=config, in_features=dim, hidden_features=mlp_hidden_dim, drop=config.hidden_dropout_prob, name="mlp"
        )
        
        # 根据 drop_path 的值创建名为 drop_path 的 TFEfficientFormerDropPath 层或线性激活层
        self.drop_path = (
            TFEfficientFormerDropPath(drop_path, name="drop_path")
            if drop_path > 0.0
            else keras.layers.Activation("linear", name="drop_path")
        )
        
        # 存储配置信息到 self.config
        self.config = config
    # 在神经网络层构建时被调用，初始化一些成员变量
    def build(self, input_shape=None):
        # 初始化用于缩放层输出的两个变量为 None
        self.layer_scale_1 = None
        self.layer_scale_2 = None

        # 如果配置指定使用层缩放
        if self.config.use_layer_scale:
            # 添加第一个层缩放权重，初始化为指定的值
            self.layer_scale_1 = self.add_weight(
                shape=(self.dim),
                initializer=keras.initializers.Constant(value=self.config.layer_scale_init_value),
                trainable=True,
                name="layer_scale_1",
            )
            # 添加第二个层缩放权重，初始化为指定的值
            self.layer_scale_2 = self.add_weight(
                shape=(self.dim),
                initializer=keras.initializers.Constant(value=self.config.layer_scale_init_value),
                trainable=True,
                name="layer_scale_2",
            )

        # 如果已经构建过网络层，则直接返回
        if self.built:
            return
        # 标记网络已构建
        self.built = True

        # 如果存在 token_mixer 层，构建其结构
        if getattr(self, "token_mixer", None) is not None:
            with tf.name_scope(self.token_mixer.name):
                self.token_mixer.build(None)
        
        # 如果存在 mlp 层，构建其结构
        if getattr(self, "mlp", None) is not None:
            with tf.name_scope(self.mlp.name):
                self.mlp.build(None)
        
        # 如果存在 drop_path 层，构建其结构
        if getattr(self, "drop_path", None) is not None:
            with tf.name_scope(self.drop_path.name):
                self.drop_path.build(None)

    # 网络层的调用函数，用于处理输入的隐藏状态
    def call(self, hidden_states: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor]:
        # 使用 token_mixer 处理隐藏状态得到输出
        outputs = self.token_mixer(hidden_states)

        # 如果配置使用层缩放
        if self.config.use_layer_scale:
            # 计算第一层输出，加上缩放后的 token_mixer 输出，并应用 drop_path 层
            layer_output = hidden_states + self.drop_path(
                tf.expand_dims(tf.expand_dims(self.layer_scale_1, 0), 0) * outputs,
                training=training,
            )

            # 计算第二层输出，加上缩放后的 MLP 处理结果，并应用 drop_path 层
            layer_output = layer_output + self.drop_path(
                tf.expand_dims(tf.expand_dims(self.layer_scale_2, 0), 0)
                * self.mlp(hidden_state=layer_output, training=training),
                training=training,
            )

        else:
            # 若不使用层缩放，直接将 token_mixer 输出应用 drop_path 层
            layer_output = hidden_states + self.drop_path(outputs, training=training)
            # 将 MLP 处理结果应用 drop_path 层后加到当前层输出上
            layer_output = layer_output + self.drop_path(
                self.mlp(hidden_state=layer_output, training=training), training=training
            )

        # 返回最终层输出
        return layer_output
class TFEfficientFormerMeta4DLayers(keras.layers.Layer):
    def __init__(self, config: EfficientFormerConfig, stage_idx: int, **kwargs):
        super().__init__(**kwargs)
        # 根据舞台索引选择层数，如果是最后一舞台，减去meta3d块数
        num_layers = (
            config.depths[stage_idx] if stage_idx != -1 else config.depths[stage_idx] - config.num_meta3d_blocks
        )
        # 计算每个块的DropPath率
        drop_paths = [
            config.drop_path_rate * (block_idx + sum(config.depths[:stage_idx])) for block_idx in range(num_layers)
        ]

        # 创建一个由多个TFEfficientFormerMeta4D组成的列表
        self.blocks = [
            TFEfficientFormerMeta4D(
                config=config, dim=config.hidden_sizes[stage_idx], drop_path=drop_paths[i], name=f"blocks.{i}"
            )
            for i in range(len(drop_paths))
        ]

    def call(self, hidden_states: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor]:
        # 依次调用每个块处理隐藏状态
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states=hidden_states, training=training)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经建立，直接返回；否则为每个块设置命名空间并构建
        if getattr(self, "blocks", None) is not None:
            for layer in self.blocks:
                with tf.name_scope(layer.name):
                    layer.build(None)


class TFEfficientFormerIntermediateStage(keras.layers.Layer):
    def __init__(self, config: EfficientFormerConfig, index: int, **kwargs):
        super().__init__(**kwargs)
        # 创建一个TFEfficientFormerMeta4DLayers实例作为meta4D层
        self.meta4D_layers = TFEfficientFormerMeta4DLayers(config=config, stage_idx=index, name="meta4D_layers")

    def call(self, hidden_states: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor]:
        # 调用meta4D层处理隐藏状态
        hidden_states = self.meta4D_layers(hidden_states=hidden_states, training=training)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经建立，直接返回；否则为meta4D层设置命名空间并构建
        if getattr(self, "meta4D_layers", None) is not None:
            with tf.name_scope(self.meta4D_layers.name):
                self.meta4D_layers.build(None)


class TFEfficientFormerLastStage(keras.layers.Layer):
    def __init__(self, config: EfficientFormerConfig, **kwargs):
        super().__init__(**kwargs)
        # 创建TFEfficientFormerMeta4DLayers实例作为meta4D层，使用-1作为舞台索引
        self.meta4D_layers = TFEfficientFormerMeta4DLayers(config=config, stage_idx=-1, name="meta4D_layers")
        # 创建TFEfficientFormerFlat实例作为flat层
        self.flat = TFEfficientFormerFlat(name="flat")
        # 创建TFEfficientFormerMeta3DLayers实例作为meta3D层
        self.meta3D_layers = TFEfficientFormerMeta3DLayers(config, name="meta3D_layers")

    def call(
        self, hidden_states: tf.Tensor, output_attentions: bool = False, training: bool = False
    ) -> Tuple[tf.Tensor]:
        # 依次调用meta4D层、flat层和meta3D层处理隐藏状态
        hidden_states = self.meta4D_layers(hidden_states=hidden_states, training=training)
        hidden_states = self.flat(hidden_states=hidden_states)
        hidden_states = self.meta3D_layers(
            hidden_states=hidden_states, output_attentions=output_attentions, training=training
        )

        return hidden_states
    # 如果模型已经构建完成，则直接返回，不再重复构建
    if self.built:
        return
    # 将模型标记为已构建状态
    self.built = True
    # 如果存在 meta4D_layers 属性，并且不为 None，则构建 meta4D_layers
    if getattr(self, "meta4D_layers", None) is not None:
        # 在 TensorFlow 中为 meta4D_layers 创建命名空间，并进行构建
        with tf.name_scope(self.meta4D_layers.name):
            self.meta4D_layers.build(None)
    # 如果存在 flat 属性，并且不为 None，则构建 flat
    if getattr(self, "flat", None) is not None:
        # 在 TensorFlow 中为 flat 创建命名空间，并进行构建
        with tf.name_scope(self.flat.name):
            self.flat.build(None)
    # 如果存在 meta3D_layers 属性，并且不为 None，则构建 meta3D_layers
    if getattr(self, "meta3D_layers", None) is not None:
        # 在 TensorFlow 中为 meta3D_layers 创建命名空间，并进行构建
        with tf.name_scope(self.meta3D_layers.name):
            self.meta3D_layers.build(None)
# 定义 TF EfficientFormer 编码器的自定义层
class TFEfficientFormerEncoder(keras.layers.Layer):
    # 初始化方法，接受 EfficientFormerConfig 对象和其他关键字参数
    def __init__(self, config: EfficientFormerConfig, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        # 计算中间阶段的数量
        num_intermediate_stages = len(config.depths) - 1
        # 判断每个中间阶段是否需要下采样
        downsamples = [
            config.downsamples[i] or config.hidden_sizes[i] != config.hidden_sizes[i + 1]
            for i in range(num_intermediate_stages)
        ]

        intermediate_stages = []
        layer_count = -1
        # 循环创建中间阶段的模块
        for i in range(num_intermediate_stages):
            layer_count += 1
            # 添加 EfficientFormer 中间阶段模块
            intermediate_stages.append(
                TFEfficientFormerIntermediateStage(config, i, name=f"intermediate_stages.{layer_count}")
            )
            # 如果需要下采样，则添加 Patch Embeddings 模块
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
        # 将创建的中间阶段模块列表赋给实例变量
        self.intermediate_stages = intermediate_stages
        # 创建最后一个阶段的模块并赋给实例变量
        self.last_stage = TFEfficientFormerLastStage(config, name="last_stage")

    # 调用方法，执行编码器的前向传播
    def call(
        self,
        hidden_states: tf.Tensor,
        output_hidden_states: bool,
        output_attentions: bool,
        return_dict: bool,
        training: bool = False,
    ) -> TFBaseModelOutput:
        # 初始化空元组或 None，用于存储所有隐藏状态和自注意力
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # 如果需要输出隐藏状态，则将输入的隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 遍历中间阶段模块，对隐藏状态进行处理
        for layer_module in self.intermediate_stages:
            hidden_states = layer_module(hidden_states, training=training)

            # 如果需要输出隐藏状态，则将当前模块处理后的隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        # 对最后一个阶段模块处理隐藏状态，并获取输出
        layer_output = self.last_stage(hidden_states, output_attentions=output_attentions, training=training)

        # 如果需要输出自注意力，则将其添加到 all_self_attentions 中
        if output_attentions:
            all_self_attentions = all_self_attentions + layer_output[1:]

        # 如果需要输出隐藏状态，则将最后阶段模块的输出添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (layer_output[0],)

        # 如果不需要以字典形式返回结果，则返回非 None 的元组值
        if not return_dict:
            return tuple(v for v in [layer_output[0], all_hidden_states, all_self_attentions] if v is not None)

        # 否则以 TFBaseModelOutput 对象返回结果字典
        return TFBaseModelOutput(
            last_hidden_state=layer_output[0],
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    # 构建方法，在第一次调用 call 方法前被自动调用
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在最后一个阶段模块，则构建它
        if getattr(self, "last_stage", None) is not None:
            with tf.name_scope(self.last_stage.name):
                self.last_stage.build(None)
        # 构建所有中间阶段模块
        for layer in self.intermediate_stages:
            with tf.name_scope(layer.name):
                layer.build(None)


@keras_serializable
# 定义 TF EfficientFormer 主层的自定义 Keras 层
class TFEfficientFormerMainLayer(keras.layers.Layer):
    # 将配置类指定为 EfficientFormerConfig 类
    config_class = EfficientFormerConfig

    # 初始化方法，接收 EfficientFormerConfig 对象和其他关键字参数
    def __init__(self, config: EfficientFormerConfig, **kwargs) -> None:
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 将传入的配置对象赋值给实例的 config 属性
        self.config = config

        # 创建一个 TFEfficientFormerConvStem 对象作为 patch_embed 属性，用于处理输入的像素值
        self.patch_embed = TFEfficientFormerConvStem(config, config.hidden_sizes[0], name="patch_embed")
        
        # 创建一个 TFEfficientFormerEncoder 对象作为 encoder 属性，用于对输入进行编码
        self.encoder = TFEfficientFormerEncoder(config, name="encoder")
        
        # 创建一个 LayerNormalization 层作为 layernorm 属性，用于对输出进行归一化处理
        self.layernorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm")

    # 使用装饰器 unpack_inputs，将 call 方法的输入参数解包为具名参数
    @unpack_inputs
    def call(
        self,
        pixel_values: Optional[tf.Tensor] = None,
        output_attentions: Optional[tf.Tensor] = None,
        output_hidden_states: Optional[tf.Tensor] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor, ...]]:
        # 如果没有显式设置输出注意力机制，则使用模型配置中的默认设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        # 如果没有显式设置输出隐藏状态，则使用模型配置中的默认设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # 如果没有显式设置返回字典，则使用模型配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            # 如果像素值为 None，则抛出数值错误异常
            raise ValueError("You have to specify pixel_values")

        # 当在 CPU 上运行时，keras.layers.Conv2D 和 keras.layers.AveragePool2D 不支持通道优先的 NCHW 格式。
        # 一些块包含两者。因此在此处将输入格式从 (batch_size, num_channels, height, width) 转换为
        # (batch_size, height, width, num_channels)。
        # shape = (batch_size, in_height, in_width, in_channels=num_channels)
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))

        # 使用 patch_embed 方法嵌入像素值，用于训练模式
        embedding_output = self.patch_embed(pixel_values, training=training)

        # 使用 encoder 处理嵌入的隐藏状态，支持输出注意力和隐藏状态，返回字典模式
        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 获取编码器输出的序列输出（第一个元素）
        sequence_output = encoder_outputs[0]

        # 对序列输出进行 LayerNormalization 处理，用于训练模式
        sequence_output = self.layernorm(sequence_output, training=training)

        # 如果需要输出隐藏状态，则将隐藏状态从 (batch_size, height, width, num_channels) 转换为
        # (batch_size, num_channels, height, width)
        if output_hidden_states:
            hidden_states = tuple([tf.transpose(h, perm=(0, 3, 1, 2)) for h in encoder_outputs[1][:-1]]) + (
                encoder_outputs[1][-1],
            )

        # 如果不使用返回字典模式，则返回序列输出和所有的隐藏状态
        if not return_dict:
            head_outputs = (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        # 否则，返回 TFBaseModelOutput 对象，包括最后隐藏状态、隐藏状态和注意力机制
        return TFBaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=hidden_states if output_hidden_states else encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果已经建立了模型，则直接返回
        if self.built:
            return

        # 设置模型已建立标志为 True
        self.built = True

        # 如果存在 patch_embed 属性，则建立 patch_embed 层
        if getattr(self, "patch_embed", None) is not None:
            with tf.name_scope(self.patch_embed.name):
                self.patch_embed.build(None)

        # 如果存在 encoder 属性，则建立 encoder 层
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)

        # 如果存在 layernorm 属性，则建立 layernorm 层
        if getattr(self, "layernorm", None) is not None:
            with tf.name_scope(self.layernorm.name):
                self.layernorm.build([None, None, self.config.hidden_sizes[-1]])
@add_start_docstrings(
    "The bare EfficientFormer Model transformer outputting raw hidden-states without any specific head on top.",
    EFFICIENTFORMER_START_DOCSTRING,
)
class TFEfficientFormerModel(TFEfficientFormerPreTrainedModel):
    def __init__(self, config: EfficientFormerConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)

        # 初始化 EfficientFormer 主层，并命名为 "efficientformer"
        self.efficientformer = TFEfficientFormerMainLayer(config, name="efficientformer")

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
        **kwargs,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        """
        Call function to forward pass through EfficientFormer model.

        Args:
            pixel_values ((`tf.Tensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values. Pixel values can be obtained using `AutoImageProcessor`. See
                `EfficientFormerImageProcessor.__call__` for details.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
                See `attentions` under returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.
                See `hidden_states` under returned tensors for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a `ModelOutput` instead of a plain tuple.
            training (`bool`, *optional*):
                Whether the model is in training mode or evaluation mode.

        Returns:
            Either a `TFBaseModelOutputWithPooling` or a tuple containing a `tf.Tensor`.

        """
        # Forward pass through the EfficientFormer model
        return self.efficientformer(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            **kwargs,
        )
    ) -> Union[Tuple, TFBaseModelOutput]:
        # 定义函数的输入参数和返回类型注解
        outputs = self.efficientformer(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 调用 efficientformer 模型，传入参数并获取输出结果
        return outputs

    def build(self, input_shape=None):
        # 如果模型已经建立，直接返回，不执行后续操作
        if self.built:
            return
        # 将模型标记为已经建立
        self.built = True
        # 如果 efficientformer 存在
        if getattr(self, "efficientformer", None) is not None:
            # 在 TensorFlow 中，使用 name_scope 命名空间来管理计算图中的节点
            with tf.name_scope(self.efficientformer.name):
                # 构建 efficientformer 模型，此处不传入具体的输入形状（None 表示动态输入形状）
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

        self.num_labels = config.num_labels
        self.efficientformer = TFEfficientFormerMainLayer(config, name="efficientformer")

        # Classifier head
        self.classifier = (
            keras.layers.Dense(config.num_labels, name="classifier")
            if config.num_labels > 0
            else keras.layers.Activation("linear", name="classifier")
        )
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(EFFICIENTFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=TFImageClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def call(
        self,
        pixel_values: Optional[tf.Tensor] = None,
        labels: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[tf.Tensor, TFImageClassifierOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # Determine the value of return_dict based on input or default from config
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass inputs to EfficientFormer model for processing
        outputs = self.efficientformer(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # Get the sequence output from EfficientFormer model
        sequence_output = outputs[0]

        # Compute logits for classification using pooled representation
        logits = self.classifier(tf.reduce_mean(sequence_output, axis=-2))

        # Compute loss if labels are provided using helper function hf_compute_loss
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # Return outputs based on whether return_dict is enabled
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # Return structured output using TFImageClassifierOutput
        return TFImageClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回，避免重复构建
        if self.built:
            return
        # 标记为已构建状态
        self.built = True
        # 如果存在 efficientformer 属性，则构建 efficientformer
        if getattr(self, "efficientformer", None) is not None:
            # 在 TensorFlow 中使用 name_scope 来管理命名空间，这里使用 efficientformer 的名称作为命名空间
            with tf.name_scope(self.efficientformer.name):
                # 调用 efficientformer 的 build 方法，传入 None 作为输入形状
                self.efficientformer.build(None)
        # 如果存在 classifier 属性，则构建 classifier
        if getattr(self, "classifier", None) is not None:
            # 如果 classifier 具有 name 属性，将其名称作为命名空间
            if hasattr(self.classifier, "name"):
                with tf.name_scope(self.classifier.name):
                    # 调用 classifier 的 build 方法，传入输入形状 [None, None, self.config.hidden_sizes[-1]]
                    self.classifier.build([None, None, self.config.hidden_sizes[-1]])
# 使用 dataclass 装饰器定义一个数据类，用于存储 EfficientFormer 模型的输出结果
@dataclass
class TFEfficientFormerForImageClassificationWithTeacherOutput(ModelOutput):
    """
    Args:
    Output type of [`EfficientFormerForImageClassificationWithTeacher`].
        logits (`tf.Tensor` of shape `(batch_size, config.num_labels)`):
            预测分数，作为 cls_logits 和 distillation_logits 的平均值。
        cls_logits (`tf.Tensor` of shape `(batch_size, config.num_labels)`):
            分类头部的预测分数（即最终类令牌的隐藏状态上的线性层）。
        distillation_logits (`tf.Tensor` of shape `(batch_size, config.num_labels)`):
            蒸馏头部的预测分数（即蒸馏令牌的隐藏状态上的线性层）。
        hidden_states (`tuple(tf.Tensor)`, *optional*, 当 `output_hidden_states=True` 时返回或当
        `config.output_hidden_states=True` 时返回):
            `tf.Tensor` 元组（一个用于嵌入的输出 + 每层输出的一个），形状为 `(batch_size, sequence_length, hidden_size)`。
            模型在每一层输出的隐藏状态加上初始嵌入输出。
        attentions (`tuple(tf.Tensor)`, *optional*, 当 `output_attentions=True` 时返回或当
        `config.output_attentions=True` 时返回):
            `tf.Tensor` 元组（每层一个），形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
            注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    logits: tf.Tensor = None
    cls_logits: tf.Tensor = None
    distillation_logits: tf.Tensor = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None


# 使用 add_start_docstrings 函数添加类的文档字符串，描述了 EfficientFormer 模型的转换器特征及其图像分类头部（最终隐藏状态上的线性层和蒸馏令牌最终隐藏状态上的线性层），
# 例如用于 ImageNet 的情况。
# 警告：此模型仅支持推断。目前尚不支持使用蒸馏进行微调（即带有教师的微调）。
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
class TFEfficientFormerForImageClassificationWithTeacher(TFEfficientFormerPreTrainedModel):
    def __init__(self, config: EfficientFormerConfig) -> None:
        # 调用父类构造函数，初始化模型的配置
        super().__init__(config)

        # 设置模型的标签数量
        self.num_labels = config.num_labels
        # 创建 EfficientFormer 主层，并命名为 efficientformer
        self.efficientformer = TFEfficientFormerMainLayer(config, name="efficientformer")

        # 分类器头部
        # 如果标签数量大于 0，则创建密集层作为分类器，否则创建线性激活层作为分类器
        self.classifier = (
            keras.layers.Dense(config.num_labels, name="classifier")
            if config.num_labels > 0
            else keras.layers.Activation("linear", name="classifier")
        )
        # 如果标签数量大于 0，则创建密集层作为蒸馏分类器，否则创建线性激活层作为蒸馏分类器
        self.distillation_classifier = (
            keras.layers.Dense(config.num_labels, name="distillation_classifier")
            if config.num_labels > 0
            else keras.layers.Activation("linear", name="distillation_classifier")
        )

    @unpack_inputs
    @add_start_docstrings_to_model_forward(EFFICIENTFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=TFEfficientFormerForImageClassificationWithTeacherOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def call(
        self,
        pixel_values: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[tuple, TFEfficientFormerForImageClassificationWithTeacherOutput]:
        # 如果 return_dict 未提供，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果处于训练模式，则抛出异常，该模型仅支持推断
        if training:
            raise Exception(
                "This model supports inference-only. Fine-tuning with distillation (i.e. with a teacher) is not yet supported."
            )

        # 调用 EfficientFormer 主层，获取输出
        outputs = self.efficientformer(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 获取序列输出（通常是模型输出的第一个元素）
        sequence_output = outputs[0]

        # 使用分类器对序列输出的平均值进行分类预测
        cls_logits = self.classifier(tf.reduce_mean(sequence_output, axis=-2))
        # 使用蒸馏分类器对序列输出的平均值进行分类预测
        distillation_logits = self.distillation_classifier(tf.reduce_mean(sequence_output, axis=-2))
        # 聚合分类器和蒸馏分类器的输出，计算最终的逻辑回归结果
        logits = (cls_logits + distillation_logits) / 2

        # 如果不需要返回字典，则返回一个元组作为模型输出
        if not return_dict:
            output = (logits, cls_logits, distillation_logits) + outputs[1:]
            return output

        # 如果需要返回字典，则创建一个带有详细输出信息的类实例并返回
        return TFEfficientFormerForImageClassificationWithTeacherOutput(
            logits=logits,
            cls_logits=cls_logits,
            distillation_logits=distillation_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    # 构建方法用于在给定输入形状的情况下构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回，不再重复构建
        if self.built:
            return
        # 标记模型为已构建状态
        self.built = True
        
        # 如果存在 efficientformer 属性，进行进一步处理
        if getattr(self, "efficientformer", None) is not None:
            # 使用 efficientformer 的名称创建一个命名空间
            with tf.name_scope(self.efficientformer.name):
                # 对 efficientformer 进行构建，此处输入形状为 None，表示未指定具体输入形状
                self.efficientformer.build(None)
        
        # 如果存在 classifier 属性，进行进一步处理
        if getattr(self, "classifier", None) is not None:
            # 如果 classifier 具有名称属性
            if hasattr(self.classifier, "name"):
                # 使用 classifier 的名称创建一个命名空间
                with tf.name_scope(self.classifier.name):
                    # 对 classifier 进行构建，输入形状为 [None, None, self.config.hidden_sizes[-1]]
                    # 其中第一个维度为批量大小，第二和第三个维度为未指定大小
                    self.classifier.build([None, None, self.config.hidden_sizes[-1]])
        
        # 如果存在 distillation_classifier 属性，进行进一步处理
        if getattr(self, "distillation_classifier", None) is not None:
            # 如果 distillation_classifier 具有名称属性
            if hasattr(self.distillation_classifier, "name"):
                # 使用 distillation_classifier 的名称创建一个命名空间
                with tf.name_scope(self.distillation_classifier.name):
                    # 对 distillation_classifier 进行构建，输入形状同上
                    self.distillation_classifier.build([None, None, self.config.hidden_sizes[-1]])
```