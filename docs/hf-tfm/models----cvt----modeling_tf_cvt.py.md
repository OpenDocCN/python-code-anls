# `.\models\cvt\modeling_tf_cvt.py`

```py
# 设置编码格式为 utf-8
# 版权声明

# 引入collections.abc标准库
import collections.abc
# 引入dataclass模块
from dataclasses import dataclass
# 引入Optional, Tuple, Union类型
from typing import Optional, Tuple, Union

# 引入tensorflow库
import tensorflow as tf

# 引入模型输出相关函数和类
from ...modeling_tf_outputs import TFImageClassifierOutputWithNoAttention
# 引入模型工具函数
from ...modeling_tf_utils import (
    TFModelInputType,
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    get_initializer,
    keras,
    keras_serializable,
    unpack_inputs,
)
# 引入tensorflow工具函数
from ...tf_utils import shape_list, stable_softmax
# 引入相关类和函数
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# 引入cvt配置类
from .configuration_cvt import CvtConfig

# 获取logger
logger = logging.get_logger(__name__)

# 基本文档字符串
_CONFIG_FOR_DOC = "CvtConfig"

# 预训练模型列表
TF_CVT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/cvt-13",
    "microsoft/cvt-13-384",
    "microsoft/cvt-13-384-22k",
    "microsoft/cvt-21",
    "microsoft/cvt-21-384",
    "microsoft/cvt-21-384-22k",
    # 查看所有Cvt模型：https://huggingface.co/models?filter=cvt
]

# 数据类
@dataclass
class TFBaseModelOutputWithCLSToken(ModelOutput):
    """
    模型输出的基本类。

    参数:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层的隐态输出序列。
        cls_token_value (`tf.Tensor` of shape `(batch_size, 1, hidden_size)`):
            模型最后一层的分类标记。
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`. 每层模型的隐藏状态加上初始嵌入输出。
    """

    # 隐含态输出序列
    last_hidden_state: tf.Tensor = None
    # 分类标记
    cls_token_value: tf.Tensor = None
    # 隐藏状态
    hidden_states: Tuple[tf.Tensor, ...] | None = None


# TFCvtDropPath类
class TFCvtDropPath(keras.layers.Layer):
    """在残差块的主路径上对每个样本进行辍学（随机深度）。
    参考：(1) github.com:rwightman/pytorch-image-models
    """
    # 初始化函数，用于设置Dropout层的丢弃概率
    def __init__(self, drop_prob: float, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 设置实例变量来存储丢弃概率
        self.drop_prob = drop_prob

    # 覆盖父类的call方法，实现自定义的Dropout逻辑
    def call(self, x: tf.Tensor, training=None):
        # 如果丢弃概率为0或者非训练模式，则直接返回输入张量x
        if self.drop_prob == 0.0 or not training:
            return x
        # 计算保留概率
        keep_prob = 1 - self.drop_prob
        # 获取输入张量x的形状信息，并创建一个与其相同维度的随机张量
        shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
        random_tensor = keep_prob + tf.random.uniform(shape, 0, 1, dtype=self.compute_dtype)
        # 对随机张量进行下取整操作
        random_tensor = tf.floor(random_tensor)
        # 返回经过Dropout处理后的张量
        return (x / keep_prob) * random_tensor
class TFCvtEmbeddings(keras.layers.Layer):
    """Construct the Convolutional Token Embeddings."""

    def __init__(
        self,
        config: CvtConfig,
        patch_size: int,
        num_channels: int,
        embed_dim: int,
        stride: int,
        padding: int,
        dropout_rate: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # 初始化卷积嵌入层，用于处理输入像素数据
        self.convolution_embeddings = TFCvtConvEmbeddings(
            config,
            patch_size=patch_size,
            num_channels=num_channels,
            embed_dim=embed_dim,
            stride=stride,
            padding=padding,
            name="convolution_embeddings",
        )
        # 添加一个丢弃层，用于在训练时随机丢弃部分输出，防止过拟合
        self.dropout = keras.layers.Dropout(dropout_rate)

    def call(self, pixel_values: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 调用卷积嵌入层处理输入像素数据
        hidden_state = self.convolution_embeddings(pixel_values)
        # 在训练时应用丢弃层，随机丢弃部分输出
        hidden_state = self.dropout(hidden_state, training=training)
        return hidden_state

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "convolution_embeddings", None) is not None:
            # 构建卷积嵌入层，如果已构建则跳过
            with tf.name_scope(self.convolution_embeddings.name):
                self.convolution_embeddings.build(None)


class TFCvtConvEmbeddings(keras.layers.Layer):
    """Image to Convolution Embeddings. This convolutional operation aims to model local spatial contexts."""

    def __init__(
        self,
        config: CvtConfig,
        patch_size: int,
        num_channels: int,
        embed_dim: int,
        stride: int,
        padding: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # 添加零填充层，用于在输入图像边界周围填充零值，以处理卷积操作
        self.padding = keras.layers.ZeroPadding2D(padding=padding)
        # 将 patch_size 转为元组形式，若 patch_size 是迭代对象则保持原样，否则转为 (patch_size, patch_size)
        self.patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        # 添加卷积层，用于从输入中提取局部特征，并投影到指定的嵌入维度
        self.projection = keras.layers.Conv2D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=stride,
            padding="valid",
            data_format="channels_last",
            kernel_initializer=get_initializer(config.initializer_range),
            name="projection",
        )
        # 添加层归一化层，用于在卷积输出上应用层归一化，增强模型的稳定性和训练效果
        # 使用与 PyTorch 相同的默认 epsilon 值
        self.normalization = keras.layers.LayerNormalization(epsilon=1e-5, name="normalization")
        self.num_channels = num_channels
        self.embed_dim = embed_dim
    # 定义一个方法，接受一个 tf.Tensor 类型的参数 pixel_values，并返回一个 tf.Tensor 类型的结果
    def call(self, pixel_values: tf.Tensor) -> tf.Tensor:
        # 如果输入的 pixel_values 是一个字典，则将其转换为字典中的 "pixel_values" 键对应的值
        if isinstance(pixel_values, dict):
            pixel_values = pixel_values["pixel_values"]

        # 对输入的 pixel_values 进行填充和投影操作
        pixel_values = self.projection(self.padding(pixel_values))

        # 获取处理后的 pixel_values 的形状信息
        # "batch_size, height, width, num_channels -> batch_size, (height*width), num_channels"
        batch_size, height, width, num_channels = shape_list(pixel_values)
        hidden_size = height * width
        # 将 pixel_values 重塑为指定形状
        pixel_values = tf.reshape(pixel_values, shape=(batch_size, hidden_size, num_channels))
        # 对重塑后的 pixel_values 进行归一化处理
        pixel_values = self.normalization(pixel_values)

        # 将 pixel_values 重新重塑为原始输入的形状
        # "batch_size, (height*width), num_channels -> batch_size, height, width, num_channels"
        pixel_values = tf.reshape(pixel_values, shape=(batch_size, height, width, num_channels))
        # 返回处理后的 pixel_values
        return pixel_values

    # 构建方法，用于建立模型的层次结构
    def build(self, input_shape=None):
        # 如果已经建立过模型，则直接返回
        if self.built:
            return
        self.built = True
        # 如果模型具有投影层，则构建投影层
        if getattr(self, "projection", None) is not None:
            with tf.name_scope(self.projection.name):
                self.projection.build([None, None, None, self.num_channels])
        # 如果模型具有归一化层，则构建归一化层
        if getattr(self, "normalization", None) is not None:
            with tf.name_scope(self.normalization.name):
                self.normalization.build([None, None, self.embed_dim])
class TFCvtSelfAttentionConvProjection(keras.layers.Layer):
    """Convolutional projection layer."""

    def __init__(self, config: CvtConfig, embed_dim: int, kernel_size: int, stride: int, padding: int, **kwargs):
        super().__init__(**kwargs)
        # 设置 ZeroPadding2D 层，用于在输入数据的边缘填充指定数量的零值
        self.padding = keras.layers.ZeroPadding2D(padding=padding)
        # 设置 Conv2D 层，用于执行二维卷积操作
        self.convolution = keras.layers.Conv2D(
            filters=embed_dim,
            kernel_size=kernel_size,
            kernel_initializer=get_initializer(config.initializer_range),
            padding="valid",
            strides=stride,
            use_bias=False,
            name="convolution",
            groups=embed_dim,  # 指定卷积操作的分组数
        )
        # 设置 BatchNormalization 层，用于对卷积层的输出进行批量归一化处理
        # 使用与 PyTorch 相同的默认 epsilon，TF 使用 (1 - PyTorch momentum)
        self.normalization = keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="normalization")
        self.embed_dim = embed_dim  # 记录嵌入维度

    def call(self, hidden_state: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 执行卷积操作，并在输入数据的边缘填充零值
        hidden_state = self.convolution(self.padding(hidden_state))
        # 对卷积输出进行批量归一化处理，可选择是否在训练过程中使用
        hidden_state = self.normalization(hidden_state, training=training)
        return hidden_state

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "convolution", None) is not None:
            with tf.name_scope(self.convolution.name):
                # 构建 Conv2D 层，指定输入张量的形状
                self.convolution.build([None, None, None, self.embed_dim])
        if getattr(self, "normalization", None) is not None:
            with tf.name_scope(self.normalization.name):
                # 构建 BatchNormalization 层，指定输入张量的形状
                self.normalization.build([None, None, None, self.embed_dim])


class TFCvtSelfAttentionLinearProjection(keras.layers.Layer):
    """Linear projection layer used to flatten tokens into 1D."""

    def call(self, hidden_state: tf.Tensor) -> tf.Tensor:
        # 将输入张量从四维(batch_size, height, width, num_channels)转换为三维(batch_size, height*width, num_channels)
        batch_size, height, width, num_channels = shape_list(hidden_state)
        hidden_size = height * width
        hidden_state = tf.reshape(hidden_state, shape=(batch_size, hidden_size, num_channels))
        return hidden_state


class TFCvtSelfAttentionProjection(keras.layers.Layer):
    """Convolutional Projection for Attention."""

    def __init__(
        self,
        config: CvtConfig,
        embed_dim: int,
        kernel_size: int,
        stride: int,
        padding: int,
        projection_method: str = "dw_bn",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if projection_method == "dw_bn":
            # 根据投影方法选择使用卷积投影层进行初始化
            self.convolution_projection = TFCvtSelfAttentionConvProjection(
                config, embed_dim, kernel_size, stride, padding, name="convolution_projection"
            )
        # 初始化线性投影层对象
        self.linear_projection = TFCvtSelfAttentionLinearProjection()
    # 定义一个方法 `call`，用于接收隐藏状态 `hidden_state` 和训练标志 `training`，返回处理后的隐藏状态
    def call(self, hidden_state: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 调用卷积投影层 `convolution_projection` 处理隐藏状态，根据训练标志 `training` 决定是否使用训练模式
        hidden_state = self.convolution_projection(hidden_state, training=training)
        # 调用线性投影层 `linear_projection` 处理卷积处理后的隐藏状态
        hidden_state = self.linear_projection(hidden_state)
        # 返回处理后的隐藏状态
        return hidden_state

    # 定义一个方法 `build`，用于构建层的网络结构
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回，避免重复构建
        if self.built:
            return
        # 标记这个层已经构建过
        self.built = True
        # 检查是否存在卷积投影层 `convolution_projection`
        if getattr(self, "convolution_projection", None) is not None:
            # 在命名作用域 `self.convolution_projection.name` 下构建卷积投影层
            with tf.name_scope(self.convolution_projection.name):
                self.convolution_projection.build(None)
# 定义一个自注意力层的类 TFCvtSelfAttention，用于处理自注意力机制。这一层包含了为查询（query）、键（key）和值（value）嵌入应用的深度可分离卷积操作（卷积投影）。
class TFCvtSelfAttention(keras.layers.Layer):
    """
    Self-attention layer. A depth-wise separable convolution operation (Convolutional Projection), is applied for
    query, key, and value embeddings.
    """

    # 初始化方法，接受多个参数来配置层的行为和特性
    def __init__(
        self,
        config: CvtConfig,  # 用于配置的对象 CvtConfig
        num_heads: int,  # 注意力头的数量
        embed_dim: int,  # 嵌入维度
        kernel_size: int,  # 卷积核大小
        stride_q: int,  # 查询的步长
        stride_kv: int,  # 键值对的步长
        padding_q: int,  # 查询的填充
        padding_kv: int,  # 键值对的填充
        qkv_projection_method: str,  # 查询、键、值投影的方法
        qkv_bias: bool,  # 是否使用偏置项
        attention_drop_rate: float,  # 注意力机制中的丢弃率
        with_cls_token: bool = True,  # 是否包含类别标记
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scale = embed_dim**-0.5  # 缩放因子，用于缩放嵌入维度
        self.with_cls_token = with_cls_token  # 是否包含类别标记
        self.embed_dim = embed_dim  # 嵌入维度
        self.num_heads = num_heads  # 注意力头的数量

        # 创建查询的卷积投影层
        self.convolution_projection_query = TFCvtSelfAttentionProjection(
            config,
            embed_dim,
            kernel_size,
            stride_q,
            padding_q,
            projection_method="linear" if qkv_projection_method == "avg" else qkv_projection_method,
            name="convolution_projection_query",
        )
        
        # 创建键的卷积投影层
        self.convolution_projection_key = TFCvtSelfAttentionProjection(
            config,
            embed_dim,
            kernel_size,
            stride_kv,
            padding_kv,
            projection_method=qkv_projection_method,
            name="convolution_projection_key",
        )
        
        # 创建值的卷积投影层
        self.convolution_projection_value = TFCvtSelfAttentionProjection(
            config,
            embed_dim,
            kernel_size,
            stride_kv,
            padding_kv,
            projection_method=qkv_projection_method,
            name="convolution_projection_value",
        )

        # 创建查询的全连接投影层
        self.projection_query = keras.layers.Dense(
            units=embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            use_bias=qkv_bias,
            bias_initializer="zeros",
            name="projection_query",
        )
        
        # 创建键的全连接投影层
        self.projection_key = keras.layers.Dense(
            units=embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            use_bias=qkv_bias,
            bias_initializer="zeros",
            name="projection_key",
        )
        
        # 创建值的全连接投影层
        self.projection_value = keras.layers.Dense(
            units=embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            use_bias=qkv_bias,
            bias_initializer="zeros",
            name="projection_value",
        )
        
        # 创建注意力机制中的丢弃层
        self.dropout = keras.layers.Dropout(attention_drop_rate)
    # 重新排列张量以供多头注意力机制使用
    def rearrange_for_multi_head_attention(self, hidden_state: tf.Tensor) -> tf.Tensor:
        # 获取张量的维度信息：batch_size为批大小，hidden_size为隐藏状态的大小，_表示不关心的维度
        batch_size, hidden_size, _ = shape_list(hidden_state)
        # 计算每个注意力头的维度
        head_dim = self.embed_dim // self.num_heads
        # 将隐藏状态张量重新整形为(batch_size, hidden_size, num_heads, head_dim)
        hidden_state = tf.reshape(hidden_state, shape=(batch_size, hidden_size, self.num_heads, head_dim))
        # 转置张量以匹配多头注意力机制的输入要求
        hidden_state = tf.transpose(hidden_state, perm=(0, 2, 1, 3))
        return hidden_state

    # 模型的前向传播函数
    def call(self, hidden_state: tf.Tensor, height: int, width: int, training: bool = False) -> tf.Tensor:
        # 如果模型包含分类标记
        if self.with_cls_token:
            # 分离分类标记和隐藏状态
            cls_token, hidden_state = tf.split(hidden_state, [1, height * width], 1)

        # "batch_size, (height*width), num_channels -> batch_size, height, width, num_channels"
        # 获取隐藏状态张量的维度信息：batch_size为批大小，hidden_size为高度乘宽度的大小，num_channels为通道数
        batch_size, hidden_size, num_channels = shape_list(hidden_state)
        # 将隐藏状态张量重新整形为(batch_size, height, width, num_channels)
        hidden_state = tf.reshape(hidden_state, shape=(batch_size, height, width, num_channels))

        # 使用三个卷积函数进行投影得到key、query、value
        key = self.convolution_projection_key(hidden_state, training=training)
        query = self.convolution_projection_query(hidden_state, training=training)
        value = self.convolution_projection_value(hidden_state, training=training)

        # 如果模型包含分类标记，则在query、key、value中添加分类标记
        if self.with_cls_token:
            query = tf.concat((cls_token, query), axis=1)
            key = tf.concat((cls_token, key), axis=1)
            value = tf.concat((cls_token, value), axis=1)

        # 计算每个注意力头的维度
        head_dim = self.embed_dim // self.num_heads

        # 将query、key、value张量重新排列以供多头注意力机制使用
        query = self.rearrange_for_multi_head_attention(self.projection_query(query))
        key = self.rearrange_for_multi_head_attention(self.projection_key(key))
        value = self.rearrange_for_multi_head_attention(self.projection_value(value))

        # 计算注意力分数
        attention_score = tf.matmul(query, key, transpose_b=True) * self.scale
        # 对注意力分数进行稳定的softmax操作
        attention_probs = stable_softmax(logits=attention_score, axis=-1)
        # 在训练时对注意力概率进行dropout
        attention_probs = self.dropout(attention_probs, training=training)

        # 计算上下文张量
        context = tf.matmul(attention_probs, value)
        # "batch_size, num_heads, hidden_size, head_dim -> batch_size, hidden_size, (num_heads*head_dim)"
        # 获取上下文张量的维度信息：batch_size为批大小，hidden_size为隐藏状态的大小，_表示不关心的维度
        _, _, hidden_size, _ = shape_list(context)
        # 转置上下文张量以匹配输出格式要求
        context = tf.transpose(context, perm=(0, 2, 1, 3))
        # 将上下文张量重新整形为(batch_size, hidden_size, (num_heads*head_dim))
        context = tf.reshape(context, (batch_size, hidden_size, self.num_heads * head_dim))
        return context
    # 如果已经构建过网络结构，则直接返回，避免重复构建
    if self.built:
        return
    # 将标志位设置为已构建
    self.built = True
    
    # 如果存在卷积投影查询层，则构建该层
    if getattr(self, "convolution_projection_query", None) is not None:
        # 在命名空间下构建卷积投影查询层
        with tf.name_scope(self.convolution_projection_query.name):
            self.convolution_projection_query.build(None)
    
    # 如果存在卷积投影键层，则构建该层
    if getattr(self, "convolution_projection_key", None) is not None:
        # 在命名空间下构建卷积投影键层
        with tf.name_scope(self.convolution_projection_key.name):
            self.convolution_projection_key.build(None)
    
    # 如果存在卷积投影值层，则构建该层
    if getattr(self, "convolution_projection_value", None) is not None:
        # 在命名空间下构建卷积投影值层
        with tf.name_scope(self.convolution_projection_value.name):
            self.convolution_projection_value.build(None)
    
    # 如果存在投影查询层，则构建该层
    if getattr(self, "projection_query", None) is not None:
        # 在命名空间下构建投影查询层，预期输入形状为 [None, None, self.embed_dim]
        with tf.name_scope(self.projection_query.name):
            self.projection_query.build([None, None, self.embed_dim])
    
    # 如果存在投影键层，则构建该层
    if getattr(self, "projection_key", None) is not None:
        # 在命名空间下构建投影键层，预期输入形状为 [None, None, self.embed_dim]
        with tf.name_scope(self.projection_key.name):
            self.projection_key.build([None, None, self.embed_dim])
    
    # 如果存在投影值层，则构建该层
    if getattr(self, "projection_value", None) is not None:
        # 在命名空间下构建投影值层，预期输入形状为 [None, None, self.embed_dim]
        with tf.name_scope(self.projection_value.name):
            self.projection_value.build([None, None, self.embed_dim])
# 自定义 Keras 层，用于表示注意力层的输出
class TFCvtSelfOutput(keras.layers.Layer):
    """Output of the Attention layer ."""

    def __init__(self, config: CvtConfig, embed_dim: int, drop_rate: float, **kwargs):
        super().__init__(**kwargs)
        # 创建一个全连接层，用于转换隐藏状态到指定维度的输出
        self.dense = keras.layers.Dense(
            units=embed_dim, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # Dropout 层，用于在训练过程中随机丢弃部分输出，防止过拟合
        self.dropout = keras.layers.Dropout(drop_rate)
        self.embed_dim = embed_dim

    def call(self, hidden_state: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 将隐藏状态输入全连接层，得到转换后的输出
        hidden_state = self.dense(inputs=hidden_state)
        # 在训练时，对输出进行 Dropout 处理
        hidden_state = self.dropout(inputs=hidden_state, training=training)
        return hidden_state

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                # 构建全连接层，指定输入维度为 [None, None, self.embed_dim]
                self.dense.build([None, None, self.embed_dim])


class TFCvtAttention(keras.layers.Layer):
    """Attention layer. First chunk of the convolutional transformer block."""

    def __init__(
        self,
        config: CvtConfig,
        num_heads: int,
        embed_dim: int,
        kernel_size: int,
        stride_q: int,
        stride_kv: int,
        padding_q: int,
        padding_kv: int,
        qkv_projection_method: str,
        qkv_bias: bool,
        attention_drop_rate: float,
        drop_rate: float,
        with_cls_token: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # 创建自注意力层，这是卷积变换器块的第一部分
        self.attention = TFCvtSelfAttention(
            config,
            num_heads,
            embed_dim,
            kernel_size,
            stride_q,
            stride_kv,
            padding_q,
            padding_kv,
            qkv_projection_method,
            qkv_bias,
            attention_drop_rate,
            with_cls_token,
            name="attention",
        )
        # 创建自定义的输出层
        self.dense_output = TFCvtSelfOutput(config, embed_dim, drop_rate, name="output")

    def prune_heads(self, heads):
        # 当前未实现剪枝头部的方法，抛出未实现异常
        raise NotImplementedError

    def call(self, hidden_state: tf.Tensor, height: int, width: int, training: bool = False):
        # 调用注意力层，得到自注意力的输出
        self_output = self.attention(hidden_state, height, width, training=training)
        # 将自注意力层的输出传递给自定义的输出层，得到最终的注意力输出
        attention_output = self.dense_output(self_output, training=training)
        return attention_output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                # 构建注意力层，没有指定输入形状，因为它可以是任意形状
                self.attention.build(None)
        if getattr(self, "dense_output", None) is not None:
            with tf.name_scope(self.dense_output.name):
                # 构建自定义输出层，同样没有指定输入形状，因为它可以是任意形状
                self.dense_output.build(None)


class TFCvtIntermediate(keras.layers.Layer):
    """Intermediate dense layer. Second chunk of the convolutional transformer block."""
    # 初始化函数，用于初始化类实例
    def __init__(self, config: CvtConfig, embed_dim: int, mlp_ratio: int, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)
        # 创建一个全连接层(Dense)，设置单元数为 embed_dim * mlp_ratio
        # 使用指定的初始化器初始化权重矩阵
        # 激活函数选择为 GELU
        # 层的名称为 "dense"
        self.dense = keras.layers.Dense(
            units=int(embed_dim * mlp_ratio),
            kernel_initializer=get_initializer(config.initializer_range),
            activation="gelu",
            name="dense",
        )
        # 存储 embed_dim 到实例变量 embed_dim
        self.embed_dim = embed_dim

    # 定义类的调用方法，用于执行实际的前向传播计算
    def call(self, hidden_state: tf.Tensor) -> tf.Tensor:
        # 将输入 hidden_state 通过全连接层 self.dense 进行变换
        hidden_state = self.dense(hidden_state)
        # 返回变换后的张量
        return hidden_state

    # 构建方法，在首次调用 call 方法前调用，用于构建层的结构
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 设置该层为已构建状态
        self.built = True
        # 如果存在 self.dense 属性，则进入下面的逻辑
        if getattr(self, "dense", None) is not None:
            # 在 TensorFlow 的命名作用域下构建 self.dense 层
            with tf.name_scope(self.dense.name):
                # 构建 self.dense 层，输入形状为 [None, None, self.embed_dim]
                self.dense.build([None, None, self.embed_dim])
        super().__init__(**kwargs)
        # 初始化方法，接收多个参数配置以及超参数
        self.attention = AttentionLayer(
            num_heads=num_heads,
            embed_dim=embed_dim,
            kernel_size=kernel_size,
            stride_q=stride_q,
            stride_kv=stride_kv,
            padding_q=padding_q,
            padding_kv=padding_kv,
            qkv_projection_method=qkv_projection_method,
            qkv_bias=qkv_bias,
            attention_drop_rate=attention_drop_rate,
            **kwargs
        )
        # 创建注意力层对象，用于处理输入数据
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-5)
        # 创建层归一化对象，对注意力层输出进行归一化
        self.dense1 = keras.layers.Dense(
            units=int(embed_dim * mlp_ratio),
            kernel_initializer=get_initializer(config.initializer_range),
            activation=gelu,
            name="dense1"
        )
        # 创建第一个全连接层对象，处理注意力层输出
        self.dropout = keras.layers.Dropout(drop_rate)
        # 创建丢弃层对象，用于随机丢弃部分数据以防止过拟合
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-5)
        # 创建第二个层归一化对象，对全连接层输出进行归一化
        self.dense2 = keras.layers.Dense(units=embed_dim, kernel_initializer=get_initializer(config.initializer_range), name="dense2")
        # 创建第二个全连接层对象，处理第一个全连接层输出
        self.drop_path = DropPath(drop_path_rate)
        # 创建路径丢弃层对象，用于在训练过程中以概率丢弃路径
        self.with_cls_token = with_cls_token
        # 设置是否使用类别令牌标志位

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 定义调用方法，接收输入张量和训练标志位
        attn_output = self.attention(inputs, training=training)
        # 调用注意力层处理输入数据
        attn_output = self.norm1(attn_output + inputs)
        # 对注意力层输出和输入数据进行残差连接并归一化
        mlp_output = self.dense1(attn_output)
        # 将注意力层输出输入全连接层1中进行处理
        mlp_output = self.dropout(mlp_output, training=training)
        # 对全连接层1输出进行丢弃处理
        mlp_output = self.dense2(mlp_output)
        # 将全连接层1输出输入全连接层2中进行处理
        mlp_output = self.drop_path(mlp_output, training=training)
        # 对全连接层2输出进行路径丢弃处理
        mlp_output = self.norm2(mlp_output + attn_output)
        # 对全连接层2输出和注意力层输出进行残差连接并归一化
        if self.with_cls_token:
            return mlp_output[:, 0]
        else:
            return mlp_output
        # 如果设置使用类别令牌，则返回全连接层2输出的第一个元素；否则返回全连接层2输出
    ):
        # 调用父类的初始化方法，传递所有的关键字参数
        super().__init__(**kwargs)
        # 初始化自注意力层，使用 TFCvtAttention 类
        self.attention = TFCvtAttention(
            config,
            num_heads,
            embed_dim,
            kernel_size,
            stride_q,
            stride_kv,
            padding_q,
            padding_kv,
            qkv_projection_method,
            qkv_bias,
            attention_drop_rate,
            drop_rate,
            with_cls_token,
            name="attention",
        )
        # 初始化中间层，使用 TFCvtIntermediate 类
        self.intermediate = TFCvtIntermediate(config, embed_dim, mlp_ratio, name="intermediate")
        # 初始化输出层，使用 TFCvtOutput 类
        self.dense_output = TFCvtOutput(config, embed_dim, mlp_ratio, drop_rate, name="output")
        # 使用 `layers.Activation` 替代 `tf.identity` 来更好地控制 `training` 行为
        self.drop_path = (
            # 如果 drop_path_rate 大于 0.0，则使用 TFCvtDropPath 初始化
            TFCvtDropPath(drop_path_rate, name="drop_path")
            # 否则使用 keras.layers.Activation("linear") 初始化
            if drop_path_rate > 0.0
            else keras.layers.Activation("linear", name="drop_path")
        )
        # 使用与 PyTorch 相同的默认 epsilon 初始化前层归一化层
        self.layernorm_before = keras.layers.LayerNormalization(epsilon=1e-5, name="layernorm_before")
        # 使用与 PyTorch 相同的默认 epsilon 初始化后层归一化层
        self.layernorm_after = keras.layers.LayerNormalization(epsilon=1e-5, name="layernorm_after")
        # 初始化 embed_dim 属性
        self.embed_dim = embed_dim

    def call(self, hidden_state: tf.Tensor, height: int, width: int, training: bool = False) -> tf.Tensor:
        # 在 Cvt 中，self-attention 前应用层归一化
        attention_output = self.attention(self.layernorm_before(hidden_state), height, width, training=training)
        # 应用 drop_path 层
        attention_output = self.drop_path(attention_output, training=training)

        # 第一个残差连接
        hidden_state = attention_output + hidden_state

        # 在 Cvt 中，self-attention 后也应用层归一化
        layer_output = self.layernorm_after(hidden_state)
        # 应用中间层
        layer_output = self.intermediate(layer_output)

        # 第二个残差连接
        layer_output = self.dense_output(layer_output, hidden_state)
        # 应用 drop_path 层
        layer_output = self.drop_path(layer_output, training=training)
        # 返回层输出
        return layer_output
    # 构建模型方法，初始化模型结构
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 标记模型为已构建状态
        self.built = True
        
        # 如果存在注意力层，构建其内部结构
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        
        # 如果存在中间层，构建其内部结构
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        
        # 如果存在密集输出层，构建其内部结构
        if getattr(self, "dense_output", None) is not None:
            with tf.name_scope(self.dense_output.name):
                self.dense_output.build(None)
        
        # 如果存在 dropout 路径，构建其内部结构
        if getattr(self, "drop_path", None) is not None:
            with tf.name_scope(self.drop_path.name):
                self.drop_path.build(None)
        
        # 如果存在 layernorm 在前，构建其内部结构
        if getattr(self, "layernorm_before", None) is not None:
            with tf.name_scope(self.layernorm_before.name):
                self.layernorm_before.build([None, None, self.embed_dim])
        
        # 如果存在 layernorm 在后，构建其内部结构
        if getattr(self, "layernorm_after", None) is not None:
            with tf.name_scope(self.layernorm_after.name):
                self.layernorm_after.build([None, None, self.embed_dim])
    """
    Cvt stage (encoder block). Each stage has 2 parts :
    - (1) A Convolutional Token Embedding layer
    - (2) A Convolutional Transformer Block (layer).
    The classification token is added only in the last stage.

    Args:
        config ([`CvtConfig`]): Model configuration class.
        stage (`int`): Stage number.
    """

    def __init__(self, config: CvtConfig, stage: int, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.stage = stage
        
        # Check if classification token should be added for this stage
        if self.config.cls_token[self.stage]:
            # Initialize classification token weights
            self.cls_token = self.add_weight(
                shape=(1, 1, self.config.embed_dim[-1]),
                initializer=get_initializer(self.config.initializer_range),
                trainable=True,
                name="cvt.encoder.stages.2.cls_token",
            )

        # Initialize token embeddings layer for the current stage
        self.embedding = TFCvtEmbeddings(
            self.config,
            patch_size=config.patch_sizes[self.stage],
            num_channels=config.num_channels if self.stage == 0 else config.embed_dim[self.stage - 1],
            stride=config.patch_stride[self.stage],
            embed_dim=config.embed_dim[self.stage],
            padding=config.patch_padding[self.stage],
            dropout_rate=config.drop_rate[self.stage],
            name="embedding",
        )

        # Compute drop path rates based on the current stage's depth
        drop_path_rates = tf.linspace(0.0, config.drop_path_rate[self.stage], config.depth[stage])
        drop_path_rates = [x.numpy().item() for x in drop_path_rates]
        
        # Initialize convolutional transformer layers for the current stage
        self.layers = [
            TFCvtLayer(
                config,
                num_heads=config.num_heads[self.stage],
                embed_dim=config.embed_dim[self.stage],
                kernel_size=config.kernel_qkv[self.stage],
                stride_q=config.stride_q[self.stage],
                stride_kv=config.stride_kv[self.stage],
                padding_q=config.padding_q[self.stage],
                padding_kv=config.padding_kv[self.stage],
                qkv_projection_method=config.qkv_projection_method[self.stage],
                qkv_bias=config.qkv_bias[self.stage],
                attention_drop_rate=config.attention_drop_rate[self.stage],
                drop_rate=config.drop_rate[self.stage],
                mlp_ratio=config.mlp_ratio[self.stage],
                drop_path_rate=drop_path_rates[self.stage],
                with_cls_token=config.cls_token[self.stage],
                name=f"layers.{j}",
            )
            for j in range(config.depth[self.stage])
        ]
    # 定义一个方法，用于调用模型，接受隐藏状态和训练标志作为参数
    def call(self, hidden_state: tf.Tensor, training: bool = False):
        # 初始化 cls_token 为 None
        cls_token = None
        # 对隐藏状态进行嵌入操作，根据训练标志调用嵌入层的函数
        hidden_state = self.embedding(hidden_state, training)

        # 获取隐藏状态的形状信息：batch_size, height, width, num_channels
        batch_size, height, width, num_channels = shape_list(hidden_state)
        # 计算展平后的隐藏状态大小
        hidden_size = height * width
        # 对隐藏状态进行重新形状操作，将其转换为 batch_size, hidden_size, num_channels
        hidden_state = tf.reshape(hidden_state, shape=(batch_size, hidden_size, num_channels))

        # 如果配置要求在当前阶段使用 cls_token
        if self.config.cls_token[self.stage]:
            # 复制 cls_token，并在 batch 维度上重复
            cls_token = tf.repeat(self.cls_token, repeats=batch_size, axis=0)
            # 将 cls_token 连接到隐藏状态的开头，沿着第二个维度拼接
            hidden_state = tf.concat((cls_token, hidden_state), axis=1)

        # 对每一层进行迭代操作
        for layer in self.layers:
            # 调用当前层的函数，处理隐藏状态，并传入高度、宽度和训练标志
            layer_outputs = layer(hidden_state, height, width, training=training)
            # 更新隐藏状态为当前层的输出
            hidden_state = layer_outputs

        # 如果配置要求在当前阶段使用 cls_token
        if self.config.cls_token[self.stage]:
            # 将隐藏状态分割为 cls_token 和其余部分
            cls_token, hidden_state = tf.split(hidden_state, [1, height * width], 1)

        # 将隐藏状态重新形状为 batch_size, height, width, num_channels
        hidden_state = tf.reshape(hidden_state, shape=(batch_size, height, width, num_channels))
        # 返回最终的隐藏状态和 cls_token
        return hidden_state, cls_token

    # 构建方法，用于构建模型
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 标记模型已构建
        self.built = True
        # 如果存在嵌入层，构建嵌入层
        if getattr(self, "embedding", None) is not None:
            with tf.name_scope(self.embedding.name):
                self.embedding.build(None)
        # 如果存在层列表，逐层构建每一层
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)
class TFCvtEncoder(keras.layers.Layer):
    """
    Convolutional Vision Transformer encoder. CVT has 3 stages of encoder blocks with their respective number of layers
    (depth) being 1, 2 and 10.

    Args:
        config ([`CvtConfig`]): Model configuration class.
    """

    config_class = CvtConfig

    def __init__(self, config: CvtConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        # 初始化 CVT 编码器的各个阶段
        self.stages = [
            TFCvtStage(config, stage_idx, name=f"stages.{stage_idx}") for stage_idx in range(len(config.depth))
        ]

    def call(
        self,
        pixel_values: TFModelInputType,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        training: Optional[bool] = False,
    ) -> Union[TFBaseModelOutputWithCLSToken, Tuple[tf.Tensor]]:
        all_hidden_states = () if output_hidden_states else None
        hidden_state = pixel_values
        # 当在 CPU 上运行时，`keras.layers.Conv2D` 不支持 (batch_size, num_channels, height, width) 作为输入格式。
        # 所以将输入格式更改为 (batch_size, height, width, num_channels)。
        hidden_state = tf.transpose(hidden_state, perm=(0, 2, 3, 1))

        cls_token = None
        for _, (stage_module) in enumerate(self.stages):
            # 逐阶段应用 CVT 编码器，并获取输出的隐藏状态和 CLS token
            hidden_state, cls_token = stage_module(hidden_state, training=training)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

        # 将隐藏状态格式变回 (batch_size, num_channels, height, width)，以保持模块的统一性
        hidden_state = tf.transpose(hidden_state, perm=(0, 3, 1, 2))
        if output_hidden_states:
            all_hidden_states = tuple([tf.transpose(hs, perm=(0, 3, 1, 2)) for hs in all_hidden_states])

        if not return_dict:
            return tuple(v for v in [hidden_state, cls_token, all_hidden_states] if v is not None)

        # 返回 TFBaseModelOutputWithCLSToken 对象，其中包括最终的隐藏状态、CLS token 值和所有隐藏状态的元组
        return TFBaseModelOutputWithCLSToken(
            last_hidden_state=hidden_state,
            cls_token_value=cls_token,
            hidden_states=all_hidden_states,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经构建过，则直接返回
        if getattr(self, "stages", None) is not None:
            # 为每个阶段的层构建名称作用域
            for layer in self.stages:
                with tf.name_scope(layer.name):
                    layer.build(None)


@keras_serializable
class TFCvtMainLayer(keras.layers.Layer):
    """Construct the Cvt model."""

    config_class = CvtConfig

    def __init__(self, config: CvtConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        # 初始化 CVT 主层的编码器部分
        self.encoder = TFCvtEncoder(config, name="encoder")

    @unpack_inputs
    def call(
        self,
        pixel_values: TFModelInputType | None = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[TFBaseModelOutputWithCLSToken, Tuple[tf.Tensor]]:
        # 调用 CVT 编码器的 call 方法来处理输入数据
    # 定义方法，返回值类型为 TFBaseModelOutputWithCLSToken 或者 tf.Tensor 元组
    def __call__(self, pixel_values=None) -> Union[TFBaseModelOutputWithCLSToken, Tuple[tf.Tensor]]:
        # 如果 pixel_values 为空，则抛出数值错误异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 使用 self.encoder 对象处理 pixel_values
        encoder_outputs = self.encoder(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 获取编码器的输出序列
        sequence_output = encoder_outputs[0]

        # 如果 return_dict 为 False，则返回一个元组，包含序列输出和编码器的其他输出
        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        # 如果 return_dict 为 True，则返回 TFBaseModelOutputWithCLSToken 对象
        return TFBaseModelOutputWithCLSToken(
            last_hidden_state=sequence_output,
            cls_token_value=encoder_outputs.cls_token_value,
            hidden_states=encoder_outputs.hidden_states,
        )

    # 定义 build 方法，用于构建模型结构
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记该模型已经构建
        self.built = True
        # 如果 self.encoder 存在，则在命名空间下构建 self.encoder 对象
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
# 导入所需模块或类
class TFCvtPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定配置类为CvtConfig
    config_class = CvtConfig
    # 模型的基础名称前缀为"cvt"
    base_model_prefix = "cvt"
    # 主要输入的名称为"pixel_values"
    main_input_name = "pixel_values"


# 定义模型开始文档字符串的常量
TFCVT_START_DOCSTRING = r"""

    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a [keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
    behavior.

    <Tip>

    TF 2.0 models accepts two formats as inputs:

    - having all inputs as keyword arguments (like PyTorch models), or
    - having all inputs as a list, tuple or dict in the first positional arguments.

    This second option is useful when using [`keras.Model.fit`] method which currently requires having all the
    tensors in the first argument of the model call function: `model(inputs)`.

    </Tip>

    Args:
        config ([`CvtConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义模型输入文档字符串的常量
TFCVT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` ``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`CvtImageProcessor.__call__`]
            for details.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
            used instead.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in
            eager mode, in graph mode the value will always be set to True.
        training (`bool`, *optional*, defaults to `False``):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""


@add_start_docstrings(
    "The bare Cvt Model transformer outputting raw hidden-states without any specific head on top.",
    TFCVT_START_DOCSTRING,
)
# 定义TFCvtModel类，继承自TFCvtPreTrainedModel
class TFCvtModel(TFCvtPreTrainedModel):
    def __init__(self, config: CvtConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        
        # 创建一个 TFCvtMainLayer 类的实例，命名为 cvt
        self.cvt = TFCvtMainLayer(config, name="cvt")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(TFCVT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBaseModelOutputWithCLSToken, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        pixel_values: tf.Tensor | None = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[TFBaseModelOutputWithCLSToken, Tuple[tf.Tensor]]:
        r"""
        模型的前向传播方法，接收输入像素值和一些可选参数，返回模型输出。

        Args:
            pixel_values (tf.Tensor | None): 输入像素值的张量，可以为 None。
            output_hidden_states (Optional[bool]): 是否输出隐藏状态，默认为 None。
            return_dict (Optional[bool]): 是否返回字典格式的输出，默认为 None。
            training (Optional[bool]): 是否在训练模式下，默认为 False。

        Returns:
            Union[TFBaseModelOutputWithCLSToken, Tuple[tf.Tensor]]: 根据 return_dict 参数返回不同类型的输出。

        Examples:

        ```
        >>> from transformers import AutoImageProcessor, TFCvtModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("microsoft/cvt-13")
        >>> model = TFCvtModel.from_pretrained("microsoft/cvt-13")

        >>> inputs = image_processor(images=image, return_tensors="tf")
        >>> outputs = model(**inputs)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""

        # 如果 pixel_values 为 None，则抛出 ValueError
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 调用 self.cvt 的前向传播方法，传入相应的参数
        outputs = self.cvt(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 如果 return_dict 为 False，则返回一个元组
        if not return_dict:
            return (outputs[0],) + outputs[1:]

        # 如果 return_dict 为 True，则返回 TFBaseModelOutputWithCLSToken 类的实例
        return TFBaseModelOutputWithCLSToken(
            last_hidden_state=outputs.last_hidden_state,
            cls_token_value=outputs.cls_token_value,
            hidden_states=outputs.hidden_states,
        )

    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        
        # 如果 self.cvt 存在，则在 tf 的命名空间下构建 self.cvt
        if getattr(self, "cvt", None) is not None:
            with tf.name_scope(self.cvt.name):
                self.cvt.build(None)
    """
    Cvt Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.
    """
    # 基于Cvt模型转换器，顶部带有图像分类头部（在[CLS]标记的最终隐藏状态之上的线性层），例如用于ImageNet。
    TFCvtForImageClassification(TFCvtPreTrainedModel, TFSequenceClassificationLoss):
    
    def __init__(self, config: CvtConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 初始化类别数
        self.num_labels = config.num_labels
        # 创建Cvt主层
        self.cvt = TFCvtMainLayer(config, name="cvt")
        # 使用与原始实现相同的默认epsilon
        self.layernorm = keras.layers.LayerNormalization(epsilon=1e-5, name="layernorm")

        # 分类器头部
        self.classifier = keras.layers.Dense(
            units=config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            use_bias=True,
            bias_initializer="zeros",
            name="classifier",
        )
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(TFCVT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFImageClassifierOutputWithNoAttention, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        pixel_values: tf.Tensor | None = None,
        labels: tf.Tensor | None = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
        # 模型调用函数，接受像素值、标签等参数
        outputs = self.cvt(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )



        # 使用自定义视觉Transformer模型进行前向推断
        # pixel_values: 输入的像素值
        # output_hidden_states: 是否输出隐藏状态
        # return_dict: 是否返回字典格式的输出
        # training: 模型是否处于训练模式



        sequence_output = outputs[0]
        cls_token = outputs[1]



        # 获取模型输出中的序列输出和分类令牌
        # sequence_output: 序列输出
        # cls_token: 分类令牌



        if self.config.cls_token[-1]:
            sequence_output = self.layernorm(cls_token)
        else:



        # 根据配置中的分类令牌，决定如何处理序列输出
        # 若分类令牌存在，则对分类令牌进行 LayerNormalization 处理后作为最终序列输出
        # 否则，进行形状重排以及转置操作，以便进一步处理



            # rearrange "batch_size, num_channels, height, width -> batch_size, (height*width), num_channels"
            batch_size, num_channels, height, width = shape_list(sequence_output)
            sequence_output = tf.reshape(sequence_output, shape=(batch_size, num_channels, height * width))
            sequence_output = tf.transpose(sequence_output, perm=(0, 2, 1))
            sequence_output = self.layernorm(sequence_output)



        # 对序列输出进行形状重排和转置操作，以便后续处理
        # batch_size, num_channels, height, width: 提取序列输出的形状信息
        # sequence_output: 重排和转置后的序列输出，经过 LayerNormalization 处理



        sequence_output_mean = tf.reduce_mean(sequence_output, axis=1)
        logits = self.classifier(sequence_output_mean)



        # 计算序列输出的平均值，并通过分类器生成 logits
        # sequence_output_mean: 序列输出的平均值
        # logits: 经过分类器处理后得到的预测 logits



        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)



        # 如果存在标签，则计算损失值
        # labels: 用于计算分类/回归损失的标签
        # loss: 计算得到的损失值，若无标签则为 None



        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output



        # 根据 return_dict 的设置决定输出格式
        # 如果不返回字典，则将 logits 和其他输出组成元组输出
        # output: 包含 logits 和其他输出的元组



        return TFImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)



        # 返回以 TFImageClassifierOutputWithNoAttention 格式封装的输出
        # loss: 损失值
        # logits: 预测 logits
        # hidden_states: 隐藏状态
    # 如果已经构建过，则直接返回，不重复构建
    if self.built:
        return
    # 设置标志位，表示已经构建
    self.built = True
    
    # 如果存在名为 'cvt' 的属性，并且不为 None，则构建 'cvt' 对象
    if getattr(self, "cvt", None) is not None:
        # 使用 'cvt' 对象的名称作为命名空间，构建它
        with tf.name_scope(self.cvt.name):
            self.cvt.build(None)
    
    # 如果存在名为 'layernorm' 的属性，并且不为 None，则构建 'layernorm' 对象
    if getattr(self, "layernorm", None) is not None:
        # 使用 'layernorm' 对象的名称作为命名空间，构建它
        with tf.name_scope(self.layernorm.name):
            # 构建 'layernorm' 对象，传入输入形状 [None, None, self.config.embed_dim[-1]]
            self.layernorm.build([None, None, self.config.embed_dim[-1]])
    
    # 如果存在名为 'classifier' 的属性，并且不为 None，则构建 'classifier' 对象
    if getattr(self, "classifier", None) is not None:
        # 如果 'classifier' 对象有 'name' 属性，则使用其名称作为命名空间
        if hasattr(self.classifier, "name"):
            with tf.name_scope(self.classifier.name):
                # 构建 'classifier' 对象，传入输入形状 [None, None, self.config.embed_dim[-1]]
                self.classifier.build([None, None, self.config.embed_dim[-1]])
```