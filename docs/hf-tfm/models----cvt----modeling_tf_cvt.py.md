# `.\models\cvt\modeling_tf_cvt.py`

```
# 设置字符编码为 UTF-8
# 版权声明和许可证
# 版权声明为 2022 年版权归 Microsoft Research 和 The HuggingFace Inc. 团队所有，保留所有权利
# 根据 Apache 许可证版本 2.0 许可使用本文件
# 除非适用法律要求或以书面形式同意，否则不得使用此文件
# 您可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或以书面形式同意，否则不得分发的软件以"原样"分发，不附带任何形式的担保或条件，无论是明示的还是暗示的
# 请参阅许可证以了解特定语言管理权限和限制
""" TF 2.0 Cvt model."""
# 导入相应的库
# 注：注释符后面是代码的空格

from __future__ import annotations

import collections.abc
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import tensorflow as tf

# 导入模型输出、TFModelBase 类等
# 注：注释符后面是代码的空格
from ...modeling_tf_outputs import TFImageClassifierOutputWithNoAttention
from ...modeling_tf_utils import (
    TFModelInputType,
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    get_initializer,
    keras_serializable,
    unpack_inputs,
)
# 导入帮助函数、日志、配置文件等
# 注：注释符后面是代码的空格
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# 导入 CvtConfig 配置文件类
# 注：注释符后面是代码的空格
from .configuration_cvt import CvtConfig

# 获取相应的日志记录器
logger = logging.get_logger(__name__)

# Cvt 模型的配置文件
_CONFIG_FOR_DOC = "CvtConfig"

# 预训练的 Cvt 模型归档列表
TF_CVT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/cvt-13",
    "microsoft/cvt-13-384",
    "microsoft/cvt-13-384-22k",
    "microsoft/cvt-21",
    "microsoft/cvt-21-384",
    "microsoft/cvt-21-384-22k",
    # 在此处查看所有的 Cvt 模型：https://huggingface.co/models?filter=cvt
]

# 模型输出的基类，包含了模型的输出数据
@dataclass
class TFBaseModelOutputWithCLSToken(ModelOutput):
    """
    Base class for model's outputs.

    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        cls_token_value (`tf.Tensor` of shape `(batch_size, 1, hidden_size)`):
            Classification token at the output of the last layer of the model.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer plus
            the initial embedding outputs.
    """

    last_hidden_state: tf.Tensor = None
    cls_token_value: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None

# Cvt 模型的 DropPath 类，其中包含注释引用信息
# 注：注释符后面是代码的空格
class TFCvtDropPath(tf.keras.layers.Layer):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    References:
        (1) github.com:rwightman/pytorch-image-models
    """
    # 初始化函数，设置丢弃概率，并调用父类的初始化方法
    def __init__(self, drop_prob: float, **kwargs):
        super().__init__(**kwargs)
        # 存储丢弃概率
        self.drop_prob = drop_prob

    # 调用层的函数，用于执行层的正向传播
    def call(self, x: tf.Tensor, training=None):
        # 如果丢弃概率为零或者不处于训练模式，则直接返回输入张量
        if self.drop_prob == 0.0 or not training:
            return x
        # 计算保留概率
        keep_prob = 1 - self.drop_prob
        # 获取张量 x 的形状
        shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
        # 生成服从均匀分布的随机张量，用于控制是否丢弃
        random_tensor = keep_prob + tf.random.uniform(shape, 0, 1, dtype=self.compute_dtype)
        # 将随机张量取整，得到二值掩码，用于标记要保留的元素
        random_tensor = tf.floor(random_tensor)
        # 对输入张量进行按元素的丢弃处理，保证期望值不变
        return (x / keep_prob) * random_tensor
# 定义一个名为TFCvtEmbeddings的类，用于构建卷积令牌嵌入
class TFCvtEmbeddings(tf.keras.layers.Layer):
    """Construct the Convolutional Token Embeddings."""

    # 初始化方法，接受一系列参数，并调用父类的初始化方法
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
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 创建一个名为convolution_embeddings的TFCvtConvEmbeddings对象
        self.convolution_embeddings = TFCvtConvEmbeddings(
            config,
            patch_size=patch_size,
            num_channels=num_channels,
            embed_dim=embed_dim,
            stride=stride,
            padding=padding,
            name="convolution_embeddings",
        )
        # 创建一个名为dropout的Dropout层
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    # 用于调用类实例时执行的方法，接受像素值和训练标志作为输入，返回隐藏状态
    def call(self, pixel_values: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 使用convolution_embeddings对象处理像素值，得到隐藏状态
        hidden_state = self.convolution_embeddings(pixel_values)
        # 对隐藏状态进行dropout操作
        hidden_state = self.dropout(hidden_state, training=training)
        # 返回处理后的隐藏状态
        return hidden_state

    # 用于构建层的方法
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 设置为已构建状态
        self.built = True
        # 如果convolution_embeddings存在
        if getattr(self, "convolution_embeddings", None) is not None:
            # 在convolution_embeddings的命名空间下构建该层
            with tf.name_scope(self.convolution_embeddings.name):
                self.convolution_embeddings.build(None)


# 定义一个名为TFCvtConvEmbeddings的类，用于将图像转换为卷积嵌入，模拟局部空间上下文
class TFCvtConvEmbeddings(tf.keras.layers.Layer):
    """Image to Convolution Embeddings. This convolutional operation aims to model local spatial contexts."""

    # 初始化方法，接受一系列参数，并调用父类的初始化方法
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
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 创建一个ZeroPadding2D层，用于填充
        self.padding = tf.keras.layers.ZeroPadding2D(padding=padding)
        # 定义一个投影层，用于卷积操作
        self.patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        self.projection = tf.keras.layers.Conv2D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=stride,
            padding="valid",
            data_format="channels_last",
            kernel_initializer=get_initializer(config.initializer_range),
            name="projection",
        )
        # 使用与PyTorch相同的默认epsilon值
        self.normalization = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="normalization")
        self.num_channels = num_channels
        self.embed_dim = embed_dim
    # 定义一个函数，接受像素值的张量输入并返回张量输出
    def call(self, pixel_values: tf.Tensor) -> tf.Tensor:
        # 检查像素值是否为字典类型，如果是则取"pixel_values"对应的值
        if isinstance(pixel_values, dict):
            pixel_values = pixel_values["pixel_values"]

        # 对输入的像素值进行填充和投影操作
        pixel_values = self.projection(self.padding(pixel_values))

        # 获取输入像素值的形状信息
        batch_size, height, width, num_channels = shape_list(pixel_values)
        hidden_size = height * width

        # 重新调整像素值的形状为(batch_size, hidden_size, num_channels)
        pixel_values = tf.reshape(pixel_values, shape=(batch_size, hidden_size, num_channels))

        # 对像素值进行规范化操作
        pixel_values = self.normalization(pixel_values)

        # 将像素值重新调整为原始形状(batch_size, height, width, num_channels)
        pixel_values = tf.reshape(pixel_values, shape=(batch_size, height, width, num_channels))

        # 返回处理后的像素值
        return pixel_values

    # 构建函数，在该函数中对投影和规范化进行构建
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True

        # 如果存在投影函数，则进行构建
        if getattr(self, "projection", None) is not None:
            with tf.name_scope(self.projection.name):
                self.projection.build([None, None, None, self.num_channels])

        # 如果存在规范化函数，则进行构建
        if getattr(self, "normalization", None) is not None:
            with tf.name_scope(self.normalization.name):
                self.normalization.build([None, None, self.embed_dim])
class TFCvtSelfAttentionConvProjection(tf.keras.layers.Layer):
    """Convolutional projection layer."""
    # 卷积投影层的定义与功能注释

    def __init__(self, config: CvtConfig, embed_dim: int, kernel_size: int, stride: int, padding: int, **kwargs):
        super().__init__(**kwargs)
        # 继承父类的初始化函数
        self.padding = tf.keras.layers.ZeroPadding2D(padding=padding)
        # 创建ZeroPadding2D层对象，用于对输入进行零填充
        self.convolution = tf.keras.layers.Conv2D(
            filters=embed_dim,
            kernel_size=kernel_size,
            kernel_initializer=get_initializer(config.initializer_range),
            padding="valid",
            strides=stride,
            use_bias=False,
            name="convolution",
            groups=embed_dim,
        )
        # 创建Conv2D层对象，实现二维卷积
        # filters为输出空间的维度
        # kernel_size为卷积核大小
        # kernel_initializer为卷积核初始化方式
        # padding为填充方式
        # strides为卷积步幅
        # use_bias为是否使用偏置项
        # name为层的名称
        # groups为卷积的分组数
        self.normalization = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="normalization")
        # 创建BatchNormalization层对象，实现批量归一化
        # epsilon为用于增加计算稳定性的小常数
        # momentum为移动平均算法的动量
        self.embed_dim = embed_dim

    def call(self, hidden_state: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_state = self.convolution(self.padding(hidden_state))
        # 卷积层对输入进行卷积操作，并对输入进行填充
        hidden_state = self.normalization(hidden_state, training=training)
        # 批量归一化输入特征
        return hidden_state

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "convolution", None) is not None:
            with tf.name_scope(self.convolution.name):
                self.convolution.build([None, None, None, self.embed_dim])
        if getattr(self, "normalization", None) is not None:
            with tf.name_scope(self.normalization.name):
                self.normalization.build([None, None, None, self.embed_dim])


class TFCvtSelfAttentionLinearProjection(tf.keras.layers.Layer):
    """Linear projection layer used to flatten tokens into 1D."""
    # 用于将标记展平为1D的线性投影层的定义与功能注释

    def call(self, hidden_state: tf.Tensor) -> tf.Tensor:
        # "batch_size, height, width, num_channels -> batch_size, (height*width), num_channels"
        # 将张量形状从"批次大小、高度、宽度、通道数"变为"��次大小、(高度*宽度)、通道数"
        batch_size, height, width, num_channels = shape_list(hidden_state)
        hidden_size = height * width
        hidden_state = tf.reshape(hidden_state, shape=(batch_size, hidden_size, num_channels))
        return hidden_state


class TFCvtSelfAttentionProjection(tf.keras.layers.Layer):
    """Convolutional Projection for Attention."""
    # 用于注意力的卷积投影定义与功能注释

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
        # 继承父类的初始化函数
        if projection_method == "dw_bn":
            self.convolution_projection = TFCvtSelfAttentionConvProjection(
                config, embed_dim, kernel_size, stride, padding, name="convolution_projection"
            )
        # 根据projection_method选择相应的投影方法
        # 创建TFCvtSelfAttentionConvProjection层对象，实现卷积投影
        self.linear_projection = TFCvtSelfAttentionLinearProjection()
    # 定义一个类方法call，用于对隐藏状态进行一系列处理，并返回结果
    def call(self, hidden_state: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 对隐藏状态进行卷积投影处理，并根据训练模式选择是否训练
        hidden_state = self.convolution_projection(hidden_state, training=training)
        # 对处理后的隐藏状态进行线性投影
        hidden_state = self.linear_projection(hidden_state)
        # 返回处理后的隐藏状态
        return hidden_state

    # 定义一个build方法
    def build(self, input_shape=None):
        # 如果已经构建过模型，则直接返回
        if self.built:
            return
        # 标记模型已经构建过
        self.built = True
        # 如果类属性convolution_projection存在
        if getattr(self, "convolution_projection", None) is not None:
            # 使用tf.name_scope创建一个名称作用域，并将名称作用域设置为属性convolution_projection的name属性
            with tf.name_scope(self.convolution_projection.name):
                # 调用convolution_projection的build方法构建模型
                self.convolution_projection.build(None)
# 定义一个名为TFCvtSelfAttention的类，继承自tf.keras.layers.Layer
class TFCvtSelfAttention(tf.keras.layers.Layer):
    """
    Self-attention layer. A depth-wise separable convolution operation (Convolutional Projection), is applied for
    query, key, and value embeddings.
    """
    # 初始化方法
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
        with_cls_token: bool = True,
        **kwargs,
    ):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 定义缩放因子，公式为embed_dim的-0.5次方
        self.scale = embed_dim**-0.5
        # 是否包含CLS token标志位
        self.with_cls_token = with_cls_token
        # embedding维度
        self.embed_dim = embed_dim
        # 多头注意力的头数
        self.num_heads = num_heads

        # 创建一个TFCvtSelfAttentionProjection对象用于query投影
        self.convolution_projection_query = TFCvtSelfAttentionProjection(
            config,
            embed_dim,
            kernel_size,
            stride_q,
            padding_q,
            projection_method="linear" if qkv_projection_method == "avg" else qkv_projection_method,
            name="convolution_projection_query",
        )
        # 创建一个TFCvtSelfAttentionProjection对象用于key投影
        self.convolution_projection_key = TFCvtSelfAttentionProjection(
            config,
            embed_dim,
            kernel_size,
            stride_kv,
            padding_kv,
            projection_method=qkv_projection_method,
            name="convolution_projection_key",
        )
        # 创建一个TFCvtSelfAttentionProjection对象用于value投影
        self.convolution_projection_value = TFCvtSelfAttentionProjection(
            config,
            embed_dim,
            kernel_size,
            stride_kv,
            padding_kv,
            projection_method=qkv_projection_method,
            name="convolution_projection_value",
        )

        # 创建一个全连接层用于query投影
        self.projection_query = tf.keras.layers.Dense(
            units=embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            use_bias=qkv_bias,
            bias_initializer="zeros",
            name="projection_query",
        )
        # 创建一个全连接层用于key投影
        self.projection_key = tf.keras.layers.Dense(
            units=embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            use_bias=qkv_bias,
            bias_initializer="zeros",
            name="projection_key",
        )
        # 创建一个全连接层用于value投影
        self.projection_value = tf.keras.layers.Dense(
            units=embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            use_bias=qkv_bias,
            bias_initializer="zeros",
            name="projection_value",
        )
        # 创建一个Dropout层，用于dropout操作
        self.dropout = tf.keras.layers.Dropout(attention_drop_rate)
    def rearrange_for_multi_head_attention(self, hidden_state: tf.Tensor) -> tf.Tensor:
        # 获取 hidden_state 的形状信息
        batch_size, hidden_size, _ = shape_list(hidden_state)
        # 计算每个 head 的维度
        head_dim = self.embed_dim // self.num_heads
        # 重新排列 hidden_state 的形状
        hidden_state = tf.reshape(hidden_state, shape=(batch_size, hidden_size, self.num_heads, head_dim))
        # 调换维度
        hidden_state = tf.transpose(hidden_state, perm=(0, 2, 1, 3))
        # 返回重新排列的 hidden_state
        return hidden_state

    def call(self, hidden_state: tf.Tensor, height: int, width: int, training: bool = False) -> tf.Tensor:
        if self.with_cls_token:
            # 如果存在 cls_token，将其从 hidden_state 中分离出来
            cls_token, hidden_state = tf.split(hidden_state, [1, height * width], 1)

        # 重塑 hidden_state 的形状
        batch_size, hidden_size, num_channels = shape_list(hidden_state)
        hidden_state = tf.reshape(hidden_state, shape=(batch_size, height, width, num_channels))

        # 获取 key、query、value
        key = self.convolution_projection_key(hidden_state, training=training)
        query = self.convolution_projection_query(hidden_state, training=training)
        value = self.convolution_projection_value(hidden_state, training=training)

        if self.with_cls_token:
            # 如果存在 cls_token，将其添加到 query、key、value 中
            query = tf.concat((cls_token, query), axis=1)
            key = tf.concat((cls_token, key), axis=1)
            value = tf.concat((cls_token, value), axis=1)

        # 计算每个 head 的维度
        head_dim = self.embed_dim // self.num_heads

        # 对 query、key、value 进行多头注意力机制的重排列
        query = self.rearrange_for_multi_head_attention(self.projection_query(query))
        key = self.rearrange_for_multi_head_attention(self.projection_key(key))
        value = self.rearrange_for_multi_head_attention(self.projection_value(value))

        # 计算注意力分数
        attention_score = tf.matmul(query, key, transpose_b=True) * self.scale
        # 计算注意力概率
        attention_probs = stable_softmax(logits=attention_score, axis=-1)
        attention_probs = self.dropout(attention_probs, training=training)

        # 计算上下文
        context = tf.matmul(attention_probs, value)
        # 调整 context 的形状
        _, _, hidden_size, _ = shape_list(context)
        context = tf.transpose(context, perm=(0, 2, 1, 3))
        context = tf.reshape(context, (batch_size, hidden_size, self.num_heads * head_dim))
        # 返回 context
        return context
    # 构建函数，用于构建自定义层
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 标记该层已经构建
        self.built = True
        # 如果存在卷积投影查询对象，构建该对象
        if getattr(self, "convolution_projection_query", None) is not None:
            with tf.name_scope(self.convolution_projection_query.name):
                self.convolution_projection_query.build(None)
        # 如果存在卷积投影关键字对象，构建该对象
        if getattr(self, "convolution_projection_key", None) is not None:
            with tf.name_scope(self.convolution_projection_key.name):
                self.convolution_projection_key.build(None)
        # 如果存在卷积投影值对象，构建该对象
        if getattr(self, "convolution_projection_value", None) is not None:
            with tf.name_scope(self.convolution_projection_value.name):
                self.convolution_projection_value.build(None)
        # 如果存在投影查询对象，构建该对象
        if getattr(self, "projection_query", None) is not None:
            with tf.name_scope(self.projection_query.name):
                self.projection_query.build([None, None, self.embed_dim])
        # 如果存在投影关键字对象，构建该对象
        if getattr(self, "projection_key", None) is not None:
            with tf.name_scope(self.projection_key.name):
                self.projection_key.build([None, None, self.embed_dim])
        # 如果存在投影值对象，构建该对象
        if getattr(self, "projection_value", None) is not None:
            with tf.name_scope(self.projection_value.name):
                self.projection_value.build([None, None, self.embed_dim])
```  
class TFCvtSelfOutput(tf.keras.layers.Layer):
    """Attention层的输出。"""

    def __init__(self, config: CvtConfig, embed_dim: int, drop_rate: float, **kwargs):
        super().__init__(**kwargs)
        # 创建一个全连接层，用于将输入映射到指定维度的输出
        self.dense = tf.keras.layers.Dense(
            units=embed_dim, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建一个用于随机失活的层，用于防止过拟合
        self.dropout = tf.keras.layers.Dropout(drop_rate)
        # 保存输出的维度
        self.embed_dim = embed_dim

    def call(self, hidden_state: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 输入通过全连接层进行变换
        hidden_state = self.dense(inputs=hidden_state)
        # 对输出进行随机失活，有选择地丢弃部分神经元以减少过拟合
        hidden_state = self.dropout(inputs=hidden_state, training=training)
        return hidden_state

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                # 构建全连接层
                self.dense.build([None, None, self.embed_dim])


class TFCvtAttention(tf.keras.layers.Layer):
    """注意力层。卷积转换块的第一部分。"""

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
        # 创建自注意力层，用于学习输入序列内元素之间的依赖关系
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
        # 创建输出层，用于处理自注意力层的输出
        self.dense_output = TFCvtSelfOutput(config, embed_dim, drop_rate, name="output")

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(self, hidden_state: tf.Tensor, height: int, width: int, training: bool = False):
        # 输入通过自注意力层进行处理
        self_output = self.attention(hidden_state, height, width, training=training)
        # 处理自注意力层的输出
        attention_output = self.dense_output(self_output, training=training)
        return attention_output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                # 构建自注意力层
                self.attention.build(None)
        if getattr(self, "dense_output", None) is not None:
            with tf.name_scope(self.dense_output.name):
                # 构建输出层
                self.dense_output.build(None)


class TFCvtIntermediate(tf.keras.layers.Layer):
    """中间密集层。卷积转换块的第二部分。"""
    # 初始化函数，用于创建一个新的实例
    def __init__(self, config: CvtConfig, embed_dim: int, mlp_ratio: int, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)
        # 创建一个全连接层，设置输出维度为 embed_dim * mlp_ratio，激活函数为 gelu
        self.dense = tf.keras.layers.Dense(
            units=int(embed_dim * mlp_ratio),
            kernel_initializer=get_initializer(config.initializer_range),
            activation="gelu",
            name="dense",
        )
        # 保存嵌入维度信息
        self.embed_dim = embed_dim

    # 调用函数，用于将输入 hidden_state 经过全连接层处理后返回
    def call(self, hidden_state: tf.Tensor) -> tf.Tensor:
        # 将隐藏状态输入全连接层，并返回处理后的结果
        hidden_state = self.dense(hidden_state)
        return hidden_state

    # 构建函数，用于构建模型的层次结构
    def build(self, input_shape=None):
        # 如果模型已经构建完成，则直接返回
        if self.built:
            return
        # 设置模型为已构建状态
        self.built = True
        # 如果存在全连接层 dense
        if getattr(self, "dense", None) is not None:
            # 在 dense 层的命名空间下，构建 dense 层，输入形状为 [None, None, self.embed_dim]
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.embed_dim])
class TFCvtOutput(tf.keras.layers.Layer):
    """
    Output of the Convolutional Transformer Block (last chunk). It consists of a MLP and a residual connection.
    """

    def __init__(self, config: CvtConfig, embed_dim: int, mlp_ratio: int, drop_rate: int, **kwargs):
        # 初始化函数，定义层的属性和参数
        super().__init__(**kwargs)
        # 创建一个全连接层，指定输出维度和初始化方法
        self.dense = tf.keras.layers.Dense(
            units=embed_dim, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建一个 Dropout 层，用于防止过拟合
        self.dropout = tf.keras.layers.Dropout(drop_rate)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio

    def call(self, hidden_state: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 定义层的前向传播逻辑
        hidden_state = self.dense(inputs=hidden_state)
        hidden_state = self.dropout(inputs=hidden_state, training=training)
        hidden_state = hidden_state + input_tensor
        return hidden_state

    def build(self, input_shape=None):
        # 构建层的参数
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, int(self.embed_dim * self.mlp_ratio)])


class TFCvtLayer(tf.keras.layers.Layer):
    """
    Convolutional Transformer Block composed by attention layers, normalization and multi-layer perceptrons (mlps). It
    consists of 3 chunks : an attention layer, an intermediate dense layer and an output layer. This corresponds to the
    `Block` class in the original implementation.
    """

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
        mlp_ratio: float,
        drop_path_rate: float,
        with_cls_token: bool = True,
        **kwargs,
    # 定义一个名为TFCvtAttention的类，继承自tf.keras.layers.Layer类
    def __init__(
        self,
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
    ):
        super().__init__(**kwargs)
        
        # 创建一个名为attention的TFCvtAttention对象，用于自注意力计算
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
        
        # 创建一个名为intermediate的TFCvtIntermediate对象，用于中间层计算
        self.intermediate = TFCvtIntermediate(config, embed_dim, mlp_ratio, name="intermediate")
        
        # 创建一个名为dense_output的TFCvtOutput对象，用于输出层计算
        self.dense_output = TFCvtOutput(config, embed_dim, mlp_ratio, drop_rate, name="output")
        
        # 判断是否需要Drop Path操作
        if drop_path_rate > 0.0:
            # 如果需要Drop Path操作，创建一个名为drop_path的TFCvtDropPath对象
            # 使用TFCvtDropPath类进行Drop Path操作
            self.drop_path = TFCvtDropPath(drop_path_rate, name="drop_path")
        else:
            # 如果不需要Drop Path操作，创建一个名为drop_path的Activation对象
            # 使用tf.keras.layers.Activation类的"linear"激活函数
            self.drop_path = tf.keras.layers.Activation("linear", name="drop_path")
        
        # 创建一个名为layernorm_before的LayerNormalization对象，用于注意力之前的LayerNorm计算
        # 设置epsilon参数为1e-5，这个参数用于防止除以零，保证数值稳定性
        self.layernorm_before = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layernorm_before")
        
        # 创建一个名为layernorm_after的LayerNormalization对象，用于注意力之后的LayerNorm计算
        # 设置epsilon参数为1e-5，这个参数用于防止除以零，保证数值稳定性
        self.layernorm_after = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layernorm_after")
        
        # 将嵌入维度赋值给成员变量embed_dim
        self.embed_dim = embed_dim

    # 实现call方法，用于定义模型的前向传播过程
    def call(self, hidden_state: tf.Tensor, height: int, width: int, training: bool = False) -> tf.Tensor:
        # 在hidden_state上使用layernorm_before进行LayerNorm计算
        # 然后传入注意力模块self.attention进行自注意力计算
        attention_output = self.attention(self.layernorm_before(hidden_state), height, width, training=training)
        
        # 根据training参数判断是否执行Drop Path操作，对attention_output进行Drop Path
        attention_output = self.drop_path(attention_output, training=training)

        # 实现第一个残差连接，将attention_output与hidden_state相加
        hidden_state = attention_output + hidden_state

        # 在hidden_state上使用layernorm_after进行LayerNorm计算
        layer_output = self.layernorm_after(hidden_state)
        
        # 通过中间层计算模块self.intermediate对layer_output进行中间层计算
        layer_output = self.intermediate(layer_output)

        # 实现第二个残差连接，将layer_output与hidden_state相加
        layer_output = self.dense_output(layer_output, hidden_state)
        
        # 根据training参数判断是否执行Drop Path操作，对layer_output进行Drop Path
        layer_output = self.drop_path(layer_output, training=training)
        
        # 返回layer_output作为函数输出
        return layer_output
    # 建立模型，如果已经建立则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 设置模型为已建立状态
        self.built = True
        # 如果存在注意力机制，则建立并设置相应的名字作用域
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        # 如果存在中间层，则建立并设置相应的名字作用域
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        # 如果存在密集层输出，则建立并设置相应的名字作用域
        if getattr(self, "dense_output", None) is not None:
            with tf.name_scope(self.dense_output.name):
                self.dense_output.build(None)
        # 如果存在随机路径，则建立并设置相应的名字作用域
        if getattr(self, "drop_path", None) is not None:
            with tf.name_scope(self.drop_path.name):
                self.drop_path.build(None)
        # 如果存在前置层归一化，则建立并设置相应的名字作用域
        if getattr(self, "layernorm_before", None) is not None:
            with tf.name_scope(self.layernorm_before.name):
                self.layernorm_before.build([None, None, self.embed_dim])
        # 如果存在后置层归一化，则建立并设置相应的名字作用域
        if getattr(self, "layernorm_after", None) is not None:
            with tf.name_scope(self.layernorm_after.name):
                self.layernorm_after.build([None, None, self.embed_dim])
class TFCvtStage(tf.keras.layers.Layer):
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
        self.config = config  # 设置类属性 config 为传入的参数
        self.stage = stage  # 设置类属性 stage 为传入的参数
        if self.config.cls_token[self.stage]:  # 如果当前阶段需要添加分类 token
            self.cls_token = self.add_weight(  # 添加分类 token 的权重
                shape=(1, 1, self.config.embed_dim[-1]),
                initializer=get_initializer(self.config.initializer_range),
                trainable=True,
                name="cvt.encoder.stages.2.cls_token",
            )

        self.embedding = TFCvtEmbeddings(  # 创建 Convolutional Token Embedding 层
            self.config,
            patch_size=config.patch_sizes[self.stage],
            num_channels=config.num_channels if self.stage == 0 else config.embed_dim[self.stage - 1],
            stride=config.patch_stride[self.stage],
            embed_dim=config.embed_dim[self.stage],
            padding=config.patch_padding[self.stage],
            dropout_rate=config.drop_rate[self.stage],
            name="embedding",
        )

        drop_path_rates = tf.linspace(0.0, config.drop_path_rate[self.stage], config.depth[stage])  # 计算 drop path 的比率
        drop_path_rates = [x.numpy().item() for x in drop_path_rates]  # 转换为可迭代的列表
        self.layers = [
            TFCvtLayer(  # 创建 Convolutional Transformer Block 层
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
            for j in range(config.depth[self.stage])  # 根据阶段的深度创建对应数量的层
        ]
    # 定义一个call方法，接受隐藏状态和训练标志作为输入
    def call(self, hidden_state: tf.Tensor, training: bool = False):
        # 初始化cls_token为空
        cls_token = None
        # 对隐藏状态应用嵌入层，根据训练标志进行处理
        hidden_state = self.embedding(hidden_state, training)
        
        # 获取隐藏状态的形状信息：batch_size, height, width, num_channels
        batch_size, height, width, num_channels = shape_list(hidden_state)
        # 计算隐藏大小
        hidden_size = height * width
        # 调整隐藏状态的形状为batch_size, hidden_size, num_channels
        hidden_state = tf.reshape(hidden_state, shape=(batch_size, hidden_size, num_channels))
        
        # 如果配置的cls_token为真，重复cls_token并与hidden_state连接
        if self.config.cls_token[self.stage]:
            cls_token = tf.repeat(self.cls_token, repeats=batch_size, axis=0)
            hidden_state = tf.concat((cls_token, hidden_state), axis=1)
        
        # 遍历所有层并调用它们，将输出赋给hidden_state
        for layer in self.layers:
            layer_outputs = layer(hidden_state, height, width, training=training)
            hidden_state = layer_outputs
        
        # 如果配置的cls_token为真，将hidden_state拆分为cls_token和hidden_state
        if self.config.cls_token[self.stage]:
            cls_token, hidden_state = tf.split(hidden_state, [1, height * width], 1)
        
        # 调整hidden_state的形状为(batch_size, height, width, num_channels)
        hidden_state = tf.reshape(hidden_state, shape=(batch_size, height, width, num_channels))
        # 返回隐藏状态和cls_token
        return hidden_state, cls_token

    # 构建模型
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 将built标记为True，表示已构建过
        self.built = True
        # 如果embedding层存在，则遍历调用embedding层的build方法
        if getattr(self, "embedding", None) is not None:
            with tf.name_scope(self.embedding.name):
                self.embedding.build(None)
        # 如果layers存在，则遍历调用每个层的build方法
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)
class TFCvtEncoder(tf.keras.layers.Layer):
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
        # 初始化三个阶段的编码器块，每个块有不同数量的层
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
        # 如果需要输出隐藏状态，则初始化一个空元组
        all_hidden_states = () if output_hidden_states else None
        # 将输入的像素值转置，以适应 Conv2D 层的输入格式要求
        hidden_state = pixel_values
        hidden_state = tf.transpose(hidden_state, perm=(0, 2, 3, 1))

        cls_token = None
        # 遍历每个阶段的编码器块
        for _, (stage_module) in enumerate(self.stages):
            # 调用每个编码器块
            hidden_state, cls_token = stage_module(hidden_state, training=training)
            # 如果需要输出隐藏状态，则将当前隐藏状态加入到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

        # 将输入格式还原为(batch_size, num_channels, height, width)以保持模块的一致性
        hidden_state = tf.transpose(hidden_state, perm=(0, 3, 1, 2))
        if output_hidden_states:
            # 将所有隐藏状态转置回原始格式
            all_hidden_states = tuple([tf.transpose(hs, perm=(0, 3, 1, 2)) for hs in all_hidden_states])

        # 如果不需要返回字典形式的输出，就返回隐藏状态、CLS 标记和所有隐藏状态
        if not return_dict:
            return tuple(v for v in [hidden_state, cls_token, all_hidden_states] if v is not None)

        # 如果需要返回字典形式的输出，就返回 TFBaseModelOutputWithCLSToken 对象
        return TFBaseModelOutputWithCLSToken(
            last_hidden_state=hidden_state,
            cls_token_value=cls_token,
            hidden_states=all_hidden_states,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "stages", None) is not None:
            # 构建每个编码器块的层
            for layer in self.stages:
                with tf.name_scope(layer.name):
                    layer.build(None)


@keras_serializable
class TFCvtMainLayer(tf.keras.layers.Layer):
    """Construct the Cvt model."""

    config_class = CvtConfig

    def __init__(self, config: CvtConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        # 初始化 CVT 编码器
        self.encoder = TFCvtEncoder(config, name="encoder")

    @unpack_inputs
    def call(
        self,
        pixel_values: TFModelInputType | None = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    # 定义一个方法，接受像素值作为输入，并返回包含 CLS 标记的模型输出或包含张量的元组
    def call(self, pixel_values: tf.Tensor, training=False, output_hidden_states=False, return_dict=True
        # 如果像素值为 None，则抛出值错误异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 使用编码器对像素值进行编码，获取编码器的输出
        encoder_outputs = self.encoder(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 从编码器的输出中获取序列输出
        sequence_output = encoder_outputs[0]

        # 如果不返回字典，则返回一个包含序列输出和其他编码器输出的元组
        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        # 如果返回字典，则返回一个带有 CLS 标记值的 TFBaseModelOutputWithCLSToken 对象
        return TFBaseModelOutputWithCLSToken(
            last_hidden_state=sequence_output,
            cls_token_value=encoder_outputs.cls_token_value,
            hidden_states=encoder_outputs.hidden_states,
        )

    # 构建模型
    def build(self, input_shape=None):
        # 如果已经构建了模型，则直接返回
        if self.built:
            return
        # 设置模型已经构建的标志
        self.built = True
        # 如果编码器存在，则在编码器的命名范围内构建编码器
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                # 构建编码器，传入空输入形状
                self.encoder.build(None)
# 定义一个 TFCvtPreTrainedModel 类，作为处理权重初始化以及下载和加载预训练模型的抽象类
class TFCvtPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 配置文件类为 CvtConfig
    config_class = CvtConfig
    # 基础模型前缀为 "cvt"
    base_model_prefix = "cvt"
    # 主输入名称为 "pixel_values"
    main_input_name = "pixel_values"


# 下面是 TFCvtModel 类的文档说明
# 继承自 TFPreTrainedModel 类，查看超类文档以了解库为所有模型实现的通用方法（如下载或保存、调整输入嵌入、修剪头等）
# 该模型也是 tf.keras.Model 的子类，可以将其用作常规的 TF 2.0 Keras 模型，并参考 TF 2.0 文档以了解一切与通用用法和行为相关的事宜
# 提示：TF 2.0 模型接受两种格式的输入：将所有输入作为关键字参数（如 PyTorch 模型）；或将所有输入作为列表、元组或字典置于第一个位置参数
# 第二种选项在使用 tf.keras.Model.fit 方法时特别有用，该方法目前要求将所有张量放在模型调用函数的第一个参数中，如：model(inputs)
# 参数：config（CvtConfig 类）- 具有模型所有参数的模型配置类。使用配置文件初始化只会加载与模型相关联的权重，而不是配置。查看 TFPreTrainedModel.from_pretrained 方法以加载模型权重
TFCVT_START_DOCSTRING = r"""

    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
    behavior.

    <Tip>

    TF 2.0 models accepts two formats as inputs:

    - having all inputs as keyword arguments (like PyTorch models), or
    - having all inputs as a list, tuple or dict in the first positional arguments.

    This second option is useful when using [`tf.keras.Model.fit`] method which currently requires having all the
    tensors in the first argument of the model call function: `model(inputs)`.

    </Tip>

    Args:
        config ([`CvtConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 输入文档字���串，描述了模型的预期输入
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
        training (`bool`, *optional*, defaults to `False`):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between
    def __init__(self, config: CvtConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 创建TFCvtMainLayer对象并且命名为"cvt"
        self.cvt = TFCvtMainLayer(config, name="cvt")

    # 装饰器，将函数输入参数解包
    # 增加model_forward()的文档字符串
    # 替换返回值文档字符串
    def call(
        self,
        pixel_values: tf.Tensor | None = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[TFBaseModelOutputWithCLSToken, Tuple[tf.Tensor]]:
        r"""
        Returns:

        Examples:

        ```python
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

        # 如果pixel_values为None，则抛出数值错误
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 调用self.cvt对象的call方法，并传入参数
        outputs = self.cvt(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 如果return_dict为False，则返回元组(输出的第一个元素，其余输出)
        if not return_dict:
            return (outputs[0],) + outputs[1:]

        # 返回TFBaseModelOutputWithCLSToken对象，包含last_hidden_state, cls_token_value, hidden_states
        return TFBaseModelOutputWithCLSToken(
            last_hidden_state=outputs.last_hidden_state,
            cls_token_value=outputs.cls_token_value,
            hidden_states=outputs.hidden_states,
        )

    # 构建方法，如果已构建则返回，构建cvt对象
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "cvt", None) is not None:
            with tf.name_scope(self.cvt.name):
                self.cvt.build(None)
# 使用装饰器为类添加起始文档字符串，描述了该模型的作用，以及在顶部添加了一个图像分类头的 Cvt 模型转换器（在 [CLS] 标记的最终隐藏状态之上的线性层）的示例，例如用于 ImageNet。
@add_start_docstrings(
    """
    Cvt Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.
    """,
    TFCVT_START_DOCSTRING,  # 添加了一个起始文档字符串
)
class TFCvtForImageClassification(TFCvtPreTrainedModel, TFSequenceClassificationLoss):
    # 初始化函数
    def __init__(self, config: CvtConfig, *inputs, **kwargs):
        # 调用父类的初始化函数
        super().__init__(config, *inputs, **kwargs)

        # 设置模型的标签数量
        self.num_labels = config.num_labels
        # 创建一个 Cvt 主层，使用给定的配置
        self.cvt = TFCvtMainLayer(config, name="cvt")
        # 使用与原始实现中相同的默认 epsilon 初始化 LayerNormalization 层
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layernorm")

        # 分类器头部
        # 创建一个密集连接层，用于分类，单元数为配置中的标签数量
        self.classifier = tf.keras.layers.Dense(
            units=config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            use_bias=True,
            bias_initializer="zeros",
            name="classifier",
        )
        # 保存配置
        self.config = config

    # 装饰器，解包输入并添加前向传播的起始文档字符串
    @unpack_inputs
    @add_start_docstrings_to_model_forward(TFCVT_INPUTS_DOCSTRING)
    # 替换返回值的文档字符串
    @replace_return_docstrings(output_type=TFImageClassifierOutputWithNoAttention, config_class=_CONFIG_FOR_DOC)
    # 前向传播函数
    def call(
        self,
        pixel_values: tf.Tensor | None = None,
        labels: tf.Tensor | None = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[TFImageClassifierOutputWithNoAttention, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, TFCvtForImageClassification
        >>> import tensorflow as tf
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("microsoft/cvt-13")
        >>> model = TFCvtForImageClassification.from_pretrained("microsoft/cvt-13")

        >>> inputs = image_processor(images=image, return_tensors="tf")
        >>> outputs = model(**inputs)
        >>> logits = outputs.logits
        >>> # model predicts one of the 1000 ImageNet classes
        >>> predicted_class_idx = tf.math.argmax(logits, axis=-1)[0]
        >>> print("Predicted class:", model.config.id2label[int(predicted_class_idx)])
        ```"""

        # 调用 CVT 模型进行图像处理，返回模型输出
        outputs = self.cvt(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 从模型输出中提取序列输出和类别 token
        sequence_output = outputs[0]
        cls_token = outputs[1]
        # 如果配置中包含类别 token，则使用类别 token 更新序列输出
        if self.config.cls_token[-1]:
            sequence_output = self.layernorm(cls_token)
        else:
            # 重新排列形状为 "batch_size, num_channels, height, width" 的序列输出
            batch_size, num_channels, height, width = shape_list(sequence_output)
            sequence_output = tf.reshape(sequence_output, shape=(batch_size, num_channels, height * width))
            sequence_output = tf.transpose(sequence_output, perm=(0, 2, 1))
            sequence_output = self.layernorm(sequence_output)

        # 计算序列输出的平均值
        sequence_output_mean = tf.reduce_mean(sequence_output, axis=1)
        # 使用分类器模型对平均序列输出进行分类
        logits = self.classifier(sequence_output_mean)
        # 如果提供了标签，则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        # 如果不返回字典，则组织输出并返回
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TFImageClassifierOutputWithNoAttention 类型的输出
        return TFImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)
    # 定义一个方法用于构建模型，指定输入形状
    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回，不重复构建
        if self.built:
            return
        # 标记模型为已构建状态
        self.built = True
        # 检查是否存在要构建的"self.cvt"属性
        if getattr(self, "cvt", None) is not None:
            # 使用 TensorFlow 的命名空间，构建 self.cvt 属性指定的子层
            with tf.name_scope(self.cvt.name):
                self.cvt.build(None)
        # 检查是否存在要构建的"self.layernorm"属性
        if getattr(self, "layernorm", None) is not None:
            # 使用 TensorFlow 的命名空间，构建 self.layernorm 属性指定的子层
            with tf.name_scope(self.layernorm.name):
                # 指定 layernorm 层的构建参数，设置其输入形状为 [None, None, self.config.embed_dim[-1]]
                self.layernorm.build([None, None, self.config.embed_dim[-1]])
        # 检查是否存在要构建的"self.classifier"属性
        if getattr(self, "classifier", None) is not None:
            # 检查 classifier 属性是否有 "name" 属性
            if hasattr(self.classifier, "name"):
                # 使用 TensorFlow 的命名空间，构建 self.classifier 属性指定的子层
                with tf.name_scope(self.classifier.name):
                    # 指定 classifier 层的构建参数，设置其输入形状为 [None, None, self.config.embed_dim[-1]]
                    self.classifier.build([None, None, self.config.embed_dim[-1]])
```