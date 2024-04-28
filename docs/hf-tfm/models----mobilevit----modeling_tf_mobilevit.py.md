# `.\transformers\models\mobilevit\modeling_tf_mobilevit.py`

```
# 设置编码为 UTF-8
# 版权声明
# 版权所有 2022 苹果公司和 HuggingFace 公司
#
# 根据 Apache 许可证 2.0 版 ("许可证") 许可；
# 除非符合许可证的约定，否则您不得使用此文件。
# 您可以在以下位置获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按"现状"分发软件，
# 不附带任何明示或暗示的担保或条件。
# 请参阅许可证以了解特定语言管理权限和限制
#
# 原始许可证: https://github.com/apple/ml-cvnets/blob/main/LICENSE
""" TensorFlow 2.0 MobileViT 模型 """

# 引入模块
from __future__ import annotations
from typing import Dict, Optional, Tuple, Union
import tensorflow as tf
# 从其它模块引入方法
from ...activations_tf import get_tf_activation
from ...file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFBaseModelOutputWithPooling,
    TFImageClassifierOutputWithNoAttention,
    TFSemanticSegmenterOutputWithNoAttention,
)
from ...modeling_tf_utils import TFPreTrainedModel, TFSequenceClassificationLoss, keras_serializable, unpack_inputs
from ...tf_utils import shape_list, stable_softmax
from ...utils import logging
from .configuration_mobilevit import MobileViTConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的通用字符串
_CONFIG_FOR_DOC = "MobileViTConfig"
# 基础文档字符串
_CHECKPOINT_FOR_DOC = "apple/mobilevit-small"
_EXPECTED_OUTPUT_SHAPE = [1, 640, 8, 8]

# 图像分类文档字符串
_IMAGE_CLASS_CHECKPOINT = "apple/mobilevit-small"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

# MobileViT 预训练模型列表
TF_MOBILEVIT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "apple/mobilevit-small",
    "apple/mobilevit-x-small",
    "apple/mobilevit-xx-small",
    "apple/deeplabv3-mobilevit-small",
    "apple/deeplabv3-mobilevit-x-small",
    "apple/deeplabv3-mobilevit-xx-small",
    # 查看所有 MobileViT 模型 https://huggingface.co/models?filter=mobilevit
]

def make_divisible(value: int, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """
    确保所有层的通道数可被 `divisor` 整除。此函数源自原始的 TensorFlow 仓库。
    可在此处找到：https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # 确保向下取整不会减少超过 10%
    if new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)

class TFMobileViTConvLayer(tf.keras.layers.Layer):
    # 请完成剩下的部分
    ```
    # 初始化函数，接受模型配置、输入通道数、输出通道数、卷积核大小、步长、分组数量、是否使用偏置、扩张率、是否使用归一化、是否使用激活函数等参数
    def __init__(
        self,
        config: MobileViTConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias: bool = False,
        dilation: int = 1,
        use_normalization: bool = True,
        use_activation: Union[bool, str] = True,
        **kwargs,
    ) -> None:
        # 调用父类初始化函数
        super().__init__(**kwargs)
        # 打印警告信息，提醒此模型不支持在 CPU 上进行反向传播操作，需使用 GPU 或 TPU
        logger.warning(
            f"\n{self.__class__.__name__} has backpropagation operations that are NOT supported on CPU. If you wish "
            "to train/fine-tune this model, you need a GPU or a TPU"
        )

        # 计算填充大小
        padding = int((kernel_size - 1) / 2) * dilation
        # 使用 ZeroPadding2D 层进行填充
        self.padding = tf.keras.layers.ZeroPadding2D(padding)

        # 检查输出通道是否能被分组数整除，若不能则引发 ValueError
        if out_channels % groups != 0:
            raise ValueError(f"Output channels ({out_channels}) are not divisible by {groups} groups.")

        # 创建卷积层
        self.convolution = tf.keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=kernel_size,
            strides=stride,
            padding="VALID",
            dilation_rate=dilation,
            groups=groups,
            use_bias=bias,
            name="convolution",
        )

        # 根据是否使用归一化进行处理
        if use_normalization:
            self.normalization = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.1, name="normalization")
        else:
            self.normalization = None

        # 根据是否使用激活函数进行处理
        if use_activation:
            if isinstance(use_activation, str):
                self.activation = get_tf_activation(use_activation)
            elif isinstance(config.hidden_act, str):
                self.activation = get_tf_activation(config.hidden_act)
            else:
                self.activation = config.hidden_act
        else:
            self.activation = None
        self.in_channels = in_channels
        self.out_channels = out_channels

    # 定义调用函数，接受特征张量和训练标识符，返回处理后的特征张量
    def call(self, features: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 对输入特征进行填充
        padded_features = self.padding(features)
        # 通过卷积层处理填充后的特征
        features = self.convolution(padded_features)
        # 若存在归一化层，则对特征进行归一化
        if self.normalization is not None:
            features = self.normalization(features, training=training)
        # 若存在激活函数，则对特征进行激活
        if self.activation is not None:
            features = self.activation(features)
        return features

    # 定义构建函数，根据输入形状构建模型
    def build(self, input_shape=None):
        # 若已构建过模型，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在卷积层，则根据输入通道数构建卷积层
        if getattr(self, "convolution", None) is not None:
            with tf.name_scope(self.convolution.name):
                self.convolution.build([None, None, None, self.in_channels])
        # 如果存在归一化层，则根据输出通道数构建归一化层
        if getattr(self, "normalization", None) is not None:
            if hasattr(self.normalization, "name"):
                with tf.name_scope(self.normalization.name):
                    self.normalization.build([None, None, None, self.out_channels])
```  
# 定义一个名为 TFMobileViTInvertedResidual 的类，继承自 tf.keras.layers.Layer
class TFMobileViTInvertedResidual(tf.keras.layers.Layer):
    """
    Inverted residual block (MobileNetv2): https://arxiv.org/abs/1801.04381
    """

    # 初始化函数，接受 MobileViTConfig 对象、输入通道数、输出通道数、步长、扩张率和其他参数
    def __init__(
        self, config: MobileViTConfig, in_channels: int, out_channels: int, stride: int, dilation: int = 1, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        # 计算扩张后的通道数
        expanded_channels = make_divisible(int(round(in_channels * config.expand_ratio)), 8)

        # 如果步长不是1或2，则引发值错误
        if stride not in [1, 2]:
            raise ValueError(f"Invalid stride {stride}.")

        # 根据条件确定是否使用残差连接
        self.use_residual = (stride == 1) and (in_channels == out_channels)

        # 创建 1x1 扩张卷积层
        self.expand_1x1 = TFMobileViTConvLayer(
            config, in_channels=in_channels, out_channels=expanded_channels, kernel_size=1, name="expand_1x1"
        )

        # 创建 3x3 卷积层
        self.conv_3x3 = TFMobileViTConvLayer(
            config,
            in_channels=expanded_channels,
            out_channels=expanded_channels,
            kernel_size=3,
            stride=stride,
            groups=expanded_channels,
            dilation=dilation,
            name="conv_3x3",
        )

        # 创建 1x1 缩减卷积层
        self.reduce_1x1 = TFMobileViTConvLayer(
            config,
            in_channels=expanded_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_activation=False,
            name="reduce_1x1",
        )

    # 前向传播函数，接受特征和训练标志，返回特征张量
    def call(self, features: tf.Tensor, training: bool = False) -> tf.Tensor:
        residual = features  # 保存输入特征

        # 经过 1x1 扩张卷积层
        features = self.expand_1x1(features, training=training)
        # 经过 3x3 卷积层
        features = self.conv_3x3(features, training=training)
        # 经过 1x1 缩减卷积层
        features = self.reduce_1x1(features, training=training)

        # 如果需要残差连接，则返回输入特征与处理后的特征相加，否则返回处理后的特征
        return residual + features if self.use_residual else features

    # 构建函数，用于构建卷积层
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "expand_1x1", None) is not None:  # 如果存在 1x1 扩张卷积层
            with tf.name_scope(self.expand_1x1.name):  # 设置命名空间
                self.expand_1x1.build(None)  # 构建卷积层
        if getattr(self, "conv_3x3", None) is not None:  # 如果存在 3x3 卷积层
            with tf.name_scope(self.conv_3x3.name):  # 设置命名空间
                self.conv_3x3.build(None)  # 构建卷积层
        if getattr(self, "reduce_1x1", None) is not None:  # 如果存在 1x1 缩减卷积层
            with tf.name_scope(self.reduce_1x1.name):  # 设置命名空间
                self.reduce_1x1.build(None)  # 构建卷积层


# 定义一个名为 TFMobileViTMobileNetLayer 的类，继承自 tf.keras.layers.Layer
class TFMobileViTMobileNetLayer(tf.keras.layers.Layer):
    # 初始化函数，接受 MobileViTConfig 对象、输入通道数、输出通道数、步长以及其他参数
    def __init__(
        self,
        config: MobileViTConfig,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        num_stages: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.layers = []  # 用于存储 TFMobileViTInvertedResidual 对象的列表
        for i in range(num_stages):  # 根据 num_stages 创建 TFMobileViTInvertedResidual 对象
            layer = TFMobileViTInvertedResidual(
                config,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride if i == 0 else 1,
                name=f"layer.{i}",
            )
            self.layers.append(layer)  # 将 TFMobileViTInvertedResidual 对象添加到列表中
            in_channels = out_channels  # 更新输入通道数
    # 定义一个方法用于调用神经网络模型，传入特征和是否训练的标志，返回处理后的特征
    def call(self, features: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 遍历每一层神经网络模块，依次对特征进行处理
        for layer_module in self.layers:
            features = layer_module(features, training=training)
        # 返回处理后的特征
        return features

    # 构建神经网络模型
    def build(self, input_shape=None):
        # 如果神经网络模型已经构建过，则直接返回
        if self.built:
            return
        # 标记神经网络模型已经构建
        self.built = True
        # 如果存在layers属性（神经网络的每一层），则对每一层进行构建
        if getattr(self, "layers", None) is not None:
            for layer_module in self.layers:
                # 使用带有名称作用域的方式构建每一层
                with tf.name_scope(layer_module.name):
                    layer_module.build(None)
# 定义 TFMobileViTSelfAttention 类，继承自 tf.keras.layers.Layer
class TFMobileViTSelfAttention(tf.keras.layers.Layer):
    # 初始化方法，接受 MobileViTConfig 和 hidden_size 两个参数
    def __init__(self, config: MobileViTConfig, hidden_size: int, **kwargs) -> None:
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 检查 hidden_size 是否是 config.num_attention_heads 的整数倍
        if hidden_size % config.num_attention_heads != 0:
            # 如果不是，则抛出异常
            raise ValueError(
                f"The hidden size {hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        # 将 config.num_attention_heads 赋值给 self.num_attention_heads
        self.num_attention_heads = config.num_attention_heads
        # 计算每个 attention head 的大小
        self.attention_head_size = int(hidden_size / config.num_attention_heads)
        # 计算所有 attention head 的总大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # 计算 scale 值，用于点积注意力机制
        scale = tf.cast(self.attention_head_size, dtype=tf.float32)
        self.scale = tf.math.sqrt(scale)

        # 创建三个全连接层，分别用于计算查询、键和值
        self.query = tf.keras.layers.Dense(self.all_head_size, use_bias=config.qkv_bias, name="query")
        self.key = tf.keras.layers.Dense(self.all_head_size, use_bias=config.qkv_bias, name="key")
        self.value = tf.keras.layers.Dense(self.all_head_size, use_bias=config.qkv_bias, name="value")

        # 创建一个 Dropout 层，用于 attention 概率的 dropout
        self.dropout = tf.keras.layers.Dropout(config.attention_probs_dropout_prob)
        # 保存 hidden_size 值
        self.hidden_size = hidden_size

    # 定义一个方法用于调整张量的形状
    def transpose_for_scores(self, x: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(x)[0]
        # 将张量的形状调整为 (batch_size, -1, self.num_attention_heads, self.attention_head_size)
        x = tf.reshape(x, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))
        # 将张量的维度顺序调整为 (0, 2, 1, 3)
        return tf.transpose(x, perm=[0, 2, 1, 3])

    # 定义前向传播方法
    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        batch_size = tf.shape(hidden_states)[0]

        # 计算 key、value 和 query
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        # 计算注意力分数
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        attention_scores = attention_scores / self.scale

        # 将注意力分数归一化为概率
        attention_probs = stable_softmax(attention_scores, axis=-1)

        # 将 attention 概率 dropout
        attention_probs = self.dropout(attention_probs, training=training)

        # 计算上下文向量
        context_layer = tf.matmul(attention_probs, value_layer)

        # 调整上下文向量的形状
        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        context_layer = tf.reshape(context_layer, shape=(batch_size, -1, self.all_head_size))

        # 返回上下文向量
        return context_layer
    # 构建自定义层的方法
    def build(self, input_shape=None):
        # 如果已经构建过，则返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在查询（query）属性，则构建查询的神经网络层
        if getattr(self, "query", None) is not None:
            # 使用查询的名称创建命名空间
            with tf.name_scope(self.query.name):
                # 构建查询神经网络层，输入形状为[None, None, self.hidden_size]
                self.query.build([None, None, self.hidden_size])
        # 如果存在键（key）属性，则构建键的神经网络层
        if getattr(self, "key", None) is not None:
            # 使用键的名称创建命名空间
            with tf.name_scope(self.key.name):
                # 构建键的神经网络层，输入形状为[None, None, self.hidden_size]
                self.key.build([None, None, self.hidden_size])
        # 如果存在数值（value）属性，则构建数值的神经网络层
        if getattr(self, "value", None) is not None:
            # 使用数值的名称创建命名空间
            with tf.name_scope(self.value.name):
                # 构建数值的神经网络层，输入形状为[None, None, self.hidden_size]
                self.value.build([None, None, self.hidden_size])
# 定义一个名为TFMobileViTSelfOutput的自定义层，继承自tf.keras.layers.Layer类
class TFMobileViTSelfOutput(tf.keras.layers.Layer):
    # 构造函数，初始化自定义层的属性
    def __init__(self, config: MobileViTConfig, hidden_size: int, **kwargs) -> None:
        # 调用父类的构造函数
        super().__init__(**kwargs)
        # 创建一个全连接层，输出维度为hidden_size，命名为"dense"
        self.dense = tf.keras.layers.Dense(hidden_size, name="dense")
        # 创建一个dropout层，使用config中的隐藏单元丢弃概率
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        # 将hidden_size属性设为传入的hidden_size值
        self.hidden_size = hidden_size

    # 定义call方法，定义自定义层的前向传播逻辑
    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 使用全连接层处理输入张量
        hidden_states = self.dense(hidden_states)
        # 使用dropout层处理全连接层输出的张量，传入training参数控制是否训练模式
        hidden_states = self.dropout(hidden_states, training=training)
        # 返回处理后的张量
        return hidden_states

    # 定义build方法，用于构建自定义层的参数
    def build(self, input_shape=None):
        # 如果自定义层已经构建完成，则直接返回
        if self.built:
            return
        # 将自定义层标记为已构建状态
        self.built = True
        # 如果存在dense属性（全连接层），则对其进行构建，指定输入形状为[None, None, self.hidden_size]
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.hidden_size])


# 定义一个名为TFMobileViTAttention的自定义层，继承自tf.keras.layers.Layer类
class TFMobileViTAttention(tf.keras.layers.Layer):
    # 构造函数，初始化自定义层的属性
    def __init__(self, config: MobileViTConfig, hidden_size: int, **kwargs) -> None:
        # 调用父类的构造函数
        super().__init__(**kwargs)
        # 创建一个TFMobileViTSelfAttention层，命名为"attention"
        self.attention = TFMobileViTSelfAttention(config, hidden_size, name="attention")
        # 创建一个TFMobileViTSelfOutput层，命名为"output"
        self.dense_output = TFMobileViTSelfOutput(config, hidden_size, name="output")

    # 定义用于剪枝heads的方法
    def prune_heads(self, heads):
        # 抛出未实现的方法异常
        raise NotImplementedError

    # 定义call方法，定义自定义层的前向传播逻辑
    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 使用TFMobileViTSelfAttention层处理输入张量，传入training参数控制是否训练模式
        self_outputs = self.attention(hidden_states, training=training)
        # 使用TFMobileViTSelfOutput层处理TFMobileViTSelfAttention层的输出张量，传入training参数控制是否训练模式
        attention_output = self.dense_output(self_outputs, training=training)
        # 返回处理后的张量
        return attention_output

    # 定义build方法，用于构建自定义层的参数
    def build(self, input_shape=None):
        # 如果自定义层已经构建完成，则直接返回
        if self.built:
            return
        # 将自定义层标记为已构建状态
        self.built = True
        # 如果存在attention属性（TFMobileViTSelfAttention层），则对其进行构建
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        # 如果存在dense_output属性（TFMobileViTSelfOutput层），则对其进行构建
        if getattr(self, "dense_output", None) is not None:
            with tf.name_scope(self.dense_output.name):
                self.dense_output.build(None)


# 定义一个名为TFMobileViTIntermediate的自定义层，继承自tf.keras.layers.Layer类
class TFMobileViTIntermediate(tf.keras.layers.Layer):
    # 构造函数，初始化自定义层的属性
    def __init__(self, config: MobileViTConfig, hidden_size: int, intermediate_size: int, **kwargs) -> None:
        # 调用父类的构造函数
        super().__init__(**kwargs)
        # 创建一个全连接层，输出维度为intermediate_size，命名为"dense"
        self.dense = tf.keras.layers.Dense(intermediate_size, name="dense")
        # 如果config.hidden_act是字符串类型，则将intermediate_act_fn设为对应的激活函数
        # 否则直接使用config.hidden_act作为激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        # 将hidden_size属性设为传入的hidden_size值
        self.hidden_size = hidden_size

    # 定义call方法，定义自定义层的前向传播逻辑
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 使用全连接层处理输入张量
        hidden_states = self.dense(hidden_states)
        # 使用中间激活函数处理全连接层输出的张量
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回处理后的张量
        return hidden_states

    # 定义build方法，用于构建自定义层的参数
    def build(self, input_shape=None):
        # 如果自定义层已经构建完成，则直接返回
        if self.built:
            return
        # 将自定义层标记为已构建状态
        self.built = True
        # 如果存在dense属性（全连接层），则对其进行构建，指定输入形状为[None, None, self.hidden_size]
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.hidden_size])
class TFMobileViTOutput(tf.keras.layers.Layer):
    def __init__(self, config: MobileViTConfig, hidden_size: int, intermediate_size: int, **kwargs) -> None:
        super().__init__(**kwargs)
        # 创建一个全连接层，用于变换隐藏状态的维度
        self.dense = tf.keras.layers.Dense(hidden_size, name="dense")
        # 创建一个dropout层，用于随机丢弃一部分神经元，以防止过拟合
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        # 中间状态的大小
        self.intermediate_size = intermediate_size

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 将隐藏状态通过全连接层进行维度变换
        hidden_states = self.dense(hidden_states)
        # 对变换后的隐藏状态进行dropout处理
        hidden_states = self.dropout(hidden_states, training=training)
        # 将dropout后的隐藏状态与输入的张量进行相加
        hidden_states = hidden_states + input_tensor
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                # 构建全连接层
                self.dense.build([None, None, self.intermediate_size])


class TFMobileViTTransformerLayer(tf.keras.layers.Layer):
    def __init__(self, config: MobileViTConfig, hidden_size: int, intermediate_size: int, **kwargs) -> None:
        super().__init__(**kwargs)
        # 创建注意力层
        self.attention = TFMobileViTAttention(config, hidden_size, name="attention")
        # 创建中间层
        self.intermediate = TFMobileViTIntermediate(config, hidden_size, intermediate_size, name="intermediate")
        # 创建输出层
        self.mobilevit_output = TFMobileViTOutput(config, hidden_size, intermediate_size, name="output")
        # 创建层归一化层，用于对输入数据进行归一化处理
        self.layernorm_before = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="layernorm_before"
        )
        # 创建层归一化层，用于对输出数据进行归一化处理
        self.layernorm_after = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="layernorm_after"
        )
        # 隐藏状态的大小
        self.hidden_size = hidden_size

    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 使用注意力层对输入的隐藏状态进行处理
        attention_output = self.attention(self.layernorm_before(hidden_states), training=training)
        # 将注意力输出与输入的隐藏状态相加
        hidden_states = attention_output + hidden_states

        # 对相加后的隐藏状态进行层归一化处理
        layer_output = self.layernorm_after(hidden_states)
        # 中间层的处理
        layer_output = self.intermediate(layer_output)
        # 输出层的处理
        layer_output = self.mobilevit_output(layer_output, hidden_states, training=training)
        return layer_output
    # 构建该层网络，如果已经构建过了则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 设置标记为已构建
        self.built = True
        # 如果存在注意力层，则构建注意力层
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        # 如果存在中间层，则构建中间层
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        # 如果存在输出层，则构建输出层
        if getattr(self, "mobilevit_output", None) is not None:
            with tf.name_scope(self.mobilevit_output.name):
                self.mobilevit_output.build(None)
        # 如果存在层归一化前操作，则构建层归一化前操作
        if getattr(self, "layernorm_before", None) is not None:
            with tf.name_scope(self.layernorm_before.name):
                self.layernorm_before.build([None, None, self.hidden_size])
        # 如果存在层归一化后操作，则构建层归一化后操作
        if getattr(self, "layernorm_after", None) is not None:
            with tf.name_scope(self.layernorm_after.name):
                self.layernorm_after.build([None, None, self.hidden_size])
class TFMobileViTTransformer(tf.keras.layers.Layer):
    def __init__(self, config: MobileViTConfig, hidden_size: int, num_stages: int, **kwargs) -> None:
        # 初始化函数，传入配置、隐藏层大小、阶段数等参数
        super().__init__(**kwargs)

        # 初始化空列表用于存储Transformer层
        self.layers = []
        # 根据阶段数循环创建TFMobileViTTransformerLayer对象并加入layers列表
        for i in range(num_stages):
            transformer_layer = TFMobileViTTransformerLayer(
                config,
                hidden_size=hidden_size,
                intermediate_size=int(hidden_size * config.mlp_ratio),
                name=f"layer.{i}",
            )
            self.layers.append(transformer_layer)

    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 对每个Transformer层进行调用，并传递隐藏状态和训练标志
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states, training=training)
        return hidden_states

    def build(self, input_shape=None):
        # 如果已经构建了模型，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在layers，循环build每个Transformer层
        if getattr(self, "layers", None) is not None:
            for layer_module in self.layers:
                with tf.name_scope(layer_module.name):
                    layer_module.build(None)


class TFMobileViTLayer(tf.keras.layers.Layer):
    """
    MobileViT block: https://arxiv.org/abs/2110.02178
    """

    def __init__(
        self,
        config: MobileViTConfig,
        in_channels: int,
        out_channels: int,
        stride: int,
        hidden_size: int,
        num_stages: int,
        dilation: int = 1,
        **kwargs,
    # 定义一个函数，参数为None，表示没有返回值
    def __init__(self, config, in_channels, out_channels, stride, dilation, hidden_size, num_stages, **kwargs):
        # 调用父类的构造函数，并传入kwargs参数
        super().__init__(**kwargs)
        # 设置patch宽度为config中的patch_size
        self.patch_width = config.patch_size
        # 设置patch高度为config中的patch_size
        self.patch_height = config.patch_size

        # 如果步幅为2，创建下采样层
        if stride == 2:
            self.downsampling_layer = TFMobileViTInvertedResidual(
                config,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride if dilation == 1 else 1,
                dilation=dilation // 2 if dilation > 1 else 1,
                name="downsampling_layer",
            )
            in_channels = out_channels
        # 如果步幅不为2，下采样层为空
        else:
            self.downsampling_layer = None

        # 创建kxk卷积层
        self.conv_kxk = TFMobileViTConvLayer(
            config,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=config.conv_kernel_size,
            name="conv_kxk",
        )

        # 创建1x1卷积层
        self.conv_1x1 = TFMobileViTConvLayer(
            config,
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=1,
            use_normalization=False,
            use_activation=False,
            name="conv_1x1",
        )

        # 创建transformer层
        self.transformer = TFMobileViTTransformer(
            config, hidden_size=hidden_size, num_stages=num_stages, name="transformer"
        )

        # 创建layernorm层
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm")

        # 创建投影卷积层
        self.conv_projection = TFMobileViTConvLayer(
            config, in_channels=hidden_size, out_channels=in_channels, kernel_size=1, name="conv_projection"
        )

        # 创建融合卷积层
        self.fusion = TFMobileViTConvLayer(
            config,
            in_channels=2 * in_channels,
            out_channels=in_channels,
            kernel_size=config.conv_kernel_size,
            name="fusion",
        )
        # 设置隐藏层的大小为hidden_size
        self.hidden_size = hidden_size
    # 定义一个方法，用于将输入的特征张量展开成补丁（patch）形式，并返回展开后的特征张量以及相关信息字典
    def unfolding(self, features: tf.Tensor) -> Tuple[tf.Tensor, Dict]:
        # 获取补丁的宽度和高度
        patch_width, patch_height = self.patch_width, self.patch_height
        # 计算补丁的面积
        patch_area = tf.cast(patch_width * patch_height, "int32")

        # 获取特征张量的批次大小、原始高度、原始宽度和通道数
        batch_size = tf.shape(features)[0]
        orig_height = tf.shape(features)[1]
        orig_width = tf.shape(features)[2]
        channels = tf.shape(features)[3]

        # 计算展开后的新高度和宽度
        new_height = tf.cast(tf.math.ceil(orig_height / patch_height) * patch_height, "int32")
        new_width = tf.cast(tf.math.ceil(orig_width / patch_width) * patch_width, "int32")

        # 判断是否需要插值
        interpolate = new_width != orig_width or new_height != orig_height
        if interpolate:
            # 注意：可以进行填充，但需要在注意力函数中处理
            features = tf.image.resize(features, size=(new_height, new_width), method="bilinear")

        # 计算沿着宽度和高度的补丁数量
        num_patch_width = new_width // patch_width
        num_patch_height = new_height // patch_height
        num_patches = num_patch_height * num_patch_width

        # 将形状从 (batch_size, orig_height, orig_width, channels) 转换为形状 (batch_size * patch_area, num_patches, channels)
        features = tf.transpose(features, [0, 3, 1, 2])
        patches = tf.reshape(
            features, (batch_size * channels * num_patch_height, patch_height, num_patch_width, patch_width)
        )
        patches = tf.transpose(patches, [0, 2, 1, 3])
        patches = tf.reshape(patches, (batch_size, channels, num_patches, patch_area))
        patches = tf.transpose(patches, [0, 3, 2, 1])
        patches = tf.reshape(patches, (batch_size * patch_area, num_patches, channels))

        # 构建信息字典，包含原始尺寸、批次大小、通道数、是否插值、补丁数量以及沿宽度和高度的补丁数量
        info_dict = {
            "orig_size": (orig_height, orig_width),
            "batch_size": batch_size,
            "channels": channels,
            "interpolate": interpolate,
            "num_patches": num_patches,
            "num_patches_width": num_patch_width,
            "num_patches_height": num_patch_height,
        }
        # 返回展开后的补丁张量和信息字典
        return patches, info_dict
    def folding(self, patches: tf.Tensor, info_dict: Dict) -> tf.Tensor:
        patch_width, patch_height = self.patch_width, self.patch_height  # 从self对象中获取patch_width和patch_height
        patch_area = int(patch_width * patch_height)  # 计算patch的面积

        batch_size = info_dict["batch_size"]  # 从info_dict中获取batch_size
        channels = info_dict["channels"]  # 从info_dict中获取channels
        num_patches = info_dict["num_patches"]  # 从info_dict中获取num_patches
        num_patch_height = info_dict["num_patches_height"]  # 从info_dict中获取num_patch_height
        num_patch_width = info_dict["num_patches_width"]  # 从info_dict中获取num_patch_width

        # convert from shape (batch_size * patch_area, num_patches, channels)
        # back to shape (batch_size, channels, orig_height, orig_width)
        features = tf.reshape(patches, (batch_size, patch_area, num_patches, -1))  # 重塑patches的形状
        features = tf.transpose(features, perm=(0, 3, 2, 1))  # 转置features的维度
        features = tf.reshape(
            features, (batch_size * channels * num_patch_height, num_patch_width, patch_height, patch_width)
        )  # 重塑features的形状
        features = tf.transpose(features, perm=(0, 2, 1, 3))  # 再次转置features的维度
        features = tf.reshape(
            features, (batch_size, channels, num_patch_height * patch_height, num_patch_width * patch_width)
        )  # 最后重塑features的形状
        features = tf.transpose(features, perm=(0, 2, 3, 1))  # 最后一次转置features的维度

        if info_dict["interpolate"]:  # 如果interpolate存在于info_dict中
            features = tf.image.resize(features, size=info_dict["orig_size"], method="bilinear")  # 使用双线性插值对features进行调整大小

        return features  # 返回features

    def call(self, features: tf.Tensor, training: bool = False) -> tf.Tensor:
        # reduce spatial dimensions if needed
        if self.downsampling_layer:  # 如果存在downsampling_layer
            features = self.downsampling_layer(features, training=training)  # 使用downsampling_layer对features进行降维处理

        residual = features  # 将features赋给residual作为备份

        # local representation
        features = self.conv_kxk(features, training=training)  # 使用conv_kxk对features进行卷积处理
        features = self.conv_1x1(features, training=training)  # 使用conv_1x1对features进行卷积处理

        # convert feature map to patches
        patches, info_dict = self.unfolding(features)  # 使用unfolding将feature map转换成patches，并获取info_dict

        # learn global representations
        patches = self.transformer(patches, training=training)  # 使用transformer学习全局表示
        patches = self.layernorm(patches)  # 使用layernorm对patches进行归一化处理

        # convert patches back to feature maps
        features = self.folding(patches, info_dict)  # 使用folding将patches转换回feature maps

        features = self.conv_projection(features, training=training)  # 使用conv_projection对features进行卷积处理
        features = self.fusion(tf.concat([residual, features], axis=-1), training=training)  # 使用fusion对features进行融合处理
        return features  # 返回features
    # 构建当前层
    def build(self, input_shape=None):
        # 如果已经构建过了，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        
        # 如果有 conv_kxk 层，则构建它
        if getattr(self, "conv_kxk", None) is not None:
            with tf.name_scope(self.conv_kxk.name):
                self.conv_kxk.build(None)
        
        # 如果有 conv_1x1 层，则构建它
        if getattr(self, "conv_1x1", None) is not None:
            with tf.name_scope(self.conv_1x1.name):
                self.conv_1x1.build(None)
        
        # 如果有 transformer 层，则构建它
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
        
        # 如果有 layernorm 层，则构建它，输入形状为 [None, None, self.hidden_size]
        if getattr(self, "layernorm", None) is not None:
            with tf.name_scope(self.layernorm.name):
                self.layernorm.build([None, None, self.hidden_size])
        
        # 如果有 conv_projection 层，则构建它
        if getattr(self, "conv_projection", None) is not None:
            with tf.name_scope(self.conv_projection.name):
                self.conv_projection.build(None)
        
        # 如果有 fusion 层，则构建它
        if getattr(self, "fusion", None) is not None:
            with tf.name_scope(self.fusion.name):
                self.fusion.build(None)
        
        # 如果有 downsampling_layer 层，则构建它
        if getattr(self, "downsampling_layer", None) is not None:
            with tf.name_scope(self.downsampling_layer.name):
                self.downsampling_layer.build(None)
# 定义了一个 TFMobileViTEncoder 类,继承自 tf.keras.layers.Layer
class TFMobileViTEncoder(tf.keras.layers.Layer):
    def __init__(self, config: MobileViTConfig, **kwargs) -> None:
        # 调用父类的 __init__ 方法,传递 **kwargs 参数
        super().__init__(**kwargs)
        # 保存 MobileViTConfig 对象到 self.config 属性
        self.config = config
        # 创建一个空列表,用于存储各个层
        self.layers = []

        # 根据 config.output_stride 的值,决定是否需要对第4层和第5层进行膨胀卷积
        dilate_layer_4 = dilate_layer_5 = False
        if config.output_stride == 8:
            dilate_layer_4 = True
            dilate_layer_5 = True
        elif config.output_stride == 16:
            dilate_layer_5 = True

        # 初始化膨胀系数为1
        dilation = 1

        # 创建第1层 TFMobileViTMobileNetLayer 并添加到 self.layers 列表中
        layer_1 = TFMobileViTMobileNetLayer(
            config,
            in_channels=config.neck_hidden_sizes[0],
            out_channels=config.neck_hidden_sizes[1],
            stride=1,
            num_stages=1,
            name="layer.0",
        )
        self.layers.append(layer_1)

        # 创建第2层 TFMobileViTMobileNetLayer 并添加到 self.layers 列表中
        layer_2 = TFMobileViTMobileNetLayer(
            config,
            in_channels=config.neck_hidden_sizes[1],
            out_channels=config.neck_hidden_sizes[2],
            stride=2,
            num_stages=3,
            name="layer.1",
        )
        self.layers.append(layer_2)

        # 创建第3层 TFMobileViTLayer 并添加到 self.layers 列表中
        layer_3 = TFMobileViTLayer(
            config,
            in_channels=config.neck_hidden_sizes[2],
            out_channels=config.neck_hidden_sizes[3],
            stride=2,
            hidden_size=config.hidden_sizes[0],
            num_stages=2,
            name="layer.2",
        )
        self.layers.append(layer_3)

        # 如果需要对第4层进行膨胀卷积,则将膨胀系数乘以2
        if dilate_layer_4:
            dilation *= 2

        # 创建第4层 TFMobileViTLayer 并添加到 self.layers 列表中
        layer_4 = TFMobileViTLayer(
            config,
            in_channels=config.neck_hidden_sizes[3],
            out_channels=config.neck_hidden_sizes[4],
            stride=2,
            hidden_size=config.hidden_sizes[1],
            num_stages=4,
            dilation=dilation,
            name="layer.3",
        )
        self.layers.append(layer_4)

        # 如果需要对第5层进行膨胀卷积,则将膨胀系数再次乘以2
        if dilate_layer_5:
            dilation *= 2

        # 创建第5层 TFMobileViTLayer 并添加到 self.layers 列表中
        layer_5 = TFMobileViTLayer(
            config,
            in_channels=config.neck_hidden_sizes[4],
            out_channels=config.neck_hidden_sizes[5],
            stride=2,
            hidden_size=config.hidden_sizes[2],
            num_stages=3,
            dilation=dilation,
            name="layer.4",
        )
        self.layers.append(layer_5)

    def call(
        self,
        hidden_states: tf.Tensor,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        training: bool = False,
    ):
        pass
        ) -> Union[tuple, TFBaseModelOutput]:
        # 定义函数的返回类型注释，表示返回值为元组或TFBaseModelOutput类型

        all_hidden_states = () if output_hidden_states else None
        # 设置变量all_hidden_states为空元组或None，取决于output_hidden_states的值

        for i, layer_module in enumerate(self.layers):
            # 遍历self.layers中的元素，使用enumerate获取索引和元素值
            hidden_states = layer_module(hidden_states, training=training)
            # 调用layer_module方法，传入hidden_states和training参数，获取hidden_states

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                # 如果output_hidden_states为True，则将hidden_states追加到all_hidden_states中

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)
            # 如果return_dict为False，则返回包含hidden_states和all_hidden_states中非None值的元组

        return TFBaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states)
        # 返回TFBaseModelOutput对象，包含last_hidden_state和hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        # 如果已经构建过模型，则直接返回

        self.built = True
        # 标记模型已经构建

        if getattr(self, "layers", None) is not None:
            # 检查self.layers是否存在
            for layer_module in self.layers:
                # 遍历self.layers中的元素
                with tf.name_scope(layer_module.name):
                    # 使用tf.name_scope给每个layer_module设置命名空间
                    layer_module.build(None)
                    # 调用layer_module的build方法，传入None作为输入形状
# 将类标记为可序列化的，用于保存/加载模型
@keras_serializable
class TFMobileViTMainLayer(tf.keras.layers.Layer):
    # 配置类为MobileViTConfig
    config_class = MobileViTConfig

    # 初始化函数，接受MobileViTConfig类型的config参数，以及可选的expand_output参数
    def __init__(self, config: MobileViTConfig, expand_output: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.expand_output = expand_output

        # 创建一个卷积层，用于处理输入数据
        self.conv_stem = TFMobileViTConvLayer(
            config,
            in_channels=config.num_channels,
            out_channels=config.neck_hidden_sizes[0],
            kernel_size=3,
            stride=2,
            name="conv_stem",
        )

        # 创建一个Mobile ViT编码器层
        self.encoder = TFMobileViTEncoder(config, name="encoder")

        # 如果expand_output为真，则创建一个用于扩展输出维度的卷积层
        if self.expand_output:
            self.conv_1x1_exp = TFMobileViTConvLayer(
                config,
                in_channels=config.neck_hidden_sizes[5],
                out_channels=config.neck_hidden_sizes[6],
                kernel_size=1,
                name="conv_1x1_exp",
            )

        # 创建一个全局平均池化层，用于池化特征图
        self.pooler = tf.keras.layers.GlobalAveragePooling2D(data_format="channels_first", name="pooler")

    # 用于剪枝模型头的私有方法
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    # 调用方法，接受一系列参数，并返回模型输出
    @unpack_inputs
    def call(
        self,
        pixel_values: tf.Tensor | None = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    # 定义一个函数，接收像素值作为输入，返回包含池化层输出的元组或 TFBaseModelOutputWithPooling 对象
    ) -> Union[Tuple[tf.Tensor], TFBaseModelOutputWithPooling]:
        # 如果 output_hidden_states 为空，则使用配置中的 output_hidden_states
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果 return_dict 为空，则使用配置中的 use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # 当在 CPU 上运行时，`tf.keras.layers.Conv2D` 不支持 `NCHW` 格式
        # 所以需要将输入格式从 `NCHW` 转换为 `NHWC`
        # 形状为 (batch_size, in_height, in_width, in_channels=num_channels)
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))
    
        # 利用卷积层处理像素值，获取嵌入输出
        embedding_output = self.conv_stem(pixel_values, training=training)
    
        # 将嵌入输出传给编码器，获取编码器输出
        encoder_outputs = self.encoder(
            embedding_output, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training
        )
    
        # 如果要扩展输出
        if self.expand_output:
            # 使用 1x1 卷积层处理编码器输出，获取最终隐藏状态
            last_hidden_state = self.conv_1x1_exp(encoder_outputs[0])
    
            # 将输出格式转换为NCHW，以保持模块的一致性
            last_hidden_state = tf.transpose(last_hidden_state, perm=[0, 3, 1, 2])
    
            # 全局平均池化：(batch_size, channels, height, width) -> (batch_size, channels)
            pooled_output = self.pooler(last_hidden_state)
        else:
            # 否则直接使用编码器输出作为最终隐藏状态
            last_hidden_state = encoder_outputs[0]
            # 将输出格式转换为NCHW，以保持模块的一致性
            last_hidden_state = tf.transpose(last_hidden_state, perm=[0, 3, 1, 2])
            pooled_output = None
    
        # 如果不使用 return_dict
        if not return_dict:
            # 返回最终隐藏状态和池化输出（如果有）
            output = (last_hidden_state, pooled_output) if pooled_output is not None else (last_hidden_state,)
    
            # 转换输出格式为NCHW，以保持模块的一致性
            if not self.expand_output:
                remaining_encoder_outputs = encoder_outputs[1:]
                remaining_encoder_outputs = tuple(
                    [tf.transpose(h, perm=(0, 3, 1, 2)) for h in remaining_encoder_outputs[0]]
                )
                remaining_encoder_outputs = (remaining_encoder_outputs,)
                return output + remaining_encoder_outputs
            else:
                return output + encoder_outputs[1:]
    
        # 如果输出隐藏状态
        if output_hidden_states:
            hidden_states = tuple([tf.transpose(h, perm=(0, 3, 1, 2)) for h in encoder_outputs[1]])
    
        # 返回 TFBaseModelOutputWithPooling 对象，包括最终隐藏状态、池化输出和隐藏状态列表（如果有隐藏状态输出）
        return TFBaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=hidden_states if output_hidden_states else encoder_outputs.hidden_states,
        )
    # 构建模型，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 设置标记表示已经构建过
        self.built = True
        # 构建卷积层，如果存在卷积层
        if getattr(self, "conv_stem", None) is not None:
            with tf.name_scope(self.conv_stem.name):
                self.conv_stem.build(None)
        # 构建编码器，如果存在编码器
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 构建池化层，如果存在池化层
        if getattr(self, "pooler", None) is not None:
            with tf.name_scope(self.pooler.name):
                self.pooler.build([None, None, None, None])
        # 构建 1x1 卷积层，如果存在 1x1 卷积层
        if getattr(self, "conv_1x1_exp", None) is not None:
            with tf.name_scope(self.conv_1x1_exp.name):
                self.conv_1x1_exp.build(None)
class TFMobileViTPreTrainedModel(TFPreTrainedModel):
    """
    一个抽象类，用于处理权重初始化、简单接口用于下载和加载预训练模型。
    """

    # MobileViT 模型的配置类
    config_class = MobileViTConfig
    # MobileViT 模型的前缀
    base_model_prefix = "mobilevit"
    # 主输入的名称
    main_input_name = "pixel_values"


MOBILEVIT_START_DOCSTRING = r"""
    这个模型继承自 [`TFPreTrainedModel`]。查看父类文档，了解库为所有模型实现的通用方法（如下载或保存、调整输入嵌入、剪枝头等）。

    这个模型也是 [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) 的子类。将其用作常规的 TF 2.0 Keras 模型，并参考 TF 2.0 文档，了解与一般用法和行为相关的一切。

    <提示>

    `transformers` 中的 TensorFlow 模型和层接受两种格式的输入：

    - 将所有输入作为关键字参数（类似于 PyTorch 模型），或
    - 将所有输入作为列表、元组或字典的第一个位置参数。

    支持第二种格式的原因是 Keras 方法在向模型和层传递输入时更偏好这种格式。由于这种支持，在使用像 `model.fit()` 这样的方法时，应该 "只需工作" - 只需传递您的输入和标签，任何 `model.fit()` 支持的格式都可以！如果您想要在 Keras 方法之外使用第二种格式，比如在使用 Keras `Functional` API 创建自己的层或模型时，您可以使用三种可能性来收集所有输入张量到第一个位置参数中：

    - 仅包含 `pixel_values`，没有其他内容的单个张量：`model(pixel_values)`
    - 在指定的顺序中有一个或多个输入张量的可变长度列表：`model([pixel_values, attention_mask])` 或 `model([pixel_values, attention_mask, token_type_ids])`
    - 一个字典，带有与文档字符串中给定的输入名称相关联的一个或多个输入张量：`model({"pixel_values": pixel_values, "token_type_ids": token_type_ids})`

    请注意，当使用
    [子类化](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) 创建模型和层时，您无需担心任何这些，因为您可以像将输入传递给任何其他 Python 函数一样直接传递！

    </提示>

    参数:
        config ([`MobileViTConfig`]): 带有模型所有参数的模型配置类。
            使用配置文件进行初始化不会加载与模型相关的权重，仅加载配置。查看 [`~TFPreTrainedModel.from_pretrained`] 方法加载模型权重。
"""

MOBILEVIT_INPUTS_DOCSTRING = r"""
    # 定义函数参数说明
    Args:
        # 像素值，可以是 NumPy 数组、TensorFlow Tensor、Tensor 列表或字典类型
        # 每个样本的形状应为 (batch_size, num_channels, height, width)
        pixel_values (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]`, `Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`MobileViTImageProcessor.__call__`] for details.
    
        # 是否返回所有层的隐藏状态，详情见返回值中的 hidden_states
        # 此参数只能在急切执行模式下使用，在图形模式下使用配置文件中的值
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
            used instead.
        # 是否返回 ModelOutput 对象而不是普通元组
        # 此参数可在急切模式下使用，在图形模式下总是设置为 True
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in
            eager mode, in graph mode the value will always be set to True.
# 添加模型文档字符串和MobileViT模型类
@add_start_docstrings(
    "The bare MobileViT model outputting raw hidden-states without any specific head on top.",
    MOBILEVIT_START_DOCSTRING,
)
class TFMobileViTModel(TFMobileViTPreTrainedModel):
    # 初始化方法
    def __init__(self, config: MobileViTConfig, expand_output: bool = True, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config
        self.expand_output = expand_output

        # 创建TFMobileViTMainLayer实例
        self.mobilevit = TFMobileViTMainLayer(config, expand_output=expand_output, name="mobilevit")

    # call方法
    @unpack_inputs
    @add_start_docstrings_to_model_forward(MOBILEVIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    # call方法
    def call(
        self,
        pixel_values: tf.Tensor | None = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[Tuple[tf.Tensor], TFBaseModelOutputWithPooling]:
        # 调用self.mobilevit
        output = self.mobilevit(pixel_values, output_hidden_states, return_dict, training=training)
        return output

    # build方法
    def build(self, input_shape=None):
        # 如果已经构建完成，直接返回
        if self.built:
            return
        self.built = True
        # 构建self.mobilevit
        if getattr(self, "mobilevit", None) is not None:
            with tf.name_scope(self.mobilevit.name):
                self.mobilevit.build(None)

# 添加模型分类任务文档字符串和TFMobileViTForImageClassification类
@add_start_docstrings(
    """
    MobileViT model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    MOBILEVIT_START_DOCSTRING,
)
class TFMobileViTForImageClassification(TFMobileViTPreTrainedModel, TFSequenceClassificationLoss):
    # 初始化方法
    def __init__(self, config: MobileViTConfig, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels
        # 创建TFMobileViTMainLayer实例
        self.mobilevit = TFMobileViTMainLayer(config, name="mobilevit")

        # 分类器头部
        self.dropout = tf.keras.layers.Dropout(config.classifier_dropout_prob)
        self.classifier = (
            tf.keras.layers.Dense(config.num_labels, name="classifier") if config.num_labels > 0 else tf.identity
        )
        self.config = config

    # call方���
    @unpack_inputs
    @add_start_docstrings_to_model_forward(MOBILEVIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=TFImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    # call方法
    def call(
        self,
        pixel_values: tf.Tensor | None = None,
        output_hidden_states: Optional[bool] = None,
        labels: tf.Tensor | None = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[tuple, TFImageClassifierOutputWithNoAttention]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss). If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 设置返回字典是否为空，默认为模型配置中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 MobileViT 模型进行推理
        outputs = self.mobilevit(
            pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training
        )

        # 如果需要返回字典，则获取池化输出；否则，获取第二个输出
        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        # 通过分类器获取分类的逻辑概率
        logits = self.classifier(self.dropout(pooled_output, training=training))
        
        # 如果没有标签，损失为 None；否则，计算损失
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        # 如果不需要返回字典，则构建输出元组
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TFImageClassifierOutputWithNoAttention 类型的输出
        return TFImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)

    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 标记模型已构建
        self.built = True
        # 如果存在 MobileViT 模型，则构建之
        if getattr(self, "mobilevit", None) is not None:
            with tf.name_scope(self.mobilevit.name):
                self.mobilevit.build(None)
        # 如果存在分类器，则构建之
        if getattr(self, "classifier", None) is not None:
            if hasattr(self.classifier, "name"):
                with tf.name_scope(self.classifier.name):
                    # 构建分类器，输入形状为 [None, None, config.neck_hidden_sizes[-1]]
                    self.classifier.build([None, None, self.config.neck_hidden_sizes[-1]])
```  
# 定义一个自定义的 Keras 层 TFMobileViTASPPPooling
class TFMobileViTASPPPooling(tf.keras.layers.Layer):

    # 初始化方法，接受配置参数、输入通道数和输出通道数
    def __init__(self, config: MobileViTConfig, in_channels: int, out_channels: int, **kwargs) -> None:
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 创建一个全局平均池化层，保持维度不变，命名为 global_pool
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D(keepdims=True, name="global_pool")

        # 创建一个 TFMobileViTConvLayer 对象，用于进行 1x1 卷积操作
        self.conv_1x1 = TFMobileViTConvLayer(
            config,  # MobileViT 的配置参数
            in_channels=in_channels,  # 输入通道数
            out_channels=out_channels,  # 输出通道数
            kernel_size=1,  # 卷积核大小为 1
            stride=1,  # 步长为 1
            use_normalization=True,  # 使用归一化
            use_activation="relu",  # 使用 ReLU 激活函数
            name="conv_1x1",  # 层的名称为 conv_1x1
        )

    # 前向传播方法，接受特征和训练标志作为输入，返回处理后的特征
    def call(self, features: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 计算特征的空间大小
        spatial_size = shape_list(features)[1:-1]
        # 对特征进行全局平均池化
        features = self.global_pool(features)
        # 对池化后的特征进行 1x1 卷积
        features = self.conv_1x1(features, training=training)
        # 将池化和卷积后的特征进行双线性插值，恢复原始空间大小
        features = tf.image.resize(features, size=spatial_size, method="bilinear")
        # 返回处理后的特征
        return features

    # 构建方法，用于构建层的权重
    def build(self, input_shape=None):
        # 如果层已经构建过，则直接返回
        if self.built:
            return
        # 设置层为已构建状态
        self.built = True
        # 如果存在全局平均池化层，构建该层的权重
        if getattr(self, "global_pool", None) is not None:
            with tf.name_scope(self.global_pool.name):
                self.global_pool.build([None, None, None, None])
        # 如果存在 1x1 卷积层，构建该层的权重
        if getattr(self, "conv_1x1", None) is not None:
            with tf.name_scope(self.conv_1x1.name):
                self.conv_1x1.build(None)


# 定义一个自定义的 Keras 层 TFMobileViTASPP
class TFMobileViTASPP(tf.keras.layers.Layer):
    """
    ASPP module defined in DeepLab papers: https://arxiv.org/abs/1606.00915, https://arxiv.org/abs/1706.05587
    """
    # 初始化函数，接受一个 MobileViTConfig 类型的配置对象和其他关键字参数
    def __init__(self, config: MobileViTConfig, **kwargs) -> None:
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 获取输入通道数和输出通道数
        in_channels = config.neck_hidden_sizes[-2]
        out_channels = config.aspp_out_channels

        # 检查是否有3个空洞卷积的扩张率
        if len(config.atrous_rates) != 3:
            raise ValueError("Expected 3 values for atrous_rates")

        # 初始化一个空的卷积层列表
        self.convs = []

        # 使用配置对象创建第一个卷积层
        in_projection = TFMobileViTConvLayer(
            config,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_activation="relu",
            name="convs.0",
        )
        self.convs.append(in_projection)

        # 循环创建空洞卷积层并添加到卷积层列表中
        self.convs.extend(
            [
                TFMobileViTConvLayer(
                    config,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    dilation=rate,
                    use_activation="relu",
                    name=f"convs.{i + 1}",
                )
                for i, rate in enumerate(config.atrous_rates)
            ]
        )

        # 创建 ASPP 池化层
        pool_layer = TFMobileViTASPPPooling(
            config, in_channels, out_channels, name=f"convs.{len(config.atrous_rates) + 1}"
        )
        self.convs.append(pool_layer)

        # 创建投影卷积层
        self.project = TFMobileViTConvLayer(
            config,
            in_channels=5 * out_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_activation="relu",
            name="project",
        )

        # 创建丢弃层
        self.dropout = tf.keras.layers.Dropout(config.aspp_dropout_prob)

    # 前向传播函数，接受特征张量和训练标志，返回处理后的特征张量
    def call(self, features: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 将隐藏状态转置为 `(batch_size, channels, height, width)` 格式，再转置为 `(batch_size, height, width, channels)` 格式
        features = tf.transpose(features, perm=[0, 2, 3, 1])
        pyramid = []
        # 遍历卷积层列表，并对特征进行处理
        for conv in self.convs:
            pyramid.append(conv(features, training=training))
        # 沿着最后一个维度进行拼接
        pyramid = tf.concat(pyramid, axis=-1)

        # 对拼接后的特征进行投影
        pooled_features = self.project(pyramid, training=training)
        # 对投影后的特征进行丢弃
        pooled_features = self.dropout(pooled_features, training=training)
        return pooled_features

    # 构建函数，根据输入形状建立层
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 设置为已构建状态
        self.built = True
        # 如果存在投影层，则建立投影层
        if getattr(self, "project", None) is not None:
            with tf.name_scope(self.project.name):
                self.project.build(None)
        # 如果存在卷积层列表，则逐一建立每个卷积层
        if getattr(self, "convs", None) is not None:
            for conv in self.convs:
                with tf.name_scope(conv.name):
                    conv.build(None)
class TFMobileViTDeepLabV3(tf.keras.layers.Layer):
    """
    DeepLabv3架构：https://arxiv.org/abs/1706.05587
    """

    def __init__(self, config: MobileViTConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.aspp = TFMobileViTASPP(config, name="aspp")  # 创建ASPP模块对象

        self.dropout = tf.keras.layers.Dropout(config.classifier_dropout_prob)  # 创建Dropout层对象

        self.classifier = TFMobileViTConvLayer(
            config,
            in_channels=config.aspp_out_channels,
            out_channels=config.num_labels,
            kernel_size=1,
            use_normalization=False,
            use_activation=False,
            bias=True,
            name="classifier",
        )  # 创建分类器对象

    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        features = self.aspp(hidden_states[-1], training=training)  # 使用ASPP模块提取特征
        features = self.dropout(features, training=training)  # 对提取的特征进行Dropout处理
        features = self.classifier(features, training=training)  # 使用分类器进行分类
        return features

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "aspp", None) is not None:
            with tf.name_scope(self.aspp.name):
                self.aspp.build(None)  # 构建ASPP模块
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build(None)  # 构建分类器


@add_start_docstrings(
    """
    MobileViT模型的语义分割头，例如用于Pascal VOC。
    """,
    MOBILEVIT_START_DOCSTRING,
)
class TFMobileViTForSemanticSegmentation(TFMobileViTPreTrainedModel):
    def __init__(self, config: MobileViTConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)

        self.num_labels = config.num_labels  # 获取类别数
        self.mobilevit = TFMobileViTMainLayer(config, expand_output=False, name="mobilevit")  # 创建MobileViT主模块对象
        self.segmentation_head = TFMobileViTDeepLabV3(config, name="segmentation_head")  # 创建DeepLabv3分割头对象
    def hf_compute_loss(self, logits, labels):
        # 计算损失函数，根据模型输出的logits和真实标签labels
        # 将logits插值至原始图像的大小
        # `labels` 的形状为 (batch_size, height, width)
        label_interp_shape = shape_list(labels)[1:]

        upsampled_logits = tf.image.resize(logits, size=label_interp_shape, method="bilinear")
        # 使用稀疏分类交叉熵损失函数
        loss_fct = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

        def masked_loss(real, pred):
            # 未蒙版的损失
            unmasked_loss = loss_fct(real, pred)
            # 创建掩码，过滤掉忽略的索引值
            mask = tf.cast(real != self.config.semantic_loss_ignore_index, dtype=unmasked_loss.dtype)
            # 带掩码的损失
            masked_loss = unmasked_loss * mask
            # 根据掩码求减少过的损失，并进行归一化处理
            reduced_masked_loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(mask)
            return tf.reshape(reduced_masked_loss, (1,))

        return masked_loss(labels, upsampled_logits)

    @unpack_inputs
    @add_start_docstrings_to_model_forward(MOBILEVIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSemanticSegmenterOutputWithNoAttention, config_class=_CONFIG_FOR_DOC)
    # 模型调用函数，接收输入 pixel_values 和 labels，可控制是否输出隐藏状态，返回字典形式输出
    def call(
        self,
        pixel_values: tf.Tensor | None = None,
        labels: tf.Tensor | None = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    def forward(
        self,
        pixel_values,
        labels=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
    ) -> Union[tuple, TFSemanticSegmenterOutputWithNoAttention]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).
            是否提供标签用于计算损失函数，标签形状为`（batch_size，高度，宽度）`，若`config.num_labels > 1`，则计算分类损失（交叉熵）。
    
        Returns:
            返回值：
    
        Examples:
            示例：
            ```python
            >>> from transformers import AutoImageProcessor, TFMobileViTForSemanticSegmentation
            >>> from PIL import Image
            >>> import requests
    
            >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            >>> image = Image.open(requests.get(url, stream=True).raw)
    
            >>> image_processor = AutoImageProcessor.from_pretrained("apple/deeplabv3-mobilevit-small")
            >>> model = TFMobileViTForSemanticSegmentation.from_pretrained("apple/deeplabv3-mobilevit-small")
    
            >>> inputs = image_processor(images=image, return_tensors="tf")
    
            >>> outputs = model(**inputs)
    
            >>> # logits are of shape (batch_size, num_labels, height, width)
            >>> logits = outputs.logits
            ```"""
    
        # 确定是否使用默认参数
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # 通过MobileViT进行前向传播，获取中间隐藏状态
        outputs = self.mobilevit(
            pixel_values,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
            training=training,
        )
    
        # 获取MobileViT的隐藏状态
        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]
    
        # 利用分割头对隐藏状态进行语义分割，得到预测结果logits
        logits = self.segmentation_head(encoder_hidden_states, training=training)
    
        loss = None
    
        # 利用标签计算损失
        if labels is not None:
            if not self.config.num_labels > 1:
                raise ValueError("The number of labels should be greater than one")
            else:
                loss = self.hf_compute_loss(logits=logits, labels=labels)
    
        # 为了保持接口的一致性，将logits的形状变成(batch_size, num_labels, height, width)
        logits = tf.transpose(logits, perm=[0, 3, 1, 2])
    
        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
    
        return TFSemanticSegmenterOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
        )
    # 构建函数，用于构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回，避免重复构建
        if self.built:
            return
        # 设置模型已经构建的标志为 True
        self.built = True
        # 检查是否存在 mobilevit 属性，并且不为 None
        if getattr(self, "mobilevit", None) is not None:
            # 在 TensorFlow 中创建一个命名空间，命名空间的名称为 self.mobilevit.name
            with tf.name_scope(self.mobilevit.name):
                # 构建 mobilevit 属性指向的模型，输入形状为 None（表示不指定）
                self.mobilevit.build(None)
        # 检查是否存在 segmentation_head 属性，并且不为 None
        if getattr(self, "segmentation_head", None) is not None:
            # 在 TensorFlow 中创建一个命名空间，命名空间的名称为 self.segmentation_head.name
            with tf.name_scope(self.segmentation_head.name):
                # 构建 segmentation_head 属性指向的模型，输入形状为 None（表示不指定）
                self.segmentation_head.build(None)
```