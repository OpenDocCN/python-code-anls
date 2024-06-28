# `.\models\mobilevit\modeling_tf_mobilevit.py`

```py
# coding=utf-8
# 版权 2022 年 Apple Inc. 和 HuggingFace Inc. 团队保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）授权；
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件根据“原样”分发，
# 不附带任何明示或暗示的担保或条件。
# 有关详细信息，请参阅许可证。
#
# 原始许可证：https://github.com/apple/ml-cvnets/blob/main/LICENSE
""" TensorFlow 2.0 MobileViT 模型。"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import tensorflow as tf

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
from ...modeling_tf_utils import (
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import shape_list, stable_softmax
from ...utils import logging
from .configuration_mobilevit import MobileViTConfig

logger = logging.get_logger(__name__)

# 一般文档字符串
_CONFIG_FOR_DOC = "MobileViTConfig"

# 基础文档字符串
_CHECKPOINT_FOR_DOC = "apple/mobilevit-small"
_EXPECTED_OUTPUT_SHAPE = [1, 640, 8, 8]

# 图像分类文档字符串
_IMAGE_CLASS_CHECKPOINT = "apple/mobilevit-small"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

# TF MobileViT 预训练模型存档列表
TF_MOBILEVIT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "apple/mobilevit-small",
    "apple/mobilevit-x-small",
    "apple/mobilevit-xx-small",
    "apple/deeplabv3-mobilevit-small",
    "apple/deeplabv3-mobilevit-x-small",
    "apple/deeplabv3-mobilevit-xx-small",
    # 请访问 https://huggingface.co/models?filter=mobilevit 查看所有 MobileViT 模型
]


def make_divisible(value: int, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """
    确保所有层的通道数量可被 `divisor` 整除。此函数源自原始 TensorFlow 仓库，可在以下链接找到：
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # 确保向下舍入不会减少超过 10%。
    if new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


class TFMobileViTConvLayer(keras.layers.Layer):
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
        super().__init__(**kwargs)
        logger.warning(
            f"\n{self.__class__.__name__} has backpropagation operations that are NOT supported on CPU. If you wish "
            "to train/fine-tune this model, you need a GPU or a TPU"
        )

        # 计算要应用的填充量，以使卷积操作保持输入输出大小相同
        padding = int((kernel_size - 1) / 2) * dilation
        self.padding = keras.layers.ZeroPadding2D(padding)

        if out_channels % groups != 0:
            # 如果输出通道数不能被组数整除，抛出错误
            raise ValueError(f"Output channels ({out_channels}) are not divisible by {groups} groups.")

        # 创建卷积层对象
        self.convolution = keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=kernel_size,
            strides=stride,
            padding="VALID",
            dilation_rate=dilation,
            groups=groups,
            use_bias=bias,
            name="convolution",
        )

        if use_normalization:
            # 如果需要使用标准化层，则创建批量标准化对象
            self.normalization = keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.1, name="normalization")
        else:
            self.normalization = None

        if use_activation:
            if isinstance(use_activation, str):
                # 如果指定了激活函数名称，则根据名称获取激活函数对象
                self.activation = get_tf_activation(use_activation)
            elif isinstance(config.hidden_act, str):
                # 否则，根据配置文件中的隐藏层激活函数名称获取激活函数对象
                self.activation = get_tf_activation(config.hidden_act)
            else:
                # 否则，使用配置文件中的隐藏层激活函数
                self.activation = config.hidden_act
        else:
            self.activation = None
        self.in_channels = in_channels
        self.out_channels = out_channels

    def call(self, features: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 对输入特征进行填充
        padded_features = self.padding(features)
        # 应用卷积操作
        features = self.convolution(padded_features)
        if self.normalization is not None:
            # 如果存在标准化层，则应用标准化
            features = self.normalization(features, training=training)
        if self.activation is not None:
            # 如果存在激活函数，则应用激活函数
            features = self.activation(features)
        return features

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "convolution", None) is not None:
            with tf.name_scope(self.convolution.name):
                # 构建卷积层
                self.convolution.build([None, None, None, self.in_channels])
        if getattr(self, "normalization", None) is not None:
            if hasattr(self.normalization, "name"):
                with tf.name_scope(self.normalization.name):
                    # 构建标准化层
                    self.normalization.build([None, None, None, self.out_channels])
# 定义一个自定义层 TFMobileViTInvertedResidual，用于实现 MobileNetv2 中的反向残差块
class TFMobileViTInvertedResidual(keras.layers.Layer):
    """
    Inverted residual block (MobileNetv2): https://arxiv.org/abs/1801.04381
    """

    # 初始化方法，接收配置 config，输入通道数 in_channels，输出通道数 out_channels，步长 stride，扩张率 dilation 等参数
    def __init__(
        self, config: MobileViTConfig, in_channels: int, out_channels: int, stride: int, dilation: int = 1, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        
        # 根据配置计算扩展后的通道数，使其能被 8 整除
        expanded_channels = make_divisible(int(round(in_channels * config.expand_ratio)), 8)

        # 检查步长是否合法，只能是 1 或 2
        if stride not in [1, 2]:
            raise ValueError(f"Invalid stride {stride}.")

        # 判断是否使用残差连接，条件为步长为 1 且输入通道数等于输出通道数
        self.use_residual = (stride == 1) and (in_channels == out_channels)

        # 创建 1x1 卷积扩展层，输入通道数为 in_channels，输出通道数为 expanded_channels
        self.expand_1x1 = TFMobileViTConvLayer(
            config, in_channels=in_channels, out_channels=expanded_channels, kernel_size=1, name="expand_1x1"
        )

        # 创建 3x3 卷积层，输入通道数为 expanded_channels，输出通道数为 expanded_channels
        # 使用组卷积（groups=expanded_channels）和指定的步长和空洞卷积率（dilation）
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

        # 创建 1x1 卷积减少层，输入通道数为 expanded_channels，输出通道数为 out_channels
        # 不使用激活函数
        self.reduce_1x1 = TFMobileViTConvLayer(
            config,
            in_channels=expanded_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_activation=False,
            name="reduce_1x1",
        )

    # 前向传播方法，接收特征张量 features 和训练标志 training，返回处理后的特征张量
    def call(self, features: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 保存残差连接
        residual = features

        # 经过 1x1 卷积扩展层
        features = self.expand_1x1(features, training=training)
        # 经过 3x3 卷积层
        features = self.conv_3x3(features, training=training)
        # 经过 1x1 卷积减少层
        features = self.reduce_1x1(features, training=training)

        # 如果使用残差连接，则将原始特征张量和处理后的特征张量相加
        return residual + features if self.use_residual else features

    # 构建方法，用于构建层，检查是否已经构建过
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在扩展层，则构建扩展层
        if getattr(self, "expand_1x1", None) is not None:
            with tf.name_scope(self.expand_1x1.name):
                self.expand_1x1.build(None)
        # 如果存在 3x3 卷积层，则构建 3x3 卷积层
        if getattr(self, "conv_3x3", None) is not None:
            with tf.name_scope(self.conv_3x3.name):
                self.conv_3x3.build(None)
        # 如果存在减少层，则构建减少层
        if getattr(self, "reduce_1x1", None) is not None:
            with tf.name_scope(self.reduce_1x1.name):
                self.reduce_1x1.build(None)


# 定义 MobileNet 层，包含多个 TFMobileViTInvertedResidual 反向残差块
class TFMobileViTMobileNetLayer(keras.layers.Layer):
    # 初始化方法，接收配置 config，输入通道数 in_channels，输出通道数 out_channels，步长 stride 等参数
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

        self.layers = []
        # 根据 num_stages 创建多个 TFMobileViTInvertedResidual 层
        for i in range(num_stages):
            layer = TFMobileViTInvertedResidual(
                config,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride if i == 0 else 1,  # 第一个阶段使用给定的 stride，其余阶段使用步长 1
                name=f"layer.{i}",
            )
            self.layers.append(layer)
            in_channels = out_channels  # 更新下一层的输入通道数为当前层的输出通道数
    # 对神经网络模型进行调用，传入特征张量，并根据训练模式决定是否进行训练
    def call(self, features: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 遍历神经网络的每一层模块
        for layer_module in self.layers:
            # 调用每一层模块，将特征张量作为输入，根据训练模式进行处理
            features = layer_module(features, training=training)
        # 返回处理后的特征张量
        return features

    # 构建神经网络模型
    def build(self, input_shape=None):
        # 如果模型已经构建完成，则直接返回
        if self.built:
            return
        # 将模型标记为已构建
        self.built = True
        # 如果模型具有"layers"属性
        if getattr(self, "layers", None) is not None:
            # 遍历神经网络的每一层模块
            for layer_module in self.layers:
                # 使用 TensorFlow 的名称作用域，为每一层模块设置名称空间
                with tf.name_scope(layer_module.name):
                    # 调用每一层模块的build方法，传入input_shape参数为None
                    layer_module.build(None)
# 定义 TFMobileViTSelfAttention 类，继承自 keras.layers.Layer
class TFMobileViTSelfAttention(keras.layers.Layer):
    # 初始化函数，接受 MobileViTConfig 对象和隐藏层大小参数
    def __init__(self, config: MobileViTConfig, hidden_size: int, **kwargs) -> None:
        super().__init__(**kwargs)

        # 检查隐藏层大小是否能被注意力头数整除
        if hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        # 设置注意力头数和每个注意力头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # 计算缩放因子，用于注意力分数的缩放
        scale = tf.cast(self.attention_head_size, dtype=tf.float32)
        self.scale = tf.math.sqrt(scale)

        # 定义用于查询、键和值的全连接层
        self.query = keras.layers.Dense(self.all_head_size, use_bias=config.qkv_bias, name="query")
        self.key = keras.layers.Dense(self.all_head_size, use_bias=config.qkv_bias, name="key")
        self.value = keras.layers.Dense(self.all_head_size, use_bias=config.qkv_bias, name="value")

        # 定义用于 dropout 的层，以及隐藏层的大小
        self.dropout = keras.layers.Dropout(config.attention_probs_dropout_prob)
        self.hidden_size = hidden_size

    # 将输入张量 x 转置以便计算注意力分数
    def transpose_for_scores(self, x: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    # 调用函数，计算自注意力机制的输出
    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        batch_size = tf.shape(hidden_states)[0]

        # 计算查询、键和值的张量并转置以便计算注意力分数
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        # 计算注意力分数
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        attention_scores = attention_scores / self.scale

        # 对注意力分数进行归一化处理，得到注意力概率
        attention_probs = stable_softmax(attention_scores, axis=-1)

        # 使用 dropout 层进行随机失活
        attention_probs = self.dropout(attention_probs, training=training)

        # 计算上下文张量，即注意力加权的值张量
        context_layer = tf.matmul(attention_probs, value_layer)

        # 将上下文张量转置和重塑以便输出
        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        context_layer = tf.reshape(context_layer, shape=(batch_size, -1, self.all_head_size))
        
        # 返回最终的上下文张量作为输出
        return context_layer
    # 构建方法，用于构建神经网络层
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回，避免重复构建
        if self.built:
            return
        # 设置标记，表明已经完成构建
        self.built = True
        # 如果有查询（query）属性，则构建查询张量，并命名作用域为查询张量的名称
        if getattr(self, "query", None) is not None:
            with tf.name_scope(self.query.name):
                # 使用隐藏大小构建查询张量，输入形状为 [None, None, self.hidden_size]
                self.query.build([None, None, self.hidden_size])
        # 如果有键（key）属性，则构建键张量，并命名作用域为键张量的名称
        if getattr(self, "key", None) is not None:
            with tf.name_scope(self.key.name):
                # 使用隐藏大小构建键张量，输入形状为 [None, None, self.hidden_size]
                self.key.build([None, None, self.hidden_size])
        # 如果有值（value）属性，则构建值张量，并命名作用域为值张量的名称
        if getattr(self, "value", None) is not None:
            with tf.name_scope(self.value.name):
                # 使用隐藏大小构建值张量，输入形状为 [None, None, self.hidden_size]
                self.value.build([None, None, self.hidden_size])
class TFMobileViTSelfOutput(keras.layers.Layer):
    def __init__(self, config: MobileViTConfig, hidden_size: int, **kwargs) -> None:
        super().__init__(**kwargs)
        # 定义一个全连接层，用于变换隐藏状态到指定大小的输出
        self.dense = keras.layers.Dense(hidden_size, name="dense")
        # 定义一个 Dropout 层，用于在训练时随机丢弃部分输出，防止过拟合
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)
        self.hidden_size = hidden_size

    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 将输入的隐藏状态通过全连接层进行线性变换
        hidden_states = self.dense(hidden_states)
        # 在训练时使用 Dropout 层处理输出，以防止过拟合
        hidden_states = self.dropout(hidden_states, training=training)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果 dense 层已经定义，则使用 tf.name_scope 构建 dense 层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.hidden_size])


class TFMobileViTAttention(keras.layers.Layer):
    def __init__(self, config: MobileViTConfig, hidden_size: int, **kwargs) -> None:
        super().__init__(**kwargs)
        # 定义注意力层，用于处理输入的隐藏状态
        self.attention = TFMobileViTSelfAttention(config, hidden_size, name="attention")
        # 定义输出层，用于处理注意力层的输出
        self.dense_output = TFMobileViTSelfOutput(config, hidden_size, name="output")

    def prune_heads(self, heads):
        # 暂未实现的方法，用于裁剪注意力头部
        raise NotImplementedError

    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 使用注意力层处理隐藏状态并获取自身输出
        self_outputs = self.attention(hidden_states, training=training)
        # 使用输出层处理注意力层的自身输出
        attention_output = self.dense_output(self_outputs, training=training)
        return attention_output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果 attention 层已经定义，则使用 tf.name_scope 构建 attention 层
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        # 如果 dense_output 层已经定义，则使用 tf.name_scope 构建 dense_output 层
        if getattr(self, "dense_output", None) is not None:
            with tf.name_scope(self.dense_output.name):
                self.dense_output.build(None)


class TFMobileViTIntermediate(keras.layers.Layer):
    def __init__(self, config: MobileViTConfig, hidden_size: int, intermediate_size: int, **kwargs) -> None:
        super().__init__(**kwargs)
        # 定义一个全连接层，用于将隐藏状态映射到中间层的大小
        self.dense = keras.layers.Dense(intermediate_size, name="dense")
        # 根据配置获取激活函数，用于处理中间层的输出
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        self.hidden_size = hidden_size

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 将隐藏状态通过全连接层进行线性变换
        hidden_states = self.dense(hidden_states)
        # 使用中间激活函数处理全连接层的输出
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果 dense 层已经定义，则使用 tf.name_scope 构建 dense 层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.hidden_size])


class TFMobileViTOutput(keras.layers.Layer):
    # 待继续完善
    # 初始化方法，用于设置类的初始状态
    def __init__(self, config: MobileViTConfig, hidden_size: int, intermediate_size: int, **kwargs) -> None:
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 创建一个全连接层，用于处理输入数据
        self.dense = keras.layers.Dense(hidden_size, name="dense")
        # 创建一个 Dropout 层，用于在训练过程中随机断开输入神经元，防止过拟合
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)
        # 设置中间层的大小
        self.intermediate_size = intermediate_size

    # 调用方法，用于定义模型的前向传播逻辑
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 将输入张量通过全连接层进行线性变换
        hidden_states = self.dense(hidden_states)
        # 在训练过程中，通过 Dropout 层进行随机失活
        hidden_states = self.dropout(hidden_states, training=training)
        # 将全连接层的输出与输入张量相加，实现残差连接
        hidden_states = hidden_states + input_tensor
        # 返回处理后的张量作为输出
        return hidden_states

    # 构建方法，用于在第一次调用 call 方法时构建层的权重
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记当前层已经构建
        self.built = True
        # 检查是否存在 dense 层，并在命名作用域下构建它的权重
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.intermediate_size])
# 定义一个自定义的 Keras 层，实现 MobileViT 的变压器层
class TFMobileViTTransformerLayer(keras.layers.Layer):
    def __init__(self, config: MobileViTConfig, hidden_size: int, intermediate_size: int, **kwargs) -> None:
        super().__init__(**kwargs)
        # 初始化注意力机制层，使用 MobileViTAttention 类
        self.attention = TFMobileViTAttention(config, hidden_size, name="attention")
        # 初始化变换层，使用 TFMobileViTIntermediate 类
        self.intermediate = TFMobileViTIntermediate(config, hidden_size, intermediate_size, name="intermediate")
        # 初始化输出层，使用 TFMobileViTOutput 类
        self.mobilevit_output = TFMobileViTOutput(config, hidden_size, intermediate_size, name="output")
        # 初始化层归一化层（之前），epsilon 参数从配置中获取
        self.layernorm_before = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm_before")
        # 初始化层归一化层（之后），epsilon 参数从配置中获取
        self.layernorm_after = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm_after")
        # 记录隐藏层大小
        self.hidden_size = hidden_size

    # 定义 call 方法，实现层的前向传播
    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 执行注意力机制，传入归一化之后的隐藏状态
        attention_output = self.attention(self.layernorm_before(hidden_states), training=training)
        # 加上残差连接，更新隐藏状态
        hidden_states = attention_output + hidden_states

        # 对更新后的隐藏状态执行层归一化（之后）
        layer_output = self.layernorm_after(hidden_states)
        # 执行变换层操作
        layer_output = self.intermediate(layer_output)
        # 执行输出层操作，传入变换层输出和之前的隐藏状态，支持训练模式
        layer_output = self.mobilevit_output(layer_output, hidden_states, training=training)
        # 返回层的输出结果
        return layer_output

    # 实现 build 方法，用于手动构建层
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记该层已经构建
        self.built = True
        # 逐个构建该层的子层，如果已经存在则跳过
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        if getattr(self, "mobilevit_output", None) is not None:
            with tf.name_scope(self.mobilevit_output.name):
                self.mobilevit_output.build(None)
        if getattr(self, "layernorm_before", None) is not None:
            with tf.name_scope(self.layernorm_before.name):
                self.layernorm_before.build([None, None, self.hidden_size])
        if getattr(self, "layernorm_after", None) is not None:
            with tf.name_scope(self.layernorm_after.name):
                self.layernorm_after.build([None, None, self.hidden_size])


# 定义一个自定义的 Keras 层，实现多层 MobileViT 变压器的堆叠
class TFMobileViTTransformer(keras.layers.Layer):
    def __init__(self, config: MobileViTConfig, hidden_size: int, num_stages: int, **kwargs) -> None:
        super().__init__(**kwargs)
        # 初始化存储变压器层的列表
        self.layers = []
        # 根据指定的层数，逐层创建 MobileViT 变压器层并添加到列表中
        for i in range(num_stages):
            transformer_layer = TFMobileViTTransformerLayer(
                config,
                hidden_size=hidden_size,
                intermediate_size=int(hidden_size * config.mlp_ratio),
                name=f"layer.{i}",
            )
            self.layers.append(transformer_layer)
    # 定义一个方法，接收隐藏状态和训练标志作为输入，返回处理后的隐藏状态张量
    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 遍历神经网络模型的每一层
        for layer_module in self.layers:
            # 调用每一层的call方法，处理隐藏状态张量，并根据训练标志进行适当的处理
            hidden_states = layer_module(hidden_states, training=training)
        # 返回处理后的最终隐藏状态张量
        return hidden_states

    # 定义一个方法，用于构建神经网络模型
    def build(self, input_shape=None):
        # 如果模型已经构建过，直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果模型包含了layers属性
        if getattr(self, "layers", None) is not None:
            # 遍历模型的每一层
            for layer_module in self.layers:
                # 在TensorFlow中，使用命名空间来组织张量和操作，这里为当前层设置一个命名空间
                with tf.name_scope(layer_module.name):
                    # 调用每一层的build方法，传入input_shape参数，实现层的构建
                    layer_module.build(None)
    """
    MobileViT block: https://arxiv.org/abs/2110.02178
    """

    # 初始化函数，定义了 TFMobileViTLayer 类的构造方法
    def __init__(
        self,
        config: MobileViTConfig,  # MobileViTConfig 类型的配置参数对象
        in_channels: int,         # 输入通道数
        out_channels: int,        # 输出通道数
        stride: int,              # 步长
        hidden_size: int,         # 隐藏层大小
        num_stages: int,          # 阶段数
        dilation: int = 1,        # 膨胀率，默认为1
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)  # 调用父类的构造方法

        self.patch_width = config.patch_size   # 设置补丁宽度
        self.patch_height = config.patch_size  # 设置补丁高度

        # 根据步长选择是否创建下采样层
        if stride == 2:
            self.downsampling_layer = TFMobileViTInvertedResidual(
                config,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride if dilation == 1 else 1,  # 如果膨胀率为1，则使用给定的步长，否则步长为1
                dilation=dilation // 2 if dilation > 1 else 1,  # 计算膨胀率的一半，如果膨胀率大于1，否则为1
                name="downsampling_layer",
            )
            in_channels = out_channels  # 更新输入通道数为输出通道数
        else:
            self.downsampling_layer = None  # 否则不创建下采样层

        # 创建 kxk 卷积层
        self.conv_kxk = TFMobileViTConvLayer(
            config,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=config.conv_kernel_size,
            name="conv_kxk",
        )

        # 创建 1x1 卷积层，用于调整隐藏层大小
        self.conv_1x1 = TFMobileViTConvLayer(
            config,
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=1,
            use_normalization=False,
            use_activation=False,
            name="conv_1x1",
        )

        # 创建 MobileViTTransformer 实例，用于执行转换操作
        self.transformer = TFMobileViTTransformer(
            config, hidden_size=hidden_size, num_stages=num_stages, name="transformer"
        )

        # 创建层归一化层，使用给定的 epsilon 值
        self.layernorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm")

        # 创建投影卷积层，将隐藏层特征映射回原始输入通道数
        self.conv_projection = TFMobileViTConvLayer(
            config, in_channels=hidden_size, out_channels=in_channels, kernel_size=1, name="conv_projection"
        )

        # 创建融合卷积层，用于融合两个输入特征
        self.fusion = TFMobileViTConvLayer(
            config,
            in_channels=2 * in_channels,
            out_channels=in_channels,
            kernel_size=config.conv_kernel_size,
            name="fusion",
        )
        self.hidden_size = hidden_size  # 设置隐藏层大小
    def unfolding(self, features: tf.Tensor) -> Tuple[tf.Tensor, Dict]:
        # 获取补丁的宽度和高度
        patch_width, patch_height = self.patch_width, self.patch_height
        # 计算补丁的面积
        patch_area = tf.cast(patch_width * patch_height, "int32")

        # 获取输入特征张量的批量大小、原始高度、原始宽度和通道数
        batch_size = tf.shape(features)[0]
        orig_height = tf.shape(features)[1]
        orig_width = tf.shape(features)[2]
        channels = tf.shape(features)[3]

        # 计算新的高度和宽度，确保能被补丁大小整除
        new_height = tf.cast(tf.math.ceil(orig_height / patch_height) * patch_height, "int32")
        new_width = tf.cast(tf.math.ceil(orig_width / patch_width) * patch_width, "int32")

        # 判断是否需要插值
        interpolate = new_width != orig_width or new_height != orig_height
        if interpolate:
            # 如果需要插值，使用双线性插值方法调整特征大小
            features = tf.image.resize(features, size=(new_height, new_width), method="bilinear")

        # 计算沿宽度和高度的补丁数量
        num_patch_width = new_width // patch_width
        num_patch_height = new_height // patch_height
        num_patches = num_patch_height * num_patch_width

        # 转置特征张量的维度顺序，使得通道数排在第二维度
        features = tf.transpose(features, [0, 3, 1, 2])
        # 重塑张量，将其从 (batch_size, channels, orig_height, orig_width) 转变为 (batch_size * channels * num_patch_height, patch_height, num_patch_width, patch_width)
        patches = tf.reshape(
            features, (batch_size * channels * num_patch_height, patch_height, num_patch_width, patch_width)
        )
        # 再次转置张量的维度顺序，重新安排其形状为 (batch_size, num_patch_width, patch_height, patch_width)
        patches = tf.transpose(patches, [0, 2, 1, 3])
        # 重塑张量，将其形状从 (batch_size, num_patch_width, patch_height, patch_width) 转变为 (batch_size, channels, num_patches, patch_area)
        patches = tf.reshape(patches, (batch_size, channels, num_patches, patch_area))
        # 转置张量的维度顺序，使得补丁数量成为第三维度
        patches = tf.transpose(patches, [0, 3, 2, 1])
        # 重塑张量，将其形状从 (batch_size, patch_area, num_patches, channels) 转变为 (batch_size * patch_area, num_patches, channels)
        patches = tf.reshape(patches, (batch_size * patch_area, num_patches, channels))

        # 创建包含各种信息的字典
        info_dict = {
            "orig_size": (orig_height, orig_width),
            "batch_size": batch_size,
            "channels": channels,
            "interpolate": interpolate,
            "num_patches": num_patches,
            "num_patches_width": num_patch_width,
            "num_patches_height": num_patch_height,
        }
        # 返回补丁张量和信息字典
        return patches, info_dict
    def folding(self, patches: tf.Tensor, info_dict: Dict) -> tf.Tensor:
        # 获取每个补丁的宽度和高度
        patch_width, patch_height = self.patch_width, self.patch_height
        # 计算每个补丁的总像素数
        patch_area = int(patch_width * patch_height)

        # 从信息字典中获取批处理大小、通道数、补丁数量、补丁高度和补丁宽度
        batch_size = info_dict["batch_size"]
        channels = info_dict["channels"]
        num_patches = info_dict["num_patches"]
        num_patch_height = info_dict["num_patches_height"]
        num_patch_width = info_dict["num_patches_width"]

        # 将补丁重新整形成 (batch_size, patch_area, num_patches, -1)
        features = tf.reshape(patches, (batch_size, patch_area, num_patches, -1))
        # 调换维度顺序为 (batch_size, -1, num_patches, patch_area)
        features = tf.transpose(features, perm=(0, 3, 2, 1))
        # 将特征张量重新整形为 (batch_size * channels * num_patch_height, num_patch_width, patch_height, patch_width)
        features = tf.reshape(
            features, (batch_size * channels * num_patch_height, num_patch_width, patch_height, patch_width)
        )
        # 再次调换维度顺序为 (batch_size * channels * num_patch_height, patch_height, num_patch_width, patch_width)
        features = tf.transpose(features, perm=(0, 2, 1, 3))
        # 最终将特征张量重新整形为 (batch_size, channels, num_patch_height * patch_height, num_patch_width * patch_width)
        features = tf.reshape(
            features, (batch_size, channels, num_patch_height * patch_height, num_patch_width * patch_width)
        )
        # 再次调换维度顺序为 (batch_size, num_patch_height * patch_height, num_patch_width * patch_width, channels)
        features = tf.transpose(features, perm=(0, 2, 3, 1))

        # 如果需要插值，对特征图像素进行双线性插值
        if info_dict["interpolate"]:
            features = tf.image.resize(features, size=info_dict["orig_size"], method="bilinear")

        return features

    def call(self, features: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 如果有下采样层，对特征进行空间维度缩减
        if self.downsampling_layer:
            features = self.downsampling_layer(features, training=training)

        residual = features

        # 本地表示
        # 将特征经过 kxk 卷积层
        features = self.conv_kxk(features, training=training)
        # 再将特征经过 1x1 卷积层
        features = self.conv_1x1(features, training=training)

        # 将特征图转换为补丁
        patches, info_dict = self.unfolding(features)

        # 学习全局表示
        # 经过 Transformer 处理补丁
        patches = self.transformer(patches, training=training)
        # Layer normalization
        patches = self.layernorm(patches)

        # 将补丁转换回特征图
        features = self.folding(patches, info_dict)

        # 投影卷积层
        features = self.conv_projection(features, training=training)
        # 特征融合
        features = self.fusion(tf.concat([residual, features], axis=-1), training=training)
        return features
    # 如果已经构建过网络结构，则直接返回，不再重复构建
    if self.built:
        return
    
    # 设置标志位，表示网络已经构建完成
    self.built = True
    
    # 如果存在卷积核大小不为None的属性，构建对应的卷积层
    if getattr(self, "conv_kxk", None) is not None:
        # 在命名空间中构建conv_kxk层
        with tf.name_scope(self.conv_kxk.name):
            self.conv_kxk.build(None)
    
    # 如果存在1x1卷积层不为None的属性，构建对应的1x1卷积层
    if getattr(self, "conv_1x1", None) is not None:
        # 在命名空间中构建conv_1x1层
        with tf.name_scope(self.conv_1x1.name):
            self.conv_1x1.build(None)
    
    # 如果存在transformer层不为None的属性，构建transformer层
    if getattr(self, "transformer", None) is not None:
        # 在命名空间中构建transformer层
        with tf.name_scope(self.transformer.name):
            self.transformer.build(None)
    
    # 如果存在layernorm层不为None的属性，构建layernorm层
    if getattr(self, "layernorm", None) is not None:
        # 在命名空间中构建layernorm层，输入形状为[None, None, self.hidden_size]
        with tf.name_scope(self.layernorm.name):
            self.layernorm.build([None, None, self.hidden_size])
    
    # 如果存在投影卷积层不为None的属性，构建对应的投影卷积层
    if getattr(self, "conv_projection", None) is not None:
        # 在命名空间中构建conv_projection层
        with tf.name_scope(self.conv_projection.name):
            self.conv_projection.build(None)
    
    # 如果存在融合层不为None的属性，构建对应的融合层
    if getattr(self, "fusion", None) is not None:
        # 在命名空间中构建fusion层
        with tf.name_scope(self.fusion.name):
            self.fusion.build(None)
    
    # 如果存在下采样层不为None的属性，构建对应的下采样层
    if getattr(self, "downsampling_layer", None) is not None:
        # 在命名空间中构建downsampling_layer层
        with tf.name_scope(self.downsampling_layer.name):
            self.downsampling_layer.build(None)
# 定义 TFMobileViTEncoder 类，继承自 keras.layers.Layer
class TFMobileViTEncoder(keras.layers.Layer):
    # 初始化方法，接受 MobileViTConfig 类型的 config 参数和其他关键字参数
    def __init__(self, config: MobileViTConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        # 将传入的 config 参数赋值给实例变量 self.config
        self.config = config

        # 初始化空列表 self.layers，用于存储编码器的各个层
        self.layers = []

        # 根据输出步长 output_stride 调整分类主干网络的步幅
        dilate_layer_4 = dilate_layer_5 = False
        if config.output_stride == 8:
            dilate_layer_4 = True
            dilate_layer_5 = True
        elif config.output_stride == 16:
            dilate_layer_5 = True

        # 初始的空间卷积的扩张率设为1
        dilation = 1

        # 创建第一个 MobileNet 层 layer_1，并添加到 self.layers 列表中
        layer_1 = TFMobileViTMobileNetLayer(
            config,
            in_channels=config.neck_hidden_sizes[0],
            out_channels=config.neck_hidden_sizes[1],
            stride=1,
            num_stages=1,
            name="layer.0",
        )
        self.layers.append(layer_1)

        # 创建第二个 MobileNet 层 layer_2，并添加到 self.layers 列表中
        layer_2 = TFMobileViTMobileNetLayer(
            config,
            in_channels=config.neck_hidden_sizes[1],
            out_channels=config.neck_hidden_sizes[2],
            stride=2,
            num_stages=3,
            name="layer.1",
        )
        self.layers.append(layer_2)

        # 创建第三个通用层 layer_3，并添加到 self.layers 列表中
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

        # 如果 dilate_layer_4 为真，则将 dilation 增加到当前值的两倍
        if dilate_layer_4:
            dilation *= 2

        # 创建第四个通用层 layer_4，并添加到 self.layers 列表中
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

        # 如果 dilate_layer_5 为真，则将 dilation 增加到当前值的两倍
        if dilate_layer_5:
            dilation *= 2

        # 创建第五个通用层 layer_5，并添加到 self.layers 列表中
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

    # 定义 call 方法，用于执行前向传播
    def call(
        self,
        hidden_states: tf.Tensor,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        training: bool = False,
        # 剩余未注释的参数在这里
        ):
    ) -> Union[tuple, TFBaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        # 如果不需要输出所有隐藏状态，则初始化为空元组，否则初始化为None

        for i, layer_module in enumerate(self.layers):
            hidden_states = layer_module(hidden_states, training=training)
            # 依次对每个层模块进行前向传播计算，并更新隐藏状态

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                # 如果需要输出所有隐藏状态，则将当前隐藏状态添加到 all_hidden_states 元组中

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)
            # 如果不需要以字典形式返回结果，则返回包含非空值的元组

        return TFBaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states)
        # 以 TFBaseModelOutput 对象形式返回结果，包括最后一个隐藏状态和所有隐藏状态组成的元组

    def build(self, input_shape=None):
        if self.built:
            return
        # 如果模型已经建立，直接返回

        self.built = True
        # 标记模型已经建立

        if getattr(self, "layers", None) is not None:
            for layer_module in self.layers:
                with tf.name_scope(layer_module.name):
                    layer_module.build(None)
        # 对每个层模块进行建立操作，使用各自的名称作为命名空间
# 使用装饰器标记该类可序列化为 Keras 模型
@keras_serializable
class TFMobileViTMainLayer(keras.layers.Layer):
    # 指定配置类为 MobileViTConfig
    config_class = MobileViTConfig

    # 初始化方法，接受 MobileViTConfig 对象和一个扩展输出的布尔值作为参数
    def __init__(self, config: MobileViTConfig, expand_output: bool = True, **kwargs):
        super().__init__(**kwargs)
        # 将传入的配置对象和扩展输出标志保存为实例属性
        self.config = config
        self.expand_output = expand_output

        # 创建 MobileViT 的卷积处理层 conv_stem
        self.conv_stem = TFMobileViTConvLayer(
            config,
            in_channels=config.num_channels,
            out_channels=config.neck_hidden_sizes[0],
            kernel_size=3,
            stride=2,
            name="conv_stem",
        )

        # 创建 MobileViT 的编码器部分 encoder
        self.encoder = TFMobileViTEncoder(config, name="encoder")

        # 如果需要扩展输出，则创建 1x1 卷积层 conv_1x1_exp
        if self.expand_output:
            self.conv_1x1_exp = TFMobileViTConvLayer(
                config,
                in_channels=config.neck_hidden_sizes[5],
                out_channels=config.neck_hidden_sizes[6],
                kernel_size=1,
                name="conv_1x1_exp",
            )

        # 创建全局平均池化层 pooler，用于提取特征图的全局平均值
        self.pooler = keras.layers.GlobalAveragePooling2D(data_format="channels_first", name="pooler")

    # 私有方法，用于剪枝模型的注意力头
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    # 使用装饰器 unpack_inputs 标记的 call 方法，接受多个输入参数，并进行模型的前向传播
    @unpack_inputs
    def call(
        self,
        pixel_values: tf.Tensor | None = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,

        # pixel_values: 输入的像素值张量，可以为 None
        # output_hidden_states: 是否返回隐藏状态的标志，可选布尔值
        # return_dict: 是否返回字典格式的输出，可选布尔值
        # training: 是否处于训练模式的标志，布尔类型

        # 在这里可以添加更多的代码，继续构建模型的前向传播过程...
    ) -> Union[Tuple[tf.Tensor], TFBaseModelOutputWithPooling]:
        # 确定是否输出隐藏状态，默认为模型配置中的设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 确定是否使用返回字典，默认为模型配置中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 当在 CPU 上运行时，`keras.layers.Conv2D` 不支持 `NCHW` 格式。
        # 因此将输入格式从 `NCHW` 转换为 `NHWC`。
        # 形状为 (batch_size, in_height, in_width, in_channels=num_channels)
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))

        # 使用卷积 stem 层处理像素值
        embedding_output = self.conv_stem(pixel_values, training=training)

        # 使用编码器处理嵌入输出
        encoder_outputs = self.encoder(
            embedding_output, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training
        )

        # 如果设置了扩展输出，对最后隐藏状态进行处理
        if self.expand_output:
            # 对最后隐藏状态进行 1x1 卷积处理
            last_hidden_state = self.conv_1x1_exp(encoder_outputs[0])

            # 将输出格式改回 NCHW 以保持模块间的一致性
            last_hidden_state = tf.transpose(last_hidden_state, perm=[0, 3, 1, 2])

            # 全局平均池化：(batch_size, channels, height, width) -> (batch_size, channels)
            pooled_output = self.pooler(last_hidden_state)
        else:
            # 如果没有扩展输出，则直接使用编码器的最后隐藏状态
            last_hidden_state = encoder_outputs[0]

            # 将输出格式改回 NCHW 以保持模块间的一致性
            last_hidden_state = tf.transpose(last_hidden_state, perm=[0, 3, 1, 2])
            pooled_output = None

        # 如果不使用返回字典，根据是否扩展输出返回相应的输出格式
        if not return_dict:
            output = (last_hidden_state, pooled_output) if pooled_output is not None else (last_hidden_state,)

            # 将输出格式改回 NCHW 以保持模块间的一致性
            if not self.expand_output:
                remaining_encoder_outputs = encoder_outputs[1:]
                remaining_encoder_outputs = tuple(
                    [tf.transpose(h, perm=(0, 3, 1, 2)) for h in remaining_encoder_outputs[0]]
                )
                remaining_encoder_outputs = (remaining_encoder_outputs,)
                return output + remaining_encoder_outputs
            else:
                return output + encoder_outputs[1:]

        # 如果需要输出隐藏状态，则将所有隐藏状态输出的格式改回 NCHW
        if output_hidden_states:
            hidden_states = tuple([tf.transpose(h, perm=(0, 3, 1, 2)) for h in encoder_outputs[1]])

        # 返回 TFBaseModelOutputWithPooling 类型的结果
        return TFBaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=hidden_states if output_hidden_states else encoder_outputs.hidden_states,
        )
    # 如果模型已经构建完成，则直接返回，不再重复构建
    if self.built:
        return
    # 设置标志表示模型已经构建
    self.built = True

    # 如果存在卷积层(conv_stem)，则构建该层
    if getattr(self, "conv_stem", None) is not None:
        # 使用卷积层的命名空间，构建卷积层
        with tf.name_scope(self.conv_stem.name):
            self.conv_stem.build(None)

    # 如果存在编码器(encoder)，则构建该编码器
    if getattr(self, "encoder", None) is not None:
        # 使用编码器的命名空间，构建编码器
        with tf.name_scope(self.encoder.name):
            self.encoder.build(None)

    # 如果存在池化层(pooler)，则构建该层
    if getattr(self, "pooler", None) is not None:
        # 使用池化层的命名空间，构建池化层，输入维度为[None, None, None, None]
        with tf.name_scope(self.pooler.name):
            self.pooler.build([None, None, None, None])

    # 如果存在1x1卷积层(conv_1x1_exp)，则构建该层
    if getattr(self, "conv_1x1_exp", None) is not None:
        # 使用1x1卷积层的命名空间，构建1x1卷积层
        with tf.name_scope(self.conv_1x1_exp.name):
            self.conv_1x1_exp.build(None)
    """
    Documentation string defining the format of inputs accepted by models and layers in the MobileViT architecture.
    It explains the two supported input formats: keyword arguments and positional list/tuple/dict for input tensors.

    When using TensorFlow 2.0 Keras methods like `model.fit()`, the second format (list, tuple, dict) is preferred.
    This enables flexibility in passing inputs such as `pixel_values`, `attention_mask`, and `token_type_ids`.

    For Keras Functional API or subclassing, inputs can be:
    - A single tensor: `model(pixel_values)`
    - A list of tensors: `model([pixel_values, attention_mask])`
    - A dictionary of tensors: `model({"pixel_values": pixel_values, "token_type_ids": token_type_ids})`

    This documentation guides users on how to interface with MobileViT models and layers effectively.

    Parameters:
        config ([`MobileViTConfig`]): Configuration class containing all model parameters.
            Loading weights requires using [`~TFPreTrainedModel.from_pretrained`], which initializes the model with weights.

    """
    # Args: 声明函数的参数和类型
    # pixel_values (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]`, `Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`):
    #     像素数值。可以使用 [`AutoImageProcessor`] 获得像素值。详见 [`MobileViTImageProcessor.__call__`]。
    # output_hidden_states (`bool`, *optional*):
    #     是否返回所有层的隐藏状态。请查看返回的张量中的 `hidden_states` 以获取更多详细信息。此参数仅在 eager 模式下可用，在图模式下将使用配置中的值。
    # return_dict (`bool`, *optional*):
    #     是否返回一个 [`~utils.ModelOutput`] 而不是普通的元组。此参数在 eager 模式下可用，在图模式下将始终设置为 True。
"""
MobileViT model outputting raw hidden-states without any specific head on top.

此类定义了一个MobileViT模型，它没有特定的输出头部。

MOBILEVIT_START_DOCSTRING: 在此处未提供具体内容的示例文档字符串。

"""
class TFMobileViTModel(TFMobileViTPreTrainedModel):
    def __init__(self, config: MobileViTConfig, expand_output: bool = True, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config  # 初始化配置对象
        self.expand_output = expand_output  # 是否扩展输出标志

        self.mobilevit = TFMobileViTMainLayer(config, expand_output=expand_output, name="mobilevit")
        # 创建MobileViT主层对象

    @unpack_inputs
    @add_start_docstrings_to_model_forward(MOBILEVIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def call(
        self,
        pixel_values: tf.Tensor | None = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[Tuple[tf.Tensor], TFBaseModelOutputWithPooling]:
        output = self.mobilevit(pixel_values, output_hidden_states, return_dict, training=training)
        return output
        # 调用MobileViT主层对象进行前向传播，返回输出

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "mobilevit", None) is not None:
            with tf.name_scope(self.mobilevit.name):
                self.mobilevit.build(None)
        # 构建模型，确保MobileViT主层对象已建立



"""
MobileViT model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
ImageNet.

此类定义了一个带有图像分类头部的MobileViT模型，例如用于ImageNet。

MOBILEVIT_START_DOCSTRING: 在此处未提供具体内容的示例文档字符串。

"""
class TFMobileViTForImageClassification(TFMobileViTPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config: MobileViTConfig, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels  # 分类标签数量
        self.mobilevit = TFMobileViTMainLayer(config, name="mobilevit")  # 创建MobileViT主层对象

        # 分类器头部
        self.dropout = keras.layers.Dropout(config.classifier_dropout_prob)  # Dropout层
        self.classifier = (
            keras.layers.Dense(config.num_labels, name="classifier") if config.num_labels > 0 else tf.identity
        )  # 分类器，如果标签数量大于0则创建密集层，否则为恒等映射
        self.config = config  # 配置对象

    @unpack_inputs
    @add_start_docstrings_to_model_forward(MOBILEVIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=TFImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def call(
        self,
        pixel_values: tf.Tensor | None = None,
        output_hidden_states: Optional[bool] = None,
        labels: tf.Tensor | None = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[TFImageClassifierOutputWithNoAttention]:
        output = self.mobilevit(pixel_values, output_hidden_states, return_dict, training=training)
        return output
        # 调用MobileViT主层对象进行前向传播，返回输出
    ) -> Union[tuple, TFImageClassifierOutputWithNoAttention]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss). If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 如果 return_dict 不为 None，则使用给定的 return_dict；否则使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 mobilevit 模型进行前向传播
        outputs = self.mobilevit(
            pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training
        )

        # 如果 return_dict 为 True，则使用 outputs 的 pooler_output；否则使用 outputs 的第二个元素作为 pooled_output
        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        # 将 pooled_output 经过 dropout 和 classifier 模型，得到 logits
        logits = self.classifier(self.dropout(pooled_output, training=training))

        # 如果 labels 不为 None，则计算损失；否则损失为 None
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        # 如果 return_dict 为 False，则返回 logits 和 outputs 的其他隐藏状态
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则返回 TFImageClassifierOutputWithNoAttention 对象，包含损失、logits 和隐藏状态
        return TFImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)

    def build(self, input_shape=None):
        # 如果已经构建过模型，则直接返回
        if self.built:
            return
        
        # 将模型标记为已构建
        self.built = True
        
        # 如果 mobilevit 模型存在，则构建它
        if getattr(self, "mobilevit", None) is not None:
            with tf.name_scope(self.mobilevit.name):
                self.mobilevit.build(None)
        
        # 如果 classifier 模型存在，则构建它
        if getattr(self, "classifier", None) is not None:
            if hasattr(self.classifier, "name"):
                with tf.name_scope(self.classifier.name):
                    self.classifier.build([None, None, self.config.neck_hidden_sizes[-1]])
class TFMobileViTASPPPooling(keras.layers.Layer):
    # 初始化函数，定义了 TFMobileViTASPPPooling 类的构造方法
    def __init__(self, config: MobileViTConfig, in_channels: int, out_channels: int, **kwargs) -> None:
        super().__init__(**kwargs)

        # 创建全局平均池化层，保持维度，命名为 "global_pool"
        self.global_pool = keras.layers.GlobalAveragePooling2D(keepdims=True, name="global_pool")

        # 创建 TFMobileViTConvLayer 实例 conv_1x1，用于 1x1 卷积
        self.conv_1x1 = TFMobileViTConvLayer(
            config,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            use_normalization=True,
            use_activation="relu",
            name="conv_1x1",
        )

    # 定义调用方法，对输入特征进行处理
    def call(self, features: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 获取特征的空间尺寸（排除批量和通道维度）
        spatial_size = shape_list(features)[1:-1]
        # 应用全局池化到特征上
        features = self.global_pool(features)
        # 应用 1x1 卷积层到全局池化后的特征上
        features = self.conv_1x1(features, training=training)
        # 使用双线性插值方法将特征尺寸调整回原始空间尺寸
        features = tf.image.resize(features, size=spatial_size, method="bilinear")
        return features

    # 构建方法，用于建立层次结构
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在全局池化层，则建立其内部结构
        if getattr(self, "global_pool", None) is not None:
            with tf.name_scope(self.global_pool.name):
                self.global_pool.build([None, None, None, None])
        # 如果存在 conv_1x1 层，则建立其内部结构
        if getattr(self, "conv_1x1", None) is not None:
            with tf.name_scope(self.conv_1x1.name):
                self.conv_1x1.build(None)


class TFMobileViTASPP(keras.layers.Layer):
    """
    ASPP module defined in DeepLab papers: https://arxiv.org/abs/1606.00915, https://arxiv.org/abs/1706.05587
    """
    def __init__(self, config: MobileViTConfig, **kwargs) -> None:
        super().__init__(**kwargs)

        # 从配置中获取输入通道数作为ASPP模块的输入通道数
        in_channels = config.neck_hidden_sizes[-2]
        # 从配置中获取ASPP模块的输出通道数
        out_channels = config.aspp_out_channels

        # 检查配置中空洞卷积的扩张率是否为3个值，如果不是则抛出数值错误异常
        if len(config.atrous_rates) != 3:
            raise ValueError("Expected 3 values for atrous_rates")

        # 初始化空洞卷积层列表
        self.convs = []

        # 创建ASPP模块的第一个投影层，使用1x1卷积核
        in_projection = TFMobileViTConvLayer(
            config,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_activation="relu",
            name="convs.0",
        )
        self.convs.append(in_projection)

        # 创建并添加多个空洞卷积层到ASPP模块中
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

        # 创建ASPP模块的池化层
        pool_layer = TFMobileViTASPPPooling(
            config, in_channels, out_channels, name=f"convs.{len(config.atrous_rates) + 1}"
        )
        self.convs.append(pool_layer)

        # 创建ASPP模块的投影层，使用1x1卷积核，将所有特征图通道合并
        self.project = TFMobileViTConvLayer(
            config,
            in_channels=5 * out_channels,  # 合并后的输入通道数
            out_channels=out_channels,
            kernel_size=1,
            use_activation="relu",
            name="project",
        )

        # 创建ASPP模块的Dropout层，使用配置中的丢弃概率
        self.dropout = keras.layers.Dropout(config.aspp_dropout_prob)

    def call(self, features: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 将特征张量的通道维度调整为`(batch_size, height, width, channels)`的顺序
        features = tf.transpose(features, perm=[0, 2, 3, 1])
        pyramid = []
        # 对ASPP模块中的每一层进行前向传播计算
        for conv in self.convs:
            pyramid.append(conv(features, training=training))
        # 将所有ASPP模块层的输出在通道维度上拼接起来
        pyramid = tf.concat(pyramid, axis=-1)

        # 对合并后的特征进行投影操作
        pooled_features = self.project(pyramid, training=training)
        # 对投影后的特征进行Dropout操作
        pooled_features = self.dropout(pooled_features, training=training)
        return pooled_features

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经构建，则直接返回
        if getattr(self, "project", None) is not None:
            # 构建ASPP模块的投影层
            with tf.name_scope(self.project.name):
                self.project.build(None)
        if getattr(self, "convs", None) is not None:
            # 构建ASPP模块中的每一层
            for conv in self.convs:
                with tf.name_scope(conv.name):
                    conv.build(None)
class TFMobileViTDeepLabV3(keras.layers.Layer):
    """
    DeepLabv3 architecture: https://arxiv.org/abs/1706.05587
    """

    def __init__(self, config: MobileViTConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        # 初始化 ASPP 模块，配置来自 MobileViTConfig
        self.aspp = TFMobileViTASPP(config, name="aspp")

        # Dropout 层，使用给定的分类器 dropout 概率
        self.dropout = keras.layers.Dropout(config.classifier_dropout_prob)

        # 分类器层，用于输出分类标签
        self.classifier = TFMobileViTConvLayer(
            config,
            in_channels=config.aspp_out_channels,
            out_channels=config.num_labels,
            kernel_size=1,
            use_normalization=False,
            use_activation=False,
            bias=True,
            name="classifier",
        )

    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        # ASPP 模块处理最后一个隐藏状态的特征
        features = self.aspp(hidden_states[-1], training=training)
        # 应用 Dropout 操作到特征上
        features = self.dropout(features, training=training)
        # 使用分类器层进行最终的分类预测
        features = self.classifier(features, training=training)
        return features

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果 ASPP 模块存在，则构建 ASPP 模块
        if getattr(self, "aspp", None) is not None:
            with tf.name_scope(self.aspp.name):
                self.aspp.build(None)
        # 如果分类器存在，则构建分类器
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build(None)


@add_start_docstrings(
    """
    MobileViT model with a semantic segmentation head on top, e.g. for Pascal VOC.
    """,
    MOBILEVIT_START_DOCSTRING,
)
class TFMobileViTForSemanticSegmentation(TFMobileViTPreTrainedModel):
    def __init__(self, config: MobileViTConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)

        # 类别数目
        self.num_labels = config.num_labels
        # MobileViT 主层，不扩展输出，命名为 'mobilevit'
        self.mobilevit = TFMobileViTMainLayer(config, expand_output=False, name="mobilevit")
        # 语义分割头部，基于 TFMobileViTDeepLabV3 构建
        self.segmentation_head = TFMobileViTDeepLabV3(config, name="segmentation_head")

    def hf_compute_loss(self, logits, labels):
        # 将 logits 上采样到原始图像大小
        # `labels` 的形状为 (batch_size, height, width)
        label_interp_shape = shape_list(labels)[1:]

        upsampled_logits = tf.image.resize(logits, size=label_interp_shape, method="bilinear")
        # 计算加权损失
        loss_fct = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

        def masked_loss(real, pred):
            unmasked_loss = loss_fct(real, pred)
            mask = tf.cast(real != self.config.semantic_loss_ignore_index, dtype=unmasked_loss.dtype)
            masked_loss = unmasked_loss * mask
            # 与 https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_tf_utils.py#L210 类似的减少策略
            reduced_masked_loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(mask)
            return tf.reshape(reduced_masked_loss, (1,))

        return masked_loss(labels, upsampled_logits)
    # 应用装饰器 @unpack_inputs，用于解包输入参数
    @unpack_inputs
    # 应用装饰器 @add_start_docstrings_to_model_forward，向模型的前向传播函数添加起始文档字符串
    @add_start_docstrings_to_model_forward(MOBILEVIT_INPUTS_DOCSTRING)
    # 应用装饰器 @replace_return_docstrings，替换返回值的文档字符串，指定输出类型为 TFSemanticSegmenterOutputWithNoAttention，并指定配置类为 _CONFIG_FOR_DOC
    @replace_return_docstrings(output_type=TFSemanticSegmenterOutputWithNoAttention, config_class=_CONFIG_FOR_DOC)
    # 定义模型的前向传播函数，接受以下参数：
    def call(
        self,
        pixel_values: tf.Tensor | None = None,  # 像素值张量，可以为 None
        labels: tf.Tensor | None = None,         # 标签张量，可以为 None
        output_hidden_states: Optional[bool] = None,  # 可选的布尔值，控制是否输出隐藏状态
        return_dict: Optional[bool] = None,      # 可选的布尔值，控制是否以字典形式返回结果
        training: bool = False,                  # 布尔值，指示当前是否处于训练模式
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.mobilevit(
            pixel_values,
            output_hidden_states=True,  # 指定输出中间隐藏状态
            return_dict=return_dict,
            training=training,
        )

        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]
        # 提取编码器的隐藏状态用于后续的语义分割任务

        logits = self.segmentation_head(encoder_hidden_states, training=training)
        # 使用编码器的隐藏状态生成语义分割的 logits

        loss = None
        if labels is not None:
            if not self.config.num_labels > 1:
                raise ValueError("The number of labels should be greater than one")
            else:
                loss = self.hf_compute_loss(logits=logits, labels=labels)
                # 计算损失函数，要求标签数量大于1

        # 将 logits 的形状转换为 (batch_size, num_labels, height, width)，以保持 API 的一致性
        logits = tf.transpose(logits, perm=[0, 3, 1, 2])

        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
                # 输出包含 logits 和可能的其他隐藏状态
            else:
                output = (logits,) + outputs[2:]
                # 输出包含 logits 和可能的其他输出信息
            return ((loss,) + output) if loss is not None else output
            # 返回输出元组，可能包含损失和额外输出信息

        return TFSemanticSegmenterOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            # 返回带有损失、logits 和隐藏状态的 TFSemanticSegmenterOutputWithNoAttention 对象
        )
    # 如果模型已经构建，则直接返回，不进行重复构建
    if self.built:
        return
    # 将模型标记为已构建状态
    self.built = True
    
    # 如果存在名为"mobilevit"的属性，并且不为None，则构建其对应的部分
    if getattr(self, "mobilevit", None) is not None:
        # 在 TensorFlow 中为"mobilevit"部分创建命名作用域
        with tf.name_scope(self.mobilevit.name):
            # 调用"mobilevit"部分的build方法，传入None作为输入形状
            self.mobilevit.build(None)
    
    # 如果存在名为"segmentation_head"的属性，并且不为None，则构建其对应的部分
    if getattr(self, "segmentation_head", None) is not None:
        # 在 TensorFlow 中为"segmentation_head"部分创建命名作用域
        with tf.name_scope(self.segmentation_head.name):
            # 调用"segmentation_head"部分的build方法，传入None作为输入形状
            self.segmentation_head.build(None)
```