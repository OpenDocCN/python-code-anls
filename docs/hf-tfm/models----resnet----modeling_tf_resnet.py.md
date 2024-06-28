# `.\models\resnet\modeling_tf_resnet.py`

```py
# coding=utf-8
# Copyright 2022 Microsoft Research, Inc. and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" TensorFlow ResNet model."""

from typing import Optional, Tuple, Union

import tensorflow as tf

from ...activations_tf import ACT2FN
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithNoAttention,
    TFBaseModelOutputWithPoolingAndNoAttention,
    TFImageClassifierOutputWithNoAttention,
)
from ...modeling_tf_utils import (
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import shape_list
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_resnet import ResNetConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "ResNetConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "microsoft/resnet-50"
_EXPECTED_OUTPUT_SHAPE = [1, 2048, 7, 7]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "microsoft/resnet-50"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tiger cat"

TF_RESNET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/resnet-50",
    # See all resnet models at https://huggingface.co/models?filter=resnet
]


class TFResNetConvLayer(keras.layers.Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        activation: str = "relu",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        # Calculate padding value based on kernel size for valid padding
        self.pad_value = kernel_size // 2
        # Define convolutional layer with specified parameters
        self.conv = keras.layers.Conv2D(
            out_channels, kernel_size=kernel_size, strides=stride, padding="valid", use_bias=False, name="convolution"
        )
        # Batch normalization layer with predefined epsilon and momentum values
        self.normalization = keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="normalization")
        # Activation function based on provided string or default to linear activation
        self.activation = ACT2FN[activation] if activation is not None else keras.layers.Activation("linear")
        # Store input and output channel counts for the layer
        self.in_channels = in_channels
        self.out_channels = out_channels
    # 对输入的 hidden_state 进行卷积操作
    def convolution(self, hidden_state: tf.Tensor) -> tf.Tensor:
        # 在高度和宽度两个维度上进行填充，以匹配 PyTorch Conv2D 模型的填充方式
        height_pad = width_pad = (self.pad_value, self.pad_value)
        # 使用 TensorFlow 的 tf.pad 函数对 hidden_state 进行填充操作
        hidden_state = tf.pad(hidden_state, [(0, 0), height_pad, width_pad, (0, 0)])
        # 使用预先定义的卷积层 conv 对填充后的 hidden_state 进行卷积操作
        hidden_state = self.conv(hidden_state)
        # 返回卷积后的结果
        return hidden_state

    # 模型的调用方法，用于执行前向传播
    def call(self, hidden_state: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 调用 convolution 方法对输入的 hidden_state 进行卷积处理
        hidden_state = self.convolution(hidden_state)
        # 使用 normalization 方法对卷积后的 hidden_state 进行归一化处理
        hidden_state = self.normalization(hidden_state, training=training)
        # 对归一化后的 hidden_state 应用激活函数 activation
        hidden_state = self.activation(hidden_state)
        # 返回经过激活函数处理后的结果
        return hidden_state

    # 在构建模型时被调用，用于定义模型的各个层
    def build(self, input_shape=None):
        # 如果模型已经构建过，直接返回
        if self.built:
            return
        # 将模型标记为已构建
        self.built = True
        # 如果已定义卷积层 conv，则构建卷积层，指定输入通道数为 self.in_channels
        if getattr(self, "conv", None) is not None:
            with tf.name_scope(self.conv.name):
                self.conv.build([None, None, None, self.in_channels])
        # 如果已定义归一化层 normalization，则构建归一化层，指定输出通道数为 self.out_channels
        if getattr(self, "normalization", None) is not None:
            with tf.name_scope(self.normalization.name):
                self.normalization.build([None, None, None, self.out_channels])
class TFResNetEmbeddings(keras.layers.Layer):
    """
    ResNet Embeddings (stem) composed of a single aggressive convolution.
    """

    def __init__(self, config: ResNetConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        # 创建一个 ResNet 的卷积层，用于嵌入处理
        self.embedder = TFResNetConvLayer(
            config.num_channels,
            config.embedding_size,
            kernel_size=7,
            stride=2,
            activation=config.hidden_act,
            name="embedder",
        )
        # 创建一个最大池化层，用于池化处理
        self.pooler = keras.layers.MaxPool2D(pool_size=3, strides=2, padding="valid", name="pooler")
        self.num_channels = config.num_channels

    def call(self, pixel_values: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 获取输入张量的形状信息
        _, _, _, num_channels = shape_list(pixel_values)
        # 如果是即时执行模式并且通道数不匹配，抛出值错误
        if tf.executing_eagerly() and num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        hidden_state = pixel_values
        # 将输入张量传入嵌入器进行处理
        hidden_state = self.embedder(hidden_state)
        # 对处理后的张量进行填充操作
        hidden_state = tf.pad(hidden_state, [[0, 0], [1, 1], [1, 1], [0, 0]])
        # 将填充后的张量传入池化层进行处理
        hidden_state = self.pooler(hidden_state)
        return hidden_state

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果嵌入器已经存在，建立嵌入器层
        if getattr(self, "embedder", None) is not None:
            with tf.name_scope(self.embedder.name):
                self.embedder.build(None)
        # 如果池化层已经存在，建立池化层
        if getattr(self, "pooler", None) is not None:
            with tf.name_scope(self.pooler.name):
                self.pooler.build(None)


class TFResNetShortCut(keras.layers.Layer):
    """
    ResNet shortcut, used to project the residual features to the correct size. If needed, it is also used to
    downsample the input using `stride=2`.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2, **kwargs) -> None:
        super().__init__(**kwargs)
        # 创建一个卷积层，用于调整残差特征到正确的大小，可以选择性地进行下采样
        self.convolution = keras.layers.Conv2D(
            out_channels, kernel_size=1, strides=stride, use_bias=False, name="convolution"
        )
        # 使用与 PyTorch 等效部分相同的默认动量和 epsilon 参数
        self.normalization = keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="normalization")
        self.in_channels = in_channels
        self.out_channels = out_channels

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_state = x
        # 通过卷积层处理输入张量
        hidden_state = self.convolution(hidden_state)
        # 通过批量归一化层处理卷积后的特征
        hidden_state = self.normalization(hidden_state, training=training)
        return hidden_state
    # 定义 build 方法，用于构建网络层
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回，避免重复构建
        if self.built:
            return
        # 将标记设置为已构建
        self.built = True
        
        # 如果存在卷积层对象
        if getattr(self, "convolution", None) is not None:
            # 在命名空间中为卷积层设置名字作用域
            with tf.name_scope(self.convolution.name):
                # 使用输入通道数构建卷积层
                self.convolution.build([None, None, None, self.in_channels])
        
        # 如果存在归一化层对象
        if getattr(self, "normalization", None) is not None:
            # 在命名空间中为归一化层设置名字作用域
            with tf.name_scope(self.normalization.name):
                # 使用输出通道数构建归一化层
                self.normalization.build([None, None, None, self.out_channels])
    # 定义 TFResNetBasicLayer 类，表示经典 ResNet 的基本残差层，由两个 3x3 卷积组成
    class TFResNetBasicLayer(keras.layers.Layer):
        """
        A classic ResNet's residual layer composed by two `3x3` convolutions.
        """

        def __init__(
            self, in_channels: int, out_channels: int, stride: int = 1, activation: str = "relu", **kwargs
        ) -> None:
            super().__init__(**kwargs)
            # 确定是否应用快捷连接（shortcut），当输入通道数不等于输出通道数或步长不为 1 时应用
            should_apply_shortcut = in_channels != out_channels or stride != 1
            # 第一个 3x3 卷积层，初始化为 TFResNetConvLayer 类的实例
            self.conv1 = TFResNetConvLayer(in_channels, out_channels, stride=stride, name="layer.0")
            # 第二个 3x3 卷积层，初始化为 TFResNetConvLayer 类的实例，激活函数设为 None
            self.conv2 = TFResNetConvLayer(out_channels, out_channels, activation=None, name="layer.1")
            # 快捷连接层，如果需要应用快捷连接，则初始化为 TFResNetShortCut 类的实例；否则使用线性激活函数
            self.shortcut = (
                TFResNetShortCut(in_channels, out_channels, stride=stride, name="shortcut")
                if should_apply_shortcut
                else keras.layers.Activation("linear", name="shortcut")
            )
            # 激活函数，根据 activation 参数选择对应的激活函数
            self.activation = ACT2FN[activation]

        def call(self, hidden_state: tf.Tensor, training: bool = False) -> tf.Tensor:
            # 保存输入的隐藏状态作为残差（residual）
            residual = hidden_state
            # 经过第一层卷积
            hidden_state = self.conv1(hidden_state, training=training)
            # 经过第二层卷积
            hidden_state = self.conv2(hidden_state, training=training)
            # 经过快捷连接层
            residual = self.shortcut(residual, training=training)
            # 将残差与卷积结果相加
            hidden_state += residual
            # 经过激活函数
            hidden_state = self.activation(hidden_state)
            # 返回处理后的隐藏状态
            return hidden_state

        def build(self, input_shape=None):
            # 如果已经建立，则直接返回
            if self.built:
                return
            # 标记为已建立
            self.built = True
            # 构建第一个卷积层 conv1
            if getattr(self, "conv1", None) is not None:
                with tf.name_scope(self.conv1.name):
                    self.conv1.build(None)
            # 构建第二个卷积层 conv2
            if getattr(self, "conv2", None) is not None:
                with tf.name_scope(self.conv2.name):
                    self.conv2.build(None)
            # 构建快捷连接层 shortcut
            if getattr(self, "shortcut", None) is not None:
                with tf.name_scope(self.shortcut.name):
                    self.shortcut.build(None)


    # 定义 TFResNetBottleNeckLayer 类，表示经典 ResNet 的瓶颈残差层，由三个 3x3 卷积组成
    class TFResNetBottleNeckLayer(keras.layers.Layer):
        """
        A classic ResNet's bottleneck layer composed by three `3x3` convolutions.

        The first `1x1` convolution reduces the input by a factor of `reduction` in order to make the second `3x3`
        convolution faster. The last `1x1` convolution remaps the reduced features to `out_channels`.
        """

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            activation: str = "relu",
            reduction: int = 4,
            **kwargs,
    ) -> None:
        # 调用父类的初始化方法，传递所有参数
        super().__init__(**kwargs)
        # 判断是否应用快捷方式，根据输入通道数、输出通道数和步长来确定
        should_apply_shortcut = in_channels != out_channels or stride != 1
        # 计算减少的通道数，用于第一个卷积层的输出通道数
        reduces_channels = out_channels // reduction
        # 创建第一个卷积层，将输入通道数转换为减少的通道数
        self.conv0 = TFResNetConvLayer(in_channels, reduces_channels, kernel_size=1, name="layer.0")
        # 创建第二个卷积层，将减少的通道数转换为相同的通道数，应用给定的步长
        self.conv1 = TFResNetConvLayer(reduces_channels, reduces_channels, stride=stride, name="layer.1")
        # 创建第三个卷积层，将通道数转换为输出通道数，应用 1x1 的卷积核
        self.conv2 = TFResNetConvLayer(reduces_channels, out_channels, kernel_size=1, activation=None, name="layer.2")
        # 创建快捷连接层，如果应用快捷方式则使用 TFResNetShortCut 类，否则使用线性激活
        self.shortcut = (
            TFResNetShortCut(in_channels, out_channels, stride=stride, name="shortcut")
            if should_apply_shortcut
            else keras.layers.Activation("linear", name="shortcut")
        )
        # 选择激活函数，根据给定的激活函数名称从预定义字典中获取对应的函数
        self.activation = ACT2FN[activation]

    def call(self, hidden_state: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 将输入状态保存为残差
        residual = hidden_state
        # 通过第一层卷积层
        hidden_state = self.conv0(hidden_state, training=training)
        # 通过第二层卷积层
        hidden_state = self.conv1(hidden_state, training=training)
        # 通过第三层卷积层
        hidden_state = self.conv2(hidden_state, training=training)
        # 应用快捷连接，并传入训练状态
        residual = self.shortcut(residual, training=training)
        # 将残差与卷积结果相加
        hidden_state += residual
        # 应用激活函数到加和的结果
        hidden_state = self.activation(hidden_state)
        # 返回最终的隐藏状态
        return hidden_state

    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回
        if self.built:
            return
        # 标记模型为已构建
        self.built = True
        # 如果存在 conv0 属性，则构建 conv0
        if getattr(self, "conv0", None) is not None:
            with tf.name_scope(self.conv0.name):
                self.conv0.build(None)
        # 如果存在 conv1 属性，则构建 conv1
        if getattr(self, "conv1", None) is not None:
            with tf.name_scope(self.conv1.name):
                self.conv1.build(None)
        # 如果存在 conv2 属性，则构建 conv2
        if getattr(self, "conv2", None) is not None:
            with tf.name_scope(self.conv2.name):
                self.conv2.build(None)
        # 如果存在 shortcut 属性，则构建 shortcut
        if getattr(self, "shortcut", None) is not None:
            with tf.name_scope(self.shortcut.name):
                self.shortcut.build(None)
class TFResNetStage(keras.layers.Layer):
    """
    A ResNet stage composed of stacked layers.
    """

    def __init__(
        self, config: ResNetConfig, in_channels: int, out_channels: int, stride: int = 2, depth: int = 2, **kwargs
    ) -> None:
        super().__init__(**kwargs)

        # 根据配置选择使用瓶颈块或基本块作为每一层的构建单元
        layer = TFResNetBottleNeckLayer if config.layer_type == "bottleneck" else TFResNetBasicLayer

        # 创建当前阶段的层列表，第一层有可能对输入进行下采样
        layers = [layer(in_channels, out_channels, stride=stride, activation=config.hidden_act, name="layers.0")]
        layers += [
            layer(out_channels, out_channels, activation=config.hidden_act, name=f"layers.{i + 1}")
            for i in range(depth - 1)
        ]
        self.stage_layers = layers

    def call(self, hidden_state: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 依次通过每一层处理隐藏状态
        for layer in self.stage_layers:
            hidden_state = layer(hidden_state, training=training)
        return hidden_state

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "stage_layers", None) is not None:
            # 对每一层进行构建
            for layer in self.stage_layers:
                with tf.name_scope(layer.name):
                    layer.build(None)


class TFResNetEncoder(keras.layers.Layer):
    def __init__(self, config: ResNetConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        # 根据配置创建多个 ResNet 阶段
        self.stages = [
            TFResNetStage(
                config,
                config.embedding_size,
                config.hidden_sizes[0],
                stride=2 if config.downsample_in_first_stage else 1,
                depth=config.depths[0],
                name="stages.0",
            )
        ]
        for i, (in_channels, out_channels, depth) in enumerate(
            zip(config.hidden_sizes, config.hidden_sizes[1:], config.depths[1:])
        ):
            self.stages.append(TFResNetStage(config, in_channels, out_channels, depth=depth, name=f"stages.{i + 1}"))

    def call(
        self,
        hidden_state: tf.Tensor,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        training: bool = False,
    ) -> TFBaseModelOutputWithNoAttention:
        # 初始化隐藏状态元组
        hidden_states = () if output_hidden_states else None

        # 依次通过每个阶段模块处理隐藏状态
        for stage_module in self.stages:
            if output_hidden_states:
                hidden_states = hidden_states + (hidden_state,)

            hidden_state = stage_module(hidden_state, training=training)

        # 如果需要输出隐藏状态，将当前隐藏状态添加到元组中
        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)

        # 根据需要返回输出形式
        if not return_dict:
            return tuple(v for v in [hidden_state, hidden_states] if v is not None)

        # 返回一个 TFBaseModelOutputWithNoAttention 对象，包含最后的隐藏状态和所有隐藏状态元组
        return TFBaseModelOutputWithNoAttention(last_hidden_state=hidden_state, hidden_states=hidden_states)
    # 定义 build 方法，用于构建模型层次结构
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回，避免重复构建
        if self.built:
            return
        # 将标志位设置为已构建
        self.built = True
        # 检查是否定义了 stages 属性
        if getattr(self, "stages", None) is not None:
            # 遍历每一个层并构建它们
            for layer in self.stages:
                # 使用层的名字作为命名空间，构建该层
                with tf.name_scope(layer.name):
                    layer.build(None)
@keras_serializable
class TFResNetMainLayer(keras.layers.Layer):
    # 设置该层使用的配置类
    config_class = ResNetConfig

    def __init__(self, config: ResNetConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        # 初始化层的配置
        self.config = config
        # 创建 TFResNetEmbeddings 实例作为嵌入器
        self.embedder = TFResNetEmbeddings(config, name="embedder")
        # 创建 TFResNetEncoder 实例作为编码器
        self.encoder = TFResNetEncoder(config, name="encoder")
        # 创建全局平均池化层，用于池化特征图
        self.pooler = keras.layers.GlobalAveragePooling2D(keepdims=True)

    @unpack_inputs
    def call(
        self,
        pixel_values: tf.Tensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[Tuple[tf.Tensor], TFBaseModelOutputWithPoolingAndNoAttention]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # TF 2.0 image layers can't use NCHW format when running on CPU.
        # We transpose to NHWC format and then transpose back after the full forward pass.
        # (batch_size, num_channels, height, width) -> (batch_size, height, width, num_channels)
        pixel_values = tf.transpose(pixel_values, perm=[0, 2, 3, 1])
        # 使用嵌入器将像素值转换为嵌入输出
        embedding_output = self.embedder(pixel_values, training=training)

        # 使用编码器进行编码
        encoder_outputs = self.encoder(
            embedding_output, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training
        )

        # 获取最后一个隐藏状态
        last_hidden_state = encoder_outputs[0]

        # 使用池化器获取池化输出
        pooled_output = self.pooler(last_hidden_state)

        # 将所有输出转置为NCHW格式
        # (batch_size, height, width, num_channels) -> (batch_size, num_channels, height, width)
        last_hidden_state = tf.transpose(last_hidden_state, (0, 3, 1, 2))
        pooled_output = tf.transpose(pooled_output, (0, 3, 1, 2))
        hidden_states = ()
        for hidden_state in encoder_outputs[1:]:
            # 对所有隐藏状态进行转置为NCHW格式
            hidden_states = hidden_states + tuple(tf.transpose(h, (0, 3, 1, 2)) for h in hidden_state)

        if not return_dict:
            # 如果不返回字典，则返回元组形式的输出
            return (last_hidden_state, pooled_output) + hidden_states

        hidden_states = hidden_states if output_hidden_states else None

        # 返回带池化和无注意力机制的基础模型输出
        return TFBaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=hidden_states,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "embedder", None) is not None:
            with tf.name_scope(self.embedder.name):
                # 构建嵌入器
                self.embedder.build(None)
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                # 构建编码器
                self.encoder.build(None)
# 使用装饰器为 TFResNetModel 类添加文档字符串，描述其为不带特定顶部头部的裸 ResNet 模型输出原始特征
@add_start_docstrings(
    "The bare ResNet model outputting raw features without any specific head on top.",
    RESNET_START_DOCSTRING,
)
# 定义 TFResNetModel 类，继承自 TFResNetPreTrainedModel 类
class TFResNetModel(TFResNetPreTrainedModel):
    # 初始化方法，接受一个 ResNetConfig 类型的 config 参数
    def __init__(self, config: ResNetConfig, **kwargs) -> None:
        # 调用父类的初始化方法
        super().__init__(config, **kwargs)
        # 创建一个 TFResNetMainLayer 实例，命名为 "resnet"
        self.resnet = TFResNetMainLayer(config=config, name="resnet")

    # 使用装饰器为 call 方法添加文档字符串，描述其输入和输出
    @add_start_docstrings_to_model_forward(RESNET_INPUTS_DOCSTRING)
    # 使用装饰器添加代码示例的文档字符串
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPoolingAndNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    # 使用装饰器对输入进行解包，即解开输入的包装
    @unpack_inputs
    # 定义 call 方法，接受多个参数并返回相应的值
    def call(
        self,
        pixel_values: tf.Tensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[Tuple[tf.Tensor], TFBaseModelOutputWithPoolingAndNoAttention]:
        # 如果 output_hidden_states 为 None，则使用 self.config.output_hidden_states
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果 return_dict 为 None，则使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 self.resnet 的 __call__ 方法，传递相应的参数
        resnet_outputs = self.resnet(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 返回 resnet_outputs
        return resnet_outputs

    # 定义 build 方法，用于构建模型
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 标记模型为已构建
        self.built = True
        # 如果 self.resnet 存在，则在名为 self.resnet 的命名空间下构建它
        if getattr(self, "resnet", None) is not None:
            with tf.name_scope(self.resnet.name):
                self.resnet.build(None)


# 使用装饰器为 TFResNetForImageClassification 类添加文档字符串，描述其为在顶部带有图像分类头部（线性层位于池化特征之上）的 ResNet 模型
@add_start_docstrings(
    """
    ResNet Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    RESNET_START_DOCSTRING,
)
# 定义 TFResNetForImageClassification 类，继承自 TFResNetPreTrainedModel 和 TFSequenceClassificationLoss 类
class TFResNetForImageClassification(TFResNetPreTrainedModel, TFSequenceClassificationLoss):
    # 初始化方法，接受一个 ResNetConfig 类型的 config 参数
    def __init__(self, config: ResNetConfig, **kwargs) -> None:
        # 调用父类的初始化方法
        super().__init__(config, **kwargs)
        # 设置 self.num_labels 为 config.num_labels
        self.num_labels = config.num_labels
        # 创建一个 TFResNetMainLayer 实例，命名为 "resnet"
        self.resnet = TFResNetMainLayer(config, name="resnet")
        # 分类头部
        self.classifier_layer = (
            keras.layers.Dense(config.num_labels, name="classifier.1")
            if config.num_labels > 0
            else keras.layers.Activation("linear", name="classifier.1")
        )
        # 设置 self.config 为 config
        self.config = config

    # 定义 classifier 方法，接受一个 tf.Tensor 类型的参数 x，并返回分类器的 logits
    def classifier(self, x: tf.Tensor) -> tf.Tensor:
        # 使用 Flatten 层展平输入 x
        x = keras.layers.Flatten()(x)
        # 将展平后的结果传递给分类器层，得到 logits
        logits = self.classifier_layer(x)
        # 返回 logits
        return logits

    # 使用装饰器为 call 方法添加文档字符串，描述其输入和输出
    @add_start_docstrings_to_model_forward(RESNET_INPUTS_DOCSTRING)
    # 使用装饰器添加代码示例的文档字符串
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=TFImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    # 使用装饰器对输入进行解包，即解开输入的包装
    @unpack_inputs
    def call(
        self,
        pixel_values: tf.Tensor = None,
        labels: tf.Tensor = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        training: bool = False,
    ) -> Union[Tuple[tf.Tensor], TFImageClassifierOutputWithNoAttention]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 设置返回字典选项，如果未提供则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 ResNet 模型进行前向传播计算
        outputs = self.resnet(
            pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training
        )

        # 如果 return_dict 为 True，则使用 pooler_output 作为输出；否则使用 outputs 的第二个元素作为输出
        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        # 将池化输出传入分类器进行分类
        logits = self.classifier(pooled_output)

        # 如果 labels 不为 None，则计算损失；否则损失设为 None
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果 return_dict 为 False，则返回 logits 和额外的 hidden states；否则返回带有损失、logits 和 hidden states 的对象
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output

        return TFImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)

    def build(self, input_shape=None):
        # 如果已经构建过网络，直接返回
        if self.built:
            return
        self.built = True

        # 如果存在 resnet 模型，则构建 resnet
        if getattr(self, "resnet", None) is not None:
            with tf.name_scope(self.resnet.name):
                self.resnet.build(None)

        # 如果存在 classifier_layer，则构建 classifier_layer
        if getattr(self, "classifier_layer", None) is not None:
            with tf.name_scope(self.classifier_layer.name):
                self.classifier_layer.build([None, None, self.config.hidden_sizes[-1]])
```