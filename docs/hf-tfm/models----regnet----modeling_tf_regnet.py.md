# `.\models\regnet\modeling_tf_regnet.py`

```
# 设置文件编码为 UTF-8
# 版权声明和版权信息，表明该文件的版权归 Meta Platforms, Inc. 和 The HuggingFace Inc. 团队所有
#
# 根据 Apache 许可证 2.0 版本（“许可证”）授权；
# 除非符合许可证的要求，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发的软件
# 无任何明示或暗示的担保或条件。
# 请参阅许可证了解特定语言下的权限和限制。
""" TensorFlow RegNet 模型."""

from typing import Optional, Tuple, Union

import tensorflow as tf

# 从相应模块导入必要的功能和类
from ...activations_tf import ACT2FN
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithNoAttention,
    TFBaseModelOutputWithPoolingAndNoAttention,
    TFSequenceClassifierOutput,
)
from ...modeling_tf_utils import (
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import shape_list
from ...utils import logging
from .configuration_regnet import RegNetConfig

# 获取日志记录器实例
logger = logging.get_logger(__name__)

# 用于文档的通用配置
_CONFIG_FOR_DOC = "RegNetConfig"

# 用于文档的基本检查点
_CHECKPOINT_FOR_DOC = "facebook/regnet-y-040"
# 预期输出的形状
_EXPECTED_OUTPUT_SHAPE = [1, 1088, 7, 7]

# 图像分类相关的检查点
_IMAGE_CLASS_CHECKPOINT = "facebook/regnet-y-040"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

# TFRegNet 模型的预训练模型存档列表
TF_REGNET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/regnet-y-040",
    # 查看所有 RegNet 模型：https://huggingface.co/models?filter=regnet
]

# 定义 TFRegNetConvLayer 类，继承自 keras.layers.Layer
class TFRegNetConvLayer(keras.layers.Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activation: Optional[str] = "relu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        # 对输入进行零填充，以确保输出大小与输入大小相同
        self.padding = keras.layers.ZeroPadding2D(padding=kernel_size // 2)
        # 定义卷积层，设置卷积核大小、步长、填充方式和组数
        self.convolution = keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=kernel_size,
            strides=stride,
            padding="VALID",
            groups=groups,
            use_bias=False,
            name="convolution",
        )
        # 批量归一化层，用于规范化卷积层的输出
        self.normalization = keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="normalization")
        # 激活函数，根据给定的激活函数名称选择激活函数或者返回标识函数
        self.activation = ACT2FN[activation] if activation is not None else tf.identity
        self.in_channels = in_channels
        self.out_channels = out_channels
    # 定义一个方法，用于调用神经网络层的操作
    def call(self, hidden_state):
        # 对输入的隐藏状态进行填充，并进行卷积操作
        hidden_state = self.convolution(self.padding(hidden_state))
        # 对卷积后的结果进行规范化（例如批量归一化）
        hidden_state = self.normalization(hidden_state)
        # 对规范化后的结果应用激活函数（如ReLU）
        hidden_state = self.activation(hidden_state)
        # 返回处理后的隐藏状态
        return hidden_state

    # 定义一个方法，用于构建神经网络层
    def build(self, input_shape=None):
        # 如果网络层已经构建，则直接返回
        if self.built:
            return
        # 标记网络层为已构建状态
        self.built = True
        # 如果存在卷积操作，并且未被构建，则构建卷积操作
        if getattr(self, "convolution", None) is not None:
            with tf.name_scope(self.convolution.name):
                # 使用输入通道数构建卷积层
                self.convolution.build([None, None, None, self.in_channels])
        # 如果存在规范化操作，并且未被构建，则构建规范化操作
        if getattr(self, "normalization", None) is not None:
            with tf.name_scope(self.normalization.name):
                # 使用输出通道数构建规范化层
                self.normalization.build([None, None, None, self.out_channels])
class TFRegNetEmbeddings(keras.layers.Layer):
    """
    RegNet Embeddings (stem) composed of a single aggressive convolution.
    """

    def __init__(self, config: RegNetConfig, **kwargs):
        super().__init__(**kwargs)
        self.num_channels = config.num_channels  # 从配置中获取通道数
        self.embedder = TFRegNetConvLayer(
            in_channels=config.num_channels,  # 输入通道数
            out_channels=config.embedding_size,  # 输出通道数（嵌入维度）
            kernel_size=3,  # 卷积核大小
            stride=2,  # 步长
            activation=config.hidden_act,  # 激活函数
            name="embedder",  # 层的名称
        )

    def call(self, pixel_values):
        num_channels = shape_list(pixel_values)[1]  # 获取像素值的通道数
        if tf.executing_eagerly() and num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )

        # 当在 CPU 上运行时，`keras.layers.Conv2D` 不支持 `NCHW` 格式。
        # 因此将输入格式从 `NCHW` 转换为 `NHWC`。
        # shape = (batch_size, in_height, in_width, in_channels=num_channels)
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))  # 转置像素值的维度顺序
        hidden_state = self.embedder(pixel_values)  # 嵌入器处理像素值
        return hidden_state

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "embedder", None) is not None:
            with tf.name_scope(self.embedder.name):
                self.embedder.build(None)  # 构建嵌入器层


class TFRegNetShortCut(keras.layers.Layer):
    """
    RegNet shortcut, used to project the residual features to the correct size. If needed, it is also used to
    downsample the input using `stride=2`.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.convolution = keras.layers.Conv2D(
            filters=out_channels, kernel_size=1, strides=stride, use_bias=False, name="convolution"
        )  # 1x1 卷积层，用于投影和下采样
        self.normalization = keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="normalization")  # 批量归一化层
        self.in_channels = in_channels  # 输入通道数
        self.out_channels = out_channels  # 输出通道数

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        return self.normalization(self.convolution(inputs), training=training)  # 应用卷积和归一化操作

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "convolution", None) is not None:
            with tf.name_scope(self.convolution.name):
                self.convolution.build([None, None, None, self.in_channels])  # 构建卷积层
        if getattr(self, "normalization", None) is not None:
            with tf.name_scope(self.normalization.name):
                self.normalization.build([None, None, None, self.out_channels])  # 构建归一化层


class TFRegNetSELayer(keras.layers.Layer):
    """
    Placeholder for the SE (Squeeze-and-Excitation) Layer in RegNet, to be implemented.
    This layer is intended for enhancing channel-wise relationships adaptively.
    """
    Squeeze and Excitation layer (SE) proposed in [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507).
    """

    # 定义 Squeeze-and-Excitation（SE）层的类
    def __init__(self, in_channels: int, reduced_channels: int, **kwargs):
        super().__init__(**kwargs)
        # 创建全局平均池化层，用于计算特征图的平均值
        self.pooler = keras.layers.GlobalAveragePooling2D(keepdims=True, name="pooler")
        # 创建注意力机制的两个卷积层，用于生成注意力权重
        self.attention = [
            keras.layers.Conv2D(filters=reduced_channels, kernel_size=1, activation="relu", name="attention.0"),
            keras.layers.Conv2D(filters=in_channels, kernel_size=1, activation="sigmoid", name="attention.2"),
        ]
        # 记录输入通道数和降维后的通道数
        self.in_channels = in_channels
        self.reduced_channels = reduced_channels

    # 定义 SE 层的前向传播函数
    def call(self, hidden_state):
        # 对输入的特征图进行全局平均池化，生成池化后的结果
        pooled = self.pooler(hidden_state)
        # 对池化后的结果分别通过两个注意力卷积层，生成注意力权重
        for layer_module in self.attention:
            pooled = layer_module(pooled)
        # 将原始特征图与注意力权重相乘，增强特征表示
        hidden_state = hidden_state * pooled
        return hidden_state

    # 构建 SE 层，确保每个组件都被正确地构建和连接
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建全局平均池化层
        if getattr(self, "pooler", None) is not None:
            with tf.name_scope(self.pooler.name):
                self.pooler.build((None, None, None, None))
        # 构建注意力卷积层
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention[0].name):
                self.attention[0].build([None, None, None, self.in_channels])
            with tf.name_scope(self.attention[1].name):
                self.attention[1].build([None, None, None, self.reduced_channels])
# 定义 TFRegNetXLayer 类，表示 RegNet 模型中的一个层，类似于 ResNet 的瓶颈层，但具有不同的特性。
class TFRegNetXLayer(keras.layers.Layer):
    """
    RegNet's layer composed by three `3x3` convolutions, same as a ResNet bottleneck layer with reduction = 1.
    """

    # 初始化方法，设置层的参数和结构
    def __init__(self, config: RegNetConfig, in_channels: int, out_channels: int, stride: int = 1, **kwargs):
        super().__init__(**kwargs)
        # 检查是否需要应用快捷连接，根据输入和输出通道数以及步长来判断
        should_apply_shortcut = in_channels != out_channels or stride != 1
        # 如果需要应用快捷连接，则创建 TFRegNetShortCut 实例作为 shortcut 属性；否则创建线性激活函数作为 shortcut 属性
        self.shortcut = (
            TFRegNetShortCut(in_channels, out_channels, stride=stride, name="shortcut")
            if should_apply_shortcut
            else keras.layers.Activation("linear", name="shortcut")
        )
        # 定义三个卷积层的列表，每一层都是 TFRegNetConvLayer 类的实例，用于构建层内部的特征提取流程
        self.layers = [
            TFRegNetConvLayer(in_channels, out_channels, kernel_size=1, activation=config.hidden_act, name="layer.0"),
            TFRegNetConvLayer(
                out_channels, out_channels, stride=stride, groups=max(1, out_channels // config.groups_width),
                activation=config.hidden_act, name="layer.1"
            ),
            TFRegNetConvLayer(out_channels, out_channels, kernel_size=1, activation=None, name="layer.2"),
        ]
        # 激活函数根据配置文件中的隐藏激活函数来选择
        self.activation = ACT2FN[config.hidden_act]

    # 定义层的前向传播逻辑
    def call(self, hidden_state):
        # 保存输入的残差连接
        residual = hidden_state
        # 遍历每一层卷积，依次对 hidden_state 进行特征提取
        for layer_module in self.layers:
            hidden_state = layer_module(hidden_state)
        # 将残差连接通过快捷连接层进行处理
        residual = self.shortcut(residual)
        # 将特征提取后的 hidden_state 与处理后的残差相加
        hidden_state += residual
        # 使用预定义的激活函数对输出进行激活
        hidden_state = self.activation(hidden_state)
        return hidden_state

    # 构建方法，用于在第一次调用前构建层的变量
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果定义了快捷连接，则构建快捷连接层
        if getattr(self, "shortcut", None) is not None:
            with tf.name_scope(self.shortcut.name):
                self.shortcut.build(None)
        # 构建每一个卷积层
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)


class TFRegNetYLayer(keras.layers.Layer):
    """
    RegNet's Y layer: an X layer with Squeeze and Excitation.
    """
    # 初始化函数，用于初始化模型对象
    def __init__(self, config: RegNetConfig, in_channels: int, out_channels: int, stride: int = 1, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 确定是否应用快捷连接（shortcut），条件是输入通道数不等于输出通道数或步长不为1
        should_apply_shortcut = in_channels != out_channels or stride != 1
        # 计算组数，确保至少有一个组
        groups = max(1, out_channels // config.groups_width)
        # 如果应用快捷连接，则创建一个 TFRegNetShortCut 对象作为快捷连接，否则创建线性激活函数作为快捷连接
        self.shortcut = (
            TFRegNetShortCut(in_channels, out_channels, stride=stride, name="shortcut")
            if should_apply_shortcut
            else keras.layers.Activation("linear", name="shortcut")
        )
        # 定义模型的层列表，包括几个 TFRegNetConvLayer 层和一个 TFRegNetSELayer 层
        self.layers = [
            TFRegNetConvLayer(in_channels, out_channels, kernel_size=1, activation=config.hidden_act, name="layer.0"),
            TFRegNetConvLayer(
                out_channels, out_channels, stride=stride, groups=groups, activation=config.hidden_act, name="layer.1"
            ),
            TFRegNetSELayer(out_channels, reduced_channels=int(round(in_channels / 4)), name="layer.2"),
            TFRegNetConvLayer(out_channels, out_channels, kernel_size=1, activation=None, name="layer.3"),
        ]
        # 激活函数使用根据配置选择的激活函数
        self.activation = ACT2FN[config.hidden_act]

    # 调用函数，用于模型的前向传播
    def call(self, hidden_state):
        # 将输入状态作为残差
        residual = hidden_state
        # 遍历模型的每一层，并对输入状态进行处理
        for layer_module in self.layers:
            hidden_state = layer_module(hidden_state)
        # 将残差通过快捷连接处理
        residual = self.shortcut(residual)
        # 将处理后的状态与残差相加
        hidden_state += residual
        # 应用激活函数到最终的隐藏状态
        hidden_state = self.activation(hidden_state)
        # 返回最终的隐藏状态
        return hidden_state

    # 构建函数，用于构建模型的层次结构
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果存在快捷连接，则构建快捷连接
        if getattr(self, "shortcut", None) is not None:
            with tf.name_scope(self.shortcut.name):
                self.shortcut.build(None)
        # 遍历每一层，并构建每一层
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)
class TFRegNetStage(keras.layers.Layer):
    """
    A RegNet stage composed by stacked layers.
    """

    def __init__(
        self, config: RegNetConfig, in_channels: int, out_channels: int, stride: int = 2, depth: int = 2, **kwargs
    ):
        super().__init__(**kwargs)

        # 根据配置选择使用 TFRegNetXLayer 或 TFRegNetYLayer 作为层
        layer = TFRegNetXLayer if config.layer_type == "x" else TFRegNetYLayer

        # 创建层列表，第一层可能使用 stride=2 进行下采样
        self.layers = [
            layer(config, in_channels, out_channels, stride=stride, name="layers.0"),
            *[layer(config, out_channels, out_channels, name=f"layers.{i+1}") for i in range(depth - 1)],
        ]

    def call(self, hidden_state):
        # 逐层调用各层的 call 方法
        for layer_module in self.layers:
            hidden_state = layer_module(hidden_state)
        return hidden_state

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)


class TFRegNetEncoder(keras.layers.Layer):
    def __init__(self, config: RegNetConfig, **kwargs):
        super().__init__(**kwargs)
        self.stages = []

        # 根据配置中的 downsample_in_first_stage 决定第一阶段是否进行输入的下采样
        self.stages.append(
            TFRegNetStage(
                config,
                config.embedding_size,
                config.hidden_sizes[0],
                stride=2 if config.downsample_in_first_stage else 1,
                depth=config.depths[0],
                name="stages.0",
            )
        )

        # 构建多个阶段，每个阶段包含多个 TFRegNetStage
        in_out_channels = zip(config.hidden_sizes, config.hidden_sizes[1:])
        for i, ((in_channels, out_channels), depth) in enumerate(zip(in_out_channels, config.depths[1:])):
            self.stages.append(TFRegNetStage(config, in_channels, out_channels, depth=depth, name=f"stages.{i+1}"))

    def call(
        self, hidden_state: tf.Tensor, output_hidden_states: bool = False, return_dict: bool = True
    ) -> TFBaseModelOutputWithNoAttention:
        hidden_states = () if output_hidden_states else None

        # 逐阶段调用 TFRegNetStage 的 call 方法，收集隐藏状态
        for stage_module in self.stages:
            if output_hidden_states:
                hidden_states = hidden_states + (hidden_state,)
            hidden_state = stage_module(hidden_state)

        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)

        # 根据 return_dict 决定返回的结果类型
        if not return_dict:
            return tuple(v for v in [hidden_state, hidden_states] if v is not None)
        return TFBaseModelOutputWithNoAttention(last_hidden_state=hidden_state, hidden_states=hidden_states)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        for stage in self.stages:
            with tf.name_scope(stage.name):
                stage.build(None)
class TFRegNetMainLayer(keras.layers.Layer):
    # 使用 RegNetConfig 类来配置模型参数
    config_class = RegNetConfig

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        # 创建 TFRegNetEmbeddings 实例作为嵌入层
        self.embedder = TFRegNetEmbeddings(config, name="embedder")
        # 创建 TFRegNetEncoder 实例作为编码器
        self.encoder = TFRegNetEncoder(config, name="encoder")
        # 创建全局平均池化层，用于池化特征
        self.pooler = keras.layers.GlobalAveragePooling2D(keepdims=True, name="pooler")

    @unpack_inputs
    def call(
        self,
        pixel_values: tf.Tensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> TFBaseModelOutputWithPoolingAndNoAttention:
        # 根据需要设置是否输出隐藏状态和是否返回字典形式结果
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 通过嵌入层处理输入数据
        embedding_output = self.embedder(pixel_values, training=training)

        # 使用编码器处理嵌入输出
        encoder_outputs = self.encoder(
            embedding_output, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training
        )

        # 获取最后一个隐藏状态
        last_hidden_state = encoder_outputs[0]
        # 对最终池化的输出进行全局维度转换
        pooled_output = self.pooler(last_hidden_state)

        # 将池化的输出格式转换为 NCHW 格式，确保模块的一致性
        pooled_output = tf.transpose(pooled_output, perm=(0, 3, 1, 2))
        last_hidden_state = tf.transpose(last_hidden_state, perm=(0, 3, 1, 2))

        # 如果需要输出隐藏状态，则将所有隐藏状态也转换为 NCHW 格式
        if output_hidden_states:
            hidden_states = tuple([tf.transpose(h, perm=(0, 3, 1, 2)) for h in encoder_outputs[1]])

        # 如果不返回字典形式结果，则返回元组形式的输出
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 如果返回字典形式结果，则构造 TFBaseModelOutputWithPoolingAndNoAttention 对象
        return TFBaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=hidden_states if output_hidden_states else encoder_outputs.hidden_states,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果嵌入层已定义，则构建嵌入层
        if getattr(self, "embedder", None) is not None:
            with tf.name_scope(self.embedder.name):
                self.embedder.build(None)
        # 如果编码器已定义，则构建编码器
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 如果池化层已定义，则构建池化层
        if getattr(self, "pooler", None) is not None:
            with tf.name_scope(self.pooler.name):
                self.pooler.build((None, None, None, None))


class TFRegNetPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 使用 RegNetConfig 类来配置模型参数
    config_class = RegNetConfig
    # 指定基础模型的前缀名称为 "regnet"
    base_model_prefix = "regnet"
    # 模型的主要输入名称为 "pixel_values"
    main_input_name = "pixel_values"

    @property
    # 定义一个方法input_signature，用于返回输入数据的签名信息，通常在 TensorFlow 的模型定义中使用
    def input_signature(self):
        # 返回一个字典，描述了输入张量的规格和数据类型
        return {"pixel_values": tf.TensorSpec(shape=(None, self.config.num_channels, 224, 224), dtype=tf.float32)}
# 定义用于文档字符串的模型描述和参数说明，使用原始的三重引号格式化字符串
REGNET_START_DOCSTRING = r"""
    This model is a Tensorflow
    [keras.layers.Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer) sub-class. Use it as a
    regular Tensorflow Module and refer to the Tensorflow documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`RegNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义用于输入参数文档字符串的格式化字符串
REGNET_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`tf.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`ConveNextImageProcessor.__call__`] for details.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# 使用装饰器为类添加起始文档字符串和额外的模型前向传播方法文档
@add_start_docstrings(
    "The bare RegNet model outputting raw features without any specific head on top.",
    REGNET_START_DOCSTRING,
)
class TFRegNetModel(TFRegNetPreTrainedModel):
    def __init__(self, config: RegNetConfig, *inputs, **kwargs):
        # 调用父类的初始化方法，传递模型配置和额外的输入参数
        super().__init__(config, *inputs, **kwargs)
        # 创建主要的RegNet层，使用给定的配置和命名为"regnet"
        self.regnet = TFRegNetMainLayer(config, name="regnet")

    # 使用装饰器为call方法添加起始文档字符串、输入参数和代码示例文档
    @unpack_inputs
    @add_start_docstrings_to_model_forward(REGNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPoolingAndNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def call(
        self,
        pixel_values: tf.Tensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPoolingAndNoAttention, Tuple[tf.Tensor]]:
        # 如果没有明确指定输出隐藏状态，使用模型配置中的设定
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果没有明确指定返回字典形式的输出，使用模型配置中的设定
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用RegNet主层进行前向传播，传递像素值、输出隐藏状态选项、返回字典选项和训练模式
        outputs = self.regnet(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 如果不返回字典形式的输出，以元组形式返回
        if not return_dict:
            return (outputs[0],) + outputs[1:]

        # 返回TFBaseModelOutputWithPoolingAndNoAttention类型的输出，包括最终隐藏状态和池化输出
        return TFBaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=outputs.last_hidden_state,
            pooler_output=outputs.pooler_output,
            hidden_states=outputs.hidden_states,
        )
    # 如果模型已经构建完成，则直接返回，不进行重复构建
    if self.built:
        return
    
    # 将模型标记为已构建状态
    self.built = True
    
    # 检查是否存在名为 "regnet" 的属性，如果存在则执行以下操作
    if getattr(self, "regnet", None) is not None:
        # 使用 TensorFlow 的命名空间为 regnet 构建模型
        with tf.name_scope(self.regnet.name):
            # 调用 regnet 对象的 build 方法，传入 None 作为输入形状
            self.regnet.build(None)
@add_start_docstrings(
    """
    RegNet Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    REGNET_START_DOCSTRING,
)
class TFRegNetForImageClassification(TFRegNetPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config: RegNetConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.regnet = TFRegNetMainLayer(config, name="regnet")
        # classification head
        self.classifier = [
            keras.layers.Flatten(),  # 将输入展平以供后续全连接层使用
            keras.layers.Dense(config.num_labels, name="classifier.1") if config.num_labels > 0 else tf.identity,  # 分类器的全连接层
        ]

    @unpack_inputs
    @add_start_docstrings_to_model_forward(REGNET_INPUTS_DOCSTRING)  # 添加模型前向传播的文档字符串
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def call(
        self,
        pixel_values: Optional[tf.Tensor] = None,
        labels: Optional[tf.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states  # 设置是否输出隐藏状态
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # 设置是否使用返回字典

        outputs = self.regnet(
            pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training  # 调用 RegNet 主层进行前向传播
        )

        pooled_output = outputs.pooler_output if return_dict else outputs[1]  # 获取汇聚输出或指定位置的输出

        flattened_output = self.classifier[0](pooled_output)  # 使用展平层处理汇聚输出
        logits = self.classifier[1](flattened_output)  # 使用全连接层计算 logits

        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)  # 计算损失，若无标签则损失为 None

        if not return_dict:
            output = (logits,) + outputs[2:]  # 组合输出，包括 logits 和可能的其他输出
            return ((loss,) + output) if loss is not None else output  # 返回损失与输出，或者仅输出

        return TFSequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states)  # 返回包装的输出对象
    # 定义神经网络层的构建方法，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 将标志位设置为已构建
        self.built = True
        # 如果存在名为"regnet"的属性，并且不为None，则构建regnet部分
        if getattr(self, "regnet", None) is not None:
            # 在命名空间内构建regnet
            with tf.name_scope(self.regnet.name):
                self.regnet.build(None)
        # 如果存在名为"classifier"的属性，并且不为None，则构建classifier部分
        if getattr(self, "classifier", None) is not None:
            # 在命名空间内构建classifier[1]
            with tf.name_scope(self.classifier[1].name):
                # 构建classifier[1]，输入形状为[None, None, None, self.config.hidden_sizes[-1]]
                self.classifier[1].build([None, None, None, self.config.hidden_sizes[-1]])
```