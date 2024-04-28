# `.\transformers\models\resnet\modeling_tf_resnet.py`

```
# 设置编码格式为utf-8
# 版权信息
# 有关许可证的信息
""" TensorFlow ResNet model."""

# 引入必要的模块
from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithNoAttention,
    TFBaseModelOutputWithPoolingAndNoAttention,
    TFImageClassifierOutputWithNoAttention,
)
from ...modeling_tf_utils import TFPreTrainedModel, TFSequenceClassificationLoss, keras_serializable, unpack_inputs
from ...tf_utils import shape_list
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_resnet import ResNetConfig

# 获取logger对象
logger = logging.get_logger(__name__)

# 用于文档
_CONFIG_FOR_DOC = "ResNetConfig"
_CHECKPOINT_FOR_DOC = "microsoft/resnet-50"
_EXPECTED_OUTPUT_SHAPE = [1, 2048, 7, 7]
_IMAGE_CLASS_CHECKPOINT = "microsoft/resnet-50"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tiger cat"
TF_RESNET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/resnet-50",
    # See all resnet models at https://huggingface.co/models?filter=resnet
]

# 定义一个卷积层
class TFResNetConvLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        activation: str = "relu",
        **kwargs,
    ) -> None:
        # 初始化函数
        super().__init__(**kwargs)
        self.pad_value = kernel_size // 2
        # 创建一个卷积层
        self.conv = tf.keras.layers.Conv2D(
            out_channels, kernel_size=kernel_size, strides=stride, padding="valid", use_bias=False, name="convolution"
        )
        # 使用和PyTorch等效的默认动量和epsilon
        self.normalization = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="normalization")
        # 激活函数
        self.activation = ACT2FN[activation] if activation is not None else tf.keras.layers.Activation("linear")
        self.in_channels = in_channels
        self.out_channels = out_channels

    def convolution(self, hidden_state: tf.Tensor) -> tf.Tensor:
        # 填充以匹配PyTorch Conv2D模型中的填充
        height_pad = width_pad = (self.pad_value, self.pad_value)
        hidden_state = tf.pad(hidden_state, [(0, 0), height_pad, width_pad, (0, 0)])
        hidden_state = self.conv(hidden_state)
        return hidden_state
    # 定义神经网络层的调用方法，接受隐藏状态张量和训练标志，返回经过卷积、标准化和激活后的隐藏状态张量
    def call(self, hidden_state: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 将隐藏状态张量输入卷积层
        hidden_state = self.convolution(hidden_state)
        # 对卷积后的张量进行标准化处理，根据训练标志决定是否使用训练模式
        hidden_state = self.normalization(hidden_state, training=training)
        # 对标准化后的张量进行激活函数处理
        hidden_state = self.activation(hidden_state)
        # 返回处理后的隐藏状态张量
        return hidden_state

    # 构建神经网络层
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回，不再重复构建
        if self.built:
            return
        # 将构建标志设置为已构建
        self.built = True
        # 如果存在卷积层，构建卷积层
        if getattr(self, "conv", None) is not None:
            with tf.name_scope(self.conv.name):
                # 构建卷积层，输入张量的shape为 [None, None, None, self.in_channels]
                self.conv.build([None, None, None, self.in_channels])
        # 如果存在标准化层，构建标准化层
        if getattr(self, "normalization", None) is not None:
            with tf.name_scope(self.normalization.name):
                # 构建标准化层，输入张量的shape为 [None, None, None, self.out_channels]
                self.normalization.build([None, None, None, self.out_channels])
class TFResNetEmbeddings(tf.keras.layers.Layer):
    """
    ResNet Embeddings (stem) composed of a single aggressive convolution.
    """

    def __init__(self, config: ResNetConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        # 创建一个卷积层，用于处理嵌入特征
        self.embedder = TFResNetConvLayer(
            config.num_channels,
            config.embedding_size,
            kernel_size=7,
            stride=2,
            activation=config.hidden_act,
            name="embedder",
        )
        # 创建一个最大池化层
        self.pooler = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="valid", name="pooler")
        # 设置通道数
        self.num_channels = config.num_channels

    def call(self, pixel_values: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 获取输入张量的通道数
        _, _, _, num_channels = shape_list(pixel_values)
        # 如果处在 `eager execution` 环境中，并且通道数与配置中的不一致，则引发 ValueError
        if tf.executing_eagerly() and num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # 将嵌入层应用于输入张量
        hidden_state = self.embedder(pixel_values)
        # 对嵌入特征进行 padding 操作
        hidden_state = tf.pad(hidden_state, [[0, 0], [1, 1], [1, 1], [0, 0]])
        # 使用最大池化层对特征进行池化
        hidden_state = self.pooler(hidden_state)
        # 返回池化后的特征
        return hidden_state

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在 embedder 属性，则对其进行构建
        if getattr(self, "embedder", None) is not None:
            with tf.name_scope(self.embedder.name):
                self.embedder.build(None)
        # 如果存在 pooler 属性，则对其进行构建
        if getattr(self, "pooler", None) is not None:
            with tf.name_scope(self.pooler.name):
                self.pooler.build(None)


class TFResNetShortCut(tf.keras.layers.Layer):
    """
    ResNet shortcut, used to project the residual features to the correct size. If needed, it is also used to
    downsample the input using `stride=2`.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2, **kwargs) -> None:
        super().__init__(**kwargs)
        # 创建一个卷积层，用于调整残差特征的通道数
        self.convolution = tf.keras.layers.Conv2D(
            out_channels, kernel_size=1, strides=stride, use_bias=False, name="convolution"
        )
        # 创建一个批归一化层，用于规范化特征
        self.normalization = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="normalization")
        # 设置输入通道数和输出通道数
        self.in_channels = in_channels
        self.out_channels = out_channels

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 获取输入张量
        hidden_state = x
        # 使用卷积层处理输入特征
        hidden_state = self.convolution(hidden_state)
        # 使用批归一化层对特征进行规范化
        hidden_state = self.normalization(hidden_state, training=training)
        # 返回处理后的特征
        return hidden_state
    # 定义 build 方法，用于构建模型层
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回，避免重复构建
        if self.built:
            return
        # 标记模型层已经构建
        self.built = True
        # 检查是否存在卷积操作，如果存在，则构建卷积层
        if getattr(self, "convolution", None) is not None:
            # 使用 tf.name_scope 确保命名空间的独立性，将卷积层的构建过程包裹起来
            with tf.name_scope(self.convolution.name):
                # 构建卷积层，指定输入形状为 [None, None, None, self.in_channels]
                self.convolution.build([None, None, None, self.in_channels])
        # 检查是否存在归一化操作，如果存在，则构建归一化层
        if getattr(self, "normalization", None) is not None:
            # 使用 tf.name_scope 确保命名空间的独立性，将归一化层的构建过程包裹起来
            with tf.name_scope(self.normalization.name):
                # 构建归一化层，指定输入形状为 [None, None, None, self.out_channels]
                self.normalization.build([None, None, None, self.out_channels])
class TFResNetBasicLayer(tf.keras.layers.Layer):
    """
    一个经典的 ResNet 残差层，由两个 `3x3` 卷积层组成。
    """

    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1, activation: str = "relu", **kwargs
    ) -> None:
        # 调用父类的初始化方法，传递额外的参数
        super().__init__(**kwargs)
        # 判断是否需要应用捷径，即输入和输出通道不同或步长不为1
        should_apply_shortcut = in_channels != out_channels or stride != 1
        # 创建第一个卷积层，名称为"layer.0"
        self.conv1 = TFResNetConvLayer(in_channels, out_channels, stride=stride, name="layer.0")
        # 创建第二个卷积层，名称为"layer.1"，不使用激活函数
        self.conv2 = TFResNetConvLayer(out_channels, out_channels, activation=None, name="layer.1")
        # 根据是否需要应用捷径，创建捷径层，名称为"shortcut"
        self.shortcut = (
            TFResNetShortCut(in_channels, out_channels, stride=stride, name="shortcut")
            if should_apply_shortcut
            else tf.keras.layers.Activation("linear", name="shortcut")
        )
        # 使用指定的激活函数
        self.activation = ACT2FN[activation]

    def call(self, hidden_state: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 保存输入状态作为残差
        residual = hidden_state
        # 通过第一个卷积层
        hidden_state = self.conv1(hidden_state, training=training)
        # 通过第二个卷积层
        hidden_state = self.conv2(hidden_state, training=training)
        # 应用捷径层到残差
        residual = self.shortcut(residual, training=training)
        # 将卷积后的结果与捷径结果相加
        hidden_state += residual
        # 使用激活函数处理结果
        hidden_state = self.activation(hidden_state)
        # 返回处理后的结果
        return hidden_state

    def build(self, input_shape=None):
        # 如果已经构建，直接返回
        if self.built:
            return
        # 设置已构建标志
        self.built = True
        # 构建第一个卷积层
        if getattr(self, "conv1", None) is not None:
            with tf.name_scope(self.conv1.name):
                self.conv1.build(None)
        # 构建第二个卷积层
        if getattr(self, "conv2", None) is not None:
            with tf.name_scope(self.conv2.name):
                self.conv2.build(None)
        # 构建捷径层
        if getattr(self, "shortcut", None) is not None:
            with tf.name_scope(self.shortcut.name):
                self.shortcut.build(None)


class TFResNetBottleNeckLayer(tf.keras.layers.Layer):
    """
    一个经典的 ResNet 瓶颈层，由三个 `3x3` 卷积层组成。

    第一个 `1x1` 卷积层通过一个系数为 `reduction` 的因子减少输入以加速第二个 `3x3`
    卷积层。最后一个 `1x1` 卷积层将减少的特征重新映射到 `out_channels`。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        activation: str = "relu",
        reduction: int = 4,
        **kwargs,
    # 定义了一个 TFResNetBlock 类，继承自 tf.keras.layers.Layer
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        reduction: int = 4,
        activation: str = "gelu",
        **kwargs
    ) -> None:
        # 调用父类的构造函数
        super().__init__(**kwargs)
        # 判断是否需要应用快捷连接
        should_apply_shortcut = in_channels != out_channels or stride != 1
        # 计算缩减的通道数
        reduces_channels = out_channels // reduction
        # 定义第一个卷积层
        self.conv0 = TFResNetConvLayer(in_channels, reduces_channels, kernel_size=1, name="layer.0")
        # 定义第二个卷积层
        self.conv1 = TFResNetConvLayer(reduces_channels, reduces_channels, stride=stride, name="layer.1")
        # 定义第三个卷积层
        self.conv2 = TFResNetConvLayer(reduces_channels, out_channels, kernel_size=1, activation=None, name="layer.2")
        # 定义快捷连接层
        self.shortcut = (
            TFResNetShortCut(in_channels, out_channels, stride=stride, name="shortcut")
            if should_apply_shortcut
            else tf.keras.layers.Activation("linear", name="shortcut")
        )
        # 定义激活函数
        self.activation = ACT2FN[activation]
    
    # 定义前向传播函数
    def call(self, hidden_state: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 保存输入作为残差连接
        residual = hidden_state
        # 通过第一个卷积层
        hidden_state = self.conv0(hidden_state, training=training)
        # 通过第二个卷积层
        hidden_state = self.conv1(hidden_state, training=training)
        # 通过第三个卷积层
        hidden_state = self.conv2(hidden_state, training=training)
        # 应用快捷连接
        residual = self.shortcut(residual, training=training)
        # 将残差连接加到卷积结果上
        hidden_state += residual
        # 应用激活函数
        hidden_state = self.activation(hidden_state)
        # 返回最终结果
        return hidden_state
    
    # 定义 build 函数，用于初始化模型参数
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 初始化第一个卷积层的参数
        if getattr(self, "conv0", None) is not None:
            with tf.name_scope(self.conv0.name):
                self.conv0.build(None)
        # 初始化第二个卷积层的参数
        if getattr(self, "conv1", None) is not None:
            with tf.name_scope(self.conv1.name):
                self.conv1.build(None)
        # 初始化第三个卷积层的参数
        if getattr(self, "conv2", None) is not None:
            with tf.name_scope(self.conv2.name):
                self.conv2.build(None)
        # 初始化快捷连接层的参数
        if getattr(self, "shortcut", None) is not None:
            with tf.name_scope(self.shortcut.name):
                self.shortcut.build(None)
class TFResNetStage(tf.keras.layers.Layer):
    """
    一个由叠加层组成的ResNet阶段。
    """

    def __init__(
        self, config: ResNetConfig, in_channels: int, out_channels: int, stride: int = 2, depth: int = 2, **kwargs
    ) -> None:
        super().__init__(**kwargs)

        # 根据配置选择使用 ResNet 瓶颈层还是基础层作为该阶段的层
        layer = TFResNetBottleNeckLayer if config.layer_type == "bottleneck" else TFResNetBasicLayer

        # 创建包含层的列表，包括一个初始层和多个中间层
        layers = [layer(in_channels, out_channels, stride=stride, activation=config.hidden_act, name="layers.0")]
        layers += [
            layer(out_channels, out_channels, activation=config.hidden_act, name=f"layers.{i + 1}")
            for i in range(depth - 1)
        ]
        self.stage_layers = layers

    def call(self, hidden_state: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 依次对层进行前向传播
        for layer in self.stage_layers:
            hidden_state = layer(hidden_state, training=training)
        return hidden_state

    def build(self, input_shape=None):
        # 如果已经构建，直接返回
        if self.built:
            return
        self.built = True
        # 如果存在阶段层，对每个层进行构建
        if getattr(self, "stage_layers", None) is not None:
            for layer in self.stage_layers:
                with tf.name_scope(layer.name):
                    layer.build(None)


class TFResNetEncoder(tf.keras.layers.Layer):
    def __init__(self, config: ResNetConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        # 根据配置信息，创建 ResNet 编码器的各个阶段
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
        # 初始化隐藏状态的元组
        hidden_states = () if output_hidden_states else None

        # 依次对各个阶段进行前向传播
        for stage_module in self.stages:
            if output_hidden_states:
                hidden_states = hidden_states + (hidden_state,)

            hidden_state = stage_module(hidden_state, training=training)

        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)

        if not return_dict:
            # 根据返回值需求选择性返回结果
            return tuple(v for v in [hidden_state, hidden_states] if v is not None)

        return TFBaseModelOutputWithNoAttention(last_hidden_state=hidden_state, hidden_states=hidden_states)
    # 检查模型是否已经构建，如果已构建则直接返回，不进行重复构建
    def build(self, input_shape=None):
        if self.built:
            return
        # 将模型标记为已构建状态
        self.built = True
        # 如果模型中存在阶段（stages），则遍历每个阶段
        if getattr(self, "stages", None) is not None:
            for layer in self.stages:
                # 使用 TensorFlow 的命名空间，为当前层设置名称范围
                with tf.name_scope(layer.name):
                    # 构建当前层，传入输入形状为 None（表示动态形状）
                    layer.build(None)
class TFResNetPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定配置类
    config_class = ResNetConfig
    # 指定基础模型前缀
    base_model_prefix = "resnet"
    # 模型的主要输入名称
    main_input_name = "pixel_values"

    @property
    def input_signature(self):
        # 返回输入的签名，用于指定输入的形状和数据类型
        return {"pixel_values": tf.TensorSpec(shape=(None, self.config.num_channels, 224, 224), dtype=tf.float32)}


RESNET_START_DOCSTRING = r"""
    This model is a TensorFlow
    [tf.keras.layers.Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer) sub-class. Use it as a
    regular TensorFlow Module and refer to the TensorFlow documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ResNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""


RESNET_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`tf.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`ConvNextImageProcessor.__call__`] for details.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@keras_serializable
class TFResNetMainLayer(tf.keras.layers.Layer):
    # 指定配置类
    config_class = ResNetConfig

    def __init__(self, config: ResNetConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.config = config
        # 创建嵌入器
        self.embedder = TFResNetEmbeddings(config, name="embedder")
        # 创建编码器
        self.encoder = TFResNetEncoder(config, name="encoder")
        # 使用全局平均池化层
        self.pooler = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)

    @unpack_inputs
    def call(
        self,
        pixel_values: tf.Tensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[Tuple[tf.Tensor], TFBaseModelOutputWithPoolingAndNoAttention]:
        # 如果未指定是否返回隐藏状态，则使用模型配置中的设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定是否返回字典格式的输出，则使用模型配置中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # TF 2.0 图像层在 CPU 上运行时无法使用 NCHW 格式。
        # 我们先转置为 NHWC 格式，然后在完整的前向传播后再转置回来。
        # (batch_size, num_channels, height, width) -> (batch_size, height, width, num_channels)
        pixel_values = tf.transpose(pixel_values, perm=[0, 2, 3, 1])
        # 将像素值通过嵌入器传递
        embedding_output = self.embedder(pixel_values, training=training)

        # 将嵌入输出传递给编码器
        encoder_outputs = self.encoder(
            embedding_output, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training
        )

        # 获取编码器的最后隐藏状态
        last_hidden_state = encoder_outputs[0]

        # 通过池化器获取池化输出
        pooled_output = self.pooler(last_hidden_state)

        # 将所有输出转置为 NCHW 格式
        # (batch_size, height, width, num_channels) -> (batch_size, num_channels, height, width)
        last_hidden_state = tf.transpose(last_hidden_state, (0, 3, 1, 2))
        pooled_output = tf.transpose(pooled_output, (0, 3, 1, 2))
        hidden_states = ()
        # 对所有隐藏状态执行转置
        for hidden_state in encoder_outputs[1:]:
            hidden_states = hidden_states + tuple(tf.transpose(h, (0, 3, 1, 2)) for h in hidden_state)

        # 如果不返回字典格式的输出，则返回元组
        if not return_dict:
            return (last_hidden_state, pooled_output) + hidden_states

        # 如果不返回隐藏状态，则将隐藏状态设置为 None
        hidden_states = hidden_states if output_hidden_states else None

        # 返回 TFBaseModelOutputWithPoolingAndNoAttention 对象
        return TFBaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=hidden_states,
        )

    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回
        if self.built:
            return
        self.built = True
        # 如果嵌入器存在，则构建嵌入器
        if getattr(self, "embedder", None) is not None:
            with tf.name_scope(self.embedder.name):
                self.embedder.build(None)
        # 如果编码器存在，则构建编码器
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
```   
@add_start_docstrings(
    "The bare ResNet model outputting raw features without any specific head on top.",
    RESNET_START_DOCSTRING,
)
# 定义 TFResNetModel 类，继承自 TFResNetPreTrainedModel
class TFResNetModel(TFResNetPreTrainedModel):
    def __init__(self, config: ResNetConfig, **kwargs) -> None:
        # 调用父类构造函数初始化对象
        super().__init__(config, **kwargs)
        # 创建 TFResNetMainLayer 对象，用于特征提取
        self.resnet = TFResNetMainLayer(config=config, name="resnet")

    @add_start_docstrings_to_model_forward(RESNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPoolingAndNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    @unpack_inputs
    # 定义模型的前向传播函数
    def call(
        self,
        pixel_values: tf.Tensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[Tuple[tf.Tensor], TFBaseModelOutputWithPoolingAndNoAttention]:
        # 如果未指定，使用配置中的参数
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 TFResNetMainLayer 对象进行前向传播
        resnet_outputs = self.resnet(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        return resnet_outputs

    def build(self, input_shape=None):
        # 如果已构建，直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 resnet 属性，则构建 resnet 层
        if getattr(self, "resnet", None) is not None:
            with tf.name_scope(self.resnet.name):
                self.resnet.build(None)


@add_start_docstrings(
    """
    ResNet Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    RESNET_START_DOCSTRING,
)
# 定义 TFResNetForImageClassification 类，继承自 TFResNetPreTrainedModel 和 TFSequenceClassificationLoss
class TFResNetForImageClassification(TFResNetPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config: ResNetConfig, **kwargs) -> None:
        # 调用父类构造函数初始化对象
        super().__init__(config, **kwargs)
        # 设置分类标签数目
        self.num_labels = config.num_labels
        # 创建 TFResNetMainLayer 对象，用于特征提取
        self.resnet = TFResNetMainLayer(config, name="resnet")
        # 分类头部网络
        self.classifier_layer = (
            tf.keras.layers.Dense(config.num_labels, name="classifier.1")
            if config.num_labels > 0
            else tf.keras.layers.Activation("linear", name="classifier.1")
        )
        self.config = config

    def classifier(self, x: tf.Tensor) -> tf.Tensor:
        # 将输入展平
        x = tf.keras.layers.Flatten()(x)
        # 通过分类器层获取 logits
        logits = self.classifier_layer(x)
        return logits

    @add_start_docstrings_to_model_forward(RESNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=TFImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    @unpack_inputs
    # 定义一个方法，用于对输入的像素值进行分类或回归预测
    def call(
        self,
        pixel_values: tf.Tensor = None,  # 像素值张量，默认为空
        labels: tf.Tensor = None,  # 标签张量，默认为空
        output_hidden_states: bool = None,  # 是否输出隐藏状态，默认为空
        return_dict: bool = None,  # 是否返回字典，默认为空
        training: bool = False,  # 是否进行训练，默认为False
    ) -> Union[Tuple[tf.Tensor], TFImageClassifierOutputWithNoAttention]:  # 返回值为 tf.Tensor 或 TFImageClassifierOutputWithNoAttention 类型的元组
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional`):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # 根据传入参数或默认配置来确定是否返回字典类型的输出

        # 使用ResNet模型进行像素值处理，得到输出结果
        outputs = self.resnet(
            pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training
        )

        # 如果设置了返回字典类型的输出，则使用池化输出；否则使用第二个输出
        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        # 使用分类器对池化输出进行分类
        logits = self.classifier(pooled_output)

        # 如果没有传入标签，则损失值为空；否则计算损失值
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果不需要返回字典类型的输出，则返回分类结果和额外信息；否则返回新的TFImageClassifierOutputWithNoAttention类型的对象
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output

        return TFImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)  # 返回TFImageClassifierOutputWithNoAttention类型的对象包括损失值、分类结果和隐藏状态

    # 构建模型结构
    def build(self, input_shape=None):
        if self.built:  # 如果已经构建过，则直接返回
            return
        self.built = True  # 标记为已构建
        if getattr(self, "resnet", None) is not None:  # 如果存在ResNet模型
            with tf.name_scope(self.resnet.name):  # 使用ResNet模型的名字来定义命名空间
                self.resnet.build(None)  # 构建ResNet模型
        if getattr(self, "classifier_layer", None) is not None:  # 如果存在分类器层
            with tf.name_scope(self.classifier_layer.name):  # 使用分类器层的名字来定义命名空间
                self.classifier_layer.build([None, None, self.config.hidden_sizes[-1]])  # 构建分类器层
```