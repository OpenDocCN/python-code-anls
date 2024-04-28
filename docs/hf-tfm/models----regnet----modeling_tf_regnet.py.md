# `.\transformers\models\regnet\modeling_tf_regnet.py`

```py
# 这是一个 TensorFlow 实现的 RegNet 模型的源代码
# 它包含了一些导入、常量定义和一个自定义的 Conv 层

# 导入必要的库和函数
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_tf_outputs import TFBaseModelOutputWithNoAttention, TFBaseModelOutputWithPoolingAndNoAttention, TFSequenceClassifierOutput
from ...modeling_tf_utils import TFPreTrainedModel, TFSequenceClassificationLoss, keras_serializable, unpack_inputs
from ...tf_utils import shape_list
from ...utils import logging
from .configuration_regnet import RegNetConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义一些常量，如文档字符串
_CONFIG_FOR_DOC = "RegNetConfig"
_CHECKPOINT_FOR_DOC = "facebook/regnet-y-040"
_EXPECTED_OUTPUT_SHAPE = [1, 1088, 7, 7]
_IMAGE_CLASS_CHECKPOINT = "facebook/regnet-y-040"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"
TF_REGNET_PRETRAINED_MODEL_ARCHIVE_LIST = ["facebook/regnet-y-040"]

# 自定义的 Conv 层
class TFRegNetConvLayer(tf.keras.layers.Layer):
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
        # 使用 ZeroPadding2D 层来进行 SAME 卷积
        self.padding = tf.keras.layers.ZeroPadding2D(padding=kernel_size // 2)
        # 使用 Conv2D 层进行卷积操作
        self.convolution = tf.keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=kernel_size,
            strides=stride,
            padding="VALID",
            groups=groups,
            use_bias=False,
            name="convolution",
        )
        # 使用 BatchNormalization 层进行归一化
        self.normalization = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="normalization")
        # 使用 ACT2FN 字典来获取对应的激活函数
        self.activation = ACT2FN[activation] if activation is not None else tf.identity
        self.in_channels = in_channels
        self.out_channels = out_channels
    # 定义模型中的一个调用方法，接收隐藏状态作为输入，经过一系列处理后返回处理后的隐藏状态
    def call(self, hidden_state):
        # 对隐藏状态进行填充操作，并通过卷积层处理
        hidden_state = self.convolution(self.padding(hidden_state))
        # 对处理后的隐藏状态进行标准化操作
        hidden_state = self.normalization(hidden_state)
        # 对标准化后的隐藏状态进行激活函数处理
        hidden_state = self.activation(hidden_state)
        return hidden_state

    # 构建模型结构
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 根据需要构建卷积层
        if getattr(self, "convolution", None) is not None:
            with tf.name_scope(self.convolution.name):
                # 构建卷积层，输入通道数为self.in_channels
                self.convolution.build([None, None, None, self.in_channels])
        # 根据需要构建标准化层
        if getattr(self, "normalization", None) is not None:
            with tf.name_scope(self.normalization.name):
                # 构建标准化层，输出通道数为self.out_channels
                self.normalization.build([None, None, None, self.out_channels])
# 定义 RegNet Embeddings（stem），由一个具有激进卷积的层组成
class TFRegNetEmbeddings(tf.keras.layers.Layer):

    def __init__(self, config: RegNetConfig, **kwargs):
        # 初始化函数，设置 num_channels 和 embedder 属性
        super().__init__(**kwargs)
        self.num_channels = config.num_channels
        # 创建 TFRegNetConvLayer 对象作为 embedder 属性
        self.embedder = TFRegNetConvLayer(
            in_channels=config.num_channels,
            out_channels=config.embedding_size,
            kernel_size=3,
            stride=2,
            activation=config.hidden_act,
            name="embedder",
        )

    # 前向传播函数，执行输入的预处理和特征提取
    def call(self, pixel_values):
        # 获取输入张量 pixel_values 的通道数
        num_channels = shape_list(pixel_values)[1]
        if tf.executing_eagerly() and num_channels != self.num_channels:
            # 如果通道数不匹配，则抛出异常 ValueError
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )

        # 当在 CPU 上运行时，`tf.keras.layers.Conv2D` 不支持 `NCHW` 格式
        # 所以将输入格式从 `NCHW` 转换为 `NHWC`
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))
        # 使用 embedder 提取特征，并返回隐藏状态
        hidden_state = self.embedder(pixel_values)
        return hidden_state

    # 创建层的权重
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "embedder", None) is not None:
            with tf.name_scope(self.embedder.name):
                # 构建 embedder 层
                self.embedder.build(None)


# 定义 RegNet shortcut，用于将残差特征投影到正确的尺寸，如果需要，也用于通过 `stride=2` 进行输入下采样
class TFRegNetShortCut(tf.keras.layers.Layer):

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2, **kwargs):
        # 初始化函数，设置 convolution 和 normalization 属性
        super().__init__(**kwargs)
        # 创建 1x1 卷积层作为 convolution 属性
        self.convolution = tf.keras.layers.Conv2D(
            filters=out_channels, kernel_size=1, strides=stride, use_bias=False, name="convolution"
        )
        # 创建批归一化层作为 normalization 属性
        self.normalization = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="normalization")
        self.in_channels = in_channels
        self.out_channels = out_channels

    # 前向传播函数，执行输入特征的投影和归一化操作
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        return self.normalization(self.convolution(inputs), training=training)

    # 创建层的权重
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "convolution", None) is not None:
            with tf.name_scope(self.convolution.name):
                # 构建 convolution 层
                self.convolution.build([None, None, None, self.in_channels])
        if getattr(self, "normalization", None) is not None:
            with tf.name_scope(self.normalization.name):
                # 构建 normalization 层
                self.normalization.build([None, None, None, self.out_channels])


class TFRegNetSELayer(tf.keras.layers.Layer):
    # 略
    # Squeeze and Excitation layer (SE) proposed in [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507).
    class SqueezeAndExcitation(tf.keras.layers.Layer):
        """
        # 定义 SqueezeAndExcitation 类，继承自 tf.keras.layers.Layer
        def __init__(self, in_channels: int, reduced_channels: int, **kwargs):
            # 在初始化方法中，接收两个参数：in_channels 和 reduced_channels
            # in_channels 表示输入通道数，reduced_channels 表示压缩通道数
            # 调用父类的 __init__ 方法
            super().__init__(**kwargs)
            # 创建一个全局平均池化层，保持维度不变
            self.pooler = tf.keras.layers.GlobalAveragePooling2D(keepdims=True, name="pooler")
            # 创建两个卷积层组成注意力机制
            self.attention = [
                tf.keras.layers.Conv2D(filters=reduced_channels, kernel_size=1, activation="relu", name="attention.0"),
                tf.keras.layers.Conv2D(filters=in_channels, kernel_size=1, activation="sigmoid", name="attention.2"),
            ]
            # 保存输入通道数和压缩通道数
            self.in_channels = in_channels
            self.reduced_channels = reduced_channels
    
        def call(self, hidden_state):
            # 定义前向传播方法
            # [batch_size, h, w, num_channels] -> [batch_size, 1, 1, num_channels]
            # 使用池化层对输入特征图进行全局平均池化
            pooled = self.pooler(hidden_state)
            # 依次通过两个卷积层
            for layer_module in self.attention:
                pooled = layer_module(pooled)
            # 将注意力作用于输入特征图
            hidden_state = hidden_state * pooled
            # 返回处理后的特征图
            return hidden_state
    
        def build(self, input_shape=None):
            # 定义 build 方法，用于构建层的权重
            if self.built:
                # 如果已经构建过了，直接返回
                return
            self.built = True
            # 构建池化层的权重
            if getattr(self, "pooler", None) is not None:
                with tf.name_scope(self.pooler.name):
                    self.pooler.build((None, None, None, None))
            # 构建两个卷积层的权重
            if getattr(self, "attention", None) is not None:
                with tf.name_scope(self.attention[0].name):
                    self.attention[0].build([None, None, None, self.in_channels])
                with tf.name_scope(self.attention[1].name):
                    self.attention[1].build([None, None, None, self.reduced_channels])
class TFRegNetXLayer(tf.keras.layers.Layer):
    """
    RegNet's layer composed by three `3x3` convolutions, same as a ResNet bottleneck layer with reduction = 1.
    """

    def __init__(self, config: RegNetConfig, in_channels: int, out_channels: int, stride: int = 1, **kwargs):
        super().__init__(**kwargs)
        # 检查是否需要添加快捷连接
        should_apply_shortcut = in_channels != out_channels or stride != 1
        # 计算分组数
        groups = max(1, out_channels // config.groups_width)
        # 创建快捷连接
        self.shortcut = (
            TFRegNetShortCut(in_channels, out_channels, stride=stride, name="shortcut")
            if should_apply_shortcut
            else tf.keras.layers.Activation("linear", name="shortcut")
        )
        # 使用三个卷积层组成 X 层
        self.layers = [
            TFRegNetConvLayer(in_channels, out_channels, kernel_size=1, activation=config.hidden_act, name="layer.0"),
            TFRegNetConvLayer(
                out_channels, out_channels, stride=stride, groups=groups, activation=config.hidden_act, name="layer.1"
            ),
            TFRegNetConvLayer(out_channels, out_channels, kernel_size=1, activation=None, name="layer.2"),
        ]
        # 激活函数
        self.activation = ACT2FN[config.hidden_act]

    def call(self, hidden_state):
        # 保存残差连接
        residual = hidden_state
        # 将隐藏状态通过每个层
        for layer_module in self.layers:
            hidden_state = layer_module(hidden_state)
        # 运行快捷连接并与隐藏状态相加
        residual = self.shortcut(residual)
        hidden_state += residual
        # 使用激活函数处理结果
        hidden_state = self.activation(hidden_state)
        return hidden_state

    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在快捷连接，构建快捷连接
        if getattr(self, "shortcut", None) is not None:
            with tf.name_scope(self.shortcut.name):
                self.shortcut.build(None)
        # 构建每个层
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)


class TFRegNetYLayer(tf.keras.layers.Layer):
    """
    RegNet's Y layer: an X layer with Squeeze and Excitation.
    """
    # 初始化函数，接受配置、输入通道数、输出通道数和步长作为参数
    def __init__(self, config: RegNetConfig, in_channels: int, out_channels: int, stride: int = 1, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)
        # 判断是否应用快捷方式连接
        should_apply_shortcut = in_channels != out_channels or stride != 1
        # 计算分组数
        groups = max(1, out_channels // config.groups_width)
        # 如果应用快捷方式连接，则创建快捷方式连接层；否则创建线性激活函数层
        self.shortcut = (
            TFRegNetShortCut(in_channels, out_channels, stride=stride, name="shortcut")
            if should_apply_shortcut
            else tf.keras.layers.Activation("linear", name="shortcut")
        )
        # 创建卷积层、SE(Squeeze-and-Excitation)层和线性激活函数层
        self.layers = [
            TFRegNetConvLayer(in_channels, out_channels, kernel_size=1, activation=config.hidden_act, name="layer.0"),
            TFRegNetConvLayer(
                out_channels, out_channels, stride=stride, groups=groups, activation=config.hidden_act, name="layer.1"
            ),
            TFRegNetSELayer(out_channels, reduced_channels=int(round(in_channels / 4)), name="layer.2"),
            TFRegNetConvLayer(out_channels, out_channels, kernel_size=1, activation=None, name="layer.3"),
        ]
        # 根据配置的激活函数名称选择激活函数
        self.activation = ACT2FN[config.hidden_act]

    # 前向传播函数，接受隐藏状态作为参数
    def call(self, hidden_state):
        # 保存原始隐藏状态
        residual = hidden_state
        # 逐层调用卷积层，并更新隐藏状态
        for layer_module in self.layers:
            hidden_state = layer_module(hidden_state)
        # 对快捷方式连接进行处理
        residual = self.shortcut(residual)
        # 将卷积层的输出和快捷方式连接的结果相加
        hidden_state += residual
        # 应用激活函数
        hidden_state = self.activation(hidden_state)
        # 返回更新后的隐藏状态
        return hidden_state

    # 构建函数，用于构建网络结构
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果快捷方式连接层不为空，使用名称作为命名空间，构建快捷方式连接层
        if getattr(self, "shortcut", None) is not None:
            with tf.name_scope(self.shortcut.name):
                self.shortcut.build(None)
        # 对每一层的名称作为命名空间，依次构建卷积层
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)
class TFRegNetStage(tf.keras.layers.Layer):
    """
    A RegNet stage composed by stacked layers.
    """

    def __init__(
        self, config: RegNetConfig, in_channels: int, out_channels: int, stride: int = 2, depth: int = 2, **kwargs
    ):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 根据配置和参数初始化层
        layer = TFRegNetXLayer if config.layer_type == "x" else TFRegNetYLayer
        self.layers = [
            # 首层进行下采样，步长为2
            layer(config, in_channels, out_channels, stride=stride, name="layers.0"),
            *[layer(config, out_channels, out_channels, name=f"layers.{i+1}") for i in range(depth - 1)],
        ]

    def call(self, hidden_state):
        # 循环调用所有的层，并更新隐藏状态
        for layer_module in self.layers:
            hidden_state = layer_module(hidden_state)
        return hidden_state

    def build(self, input_shape=None):
        # 如果已经构建过了，则直接返回
        if self.built:
            return
        # 标记已经构建
        self.built = True
        if getattr(self, "layers", None) is not None:
            # 循环构建每个层
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)


class TFRegNetEncoder(tf.keras.layers.Layer):
    def __init__(self, config: RegNetConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        self.stages = []
        # 根据 `downsample_in_first_stage`，首阶段的第一层可能对输入进行下采样，也可能不下采样
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
        in_out_channels = zip(config.hidden_sizes, config.hidden_sizes[1:])
        for i, ((in_channels, out_channels), depth) in enumerate(zip(in_out_channels, config.depths[1:])):
            self.stages.append(TFRegNetStage(config, in_channels, out_channels, depth=depth, name=f"stages.{i+1}"))

    def call(
        self, hidden_state: tf.Tensor, output_hidden_states: bool = False, return_dict: bool = True
    ) -> TFBaseModelOutputWithNoAttention:
        hidden_states = () if output_hidden_states else None

        for stage_module in self.stages:
            if output_hidden_states:
                hidden_states = hidden_states + (hidden_state,)

            hidden_state = stage_module(hidden_state)

        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)

        if not return_dict:
            return tuple(v for v in [hidden_state, hidden_states] if v is not None)

        return TFBaseModelOutputWithNoAttention(last_hidden_state=hidden_state, hidden_states=hidden_states)

    def build(self, input_shape=None):
        # 如果已经构建过了，则直接返回
        if self.built:
            return
        # 标记已经构建
        self.built = True
        for stage in self.stages:
            with tf.name_scope(stage.name):
                stage.build(None)


@keras_serializable
# 定义一个 TensorFlow 自定义层，用于主体部分的 RegNet 模型
class TFRegNetMainLayer(tf.keras.layers.Layer):
    # 配置类属性指向 RegNetConfig 类
    config_class = RegNetConfig

    # 初始化函数，接受配置对象以及其他关键字参数
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 存储配置对象
        self.config = config
        # 创建 RegNetEmbeddings 层，用于嵌入输入数据
        self.embedder = TFRegNetEmbeddings(config, name="embedder")
        # 创建 RegNetEncoder 层，用于编码嵌入数据
        self.encoder = TFRegNetEncoder(config, name="encoder")
        # 创建全局平均池化层，用于从编码器输出中生成池化的表示
        self.pooler = tf.keras.layers.GlobalAveragePooling2D(keepdims=True, name="pooler")

    # 调用函数，接受输入张量并返回模型输出
    @unpack_inputs
    def call(
        self,
        pixel_values: tf.Tensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> TFBaseModelOutputWithPoolingAndNoAttention:
        # 如果未提供输出隐藏状态，则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未提供返回字典标志，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入数据嵌入到嵌入层
        embedding_output = self.embedder(pixel_values, training=training)

        # 在编码器层上调用编码函数
        encoder_outputs = self.encoder(
            embedding_output, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training
        )

        # 获取编码器的最后一个隐藏状态和池化输出
        last_hidden_state = encoder_outputs[0]
        pooled_output = self.pooler(last_hidden_state)

        # 将池化输出和最后一个隐藏状态的维度顺序更改为 NCHW
        pooled_output = tf.transpose(pooled_output, perm=(0, 3, 1, 2))
        last_hidden_state = tf.transpose(last_hidden_state, perm=(0, 3, 1, 2))

        # 如果需要输出隐藏状态，则将其维度顺序更改为 NCHW
        if output_hidden_states:
            hidden_states = tuple([tf.transpose(h, perm=(0, 3, 1, 2)) for h in encoder_outputs[1]])

        # 如果不返回字典，则返回元组形式的输出
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 如果返回字典，则返回 TFBaseModelOutputWithPoolingAndNoAttention 对象
        return TFBaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=hidden_states if output_hidden_states else encoder_outputs.hidden_states,
        )

    # 构建函数，用于构建层及其子层的权重
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在 embedder 层，则构建该层
        if getattr(self, "embedder", None) is not None:
            with tf.name_scope(self.embedder.name):
                self.embedder.build(None)
        # 如果存在 encoder 层，则构建该层
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 如果存在 pooler 层，则构建该层
        if getattr(self, "pooler", None) is not None:
            with tf.name_scope(self.pooler.name):
                self.pooler.build((None, None, None, None))


# 定义 TFRegNetPreTrainedModel 类，用于处理权重初始化以及预训练模型的下载和加载
class TFRegNetPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 配置类属性指向 RegNetConfig 类
    config_class = RegNetConfig
    # 基础模型前缀为 "regnet"
    base_model_prefix = "regnet"
    # 主输入名称为 "pixel_values"
    main_input_name = "pixel_values"

    @property
    # 定义一个输入签名，描述了输入张量的形状和数据类型
    def input_signature(self):
        # 返回一个字典，key为输入张量的名称，value为TensorSpec对象，描述了张量的形状和数据类型
        return {"pixel_values": tf.TensorSpec(shape=(None, self.config.num_channels, 224, 224), dtype=tf.float32)}
# 定义了一个原始的 RegNet 模型，该模型在顶部没有特定的头部
# 使用了 @add_start_docstrings 装饰器，将指定的文档字符串添加到模型类的文档字符串之前
# REGNET_START_DOCSTRING 包含了模型的文档字符串，提供了模型的参数信息和一般使用说明
REGNET_START_DOCSTRING = r"""
    This model is a Tensorflow
    [tf.keras.layers.Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer) sub-class. Use it as a
    regular Tensorflow Module and refer to the Tensorflow documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`RegNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义了输入的文档字符串
# 包含了输入参数的描述和用法说明
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

# 使用装饰器 @add_start_docstrings 将指定的文档字符串添加到模型类的文档字符串之前
# 提供了模型输出文档字符串和参数说明
@add_start_docstrings(
    "The bare RegNet model outputting raw features without any specific head on top.",
    REGNET_START_DOCSTRING,
)
# 定义了 TFRegNetModel 类，继承自 TFRegNetPreTrainedModel 类
# TFRegNetModel 是一个 Tensorflow Layer 子类，可以像常规的 Tensorflow 模块一样使用
# 通过继承 TFRegNetPreTrainedModel 类，初始化模型
class TFRegNetModel(TFRegNetPreTrainedModel):
    # 初始化方法，接受一个 RegNetConfig 类型的配置参数
    def __init__(self, config: RegNetConfig, *inputs, **kwargs):
        # 调用父类的初始化方法，传入配置参数及其它参数
        super().__init__(config, *inputs, **kwargs)
        # 初始化 TFRegNetMainLayer 类，传入配置参数和名称
        self.regnet = TFRegNetMainLayer(config, name="regnet")

    # call 方法，定义模型的前向传播逻辑
    # 接受输入参数，包括像素值、是否输出隐藏状态、是否返回字典格式的输出、训练标志等
    # 返回模型输出
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
        # 如果未指定是否返回隐藏状态，则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定是否返回字典格式的输出，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 TFRegNetMainLayer 的前向传播方法，获取输出
        outputs = self.regnet(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 如果不返回字典格式的输出，则返回一个元组
        if not return_dict:
            return (outputs[0],) + outputs[1:]

        # 返回 TFBaseModelOutputWithPoolingAndNoAttention 类型的输出
        return TFBaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=outputs.last_hidden_state,
            pooler_output=outputs.pooler_output,
            hidden_states=outputs.hidden_states,
        )
    # 定义一个用于构建模型的方法，可以指定输入的形状
    def build(self, input_shape=None):
        # 如果模型已经构建过了，则直接返回，不再重复构建
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果模型中有名为"regnet"的属性且不为None
        if getattr(self, "regnet", None) is not None:
            # 在 TensorFlow 中使用指定的名字为"self.regnet.name"的作用域
            with tf.name_scope(self.regnet.name):
                # 对"self.regnet"对象进行构建，参数为None表示输入形状未指定
                self.regnet.build(None)
# 添加起始文档字符串，描述 RegNet 模型及其在顶部的图像分类头部的作用（在池化特征的顶部是一个线性层），例如用于 ImageNet 数据集
@add_start_docstrings(
    """
    RegNet Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    REGNET_START_DOCSTRING,
)

# 定义 TFRegNetForImageClassification 类，继承自 TFRegNetPreTrainedModel 和 TFSequenceClassificationLoss
class TFRegNetForImageClassification(TFRegNetPreTrainedModel, TFSequenceClassificationLoss):
    # 初始化方法
    def __init__(self, config: RegNetConfig, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)
        # 设置标签数量
        self.num_labels = config.num_labels
        # 创建 RegNet 主层对象
        self.regnet = TFRegNetMainLayer(config, name="regnet")
        # 图像分类器
        self.classifier = [
            # 展开层
            tf.keras.layers.Flatten(),
            # 分类器
            tf.keras.layers.Dense(config.num_labels, name="classifier.1") if config.num_labels > 0 else tf.identity,
        ]

    # 调用方法
    @unpack_inputs
    @add_start_docstrings_to_model_forward(REGNET_INPUTS_DOCSTRING)
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
        # 设置隐藏状态输出
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 RegNet 主层对象进行前向传播
        outputs = self.regnet(
            pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training
        )

        # 池化特征输出
        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        # 展平输出
        flattened_output = self.classifier[0](pooled_output)
        # 计算 logits
        logits = self.classifier[1](flattened_output)

        # 计算损失
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        # 如果不返回字典
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TFSequenceClassifierOutput 对象
        return TFSequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states)
    # 定义神经网络的构建方法，参数为输入形状，默认为 None
    def build(self, input_shape=None):
        # 如果神经网络已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 regnet 属性，则构建 regnet 对象
        if getattr(self, "regnet", None) is not None:
            # 在名字空间中构建 regnet
            with tf.name_scope(self.regnet.name):
                self.regnet.build(None)
        # 如果存在 classifier 属性，则构建 classifier 对象的第二个元素
        if getattr(self, "classifier", None) is not None:
            # 在名字空间中构建 classifier 的第二个元素
            with tf.name_scope(self.classifier[1].name):
                # 构建 classifier 的第二个元素，输入形状为 [None, None, None, self.config.hidden_sizes[-1]]
                self.classifier[1].build([None, None, None, self.config.hidden_sizes[-1]])
```