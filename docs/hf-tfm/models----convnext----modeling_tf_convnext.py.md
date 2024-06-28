# `.\models\convnext\modeling_tf_convnext.py`

```
# 设置编码格式为utf-8
# 版权声明
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

# 导入相对当前文件的模块
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling, TFSequenceClassifierOutput
from ...modeling_tf_utils import (
    TFModelInputType,
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    get_initializer,
    keras,
    keras_serializable,
    unpack_inputs,
)
# 导入包级别的模块
from ...tf_utils import shape_list
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
# 从configuration_convnext模块中导入ConvNextConfig类
from .configuration_convnext import ConvNextConfig

# 获取logger对象
logger = logging.get_logger(__name__)

# 设置文档中的配置和检查点
_CONFIG_FOR_DOC = "ConvNextConfig"
_CHECKPOINT_FOR_DOC = "facebook/convnext-tiny-224"

# 定义TFConvNextDropPath类，继承于keras.layers.Layer
class TFConvNextDropPath(keras.layers.Layer):
    # 初始化函数
    def __init__(self, drop_path: float, **kwargs):
        super().__init__(**kwargs)
        self.drop_path = drop_path
    
    # 调用函数
    def call(self, x: tf.Tensor, training=None):
        # 如果处于训练状态
        if training:
            keep_prob = 1 - self.drop_path
            # 获取x的shape
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        # 否则直接返回x
        return x

# 定义TFConvNextEmbeddings类，继承于keras.layers.Layer
class TFConvNextEmbeddings(keras.layers.Layer):
    # 类注释
    """This class is comparable to (and inspired by) the SwinEmbeddings class
    found in src/transformers/models/swin/modeling_swin.py.
    """
    # 初始化方法，接受一个配置对象和其他关键字参数
    def __init__(self, config: ConvNextConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        
        # 创建一个卷积层用于图像块的嵌入表示
        self.patch_embeddings = keras.layers.Conv2D(
            filters=config.hidden_sizes[0],                # 卷积核数目为配置中的隐藏层大小的第一个元素
            kernel_size=config.patch_size,                 # 卷积核大小为配置中的图像块大小
            strides=config.patch_size,                     # 步长为配置中的图像块大小
            name="patch_embeddings",                       # 层名称为patch_embeddings
            kernel_initializer=get_initializer(config.initializer_range),  # 使用配置中的初始化器初始化卷积核
            bias_initializer=keras.initializers.Zeros(),  # 使用0初始化偏置项
        )
        
        # 创建一个 LayerNormalization 层，用于归一化嵌入表示
        self.layernorm = keras.layers.LayerNormalization(epsilon=1e-6, name="layernorm")
        
        # 记录配置中的通道数
        self.num_channels = config.num_channels
        
        # 记录传入的配置对象
        self.config = config

    # 模型调用方法，接受像素值作为输入，进行嵌入表示的计算
    def call(self, pixel_values):
        # 如果像素值是一个字典，取出其中的像素值
        if isinstance(pixel_values, dict):
            pixel_values = pixel_values["pixel_values"]

        # 断言像素值的通道维度与配置中设置的通道数相匹配，用于调试和错误检查
        tf.debugging.assert_equal(
            shape_list(pixel_values)[1],
            self.num_channels,
            message="Make sure that the channel dimension of the pixel values match with the one set in the configuration.",
        )

        # 当在CPU上运行时，`keras.layers.Conv2D` 不支持 `NCHW` 格式。
        # 所以将输入格式从 `NCHW` 转换为 `NHWC`。
        # shape = (batch_size, in_height, in_width, in_channels)
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))

        # 计算图像块的嵌入表示
        embeddings = self.patch_embeddings(pixel_values)
        
        # 对嵌入表示进行归一化处理
        embeddings = self.layernorm(embeddings)
        
        # 返回归一化后的嵌入表示
        return embeddings

    # 构建方法，用于构建模型的各个层次结构
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        
        # 标记模型为已构建状态
        self.built = True
        
        # 如果存在 patch_embeddings 层，则构建其内部结构
        if getattr(self, "patch_embeddings", None) is not None:
            with tf.name_scope(self.patch_embeddings.name):
                self.patch_embeddings.build([None, None, None, self.config.num_channels])
        
        # 如果存在 layernorm 层，则构建其内部结构
        if getattr(self, "layernorm", None) is not None:
            with tf.name_scope(self.layernorm.name):
                self.layernorm.build([None, None, None, self.config.hidden_sizes[0]])
# 定义自定义层 `TFConvNextLayer`，继承自 `keras.layers.Layer` 类。
"""This corresponds to the `Block` class in the original implementation.

There are two equivalent implementations: [DwConv, LayerNorm (channels_first), Conv, GELU,1x1 Conv]; all in (N, C,
H, W) (2) [DwConv, Permute to (N, H, W, C), LayerNorm (channels_last), Linear, GELU, Linear]; Permute back

The authors used (2) as they find it slightly faster in PyTorch. Since we already permuted the inputs to follow
NHWC ordering, we can just apply the operations straight-away without the permutation.
"""

# 初始化方法，接受 `config`、`dim` 和 `drop_path` 作为参数
def __init__(self, config, dim, drop_path=0.0, **kwargs):
    # 调用父类 `keras.layers.Layer` 的初始化方法
    super().__init__(**kwargs)
    # 设置实例变量 `dim` 和 `config`
    self.dim = dim
    self.config = config

    # 定义深度卷积层 `dwconv`
    self.dwconv = keras.layers.Conv2D(
        filters=dim,
        kernel_size=7,
        padding="same",
        groups=dim,
        kernel_initializer=get_initializer(config.initializer_range),
        bias_initializer="zeros",
        name="dwconv",
    )  # depthwise conv

    # 定义层归一化层 `layernorm`
    self.layernorm = keras.layers.LayerNormalization(
        epsilon=1e-6,
        name="layernorm",
    )

    # 定义第一个点卷积层 `pwconv1`
    self.pwconv1 = keras.layers.Dense(
        units=4 * dim,
        kernel_initializer=get_initializer(config.initializer_range),
        bias_initializer="zeros",
        name="pwconv1",
    )  # pointwise/1x1 convs, implemented with linear layers

    # 获取激活函数并设置为实例变量 `act`
    self.act = get_tf_activation(config.hidden_act)

    # 定义第二个点卷积层 `pwconv2`
    self.pwconv2 = keras.layers.Dense(
        units=dim,
        kernel_initializer=get_initializer(config.initializer_range),
        bias_initializer="zeros",
        name="pwconv2",
    )

    # 设置 `drop_path` 为 `TFConvNextDropPath` 层或线性激活函数 `keras.layers.Activation("linear")`
    # 根据 `drop_path` 大于 `0.0` 条件判断
    self.drop_path = (
        TFConvNextDropPath(drop_path, name="drop_path")
        if drop_path > 0.0
        else keras.layers.Activation("linear", name="drop_path")
    )
    def build(self, input_shape: tf.TensorShape = None):
        # PT's `nn.Parameters` must be mapped to a TF layer weight to inherit the same name hierarchy (and vice-versa)
        # 初始化一个可训练的层参数，如果配置的初始化值大于零
        self.layer_scale_parameter = (
            self.add_weight(
                shape=(self.dim,),
                initializer=keras.initializers.Constant(value=self.config.layer_scale_init_value),
                trainable=True,
                name="layer_scale_parameter",
            )
            if self.config.layer_scale_init_value > 0
            else None
        )

        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True

        # 如果存在深度可分离卷积层，则构建该层
        if getattr(self, "dwconv", None) is not None:
            with tf.name_scope(self.dwconv.name):
                self.dwconv.build([None, None, None, self.dim])

        # 如果存在层归一化层，则构建该层
        if getattr(self, "layernorm", None) is not None:
            with tf.name_scope(self.layernorm.name):
                self.layernorm.build([None, None, None, self.dim])

        # 如果存在第一个逐点卷积层，则构建该层
        if getattr(self, "pwconv1", None) is not None:
            with tf.name_scope(self.pwconv1.name):
                self.pwconv1.build([None, None, self.dim])

        # 如果存在第二个逐点卷积层，则构建该层
        if getattr(self, "pwconv2", None) is not None:
            with tf.name_scope(self.pwconv2.name):
                self.pwconv2.build([None, None, 4 * self.dim])

        # 如果存在 drop_path 层，则构建该层
        if getattr(self, "drop_path", None) is not None:
            with tf.name_scope(self.drop_path.name):
                self.drop_path.build(None)

    def call(self, hidden_states, training=False):
        input = hidden_states
        # 应用深度可分离卷积层
        x = self.dwconv(hidden_states)
        # 应用层归一化
        x = self.layernorm(x)
        # 应用第一个逐点卷积层
        x = self.pwconv1(x)
        # 应用激活函数
        x = self.act(x)
        # 应用第二个逐点卷积层
        x = self.pwconv2(x)

        # 如果存在层参数缩放参数，则进行缩放
        if self.layer_scale_parameter is not None:
            x = self.layer_scale_parameter * x

        # 加上原始输入和 drop_path 层的输出
        x = input + self.drop_path(x, training=training)
        return x
class TFConvNextStage(keras.layers.Layer):
    """ConvNext stage, consisting of an optional downsampling layer + multiple residual blocks.

    Args:
        config (`ConvNextV2Config`):
            Model configuration class.
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`):
            Number of output channels.
        depth (`int`):
            Number of residual blocks.
        drop_path_rates(`List[float]`):
            Stochastic depth rates for each layer.
    """

    def __init__(
        self,
        config: ConvNextConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        stride: int = 2,
        depth: int = 2,
        drop_path_rates: Optional[List[float]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # 如果输入通道数不等于输出通道数或者步幅大于1，添加下采样层
        if in_channels != out_channels or stride > 1:
            self.downsampling_layer = [
                keras.layers.LayerNormalization(
                    epsilon=1e-6,
                    name="downsampling_layer.0",
                ),
                # 由于在 `TFConvNextEmbeddings` 层中将输入从 NCHW 转置到 NHWC 格式，
                # 此处输入将按 NHWC 格式处理。从此处到模型输出，所有输出都将保持 NHWC 格式，
                # 直到最后输出时再次转换为 NCHW 格式。
                keras.layers.Conv2D(
                    filters=out_channels,
                    kernel_size=kernel_size,
                    strides=stride,
                    kernel_initializer=get_initializer(config.initializer_range),
                    bias_initializer=keras.initializers.Zeros(),
                    name="downsampling_layer.1",
                ),
            ]
        else:
            # 如果输入通道数等于输出通道数且步幅为1，则为恒等映射
            self.downsampling_layer = [tf.identity]

        # 根据 depth 和 drop_path_rates 创建 TFConvNextLayer 的列表作为网络的主要层
        drop_path_rates = drop_path_rates or [0.0] * depth
        self.layers = [
            TFConvNextLayer(
                config,
                dim=out_channels,
                drop_path=drop_path_rates[j],
                name=f"layers.{j}",
            )
            for j in range(depth)
        ]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def call(self, hidden_states):
        # 执行下采样层
        for layer in self.downsampling_layer:
            hidden_states = layer(hidden_states)
        
        # 执行主要层（残差块）
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        return hidden_states
    # 定义一个方法 `build`，用于构建神经网络层
    def build(self, input_shape=None):
        # 如果已经构建过网络，则直接返回，不重复构建
        if self.built:
            return
        # 设置标志位，表示网络已经构建
        self.built = True
        
        # 如果存在 `layers` 属性，遍历每一层进行构建
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                # 使用每一层的名称为当前层创建命名空间
                with tf.name_scope(layer.name):
                    # 调用每一层的 `build` 方法进行构建，传入 `None` 作为输入形状
                    layer.build(None)
        
        # 如果输入通道数不等于输出通道数或者步长大于 1
        if self.in_channels != self.out_channels or self.stride > 1:
            # 使用第一个下采样层的名称创建命名空间
            with tf.name_scope(self.downsampling_layer[0].name):
                # 调用第一个下采样层的 `build` 方法，传入输入形状 `[None, None, None, self.in_channels]`
                self.downsampling_layer[0].build([None, None, None, self.in_channels])
            # 使用第二个下采样层的名称创建命名空间
            with tf.name_scope(self.downsampling_layer[1].name):
                # 调用第二个下采样层的 `build` 方法，传入输入形状 `[None, None, None, self.in_channels]`
                self.downsampling_layer[1].build([None, None, None, self.in_channels])
# TFConvNextEncoder 类，继承自 keras.layers.Layer
class TFConvNextEncoder(keras.layers.Layer):
    
    # 初始化方法，接受一个 config 对象和额外的关键字参数
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        
        # 初始化阶段列表
        self.stages = []
        
        # 计算并生成一个列表，包含每个阶段的 drop_path_rate
        drop_path_rates = tf.linspace(0.0, config.drop_path_rate, sum(config.depths))
        
        # 将 drop_path_rates 按照 config.depths 切分成多个部分
        drop_path_rates = tf.split(drop_path_rates, config.depths)
        
        # 将每个 Tensor 转换为列表形式
        drop_path_rates = [x.numpy().tolist() for x in drop_path_rates]
        
        # 获取第一个隐藏层的通道数
        prev_chs = config.hidden_sizes[0]
        
        # 根据 num_stages 创建不同的 TFConvNextStage 实例
        for i in range(config.num_stages):
            # 获取当前阶段的输出通道数
            out_chs = config.hidden_sizes[i]
            
            # 创建 TFConvNextStage 实例并添加到 stages 列表中
            stage = TFConvNextStage(
                config,
                in_channels=prev_chs,
                out_channels=out_chs,
                stride=2 if i > 0 else 1,
                depth=config.depths[i],
                drop_path_rates=drop_path_rates[i],
                name=f"stages.{i}",
            )
            self.stages.append(stage)
            prev_chs = out_chs

    # call 方法，用于执行正向传播
    def call(self, hidden_states, output_hidden_states=False, return_dict=True):
        # 如果要输出所有隐藏状态，则初始化一个空元组
        all_hidden_states = () if output_hidden_states else None

        # 遍历每个阶段并执行正向传播
        for i, layer_module in enumerate(self.stages):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 执行当前阶段的正向传播
            hidden_states = layer_module(hidden_states)

        # 如果要输出所有隐藏状态，则添加最终的隐藏状态
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果 return_dict 为 False，则返回非空的结果元组
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        # 返回 TFBaseModelOutput 实例，包含最终隐藏状态和所有隐藏状态
        return TFBaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states)

    # build 方法，用于构建模型的各个阶段
    def build(self, input_shape=None):
        # 遍历每个阶段并在命名作用域内构建阶段
        for stage in self.stages:
            with tf.name_scope(stage.name):
                stage.build(None)



# TFConvNextMainLayer 类，继承自 keras.layers.Layer，并使用 keras_serializable 装饰器
@keras_serializable
class TFConvNextMainLayer(keras.layers.Layer):
    
    # 配置类属性，指定为 ConvNextConfig
    config_class = ConvNextConfig

    # 初始化方法，接受一个 config 对象和额外的关键字参数
    def __init__(self, config: ConvNextConfig, add_pooling_layer: bool = True, **kwargs):
        super().__init__(**kwargs)

        # 初始化 config 属性
        self.config = config
        
        # 创建 TFConvNextEmbeddings 实例，并命名为 "embeddings"
        self.embeddings = TFConvNextEmbeddings(config, name="embeddings")
        
        # 创建 TFConvNextEncoder 实例，并命名为 "encoder"
        self.encoder = TFConvNextEncoder(config, name="encoder")
        
        # 创建 LayerNormalization 层，并设置 epsilon 参数
        self.layernorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm")
        
        # 如果 add_pooling_layer 为 True，则创建 GlobalAvgPool2D 层，并设置 data_format
        self.pooler = keras.layers.GlobalAvgPool2D(data_format="channels_first") if add_pooling_layer else None

    # call 方法，用于执行正向传播
    @unpack_inputs
    def call(
        self,
        pixel_values: TFModelInputType | None = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        # 输入参数解包装饰器，用于接收多个输入参数
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 使用嵌入层将像素值转换为嵌入表示，根据训练状态进行操作
        embedding_output = self.embeddings(pixel_values, training=training)

        # 使用编码器处理嵌入表示，可以选择是否输出隐藏状态
        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 获取编码器的最后隐藏状态，并将通道维度移到第二个位置，确保一致性
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = tf.transpose(last_hidden_state, perm=(0, 3, 1, 2))
        
        # 通过池化器和层归一化器生成池化后的输出
        pooled_output = self.layernorm(self.pooler(last_hidden_state))

        # 如果需要输出所有隐藏状态，则将它们的通道维度移到正确的位置
        if output_hidden_states:
            hidden_states = tuple([tf.transpose(h, perm=(0, 3, 1, 2)) for h in encoder_outputs[1]])

        # 根据是否返回字典，构造返回的模型输出
        if not return_dict:
            hidden_states = hidden_states if output_hidden_states else ()
            return (last_hidden_state, pooled_output) + hidden_states

        # 如果返回字典，则使用特定格式返回模型输出
        return TFBaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=hidden_states if output_hidden_states else encoder_outputs.hidden_states,
        )

    # 构建模型的方法，初始化嵌入层、编码器和层归一化器
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        if getattr(self, "layernorm", None) is not None:
            with tf.name_scope(self.layernorm.name):
                self.layernorm.build([None, self.config.hidden_sizes[-1]])
    """
    CONVNEXT_INPUTS_DOCSTRING = r"""
    A class attribute containing a raw string that serves as documentation for the expected input formats
    of models and layers in the ConvNext model.

    This string outlines the two supported input formats for TensorFlow models and layers in the `transformers`
    library:
    - Inputs can be provided as keyword arguments, akin to PyTorch models.
    - Alternatively, inputs can be passed in a list, tuple, or dictionary within the first positional argument.

    The support for the second format is particularly beneficial for seamless integration with Keras methods like
    `model.fit()`. When using Keras methods, inputs and labels can be passed in any format supported by `model.fit()`.

    For cases outside Keras methods, such as custom layers or models using the Keras Functional API, three approaches
    are recommended for gathering input Tensors:
    - A single Tensor containing `pixel_values` exclusively: `model(pixel_values)`
    - A list of varying length containing input Tensors in specified order: `model([pixel_values, attention_mask])`
      or `model([pixel_values, attention_mask, token_type_ids])`
    - A dictionary associating input Tensors with their respective names: `model({"pixel_values": pixel_values,
      "token_type_ids": token_type_ids})`

    For users creating models and layers through subclassing, typical Python function input practices apply.

    This documentation aims to clarify the input expectations and usage guidelines for ConvNext models and layers.

    """
    """
    Args:
        pixel_values (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` ``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`):
            # `pixel_values`参数可以是`np.ndarray`、`tf.Tensor`、`List[tf.Tensor]`、`Dict[str, tf.Tensor]`或`Dict[str, np.ndarray]`类型的数据，每个示例必须具有形状为`(batch_size, num_channels, height, width)`的结构。

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。详见返回张量中的`hidden_states`。此参数仅在即时执行模式下有效，在图模式下将使用配置中的值。

        return_dict (`bool`, *optional*):
            # 是否返回[`~utils.ModelOutput`]而不是普通元组。此参数仅在即时执行模式下有效，在图模式下将始终设置为True。
"""
@add_start_docstrings(
    "The bare ConvNext model outputting raw features without any specific head on top.",
    CONVNEXT_START_DOCSTRING,
)
class TFConvNextModel(TFConvNextPreTrainedModel):
    def __init__(self, config, *inputs, add_pooling_layer=True, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 初始化 ConvNext 主层，配置是否添加池化层
        self.convnext = TFConvNextMainLayer(config, add_pooling_layer=add_pooling_layer, name="convnext")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CONVNEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        pixel_values: TFModelInputType | None = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        r"""
        Returns:
            Depending on `return_dict`, either returns:
                - TFBaseModelOutputWithPooling (if `return_dict=True`)
                - Tuple[tf.Tensor] (if `return_dict=False`)

        Examples:
            示例用法代码块，演示如何使用 TFConvNextModel 进行推理。

        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 调用 ConvNext 主层进行前向传播
        outputs = self.convnext(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        if not return_dict:
            # 如果 return_dict=False，返回元组形式的输出
            return (outputs[0],) + outputs[1:]

        # 如果 return_dict=True，返回 TFBaseModelOutputWithPooling 对象
        return TFBaseModelOutputWithPooling(
            last_hidden_state=outputs.last_hidden_state,
            pooler_output=outputs.pooler_output,
            hidden_states=outputs.hidden_states,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "convnext", None) is not None:
            with tf.name_scope(self.convnext.name):
                # 构建 ConvNext 主层
                self.convnext.build(None)


@add_start_docstrings(
    """
    ConvNext Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    CONVNEXT_START_DOCSTRING,
)
"""
        # TFConvNextForImageClassification 类的构造函数，继承自 TFConvNextPreTrainedModel 和 TFSequenceClassificationLoss
        def __init__(self, config: ConvNextConfig, *inputs, **kwargs):
            # 调用父类 TFConvNextPreTrainedModel 的构造函数
            super().__init__(config, *inputs, **kwargs)

            # 设置分类任务的标签数目
            self.num_labels = config.num_labels
            # 创建 TFConvNextMainLayer 的实例作为特征提取器
            self.convnext = TFConvNextMainLayer(config, name="convnext")

            # 分类器部分
            # 创建一个全连接层作为分类器，输出单元数为 config.num_labels
            self.classifier = keras.layers.Dense(
                units=config.num_labels,
                kernel_initializer=get_initializer(config.initializer_range),
                bias_initializer="zeros",
                name="classifier",
            )
            # 保存配置信息
            self.config = config

        @unpack_inputs
        @add_start_docstrings_to_model_forward(CONVNEXT_INPUTS_DOCSTRING)
        @replace_return_docstrings(output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
        # 定义模型的前向传播方法，接收多种输入参数并返回模型输出
        def call(
            self,
            pixel_values: TFModelInputType | None = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            labels: np.ndarray | tf.Tensor | None = None,
            training: Optional[bool] = False,
    ) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:
            Returns either TFSequenceClassifierOutput or a tuple of tf.Tensor:
                TFSequenceClassifierOutput: Output containing loss, logits, and hidden states.
                Tuple[tf.Tensor]: Tuple of logits and additional hidden states.

        Examples:
            Example usage demonstrating image processing and classification using Transformers and TensorFlow.

            ```python
            >>> from transformers import AutoImageProcessor, TFConvNextForImageClassification
            >>> import tensorflow as tf
            >>> from PIL import Image
            >>> import requests

            >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            >>> image = Image.open(requests.get(url, stream=True).raw)

            >>> image_processor = AutoImageProcessor.from_pretrained("facebook/convnext-tiny-224")
            >>> model = TFConvNextForImageClassification.from_pretrained("facebook/convnext-tiny-224")

            >>> inputs = image_processor(images=image, return_tensors="tf")
            >>> outputs = model(**inputs)
            >>> logits = outputs.logits
            >>> # model predicts one of the 1000 ImageNet classes
            >>> predicted_class_idx = tf.math.argmax(logits, axis=-1)[0]
            >>> print("Predicted class:", model.config.id2label[int(predicted_class_idx)])
            ```

        """
        # Determine whether to include hidden states in the output
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # Determine whether to use a return dictionary for the output
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Check if pixel_values are provided, raise an error if not
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Pass pixel_values through the ConvNext model
        outputs = self.convnext(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # Determine the pooled output based on return_dict flag
        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        # Calculate logits using the classifier network
        logits = self.classifier(pooled_output)

        # Compute loss if labels are provided using the helper function hf_compute_loss
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        # If return_dict is False, format the output as a tuple
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # Return TFSequenceClassifierOutput if return_dict is True
        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )

    def build(self, input_shape=None):
        # Check if the model is already built, if so, return immediately
        if self.built:
            return
        # Set built flag to True indicating the model is being built
        self.built = True

        # Build the ConvNext model if it exists
        if getattr(self, "convnext", None) is not None:
            with tf.name_scope(self.convnext.name):
                self.convnext.build(None)

        # Build the classifier model if it exists
        if getattr(self, "classifier", None) is not None:
            if hasattr(self.classifier, "name"):
                with tf.name_scope(self.classifier.name):
                    self.classifier.build([None, None, self.config.hidden_sizes[-1]])
```