# `.\transformers\models\resnet\modeling_flax_resnet.py`

```
# 导入所需模块和类
from functools import partial
from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict

# 导入自定义的输出类
from ...modeling_flax_outputs import (
    FlaxBaseModelOutputWithNoAttention,
    FlaxBaseModelOutputWithPoolingAndNoAttention,
    FlaxImageClassifierOutputWithNoAttention,
)

# 导入自定义的工具类和方法
from ...modeling_flax_utils import (
    ACT2FN,
    FlaxPreTrainedModel,
    append_replace_return_docstrings,
    overwrite_call_docstring,
)

# 导入自定义的配置类
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_resnet import ResNetConfig

# 定义一个常量，包含了一段字符串文档，用于描述该模型的基本信息和特性
RESNET_START_DOCSTRING = r"""
...
"""

# 开始定义主要模型类，并继承了FlaxPreTrainedModel和一些其他模块类
class FlaxResNetModule(nn.Module):
    ...
    Parameters:
        # 参数说明
        config ([`ResNetConfig`]): Model configuration class with all the parameters of the model.
            # 使用`ResNetConfig`类作为模型配置类，包含模型的所有参数。
            # 通过配置文件初始化不会加载与模型关联的权重，只加载配置。
            # 可以查看 [`~FlaxPreTrainedModel.from_pretrained`] 方法以加载模型权重。
        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
            # 数据类型用于计算。可以是`jax.numpy.float32`、`jax.numpy.float16`(在GPU上)和`jax.numpy.bfloat16`(在TPU上)之一。
            # 可用于启用在GPU或TPU上进行混合精度训练或半精度推断。如果指定，所有计算将使用给定的`dtype`进行。
            **请注意，这仅指定计算的数据类型，不会影响模型参数的数据类型。**
            如果您希望更改模型参数的数据类型，请参阅 [`~FlaxPreTrainedModel.to_fp16`] 和 [`~FlaxPreTrainedModel.to_bf16`]。
```  
"""
# ResNet 模型的输入参数文档字符串，包括输入张量的形状及额外参数
RESNET_INPUTS_DOCSTRING = r"""
    Args:
        # 像素值，类型为 JAX 数组，形状为 (批次大小，通道数，高度，宽度)
        pixel_values (`jax.numpy.float32` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`AutoImageProcessor.__call__`] for details.
        # 是否返回所有层的隐藏状态
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        # 是否返回 ModelOutput 对象而非普通元组
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# 定义一个简单的身份函数模块，直接返回输入
class Identity(nn.Module):
    """Identity function."""

    @nn.compact
    def __call__(self, x, **kwargs):
        # 返回输入的值
        return x


# 定义一个 ResNet 的卷积层模块
class FlaxResNetConvLayer(nn.Module):
    # 卷积输出通道数
    out_channels: int
    # 卷积核尺寸，默认为 3
    kernel_size: int = 3
    # 卷积的步长，默认为 1
    stride: int = 1
    # 激活函数，默认为 ReLU
    activation: Optional[str] = "relu"
    # 数据类型，默认为 32 位浮点
    dtype: jnp.dtype = jnp.float32

    # 设置函数，用于初始化模块
    def setup(self):
        # 创建卷积操作对象
        self.convolution = nn.Conv(
            self.out_channels,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=self.stride,
            padding=self.kernel_size // 2,
            dtype=self.dtype,
            use_bias=False,
            # 使用方差缩放初始化卷积核
            kernel_init=nn.initializers.variance_scaling(2.0, mode="fan_out", distribution="normal", dtype=self.dtype),
        )
        # 创建批归一化操作对象
        self.normalization = nn.BatchNorm(momentum=0.9, epsilon=1e-05, dtype=self.dtype)
        # 根据激活函数名称选择对应的激活函数对象，如果没有激活函数，使用 Identity
        self.activation_func = ACT2FN[self.activation] if self.activation is not None else Identity()

    # 通过模块时的调用函数
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # 应用卷积操作
        hidden_state = self.convolution(x)
        # 应用批归一化操作
        hidden_state = self.normalization(hidden_state, use_running_average=deterministic)
        # 应用激活函数
        hidden_state = self.activation_func(hidden_state)
        # 返回经过处理的隐藏状态
        return hidden_state


# 定义 ResNet 的嵌入层，主要是用于处理输入图像
class FlaxResNetEmbeddings(nn.Module):
    """
    ResNet 嵌入层，包含一个主要的卷积操作。
    """

    # ResNet 的配置
    config: ResNetConfig
    # 数据类型，默认为 32 位浮点
    dtype: jnp.dtype = jnp.float32

    # 设置函数，用于初始化模块
    def setup(self):
        # 使用配置的嵌入大小、卷积核尺寸、步长和激活函数来创建嵌入层
        self.embedder = FlaxResNetConvLayer(
            self.config.embedding_size,
            kernel_size=7,
            stride=2,
            activation=self.config.hidden_act,
            dtype=self.dtype,
        )

        # 创建最大池化操作，部分用于减少特征图的尺寸
        self.max_pool = partial(nn.max_pool, window_shape=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))

    # 通过模块时的调用函数
    def __call__(self, pixel_values: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # 获取输入张量的通道数
        num_channels = pixel_values.shape[-1]
        # 检查输入张量的通道数是否与配置中的通道数一致，如果不一致则抛出异常
        if num_channels != self.config.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # 应用卷积嵌入层
        embedding = self.embedder(pixel_values, deterministic=deterministic)
        # 应用最大池化操作
        embedding = self.max_pool(embedding)
        # 返回嵌入结果
        return embedding


# 定义 ResNet 的短连接模块，通常用于残差连接
class FlaxResNetShortCut(nn.Module):
    """
    ResNet shortcut, used to project the residual features to the correct size. If needed, it is also used to
    downsample the input using `stride=2`.
    """
    
    # 定义ResNet中的shortcut，用于将残差特征投影到正确的大小。如果需要，也可以使用`stride=2`来下采样输入。

    out_channels: int
    # 输出通道数量

    stride: int = 2
    # 步幅，默认为2，用于下采样输入

    dtype: jnp.dtype = jnp.float32
    # 数据类型，默认为jnp.float32

    def setup(self):
        # 初始化层，在该函数中初始化Conv和BatchNorm层

        self.convolution = nn.Conv(
            self.out_channels,
            kernel_size=(1, 1),
            strides=self.stride,
            use_bias=False,
            kernel_init=nn.initializers.variance_scaling(2.0, mode="fan_out", distribution="truncated_normal"),
            dtype=self.dtype,
        )
        # 创建卷积层，设置卷积核大小为(1, 1)，步幅为self.stride，不使用偏置，使用自定义的核初始化方法

        self.normalization = nn.BatchNorm(momentum=0.9, epsilon=1e-05, dtype=self.dtype)
        # 创建批量归一化层，设置动量为0.9，epsilon值为1e-05，数据类型为self.dtype

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # 定义调用函数，传入输入x和是否使用固定随机数种子deterministic

        hidden_state = self.convolution(x)
        # 使用Conv层处理输入x，得到隐藏状态

        hidden_state = self.normalization(hidden_state, use_running_average=deterministic)
        # 使用BatchNorm层对隐藏状态进行归一化，根据deterministic参数决定是否使用运行时的平均值

        return hidden_state
        # 返回处理后的隐藏状态
class FlaxResNetBasicLayerCollection(nn.Module):
    out_channels: int  # 输出通道数
    stride: int = 1  # 步幅，默认为1
    dtype: jnp.dtype = jnp.float32  # 数据类型，默认为 jnp.float32

    def setup(self):
        self.layer = [
            FlaxResNetConvLayer(self.out_channels, stride=self.stride, dtype=self.dtype),  # 创建卷积层对象
            FlaxResNetConvLayer(self.out_channels, activation=None, dtype=self.dtype),  # 创建卷积层对象，激活函数为 None
        ]

    def __call__(self, hidden_state: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        for layer in self.layer:  # 遍历卷积层
            hidden_state = layer(hidden_state, deterministic=deterministic)  # 应用卷积层到隐藏状态
        return hidden_state  # 返回隐藏状态


class FlaxResNetBasicLayer(nn.Module):
    """
    A classic ResNet's residual layer composed by two `3x3` convolutions.
    """

    in_channels: int  # 输入通道数
    out_channels: int  # 输出通道数
    stride: int = 1  # 步幅，默认为1
    activation: Optional[str] = "relu"  # 激活函数，默认为"relu"
    dtype: jnp.dtype = jnp.float32  # 数据类型，默认为 jnp.float32

    def setup(self):
        should_apply_shortcut = self.in_channels != self.out_channels or self.stride != 1  # 判断是否需要添加快捷连接
        self.shortcut = (
            FlaxResNetShortCut(self.out_channels, stride=self.stride, dtype=self.dtype)  # 创建快捷连接层对象
            if should_apply_shortcut
            else None
        )
        self.layer = FlaxResNetBasicLayerCollection(
            out_channels=self.out_channels,
            stride=self.stride,
            dtype=self.dtype,
        )
        self.activation_func = ACT2FN[self.activation]  # 获取激活函数对应的函数对象

    def __call__(self, hidden_state, deterministic: bool = True):
        residual = hidden_state  # 记录残差连接部分
        hidden_state = self.layer(hidden_state, deterministic=deterministic)  # 应用卷积层到隐藏状态

        if self.shortcut is not None:  # 如果存在快捷连接
            residual = self.shortcut(residual, deterministic=deterministic)  # 应用快捷连接到残差部分
        hidden_state += residual  # 加上残差连接的结果

        hidden_state = self.activation_func(hidden_state)  # 应用激活函数到隐藏状态
        return hidden_state  # 返回隐藏状态


class FlaxResNetBottleNeckLayerCollection(nn.Module):
    out_channels: int  # 输出通道数
    stride: int = 1  # 步幅，默认为1
    activation: Optional[str] = "relu"  # 激活函数，默认为"relu"
    reduction: int = 4  # 缩小因子，默认为4
    dtype: jnp.dtype = jnp.float32  # 数据类型，默认为 jnp.float32

    def setup(self):
        reduces_channels = self.out_channels // self.reduction  # 计算减少的通道数

        self.layer = [
            FlaxResNetConvLayer(reduces_channels, kernel_size=1, dtype=self.dtype, name="0"),  # 创建卷积层对象
            FlaxResNetConvLayer(reduces_channels, stride=self.stride, dtype=self.dtype, name="1"),  # 创建卷积层对象
            FlaxResNetConvLayer(self.out_channels, kernel_size=1, activation=None, dtype=self.dtype, name="2"),  # 创建卷积层对象
        ]

    def __call__(self, hidden_state: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        for layer in self.layer:  # 遍历卷积层
            hidden_state = layer(hidden_state, deterministic=deterministic)  # 应用卷积层到隐藏状态
        return hidden_state  # 返回隐藏状态


class FlaxResNetBottleNeckLayer(nn.Module):
    """
    A classic ResNet's bottleneck layer composed by three `3x3` convolutions. The first `1x1` convolution reduces the
    input by a factor of `reduction` in order to make the second `3x3` convolution faster. The last `1x1` convolution
    remaps the reduced features to `out_channels`.
    """
    in_channels: int                               # 输入通道数的变量
    out_channels: int                              # 输出通道数的变量
    stride: int = 1                                # 步长，默认为1
    activation: Optional[str] = "relu"             # 激活函数，默认为ReLU
    reduction: int = 4                             # 通道压缩的因子，默认为4
    dtype: jnp.dtype = jnp.float32                  # 数据类型，默认为float32

    def setup(self):
        should_apply_shortcut = self.in_channels != self.out_channels or self.stride != 1   # 判断是否应用shortcut
        self.shortcut = (
            FlaxResNetShortCut(self.out_channels, stride=self.stride, dtype=self.dtype)      # 创建shortcut对象
            if should_apply_shortcut
            else None
        )

        self.layer = FlaxResNetBottleNeckLayerCollection(
            self.out_channels,
            stride=self.stride,
            activation=self.activation,
            reduction=self.reduction,
            dtype=self.dtype,
        )                                                                                   # 创建ResNet的瓶颈层

        self.activation_func = ACT2FN[self.activation]                                     
                                                                    # 根据激活函数名称从ACT2FN字典中获取对应的激活函数

    def __call__(self, hidden_state: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
                                                                    # 定义__call__方法，可以像函数一样调用对象
        residual = hidden_state                                         # 将输入隐藏状态保存到残差变量中

        if self.shortcut is not None:
            residual = self.shortcut(residual, deterministic=deterministic)      # 若shortcut对象存在，则通过shortcut处理残差
        hidden_state = self.layer(hidden_state, deterministic)
                                                              # 将隐藏状态通过ResNet的瓶颈层处理
        hidden_state += residual                                      # 将处理后的隐藏状态与残差相加
        hidden_state = self.activation_func(hidden_state)                      # 使用激活函数处理隐藏状态
        return hidden_state                                                  # 返回处理后的隐藏状态
class FlaxResNetStageLayersCollection(nn.Module):
    """
    A ResNet stage composed by stacked layers.
    """

    config: ResNetConfig  # 保存ResNet的配置信息
    in_channels: int  # 输入通道数
    out_channels: int  # 输出通道数
    stride: int = 2  # 步幅，默认为2
    depth: int = 2  # 层深度，默认为2
    dtype: jnp.dtype = jnp.float32  # 数据类型，默认为32位浮点数

    def setup(self):
        # 根据配置选择使用瓶颈块或基本块
        layer = FlaxResNetBottleNeckLayer if self.config.layer_type == "bottleneck" else FlaxResNetBasicLayer

        layers = [
            # 在第一层中进行下采样，步幅为2
            layer(
                self.in_channels,
                self.out_channels,
                stride=self.stride,
                activation=self.config.hidden_act,
                dtype=self.dtype,
                name="0",
            ),
        ]

        for i in range(self.depth - 1):
            layers.append(
                layer(
                    self.out_channels,
                    self.out_channels,
                    activation=self.config.hidden_act,
                    dtype=self.dtype,
                    name=str(i + 1),
                )
            )

        self.layers = layers

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        hidden_state = x
        for layer in self.layers:
            # 通过堆叠层来处理输入数据
            hidden_state = layer(hidden_state, deterministic=deterministic)
        return hidden_state


class FlaxResNetStage(nn.Module):
    """
    A ResNet stage composed by stacked layers.
    """

    config: ResNetConfig  # 保存ResNet的配置信息
    in_channels: int  # 输入通道数
    out_channels: int  # 输出通道数
    stride: int = 2  # 步幅，默认为2
    depth: int = 2  # 层深度，默认为2
    dtype: jnp.dtype = jnp.float32  # 数据类型，默认为32位浮点数

    def setup(self):
        self.layers = FlaxResNetStageLayersCollection(
            self.config,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            stride=self.stride,
            depth=self.depth,
            dtype=self.dtype,
        )

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # 调用内部的层来处理输入数据
        return self.layers(x, deterministic=deterministic)


class FlaxResNetStageCollection(nn.Module):
    config: ResNetConfig  # 保存ResNet的配置信息
    dtype: jnp.dtype = jnp.float32  # 数据类型，默认为32位浮点数

    def setup(self):
        # 获取每个阶段输入输出通道的组合
        in_out_channels = zip(self.config.hidden_sizes, self.config.hidden_sizes[1:])
        stages = [
            FlaxResNetStage(
                self.config,
                self.config.embedding_size,
                self.config.hidden_sizes[0],
                stride=2 if self.config.downsample_in_first_stage else 1,
                depth=self.config.depths[0],
                dtype=self.dtype,
                name="0",
            )
        ]

        for i, ((in_channels, out_channels), depth) in enumerate(zip(in_out_channels, self.config.depths[1:])):
            stages.append(
                FlaxResNetStage(self.config, in_channels, out_channels, depth=depth, dtype=self.dtype, name=str(i + 1))
            )

        self.stages = stages
    # 定义 __call__ 方法，用于模型调用
    def __call__(
        self,
        hidden_state: jnp.ndarray,  # 输入隐藏状态，为 JAX 数组
        output_hidden_states: bool = False,  # 是否输出隐藏状态，默认为 False
        deterministic: bool = True,  # 是否使用确定性计算，默认为 True
    ) -> FlaxBaseModelOutputWithNoAttention:  # 返回值为不包含注意力的 FlaxBaseModelOutput 类型
    
        # 如果输出隐藏状态为 True，则初始化隐藏状态为空元组；否则隐藏状态为 None
        hidden_states = () if output_hidden_states else None
    
        # 遍历模型的各个阶段
        for stage_module in self.stages:
            # 如果输出隐藏状态为 True，则将隐藏状态转置后添加到隐藏状态元组中
            if output_hidden_states:
                hidden_states = hidden_states + (hidden_state.transpose(0, 3, 1, 2),)
    
            # 对当前阶段模块进行计算，更新隐藏状态
            hidden_state = stage_module(hidden_state, deterministic=deterministic)
    
        # 返回更新后的隐藏状态以及隐藏状态元组（如果需要输出隐藏状态）
        return hidden_state, hidden_states
class FlaxResNetEncoder(nn.Module):
    config: ResNetConfig  # 定义 config 属性为 ResNetConfig 类型
    dtype: jnp.dtype = jnp.float32  # 定义 dtype 属性为 jnp.float32 类型，默认为 jnp.float32

    def setup(self):  # 定义 setup 方法
        self.stages = FlaxResNetStageCollection(self.config, dtype=self.dtype)  # 初始化 self.stages 为 FlaxResNetStageCollection 对象

    def __call__(  # 定义 __call__ 方法，接收 hidden_state、output_hidden_states、return_dict 和 deterministic 四个参数
        self,
        hidden_state: jnp.ndarray,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
    ) -> FlaxBaseModelOutputWithNoAttention:  # 返回类型为 FlaxBaseModelOutputWithNoAttention

        hidden_state, hidden_states = self.stages(  # 调用 self.stages，将 hidden_state 传入，获取返回的 hidden_state 和 hidden_states
            hidden_state, output_hidden_states=output_hidden_states, deterministic=deterministic
        )

        if output_hidden_states:  # 如果 output_hidden_states 为 True
            hidden_states = hidden_states + (hidden_state.transpose(0, 3, 1, 2),)  # 对 hidden_states 进行转置并与原 hidden_states 相加

        if not return_dict:  # 如果 return_dict 为 False
            return tuple(v for v in [hidden_state, hidden_states] if v is not None)  # 返回包含非空元素的元组

        return FlaxBaseModelOutputWithNoAttention(  # 返回 FlaxBaseModelOutputWithNoAttention 对象
            last_hidden_state=hidden_state,  # 最终的隐藏状态为 hidden_state
            hidden_states=hidden_states,  # 所有隐藏状态为 hidden_states
        )


class FlaxResNetPreTrainedModel(FlaxPreTrainedModel):  # 定义 FlaxResNetPreTrainedModel 类，继承自 FlaxPreTrainedModel
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ResNetConfig  # 设置 config_class 为 ResNetConfig 类型
    base_model_prefix = "resnet"  # 设置 base_model_prefix 为 "resnet"
    main_input_name = "pixel_values"  # 设置 main_input_name 为 "pixel_values"
    module_class: nn.Module = None  # 定义 module_class 属性为 nn.Module 类型，默认为 None

    def __init__(  # 定义 __init__ 方法，接收 config、input_shape、seed、dtype、_do_init 和 **kwargs 参数
        self,
        config: ResNetConfig,
        input_shape=(1, 224, 224, 3),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)  # 使用 module_class 创建 module 对象
        if input_shape is None:  # 如果 input_shape 为 None
            input_shape = (1, config.image_size, config.image_size, config.num_channels)  # 设置默认的 input_shape
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)  # 调用父类的 __init__

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量
        pixel_values = jnp.zeros(input_shape, dtype=self.dtype)

        rngs = {"params": rng}  # 设置随机数种子

        random_params = self.module.init(rngs, pixel_values, return_dict=False)  # 使用 module 的 init 方法初始化参数

        if params is not None:  # 如果传入了 params
            random_params = flatten_dict(unfreeze(random_params))  # 将 random_params 展平
            params = flatten_dict(unfreeze(params))  # 将 params 展平
            for missing_key in self._missing_keys:  # 遍历缺失的键
                params[missing_key] = random_params[missing_key]  # 将缺失的键从 random_params 复制到 params
            self._missing_keys = set()  # 清空缺失的键集合
            return freeze(unflatten_dict(params))  # 返回重新冻结的 params
        else:
            return random_params  # 返回随机初始化的参数

    @add_start_docstrings_to_model_forward(RESNET_INPUTS_DOCSTRING)
    def __call__(
        self,
        pixel_values,
        params: dict = None,
        train: bool = False,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 将像素值维度转换为(0, 2, 3, 1)
        pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))

        # 处理任何需要的 PRNG
        rngs = {}

        # 调用模块的 apply 方法，传入参数字典和其他参数
        return self.module.apply(
            {
                "params": params["params"] if params is not None else self.params["params"],
                "batch_stats": params["batch_stats"] if params is not None else self.params["batch_stats"],
            },
            jnp.array(pixel_values, dtype=jnp.float32),
            not train,  # 是否为训练模式，与 train 取反
            output_hidden_states,
            return_dict,
            rngs=rngs,
            mutable=["batch_stats"] if train else False,  # 当 train 为 True 时，将批次统计信息放在返回的元组中
        )
# 定义了一个 FlaxResNetModule 类，继承自 nn.Module
class FlaxResNetModule(nn.Module):
    # 指定 config 属性为 ResNetConfig 类型
    config: ResNetConfig
    # 指定 dtype 属性为 jnp.float32 类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    # 模块设置方法
    def setup(self):
        # 初始化 embedder 属性为 FlaxResNetEmbeddings 类实例
        self.embedder = FlaxResNetEmbeddings(self.config, dtype=self.dtype)
        # 初始化 encoder 属性为 FlaxResNetEncoder 类实例
        self.encoder = FlaxResNetEncoder(self.config, dtype=self.dtype)

        # 定义了一个平均池化函数，用于 resnet 中的自适应平均池化
        self.pooler = partial(
            nn.avg_pool,
            padding=((0, 0), (0, 0)),
        )

    # 对象调用方法，用于执行前向传播
    def __call__(
        self,
        pixel_values,
        deterministic: bool = True,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> FlaxBaseModelOutputWithPoolingAndNoAttention:
        # 如果 output_hidden_states 为 None，则使用 config 中的设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果 return_dict 为 None，则使用 config 中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 对输入进行嵌入
        embedding_output = self.embedder(pixel_values, deterministic=deterministic)

        # 使用 encoder 对嵌入向量进行编码
        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        # 获取编码器输出的最后隐藏状态
        last_hidden_state = encoder_outputs[0]

        # 对最后隐藏状态进行池化，将其转置并进行形状调整
        pooled_output = self.pooler(
            last_hidden_state,
            window_shape=(last_hidden_state.shape[1], last_hidden_state.shape[2]),
            strides=(last_hidden_state.shape[1], last_hidden_state.shape[2]),
        ).transpose(0, 3, 1, 2)

        # 将最后隐藏状态进行转置并进行形状调整
        last_hidden_state = last_hidden_state.transpose(0, 3, 1, 2)

        # 如果 return_dict 为 False，则返回元组形式的结果
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 如果 return_dict 为 True，则返回包含池化输出和隐藏状态的 FlaxBaseModelOutputWithPoolingAndNoAttention 对象
        return FlaxBaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )


# 添加模型文档注释
@add_start_docstrings(
    "The bare ResNet model outputting raw features without any specific head on top.",
    RESNET_START_DOCSTRING,
)
class FlaxResNetModel(FlaxResNetPreTrainedModel):
    # 模型类别为 FlaxResNetModule
    module_class = FlaxResNetModule


# 定义了 FLAX_VISION_MODEL_DOCSTRING 字符串，包含模型的返回值和示例
FLAX_VISION_MODEL_DOCSTRING = """
    Returns:

    Examples:

    ```python
    >>> from transformers import AutoImageProcessor, FlaxResNetModel
    >>> from PIL import Image
    >>> import requests

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)
    >>> image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    >>> model = FlaxResNetModel.from_pretrained("microsoft/resnet-50")
    >>> inputs = image_processor(images=image, return_tensors="np")
    >>> outputs = model(**inputs)
    >>> last_hidden_states = outputs.last_hidden_state
    ```
"""

# 重写 FlaxResNetModel 类的调用方法的文档注释
overwrite_call_docstring(FlaxResNetModel, FLAX_VISION_MODEL_DOCSTRING)
# 追加或替换返回值的文档注释
append_replace_return_docstrings(
    # 导入FlaxResNetModel模块，并指定输出类型为FlaxBaseModelOutputWithPoolingAndNoAttention，使用ResNetConfig作为配置类
    FlaxResNetModel, output_type=FlaxBaseModelOutputWithPoolingAndNoAttention, config_class=ResNetConfig
# 定义一个用于图像分类的 FlaxResNetClassifierCollection 类
class FlaxResNetClassifierCollection(nn.Module):
    # 从配置中获取分类器的相关参数
    config: ResNetConfig
    # 设置数据类型为 float32
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 创建一个全连接层作为分类器，输出维度为配置中指定的类别数量
        self.classifier = nn.Dense(self.config.num_labels, dtype=self.dtype, name="1")

    # 前向传播，输入特征向量，输出分类结果
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.classifier(x)


# 定义一个用于图像分类的 FlaxResNetForImageClassificationModule 类
class FlaxResNetForImageClassificationModule(nn.Module):
    # 从配置中获取相关参数
    config: ResNetConfig
    # 设置数据类型为 float32
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 创建一个 ResNet 模型
        self.resnet = FlaxResNetModule(config=self.config, dtype=self.dtype)

        # 如果有类别数量，创建分类器
        if self.config.num_labels > 0:
            self.classifier = FlaxResNetClassifierCollection(self.config, dtype=self.dtype)
        # 否则创建一个恒等映射
        else:
            self.classifier = Identity()

    # 前向传播
    def __call__(
        self,
        pixel_values=None,
        deterministic: bool = True,
        output_hidden_states=None,
        return_dict=None,
    ):
        # 根据配置决定是否返回字典格式
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 通过 ResNet 模型获得特征输出
        outputs = self.resnet(
            pixel_values,
            deterministic=deterministic,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 取出池化后的特征向量
        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        # 通过分类器得到分类结果
        logits = self.classifier(pooled_output[:, :, 0, 0])

        # 根据是否返回字典决定输出格式
        if not return_dict:
            output = (logits,) + outputs[2:]
            return output

        return FlaxImageClassifierOutputWithNoAttention(logits=logits, hidden_states=outputs.hidden_states)


# 定义一个用于图像分类的 FlaxResNetForImageClassification 类
@add_start_docstrings(
    """
    ResNet Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    RESNET_START_DOCSTRING,
)
class FlaxResNetForImageClassification(FlaxResNetPreTrainedModel):
    module_class = FlaxResNetForImageClassificationModule
```