# `.\models\resnet\modeling_flax_resnet.py`

```
# 导入必要的模块和函数
from functools import partial  # 导入 functools 模块中的 partial 函数
from typing import Optional, Tuple  # 导入类型提示的 Optional 和 Tuple

import flax.linen as nn  # 导入 flax.linen 模块并重命名为 nn
import jax  # 导入 jax 库
import jax.numpy as jnp  # 导入 jax 库中的 numpy 并重命名为 jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze  # 从 flax.core.frozen_dict 导入相关函数
from flax.traverse_util import flatten_dict, unflatten_dict  # 从 flax.traverse_util 导入 flatten_dict 和 unflatten_dict

from ...modeling_flax_outputs import (  # 导入输出相关的模块
    FlaxBaseModelOutputWithNoAttention,
    FlaxBaseModelOutputWithPoolingAndNoAttention,
    FlaxImageClassifierOutputWithNoAttention,
)
from ...modeling_flax_utils import (  # 导入实用函数和类
    ACT2FN,
    FlaxPreTrainedModel,
    append_replace_return_docstrings,
    overwrite_call_docstring,
)
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward  # 从 utils 导入函数
from .configuration_resnet import ResNetConfig  # 导入 ResNetConfig 配置类

RESNET_START_DOCSTRING = r"""

    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading, saving and converting weights from PyTorch models)

    This model is also a
    [flax.linen.Module](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html) subclass. Use it as
    a regular Flax linen Module and refer to the Flax documentation for all matter related to general usage and
    behavior.

    Finally, this model supports inherent JAX features such as:

    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

"""

# 上述部分是对模型的开始文档字符串的定义和赋值
    # Parameters参数：
    # config ([`ResNetConfig`]): 模型配置类，包含模型的所有参数。
    #   使用配置文件初始化不会加载与模型相关的权重，仅加载配置。
    #   查看 [`~FlaxPreTrainedModel.from_pretrained`] 方法以加载模型权重。
    # dtype (`jax.numpy.dtype`, *optional*, 默认为 `jax.numpy.float32`):
    #   计算时的数据类型。可以是 `jax.numpy.float32`、`jax.numpy.float16`（在GPU上）、`jax.numpy.bfloat16`（在TPU上）之一。
    #   这可用于在GPU或TPU上启用混合精度训练或半精度推断。如果指定，所有计算将使用给定的 `dtype` 进行。
    #
    #   **注意，这仅指定计算的数据类型，不影响模型参数的数据类型。**
    #
    #   如果要更改模型参数的数据类型，请参见 [`~FlaxPreTrainedModel.to_fp16`] 和 [`~FlaxPreTrainedModel.to_bf16`]。
"""
    Args:
        pixel_values (`jax.numpy.float32` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`AutoImageProcessor.__call__`] for details.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class Identity(nn.Module):
    """Identity function."""

    @nn.compact
    def __call__(self, x, **kwargs):
        return x


class FlaxResNetConvLayer(nn.Module):
    """
    Defines a convolutional layer followed by batch normalization and activation function.
    """

    out_channels: int
    kernel_size: int = 3
    stride: int = 1
    activation: Optional[str] = "relu"
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # Define the convolutional layer
        self.convolution = nn.Conv(
            self.out_channels,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=self.stride,
            padding=self.kernel_size // 2,
            dtype=self.dtype,
            use_bias=False,
            kernel_init=nn.initializers.variance_scaling(2.0, mode="fan_out", distribution="normal", dtype=self.dtype),
        )
        # Define batch normalization layer
        self.normalization = nn.BatchNorm(momentum=0.9, epsilon=1e-05, dtype=self.dtype)
        # Define activation function
        self.activation_func = ACT2FN[self.activation] if self.activation is not None else Identity()

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # Perform convolution
        hidden_state = self.convolution(x)
        # Apply batch normalization
        hidden_state = self.normalization(hidden_state, use_running_average=deterministic)
        # Apply activation function
        hidden_state = self.activation_func(hidden_state)
        return hidden_state


class FlaxResNetEmbeddings(nn.Module):
    """
    ResNet Embeddings (stem) composed of a single aggressive convolution.
    """

    config: ResNetConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # Define the embedding layer using FlaxResNetConvLayer
        self.embedder = FlaxResNetConvLayer(
            self.config.embedding_size,
            kernel_size=7,
            stride=2,
            activation=self.config.hidden_act,
            dtype=self.dtype,
        )
        # Define max pooling operation
        self.max_pool = partial(nn.max_pool, window_shape=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))

    def __call__(self, pixel_values: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        num_channels = pixel_values.shape[-1]
        # Check if number of input channels matches the configuration
        if num_channels != self.config.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # Apply embedding layer
        embedding = self.embedder(pixel_values, deterministic=deterministic)
        # Apply max pooling
        embedding = self.max_pool(embedding)
        return embedding


class FlaxResNetShortCut(nn.Module):
    """
    Placeholder class for Flax ResNet shortcut connections.
    No implementation details provided.
    """
    ResNet shortcut, used to project the residual features to the correct size. If needed, it is also used to
    downsample the input using `stride=2`.
    """

    # 定义一个类，用于 ResNet 中的快捷连接（shortcut），将残差特征投影到正确的尺寸。如果需要，也可以用来
    # 使用 `stride=2` 对输入进行下采样。
    out_channels: int  # 输出通道数，即卷积层输出的特征图的深度
    stride: int = 2  # 步长，默认为2，用于卷积操作时的步进大小
    dtype: jnp.dtype = jnp.float32  # 数据类型，默认为 jax 的 float32 类型

    def setup(self):
        # 设置卷积层，用于实现残差块中的投影操作，将输入特征投影到输出通道大小
        self.convolution = nn.Conv(
            self.out_channels,
            kernel_size=(1, 1),  # 卷积核大小为 1x1
            strides=self.stride,  # 使用类属性中定义的步长
            use_bias=False,  # 不使用偏置项
            kernel_init=nn.initializers.variance_scaling(2.0, mode="fan_out", distribution="truncated_normal"),  # 卷积核初始化方式
            dtype=self.dtype,  # 使用指定的数据类型
        )
        # 设置批归一化层，用于规范化卷积层的输出特征
        self.normalization = nn.BatchNorm(momentum=0.9, epsilon=1e-05, dtype=self.dtype)

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # 在调用时，对输入 x 进行卷积投影操作
        hidden_state = self.convolution(x)
        # 对投影后的特征进行批归一化处理
        hidden_state = self.normalization(hidden_state, use_running_average=deterministic)
        return hidden_state
class FlaxResNetBasicLayerCollection(nn.Module):
    out_channels: int                     # 输出通道数
    stride: int = 1                       # 步长，默认为1
    dtype: jnp.dtype = jnp.float32         # 数据类型，默认为32位浮点数

    def setup(self):
        self.layer = [                    # 创建层列表
            FlaxResNetConvLayer(self.out_channels, stride=self.stride, dtype=self.dtype),  # 卷积层1
            FlaxResNetConvLayer(self.out_channels, activation=None, dtype=self.dtype),      # 卷积层2
        ]

    def __call__(self, hidden_state: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        for layer in self.layer:         # 对每一层进行迭代
            hidden_state = layer(hidden_state, deterministic=deterministic)  # 应用层到隐藏状态
        return hidden_state               # 返回最终隐藏状态


class FlaxResNetBasicLayer(nn.Module):
    """
    A classic ResNet's residual layer composed by two `3x3` convolutions.
    """
    
    in_channels: int                      # 输入通道数
    out_channels: int                     # 输出通道数
    stride: int = 1                       # 步长，默认为1
    activation: Optional[str] = "relu"    # 激活函数，默认为ReLU
    dtype: jnp.dtype = jnp.float32         # 数据类型，默认为32位浮点数

    def setup(self):
        should_apply_shortcut = self.in_channels != self.out_channels or self.stride != 1  # 是否应用快捷连接
        self.shortcut = (
            FlaxResNetShortCut(self.out_channels, stride=self.stride, dtype=self.dtype)  # 如果需要，则创建快捷连接
            if should_apply_shortcut
            else None
        )
        self.layer = FlaxResNetBasicLayerCollection(  # 创建基础层集合
            out_channels=self.out_channels,
            stride=self.stride,
            dtype=self.dtype,
        )
        self.activation_func = ACT2FN[self.activation]  # 获取激活函数

    def __call__(self, hidden_state, deterministic: bool = True):
        residual = hidden_state          # 保存残差
        hidden_state = self.layer(hidden_state, deterministic=deterministic)  # 应用基础层到隐藏状态

        if self.shortcut is not None:    # 如果存在快捷连接
            residual = self.shortcut(residual, deterministic=deterministic)  # 应用快捷连接到残差
        hidden_state += residual         # 添加残差到隐藏状态

        hidden_state = self.activation_func(hidden_state)  # 应用激活函数到隐藏状态
        return hidden_state               # 返回最终隐藏状态


class FlaxResNetBottleNeckLayerCollection(nn.Module):
    out_channels: int                     # 输出通道数
    stride: int = 1                       # 步长，默认为1
    activation: Optional[str] = "relu"    # 激活函数，默认为ReLU
    reduction: int = 4                    # 减少倍数，默认为4
    dtype: jnp.dtype = jnp.float32         # 数据类型，默认为32位浮点数

    def setup(self):
        reduces_channels = self.out_channels // self.reduction  # 减少的通道数

        self.layer = [
            FlaxResNetConvLayer(reduces_channels, kernel_size=1, dtype=self.dtype, name="0"),  # 第一个卷积层
            FlaxResNetConvLayer(reduces_channels, stride=self.stride, dtype=self.dtype, name="1"),  # 第二个卷积层
            FlaxResNetConvLayer(self.out_channels, kernel_size=1, activation=None, dtype=self.dtype, name="2"),  # 第三个卷积层
        ]

    def __call__(self, hidden_state: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        for layer in self.layer:         # 对每一层进行迭代
            hidden_state = layer(hidden_state, deterministic=deterministic)  # 应用层到隐藏状态
        return hidden_state               # 返回最终隐藏状态


class FlaxResNetBottleNeckLayer(nn.Module):
    """
    A classic ResNet's bottleneck layer composed by three `3x3` convolutions. The first `1x1` convolution reduces the
    input by a factor of `reduction` in order to make the second `3x3` convolution faster. The last `1x1` convolution
    remaps the reduced features to `out_channels`.
    """
    # 输入通道数，表示输入特征的数量
    in_channels: int
    # 输出通道数，表示输出特征的数量
    out_channels: int
    # 步长，默认为1，控制卷积操作的步长大小
    stride: int = 1
    # 激活函数类型，默认为"relu"
    activation: Optional[str] = "relu"
    # 降维参数，默认为4，用于瓶颈层中的维度缩减
    reduction: int = 4
    # 数据类型，默认为32位浮点数
    dtype: jnp.dtype = jnp.float32

    # 初始化方法，用于设置网络层的结构
    def setup(self):
        # 判断是否需要应用快捷连接（shortcut）
        should_apply_shortcut = self.in_channels != self.out_channels or self.stride != 1
        # 如果需要应用快捷连接，则创建一个FlaxResNetShortCut对象
        self.shortcut = (
            FlaxResNetShortCut(self.out_channels, stride=self.stride, dtype=self.dtype)
            if should_apply_shortcut
            else None
        )

        # 创建一个FlaxResNetBottleNeckLayerCollection对象，用于构建瓶颈层集合
        self.layer = FlaxResNetBottleNeckLayerCollection(
            self.out_channels,
            stride=self.stride,
            activation=self.activation,
            reduction=self.reduction,
            dtype=self.dtype,
        )

        # 获取指定名称的激活函数
        self.activation_func = ACT2FN[self.activation]

    # 对象调用方法，用于执行网络层的前向传播
    def __call__(self, hidden_state: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # 将输入隐藏状态作为残差进行保存
        residual = hidden_state

        # 如果存在快捷连接，则对残差应用快捷连接
        if self.shortcut is not None:
            residual = self.shortcut(residual, deterministic=deterministic)
        
        # 对输入隐藏状态应用瓶颈层集合的处理
        hidden_state = self.layer(hidden_state, deterministic)
        
        # 将原始输入和处理后的残差相加
        hidden_state += residual
        
        # 对相加后的结果应用激活函数
        hidden_state = self.activation_func(hidden_state)
        
        # 返回处理后的隐藏状态
        return hidden_state
class FlaxResNetStageLayersCollection(nn.Module):
    """
    A ResNet stage composed by stacked layers.
    """

    config: ResNetConfig  # 配置对象，包含 ResNet 的配置信息
    in_channels: int  # 输入通道数
    out_channels: int  # 输出通道数
    stride: int = 2  # 步幅，默认为 2
    depth: int = 2  # 层深度，默认为 2
    dtype: jnp.dtype = jnp.float32  # 数据类型，默认为 jnp.float32

    def setup(self):
        layer = FlaxResNetBottleNeckLayer if self.config.layer_type == "bottleneck" else FlaxResNetBasicLayer

        layers = [
            # downsampling is done in the first layer with stride of 2
            # 第一层进行下采样，步幅为 2
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
            hidden_state = layer(hidden_state, deterministic=deterministic)
        return hidden_state


class FlaxResNetStage(nn.Module):
    """
    A ResNet stage composed by stacked layers.
    """

    config: ResNetConfig  # 配置对象，包含 ResNet 的配置信息
    in_channels: int  # 输入通道数
    out_channels: int  # 输出通道数
    stride: int = 2  # 步幅，默认为 2
    depth: int = 2  # 层深度，默认为 2
    dtype: jnp.dtype = jnp.float32  # 数据类型，默认为 jnp.float32

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
        return self.layers(x, deterministic=deterministic)


class FlaxResNetStageCollection(nn.Module):
    config: ResNetConfig  # 配置对象，包含 ResNet 的配置信息
    dtype: jnp.dtype = jnp.float32  # 数据类型，默认为 jnp.float32

    def setup(self):
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
    # 定义类中的调用方法，用于执行模型推理过程，返回不包含注意力权重的模型输出对象
    def __call__(
        self,
        hidden_state: jnp.ndarray,
        output_hidden_states: bool = False,
        deterministic: bool = True,
    ) -> FlaxBaseModelOutputWithNoAttention:
        # 如果需要输出隐藏状态，则初始化一个空元组；否则置为 None
        hidden_states = () if output_hidden_states else None
    
        # 遍历模型的各个阶段模块
        for stage_module in self.stages:
            # 如果需要输出隐藏状态，则将隐藏状态进行维度转换，并添加到隐藏状态元组中
            if output_hidden_states:
                hidden_states = hidden_states + (hidden_state.transpose(0, 3, 1, 2),)
            
            # 调用当前阶段模块，更新隐藏状态
            hidden_state = stage_module(hidden_state, deterministic=deterministic)
    
        # 返回更新后的隐藏状态和可能的隐藏状态元组
        return hidden_state, hidden_states
# 定义一个继承自 nn.Module 的类 FlaxResNetEncoder，用于实现 ResNet 编码器
class FlaxResNetEncoder(nn.Module):
    # 类属性 config，类型为 ResNetConfig，用于配置模型
    config: ResNetConfig
    # 类属性 dtype，默认为 jnp.float32 的数据类型
    dtype: jnp.dtype = jnp.float32

    # 初始化方法，设置编码器的各个阶段
    def setup(self):
        # 创建 FlaxResNetStageCollection 对象，用于管理 ResNet 的阶段
        self.stages = FlaxResNetStageCollection(self.config, dtype=self.dtype)

    # 对象调用方法，实现编码器的前向传播逻辑
    def __call__(
        self,
        hidden_state: jnp.ndarray,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
    ) -> FlaxBaseModelOutputWithNoAttention:
        # 调用编码器的阶段对象进行前向传播，得到编码后的隐藏状态和可能的中间隐藏状态列表
        hidden_state, hidden_states = self.stages(
            hidden_state, output_hidden_states=output_hidden_states, deterministic=deterministic
        )

        # 如果需要输出中间隐藏状态，将当前隐藏状态加入到隐藏状态列表中
        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state.transpose(0, 3, 1, 2),)

        # 如果不需要返回字典形式的输出，则将有效的结果作为元组返回
        if not return_dict:
            return tuple(v for v in [hidden_state, hidden_states] if v is not None)

        # 返回包含最终隐藏状态和隐藏状态列表的 FlaxBaseModelOutputWithNoAttention 对象
        return FlaxBaseModelOutputWithNoAttention(
            last_hidden_state=hidden_state,
            hidden_states=hidden_states,
        )


# 定义一个继承自 FlaxPreTrainedModel 的抽象类 FlaxResNetPreTrainedModel
class FlaxResNetPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 类属性 config_class，指定模型配置类为 ResNetConfig
    config_class = ResNetConfig
    # 类属性 base_model_prefix，指定基础模型前缀为 "resnet"
    base_model_prefix = "resnet"
    # 类属性 main_input_name，指定主要输入名称为 "pixel_values"
    main_input_name = "pixel_values"
    # 类属性 module_class，用于指定模块类的类型，默认为 None
    module_class: nn.Module = None

    # 初始化方法，用于创建模型对象
    def __init__(
        self,
        config: ResNetConfig,
        input_shape=(1, 224, 224, 3),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        # 根据配置类和其他参数创建模块对象
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 如果未指定输入形状，则使用默认形状根据配置设置
        if input_shape is None:
            input_shape = (1, config.image_size, config.image_size, config.num_channels)
        # 调用父类的初始化方法，传递配置、模块、输入形状等参数
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    # 初始化权重方法，用于随机初始化模型的参数
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化像素值为全零张量作为输入
        pixel_values = jnp.zeros(input_shape, dtype=self.dtype)

        # 创建随机数生成器字典，用于参数初始化
        rngs = {"params": rng}

        # 使用模块对象的初始化方法，初始化模型参数，返回未冻结的参数字典
        random_params = self.module.init(rngs, pixel_values, return_dict=False)

        # 如果提供了已有的参数字典，则将缺失的参数从随机初始化的参数中复制过来
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    # 对象调用方法，实现模型的前向传播逻辑
    @add_start_docstrings_to_model_forward(RESNET_INPUTS_DOCSTRING)
    def __call__(
        self,
        pixel_values,
        params: dict = None,
        train: bool = False,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> FlaxBaseModelOutput:
        # 实现模型的前向传播逻辑，具体细节根据模型的实现和输入参数进行处理
        pass  # 这里未提供完整的方法实现，需要根据具体模型的实现补充
        ):
        # 如果 output_hidden_states 不为 None，则使用传入的值，否则使用模型配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果 return_dict 不为 None，则使用传入的值，否则使用模型配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 调整像素值的维度顺序，将通道维度移到最后一个维度位置
        pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))

        # 处理可能需要的随机数生成器
        rngs = {}

        # 调用模型的 apply 方法进行推断或训练
        return self.module.apply(
            {
                "params": params["params"] if params is not None else self.params["params"],
                "batch_stats": params["batch_stats"] if params is not None else self.params["batch_stats"],
            },
            jnp.array(pixel_values, dtype=jnp.float32),  # 将像素值转换为 jax 数组并传入
            not train,  # 如果不是训练模式，则传入 True，表示推断模式
            output_hidden_states,  # 传入是否需要隐藏状态的标志
            return_dict,  # 传入是否返回字典的标志
            rngs=rngs,  # 传入随机数生成器
            mutable=["batch_stats"] if train else False,  # 当 train 为 True 时返回包含 batch_stats 的元组
        )
# 定义一个名为FlaxResNetModule的类，继承自nn.Module
class FlaxResNetModule(nn.Module):
    # 类变量config，用于存储ResNet的配置信息
    config: ResNetConfig
    # 定义dtype变量，默认为jnp.float32，用于指定计算时的数据类型
    dtype: jnp.dtype = jnp.float32  # 计算中使用的数据类型

    # 定义setup方法，用于初始化模块
    def setup(self):
        # 创建FlaxResNetEmbeddings对象，使用self.config和self.dtype作为参数
        self.embedder = FlaxResNetEmbeddings(self.config, dtype=self.dtype)
        # 创建FlaxResNetEncoder对象，使用self.config和self.dtype作为参数
        self.encoder = FlaxResNetEncoder(self.config, dtype=self.dtype)

        # 创建部分应用了avg_pool函数的pooler对象，设置了padding参数
        self.pooler = partial(
            nn.avg_pool,
            padding=((0, 0), (0, 0)),
        )

    # 定义__call__方法，实现对象的调用功能
    def __call__(
        self,
        pixel_values,
        deterministic: bool = True,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> FlaxBaseModelOutputWithPoolingAndNoAttention:
        # 如果output_hidden_states为None，则使用self.config.output_hidden_states
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果return_dict为None，则使用self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 获取嵌入输出，调用self.embedder对象
        embedding_output = self.embedder(pixel_values, deterministic=deterministic)

        # 获取编码器输出，调用self.encoder对象
        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        # 获取最后隐藏状态
        last_hidden_state = encoder_outputs[0]

        # 对最后隐藏状态进行自适应平均池化操作
        pooled_output = self.pooler(
            last_hidden_state,
            window_shape=(last_hidden_state.shape[1], last_hidden_state.shape[2]),
            strides=(last_hidden_state.shape[1], last_hidden_state.shape[2]),
        ).transpose(0, 3, 1, 2)

        # 调整最后隐藏状态的维度顺序
        last_hidden_state = last_hidden_state.transpose(0, 3, 1, 2)

        # 如果return_dict为False，则返回元组
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 返回FlaxBaseModelOutputWithPoolingAndNoAttention对象
        return FlaxBaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )


# 为FlaxResNetModel类添加文档注释
@add_start_docstrings(
    "The bare ResNet model outputting raw features without any specific head on top.",
    RESNET_START_DOCSTRING,
)
class FlaxResNetModel(FlaxResNetPreTrainedModel):
    module_class = FlaxResNetModule


# 定义FLAX_VISION_MODEL_DOCSTRING常量，包含FlaxResNetModel类的文档字符串
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

# 调用overwrite_call_docstring函数，为FlaxResNetModel类覆盖文档字符串
overwrite_call_docstring(FlaxResNetModel, FLAX_VISION_MODEL_DOCSTRING)
# 调用append_replace_return_docstrings函数，为FlaxResNetModel类添加或替换返回值的文档字符串
append_replace_return_docstrings(
    # 导入FlaxResNetModel类，并指定output_type和config_class参数
    FlaxResNetModel, output_type=FlaxBaseModelOutputWithPoolingAndNoAttention, config_class=ResNetConfig
# 导入所需的库和模块
import jax
import requests

# FLAX_VISION_CLASSIF_DOCSTRING文档字符串，描述了FlaxResNetForImageClassification模型的返回和示例用法
FLAX_VISION_CLASSIF_DOCSTRING = """
    Returns:
        返回一个示例，展示如何使用该模型进行图像分类预测。

    Example:
        展示如何从URL下载图像并使用模型预测图像的分类。

    ```python
    >>> from transformers import AutoImageProcessor, FlaxResNetForImageClassification
    >>> from PIL import Image
    >>> import jax
    >>> import requests

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    >>> model = FlaxResNetForImageClassification.from_pretrained("microsoft/resnet-50")

    >>> inputs = image_processor(images=image, return_tensors="np")
    >>> outputs = model(**inputs)
    >>> logits = outputs.logits

    >>> # model predicts one of the 1000 ImageNet classes
    >>> predicted_class_idx = jax.numpy.argmax(logits, axis=-1)
    >>> print("Predicted class:", model.config.id2label[predicted_class_idx.item()])
    ```
"""

# 使用overwrite_call_docstring函数将FLAX_VISION_CLASSIF_DOCSTRING设置为FlaxResNetForImageClassification类的文档字符串
overwrite_call_docstring(FlaxResNetForImageClassification, FLAX_VISION_CLASSIF_DOCSTRING)

# 使用append_replace_return_docstrings函数，为FlaxResNetForImageClassification类追加和替换返回结果的文档字符串
append_replace_return_docstrings(
    FlaxResNetForImageClassification, output_type=FlaxImageClassifierOutputWithNoAttention, config_class=ResNetConfig
)
```