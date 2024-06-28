# `.\models\regnet\modeling_flax_regnet.py`

```
# 导入所需的模块和库
from functools import partial  # 导入partial函数，用于创建带预设参数的可调用对象
from typing import Optional, Tuple  # 引入类型提示，用于函数参数和返回类型的声明

import flax.linen as nn  # 导入Flax的linen模块，用于定义Flax模型
import jax  # 导入JAX，用于自动微分和并行计算
import jax.numpy as jnp  # 导入JAX的NumPy接口，用于数值计算
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze  # 导入FrozenDict和相关函数，用于不可变字典的管理
from flax.traverse_util import flatten_dict, unflatten_dict  # 导入flatten_dict和unflatten_dict函数，用于字典的扁平化和恢复

from transformers import RegNetConfig  # 导入RegNetConfig类，用于配置RegNet模型
from transformers.modeling_flax_outputs import (  # 导入Flax模型输出相关类
    FlaxBaseModelOutputWithNoAttention,
    FlaxBaseModelOutputWithPooling,
    FlaxBaseModelOutputWithPoolingAndNoAttention,
    FlaxImageClassifierOutputWithNoAttention,
)
from transformers.modeling_flax_utils import (  # 导入Flax模型工具函数
    ACT2FN,
    FlaxPreTrainedModel,
    append_replace_return_docstrings,
    overwrite_call_docstring,
)
from transformers.utils import (  # 导入transformers工具函数
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)

# 定义模型文档字符串
REGNET_START_DOCSTRING = r"""

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
    # 定义函数参数：
    #   config ([`RegNetConfig`]): 模型配置类，包含模型的所有参数。
    #       仅使用配置文件初始化，不加载与模型相关的权重，只加载配置信息。
    #       查看 [`~FlaxPreTrainedModel.from_pretrained`] 方法以加载模型权重。
    #   dtype (`jax.numpy.dtype`, *可选*, 默认为 `jax.numpy.float32`):
    #       计算的数据类型。可以是 `jax.numpy.float32`, `jax.numpy.float16` (在GPU上)，以及 `jax.numpy.bfloat16` (在TPU上)。
    #       
    #       这可以用于在GPU或TPU上启用混合精度训练或半精度推断。如果指定，则所有计算将使用给定的 `dtype` 执行。
    #       
    #       **请注意，这仅指定计算的数据类型，并不影响模型参数的数据类型。**
    #       
    #       如果要更改模型参数的数据类型，请参阅 [`~FlaxPreTrainedModel.to_fp16`] 和 [`~FlaxPreTrainedModel.to_bf16`]。
"""

REGNET_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`numpy.ndarray` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`RegNetImageProcessor.__call__`] for details.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


# Copied from transformers.models.resnet.modeling_flax_resnet.Identity
class Identity(nn.Module):
    """Identity function."""
    
    @nn.compact
    def __call__(self, x, **kwargs):
        return x
    RegNet shortcut, used to project the residual features to the correct size. If needed, it is also used to
    downsample the input using `stride=2`.
    """
    
    # 定义一个类，用于执行RegNet的shortcut操作，将残差特征投影到正确的大小。如果需要，还可以使用 `stride=2` 对输入进行降采样。
    class RegNetShortcut(nn.Module):
        
        # 初始化函数，设置输出通道数、步幅为2、数据类型为 jnp.float32
        def __init__(self, out_channels: int, stride: int = 2, dtype: jnp.dtype = jnp.float32):
            self.out_channels = out_channels
            self.stride = stride
            self.dtype = dtype
            
        # 在设置阶段定义操作：使用 1x1 的卷积层进行投影，无偏置，采用截断正态分布进行初始化，数据类型为 self.dtype
        def setup(self):
            self.convolution = nn.Conv(
                self.out_channels,
                kernel_size=(1, 1),
                strides=self.stride,
                use_bias=False,
                kernel_init=nn.initializers.variance_scaling(2.0, mode="fan_out", distribution="truncated_normal"),
                dtype=self.dtype,
            )
            # 使用批量归一化层，设置动量为 0.9，epsilon 为 1e-05，数据类型为 self.dtype
            self.normalization = nn.BatchNorm(momentum=0.9, epsilon=1e-05, dtype=self.dtype)
        
        # 调用实例时执行的操作：对输入 x 进行卷积投影，然后对投影结果进行归一化
        def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
            hidden_state = self.convolution(x)
            hidden_state = self.normalization(hidden_state, use_running_average=deterministic)
            return hidden_state
class FlaxRegNetSELayerCollection(nn.Module):
    in_channels: int
    reduced_channels: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 定义第一个卷积层，用于 SE 层集合
        self.conv_1 = nn.Conv(
            self.reduced_channels,
            kernel_size=(1, 1),
            kernel_init=nn.initializers.variance_scaling(2.0, mode="fan_out", distribution="truncated_normal"),
            dtype=self.dtype,
            name="0",
        )  # 0 is the name used in corresponding pytorch implementation
        # 定义第二个卷积层，用于 SE 层集合
        self.conv_2 = nn.Conv(
            self.in_channels,
            kernel_size=(1, 1),
            kernel_init=nn.initializers.variance_scaling(2.0, mode="fan_out", distribution="truncated_normal"),
            dtype=self.dtype,
            name="2",
        )  # 2 is the name used in corresponding pytorch implementation

    def __call__(self, hidden_state: jnp.ndarray) -> jnp.ndarray:
        # 对输入的隐藏状态应用第一个卷积层
        hidden_state = self.conv_1(hidden_state)
        # 应用 ReLU 激活函数
        hidden_state = nn.relu(hidden_state)
        # 对处理后的隐藏状态应用第二个卷积层
        hidden_state = self.conv_2(hidden_state)
        # 计算注意力，使用 sigmoid 激活函数
        attention = nn.sigmoid(hidden_state)

        return attention


class FlaxRegNetSELayer(nn.Module):
    """
    Squeeze and Excitation layer (SE) proposed in [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507).
    """

    in_channels: int
    reduced_channels: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 定义平均池化操作的部分函数
        self.pooler = partial(nn.avg_pool, padding=((0, 0), (0, 0)))
        # 初始化 SE 层集合作为注意力机制
        self.attention = FlaxRegNetSELayerCollection(self.in_channels, self.reduced_channels, dtype=self.dtype)

    def __call__(self, hidden_state: jnp.ndarray) -> jnp.ndarray:
        # 对输入的隐藏状态进行平均池化
        pooled = self.pooler(
            hidden_state,
            window_shape=(hidden_state.shape[1], hidden_state.shape[2]),
            strides=(hidden_state.shape[1], hidden_state.shape[2]),
        )
        # 应用注意力机制得到注意力张量
        attention = self.attention(pooled)
        # 将原始隐藏状态与注意力张量相乘，执行 SE 操作
        hidden_state = hidden_state * attention
        return hidden_state


class FlaxRegNetXLayerCollection(nn.Module):
    config: RegNetConfig
    out_channels: int
    stride: int = 1
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 计算组数，确保每个组的宽度不小于 1
        groups = max(1, self.out_channels // self.config.groups_width)

        # 定义层集合，包括三个卷积层
        self.layer = [
            FlaxRegNetConvLayer(
                self.out_channels,
                kernel_size=1,
                activation=self.config.hidden_act,
                dtype=self.dtype,
                name="0",
            ),
            FlaxRegNetConvLayer(
                self.out_channels,
                stride=self.stride,
                groups=groups,
                activation=self.config.hidden_act,
                dtype=self.dtype,
                name="1",
            ),
            FlaxRegNetConvLayer(
                self.out_channels,
                kernel_size=1,
                activation=None,
                dtype=self.dtype,
                name="2",
            ),
        ]
    # 定义一个特殊方法 __call__，使得该类的实例对象可以像函数一样被调用
    def __call__(self, hidden_state: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # 遍历该类实例对象中的每一个层（layer）
        for layer in self.layer:
            # 对隐藏状态（hidden_state）依次应用每一层（layer）的操作
            hidden_state = layer(hidden_state, deterministic=deterministic)
        # 返回经过所有层操作后的最终隐藏状态
        return hidden_state
# 定义一个 FlaxRegNetXLayer 类，表示 RegNet 的 X 层模块
class FlaxRegNetXLayer(nn.Module):
    """
    RegNet 的层，由三个 3x3 卷积组成，与 ResNet 的瓶颈层相同，但 reduction = 1。
    """

    # 类属性：RegNet 的配置
    config: RegNetConfig
    # 输入通道数
    in_channels: int
    # 输出通道数
    out_channels: int
    # 步幅，默认为 1
    stride: int = 1
    # 数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 设置方法，用于初始化层的各个组件
    def setup(self):
        # 判断是否需要应用 shortcut
        should_apply_shortcut = self.in_channels != self.out_channels or self.stride != 1
        # 如果需要应用 shortcut，则初始化为 FlaxRegNetShortCut 对象；否则初始化为 Identity 对象
        self.shortcut = (
            FlaxRegNetShortCut(
                self.out_channels,
                stride=self.stride,
                dtype=self.dtype,
            )
            if should_apply_shortcut
            else Identity()
        )
        # 初始化层对象为 FlaxRegNetXLayerCollection 实例
        self.layer = FlaxRegNetXLayerCollection(
            self.config,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            stride=self.stride,
            dtype=self.dtype,
        )
        # 激活函数为 ACT2FN 中根据配置选择的隐藏激活函数
        self.activation_func = ACT2FN[self.config.hidden_act]

    # 调用方法，用于前向传播
    def __call__(self, hidden_state: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # 将输入 hidden_state 赋值给 residual 作为残差连接的起始值
        residual = hidden_state
        # 经过 X 层的主体卷积操作
        hidden_state = self.layer(hidden_state)
        # 对残差应用 shortcut 操作
        residual = self.shortcut(residual, deterministic=deterministic)
        # 将主体卷积的输出与 shortcut 的结果相加，实现残差连接
        hidden_state += residual
        # 应用激活函数到最终输出
        hidden_state = self.activation_func(hidden_state)
        return hidden_state


# 定义一个 FlaxRegNetYLayerCollection 类，表示 RegNet 的 Y 层的卷积集合
class FlaxRegNetYLayerCollection(nn.Module):
    config: RegNetConfig
    in_channels: int
    out_channels: int
    stride: int = 1
    dtype: jnp.dtype = jnp.float32

    # 设置方法，用于初始化层的各个组件
    def setup(self):
        # 计算组数，用于分组卷积
        groups = max(1, self.out_channels // self.config.groups_width)

        # 初始化层对象为包含四个子层的列表
        self.layer = [
            # 第一个卷积层，1x1 卷积
            FlaxRegNetConvLayer(
                self.out_channels,
                kernel_size=1,
                activation=self.config.hidden_act,
                dtype=self.dtype,
                name="0",
            ),
            # 第二个卷积层，3x3 卷积，带分组卷积
            FlaxRegNetConvLayer(
                self.out_channels,
                stride=self.stride,
                groups=groups,
                activation=self.config.hidden_act,
                dtype=self.dtype,
                name="1",
            ),
            # Squeeze and Excitation 模块
            FlaxRegNetSELayer(
                self.out_channels,
                reduced_channels=int(round(self.in_channels / 4)),
                dtype=self.dtype,
                name="2",
            ),
            # 第四个卷积层，1x1 卷积，不带激活函数
            FlaxRegNetConvLayer(
                self.out_channels,
                kernel_size=1,
                activation=None,
                dtype=self.dtype,
                name="3",
            ),
        ]

    # 调用方法，用于前向传播
    def __call__(self, hidden_state: jnp.ndarray) -> jnp.ndarray:
        # 依次对每个子层进行前向传播
        for layer in self.layer:
            hidden_state = layer(hidden_state)
        return hidden_state


# 定义一个 FlaxRegNetYLayer 类，表示 RegNet 的 Y 层模块，是 X 层与 Squeeze and Excitation 的组合
class FlaxRegNetYLayer(nn.Module):
    """
    RegNet 的 Y 层：包含一个 X 层和 Squeeze and Excitation 模块。
    """

    config: RegNetConfig
    in_channels: int
    out_channels: int
    stride: int = 1
    dtype: jnp.dtype = jnp.float32
    # 定义设置方法，用于初始化网络层
    def setup(self):
        # 检查是否需要应用快捷连接，条件为输入通道数不等于输出通道数或步长不为1
        should_apply_shortcut = self.in_channels != self.out_channels or self.stride != 1

        # 根据条件选择不同的快捷连接方式，若需要则使用 FlaxRegNetShortCut，否则使用 Identity()
        self.shortcut = (
            FlaxRegNetShortCut(
                self.out_channels,
                stride=self.stride,
                dtype=self.dtype,
            )
            if should_apply_shortcut
            else Identity()
        )

        # 创建网络层集合对象 FlaxRegNetYLayerCollection
        self.layer = FlaxRegNetYLayerCollection(
            self.config,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            stride=self.stride,
            dtype=self.dtype,
        )

        # 选择激活函数，根据配置选择对应的激活函数
        self.activation_func = ACT2FN[self.config.hidden_act]

    # 定义调用方法，用于执行前向传播计算
    def __call__(self, hidden_state: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # 将输入的隐藏状态作为残差保存
        residual = hidden_state
        # 通过网络层集合对象计算新的隐藏状态
        hidden_state = self.layer(hidden_state)
        # 根据快捷连接计算残差项
        residual = self.shortcut(residual, deterministic=deterministic)
        # 将残差项加到新的隐藏状态上
        hidden_state += residual
        # 应用选择的激活函数到更新后的隐藏状态上
        hidden_state = self.activation_func(hidden_state)
        # 返回更新后的隐藏状态作为输出
        return hidden_state
class FlaxRegNetStageLayersCollection(nn.Module):
    """
    A RegNet stage composed by stacked layers.
    """

    config: RegNetConfig  # 存储RegNet配置的对象
    in_channels: int  # 输入通道数
    out_channels: int  # 输出通道数
    stride: int = 2  # 步幅，默认为2
    depth: int = 2  # 层的深度，默认为2
    dtype: jnp.dtype = jnp.float32  # 数据类型，默认为32位浮点数

    def setup(self):
        layer = FlaxRegNetXLayer if self.config.layer_type == "x" else FlaxRegNetYLayer

        layers = [
            # downsampling is done in the first layer with stride of 2
            layer(
                self.config,
                self.in_channels,
                self.out_channels,
                stride=self.stride,
                dtype=self.dtype,
                name="0",
            )
        ]

        for i in range(self.depth - 1):
            layers.append(
                layer(
                    self.config,
                    self.out_channels,
                    self.out_channels,
                    dtype=self.dtype,
                    name=str(i + 1),
                )
            )

        self.layers = layers  # 存储所有层的列表

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        hidden_state = x
        for layer in self.layers:
            hidden_state = layer(hidden_state, deterministic=deterministic)
        return hidden_state


# Copied from transformers.models.resnet.modeling_flax_resnet.FlaxResNetStage with ResNet->RegNet
class FlaxRegNetStage(nn.Module):
    """
    A RegNet stage composed by stacked layers.
    """

    config: RegNetConfig  # 存储RegNet配置的对象
    in_channels: int  # 输入通道数
    out_channels: int  # 输出通道数
    stride: int = 2  # 步幅，默认为2
    depth: int = 2  # 层的深度，默认为2
    dtype: jnp.dtype = jnp.float32  # 数据类型，默认为32位浮点数

    def setup(self):
        self.layers = FlaxRegNetStageLayersCollection(
            self.config,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            stride=self.stride,
            depth=self.depth,
            dtype=self.dtype,
        )

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        return self.layers(x, deterministic=deterministic)


# Copied from transformers.models.resnet.modeling_flax_resnet.FlaxResNetStageCollection with ResNet->RegNet
class FlaxRegNetStageCollection(nn.Module):
    config: RegNetConfig  # 存储RegNet配置的对象
    dtype: jnp.dtype = jnp.float32  # 数据类型，默认为32位浮点数
    # 定义初始化方法，用于设置网络结构
    def setup(self):
        # 计算每个阶段输入输出通道数的元组列表
        in_out_channels = zip(self.config.hidden_sizes, self.config.hidden_sizes[1:])
        # 创建阶段列表，并添加第一个阶段的配置
        stages = [
            FlaxRegNetStage(
                self.config,
                self.config.embedding_size,
                self.config.hidden_sizes[0],
                stride=2 if self.config.downsample_in_first_stage else 1,
                depth=self.config.depths[0],
                dtype=self.dtype,
                name="0",
            )
        ]

        # 遍历计算后续阶段的配置并添加到阶段列表中
        for i, ((in_channels, out_channels), depth) in enumerate(zip(in_out_channels, self.config.depths[1:])):
            stages.append(
                FlaxRegNetStage(self.config, in_channels, out_channels, depth=depth, dtype=self.dtype, name=str(i + 1))
            )

        # 将创建好的阶段列表赋值给对象的stages属性
        self.stages = stages

    # 定义调用方法，实现模型的前向传播
    def __call__(
        self,
        hidden_state: jnp.ndarray,
        output_hidden_states: bool = False,
        deterministic: bool = True,
    ) -> FlaxBaseModelOutputWithNoAttention:
        # 如果需要输出隐藏状态，则初始化一个空元组用于存储隐藏状态
        hidden_states = () if output_hidden_states else None

        # 遍历每个阶段模块进行前向传播
        for stage_module in self.stages:
            # 如果需要输出隐藏状态，则将当前隐藏状态转置后添加到隐藏状态元组中
            if output_hidden_states:
                hidden_states = hidden_states + (hidden_state.transpose(0, 3, 1, 2),)

            # 调用当前阶段模块进行前向传播，更新隐藏状态
            hidden_state = stage_module(hidden_state, deterministic=deterministic)

        # 返回最终的隐藏状态和可能的隐藏状态元组
        return hidden_state, hidden_states
# 从 transformers.models.resnet.modeling_flax_resnet.FlaxResNetEncoder 复制而来，将 ResNet 修改为 RegNet
class FlaxRegNetEncoder(nn.Module):
    # 使用 RegNetConfig 类型的 config 属性
    config: RegNetConfig
    # 使用 jnp.float32 类型的 dtype 属性
    dtype: jnp.dtype = jnp.float32

    # 模块初始化方法
    def setup(self):
        # 使用 RegNetConfig 和 dtype 创建 FlaxRegNetStageCollection 实例
        self.stages = FlaxRegNetStageCollection(self.config, dtype=self.dtype)

    # 调用方法，接受 hidden_state, output_hidden_states, return_dict, deterministic 参数，并返回 FlaxBaseModelOutputWithNoAttention 类型的值
    def __call__(
        self,
        hidden_state: jnp.ndarray,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
    ) -> FlaxBaseModelOutputWithNoAttention:
        # 调用 self.stages 处理 hidden_state，并根据参数设置返回 hidden_state 和 hidden_states
        hidden_state, hidden_states = self.stages(
            hidden_state, output_hidden_states=output_hidden_states, deterministic=deterministic
        )

        # 如果 output_hidden_states 为真，则添加转置后的 hidden_state 到 hidden_states 元组中
        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state.transpose(0, 3, 1, 2),)

        # 如果 return_dict 为假，则返回非空的 hidden_state 和 hidden_states 元组
        if not return_dict:
            return tuple(v for v in [hidden_state, hidden_states] if v is not None)

        # 返回 FlaxBaseModelOutputWithNoAttention 实例，包含 last_hidden_state 和 hidden_states
        return FlaxBaseModelOutputWithNoAttention(
            last_hidden_state=hidden_state,
            hidden_states=hidden_states,
        )


# 从 transformers.models.resnet.modeling_flax_resnet.FlaxResNetPreTrainedModel 复制而来，将 ResNet 修改为 RegNet，resnet 修改为 regnet，RESNET 修改为 REGNET
class FlaxRegNetPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 使用 RegNetConfig 类型的 config_class 属性
    config_class = RegNetConfig
    # 使用 "regnet" 字符串的 base_model_prefix 属性
    base_model_prefix = "regnet"
    # 使用 "pixel_values" 字符串的 main_input_name 属性
    main_input_name = "pixel_values"
    # 使用 NoneType 的 module_class 属性
    module_class: nn.Module = None

    # 初始化方法，接受 config, input_shape, seed, dtype, _do_init 等参数
    def __init__(
        self,
        config: RegNetConfig,
        input_shape=(1, 224, 224, 3),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        # 使用 config, dtype, **kwargs 创建 module 实例
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 如果 input_shape 为 None，则使用 (1, config.image_size, config.image_size, config.num_channels)
        if input_shape is None:
            input_shape = (1, config.image_size, config.image_size, config.num_channels)
        # 调用父类的初始化方法，传递 config, module, input_shape, seed, dtype, _do_init 等参数
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    # 初始化权重方法，接受 rng, input_shape, params 等参数，并返回 FrozenDict 类型的值
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 创建 dtype 为 self.dtype 的 jnp.zeros 像素值数组 pixel_values
        pixel_values = jnp.zeros(input_shape, dtype=self.dtype)

        # 创建包含 rng 的 rngs 字典
        rngs = {"params": rng}

        # 使用 self.module.init 初始化随机参数 random_params，并设置 return_dict 为 False
        random_params = self.module.init(rngs, pixel_values, return_dict=False)

        # 如果 params 非空，则扁平化并合并 random_params 和 params 中缺失的键
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            # 否则返回 random_params
            return random_params

    # 将 REGNET_INPUTS_DOCSTRING 添加到模型前向传播方法的装饰器
    @add_start_docstrings_to_model_forward(REGNET_INPUTS_DOCSTRING)
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

        # 将输入的像素值进行维度转换，调整通道顺序为 (0, 2, 3, 1)
        pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))

        # 处理可能需要的伪随机数生成器
        rngs = {}

        # 调用模块的应用方法，传递参数和数据
        return self.module.apply(
            {
                "params": params["params"] if params is not None else self.params["params"],
                "batch_stats": params["batch_stats"] if params is not None else self.params["batch_stats"],
            },
            jnp.array(pixel_values, dtype=jnp.float32),  # 转换后的像素值数组
            not train,  # 训练模式取反，传递给模块的参数
            output_hidden_states,  # 是否返回隐藏状态的标志
            return_dict,  # 是否返回字典形式的输出
            rngs=rngs,  # 伪随机数生成器
            mutable=["batch_stats"] if train else False,  # 当训练为真时，返回包含 batch_stats 的元组
        )
# 从transformers.models.resnet.modeling_flax_resnet.FlaxResNetModule复制到此处，修改ResNet为RegNet
class FlaxRegNetModule(nn.Module):
    config: RegNetConfig  # 模型配置为RegNetConfig类型
    dtype: jnp.dtype = jnp.float32  # 计算中使用的数据类型为jnp.float32

    def setup(self):
        self.embedder = FlaxRegNetEmbeddings(self.config, dtype=self.dtype)
        self.encoder = FlaxRegNetEncoder(self.config, dtype=self.dtype)

        # 在ResNet中使用的自适应平均池化
        self.pooler = partial(
            nn.avg_pool,
            padding=((0, 0), (0, 0)),
        )

    def __call__(
        self,
        pixel_values,
        deterministic: bool = True,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> FlaxBaseModelOutputWithPoolingAndNoAttention:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        embedding_output = self.embedder(pixel_values, deterministic=deterministic)

        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        last_hidden_state = encoder_outputs[0]

        # 对最后隐藏状态进行自适应平均池化
        pooled_output = self.pooler(
            last_hidden_state,
            window_shape=(last_hidden_state.shape[1], last_hidden_state.shape[2]),
            strides=(last_hidden_state.shape[1], last_hidden_state.shape[2]),
        ).transpose(0, 3, 1, 2)

        last_hidden_state = last_hidden_state.transpose(0, 3, 1, 2)

        if not return_dict:
            # 如果不返回字典形式，则返回元组形式的输出
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 返回带池化和无注意力的基础模型输出的字典形式
        return FlaxBaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )


@add_start_docstrings(
    "The bare RegNet model outputting raw features without any specific head on top.",
    REGNET_START_DOCSTRING,
)
class FlaxRegNetModel(FlaxRegNetPreTrainedModel):
    module_class = FlaxRegNetModule


# FLAX_VISION_MODEL_DOCSTRING字符串文档
FLAX_VISION_MODEL_DOCSTRING = """
    Returns:

    Examples:

    ```python
    >>> from transformers import AutoImageProcessor, FlaxRegNetModel
    >>> from PIL import Image
    >>> import requests

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> image_processor = AutoImageProcessor.from_pretrained("facebook/regnet-y-040")
    >>> model = FlaxRegNetModel.from_pretrained("facebook/regnet-y-040")

    >>> inputs = image_processor(images=image, return_tensors="np")
    >>> outputs = model(**inputs)
    >>> last_hidden_states = outputs.last_hidden_state
    ```
"""

# 覆盖FlaxRegNetModel的调用文档字符串
overwrite_call_docstring(FlaxRegNetModel, FLAX_VISION_MODEL_DOCSTRING)
# 使用函数`append_replace_return_docstrings`设置FlaxRegNetModel的文档字符串，指定输出类型为FlaxBaseModelOutputWithPooling，
# 并使用RegNetConfig进行配置。

# 从`transformers.models.resnet.modeling_flax_resnet.FlaxResNetClassifierCollection`复制代码到`FlaxRegNetClassifierCollection`，
# 将ResNet更改为RegNet。这个类用于创建RegNet模型的分类器集合。

class FlaxRegNetClassifierCollection(nn.Module):
    # 使用RegNetConfig配置模型
    config: RegNetConfig
    # 默认数据类型为jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 模块设置方法
    def setup(self):
        # 创建具有config.num_labels输出的全连接层作为分类器，数据类型为dtype，名称为"1"
        self.classifier = nn.Dense(self.config.num_labels, dtype=self.dtype, name="1")

    # 调用方法，将输入x经过分类器处理后返回结果
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.classifier(x)


# 从`transformers.models.resnet.modeling_flax_resnet.FlaxResNetForImageClassificationModule`复制代码到`FlaxRegNetForImageClassificationModule`，
# 将ResNet更改为RegNet，同时修改resnet->regnet, RESNET->REGNET。这个类用于创建RegNet用于图像分类的模块。

class FlaxRegNetForImageClassificationModule(nn.Module):
    # 使用RegNetConfig配置模型
    config: RegNetConfig
    # 默认数据类型为jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 模块设置方法
    def setup(self):
        # 创建RegNet模块，使用给定的config和dtype
        self.regnet = FlaxRegNetModule(config=self.config, dtype=self.dtype)

        # 根据配置决定是否创建分类器集合或保持身份映射
        if self.config.num_labels > 0:
            self.classifier = FlaxRegNetClassifierCollection(self.config, dtype=self.dtype)
        else:
            self.classifier = Identity()

    # 调用方法，根据输入参数进行前向传播，返回分类结果
    def __call__(
        self,
        pixel_values=None,
        deterministic: bool = True,
        output_hidden_states=None,
        return_dict=None,
    ):
        # 根据参数设置返回字典的使用与否
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用RegNet模块进行前向传播
        outputs = self.regnet(
            pixel_values,
            deterministic=deterministic,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果使用返回字典，则获取池化后的输出；否则直接从输出中获取池化后的特征
        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        # 将池化后的特征输入到分类器中，获取最终的logits
        logits = self.classifier(pooled_output[:, :, 0, 0])

        # 如果不使用返回字典，则将logits与额外的隐藏状态一起返回
        if not return_dict:
            output = (logits,) + outputs[2:]
            return output

        # 使用FlaxImageClassifierOutputWithNoAttention将logits和隐藏状态输出
        return FlaxImageClassifierOutputWithNoAttention(logits=logits, hidden_states=outputs.hidden_states)


@add_start_docstrings(
    """
    用于在RegNet模型顶部添加图像分类头的模型，例如在ImageNet上使用线性层对池化特征进行分类。
    """,
    REGNET_START_DOCSTRING,
)
# 使用`add_start_docstrings`添加模型文档字符串，描述其用途和示例。
class FlaxRegNetForImageClassification(FlaxRegNetPreTrainedModel):
    # 模块类别设置为FlaxRegNetForImageClassificationModule
    module_class = FlaxRegNetForImageClassificationModule


FLAX_VISION_CLASSIF_DOCSTRING = """
    Returns:

    Example:

    ```python
    >>> from transformers import AutoImageProcessor, FlaxRegNetForImageClassification
    >>> from PIL import Image
    >>> import jax
    >>> import requests

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> image_processor = AutoImageProcessor.from_pretrained("facebook/regnet-y-040")
    >>> model = FlaxRegNetForImageClassification.from_pretrained("facebook/regnet-y-040")

    >>> inputs = image_processor(images=image, return_tensors="np")
    >>> outputs = model(**inputs)
    >>> logits = outputs.logits

    >>> # model predicts one of the 1000 ImageNet classes


# 注释：
# 此部分为文档字符串`FLAX_VISION_CLASSIF_DOCSTRING`，提供了该模型的返回值说明和使用示例。
    # 使用 JAX 提供的 numpy 模块计算 logits 中每个样本预测的类别索引
    predicted_class_idx = jax.numpy.argmax(logits, axis=-1)
    # 打印预测的类别，根据模型配置中的 id2label 映射将索引转换为标签名称并输出
    print("Predicted class:", model.config.id2label[predicted_class_idx.item()])
"""
覆盖调用函数的文档字符串为指定的文档字符串。
"""
overwrite_call_docstring(FlaxRegNetForImageClassification, FLAX_VISION_CLASSIF_DOCSTRING)

"""
向指定类追加或替换返回值文档字符串。
"""
append_replace_return_docstrings(
    FlaxRegNetForImageClassification,
    output_type=FlaxImageClassifierOutputWithNoAttention,
    config_class=RegNetConfig,
)
```