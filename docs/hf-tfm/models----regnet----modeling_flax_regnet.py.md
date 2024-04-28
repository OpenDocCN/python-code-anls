# `.\transformers\models\regnet\modeling_flax_regnet.py`

```py
# 导入所需的库
from functools import partial
from typing import Optional, Tuple
# 导入 flax 和 jax 库
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
# 导入其他相关的模型配置和输出
from transformers import RegNetConfig
from transformers.modeling_flax_outputs import (
    FlaxBaseModelOutputWithNoAttention,
    FlaxBaseModelOutputWithPooling,
    FlaxBaseModelOutputWithPoolingAndNoAttention,
    FlaxImageClassifierOutputWithNoAttention,
)
from transformers.modeling_flax_utils import (
    ACT2FN,
    FlaxPreTrainedModel,
    append_replace_return_docstrings,
    overwrite_call_docstring,
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)

# 定义模型的起始文档字符串
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
    # 参数：
    # config：[`RegNetConfig`]，模型配置类，包含模型的所有参数。使用配置文件初始化时不会加载与模型关联的权重，只会加载配置。查看 [`~FlaxPreTrainedModel.from_pretrained`] 方法以加载模型权重。
    # dtype：`jax.numpy.dtype`，*可选*，默认为 `jax.numpy.float32`
    # 计算的数据类型。可以是 `jax.numpy.float32`、`jax.numpy.float16`（在 GPU 上）、`jax.numpy.bfloat16`（在 TPU 上）之一。
    # 这可用于在 GPU 或 TPU 上启用混合精度训练或半精度推理。如果指定，则所有计算将使用给定的 `dtype` 执行。
    # **请注意，这仅指定了计算的数据类型，不会影响模型参数的数据类型。**
    # 如果要更改模型参数的数据类型，请参阅 [`~FlaxPreTrainedModel.to_fp16`] 和 [`~FlaxPreTrainedModel.to_bf16`]。
# 此处定义了一个文档字符串，用于描述函数的输入参数和用法
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

# 定义一个名为 Identity 的神经网络模块，用于实现恒等映射函数
class Identity(nn.Module):
    """Identity function."""

    @nn.compact
    def __call__(self, x, **kwargs):
        return x

# 定义一个名为 FlaxRegNetConvLayer 的神经网络模块，用于实现 RegNet 中的卷积层
class FlaxRegNetConvLayer(nn.Module):
    out_channels: int
    kernel_size: int = 3
    stride: int = 1
    groups: int = 1
    activation: Optional[str] = "relu"
    dtype: jnp.dtype = jnp.float32

    # 初始化函数，设置网络结构
    def setup(self):
        # 定义卷积层
        self.convolution = nn.Conv(
            self.out_channels,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=self.stride,
            padding=self.kernel_size // 2,
            feature_group_count=self.groups,
            use_bias=False,
            # 使用截断正态分布初始化卷积核参数
            kernel_init=nn.initializers.variance_scaling(2.0, mode="fan_out", distribution="truncated_normal"),
            dtype=self.dtype,
        )
        # 定义批标准化层
        self.normalization = nn.BatchNorm(momentum=0.9, epsilon=1e-05, dtype=self.dtype)
        # 定义激活函数
        self.activation_func = ACT2FN[self.activation] if self.activation is not None else Identity()

    # 正向传播函数，实现卷积、标准化和激活操作
    def __call__(self, hidden_state: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # 卷积操作
        hidden_state = self.convolution(hidden_state)
        # 批标准化操作
        hidden_state = self.normalization(hidden_state, use_running_average=deterministic)
        # 激活操作
        hidden_state = self.activation_func(hidden_state)
        return hidden_state

# 定义一个名为 FlaxRegNetEmbeddings 的神经网络模块，用于实现 RegNet 中的嵌入层
class FlaxRegNetEmbeddings(nn.Module):
    config: RegNetConfig
    dtype: jnp.dtype = jnp.float32

    # 初始化函数，设置网络结构
    def setup(self):
        # 定义嵌入器，即一个卷积层
        self.embedder = FlaxRegNetConvLayer(
            self.config.embedding_size,
            kernel_size=3,
            stride=2,
            activation=self.config.hidden_act,
            dtype=self.dtype,
        )

    # 正向传播函数，用于实现嵌入操作
    def __call__(self, pixel_values: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        num_channels = pixel_values.shape[-1]
        # 检查输入像素值的通道数是否与配置中指定的一致
        if num_channels != self.config.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # 执行嵌入操作
        hidden_state = self.embedder(pixel_values, deterministic=deterministic)
        return hidden_state

# 定义一个名为 FlaxRegNetShortCut 的神经网络模块，用于实现 RegNet 中的跳跃连接
class FlaxRegNetShortCut(nn.Module):
    """
    # 一个 RegNet 的快捷连接块(shortcut)，用于将输入特征投射到正确的尺寸。
    # 如果需要的话，它也可以用 `stride=2` 来下采样输入。
    class RegNetShortcut(nn.Module):
        # 输出通道数
        out_channels: int
        # 步长，默认为 2
        stride: int = 2
        # 数据类型，默认为 float32
        dtype: jnp.dtype = jnp.float32
    
        def setup(self):
            # 创建一个 1x1 卷积层，用于通道数转换和下采样
            self.convolution = nn.Conv(
                self.out_channels,
                kernel_size=(1, 1),
                strides=self.stride,
                use_bias=False,
                # 使用合适的卷积核初始化方式
                kernel_init=nn.initializers.variance_scaling(2.0, mode="fan_out", distribution="truncated_normal"),
                dtype=self.dtype,
            )
            # 创建批归一化层
            self.normalization = nn.BatchNorm(momentum=0.9, epsilon=1e-05, dtype=self.dtype)
    
        def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
            # 通过卷积和批归一化处理输入特征
            hidden_state = self.convolution(x)
            hidden_state = self.normalization(hidden_state, use_running_average=deterministic)
            # 返回处理后的特征
            return hidden_state
class FlaxRegNetSELayerCollection(nn.Module):
    # 定义一个类用于包含Squeeze and Excitation(SE)层
    in_channels: int
    reduced_channels: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 初始化第一个卷积层，用于减少通道数
        self.conv_1 = nn.Conv(
            self.reduced_channels,
            kernel_size=(1, 1),
            kernel_init=nn.initializers.variance_scaling(2.0, mode="fan_out", distribution="truncated_normal"),
            dtype=self.dtype,
            name="0",
        )  # 0 is the name used in corresponding pytorch implementation
        # 初始化第二个卷积层，用于还原通道数
        self.conv_2 = nn.Conv(
            self.in_channels,
            kernel_size=(1, 1),
            kernel_init=nn.initializers.variance_scaling(2.0, mode="fan_out", distribution="truncated_normal"),
            dtype=self.dtype,
            name="2",
        )  # 2 is the name used in corresponding pytorch implementation

    def __call__(self, hidden_state: jnp.ndarray) -> jnp.ndarray:
        # 对隐藏状态应用第一个卷积层
        hidden_state = self.conv_1(hidden_state)
        # 使用ReLU激活函数
        hidden_state = nn.relu(hidden_state)
        # 对隐藏状态应用第二个卷积层
        hidden_state = self.conv_2(hidden_state)
        # 使用Sigmoid函数求得注意力
        attention = nn.sigmoid(hidden_state)

        return attention


class FlaxRegNetSELayer(nn.Module):
    # 定义Squeeze and Excitation (SE)层
    """
    Squeeze and Excitation layer (SE) proposed in [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507).
    """

    in_channels: int
    reduced_channels: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 初始化池化函数
        self.pooler = partial(nn.avg_pool, padding=((0, 0), (0, 0)))
        # 初始化SE集合
        self.attention = FlaxRegNetSELayerCollection(self.in_channels, self.reduced_channels, dtype=self.dtype)

    def __call__(self, hidden_state: jnp.ndarray) -> jnp.ndarray:
        # 对隐藏状态进行池化
        pooled = self.pooler(
            hidden_state,
            window_shape=(hidden_state.shape[1], hidden_state.shape[2]),
            strides=(hidden_state.shape[1], hidden_state.shape[2]),
        )
        # 在池化后应用SE集合
        attention = self.attention(pooled)
        # 通过注意力调整隐藏状态
        hidden_state = hidden_state * attention
        return hidden_state


class FlaxRegNetXLayerCollection(nn.Module):
    # 定义一个类用于包含RegNet X层的不同卷积层
    config: RegNetConfig
    out_channels: int
    stride: int = 1
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 确定组数
        groups = max(1, self.out_channels // self.config.groups_width)

        # 创建包含不同卷积层的列表
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
    # 定义一个 __call__ 方法，接收隐藏状态和一个布尔值参数，并返回隐藏状态
    def __call__(self, hidden_state: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # 遍历神经网络的每一层
        for layer in self.layer:
            # 根据每一层的权重和隐藏状态计算新的隐藏状态
            hidden_state = layer(hidden_state, deterministic=deterministic)
        # 返回经过所有层处理后的最终隐藏状态
        return hidden_state
class FlaxRegNetXLayer(nn.Module):
    """
    RegNet的层，由三个3x3的卷积组成，与具有reduction=1的ResNet瓶颈层相同。
    """

    config: RegNetConfig  # RegNet的配置
    in_channels: int  # 输入通道数
    out_channels: int  # 输出通道数
    stride: int = 1  # 步长，默认为1
    dtype: jnp.dtype = jnp.float32  # 数据类型，默认为jnp.float32

    def setup(self):
        # 是否应用shortcut取决于输入通道数和输出通道数是否相等，以及步长是否为1
        should_apply_shortcut = self.in_channels != self.out_channels or self.stride != 1
        # 如果应用shortcut，则创建FlaxRegNetShortCut对象，否则创建Identity对象
        self.shortcut = (
            FlaxRegNetShortCut(
                self.out_channels,
                stride=self.stride,
                dtype=self.dtype,
            )
            if should_apply_shortcut
            else Identity()
        )
        # 创建FlaxRegNetXLayerCollection对象
        self.layer = FlaxRegNetXLayerCollection(
            self.config,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            stride=self.stride,
            dtype=self.dtype,
        )
        # 激活函数为配置中指定的隐藏激活函数
        self.activation_func = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_state: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        residual = hidden_state  # 保存输入状态作为残差连接
        hidden_state = self.layer(hidden_state)  # 通过卷积层
        residual = self.shortcut(residual, deterministic=deterministic)  # 应用shortcut
        hidden_state += residual  # 残差连接
        hidden_state = self.activation_func(hidden_state)  # 应用激活函数
        return hidden_state  # 返回处理后的隐藏状态


class FlaxRegNetYLayerCollection(nn.Module):
    config: RegNetConfig  # RegNet的配置
    in_channels: int  # 输入通道数
    out_channels: int  # 输出通道数
    stride: int = 1  # 步长，默认为1
    dtype: jnp.dtype = jnp.float32  # 数据类型，默认为jnp.float32

    def setup(self):
        # 计算分组数
        groups = max(1, self.out_channels // self.config.groups_width)

        # 创建包含卷积层、SE层的列表
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
            FlaxRegNetSELayer(
                self.out_channels,
                reduced_channels=int(round(self.in_channels / 4)),
                dtype=self.dtype,
                name="2",
            ),
            FlaxRegNetConvLayer(
                self.out_channels,
                kernel_size=1,
                activation=None,
                dtype=self.dtype,
                name="3",
            ),
        ]

    def __call__(self, hidden_state: jnp.ndarray) -> jnp.ndarray:
        # 遍历层列表，逐层处理隐藏状态
        for layer in self.layer:
            hidden_state = layer(hidden_state)
        return hidden_state  # 返回处理后的隐藏状态


class FlaxRegNetYLayer(nn.Module):
    """
    RegNet的Y层：带有Squeeze and Excitation的X层。
    """

    config: RegNetConfig  # RegNet的配置
    in_channels: int  # 输入通道数
    out_channels: int  # 输出通道数
    stride: int = 1  # 步长，默认为1
    dtype: jnp.dtype = jnp.float32  # 数据类型，默认为jnp.float32
    def setup(self):
        # 判断是否需要应用快捷方式，判断条件是输入通道数不等于输出通道数，或者步长不等于1
        should_apply_shortcut = self.in_channels != self.out_channels or self.stride != 1

        # 如果需要应用快捷方式，则创建一个FlaxRegNetShortCut对象，并赋值给self.shortcut
        # 否则创建一个Identity对象，并赋值给self.shortcut
        self.shortcut = (
            FlaxRegNetShortCut(
                self.out_channels,
                stride=self.stride,
                dtype=self.dtype,
            )
            if should_apply_shortcut
            else Identity()
        )
        
        # 创建一个FlaxRegNetYLayerCollection对象，并赋值给self.layer
        self.layer = FlaxRegNetYLayerCollection(
            self.config,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            stride=self.stride,
            dtype=self.dtype,
        )
        
        # 根据配置中的隐藏激活函数名获取对应的函数，并赋值给self.activation_func
        self.activation_func = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_state: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # 将hidden_state保存为residual
        residual = hidden_state
        # 将hidden_state输入到FlaxRegNetYLayerCollection中进行计算，并重新赋值给hidden_state
        hidden_state = self.layer(hidden_state)
        # 将residual输入到shortcut中进行计算，并重新赋值给residual
        residual = self.shortcut(residual, deterministic=deterministic)
        # 将hidden_state和residual相加，并重新赋值给hidden_state
        hidden_state += residual
        # 将hidden_state输入到激活函数中进行计算，并将结果返回
        hidden_state = self.activation_func(hidden_state)
        return hidden_state
# 定义一个由堆叠层组成的 RegNet 阶段
class FlaxRegNetStageLayersCollection(nn.Module):
    """
    由堆叠层组成的 RegNet 阶段。
    """

    # RegNet 配置
    config: RegNetConfig
    # 输入通道数
    in_channels: int
    # 输出通道数
    out_channels: int
    # 步长，默认为2
    stride: int = 2
    # 层数，默认为2
    depth: int = 2
    # 数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 设置函数
    def setup(self):
        # 根据配置选择不同的层类型
        layer = FlaxRegNetXLayer if self.config.layer_type == "x" else FlaxRegNetYLayer

        layers = [
            # 第一层进行下采样，步长为2
            layer(
                self.config,
                self.in_channels,
                self.out_channels,
                stride=self.stride,
                dtype=self.dtype,
                name="0",
            )
        ]

        # 根据深度堆叠层
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

        # 将层保存到对象中
        self.layers = layers

    # 调用函数
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # 初始化隐藏状态为输入
        hidden_state = x
        # 遍历所有层，将隐藏状态传递给每一层
        for layer in self.layers:
            hidden_state = layer(hidden_state, deterministic=deterministic)
        # 返回最终隐藏状态
        return hidden_state


# 从 transformers.models.resnet.modeling_flax_resnet.FlaxResNetStage 复制并修改为 RegNet
class FlaxRegNetStage(nn.Module):
    """
    由堆叠层组成的 RegNet 阶段。
    """

    # RegNet 配置
    config: RegNetConfig
    # 输入通道数
    in_channels: int
    # 输出通道数
    out_channels: int
    # 步长，默认为2
    stride: int = 2
    # 层数，默认为2
    depth: int = 2
    # 数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 设置函数
    def setup(self):
        # 初始化层对象
        self.layers = FlaxRegNetStageLayersCollection(
            # 传递配置
            self.config,
            # 传递输入通道数
            in_channels=self.in_channels,
            # 传递输出通道数
            out_channels=self.out_channels,
            # 传递步长
            stride=self.stride,
            # 传递层数
            depth=self.depth,
            # 传递数据类型
            dtype=self.dtype,
        )

    # 调用函数
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # 调用层对象，传递输入数据
        return self.layers(x, deterministic=deterministic)


# 从 transformers.models.resnet.modeling_flax_resnet.FlaxResNetStageCollection 复制并修改为 RegNet
class FlaxRegNetStageCollection(nn.Module):
    # RegNet 配置
    config: RegNetConfig
    # 数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 定义一个setup方法，在其中初始化模型的各个阶段
    def setup(self):
        # 定义输入和输出通道，zip函数将hidden_sizes列表中的相邻两个元素组成一个元组
        in_out_channels = zip(self.config.hidden_sizes, self.config.hidden_sizes[1:])
        # 创建一个用于存储模型阶段的列表，并将第一个阶段添加到其中
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

        # 遍历输入输出通道和深度列表，并将对应的阶段添加到列表中
        for i, ((in_channels, out_channels), depth) in enumerate(zip(in_out_channels, self.config.depths[1:])):
            stages.append(
                FlaxRegNetStage(self.config, in_channels, out_channels, depth=depth, dtype=self.dtype, name=str(i + 1))
            )

        # 将创建好的阶段列表赋值给类的stages属性
        self.stages = stages

    # 定义一个__call__方法，用于模型的前向传播
    def __call__(
        self,
        hidden_state: jnp.ndarray,
        output_hidden_states: bool = False,
        deterministic: bool = True,
    ) -> FlaxBaseModelOutputWithNoAttention:
        # 如果需要输出隐藏状态，则创建一个空的元组用于存储隐藏状态
        hidden_states = () if output_hidden_states else None

        # 遍历所有阶段模块，对输入的隐藏状态进行处理
        for stage_module in self.stages:
            # 如果需要输出隐藏状态，则将当前隐藏状态加入到隐藏状态元组中
            if output_hidden_states:
                hidden_states = hidden_states + (hidden_state.transpose(0, 3, 1, 2),)

            # 对输入的隐藏状态使用当前阶段模块进行处理
            hidden_state = stage_module(hidden_state, deterministic=deterministic)

        # 返回处理后的隐藏状态以及隐藏状态元组（如果需要输出隐藏状态）
        return hidden_state, hidden_states
# 从transformers.models.resnet.modeling_flax_resnet.FlaxResNetEncoder复制而来，将ResNet修改为RegNet
class FlaxRegNetEncoder(nn.Module):
    # 设置类属性config为RegNetConfig，dtype为jnp.float32
    config: RegNetConfig
    dtype: jnp.dtype = jnp.float32

    # 定义初始化方法
    def setup(self):
        # 创建FlaxRegNetStageCollection对象，并传入config和dtype
        self.stages = FlaxRegNetStageCollection(self.config, dtype=self.dtype)

    # 定义调用方法
    def __call__(
        self,
        hidden_state: jnp.ndarray,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
    ) -> FlaxBaseModelOutputWithNoAttention:
        # 调用self.stages，处理hidden_state，根据output_hidden_states和deterministic决定是否返回hidden_states
        hidden_state, hidden_states = self.stages(
            hidden_state, output_hidden_states=output_hidden_states, deterministic=deterministic
        )

        # 如果需要输出hidden_states，则将hidden_state进行转置后加到hidden_states中
        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state.transpose(0, 3, 1, 2),)

        # 如果不需要返回字典，则返回不为None的值的元组
        if not return_dict:
            return tuple(v for v in [hidden_state, hidden_states] if v is not None)

        # 返回FlaxBaseModelOutputWithNoAttention对象，包含last_hidden_state和hidden_states
        return FlaxBaseModelOutputWithNoAttention(
            last_hidden_state=hidden_state,
            hidden_states=hidden_states,
        )


# 从transformers.models.resnet.modeling_flax_resnet.FlaxResNetPreTrainedModel复制而来，将ResNet修改为RegNet，resnet修改为regnet，RESNET修改为REGNET
class FlaxRegNetPreTrainedModel(FlaxPreTrainedModel):
    """
    处理权重初始化、预训练模型下载和加载的抽象类。
    """

    # 设置配置类为RegNetConfig，基础模型前缀为regnet，主输入名称为pixel_values，模块类为None
    config_class = RegNetConfig
    base_model_prefix = "regnet"
    main_input_name = "pixel_values"
    module_class: nn.Module = None

    # 初始化方法
    def __init__(
        self,
        config: RegNetConfig,
        input_shape=(1, 224, 224, 3),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        # 创建模块对象，传入config、dtype和其他关键字参数
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 如果input_shape为None，则设置为默认值
        if input_shape is None:
            input_shape = (1, config.image_size, config.image_size, config.num_channels)
        # 调用父类初始化方法
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    # 初始化权重方法
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量pixel_values为全零张量
        pixel_values = jnp.zeros(input_shape, dtype=self.dtype)

        # 创建随机数生成器字典
        rngs = {"params": rng}

        # 使用模块的init方法初始化参数
        random_params = self.module.init(rngs, pixel_values, return_dict=False)

        # 如果存在已有参数，则使用已有参数替换随机初始化的参数
        if params is not None:
            # 展平参数字典
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            # 遍历已有参数字典，将缺失的参数用随机初始化的参数替换
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            # 冻结参数字典并返回
            return freeze(unflatten_dict(params))
        else:
            # 返回随机初始化的参数字典
            return random_params

    # 将输入参数注释添加到模型前向方法中
    @add_start_docstrings_to_model_forward(REGNET_INPUTS_DOCSTRING)
    def __call__(
        self,
        pixel_values,
        params: dict = None,
        train: bool = False,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 将输入的像素值按照维度转置为 (batch_size, height, width, channels) 的形式
        pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))

        # 处理需要的伪随机数生成器
        rngs = {}

        return self.module.apply(
            {
                # 如果传入了自定义参数，则使用传入的参数；否则使用初始化的参数
                "params": params["params"] if params is not None else self.params["params"],
                # 如果传入了批次统计数据，则使用传入的批次统计数据；否则使用初始化的批次统计数据
                "batch_stats": params["batch_stats"] if params is not None else self.params["batch_stats"],
            },
            # 将像素值转换为 jnp.float32 类型的数组，并传入模型进行前向计算
            jnp.array(pixel_values, dtype=jnp.float32),
            # 如果是训练模式，则传入参数 "False"，如果是推理模式，则传入参数 "True"
            not train,
            # 如果传入了 output_hidden_states 参数，则使用传入的参数，否则使用模型的配置中的参数
            output_hidden_states,
            # 如果传入了 return_dict 参数，则使用传入的参数，否则使用模型的配置中的参数
            return_dict,
            # 传入需要的伪随机数生成器
            rngs=rngs,
            # 如果是训练模式，则将 batch_stats 设置为可变，只在训练时返回包含 batch_stats 的元组，否则设置为 False
            mutable=["batch_stats"] if train else False,
        )
# 从transformers.models.resnet.modeling_flax_resnet.FlaxResNetModule复制并将ResNet->RegNet
class FlaxRegNetModule(nn.Module):
    config: RegNetConfig  # RegNet配置
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型

    def setup(self):  # 设置函数
        self.embedder = FlaxRegNetEmbeddings(self.config, dtype=self.dtype)  # 设置嵌入器
        self.encoder = FlaxRegNetEncoder(self.config, dtype=self.dtype)  # 设置编码器

        # 用于ResNet的自适应平均池化
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
    ) -> FlaxBaseModelOutputWithPoolingAndNoAttention:  # 返回特征和无关注的基础模型输出
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        embedding_output = self.embedder(pixel_values, deterministic=deterministic)  # 使用嵌入器

        encoder_outputs = self.encoder(  # 编码器输出
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        last_hidden_state = encoder_outputs[0]  # 最后的隐藏状态

        pooled_output = self.pooler(  # 池化输出
            last_hidden_state,
            window_shape=(last_hidden_state.shape[1], last_hidden_state.shape[2]),
            strides=(last_hidden_state.shape[1], last_hidden_state.shape[2]),
        ).transpose(0, 3, 1, 2)

        last_hidden_state = last_hidden_state.transpose(0, 3, 1, 2)

        if not return_dict:  # 如果不返回字典
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return FlaxBaseModelOutputWithPoolingAndNoAttention(  # 返回基础模型带池化和无注意力的输出
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )


@add_start_docstrings(  # 添加起始文档字符串
    "The bare RegNet model outputting raw features without any specific head on top.",  # 输出原始特征的裸RegNet模型
    REGNET_START_DOCSTRING,  # RegNet起始文档字符串
)
class FlaxRegNetModel(FlaxRegNetPreTrainedModel):  # FlaxRegNet模型类
    module_class = FlaxRegNetModule  # 模块类为FlaxRegNetModule


FLAX_VISION_MODEL_DOCSTRING = """
    Returns:  # 返回

    Examples:  # 示例

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
    ```py
"""

overwrite_call_docstring(FlaxRegNetModel, FLAX_VISION_MODEL_DOCSTRING)  # 覆盖调用文档字符串
# 将 FlaxRegNetModel 的基类与输出类型、配置类添加到它的文档中
append_replace_return_docstrings(
    FlaxRegNetModel,
    output_type=FlaxBaseModelOutputWithPooling,
    config_class=RegNetConfig,
)

# 定义一个集成了分类器的 RegNet 模型
class FlaxRegNetClassifierCollection(nn.Module):
    # 从配置中读取分类器的输出标签数量
    config: RegNetConfig
    # 设置数据类型为默认的 float32
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 创建一个全连接层作为分类器
        self.classifier = nn.Dense(self.config.num_labels, dtype=self.dtype, name="1")

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # 对输入 x 应用分类器,返回分类结果
        return self.classifier(x)

# 定义一个由 RegNet 模型和分类器组成的复合模型
class FlaxRegNetForImageClassificationModule(nn.Module):
    # 从配置中读取模型配置
    config: RegNetConfig
    # 设置数据类型为默认的 float32
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 创建 RegNet 模型
        self.regnet = FlaxRegNetModule(config=self.config, dtype=self.dtype)

        # 如果配置中有分类标签数量,则创建分类器
        if self.config.num_labels > 0:
            self.classifier = FlaxRegNetClassifierCollection(self.config, dtype=self.dtype)
        # 否则使用恒等映射作为分类器
        else:
            self.classifier = Identity()

    def __call__(
        self,
        pixel_values=None,
        deterministic: bool = True,
        output_hidden_states=None,
        return_dict=None,
    ):
        # 设置返回字典的选项
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 传入像素值,获取 RegNet 模型的输出
        outputs = self.regnet(
            pixel_values,
            deterministic=deterministic,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 取出池化后的输出
        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        # 将池化输出传入分类器,获得分类结果
        logits = self.classifier(pooled_output[:, :, 0, 0])

        # 根据是否返回字典,组装最终输出
        if not return_dict:
            output = (logits,) + outputs[2:]
            return output

        return FlaxImageClassifierOutputWithNoAttention(logits=logits, hidden_states=outputs.hidden_states)

# 定义一个 RegNet 的图像分类模型
@add_start_docstrings(
    """
    RegNet Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    REGNET_START_DOCSTRING,
)
class FlaxRegNetForImageClassification(FlaxRegNetPreTrainedModel):
    module_class = FlaxRegNetForImageClassificationModule
    # 使用 JAX 库的 argmax 函数，沿着最后一个维度(axis=-1)返回 logits 中的最大值的索引，即预测的类别索引
    predicted_class_idx = jax.numpy.argmax(logits, axis=-1)
    # 打印预测的类别，根据索引在模型配置中查找对应的标签并打印
    print("Predicted class:", model.config.id2label[predicted_class_idx.item()])
"""
# 使用指定的文档字符串覆盖FlaxRegNetForImageClassification类的文档字符串
overwrite_call_docstring(FlaxRegNetForImageClassification, FLAX_VISION_CLASSIF_DOCSTRING)
# 替换FlaxRegNetForImageClassification类的返回文档字符串，使用指定的输出类型和配置类
append_replace_return_docstrings(
    FlaxRegNetForImageClassification,
    output_type=FlaxImageClassifierOutputWithNoAttention,
    config_class=RegNetConfig,
)
```