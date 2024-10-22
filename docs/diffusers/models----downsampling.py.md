# `.\diffusers\models\downsampling.py`

```py
# 版权声明，标识此文件的版权归 HuggingFace 团队所有
# 
# 根据 Apache 许可证第 2.0 版（“许可证”）授权；
# 除非遵守该许可证，否则您不得使用此文件。
# 您可以在以下网址获得许可证的副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律或书面同意，否则根据许可证分发的软件是“按原样”分发的，
# 不提供任何形式的保证或条件，无论是明示或暗示的。
# 请参阅许可证以获取特定的语言治理权限和
# 限制条款。

# 导入可选类型和元组类型
from typing import Optional, Tuple

# 导入 PyTorch 库
import torch
# 导入 PyTorch 的神经网络模块
import torch.nn as nn
# 导入 PyTorch 的功能模块
import torch.nn.functional as F

# 导入工具函数，可能用于标记已弃用的功能
from ..utils import deprecate
# 从 normalization 模块导入 RMSNorm 类
from .normalization import RMSNorm
# 从 upsampling 模块导入 upfirdn2d_native 函数
from .upsampling import upfirdn2d_native


class Downsample1D(nn.Module):
    """1D 下采样层，支持可选的卷积操作。

    参数：
        channels (`int`):
            输入和输出的通道数。
        use_conv (`bool`, default `False`):
            是否使用卷积。
        out_channels (`int`, optional):
            输出通道数，默认为 `channels`。
        padding (`int`, default `1`):
            卷积的填充大小。
        name (`str`, default `conv`):
            下采样 1D 层的名称。
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        padding: int = 1,
        name: str = "conv",
    ):
        # 调用父类初始化方法
        super().__init__()
        # 保存输入通道数
        self.channels = channels
        # 设置输出通道数，如果未指定则默认为输入通道数
        self.out_channels = out_channels or channels
        # 保存是否使用卷积的标志
        self.use_conv = use_conv
        # 保存卷积的填充大小
        self.padding = padding
        # 设置步幅为 2
        stride = 2
        # 保存层的名称
        self.name = name

        # 如果选择使用卷积，则初始化卷积层
        if use_conv:
            self.conv = nn.Conv1d(self.channels, self.out_channels, 3, stride=stride, padding=padding)
        # 否则，进行平均池化操作，并确保输入和输出通道数相同
        else:
            assert self.channels == self.out_channels
            self.conv = nn.AvgPool1d(kernel_size=stride, stride=stride)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # 确保输入的通道数与预期的通道数相同
        assert inputs.shape[1] == self.channels
        # 返回经过下采样的结果
        return self.conv(inputs)


class Downsample2D(nn.Module):
    """2D 下采样层，支持可选的卷积操作。

    参数：
        channels (`int`):
            输入和输出的通道数。
        use_conv (`bool`, default `False`):
            是否使用卷积。
        out_channels (`int`, optional):
            输出通道数，默认为 `channels`。
        padding (`int`, default `1`):
            卷积的填充大小。
        name (`str`, default `conv`):
            下采样 2D 层的名称。
    """
    # 初始化方法，设置网络层的参数
        def __init__(
            self,
            channels: int,  # 输入通道数
            use_conv: bool = False,  # 是否使用卷积层的标志
            out_channels: Optional[int] = None,  # 输出通道数（默认为输入通道数）
            padding: int = 1,  # 填充大小
            name: str = "conv",  # 层名称
            kernel_size=3,  # 卷积核大小
            norm_type=None,  # 归一化类型
            eps=None,  # 小常数，用于数值稳定性
            elementwise_affine=None,  # 是否使用逐元素仿射变换
            bias=True,  # 是否使用偏置
        ):
            super().__init__()  # 调用父类构造函数
            self.channels = channels  # 保存输入通道数
            self.out_channels = out_channels or channels  # 输出通道数，如果未提供则默认为输入通道数
            self.use_conv = use_conv  # 保存是否使用卷积层的标志
            self.padding = padding  # 保存填充大小
            stride = 2  # 设置步幅为2
            self.name = name  # 保存层名称
    
            # 根据归一化类型初始化归一化层
            if norm_type == "ln_norm":
                self.norm = nn.LayerNorm(channels, eps, elementwise_affine)  # 使用层归一化
            elif norm_type == "rms_norm":
                self.norm = RMSNorm(channels, eps, elementwise_affine)  # 使用 RMS 归一化
            elif norm_type is None:
                self.norm = None  # 不使用归一化
            else:
                raise ValueError(f"unknown norm_type: {norm_type}")  # 抛出未知归一化类型的错误
    
            # 根据是否使用卷积层初始化卷积操作
            if use_conv:
                conv = nn.Conv2d(
                    self.channels, self.out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
                )  # 创建卷积层
            else:
                assert self.channels == self.out_channels  # 确保输入通道数等于输出通道数
                conv = nn.AvgPool2d(kernel_size=stride, stride=stride)  # 创建平均池化层
    
            # 根据层名称设置卷积层的别名
            if name == "conv":
                self.Conv2d_0 = conv  # 保存卷积层到属性 Conv2d_0
                self.conv = conv  # 保存卷积层到属性 conv
            elif name == "Conv2d_0":
                self.conv = conv  # 保存卷积层到属性 conv
            else:
                self.conv = conv  # 保存卷积层到属性 conv
    
        # 前向传播方法，处理输入的隐藏状态
        def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            # 检查额外参数或过时参数是否存在
            if len(args) > 0 or kwargs.get("scale", None) is not None:
                deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
                deprecate("scale", "1.0.0", deprecation_message)  # 发出过时警告
            assert hidden_states.shape[1] == self.channels  # 确保输入通道数与预期一致
    
            # 如果存在归一化层，则应用归一化
            if self.norm is not None:
                hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # 归一化并重新排列维度
    
            # 如果使用卷积且填充为0，则进行填充
            if self.use_conv and self.padding == 0:
                pad = (0, 1, 0, 1)  # 定义填充大小
                hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)  # 执行填充操作
    
            assert hidden_states.shape[1] == self.channels  # 确保输入通道数与预期一致
    
            hidden_states = self.conv(hidden_states)  # 应用卷积或池化操作
    
            return hidden_states  # 返回处理后的隐藏状态
# 定义一个2D FIR下采样层，继承自 nn.Module
class FirDownsample2D(nn.Module):
    """一个可选卷积的2D FIR下采样层。

    参数：
        channels (`int`):
            输入和输出的通道数。
        use_conv (`bool`, default `False`):
            是否使用卷积的选项。
        out_channels (`int`, optional):
            输出通道数，默认为 `channels`。
        fir_kernel (`tuple`, default `(1, 3, 3, 1)`):
            FIR滤波器的核。
    """

    # 初始化方法，设置层的参数
    def __init__(
        self,
        channels: Optional[int] = None,  # 输入通道数
        out_channels: Optional[int] = None,  # 输出通道数
        use_conv: bool = False,  # 是否使用卷积
        fir_kernel: Tuple[int, int, int, int] = (1, 3, 3, 1),  # FIR核
    ):
        super().__init__()  # 调用父类的初始化方法
        # 如果未指定输出通道数，则使用输入通道数
        out_channels = out_channels if out_channels else channels
        # 如果选择使用卷积，定义卷积层
        if use_conv:
            self.Conv2d_0 = nn.Conv2d(channels, out_channels, kernel_size=3, stride=1, padding=1)
        # 设置 FIR 核
        self.fir_kernel = fir_kernel
        # 保存是否使用卷积的标志
        self.use_conv = use_conv
        # 保存输出通道数
        self.out_channels = out_channels

    # 定义一个私有方法用于2D下采样
    def _downsample_2d(
        self,
        hidden_states: torch.Tensor,  # 输入的张量
        weight: Optional[torch.Tensor] = None,  # 可选的权重张量
        kernel: Optional[torch.Tensor] = None,  # 可选的核张量
        factor: int = 2,  # 下采样因子，默认为2
        gain: float = 1,  # 增益，默认为1
    ) -> torch.Tensor:
        """Fused `Conv2d()` followed by `downsample_2d()`.
        Padding is performed only once at the beginning, not between the operations. The fused op is considerably more
        efficient than performing the same calculation using standard TensorFlow ops. It supports gradients of
        arbitrary order.

        Args:
            hidden_states (`torch.Tensor`):
                Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`.
            weight (`torch.Tensor`, *optional*):
                Weight tensor of the shape `[filterH, filterW, inChannels, outChannels]`. Grouped convolution can be
                performed by `inChannels = x.shape[0] // numGroups`.
            kernel (`torch.Tensor`, *optional*):
                FIR filter of the shape `[firH, firW]` or `[firN]` (separable). The default is `[1] * factor`, which
                corresponds to average pooling.
            factor (`int`, *optional*, default to `2`):
                Integer downsampling factor.
            gain (`float`, *optional*, default to `1.0`):
                Scaling factor for signal magnitude.

        Returns:
            output (`torch.Tensor`):
                Tensor of the shape `[N, C, H // factor, W // factor]` or `[N, H // factor, W // factor, C]`, and same
                datatype as `x`.
        """

        # 检查因子是否为整数且大于等于 1
        assert isinstance(factor, int) and factor >= 1
        # 如果未提供卷积核，默认生成一个大小为 factor 的卷积核
        if kernel is None:
            kernel = [1] * factor

        # 将卷积核转换为 float32 类型的张量
        kernel = torch.tensor(kernel, dtype=torch.float32)
        # 如果卷积核是一维的，生成外积形成二维卷积核
        if kernel.ndim == 1:
            kernel = torch.outer(kernel, kernel)
        # 归一化卷积核
        kernel /= torch.sum(kernel)

        # 将卷积核乘以增益因子
        kernel = kernel * gain

        # 如果使用卷积操作
        if self.use_conv:
            # 获取权重的高度和宽度
            _, _, convH, convW = weight.shape
            # 计算填充值
            pad_value = (kernel.shape[0] - factor) + (convW - 1)
            # 定义步幅值
            stride_value = [factor, factor]
            # 调用 upfirdn2d_native 进行上采样和填充
            upfirdn_input = upfirdn2d_native(
                hidden_states,
                torch.tensor(kernel, device=hidden_states.device),
                pad=((pad_value + 1) // 2, pad_value // 2),
            )
            # 使用卷积层进行卷积操作
            output = F.conv2d(upfirdn_input, weight, stride=stride_value, padding=0)
        else:
            # 计算填充值
            pad_value = kernel.shape[0] - factor
            # 调用 upfirdn2d_native 进行下采样
            output = upfirdn2d_native(
                hidden_states,
                torch.tensor(kernel, device=hidden_states.device),
                down=factor,
                pad=((pad_value + 1) // 2, pad_value // 2),
            )

        # 返回输出张量
        return output

    # 定义前向传播函数
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 如果使用卷积
        if self.use_conv:
            # 进行 2D 下采样并添加偏置
            downsample_input = self._downsample_2d(hidden_states, weight=self.Conv2d_0.weight, kernel=self.fir_kernel)
            hidden_states = downsample_input + self.Conv2d_0.bias.reshape(1, -1, 1, 1)
        else:
            # 进行 2D 下采样
            hidden_states = self._downsample_2d(hidden_states, kernel=self.fir_kernel, factor=2)

        # 返回处理后的隐状态
        return hidden_states
# 用于 K-upscaler 的下采样/上采样层，可能可以使用 FirDownsample2D/DirUpsample2D 代替
class KDownsample2D(nn.Module):
    r"""一个 2D K-下采样层。

    参数：
        pad_mode (`str`, *可选*, 默认值为 `"reflect"`): 使用的填充模式。
    """

    def __init__(self, pad_mode: str = "reflect"):
        # 调用父类初始化方法
        super().__init__()
        # 设置填充模式
        self.pad_mode = pad_mode
        # 定义 1D 卷积核并归一化
        kernel_1d = torch.tensor([[1 / 8, 3 / 8, 3 / 8, 1 / 8]])
        # 计算填充大小
        self.pad = kernel_1d.shape[1] // 2 - 1
        # 将卷积核的转置注册为缓冲区
        self.register_buffer("kernel", kernel_1d.T @ kernel_1d, persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # 根据填充模式对输入进行填充
        inputs = F.pad(inputs, (self.pad,) * 4, self.pad_mode)
        # 创建与输入通道数相同的零权重张量
        weight = inputs.new_zeros(
            [
                inputs.shape[1],
                inputs.shape[1],
                self.kernel.shape[0],
                self.kernel.shape[1],
            ]
        )
        # 创建输入通道的索引
        indices = torch.arange(inputs.shape[1], device=inputs.device)
        # 将卷积核扩展为权重张量
        kernel = self.kernel.to(weight)[None, :].expand(inputs.shape[1], -1, -1)
        # 在权重张量中设置相应的卷积核值
        weight[indices, indices] = kernel
        # 对输入进行 2D 卷积并返回结果
        return F.conv2d(inputs, weight, stride=2)


class CogVideoXDownsample3D(nn.Module):
    # 待办事项：等待论文发布。
    r"""
    一个 3D 下采样层，使用于 [CogVideoX]() 由清华大学和智谱AI 提供

    参数：
        in_channels (`int`):
            输入图像中的通道数。
        out_channels (`int`):
            卷积产生的通道数。
        kernel_size (`int`, 默认值为 `3`):
            卷积核的大小。
        stride (`int`, 默认值为 `2`):
            卷积的步幅。
        padding (`int`, 默认值为 `0`):
            添加到输入四个边的填充。
        compress_time (`bool`, 默认值为 `False`):
            是否压缩时间维度。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 0,
        compress_time: bool = False,
    ):
        # 调用父类初始化方法
        super().__init__()

        # 初始化 2D 卷积层
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        # 设置时间压缩标志
        self.compress_time = compress_time
    # 定义前向传播方法，接受一个张量 x，返回一个张量
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 如果需要压缩时间维度
        if self.compress_time:
            # 获取输入张量的形状，分别是批量大小、通道数、帧数、高度和宽度
            batch_size, channels, frames, height, width = x.shape

            # 重新排列和重塑张量的形状，变为 (batch_size * height * width, channels, frames)
            x = x.permute(0, 3, 4, 1, 2).reshape(batch_size * height * width, channels, frames)

            # 检查帧数是否为奇数
            if x.shape[-1] % 2 == 1:
                # 分离出第一帧和剩余帧
                x_first, x_rest = x[..., 0], x[..., 1:]
                # 如果剩余帧存在
                if x_rest.shape[-1] > 0:
                    # 对剩余帧进行平均池化，减少帧数
                    x_rest = F.avg_pool1d(x_rest, kernel_size=2, stride=2)

                # 将第一帧和池化后的剩余帧拼接在一起
                x = torch.cat([x_first[..., None], x_rest], dim=-1)
                # 重新排列和重塑张量，变为 (batch_size, channels, (frames // 2) + 1, height, width)
                x = x.reshape(batch_size, height, width, channels, x.shape[-1]).permute(0, 3, 4, 1, 2)
            else:
                # 对输入张量进行平均池化，减少帧数
                x = F.avg_pool1d(x, kernel_size=2, stride=2)
                # 重新排列和重塑张量，变为 (batch_size, channels, frames // 2, height, width)
                x = x.reshape(batch_size, height, width, channels, x.shape[-1]).permute(0, 3, 4, 1, 2)

        # 对张量进行填充
        pad = (0, 1, 0, 1)
        # 使用常数值 0 进行填充
        x = F.pad(x, pad, mode="constant", value=0)
        # 获取填充后张量的形状
        batch_size, channels, frames, height, width = x.shape
        # 重新排列和重塑张量，变为 (batch_size * frames, channels, height, width)
        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channels, height, width)
        # 通过卷积层处理张量
        x = self.conv(x)
        # 重新排列和重塑张量，变为 (batch_size, channels, frames, height, width)
        x = x.reshape(batch_size, frames, x.shape[1], x.shape[2], x.shape[3]).permute(0, 2, 1, 3, 4)
        # 返回处理后的张量
        return x
# 定义一个用于二维图像降采样的函数
def downsample_2d(
    # 输入的张量，形状为[N, C, H, W]或[N, H, W, C]
    hidden_states: torch.Tensor,
    # 可选的 FIR 滤波器
    kernel: Optional[torch.Tensor] = None,
    # 降采样因子，默认为2
    factor: int = 2,
    # 信号幅度的缩放因子，默认为1
    gain: float = 1,
) -> torch.Tensor:
    r"""Downsample2D a batch of 2D images with the given filter.
    # 文档字符串，说明函数用途和参数

    Args:
        hidden_states (`torch.Tensor`)
            # 输入张量的形状说明
        kernel (`torch.Tensor`, *optional*):
            # FIR 滤波器的形状说明
        factor (`int`, *optional*, default to `2`):
            # 降采样因子的说明
        gain (`float`, *optional*, default to `1.0`):
            # 信号幅度缩放因子的说明

    Returns:
        output (`torch.Tensor`):
            # 返回张量的形状说明
    """

    # 确保 factor 是正整数
    assert isinstance(factor, int) and factor >= 1
    # 如果未提供 kernel，使用默认值
    if kernel is None:
        kernel = [1] * factor

    # 将 kernel 转换为浮点类型的张量
    kernel = torch.tensor(kernel, dtype=torch.float32)
    # 如果 kernel 是一维的，生成外积形成二维滤波器
    if kernel.ndim == 1:
        kernel = torch.outer(kernel, kernel)
    # 将 kernel 归一化
    kernel /= torch.sum(kernel)

    # 根据 gain 调整 kernel 的值
    kernel = kernel * gain
    # 计算填充值
    pad_value = kernel.shape[0] - factor
    # 调用 upfirdn2d_native 函数进行降采样
    output = upfirdn2d_native(
        hidden_states,
        # 将 kernel 移动到与输入相同的设备
        kernel.to(device=hidden_states.device),
        # 设置降采样因子
        down=factor,
        # 设置填充参数
        pad=((pad_value + 1) // 2, pad_value // 2),
    )
    # 返回降采样后的输出
    return output
```