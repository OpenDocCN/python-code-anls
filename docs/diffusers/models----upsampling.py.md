# `.\diffusers\models\upsampling.py`

```py
# 版权所有 2024 The HuggingFace Team. 保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（“许可证”）授权；
# 除非遵循许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是以“原样”基础提供，
# 不提供任何形式的担保或条件，无论是明示或暗示的。
# 有关许可证的特定语言、治理权限和
# 许可证限制，请参阅许可证。

# 从 typing 模块导入 Optional 和 Tuple，用于类型提示
from typing import Optional, Tuple

# 导入 PyTorch 库及其神经网络模块
import torch
import torch.nn as nn
import torch.nn.functional as F

# 从 utils 模块导入 deprecate 装饰器
from ..utils import deprecate
# 从 normalization 模块导入 RMSNorm 类
from .normalization import RMSNorm

# 定义一维上采样层类，继承自 nn.Module
class Upsample1D(nn.Module):
    """一维上采样层，带可选的卷积。

    参数：
        channels (`int`):
            输入和输出的通道数。
        use_conv (`bool`, default `False`):
            是否使用卷积的选项。
        use_conv_transpose (`bool`, default `False`):
            是否使用转置卷积的选项。
        out_channels (`int`, optional):
            输出通道的数量。默认为 `channels`。
        name (`str`, default `conv`):
            一维上采样层的名称。
    """

    # 初始化函数，定义层的参数和卷积层
    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: Optional[int] = None,
        name: str = "conv",
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 设置输入通道数
        self.channels = channels
        # 设置输出通道数，默认为输入通道数
        self.out_channels = out_channels or channels
        # 设置是否使用卷积
        self.use_conv = use_conv
        # 设置是否使用转置卷积
        self.use_conv_transpose = use_conv_transpose
        # 设置层的名称
        self.name = name

        # 初始化卷积层为 None
        self.conv = None
        # 如果选择使用转置卷积，则初始化相应的卷积层
        if use_conv_transpose:
            self.conv = nn.ConvTranspose1d(channels, self.out_channels, 4, 2, 1)
        # 否则，如果选择使用卷积，则初始化卷积层
        elif use_conv:
            self.conv = nn.Conv1d(self.channels, self.out_channels, 3, padding=1)

    # 定义前向传播函数，接收输入并返回输出
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # 断言输入的通道数与设置的通道数一致
        assert inputs.shape[1] == self.channels
        # 如果选择使用转置卷积，直接返回卷积结果
        if self.use_conv_transpose:
            return self.conv(inputs)

        # 否则，使用最近邻插值法对输入进行上采样
        outputs = F.interpolate(inputs, scale_factor=2.0, mode="nearest")

        # 如果选择使用卷积，则对上采样结果进行卷积
        if self.use_conv:
            outputs = self.conv(outputs)

        # 返回最终的输出结果
        return outputs

# 定义二维上采样层类，继承自 nn.Module
class Upsample2D(nn.Module):
    """二维上采样层，带可选的卷积。

    参数：
        channels (`int`):
            输入和输出的通道数。
        use_conv (`bool`, default `False`):
            是否使用卷积的选项。
        use_conv_transpose (`bool`, default `False`):
            是否使用转置卷积的选项。
        out_channels (`int`, optional):
            输出通道的数量。默认为 `channels`。
        name (`str`, default `conv`):
            二维上采样层的名称。
    """
    # 初始化方法，用于创建该类的实例
    def __init__(
        # 输入通道数
        self,
        channels: int,
        # 是否使用卷积
        use_conv: bool = False,
        # 是否使用转置卷积
        use_conv_transpose: bool = False,
        # 输出通道数，可选，默认为输入通道数
        out_channels: Optional[int] = None,
        # 模块名称，默认为 "conv"
        name: str = "conv",
        # 卷积核大小，可选
        kernel_size: Optional[int] = None,
        # 填充大小，默认为 1
        padding=1,
        # 归一化类型
        norm_type=None,
        # 归一化时的 epsilon
        eps=None,
        # 是否使用逐元素仿射
        elementwise_affine=None,
        # 是否使用偏置，默认为 True
        bias=True,
        # 是否进行插值，默认为 True
        interpolate=True,
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 保存输入通道数
        self.channels = channels
        # 保存输出通道数，默认为输入通道数
        self.out_channels = out_channels or channels
        # 保存是否使用卷积的标志
        self.use_conv = use_conv
        # 保存是否使用转置卷积的标志
        self.use_conv_transpose = use_conv_transpose
        # 保存模块名称
        self.name = name
        # 保存是否进行插值的标志
        self.interpolate = interpolate

        # 根据归一化类型初始化归一化层
        if norm_type == "ln_norm":
            # 使用层归一化
            self.norm = nn.LayerNorm(channels, eps, elementwise_affine)
        elif norm_type == "rms_norm":
            # 使用 RMS 归一化
            self.norm = RMSNorm(channels, eps, elementwise_affine)
        elif norm_type is None:
            # 不使用归一化
            self.norm = None
        else:
            # 抛出未知归一化类型的错误
            raise ValueError(f"unknown norm_type: {norm_type}")

        # 初始化卷积层
        conv = None
        if use_conv_transpose:
            # 如果使用转置卷积且未指定卷积核大小，则默认为 4
            if kernel_size is None:
                kernel_size = 4
            # 创建转置卷积层
            conv = nn.ConvTranspose2d(
                channels, self.out_channels, kernel_size=kernel_size, stride=2, padding=padding, bias=bias
            )
        elif use_conv:
            # 如果使用卷积且未指定卷积核大小，则默认为 3
            if kernel_size is None:
                kernel_size = 3
            # 创建卷积层
            conv = nn.Conv2d(self.channels, self.out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

        # TODO(Suraj, Patrick) - 在权重字典正确重命名后进行清理
        if name == "conv":
            # 如果名称为 "conv"，则保存卷积层
            self.conv = conv
        else:
            # 否则将卷积层保存为另一个属性
            self.Conv2d_0 = conv
    # 前向传播函数，接受隐藏状态和可选输出大小，返回张量
    def forward(self, hidden_states: torch.Tensor, output_size: Optional[int] = None, *args, **kwargs) -> torch.Tensor:
        # 检查是否传入多余参数或废弃的 scale 参数
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            # 设置废弃提示信息，告知用户 scale 参数将来会引发错误
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            # 调用 deprecate 函数，记录废弃信息
            deprecate("scale", "1.0.0", deprecation_message)
    
        # 确保隐藏状态的通道数与当前对象的通道数匹配
        assert hidden_states.shape[1] == self.channels
    
        # 如果存在归一化层，则对隐藏状态进行归一化
        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    
        # 如果使用转置卷积，则调用卷积层处理隐藏状态
        if self.use_conv_transpose:
            return self.conv(hidden_states)
    
        # 将数据类型转换为 float32，解决 bfloat16 在特定操作中不支持的问题
        # TODO(Suraj): 一旦问题修复，移除此转换
        # https://github.com/pytorch/pytorch/issues/86679
        dtype = hidden_states.dtype
        # 检查数据类型是否为 bfloat16
        if dtype == torch.bfloat16:
            # 转换为 float32
            hidden_states = hidden_states.to(torch.float32)
    
        # 对于大批量大小的情况，确保数据是连续的
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()
    
        # 如果传入了 output_size，则强制进行插值输出
        # 并不使用 scale_factor=2
        if self.interpolate:
            # 如果没有传入 output_size，则使用默认插值因子
            if output_size is None:
                hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
            else:
                # 使用传入的 output_size 进行插值
                hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")
    
        # 如果输入为 bfloat16，转换回 bfloat16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)
    
        # TODO(Suraj, Patrick) - 在权重字典正确重命名后进行清理
        if self.use_conv:
            # 如果使用卷积，判断卷积层名称
            if self.name == "conv":
                hidden_states = self.conv(hidden_states)  # 调用常规卷积
            else:
                hidden_states = self.Conv2d_0(hidden_states)  # 调用特定卷积层
    
        # 返回处理后的隐藏状态
        return hidden_states
# 定义一个二维 FIR 上采样层，包含可选的卷积操作
class FirUpsample2D(nn.Module):
    """A 2D FIR upsampling layer with an optional convolution.

    Parameters:
        channels (`int`, optional):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        fir_kernel (`tuple`, default `(1, 3, 3, 1)`):
            kernel for the FIR filter.
    """

    # 初始化 FIR 上采样层
    def __init__(
        self,
        channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        use_conv: bool = False,
        fir_kernel: Tuple[int, int, int, int] = (1, 3, 3, 1),
    ):
        # 调用父类构造函数
        super().__init__()
        # 确定输出通道数，如果未提供则使用输入通道数
        out_channels = out_channels if out_channels else channels
        # 如果选择使用卷积，初始化卷积层
        if use_conv:
            self.Conv2d_0 = nn.Conv2d(channels, out_channels, kernel_size=3, stride=1, padding=1)
        # 保存卷积使用状态、FIR 核心和输出通道数
        self.use_conv = use_conv
        self.fir_kernel = fir_kernel
        self.out_channels = out_channels

    # 定义一个用于 2D 上采样的私有方法
    def _upsample_2d(
        self,
        hidden_states: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        kernel: Optional[torch.Tensor] = None,
        factor: int = 2,
        gain: float = 1,
    ):
        # 定义前向传播方法
        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            # 如果使用卷积，执行卷积操作并加上偏置
            if self.use_conv:
                height = self._upsample_2d(hidden_states, self.Conv2d_0.weight, kernel=self.fir_kernel)
                height = height + self.Conv2d_0.bias.reshape(1, -1, 1, 1)
            # 否则仅执行 FIR 上采样
            else:
                height = self._upsample_2d(hidden_states, kernel=self.fir_kernel, factor=2)

            # 返回上采样后的高度
            return height


# 定义一个二维 K 上采样层
class KUpsample2D(nn.Module):
    r"""A 2D K-upsampling layer.

    Parameters:
        pad_mode (`str`, *optional*, default to `"reflect"`): the padding mode to use.
    """

    # 初始化 K 上采样层，设置填充模式
    def __init__(self, pad_mode: str = "reflect"):
        # 调用父类构造函数
        super().__init__()
        # 保存填充模式
        self.pad_mode = pad_mode
        # 创建一维卷积核，进行标准化
        kernel_1d = torch.tensor([[1 / 8, 3 / 8, 3 / 8, 1 / 8]]) * 2
        # 计算填充大小
        self.pad = kernel_1d.shape[1] // 2 - 1
        # 注册卷积核缓冲区
        self.register_buffer("kernel", kernel_1d.T @ kernel_1d, persistent=False)

    # 定义前向传播方法
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # 根据填充模式填充输入
        inputs = F.pad(inputs, ((self.pad + 1) // 2,) * 4, self.pad_mode)
        # 初始化权重张量
        weight = inputs.new_zeros(
            [
                inputs.shape[1],
                inputs.shape[1],
                self.kernel.shape[0],
                self.kernel.shape[1],
            ]
        )
        # 创建索引张量
        indices = torch.arange(inputs.shape[1], device=inputs.device)
        # 扩展卷积核
        kernel = self.kernel.to(weight)[None, :].expand(inputs.shape[1], -1, -1)
        # 设置权重的对应索引
        weight[indices, indices] = kernel
        # 返回经过反卷积后的结果
        return F.conv_transpose2d(inputs, weight, stride=2, padding=self.pad * 2 + 1)


# 定义一个三维上采样层
class CogVideoXUpsample3D(nn.Module):
    r"""
    A 3D Upsample layer using in CogVideoX by Tsinghua University & ZhipuAI # Todo: Wait for paper relase.
    # 参数说明
    Args:
        in_channels (`int`):
            # 输入图像的通道数
            Number of channels in the input image.
        out_channels (`int`):
            # 卷积操作产生的通道数
            Number of channels produced by the convolution.
        kernel_size (`int`, defaults to `3`):
            # 卷积核的大小，默认值为3
            Size of the convolving kernel.
        stride (`int`, defaults to `1`):
            # 卷积的步幅，默认值为1
            Stride of the convolution.
        padding (`int`, defaults to `1`):
            # 输入数据四周填充的大小，默认值为1
            Padding added to all four sides of the input.
        compress_time (`bool`, defaults to `False`):
            # 是否压缩时间维度，默认值为False
            Whether or not to compress the time dimension.
    """

    # 初始化方法
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        compress_time: bool = False,
    ) -> None:
        # 调用父类的初始化方法
        super().__init__()

        # 定义卷积层
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        # 保存压缩时间的标志
        self.compress_time = compress_time

    # 前向传播方法
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # 如果需要压缩时间维度
        if self.compress_time:
            # 检查时间维度是否大于1且为奇数
            if inputs.shape[2] > 1 and inputs.shape[2] % 2 == 1:
                # 分离第一个帧
                x_first, x_rest = inputs[:, :, 0], inputs[:, :, 1:]

                # 对第一个帧进行插值放大
                x_first = F.interpolate(x_first, scale_factor=2.0)
                # 对其余帧进行插值放大
                x_rest = F.interpolate(x_rest, scale_factor=2.0)
                # 增加一个维度
                x_first = x_first[:, :, None, :, :]
                # 合并第一个帧和其余帧
                inputs = torch.cat([x_first, x_rest], dim=2)
            # 如果时间维度大于1
            elif inputs.shape[2] > 1:
                # 对输入进行插值放大
                inputs = F.interpolate(inputs, scale_factor=2.0)
            else:
                # 如果时间维度等于1，进行处理
                inputs = inputs.squeeze(2)
                # 对输入进行插值放大
                inputs = F.interpolate(inputs, scale_factor=2.0)
                # 增加一个维度
                inputs = inputs[:, :, None, :, :]
        else:
            # 仅对2D进行插值处理
            b, c, t, h, w = inputs.shape
            # 重新排列维度，准备卷积操作
            inputs = inputs.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            # 对输入进行插值放大
            inputs = F.interpolate(inputs, scale_factor=2.0)
            # 还原维度顺序
            inputs = inputs.reshape(b, t, c, *inputs.shape[2:]).permute(0, 2, 1, 3, 4)

        # 再次获取当前形状
        b, c, t, h, w = inputs.shape
        # 重新排列维度，为卷积操作做准备
        inputs = inputs.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        # 通过卷积层处理输入
        inputs = self.conv(inputs)
        # 还原维度顺序
        inputs = inputs.reshape(b, t, *inputs.shape[1:]).permute(0, 2, 1, 3, 4)

        # 返回处理后的输入
        return inputs
# 定义一个用于二维上采样的函数，支持可选的上采样核和因子
def upfirdn2d_native(
    # 输入的张量，通常是图像数据
    tensor: torch.Tensor,
    # 卷积核，决定上采样的滤波效果
    kernel: torch.Tensor,
    # 上采样因子，默认值为1
    up: int = 1,
    # 下采样因子，默认值为1
    down: int = 1,
    # 填充大小，默认不填充
    pad: Tuple[int, int] = (0, 0),
) -> torch.Tensor:
    # 将上采样因子赋值给x和y方向
    up_x = up_y = up
    # 将下采样因子赋值给x和y方向
    down_x = down_y = down
    # 获取y方向的填充大小
    pad_x0 = pad_y0 = pad[0]
    # 获取y方向的填充大小
    pad_x1 = pad_y1 = pad[1]

    # 获取输入张量的形状信息，包括通道数和高宽
    _, channel, in_h, in_w = tensor.shape
    # 将张量重塑为适合卷积操作的形状
    tensor = tensor.reshape(-1, in_h, in_w, 1)

    # 获取重塑后的张量的形状信息
    _, in_h, in_w, minor = tensor.shape
    # 获取卷积核的高和宽
    kernel_h, kernel_w = kernel.shape

    # 将张量视图转换为适合处理的格式
    out = tensor.view(-1, in_h, 1, in_w, 1, minor)
    # 对张量进行填充以便进行上采样
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    # 重塑填充后的张量为新的形状
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    # 应用额外的填充，以便在后续处理中保持一致
    out = F.pad(out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    # 将张量移动回原始设备（如果需要）
    out = out.to(tensor.device)  # Move back to mps if necessary
    # 应用负填充以调整输出的边界
    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    # 重新排列张量的维度，以便于卷积操作
    out = out.permute(0, 3, 1, 2)
    # 重塑输出张量以匹配卷积的输入格式
    out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    # 翻转卷积核以应用于卷积操作
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    # 执行卷积操作
    out = F.conv2d(out, w)
    # 重塑输出张量的形状以匹配所需的输出格式
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    # 重新排列输出张量的维度
    out = out.permute(0, 2, 3, 1)
    # 根据下采样因子对输出张量进行下采样
    out = out[:, ::down_y, ::down_x, :]

    # 计算输出张量的高和宽
    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1

    # 返回最终输出张量，调整为指定形状
    return out.view(-1, channel, out_h, out_w)


# 定义用于二维上采样的辅助函数，支持自定义卷积核和因子
def upsample_2d(
    # 输入的张量，通常是图像数据
    hidden_states: torch.Tensor,
    # 可选的卷积核，用于滤波
    kernel: Optional[torch.Tensor] = None,
    # 上采样因子，默认值为2
    factor: int = 2,
    # 信号幅度的缩放因子，默认值为1
    gain: float = 1,
) -> torch.Tensor:
    r"""Upsample2D a batch of 2D images with the given filter.
    Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]` and upsamples each image with the given
    filter. The filter is normalized so that if the input pixels are constant, they will be scaled by the specified
    `gain`. Pixels outside the image are assumed to be zero, and the filter is padded with zeros so that its shape is
    a: multiple of the upsampling factor.

    Args:
        hidden_states (`torch.Tensor`):
            Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`.
        kernel (`torch.Tensor`, *optional*):
            FIR filter of the shape `[firH, firW]` or `[firN]` (separable). The default is `[1] * factor`, which
            corresponds to nearest-neighbor upsampling.
        factor (`int`, *optional*, default to `2`):
            Integer upsampling factor.
        gain (`float`, *optional*, default to `1.0`):
            Scaling factor for signal magnitude (default: 1.0).

    Returns:
        output (`torch.Tensor`):
            Tensor of the shape `[N, C, H * factor, W * factor]`
    """
    # 确保因子是一个正整数
    assert isinstance(factor, int) and factor >= 1
    # 如果没有提供卷积核，则使用默认的最近邻上采样核
    if kernel is None:
        kernel = [1] * factor
    # 将输入的 kernel 转换为张量，并指定数据类型为 float32
        kernel = torch.tensor(kernel, dtype=torch.float32)
        # 如果 kernel 是一维的，则计算其外积，生成二维卷积核
        if kernel.ndim == 1:
            kernel = torch.outer(kernel, kernel)
        # 将 kernel 归一化，使其所有元素之和为 1
        kernel /= torch.sum(kernel)
    
        # 根据增益和因子调整 kernel 的值
        kernel = kernel * (gain * (factor**2))
        # 计算 padding 的值，用于图像处理
        pad_value = kernel.shape[0] - factor
        # 使用 upfirdn2d_native 函数进行上采样和滤波处理
        output = upfirdn2d_native(
            hidden_states,  # 输入的隐藏状态
            kernel.to(device=hidden_states.device),  # 将 kernel 移动到隐藏状态的设备上
            up=factor,  # 上采样因子
            # 设置 padding，确保输出的尺寸正确
            pad=((pad_value + 1) // 2 + factor - 1, pad_value // 2),
        )
        # 返回处理后的输出
        return output
```