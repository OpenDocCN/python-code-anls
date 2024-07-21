# `.\pytorch\torch\nn\modules\conv.py`

```py
# mypy: allow-untyped-defs
# 引入 math 模块，用于数学运算
import math
# 引入类型注解相关模块
from typing import List, Optional, Tuple, Union
# 引入已弃用的类型扩展
from typing_extensions import deprecated

# 引入 PyTorch 模块
import torch
# 引入 PyTorch 的 Tensor 类型
from torch import Tensor
# 引入 PyTorch 中的文档注释相关内容
from torch._torch_docs import reproducibility_notes
# 引入 PyTorch 中的函数模块和初始化模块
from torch.nn import functional as F, init
# 引入 PyTorch 中关于 tensor 尺寸类型的定义
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
# 引入 PyTorch 中的参数和未初始化参数
from torch.nn.parameter import Parameter, UninitializedParameter

# 引入 LazyModuleMixin 类
from .lazy import LazyModuleMixin
# 引入 Module 类
from .module import Module
# 引入 utils 中的尺寸处理函数
from .utils import _pair, _reverse_repeat_tuple, _single, _triple

# 定义模块的公开接口
__all__ = [
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "LazyConv1d",
    "LazyConv2d",
    "LazyConv3d",
    "LazyConvTranspose1d",
    "LazyConvTranspose2d",
    "LazyConvTranspose3d",
]

# 定义卷积操作的相关注释
convolution_notes = {
    "groups_note": r"""* :attr:`groups` 控制输入和输出之间的连接。
      :attr:`in_channels` 和 :attr:`out_channels` 必须同时被
      :attr:`groups` 整除。例如，

        * 当 groups=1 时，所有输入都与所有输出进行卷积。
        * 当 groups=2 时，该操作等效于并排放置两个卷积层，
          每个卷积层看到输入通道的一半，并生成输出通道的一半，
          然后将两者串联。
        * 当 groups= :attr:`in_channels` 时，每个输入通道与
          其自身的一组滤波器（大小为
          :math:`\frac{\text{out\_channels}}{\text{in\_channels}}`）进行卷积。""",
    "depthwise_separable_note": r"""当 `groups == in_channels` 且 `out_channels == K * in_channels` 时，
        此操作也称为“深度可分离卷积”。

        换句话说，对于尺寸为 :math:`(N, C_{in}, L_{in})` 的输入，
        使用深度乘数 `K` 可以执行深度可分离卷积，其参数为
        :math:`(C_\text{in}=C_\text{in}, C_\text{out}=C_\text{in} \times \text{K}, ..., \text{groups}=C_\text{in})`。""",
}  # noqa: B950

# 定义 _ConvNd 类，继承自 Module 类
class _ConvNd(Module):
    # 常量列表，表示不可变的属性名称
    __constants__ = [
        "stride",
        "padding",
        "dilation",
        "groups",
        "padding_mode",
        "output_padding",
        "in_channels",
        "out_channels",
        "kernel_size",
    ]
    # 注解字典，标明 bias 属性的类型为 Optional[Tensor]
    __annotations__ = {"bias": Optional[Tensor]}

    # 定义 _conv_forward 方法，用于卷积操作的前向传播
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:  # type: ignore[empty-body]
        ...

    # 输入通道数
    in_channels: int
    # 反转后的填充重复两次的列表
    _reversed_padding_repeated_twice: List[int]
    # 输出通道数
    out_channels: int
    # 卷积核大小的元组
    kernel_size: Tuple[int, ...]
    # 步幅的元组
    stride: Tuple[int, ...]
    # 填充的字符串或元组
    padding: Union[str, Tuple[int, ...]]
    # 扩张的元组
    dilation: Tuple[int, ...]
    # 是否是转置卷积
    transposed: bool
    # 输出填充的元组
    output_padding: Tuple[int, ...]
    # 分组数
    groups: int
    # 填充模式的字符串
    padding_mode: str
    # 权重张量
    weight: Tensor
    # 偏置张量（可选）
    bias: Optional[Tensor]
    def reset_parameters(self) -> None:
        # 使用 Kaiming 均匀分布初始化权重矩阵 self.weight
        # 根据论文建议设置 a=sqrt(5)，相当于在 [-1/sqrt(k), 1/sqrt(k)] 范围内均匀初始化，
        # 其中 k = weight.size(1) * prod(*kernel_size)
        # 参考链接: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        # 如果存在偏置项 self.bias，则初始化其值
        if self.bias is not None:
            # 计算输入和输出的 fan_in 和 fan_out
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                # 根据 fan_in 设置均匀分布的边界
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        # 生成表示网络层参数的字符串描述
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}"
        )
        # 如果 padding 不全为 0，则加入到描述中
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        # 如果 dilation 不全为 1，则加入到描述中
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        # 如果 output_padding 不全为 0，则加入到描述中
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        # 如果 groups 不为 1，则加入到描述中
        if self.groups != 1:
            s += ", groups={groups}"
        # 如果没有偏置项，则加入 bias=False 到描述中
        if self.bias is None:
            s += ", bias=False"
        # 如果 padding_mode 不是 "zeros"，则加入到描述中
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        # 使用实例的字典来格式化描述字符串
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        # 调用父类的 __setstate__ 方法来设置对象状态
        super().__setstate__(state)
        # 如果对象没有 padding_mode 属性，则设置默认值 "zeros"
        if not hasattr(self, "padding_mode"):
            self.padding_mode = "zeros"
# 定义 Conv1d 类，继承自 _ConvNd 类，用于实现一维卷积操作
class Conv1d(_ConvNd):
    # 文档字符串，描述了一维卷积层对输入信号进行处理的方式及输出结果的形状
    __doc__ = (
        r"""Applies a 1D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, L)` and output :math:`(N, C_{\text{out}}, L_{\text{out}})` can be
    precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{in} - 1} \text{weight}(C_{\text{out}_j}, k)
        \star \text{input}(N_i, k)

    where :math:`\star` is the valid `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`L` is a length of signal sequence.
    """
        + r"""

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a one-element tuple.

    * :attr:`padding` controls the amount of padding applied to the input. It
      can be either a string {{'valid', 'same'}} or a tuple of ints giving the
      amount of implicit padding applied on both sides.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the \uue0 trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

    {groups_note}  # 描述了 groups 参数的作用

    Note:
        {depthwise_separable_note}  # 深度可分离卷积的特殊说明
    Note:
        {cudnn_reproducibility_note}  # CuDNN 的可重现性说明

    Note:
        ``padding='valid'`` is the same as no padding. ``padding='same'`` pads
        the input so the output has the shape as the input. However, this mode
        doesn't support any stride values other than 1.

    Note:
        This module supports complex data types i.e. ``complex32, complex64, complex128``.

    Args:
        in_channels (int): Number of channels in the input image  # 输入图像的通道数
        out_channels (int): Number of channels produced by the convolution  # 卷积操作输出的通道数
        kernel_size (int or tuple): Size of the convolving kernel  # 卷积核的大小
        stride (int or tuple, optional): Stride of the convolution. Default: 1  # 卷积的步幅
        padding (int, tuple or str, optional): Padding added to both sides of
            the input. Default: 0  # 输入信号两侧添加的填充
        padding_mode (str, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``  # 填充模式
        dilation (int or tuple, optional): Spacing between kernel
            elements. Default: 1  # 卷积核元素之间的间距
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1  # 输入通道到输出通道的分组连接数
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``  # 是否添加偏置项到输出

    """.format(
            **reproducibility_notes, **convolution_notes  # 使用外部定义的 reproducibility_notes 和 convolution_notes
        )
        + r"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: Union[str, _size_1_t] = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",  # TODO: refine this type
        device=None,
        dtype=None,
    ) -> None:
        # 创建一个包含设备和数据类型的工厂参数字典
        factory_kwargs = {"device": device, "dtype": dtype}

        # 根据输入的 kernel_size 类型，确定正确的 kernel_size_
        # 这里使用 _single 函数确保 kernel_size_ 是一个元组
        kernel_size_ = _single(kernel_size)

        # 根据输入的 stride 类型，确定正确的 stride_
        # 这里使用 _single 函数确保 stride_ 是一个元组
        stride_ = _single(stride)

        # 如果 padding 是字符串，则直接使用它；否则，使用 _single 函数转换成元组
        padding_ = padding if isinstance(padding, str) else _single(padding)

        # 根据输入的 dilation 类型，确定正确的 dilation_
        # 这里使用 _single 函数确保 dilation_ 是一个元组
        dilation_ = _single(dilation)

        # 调用父类的构造函数，初始化卷积层的各个参数
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            False,  # 这里设置的是 transposed=False，表示不是转置卷积
            _single(0),  # 这里设置的是 output_padding，对于普通卷积设为零
            groups,
            bias,
            padding_mode,
            **factory_kwargs,  # 传递工厂参数字典
        )
    # 定义卷积层的前向传播函数，接受输入、权重和偏置参数
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        # 如果填充模式不是"zeros"，则进行以下操作
        if self.padding_mode != "zeros":
            # 对输入进行零填充，使用类的内部计算的填充方式和模式
            return F.conv1d(
                F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight,
                bias,
                self.stride,
                _single(0),  # 一个长度为1的元组，表示 padding 为0
                self.dilation,
                self.groups,
            )
        # 如果填充模式是"zeros"，则进行以下操作
        return F.conv1d(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

    # 定义整个卷积层的前向传播函数，接受输入张量，并返回处理后的张量
    def forward(self, input: Tensor) -> Tensor:
        # 调用内部定义的 _conv_forward 函数，传入输入、权重和偏置参数
        return self._conv_forward(input, self.weight, self.bias)
# 定义 Conv2d 类，继承自 _ConvNd 类
class Conv2d(_ConvNd):
    # 设置类文档字符串，描述该类的作用和功能
    __doc__ = (
        r"""Applies a 2D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, H, W)` and output :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`
    can be precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)


    where :math:`\star` is the valid 2D `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`H` is a height of input planes in pixels, and :math:`W` is
    width in pixels.
    """
        + r"""

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a tuple.

    * :attr:`padding` controls the amount of padding applied to the input. It
      can be either a string {{'valid', 'same'}} or an int / a tuple of ints giving the
      amount of implicit padding applied on both sides.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

    {groups_note}

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Note:
        {depthwise_separable_note}

    Note:
        {cudnn_reproducibility_note}

    Note:
        ``padding='valid'`` is the same as no padding. ``padding='same'`` pads
        the input so the output has the shape as the input. However, this mode
        doesn't support any stride values other than 1.

    Note:
        This module supports complex data types i.e. ``complex32, complex64, complex128``.
    """
    Args:
        in_channels (int): 输入图像的通道数
        out_channels (int): 卷积操作输出的通道数
        kernel_size (int or tuple): 卷积核的大小
        stride (int or tuple, optional): 卷积操作的步长。默认为 1
        padding (int, tuple or str, optional): 输入图像四周的填充大小或类型。默认为 0
        padding_mode (str, optional): 填充模式，可以是 'zeros', 'reflect', 'replicate' 或 'circular'。默认为 'zeros'
        dilation (int or tuple, optional): 卷积核中各元素之间的间隔。默认为 1
        groups (int, optional): 输入通道到输出通道的分组连接数。默认为 1
        bias (bool, optional): 如果为 True，则输出中加入一个可学习的偏置。默认为 True
    """.format(
            **reproducibility_notes, **convolution_notes
        )
        + r"""
        
    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})` 或 :math:`(C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` 或 :math:`(C_{out}, H_{out}, W_{out})`, 其中

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    Attributes:
        weight (Tensor): 模块的可学习权重，形状为
            :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
            :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]})`.
            这些权重的值从 :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` 中抽样，其中
            :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
        bias (Tensor): 模块的可学习偏置，形状为 (out_channels)。
            如果 :attr:`bias` 是 ``True``，则这些权重的值从 :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` 中抽样，其中
            :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`

    Examples:

        >>> # 使用方形卷积核和相同步长
        >>> m = nn.Conv2d(16, 33, 3, stride=2)
        >>> # 使用非方形卷积核、不同步长和填充
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # 使用非方形卷积核、不同步长、填充和扩张
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation
    # 定义一个链接到 GitHub 上某个 README 文档的链接
    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    
    # 定义一个名为 Conv2d 的类，用于二维卷积操作
    class Conv2d(nn.Module):
        # 初始化函数，设置卷积层的参数和设备信息
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: Union[str, _size_2_t] = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = "zeros",  # 待优化的填充模式类型
            device=None,
            dtype=None,
        ) -> None:
            # 根据设备和数据类型创建工厂参数字典
            factory_kwargs = {"device": device, "dtype": dtype}
            # 将 kernel_size 转换为二元组
            kernel_size_ = _pair(kernel_size)
            # 将 stride 转换为二元组
            stride_ = _pair(stride)
            # 如果 padding 是字符串，则保留原样，否则转换为二元组
            padding_ = padding if isinstance(padding, str) else _pair(padding)
            # 将 dilation 转换为二元组
            dilation_ = _pair(dilation)
            
            # 调用父类的初始化方法，传入相关参数
            super().__init__(
                in_channels,
                out_channels,
                kernel_size_,
                stride_,
                padding_,
                dilation_,
                False,  # 此处的参数不明确
                _pair(0),  # 此处的参数不明确
                groups,
                bias,
                padding_mode,
                **factory_kwargs,
            )
    
        # 定义卷积操作的前向传播函数
        def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
            # 如果 padding_mode 不是 "zeros"，则进行指定填充模式的卷积计算
            if self.padding_mode != "zeros":
                return F.conv2d(
                    F.pad(
                        input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                    ),
                    weight,
                    bias,
                    self.stride,
                    _pair(0),  # 此处的参数不明确
                    self.dilation,
                    self.groups,
                )
            # 否则，进行普通的零填充卷积计算
            return F.conv2d(
                input, weight, bias, self.stride, self.padding, self.dilation, self.groups
            )
    
        # 定义整个模型的前向传播函数，调用 _conv_forward 方法
        def forward(self, input: Tensor) -> Tensor:
            return self._conv_forward(input, self.weight, self.bias)
class Conv3d(_ConvNd):
    # 3D 卷积类，继承自 _ConvNd 类
    __doc__ = (
        r"""Applies a 3D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C_{in}, D, H, W)`
    and output :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` can be precisely described as:

    .. math::
        out(N_i, C_{out_j}) = bias(C_{out_j}) +
                                \sum_{k = 0}^{C_{in} - 1} weight(C_{out_j}, k) \star input(N_i, k)

    where :math:`\star` is the valid 3D `cross-correlation`_ operator
    """
        + r"""

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    * :attr:`stride` controls the stride for the cross-correlation.

    * :attr:`padding` controls the amount of padding applied to the input. It
      can be either a string {'valid', 'same'} or a tuple of ints giving the
      amount of implicit padding applied on both sides.

    * :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm.
      It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    {groups_note}  # 描述组的注意事项

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the depth, height and width dimension
        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,
          the second `int` for the height dimension and the third `int` for the width dimension

    Note:
        {depthwise_separable_note}  # 深度可分离卷积的注意事项

    Note:
        {cudnn_reproducibility_note}  # CuDNN 可重现性的注意事项

    Note:
        ``padding='valid'`` is the same as no padding. ``padding='same'`` pads
        the input so the output has the shape as the input. However, this mode
        doesn't support any stride values other than 1.

    Note:
        This module supports complex data types i.e. ``complex32, complex64, complex128``.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all six sides of
            the input. Default: 0
        padding_mode (str, optional): ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """
    """
    """.format(
            **reproducibility_notes, **convolution_notes
        )
        + r"""

    Shape:
        - Input: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})` or :math:`(C_{in}, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` or :math:`(C_{out}, D_{out}, H_{out}, W_{out})`,
          where

          .. math::
              D_{out} = \left\lfloor\frac{D_{in} + 2 \times \text{padding}[0] - \text{dilation}[0]
                    \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 \times \text{padding}[1] - \text{dilation}[1]
                    \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 \times \text{padding}[2] - \text{dilation}[2]
                    \times (\text{kernel\_size}[2] - 1) - 1}{\text{stride}[2]} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
                         :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]}, \text{kernel\_size[2]})`.
                         The values of these weights are sampled from
                         :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{2}\text{kernel\_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape (out_channels). If :attr:`bias` is ``True``,
                         then the values of these weights are
                         sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{2}\text{kernel\_size}[i]}`

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.Conv3d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))
        >>> input = torch.randn(20, 16, 10, 50, 100)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """
    )

    # 定义了一个新的类，用于创建 3D 卷积层
    def __init__(
        self,
        in_channels: int,                   # 输入通道数
        out_channels: int,                  # 输出通道数
        kernel_size: _size_3_t,             # 卷积核大小，可以是一个数或者三个数的元组
        stride: _size_3_t = 1,              # 步长大小，可以是一个数或者三个数的元组，默认为 1
        padding: Union[str, _size_3_t] = 0, # 填充大小，可以是一个字符串或者三个数的元组，默认为 0
        dilation: _size_3_t = 1,            # 空洞卷积的空洞大小，可以是一个数或者三个数的元组，默认为 1
        groups: int = 1,                    # 分组卷积时的分组数，默认为 1
        bias: bool = True,                  # 是否使用偏置，默认为 True
        padding_mode: str = "zeros",        # 填充模式，默认为 "zeros"
        device=None,                        # 指定设备，如 GPU
        dtype=None,                         # 指定数据类型
    ) -> None:
        # 构造函数的参数设置，包括设备、数据类型等
        factory_kwargs = {"device": device, "dtype": dtype}
        # 将 kernel_size 转换为三元组形式
        kernel_size_ = _triple(kernel_size)
        # 将 stride 转换为三元组形式
        stride_ = _triple(stride)
        # 如果 padding 是字符串，则转换为三元组形式；否则保持原样
        padding_ = padding if isinstance(padding, str) else _triple(padding)
        # 将 dilation 转换为三元组形式
        dilation_ = _triple(dilation)
        # 调用父类的构造方法，传入各种参数
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            False,
            _triple(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        # 如果 padding_mode 不是 "zeros"，则进行填充处理并进行 3D 卷积运算
        if self.padding_mode != "zeros":
            return F.conv3d(
                F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight,
                bias,
                self.stride,
                _triple(0),  # 使用零作为输出填充值
                self.dilation,
                self.groups,
            )
        # 如果 padding_mode 是 "zeros"，则直接进行 3D 卷积运算
        return F.conv3d(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

    def forward(self, input: Tensor) -> Tensor:
        # 前向传播方法，调用 _conv_forward 方法执行卷积操作
        return self._conv_forward(input, self.weight, self.bias)
class _ConvTransposeNd(_ConvNd):
    # 初始化函数，用于构造转置卷积层对象
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        bias,
        padding_mode,
        device=None,
        dtype=None,
    ) -> None:
        # 如果填充模式不是 "zeros"，则抛出数值错误异常
        if padding_mode != "zeros":
            raise ValueError(
                f'Only "zeros" padding mode is supported for {self.__class__.__name__}'
            )

        # 创建工厂关键字参数字典，包括设备和数据类型
        factory_kwargs = {"device": device, "dtype": dtype}
        # 调用父类 _ConvNd 的初始化方法，传递所有参数和工厂关键字参数
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    # _output_padding 方法用于计算输出填充
    # dilation 是一个可选参数，用于向后兼容性
    def _output_padding(
        self,
        input: Tensor,
        output_size: Optional[List[int]],
        stride: List[int],
        padding: List[int],
        kernel_size: List[int],
        num_spatial_dims: int,
        dilation: Optional[List[int]] = None,
    ) -> List[int]:
        # 如果未指定输出大小，则使用默认的输出填充值
        if output_size is None:
            ret = _single(self.output_padding)  # 如果之前不是列表，则转换为列表
        else:
            # 检查输入张量是否有批次维度
            has_batch_dim = input.dim() == num_spatial_dims + 2
            num_non_spatial_dims = 2 if has_batch_dim else 1
            
            # 调整输出大小以匹配空间维度的数量
            if len(output_size) == num_non_spatial_dims + num_spatial_dims:
                output_size = output_size[num_non_spatial_dims:]
                
            # 检查输出大小的有效性
            if len(output_size) != num_spatial_dims:
                raise ValueError(
                    f"ConvTranspose{num_spatial_dims}D: for {input.dim()}D input, output_size must have {num_spatial_dims} "
                    f"or {num_non_spatial_dims + num_spatial_dims} elements (got {len(output_size)})"
                )

            # 初始化最小和最大大小的列表
            min_sizes = torch.jit.annotate(List[int], [])
            max_sizes = torch.jit.annotate(List[int], [])
            
            # 计算每个空间维度的最小和最大大小
            for d in range(num_spatial_dims):
                dim_size = (
                    (input.size(d + num_non_spatial_dims) - 1) * stride[d]
                    - 2 * padding[d]
                    + (dilation[d] if dilation is not None else 1)
                    * (kernel_size[d] - 1)
                    + 1
                )
                min_sizes.append(dim_size)
                max_sizes.append(min_sizes[d] + stride[d] - 1)

            # 检查输出大小是否在有效范围内
            for i in range(len(output_size)):
                size = output_size[i]
                min_size = min_sizes[i]
                max_size = max_sizes[i]
                if size < min_size or size > max_size:
                    raise ValueError(
                        f"requested an output size of {output_size}, but valid sizes range "
                        f"from {min_sizes} to {max_sizes} (for an input of {input.size()[2:]})"
                    )

            # 计算输出相对于最小大小的偏移量
            res = torch.jit.annotate(List[int], [])
            for d in range(num_spatial_dims):
                res.append(output_size[d] - min_sizes[d])

            # 返回计算的输出大小列表
            ret = res
        # 返回结果
        return ret
# 定义 ConvTranspose1d 类，继承自 _ConvTransposeNd 类
class ConvTranspose1d(_ConvTransposeNd):
    # 文档字符串，描述该类的作用和特性
    __doc__ = (
        r"""Applies a 1D transposed convolution operator over an input image
    composed of several input planes.

    This module can be seen as the gradient of Conv1d with respect to its input.
    It is also known as a fractionally-strided convolution or
    a deconvolution (although it is not an actual deconvolution operation as it does
    not compute a true inverse of convolution). For more information, see the visualizations
    `here`_ and the `Deconvolutional Networks`_ paper.

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    * :attr:`stride` controls the stride for the cross-correlation.

    * :attr:`padding` controls the amount of implicit zero padding on both
      sides for ``dilation * (kernel_size - 1) - padding`` number of points. See note
      below for details.

    * :attr:`output_padding` controls the additional size added to one side
      of the output shape. See note below for details.

    * :attr:`dilation` controls the spacing between the kernel points; also known as the \u00e0 trous algorithm.
      It is harder to describe, but the link `here`_ has a nice visualization of what :attr:`dilation` does.

    {groups_note}

    Note:
        The :attr:`padding` argument effectively adds ``dilation * (kernel_size - 1) - padding``
        amount of zero padding to both sizes of the input. This is set so that
        when a :class:`~torch.nn.Conv1d` and a :class:`~torch.nn.ConvTranspose1d`
        are initialized with same parameters, they are inverses of each other in
        regard to the input and output shapes. However, when ``stride > 1``,
        :class:`~torch.nn.Conv1d` maps multiple input shapes to the same output
        shape. :attr:`output_padding` is provided to resolve this ambiguity by
        effectively increasing the calculated output shape on one side. Note
        that :attr:`output_padding` is only used to find output shape, but does
        not actually add zero-padding to output.

    Note:
        In some circumstances when using the CUDA backend with CuDNN, this operator
        may select a nondeterministic algorithm to increase performance. If this is
        undesirable, you can try to make the operation deterministic (potentially at
        a performance cost) by setting ``torch.backends.cudnn.deterministic =
        True``.
        Please see the notes on :doc:`/notes/randomness` for background.
    """
    """
    Args:
        in_channels (int): 输入图像的通道数
        out_channels (int): 卷积层输出的通道数
        kernel_size (int or tuple): 卷积核的大小
        stride (int or tuple, optional): 卷积的步长。默认为 1
        padding (int or tuple, optional): 输入两侧的填充量，计算公式为 ``dilation * (kernel_size - 1) - padding``。默认为 0
        output_padding (int or tuple, optional): 输出形状增加的额外量。默认为 0
        groups (int, optional): 输入通道到输出通道的分组连接数。默认为 1
        bias (bool, optional): 如果为 ``True``，则输出中添加一个可学习的偏置。默认为 ``True``
        dilation (int or tuple, optional): 卷积核元素之间的间距。默认为 1
    """.format(
        **reproducibility_notes, **convolution_notes
    ) + r"""
        
    Shape:
        - Input: :math:`(N, C_{in}, L_{in})` 或 :math:`(C_{in}, L_{in})`
        - Output: :math:`(N, C_{out}, L_{out})` 或 :math:`(C_{out}, L_{out})`，其中
    
          .. math::
              L_{out} = (L_{in} - 1) \times \text{stride} - 2 \times \text{padding} + \text{dilation}
                        \times (\text{kernel\_size} - 1) + \text{output\_padding} + 1
    
    Attributes:
        weight (Tensor): 模块的可学习权重，形状为
                         :math:`(\text{in\_channels}, \frac{\text{out\_channels}}{\text{groups}},`
                         :math:`\text{kernel\_size})`。
                         这些权重的值从 :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` 中采样，其中
                         :math:`k = \frac{groups}{C_\text{out} * \text{kernel\_size}}`
        bias (Tensor):   模块的可学习偏置，形状为 (out_channels)。
                         如果 :attr:`bias` 是 ``True``，则偏置的值从 :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` 中采样，
                         其中 :math:`k = \frac{groups}{C_\text{out} * \text{kernel\_size}}`
    
    .. _`here`:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    
    .. _`Deconvolutional Networks`:
        https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf
    """
    )
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        output_padding: _size_1_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_1_t = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size = _single(kernel_size)  # 将 kernel_size 转换为单值或单元素元组
        stride = _single(stride)  # 将 stride 转换为单值或单元素元组
        padding = _single(padding)  # 将 padding 转换为单值或单元素元组
        dilation = _single(dilation)  # 将 dilation 转换为单值或单元素元组
        output_padding = _single(output_padding)  # 将 output_padding 转换为单值或单元素元组
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            True,
            output_padding,
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def forward(self, input: Tensor, output_size: Optional[List[int]] = None) -> Tensor:
        if self.padding_mode != "zeros":
            raise ValueError(
                "Only `zeros` padding mode is supported for ConvTranspose1d"
            )

        assert isinstance(self.padding, tuple)
        # 在 "_output_padding" 中不能用 Tuple 或 Sequence 替代 List，
        # 因为 TorchScript 不支持 `Sequence[T]` 或 `Tuple[T, ...]`。
        num_spatial_dims = 1
        output_padding = self._output_padding(
            input,
            output_size,
            self.stride,  # type: ignore[arg-type] 表示在类型检查时忽略这里的参数类型检查
            self.padding,  # type: ignore[arg-type]
            self.kernel_size,  # type: ignore[arg-type]
            num_spatial_dims,
            self.dilation,  # type: ignore[arg-type]
        )
        return F.conv_transpose1d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )
# 定义一个继承自 _ConvTransposeNd 的类 ConvTranspose2d
class ConvTranspose2d(_ConvTransposeNd):
    # 类的文档字符串，描述了对输入图像应用二维转置卷积操作的作用
    __doc__ = (
        r"""Applies a 2D transposed convolution operator over an input image
    composed of several input planes.

    This module can be seen as the gradient of Conv2d with respect to its input.
    It is also known as a fractionally-strided convolution or
    a deconvolution (although it is not an actual deconvolution operation as it does
    not compute a true inverse of convolution). For more information, see the visualizations
    `here`_ and the `Deconvolutional Networks`_ paper.

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    * :attr:`stride` controls the stride for the cross-correlation.

    * :attr:`padding` controls the amount of implicit zero padding on both
      sides for ``dilation * (kernel_size - 1) - padding`` number of points. See note
      below for details.

    * :attr:`output_padding` controls the additional size added to one side
      of the output shape. See note below for details.

    * :attr:`dilation` controls the spacing between the kernel points; also known as the \u00e0 trous algorithm.
      It is harder to describe, but the link `here`_ has a nice visualization of what :attr:`dilation` does.

    {groups_note}

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`output_padding`
    can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimensions
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Note:
        The :attr:`padding` argument effectively adds ``dilation * (kernel_size - 1) - padding``
        amount of zero padding to both sizes of the input. This is set so that
        when a :class:`~torch.nn.Conv2d` and a :class:`~torch.nn.ConvTranspose2d`
        are initialized with same parameters, they are inverses of each other in
        regard to the input and output shapes. However, when ``stride > 1``,
        :class:`~torch.nn.Conv2d` maps multiple input shapes to the same output
        shape. :attr:`output_padding` is provided to resolve this ambiguity by
        effectively increasing the calculated output shape on one side. Note
        that :attr:`output_padding` is only used to find output shape, but does
        not actually add zero-padding to output.

    Note:
        {cudnn_reproducibility_note}
    """
    Args:
        in_channels (int): 输入图像的通道数
        out_channels (int): 卷积层输出的通道数
        kernel_size (int or tuple): 卷积核的大小
        stride (int or tuple, optional): 卷积的步长。默认值：1
        padding (int or tuple, optional): 输入各维度两端补充零的大小，默认是 ``dilation * (kernel_size - 1) - padding``。默认值：0
        output_padding (int or tuple, optional): 输出形状各维度一端增加的大小。默认值：0
        groups (int, optional): 输入通道到输出通道的分组连接数。默认值：1
        bias (bool, optional): 如果为 ``True``，输出添加可学习的偏置。默认值：``True``
        dilation (int or tuple, optional): 卷积核元素之间的间隔。默认值：1
    """.format(
            **reproducibility_notes, **convolution_notes
        )
        + r"""

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})` 或者 :math:`(C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` 或者 :math:`(C_{out}, H_{out}, W_{out})`，其中

        .. math::
              H_{out} = (H_{in} - 1) \times \text{stride}[0] - 2 \times \text{padding}[0] + \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) + \text{output\_padding}[0] + 1
        .. math::
              W_{out} = (W_{in} - 1) \times \text{stride}[1] - 2 \times \text{padding}[1] + \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) + \text{output\_padding}[1] + 1

    Attributes:
        weight (Tensor): 模块的可学习权重，形状为
                         :math:`(\text{in\_channels}, \frac{\text{out\_channels}}{\text{groups}},`
                         :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]})`。
                         这些权重的值从 :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` 中采样，
                         其中 :math:`k = \frac{groups}{C_\text{out} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
        bias (Tensor): 模块的可学习偏置，形状为 (out_channels)。
                       如果 :attr:`bias` 是 ``True``，那么这些权重的值从 :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` 中采样，
                       其中 :math:`k = \frac{groups}{C_\text{out} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
    Examples::

        >>> # 使用正方形核和相同步长
        >>> m = nn.ConvTranspose2d(16, 33, 3, stride=2)
        >>> # 非正方形核、不同步长，并带有填充
        >>> m = nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> output = m(input)
        >>> # 也可以指定精确的输出大小作为参数
        >>> input = torch.randn(1, 16, 12, 12)
        >>> downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(input)
        >>> h.size()
        torch.Size([1, 16, 6, 6])
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12, 12])

    .. _`here`:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md

    .. _`Deconvolutional Networks`:
        https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf
    """
    )

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        output_padding: _size_2_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_2_t = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        # 创建包含设备和数据类型的关键字参数字典
        factory_kwargs = {"device": device, "dtype": dtype}
        # 将核大小转换为二元组形式
        kernel_size = _pair(kernel_size)
        # 将步长转换为二元组形式
        stride = _pair(stride)
        # 将填充转换为二元组形式
        padding = _pair(padding)
        # 将膨胀转换为二元组形式
        dilation = _pair(dilation)
        # 将输出填充转换为二元组形式
        output_padding = _pair(output_padding)
        # 调用父类的初始化方法，传入所有参数
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            True,  # 固定为True
            output_padding,
            groups,
            bias,
            padding_mode,
            **factory_kwargs,  # 传递设备和数据类型的关键字参数
        )
    def forward(self, input: Tensor, output_size: Optional[List[int]] = None) -> Tensor:
        # 检查 padding_mode 是否为 "zeros"，如果不是则抛出数值错误异常
        if self.padding_mode != "zeros":
            raise ValueError(
                "Only `zeros` padding mode is supported for ConvTranspose2d"
            )

        # 断言 padding 是一个元组
        assert isinstance(self.padding, tuple)
        
        # 获取输出填充 output_padding，调用 self._output_padding 方法计算
        # 注意：在 TorchScript 中不能将 List 替换为 Tuple 或 Sequence，因为不支持 `Sequence[T]` 或 `Tuple[T, ...]`
        num_spatial_dims = 2
        output_padding = self._output_padding(
            input,
            output_size,
            self.stride,  # type: ignore[arg-type]
            self.padding,  # type: ignore[arg-type]
            self.kernel_size,  # type: ignore[arg-type]
            num_spatial_dims,
            self.dilation,  # type: ignore[arg-type]
        )

        # 返回使用 F.conv_transpose2d 函数进行的转置卷积操作结果
        return F.conv_transpose2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )
class ConvTranspose3d(_ConvTransposeNd):
    # ConvTranspose3d 类，继承自 _ConvTransposeNd
    __doc__ = (
        r"""Applies a 3D transposed convolution operator over an input image composed of several input
    planes.
    The transposed convolution operator multiplies each input value element-wise by a learnable kernel,
    and sums over the outputs from all input feature planes.

    This module can be seen as the gradient of Conv3d with respect to its input.
    It is also known as a fractionally-strided convolution or
    a deconvolution (although it is not an actual deconvolution operation as it does
    not compute a true inverse of convolution). For more information, see the visualizations
    `here`_ and the `Deconvolutional Networks`_ paper.

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    * :attr:`stride` controls the stride for the cross-correlation.

    * :attr:`padding` controls the amount of implicit zero padding on both
      sides for ``dilation * (kernel_size - 1) - padding`` number of points. See note
      below for details.

    * :attr:`output_padding` controls the additional size added to one side
      of the output shape. See note below for details.

    * :attr:`dilation` controls the spacing between the kernel points; also known as the \u00e0 trous algorithm.
      It is harder to describe, but the link `here`_ has a nice visualization of what :attr:`dilation` does.

    {groups_note}

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`output_padding`
    can either be:

        - a single ``int`` -- in which case the same value is used for the depth, height and width dimensions
        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,
          the second `int` for the height dimension and the third `int` for the width dimension

    Note:
        The :attr:`padding` argument effectively adds ``dilation * (kernel_size - 1) - padding``
        amount of zero padding to both sizes of the input. This is set so that
        when a :class:`~torch.nn.Conv3d` and a :class:`~torch.nn.ConvTranspose3d`
        are initialized with same parameters, they are inverses of each other in
        regard to the input and output shapes. However, when ``stride > 1``,
        :class:`~torch.nn.Conv3d` maps multiple input shapes to the same output
        shape. :attr:`output_padding` is provided to resolve this ambiguity by
        effectively increasing the calculated output shape on one side. Note
        that :attr:`output_padding` is only used to find output shape, but does
        not actually add zero-padding to output.

    Note:
        {cudnn_reproducibility_note}
    """  # 类的文档字符串，描述了 ConvTranspose3d 类的功能和用法
        Args:
            in_channels (int): 输入图像的通道数
            out_channels (int): 卷积操作产生的通道数
            kernel_size (int or tuple): 卷积核的大小
            stride (int or tuple, optional): 卷积的步长，默认为1
            padding (int or tuple, optional): 输入每个维度两侧的零填充数，默认为0
            output_padding (int or tuple, optional): 输出形状每个维度增加的额外大小，默认为0
            groups (int, optional): 输入通道到输出通道的分组连接数，默认为1
            bias (bool, optional): 如果为True，则在输出上添加可学习的偏置，默认为True
            dilation (int or tuple, optional): 卷积核元素之间的间距，默认为1

        Shape:
            - Input: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})` 或 :math:`(C_{in}, D_{in}, H_{in}, W_{in})`
            - Output: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` 或 :math:`(C_{out}, D_{out}, H_{out}, W_{out})`，其中

            .. math::
                  D_{out} = (D_{in} - 1) \times \text{stride}[0] - 2 \times \text{padding}[0] + \text{dilation}[0]
                            \times (\text{kernel\_size}[0] - 1) + \text{output\_padding}[0] + 1
            .. math::
                  H_{out} = (H_{in} - 1) \times \text{stride}[1] - 2 \times \text{padding}[1] + \text{dilation}[1]
                            \times (\text{kernel\_size}[1] - 1) + \text{output\_padding}[1] + 1
            .. math::
                  W_{out} = (W_{in} - 1) \times \text{stride}[2] - 2 \times \text{padding}[2] + \text{dilation}[2]
                            \times (\text{kernel\_size}[2] - 1) + \text{output\_padding}[2] + 1

        Attributes:
            weight (Tensor): 模块的可学习权重，形状为
                             :math:`(\text{in\_channels}, \frac{\text{out\_channels}}{\text{groups}},`
                             :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]}, \text{kernel\_size[2]})`.
                             这些权重的值从 :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` 中采样，其中
                             :math:`k = \frac{groups}{C_\text{out} * \prod_{i=0}^{2}\text{kernel\_size}[i]}`
            bias (Tensor): 模块的可学习偏置，形状为 (out_channels)。
                           如果 :attr:`bias` 为 ``True``，则这些权重的值从 :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` 中采样，其中
                           :math:`k = \frac{groups}{C_\text{out} * \prod_{i=0}^{2}\text{kernel\_size}[i]}`
    Examples::

        >>> # 使用方形卷积核和相同步长
        >>> m = nn.ConvTranspose3d(16, 33, 3, stride=2)
        >>> # 使用非方形卷积核、不同步长和填充
        >>> m = nn.ConvTranspose3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(0, 4, 2))
        >>> input = torch.randn(20, 16, 10, 50, 100)
        >>> output = m(input)

    .. _`here`:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md

    .. _`Deconvolutional Networks`:
        https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf
    """
    )

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: _size_3_t = 0,
        output_padding: _size_3_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_3_t = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        # 准备工厂参数字典，包括设备和数据类型
        factory_kwargs = {"device": device, "dtype": dtype}
        # 将核大小、步长、填充、扩张、输出填充转换为三元组形式
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        output_padding = _triple(output_padding)
        # 调用父类构造函数初始化
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            True,  # 强制传递`True`给`transposed`参数
            output_padding,
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def forward(self, input: Tensor, output_size: Optional[List[int]] = None) -> Tensor:
        # 如果填充模式不是“zeros”，则抛出值错误异常
        if self.padding_mode != "zeros":
            raise ValueError(
                "Only `zeros` padding mode is supported for ConvTranspose3d"
            )

        assert isinstance(self.padding, tuple)
        # 计算输出填充，调用内部方法`_output_padding`
        num_spatial_dims = 3
        output_padding = self._output_padding(
            input,
            output_size,
            self.stride,  # type: ignore[arg-type] - TorchScript不支持`Sequence[T]`或`Tuple[T, ...]`
            self.padding,  # type: ignore[arg-type] - TorchScript不支持`Sequence[T]`或`Tuple[T, ...]`
            self.kernel_size,  # type: ignore[arg-type] - TorchScript不支持`Sequence[T]`或`Tuple[T, ...]`
            num_spatial_dims,
            self.dilation,  # type: ignore[arg-type] - TorchScript不支持`Sequence[T]`或`Tuple[T, ...]`
        )

        # 调用`F.conv_transpose3d`函数进行三维转置卷积操作
        return F.conv_transpose3d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )
# TODO: Deprecate and remove the following alias `_ConvTransposeMixin`.
#
# `_ConvTransposeMixin` was a mixin that was removed.  It is meant to be used
# with `_ConvNd` to construct actual module classes that implements conv
# transpose ops:
#
#   class MyConvTranspose(_ConvNd, _ConvTransposeMixin):
#       ...
#
# In PyTorch, it has been replaced by `_ConvTransposeNd`, which is a proper
# subclass of `_ConvNd`.  However, some user code in the wild still (incorrectly)
# use the internal class `_ConvTransposeMixin`.  Hence, we provide this alias
# for BC, because it is cheap and easy for us to do so, even though that
# `_ConvTransposeNd` is really not a mixin anymore (but multiple inheritance as
# above would still work).
class _ConvTransposeMixin(_ConvTransposeNd):
    @deprecated(
        "`_ConvTransposeMixin` is a deprecated internal class. "
        "Please consider using public APIs.",
        category=FutureWarning,
    )
    # 初始化函数，继承自 _ConvTransposeNd，被标记为已废弃
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# TODO: Conv2dLocal
# TODO: Conv2dMap
# TODO: ConvTranspose2dMap


class _LazyConvXdMixin(LazyModuleMixin):
    # 定义了一些属性，用于卷积操作
    groups: int
    transposed: bool
    in_channels: int
    out_channels: int
    kernel_size: Tuple[int, ...]
    weight: UninitializedParameter
    bias: UninitializedParameter

    # 重置参数的方法
    def reset_parameters(self) -> None:
        # 如果没有未初始化的参数并且输入通道不为零
        if not self.has_uninitialized_params() and self.in_channels != 0:  # type: ignore[misc]
            # "type:ignore[..]" 是因为 mypy 认为在父类中未定义 "reset_parameters" 方法，
            # 实际上它在 _ConvND 中定义，而 _LazyConvXdMixin 继承了该类
            super().reset_parameters()  # type: ignore[misc]

    # "initialize_parameters" 的签名与超类型 LazyModuleMixin 中的定义不兼容
    # 根据输入和参数初始化模型参数
    def initialize_parameters(self, input: Tensor, *args, **kwargs) -> None:  # type: ignore[override]
        # 检查是否有未初始化的参数
        if self.has_uninitialized_params():  # type: ignore[misc]
            # 获取输入数据的通道数
            self.in_channels = self._get_in_channels(input)
            # 检查输入通道数是否能被分组数整除
            if self.in_channels % self.groups != 0:
                raise ValueError("in_channels must be divisible by groups")
            # 确保权重是未初始化的参数对象
            assert isinstance(self.weight, UninitializedParameter)
            # 根据是否转置设置权重的形状
            if self.transposed:
                self.weight.materialize(
                    (
                        self.in_channels,
                        self.out_channels // self.groups,
                        *self.kernel_size,
                    )
                )
            else:
                self.weight.materialize(
                    (
                        self.out_channels,
                        self.in_channels // self.groups,
                        *self.kernel_size,
                    )
                )
            # 如果存在偏置项，确保偏置项是未初始化的参数对象，并设置其形状
            if self.bias is not None:
                assert isinstance(self.bias, UninitializedParameter)
                self.bias.materialize((self.out_channels,))
            # 重置模型参数
            self.reset_parameters()

    # 从输入中提取通道数的函数
    def _get_in_channels(self, input: Tensor) -> int:
        # 获取输入数据的空间维度数
        num_spatial_dims = self._get_num_spatial_dims()
        # 计算没有批次维度的总维度数，加上通道维度
        num_dims_no_batch = num_spatial_dims + 1  # +1 for channels dim
        # 加上批次维度后的总维度数
        num_dims_batch = num_dims_no_batch + 1
        # 检查输入数据的维度是否符合预期
        if input.dim() not in (num_dims_no_batch, num_dims_batch):
            raise RuntimeError(
                f"Expected {num_dims_no_batch}D (unbatched) or {num_dims_batch}D (batched) input "
                f"to {self.__class__.__name__}, but "
                f"got input of size: {input.shape}"
            )
        # 返回输入数据的通道数
        return input.shape[1] if input.dim() == num_dims_batch else input.shape[0]

    # 返回模块期望的空间维度数，子类需要实现该方法
    def _get_num_spatial_dims(self) -> int:
        raise NotImplementedError
# LazyConv1d继承自_LazyConvXdMixin和Conv1d类，忽略类型检查misc
class LazyConv1d(_LazyConvXdMixin, Conv1d):  # type: ignore[misc]
    r"""A :class:`torch.nn.Conv1d` module with lazy initialization of the ``in_channels`` argument.

    The ``in_channels`` argument of the :class:`Conv1d` is inferred from the ``input.size(1)``.
    The attributes that will be lazily initialized are `weight` and `bias`.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.

    Args:
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        padding_mode (str, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel
            elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``

    .. seealso:: :class:`torch.nn.Conv1d` and :class:`torch.nn.modules.lazy.LazyModuleMixin`
    """

    # 超类将此变量定义为None，因此需要"类型：忽略"以便我们重新定义该变量。
    cls_to_become = Conv1d  # type: ignore[assignment]

    def __init__(
        self,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        # 调用父类的构造函数进行初始化，将bias参数硬编码为False，以避免创建即将被覆写的张量。
        super().__init__(
            0,
            0,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            False,  # bias参数硬编码为False，避免创建即将被覆写的张量
            padding_mode,
            **factory_kwargs,
        )
        # 使用UninitializedParameter延迟初始化weight属性
        self.weight = UninitializedParameter(**factory_kwargs)
        self.out_channels = out_channels
        if bias:
            # 如果bias为True，则使用UninitializedParameter延迟初始化bias属性
            self.bias = UninitializedParameter(**factory_kwargs)

    def _get_num_spatial_dims(self) -> int:
        # 返回空间维度数目为1
        return 1


# LazyConv2d继承自_LazyConvXdMixin和Conv2d类，忽略类型检查misc
class LazyConv2d(_LazyConvXdMixin, Conv2d):  # type: ignore[misc]
    r"""A :class:`torch.nn.Conv2d` module with lazy initialization of the ``in_channels`` argument.
    # 定义一个类，继承自 torch.nn.Conv2d，并重定义了一些属性和方法
    class Conv2d(torch.nn.Conv2d):
        """
        The ``in_channels`` argument of the :class:`Conv2d` that is inferred from the ``input.size(1)``.
        The attributes that will be lazily initialized are `weight` and `bias`.
    
        Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
        on lazy modules and their limitations.
    
        Args:
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of
                the input. Default: 0
            padding_mode (str, optional): ``'zeros'``, ``'reflect'``,
                ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
            dilation (int or tuple, optional): Spacing between kernel
                elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            bias (bool, optional): If ``True``, adds a learnable bias to the
                output. Default: ``True``
    
        .. seealso:: :class:`torch.nn.Conv2d` and :class:`torch.nn.modules.lazy.LazyModuleMixin`
        """
    
        # 定义一个变量 cls_to_become 为 Conv2d 类型，用于指定 super 类
        cls_to_become = Conv2d  # type: ignore[assignment]
    
        def __init__(
            self,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: _size_2_t = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = "zeros",  # TODO: refine this type
            device=None,
            dtype=None,
        ) -> None:
            # 准备工厂参数字典，包含设备和数据类型信息
            factory_kwargs = {"device": device, "dtype": dtype}
            
            # 调用父类构造方法初始化 Conv2d 对象，部分参数被硬编码，例如 bias 被设为 False
            super().__init__(
                0,
                0,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                False,  # bias 被设为 False，避免创建一个即将被覆盖的张量
                padding_mode,
                **factory_kwargs,
            )
            
            # 初始化 weight 属性为 UninitializedParameter 对象
            self.weight = UninitializedParameter(**factory_kwargs)
            self.out_channels = out_channels
            
            # 如果 bias 为 True，则初始化 bias 属性为 UninitializedParameter 对象
            if bias:
                self.bias = UninitializedParameter(**factory_kwargs)
    
        # 返回空间维度的数量，固定返回值为 2
        def _get_num_spatial_dims(self) -> int:
            return 2
# LazyConv3d继承自_LazyConvXdMixin和Conv3d，忽略类型检查错误
class LazyConv3d(_LazyConvXdMixin, Conv3d):  # type: ignore[misc]
    r"""A :class:`torch.nn.Conv3d` module with lazy initialization of the ``in_channels`` argument.

    The ``in_channels`` argument of the :class:`Conv3d` that is inferred from
    the ``input.size(1)``.
    The attributes that will be lazily initialized are `weight` and `bias`.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.

    Args:
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        padding_mode (str, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel
            elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``

    .. seealso:: :class:`torch.nn.Conv3d` and :class:`torch.nn.modules.lazy.LazyModuleMixin`
    """

    # 超类将此变量定义为None，因此需要"类型：忽略"以避免重新定义变量时的错误提示
    # cls_to_become被定义为Conv3d类型，忽略了类型分配检查
    cls_to_become = Conv3d  # type: ignore[assignment]

    def __init__(
        self,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: _size_3_t = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        # 调用超类的初始化方法，设置初始参数并创建卷积层
        super().__init__(
            0,  # in_channels设置为0，将在之后的forward中动态设置
            0,  # out_channels设置为0，将在之后的forward中动态设置
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            # bias被硬编码为False，以避免创建即将被覆盖的张量
            False,
            padding_mode,
            **factory_kwargs,
        )
        # 使用UninitializedParameter创建权重张量，并将其分配给self.weight
        self.weight = UninitializedParameter(**factory_kwargs)
        self.out_channels = out_channels
        # 如果bias为True，则使用UninitializedParameter创建偏置张量，并将其分配给self.bias
        if bias:
            self.bias = UninitializedParameter(**factory_kwargs)

    def _get_num_spatial_dims(self) -> int:
        return 3


# LazyConvTranspose1d继承自_LazyConvXdMixin和ConvTranspose1d，忽略类型检查错误
class LazyConvTranspose1d(_LazyConvXdMixin, ConvTranspose1d):  # type: ignore[misc]
    r"""A :class:`torch.nn.ConvTranspose1d` module with lazy initialization of the ``in_channels`` argument.
    """
    The ``in_channels`` argument of the :class:`ConvTranspose1d` that is inferred from
    the ``input.size(1)``.
    The attributes that will be lazily initialized are `weight` and `bias`.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.

    Args:
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): ``dilation * (kernel_size - 1) - padding`` zero-padding
            will be added to both sides of the input. Default: 0
        output_padding (int or tuple, optional): Additional size added to one side
            of the output shape. Default: 0
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1

    .. seealso:: :class:`torch.nn.ConvTranspose1d` and :class:`torch.nn.modules.lazy.LazyModuleMixin`
    """

    # super class define this variable as None. "type: ignore[..] is required
    # since we are redefining the variable.
    cls_to_become = ConvTranspose1d  # type: ignore[assignment]

    # Initialize method for ConvTranspose1d
    def __init__(
        self,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        output_padding: _size_1_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_1_t = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        # Keyword arguments for tensor factory
        factory_kwargs = {"device": device, "dtype": dtype}
        # Call superclass initialization with specified parameters
        super().__init__(
            0,  # in_channels set to 0 as it will be determined dynamically
            0,  # out_channels set to 0 as it will be determined dynamically
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            # bias is hardcoded to False to avoid creating tensor
            # that will soon be overwritten.
            False,
            dilation,
            padding_mode,
            **factory_kwargs,
        )
        # Initialize weight attribute as an uninitialized parameter
        self.weight = UninitializedParameter(**factory_kwargs)
        # Set the number of output channels
        self.out_channels = out_channels
        # Conditionally initialize bias attribute as an uninitialized parameter
        if bias:
            self.bias = UninitializedParameter(**factory_kwargs)

    # Method to retrieve the number of spatial dimensions
    def _get_num_spatial_dims(self) -> int:
        return 1
# LazyConvTranspose2d 类继承自 _LazyConvXdMixin 和 ConvTranspose2d 类，其中 weight 被定义为 Tensor，
# 但派生类中定义为 UninitializedParameter
class LazyConvTranspose2d(_LazyConvXdMixin, ConvTranspose2d):  # type: ignore[misc]
    r"""A :class:`torch.nn.ConvTranspose2d` module with lazy initialization of the ``in_channels`` argument.

    The ``in_channels`` argument of the :class:`ConvTranspose2d` is inferred from
    the ``input.size(1)``.
    The attributes that will be lazily initialized are `weight` and `bias`.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.

    Args:
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): ``dilation * (kernel_size - 1) - padding`` zero-padding
            will be added to both sides of each dimension in the input. Default: 0
        output_padding (int or tuple, optional): Additional size added to one side
            of each dimension in the output shape. Default: 0
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1

    .. seealso:: :class:`torch.nn.ConvTranspose2d` and :class:`torch.nn.modules.lazy.LazyModuleMixin`
    """

    # 超类将此变量定义为 None。"type: ignore[..]" 是必需的，因为我们正在重新定义变量。
    cls_to_become = ConvTranspose2d  # type: ignore[assignment]

    def __init__(
        self,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        output_padding: _size_2_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        # 准备用于创建 tensor 的参数字典
        factory_kwargs = {"device": device, "dtype": dtype}
        # 调用父类的初始化方法，设置各种参数，并将 bias 强制设置为 False，以避免创建即将被覆盖的 tensor
        super().__init__(
            0,
            0,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            False,  # bias 被硬编码为 False，以避免创建即将被覆盖的 tensor
            dilation,
            padding_mode,
            **factory_kwargs,
        )
        # 初始化 weight 属性为 UninitializedParameter 对象
        self.weight = UninitializedParameter(**factory_kwargs)
        self.out_channels = out_channels
        # 如果 bias 为 True，则初始化 bias 属性为 UninitializedParameter 对象
        if bias:
            self.bias = UninitializedParameter(**factory_kwargs)

    def _get_num_spatial_dims(self) -> int:
        # 返回空间维度的数量，此处为 2
        return 2


# LazyConvTranspose3d 定义了 weight 为 Tensor，但派生类中定义为 UninitializedParameter
class LazyConvTranspose3d(_LazyConvXdMixin, ConvTranspose3d):  # type: ignore[misc]
    r"""A :class:`torch.nn.ConvTranspose3d` module with lazy initialization of the ``in_channels`` argument.

    The ``in_channels`` argument of the :class:`ConvTranspose3d` is inferred from
    the ``input.size(1)``.
    The attributes that will be lazily initialized are `weight` and `bias`.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.

    Args:
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): ``dilation * (kernel_size - 1) - padding`` zero-padding
            will be added to both sides of each dimension in the input. Default: 0
        output_padding (int or tuple, optional): Additional size added to one side
            of each dimension in the output shape. Default: 0
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1

    .. seealso:: :class:`torch.nn.ConvTranspose3d` and :class:`torch.nn.modules.lazy.LazyModuleMixin`
    """

    # super class define this variable as None. "type: ignore[..] is required
    # since we are redefining the variable.
    cls_to_become = ConvTranspose3d  # type: ignore[assignment]

    def __init__(
        self,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: _size_3_t = 0,
        output_padding: _size_3_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_3_t = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        # Specify device and data type for factory kwargs
        factory_kwargs = {"device": device, "dtype": dtype}
        # Initialize parent class with necessary parameters
        super().__init__(
            0,  # in_channels is set to 0 as it will be lazily initialized
            0,  # unused parameter for padding
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            # bias is hardcoded to False to avoid creating tensor
            # that will soon be overwritten.
            False,
            dilation,
            padding_mode,
            **factory_kwargs,
        )
        # Initialize weight as an UninitializedParameter
        self.weight = UninitializedParameter(**factory_kwargs)
        # Set the number of output channels
        self.out_channels = out_channels
        # Initialize bias as UninitializedParameter if bias is True
        if bias:
            self.bias = UninitializedParameter(**factory_kwargs)

    def _get_num_spatial_dims(self) -> int:
        # Return the number of spatial dimensions (3 for ConvTranspose3d)
        return 3
```