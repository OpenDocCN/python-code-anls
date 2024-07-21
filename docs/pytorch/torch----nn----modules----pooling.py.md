# `.\pytorch\torch\nn\modules\pooling.py`

```
# 导入所需的类型定义
from typing import List, Optional

# 导入 torch 中的函数和数据类型
import torch.nn.functional as F
from torch import Tensor

# 导入 torch.nn.common_types 中的特定类型
from torch.nn.common_types import (
    _ratio_2_t,
    _ratio_3_t,
    _size_1_t,
    _size_2_opt_t,
    _size_2_t,
    _size_3_opt_t,
    _size_3_t,
    _size_any_opt_t,
    _size_any_t,
)

# 从当前目录下的 module.py 文件导入 Module 类
from .module import Module
# 从当前目录下的 utils.py 文件导入一些函数：_pair, _single, _triple
from .utils import _pair, _single, _triple

# 定义公开的类和函数列表，这些可以在模块外部使用
__all__ = [
    "MaxPool1d",
    "MaxPool2d",
    "MaxPool3d",
    "MaxUnpool1d",
    "MaxUnpool2d",
    "MaxUnpool3d",
    "AvgPool1d",
    "AvgPool2d",
    "AvgPool3d",
    "FractionalMaxPool2d",
    "FractionalMaxPool3d",
    "LPPool1d",
    "LPPool2d",
    "LPPool3d",
    "AdaptiveMaxPool1d",
    "AdaptiveMaxPool2d",
    "AdaptiveMaxPool3d",
    "AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d",
]

# 定义一个抽象基类 _MaxPoolNd，继承自 Module 类
class _MaxPoolNd(Module):
    # 常量列表，指定了该类实例的固定属性
    __constants__ = [
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "return_indices",
        "ceil_mode",
    ]
    return_indices: bool  # 返回索引的标志
    ceil_mode: bool  # 向上取整模式的标志

    # 初始化方法，接受多个参数并进行初始化
    def __init__(
        self,
        kernel_size: _size_any_t,         # 池化核大小，可以是多种类型之一
        stride: Optional[_size_any_t] = None,  # 步幅大小，可选参数，默认为池化核大小
        padding: _size_any_t = 0,         # 填充大小，可以是多种类型之一，默认为 0
        dilation: _size_any_t = 1,        # 空洞卷积的膨胀率，可以是多种类型之一，默认为 1
        return_indices: bool = False,     # 是否返回池化结果的索引，默认为 False
        ceil_mode: bool = False,          # 是否使用向上取整模式，默认为 False
    ) -> None:
        super().__init__()  # 调用父类 Module 的初始化方法
        self.kernel_size = kernel_size  # 设置池化核大小
        self.stride = stride if (stride is not None) else kernel_size  # 设置步幅大小，默认为池化核大小
        self.padding = padding  # 设置填充大小
        self.dilation = dilation  # 设置空洞卷积的膨胀率
        self.return_indices = return_indices  # 设置是否返回池化结果的索引
        self.ceil_mode = ceil_mode  # 设置是否使用向上取整模式

    # 返回一个描述对象的额外字符串表示形式
    def extra_repr(self) -> str:
        return (
            "kernel_size={kernel_size}, stride={stride}, padding={padding}"
            ", dilation={dilation}, ceil_mode={ceil_mode}".format(**self.__dict__)
        )

# MaxPool1d 类继承自 _MaxPoolNd 类，用于执行一维最大池化操作
class MaxPool1d(_MaxPoolNd):
    r"""Applies a 1D max pooling over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, L)`
    and output :math:`(N, C, L_{out})` can be precisely described as:

    .. math::
        out(N_i, C_j, k) = \max_{m=0, \ldots, \text{kernel\_size} - 1}
                input(N_i, C_j, stride \times k + m)

    If :attr:`padding` is non-zero, then the input is implicitly padded with negative infinity on both sides
    for :attr:`padding` number of points. :attr:`dilation` is the stride between the elements within the
    sliding window. This `link`_ has a nice visualization of the pooling parameters.

    Note:
        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
        or the input. Sliding windows that would start in the right padded region are ignored.
    ```
    ```
    # 定义一个类，实现一维最大池化操作
    kernel_size: _size_1_t
    stride: _size_1_t
    padding: _size_1_t
    dilation: _size_1_t
    
    # 类的初始化方法，接收输入张量并进行最大池化操作
    def forward(self, input: Tensor):
        # 使用 torch.nn.functional 中的 max_pool1d 函数进行一维最大池化
        return F.max_pool1d(
            input,  # 输入张量
            self.kernel_size,  # 池化窗口大小
            self.stride,  # 池化窗口的步幅
            self.padding,  # 输入的填充数
            self.dilation,  # 卷积核之间的步长
            ceil_mode=self.ceil_mode,  # 是否使用 ceil 模式计算输出形状
            return_indices=self.return_indices,  # 如果为 True，则同时返回最大值的位置索引
        )
class MaxPool2d(_MaxPoolNd):
    r"""Applies a 2D max pooling over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,
    output :math:`(N, C, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kH, kW)`
    can be precisely described as:

    .. math::
        \begin{aligned}
            out(N_i, C_j, h, w) ={} & \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1} \\
                                    & \text{input}(N_i, C_j, \text{stride[0]} \times h + m,
                                                   \text{stride[1]} \times w + n)
        \end{aligned}

    If :attr:`padding` is non-zero, then the input is implicitly padded with negative infinity on both sides
    for :attr:`padding` number of points. :attr:`dilation` controls the spacing between the kernel points.
    It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    Note:
        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
        or the input. Sliding windows that would start in the right padded region are ignored.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Args:
        kernel_size: the size of the window to take a max over
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: Implicit negative infinity padding to be added on both sides
        dilation: a parameter that controls the stride of elements in the window
        return_indices: if ``True``, will return the max indices along with the outputs.
                        Useful for :class:`torch.nn.MaxUnpool2d` later
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 * \text{padding[0]} - \text{dilation[0]}
                    \times (\text{kernel\_size[0]} - 1) - 1}{\text{stride[0]}} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 * \text{padding[1]} - \text{dilation[1]}
                    \times (\text{kernel\_size[1]} - 1) - 1}{\text{stride[1]}} + 1\right\rfloor
    """
    Examples::

        # 创建一个最大池化层，使用方形窗口大小为3，步幅为2
        m = nn.MaxPool2d(3, stride=2)
        # 创建一个最大池化层，使用非方形窗口大小为(3, 2)，步幅为(2, 1)
        m = nn.MaxPool2d((3, 2), stride=(2, 1))
        # 创建一个输入张量，大小为(20, 16, 50, 32)
        input = torch.randn(20, 16, 50, 32)
        # 对输入张量应用最大池化层，将结果存储在output中
        output = m(input)

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    # 定义最大池化层的核大小、步幅、填充和扩展率
    kernel_size: _size_2_t
    stride: _size_2_t
    padding: _size_2_t
    dilation: _size_2_t

    # 前向传播函数，将输入张量input应用到最大池化操作中
    def forward(self, input: Tensor):
        return F.max_pool2d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=self.return_indices,
        )
# 定义 MaxPool3d 类，继承自 _MaxPoolNd 类
class MaxPool3d(_MaxPoolNd):
    r"""Applies a 3D max pooling over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, D, H, W)`,
    output :math:`(N, C, D_{out}, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kD, kH, kW)`
    can be precisely described as:

    .. math::
        \begin{aligned}
            \text{out}(N_i, C_j, d, h, w) ={} & \max_{k=0, \ldots, kD-1} \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1} \\
                                              & \text{input}(N_i, C_j, \text{stride[0]} \times d + k,
                                                             \text{stride[1]} \times h + m, \text{stride[2]} \times w + n)
        \end{aligned}

    If :attr:`padding` is non-zero, then the input is implicitly padded with negative infinity on both sides
    for :attr:`padding` number of points. :attr:`dilation` controls the spacing between the kernel points.
    It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    Note:
        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
        or the input. Sliding windows that would start in the right padded region are ignored.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the depth, height and width dimension
        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,
          the second `int` for the height dimension and the third `int` for the width dimension

    Args:
        kernel_size: the size of the window to take a max over
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: Implicit negative infinity padding to be added on all three sides
        dilation: a parameter that controls the stride of elements in the window
        return_indices: if ``True``, will return the max indices along with the outputs.
                        Useful for :class:`torch.nn.MaxUnpool3d` later
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape
    # 定义网络层的输入和输出形状描述
    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})` or :math:`(C, D_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` or :math:`(C, D_{out}, H_{out}, W_{out})`, where

          .. math::
              D_{out} = \left\lfloor\frac{D_{in} + 2 \times \text{padding}[0] - \text{dilation}[0] \times
                (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 \times \text{padding}[1] - \text{dilation}[1] \times
                (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 \times \text{padding}[2] - \text{dilation}[2] \times
                (\text{kernel\_size}[2] - 1) - 1}{\text{stride}[2]} + 1\right\rfloor

    Examples::

        >>> # pool of square window of size=3, stride=2
        >>> m = nn.MaxPool3d(3, stride=2)
        >>> # pool of non-square window
        >>> m = nn.MaxPool3d((3, 2, 2), stride=(2, 1, 2))
        >>> input = torch.randn(20, 16, 50, 44, 31)
        >>> output = m(input)

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """  # noqa: E501

    # 定义卷积核大小、步幅、填充、扩张参数
    kernel_size: _size_3_t
    stride: _size_3_t
    padding: _size_3_t
    dilation: _size_3_t

    # 定义前向传播函数，接收一个张量输入，并进行最大池化操作
    def forward(self, input: Tensor):
        return F.max_pool3d(
            input,                      # 输入张量
            self.kernel_size,           # 卷积核大小
            self.stride,                # 步幅
            self.padding,               # 填充
            self.dilation,              # 扩张
            ceil_mode=self.ceil_mode,   # 是否使用天花板模式
            return_indices=self.return_indices,  # 是否返回池化最大值的索引
        )
class _MaxUnpoolNd(Module):
    # 定义一个名为 _MaxUnpoolNd 的类，继承自 Module
    def extra_repr(self) -> str:
        # 返回一个字符串，描述了该类的核大小、步幅和填充属性
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"


class MaxUnpool1d(_MaxUnpoolNd):
    r"""Computes a partial inverse of :class:`MaxPool1d`.

    :class:`MaxPool1d` is not fully invertible, since the non-maximal values are lost.

    :class:`MaxUnpool1d` takes in as input the output of :class:`MaxPool1d`
    including the indices of the maximal values and computes a partial inverse
    in which all non-maximal values are set to zero.

    Note:
        This operation may behave nondeterministically when the input indices has repeat values.
        See https://github.com/pytorch/pytorch/issues/80827 and :doc:`/notes/randomness` for more information.

    .. note:: :class:`MaxPool1d` can map several input sizes to the same output
              sizes. Hence, the inversion process can get ambiguous.
              To accommodate this, you can provide the needed output size
              as an additional argument :attr:`output_size` in the forward call.
              See the Inputs and Example below.

    Args:
        kernel_size (int or tuple): Size of the max pooling window.
        stride (int or tuple): Stride of the max pooling window.
            It is set to :attr:`kernel_size` by default.
        padding (int or tuple): Padding that was added to the input

    Inputs:
        - `input`: the input Tensor to invert
        - `indices`: the indices given out by :class:`~torch.nn.MaxPool1d`
        - `output_size` (optional): the targeted output size

    Shape:
        - Input: :math:`(N, C, H_{in})` or :math:`(C, H_{in})`.
        - Output: :math:`(N, C, H_{out})` or :math:`(C, H_{out})`, where

          .. math::
              H_{out} = (H_{in} - 1) \times \text{stride}[0] - 2 \times \text{padding}[0] + \text{kernel\_size}[0]

          or as given by :attr:`output_size` in the call operator

    Example::

        >>> # xdoctest: +IGNORE_WANT("do other tests modify the global state?")
        >>> pool = nn.MaxPool1d(2, stride=2, return_indices=True)
        >>> unpool = nn.MaxUnpool1d(2, stride=2)
        >>> input = torch.tensor([[[1., 2, 3, 4, 5, 6, 7, 8]]])
        >>> output, indices = pool(input)
        >>> unpool(output, indices)
        tensor([[[ 0.,  2.,  0.,  4.,  0.,  6.,  0., 8.]]])

        >>> # Example showcasing the use of output_size
        >>> input = torch.tensor([[[1., 2, 3, 4, 5, 6, 7, 8, 9]]])
        >>> output, indices = pool(input)
        >>> unpool(output, indices, output_size=input.size())
        tensor([[[ 0.,  2.,  0.,  4.,  0.,  6.,  0., 8.,  0.]]])

        >>> unpool(output, indices)
        tensor([[[ 0.,  2.,  0.,  4.,  0.,  6.,  0., 8.]]])
    """

    kernel_size: _size_1_t
    stride: _size_1_t
    padding: _size_1_t

    def __init__(
        self,
        kernel_size: _size_1_t,
        stride: Optional[_size_1_t] = None,
        padding: _size_1_t = 0,
    ):
        # 初始化 MaxUnpool1d 类，设置核大小、步幅和填充属性
        super(MaxUnpool1d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
    ) -> None:
        # 调用父类的初始化方法
        super().__init__()
        # 确定卷积核大小，并转换为单元素元组
        self.kernel_size = _single(kernel_size)
        # 确定步长，如果未指定则使用卷积核大小，并转换为单元素元组
        self.stride = _single(stride if (stride is not None) else kernel_size)
        # 确定填充大小，并转换为单元素元组
        self.padding = _single(padding)

    def forward(
        self, input: Tensor, indices: Tensor, output_size: Optional[List[int]] = None
    ) -> Tensor:
        # 调用 torch.nn.functional 中的 max_unpool1d 函数，进行最大池化的反池化操作
        return F.max_unpool1d(
            input, indices, self.kernel_size, self.stride, self.padding, output_size
        )
class MaxUnpool2d(_MaxUnpoolNd):
    r"""Computes a partial inverse of :class:`MaxPool2d`.

    :class:`MaxPool2d` is not fully invertible, since the non-maximal values are lost.

    :class:`MaxUnpool2d` takes in as input the output of :class:`MaxPool2d`
    including the indices of the maximal values and computes a partial inverse
    in which all non-maximal values are set to zero.

    Note:
        This operation may behave nondeterministically when the input indices has repeat values.
        See https://github.com/pytorch/pytorch/issues/80827 and :doc:`/notes/randomness` for more information.

    .. note:: :class:`MaxPool2d` can map several input sizes to the same output
              sizes. Hence, the inversion process can get ambiguous.
              To accommodate this, you can provide the needed output size
              as an additional argument :attr:`output_size` in the forward call.
              See the Inputs and Example below.

    Args:
        kernel_size (int or tuple): Size of the max pooling window.
        stride (int or tuple): Stride of the max pooling window.
            It is set to :attr:`kernel_size` by default.
        padding (int or tuple): Padding that was added to the input

    Inputs:
        - `input`: the input Tensor to invert
        - `indices`: the indices given out by :class:`~torch.nn.MaxPool2d`
        - `output_size` (optional): the targeted output size

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where

          .. math::
            H_{out} = (H_{in} - 1) \times \text{stride[0]} - 2 \times \text{padding[0]} + \text{kernel\_size[0]}

          .. math::
            W_{out} = (W_{in} - 1) \times \text{stride[1]} - 2 \times \text{padding[1]} + \text{kernel\_size[1]}

          or as given by :attr:`output_size` in the call operator
    Example::

        >>> pool = nn.MaxPool2d(2, stride=2, return_indices=True)
        >>> unpool = nn.MaxUnpool2d(2, stride=2)
        >>> input = torch.tensor([[[[ 1.,  2.,  3.,  4.],
                                    [ 5.,  6.,  7.,  8.],
                                    [ 9., 10., 11., 12.],
                                    [13., 14., 15., 16.]]]])
        >>> output, indices = pool(input)
        >>> unpool(output, indices)
        tensor([[[[  0.,   0.,   0.,   0.],
                  [  0.,   6.,   0.,   8.],
                  [  0.,   0.,   0.,   0.],
                  [  0.,  14.,   0.,  16.]]]])
        >>> # Now using output_size to resolve an ambiguous size for the inverse
        >>> input = torch.tensor([[[[ 1.,  2.,  3., 4., 5.],
                                    [ 6.,  7.,  8., 9., 10.],
                                    [11., 12., 13., 14., 15.],
                                    [16., 17., 18., 19., 20.]]]])
        >>> output, indices = pool(input)
        >>> # This call will not work without specifying output_size
        >>> unpool(output, indices, output_size=input.size())
        tensor([[[[ 0.,  0.,  0.,  0.,  0.],
                  [ 0.,  7.,  0.,  9.,  0.],
                  [ 0.,  0.,  0.,  0.,  0.],
                  [ 0., 17.,  0., 19.,  0.]]]])

    """

    kernel_size: _size_2_t
    stride: _size_2_t
    padding: _size_2_t

    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: Optional[_size_2_t] = None,
        padding: _size_2_t = 0,
    ) -> None:
        # 初始化函数，设置最大池化的核大小、步长和填充
        super().__init__()
        # 将核大小转换为元组形式
        self.kernel_size = _pair(kernel_size)
        # 如果未指定步长，则默认与核大小相同
        self.stride = _pair(stride if (stride is not None) else kernel_size)
        # 设置填充大小为指定值或默认为0
        self.padding = _pair(padding)

    def forward(
        self, input: Tensor, indices: Tensor, output_size: Optional[List[int]] = None
    ) -> Tensor:
        # 调用PyTorch中的最大反池化函数进行前向传播
        return F.max_unpool2d(
            input, indices, self.kernel_size, self.stride, self.padding, output_size
        )
class MaxUnpool3d(_MaxUnpoolNd):
    r"""Computes a partial inverse of :class:`MaxPool3d`.

    :class:`MaxPool3d` is not fully invertible, since the non-maximal values are lost.
    :class:`MaxUnpool3d` takes in as input the output of :class:`MaxPool3d`
    including the indices of the maximal values and computes a partial inverse
    in which all non-maximal values are set to zero.

    Note:
        This operation may behave nondeterministically when the input indices has repeat values.
        See https://github.com/pytorch/pytorch/issues/80827 and :doc:`/notes/randomness` for more information.

    .. note:: :class:`MaxPool3d` can map several input sizes to the same output
              sizes. Hence, the inversion process can get ambiguous.
              To accommodate this, you can provide the needed output size
              as an additional argument :attr:`output_size` in the forward call.
              See the Inputs section below.

    Args:
        kernel_size (int or tuple): Size of the max pooling window.
        stride (int or tuple): Stride of the max pooling window.
            It is set to :attr:`kernel_size` by default.
        padding (int or tuple): Padding that was added to the input

    Attributes:
        kernel_size (_size_3_t): Size of the max pooling window as a triple.
        stride (_size_3_t): Stride of the max pooling window as a triple.
        padding (_size_3_t): Padding added to the input as a triple.

    Inputs:
        - `input`: the input Tensor to invert
        - `indices`: the indices given out by :class:`~torch.nn.MaxPool3d`
        - `output_size` (optional): the targeted output size

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})` or :math:`(C, D_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` or :math:`(C, D_{out}, H_{out}, W_{out})`, where

          .. math::
              D_{out} = (D_{in} - 1) \times \text{stride[0]} - 2 \times \text{padding[0]} + \text{kernel\_size[0]}

          .. math::
              H_{out} = (H_{in} - 1) \times \text{stride[1]} - 2 \times \text{padding[1]} + \text{kernel\_size[1]}

          .. math::
              W_{out} = (W_{in} - 1) \times \text{stride[2]} - 2 \times \text{padding[2]} + \text{kernel\_size[2]}

          or as given by :attr:`output_size` in the call operator

    Example::

        >>> # pool of square window of size=3, stride=2
        >>> pool = nn.MaxPool3d(3, stride=2, return_indices=True)
        >>> unpool = nn.MaxUnpool3d(3, stride=2)
        >>> output, indices = pool(torch.randn(20, 16, 51, 33, 15))
        >>> unpooled_output = unpool(output, indices)
        >>> unpooled_output.size()
        torch.Size([20, 16, 51, 33, 15])
    """

    kernel_size: _size_3_t
    stride: _size_3_t
    padding: _size_3_t

    def __init__(
        self,
        kernel_size: _size_3_t,
        stride: Optional[_size_3_t] = None,
        padding: _size_3_t = 0,
    ) -> None:
        """
        Initialize MaxUnpool3d module with kernel size, stride, and padding.
        
        Args:
            kernel_size (int or tuple): Size of the max pooling window.
            stride (int or tuple, optional): Stride of the max pooling window.
                Defaults to `kernel_size`.
            padding (int or tuple): Padding that was added to the input.
        """
        super().__init__()
        # Convert kernel_size to a triple if it's not already
        self.kernel_size = _triple(kernel_size)
        # Set stride to kernel_size if not provided
        self.stride = _triple(stride if (stride is not None) else kernel_size)
        # Convert padding to a triple
        self.padding = _triple(padding)
    # 定义一个方法 `forward`，用于执行神经网络的前向传播操作
    def forward(
        self, input: Tensor, indices: Tensor, output_size: Optional[List[int]] = None
    ) -> Tensor:
        # 调用 PyTorch 的 `F.max_unpool3d` 函数，执行3D最大池化的反向操作，实现最大解池化
        # 参数解释：
        # input: 输入的张量，通常是经过池化后的结果
        # indices: 池化过程中记录的最大值位置的张量
        # self.kernel_size: 池化核的大小
        # self.stride: 池化时的步长
        # self.padding: 池化时的填充
        # output_size: 可选参数，输出张量的大小，用于指定解池化后的大小
        return F.max_unpool3d(
            input, indices, self.kernel_size, self.stride, self.padding, output_size
        )
class _AvgPoolNd(Module):
    __constants__ = [
        "kernel_size",
        "stride",
        "padding",
        "ceil_mode",
        "count_include_pad",
    ]

    # 返回池化层的额外描述信息，包括核大小、步幅和填充
    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"


class AvgPool1d(_AvgPoolNd):
    r"""Applies a 1D average pooling over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, L)`,
    output :math:`(N, C, L_{out})` and :attr:`kernel_size` :math:`k`
    can be precisely described as:

    .. math::

        \text{out}(N_i, C_j, l) = \frac{1}{k} \sum_{m=0}^{k-1}
                               \text{input}(N_i, C_j, \text{stride} \times l + m)

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
    for :attr:`padding` number of points.

    Note:
        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
        or the input. Sliding windows that would start in the right padded region are ignored.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding` can each be
    an ``int`` or a one-element tuple.

    Args:
        kernel_size: the size of the window
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on both sides
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape
        count_include_pad: when True, will include the zero-padding in the averaging calculation

    Shape:
        - Input: :math:`(N, C, L_{in})` or :math:`(C, L_{in})`.
        - Output: :math:`(N, C, L_{out})` or :math:`(C, L_{out})`, where

          .. math::
              L_{out} = \left\lfloor \frac{L_{in} +
              2 \times \text{padding} - \text{kernel\_size}}{\text{stride}} + 1\right\rfloor

          Per the note above, if ``ceil_mode`` is True and :math:`(L_{out} - 1) \times \text{stride} \geq L_{in}
          + \text{padding}`, we skip the last window as it would start in the right padded region, resulting in
          :math:`L_{out}` being reduced by one.

    Examples::

        >>> # pool with window of size=3, stride=2
        >>> m = nn.AvgPool1d(3, stride=2)
        >>> m(torch.tensor([[[1., 2, 3, 4, 5, 6, 7]]]))
        tensor([[[2., 4., 6.]]])
    """

    kernel_size: _size_1_t
    stride: _size_1_t
    padding: _size_1_t
    ceil_mode: bool
    count_include_pad: bool

    # 初始化函数，设定池化层的参数
    def __init__(
        self,
        kernel_size: _size_1_t,
        stride: _size_1_t = None,
        padding: _size_1_t = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> None:
        # 调用父类构造函数，初始化基类
        super().__init__()
        # 设置卷积核大小，确保是单值或者元组
        self.kernel_size = _single(kernel_size)
        # 设置步长，如果未指定步长则使用卷积核大小
        self.stride = _single(stride if stride is not None else kernel_size)
        # 设置填充大小，确保是单值或者元组
        self.padding = _single(padding)
        # 设置是否使用向上取整模式
        self.ceil_mode = ceil_mode
        # 设置是否包含填充计数
        self.count_include_pad = count_include_pad

    def forward(self, input: Tensor) -> Tensor:
        # 调用PyTorch中的avg_pool1d函数进行平均池化操作
        return F.avg_pool1d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
        )
class AvgPool2d(_AvgPoolNd):
    r"""Applies a 2D average pooling over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,
    output :math:`(N, C, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kH, kW)`
    can be precisely described as:

    .. math::

        out(N_i, C_j, h, w)  = \frac{1}{kH * kW} \sum_{m=0}^{kH-1} \sum_{n=0}^{kW-1}
                               input(N_i, C_j, stride[0] \times h + m, stride[1] \times w + n)

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
    for :attr:`padding` number of points.

    Note:
        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
        or the input. Sliding windows that would start in the right padded region are ignored.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Args:
        kernel_size: the size of the window
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on both sides
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape
        count_include_pad: when True, will include the zero-padding in the averaging calculation
        divisor_override: if specified, it will be used as divisor, otherwise size of the pooling region will be used.


    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] -
                \text{kernel\_size}[0]}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] -
                \text{kernel\_size}[1]}{\text{stride}[1]} + 1\right\rfloor

          Per the note above, if ``ceil_mode`` is True and :math:`(H_{out} - 1)\times \text{stride}[0]\geq H_{in}
          + \text{padding}[0]`, we skip the last window as it would start in the bottom padded region,
          resulting in :math:`H_{out}` being reduced by one.

          The same applies for :math:`W_{out}`.

    Examples::

        >>> # pool of square window of size=3, stride=2
        >>> m = nn.AvgPool2d(3, stride=2)
        >>> # pool of non-square window
        >>> m = nn.AvgPool2d((3, 2), stride=(2, 1))
        >>> input = torch.randn(20, 16, 50, 32)
        >>> output = m(input)
    """
    # 定义一个类的常量列表，包含了池化层的参数信息
    __constants__ = [
        "kernel_size",             # 池化核大小
        "stride",                  # 步长
        "padding",                 # 填充大小
        "ceil_mode",               # 是否启用向上取整模式
        "count_include_pad",       # 是否包含填充在内
        "divisor_override",        # 除数覆盖（用于平均池化）
    ]
    
    # 定义类的属性，用于指定池化核大小、步长、填充大小等信息
    kernel_size: _size_2_t         # 池化核大小，类型为 _size_2_t
    stride: _size_2_t              # 步长，类型为 _size_2_t
    padding: _size_2_t             # 填充大小，类型为 _size_2_t
    ceil_mode: bool                # 是否启用向上取整模式，类型为布尔值
    count_include_pad: bool        # 是否包含填充在内，类型为布尔值
    
    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: Optional[_size_2_t] = None,
        padding: _size_2_t = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Optional[int] = None,
    ) -> None:
        # 调用父类的构造函数
        super().__init__()
        # 初始化池化层的参数
        self.kernel_size = kernel_size                          # 设置池化核大小
        self.stride = stride if (stride is not None) else kernel_size  # 设置步长，如果未指定则使用池化核大小
        self.padding = padding                                  # 设置填充大小
        self.ceil_mode = ceil_mode                              # 设置是否启用向上取整模式
        self.count_include_pad = count_include_pad              # 设置是否包含填充在内
        self.divisor_override = divisor_override                # 设置除数覆盖（用于平均池化）
    
    def forward(self, input: Tensor) -> Tensor:
        # 调用 PyTorch 的平均池化函数 F.avg_pool2d 进行前向传播
        return F.avg_pool2d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
            self.divisor_override,
        )
class AvgPool3d(_AvgPoolNd):
    r"""Applies a 3D average pooling over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, D, H, W)`,
    output :math:`(N, C, D_{out}, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kD, kH, kW)`
    can be precisely described as:

    .. math::
        \begin{aligned}
            \text{out}(N_i, C_j, d, h, w) ={} & \sum_{k=0}^{kD-1} \sum_{m=0}^{kH-1} \sum_{n=0}^{kW-1} \\
                                              & \frac{\text{input}(N_i, C_j, \text{stride}[0] \times d + k,
                                                      \text{stride}[1] \times h + m, \text{stride}[2] \times w + n)}
                                                     {kD \times kH \times kW}
        \end{aligned}

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on all three sides
    for :attr:`padding` number of points.

    Note:
        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
        or the input. Sliding windows that would start in the right padded region are ignored.

    The parameters :attr:`kernel_size`, :attr:`stride` can either be:

        - a single ``int`` -- in which case the same value is used for the depth, height and width dimension
        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,
          the second `int` for the height dimension and the third `int` for the width dimension

    Args:
        kernel_size: the size of the window
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on all three sides
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape
        count_include_pad: when True, will include the zero-padding in the averaging calculation
        divisor_override: if specified, it will be used as divisor, otherwise :attr:`kernel_size` will be used
    """
    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})` or :math:`(C, D_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` or
          :math:`(C, D_{out}, H_{out}, W_{out})`, where

          .. math::
              D_{out} = \left\lfloor\frac{D_{in} + 2 \times \text{padding}[0] -
                    \text{kernel\_size}[0]}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 \times \text{padding}[1] -
                    \text{kernel\_size}[1]}{\text{stride}[1]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 \times \text{padding}[2] -
                    \text{kernel\_size}[2]}{\text{stride}[2]} + 1\right\rfloor

          根据输入和卷积操作参数计算输出的空间尺寸，包括通过向下取整计算输出体积的公式。

          根据上述注意事项，如果 ``ceil_mode`` 为 True 并且 :math:`(D_{out} - 1)\times \text{stride}[0]\geq D_{in}
          + \text{padding}[0]`，则会跳过最后一个窗口，因为它会开始在填充区域内，导致 :math:`D_{out}` 减少一次。

          对于 :math:`W_{out}` 和 :math:`H_{out}`，也适用相同的规则。

    Examples::

        >>> # pool of square window of size=3, stride=2
        >>> m = nn.AvgPool3d(3, stride=2)
        >>> # pool of non-square window
        >>> m = nn.AvgPool3d((3, 2, 2), stride=(2, 1, 2))
        >>> input = torch.randn(20, 16, 50, 44, 31)
        >>> output = m(input)
    """

    __constants__ = [
        "kernel_size",
        "stride",
        "padding",
        "ceil_mode",
        "count_include_pad",
        "divisor_override",
    ]

    # 定义类的常量，包括卷积操作的参数名称

    kernel_size: _size_3_t
    stride: _size_3_t
    padding: _size_3_t
    ceil_mode: bool
    count_include_pad: bool

    def __init__(
        self,
        kernel_size: _size_3_t,
        stride: Optional[_size_3_t] = None,
        padding: _size_3_t = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Optional[int] = None,
    ) -> None:
        # 初始化方法，设置卷积操作的参数
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def forward(self, input: Tensor) -> Tensor:
        # 前向传播方法，调用 F.avg_pool3d 进行平均池化操作
        return F.avg_pool3d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
            self.divisor_override,
        )

    def __setstate__(self, d):
        # 设置状态的方法，确保对象的属性正确初始化
        super().__setstate__(d)
        self.__dict__.setdefault("padding", 0)
        self.__dict__.setdefault("ceil_mode", False)
        self.__dict__.setdefault("count_include_pad", True)
class FractionalMaxPool2d(Module):
    r"""Applies a 2D fractional max pooling over an input signal composed of several input planes.

    Fractional MaxPooling is described in detail in the paper `Fractional MaxPooling`_ by Ben Graham

    The max-pooling operation is applied in :math:`kH \times kW` regions by a stochastic
    step size determined by the target output size.
    The number of output features is equal to the number of input planes.

    .. note:: Exactly one of ``output_size`` or ``output_ratio`` must be defined.

    Args:
        kernel_size: the size of the window to take a max over.
                     Can be a single number k (for a square kernel of k x k) or a tuple `(kh, kw)`
        output_size: the target output size of the image of the form `oH x oW`.
                     Can be a tuple `(oH, oW)` or a single number oH for a square image `oH x oH`.
                     Note that we must have :math:`kH + oH - 1 <= H_{in}` and :math:`kW + oW - 1 <= W_{in}`
        output_ratio: If one wants to have an output size as a ratio of the input size, this option can be given.
                      This has to be a number or tuple in the range (0, 1).
                      Note that we must have :math:`kH + (output\_ratio\_H * H_{in}) - 1 <= H_{in}`
                      and :math:`kW + (output\_ratio\_W * W_{in}) - 1 <= W_{in}`
        return_indices: if ``True``, will return the indices along with the outputs.
                        Useful to pass to :meth:`nn.MaxUnpool2d`. Default: ``False``

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where
          :math:`(H_{out}, W_{out})=\text{output\_size}` or
          :math:`(H_{out}, W_{out})=\text{output\_ratio} \times (H_{in}, W_{in})`.

    Examples:
        >>> # pool of square window of size=3, and target output size 13x12
        >>> m = nn.FractionalMaxPool2d(3, output_size=(13, 12))
        >>> # pool of square window and target output size being half of input image size
        >>> m = nn.FractionalMaxPool2d(3, output_ratio=(0.5, 0.5))
        >>> input = torch.randn(20, 16, 50, 32)
        >>> output = m(input)

    .. _Fractional MaxPooling:
        https://arxiv.org/abs/1412.6071
    """

    __constants__ = ["kernel_size", "return_indices", "output_size", "output_ratio"]

    kernel_size: _size_2_t  # Size of the pooling window in 2D, can be a single number or a tuple (kh, kw)
    return_indices: bool  # Indicates whether to return pooling indices along with outputs
    output_size: _size_2_t  # Target output size of the pooled image, either as a tuple (oH, oW) or a single number oH
    output_ratio: _ratio_2_t  # Output size as a ratio of the input size, either a number or a tuple (ratio_H, ratio_W)

    def __init__(
        self,
        kernel_size: _size_2_t,
        output_size: Optional[_size_2_t] = None,
        output_ratio: Optional[_ratio_2_t] = None,
        return_indices: bool = False,
        _random_samples=None,
    # 初始化函数，用于设置 FractionalMaxPool2d 层的参数和检查合法性
    ) -> None:
        super().__init__()  # 调用父类的初始化方法
        self.kernel_size = _pair(kernel_size)  # 设置池化核大小，并转换为二元组
        self.return_indices = return_indices  # 是否返回池化操作的索引
        self.register_buffer("_random_samples", _random_samples)  # 注册随机样本的缓冲区
        self.output_size = _pair(output_size) if output_size is not None else None  # 设置输出大小，并转换为二元组，如果未指定则为 None
        self.output_ratio = _pair(output_ratio) if output_ratio is not None else None  # 设置输出比例，并转换为二元组，如果未指定则为 None
        if output_size is None and output_ratio is None:
            # 如果未指定输出大小和输出比例，则引发值错误
            raise ValueError(
                "FractionalMaxPool2d requires specifying either "
                "an output size, or a pooling ratio"
            )
        if output_size is not None and output_ratio is not None:
            # 如果同时指定了输出大小和输出比例，则引发值错误
            raise ValueError(
                "only one of output_size and output_ratio may be specified"
            )
        if self.output_ratio is not None:
            # 如果指定了输出比例，则检查其合法性
            if not (0 < self.output_ratio[0] < 1 and 0 < self.output_ratio[1] < 1):
                raise ValueError(
                    f"output_ratio must be between 0 and 1 (got {output_ratio})"
                )

    def forward(self, input: Tensor):
        # 执行前向传播，调用 PyTorch 中的 fractional_max_pool2d 函数
        return F.fractional_max_pool2d(
            input,  # 输入张量
            self.kernel_size,  # 池化核大小
            self.output_size,  # 输出大小
            self.output_ratio,  # 输出比例
            self.return_indices,  # 是否返回索引
            _random_samples=self._random_samples,  # 随机样本
        )
class FractionalMaxPool3d(Module):
    r"""Applies a 3D fractional max pooling over an input signal composed of several input planes.

    Fractional MaxPooling is described in detail in the paper `Fractional MaxPooling`_ by Ben Graham

    The max-pooling operation is applied in :math:`kT \times kH \times kW` regions by a stochastic
    step size determined by the target output size.
    The number of output features is equal to the number of input planes.

    .. note:: Exactly one of ``output_size`` or ``output_ratio`` must be defined.

    Args:
        kernel_size: the size of the window to take a max over.
                     Can be a single number k (for a square kernel of k x k x k) or a tuple `(kt x kh x kw)`
        output_size: the target output size of the image of the form `oT x oH x oW`.
                     Can be a tuple `(oT, oH, oW)` or a single number oH for a square image `oH x oH x oH`
        output_ratio: If one wants to have an output size as a ratio of the input size, this option can be given.
                      This has to be a number or tuple in the range (0, 1)
        return_indices: if ``True``, will return the indices along with the outputs.
                        Useful to pass to :meth:`nn.MaxUnpool3d`. Default: ``False``

    Shape:
        - Input: :math:`(N, C, T_{in}, H_{in}, W_{in})` or :math:`(C, T_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, T_{out}, H_{out}, W_{out})` or :math:`(C, T_{out}, H_{out}, W_{out})`, where
          :math:`(T_{out}, H_{out}, W_{out})=\text{output\_size}` or
          :math:`(T_{out}, H_{out}, W_{out})=\text{output\_ratio} \times (T_{in}, H_{in}, W_{in})`

    Examples:
        >>> # pool of cubic window of size=3, and target output size 13x12x11
        >>> m = nn.FractionalMaxPool3d(3, output_size=(13, 12, 11))
        >>> # pool of cubic window and target output size being half of input size
        >>> m = nn.FractionalMaxPool3d(3, output_ratio=(0.5, 0.5, 0.5))
        >>> input = torch.randn(20, 16, 50, 32, 16)
        >>> output = m(input)

    .. _Fractional MaxPooling:
        https://arxiv.org/abs/1412.6071
    """

    __constants__ = ["kernel_size", "return_indices", "output_size", "output_ratio"]
    kernel_size: _size_3_t       # 3D池化窗口的大小，可以是一个数字k，或者一个元组`(kt x kh x kw)`
    return_indices: bool         # 是否返回池化操作的索引，默认为False
    output_size: _size_3_t       # 目标输出大小，可以是一个元组`(oT, oH, oW)`或一个数字oH，表示一个立方体图像`oH x oH x oH`
    output_ratio: _ratio_3_t     # 输出大小相对于输入大小的比例，范围在(0, 1)之间的数字或元组

    def __init__(
        self,
        kernel_size: _size_3_t,
        output_size: Optional[_size_3_t] = None,
        output_ratio: Optional[_ratio_3_t] = None,
        return_indices: bool = False,
        _random_samples=None,
        ):
        """
        初始化函数，设置FractionalMaxPool3d的各种参数和选项

        Args:
            kernel_size: 池化窗口的大小，可以是一个数字k或者一个元组`(kt x kh x kw)`
            output_size: 输出的目标大小，可以是一个元组`(oT, oH, oW)`或一个数字oH
            output_ratio: 输出大小相对于输入大小的比例，范围在(0, 1)之间的数字或元组
            return_indices: 如果为True，将返回索引和输出，可用于nn.MaxUnpool3d方法，默认为False
            _random_samples: 内部使用的随机采样参数，默认为None
        """
        super(FractionalMaxPool3d, self).__init__()
        self.kernel_size = kernel_size
        self.output_size = output_size
        self.output_ratio = output_ratio
        self.return_indices = return_indices
    # 初始化函数，设置 FractionalMaxPool3d 的参数
    def __init__(
        self,
        kernel_size: Tuple[int, int, int],
        output_size: Optional[Tuple[int, int, int]] = None,
        output_ratio: Optional[Tuple[float, float, float]] = None,
        return_indices: bool = False,
        _random_samples: Optional[Tensor] = None
    ) -> None:
        # 调用父类初始化函数
        super().__init__()
        # 将 kernel_size 转换为三元组形式
        self.kernel_size = _triple(kernel_size)
        # 是否返回 pooling 的 indices
        self.return_indices = return_indices
        # 注册 _random_samples 为 buffer
        self.register_buffer("_random_samples", _random_samples)
        # 如果指定了 output_size，则转换为三元组形式；否则为 None
        self.output_size = _triple(output_size) if output_size is not None else None
        # 如果指定了 output_ratio，则转换为三元组形式；否则为 None
        self.output_ratio = _triple(output_ratio) if output_ratio is not None else None
        # 如果既没有指定 output_size 也没有指定 output_ratio，则抛出异常
        if output_size is None and output_ratio is None:
            raise ValueError(
                "FractionalMaxPool3d requires specifying either "
                "an output size, or a pooling ratio"
            )
        # 如果同时指定了 output_size 和 output_ratio，则抛出异常
        if output_size is not None and output_ratio is not None:
            raise ValueError(
                "only one of output_size and output_ratio may be specified"
            )
        # 如果指定了 output_ratio，则检查其值范围是否在 (0, 1) 之间
        if self.output_ratio is not None:
            if not (
                0 < self.output_ratio[0] < 1
                and 0 < self.output_ratio[1] < 1
                and 0 < self.output_ratio[2] < 1
            ):
                raise ValueError(
                    f"output_ratio must be between 0 and 1 (got {output_ratio})"
                )

    # 前向传播函数，调用 F 模块的 fractional_max_pool3d 函数进行计算
    def forward(self, input: Tensor):
        return F.fractional_max_pool3d(
            input,
            self.kernel_size,
            self.output_size,
            self.output_ratio,
            self.return_indices,
            _random_samples=self._random_samples,
        )
class _LPPoolNd(Module):
    # 类常量，指定了该类实例化时必需的属性
    __constants__ = ["norm_type", "kernel_size", "stride", "ceil_mode"]

    # 类属性定义
    norm_type: float
    ceil_mode: bool

    def __init__(
        self,
        norm_type: float,
        kernel_size: _size_any_t,
        stride: Optional[_size_any_t] = None,
        ceil_mode: bool = False,
    ) -> None:
        # 调用父类的构造函数初始化基础类Module
        super().__init__()
        # 初始化对象的属性
        self.norm_type = norm_type
        self.kernel_size = kernel_size
        self.stride = stride
        self.ceil_mode = ceil_mode

    def extra_repr(self) -> str:
        # 返回对象的字符串表示，描述对象的各个属性
        return (
            "norm_type={norm_type}, kernel_size={kernel_size}, stride={stride}, "
            "ceil_mode={ceil_mode}".format(**self.__dict__)
        )


class LPPool1d(_LPPoolNd):
    r"""Applies a 1D power-average pooling over an input signal composed of several input planes.

    On each window, the function computed is:

    .. math::
        f(X) = \sqrt[p]{\sum_{x \in X} x^{p}}

    - At p = :math:`\infty`, one gets Max Pooling
    - At p = 1, one gets Sum Pooling (which is proportional to Average Pooling)

    .. note:: If the sum to the power of `p` is zero, the gradient of this function is
              not defined. This implementation will set the gradient to zero in this case.

    Args:
        kernel_size: a single int, the size of the window
        stride: a single int, the stride of the window. Default value is :attr:`kernel_size`
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape

    Shape:
        - Input: :math:`(N, C, L_{in})` or :math:`(C, L_{in})`.
        - Output: :math:`(N, C, L_{out})` or :math:`(C, L_{out})`, where

          .. math::
              L_{out} = \left\lfloor\frac{L_{in} - \text{kernel\_size}}{\text{stride}} + 1\right\rfloor

    Examples::
        >>> # power-2 pool of window of length 3, with stride 2.
        >>> m = nn.LPPool1d(2, 3, stride=2)
        >>> input = torch.randn(20, 16, 50)
        >>> output = m(input)
    """

    # 类属性定义，继承自 _LPPoolNd
    kernel_size: _size_1_t
    stride: _size_1_t

    def forward(self, input: Tensor) -> Tensor:
        # 调用 F.lp_pool1d 函数进行 1D Lp pooling 操作
        return F.lp_pool1d(
            input, float(self.norm_type), self.kernel_size, self.stride, self.ceil_mode
        )


class LPPool2d(_LPPoolNd):
    r"""Applies a 2D power-average pooling over an input signal composed of several input planes.

    On each window, the function computed is:

    .. math::
        f(X) = \sqrt[p]{\sum_{x \in X} x^{p}}

    - At p = :math:`\infty`, one gets Max Pooling
    - At p = 1, one gets Sum Pooling (which is proportional to average pooling)

    The parameters :attr:`kernel_size`, :attr:`stride` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension
    """
        .. note:: If the sum to the power of `p` is zero, the gradient of this function is
                  not defined. This implementation will set the gradient to zero in this case.
    
        Args:
            kernel_size: the size of the window
            stride: the stride of the window. Default value is :attr:`kernel_size`
            ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape
    
        Shape:
            - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
            - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where
    
              .. math::
                  H_{out} = \left\lfloor\frac{H_{in} - \text{kernel\_size}[0]}{\text{stride}[0]} + 1\right\rfloor
    
              .. math::
                  W_{out} = \left\lfloor\frac{W_{in} - \text{kernel\_size}[1]}{\text{stride}[1]} + 1\right\rfloor
    
        Examples::
    
            >>> # power-2 pool of square window of size=3, stride=2
            >>> m = nn.LPPool2d(2, 3, stride=2)
            >>> # pool of non-square window of power 1.2
            >>> m = nn.LPPool2d(1.2, (3, 2), stride=(2, 1))
            >>> input = torch.randn(20, 16, 50, 32)
            >>> output = m(input)
    
        """
    
        kernel_size: _size_2_t  # 定义窗口的大小，类型为 _size_2_t
        stride: _size_2_t  # 定义窗口的步幅，类型为 _size_2_t
    
        def forward(self, input: Tensor) -> Tensor:
            # 调用 Torch 中的 lp_pool2d 函数进行 Lp 池化操作
            return F.lp_pool2d(
                input, float(self.norm_type), self.kernel_size, self.stride, self.ceil_mode
            )
class LPPool3d(_LPPoolNd):
    r"""Applies a 3D power-average pooling over an input signal composed of several input planes.

    On each window, the function computed is:

    .. math::
        f(X) = \sqrt[p]{\sum_{x \in X} x^{p}}

    - At p = :math:`\infty`, one gets Max Pooling
    - At p = 1, one gets Sum Pooling (which is proportional to average pooling)

    The parameters :attr:`kernel_size`, :attr:`stride` can either be:

        - a single ``int`` -- in which case the same value is used for the height, width and depth dimension
        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,
          the second `int` for the height dimension and the third `int` for the width dimension

    .. note:: If the sum to the power of `p` is zero, the gradient of this function is
              not defined. This implementation will set the gradient to zero in this case.

    Args:
        kernel_size: the size of the window
        stride: the stride of the window. Default value is :attr:`kernel_size`
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})` or :math:`(C, D_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` or
          :math:`(C, D_{out}, H_{out}, W_{out})`, where

          .. math::
              D_{out} = \left\lfloor\frac{D_{in} - \text{kernel\_size}[0]}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} - \text{kernel\_size}[1]}{\text{stride}[1]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} - \text{kernel\_size}[2]}{\text{stride}[2]} + 1\right\rfloor

    Examples::

        >>> # power-2 pool of square window of size=3, stride=2
        >>> m = nn.LPPool3d(2, 3, stride=2)
        >>> # pool of non-square window of power 1.2
        >>> m = nn.LPPool3d(1.2, (3, 2, 2), stride=(2, 1, 2))
        >>> input = torch.randn(20, 16, 50, 44, 31)
        >>> output = m(input)

    """

    kernel_size: _size_3_t
    stride: _size_3_t

    # 定义 forward 方法，接受输入张量 input，并返回处理后的张量
    def forward(self, input: Tensor) -> Tensor:
        # 调用 torch.nn.functional 中的 lp_pool3d 函数进行 3D Lp pooling 操作
        return F.lp_pool3d(
            input, float(self.norm_type), self.kernel_size, self.stride, self.ceil_mode
        )


class _AdaptiveMaxPoolNd(Module):
    __constants__ = ["output_size", "return_indices"]
    return_indices: bool

    # 初始化方法，接受输出尺寸 output_size 和 return_indices 参数
    def __init__(
        self, output_size: _size_any_opt_t, return_indices: bool = False
    ) -> None:
        super().__init__()
        # 设置实例的 output_size 属性
        self.output_size = output_size
        # 设置实例的 return_indices 属性
        self.return_indices = return_indices

    # 返回额外的表示信息，描述实例的 output_size 属性
    def extra_repr(self) -> str:
        return f"output_size={self.output_size}"


# FIXME (by @ssnl): Improve adaptive pooling docs: specify what the input and
#   output shapes are, and how the operation computes output.


class AdaptiveMaxPool1d(_AdaptiveMaxPoolNd):
    r"""Applies a 1D adaptive max pooling over an input signal composed of several input planes.

    The output size is :math:`L_{out}`, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size :math:`L_{out}`.
        return_indices: if ``True``, will return the indices along with the outputs.
                        Useful to pass to nn.MaxUnpool1d. Default: ``False``

    Shape:
        - Input: :math:`(N, C, L_{in})` or :math:`(C, L_{in})`.
        - Output: :math:`(N, C, L_{out})` or :math:`(C, L_{out})`, where
          :math:`L_{out}=\text{output\_size}`.

    Examples:
        >>> # target output size of 5
        >>> m = nn.AdaptiveMaxPool1d(5)
        >>> input = torch.randn(1, 64, 8)
        >>> output = m(input)

    """

    output_size: _size_1_t
    # 定义了一个类成员变量 output_size，表示自适应最大池化的目标输出尺寸

    def forward(self, input: Tensor):
        # 前向传播函数，接收一个张量 input 作为输入，应用自适应最大池化操作
        return F.adaptive_max_pool1d(input, self.output_size, self.return_indices)
        # 调用 PyTorch 的 F 模块中的 adaptive_max_pool1d 函数进行自适应最大池化操作，传入输入张量、目标输出尺寸和是否返回索引
class AdaptiveMaxPool2d(_AdaptiveMaxPoolNd):
    r"""Applies a 2D adaptive max pooling over an input signal composed of several input planes.

    The output is of size :math:`H_{out} \times W_{out}`, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size of the image of the form :math:`H_{out} \times W_{out}`.
                     Can be a tuple :math:`(H_{out}, W_{out})` or a single :math:`H_{out}` for a
                     square image :math:`H_{out} \times H_{out}`. :math:`H_{out}` and :math:`W_{out}`
                     can be either a ``int``, or ``None`` which means the size will be the same as that
                     of the input.
        return_indices: if ``True``, will return the indices along with the outputs.
                        Useful to pass to nn.MaxUnpool2d. Default: ``False``

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where
          :math:`(H_{out}, W_{out})=\text{output\_size}`.

    Examples:
        >>> # target output size of 5x7
        >>> m = nn.AdaptiveMaxPool2d((5, 7))
        >>> input = torch.randn(1, 64, 8, 9)
        >>> output = m(input)
        >>> # target output size of 7x7 (square)
        >>> m = nn.AdaptiveMaxPool2d(7)
        >>> input = torch.randn(1, 64, 10, 9)
        >>> output = m(input)
        >>> # target output size of 10x7
        >>> m = nn.AdaptiveMaxPool2d((None, 7))
        >>> input = torch.randn(1, 64, 10, 9)
        >>> output = m(input)

    """

    output_size: _size_2_opt_t  # 目标输出大小的类型，可以是 int 或 tuple，表示输出的高度和宽度

    def forward(self, input: Tensor):
        # 调用 PyTorch 中的 adaptive_max_pool2d 函数，对输入进行 2D 自适应最大池化
        return F.adaptive_max_pool2d(input, self.output_size, self.return_indices)


class AdaptiveMaxPool3d(_AdaptiveMaxPoolNd):
    r"""Applies a 3D adaptive max pooling over an input signal composed of several input planes.

    The output is of size :math:`D_{out} \times H_{out} \times W_{out}`, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size of the image of the form :math:`D_{out} \times H_{out} \times W_{out}`.
                     Can be a tuple :math:`(D_{out}, H_{out}, W_{out})` or a single
                     :math:`D_{out}` for a cube :math:`D_{out} \times D_{out} \times D_{out}`.
                     :math:`D_{out}`, :math:`H_{out}` and :math:`W_{out}` can be either a
                     ``int``, or ``None`` which means the size will be the same as that of the input.

        return_indices: if ``True``, will return the indices along with the outputs.
                        Useful to pass to nn.MaxUnpool3d. Default: ``False``

    """
    """
    该模块定义了一个自适应最大池化的三维版本，允许根据给定的输出尺寸自动调整输入的大小进行池化操作。

    Attributes:
        output_size: 定义输出的目标尺寸，可以是一个整数或一个三元组，用于指定输出的深度、高度和宽度。
                     如果其中某个维度为None，则该维度将根据输入的相应维度进行自动调整。

    Examples:
        >>> # target output size of 5x7x9
        >>> m = nn.AdaptiveMaxPool3d((5, 7, 9))
        >>> input = torch.randn(1, 64, 8, 9, 10)
        >>> output = m(input)
        >>> # target output size of 7x7x7 (cube)
        >>> m = nn.AdaptiveMaxPool3d(7)
        >>> input = torch.randn(1, 64, 10, 9, 8)
        >>> output = m(input)
        >>> # target output size of 7x9x8
        >>> m = nn.AdaptiveMaxPool3d((7, None, None))
        >>> input = torch.randn(1, 64, 10, 9, 8)
        >>> output = m(input)

    """

    output_size: _size_3_opt_t

    def forward(self, input: Tensor):
        # 使用 torch.nn.functional.adaptive_max_pool3d 函数进行自适应最大池化操作，
        # 根据指定的 output_size 和 self.return_indices 参数
        return F.adaptive_max_pool3d(input, self.output_size, self.return_indices)
# 定义一个私有类 _AdaptiveAvgPoolNd，继承自 Module 类
class _AdaptiveAvgPoolNd(Module):
    # 类的常量列表，包含 output_size
    __constants__ = ["output_size"]

    # 初始化方法，接收 output_size 参数并赋值给实例变量 self.output_size
    def __init__(self, output_size: _size_any_opt_t) -> None:
        super().__init__()
        self.output_size = output_size

    # 返回描述对象的字符串，包括 output_size 的信息
    def extra_repr(self) -> str:
        return f"output_size={self.output_size}"


# 定义一维自适应平均池化类 AdaptiveAvgPool1d，继承自 _AdaptiveAvgPoolNd 类
class AdaptiveAvgPool1d(_AdaptiveAvgPoolNd):
    # 文档字符串，描述了一维自适应平均池化的作用、输入输出形状、示例用法等
    r"""Applies a 1D adaptive average pooling over an input signal composed of several input planes.

    The output size is :math:`L_{out}`, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size :math:`L_{out}`.

    Shape:
        - Input: :math:`(N, C, L_{in})` or :math:`(C, L_{in})`.
        - Output: :math:`(N, C, L_{out})` or :math:`(C, L_{out})`, where
          :math:`L_{out}=\text{output\_size}`.

    Examples:
        >>> # target output size of 5
        >>> m = nn.AdaptiveAvgPool1d(5)
        >>> input = torch.randn(1, 64, 8)
        >>> output = m(input)

    """

    # output_size 属性的类型注解
    output_size: _size_1_t

    # 前向传播方法，接收输入张量 input，返回经过自适应平均池化后的张量
    def forward(self, input: Tensor) -> Tensor:
        return F.adaptive_avg_pool1d(input, self.output_size)


# 定义二维自适应平均池化类 AdaptiveAvgPool2d，继承自 _AdaptiveAvgPoolNd 类
class AdaptiveAvgPool2d(_AdaptiveAvgPoolNd):
    # 文档字符串，描述了二维自适应平均池化的作用、输入输出形状、示例用法等
    r"""Applies a 2D adaptive average pooling over an input signal composed of several input planes.

    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H.
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, S_{0}, S_{1})` or :math:`(C, S_{0}, S_{1})`, where
          :math:`S=\text{output\_size}`.

    Examples:
        >>> # target output size of 5x7
        >>> m = nn.AdaptiveAvgPool2d((5, 7))
        >>> input = torch.randn(1, 64, 8, 9)
        >>> output = m(input)
        >>> # target output size of 7x7 (square)
        >>> m = nn.AdaptiveAvgPool2d(7)
        >>> input = torch.randn(1, 64, 10, 9)
        >>> output = m(input)
        >>> # target output size of 10x7
        >>> m = nn.AdaptiveAvgPool2d((None, 7))
        >>> input = torch.randn(1, 64, 10, 9)
        >>> output = m(input)

    """

    # output_size 属性的类型注解
    output_size: _size_2_opt_t

    # 前向传播方法，接收输入张量 input，返回经过自适应平均池化后的张量
    def forward(self, input: Tensor) -> Tensor:
        return F.adaptive_avg_pool2d(input, self.output_size)


# 定义三维自适应平均池化类 AdaptiveAvgPool3d，继承自 _AdaptiveAvgPoolNd 类
class AdaptiveAvgPool3d(_AdaptiveAvgPoolNd):
    # 文档字符串，描述了三维自适应平均池化的作用、输入输出形状
    r"""Applies a 3D adaptive average pooling over an input signal composed of several input planes.

    The output is of size D x H x W, for any input size.
    The number of output features is equal to the number of input planes.
    """
    Args:
        output_size: the target output size of the form D x H x W.
                     Can be a tuple (D, H, W) or a single number D for a cube D x D x D.
                     D, H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})` or :math:`(C, D_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, S_{0}, S_{1}, S_{2})` or :math:`(C, S_{0}, S_{1}, S_{2})`,
          where :math:`S=\text{output\_size}`.

    Examples:
        >>> # target output size of 5x7x9
        >>> m = nn.AdaptiveAvgPool3d((5, 7, 9))
        >>> input = torch.randn(1, 64, 8, 9, 10)
        >>> output = m(input)
        >>> # target output size of 7x7x7 (cube)
        >>> m = nn.AdaptiveAvgPool3d(7)
        >>> input = torch.randn(1, 64, 10, 9, 8)
        >>> output = m(input)
        >>> # target output size of 7x9x8
        >>> m = nn.AdaptiveAvgPool3d((7, None, None))
        >>> input = torch.randn(1, 64, 10, 9, 8)
        >>> output = m(input)

    """

    output_size: _size_3_opt_t

    # 定义前向传播函数，接受输入张量 input 和输出尺寸 output_size，返回处理后的张量
    def forward(self, input: Tensor) -> Tensor:
        # 使用 PyTorch 的 adaptive_avg_pool3d 函数对输入张量进行自适应平均池化，输出尺寸由 output_size 指定
        return F.adaptive_avg_pool3d(input, self.output_size)
```