# `.\pytorch\torch\ao\nn\quantized\functional.py`

```
# mypy: allow-untyped-defs
r""" Functional interface (quantized)."""
# 引入必要的库和模块
from typing import List, Optional
import warnings

import torch
from torch import Tensor
from torch.nn.modules.utils import _pair, _triple
from torch.jit.annotations import BroadcastingList2

# 从自定义的模块中引入一个函数
from .modules.utils import _pair_from_first

# 虽然部分函数和文档字符串与torch.nn中的镜像，但我们为了未来的更改保留在这里。

# 定义公开的函数列表，这些函数可能在其他地方被调用
__all__ = [
    "avg_pool2d",
    "avg_pool3d",
    "adaptive_avg_pool2d",
    "adaptive_avg_pool3d",
    "conv1d",
    "conv2d",
    "conv3d",
    "interpolate",
    "linear",
    "max_pool1d",
    "max_pool2d",
    "celu",
    "leaky_relu",
    "hardtanh",
    "hardswish",
    "threshold",
    "elu",
    "hardsigmoid",
    "clamp",
    "upsample",
    "upsample_bilinear",
    "upsample_nearest",
]

# 定义二维平均池化函数，用于量化输入数据
def avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False,
               count_include_pad=True, divisor_override=None):
    r"""
    Applies 2D average-pooling operation in :math:`kH \times kW` regions by step size
    :math:`sH \times sW` steps. The number of output features is equal to the number of
    input planes.

    .. note:: The input quantization parameters propagate to the output.

    See :class:`~torch.ao.nn.quantized.AvgPool2d` for details and output shape.

    Args:
        input: quantized input tensor :math:`(\text{minibatch} , \text{in\_channels} , iH , iW)`
        kernel_size: size of the pooling region. Can be a single number or a
          tuple `(kH, kW)`
        stride: stride of the pooling operation. Can be a single number or a
          tuple `(sH, sW)`. Default: :attr:`kernel_size`
        padding: implicit zero paddings on both sides of the input. Can be a
          single number or a tuple `(padH, padW)`. Default: 0
        ceil_mode: when True, will use `ceil` instead of `floor` in the formula
            to compute the output shape. Default: ``False``
        count_include_pad: when True, will include the zero-padding in the
            averaging calculation. Default: ``True``
        divisor_override: if specified, it will be used as divisor, otherwise
             size of the pooling region will be used. Default: None
    """
    # 如果输入数据未被量化，则抛出错误
    if not input.is_quantized:
        raise ValueError("Input to 'quantized.avg_pool2d' must be quantized!")
    # 调用PyTorch的函数，执行二维平均池化操作
    return torch.nn.functional.avg_pool2d(input, kernel_size, stride, padding,
                                          ceil_mode, count_include_pad,
                                          divisor_override)

# 定义三维平均池化函数，用于量化输入数据
def avg_pool3d(input, kernel_size, stride=None, padding=0, ceil_mode=False,
               count_include_pad=True, divisor_override=None):
    r"""
    Applies 3D average-pooling operation in :math:`kD \ times kH \times kW` regions by step size
    :math:`sD \times sH \times sW` steps. The number of output features is equal to the number of
    input planes.

    .. note:: The input quantization parameters propagate to the output.

    ```
    # 检查输入张量是否已量化，如果没有，抛出值错误异常
    if not input.is_quantized:
        raise ValueError("Input to 'quantized.avg_pool3d' must be quantized!")
    # 调用 PyTorch 中的函数式接口进行三维平均池化操作，返回池化后的结果张量
    return torch.nn.functional.avg_pool3d(input, kernel_size, stride, padding,
                                          ceil_mode, count_include_pad,
                                          divisor_override)
# 对输入的 quantized 1D 数据进行一维量化卷积操作
def conv1d(input, weight, bias,
           stride=1, padding=0, dilation=1, groups=1,
           padding_mode='zeros',
           scale=1.0, zero_point=0,
           dtype=torch.quint8):
    r"""
    Applies a 1D convolution over a quantized 1D input composed of several input
    planes.

    See :class:`~torch.ao.nn.quantized.Conv1d` for details and output shape.

    Args:
        input: quantized input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iW)`
        weight: quantized filters of shape :math:`(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , iW)`
        bias: **non-quantized** bias tensor of shape :math:`(\text{out\_channels})`. The tensor type must be `torch.float`.
        stride: the stride of the convolving kernel. Can be a single number or a
          tuple `(sW,)`. Default: 1
        padding: implicit paddings on both sides of the input. Can be a
          single number or a tuple `(padW,)`. Default: 0
        dilation: the spacing between kernel elements. Can be a single number or
          a tuple `(dW,)`. Default: 1
        groups: split input into groups, :math:`\text{in\_channels}` should be divisible by the
          number of groups. Default: 1
        padding_mode: the padding mode to use. Only "zeros" is supported for quantized convolution at the moment. Default: "zeros"
        scale: quantization scale for the output. Default: 1.0
        zero_point: quantization zero_point for the output. Default: 0
        dtype: quantization data type to use. Default: ``torch.quint8``
    """
    # 检查输入是否已经进行了量化
    if not input.is_quantized:
        raise ValueError("Input to 'quantized.functional.conv1d' must be quantized!")
    # 调用 PyTorch 的函数进行一维量化卷积操作
    return torch.nn.functional.conv1d(input, weight, bias, stride, padding, dilation, groups)
    """  # noqa: E501
    # 检查填充模式是否为零填充，否则抛出错误
    if padding_mode != 'zeros':
        raise NotImplementedError("Only zero-padding is supported!")
    # 检查输入张量的数据类型是否为 torch.quint8，否则抛出错误
    if input.dtype != torch.quint8:
        raise NotImplementedError("Only torch.quint8 is supported for activation tensor!")
    # 检查权重张量的数据类型是否为 torch.qint8，否则抛出错误
    if weight.dtype != torch.qint8:
        raise NotImplementedError("Only torch.qint8 is supported for weight tensor!")
    # 检查输入张量的维度是否为 3，即形状必须为 `(N, C, L)`
    if input.ndim != 3:
        raise ValueError("Input shape must be `(N, C, L)`!")
    # 从步长参数中提取成对的值
    stride = _pair_from_first(stride)
    # 从填充参数中提取成对的值
    padding = _pair_from_first(padding)
    # 从扩展参数中提取成对的值
    dilation = _pair_from_first(dilation)

    # 使用 torch.ops.quantized.conv1d_prepack 打包参数
    packed_params = torch.ops.quantized.conv1d_prepack(
        weight, bias, stride, padding, dilation, groups)
    # 调用 quantized conv1d 操作执行量化卷积运算
    return torch.ops.quantized.conv1d(input, packed_params, scale, zero_point)
    """
# 定义一个函数，用于对量化的二维输入进行卷积操作，使用给定的量化卷积参数。
def conv2d(input, weight, bias,
           stride=1, padding=0, dilation=1, groups=1,
           padding_mode='zeros',
           scale=1.0, zero_point=0,
           dtype=torch.quint8):
    r"""
    Applies a 2D convolution over a quantized 2D input composed of several input
    planes.

    See :class:`~torch.ao.nn.quantized.Conv2d` for details and output shape.

    Args:
        input: quantized input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iH , iW)`
        weight: quantized filters of shape :math:`(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kH , kW)`
        bias: **non-quantized** bias tensor of shape :math:`(\text{out\_channels})`. The tensor type must be `torch.float`.
        stride: the stride of the convolving kernel. Can be a single number or a
          tuple `(sH, sW)`. Default: 1
        padding: implicit paddings on both sides of the input. Can be a
          single number or a tuple `(padH, padW)`. Default: 0
        dilation: the spacing between kernel elements. Can be a single number or
          a tuple `(dH, dW)`. Default: 1
        groups: split input into groups, :math:`\text{in\_channels}` should be divisible by the
          number of groups. Default: 1
        padding_mode: the padding mode to use. Only "zeros" is supported for quantized convolution at the moment. Default: "zeros"
        scale: quantization scale for the output. Default: 1.0
        zero_point: quantization zero_point for the output. Default: 0
        dtype: quantization data type to use. Default: ``torch.quint8``

    Examples::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_QENGINE)
        >>> from torch.ao.nn.quantized import functional as qF
        >>> filters = torch.randn(8, 4, 3, 3, dtype=torch.float)
        >>> inputs = torch.randn(1, 4, 5, 5, dtype=torch.float)
        >>> bias = torch.randn(8, dtype=torch.float)
        >>>
        >>> scale, zero_point = 1.0, 0
        >>> dtype_inputs = torch.quint8
        >>> dtype_filters = torch.qint8
        >>>
        >>> q_filters = torch.quantize_per_tensor(filters, scale, zero_point, dtype_filters)
        >>> q_inputs = torch.quantize_per_tensor(inputs, scale, zero_point, dtype_inputs)
        >>> qF.conv2d(q_inputs, q_filters, bias, padding=1, scale=scale, zero_point=zero_point)
    """  # noqa: E501
    # 如果指定的填充模式不是 'zeros'，则抛出错误，目前只支持零填充
    if padding_mode != 'zeros':
        raise NotImplementedError("Only zero-padding is supported!")
    # 如果输入的数据类型不是 torch.quint8，抛出错误，目前只支持激活张量为 torch.quint8 类型
    if input.dtype != torch.quint8:
        raise NotImplementedError("Only torch.quint8 is supported for activation tensor!")
    # 如果权重的数据类型不是 torch.qint8，抛出错误，目前只支持权重张量为 torch.qint8 类型
    if weight.dtype != torch.qint8:
        raise NotImplementedError("Only torch.qint8 is supported for weight tensor!")
    # 如果输入的维度不是 4，抛出值错误，输入形状必须为 `(N, C, H, W)`
    if input.ndim != 4:
        raise ValueError("Input shape must be `(N, C, H, W)`!")
    # 将步长、填充、扩张参数转换为元组形式
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    # 使用 torch.ops.quantized.conv2d_prepack 函数打包参数，准备进行量化卷积操作
    packed_params = torch.ops.quantized.conv2d_prepack(
        weight, bias, stride, padding, dilation, groups)
    # 调用 PyTorch 提供的量化操作（quantized conv2d），执行量化卷积操作。
    return torch.ops.quantized.conv2d(input, packed_params, scale, zero_point)
# 定义一个函数，执行三维量化卷积操作，作用于由多个输入平面组成的三维量化输入。
def conv3d(input, weight, bias, stride=1, padding=0, dilation=1, groups=1,
           padding_mode='zeros', scale=1.0, zero_point=0, dtype=torch.quint8):
    r"""
    Applies a 3D convolution over a quantized 3D input composed of several input
    planes.

    See :class:`~torch.ao.nn.quantized.Conv3d` for details and output shape.

    Args:
        input: quantized input tensor of shape
          :math:`(\text{minibatch} , \text{in\_channels} , iD , iH , iW)`
        weight: quantized filters of shape
          :math:`(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kD , kH , kW)`
        bias: **non-quantized** bias tensor of shape
          :math:`(\text{out\_channels})`. The tensor type must be `torch.float`.
        stride: the stride of the convolving kernel. Can be a single number or a
          tuple `(sD, sH, sW)`. Default: 1
        padding: implicit paddings on both sides of the input. Can be a
          single number or a tuple `(padD, padH, padW)`. Default: 0
        dilation: the spacing between kernel elements. Can be a single number or
          a tuple `(dD, dH, dW)`. Default: 1
        groups: split input into groups, :math:`\text{in\_channels}` should be
          divisible by the number of groups. Default: 1
        padding_mode: the padding mode to use. Only "zeros" is supported for
          quantized convolution at the moment. Default: "zeros"
        scale: quantization scale for the output. Default: 1.0
        zero_point: quantization zero_point for the output. Default: 0
        dtype: quantization data type to use. Default: ``torch.quint8``

    Examples::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_QENGINE)
        >>> from torch.ao.nn.quantized import functional as qF
        >>> filters = torch.randn(8, 4, 3, 3, 3, dtype=torch.float)
        >>> inputs = torch.randn(1, 4, 5, 5, 5, dtype=torch.float)
        >>> bias = torch.randn(8, dtype=torch.float)
        >>>
        >>> scale, zero_point = 1.0, 0
        >>> dtype_inputs = torch.quint8
        >>> dtype_filters = torch.qint8
        >>>
        >>> q_filters = torch.quantize_per_tensor(filters, scale, zero_point, dtype_filters)
        >>> q_inputs = torch.quantize_per_tensor(inputs, scale, zero_point, dtype_inputs)
        >>> qF.conv3d(q_inputs, q_filters, bias, padding=1, scale=scale, zero_point=zero_point)
    """  # noqa: E501

    # 检查是否选择了除 "zeros" 以外的填充模式，若是则引发错误
    if padding_mode != 'zeros':
        raise NotImplementedError("Only zero-padding is supported!")

    # 检查输入张量是否为 torch.quint8 类型，若不是则引发错误
    if input.dtype != torch.quint8:
        raise NotImplementedError("Only torch.quint8 is supported for activation tensor!")

    # 检查权重张量是否为 torch.qint8 类型，若不是则引发错误
    if weight.dtype != torch.qint8:
        raise NotImplementedError("Only torch.qint8 is supported for weight tensor!")

    # 检查输入张量的维度是否为 5，即 (N, C, D, H, W)，若不是则引发错误
    if input.ndim != 5:
        raise ValueError("Input shape must be `(N, C, D, H, W)`!")

    # 将步幅、填充和膨胀参数转换为三元组形式
    stride = _triple(stride)
    padding = _triple(padding)
    dilation = _triple(dilation)
    # 使用 PyTorch 提供的 quantized 操作，对 3D 卷积的权重、偏置、步长、填充、扩张和分组进行预打包
    packed_params = torch.ops.quantized.conv3d_prepack(
        weight, bias, stride, padding, dilation, groups)
    # 调用 quantized 操作执行量化的 3D 卷积计算，使用预打包的参数、量化参数（scale 和 zero_point）
    return torch.ops.quantized.conv3d(input, packed_params, scale, zero_point)
# 对输入进行插值操作，调整大小到给定的尺寸或者按照给定的尺度因子进行缩放
def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
    r"""Down/up samples the input to either the given :attr:`size` or the given
    :attr:`scale_factor`

    See :func:`torch.nn.functional.interpolate` for implementation details.

    The input dimensions are interpreted in the form:
    `mini-batch x channels x [optional depth] x [optional height] x width`.

    .. note:: The input quantization parameters propagate to the output.

    .. note:: Only 2D/3D input is supported for quantized inputs

    .. note:: Only the following modes are supported for the quantized inputs:

        - `bilinear`
        - `nearest`

    Args:
        input (Tensor): the input tensor
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]):
            output spatial size.
        scale_factor (float or Tuple[float]): multiplier for spatial size. Has to match input size if it is a tuple.
        mode (str): algorithm used for upsampling:
            ``'nearest'`` | ``'bilinear'``
        align_corners (bool, optional): Geometrically, we consider the pixels of the
            input and output as squares rather than points.
            If set to ``True``, the input and output tensors are aligned by the
            center points of their corner pixels, preserving the values at the corner pixels.
            If set to ``False``, the input and output tensors are aligned by the corner
            points of their corner pixels, and the interpolation uses edge value padding
            for out-of-boundary values, making this operation *independent* of input size
            when :attr:`scale_factor` is kept the same. This only has an effect when :attr:`mode`
            is ``'bilinear'``.
            Default: ``False``
    """
    # 检查输入是否为量化的数据类型，如果不是，则抛出值错误异常
    if not input.is_quantized:
        raise ValueError("Input to 'quantized.interpolate' must be quantized!")
    # 调用 PyTorch 的函数式插值方法，返回插值后的结果
    return torch.nn.functional.interpolate(input, size, scale_factor, mode,
                                           align_corners)


# 对输入的量化数据应用线性变换操作：y = xA^T + b
def linear(
    input: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
    scale: Optional[float] = None, zero_point: Optional[int] = None
) -> Tensor:
    r"""
    Applies a linear transformation to the incoming quantized data:
    :math:`y = xA^T + b`.
    See :class:`~torch.ao.nn.quantized.Linear`

    .. note::

      Current implementation packs weights on every call, which has penalty on performance.
      If you want to avoid the overhead, use :class:`~torch.ao.nn.quantized.Linear`.

    Args:
      input (Tensor): Quantized input of type `torch.quint8`
      weight (Tensor): Quantized weight of type `torch.qint8`
      bias (Tensor): None or fp32 bias of type `torch.float`
      scale (double): output scale. If None, derived from the input scale
      zero_point (long): output zero point. If None, derived from the input zero_point
    """
    """
    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """
    # 如果未提供缩放因子，则从输入张量中获取缩放因子
    if scale is None:
        scale = input.q_scale()
    # 如果未提供零点，则从输入张量中获取零点
    if zero_point is None:
        zero_point = input.q_zero_point()
    # 将权重和偏置预打包成内部结构
    _packed_params = torch.ops.quantized.linear_prepack(weight, bias)
    # 调用量化线性运算，传入输入张量、预打包的参数、缩放因子和零点
    return torch.ops.quantized.linear(input, _packed_params, scale, zero_point)
# 定义一个函数，对输入的一维量化信号应用最大池化操作。
def max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1,
               ceil_mode=False, return_indices=False):
    r"""Applies a 1D max pooling over a quantized input signal composed of
    several quantized input planes.

    .. note:: The input quantization parameters are propagated to the output.

    See :class:`~torch.ao.nn.quantized.MaxPool1d` for details.
    """

    # 如果要返回池化操作的索引，目前不支持该功能，抛出未实现的错误。
    if return_indices:
        raise NotImplementedError("return_indices is not yet implemented!")

    # 如果未指定步幅（stride），则设定为空列表。
    if stride is None:
        stride = torch.jit.annotate(List[int], [])

    # 调用PyTorch中的函数式接口实现一维最大池化操作，并返回结果。
    return torch.nn.functional.max_pool1d(input, kernel_size, stride, padding,
                                          dilation, ceil_mode=ceil_mode, return_indices=return_indices)


# 定义一个函数，对输入的二维量化信号应用最大池化操作。
def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1,
               ceil_mode=False, return_indices=False):
    r"""Applies a 2D max pooling over a quantized input signal composed of
    several quantized input planes.

    .. note:: The input quantization parameters are propagated to the output.

    See :class:`~torch.ao.nn.quantized.MaxPool2d` for details.
    """

    # 如果要返回池化操作的索引，目前不支持该功能，抛出未实现的错误。
    if return_indices:
        raise NotImplementedError("return_indices is not yet implemented!")

    # 如果未指定步幅（stride），则设定为空列表。
    if stride is None:
        stride = torch.jit.annotate(List[int], [])

    # 调用PyTorch中的函数式接口实现二维最大池化操作，并返回结果。
    return torch.nn.functional.max_pool2d(input, kernel_size, stride, padding,
                                          dilation, ceil_mode=ceil_mode, return_indices=return_indices)


# 定义一个函数，对量化输入应用CELu（Continuous Exponential Linear Units）函数的元素级操作。
def celu(input: Tensor, scale: float, zero_point: int, alpha: float = 1.) -> Tensor:
    r"""celu(input, scale, zero_point, alpha=1.) -> Tensor

    Applies the quantized CELU function element-wise.

    .. math::
        \text{CELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x / \alpha) - 1))

    Args:
        input: quantized input
        alpha: the :math:`\alpha` value for the CELU formulation. Default: 1.0
    """

    # 如果输入不是量化的，则抛出值错误。
    if not input.is_quantized:
        raise ValueError("Input to 'quantized.celu' must be quantized!")

    # 调用PyTorch运算符，对输入应用量化CELu函数，并返回结果张量。
    return torch.ops.quantized.celu(input, scale, zero_point, alpha)


# 定义一个函数，对输入的量化信号应用Leaky ReLU函数的元素级操作。
def leaky_relu(input: Tensor, negative_slope: float = 0.01, inplace: bool = False,
               scale: Optional[float] = None, zero_point: Optional[int] = None):
    r"""
    Quantized version of the LeakyReLU function.

    leaky_relu(input, negative_slope=0.01, inplace=False, scale, zero_point) -> Tensor

    Applies element-wise,
    :math:`\text{LeakyReLU}(x) = \max(0, x) + \text{negative_slope} * \min(0, x)`

    Args:
        input: Quantized input
        negative_slope: The slope of the negative input
        inplace: Inplace modification of the input tensor
        scale, zero_point: Scale and zero point of the output tensor.

    See :class:`~torch.nn.LeakyReLU` for more details.
    """
    # 如果同时提供了 scale 和 zero_point 参数，则确保不使用 inplace 操作，因为无法在原地进行重新缩放操作
    if scale is not None and zero_point is not None:
        assert not inplace, "Cannot rescale with `inplace`"
        # 使用 torch._empty_affine_quantized 创建一个量化张量，根据指定的 scale 和 zero_point 进行初始化
        output = torch._empty_affine_quantized(
            input.shape, scale=scale, zero_point=int(zero_point), dtype=input.dtype)
        # 对输入张量应用 leaky ReLU 激活函数，将结果写入到预先创建的 output 张量中
        torch._C._nn.leaky_relu(input, negative_slope, out=output)
        return output
    
    # 如果 inplace 参数为 True，则在原地执行 leaky ReLU 操作，结果覆盖原始输入张量
    if inplace:
        result = torch._C._nn.leaky_relu_(input, negative_slope)
    else:
        # 如果 inplace 参数为 False，则创建一个新的张量，并应用 leaky ReLU 操作
        result = torch._C._nn.leaky_relu(input, negative_slope)
    
    # 返回经过 leaky ReLU 处理后的结果张量
    return result
# 定义了一个函数 `hardtanh`，实现了量化版本的硬切线函数
def hardtanh(input: Tensor, min_val: float = -1., max_val: float = 1., inplace: bool = False) -> Tensor:
    r"""This is the quantized version of :func:`~torch.nn.functional.hardtanh`.
    """
    # 如果输入张量未量化，抛出数值错误
    if not input.is_quantized:
        raise ValueError("Input to 'quantized.hardtanh' must be quantized!")
    # 如果 inplace 为真，则调用就地更新的硬切线函数
    if inplace:
        return torch._C._nn.hardtanh_(input, min_val, max_val)
    # 否则调用标准的硬切线函数
    return torch._C._nn.hardtanh(input, min_val, max_val)

# 定义了一个函数 `hardswish`，实现了量化版本的硬门控整流线性函数
def hardswish(input: Tensor, scale: float, zero_point: int) -> Tensor:
    r"""This is the quantized version of :func:`~torch.nn.functional.hardswish`.

    Args:
        input: quantized input
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
    """
    # 如果输入张量未量化，抛出数值错误
    if not input.is_quantized:
        raise ValueError("Input to 'quantized.hardswish' must be quantized!")
    # 调用量化版本的硬门控整流线性函数
    return torch._ops.ops.quantized.hardswish(input, scale, zero_point)

# 定义了一个函数 `threshold`，实现了元素级的量化阈值函数
def threshold(input: Tensor, threshold: float, value: float) -> Tensor:
    r"""Applies the quantized version of the threshold function element-wise:

    .. math::
        x = \begin{cases}
                x & \text{if~} x > \text{threshold} \\
                \text{value} & \text{otherwise}
            \end{cases}

    See :class:`~torch.nn.Threshold` for more details.
    """
    # 如果输入张量未量化，抛出数值错误
    if not input.is_quantized:
        raise ValueError("Input to 'quantized.threshold' must be quantized!")
    # 如果未指定阈值，抛出数值错误
    if threshold is None:
        raise ValueError("Input to 'threshold' must be specified!")
    # 如果未指定值，抛出数值错误
    if value is None:
        raise ValueError("Input to 'value' must be specified!")
    # 调用量化版本的阈值函数
    return torch._ops.ops.quantized.threshold(input, threshold, value)

# 定义了一个函数 `elu`，实现了量化版本的指数线性单元函数
def elu(input: Tensor, scale: float, zero_point: int, alpha: float = 1.) -> Tensor:
    r"""This is the quantized version of :func:`~torch.nn.functional.elu`.

    Args:
        input: quantized input
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
        alpha: the alpha constant
    """
    # 如果输入张量未量化，抛出数值错误
    if not input.is_quantized:
        raise ValueError("Input to 'quantized.elu' must be quantized!")
    # 调用量化版本的指数线性单元函数
    return torch.ops.quantized.elu(input, scale, zero_point, alpha)

# 定义了一个函数 `hardsigmoid`，实现了量化版本的硬 sigmoid 函数
def hardsigmoid(input: Tensor, inplace: bool = False) -> Tensor:
    r"""This is the quantized version of :func:`~torch.nn.functional.hardsigmoid`.
    """
    # 如果输入张量未量化，抛出数值错误
    if not input.is_quantized:
        raise ValueError("Input to 'quantized.hardsigmoid' must be quantized!")
    # 如果 inplace 为真，则调用就地更新的硬 sigmoid 函数
    if inplace:
        return torch._C._nn.hardsigmoid_(input)  # type: ignore[attr-defined]
    # 否则调用标准的硬 sigmoid 函数
    return torch._C._nn.hardsigmoid(input)

# 定义了一个函数 `clamp`，实现了元素级的量化 clamp 函数
def clamp(input: Tensor, min_: float, max_: float) -> Tensor:
    r"""float(input, min_, max_) -> Tensor

    Applies the clamp function element-wise.
    See :class:`~torch.ao.nn.quantized.clamp` for more details.

    Args:
        input: quantized input
        min_: minimum value for clamping
        max_: maximum value for clamping
    """
    # 检查输入张量是否已经量化，如果没有则抛出值错误异常
    if not input.is_quantized:
        raise ValueError("Input to 'quantized.clamp' must be quantized!")
    # 对输入张量进行截断操作，限制数值在[min_, max_]之间
    return torch.clamp(input, min_, max_)
# 引入警告模块，用于发出函数已弃用的警告信息
import warnings

# 定义一个函数，用于对输入进行上采样到指定尺寸或按指定比例
def upsample(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
    r"""Upsamples the input to either the given :attr:`size` or the given
    :attr:`scale_factor`

    .. warning::
        This function is deprecated in favor of
        :func:`torch.ao.nn.quantized.functional.interpolate`.
        This is equivalent with ``nn.quantized.functional.interpolate(...)``.

    See :func:`torch.nn.functional.interpolate` for implementation details.

    The input dimensions are interpreted in the form:
    `mini-batch x channels x [optional depth] x [optional height] x width`.

    .. note:: The input quantization parameters propagate to the output.

    .. note:: Only 2D input is supported for quantized inputs

    .. note:: Only the following modes are supported for the quantized inputs:

        - `bilinear`
        - `nearest`

    Args:
        input (Tensor): quantized input tensor
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]):
            output spatial size.
        scale_factor (float or Tuple[float]): multiplier for spatial size. Has to be an integer.
        mode (str): algorithm used for upsampling:
            ``'nearest'`` | ``'bilinear'``
        align_corners (bool, optional): Geometrically, we consider the pixels of the
            input and output as squares rather than points.
            If set to ``True``, the input and output tensors are aligned by the
            center points of their corner pixels, preserving the values at the corner pixels.
            If set to ``False``, the input and output tensors are aligned by the corner
            points of their corner pixels, and the interpolation uses edge value padding
            for out-of-boundary values, making this operation *independent* of input size
            when :attr:`scale_factor` is kept the same. This only has an effect when :attr:`mode`
            is ``'bilinear'``.
            Default: ``False``

    .. warning::
        With ``align_corners = True``, the linearly interpolating modes
        (`bilinear`) don't proportionally align the
        output and input pixels, and thus the output values can depend on the
        input size. This was the default behavior for these modes up to version
        0.3.1. Since then, the default behavior is ``align_corners = False``.
        See :class:`~torch.nn.Upsample` for concrete examples on how this
        affects the outputs.
    """
    # 发出警告，提示函数已弃用，建议使用新的函数 interpolate
    warnings.warn("nn.quantized.functional.upsample is deprecated. Use nn.quantized.functional.interpolate instead.")
    # 调用 interpolate 函数进行上采样操作，返回结果
    return interpolate(input, size, scale_factor, mode, align_corners)

# 定义一个函数，用于对输入进行双线性插值上采样
def upsample_bilinear(input, size=None, scale_factor=None):
    r"""Upsamples the input, using bilinear upsampling.
    """
    .. warning::
        This function is deprecated in favor of
        :func:`torch.ao.nn.quantized.functional.interpolate`.
        This is equivalent with
        ``nn.quantized.functional.interpolate(..., mode='bilinear', align_corners=True)``.

    .. note:: The input quantization parameters propagate to the output.

    .. note:: Only 2D inputs are supported

    Args:
        input (Tensor): quantized input
        size (int or Tuple[int, int]): output spatial size.
        scale_factor (int or Tuple[int, int]): multiplier for spatial size
    """
    # 发出警告，提示函数已废弃，推荐使用 nn.quantized.functional.interpolate 代替
    warnings.warn("nn.quantized.functional.upsample_bilinear is deprecated. Use nn.quantized.functional.interpolate instead.")
    # 调用 interpolate 函数，对输入进行双线性插值，保持 align_corners=True
    return interpolate(input, size, scale_factor, mode='bilinear', align_corners=True)
def upsample_nearest(input, size=None, scale_factor=None):
    r"""Upsamples the input, using nearest neighbours' pixel values.

    .. warning::
        This function is deprecated in favor of
        :func:`torch.ao.nn.quantized.functional.interpolate`.
        This is equivalent with ``nn.quantized.functional.interpolate(..., mode='nearest')``.

    .. note:: The input quantization parameters propagate to the output.

    .. note:: Only 2D inputs are supported

    Args:
        input (Tensor): quantized input
        size (int or Tuple[int, int] or Tuple[int, int, int]): output spatial
            size.
        scale_factor (int): multiplier for spatial size. Has to be an integer.
    """
    # DeprecationWarning is ignored by default
    忽略默认情况下的 DeprecationWarning 警告
    warnings.warn("nn.quantized.functional.upsample_nearest is deprecated. Use nn.quantized.functional.interpolate instead.")
    # 调用 interpolate 函数进行最近邻插值，返回结果
    return interpolate(input, size, scale_factor, mode='nearest')
```