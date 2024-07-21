# `.\pytorch\torch\nn\grad.py`

```py
# mypy: allow-untyped-defs
"""Gradient interface."""

import torch
from torch.nn.modules.utils import _pair, _single, _triple


def conv1d_input(
    input_size,
    weight,
    grad_output,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
    r"""Compute the gradient of conv1d with respect to the input of the convolution.

    This is same as the 1D transposed convolution operator under the hood but requires
    the shape of the gradient w.r.t. input to be specified explicitly.

    Args:
        input_size : Shape of the input gradient tensor
        weight: weight tensor (out_channels x in_channels/groups x kW)
        grad_output : output gradient tensor (minibatch x out_channels x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1

    Examples::

        >>> input = torch.randn(1, 1, 3, requires_grad=True)
        >>> weight = torch.randn(1, 1, 1, requires_grad=True)
        >>> output = F.conv1d(input, weight)
        >>> grad_output = torch.randn(output.shape)
        >>> grad_input = torch.autograd.grad(output, input, grad_output)
        >>> F.grad.conv1d_input(input.shape, weight, grad_output)

    """
    # 创建一个与输入梯度大小相同的空张量，用于存储输入梯度
    input = grad_output.new_empty(1).expand(input_size)

    # 调用底层的 ATen 函数计算卷积反向传播的输入梯度
    return torch.ops.aten.convolution_backward(
        grad_output,
        input,
        weight,
        None,
        _single(stride),
        _single(padding),
        _single(dilation),
        False,
        [0],
        groups,
        (True, False, False),
    )[0]


def conv1d_weight(
    input,
    weight_size,
    grad_output,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
    r"""Compute the gradient of conv1d with respect to the weight of the convolution.

    Args:
        input: input tensor of shape (minibatch x in_channels x iW)
        weight_size : Shape of the weight gradient tensor
        grad_output : output gradient tensor (minibatch x out_channels x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
    Examples::

        >>> input = torch.randn(1, 1, 3, requires_grad=True)  # 创建一个形状为 (1, 1, 3) 的随机输入张量，开启梯度跟踪
        >>> weight = torch.randn(1, 1, 1, requires_grad=True)  # 创建一个形状为 (1, 1, 1) 的随机权重张量，开启梯度跟踪
        >>> output = F.conv1d(input, weight)  # 对输入张量应用一维卷积操作，使用给定的权重
        >>> grad_output = torch.randn(output.shape)  # 创建一个形状与输出相同的随机梯度张量
        >>> # xdoctest: +SKIP  # 跳过测试框架中的特定注释标记
        >>> grad_weight = torch.autograd.grad(output, filter, grad_output)  # 计算卷积操作的权重梯度
        >>> F.grad.conv1d_weight(input, weight.shape, grad_output)  # 返回卷积操作的权重梯度

    """
    weight = grad_output.new_empty(1).expand(weight_size)  # 创建一个与权重相同形状的新空张量，用于存储权重梯度

    return torch.ops.aten.convolution_backward(
        grad_output,  # 输出的梯度
        input,  # 输入张量
        weight,  # 权重张量
        None,  # 偏置参数，这里不使用
        _single(stride),  # 卷积操作的步幅，转换为单元素元组
        _single(padding),  # 卷积操作的填充大小，转换为单元素元组
        _single(dilation),  # 卷积操作的扩展大小，转换为单元素元组
        False,  # 是否使用输出偏置，这里为 False
        [0],  # 指定输出通道的索引列表
        groups,  # 分组数
        (False, True, False),  # 是否转置权重、是否需要计算权重梯度、是否使用混合卷积
    )[1]  # 返回计算得到的权重梯度
# 定义函数，计算卷积操作对输入的梯度
def conv2d_input(
    input_size,
    weight,
    grad_output,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
    r"""Compute the gradient of conv2d with respect to the input of the convolution.

    This is same as the 2D transposed convolution operator under the hood but requires
    the shape of the gradient w.r.t. input to be specified explicitly.

    Args:
        input_size : Shape of the input gradient tensor
        weight: weight tensor (out_channels x in_channels/groups x kH x kW)
        grad_output : output gradient tensor (minibatch x out_channels x oH x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1

    Examples::

        >>> input = torch.randn(1, 1, 3, 3, requires_grad=True)
        >>> weight = torch.randn(1, 1, 1, 2, requires_grad=True)
        >>> output = F.conv2d(input, weight)
        >>> grad_output = torch.randn(output.shape)
        >>> grad_input = torch.autograd.grad(output, input, grad_output)
        >>> F.grad.conv2d_input(input.shape, weight, grad_output)

    """
    # 创建一个与梯度输出形状相同的空张量，作为输入的梯度
    input = grad_output.new_empty(1).expand(input_size)

    # 调用底层的 convolution_backward 函数计算卷积操作对输入的梯度，并返回结果
    return torch.ops.aten.convolution_backward(
        grad_output,
        input,
        weight,
        None,
        _pair(stride),
        _pair(padding),
        _pair(dilation),
        False,
        [0],
        groups,
        (True, False, False),
    )[0]


# 定义函数，计算卷积操作对权重的梯度
def conv2d_weight(
    input,
    weight_size,
    grad_output,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
    r"""Compute the gradient of conv2d with respect to the weight of the convolution.

    Args:
        input: input tensor of shape (minibatch x in_channels x iH x iW)
        weight_size : Shape of the weight gradient tensor
        grad_output : output gradient tensor (minibatch x out_channels x oH x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1

    Examples::

        >>> input = torch.randn(1, 1, 3, 3, requires_grad=True)
        >>> weight = torch.randn(1, 1, 1, 2, requires_grad=True)
        >>> output = F.conv2d(input, weight)
        >>> grad_output = torch.randn(output.shape)
        >>> # xdoctest: +SKIP
        >>> grad_weight = torch.autograd.grad(output, filter, grad_output)
        >>> F.grad.conv2d_weight(input, weight.shape, grad_output)

    """
    # 创建一个与梯度输出形状相同的空张量，作为权重的梯度
    weight = grad_output.new_empty(1).expand(weight_size)
    # 调用 PyTorch ATen 库中的卷积反向传播函数
    return torch.ops.aten.convolution_backward(
        # 反向传播时的梯度输出
        grad_output,
        # 原始输入数据
        input,
        # 卷积核参数
        weight,
        # 空值占位符，表示无额外的偏置项梯度
        None,
        # 步幅，作为二元组传入
        _pair(stride),
        # 填充，作为二元组传入
        _pair(padding),
        # 膨胀系数，作为二元组传入
        _pair(dilation),
        # 是否使用转置卷积（通常为False）
        False,
        # 输入的输入通道索引（通常为[0]，表示单一输入）
        [0],
        # 分组卷积的数量
        groups,
        # 是否进行输出偏置的梯度计算的元组（通常为(False, True, False)）
        (False, True, False),
    )[1]
# 计算 conv3d 对输入的梯度，实际上是进行了3D转置卷积操作，但需要明确指定输入梯度的形状
def conv3d_input(
    input_size,
    weight,
    grad_output,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
    """
    计算卷积操作对输入的梯度。

    Args:
        input_size : 输入梯度张量的形状
        weight: 权重张量 (out_channels x in_channels/groups x kT x kH x kW)
        grad_output : 输出梯度张量 (minibatch x out_channels x oT x oH x oW)
        stride (int or tuple, optional): 卷积的步长。默认: 1
        padding (int or tuple, optional): 输入两侧的零填充。默认: 0
        dilation (int or tuple, optional): 卷积核元素之间的间距。默认: 1
        groups (int, optional): 输入通道到输出通道的阻塞连接数。默认: 1

    Examples::

        >>> input = torch.randn(2, 8, 10, 10, 20, requires_grad=True)
        >>> weight = torch.randn(4, 8, 2, 3, 3, requires_grad=True)
        >>> output = F.conv3d(input, weight)
        >>> grad_output = torch.randn(output.shape)
        >>> grad_input = torch.autograd.grad(output, input, grad_output)
        >>> F.grad.conv3d_input(input.shape, weight, grad_output)

    """
    # 创建一个与 grad_output 具有相同形状的空张量，作为输入梯度
    input = grad_output.new_empty(1).expand(input_size)

    # 调用底层的 ATen 函数，计算卷积操作的反向传播，返回输入梯度
    return torch.ops.aten.convolution_backward(
        grad_output,
        input,
        weight,
        None,
        _triple(stride),
        _triple(padding),
        _triple(dilation),
        False,
        [0],
        groups,
        (True, False, False),
    )[0]


# 计算 conv3d 对权重的梯度
def conv3d_weight(
    input,
    weight_size,
    grad_output,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
    """
    计算卷积操作对权重的梯度。

    Args:
        input: 输入张量的形状 (minibatch x in_channels x iT x iH x iW)
        weight_size : 权重梯度张量的形状
        grad_output : 输出梯度张量 (minibatch x out_channels x oT x oH x oW)
        stride (int or tuple, optional): 卷积的步长。默认: 1
        padding (int or tuple, optional): 输入两侧的零填充。默认: 0
        dilation (int or tuple, optional): 卷积核元素之间的间距。默认: 1
        groups (int, optional): 输入通道到输出通道的阻塞连接数。默认: 1

    Examples::

        >>> input = torch.randn(2, 8, 10, 10, 20, requires_grad=True)
        >>> weight = torch.randn(4, 8, 2, 3, 3, requires_grad=True)
        >>> output = F.conv3d(input, weight)
        >>> grad_output = torch.randn(output.shape)
        >>> grad_weight = torch.autograd.grad(output, weight, grad_output)
        >>> F.grad.conv3d_weight(input, weight.shape, grad_output)

    """
    # 创建一个与 grad_output 形状相同的新张量，但不包含任何数据，其形状为 (1,)，然后通过扩展将其形状调整为 weight_size
    weight = grad_output.new_empty(1).expand(weight_size)
    
    # 调用底层的 PyTorch 函数，执行卷积反向传播的计算
    # 参数说明：
    # - grad_output: 上游梯度，即卷积操作的输出的梯度
    # - input: 卷积操作的输入张量
    # - weight: 卷积核的权重张量
    # - None: 在反向传播中不使用 bias，因此为 None
    # - _triple(stride): 卷积核的步幅，通过 _triple 函数将其转换为三维形式
    # - _triple(padding): 卷积操作的填充，通过 _triple 函数将其转换为三维形式
    # - _triple(dilation): 卷积核的扩展，通过 _triple 函数将其转换为三维形式
    # - False: 是否使用 transposed 卷积，默认为 False
    # - [0]: 卷积操作的输出张量维度的偏移
    # - groups: 卷积操作中的分组数
    # - (False, True, False): 控制输入、输出、权重在卷积操作中的反转
    torch.ops.aten.convolution_backward(
        grad_output,
        input,
        weight,
        None,
        _triple(stride),
        _triple(padding),
        _triple(dilation),
        False,
        [0],
        groups,
        (False, True, False),
    )[1]
```