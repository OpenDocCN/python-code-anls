# `.\pytorch\torch\nn\functional.py`

```
# 导入模块和库
import importlib  # 导入模块导入工具
import math  # 导入数学函数库
import warnings  # 导入警告模块
from typing import Callable, List, Optional, Tuple, TYPE_CHECKING, Union  # 导入类型提示相关的工具

import torch  # 导入PyTorch库
from torch import _VF, sym_int as _sym_int, Tensor  # 从torch导入私有子模块和Tensor类
from torch._C import _add_docstr, _infer_size  # 从torch._C导入添加文档字符串和推断尺寸功能
from torch._jit_internal import (  # 从torch._jit_internal导入多个函数和类
    _overload,
    boolean_dispatch,
    BroadcastingList1,
    BroadcastingList2,
    BroadcastingList3,
)
from torch._torch_docs import reproducibility_notes, sparse_support_notes, tf32_notes  # 从torch._torch_docs导入多个文档注释

from torch.nn import _reduction as _Reduction, grad  # 从torch.nn导入重命名为_Reduction的模块和grad函数  # noqa: F401

from torch.nn.modules.utils import _list_with_default, _pair, _single, _triple  # 从torch.nn.modules.utils导入多个工具函数
from torch.overrides import (  # 从torch.overrides导入多个函数
    handle_torch_function,
    has_torch_function,
    has_torch_function_unary,
    has_torch_function_variadic,
)

# 如果TYPE_CHECKING为真，则从torch.types导入_dtype作为DType类型，否则将DType设置为整数类型
if TYPE_CHECKING:
    from torch.types import _dtype as DType
else:
    DType = int

# 尝试导入numpy库，如果失败则将np设置为None
try:
    import numpy as np
except ModuleNotFoundError:
    np = None

# 使用_add_docstr函数为torch.conv1d函数添加文档字符串
conv1d = _add_docstr(
    torch.conv1d,
    r"""
conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor

Applies a 1D convolution over an input signal composed of several input
planes.

{tf32_note}

See :class:`~torch.nn.Conv1d` for details and output shape.

Note:
    {cudnn_reproducibility_note}

Note:
    This operator supports complex data types i.e. ``complex32, complex64, complex128``.
""".format(
        **reproducibility_notes, **tf32_notes
    )
    + r"""

Args:
    input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iW)`
    weight: filters of shape :math:`(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kW)`
    bias: optional bias of shape :math:`(\text{out\_channels})`. Default: ``None``
    stride: the stride of the convolving kernel. Can be a single number or
      a one-element tuple `(sW,)`. Default: 1
    padding: implicit paddings on both sides of the input. Can be a string {'valid', 'same'},
      single number or a one-element tuple `(padW,)`. Default: 0
      ``padding='valid'`` is the same as no padding. ``padding='same'`` pads
      the input so the output has the same shape as the input. However, this mode
      doesn't support any stride values other than 1.

      .. warning::
          For ``padding='same'``, if the ``weight`` is even-length and
          ``dilation`` is odd in any dimension, a full :func:`pad` operation
          may be needed internally. Lowering performance.
    dilation: the spacing between kernel elements. Can be a single number or
      a one-element tuple `(dW,)`. Default: 1
    groups: split input into groups, :math:`\text{in\_channels}` should be divisible by
      the number of groups. Default: 1

Examples::

    >>> inputs = torch.randn(33, 16, 30)
    >>> filters = torch.randn(20, 16, 5)
    >>> F.conv1d(inputs, filters)
""",
)

# 使用_add_docstr函数为torch.conv2d函数添加文档字符串
conv2d = _add_docstr(
    torch.conv2d,
    r"""
conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor

Applies a 2D convolution over an input signal composed of several input
planes.

{tf32_note}

See :class:`~torch.nn.Conv2d` for details and output shape.

Note:
    {cudnn_reproducibility_note}

Note:
    This operator supports complex data types i.e. ``complex32, complex64, complex128``.
""".format(
        **reproducibility_notes, **tf32_notes
    )
    + r"""

Args:
    input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iH , iW)`
    weight: filters of shape :math:`(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kH , kW)`
    bias: optional bias of shape :math:`(\text{out\_channels})`. Default: ``None``
    stride: the stride of the convolving kernel. Can be a single number or
      a tuple `(sH, sW)`. Default: 1
    padding: implicit paddings on both sides of the input. Can be a string {'valid', 'same'},
      a single number or a tuple `(padH, padW)`. Default: 0
      ``padding='valid'`` is the same as no padding. ``padding='same'`` pads
      the input so the output has the same shape as the input. However, this mode
      doesn't support any stride values other than 1.

      .. warning::
          For ``padding='same'``, if the ``weight`` is even-length and
          ``dilation`` is odd in any dimension, a full :func:`pad` operation
          may be needed internally. Lowering performance.
    dilation: the spacing between kernel elements. Can be a single number or
      a tuple `(dH, dW)`. Default: 1
    groups: split input into groups, :math:`\text{in\_channels}` should be divisible by
      the number of groups. Default: 1

Examples::

    >>> inputs = torch.randn(20, 16, 50, 100)
    >>> filters = torch.randn(33, 16, 3, 3)
    >>> F.conv2d(inputs, filters)
""",
)
conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor
# 定义了一个二维卷积操作函数，用于处理由多个输入平面组成的输入图像。

Applies a 2D convolution over an input image composed of several input
planes.

{tf32_note}

See :class:`~torch.nn.Conv2d` for details and output shape.

Note:
    {cudnn_reproducibility_note}
# 从格式化字符串中获取 cudnn 的可重现性注释。

Note:
    This operator supports complex data types i.e. ``complex32, complex64, complex128``.
# 说明此操作支持复杂数据类型，如 complex32, complex64, complex128。

""".format(
        **reproducibility_notes, **tf32_notes
    )
    + r"""

Args:
    input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iH , iW)`
    # 输入张量的形状应为 (批次大小, 输入通道数, 输入高度, 输入宽度)
    
    weight: filters of shape :math:`(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kH , kW)`
    # 过滤器的形状应为 (输出通道数, 输入通道数/分组数, 卷积核高度, 卷积核宽度)
    
    bias: optional bias tensor of shape :math:`(\text{out\_channels})`. Default: ``None``
    # 可选的偏置张量形状为 (输出通道数)，默认为 None
    
    stride: the stride of the convolving kernel. Can be a single number or a
      tuple `(sH, sW)`. Default: 1
    # 卷积核的步幅，可以是单个数字或元组 (高度方向步幅, 宽度方向步幅)，默认为 1
    
    padding: implicit paddings on both sides of the input. Can be a string {'valid', 'same'},
      single number or a tuple `(padH, padW)`. Default: 0
      ``padding='valid'`` is the same as no padding. ``padding='same'`` pads
      the input so the output has the same shape as the input. However, this mode
      doesn't support any stride values other than 1.
    # 输入两侧的隐式填充。可以是字符串 {'valid', 'same'}、单个数字或元组 (填充高度, 填充宽度)，默认为 0。
    # ``padding='valid'`` 表示不填充，``padding='same'`` 表示填充输入使输出与输入形状相同。
    # 然而，此模式不支持除 1 以外的任何步幅值。
    
      .. warning::
          For ``padding='same'``, if the ``weight`` is even-length and
          ``dilation`` is odd in any dimension, a full :func:`pad` operation
          may be needed internally. Lowering performance.
    # 警告：对于 ``padding='same'``，如果权重是偶数长度且 dilation 在任何维度上为奇数，可能需要内部执行完整的 pad 操作，降低性能。

    dilation: the spacing between kernel elements. Can be a single number or
      a tuple `(dH, dW)`. Default: 1
    # 卷积核元素之间的间距，可以是单个数字或元组 (高度方向间距, 宽度方向间距)，默认为 1
    
    groups: split input into groups, both :math:`\text{in\_channels}` and :math:`\text{out\_channels}`
      should be divisible by the number of groups. Default: 1
    # 将输入分成多个组，要求输入通道数和输出通道数均可被分组数整除，默认为 1

Examples::

    >>> # With square kernels and equal stride
    >>> filters = torch.randn(8, 4, 3, 3)
    >>> inputs = torch.randn(1, 4, 5, 5)
    >>> F.conv2d(inputs, filters, padding=1)
# 示例：使用方形卷积核和相同步幅

"""  # noqa: E501

conv3d = _add_docstr(
    torch.conv3d,
    r"""
conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor

Applies a 3D convolution over an input image composed of several input
planes.

{tf32_note}

See :class:`~torch.nn.Conv3d` for details and output shape.

Note:
    {cudnn_reproducibility_note}

Note:
    This operator supports complex data types i.e. ``complex32, complex64, complex128``.
# 说明此操作支持复杂数据类型，如 complex32, complex64, complex128。

""".format(
        **reproducibility_notes, **tf32_notes
    )
    + r"""

Args:
    input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iT , iH , iW)`
    # 输入张量的形状应为 (批次大小, 输入通道数, 输入时间维度, 输入高度, 输入宽度)
    
    weight: filters of shape :math:`(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kT , kH , kW)`
    # 过滤器的形状应为 (输出通道数, 输入通道数/分组数, 卷积核时间维度, 卷积核高度, 卷积核宽度)
    
    bias: optional bias tensor of shape :math:`(\text{out\_channels})`. Default: None
    # 可选的偏置张量形状为 (输出通道数)，默认为 None
    
    stride: the stride of the convolving kernel. Can be a single number or a
      tuple `(sT, sH, sW)`. Default: 1
    # 卷积核的步幅，可以是单个数字或元组 (时间方向步幅, 高度方向步幅, 宽度方向步幅)，默认为 1
    
"""```python
conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor
# 定义一个对输入图像进行二维卷积操作的函数，输入图像由多个输入平面组成。

Applies a 2D convolution over an input image composed of several input
planes.

{tf32_note}
# 插入 tf32_note 的内容，具体细节和输出形状请参见 torch.nn.Conv2d。

See :class:`~torch.nn.Conv2d` for details and output shape.
# 查看 :class:`~torch.nn.Conv2d` 获取详细信息和输出形状。

Note:
    {cudnn_reproducibility_note}
# 插入 cudnn_reproducibility_note 的内容，说明注意事项和 cudnn 可重现性。

Note:
    This operator supports complex data types i.e. ``complex32, complex64, complex128``.
# 说明此运算支持复杂数据类型，例如 complex32, complex64, complex128。

""".format(
        **reproducibility_notes, **tf32_notes
    )
    + r"""

Args:
    input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iH , iW)`
    # 输入张量的形状应为 :math:`(\text{minibatch} , \text{in\_channels} , iH , iW)`
    
    weight: filters of shape :math:`(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kH , kW)`
    # 权重的形状应为 :math:`(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kH , kW)`
    
    bias: optional bias tensor of shape :math:`(\text{out\_channels})`. Default: ``None``
    # 可选的偏置张量的形状为 :math:`(\text{out\_channels})`，默认为 ``None``
    
    stride: the stride of the convolving kernel. Can be a single number or a
      tuple `(sH, sW)`. Default: 1
    # 卷积核的步幅，可以是单个数字或元组 `(sH, sW)`，默认为 1
    
    padding: implicit paddings on both sides of the input. Can be a string {'valid', 'same'},
      single number or a tuple `(padH, padW)`. Default: 0
      ``padding='valid'`` is the same as no padding. ``padding='same'`` pads
      the input so the output has the same shape as the input. However, this mode
      doesn't support any stride values other than 1.
    # 输入两侧的隐式填充。可以是字符串 {'valid', 'same'}、单个数字或元组 `(padH, padW)`，默认为 0。
    # ``padding='valid'`` 表示没有填充。``padding='same'`` 对输入进行填充，使输出与输入形状相同。
    # 然而，此模式不支持除 1 以外的任何步幅值。
    
      .. warning::
          For ``padding='same'``, if the ``weight`` is even-length and
          ``dilation`` is odd in any dimension, a full :func:`pad` operation
          may be needed internally. Lowering performance.
    # 警告：对于 ``padding='same'``，如果 ``weight`` 是偶数长度且 ``dilation`` 在任何维度上为奇数，可能需要内部执行完整的 :func:`pad` 操作，从而降低性能。

    dilation: the spacing between kernel elements. Can be a single number or
      a tuple `(dH, dW)`. Default: 1
    # 卷积核元素之间的间距，可以是单个数字或元组 `(dH, dW)`，默认为 1
    
    groups: split input into groups, both :math:`\text{in\_channels}` and :math:`\text{out\_channels}`
      should be divisible by the number of groups. Default: 1
    # 将输入分成多个组，要求输入通道数和输出通道数均可被分组数整除，默认为 1

Examples::

    >>> # With square kernels and equal stride
    >>> filters = torch.randn(8, 4, 3, 3)
    >>> inputs = torch.randn(1, 4, 5, 5)
    >>> F.conv2d(inputs, filters, padding=1)
# 示例：使用方形卷积核和相同步幅

"""  # noqa: E501

conv3d = _add_docstr(
    torch.conv3d,
    r"""
conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor

Applies a 3D convolution over an input image composed of several input
planes.

{tf32_note}
# 插入 tf32_note 的内容，具体细节和输出形状请参见 torch.nn.Conv3d。

See :class:`~torch.nn.Conv3d` for details and output shape.
# 查看 :class:`~torch.nn.Conv3d` 获取详细信息和输出形状。

Note:
    {cudnn_reproducibility_note}
# 插入 cudnn_reproducibility_note 的内容，说明注意事项和 cudnn 可重现性。

Note:
    This operator supports complex data types i.e. ``complex32, complex64, complex128``.
# 说明此运算支持复杂数据类型，例如 complex32, complex64, complex128。

""".format(
        **reproducibility_notes, **tf32_notes
    )
    + r"""

Args:
    input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iT , iH , iW)`
    # 输入张量的形状应为 :math:`(\text{minibatch} , \text{in\_channels} , iT , iH , iW)`
    
    weight: filters of shape :math:`(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kT , kH , kW)`
    # 权重的形状应为 :math:`(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kT , kH , kW)`
    
    bias: optional bias tensor of shape :math:`(\text{out\_channels})`. Default: None
    # 可选的偏置张量的形状为 :math:`(\text{out\_channels})`，默认为 None
    
    stride: the stride of the convolving kernel. Can be a single number or a
      tuple `(sT, sH, sW)`. Default: 1
    # 卷积核的步幅，可以是单个数字或元组 `(sT, sH, sW)`，默认为 1
    
"""
    padding: 输入数据两侧的隐式填充。可以是字符串 {'valid', 'same'}，单个数字或元组 `(padT, padH, padW)`。默认为 0。
      ``padding='valid'`` 表示没有填充。 ``padding='same'`` 表示填充输入使得输出具有与输入相同的形状。然而，这种模式不支持除了 1 以外的任何步长值。

      .. warning::
          对于 ``padding='same'``，如果 ``weight`` 的长度是偶数，并且在任何维度上 ``dilation`` 是奇数，可能需要进行完整的 :func:`pad` 操作，从而降低性能。

    dilation: 卷积核元素之间的间隔。可以是单个数字或元组 `(dT, dH, dW)`。默认为 1。
    groups: 将输入分成多个组，:math:`\text{in\_channels}` 应该能够被组数整除。默认为 1。
# 使用 torch.randn 创建一个形状为 (33, 16, 3, 3, 3) 的张量作为卷积滤波器
filters = torch.randn(33, 16, 3, 3, 3)
# 使用 torch.randn 创建一个形状为 (20, 16, 50, 10, 20) 的张量作为输入数据
inputs = torch.randn(20, 16, 50, 10, 20)
# 调用 F.conv3d 对输入数据和滤波器进行 3D 卷积操作，返回卷积结果张量
F.conv3d(inputs, filters)



conv_transpose1d = _add_docstr(
    torch.conv_transpose1d,
    r"""
conv_transpose1d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor

Applies a 1D transposed convolution operator over an input signal
composed of several input planes, sometimes also called "deconvolution".

{tf32_note}

See :class:`~torch.nn.ConvTranspose1d` for details and output shape.

Note:
    {cudnn_reproducibility_note}
""".format(
        **reproducibility_notes, **tf32_notes
    )
    + r"""

Args:
    input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iW)`
    weight: filters of shape :math:`(\text{in\_channels} , \frac{\text{out\_channels}}{\text{groups}} , kW)`
    bias: optional bias of shape :math:`(\text{out\_channels})`. Default: None
    stride: the stride of the convolving kernel. Can be a single number or a
      tuple ``(sW,)``. Default: 1
    padding: ``dilation * (kernel_size - 1) - padding`` zero-padding will be added to both
      sides of each dimension in the input. Can be a single number or a tuple
      ``(padW,)``. Default: 0
    output_padding: additional size added to one side of each dimension in the
      output shape. Can be a single number or a tuple ``(out_padW)``. Default: 0
    groups: split input into groups, :math:`\text{in\_channels}` should be divisible by the
      number of groups. Default: 1
    dilation: the spacing between kernel elements. Can be a single number or
      a tuple ``(dW,)``. Default: 1

Examples::

    >>> inputs = torch.randn(20, 16, 50)
    >>> weights = torch.randn(16, 33, 5)
    >>> F.conv_transpose1d(inputs, weights)
"""
)
# 定义 conv_transpose1d 函数，对应 torch.conv_transpose1d，添加了详细的文档字符串来描述函数的作用、参数和示例用法



conv_transpose2d = _add_docstr(
    torch.conv_transpose2d,
    r"""
conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor

Applies a 2D transposed convolution operator over an input image
composed of several input planes, sometimes also called "deconvolution".

{tf32_note}

See :class:`~torch.nn.ConvTranspose2d` for details and output shape.

Note:
    {cudnn_reproducibility_note}
""".format(
        **reproducibility_notes, **tf32_notes
    )
    + r"""

Args:
    input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iH , iW)`
    weight: filters of shape :math:`(\text{in\_channels} , \frac{\text{out\_channels}}{\text{groups}} , kH , kW)`
    bias: optional bias of shape :math:`(\text{out\_channels})`. Default: None
    stride: the stride of the convolving kernel. Can be a single number or a
      tuple ``(sH, sW)``. Default: 1
    padding: ``dilation * (kernel_size - 1) - padding`` zero-padding will be added to both
      sides of each dimension in the input. Can be a single number or a tuple
      ``(padH, padW)``. Default: 0
    output_padding: additional size added to one side of each dimension in the
      output shape. Can be a single number or a tuple ``(out_padH, out_padW)``. Default: 0

"""
)
# 定义 conv_transpose2d 函数，对应 torch.conv_transpose2d，添加了详细的文档字符串来描述函数的作用、参数和注意事项
    output_padding: additional size added to one side of each dimension in the
      output shape. Can be a single number or a tuple ``(out_padH, out_padW)``.
      Default: 0
    groups: split input into groups, :math:`\text{in\_channels}` should be divisible by the
      number of groups. Default: 1
    dilation: the spacing between kernel elements. Can be a single number or
      a tuple ``(dH, dW)``. Default: 1



# output_padding：在输出形状的每个维度的一侧添加的额外大小。可以是一个单独的数字或元组 ``(out_padH, out_padW)``
# 默认值：0

# groups：将输入分成多个组，:math:`\text{in\_channels}` 应该能够被组数整除。
# 默认值：1

# dilation：核元素之间的间隔。可以是一个单独的数字或元组 ``(dH, dW)``
# 默认值：1
# 定义了一个函数 avg_pool1d，用于对输入的一维信号进行平均池化操作
avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True) -> Tensor

# 应用一维平均池化到输入信号中的每个通道
Applies a 1D average pooling over an input signal composed of several
input planes.

# 查看 torch.nn.AvgPool1d 类获取更多细节和输出形状信息
See :class:`~torch.nn.AvgPool1d` for details and output shape.

# 参数说明：
Args:
    input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iW)`
    输入参数：形状为 :math:`(\text{minibatch} , \text{in\_channels} , iW)` 的输入张量
    kernel_size: 池化核大小，定义了池化窗口的宽度
    stride: 池化操作的步长。可以是单个数字或元组 ``(strideT)``
    padding: 在输入的两端各添加的零填充的数目。可以是单个数字或元组 ``(padT)``
    ceil_mode: 是否使用天花板模式计算输出形状。默认为 False
    count_include_pad: 是否包括填充值在内。默认为 True
    kernel_size: the size of the window. Can be a single number or a
      tuple `(kW,)`
    stride: the stride of the window. Can be a single number or a tuple
      `(sW,)`. Default: :attr:`kernel_size`
    padding: implicit zero paddings on both sides of the input. Can be a
      single number or a tuple `(padW,)`. Default: 0
    ceil_mode: when True, will use `ceil` instead of `floor` to compute the
        output shape. Default: ``False``
    count_include_pad: when True, will include the zero-padding in the
        averaging calculation. Default: ``True``
"""
avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None) -> Tensor

2D 平均池化操作，将输入的每个 kH x kW 区域按步长 sH x sW 进行池化。输出特征数量等于输入平面数。

详见 torch.nn.AvgPool2d 以获取更多详情和输出形状。

Args:
    input: 输入张量 (\text{minibatch} , \text{in_channels} , iH , iW)
    kernel_size: 池化区域的大小。可以是单个数字或元组 (kH, kW)
    stride: 池化操作的步长。可以是单个数字或元组 (sH, sW)。默认为 kernel_size
    padding: 输入两侧的隐式零填充。可以是单个数字或元组 (padH, padW)。默认为 0
    ceil_mode: 当为 True 时，在计算输出形状时使用 `ceil` 而非 `floor`。默认为 False
    count_include_pad: 当为 True 时，将零填充包含在平均计算中。默认为 True
    divisor_override: 如果指定，将用作除数；否则将使用池化区域的大小。默认为 None
"""

"""
avg_pool3d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None) -> Tensor

3D 平均池化操作，将输入的每个 kT x kH x kW 区域按步长 sT x sH x sW 进行池化。输出特征数量等于输入平面数除以 sT。

详见 torch.nn.AvgPool3d 以获取更多详情和输出形状。

Args:
    input: 输入张量 (\text{minibatch} , \text{in_channels} , iT \times iH , iW)
    kernel_size: 池化区域的大小。可以是单个数字或元组 (kT, kH, kW)
    stride: 池化操作的步长。可以是单个数字或元组 (sT, sH, sW)。默认为 kernel_size
    padding: 输入两侧的隐式零填充。可以是单个数字或元组 (padT, padH, padW)。默认为 0
    ceil_mode: 当为 True 时，在计算输出形状时使用 `ceil` 而非 `floor`。默认为 False
    count_include_pad: 当为 True 时，将零填充包含在平均计算中。默认为 True
    divisor_override: 如果指定，将用作除数；否则将使用池化区域的大小。默认为 None
"""
    output_ratio: Optional[BroadcastingList2[float]] = None,
    # 定义一个名为 output_ratio 的变量，类型为 Optional[BroadcastingList2[float]]，默认值为 None
    return_indices: bool = False,
    # 定义一个名为 return_indices 的变量，类型为 bool，默认值为 False
    _random_samples: Optional[Tensor] = None,
    # 定义一个名为 _random_samples 的变量，类型为 Optional[Tensor]，默认值为 None
def fractional_max_pool2d(input, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None) -> Tuple[Tensor, Tensor]:  # noqa: D400
    r"""
    fractional_max_pool2d(input, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None)

    Applies 2D fractional max pooling over an input signal composed of several input planes.

    Fractional MaxPooling is described in detail in the paper `Fractional MaxPooling`_ by Ben Graham

    The max-pooling operation is applied in :math:`kH \times kW` regions by a stochastic
    step size determined by the target output size.
    The number of output features is equal to the number of input planes.

    Args:
        kernel_size: the size of the window to take a max over.
                     Can be a single number :math:`k` (for a square kernel of :math:`k \times k`)
                     or a tuple `(kH, kW)`
        output_size: the target output size of the image of the form :math:`oH \times oW`.
                     Can be a tuple `(oH, oW)` or a single number :math:`oH` for a square image :math:`oH \times oH`
        output_ratio: If one wants to have an output size as a ratio of the input size, this option can be given.
                      This has to be a number or tuple in the range (0, 1)
        return_indices: if ``True``, will return the indices along with the outputs.
                        Useful to pass to :func:`~torch.nn.functional.max_unpool2d`.

    Examples::
        >>> input = torch.randn(20, 16, 50, 32)
        >>> # pool of square window of size=3, and target output size 13x12
        >>> F.fractional_max_pool2d(input, 3, output_size=(13, 12))
        >>> # pool of square window and target output size being half of input image size
        >>> F.fractional_max_pool2d(input, 3, output_ratio=(0.5, 0.5))

    .. _Fractional MaxPooling:
        http://arxiv.org/abs/1412.6071
    """

    # Check if input has a Torch function variation with variadic arguments
    if has_torch_function_variadic(input, _random_samples):
        # If true, handle Torch function with fractional_max_pool2d_with_indices
        return handle_torch_function(
            fractional_max_pool2d_with_indices,
            (input, _random_samples),
            input,
            kernel_size,
            output_size=output_size,
            output_ratio=output_ratio,
            return_indices=return_indices,
            _random_samples=_random_samples,
        )

    # If output_size is not provided and output_ratio is not specified, raise an error
    if output_size is None and output_ratio is None:
        raise ValueError(
            "fractional_max_pool2d requires specifying either an output_size or an output_ratio"
        )

    # If output_size is not provided, calculate it based on the specified output_ratio
    if output_size is None:
        assert output_ratio is not None
        if len(output_ratio) > 2:
            raise ValueError(
                "fractional_max_pool2d requires output_ratio to either be a single Int or tuple of Ints."
            )
        _output_ratio = _pair(output_ratio)
        output_size = [
            int(input.size(-2) * _output_ratio[0]),
            int(input.size(-1) * _output_ratio[1]),
        ]
    # 如果 _random_samples 为 None，则进行以下操作
    if _random_samples is None:
        # 如果输入的维度是 3 维，则设定批次数为 1，否则设定为输入的第一维大小
        n_batch = 1 if input.dim() == 3 else input.size(0)
        # 生成一个与输入大小相同的随机张量，形状为 (n_batch, input.size(-3), 2)
        # 数据类型与输入相同，存储设备与输入相同
        _random_samples = torch.rand(
            n_batch, input.size(-3), 2, dtype=input.dtype, device=input.device
        )
    
    # 调用 Torch 库中的 fractional_max_pool2d 函数进行分数最大池化操作
    # 使用给定的输入、核大小、输出大小以及之前生成的随机样本 _random_samples
    return torch._C._nn.fractional_max_pool2d(
        input, kernel_size, output_size, _random_samples
    )
def _fractional_max_pool2d(
    input: Tensor,
    kernel_size: BroadcastingList2[int],
    output_size: Optional[BroadcastingList2[int]] = None,
    output_ratio: Optional[BroadcastingList2[float]] = None,
    return_indices: bool = False,
    _random_samples: Optional[Tensor] = None,
) -> Tensor:
    # 检查输入参数是否包含torch函数的可变参数，如果是则调用torch函数处理
    if has_torch_function_variadic(input, _random_samples):
        return handle_torch_function(
            fractional_max_pool2d,
            (input, _random_samples),
            input,
            kernel_size,
            output_size=output_size,
            output_ratio=output_ratio,
            return_indices=return_indices,
            _random_samples=_random_samples,
        )
    # 否则调用自定义的fractional_max_pool2d_with_indices函数处理
    return fractional_max_pool2d_with_indices(
        input, kernel_size, output_size, output_ratio, return_indices, _random_samples
    )[0]


fractional_max_pool2d = boolean_dispatch(
    arg_name="return_indices",
    arg_index=4,
    default=False,
    if_true=fractional_max_pool2d_with_indices,
    if_false=_fractional_max_pool2d,
    module_name=__name__,
    func_name="fractional_max_pool2d",
)


def fractional_max_pool3d_with_indices(
    input: Tensor,
    kernel_size: BroadcastingList3[int],
    output_size: Optional[BroadcastingList3[int]] = None,
    output_ratio: Optional[BroadcastingList3[float]] = None,
    return_indices: bool = False,
    _random_samples: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:  # noqa: D400
    r"""
    fractional_max_pool3d(input, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None)

    Applies 3D fractional max pooling over an input signal composed of several input planes.

    Fractional MaxPooling is described in detail in the paper `Fractional MaxPooling`_ by Ben Graham

    The max-pooling operation is applied in :math:`kT \times kH \times kW` regions by a stochastic
    step size determined by the target output size.
    The number of output features is equal to the number of input planes.

    Args:
        kernel_size: the size of the window to take a max over.
                     Can be a single number :math:`k` (for a cubic kernel of :math:`k \times k \times k`)
                     or a tuple `(kT, kH, kW)`
        output_size: the target output size of the form :math:`oT \times oH \times oW`.
                     Can be a tuple `(oT, oH, oW)` or a single number :math:`oH` for a cubic output
                     :math:`oH \times oH \times oH`
        output_ratio: If one wants to have an output size as a ratio of the input size, this option can be given.
                      This has to be a number or tuple in the range (0, 1)
        return_indices: if ``True``, will return the indices along with the outputs.
                        Useful to pass to :func:`~torch.nn.functional.max_unpool3d`.
    """
    # 实现对输入的3D分数最大池化操作，并返回输出张量及对应的索引张量
    pass
    # 检查输入是否具有Torch函数的变体和随机样本
    if has_torch_function_variadic(input, _random_samples):
        # 如果有，调用处理Torch函数的方法，返回处理结果
        return handle_torch_function(
            fractional_max_pool3d_with_indices,
            (input, _random_samples),
            input,
            kernel_size,
            output_size=output_size,
            output_ratio=output_ratio,
            return_indices=return_indices,
            _random_samples=_random_samples,
        )
    
    # 如果未指定输出大小和输出比率，则引发值错误
    if output_size is None and output_ratio is None:
        raise ValueError(
            "fractional_max_pool3d需要指定output_size或output_ratio之一"
        )
    
    # 如果未指定输出大小，则根据输出比率计算输出大小
    if output_size is None:
        assert output_ratio is not None
        _output_ratio = _triple(output_ratio)
        output_size = [
            int(input.size(-3) * _output_ratio[0]),
            int(input.size(-2) * _output_ratio[1]),
            int(input.size(-1) * _output_ratio[2]),
        ]

    # 如果随机样本未提供，则根据输入的维度创建随机样本
    if _random_samples is None:
        n_batch = 1 if input.dim() == 4 else input.size(0)
        _random_samples = torch.rand(
            n_batch, input.size(-4), 3, dtype=input.dtype, device=input.device
        )
    
    # 调用底层的C++函数实现分数最大池化操作
    return torch._C._nn.fractional_max_pool3d(
        input, kernel_size, output_size, _random_samples
    )
def _fractional_max_pool3d(
    input: Tensor,
    kernel_size: BroadcastingList3[int],
    output_size: Optional[BroadcastingList3[int]] = None,
    output_ratio: Optional[BroadcastingList3[float]] = None,
    return_indices: bool = False,
    _random_samples: Optional[Tensor] = None,
) -> Tensor:
    # 检查输入参数是否具有 torch 函数的可变性
    if has_torch_function_variadic(input, _random_samples):
        # 处理 torch 函数的调用
        return handle_torch_function(
            fractional_max_pool3d,
            (input, _random_samples),
            input,
            kernel_size,
            output_size=output_size,
            output_ratio=output_ratio,
            return_indices=return_indices,
            _random_samples=_random_samples,
        )
    # 调用内部函数处理分数最大池化
    return fractional_max_pool3d_with_indices(
        input, kernel_size, output_size, output_ratio, return_indices, _random_samples
    )[0]


fractional_max_pool3d = boolean_dispatch(
    arg_name="return_indices",
    arg_index=4,
    default=False,
    if_true=fractional_max_pool3d_with_indices,
    if_false=_fractional_max_pool3d,
    module_name=__name__,
    func_name="fractional_max_pool3d",
)


def max_pool1d_with_indices(
    input: Tensor,
    kernel_size: BroadcastingList1[int],
    stride: Optional[BroadcastingList1[int]] = None,
    padding: BroadcastingList1[int] = 0,
    dilation: BroadcastingList1[int] = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Tuple[Tensor, Tensor]:  # noqa: D400
    r"""
    max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)

    对输入信号的每个输入平面应用 1D 最大池化。

    .. note::
        :attr:`ceil_mode` 和 :attr:`return_indices` 的顺序与 :class:`~torch.nn.MaxPool1d` 中的不同，
        将在将来的版本中更改。

    详见 :class:`~torch.nn.MaxPool1d` 获取详情。

    Args:
        input: 输入张量的形状为 :math:`(\text{minibatch} , \text{in\_channels} , iW)`，可选的小批量维度。
        kernel_size: 窗口的大小。可以是一个数字或元组 `(kW,)`
        stride: 窗口的步幅。可以是一个数字或元组 `(sW,)`。默认值：:attr:`kernel_size`
        padding: 隐式负无穷填充，必须 >= 0 且 <= kernel_size / 2。
        dilation: 滑动窗口内元素之间的步幅，必须 > 0。
        ceil_mode: 如果 ``True``，将使用 `ceil` 而不是 `floor` 来计算输出形状。这
                   确保每个输入张量中的每个元素都被滑动窗口覆盖。
        return_indices: 如果 ``True``，将返回最大值的 argmax。
                        对 :class:`torch.nn.functional.max_unpool1d` 很有用。
    """
    # 检查输入是否具有torch函数的一元操作
    if has_torch_function_unary(input):
        # 如果有，调用处理torch函数的方法，返回处理结果
        return handle_torch_function(
            max_pool1d_with_indices,  # 要处理的torch函数
            (input,),  # 输入参数的元组
            input,  # 实际输入
            kernel_size,  # 池化核大小
            stride=stride,  # 步幅
            padding=padding,  # 填充
            dilation=dilation,  # 扩张率
            ceil_mode=ceil_mode,  # 是否使用天花板模式
            return_indices=return_indices,  # 是否返回池化的索引
        )
    
    # 如果未指定步幅，则用torch.jit.annotate标注一个空的int列表作为步幅
    if stride is None:
        stride = torch.jit.annotate(List[int], [])

    # 返回torch.max_pool1d_with_indices的结果，执行一维最大池化操作
    return torch.max_pool1d_with_indices(
        input, kernel_size, stride, padding, dilation, ceil_mode
    )
# 定义了一个函数_max_pool1d，用于执行一维最大池化操作
def _max_pool1d(
    input: Tensor,
    kernel_size: BroadcastingList1[int],
    stride: Optional[BroadcastingList1[int]] = None,
    padding: BroadcastingList1[int] = 0,
    dilation: BroadcastingList1[int] = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Tensor:
    # 检查输入是否具有torch函数的重载，如果是，则调用torch函数处理
    if has_torch_function_unary(input):
        return handle_torch_function(
            max_pool1d,
            (input,),  # 输入作为元组传递给torch函数处理
            input,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            return_indices=return_indices,
        )
    # 如果未提供stride参数，则初始化为空列表
    if stride is None:
        stride = torch.jit.annotate(List[int], [])
    # 调用torch库的一维最大池化函数，并返回结果
    return torch.max_pool1d(input, kernel_size, stride, padding, dilation, ceil_mode)


# 使用boolean_dispatch函数根据return_indices参数的值来选择执行哪个函数
max_pool1d = boolean_dispatch(
    arg_name="return_indices",
    arg_index=6,
    default=False,
    if_true=max_pool1d_with_indices,  # 如果return_indices为True，则执行max_pool1d_with_indices函数
    if_false=_max_pool1d,  # 如果return_indices为False，则执行_max_pool1d函数
    module_name=__name__,
    func_name="max_pool1d",
)


# 定义了一个函数max_pool2d_with_indices，用于执行二维最大池化操作并返回索引
def max_pool2d_with_indices(
    input: Tensor,
    kernel_size: BroadcastingList2[int],
    stride: Optional[BroadcastingList2[int]] = None,
    padding: BroadcastingList2[int] = 0,
    dilation: BroadcastingList2[int] = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Tuple[Tensor, Tensor]:  # noqa: D400
    r"""
    max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)

    对由多个输入平面组成的输入信号应用二维最大池化。

    .. note::
        ceil_mode和return_indices的顺序与torch.nn.MaxPool2d中的顺序不同，并将在未来版本中更改。

    Args:
        input: 输入张量 :math:`(\text{minibatch} , \text{in\_channels} , iH , iW)`，可以选择小批次维度。
        kernel_size: 池化区域的大小。可以是一个数或元组 `(kH, kW)`
        stride: 池化操作的步幅。可以是一个数或元组 `(sH, sW)`。默认值为 `kernel_size`
        padding: 在两侧添加的隐式负无穷填充，必须 >= 0 且 <= kernel_size / 2。
        dilation: 滑动窗口内元素之间的步幅，必须 > 0。
        ceil_mode: 如果 ``True``，将使用 `ceil` 而不是 `floor` 来计算输出形状。这确保了输入张量的每个元素都被滑动窗口覆盖。
        return_indices: 如果 ``True``，将返回最大值的argmax值。对后续的torch.nn.functional.max_unpool2d很有用
    """
    # 如果输入参数 input 满足 torch 函数的一元操作条件
    if has_torch_function_unary(input):
        # 调用处理 torch 函数的方法，并返回结果
        return handle_torch_function(
            max_pool2d_with_indices,    # 使用的 torch 函数 max_pool2d_with_indices
            (input,),                   # 参数元组，包含输入参数 input
            input,                      # 再次传递输入参数 input
            kernel_size,                # 池化核大小
            stride=stride,              # 步幅
            padding=padding,            # 填充
            dilation=dilation,          # 膨胀率
            ceil_mode=ceil_mode,        # 是否使用 ceil 模式
            return_indices=return_indices   # 是否返回索引
        )
    
    # 如果未指定 stride 参数
    if stride is None:
        # 使用 torch.jit.annotate 声明一个空的整数列表
        stride = torch.jit.annotate(List[int], [])
    
    # 调用底层 C 函数执行最大池化操作，返回结果
    return torch._C._nn.max_pool2d_with_indices(
        input, kernel_size, stride, padding, dilation, ceil_mode
    )
# 根据是否具有 torch 函数的一元操作来决定返回处理后的函数结果
if has_torch_function_unary(input):
    return handle_torch_function(
        max_pool2d,  # 调用处理 torch 函数的方法
        (input,),  # 传入输入参数元组
        input,  # 输入张量
        kernel_size,  # 池化核大小
        stride=stride,  # 池化步长
        padding=padding,  # 填充
        dilation=dilation,  # 扩张率
        ceil_mode=ceil_mode,  # 是否使用向上取整模式
        return_indices=return_indices,  # 是否返回池化结果索引
    )

# 如果未提供池化的步长参数，则设置为空列表
if stride is None:
    stride = torch.jit.annotate(List[int], [])

# 调用 torch 自带的最大池化函数 torch.max_pool2d
return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
    # 检查输入是否已经实现了 torch_function_unary 方法，如果是，则调用 torch 函数处理
    if has_torch_function_unary(input):
        # 调用处理 torch 函数的方法，传入 max_pool3d_with_indices 函数和参数
        return handle_torch_function(
            max_pool3d_with_indices,
            (input,),
            input,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            return_indices=return_indices,
        )
    
    # 如果未提供 stride 参数，则使用空的 List[int] 类型对象进行注释
    if stride is None:
        stride = torch.jit.annotate(List[int], [])
    
    # 调用底层 C++ 函数实现的 max_pool3d_with_indices 方法，进行最大池化操作
    return torch._C._nn.max_pool3d_with_indices(
        input, kernel_size, stride, padding, dilation, ceil_mode
    )
def _max_pool3d(
    input: Tensor,
    kernel_size: BroadcastingList3[int],
    stride: Optional[BroadcastingList3[int]] = None,
    padding: BroadcastingList3[int] = 0,
    dilation: BroadcastingList3[int] = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Tensor:
    # 检查输入是否具有 Torch 函数的一元操作，如果是，则调用 Torch 函数处理
    if has_torch_function_unary(input):
        return handle_torch_function(
            max_pool3d,
            (input,),
            input,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            return_indices=return_indices,
        )
    # 如果未指定步长，则初始化为空列表
    if stride is None:
        stride = torch.jit.annotate(List[int], [])
    # 调用 Torch 提供的最大池化函数 max_pool3d
    return torch.max_pool3d(input, kernel_size, stride, padding, dilation, ceil_mode)


max_pool3d = boolean_dispatch(
    arg_name="return_indices",
    arg_index=6,
    default=False,
    if_true=max_pool3d_with_indices,
    if_false=_max_pool3d,
    module_name=__name__,
    func_name="max_pool3d",
)


def _unpool_output_size(
    input: Tensor,
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
    output_size: Optional[List[int]],
) -> List[int]:
    # 获取输入张量的尺寸
    input_size = input.size()
    # 初始化默认大小列表
    default_size = torch.jit.annotate(List[int], [])
    # 根据卷积核大小、步长和填充计算输出的默认大小
    for d in range(len(kernel_size)):
        default_size.append(
            (input_size[-len(kernel_size) + d] - 1) * stride[d]
            + kernel_size[d]
            - 2 * padding[d]
        )
    # 如果未提供输出大小，则返回默认大小
    if output_size is None:
        ret = default_size
    else:
        # 如果提供了输出大小，根据情况进行验证
        if len(output_size) == len(kernel_size) + 2:
            output_size = output_size[2:]
        # 检查输出大小与核大小的一致性
        if len(output_size) != len(kernel_size):
            raise ValueError(
                "output_size should be a sequence containing "
                f"{len(kernel_size)} or {len(kernel_size) + 2} elements, but it has a length of '{len(output_size)}'"
            )
        # 检查输出大小是否在合理范围内
        for d in range(len(kernel_size)):
            min_size = default_size[d] - stride[d]
            max_size = default_size[d] + stride[d]
            if not (min_size < output_size[d] < max_size):
                raise ValueError(
                    f'invalid output_size "{output_size}" (dim {d} must be between {min_size} and {max_size})'
                )

        ret = output_size
    return ret


def max_unpool1d(
    input: Tensor,
    indices: Tensor,
    kernel_size: BroadcastingList1[int],
    stride: Optional[BroadcastingList1[int]] = None,
    padding: BroadcastingList1[int] = 0,
    output_size: Optional[BroadcastingList1[int]] = None,
) -> Tensor:
    r"""Compute a partial inverse of :class:`MaxPool1d`.

    See :class:`~torch.nn.MaxUnpool1d` for details.
    """
    # 检查输入是否具有 Torch 函数的一元操作，如果是，则调用 Torch 函数处理
    if has_torch_function_unary(input):
        return handle_torch_function(
            max_unpool1d,
            (input,),
            input,
            indices,
            kernel_size,
            stride=stride,
            padding=padding,
            output_size=output_size,
        )
    # 将 kernel_size 转换为一维列表，确保统一处理方式
    kernel_size = _single(kernel_size)
    
    # 如果指定了 stride，则将其转换为一维列表；否则使用 kernel_size 的值作为默认值
    if stride is not None:
        _stride = _single(stride)
    else:
        _stride = kernel_size
    
    # 将 padding 参数转换为一维列表，确保统一处理方式
    padding = _single(padding)
    
    # 根据输入、核大小、步长、填充和输出大小计算输出尺寸
    output_size = _unpool_output_size(input, kernel_size, _stride, padding, output_size)
    
    # 如果输出大小是列表，则添加额外的维度 1；否则添加额外的元组维度 (1,)
    if isinstance(output_size, list):
        output_size = output_size + [1]
    else:
        output_size = output_size + (1,)
    
    # 调用 PyTorch C++ 扩展函数 max_unpool2d 对输入进行最大解池操作
    # 将输入、索引和计算的输出大小作为参数传入，返回解池后的结果
    return torch._C._nn.max_unpool2d(
        input.unsqueeze(-1), indices.unsqueeze(-1), output_size
    ).squeeze(-1)
# 定义函数 max_unpool2d，用于执行二维最大池化的部分反操作
def max_unpool2d(
    input: Tensor,
    indices: Tensor,
    kernel_size: BroadcastingList2[int],
    stride: Optional[BroadcastingList2[int]] = None,
    padding: BroadcastingList2[int] = 0,
    output_size: Optional[BroadcastingList2[int]] = None,
) -> Tensor:
    r"""Compute a partial inverse of :class:`MaxPool2d`.

    See :class:`~torch.nn.MaxUnpool2d` for details.
    """
    # 如果输入具有 Torch 函数，处理 Torch 函数操作
    if has_torch_function_unary(input):
        return handle_torch_function(
            max_unpool2d,
            (input,),
            input,
            indices,
            kernel_size,
            stride=stride,
            padding=padding,
            output_size=output_size,
        )
    # 转换 kernel_size 为二维元组
    kernel_size = _pair(kernel_size)
    # 如果指定了 stride，则转换为二维元组；否则使用 kernel_size
    if stride is not None:
        _stride = _pair(stride)
    else:
        _stride = kernel_size
    # 转换 padding 为二维元组
    padding = _pair(padding)
    # 计算输出尺寸
    output_size = _unpool_output_size(input, kernel_size, _stride, padding, output_size)
    # 调用 Torch C++ 前端函数执行最大池化的反操作
    return torch._C._nn.max_unpool2d(input, indices, output_size)


# 定义函数 max_unpool3d，用于执行三维最大池化的部分反操作
def max_unpool3d(
    input: Tensor,
    indices: Tensor,
    kernel_size: BroadcastingList3[int],
    stride: Optional[BroadcastingList3[int]] = None,
    padding: BroadcastingList3[int] = 0,
    output_size: Optional[BroadcastingList3[int]] = None,
) -> Tensor:
    r"""Compute a partial inverse of :class:`MaxPool3d`.

    See :class:`~torch.nn.MaxUnpool3d` for details.
    """
    # 如果输入具有 Torch 函数，处理 Torch 函数操作
    if has_torch_function_unary(input):
        return handle_torch_function(
            max_unpool3d,
            (input,),
            input,
            indices,
            kernel_size,
            stride=stride,
            padding=padding,
            output_size=output_size,
        )
    # 转换 kernel_size 为三维元组
    kernel_size = _triple(kernel_size)
    # 如果指定了 stride，则转换为三维元组；否则使用 kernel_size
    if stride is not None:
        _stride = _triple(stride)
    else:
        _stride = kernel_size
    # 转换 padding 为三维元组
    padding = _triple(padding)
    # 计算输出尺寸
    output_size = _unpool_output_size(input, kernel_size, _stride, padding, output_size)
    # 调用 Torch C++ 前端函数执行最大池化的反操作
    return torch._C._nn.max_unpool3d(input, indices, output_size, _stride, padding)


# 定义函数 lp_pool3d，用于执行三维 Lp 池化
def lp_pool3d(
    input: Tensor,
    norm_type: Union[int, float],
    kernel_size: BroadcastingList3[int],
    stride: Optional[BroadcastingList3[int]] = None,
    ceil_mode: bool = False,
) -> Tensor:
    r"""
    Apply a 3D power-average pooling over an input signal composed of several input planes.

    If the sum of all inputs to the power of `p` is
    zero, the gradient is set to zero as well.

    See :class:`~torch.nn.LPPool3d` for details.
    """
    # 如果输入具有 Torch 函数，处理 Torch 函数操作
    if has_torch_function_unary(input):
        return handle_torch_function(
            lp_pool3d,
            (input,),
            input,
            norm_type,
            kernel_size,
            stride=stride,
            ceil_mode=ceil_mode,
        )
    # 转换 kernel_size 为三维元组
    kd, kw, kh = _triple(kernel_size)
    # 如果指定了 stride，则应用三维元组；否则使用默认
    if stride is not None:
        out = avg_pool3d(input.pow(norm_type), kernel_size, stride, 0, ceil_mode)
    else:
        # 对输入进行幂运算，使用给定的规范化类型
        out = avg_pool3d(
            input.pow(norm_type), kernel_size, padding=0, ceil_mode=ceil_mode
        )

    # 对输出进行符号函数和修正线性单元操作，然后乘以核大小的乘积再进行幂运算
    return (
        (torch.sign(out) * relu(torch.abs(out))).mul(kd * kw * kh).pow(1.0 / norm_type)
    )
# 定义一个函数 lp_pool2d，接受以下参数：输入张量 input，规范类型 norm_type（可以是 int 或 float），卷积核大小 kernel_size，可选的步长 stride，默认为 None，是否向上取整 ceil_mode，默认为 False，返回值为张量。
def lp_pool2d(
    input: Tensor,
    norm_type: Union[int, float],
    kernel_size: BroadcastingList2[int],
    stride: Optional[BroadcastingList2[int]] = None,
    ceil_mode: bool = False,
) -> Tensor:
    r"""
    Apply a 2D power-average pooling over an input signal composed of several input planes.

    If the sum of all inputs to the power of `p` is
    zero, the gradient is set to zero as well.

    See :class:`~torch.nn.LPPool2d` for details.
    """
    # 如果输入张量 input 具有 Torch 函数的一元操作，调用 handle_torch_function 处理
    if has_torch_function_unary(input):
        return handle_torch_function(
            lp_pool2d,
            (input,),
            input,
            norm_type,
            kernel_size,
            stride=stride,
            ceil_mode=ceil_mode,
        )
    
    # 解析 kernel_size 到 kw, kh 两个变量中
    kw, kh = _pair(kernel_size)
    
    # 如果指定了步长 stride
    if stride is not None:
        # 对输入张量 input 的 norm_type 次幂应用平均池化，使用 kernel_size 和 stride 进行池化操作，ceil_mode 为 0
        out = avg_pool2d(input.pow(norm_type), kernel_size, stride, 0, ceil_mode)
    else:
        # 否则，对输入张量 input 的 norm_type 次幂应用平均池化，使用 kernel_size 进行池化操作，padding 为 0，ceil_mode 为 ceil_mode
        out = avg_pool2d(
            input.pow(norm_type), kernel_size, padding=0, ceil_mode=ceil_mode
        )

    # 返回对池化结果 out 进行符号函数、ReLU 函数和乘法操作，乘数为 kw * kh，然后取 norm_type 次方根的结果
    return (torch.sign(out) * relu(torch.abs(out))).mul(kw * kh).pow(1.0 / norm_type)


# 定义一个函数 lp_pool1d，接受以下参数：输入张量 input，规范类型 norm_type（可以是 int 或 float），卷积核大小 kernel_size，可选的步长 stride，默认为 None，是否向上取整 ceil_mode，默认为 False，返回值为张量。
def lp_pool1d(
    input: Tensor,
    norm_type: Union[int, float],
    kernel_size: int,
    stride: Optional[BroadcastingList1[int]] = None,
    ceil_mode: bool = False,
) -> Tensor:
    r"""Apply a 1D power-average pooling over an input signal composed of several input planes.

    If the sum of all inputs to the power of `p` is
    zero, the gradient is set to zero as well.

    See :class:`~torch.nn.LPPool1d` for details.
    """
    # 如果输入张量 input 具有 Torch 函数的一元操作，调用 handle_torch_function 处理
    if has_torch_function_unary(input):
        return handle_torch_function(
            lp_pool1d,
            (input,),
            input,
            norm_type,
            kernel_size,
            stride=stride,
            ceil_mode=ceil_mode,
        )
    
    # 如果指定了步长 stride
    if stride is not None:
        # 对输入张量 input 的 norm_type 次幂应用平均池化，使用 kernel_size 和 stride 进行池化操作，ceil_mode 为 0
        out = avg_pool1d(input.pow(norm_type), kernel_size, stride, 0, ceil_mode)
    else:
        # 否则，对输入张量 input 的 norm_type 次幂应用平均池化，使用 kernel_size 进行池化操作，padding 为 0，ceil_mode 为 ceil_mode
        out = avg_pool1d(
            input.pow(norm_type), kernel_size, padding=0, ceil_mode=ceil_mode
        )

    # 返回对池化结果 out 进行符号函数、ReLU 函数和乘法操作，乘数为 kernel_size，然后取 norm_type 次方根的结果
    return (
        (torch.sign(out) * relu(torch.abs(out))).mul(kernel_size).pow(1.0 / norm_type)
    )


# 定义一个函数 adaptive_max_pool1d_with_indices，接受以下参数：输入张量 input，目标输出大小 output_size，是否返回池化索引 return_indices，默认为 False，返回值为元组（张量，张量）。
def adaptive_max_pool1d_with_indices(
    input: Tensor,
    output_size: BroadcastingList1[int],
    return_indices: bool = False,
) -> Tuple[Tensor, Tensor]:  # noqa: D400
    r"""
    adaptive_max_pool1d(input, output_size, return_indices=False)

    Applies a 1D adaptive max pooling over an input signal composed of
    several input planes.

    See :class:`~torch.nn.AdaptiveMaxPool1d` for details and output shape.

    Args:
        output_size: the target output size (single integer)
        return_indices: whether to return pooling indices. Default: ``False``
    """
    # 如果输入张量 input 具有 Torch 函数的一元操作，调用 handle_torch_function 处理
    if has_torch_function_unary(input):
        return handle_torch_function(
            adaptive_max_pool1d_with_indices,
            (input,),
            input,
            output_size,
            return_indices=return_indices,
        )
    # 使用 PyTorch 的 adaptive_max_pool1d 函数对输入进行自适应最大池化操作，将其调整为指定的输出大小后返回结果
    return torch.adaptive_max_pool1d(input, output_size)
# 定义一个函数，对输入的一维张量进行自适应最大池化操作
def _adaptive_max_pool1d(
    input: Tensor,
    output_size: BroadcastingList1[int],
    return_indices: bool = False,
) -> Tensor:
    # 如果输入具有torch函数，调用torch函数处理
    if has_torch_function_unary(input):
        return handle_torch_function(
            adaptive_max_pool1d,
            (input,),  # 输入参数元组
            input,
            output_size,
            return_indices=return_indices,
        )
    # 否则调用自定义的自适应最大池化函数，返回结果的第一个元素
    return adaptive_max_pool1d_with_indices(input, output_size)[0]


# 创建一个布尔分发，根据输入参数决定调用哪个自适应最大池化函数
adaptive_max_pool1d = boolean_dispatch(
    arg_name="return_indices",  # 参数名称
    arg_index=2,  # 参数索引
    default=False,  # 默认值
    if_true=adaptive_max_pool1d_with_indices,  # 如果为True时调用的函数
    if_false=_adaptive_max_pool1d,  # 如果为False时调用的函数
    module_name=__name__,  # 模块名称
    func_name="adaptive_max_pool1d",  # 函数名称
)


# 定义一个函数，对输入的二维张量进行自适应最大池化操作并返回池化结果和索引
def adaptive_max_pool2d_with_indices(
    input: Tensor,
    output_size: BroadcastingList2[int],
    return_indices: bool = False,
) -> Tuple[Tensor, Tensor]:  # noqa: D400
    r"""adaptive_max_pool2d(input, output_size, return_indices=False)

    对由多个输入平面组成的输入信号应用二维自适应最大池化操作。

    查看:class:`~torch.nn.AdaptiveMaxPool2d`了解详细信息和输出形状。

    Args:
        output_size: 目标输出大小（单个整数或双整数元组）
        return_indices: 是否返回池化索引。默认为``False``
    """
    # 如果输入具有torch函数，调用torch函数处理
    if has_torch_function_unary(input):
        return handle_torch_function(
            adaptive_max_pool2d_with_indices,
            (input,),  # 输入参数元组
            input,
            output_size,
            return_indices=return_indices,
        )
    # 否则，根据默认的输出大小和输入的尺寸调用C++扩展实现的自适应最大池化函数
    output_size = _list_with_default(output_size, input.size())
    return torch._C._nn.adaptive_max_pool2d(input, output_size)


# 定义一个函数，对输入的二维张量进行自适应最大池化操作并返回池化结果
def _adaptive_max_pool2d(
    input: Tensor,
    output_size: BroadcastingList2[int],
    return_indices: bool = False,
) -> Tensor:
    # 如果输入具有torch函数，调用torch函数处理
    if has_torch_function_unary(input):
        return handle_torch_function(
            adaptive_max_pool2d,
            (input,),  # 输入参数元组
            input,
            output_size,
            return_indices=return_indices,
        )
    # 否则调用自定义的自适应最大池化函数，返回结果的第一个元素
    return adaptive_max_pool2d_with_indices(input, output_size)[0]


# 创建一个布尔分发，根据输入参数决定调用哪个自适应最大池化函数
adaptive_max_pool2d = boolean_dispatch(
    arg_name="return_indices",  # 参数名称
    arg_index=2,  # 参数索引
    default=False,  # 默认值
    if_true=adaptive_max_pool2d_with_indices,  # 如果为True时调用的函数
    if_false=_adaptive_max_pool2d,  # 如果为False时调用的函数
    module_name=__name__,  # 模块名称
    func_name="adaptive_max_pool2d",  # 函数名称
)


# 定义一个函数，对输入的三维张量进行自适应最大池化操作并返回池化结果和索引
def adaptive_max_pool3d_with_indices(
    input: Tensor,
    output_size: BroadcastingList3[int],
    return_indices: bool = False,
) -> Tuple[Tensor, Tensor]:  # noqa: D400
    r"""
    adaptive_max_pool3d(input, output_size, return_indices=False)

    对由多个输入平面组成的输入信号应用三维自适应最大池化操作。

    查看:class:`~torch.nn.AdaptiveMaxPool3d`了解详细信息和输出形状。

    Args:
        output_size: 目标输出大小（单个整数或三整数元组）
        return_indices: 是否返回池化索引。默认为``False``
    """
    # 如果输入的张量具有 torch 函数的单目处理能力
    if has_torch_function_unary(input):
        # 调用处理 torch 函数的方法，执行自适应三维最大池化操作
        return handle_torch_function(
            adaptive_max_pool3d_with_indices,  # 处理的 torch 函数为自适应最大池化
            (input,),  # 将输入作为参数元组传递
            input,  # 输入张量
            output_size,  # 输出尺寸参数
            return_indices=return_indices,  # 是否返回索引的标志
        )
    
    # 如果没有 torch 函数的单目处理能力，则进行以下操作
    output_size = _list_with_default(output_size, input.size())
    # 调用 torch 库中的 C++ 扩展方法进行自适应三维最大池化操作，并返回结果
    return torch._C._nn.adaptive_max_pool3d(input, output_size)
# 定义一个三维自适应最大池化函数，对输入进行池化操作，并返回池化后的结果张量
def _adaptive_max_pool3d(
    input: Tensor,
    output_size: BroadcastingList3[int],
    return_indices: bool = False,
) -> Tensor:
    # 如果输入对象有自定义的Torch函数，使用Torch函数处理自适应最大池化操作
    if has_torch_function_unary(input):
        return handle_torch_function(
            adaptive_max_pool3d,
            (input,),
            input,
            output_size,
            return_indices=return_indices,
        )
    # 否则，调用内置的自适应最大池化函数进行操作，并返回池化后的结果张量
    return adaptive_max_pool3d_with_indices(input, output_size)[0]


# 创建一个自适应最大池化的分发函数，根据参数`return_indices`的布尔值分发到对应的处理函数
adaptive_max_pool3d = boolean_dispatch(
    arg_name="return_indices",
    arg_index=2,
    default=False,
    if_true=adaptive_max_pool3d_with_indices,
    if_false=_adaptive_max_pool3d,
    module_name=__name__,
    func_name="adaptive_max_pool3d",
)


# 添加文档字符串给`adaptive_avg_pool1d`函数，描述其功能和参数信息
adaptive_avg_pool1d = _add_docstr(
    torch.adaptive_avg_pool1d,
    r"""
adaptive_avg_pool1d(input, output_size) -> Tensor

Applies a 1D adaptive average pooling over an input signal composed of
several input planes.

See :class:`~torch.nn.AdaptiveAvgPool1d` for details and output shape.

Args:
    output_size: the target output size (single integer)
""",
)


# 定义一个二维自适应平均池化函数，对输入进行池化操作，并返回池化后的结果张量
def adaptive_avg_pool2d(input: Tensor, output_size: BroadcastingList2[int]) -> Tensor:
    r"""Apply a 2D adaptive average pooling over an input signal composed of several input planes.

    See :class:`~torch.nn.AdaptiveAvgPool2d` for details and output shape.

    Args:
        output_size: the target output size (single integer or
            double-integer tuple)
    """
    # 如果输入对象有自定义的Torch函数，使用Torch函数处理自适应平均池化操作
    if has_torch_function_unary(input):
        return handle_torch_function(adaptive_avg_pool2d, (input,), input, output_size)
    # 否则，根据给定的输出大小进行自适应平均池化操作，并返回池化后的结果张量
    _output_size = _list_with_default(output_size, input.size())
    return torch._C._nn.adaptive_avg_pool2d(input, _output_size)


# 定义一个三维自适应平均池化函数，对输入进行池化操作，并返回池化后的结果张量
def adaptive_avg_pool3d(input: Tensor, output_size: BroadcastingList3[int]) -> Tensor:
    r"""Apply a 3D adaptive average pooling over an input signal composed of several input planes.

    See :class:`~torch.nn.AdaptiveAvgPool3d` for details and output shape.

    Args:
        output_size: the target output size (single integer or
            triple-integer tuple)
    """
    # 如果输入对象有自定义的Torch函数，使用Torch函数处理自适应平均池化操作
    if has_torch_function_unary(input):
        return handle_torch_function(adaptive_avg_pool3d, (input,), input, output_size)
    # 否则，根据给定的输出大小进行自适应平均池化操作，并返回池化后的结果张量
    _output_size = _list_with_default(output_size, input.size())
    return torch._C._nn.adaptive_avg_pool3d(input, _output_size)


# 定义一个dropout函数，对输入进行dropout操作，在训练过程中随机将部分元素置为零
def dropout(
    input: Tensor,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
) -> Tensor:
    r"""During training, randomly zeroes some elements of the input tensor with probability :attr:`p`.

    Uses samples from a Bernoulli distribution.

    See :class:`~torch.nn.Dropout` for details.

    Args:
        p: probability of an element to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """
    # 如果输入参数具有 torch_function 的一元方法，则调用处理 torch_function 的函数
    if has_torch_function_unary(input):
        # 调用 handle_torch_function 处理 dropout 函数的 torch_function
        return handle_torch_function(
            dropout, (input,), input, p=p, training=training, inplace=inplace
        )
    
    # 如果 dropout 概率 p 不在 [0, 1] 的范围内，抛出数值错误异常
    if p < 0.0 or p > 1.0:
        raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
    
    # 如果 inplace 标志为 True，则调用 _VF.dropout_ 执行原地 dropout 操作，否则调用 _VF.dropout 执行非原地操作
    return (
        _VF.dropout_(input, p, training) if inplace else _VF.dropout(input, p, training)
    )
# 应用 alpha dropout 到输入张量上
def alpha_dropout(
    input: Tensor,
    p: float = 0.5,
    training: bool = False,
    inplace: bool = False,
) -> Tensor:
    # 如果输入张量有 Torch 函数的重载，使用处理 Torch 函数的方法
    if has_torch_function_unary(input):
        return handle_torch_function(
            alpha_dropout, (input,), input, p=p, training=training, inplace=inplace
        )
    # 如果 dropout 概率不在 [0, 1] 之间，抛出 ValueError
    if p < 0.0 or p > 1.0:
        raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
    # 根据 inplace 参数选择执行 alpha dropout 的底层函数
    return (
        _VF.alpha_dropout_(input, p, training)
        if inplace
        else _VF.alpha_dropout(input, p, training)
    )


# 对 1 维输入执行 dropout 操作
def dropout1d(
    input: Tensor,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
) -> Tensor:
    # 如果输入张量有 Torch 函数的重载，使用处理 Torch 函数的方法
    if has_torch_function_unary(input):
        return handle_torch_function(
            dropout1d, (input,), input, p=p, training=training, inplace=inplace
        )
    # 如果 dropout 概率不在 [0, 1] 之间，抛出 ValueError
    if p < 0.0 or p > 1.0:
        raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
    # 获取输入张量的维度
    inp_dim = input.dim()
    # 如果输入张量不是 2 维或 3 维，则抛出 RuntimeError
    if inp_dim not in (2, 3):
        raise RuntimeError(
            f"dropout1d: Expected 2D or 3D input, but received a {inp_dim}D input. "
            "Note that dropout1d exists to provide channel-wise dropout on inputs with 1 "
            "spatial dimension, a channel dimension, and an optional batch dimension "
            "(i.e. 2D or 3D inputs)."
        )

    # 检查输入张量是否是批处理的
    is_batched = inp_dim == 3
    # 如果不是批处理的，根据 inplace 参数选择是否在输入张量上添加维度
    if not is_batched:
        input = input.unsqueeze_(0) if inplace else input.unsqueeze(0)

    # 根据 inplace 参数选择执行 feature dropout 的底层函数
    result = (
        _VF.feature_dropout_(input, p, training)
        if inplace
        else _VF.feature_dropout(input, p, training)
    )

    # 如果不是批处理的，根据 inplace 参数选择是否在结果张量上去掉添加的维度
    if not is_batched:
        result = result.squeeze_(0) if inplace else result.squeeze(0)

    return result


# 对 2 维输入执行 dropout 操作
def dropout2d(
    input: Tensor,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
) -> Tensor:
    # 应用 alpha dropout 到输入张量上
    r"""Randomly zero out entire channels (a channel is a 2D feature map).

    For example, the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 2D tensor :math:`\text{input}[i, j]` of the input tensor.
    Each channel will be zeroed out independently on every forward call with
    ```
    # 如果输入具有 Torch 函数，调用 Torch 函数处理
    if has_torch_function_unary(input):
        return handle_torch_function(
            dropout2d, (input,), input, p=p, training=training, inplace=inplace
        )
    # 检查 dropout 概率 p 是否在合理范围内 (0 到 1 之间)
    if p < 0.0 or p > 1.0:
        raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
    # 获取输入的维度
    inp_dim = input.dim()
    # 如果输入维度不是 3 或 4，发出警告信息，这在将来版本中会报错
    if inp_dim not in (3, 4):
        warn_msg = (
            f"dropout2d: Received a {inp_dim}-D input to dropout2d, which is deprecated "
            "and will result in an error in a future release. To retain the behavior "
            "and silence this warning, please use dropout instead. Note that dropout2d "
            "exists to provide channel-wise dropout on inputs with 2 spatial dimensions, "
            "a channel dimension, and an optional batch dimension (i.e. 3D or 4D inputs)."
        )
        warnings.warn(warn_msg)

    # TODO: Properly support no-batch-dim inputs. For now, these are NOT supported; passing
    # a 3D input will perform dropout1d behavior instead. This was done historically and the
    # behavior is maintained here for now.
    # See https://github.com/pytorch/pytorch/issues/77081
    # 如果输入维度为 3，发出警告信息，表明将来版本将修改行为以支持无批次维度的输入
    if inp_dim == 3:
        warnings.warn(
            "dropout2d: Received a 3D input to dropout2d and assuming that channel-wise "
            "1D dropout behavior is desired - input is interpreted as shape (N, C, L), where C "
            "is the channel dim. This behavior will change in a future release to interpret the "
            "input as one without a batch dimension, i.e. shape (C, H, W). To maintain the 1D "
            "channel-wise dropout behavior, please switch to using dropout1d instead."
        )

    # 根据 inplace 参数选择调用 _VF.feature_dropout_ 或 _VF.feature_dropout 函数执行 dropout 操作
    result = (
        _VF.feature_dropout_(input, p, training)
        if inplace
        else _VF.feature_dropout(input, p, training)
    )

    # 返回 dropout 后的结果
    return result
# 随机将整个通道（即一个3D特征图）置零。

def dropout3d(
    input: Tensor,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
) -> Tensor:
    r"""Randomly zero out entire channels (a channel is a 3D feature map).

    For example, the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 3D tensor :math:`\text{input}[i, j]` of the input tensor.
    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    See :class:`~torch.nn.Dropout3d` for details.

    Args:
        p: probability of a channel to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """

    # 如果输入有 torch 函数处理，则调用相应的处理函数
    if has_torch_function_unary(input):
        return handle_torch_function(
            dropout3d, (input,), input, p=p, training=training, inplace=inplace
        )

    # 如果 p 不在 [0, 1] 范围内，则抛出值错误异常
    if p < 0.0 or p > 1.0:
        raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")

    # 获取输入的维度
    inp_dim = input.dim()

    # 如果输入维度不是 4 或 5，则发出警告
    if inp_dim not in (4, 5):
        warn_msg = (
            f"dropout3d: Received a {inp_dim}-D input to dropout3d, which is deprecated "
            "and will result in an error in a future release. To retain the behavior "
            "and silence this warning, please use dropout instead. Note that dropout3d "
            "exists to provide channel-wise dropout on inputs with 3 spatial dimensions, "
            "a channel dimension, and an optional batch dimension (i.e. 4D or 5D inputs)."
        )
        warnings.warn(warn_msg)

    # 检查是否有批次维度，如果没有则添加批次维度
    is_batched = inp_dim == 5
    if not is_batched:
        input = input.unsqueeze_(0) if inplace else input.unsqueeze(0)

    # 使用 _VF.feature_dropout_ 或 _VF.feature_dropout 执行特征丢弃操作
    result = (
        _VF.feature_dropout_(input, p, training)
        if inplace
        else _VF.feature_dropout(input, p, training)
    )

    # 如果没有批次维度，则去除添加的批次维度
    if not is_batched:
        result = result.squeeze_(0) if inplace else result.squeeze(0)
    return result


def feature_alpha_dropout(
    input: Tensor,
    p: float = 0.5,
    training: bool = False,
    inplace: bool = False,
) -> Tensor:
    r"""Randomly masks out entire channels (a channel is a feature map).

    For example, the :math:`j`-th channel of the :math:`i`-th sample in the batch input
    is a tensor :math:`\text{input}[i, j]` of the input tensor. Instead of
    setting activations to zero, as in regular Dropout, the activations are set
    to the negative saturation value of the SELU activation function.

    Each element will be masked independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.
    The elements to be masked are randomized on every forward call, and scaled
    and shifted to maintain zero mean and unit variance.

    See :class:`~torch.nn.FeatureAlphaDropout` for details.
    def feature_alpha_dropout(input, p=0.5, training=True, inplace=False):
        """
        Applies feature alpha dropout to the input tensor.
    
        Args:
            p: dropout probability of a channel to be zeroed. Default: 0.5
            training: apply dropout if is ``True``. Default: ``True``
            inplace: If set to ``True``, will do this operation in-place. Default: ``False``
        """
        # 如果输入具有单目的 Torch 函数，调用 Torch 函数处理
        if has_torch_function_unary(input):
            return handle_torch_function(
                feature_alpha_dropout,
                (input,),
                input,
                p=p,
                training=training,
                inplace=inplace,
            )
        # 如果概率 p 超出范围 [0, 1]，抛出值错误异常
        if p < 0.0 or p > 1.0:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        # 根据 inplace 参数选择性地应用 feature alpha dropout 操作
        return (
            _VF.feature_alpha_dropout_(input, p, training)
            if inplace
            else _VF.feature_alpha_dropout(input, p, training)
        )
# 定义了一个函数 _threshold，它对输入的 Tensor 的每个元素应用阈值操作
def _threshold(
    input: Tensor,
    threshold: float,
    value: float,
    inplace: bool = False,
) -> Tensor:
    r"""Apply a threshold to each element of the input Tensor.

    See :class:`~torch.nn.Threshold` for more details.
    """
    # 如果输入的 input 具有 torch 函数的重载机制，则调用处理函数
    if has_torch_function_unary(input):
        return handle_torch_function(
            _threshold, (input,), input, threshold, value, inplace=inplace
        )
    # 如果 inplace 参数为 True，则调用 _VF.threshold_ 实现就地修改
    if inplace:
        result = _VF.threshold_(input, threshold, value)
    else:
        # 否则调用 _VF.threshold 实现创建新的 Tensor
        result = _VF.threshold(input, threshold, value)
    # 返回处理后的 Tensor
    return result


# 将 _threshold 函数赋值给 threshold 变量，以避免 threshold 参数对 __torch_function__ 的递归引用问题
threshold = _threshold

# 定义了一个函数 threshold_，用于给 _VF.threshold_ 函数添加文档字符串
threshold_ = _add_docstr(
    _VF.threshold_,
    r"""
threshold_(input, threshold, value) -> Tensor

In-place version of :func:`~threshold`.
""",
)


# 定义了一个函数 relu，实现对输入 Tensor 的每个元素应用 ReLU 函数
def relu(input: Tensor, inplace: bool = False) -> Tensor:  # noqa: D400,D402
    r"""relu(input, inplace=False) -> Tensor

    Applies the rectified linear unit function element-wise. See
    :class:`~torch.nn.ReLU` for more details.
    """
    # 如果输入的 input 具有 torch 函数的重载机制，则调用处理函数
    if has_torch_function_unary(input):
        return handle_torch_function(relu, (input,), input, inplace=inplace)
    # 如果 inplace 参数为 True，则调用 torch.relu_ 实现就地修改
    if inplace:
        result = torch.relu_(input)
    else:
        # 否则调用 torch.relu 实现创建新的 Tensor
        result = torch.relu(input)
    # 返回处理后的 Tensor
    return result


# 将 torch.relu_ 函数赋值给 relu_ 变量，为其添加文档字符串
relu_ = _add_docstr(
    torch.relu_,
    r"""
relu_(input) -> Tensor

In-place version of :func:`~relu`.
""",
)


# 定义了一个函数 glu，实现对输入 Tensor 在指定维度上进行 GLU 操作
def glu(input: Tensor, dim: int = -1) -> Tensor:  # noqa: D400,D402
    r"""
    glu(input, dim=-1) -> Tensor

    The gated linear unit. Computes:

    .. math ::
        \text{GLU}(a, b) = a \otimes \sigma(b)

    where `input` is split in half along `dim` to form `a` and `b`, :math:`\sigma`
    is the sigmoid function and :math:`\otimes` is the element-wise product between matrices.

    See `Language Modeling with Gated Convolutional Networks <https://arxiv.org/abs/1612.08083>`_.

    Args:
        input (Tensor): input tensor
        dim (int): dimension on which to split the input. Default: -1
    """
    # 如果输入的 input 具有 torch 函数的重载机制，则调用处理函数
    if has_torch_function_unary(input):
        return handle_torch_function(glu, (input,), input, dim=dim)
    # 如果 input 的维度为 0，则抛出异常，因为 glu 不支持标量
    if input.dim() == 0:
        raise RuntimeError(
            "glu does not support scalars because halving size must be even"
        )
    # 调用 torch._C._nn.glu 实现 GLU 操作
    return torch._C._nn.glu(input, dim)


# 定义了一个函数 hardtanh，实现对输入 Tensor 的每个元素应用 HardTanh 函数
def hardtanh(
    input: Tensor,
    min_val: float = -1.0,
    max_val: float = 1.0,
    inplace: bool = False,
) -> Tensor:  # noqa: D400,D402
    r"""
    hardtanh(input, min_val=-1., max_val=1., inplace=False) -> Tensor

    Applies the HardTanh function element-wise. See :class:`~torch.nn.Hardtanh` for more
    details.
    """
    # 如果输入的 input 具有 torch 函数的重载机制，则调用处理函数
    if has_torch_function_unary(input):
        return handle_torch_function(
            hardtanh, (input,), input, min_val=min_val, max_val=max_val, inplace=inplace
        )
    # 检查最小值是否大于最大值，若是则抛出数值错误异常
    if min_val > max_val:
        raise ValueError("min_val cannot be greater than max_val")
    
    # 如果 inplace 参数为 True，则调用 PyTorch 库中的原位修改版本 hardtanh_
    if inplace:
        result = torch._C._nn.hardtanh_(input, min_val, max_val)
    else:
        # 如果 inplace 参数为 False，则调用 PyTorch 库中的非原位修改版本 hardtanh
        result = torch._C._nn.hardtanh(input, min_val, max_val)
    
    # 返回计算结果
    return result
# 添加文档字符串到 torch._C._nn.hardtanh_ 函数
hardtanh_ = _add_docstr(
    torch._C._nn.hardtanh_,
    r"""
hardtanh_(input, min_val=-1., max_val=1.) -> Tensor

In-place version of :func:`~hardtanh`.
""",
)


def relu6(input: Tensor, inplace: bool = False) -> Tensor:  # noqa: D400,D402
    r"""relu6(input, inplace=False) -> Tensor

    Applies the element-wise function :math:`\text{ReLU6}(x) = \min(\max(0,x), 6)`.

    See :class:`~torch.nn.ReLU6` for more details.
    """
    # 如果 input 支持 torch 函数式操作，则调用对应的处理函数
    if has_torch_function_unary(input):
        return handle_torch_function(relu6, (input,), input, inplace=inplace)
    # 如果 inplace=True，则调用 torch._C._nn.relu6_ 函数进行原地操作
    if inplace:
        result = torch._C._nn.relu6_(input)
    else:
        # 否则调用 torch._C._nn.relu6 函数
        result = torch._C._nn.relu6(input)
    return result


def elu(input: Tensor, alpha: float = 1.0, inplace: bool = False) -> Tensor:
    r"""Apply the Exponential Linear Unit (ELU) function element-wise.

    See :class:`~torch.nn.ELU` for more details.
    """
    # 如果 input 支持 torch 函数式操作，则调用对应的处理函数
    if has_torch_function_unary(input):
        return handle_torch_function(elu, (input,), input, alpha=alpha, inplace=inplace)
    # 如果 inplace=True，则调用 torch._C._nn.elu_ 函数进行原地操作
    if inplace:
        result = torch._C._nn.elu_(input, alpha)
    else:
        # 否则调用 torch._C._nn.elu 函数
        result = torch._C._nn.elu(input, alpha)
    return result


# 添加文档字符串到 torch._C._nn.elu_ 函数
elu_ = _add_docstr(
    torch._C._nn.elu_,
    r"""
elu_(input, alpha=1.) -> Tensor

In-place version of :func:`~elu`.
""",
)


def selu(input: Tensor, inplace: bool = False) -> Tensor:  # noqa: D400,D402
    r"""selu(input, inplace=False) -> Tensor

    Applies element-wise,
    :math:`\text{SELU}(x) = scale * (\max(0,x) + \min(0, \alpha * (\exp(x) - 1)))`,
    with :math:`\alpha=1.6732632423543772848170429916717` and
    :math:`scale=1.0507009873554804934193349852946`.

    See :class:`~torch.nn.SELU` for more details.
    """
    # 如果 input 支持 torch 函数式操作，则调用对应的处理函数
    if has_torch_function_unary(input):
        return handle_torch_function(selu, (input,), input, inplace=inplace)
    # 如果 inplace=True，则调用 torch.selu_ 函数进行原地操作
    if inplace:
        result = torch.selu_(input)
    else:
        # 否则调用 torch.selu 函数
        result = torch.selu(input)
    return result


# 添加文档字符串到 torch.selu_ 函数
selu_ = _add_docstr(
    torch.selu_,
    r"""
selu_(input) -> Tensor

In-place version of :func:`~selu`.
""",
)


def celu(
    input: Tensor,
    alpha: float = 1.0,
    inplace: bool = False,
) -> Tensor:  # noqa: D400,D402
    r"""celu(input, alpha=1., inplace=False) -> Tensor

    Applies element-wise,
    :math:`\text{CELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x/\alpha) - 1))`.

    See :class:`~torch.nn.CELU` for more details.
    """
    # 如果 input 支持 torch 函数式操作，则调用对应的处理函数
    if has_torch_function_unary(input):
        return handle_torch_function(
            celu, (input,), input, alpha=alpha, inplace=inplace
        )
    # 如果 inplace=True，则调用 torch.celu_ 函数进行原地操作
    if inplace:
        result = torch.celu_(input, alpha)
    else:
        # 否则调用 torch.celu 函数
        result = torch.celu(input, alpha)
    return result


# 添加文档字符串到 torch.celu_ 函数
celu_ = _add_docstr(
    torch.celu_,
    r"""
celu_(input, alpha=1.) -> Tensor

In-place version of :func:`~celu`.
""",
)


def leaky_relu(
    input: Tensor,
    negative_slope: float = 0.01,
    inplace: bool = False,
) -> Tensor:  # noqa: D400,D402
    r"""
    leaky_relu(input, negative_slope=0.01, inplace=False) -> Tensor

    Applies element-wise Leaky ReLU non-linearity.

    See :class:`~torch.nn.LeakyReLU` for more details.
    """
    Applies element-wise,
    :math:`\text{LeakyReLU}(x) = \max(0, x) + \text{negative\_slope} * \min(0, x)`
    
    See :class:`~torch.nn.LeakyReLU` for more details.
    """
    # 如果输入对象支持 Torch 函数的单参数形式，则调用 Torch 函数处理
    if has_torch_function_unary(input):
        return handle_torch_function(
            leaky_relu, (input,), input, negative_slope=negative_slope, inplace=inplace
        )
    # 如果 inplace 标志为 True，则调用原位操作的 LeakyReLU
    if inplace:
        result = torch._C._nn.leaky_relu_(input, negative_slope)
    # 如果 inplace 标志为 False，则调用非原位操作的 LeakyReLU
    else:
        result = torch._C._nn.leaky_relu(input, negative_slope)
    # 返回 LeakyReLU 操作的结果
    return result
leaky_relu_ = _add_docstr(
    torch._C._nn.leaky_relu_,
    r"""
    leaky_relu_(input, negative_slope=0.01) -> Tensor
    
    In-place version of :func:`~leaky_relu`.
    """
)

prelu = _add_docstr(
    torch.prelu,
    r"""
    prelu(input, weight) -> Tensor
    
    Applies element-wise the function
    :math:`\text{PReLU}(x) = \max(0,x) + \text{weight} * \min(0,x)` where weight is a
    learnable parameter.
    
    .. note::
        `weight` is expected to be a scalar or 1-D tensor. If `weight` is 1-D,
        its size must match the number of input channels, determined by
        `input.size(1)` when `input.dim() >= 2`, otherwise 1.
        In the 1-D case, note that when `input` has dim > 2, `weight` can be expanded
        to the shape of `input` in a way that is not possible using normal
        :ref:`broadcasting semantics<broadcasting-semantics>`.
    
    See :class:`~torch.nn.PReLU` for more details.
    """
)

def rrelu(
    input: Tensor,
    lower: float = 1.0 / 8,
    upper: float = 1.0 / 3,
    training: bool = False,
    inplace: bool = False,
) -> Tensor:  # noqa: D400,D402
    r"""
    rrelu(input, lower=1./8, upper=1./3, training=False, inplace=False) -> Tensor
    
    Randomized leaky ReLU.
    
    See :class:`~torch.nn.RReLU` for more details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            rrelu,
            (input,),
            input,
            lower=lower,
            upper=upper,
            training=training,
            inplace=inplace,
        )
    if inplace:
        result = torch.rrelu_(input, lower, upper, training)
    else:
        result = torch.rrelu(input, lower, upper, training)
    return result


rrelu_ = _add_docstr(
    torch.rrelu_,
    r"""
    rrelu_(input, lower=1./8, upper=1./3, training=False) -> Tensor
    
    In-place version of :func:`~rrelu`.
    """
)

logsigmoid = _add_docstr(
    torch._C._nn.log_sigmoid,
    r"""
    logsigmoid(input) -> Tensor
    
    Applies element-wise :math:`\text{LogSigmoid}(x_i) = \log \left(\frac{1}{1 + \exp(-x_i)}\right)`
    
    See :class:`~torch.nn.LogSigmoid` for more details.
    """
)

gelu = _add_docstr(
    torch._C._nn.gelu,
    r"""
    gelu(input, approximate = 'none') -> Tensor
    
    When the approximate argument is 'none', it applies element-wise the function
    :math:`\text{GELU}(x) = x * \Phi(x)`
    
    where :math:`\Phi(x)` is the Cumulative Distribution Function for Gaussian Distribution.
    
    When the approximate argument is 'tanh', Gelu is estimated with
    
    .. math::
        \text{GELU}(x) = 0.5 * x * (1 + \text{Tanh}(\sqrt{2 / \pi} * (x + 0.044715 * x^3)))
    
    See `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_.
    """
)

hardshrink = _add_docstr(
    torch.hardshrink,
    r"""
    hardshrink(input, lambd=0.5) -> Tensor
    
    Applies the hard shrinkage function element-wise
    
    See :class:`~torch.nn.Hardshrink` for more details.
    """
)

def tanhshrink(input):  # noqa: D400,D402
    r"""
    tanhshrink(input) -> Tensor
    
    Applies element-wise, :math:`\text{Tanhshrink}(x) = x - \text{Tanh}(x)`
    """
    # 如果输入具有 Torch 函数的重载（torch function），则调用处理 Torch 函数的方法，并返回处理后的结果
    if has_torch_function_unary(input):
        return handle_torch_function(tanhshrink, (input,), input)
    # 否则，计算输入张量的 tanhshrink（tanh(x) - x）并返回结果
    return input - input.tanh()
def softsign(input):  # noqa: D400,D402
    r"""softsign(input) -> Tensor

    Applies element-wise, the function :math:`\text{SoftSign}(x) = \frac{x}{1 + |x|}`

    See :class:`~torch.nn.Softsign` for more details.
    """
    # 检查输入是否有 Torch 函数的一元版本
    if has_torch_function_unary(input):
        # 处理 Torch 函数的调用
        return handle_torch_function(softsign, (input,), input)
    # 应用 SoftSign 函数
    return input / (input.abs() + 1)


softplus = _add_docstr(
    torch._C._nn.softplus,
    r"""
softplus(input, beta=1, threshold=20) -> Tensor

Applies element-wise, the function :math:`\text{Softplus}(x) = \frac{1}{\beta} * \log(1 + \exp(\beta * x))`.

For numerical stability the implementation reverts to the linear function
when :math:`input \times \beta > threshold`.

See :class:`~torch.nn.Softplus` for more details.
""",
)


def _get_softmax_dim(name: str, ndim: int, stacklevel: int) -> int:
    # 发出警告信息，指出隐式选择维度已经被废弃
    warnings.warn(
        f"Implicit dimension choice for {name} has been deprecated. "
        "Change the call to include dim=X as an argument.",
        stacklevel=stacklevel,
    )
    # 根据输入的维度选择 softmax 操作的维度
    if ndim == 0 or ndim == 1 or ndim == 3:
        ret = 0
    else:
        ret = 1
    return ret


def softmin(
    input: Tensor,
    dim: Optional[int] = None,
    _stacklevel: int = 3,
    dtype: Optional[DType] = None,
) -> Tensor:
    r"""Apply a softmin function.

    Note that :math:`\text{Softmin}(x) = \text{Softmax}(-x)`. See softmax definition for mathematical formula.

    See :class:`~torch.nn.Softmin` for more details.

    Args:
        input (Tensor): input
        dim (int): A dimension along which softmin will be computed (so every slice
            along dim will sum to 1).
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
          If specified, the input tensor is casted to :attr:`dtype` before the operation
          is performed. This is useful for preventing data type overflows. Default: None.
    """
    # 检查输入是否有 Torch 函数的一元版本
    if has_torch_function_unary(input):
        # 处理 Torch 函数的调用
        return handle_torch_function(
            softmin, (input,), input, dim=dim, _stacklevel=_stacklevel, dtype=dtype
        )
    # 如果没有指定维度，则根据输入的维度选择 softmin 操作的维度
    if dim is None:
        dim = _get_softmax_dim("softmin", input.dim(), _stacklevel)
    # 应用 softmin 函数，计算 softmax(-input)
    if dtype is None:
        ret = (-input).softmax(dim)
    else:
        ret = (-input).softmax(dim, dtype=dtype)
    return ret


def softmax(
    input: Tensor,
    dim: Optional[int] = None,
    _stacklevel: int = 3,
    dtype: Optional[DType] = None,
) -> Tensor:
    r"""Apply a softmax function.

    Softmax is defined as:

    :math:`\text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}`

    It is applied to all slices along dim, and will re-scale them so that the elements
    lie in the range `[0, 1]` and sum to 1.

    See :class:`~torch.nn.Softmax` for more details.
    """
    # 检查输入是否有 Torch 函数的一元版本
    if has_torch_function_unary(input):
        # 处理 Torch 函数的调用
        return handle_torch_function(
            softmax, (input,), input, dim=dim, _stacklevel=_stacklevel, dtype=dtype
        )
    # 如果没有指定维度，则根据输入的维度选择 softmax 操作的维度
    if dim is None:
        dim = _get_softmax_dim("softmax", input.dim(), _stacklevel)
    # 应用 softmax 函数
    return input.softmax(dim, dtype=dtype)
    """
    Apply softmax function along a specified dimension of the input tensor.

    Args:
        input (Tensor): The input tensor.
        dim (int): The dimension along which softmax will be computed.
        dtype (:class:`torch.dtype`, optional): The desired data type of the returned tensor.
          If specified, the input tensor is casted to :attr:`dtype` before the operation
          is performed. This helps prevent data type overflows. Default: None.

    .. note::
        This function does not directly support NLLLoss, which requires log to be computed
        between softmax and itself. For that purpose, use log_softmax instead, as it is faster
        and has better numerical stability.

    """
    # Check if the input tensor has a Torch function defined for unary operations
    if has_torch_function_unary(input):
        # If yes, invoke Torch's function handler for softmax
        return handle_torch_function(
            softmax, (input,), input, dim=dim, _stacklevel=_stacklevel, dtype=dtype
        )
    
    # If dimension `dim` is not specified, determine it using `_get_softmax_dim` helper function
    if dim is None:
        dim = _get_softmax_dim("softmax", input.dim(), _stacklevel)
    
    # Compute softmax along dimension `dim` of the input tensor
    if dtype is None:
        ret = input.softmax(dim)
    else:
        ret = input.softmax(dim, dtype=dtype)
    
    # Return the computed softmax tensor
    return ret
# 定义一个函数 gumbel_softmax，用于从 Gumbel-Softmax 分布中采样并可选地离散化
def gumbel_softmax(
    logits: Tensor,              # 输入的未归一化对数概率张量，形状为 [..., num_features]
    tau: float = 1,              # 非负标量温度参数
    hard: bool = False,          # 如果为 True，则返回的样本将被离散化为 one-hot 向量；在 autograd 中仍被视为软样本
    eps: float = 1e-10,          # 忽略参数，已弃用
    dim: int = -1,               # 沿着该维度计算 softmax，默认为最后一维
) -> Tensor:                     # 返回值为与 logits 形状相同的 Gumbel-Softmax 分布样本张量

    r"""
    Sample from the Gumbel-Softmax distribution (`Link 1`_  `Link 2`_) and optionally discretize.

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.

    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.

    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`

      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)

    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)

    .. _Link 1:
        https://arxiv.org/abs/1611.00712
    .. _Link 2:
        https://arxiv.org/abs/1611.01144
    """

    if has_torch_function_unary(logits):
        # 处理具有 torch 函数的 logits 的 Torch 函数调用
        return handle_torch_function(
            gumbel_softmax, (logits,), logits, tau=tau, hard=hard, eps=eps, dim=dim
        )
    
    if eps != 1e-10:
        # 如果 eps 参数不等于默认值，发出警告（该参数已弃用且不起作用）
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    # 从 Gumbel 分布中采样（Gumbel 分布的 -log(-log(U))，其中 U 是均匀分布）
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)  # 创建与 logits 相同形状的张量
        .exponential_()                                                         # 从指数分布中采样
        .log()                                                                  # 取对数得到 Gumbel 分布样本
    )  # ~Gumbel(0,1)
    
    # 使用 tau 对 logits 和 gumbels 进行加权平均，形成 Gumbel-Softmax 分布
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    
    # 对 Gumbel-Softmax 分布进行 softmax 操作，沿着指定维度 dim
    y_soft = gumbels.softmax(dim)

    if hard:
        # 硬化操作，使用直通（Straight-through）技巧
        index = y_soft.max(dim, keepdim=True)[1]  # 找到最大概率的索引，形成 one-hot 向量
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).scatter_(dim, index, 1.0)  # 将 one-hot 向量填充到指定位置
        ret = y_hard - y_soft.detach() + y_soft  # 通过直通技巧得到硬化样本
    else:
        # 通过重参数化技巧得到软化样本
        ret = y_soft
    
    # 返回最终的样本张量
    return ret
    # 如果输入对象 input 有定义了 Torch 函数的一元操作，则调用 Torch 函数处理 log_softmax 操作
    if has_torch_function_unary(input):
        # 处理 Torch 函数调用
        return handle_torch_function(
            log_softmax, (input,), input, dim=dim, _stacklevel=_stacklevel, dtype=dtype
        )
    
    # 如果未指定维度 dim，则根据输入张量的维度获取 softmax 的维度
    if dim is None:
        dim = _get_softmax_dim("log_softmax", input.dim(), _stacklevel)
    
    # 如果未指定数据类型 dtype，则直接调用 input 的 log_softmax 方法
    if dtype is None:
        # 计算 log_softmax，结果保存在 ret 中
        ret = input.log_softmax(dim)
    else:
        # 指定了数据类型 dtype，调用 input 的 log_softmax 方法，并指定数据类型
        ret = input.log_softmax(dim, dtype=dtype)
    
    # 返回计算得到的结果 ret
    return ret
softshrink = _add_docstr(
    torch._C._nn.softshrink,
    r"""
softshrink(input, lambd=0.5) -> Tensor

Applies the soft shrinkage function elementwise

See :class:`~torch.nn.Softshrink` for more details.
""",
)
# 定义 softshrink 函数，并添加文档字符串，描述其功能和参数说明

def tanh(input):  # noqa: D400,D402
    r"""tanh(input) -> Tensor

    Applies element-wise,
    :math:`\text{Tanh}(x) = \tanh(x) = \frac{\exp(x) - \exp(-x)}{\exp(x) + \exp(-x)}`

    See :class:`~torch.nn.Tanh` for more details.
    """
    return input.tanh()
# 定义 tanh 函数，实现输入张量的双曲正切运算，返回结果张量

def sigmoid(input):  # noqa: D400,D402
    r"""sigmoid(input) -> Tensor

    Applies the element-wise function :math:`\text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}`

    See :class:`~torch.nn.Sigmoid` for more details.
    """
    return input.sigmoid()
# 定义 sigmoid 函数，实现输入张量的 sigmoid 函数运算，返回结果张量

def hardsigmoid(input: Tensor, inplace: bool = False) -> Tensor:
    r"""Apply the Hardsigmoid function element-wise.

    .. math::
        \text{Hardsigmoid}(x) = \begin{cases}
            0 & \text{if~} x \le -3, \\
            1 & \text{if~} x \ge +3, \\
            x / 6 + 1 / 2 & \text{otherwise}
        \end{cases}

    Args:
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``

    See :class:`~torch.nn.Hardsigmoid` for more details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(hardsigmoid, (input,), input, inplace=inplace)
    if inplace:
        return torch._C._nn.hardsigmoid_(input)
    return torch._C._nn.hardsigmoid(input)
# 定义 hardsigmoid 函数，实现输入张量的硬 sigmoid 函数运算，支持原地操作和 Torch 函数处理

linear = _add_docstr(
    torch._C._nn.linear,
    r"""
linear(input, weight, bias=None) -> Tensor

Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

This operation supports 2-D :attr:`weight` with :ref:`sparse layout<sparse-docs>`

{sparse_beta_warning}

This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.

Shape:

    - Input: :math:`(*, in\_features)` where `*` means any number of
      additional dimensions, including none
    - Weight: :math:`(out\_features, in\_features)` or :math:`(in\_features)`
    - Bias: :math:`(out\_features)` or :math:`()`
    - Output: :math:`(*, out\_features)` or :math:`(*)`, based on the shape of the weight
""".format(
        **sparse_support_notes
    ),
)
# 定义 linear 函数，实现输入数据的线性变换，支持稀疏布局和 TensorFloat32，返回结果张量

bilinear = _add_docstr(
    torch.bilinear,
    r"""
bilinear(input1, input2, weight, bias=None) -> Tensor

Applies a bilinear transformation to the incoming data:
:math:`y = x_1^T A x_2 + b`

Shape:

    - input1: :math:`(N, *, H_{in1})` where :math:`H_{in1}=\text{in1\_features}`
      and :math:`*` means any number of additional dimensions.
      All but the last dimension of the inputs should be the same.
    - input2: :math:`(N, *, H_{in2})` where :math:`H_{in2}=\text{in2\_features}`
    - weight: :math:`(\text{out\_features}, \text{in1\_features},
      \text{in2\_features})`
    - bias: :math:`(\text{out\_features})`
    - output: :math:`(N, *, H_{out})` where :math:`H_{out}=\text{out\_features}`
      and all but the last dimension are the same shape as the input.
""",
)
# 定义 bilinear 函数，实现输入数据的双线性变换，返回结果张量
# 定义 SiLU（Swish）激活函数，接受一个张量输入和一个布尔型参数 inplace
def silu(input: Tensor, inplace: bool = False) -> Tensor:
    r"""Apply the Sigmoid Linear Unit (SiLU) function, element-wise.

    The SiLU function is also known as the swish function.

    .. math::
        \text{silu}(x) = x * \sigma(x), \text{where } \sigma(x) \text{ is the logistic sigmoid.}

    .. note::
        See `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_
        where the SiLU (Sigmoid Linear Unit) was originally coined, and see
        `Sigmoid-Weighted Linear Units for Neural Network Function Approximation
        in Reinforcement Learning <https://arxiv.org/abs/1702.03118>`_ and `Swish:
        a Self-Gated Activation Function <https://arxiv.org/abs/1710.05941v1>`_
        where the SiLU was experimented with later.

    See :class:`~torch.nn.SiLU` for more details.
    """
    # 如果输入具有 torch 函数的重载，调用 torch 函数处理
    if has_torch_function_unary(input):
        return handle_torch_function(silu, (input,), input, inplace=inplace)
    # 如果 inplace 参数为 True，调用 torch 库中的原地修改版本
    if inplace:
        return torch._C._nn.silu_(input)
    # 否则调用 torch 库中的标准版本
    return torch._C._nn.silu(input)


# 定义 Mish 激活函数，接受一个张量输入和一个布尔型参数 inplace
def mish(input: Tensor, inplace: bool = False) -> Tensor:
    r"""Apply the Mish function, element-wise.

    Mish: A Self Regularized Non-Monotonic Neural Activation Function.

    .. math::
        \text{Mish}(x) = x * \text{Tanh}(\text{Softplus}(x))

    .. note::
        See `Mish: A Self Regularized Non-Monotonic Neural Activation Function <https://arxiv.org/abs/1908.08681>`_

    See :class:`~torch.nn.Mish` for more details.
    """
    # 如果输入具有 torch 函数的重载，调用 torch 函数处理
    if has_torch_function_unary(input):
        return handle_torch_function(mish, (input,), input, inplace=inplace)
    # 如果 inplace 参数为 True，调用 torch 库中的原地修改版本
    if inplace:
        return torch._C._nn.mish_(input)
    # 否则调用 torch 库中的标准版本
    return torch._C._nn.mish(input)


# 定义 Hardswish 激活函数，接受一个张量输入和一个布尔型参数 inplace
def hardswish(input: Tensor, inplace: bool = False) -> Tensor:
    r"""Apply hardswish function, element-wise.

    Follows implementation as described in the paper:
    `Searching for MobileNetV3`_.

    .. math::
        \text{Hardswish}(x) = \begin{cases}
            0 & \text{if~} x \le -3, \\
            x & \text{if~} x \ge +3, \\
            x \cdot (x + 3) /6 & \text{otherwise}
        \end{cases}

    See :class:`~torch.nn.Hardswish` for more details.

    .. _`Searching for MobileNetV3`:
        https://arxiv.org/abs/1905.02244
    """
    # 如果输入具有 torch 函数的重载，调用 torch 函数处理
    if has_torch_function_unary(input):
        return handle_torch_function(hardswish, (input,), input, inplace=inplace)
    # 如果 inplace 参数为 True，调用 torch 库中的原地修改版本
    if inplace:
        return torch._C._nn.hardswish_(input)
    # 否则调用 torch 库中的标准版本
    return torch._C._nn.hardswish(input)


# 定义 _no_grad_embedding_renorm_ 函数，对权重张量进行嵌入归一化，不产生梯度
def _no_grad_embedding_renorm_(
    weight: Tensor,
    input: Tensor,
    max_norm: float,
    norm_type: float,
) -> Tuple[Tensor, Tensor]:
    # 使用 torch 的嵌入归一化函数对权重张量进行操作，但不记录梯度
    torch.embedding_renorm_(weight.detach(), input, max_norm, norm_type)


# 定义 embedding 函数，实现张量的嵌入操作
def embedding(
    input: Tensor,
    weight: Tensor,
    padding_idx: Optional[int] = None,
    max_norm: Optional[float] = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> Tensor:
    # 这里实现嵌入操作的具体逻辑，但需要在下一个代码块中进行完整注释
    r"""Generate a simple lookup table that looks up embeddings in a fixed dictionary and size.

    This module is often used to retrieve word embeddings using indices.
    The input to the module is a list of indices, and the embedding matrix,
    and the output is the corresponding word embeddings.

    See :class:`torch.nn.Embedding` for more details.

    .. note::
        Note that the analytical gradients of this function with respect to
        entries in :attr:`weight` at the row specified by :attr:`padding_idx`
        are expected to differ from the numerical ones.

    .. note::
        Note that `:class:`torch.nn.Embedding` differs from this function in
        that it initializes the row of :attr:`weight` specified by
        :attr:`padding_idx` to all zeros on construction.

    Args:
        input (LongTensor): Tensor containing indices into the embedding matrix
        weight (Tensor): The embedding matrix with number of rows equal to the maximum possible index + 1,
            and number of columns equal to the embedding size
        padding_idx (int, optional): If specified, the entries at :attr:`padding_idx` do not contribute to the gradient;
                                     therefore, the embedding vector at :attr:`padding_idx` is not updated during training,
                                     i.e. it remains as a fixed "pad".
        max_norm (float, optional): If given, each embedding vector with norm larger than :attr:`max_norm`
                                    is renormalized to have norm :attr:`max_norm`.
                                    Note: this will modify :attr:`weight` in-place.
        norm_type (float, optional): The p of the p-norm to compute for the :attr:`max_norm` option. Default ``2``.
        scale_grad_by_freq (bool, optional): If given, this will scale gradients by the inverse of frequency of
                                                the words in the mini-batch. Default ``False``.
        sparse (bool, optional): If ``True``, gradient w.r.t. :attr:`weight` will be a sparse tensor. See Notes under
                                 :class:`torch.nn.Embedding` for more details regarding sparse gradients.

    Shape:
        - Input: LongTensor of arbitrary shape containing the indices to extract
        - Weight: Embedding matrix of floating point type with shape `(V, embedding_dim)`,
          where V = maximum index + 1 and embedding_dim = the embedding size
        - Output: `(*, embedding_dim)`, where `*` is the input shape
    """


注释：定义了一个函数或者模块的文档字符串，描述了生成一个简单的查找表，用于从固定大小的字典和大小中查找嵌入。通常用于使用索引检索单词嵌入。详细说明了输入是索引列表和嵌入矩阵，输出是相应的单词嵌入。
    """
    if has_torch_function_variadic(input, weight):
        # 如果 input 和 weight 有 torch 函数的变量重载，调用处理 torch 函数的方法
        return handle_torch_function(
            embedding,
            (input, weight),
            input,
            weight,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
        )
    if padding_idx is not None:
        if padding_idx > 0:
            # 如果 padding_idx 大于 0，确保它在 weight 的大小范围内
            assert padding_idx < weight.size(
                0
            ), "Padding_idx must be within num_embeddings"
        elif padding_idx < 0:
            # 如果 padding_idx 小于 0，确保它在负数形式的 weight 的大小范围内
            assert padding_idx >= -weight.size(
                0
            ), "Padding_idx must be within num_embeddings"
            # 将 padding_idx 转换为 weight 的正数形式
            padding_idx = weight.size(0) + padding_idx
    else:
        # 如果 padding_idx 为 None，设置为默认值 -1
        padding_idx = -1
    if max_norm is not None:
        # 注意 [embedding_renorm contiguous]
        # `embedding_renorm_` 将在 input 上调用 .contiguous()，所以我们在这里也调用它，并且利用下面的 `embedding` 调用中的局部性改进。
        input = input.contiguous()
        # 注意 [embedding_renorm set_grad_enabled]
        # XXX: 相当于
        # with torch.no_grad():
        #   torch.embedding_renorm_
        # 一旦脚本支持 set_grad_enabled，将其删除
        _no_grad_embedding_renorm_(weight, input, max_norm, norm_type)
    # 调用 torch 的 embedding 方法
    return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)
    """
# 定义一个函数，用于计算嵌入袋（embedding bag）的总和、均值或最大值
def embedding_bag(
    input: Tensor,
    weight: Tensor,
    offsets: Optional[Tensor] = None,
    max_norm: Optional[float] = None,
    norm_type: float = 2,
    scale_grad_by_freq: bool = False,
    mode: str = "mean",
    sparse: bool = False,
    per_sample_weights: Optional[Tensor] = None,
    include_last_offset: bool = False,
    padding_idx: Optional[int] = None,
) -> Tensor:
    r"""Compute sums, means or maxes of `bags` of embeddings.

    Calculation is done without instantiating the intermediate embeddings.
    See :class:`torch.nn.EmbeddingBag` for more details.

    Note:
        {backward_reproducibility_note}
    """
    # Args参数列表，描述函数的输入参数
    Args:
        # input (LongTensor): 一个包含索引的张量，用于指向嵌入矩阵中的各个袋子
        input (LongTensor): Tensor containing bags of indices into the embedding matrix
        # weight (Tensor): 嵌入矩阵，行数等于最大可能索引值加1，列数等于嵌入大小
        weight (Tensor): The embedding matrix with number of rows equal to the maximum possible index + 1,
                         and number of columns equal to the embedding size
        # offsets (LongTensor, optional): 当 input 是1维时使用，offsets 确定 input 中每个袋子（序列）的起始索引位置
        offsets (LongTensor, optional): Only used when :attr:`input` is 1D. :attr:`offsets` determines
                                         the starting index position of each bag (sequence) in :attr:`input`.
        # max_norm (float, optional): 如果给定，超过 max_norm 的每个嵌入向量将被重新归一化为 max_norm 的大小
        max_norm (float, optional): If given, each embedding vector with norm larger than :attr:`max_norm`
                                    is renormalized to have norm :attr:`max_norm`.
                                    Note: this will modify :attr:`weight` in-place.
        # norm_type (float, optional): max_norm 选项中计算的 p-norm 中的 p 值，默认为 2
        norm_type (float, optional): The ``p`` in the ``p``-norm to compute for the :attr:`max_norm` option.
                                     Default ``2``.
        # scale_grad_by_freq (bool, optional): 如果给定，将根据小批量中单词的频率倒数来缩放梯度，默认为 False
        scale_grad_by_freq (bool, optional): if given, this will scale gradients by the inverse of frequency of
                                             the words in the mini-batch. Default ``False``.
                                             Note: this option is not supported when ``mode="max"``.
        # mode (str, optional): "sum"、"mean" 或 "max"，指定如何减少袋子的方式，默认为 "mean"
        mode (str, optional): ``"sum"``, ``"mean"`` or ``"max"``. Specifies the way to reduce the bag.
                              Default: ``"mean"``
        # sparse (bool, optional): 如果为 True，则对 weight 的梯度将是一个稀疏张量
        sparse (bool, optional): if ``True``, gradient w.r.t. :attr:`weight` will be a sparse tensor.
                                 Note: this option is not supported when ``mode="max"``.
        # per_sample_weights (Tensor, optional): 一个浮点/双精度权重张量，或者 None 表示所有权重都为 1
        per_sample_weights (Tensor, optional): a tensor of float / double weights, or None
                                               to indicate all weights should be taken to be 1.
        # include_last_offset (bool, optional): 如果为 True，则 offsets 的大小等于袋子数 + 1
        include_last_offset (bool, optional): if ``True``, the size of offsets is equal to the number of bags + 1.
                                              The last element is the size of the input, or the ending index position
                                              of the last bag (sequence).
        # padding_idx (int, optional): 如果指定，padding_idx 处的条目不会对梯度贡献；因此，在训练期间 padding_idx 处的
        #                             嵌入向量保持固定，即作为固定的“pad”。
        padding_idx (int, optional): If specified, the entries at :attr:`padding_idx` do not contribute to the
                                     gradient; therefore, the embedding vector at :attr:`padding_idx` is not updated
                                     during training, i.e. it remains as a fixed "pad". Note that the embedding
                                     vector at :attr:`padding_idx` is excluded from the reduction.
    """
    if has_torch_function_variadic(input, weight, offsets, per_sample_weights):
        return handle_torch_function(
            embedding_bag,
            (input, weight, offsets, per_sample_weights),
            input,
            weight,
            offsets=offsets,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            mode=mode,
            sparse=sparse,
            per_sample_weights=per_sample_weights,
            include_last_offset=include_last_offset,
            padding_idx=padding_idx,
        )
    """
    # 如果有torch函数的可变参数版本，则调用处理torch函数的方法
    if has_torch_function_variadic(input, weight, offsets, per_sample_weights):
        # 调用torch函数处理方法，返回处理结果
        return handle_torch_function(
            embedding_bag,
            # 将(input, weight, offsets, per_sample_weights)作为参数传递给处理方法
            (input, weight, offsets, per_sample_weights),
            # 返回值为input
            input,
            # 返回值为weight
            weight,
            # offsets参数设定为offsets
            offsets=offsets,
            # max_norm参数为max_norm
            max_norm=max_norm,
            # norm_type参数为norm_type
            norm_type=norm_type,
            # scale_grad_by_freq参数为scale_grad_by_freq
            # include_last_offset参数为include_last_offset
 per_sample
    # 检查 weight 的数据类型是否为 torch.long，并且 input 是否为浮点数类型
    if weight.dtype == torch.long and input.is_floating_point():
        # 发出警告，提醒使用新的函数参数顺序
        warnings.warn(
            "Argument order of nn.functional.embedding_bag was changed. "
            "Usage `embedding_bag(weight, input, ...)` is deprecated, "
            "and should now be `embedding_bag(input, weight, ...)`."
        )
        # 交换 weight 和 input 的顺序
        weight, input = input, weight

    # 如果 per_sample_weights 不为 None，并且其形状与 input 的形状不同，抛出 ValueError 异常
    if per_sample_weights is not None and input.size() != per_sample_weights.size():
        raise ValueError(
            f"embedding_bag: If per_sample_weights ({per_sample_weights.shape}) is not None, "
            f"then it must have the same shape as the input ({input.shape})"
        )

    # 检查 weight 的维度是否为 2
    if not weight.dim() == 2:
        raise ValueError(
            f"weight has to be a 2D Tensor, but got Tensor of dimension {weight.dim()}"
        )

    # 如果 input 的维度为 2
    if input.dim() == 2:
        # 如果 offsets 不为 None，则抛出异常
        if offsets is not None:
            type_str = "<unknown>"
            # TODO: Remove this once script supports type() calls
            if not torch.jit.is_scripting():
                type_str = str(type(offsets))
            raise ValueError(
                "if input is 2D, then offsets has to be None"
                ", as input is treated is a mini-batch of"
                " fixed length sequences. However, found "
                f"offsets of type {type_str}"
            )
        # 创建一个包含从 0 到 input 中元素总数的一维张量，步长为 input 的第二维度大小
        offsets = torch.arange(
            0, input.numel(), input.size(1), dtype=input.dtype, device=input.device
        )

        # 将 input 展平为一维张量
        input = input.reshape(-1)
        # 如果 per_sample_weights 不为 None，则将其也展平为一维张量
        if per_sample_weights is not None:
            per_sample_weights = per_sample_weights.reshape(-1)
    
    # 如果 input 的维度为 1
    elif input.dim() == 1:
        # 如果 offsets 为 None，则抛出异常
        if offsets is None:
            raise ValueError("offsets has to be a 1D Tensor but got None")
        # 如果 offsets 的维度不为 1，则抛出异常
        if offsets.dim() != 1:
            raise ValueError("offsets has to be a 1D Tensor")
    
    # 如果 input 的维度既不是 1 也不是 2，则抛出异常
    else:
        raise ValueError(
            f"input has to be 1D or 2D Tensor, but got Tensor of dimension {input.dim()}"
        )

    # 根据 mode 设置 mode_enum 的值
    if mode == "sum":
        mode_enum = 0
    elif mode == "mean":
        mode_enum = 1
    elif mode == "max":
        mode_enum = 2
        
        # 如果 scale_grad_by_freq 为 True，则抛出异常，因为 max 模式不支持梯度频率缩放
        if scale_grad_by_freq:
            raise ValueError(
                "max mode does not support scaling the gradient by the frequency"
            )
        
        # 如果 sparse 为 True，则抛出异常，因为 max 模式不支持稀疏权重
        if sparse:
            raise ValueError("max mode does not support sparse weights")
    
    # 如果 mode 的值不是 "sum"、"mean" 或 "max" 中的一个，则抛出异常
    else:
        raise ValueError("mode has to be one of sum, mean or max")

    # 如果 max_norm 不为 None，则进行权重的范数裁剪
    if max_norm is not None:
        # XXX: 相当于
        # with torch.no_grad():
        #   torch.nembedding_renorm_
        # 一旦脚本支持 set_grad_enabled，则删除此部分
        _no_grad_embedding_renorm_(weight, input, max_norm, norm_type)

    # 如果 per_sample_weights 不为 None，并且 mode 不是 "sum"，则抛出异常
    if per_sample_weights is not None and mode != "sum":
        raise NotImplementedError(
            "embedding_bag: per_sample_weights was not None. "
            "per_sample_weights is only supported for mode='sum' "
            f"(got mode='{mode}'). Please open a feature request on GitHub."
        )
    # 使用 torch.embedding_bag 函数进行嵌入操作，返回结果保存在 ret 变量中
    ret, _, _, _ = torch.embedding_bag(
        weight,                 # 嵌入操作的权重参数
        input,                  # 输入的张量，包含待嵌入的索引
        offsets,                # 每个样本在输入中的偏移量
        scale_grad_by_freq,     # 是否根据频率对梯度进行缩放的布尔值
        mode_enum,              # 嵌入的模式，如 'mean'、'sum' 等
        sparse,                 # 是否使用稀疏梯度的布尔值
        per_sample_weights,     # 每个样本的权重，用于加权平均的操作
        include_last_offset,    # 是否在偏移量中包含最后一个元素的布尔值
        padding_idx,            # 如果提供，指定的填充索引
    )
    # 返回嵌入操作的结果
    return ret
def _verify_batch_size(size: List[int]) -> None:
    # XXX: JIT script does not support the reduce from functools, and mul op is a
    # builtin, which cannot be used as a value to a func yet, so rewrite this size
    # check to a simple equivalent for loop
    # 
    # TODO: make use of reduce like below when JIT is ready with the missing features:
    # from operator import mul
    # from functools import reduce
    #
    #   if reduce(mul, size[2:], size[0]) == 1
    # 计算 size 中除了第一个元素以外的所有元素的乘积
    size_prods = size[0]
    for i in range(len(size) - 2):
        size_prods *= size[i + 2]
    # 如果乘积等于1，抛出数值错误异常
    if size_prods == 1:
        raise ValueError(
            f"Expected more than 1 value per channel when training, got input size {size}"
        )


def batch_norm(
    input: Tensor,
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    r"""Apply Batch Normalization for each channel across a batch of data.

    See :class:`~torch.nn.BatchNorm1d`, :class:`~torch.nn.BatchNorm2d`,
    :class:`~torch.nn.BatchNorm3d` for details.
    """
    # 如果输入的参数有 torch 函数变体，则使用 torch 函数处理
    if has_torch_function_variadic(input, running_mean, running_var, weight, bias):
        return handle_torch_function(
            batch_norm,
            (input, running_mean, running_var, weight, bias),
            input,
            running_mean,
            running_var,
            weight=weight,
            bias=bias,
            training=training,
            momentum=momentum,
            eps=eps,
        )
    # 如果处于训练模式，验证输入数据的大小
    if training:
        _verify_batch_size(input.size())

    # 调用 torch 的批量归一化函数，返回归一化后的结果张量
    return torch.batch_norm(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps,
        torch.backends.cudnn.enabled,
    )


def _verify_spatial_size(size: List[int]) -> None:
    # Verify that there is > 1 spatial element for instance norm calculation.
    # 计算 size 中从第三个元素到最后一个元素（空间元素）的乘积
    size_prods = 1
    for i in range(2, len(size)):
        size_prods *= size[i]
    # 如果乘积等于1，抛出数值错误异常
    if size_prods == 1:
        raise ValueError(
            f"Expected more than 1 spatial element when training, got input size {size}"
        )


def instance_norm(
    input: Tensor,
    running_mean: Optional[Tensor] = None,
    running_var: Optional[Tensor] = None,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    use_input_stats: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    r"""Apply Instance Normalization independently for each channel in every data sample within a batch.

    See :class:`~torch.nn.InstanceNorm1d`, :class:`~torch.nn.InstanceNorm2d`,
    :class:`~torch.nn.InstanceNorm3d` for details.
    """
    # 如果输入的参数经过 torch_function 变种处理后返回 True，则调用处理 torch_function 的函数
    if has_torch_function_variadic(input, running_mean, running_var, weight, bias):
        return handle_torch_function(
            instance_norm,  # 调用的函数是 instance_norm
            (input, running_mean, running_var, weight, bias),  # 传递给 handle_torch_function 的参数元组
            input,  # 实例规范化的输入参数
            running_mean=running_mean,  # 实例规范化的运行均值参数
            running_var=running_var,  # 实例规范化的运行方差参数
            weight=weight,  # 权重参数
            bias=bias,  # 偏置参数
            use_input_stats=use_input_stats,  # 是否使用输入的统计数据参数
            momentum=momentum,  # 动量参数
            eps=eps,  # ε 参数
        )
    
    # 如果 use_input_stats 为 True，则验证输入的空间尺寸
    if use_input_stats:
        _verify_spatial_size(input.size())
    
    # 调用 Torch 的 instance_norm 函数进行实例规范化处理，返回处理结果
    return torch.instance_norm(
        input,  # 输入参数
        weight,  # 权重参数
        bias,  # 偏置参数
        running_mean,  # 运行均值参数
        running_var,  # 运行方差参数
        use_input_stats,  # 是否使用输入的统计数据参数
        momentum,  # 动量参数
        eps,  # ε 参数
        torch.backends.cudnn.enabled,  # 是否启用 cuDNN 加速的参数
    )
# 对输入张量进行层归一化操作，归一化的维度由 normalized_shape 指定
def layer_norm(
    input: Tensor,
    normalized_shape: List[int],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tensor:
    # 检查是否有 torch 函数的变量重载，如果有，则处理 torch 函数的调用
    if has_torch_function_variadic(input, weight, bias):
        return handle_torch_function(
            layer_norm,
            (input, weight, bias),
            input,
            normalized_shape,
            weight=weight,
            bias=bias,
            eps=eps,
        )
    # 调用 torch 的层归一化函数，返回归一化后的张量
    return torch.layer_norm(
        input, normalized_shape, weight, bias, eps, torch.backends.cudnn.enabled
    )


# 对输入张量进行 RMS 归一化操作
def rms_norm(
    input: Tensor,
    normalized_shape: List[int],
    weight: Optional[Tensor] = None,
    eps: Optional[float] = None,
) -> Tensor:
    # 检查是否有 torch 函数的变量重载，如果有，则处理 torch 函数的调用
    if has_torch_function_variadic(input, weight):
        return handle_torch_function(
            rms_norm, (input, weight), input, normalized_shape, weight=weight, eps=eps
        )
    # 调用 torch 的 RMS 归一化函数，返回归一化后的张量
    return torch.rms_norm(input, normalized_shape, weight, eps)


# 对输入张量进行分组归一化操作
def group_norm(
    input: Tensor,
    num_groups: int,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tensor:
    # 检查是否有 torch 函数的变量重载，如果有，则处理 torch 函数的调用
    if has_torch_function_variadic(input, weight, bias):
        return handle_torch_function(
            group_norm,
            (
                input,
                weight,
                bias,
            ),
            input,
            num_groups,
            weight=weight,
            bias=bias,
            eps=eps,
        )
    # 如果输入张量的维度小于 2，则抛出运行时错误
    if input.dim() < 2:
        raise RuntimeError(
            f"Expected at least 2 dimensions for input tensor but received {input.dim()}"
        )
    # 验证批处理大小是否符合要求
    _verify_batch_size(
        [input.size(0) * input.size(1) // num_groups, num_groups]
        + list(input.size()[2:])
    )
    # 调用 torch 的分组归一化函数，返回归一化后的张量
    return torch.group_norm(
        input, num_groups, weight, bias, eps, torch.backends.cudnn.enabled
    )


# 对输入信号应用局部响应归一化
def local_response_norm(
    input: Tensor,
    size: int,
    alpha: float = 1e-4,
    beta: float = 0.75,
    k: float = 1.0,
) -> Tensor:
    # 检查是否有 torch 函数的一元重载，如果有，则处理 torch 函数的调用
    if has_torch_function_unary(input):
        return handle_torch_function(
            local_response_norm, (input,), input, size, alpha=alpha, beta=beta, k=k
        )
    # 获取输入张量的维度
    dim = input.dim()
    # 如果输入张量的维度小于3，则抛出数值错误异常
    if dim < 3:
        raise ValueError(
            f"Expected 3D or higher dimensionality input (got {dim} dimensions)"
        )

    # 如果输入张量元素个数为0，则直接返回输入张量
    if input.numel() == 0:
        return input

    # 计算输入张量的每个元素的平方
    div = input.mul(input)

    # 如果维度为3，则进行以下操作
    if dim == 3:
        # 在第1维度上添加一个维度，变成四维张量
        div = div.unsqueeze(1)
        # 对第2维度和第3维度进行零填充，左右各填充size//2和(size-1)//2个元素
        div = pad(div, (0, 0, size // 2, (size - 1) // 2))
        # 对第1维度使用大小为(size, 1)的平均池化，步长为1，然后去除第1维度
        div = avg_pool2d(div, (size, 1), stride=1).squeeze(1)
    else:
        # 获取输入张量的尺寸
        sizes = input.size()
        # 将平方后的张量视图变换为(sizes[0], 1, sizes[1], sizes[2], -1)形状
        div = div.view(sizes[0], 1, sizes[1], sizes[2], -1)
        # 对第5维度进行零填充，左右各填充size//2和(size-1)//2个元素
        div = pad(div, (0, 0, 0, 0, size // 2, (size - 1) // 2))
        # 对第1维度使用大小为(size, 1, 1)的三维平均池化，步长为1，然后去除第1维度
        div = avg_pool3d(div, (size, 1, 1), stride=1).squeeze(1)
        # 将张量视图还原为原始尺寸
        div = div.view(sizes)

    # 将div张量每个元素乘以alpha，加上k，然后取beta次方
    div = div.mul(alpha).add(k).pow(beta)

    # 返回输入张量除以div张量后的结果
    return input / div
# 定义 CTC 损失函数
def ctc_loss(
    log_probs: Tensor,
    targets: Tensor,
    input_lengths: Tensor,
    target_lengths: Tensor,
    blank: int = 0,
    reduction: str = "mean",
    zero_infinity: bool = False,
) -> Tensor:
    r"""Apply the Connectionist Temporal Classification loss.

    See :class:`~torch.nn.CTCLoss` for details.

    Note:
        {cudnn_reproducibility_note}  # 提示关于 cuDNN 复现性的注意事项

    Note:
        {backward_reproducibility_note}  # 提示关于反向传播复现性的注意事项

    Args:
        log_probs: :math:`(T, N, C)` or :math:`(T, C)` where `C = number of characters in alphabet including blank`,
            `T = input length`, and `N = batch size`.
            The logarithmized probabilities of the outputs
            (e.g. obtained with :func:`torch.nn.functional.log_softmax`).
        targets: :math:`(N, S)` or `(sum(target_lengths))`.
            Targets cannot be blank. In the second form, the targets are assumed to be concatenated.
        input_lengths: :math:`(N)` or :math:`()`.
            Lengths of the inputs (must each be :math:`\leq T`)
        target_lengths: :math:`(N)` or :math:`()`.
            Lengths of the targets
        blank (int, optional):
            Blank label. Default :math:`0`.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the output losses will be divided by the target lengths and
            then the mean over the batch is taken, ``'sum'``: the output will be
            summed. Default: ``'mean'``
        zero_infinity (bool, optional):
            Whether to zero infinite losses and the associated gradients.
            Default: ``False``
            Infinite losses mainly occur when the inputs are too short
            to be aligned to the targets.

    Example::

        >>> log_probs = torch.randn(50, 16, 20).log_softmax(2).detach().requires_grad_()
        >>> targets = torch.randint(1, 20, (16, 30), dtype=torch.long)
        >>> input_lengths = torch.full((16,), 50, dtype=torch.long)
        >>> target_lengths = torch.randint(10, 30, (16,), dtype=torch.long)
        >>> loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        >>> loss.backward()

    Returns:
        Tensor: The computed CTC loss.

    """
    # 处理 torch 函数的多态调用
    if has_torch_function_variadic(log_probs, targets, input_lengths, target_lengths):
        return handle_torch_function(
            ctc_loss,
            (log_probs, targets, input_lengths, target_lengths),
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=blank,
            reduction=reduction,
            zero_infinity=zero_infinity,
        )
    # 调用 PyTorch 提供的 CTC 损失函数
    return torch.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank,
        _Reduction.get_enum(reduction),
        zero_infinity,
    )


# 如果 ctc_loss 函数有文档字符串，则根据 reproducibility_notes 格式化该文档字符串
if ctc_loss.__doc__:
    ctc_loss.__doc__ = ctc_loss.__doc__.format(**reproducibility_notes)


# 定义负对数似然损失函数
def nll_loss(
    input: Tensor,
    target: Tensor,
    # 目标张量，通常是模型的预测结果或者标签
    weight: Optional[Tensor] = None,
    # 权重张量，用于加权损失计算，可选参数，默认为None
    size_average: Optional[bool] = None,
    # 是否对损失进行平均，可选参数，默认为None
    ignore_index: int = -100,
    # 指定要忽略的目标类别的索引，通常用于分类任务，默认为-100
    reduce: Optional[bool] = None,
    # 是否对损失进行降维操作，可选参数，默认为None
    reduction: str = "mean",
    # 损失的降维方式，通常为"mean"（均值）或"sum"（求和），默认为"mean"
def nll_loss(input: Tensor, target: Tensor, weight: Tensor = None, size_average: bool = True,
             ignore_index: int = -100, reduce: bool = True, reduction: str = 'mean') -> Tensor:
    r"""Compute the negative log likelihood loss.

    See :class:`~torch.nn.NLLLoss` for details.

    Args:
        input: :math:`(N, C)` where `C = number of classes` or :math:`(N, C, H, W)`
            in case of 2D Loss, or :math:`(N, C, d_1, d_2, ..., d_K)` where :math:`K \geq 1`
            in the case of K-dimensional loss. `input` is expected to be log-probabilities.
        target: :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`,
            or :math:`(N, d_1, d_2, ..., d_K)` where :math:`K \geq 1` for
            K-dimensional loss.
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, has to be a Tensor of size `C`
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When :attr:`size_average` is
            ``True``, the loss is averaged over non-ignored targets. Default: -100
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Example::

        >>> # input is of size N x C = 3 x 5
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> # each element in target has to have 0 <= value < C
        >>> target = torch.tensor([1, 0, 4])
        >>> output = F.nll_loss(F.log_softmax(input, dim=1), target)
        >>> output.backward()
    """
    # Check if input, target, and weight have a torch function that handles them
    if has_torch_function_variadic(input, target, weight):
        # Handle the torch function with nll_loss and return the result
        return handle_torch_function(
            nll_loss,  # Function to handle
            (input, target, weight),  # Tuple of arguments
            input,  # Original input tensor
            target,  # Original target tensor
            weight=weight,  # Optional weight tensor
            size_average=size_average,  # Size averaging flag
            ignore_index=ignore_index,  # Ignore index value
            reduce=reduce,  # Reduce flag
            reduction=reduction,  # Reduction method
        )
    # 如果 size_average 或 reduce 任一不为 None，则根据它们获取相应的降维方式
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    # 调用 C++ 扩展方法计算 NLL 损失，并返回结果
    return torch._C._nn.nll_loss_nd(
        input, target, weight, _Reduction.get_enum(reduction), ignore_index
    )
def poisson_nll_loss(
    input: Tensor,
    target: Tensor,
    log_input: bool = True,
    full: bool = False,
    size_average: Optional[bool] = None,
    eps: float = 1e-8,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    r"""Poisson negative log likelihood loss.

    See :class:`~torch.nn.PoissonNLLLoss` for details.

    Args:
        input: expectation of underlying Poisson distribution.
        target: random sample :math:`target \sim \text{Poisson}(input)`.
        log_input: if ``True`` the loss is computed as
            :math:`\exp(\text{input}) - \text{target} * \text{input}`, if ``False`` then loss is
            :math:`\text{input} - \text{target} * \log(\text{input}+\text{eps})`. Default: ``True``
        full: whether to compute full loss, i. e. to add the Stirling
            approximation term. Default: ``False``
            :math:`\text{target} * \log(\text{target}) - \text{target} + 0.5 * \log(2 * \pi * \text{target})`.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        eps (float, optional): Small value to avoid evaluation of :math:`\log(0)` when
            :attr:`log_input`\ =\ ``False``. Default: 1e-8
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    """
    # 如果 input 和 target 通过了 Torch 函数变量参数的检查，使用处理 Torch 函数的方法
    if has_torch_function_variadic(input, target):
        # 调用 Torch 函数处理函数，返回处理结果
        return handle_torch_function(
            poisson_nll_loss,
            (input, target),
            input,
            target,
            log_input=log_input,
            full=full,
            size_average=size_average,
            eps=eps,
            reduce=reduce,
            reduction=reduction,
        )
    # 如果 size_average 或 reduce 被显式指定，则使用 legacy_get_string 方法获取字符串形式的 reduction
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    # 如果 reduction 参数不是 "none", "mean", 或 "sum" 中的任何一个，抛出数值错误异常
    if reduction != "none" and reduction != "mean" and reduction != "sum":
        # 将 ret 设为 input 并返回，此处不会继续执行后续代码
        ret = input
        raise ValueError(reduction + " is not a valid value for reduction")
    
    # 使用 PyTorch 的泊松负对数似然损失函数计算损失
    ret = torch.poisson_nll_loss(
        input, target, log_input, full, eps, _Reduction.get_enum(reduction)
    )
    # 返回计算得到的损失值
    return ret
def gaussian_nll_loss(
    input: Tensor,
    target: Tensor,
    var: Tensor,
    full: bool = False,
    eps: float = 1e-6,
    reduction: str = "mean",
) -> Tensor:
    r"""Gaussian negative log likelihood loss.

    See :class:`~torch.nn.GaussianNLLLoss` for details.

    Args:
        input: expectation of the Gaussian distribution.
        target: sample from the Gaussian distribution.
        var: tensor of positive variance(s), one for each of the expectations
            in the input (heteroscedastic), or a single one (homoscedastic).
        full (bool, optional): include the constant term in the loss calculation. Default: ``False``.
        eps (float, optional): value added to var, for stability. Default: 1e-6.
        reduction (str, optional): specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the output is the average of all batch member losses,
            ``'sum'``: the output is the sum of all batch member losses.
            Default: ``'mean'``.
    """
    # Handle torch function application if input, target, and var support it
    if has_torch_function_variadic(input, target, var):
        return handle_torch_function(
            gaussian_nll_loss,
            (input, target, var),
            input,
            target,
            var,
            full=full,
            eps=eps,
            reduction=reduction,
        )

    # Check var size to determine if the loss calculation can proceed
    if var.size() != input.size():
        # If var has one dimension less than input, assume homoscedastic case
        if input.size()[:-1] == var.size():
            var = torch.unsqueeze(var, -1)  # Ensure var.shape = (..., 1)

        # Check for heteroscedastic case where var's last dimension is 1
        elif input.size()[:-1] == var.size()[:-1] and var.size(-1) == 1:
            pass  # No further action needed for heteroscedastic case

        # Raise error if var dimensions do not match expected cases
        else:
            raise ValueError("var is of incorrect size")

    # Validate chosen reduction mode
    if reduction != "none" and reduction != "mean" and reduction != "sum":
        raise ValueError(reduction + " is not valid")

    # Ensure all values in var are non-negative
    if torch.any(var < 0):
        raise ValueError("var has negative entry/entries")

    # Clamp var to prevent log(0) instability
    var = var.clone()
    with torch.no_grad():
        var.clamp_(min=eps)

    # Compute Gaussian negative log likelihood loss
    loss = 0.5 * (torch.log(var) + (input - target) ** 2 / var)
    # 如果 `full` 参数为真，则增加 0.5 * log(2 * pi) 到损失值
    if full:
        loss += 0.5 * math.log(2 * math.pi)
    
    # 根据 `reduction` 参数的取值进行损失的归约操作
    if reduction == "mean":
        # 返回损失值的均值
        return loss.mean()
    elif reduction == "sum":
        # 返回损失值的总和
        return loss.sum()
    else:
        # 若 `reduction` 参数不是 "mean" 或 "sum"，则直接返回损失值
        return loss
def kl_div(
    input: Tensor,
    target: Tensor,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
    log_target: bool = False,
) -> Tensor:
    r"""Compute the KL Divergence loss.

    Refer - The `Kullback-Leibler divergence Loss
    <https://en.wikipedia.org/wiki/Kullback-Leibler_divergence>`__

    See :class:`~torch.nn.KLDivLoss` for details.

    Args:
        input: Tensor of arbitrary shape in log-probabilities.
        target: Tensor of the same shape as input. See :attr:`log_target` for
            the target's interpretation.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'batchmean'`` | ``'sum'`` | ``'mean'``.
            ``'none'``: no reduction will be applied
            ``'batchmean'``: the sum of the output will be divided by the batchsize
            ``'sum'``: the output will be summed
            ``'mean'``: the output will be divided by the number of elements in the output
            Default: ``'mean'``
        log_target (bool): A flag indicating whether ``target`` is passed in the log space.
            It is recommended to pass certain distributions (like ``softmax``)
            in the log space to avoid numerical issues caused by explicit ``log``.
            Default: ``False``

    .. note::
        :attr:`size_average` and :attr:`reduce` are in the process of being deprecated,
        and in the meantime, specifying either of those two args will override :attr:`reduction`.

    .. warning::
        :attr:`reduction` = ``'mean'`` doesn't return the true kl divergence value, please use
        :attr:`reduction` = ``'batchmean'`` which aligns with KL math definition.
    """
    # 如果输入参数和目标参数通过torch函数被广义定义，那么委托torch函数处理
    if has_torch_function_variadic(input, target):
        return handle_torch_function(
            kl_div,
            (input, target),
            input,
            target,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            log_target=log_target,
        )
    # 如果size_average或reduce被指定，将其转换成_Reduction枚举类型
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    # 如果 reduction 参数不是 "mean" 或 "batchmean"，发出警告信息
    else:
        if reduction == "mean":
            # 发出警告，说明 'mean' 的行为将来会与 'batchmean' 一致
            warnings.warn(
                "reduction: 'mean' divides the total loss by both the batch size and the support size."
                "'batchmean' divides only by the batch size, and aligns with the KL div math definition."
                "'mean' will be changed to behave the same as 'batchmean' in the next major release."
            )

        # 对于 reduction 为 "batchmean" 的特殊情况
        if reduction == "batchmean":
            # 设置 reduction_enum 为 sum 对应的枚举值
            reduction_enum = _Reduction.get_enum("sum")
        else:
            # 根据 reduction 参数获取对应的枚举值
            reduction_enum = _Reduction.get_enum(reduction)

    # 计算 KL 散度，并根据指定的 reduction 策略进行降维处理
    reduced = torch.kl_div(input, target, reduction_enum, log_target=log_target)

    # 如果 reduction 是 "batchmean" 并且输入张量的维度不为零
    if reduction == "batchmean" and input.dim() != 0:
        # 将 reduced 结果除以输入张量的第一个维度大小
        reduced = reduced / input.size()[0]

    # 返回处理后的降维结果
    return reduced
# 计算输入 logits 和目标之间的交叉熵损失函数
def cross_entropy(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    size_average: Optional[bool] = None,
    ignore_index: int = -100,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> Tensor:
    r"""Compute the cross entropy loss between input logits and target.

    See :class:`~torch.nn.CrossEntropyLoss` for details.

    Args:
        input (Tensor) : Predicted unnormalized logits;
            see Shape section below for supported shapes.
        target (Tensor) : Ground truth class indices or class probabilities;
            see Shape section below for supported shapes.
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, has to be a Tensor of size `C`
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When :attr:`size_average` is
            ``True``, the loss is averaged over non-ignored targets. Note that
            :attr:`ignore_index` is only applicable when the target contains class indices.
            Default: -100
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
        label_smoothing (float, optional): A float in [0.0, 1.0]. Specifies the amount
            of smoothing when computing the loss, where 0.0 means no smoothing. The targets
            become a mixture of the original ground truth and a uniform distribution as described in
            `Rethinking the Inception Architecture for Computer Vision <https://arxiv.org/abs/1512.00567>`__. Default: :math:`0.0`.
    """
    # 实现交叉熵损失计算的函数，具体实现和细节参考 torch.nn.CrossEntropyLoss

    # 返回计算后的损失值张量
    return torch.nn.functional.cross_entropy(
        input, target, weight=weight, size_average=size_average,
        ignore_index=ignore_index, reduce=reduce, reduction=reduction,
        label_smoothing=label_smoothing
    )
    """
    Calculate the cross-entropy loss between the input and target.

    Args:
        input (Tensor): The input tensor with predicted values.
        target (Tensor): The target tensor with true class labels or probabilities.
        weight (Tensor, optional): A tensor of weights to apply to each class. Default: None.
        size_average (bool, optional): Deprecated (see reduction). Default: None.
        ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the input gradient. Default: -100.
        reduce (bool, optional): Deprecated (see reduction). Default: None.
        reduction (str, optional): Specifies the reduction to apply to the output ('none', 'mean', 'sum'). Default: 'mean'.
        label_smoothing (float, optional): Applies label smoothing to the targets. Default: 0.

    Returns:
        Tensor: The computed loss.

    Notes:
        - The 'input' tensor represents predicted values from the model.
        - The 'target' tensor can contain either class indices or probabilities.
        - If 'size_average' or 'reduce' are provided, they are ignored in favor of 'reduction'.
        - The function supports handling through TorchScript if 'has_torch_function_variadic' is True.

    Examples::

        >>> # Example of target with class indices
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randint(5, (3,), dtype=torch.int64)
        >>> loss = F.cross_entropy(input, target)
        >>> loss.backward()

        >>> # Example of target with class probabilities
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5).softmax(dim=1)
        >>> loss = F.cross_entropy(input, target)
        >>> loss.backward()
    """
    # Check if there is a TorchScript function variant available for handling
    if has_torch_function_variadic(input, target, weight):
        # If available, handle the cross_entropy function through TorchScript
        return handle_torch_function(
            cross_entropy,
            (input, target, weight),
            input,
            target,
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )
    
    # Handle legacy 'size_average' and 'reduce' arguments
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    
    # Compute cross-entropy loss using C++ backend
    return torch._C._nn.cross_entropy_loss(
        input,
        target,
        weight,
        _Reduction.get_enum(reduction),
        ignore_index,
        label_smoothing,
    )
# 定义一个函数，计算输入概率和目标之间的二元交叉熵损失
def binary_cross_entropy(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    r"""Measure Binary Cross Entropy between the target and input probabilities.

    See :class:`~torch.nn.BCELoss` for details.

    Args:
        input: Tensor of arbitrary shape as probabilities. 输入张量，任意形状的概率值。
        target: Tensor of the same shape as input with values between 0 and 1. 目标张量，与输入形状相同，其值在0和1之间。
        weight (Tensor, optional): a manual rescaling weight
                if provided it's repeated to match input tensor shape 权重张量，手动重新缩放权重。如果提供，则重复以匹配输入张量的形状。
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True`` 大小平均（已弃用）。默认情况下，损失在批处理中的每个损失元素上进行平均。
            请注意，对于某些损失，每个样本有多个元素。如果字段 :attr:`size_average` 设置为 ``False``，则每个小批量的损失将被求和。当 reduce 设置为 ``False`` 时忽略。
            默认值：``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True`` 减少（已弃用）。默认情况下，损失在每个小批量的观测值上进行平均或求和，具体取决于 :attr:`size_average`。
            当 :attr:`reduce` 设置为 ``False`` 时，返回每个批次元素的损失，并忽略 :attr:`size_average`。
            默认值：``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'`` 指定应用于输出的减少方法：
            ``'none'`` | ``'mean'`` | ``'sum'``。 ``'none'``：不会应用任何减少，
            ``'mean'``：输出的总和将被输出中的元素数除以，
            ``'sum'``：输出将被总和。注意：:attr:`size_average` 和 :attr:`reduce` 正在被弃用，
            同时，指定这两个参数之一将覆盖 :attr:`reduction`。默认值： ``'mean'``

    Examples::

        >>> input = torch.randn(3, 2, requires_grad=True)
        >>> target = torch.rand(3, 2, requires_grad=False)
        >>> loss = F.binary_cross_entropy(torch.sigmoid(input), target)
        >>> loss.backward()
    """
    # 检查是否有变量函数的 Torch 功能
    if has_torch_function_variadic(input, target, weight):
        return handle_torch_function(
            binary_cross_entropy,
            (input, target, weight),
            input,
            target,
            weight=weight,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
        )
    # 处理大小平均和减少参数的退化情况
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction_enum = _Reduction.get_enum(reduction)
    # 检查目标张量和输入张量的尺寸是否相同，否则引发 ValueError
    if target.size() != input.size():
        raise ValueError(
            f"Using a target size ({target.size()}) that is different to the input size ({input.size()}) is deprecated. "
            "Please ensure they have the same size."
        )
    # 如果 weight 参数不为 None，则进行以下操作
    if weight is not None:
        # 调用 _infer_size 函数推断目标张量的大小，并根据权重张量的大小进行扩展
        new_size = _infer_size(target.size(), weight.size())
        weight = weight.expand(new_size)
    
    # 返回通过 torch._C._nn.binary_cross_entropy 计算得到的损失值，传入输入张量、目标张量、权重张量和减少方式枚举
    return torch._C._nn.binary_cross_entropy(input, target, weight, reduction_enum)
# 定义一个函数，计算输入 logits 和目标之间的二元交叉熵损失
def binary_cross_entropy_with_logits(
    input: Tensor,  # 输入张量，包含未归一化的分数（通常称为 logits）
    target: Tensor,  # 与输入张量相同形状的张量，其值在 0 到 1 之间
    weight: Optional[Tensor] = None,  # 可选的手动重新缩放权重，如果提供，则重复以匹配输入张量的形状
    size_average: Optional[bool] = None,  # 已弃用（参见 reduction）。默认情况下，对批次中每个损失元素进行平均。注意，对于某些损失，每个样本可能有多个元素。
    reduce: Optional[bool] = None,  # 已弃用（参见 reduction）。默认情况下，对每个批次观测进行平均或求和，具体取决于 size_average。当 reduce 为 False 时，返回每个批次元素的损失，并忽略 size_average。
    reduction: str = "mean",  # 指定应用于输出的减少方式：'none'：不应用任何减少，'mean'：输出总和将被输出元素的数量除以，'sum'：输出将被求和。
    pos_weight: Optional[Tensor] = None,  # 正样本的权重，将与目标广播。必须是沿着类维度大小相等的张量数。
) -> Tensor:  # 函数返回一个张量

    r"""Calculate Binary Cross Entropy between target and input logits.

    See :class:`~torch.nn.BCEWithLogitsLoss` for details.

    Args:
        input: Tensor of arbitrary shape as unnormalized scores (often referred to as logits).
        target: Tensor of the same shape as input with values between 0 and 1
        weight (Tensor, optional): a manual rescaling weight
            if provided it's repeated to match input tensor shape
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
        pos_weight (Tensor, optional): a weight of positive examples to be broadcasted with target.
            Must be a tensor with equal size along the class dimension to the number of classes.
            Pay close attention to PyTorch's broadcasting semantics in order to achieve the desired
            operations. For a target of size [B, C, H, W] (where B is batch size) pos_weight of
            size [B, C, H, W] will apply different pos_weights to each element of the batch or
            [C, H, W] the same pos_weights across the batch. To apply the same positive weight
            along all spatial dimensions for a 2D multi-class target [C, H, W] use: [C, 1, 1].
            Default: ``None``

    Examples::

         >>> input = torch.randn(3, requires_grad=True)
         >>> target = torch.empty(3).random_(2)
         >>> loss = F.binary_cross_entropy_with_logits(input, target)
         >>> loss.backward()
    """
    # 检查是否存在torch_function的可变参数，根据情况调用对应的处理函数
    if has_torch_function_variadic(input, target, weight, pos_weight):
        return handle_torch_function(
            binary_cross_entropy_with_logits,  # 调用torch_function处理二元交叉熵函数
            (input, target, weight, pos_weight),  # 传递给torch_function的参数元组
            input,  # 输入张量
            target,  # 目标张量
            weight=weight,  # 权重张量
            size_average=size_average,  # 平均大小参数
            reduce=reduce,  # 减少参数
            reduction=reduction,  # 缩减方式参数
            pos_weight=pos_weight,  # 正权重参数
        )
    
    # 如果size_average或reduce有值，则使用旧的减少枚举；否则使用新的减少枚举
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction_enum = _Reduction.get_enum(reduction)
    
    # 检查目标张量与输入张量的大小是否一致，否则抛出值错误异常
    if not (target.size() == input.size()):
        raise ValueError(
            f"Target size ({target.size()}) must be the same as input size ({input.size()})"
        )
    
    # 调用torch库中的二元交叉熵函数，并传递输入、目标、权重、正权重和减少枚举参数
    return torch.binary_cross_entropy_with_logits(
        input, target, weight, pos_weight, reduction_enum
    )
def smooth_l1_loss(
    input: Tensor,
    target: Tensor,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
    beta: float = 1.0,
) -> Tensor:
    r"""Compute the Smooth L1 loss.

    Function uses a squared term if the absolute
    element-wise error falls below beta and an L1 term otherwise.

    See :class:`~torch.nn.SmoothL1Loss` for details.
    """
    # 如果输入和目标张量具有可变参数的 Torch 函数，交给 Torch 函数处理
    if has_torch_function_variadic(input, target):
        return handle_torch_function(
            smooth_l1_loss,
            (input, target),
            input,
            target,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            beta=beta,
        )
    # 如果目标张量的尺寸与输入张量的尺寸不同，发出警告
    if not (target.size() == input.size()):
        warnings.warn(
            f"Using a target size ({target.size()}) that is different to the input size ({input.size()}). "
            "This will likely lead to incorrect results due to broadcasting. "
            "Please ensure they have the same size.",
            stacklevel=2,
        )
    # 如果指定了 size_average 或 reduce 参数，则使用遗留的减少方式
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)

    # 扩展输入和目标张量，使它们能够进行广播
    expanded_input, expanded_target = torch.broadcast_tensors(input, target)

    # 如果 beta 等于 0.0，则返回 L1 损失
    if beta == 0.0:
        return torch._C._nn.l1_loss(
            expanded_input, expanded_target, _Reduction.get_enum(reduction)
        )
    # 否则，返回 Smooth L1 损失
    else:
        return torch._C._nn.smooth_l1_loss(
            expanded_input, expanded_target, _Reduction.get_enum(reduction), beta
        )


def huber_loss(
    input: Tensor,
    target: Tensor,
    reduction: str = "mean",
    delta: float = 1.0,
) -> Tensor:
    r"""Compute the Huber loss.

    Function uses a squared term if the absolute
    element-wise error falls below delta and a delta-scaled L1 term otherwise.

    When delta equals 1, this loss is equivalent to SmoothL1Loss.
    In general, Huber loss differs from SmoothL1Loss by a factor of delta (AKA beta in Smooth L1).

    See :class:`~torch.nn.HuberLoss` for details.
    """
    # 如果输入和目标张量具有可变参数的 Torch 函数，交给 Torch 函数处理
    if has_torch_function_variadic(input, target):
        return handle_torch_function(
            huber_loss,
            (input, target),
            input,
            target,
            reduction=reduction,
            delta=delta,
        )
    # 如果目标张量的尺寸与输入张量的尺寸不同，发出警告
    if not (target.size() == input.size()):
        warnings.warn(
            f"Using a target size ({target.size()}) that is different to the input size ({input.size()}). "
            "This will likely lead to incorrect results due to broadcasting. "
            "Please ensure they have the same size.",
            stacklevel=2,
        )

    # 扩展输入和目标张量，使它们能够进行广播
    expanded_input, expanded_target = torch.broadcast_tensors(input, target)
    
    # 返回 Huber 损失
    return torch._C._nn.huber_loss(
        expanded_input, expanded_target, _Reduction.get_enum(reduction), delta
    )


def l1_loss(
    input: Tensor,
    target: Tensor,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    # 定义一个字符串类型的变量 reduction，并初始化为 "mean"
    reduction: str = "mean",
def margin_ranking_loss(
    input1: Tensor,
    input2: Tensor,
    target: Tensor,
    margin: float = 0,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:  # noqa: D400,D402
    r"""margin_ranking_loss(input1, input2, target, margin=0, size_average=None, reduce=None, reduction='mean') -> Tensor

    See :class:`~torch.nn.MarginRankingLoss` for details.
    """

    # 如果输入具有变量功能，则调用 torch 函数处理
    if has_torch_function_variadic(input1, input2, target):
        return handle_torch_function(
            margin_ranking_loss,
            (input1, input2, target),
            input1,
            input2,
            target,
            margin=margin,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
        )

    # 如果目标张量的尺寸与输入张量的尺寸不匹配，则发出警告
    if not (target.size() == input1.size()):
        warnings.warn(
            f"Using a target size ({target.size()}) that is different to the input size ({input1.size()}). "
            "This will likely lead to incorrect results due to broadcasting. "
            "Please ensure they have the same size.",
            stacklevel=2,
        )

    # 如果指定了 size_average 或 reduce 参数，则使用旧版本的字符串表示方法获取减少方式
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)

    # 扩展输入张量和目标张量，使其可以广播
    expanded_input1, expanded_input2 = torch.broadcast_tensors(input1, input2)

    # 调用 C++ 实现的 margin ranking loss 函数
    return torch._C._nn.margin_ranking_loss(
        expanded_input1,  # 扩展后的输入张量 1
        expanded_input2,  # 扩展后的输入张量 2
        target,           # 目标张量
        margin,           # 边界值
        _Reduction.get_enum(reduction)  # 减少方式的枚举值
    )
    # 检查是否需要通过 torch function 处理输入参数
    if has_torch_function_variadic(input1, input2, target):
        # 若需要，则调用 handle_torch_function 处理 margin_ranking_loss 函数
        return handle_torch_function(
            margin_ranking_loss,
            (input1, input2, target),
            input1,
            input2,
            target,
            margin=margin,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
        )
    
    # 如果 size_average 或 reduce 不为 None，则使用 legacy_get_enum 获取 reduction_enum
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        # 否则，使用 get_enum 获取 reduction_enum
        reduction_enum = _Reduction.get_enum(reduction)
    
    # 检查 input1、input2 和 target 的维度是否一致
    if input1.dim() != input2.dim() or input1.dim() != target.dim():
        # 若维度不一致，则抛出 RuntimeError 异常
        raise RuntimeError(
            f"margin_ranking_loss : All input tensors should have same dimension but got sizes: "
            f"input1: {input1.size()}, input2: {input2.size()}, target: {target.size()} "
        )
    
    # 调用 torch 的 margin_ranking_loss 函数，传入 input1、input2、target、margin 和 reduction_enum
    return torch.margin_ranking_loss(input1, input2, target, margin, reduction_enum)
# 定义 Hinge Embedding Loss 函数，用于计算输入与目标之间的 Hinge Embedding 损失
def hinge_embedding_loss(
    input: Tensor,
    target: Tensor,
    margin: float = 1.0,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:  # noqa: D400,D402
    r"""hinge_embedding_loss(input, target, margin=1.0, size_average=None, reduce=None, reduction='mean') -> Tensor

    See :class:`~torch.nn.HingeEmbeddingLoss` for details.
    """
    # 如果 input 或 target 中有可变参数的 torch 函数，交由 handle_torch_function 处理
    if has_torch_function_variadic(input, target):
        return handle_torch_function(
            hinge_embedding_loss,
            (input, target),
            input,
            target,
            margin=margin,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
        )
    # 根据 size_average 和 reduce 的值确定 reduction_enum，用于指定如何减少损失
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction_enum = _Reduction.get_enum(reduction)
    # 调用 torch.hinge_embedding_loss 函数计算 Hinge Embedding 损失
    return torch.hinge_embedding_loss(input, target, margin, reduction_enum)


# 定义 Multilabel Margin Loss 函数，用于计算多标签间隔损失
def multilabel_margin_loss(
    input: Tensor,
    target: Tensor,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:  # noqa: D400,D402
    r"""multilabel_margin_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor

    See :class:`~torch.nn.MultiLabelMarginLoss` for details.
    """
    # 如果 input 或 target 中有可变参数的 torch 函数，交由 handle_torch_function 处理
    if has_torch_function_variadic(input, target):
        return handle_torch_function(
            multilabel_margin_loss,
            (input, target),
            input,
            target,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
        )
    # 根据 size_average 和 reduce 的值确定 reduction_enum，用于指定如何减少损失
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction_enum = _Reduction.get_enum(reduction)
    # 调用 torch._C._nn.multilabel_margin_loss 函数计算多标签间隔损失
    return torch._C._nn.multilabel_margin_loss(input, target, reduction_enum)


# 定义 Soft Margin Loss 函数，用于计算输入与目标之间的 Soft Margin 损失
def soft_margin_loss(
    input: Tensor,
    target: Tensor,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:  # noqa: D400,D402
    r"""
    soft_margin_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor

    See :class:`~torch.nn.SoftMarginLoss` for details.
    """
    # 如果 input 或 target 中有可变参数的 torch 函数，交由 handle_torch_function 处理
    if has_torch_function_variadic(input, target):
        return handle_torch_function(
            soft_margin_loss,
            (input, target),
            input,
            target,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
        )
    # 根据 size_average 和 reduce 的值确定 reduction_enum，用于指定如何减少损失
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction_enum = _Reduction.get_enum(reduction)
    # 调用 torch._C._nn.soft_margin_loss 函数计算 Soft Margin 损失
    return torch._C._nn.soft_margin_loss(input, target, reduction_enum)


# 定义 Multilabel Soft Margin Loss 函数，用于计算多标签 Soft Margin 损失
def multilabel_soft_margin_loss(
    input: Tensor,
    target: Tensor,
    reduction: str = "mean",
) -> Tensor:  # noqa: D400,D402
    r"""
    multilabel_soft_margin_loss(input, target, reduction='mean') -> Tensor

    See :class:`~torch.nn.MultiLabelSoftMarginLoss` for details.
    """
    # 如果 input 或 target 中有可变参数的 torch 函数，交由 handle_torch_function 处理
    if has_torch_function_variadic(input, target):
        return handle_torch_function(
            multilabel_soft_margin_loss,
            (input, target),
            input,
            target,
            reduction=reduction,
        )
    # 调用 torch._C._nn.multilabel_soft_margin_loss 函数计算多标签 Soft Margin 损失
    return torch._C._nn.multilabel_soft_margin_loss(input, target, reduction)
    weight: Optional[Tensor] = None,
    # 权重参数，类型为可选的张量，初始值为 None
    size_average: Optional[bool] = None,
    # 是否对每个元素求平均值，类型为可选的布尔值，初始值为 None
    reduce: Optional[bool] = None,
    # 是否进行降维操作，类型为可选的布尔值，初始值为 None
    reduction: str = "mean",
    # 指定如何减少输出形状，缺省为 "mean"，表示求均值
def multi_margin_loss(
    input: Tensor,
    target: Tensor,
    p: int = 1,
    margin: float = 1.0,
    weight: Optional[Tensor] = None,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:  # noqa: D400,D402
    r"""multi_margin_loss(input, target, p=1, margin=1, weight=None, size_average=None, reduce=None, reduction='mean') -> Tensor

    See :class:`~torch.nn.MultiMarginLoss` for details.
    """
    # 检查是否有 Torch 函数的可变参数版本，如果是则处理 Torch 函数
    if has_torch_function_variadic(input, target, weight):
        return handle_torch_function(
            multi_margin_loss,
            (input, target, weight),
            input,
            target,
            weight=weight,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
        )
    
    # 如果设置了 size_average 或 reduce 参数，则根据其值设置 reduction
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction_enum = _Reduction.get_enum(reduction)
    
    # 调用 PyTorch 中的 multi_margin_loss 函数，计算损失值
    return torch.multi_margin_loss(input, target, p, margin, weight, reduction_enum)
    # 检查是否有 torch function 支持可变参数的变量，用于处理多元边界损失函数
    if has_torch_function_variadic(input, target, weight):
        # 调用处理 torch function 的函数，返回处理结果
        return handle_torch_function(
            multi_margin_loss,    # 使用的损失函数名称
            (input, target, weight),  # 传递给处理函数的参数元组
            input,    # 输入数据张量
            target,   # 目标数据张量
            p=p,      # 损失函数的 p 参数
            margin=margin,  # 损失函数的 margin 参数
            weight=weight,  # 损失函数的 weight 参数
            size_average=size_average,  # 平均大小选项
            reduce=reduce,  # 减少选项
            reduction=reduction,  # 减少方式
        )
    
    # 如果 size_average 或 reduce 参数不为空
    if size_average is not None or reduce is not None:
        # 根据 size_average 和 reduce 参数获取对应的 reduction_enum 枚举
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        # 否则，根据 reduction 参数获取对应的 reduction_enum 枚举
        reduction_enum = _Reduction.get_enum(reduction)
    
    # 如果 p 不等于 1 且不等于 2，抛出数值错误异常
    if p != 1 and p != 2:
        raise ValueError("only p == 1 and p == 2 supported")
    
    # 如果 weight 参数不为空
    if weight is not None:
        # 如果 weight 的维度不是 1，抛出数值错误异常
        if weight.dim() != 1:
            raise ValueError("weight must be one-dimensional")

    # 调用 torch._C._nn.multi_margin_loss 函数计算多元边界损失
    return torch._C._nn.multi_margin_loss(
        input, target, p, margin, weight, reduction_enum
    )
pixel_shuffle = _add_docstr(
    torch.pixel_shuffle,
    r"""
pixel_shuffle(input, upscale_factor) -> Tensor

Rearranges elements in a tensor of shape :math:`(*, C \times r^2, H, W)` to a
tensor of shape :math:`(*, C, H \times r, W \times r)`, where r is the :attr:`upscale_factor`.

See :class:`~torch.nn.PixelShuffle` for details.

Args:
    input (Tensor): the input tensor
    upscale_factor (int): factor to increase spatial resolution by

Examples::

    >>> input = torch.randn(1, 9, 4, 4)
    >>> output = torch.nn.functional.pixel_shuffle(input, 3)
    >>> print(output.size())
    torch.Size([1, 1, 12, 12])
""",
)

pixel_unshuffle = _add_docstr(
    torch.pixel_unshuffle,
    r"""
pixel_unshuffle(input, downscale_factor) -> Tensor

Reverses the :class:`~torch.nn.PixelShuffle` operation by rearranging elements in a
tensor of shape :math:`(*, C, H \times r, W \times r)` to a tensor of shape
:math:`(*, C \times r^2, H, W)`, where r is the :attr:`downscale_factor`.

See :class:`~torch.nn.PixelUnshuffle` for details.

Args:
    input (Tensor): the input tensor
    downscale_factor (int): factor to increase spatial resolution by

Examples::

    >>> input = torch.randn(1, 1, 12, 12)
    >>> output = torch.nn.functional.pixel_unshuffle(input, 3)
    >>> print(output.size())
    torch.Size([1, 9, 4, 4])
""",
)

channel_shuffle = _add_docstr(
    torch.channel_shuffle,
    r"""
channel_shuffle(input, groups) -> Tensor

Divide the channels in a tensor of shape :math:`(*, C , H, W)`
into g groups and rearrange them as :math:`(*, C \frac{C}{g}, g, H, W)`,
while keeping the original tensor shape.

See :class:`~torch.nn.ChannelShuffle` for details.

Args:
    input (Tensor): the input tensor
    groups (int): number of groups to divide channels in and rearrange.

Examples::

    >>> input = torch.randn(1, 4, 2, 2)
    >>> print(input)
    [[[[1, 2],
       [3, 4]],
      [[5, 6],
       [7, 8]],
      [[9, 10],
       [11, 12]],
      [[13, 14],
       [15, 16]],
     ]]
    >>> output = torch.nn.functional.channel_shuffle(input, 2)
    >>> print(output)
    [[[[1, 2],
       [3, 4]],
      [[9, 10],
       [11, 12]],
      [[5, 6],
       [7, 8]],
      [[13, 14],
       [15, 16]],
     ]]
""",
)

native_channel_shuffle = _add_docstr(
    torch.native_channel_shuffle,
    r"""
native_channel_shuffle(input, groups) -> Tensor

Native kernel level implementation of the `channel_shuffle`.
This function might become private in future releases, use with caution.

Divide the channels in a tensor of shape :math:`(*, C , H, W)`
into g groups and rearrange them as :math:`(*, C \frac{C}{g}, g, H, W)`,
while keeping the original tensor shape.

See :class:`~torch.nn.ChannelShuffle` for details.

Args:
    input (Tensor): the input tensor
    groups (int): number of groups to divide channels in and rearrange.

Examples::

    >>> input = torch.randn(1, 4, 2, 2)
    >>> print(input)
    # Output not shown due to code snippet cutoff
""",
)
    # 创建一个四维列表，表示一个4个通道的输入张量，每个通道包含一个2x2的矩阵
    input = [
        [[1, 2],
         [3, 4]],
        [[5, 6],
         [7, 8]],
        [[9, 10],
         [11, 12]],
        [[13, 14],
         [15, 16]]
    ]
    
    # 使用 PyTorch 的函数 native_channel_shuffle 对输入张量进行通道内元素的重新排序，通道数为2
    output = torch.nn.functional.native_channel_shuffle(input, 2)
    
    # 打印重新排序后的输出张量
    print(output)
    # 输出如下，通道内的元素顺序发生了变化，但通道之间的顺序保持不变：
    # [[[[1, 2],
    #    [3, 4]],
    #   [[9, 10],
    #    [11, 12]],
    #   [[5, 6],
    #    [7, 8]],
    #   [[13, 14],
    #    [15, 16]],
    #  ]]
# noqa: F811
# 定义了一个函数重载，接受一个Tensor类型的输入和几个可选参数，返回一个Tensor类型的输出
@_overload
def upsample(
    input: Tensor,
    size: Optional[int] = None,
    scale_factor: Optional[float] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
) -> Tensor:
    pass

# noqa: F811
# 定义了另一个函数重载，接受一个Tensor类型的输入和几个可选参数，返回一个Tensor类型的输出
@_overload
def upsample(
    input: Tensor,
    size: Optional[List[int]] = None,
    scale_factor: Optional[float] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
) -> Tensor:
    pass

# noqa: F811
# 定义了实际的upsample函数，接受一个Tensor类型的输入和几个可选参数，返回一个Tensor类型的输出
def upsample(
    input,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
):
    r"""Upsample input.

    Provided tensor is upsampled to either the given :attr:`size` or the given
    :attr:`scale_factor`

    .. warning::
        This function is deprecated in favor of :func:`torch.nn.functional.interpolate`.
        This is equivalent with ``nn.functional.interpolate(...)``.

    Note:
        {backward_reproducibility_note}

    The algorithm used for upsampling is determined by :attr:`mode`.

    Currently temporal, spatial and volumetric upsampling are supported, i.e.
    expected inputs are 3-D, 4-D or 5-D in shape.

    The input dimensions are interpreted in the form:
    `mini-batch x channels x [optional depth] x [optional height] x width`.

    The modes available for upsampling are: `nearest`, `linear` (3D-only),
    `bilinear`, `bicubic` (4D-only), `trilinear` (5D-only)

    Args:
        input (Tensor): the input tensor
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]):
            output spatial size.
        scale_factor (float or Tuple[float]): multiplier for spatial size. Has to match input size if it is a tuple.
        mode (str): algorithm used for upsampling:
            ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
            ``'trilinear'``. Default: ``'nearest'``
        align_corners (bool, optional): Geometrically, we consider the pixels of the
            input and output as squares rather than points.
            If set to ``True``, the input and output tensors are aligned by the
            center points of their corner pixels, preserving the values at the corner pixels.
            If set to ``False``, the input and output tensors are aligned by the corner
            points of their corner pixels, and the interpolation uses edge value padding
            for out-of-boundary values, making this operation *independent* of input size
            when :attr:`scale_factor` is kept the same. This only has an effect when :attr:`mode`
            is ``'linear'``, ``'bilinear'``, ``'bicubic'`` or ``'trilinear'``.
            Default: ``False``

    .. note::
        With ``mode='bicubic'``, it's possible to cause overshoot, in other words it can produce
        negative values or values greater than 255 for images.
        Explicitly call ``result.clamp(min=0, max=255)`` if you want to reduce the overshoot
        when displaying the image.
    """
        .. warning::
            当 ``align_corners = True`` 时，线性插值模式（`linear`，`bilinear` 和 `trilinear`）
            不会按比例对齐输出和输入像素，因此输出值可能依赖于输入大小。这是这些模式在版本
            0.3.1 之前的默认行为。从那时起，默认行为是 ``align_corners = False``。
            请参考 :class:`~torch.nn.Upsample` 获取关于如何影响输出的具体示例。
    
        """
        发出警告，提醒用户 `nn.functional.upsample` 已弃用。
        建议改用 `nn.functional.interpolate`。
    
        # 返回使用 interpolate 函数处理后的结果
        return interpolate(input, size, scale_factor, mode, align_corners)
if upsample.__doc__:
    upsample.__doc__ = upsample.__doc__.format(**reproducibility_notes)



# 如果 upsample 函数有文档字符串，则使用 reproducibility_notes 中的内容格式化该文档字符串
if upsample.__doc__:
    upsample.__doc__ = upsample.__doc__.format(**reproducibility_notes)



def _is_integer(x) -> bool:
    r"""Type check the input number is an integer.

    Will return True for int, SymInt, Numpy integers and Tensors with integer elements.
    """
    # 检查输入的 x 是否为整数类型，包括 int、torch.SymInt、Numpy 整数和具有整数元素的 Tensors
    if isinstance(x, (int, torch.SymInt)):
        return True
    if np is not None and isinstance(x, np.integer):
        return True
    # 如果是 Tensor 类型且不是浮点数类型，则返回 True
    return isinstance(x, Tensor) and not x.is_floating_point()



@_overload
def interpolate(  # noqa: F811
    input: Tensor,
    size: Optional[int] = None,
    scale_factor: Optional[List[float]] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None,
    antialias: bool = False,
) -> Tensor:  # noqa: B950
    pass



@_overload
def interpolate(  # noqa: F811
    input: Tensor,
    size: Optional[List[int]] = None,
    scale_factor: Optional[List[float]] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None,
    antialias: bool = False,
) -> Tensor:  # noqa: B950
    pass



@_overload
def interpolate(  # noqa: F811
    input: Tensor,
    size: Optional[int] = None,
    scale_factor: Optional[float] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None,
    antialias: bool = False,
) -> Tensor:  # noqa: B950
    pass



@_overload
def interpolate(  # noqa: F811
    input: Tensor,
    size: Optional[List[int]] = None,
    scale_factor: Optional[float] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None,
    antialias: bool = False,
) -> Tensor:
    pass



def interpolate(  # noqa: F811
    input: Tensor,
    size: Optional[int] = None,
    scale_factor: Optional[List[float]] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None,
    antialias: bool = False,
) -> Tensor:  # noqa: B950
    r"""Down/up samples the input.

    Tensor interpolated to either the given :attr:`size` or the given
    :attr:`scale_factor`

    The algorithm used for interpolation is determined by :attr:`mode`.

    Currently temporal, spatial and volumetric sampling are supported, i.e.
    expected inputs are 3-D, 4-D or 5-D in shape.

    The input dimensions are interpreted in the form:
    `mini-batch x channels x [optional depth] x [optional height] x width`.

    The modes available for resizing are: `nearest`, `linear` (3D-only),
    `bilinear`, `bicubic` (4D-only), `trilinear` (5D-only), `area`, `nearest-exact`
    """



# 对输入进行上/下采样操作

def interpolate(  # noqa: F811
    input: Tensor,
    size: Optional[int] = None,
    scale_factor: Optional[List[float]] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None,
    antialias: bool = False,
) -> Tensor:  # noqa: B950
    r"""Down/up samples the input.

    Tensor interpolated to either the given :attr:`size` or the given
    :attr:`scale_factor`

    The algorithm used for interpolation is determined by :attr:`mode`.

    Currently temporal, spatial and volumetric sampling are supported, i.e.
    expected inputs are 3-D, 4-D or 5-D in shape.

    The input dimensions are interpreted in the form:
    `mini-batch x channels x [optional depth] x [optional height] x width`.

    The modes available for resizing are: `nearest`, `linear` (3D-only),
    `bilinear`, `bicubic` (4D-only), `trilinear` (5D-only), `area`, `nearest-exact`
    """
    Args:
        input (Tensor): 输入张量
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]):
            输出空间尺寸。
        scale_factor (float or Tuple[float]): 空间尺寸的乘数。如果 `scale_factor` 是元组，
            其长度必须与空间维度数匹配；即 `input.dim() - 2`。
        mode (str): 上采样算法:
            ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
            ``'trilinear'`` | ``'area'`` | ``'nearest-exact'``。默认为 ``'nearest'``
        align_corners (bool, optional): 在几何上，将输入和输出视为方块而不是点。
            如果设置为 ``True``，则通过它们角像素的中心点对齐输入和输出张量，保留角像素的值。
            如果设置为 ``False``，则通过它们角像素的角点对齐输入和输出张量，并且在超出边界值时使用边缘值填充进行插值，
            使得该操作在保持相同 :attr:`scale_factor` 时 *与输入尺寸无关*。只有当 :attr:`mode`
            为 ``'linear'``、``'bilinear'``、``'bicubic'`` 或 ``'trilinear'`` 时才有效。
            默认为 ``False``
        recompute_scale_factor (bool, optional): 重新计算用于插值计算的 `scale_factor`。
            如果 `recompute_scale_factor` 为 ``True``，则必须传入 `scale_factor`，并且 `scale_factor` 用于计算输出 `size`。
            计算得到的输出 `size` 将用于推断插值的新比例。注意，当 `scale_factor` 是浮点数时，由于四舍五入和精度问题，它可能与重新计算的 `scale_factor` 不同。
            如果 `recompute_scale_factor` 为 ``False``，则将直接使用 `size` 或 `scale_factor` 进行插值。默认为 ``None``。
        antialias (bool, optional): 应用抗锯齿的标志。默认为 ``False``。与 ``align_corners=False`` 一起使用抗锯齿选项，
            插值结果将与 Pillow 在缩小操作上的结果匹配。支持的模式有： ``'bilinear'``、``'bicubic'``。

    .. note::
        使用 ``mode='bicubic'`` 时，可能会造成过冲，换句话说，它可能会产生负值或大于 255 的值用于图像。
        如果想要减少显示图像时的过冲，可以显式调用 ``result.clamp(min=0, max=255)``。

    .. note::
        模式 ``mode='nearest-exact'`` 与 Scikit-Image 和 PIL 的最近邻插值算法匹配，并修复了 ``mode='nearest'`` 的已知问题。
        引入此模式是为了保持向后兼容性。
        模式 ``mode='nearest'`` 与有缺陷的 OpenCV 的 ``INTER_NEAREST`` 插值算法匹配。
    # 如果输入具有 torch 函数的一元操作，调用 handle_torch_function 处理
    if has_torch_function_unary(input):
        return handle_torch_function(
            interpolate,
            (input,),
            input,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor,
            antialias=antialias,
        )

    # 如果插值模式是 "nearest", "area", "nearest-exact"，则检查 align_corners 是否为 None
    if mode in ("nearest", "area", "nearest-exact"):
        if align_corners is not None:
            raise ValueError(
                "align_corners option can only be set with the "
                "interpolating modes: linear | bilinear | bicubic | trilinear"
            )
    else:
        # 如果插值模式不是上述几种，确保 align_corners 不为 None
        if align_corners is None:
            align_corners = False

    # 计算输入张量的空间维度数量
    dim = input.dim() - 2  # Number of spatial dimensions.

    # 处理 size 和 scale_factor 参数，确保只有一个被定义
    # 如果同时定义了 size 和 scale_factor，则抛出 ValueError 异常
    if size is not None and scale_factor is not None:
        raise ValueError("only one of size or scale_factor should be defined")
    elif size is not None:
        assert scale_factor is None
        scale_factors = None
        # 如果 size 是列表或元组，则验证其长度与空间维度数量是否一致
        if isinstance(size, (list, tuple)):
            if len(size) != dim:
                raise ValueError(
                    "Input and output must have the same number of spatial dimensions, but got "
                    f"input with spatial dimensions of {list(input.shape[2:])} and output size of {size}. "
                    "Please provide input tensor in (N, C, d1, d2, ...,dK) format and "
                    "output size in (o1, o2, ...,oK) format."
                )
            # 如果不是脚本模式，则验证 size 中的每个元素是否为整数
            if not torch.jit.is_scripting():
                if not all(_is_integer(x) for x in size):
                    raise TypeError(
                        "expected size to be one of int or Tuple[int] or Tuple[int, int] or "
                        f"Tuple[int, int, int], but got size with types {[type(x) for x in size]}"
                    )
            output_size = size
        else:
            # 如果 size 是标量，则将其扩展为与空间维度数量一致的列表
            output_size = [size for _ in range(dim)]
    # 如果 scale_factor 不为空，则 size 必须为空，因为两者不能同时指定
    elif scale_factor is not None:
        assert size is None
        output_size = None
        # 如果 scale_factor 是列表或元组，则其长度必须与空间维度 dim 相同
        if isinstance(scale_factor, (list, tuple)):
            if len(scale_factor) != dim:
                raise ValueError(
                    "Input and scale_factor must have the same number of spatial dimensions, but "
                    f"got input with spatial dimensions of {list(input.shape[2:])} and "
                    f"scale_factor of shape {scale_factor}. "
                    "Please provide input tensor in (N, C, d1, d2, ...,dK) format and "
                    "scale_factor in (s1, s2, ...,sK) format."
                )
            scale_factors = scale_factor
        else:
            # 如果 scale_factor 是单个值，则将其重复 dim 次形成列表
            scale_factors = [scale_factor for _ in range(dim)]
    else:
        # 如果既没有指定 size 也没有指定 scale_factor，则抛出异常
        raise ValueError("either size or scale_factor should be defined")

    if (
        recompute_scale_factor is not None
        and recompute_scale_factor
        and size is not None
    ):
        # 如果 recompute_scale_factor 为 True，且同时指定了 size，则抛出异常
        raise ValueError(
            "recompute_scale_factor is not meaningful with an explicit size."
        )

    # 对于 "area" 模式，始终需要明确的 size 而非 scale_factor
    # 重用 recompute_scale_factor 的代码路径
    if mode == "area" and output_size is None:
        recompute_scale_factor = True

    if recompute_scale_factor is not None and recompute_scale_factor:
        # 在这里计算 output_size，然后取消设置 scale_factors
        assert scale_factors is not None
        if not torch.jit.is_scripting() and torch._C._get_tracing_state():
            # 在追踪模式下，将 scale_factor 转换为张量，以避免常数被内联
            output_size = [
                (
                    torch.floor(
                        (
                            input.size(i + 2).float()
                            * torch.tensor(scale_factors[i], dtype=torch.float32)
                        ).float()
                    )
                )
                for i in range(dim)
            ]
        elif torch.jit.is_scripting():
            # 在脚本化模式下，计算 output_size 的每个维度
            output_size = [
                int(math.floor(float(input.size(i + 2)) * scale_factors[i]))
                for i in range(dim)
            ]
        else:
            # 在其他情况下，使用符号整数函数计算每个维度的 output_size
            output_size = [
                _sym_int(input.size(i + 2) * scale_factors[i]) for i in range(dim)
            ]
        scale_factors = None

    if antialias and not (mode in ("bilinear", "bicubic") and input.ndim == 4):
        # 如果启用了反锯齿选项，并且模式不是双线性或双三次，并且输入张量维度不为 4，则抛出异常
        raise ValueError(
            "Anti-alias option is restricted to bilinear and bicubic modes and requires a 4-D tensor as input"
        )

    if input.dim() == 3 and mode == "nearest":
        # 如果输入张量维度为 3，且模式为 "nearest"，则调用最近邻插值函数处理
        return torch._C._nn.upsample_nearest1d(input, output_size, scale_factors)
    if input.dim() == 4 and mode == "nearest":
        # 如果输入张量维度为 4，且模式为 "nearest"，则调用最近邻插值函数处理
        return torch._C._nn.upsample_nearest2d(input, output_size, scale_factors)
    # 检查输入张量的维度是否为5，并且插值模式为"nearest"
    if input.dim() == 5 and mode == "nearest":
        # 调用 Torch C++ 扩展函数进行3维最近邻上采样
        return torch._C._nn.upsample_nearest3d(input, output_size, scale_factors)

    # 检查输入张量的维度是否为3，并且插值模式为"nearest-exact"
    if input.dim() == 3 and mode == "nearest-exact":
        # 调用 Torch C++ 扩展函数进行1维最近邻精确上采样
        return torch._C._nn._upsample_nearest_exact1d(input, output_size, scale_factors)
    # 检查输入张量的维度是否为4，并且插值模式为"nearest-exact"
    if input.dim() == 4 and mode == "nearest-exact":
        # 调用 Torch C++ 扩展函数进行2维最近邻精确上采样
        return torch._C._nn._upsample_nearest_exact2d(input, output_size, scale_factors)
    # 检查输入张量的维度是否为5，并且插值模式为"nearest-exact"
    if input.dim() == 5 and mode == "nearest-exact":
        # 调用 Torch C++ 扩展函数进行3维最近邻精确上采样
        return torch._C._nn._upsample_nearest_exact3d(input, output_size, scale_factors)

    # 检查输入张量的维度是否为3，并且插值模式为"area"
    if input.dim() == 3 and mode == "area":
        # 断言输出大小不为None，然后调用自适应平均池化函数对1维输入进行池化
        assert output_size is not None
        return adaptive_avg_pool1d(input, output_size)
    # 检查输入张量的维度是否为4，并且插值模式为"area"
    if input.dim() == 4 and mode == "area":
        # 断言输出大小不为None，然后调用自适应平均池化函数对2维输入进行池化
        assert output_size is not None
        return adaptive_avg_pool2d(input, output_size)
    # 检查输入张量的维度是否为5，并且插值模式为"area"
    if input.dim() == 5 and mode == "area":
        # 断言输出大小不为None，然后调用自适应平均池化函数对3维输入进行池化
        assert output_size is not None
        return adaptive_avg_pool3d(input, output_size)

    # 检查输入张量的维度是否为3，并且插值模式为"linear"
    if input.dim() == 3 and mode == "linear":
        # 断言对齐角点参数不为None，然后调用线性插值函数对1维输入进行插值
        assert align_corners is not None
        return torch._C._nn.upsample_linear1d(
            input, output_size, align_corners, scale_factors
        )

    # 检查输入张量的维度是否为4，并且插值模式为"bilinear"
    if input.dim() == 4 and mode == "bilinear":
        # 断言对齐角点参数不为None
        assert align_corners is not None
        # 如果开启抗锯齿，则调用带抗锯齿的双线性插值函数
        if antialias:
            return torch._C._nn._upsample_bilinear2d_aa(
                input, output_size, align_corners, scale_factors
            )
        # 在 TorchScript 环境下，需要两级防止触及 are_deterministic_algorithms_enabled
        if not torch.jit.is_scripting():
            # 如果确定性算法已启用且输入在CUDA上，则使用慢速的分解方法
            if torch.are_deterministic_algorithms_enabled() and input.is_cuda:
                # 使用 importlib 导入 torch._decomp.decompositions 模块进行线性插值
                return importlib.import_module(
                    "torch._decomp.decompositions"
                )._upsample_linear_vec(input, output_size, align_corners, scale_factors)
        # 否则调用普通的双线性插值函数
        return torch._C._nn.upsample_bilinear2d(
            input, output_size, align_corners, scale_factors
        )

    # 检查输入张量的维度是否为5，并且插值模式为"trilinear"
    if input.dim() == 5 and mode == "trilinear":
        # 断言对齐角点参数不为None，然后调用三线性插值函数
        assert align_corners is not None
        return torch._C._nn.upsample_trilinear3d(
            input, output_size, align_corners, scale_factors
        )

    # 检查输入张量的维度是否为4，并且插值模式为"bicubic"
    if input.dim() == 4 and mode == "bicubic":
        # 断言对齐角点参数不为None
        assert align_corners is not None
        # 如果开启抗锯齿，则调用带抗锯齿的双三次插值函数
        if antialias:
            return torch._C._nn._upsample_bicubic2d_aa(
                input, output_size, align_corners, scale_factors
            )
        # 否则调用普通的双三次插值函数
        return torch._C._nn.upsample_bicubic2d(
            input, output_size, align_corners, scale_factors
        )

    # 检查输入张量的维度是否为3，并且插值模式为"bilinear"，这种情况下不支持3D输入
    if input.dim() == 3 and mode == "bilinear":
        # 抛出未实现错误，因为双线性插值模式需要4维输入
        raise NotImplementedError("Got 3D input, but bilinear mode needs 4D input")
    # 检查输入张量的维度和插值模式是否匹配，如果不匹配则抛出未实现的错误
    if input.dim() == 3 and mode == "trilinear":
        raise NotImplementedError("Got 3D input, but trilinear mode needs 5D input")
    
    # 检查输入张量的维度和插值模式是否匹配，如果不匹配则抛出未实现的错误
    if input.dim() == 4 and mode == "linear":
        raise NotImplementedError("Got 4D input, but linear mode needs 3D input")
    
    # 检查输入张量的维度和插值模式是否匹配，如果不匹配则抛出未实现的错误
    if input.dim() == 4 and mode == "trilinear":
        raise NotImplementedError("Got 4D input, but trilinear mode needs 5D input")
    
    # 检查输入张量的维度和插值模式是否匹配，如果不匹配则抛出未实现的错误
    if input.dim() == 5 and mode == "linear":
        raise NotImplementedError("Got 5D input, but linear mode needs 3D input")
    
    # 检查输入张量的维度和插值模式是否匹配，如果不匹配则抛出未实现的错误
    if input.dim() == 5 and mode == "bilinear":
        raise NotImplementedError("Got 5D input, but bilinear mode needs 4D input")

    # 如果前面的所有条件都不满足，抛出未实现的错误，显示详细的错误消息
    raise NotImplementedError(
        "Input Error: Only 3D, 4D and 5D input Tensors supported"
        f" (got {input.dim()}D) for the modes: nearest | linear | bilinear | bicubic | trilinear | area | nearest-exact"
        f" (got {mode})"
    )
# 如果 interpolate 函数有文档字符串，则使用 reproducibility_notes 格式化它
if interpolate.__doc__:
    interpolate.__doc__ = interpolate.__doc__.format(**reproducibility_notes)


@_overload
def upsample_nearest(  # noqa: F811
    input: Tensor,
    size: Optional[int] = None,
    scale_factor: Optional[float] = None,
) -> Tensor:
    pass


@_overload
def upsample_nearest(  # noqa: F811
    input: Tensor,
    size: Optional[List[int]] = None,
    scale_factor: Optional[float] = None,
) -> Tensor:
    pass


# 定义 upsample_nearest 函数，对输入进行最近邻上采样
def upsample_nearest(input, size=None, scale_factor=None):  # noqa: F811
    r"""Upsamples the input, using nearest neighbours' pixel values.

    .. warning::
        This function is deprecated in favor of :func:`torch.nn.functional.interpolate`.
        This is equivalent with ``nn.functional.interpolate(..., mode='nearest')``.

    Currently spatial and volumetric upsampling are supported (i.e. expected
    inputs are 4 or 5 dimensional).

    Args:
        input (Tensor): input
        size (int or Tuple[int, int] or Tuple[int, int, int]): output spatial size.
        scale_factor (int): multiplier for spatial size. Has to be an integer.

    Note:
        {backward_reproducibility_note}
    """
    # 发出警告，提示该函数已被弃用
    warnings.warn(
        "`nn.functional.upsample_nearest` is deprecated. "
        "Use `nn.functional.interpolate` instead.",
        stacklevel=2,
    )
    # 调用 interpolate 函数进行最近邻插值
    return interpolate(input, size, scale_factor, mode="nearest")


# 如果 upsample_nearest 函数有文档字符串，则使用 reproducibility_notes 格式化它
if upsample_nearest.__doc__:
    upsample_nearest.__doc__ = upsample_nearest.__doc__.format(**reproducibility_notes)


@_overload
def upsample_bilinear(  # noqa: F811
    input: Tensor,
    size: Optional[int] = None,
    scale_factor: Optional[float] = None,
) -> Tensor:
    pass


@_overload
def upsample_bilinear(  # noqa: F811
    input: Tensor,
    size: Optional[List[int]] = None,
    scale_factor: Optional[float] = None,
) -> Tensor:
    pass


@_overload
def upsample_bilinear(  # noqa: F811
    input: Tensor,
    size: Optional[int] = None,
    scale_factor: Optional[List[float]] = None,
) -> Tensor:
    pass


@_overload
def upsample_bilinear(  # noqa: F811
    input: Tensor,
    size: Optional[List[int]] = None,
    scale_factor: Optional[List[float]] = None,
) -> Tensor:
    pass


# 定义 upsample_bilinear 函数，对输入进行双线性上采样
def upsample_bilinear(input, size=None, scale_factor=None):  # noqa: F811
    r"""Upsamples the input, using bilinear upsampling.

    .. warning::
        This function is deprecated in favor of :func:`torch.nn.functional.interpolate`.
        This is equivalent with
        ``nn.functional.interpolate(..., mode='bilinear', align_corners=True)``.

    Expected inputs are spatial (4 dimensional). Use `upsample_trilinear` for
    volumetric (5 dimensional) inputs.

    Args:
        input (Tensor): input
        size (int or Tuple[int, int]): output spatial size.
        scale_factor (int or Tuple[int, int]): multiplier for spatial size

    Note:
        {backward_reproducibility_note}
    """
    # DeprecationWarning 被默认忽略
    # 发出警告，提示 `nn.functional.upsample_bilinear` 已弃用，建议使用 `nn.functional.interpolate` 替代
    warnings.warn(
        "`nn.functional.upsample_bilinear` is deprecated. "
        "Use `nn.functional.interpolate` instead.",
        stacklevel=2,
    )
    # 调用 interpolate 函数，进行双线性插值操作
    return interpolate(input, size, scale_factor, mode="bilinear", align_corners=True)
# 如果 upsample_bilinear 函数有文档字符串，则格式化该文档字符串以包含 reproducibility_notes 中的内容
if upsample_bilinear.__doc__:
    upsample_bilinear.__doc__ = upsample_bilinear.__doc__.format(
        **reproducibility_notes
    )

# 定义一个字典，将插值模式名称映射到整数值
GRID_SAMPLE_INTERPOLATION_MODES = {
    "bilinear": 0,   # 双线性插值模式
    "nearest": 1,    # 最近邻插值模式
    "bicubic": 2,    # 双三次插值模式
}

# 定义一个字典，将填充模式名称映射到整数值
GRID_SAMPLE_PADDING_MODES = {
    "zeros": 0,          # 用零填充模式
    "border": 1,         # 用边界值填充模式
    "reflection": 2,     # 用反射值填充模式
}

# 定义 grid_sample 函数，用于计算网格采样
def grid_sample(
    input: Tensor,
    grid: Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: Optional[bool] = None,
) -> Tensor:
    r"""Compute grid sample.

    给定输入张量 input 和流场 grid，使用 input 值和 grid 中的像素位置计算输出。

    目前仅支持空间 (4-D) 和体积 (5-D) 输入。

    在空间 (4-D) 情况下，对于形状为 (N, C, H_in, W_in) 的 input 和形状为 (N, H_out, W_out, 2) 的 grid，
    输出将具有形状 (N, C, H_out, W_out)。

    对于每个输出位置 output[n, :, h, w]，大小为 2 的向量 grid[n, h, w] 指定了用于插值输出值 output[n, :, h, w] 的 input 像素位置 x 和 y。
    在 5D 输入情况下，grid[n, d, h, w] 指定了插值 output[n, :, d, h, w] 的 x、y、z 像素位置。mode 参数指定了最近邻或双线性插值方法来采样输入像素。

    grid 指定了由 input 空间尺寸归一化的采样像素位置。因此，它的大多数值应在 [-1, 1] 范围内。
    例如，值 x = -1, y = -1 是 input 的左上像素，值 x = 1, y = 1 是 input 的右下像素。

    如果 grid 的值超出 [-1, 1] 范围，则相应的输出由 padding_mode 定义处理。
    选项包括：

        * "padding_mode='zeros'"：对于超出边界的 grid 位置使用 0，
        * "padding_mode='border'"：对于超出边界的 grid 位置使用边界值，
        * "padding_mode='reflection'"：对于超出边界的 grid 位置使用边界反射的值。
          对于远离边界的位置，它将继续反射，直到变为内部位置。
          例如，(归一化) 像素位置 x = -3.5 反射到边界 -1 变为 x' = 1.5，然后反射到边界 1 变为 x'' = -0.5。

    注意：
        此函数通常与 affine_grid 结合使用，构建空间变换网络 (Spatial Transformer Networks)。

    注意：
        使用 CUDA 后端时，此操作可能在其反向传播中引入非确定性行为，难以关闭。
        请参阅有关 :doc:`/notes/randomness` 的说明了解背景信息。
        
    """
    Note:
        NaN values in :attr:`grid` would be interpreted as ``-1``.

    Args:
        input (Tensor): input of shape :math:`(N, C, H_\text{in}, W_\text{in})` (4-D case)
                        or :math:`(N, C, D_\text{in}, H_\text{in}, W_\text{in})` (5-D case)
            输入张量：形状为 :math:`(N, C, H_\text{in}, W_\text{in})`（4维情况）
                      或者 :math:`(N, C, D_\text{in}, H_\text{in}, W_\text{in})`（5维情况）
        grid (Tensor): flow-field of shape :math:`(N, H_\text{out}, W_\text{out}, 2)` (4-D case)
                       or :math:`(N, D_\text{out}, H_\text{out}, W_\text{out}, 3)` (5-D case)
            网格张量：形状为 :math:`(N, H_\text{out}, W_\text{out}, 2)`（4维情况）
                      或者 :math:`(N, D_\text{out}, H_\text{out}, W_\text{out}, 3)`（5维情况）
        mode (str): interpolation mode to calculate output values
            ``'bilinear'`` | ``'nearest'`` | ``'bicubic'``. Default: ``'bilinear'``
            插值模式：用于计算输出值的插值模式
            ``'bilinear'`` | ``'nearest'`` | ``'bicubic'``。默认值为 ``'bilinear'``
            注意：``mode='bicubic'`` 仅支持4维输入。
            当 ``mode='bilinear'`` 且输入为5维时，内部实际使用的是三线性插值（trilinear）。
            然而，当输入为4维时，内部使用的是合法的双线性插值（bilinear）。
        padding_mode (str): padding mode for outside grid values
            ``'zeros'`` | ``'border'`` | ``'reflection'``. Default: ``'zeros'``
            填充模式：用于网格外部值的填充模式
            ``'zeros'`` | ``'border'`` | ``'reflection'``。默认值为 ``'zeros'``
        align_corners (bool, optional): Geometrically, we consider the pixels of the
            input  as squares rather than points.
            If set to ``True``, the extrema (``-1`` and ``1``) are considered as referring
            to the center points of the input's corner pixels. If set to ``False``, they
            are instead considered as referring to the corner points of the input's corner
            pixels, making the sampling more resolution agnostic.
            This option parallels the ``align_corners`` option in
            :func:`interpolate`, and so whichever option is used here
            should also be used there to resize the input image before grid sampling.
            Default: ``False``
            几何上，我们将输入像素视为正方形而非点。
            如果设置为 ``True``，则极值（``-1`` 和 ``1``）被视为指向输入角落像素的中心点。
            如果设置为 ``False``，则极值被视为指向输入角落像素的角点，使得采样更加分辨率不可知。
            此选项与 :func:`interpolate` 中的 ``align_corners`` 选项并行，因此这里使用的选项
            应该与那里在网格采样之前调整输入图像大小时使用的选项相对应。
            默认值为 ``False``

    Returns:
        output (Tensor): output Tensor
            输出张量：输出张量

    .. _`Spatial Transformer Networks`:
        https://arxiv.org/abs/1506.02025

    .. warning::
        When ``align_corners = True``, the grid positions depend on the pixel
        size relative to the input image size, and so the locations sampled by
        :func:`grid_sample` will differ for the same input given at different
        resolutions (that is, after being upsampled or downsampled).
        The default behavior up to version 1.2.0 was ``align_corners = True``.
        Since then, the default behavior has been changed to ``align_corners = False``,
        in order to bring it in line with the default for :func:`interpolate`.
        警告：当 ``align_corners = True`` 时，网格位置依赖于像素大小相对于输入图像大小的比例，
        因此在不同分辨率（即经过上采样或下采样后）的相同输入上采样的位置将不同。
        到版本1.2.0为止的默认行为是 ``align_corners = True``。
        从那时起，默认行为已更改为 ``align_corners = False``，以使其与 :func:`interpolate` 的默认行为保持一致。
    """
        如果输入或者网格具有 torch 函数变异，调用处理 torch 函数
        grid_sample 函数，传入输入、网格、模式、填充模式和角对齐参数
    """
    if has_torch_function_variadic(input, grid):
        return handle_torch_function(
            grid_sample,
            (input, grid),
            input,
            grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
    
    """
        如果模式不是 'bilinear'、'nearest' 或 'bicubic'，引发值错误异常
    """
    if mode != "bilinear" and mode != "nearest" and mode != "bicubic":
        raise ValueError(
            f"nn.functional.grid_sample(): expected mode to be 'bilinear', 'nearest' or 'bicubic', but got: '{mode}'"
        )
    
    """
        如果填充模式不是 'zeros'、'border' 或 'reflection'，引发值错误异常
    """
    if (
        padding_mode != "zeros"
        and padding_mode != "border"
        and padding_mode != "reflection"
    ):
        raise ValueError(
            "nn.functional.grid_sample(): expected padding_mode "
            "to be 'zeros', 'border', or 'reflection', "
            f"but got: '{padding_mode}'"
        )
    
    """
        根据模式设置模式枚举值：'bilinear' -> 0, 'nearest' -> 1, 'bicubic' -> 2
    """
    if mode == "bilinear":
        mode_enum = 0
    elif mode == "nearest":
        mode_enum = 1
    else:  # mode == 'bicubic'
        mode_enum = 2
    
    """
        根据填充模式设置填充模式枚举值：'zeros' -> 0, 'border' -> 1, 'reflection' -> 2
    """
    if padding_mode == "zeros":
        padding_mode_enum = 0
    elif padding_mode == "border":
        padding_mode_enum = 1
    else:  # padding_mode == 'reflection'
        padding_mode_enum = 2
    
    """
        如果角对齐参数为 None，发出警告并设置角对齐参数为 False
    """
    if align_corners is None:
        warnings.warn(
            "Default grid_sample and affine_grid behavior has changed "
            "to align_corners=False since 1.3.0. Please specify "
            "align_corners=True if the old behavior is desired. "
            "See the documentation of grid_sample for details."
        )
        align_corners = False
    
    """
        调用 torch.grid_sampler 函数，传入输入、网格、模式枚举值、填充模式枚举值和角对齐参数
    """
    return torch.grid_sampler(input, grid, mode_enum, padding_mode_enum, align_corners)
# 定义一个函数，生成二维或三维的流场（采样网格），根据输入的仿射矩阵 theta 批量生成
def affine_grid(
    theta: Tensor,
    size: List[int],
    align_corners: Optional[bool] = None,
) -> Tensor:
    r"""Generate 2D or 3D flow field (sampling grid), given a batch of affine matrices :attr:`theta`.

    .. note::
        This function is often used in conjunction with :func:`grid_sample`
        to build `Spatial Transformer Networks`_ .

    Args:
        theta (Tensor): input batch of affine matrices with shape
            (:math:`N \times 2 \times 3`) for 2D or
            (:math:`N \times 3 \times 4`) for 3D
        size (torch.Size): the target output image size.
            (:math:`N \times C \times H \times W` for 2D or
            :math:`N \times C \times D \times H \times W` for 3D)
            Example: torch.Size((32, 3, 24, 24))
        align_corners (bool, optional): if ``True``, consider ``-1`` and ``1``
            to refer to the centers of the corner pixels rather than the image corners.
            Refer to :func:`grid_sample` for a more complete description.
            A grid generated by :func:`affine_grid` should be passed to :func:`grid_sample`
            with the same setting for this option.
            Default: ``False``

    Returns:
        output (Tensor): output Tensor of size (:math:`N \times H \times W \times 2`)

    .. _`Spatial Transformer Networks`:
        https://arxiv.org/abs/1506.02025

    .. warning::
        When ``align_corners = True``, the grid positions depend on the pixel
        size relative to the input image size, and so the locations sampled by
        :func:`grid_sample` will differ for the same input given at different
        resolutions (that is, after being upsampled or downsampled).
        The default behavior up to version 1.2.0 was ``align_corners = True``.
        Since then, the default behavior has been changed to ``align_corners = False``,
        in order to bring it in line with the default for :func:`interpolate`.
    .. warning::
        When ``align_corners = True``, 2D affine transforms on 1D data and
        3D affine transforms on 2D data (that is, when one of the spatial
        dimensions has unit size) are ill-defined, and not an intended use case.
        This is not a problem when ``align_corners = False``.
        Up to version 1.2.0, all grid points along a unit dimension were
        considered arbitrarily to be at ``-1``.
        From version 1.3.0, under ``align_corners = True`` all grid points
        along a unit dimension are considered to be at ``0``
        (the center of the input image).
    """
    # 检查 theta 是否有 torch 函数的一元版本，如果有则调用对应的 torch 处理函数
    if has_torch_function_unary(theta):
        # 调用 torch 处理函数，处理 affine_grid 函数的调用，保留 align_corners 参数设置
        return handle_torch_function(
            affine_grid, (theta,), theta, size, align_corners=align_corners
        )
    # 如果未指定 align_corners 参数，则发出警告并设置为 False
    if align_corners is None:
        warnings.warn(
            "Default grid_sample and affine_grid behavior has changed "
            "to align_corners=False since 1.3.0. Please specify "
            "align_corners=True if the old behavior is desired. "
            "See the documentation of grid_sample for details."
        )
        align_corners = False

    # 确保 theta 的数据类型为浮点型
    if not theta.is_floating_point():
        raise ValueError(
            f"Expected theta to have floating point type, but got {theta.dtype}"
        )

    # 检查 theta 的形状和尺寸匹配情况
    if len(size) == 4:
        if theta.dim() != 3 or theta.shape[-2] != 2 or theta.shape[-1] != 3:
            raise ValueError(
                f"Expected a batch of 2D affine matrices of shape Nx2x3 for size {size}. Got {theta.shape}."
            )
        spatial_size = size[-2:]  # 空间维度的尺寸
    elif len(size) == 5:
        if theta.dim() != 3 or theta.shape[-2] != 3 or theta.shape[-1] != 4:
            raise ValueError(
                f"Expected a batch of 3D affine matrices of shape Nx3x4 for size {size}. Got {theta.shape}."
            )
        spatial_size = size[-3:]  # 空间维度的尺寸
    else:
        raise NotImplementedError(
            "affine_grid only supports 4D and 5D sizes, "
            "for 2D and 3D affine transforms, respectively. "
            f"Got size {size}."
        )

    # 检查是否存在空跨度
    if align_corners and min(spatial_size) == 1:
        warnings.warn(
            "Since version 1.3.0, affine_grid behavior has changed "
            "for unit-size grids when align_corners=True. "
            "This is not an intended use case of affine_grid. "
            "See the documentation of affine_grid for details."
        )
    elif min(size) <= 0:
        raise ValueError(f"Expected non-zero, positive output size. Got {size}")

    # 调用 torch 库中的 affine_grid_generator 函数生成仿射网格
    return torch.affine_grid_generator(theta, size, align_corners)
# 定义一个函数 `pad`，用于对输入的张量进行填充操作
def pad(
    input: Tensor,
    pad: List[int],
    mode: str = "constant",
    value: Optional[float] = None,
) -> Tensor:
    r"""
    pad(input, pad, mode="constant", value=None) -> Tensor

    Pads tensor.

    Padding size:
        The padding size by which to pad some dimensions of :attr:`input`
        are described starting from the last dimension and moving forward.
        :math:`\left\lfloor\frac{\text{len(pad)}}{2}\right\rfloor` dimensions
        of ``input`` will be padded.
        For example, to pad only the last dimension of the input tensor, then
        :attr:`pad` has the form
        :math:`(\text{padding\_left}, \text{padding\_right})`;
        to pad the last 2 dimensions of the input tensor, then use
        :math:`(\text{padding\_left}, \text{padding\_right},`
        :math:`\text{padding\_top}, \text{padding\_bottom})`;
        to pad the last 3 dimensions, use
        :math:`(\text{padding\_left}, \text{padding\_right},`
        :math:`\text{padding\_top}, \text{padding\_bottom}`
        :math:`\text{padding\_front}, \text{padding\_back})`.

    Padding mode:
        See :class:`torch.nn.CircularPad2d`, :class:`torch.nn.ConstantPad2d`,
        :class:`torch.nn.ReflectionPad2d`, and :class:`torch.nn.ReplicationPad2d`
        for concrete examples on how each of the padding modes works. Constant
        padding is implemented for arbitrary dimensions. Circular, replicate and
        reflection padding are implemented for padding the last 3 dimensions of a
        4D or 5D input tensor, the last 2 dimensions of a 3D or 4D input tensor,
        or the last dimension of a 2D or 3D input tensor.

    Note:
        When using the CUDA backend, this operation may induce nondeterministic
        behaviour in its backward pass that is not easily switched off.
        Please see the notes on :doc:`/notes/randomness` for background.

    Args:
        input (Tensor): N-dimensional tensor
        pad (tuple): m-elements tuple, where
            :math:`\frac{m}{2} \leq` input dimensions and :math:`m` is even.
            Specifies the amount of padding added to each dimension of the input tensor.
        mode: ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
            Default: ``'constant'``
            Specifies the padding mode. 'constant' fills with a specified value, while
            other modes (reflect, replicate, circular) use different strategies for padding.
        value: fill value for ``'constant'`` padding. Default: ``0``
            The value used for padding when mode is 'constant'.

    Examples::

        >>> t4d = torch.empty(3, 3, 4, 2)
        >>> p1d = (1, 1) # pad last dim by 1 on each side
        >>> out = F.pad(t4d, p1d, "constant", 0)  # effectively zero padding
        >>> print(out.size())
        torch.Size([3, 3, 4, 4])
        >>> p2d = (1, 1, 2, 2) # pad last dim by (1, 1) and 2nd to last by (2, 2)
        >>> out = F.pad(t4d, p2d, "constant", 0)
        >>> print(out.size())
        torch.Size([3, 3, 8, 4])
        >>> t4d = torch.empty(3, 3, 4, 2)
        >>> p3d = (0, 1, 2, 1, 3, 3) # pad by (0, 1), (2, 1), and (3, 3)
        >>> out = F.pad(t4d, p3d, "constant", 0)
        >>> print(out.size())
        torch.Size([3, 9, 7, 3])
    """
    # 如果输入张量具有torch函数的单目处理方法，则调用handle_torch_function处理
    if has_torch_function_unary(input):
        # 调用torch函数的处理方法，传递参数(input,)和输入(input)，以及pad、mode和value
        return handle_torch_function(
            torch.nn.functional.pad, (input,), input, pad, mode=mode, value=value
        )
    
    # 如果当前环境不是torch脚本化环境
    if not torch.jit.is_scripting():
        # 如果启用了确定性算法且输入在CUDA上
        if torch.are_deterministic_algorithms_enabled() and input.is_cuda:
            # 如果模式是"replicate"
            if mode == "replicate":
                # 使用慢速分解，其反向传播将基于index_put。
                # importlib是必需的，因为导入不能是顶级的（存在循环依赖）且不能嵌套（TS不支持）
                return importlib.import_module(
                    "torch._decomp.decompositions"
                )._replication_pad(input, pad)
    
    # 使用C++实现的pad函数来对输入张量进行填充
    return torch._C._nn.pad(input, pad, mode, value)
# 将 pad 对象的模块属性修改为 "torch.nn.functional"，用于修复 https://github.com/pytorch/pytorch/issues/75798
pad.__module__ = "torch.nn.functional"

# 对 torch.pairwise_distance 函数添加文档字符串，描述其参数和功能
pairwise_distance = _add_docstr(
    torch.pairwise_distance,
    r"""
    pairwise_distance(x1, x2, p=2.0, eps=1e-6, keepdim=False) -> Tensor
    
    See :class:`torch.nn.PairwiseDistance` for details
    """,
)

# 对 torch.pdist 函数添加文档字符串，详细描述其参数、功能和用法
pdist = _add_docstr(
    torch.pdist,
    r"""
    pdist(input, p=2) -> Tensor
    
    Computes the p-norm distance between every pair of row vectors in the input.
    This is identical to the upper triangular portion, excluding the diagonal, of
    `torch.norm(input[:, None] - input, dim=2, p=p)`. This function will be faster
    if the rows are contiguous.
    
    If input has shape :math:`N \times M` then the output will have shape
    :math:`\frac{1}{2} N (N - 1)`.
    
    This function is equivalent to ``scipy.spatial.distance.pdist(input,
    'minkowski', p=p)`` if :math:`p \in (0, \infty)`. When :math:`p = 0` it is
    equivalent to ``scipy.spatial.distance.pdist(input, 'hamming') * M``.
    When :math:`p = \infty`, the closest scipy function is
    ``scipy.spatial.distance.pdist(xn, lambda x, y: np.abs(x - y).max())``.
    
    Args:
        input: input tensor of shape :math:`N \times M`.
        p: p value for the p-norm distance to calculate between each vector pair
            :math:`\in [0, \infty]`.
    """,
)

# 对 torch.cosine_similarity 函数添加文档字符串，详细描述其参数、功能和用法
cosine_similarity = _add_docstr(
    torch.cosine_similarity,
    r"""
    cosine_similarity(x1, x2, dim=1, eps=1e-8) -> Tensor
    
    Returns cosine similarity between ``x1`` and ``x2``, computed along dim. ``x1`` and ``x2`` must be broadcastable
    to a common shape. ``dim`` refers to the dimension in this common shape. Dimension ``dim`` of the output is
    squeezed (see :func:`torch.squeeze`), resulting in the
    output tensor having 1 fewer dimension.
    
    .. math ::
        \text{similarity} = \dfrac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert _2, \epsilon) \cdot \max(\Vert x_2 \Vert _2, \epsilon)}
    
    Supports :ref:`type promotion <type-promotion-doc>`.
    
    Args:
        x1 (Tensor): First input.
        x2 (Tensor): Second input.
        dim (int, optional): Dimension along which cosine similarity is computed. Default: 1
        eps (float, optional): Small value to avoid division by zero.
            Default: 1e-8
    
    Example::
    
        >>> input1 = torch.randn(100, 128)
        >>> input2 = torch.randn(100, 128)
        >>> output = F.cosine_similarity(input1, input2)
        >>> print(output)
    """,
)

# 对 torch._C._nn.one_hot 函数添加文档字符串，描述其参数、功能和用法
one_hot = _add_docstr(
    torch._C._nn.one_hot,
    r"""
    one_hot(tensor, num_classes=-1) -> LongTensor
    
    Takes LongTensor with index values of shape ``(*)`` and returns a tensor
    of shape ``(*, num_classes)`` that have zeros everywhere except where the
    index of last dimension matches the corresponding value of the input tensor,
    in which case it will be 1.
    
    See also `One-hot on Wikipedia`_ .
    
    .. _One-hot on Wikipedia:
        https://en.wikipedia.org/wiki/One-hot
    
    Arguments:
        tensor (LongTensor): class values of any shape.
    """,
)
    num_classes (int):  Total number of classes. If set to -1, the number
                        of classes will be inferred as one greater than the largest class
                        value in the input tensor.
# 返回一个长张量，其在由输入指示的最后一个维度的索引处具有值为1，其他位置为0的独热编码张量。

def triplet_margin_loss(
    anchor: Tensor,
    positive: Tensor,
    negative: Tensor,
    margin: float = 1.0,
    p: float = 2,
    eps: float = 1e-6,
    swap: bool = False,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    r"""Compute the triplet loss between given input tensors and a margin greater than 0.

    See :class:`~torch.nn.TripletMarginLoss` for details.
    """

    # 检查输入的张量是否支持 torch 函数的变参形式，如果支持，则调用处理 torch 函数的方法
    if has_torch_function_variadic(anchor, positive, negative):
        return handle_torch_function(
            triplet_margin_loss,
            (anchor, positive, negative),
            anchor,
            positive,
            negative,
            margin=margin,
            p=p,
            eps=eps,
            swap=swap,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
        )

    # 根据 size_average 和 reduce 参数确定计算结果的减少方式
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction_enum = _Reduction.get_enum(reduction)

    # 如果 margin 小于等于0，则抛出 ValueError 异常
    if margin <= 0:
        raise ValueError(f"margin must be greater than 0, got {margin}")

    # 调用 torch 库中的 triplet_margin_loss 函数计算三元组边缘损失
    return torch.triplet_margin_loss(
        anchor, positive, negative, margin, p, eps, swap, reduction_enum
    )


def triplet_margin_with_distance_loss(
    anchor: Tensor,
    positive: Tensor,
    negative: Tensor,
    *,
    distance_function: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    margin: float = 1.0,
    swap: bool = False,
    reduction: str = "mean",
) -> Tensor:
    r"""Compute the triplet margin loss for input tensors using a custom distance function.

    See :class:`~torch.nn.TripletMarginWithDistanceLoss` for details.
    """

    # 如果正在进行脚本化编译，则抛出 NotImplementedError
    if torch.jit.is_scripting():
        raise NotImplementedError(
            "F.triplet_margin_with_distance_loss does not support JIT scripting: "
            "functions requiring Callables cannot be scripted."
        )
    # 检查是否有torch函数的可变参数形式存在
    if has_torch_function_variadic(anchor, positive, negative):
        return handle_torch_function(
            triplet_margin_with_distance_loss,
            (anchor, positive, negative),
            anchor,
            positive,
            negative,
            distance_function=distance_function,
            margin=margin,
            swap=swap,
            reduction=reduction,
        )

    # 检查 reduction 模式的有效性
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"{reduction} is not a valid value for reduction")

    # 检查 margin 参数的有效性
    if margin <= 0:
        raise ValueError(f"margin must be greater than 0, got {margin}")

    # 检查张量的维度
    a_dim = anchor.ndim
    p_dim = positive.ndim
    n_dim = negative.ndim
    if not (a_dim == p_dim and p_dim == n_dim):
        raise RuntimeError(
            f"The anchor, positive, and negative tensors are expected to have "
            f"the same number of dimensions, but got: anchor {a_dim}D, "
            f"positive {p_dim}D, and negative {n_dim}D inputs"
        )

    # 计算损失函数
    # 如果 distance_function 为 None，则默认为 torch.pairwise_distance
    if distance_function is None:
        distance_function = torch.pairwise_distance

    # 计算 anchor 到 positive 的距离和 anchor 到 negative 的距离
    dist_pos = distance_function(anchor, positive)
    dist_neg = distance_function(anchor, negative)

    # 根据 swap 参数，如果需要交换 positive 和 anchor 的位置以计算损失
    # 详情见 "Learning shallow convolutional feature descriptors with triplet losses" 论文
    if swap:
        dist_swap = distance_function(positive, negative)
        dist_neg = torch.minimum(dist_neg, dist_swap)

    # 计算三元组损失
    loss = torch.clamp_min(margin + dist_pos - dist_neg, 0)

    # 应用 reduction 操作
    if reduction == "sum":
        return torch.sum(loss)
    elif reduction == "mean":
        return torch.mean(loss)
    else:  # reduction == "none"
        return loss
# 执行 L_p 范数规范化，对输入张量在指定维度上进行规范化
def normalize(
    input: Tensor,
    p: float = 2.0,
    dim: int = 1,
    eps: float = 1e-12,
    out: Optional[Tensor] = None,
) -> Tensor:
    r"""Perform :math:`L_p` normalization of inputs over specified dimension.

    For a tensor :attr:`input` of sizes :math:`(n_0, ..., n_{dim}, ..., n_k)`, each
    :math:`n_{dim}` -element vector :math:`v` along dimension :attr:`dim` is transformed as

    .. math::
        v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}.

    With the default arguments it uses the Euclidean norm over vectors along dimension :math:`1` for normalization.

    Args:
        input: input tensor of any shape
        p (float): the exponent value in the norm formulation. Default: 2
        dim (int or tuple of ints): the dimension to reduce. Default: 1
        eps (float): small value to avoid division by zero. Default: 1e-12
        out (Tensor, optional): the output tensor. If :attr:`out` is used, this
                                operation won't be differentiable.
    """
    # 处理 Torch 函数的多态
    if has_torch_function_variadic(input, out):
        # 调用 Torch 函数处理
        return handle_torch_function(
            normalize, (input, out), input, p=p, dim=dim, eps=eps, out=out
        )
    # 如果未指定输出张量
    if out is None:
        # 计算输入张量在指定维度上的 L_p 范数，并将其限制在最小值为 eps 的范围内
        denom = input.norm(p, dim, keepdim=True).clamp_min(eps).expand_as(input)
        # 返回归一化后的张量
        return input / denom
    else:
        # 计算输入张量在指定维度上的 L_p 范数，并将其限制在最小值为 eps 的范围内（原位操作）
        denom = input.norm(p, dim, keepdim=True).clamp_min_(eps).expand_as(input)
        # 在给定输出张量的情况下，执行归一化操作
        return torch.div(input, denom, out=out)


# 断言输入参数为整数或长度为 2 的列表，用于验证参数是否符合预期
def assert_int_or_pair(arg: List[int], arg_name: str, message: str) -> None:
    assert isinstance(arg, int) or len(arg) == 2, message.format(arg_name)


# 从批量输入张量中提取滑动的局部块
def unfold(
    input: Tensor,
    kernel_size: BroadcastingList2[int],
    dilation: BroadcastingList2[int] = 1,
    padding: BroadcastingList2[int] = 0,
    stride: BroadcastingList2[int] = 1,
) -> Tensor:
    r"""Extract sliding local blocks from a batched input tensor.

    .. warning::
        Currently, only 4-D input tensors (batched image-like tensors) are
        supported.

    .. warning::

        More than one element of the unfolded tensor may refer to a single
        memory location. As a result, in-place operations (especially ones that
        are vectorized) may result in incorrect behavior. If you need to write
        to the tensor, please clone it first.


    See :class:`torch.nn.Unfold` for details
    """
    # 处理 Torch 函数的单态
    if has_torch_function_unary(input):
        # 调用 Torch 函数处理
        return handle_torch_function(
            unfold,
            (input,),
            input,
            kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride,
        )
    # 调用底层 C 函数执行图像到列的转换操作
    return torch._C._nn.im2col(
        input, _pair(kernel_size), _pair(dilation), _pair(padding), _pair(stride)
    )


# 将局部块折叠回批量输入张量中
def fold(
    input: Tensor,
    output_size: BroadcastingList2[int],
    kernel_size: BroadcastingList2[int],
    dilation: BroadcastingList2[int] = 1,
    padding: BroadcastingList2[int] = 0,
    stride: BroadcastingList2[int] = 1,
) -> Tensor:
    """
    Combine an array of sliding local blocks into a large containing tensor.
    
    .. warning::
        Currently, only unbatched (3D) or batched (4D) image-like output tensors are supported.
    
    See :class:`torch.nn.Fold` for details
    """
    # 检查输入是否具有 Torch 函数的一元版本，如果是则调用处理 Torch 函数的方法
    if has_torch_function_unary(input):
        # 调用处理 Torch 函数的方法来处理 fold 函数，返回处理后的结果
        return handle_torch_function(
            fold,
            (input,),                  # 传入 fold 函数的参数元组，只有一个 input
            input,                     # 输入参数
            output_size,               # 输出大小
            kernel_size,               # 卷积核大小
            dilation=dilation,         # 扩张大小
            padding=padding,           # 填充大小
            stride=stride,             # 步幅大小
        )
    # 调用 Torch 库中的 C++ 实现，将输入数据转换为输出尺寸的图像
    return torch._C._nn.col2im(
        input,                         # 输入数据
        _pair(output_size),            # 输出尺寸转换为对应的元组形式
        _pair(kernel_size),            # 卷积核大小转换为对应的元组形式
        _pair(dilation),               # 扩张大小转换为对应的元组形式
        _pair(padding),                # 填充大小转换为对应的元组形式
        _pair(stride),                 # 步幅大小转换为对应的元组形式
    )
# 定义一个函数用于执行多头注意力机制中的投影步骤，使用打包的权重
def _in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
) -> List[Tensor]:
    r"""Perform the in-projection step of the attention operation, using packed weights.

    Output is a triple containing projection tensors for query, key and value.

    Args:
        q, k, v: query, key and value tensors to be projected. For self-attention,
            these are typically the same tensor; for encoder-decoder attention,
            k and v are typically the same tensor. (We take advantage of these
            identities for performance if they are present.) Regardless, q, k and v
            must share a common embedding dimension; otherwise their shapes may vary.
        w: projection weights for q, k and v, packed into a single tensor. Weights
            are packed along dimension 0, in q, k, v order.
        b: optional projection biases for q, k and v, packed into a single tensor
            in q, k, v order.

    Shape:
        Inputs:
        - q: :math:`(..., E)` where E is the embedding dimension
        - k: :math:`(..., E)` where E is the embedding dimension
        - v: :math:`(..., E)` where E is the embedding dimension
        - w: :math:`(E * 3, E)` where E is the embedding dimension
        - b: :math:`E * 3` where E is the embedding dimension

        Output:
        - in output list :math:`[q', k', v']`, each output tensor will have the
            same shape as the corresponding input tensor.
    """
    # 获取嵌入维度 E
    E = q.size(-1)
    if k is v:
        if q is k:
            # 如果是自注意力机制
            # 对查询进行线性投影
            proj = linear(q, w, b)
            # 将结果重新整形为 3, E 的形状，而不是 E, 3，这是为了更好地内存合并和保持与 chunk() 相同的顺序
            proj = (
                proj.unflatten(-1, (3, E))
                .unsqueeze(0)
                .transpose(0, -2)
                .squeeze(-2)
                .contiguous()
            )
            return proj[0], proj[1], proj[2]
        else:
            # 如果是编码器-解码器注意力机制
            # 拆分权重 w 为查询和键值对应的部分 w_q 和 w_kv
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                # 拆分偏置 b 为查询和键值对应的部分 b_q 和 b_kv
                b_q, b_kv = b.split([E, E * 2])
            # 对查询和键分别进行线性投影
            q_proj = linear(q, w_q, b_q)
            kv_proj = linear(k, w_kv, b_kv)
            # 将结果重新整形为 2, E 的形状，而不是 E, 2，这是为了更好地内存合并和保持与 chunk() 相同的顺序
            kv_proj = (
                kv_proj.unflatten(-1, (2, E))
                .unsqueeze(0)
                .transpose(0, -2)
                .squeeze(-2)
                .contiguous()
            )
            return (q_proj, kv_proj[0], kv_proj[1])
    else:
        # 如果查询、键和值都不是同一个张量
        # 拆分权重 w 为查询、键、值对应的部分 w_q、w_k 和 w_v
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            # 拆分偏置 b 为查询、键、值对应的部分 b_q、b_k 和 b_v
            b_q, b_k, b_v = b.chunk(3)
        # 分别对查询、键、值进行线性投影
        return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    b_q: Optional[Tensor] = None,
    b_k: Optional[Tensor] = None,
    b_v: Optional[Tensor] = None,


    # 定义神经网络中的权重张量 w_q, w_k, w_v，它们分别用于查询、键、值的线性变换
    # 可选的偏置张量 b_q, b_k, b_v，默认为 None，用于神经网络中的查询、键、值的线性变换
def in_projection_attention(
    q, k, v, w_q, w_k, w_v, b_q=None, b_k=None, b_v=None
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""Perform the in-projection step of the attention operation.

    This is simply a triple of linear projections,
    with shape constraints on the weights which
    ensure embedding dimension uniformity in the projected outputs.
    Output is a triple containing projection tensors for query, key and value.

    Args:
        q, k, v: query, key and value tensors to be projected.
        w_q, w_k, w_v: weights for q, k and v, respectively.
        b_q, b_k, b_v: optional biases for q, k and v, respectively.

    Shape:
        Inputs:
        - q: :math:`(Qdims..., Eq)` where Eq is the query embedding dimension and Qdims are any
            number of leading dimensions.
        - k: :math:`(Kdims..., Ek)` where Ek is the key embedding dimension and Kdims are any
            number of leading dimensions.
        - v: :math:`(Vdims..., Ev)` where Ev is the value embedding dimension and Vdims are any
            number of leading dimensions.
        - w_q: :math:`(Eq, Eq)`
        - w_k: :math:`(Eq, Ek)`
        - w_v: :math:`(Eq, Ev)`
        - b_q: :math:`(Eq)`
        - b_k: :math:`(Eq)`
        - b_v: :math:`(Eq)`

        Output: in output triple :math:`(q', k', v')`,
         - q': :math:`[Qdims..., Eq]`
         - k': :math:`[Kdims..., Eq]`
         - v': :math:`[Vdims..., Eq]`

    """
    # Extract embedding dimensions from input tensors
    Eq, Ek, Ev = q.size(-1), k.size(-1), v.size(-1)

    # Validate shapes of weight tensors
    assert w_q.shape == (
        Eq,
        Eq,
    ), f"expecting query weights shape of {(Eq, Eq)}, but got {w_q.shape}"
    assert w_k.shape == (
        Eq,
        Ek,
    ), f"expecting key weights shape of {(Eq, Ek)}, but got {w_k.shape}"
    assert w_v.shape == (
        Eq,
        Ev,
    ), f"expecting value weights shape of {(Eq, Ev)}, but got {w_v.shape}"

    # Validate shapes of bias tensors if provided
    assert b_q is None or b_q.shape == (
        Eq,
    ), f"expecting query bias shape of {(Eq,)}, but got {b_q.shape}"
    assert b_k is None or b_k.shape == (
        Eq,
    ), f"expecting key bias shape of {(Eq,)}, but got {b_k.shape}"
    assert b_v is None or b_v.shape == (
        Eq,
    ), f"expecting value bias shape of {(Eq,)}, but got {b_v.shape}"

    # Perform linear projections using provided weights and biases
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)
    # 定义缩放点积注意力函数，输入query, key, value张量，返回注意力权重张量
    def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
        # 获取query和key张量的长度L和S
        L, S = query.size(-2), key.size(-2)
        # 计算缩放因子，如果未指定则为特征维度的倒数平方根
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        # 创建全零注意力偏置张量，形状为(L, S)，数据类型与query相同
        attn_bias = torch.zeros(L, S, dtype=query.dtype)
        
        # 如果是因果注意力，确保没有提供attn_mask
        if is_causal:
            assert attn_mask is None
            # 创建下三角形式的临时掩码张量，形状为(L, S)，仅在对角线及以下为True
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            # 将不符合条件的部分填充为负无穷大，用于掩盖未来信息
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)  # 确保数据类型一致
        
        # 如果存在attn_mask，则根据其类型进行填充
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                # 将掩码为False的位置填充为负无穷大
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                # 直接加上给定的attn_mask张量
                attn_bias += attn_mask
        
        # 计算注意力权重，query与key的转置相乘并乘以缩放因子
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        # 加上注意力偏置
        attn_weight += attn_bias
        # 对注意力权重进行softmax归一化
        attn_weight = torch.softmax(attn_weight, dim=-1)
        # 对注意力权重进行dropout操作
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        # 返回注意力权重与value张量的乘积作为最终的注意力值
        return attn_weight @ value
# 警告：此函数处于测试阶段，可能会有更改。

# 此函数始终根据指定的 dropout_p 参数应用 dropout。
# 在评估过程中禁用 dropout，请确保在调用该函数的模块不处于训练模式时传递值为 0.0。

# 示例：
# 创建自定义神经网络模型 MyModel，初始化时可以指定 dropout 概率 p，默认为 0.5。
class MyModel(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    # 前向传播方法，根据当前模式返回 scaled dot product attention 的结果
    def forward(self, ...):
        return F.scaled_dot_product_attention(..., dropout_p=(self.p if self.training else 0.0))

# 提示：
# 当前有三种支持的 scaled dot product attention 实现方式：
# - FlashAttention-2: 更快速、更好的并行性和工作分区的注意力机制
# - Memory-Efficient Attention: 内存高效的注意力机制
# - 一个在 C++ 中定义的 PyTorch 实现，与上述公式匹配

# 函数在使用 CUDA 后端时可能调用优化的内核以提升性能。
# 对于所有其他后端，将使用 PyTorch 实现。

# 所有实现默认启用。Scaled dot product attention 会基于输入自动选择最优实现。
# 为了更精细地控制使用哪种实现，提供以下函数：
# - torch.nn.attention.sdpa_kernel: 上下文管理器，用于启用或禁用任何实现
# - torch.backends.cuda.enable_flash_sdp: 全局启用或禁用 FlashAttention
# - torch.backends.cuda.enable_mem_efficient_sdp: 全局启用或禁用 Memory-Efficient Attention
# - torch.backends.cuda.enable_math_sdp: 全局启用或禁用 PyTorch C++ 实现

# 每个融合内核都有特定的输入限制。如果用户需要使用特定的融合实现，
# 可以使用 torch.nn.attention.sdpa_kernel 禁用 PyTorch C++ 实现。
# 如果无法运行融合实现，将会发出警告，说明无法运行融合实现的原因。

# 由于融合浮点操作的性质，此函数的输出可能因所选后端内核不同而异。
# C++ 实现支持 torch.float64，可在需要更高精度时使用。
# 更多信息请参阅 /notes/numerical_accuracy 文档。

# 注意：
# {cudnn_reproducibility_note}

Args:
    query (Tensor): 查询张量；形状为 :math:`(N, ..., L, E)`。
    key (Tensor): 键张量；形状为 :math:`(N, ..., S, E)`。
    value (Tensor): 值张量；形状为 :math:`(N, ..., S, Ev)`。
    # attn_mask (optional Tensor): Attention mask; shape must be broadcastable to the shape of attention weights,
    #     which is :math:`(N,..., L, S)`. Two types of masks are supported.
    #     A boolean mask where a value of True indicates that the element *should* take part in attention.
    #     A float mask of the same type as query, key, value that is added to the attention score.
    dropout_p (float): Dropout probability; if greater than 0.0, dropout is applied
    is_causal (bool): If set to true, the attention masking is a lower triangular matrix when the mask is a
        square matrix. The attention masking has the form of the upper left causal bias due to the alignment
        (see :class:`torch.nn.attention.bias.CausalBias`) when the mask is a non-square matrix.
        An error is thrown if both attn_mask and is_causal are set.
    scale (optional float, keyword-only): Scaling factor applied prior to softmax. If None, the default value is set
        to :math:`\frac{1}{\sqrt{E}}`.
# Verifies the expected shape for `query, `key`, `value`, `key_padding_mask` and `attn_mask`
# and returns if the input is batched or not.
# Raises an error if `query` is not 2-D (unbatched) or 3-D (batched) tensor.
def _mha_shape_check(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    key_padding_mask: Optional[Tensor],
    attn_mask: Optional[Tensor],
    num_heads: int,
):
    # Shape check.
    if query.dim() == 3:
        # Batched Inputs
        is_batched = True
        assert key.dim() == 3 and value.dim() == 3, (
            "For batched (3-D) `query`, expected `key` and `value` to be 3-D"
            f" but found {key.dim()}-D and {value.dim()}-D tensors respectively"
        )
        if key_padding_mask is not None:
            assert key_padding_mask.dim() == 2, (
                "For batched (3-D) `query`, expected `key_padding_mask` to be `None` or 2-D"
                f" but found {key_padding_mask.dim()}-D tensor instead"
            )
        if attn_mask is not None:
            assert attn_mask.dim() in (2, 3), (
                "For batched (3-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                f" but found {attn_mask.dim()}-D tensor instead"
            )
    elif query.dim() == 2:
        # 如果查询张量是二维的，则表示输入未经批处理
        is_batched = False
        # 断言键（key）和值（value）张量也都是二维的
        assert key.dim() == 2 and value.dim() == 2, (
            "For unbatched (2-D) `query`, expected `key` and `value` to be 2-D"
            f" but found {key.dim()}-D and {value.dim()}-D tensors respectively"
        )

        # 如果存在键填充遮罩（key_padding_mask），则断言其为一维张量
        if key_padding_mask is not None:
            assert key_padding_mask.dim() == 1, (
                "For unbatched (2-D) `query`, expected `key_padding_mask` to be `None` or 1-D"
                f" but found {key_padding_mask.dim()}-D tensor instead"
            )

        # 如果存在注意力遮罩（attn_mask），则断言其为二维或三维张量
        if attn_mask is not None:
            assert attn_mask.dim() in (2, 3), (
                "For unbatched (2-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                f" but found {attn_mask.dim()}-D tensor instead"
            )
            # 如果注意力遮罩是三维的，则检查其形状是否符合预期
            if attn_mask.dim() == 3:
                expected_shape = (num_heads, query.shape[0], key.shape[0])
                assert (
                    attn_mask.shape == expected_shape
                ), f"Expected `attn_mask` shape to be {expected_shape} but got {attn_mask.shape}"

    else:
        # 如果查询张量不是二维的，则抛出断言错误
        raise AssertionError(
            f"query should be unbatched 2D or batched 3D tensor but received {query.dim()}-D query tensor"
        )

    # 返回是否经过批处理的布尔值
    return is_batched
# 定义私有函数 `_canonical_mask`，用于规范化掩码张量
def _canonical_mask(
    mask: Optional[Tensor],  # 输入的掩码张量，可选类型
    mask_name: str,  # 掩码张量的名称
    other_type: Optional[DType],  # 另一个输入的数据类型，可选类型
    other_name: str,  # 另一个输入的名称
    target_type: DType,  # 目标数据类型
    check_other: bool = True,  # 是否检查另一个输入，默认为 True
) -> Optional[Tensor]:  # 返回值为可选的张量类型
    # 如果掩码张量不为 None，则进行处理
    if mask is not None:
        _mask_dtype = mask.dtype  # 获取掩码张量的数据类型
        _mask_is_float = torch.is_floating_point(mask)  # 检查掩码张量是否为浮点数类型
        # 如果掩码数据类型不是 bool 并且不是浮点数类型，则抛出异常
        if _mask_dtype != torch.bool and not _mask_is_float:
            raise AssertionError(
                f"only bool and floating types of {mask_name} are supported"
            )
        # 如果需要检查另一个输入，并且另一个输入类型不为 None
        if check_other and other_type is not None:
            # 如果掩码张量的数据类型与另一个输入的类型不一致，则发出警告
            if _mask_dtype != other_type:
                warnings.warn(
                    f"Support for mismatched {mask_name} and {other_name} "
                    "is deprecated. Use same type for both instead."
                )
        # 如果掩码张量不是浮点数类型，则将其重置为指定目标类型的零张量，并根据掩码进行部分填充
        if not _mask_is_float:
            mask = torch.zeros_like(mask, dtype=target_type).masked_fill_(
                mask, float("-inf")
            )
    # 返回处理后的掩码张量
    return mask


# 定义私有函数 `_none_or_dtype`，用于获取输入的数据类型或返回 None
def _none_or_dtype(input: Optional[Tensor]) -> Optional[DType]:
    # 如果输入为 None，则直接返回 None
    if input is None:
        return None
    # 如果输入是 torch.Tensor 类型，则返回其数据类型
    elif isinstance(input, torch.Tensor):
        return input.dtype
    # 否则，抛出运行时异常，要求输入必须为 None 或 torch.Tensor 类型
    raise RuntimeError("input to _none_or_dtype() must be None or torch.Tensor")


# 定义多头注意力机制的前向传播方法 `multi_head_attention_forward`
def multi_head_attention_forward(
    query: Tensor,  # 查询张量
    key: Tensor,  # 键张量
    value: Tensor,  # 值张量
    embed_dim_to_check: int,  # 检查的嵌入维度
    num_heads: int,  # 头数
    in_proj_weight: Optional[Tensor],  # 输入投影权重，可选类型
    in_proj_bias: Optional[Tensor],  # 输入投影偏置，可选类型
    bias_k: Optional[Tensor],  # 键的偏置项，可选类型
    bias_v: Optional[Tensor],  # 值的偏置项，可选类型
    add_zero_attn: bool,  # 是否添加零注意力
    dropout_p: float,  # dropout 概率
    out_proj_weight: Tensor,  # 输出投影权重
    out_proj_bias: Optional[Tensor],  # 输出投影偏置，可选类型
    training: bool = True,  # 是否处于训练模式，默认为 True
    key_padding_mask: Optional[Tensor] = None,  # 键的填充掩码，可选类型
    need_weights: bool = True,  # 是否需要注意力权重，默认为 True
    attn_mask: Optional[Tensor] = None,  # 注意力掩码，可选类型
    use_separate_proj_weight: bool = False,  # 是否使用独立的投影权重，默认为 False
    q_proj_weight: Optional[Tensor] = None,  # 查询投影权重，可选类型
    k_proj_weight: Optional[Tensor] = None,  # 键投影权重，可选类型
    v_proj_weight: Optional[Tensor] = None,  # 值投影权重，可选类型
    static_k: Optional[Tensor] = None,  # 静态键，可选类型
    static_v: Optional[Tensor] = None,  # 静态值，可选类型
    average_attn_weights: bool = True,  # 是否平均注意力权重，默认为 True
    is_causal: bool = False,  # 是否因果注意力，默认为 False
) -> Tuple[Tensor, Optional[Tensor]]:  # 返回值为张量和可选的张量元组
    r"""Forward method for MultiHeadAttention.

    See :class:`torch.nn.MultiheadAttention` for details.
    # Args: 定义函数参数和其作用。
    # query, key, value: 定义查询和键值对，用于映射到输出。
    # embed_dim_to_check: 模型的总维度。
    # num_heads: 并行注意力头的数量。
    # in_proj_weight, in_proj_bias: 输入投影的权重和偏置。
    # bias_k, bias_v: 添加到维度0的键和值序列的偏置。
    # add_zero_attn: 在维度1上添加一个新的零批次到键和值序列。
    # dropout_p: 元素被置为零的概率。
    # out_proj_weight, out_proj_bias: 输出投影的权重和偏置。
    # training: 如果为 ``True``，应用dropout。
    # key_padding_mask: 如果提供，指定在注意力中忽略键中的填充元素。这是一个二进制掩码。当值为True时，注意力层上对应的值将被填充为 -inf。
    # need_weights: 输出 attn_output_weights。默认为 `True`。
    # attn_mask: 防止关注某些位置的2D或3D掩码。2D掩码将广播到所有批次，而3D掩码允许为每个批次的条目指定不同的掩码。
    # is_causal: 如果指定，则应用因果掩码作为注意力掩码，并忽略 attn_mask 用于计算缩放点积注意力。默认为 ``False``。
    # use_separate_proj_weight: 函数接受查询、键和值的不同形式的投影权重。如果为false，将使用 in_proj_weight，它是 q_proj_weight、k_proj_weight、v_proj_weight 的组合。
    # q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: 输入投影的权重和偏置。
    # static_k, static_v: 用于注意力操作的静态键和值。
    # average_attn_weights: 如果为True，表示返回的 ``attn_weights`` 应该在头部之间进行平均。否则，``attn_weights`` 将单独针对每个头部提供。注意，仅当 ``need_weights=True`` 时，此标志才会生效。默认为True。
    """
    Shape:
        Inputs:
        - query: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, E)` or :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, E)` or :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(S)` or :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a FloatTensor is provided, it will be directly added to the value.
          If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: Only returned when ``need_weights=True``. If ``average_attn_weights=True``, returns
          attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
          head of shape :math:`(num_heads, L, S)` when input is unbatched or :math:`(N, num_heads, L, S)`.

    Initialize a tuple `tens_ops` containing the following tensors:
    - query: Target sequence representations with shape (L, E) or (L, N, E).
    - key: Source sequence representations with shape (S, E) or (S, N, E).
    - value: Source sequence representations with shape (S, E) or (S, N, E).
    - in_proj_weight: Weight tensor for input projection.
    - in_proj_bias: Bias tensor for input projection.
    - bias_k: Bias tensor for key projection.
    - bias_v: Bias tensor for value projection.
    - out_proj_weight: Weight tensor for output projection.
    - out_proj_bias: Bias tensor for output projection.
    """
    tens_ops = (
        query,
        key,
        value,
        in_proj_weight,
        in_proj_bias,
        bias_k,
        bias_v,
        out_proj_weight,
        out_proj_bias,
    )
    # 检查是否有 torch function 处理 tens_ops
    if has_torch_function(tens_ops):
        return handle_torch_function(
            multi_head_attention_forward,
            tens_ops,
            query,
            key,
            value,
            embed_dim_to_check,
            num_heads,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            add_zero_attn,
            dropout_p,
            out_proj_weight,
            out_proj_bias,
            training=training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            is_causal=is_causal,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            static_k=static_k,
            static_v=static_v,
            average_attn_weights=average_attn_weights,
        )

    # 检查输入是否为批次化，执行形状检查
    is_batched = _mha_shape_check(
        query, key, value, key_padding_mask, attn_mask, num_heads
    )

    # 对于未批次化输入，我们在预期的批次维度上进行unsqueeze操作，以假装输入是批次化的，
    # 运行计算并在返回之前挤压批次维度，以便输出不包含这个临时批次维度。
    if not is_batched:
        # 如果输入未批次化，则进行unsqueeze操作
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)

    # 设置形状变量
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape

    # 规范化键填充掩码
    key_padding_mask = _canonical_mask(
        mask=key_padding_mask,
        mask_name="key_padding_mask",
        other_type=_none_or_dtype(attn_mask),
        other_name="attn_mask",
        target_type=query.dtype,
    )

    # 如果设置了 is_causal 并且没有指定 attn_mask，则抛出运行时错误
    if is_causal and attn_mask is None:
        raise RuntimeError(
            "Need attn_mask if specifying the is_causal hint. "
            "You may use the Transformer module method "
            "`generate_square_subsequent_mask` to create this mask."
        )

    # 如果设置了 is_causal 并且没有设置键填充掩码并且不需要权重
    if is_causal and key_padding_mask is None and not need_weights:
        # 当我们有 kpm 或者需要权重时，我们需要 attn_mask
        # 否则，我们使用 is_causal 提示传递给 SDPA。
        attn_mask = None
    else:
        # 否则，规范化注意力掩码
        attn_mask = _canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        # 如果键填充掩码不为空
        if key_padding_mask is not None:
            # 我们有 attn_mask，并使用它将 kpm 合并到其中。
            # 关闭使用 is_causal 提示，因为合并的掩码不再是因果的。
            is_causal = False

    # 断言，确保嵌入维度与 embed_dim_to_check 相等
    assert (
        embed_dim == embed_dim_to_check
    ), f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    # 检查嵌入维度是否符合预期，如果不符合，抛出异常信息

    if isinstance(embed_dim, torch.Tensor):
        # 当嵌入维度是一个张量时，通常在 JIT 追踪时出现
        head_dim = embed_dim.div(num_heads, rounding_mode="trunc")
    else:
        head_dim = embed_dim // num_heads
    # 根据嵌入维度和注意力头数计算每个头的维度

    assert (
        head_dim * num_heads == embed_dim
    ), f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    # 确保嵌入维度能够被注意力头数整除，否则抛出异常信息

    if use_separate_proj_weight:
        # 当使用单独的投影权重时，允许多头注意力具有不同的嵌入维度
        assert (
            key.shape[:2] == value.shape[:2]
        ), f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert (
            key.shape == value.shape
        ), f"key shape {key.shape} does not match value shape {value.shape}"
    # 根据是否使用单独的投影权重，检查键和值的形状是否匹配，否则抛出异常信息

    #
    # 计算投影
    #
    if not use_separate_proj_weight:
        assert (
            in_proj_weight is not None
        ), "use_separate_proj_weight is False but in_proj_weight is None"
        # 当不使用单独的投影权重时，确保投影权重不为 None
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert (
            q_proj_weight is not None
        ), "use_separate_proj_weight is True but q_proj_weight is None"
        assert (
            k_proj_weight is not None
        ), "use_separate_proj_weight is True but k_proj_weight is None"
        assert (
            v_proj_weight is not None
        ), "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        # 当使用单独的投影权重时，确保所有投影权重不为 None，否则抛出异常信息
        q, k, v = _in_projection(
            query,
            key,
            value,
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
            b_q,
            b_k,
            b_v,
        )

    # 准备注意力掩码

    if attn_mask is not None:
        # 确保注意力掩码的维度是3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(
                    f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}."
                )
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(
                    f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}."
                )
        else:
            raise RuntimeError(
                f"attn_mask's dimension {attn_mask.dim()} is not supported"
            )
    # 当存在注意力掩码时，根据其维度进行检查和调整，确保其形状符合预期

    # 沿批次维度（当前为第二维度）添加偏置
    # 如果存在 bias_k 和 bias_v，则进行以下处理
    if bias_k is not None and bias_v is not None:
        # 断言 static_k 为 None，否则报错“无法将偏置添加到静态键”
        assert static_k is None, "bias cannot be added to static key."
        # 断言 static_v 为 None，否则报错“无法将偏置添加到静态值”
        assert static_v is None, "bias cannot be added to static value."
        # 在维度 1 上重复 bias_k，然后连接到 k 中
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        # 在维度 1 上重复 bias_v，然后连接到 v 中
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        # 如果存在 attn_mask，则在其右侧填充一个单位长度的零
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        # 如果存在 key_padding_mask，则在其右侧填充一个单位长度的零
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    else:
        # 如果不存在 bias_k 和 bias_v，则断言它们均为 None
        assert bias_k is None
        assert bias_v is None

    #
    # 为多头注意力重塑 q, k, v，并确保它们以批次为首维度
    #
    # 重塑 q 为 (tgt_len, bsz * num_heads, head_dim)，然后转置前两个维度
    q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        # 如果 static_k 为 None，则重塑 k 为 (k.shape[0], bsz * num_heads, head_dim)，然后转置前两个维度
        k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # 如果 static_k 不为 None，则断言 static_k 的第一个维度为 bsz * num_heads
        assert (
            static_k.size(0) == bsz * num_heads
        ), f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        # 断言 static_k 的第三个维度为 head_dim
        assert (
            static_k.size(2) == head_dim
        ), f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        # 将 k 设置为 static_k
        k = static_k
    if static_v is None:
        # 如果 static_v 为 None，则重塑 v 为 (v.shape[0], bsz * num_heads, head_dim)，然后转置前两个维度
        v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # 如果 static_v 不为 None，则断言 static_v 的第一个维度为 bsz * num_heads
        assert (
            static_v.size(0) == bsz * num_heads
        ), f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        # 断言 static_v 的第三个维度为 head_dim
        assert (
            static_v.size(2) == head_dim
        ), f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        # 将 v 设置为 static_v

    # 如果需要添加零注意力沿批次维度（现在是第一维度）
    if add_zero_attn:
        # 定义零注意力的形状
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        # 在 k 的第二维度连接一个全零张量
        k = torch.cat(
            [k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1
        )
        # 在 v 的第二维度连接一个全零张量
        v = torch.cat(
            [v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1
        )
        # 如果存在 attn_mask，则在其右侧填充一个单位长度的零
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        # 如果存在 key_padding_mask，则在其右侧填充一个单位长度的零
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    # 更新调整后的源序列长度
    src_len = k.size(1)

    # 合并键填充和注意力掩码
    if key_padding_mask is not None:
        # 断言 key_padding_mask 的形状为 (bsz, src_len)
        assert key_padding_mask.shape == (
            bsz,
            src_len,
        ), f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        # 将 key_padding_mask 重塑为 (bsz, 1, 1, src_len)，然后扩展为 (bsz * num_heads, 1, src_len)，最后重塑为 (bsz * num_heads, 1, src_len)
        key_padding_mask = (
            key_padding_mask.view(bsz, 1, 1, src_len)
            .expand(-1, num_heads, -1, -1)
            .reshape(bsz * num_heads, 1, src_len)
        )
        # 如果 attn_mask 为 None，则设为 key_padding_mask；否则，将其加到 attn_mask 上
        if attn_mask is None:
            attn_mask = key_padding_mask
        else:
            attn_mask = attn_mask + key_padding_mask
    # 调整丢弃概率
    if not training:
        # 如果不是训练阶段，则将丢弃概率设为0
        dropout_p = 0.0

    #
    # 计算注意力和输出投影
    #

    # 如果需要权重计算
    if need_weights:
        # 获取输入张量的维度信息
        B, Nt, E = q.shape
        # 对查询张量进行缩放，乘以 sqrt(1.0 / E)，用于注意力计算
        q_scaled = q * math.sqrt(1.0 / float(E))

        # 如果是因果关系且没有提供注意力掩码，则报错
        assert not (
            is_causal and attn_mask is None
        ), "FIXME: is_causal not implemented for need_weights"

        # 根据是否有注意力掩码选择不同的注意力权重计算方法
        if attn_mask is not None:
            attn_output_weights = torch.baddbmm(
                attn_mask, q_scaled, k.transpose(-2, -1)
            )
        else:
            attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
        # 对注意力权重进行 softmax 归一化
        attn_output_weights = softmax(attn_output_weights, dim=-1)
        # 如果设定了丢弃概率，则应用丢弃操作到注意力权重上
        if dropout_p > 0.0:
            attn_output_weights = dropout(attn_output_weights, p=dropout_p)

        # 计算注意力输出，使用注意力权重加权求和得到
        attn_output = torch.bmm(attn_output_weights, v)

        # 调整注意力输出的形状，以便后续的线性投影操作
        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
        )
        # 线性投影操作，将注意力输出映射到指定维度
        attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
        # 将输出重新调整为原来的形状
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

        # 可选地对多头注意力权重进行平均操作
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.mean(dim=1)

        # 如果输入没有进行批处理，则压缩输出的维度
        if not is_batched:
            attn_output = attn_output.squeeze(1)
            attn_output_weights = attn_output_weights.squeeze(0)
        return attn_output, attn_output_weights
    else:
        # 如果不需要权重计算

        # 根据注意力掩码的形状进行适当的调整
        # 如果注意力掩码的形状是 (L, S) 或者 (N*num_heads, L, S)
        # 如果注意力掩码的形状是 (1, L, S)，则需要扩展为 (1, 1, L, S)，以匹配输入格式要求
        if attn_mask is not None:
            if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(0)
            else:
                attn_mask = attn_mask.view(bsz, num_heads, -1, src_len)

        # 重新调整 q、k、v 张量的形状，以适应多头自注意力计算
        q = q.view(bsz, num_heads, tgt_len, head_dim)
        k = k.view(bsz, num_heads, src_len, head_dim)
        v = v.view(bsz, num_heads, src_len, head_dim)

        # 执行缩放点积注意力计算
        attn_output = scaled_dot_product_attention(
            q, k, v, attn_mask, dropout_p, is_causal
        )

        # 调整注意力输出的形状，以便后续的线性投影操作
        attn_output = (
            attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
        )

        # 线性投影操作，将注意力输出映射到指定维度
        attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
        # 将输出重新调整为原来的形状
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

        # 如果输入没有进行批处理，则压缩输出的维度
        if not is_batched:
            attn_output = attn_output.squeeze(1)
        return attn_output, None
```