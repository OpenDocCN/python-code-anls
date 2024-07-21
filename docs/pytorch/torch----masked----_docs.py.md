# `.\pytorch\torch\masked\_docs.py`

```
# This file is generated, do not modify it!
#
# To update this file, run the update masked docs script as follows:
#
#   python tools/update_masked_docs.py
#
# The script must be called from an environment where the development
# version of torch package can be imported and is functional.
#

# 定义函数amax，用于计算沿给定维度dim的输入张量input的最大值，
# 在计算过程中根据布尔张量mask对input元素进行遮蔽。
amax_docstring = """amax(input, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor

Returns maximum of all the elements in the :attr:`input`
tensor along the given dimension(s) :attr:`dim` while the :attr:`input`
elements are masked out according to the boolean tensor
:attr:`mask`.

The identity value of maximum operation, which is used to start the
reduction, depends on input dtype. For instance, for float32, uint8,
and int32 dtypes, the identity values are ``-inf``, ``0``, and ``-2147483648``, respectively.

If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of
size 1. Otherwise, :attr:`dim` is squeezed (see
:func:`torch.squeeze`), resulting in the output tensor having 1 (or
``len(dim)``) fewer dimension(s).

The boolean tensor :attr:`mask` defines the "validity" of
:attr:`input` tensor elements: if :attr:`mask` element is True
then the corresponding element in :attr:`input` tensor will be
included in maximum computation, otherwise the element is
ignored.

When all elements of :attr:`input` along the given dimension
:attr:`dim` are ignored (fully masked-out), the corresponding element
of the output tensor will have undefined value: it may or may not
correspond to the identity value of maximum operation; the
choice may correspond to the value that leads to the most efficient
storage of :attr:`output` tensor.

The mask of the output tensor can be computed as
``torch.any(torch.broadcast_to(mask, input.shape), dim, keepdim=keepdim,
dtype=torch.bool)``.

The shapes of the :attr:`mask` tensor and the :attr:`input` tensor
don't need to match, but they must be :ref:`broadcastable
<broadcasting-semantics>` and the dimensionality of the :attr:`mask`
tensor must not be greater than of the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor
    dim (int or tuple of ints, optional): the dimension or dimensions to reduce.
      Default: None that is equivalent to ``tuple(range(input.ndim))``.

Keyword args:
    keepdim (bool, optional): whether the output tensor has
      :attr:`dim` retained or not. Default: False.
    dtype (:class:`torch.dtype`, optional): the desired data type
      of returned tensor.  If specified, the input tensor is
      casted to :attr:`dtype` before the operation is
      performed. Default: None.
    mask (:class:`torch.Tensor`, optional): the boolean tensor
      containing the binary mask of validity of input tensor
      elements.
      Default: None that is equivalent to ``torch.ones(input.shape, dtype=torch.bool)``.

Example::

    >>> input = tensor([[-3, -2, -1], [ 0, 1, 2]])
    >>> input
    # 创建一个二维张量，包含两行三列的整数数据
    tensor([[-3, -2, -1],
            [ 0,  1,  2]])
    
    # 创建一个与上面张量形状相同的布尔掩码张量
    mask = tensor([[ True, False, True], [False, False, False]])
    
    # 显示当前的掩码张量
    mask
    tensor([[ True, False,  True],
            [False, False, False]])
    
    # 使用 torch.masked._ops.amax 函数计算张量 input 沿着第一维度的最大值，
    # 但仅考虑掩码为 True 的位置，其他位置为 False 则被忽略
    torch.masked._ops.amax(input, 1, mask=mask)
    tensor([                  -1, -9223372036854775808])
"""

amin_docstring = """amin(input, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor

Returns minimum of all the elements in the :attr:`input`
tensor along the given dimension(s) :attr:`dim` while the :attr:`input`
elements are masked out according to the boolean tensor
:attr:`mask`.

The identity value of minimum operation, which is used to start the
reduction, depends on input dtype. For instance, for float32, uint8,
and int32 dtypes, the identity values are ``inf``, ``255``, and ``2147483647``, respectively.

If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of
size 1. Otherwise, :attr:`dim` is squeezed (see
:func:`torch.squeeze`), resulting in the output tensor having 1 (or
``len(dim)``) fewer dimension(s).

The boolean tensor :attr:`mask` defines the "validity" of
:attr:`input` tensor elements: if :attr:`mask` element is True
then the corresponding element in :attr:`input` tensor will be
included in minimum computation, otherwise the element is
ignored.

When all elements of :attr:`input` along the given dimension
:attr:`dim` are ignored (fully masked-out), the corresponding element
of the output tensor will have undefined value: it may or may not
correspond to the identity value of minimum operation; the
choice may correspond to the value that leads to the most efficient
storage of :attr:`output` tensor.

The mask of the output tensor can be computed as
``torch.any(torch.broadcast_to(mask, input.shape), dim, keepdim=keepdim,
dtype=torch.bool)``.

The shapes of the :attr:`mask` tensor and the :attr:`input` tensor
don't need to match, but they must be :ref:`broadcastable
<broadcasting-semantics>` and the dimensionality of the :attr:`mask`
tensor must not be greater than of the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor
    dim (int or tuple of ints, optional): the dimension or dimensions to reduce.
      Default: None that is equivalent to ``tuple(range(input.ndim))``.

Keyword args:
    keepdim (bool, optional): whether the output tensor has
      :attr:`dim` retained or not. Default: False.
    dtype (:class:`torch.dtype`, optional): the desired data type
      of returned tensor.  If specified, the input tensor is
      casted to :attr:`dtype` before the operation is
      performed. Default: None.
    mask (:class:`torch.Tensor`, optional): the boolean tensor
      containing the binary mask of validity of input tensor
      elements.
      Default: None that is equivalent to ``torch.ones(input.shape, dtype=torch.bool)``.

Example::

    >>> input = tensor([[-3, -2, -1], [ 0, 1, 2]])
    >>> input
    tensor([[-3, -2, -1],
            [ 0,  1,  2]])
    >>> mask = tensor([[ True, False, True], [False, False, False]])
    >>> mask
    tensor([[ True, False,  True],
            [False, False, False]])
    >>> torch.masked._ops.amin(input, 1, mask=mask)
    tensor([                 -3, 9223372036854775807])
"""
# 定义 argmax 函数的文档字符串，描述函数的作用、参数、和返回值
argmax_docstring = """argmax(input, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor
Returns argmax of all the elements in the :attr:`input`
tensor along the given dimension(s) :attr:`dim` while the :attr:`input`
elements are masked out according to the boolean tensor
:attr:`mask`.
The identity value of argmax operation, which is used to start the
reduction, depends on input dtype. For instance, for float32, uint8,
and int32 dtypes, the identity values are ``-inf``, ``0``, and ``-2147483648``, respectively.
If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of
size 1. Otherwise, :attr:`dim` is squeezed (see
:func:`torch.squeeze`), resulting in the output tensor having 1 (or
``len(dim)``) fewer dimension(s).

The boolean tensor :attr:`mask` defines the "validity" of
:attr:`input` tensor elements: if :attr:`mask` element is True
then the corresponding element in :attr:`input` tensor will be
included in argmax computation, otherwise the element is
ignored.

When all elements of :attr:`input` along the given dimension
:attr:`dim` are ignored (fully masked-out), the corresponding element
of the output tensor will have undefined value: it may or may not
correspond to the identity value of argmax operation; the
choice may correspond to the value that leads to the most efficient
storage of :attr:`output` tensor.

The mask of the output tensor can be computed as
``torch.any(torch.broadcast_to(mask, input.shape), dim, keepdim=keepdim,
dtype=torch.bool)``.

The shapes of the :attr:`mask` tensor and the :attr:`input` tensor
don't need to match, but they must be :ref:`broadcastable
<broadcasting-semantics>` and the dimensionality of the :attr:`mask`
tensor must not be greater than of the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor
    dim (int): the dimension along which argmax is computed.

Keyword args:
    keepdim (bool, optional): whether the output tensor has
      :attr:`dim` retained or not. Default: False.
    dtype (:class:`torch.dtype`, optional): the desired data type
      of returned tensor.  If specified, the input tensor is
      casted to :attr:`dtype` before the operation is
      performed. Default: None.
    mask (:class:`torch.Tensor`, optional): the boolean tensor
      containing the binary mask of validity of input tensor
      elements.
      Default: None that is equivalent to ``torch.ones(input.shape, dtype=torch.bool)``.
Example::

    >>> input = tensor([[-3, -2, -1], [ 0, 1, 2]])
    >>> input
    tensor([[-3, -2, -1],
            [ 0,  1,  2]])
    >>> mask = tensor([[ True, False, True], [False, False, False]])
    >>> mask
    tensor([[ True, False,  True],
            [False, False, False]])
    >>> torch.masked._ops.argmax(input, 1, mask=mask)
    tensor([2, 0])
"""

# 定义 argmin 函数的文档字符串，描述函数的作用、参数、和返回值
argmin_docstring = """argmin(input, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor
Returns argmin of all the elements in the :attr:`input`
tensor along the given dimension(s) :attr:`dim` while the :attr:`input`
elements are masked out according to the boolean tensor
:attr:`mask`.
The identity value of argmin operation, which is used to start the
reduction, depends on input dtype. For instance, for float32, uint8,
and int32 dtypes, the identity values are ``inf``, ``255``, and ``2147483647``, respectively.
If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of
size 1. Otherwise, :attr:`dim` is squeezed (see
:func:`torch.squeeze`), resulting in the output tensor having 1 (or
``len(dim)``) fewer dimension(s).

The boolean tensor :attr:`mask` defines the "validity" of
:attr:`input` tensor elements: if :attr:`mask` element is True
then the corresponding element in :attr:`input` tensor will be
included in argmin computation, otherwise the element is
ignored.

When all elements of :attr:`input` along the given dimension
:attr:`dim` are ignored (fully masked-out), the corresponding element
of the output tensor will have undefined value: it may or may not
correspond to the identity value of argmin operation; the
choice may correspond to the value that leads to the most efficient
storage of :attr:`output` tensor.

The mask of the output tensor can be computed as
``torch.any(torch.broadcast_to(mask, input.shape), dim, keepdim=keepdim,
dtype=torch.bool)``.

The shapes of the :attr:`mask` tensor and the :attr:`input` tensor
don't need to match, but they must be :ref:`broadcastable
<broadcasting-semantics>` and the dimensionality of the :attr:`mask`
tensor must not be greater than of the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor
    dim (int): the dimension along which argmin is computed.

Keyword args:
    keepdim (bool, optional): whether the output tensor has
      :attr:`dim` retained or not. Default: False.
    dtype (:class:`torch.dtype`, optional): the desired data type
      of returned tensor.  If specified, the input tensor is
      casted to :attr:`dtype` before the operation is
      performed. Default: None.
    mask (:class:`torch.Tensor`, optional): the boolean tensor
      containing the binary mask of validity of input tensor
      elements.
      Default: None that is equivalent to ``torch.ones(input.shape, dtype=torch.bool)``.
Example::

    >>> input = tensor([[-3, -2, -1], [ 0, 1, 2]])
    >>> input
    tensor([[-3, -2, -1],
            [ 0,  1,  2]])
    >>> mask = tensor([[ True, False, True], [False, False, False]])
    >>> mask
    tensor([[ True, False,  True],
            [False, False, False]])
    >>> torch.masked._ops.argmin(input, 1, mask=mask)
    tensor([0, 1])
"""
"""
cumprod(input, dim, *, dtype=None, mask=None) -> Tensor

返回沿着给定维度 `dim` 的累积乘积的结果，同时根据布尔型张量 `mask` 屏蔽输入张量的元素。

Args:
    input (Tensor): 输入张量
    dim (int): 执行累积乘积操作的维度

Keyword args:
    dtype (:class:`torch.dtype`, optional): 返回张量的期望数据类型。如果指定，操作执行前将输入张量转换为 `dtype` 类型。默认为 `None`。
    mask (:class:`torch.Tensor`, optional): 包含输入张量元素有效性的布尔型张量。默认为 `None`，等同于 `torch.ones(input.shape, dtype=torch.bool)`。

Returns:
    Tensor: 返回累积乘积的结果张量

Example::

    >>> input = tensor([[1, 2, 3], [4, 5, 6]])
    >>> input
    tensor([[1, 2, 3],
            [4, 5, 6]])
    >>> mask = tensor([[ True, False, True], [False, False, False]])
    >>> mask
    tensor([[ True, False,  True],
            [False, False, False]])
    >>> torch.masked._ops.cumprod(input, 1, mask=mask)
    tensor([[ 1,  2,  6],
            [ 4, 20, 120]])

"""
"""
cumsum_docstring = """cumsum(input, dim, *, dtype=None, mask=None) -> Tensor

Returns cumulative_sum of all the slices in the :attr:`input` tensor
along :attr:`dim` while the :attr:`input` elements are masked out
according to the boolean tensor :attr:`mask`.

Let ``x`` be a sequence of unmasked elements of one-dimensional slice
of the :attr:`input` tensor. Cumsum of i-th element in ``x`` is
defined as ``sum(x[:i])``.

The boolean tensor :attr:`mask` defines the "validity" of
:attr:`input` tensor elements: if :attr:`mask` element is True then
the corresponding element in :attr:`input` tensor will be included in
cumulative_sum computation, otherwise the element is ignored.

The values of masked-out elements of the output tensor have undefined
value: it may or may not be set to zero or nan; the choice may correspond to
the value that leads to the most efficient storage of :attr:`output`
tensor.


"""
"""
log_softmax(input, dim, *, dtype=None, mask=None) -> Tensor

Returns log_softmax of all the slices in the :attr:`input` tensor
along :attr:`dim` while the :attr:`input` elements are masked out
according to the boolean tensor :attr:`mask`.

Let ``x`` be a sequence of unmasked elements of one-dimensional slice
of the :attr:`input` tensor. LogSoftmax of i-th element in ``x`` is
defined as ``log(exp(x[i])/sum(exp(x)))``.

The boolean tensor :attr:`mask` defines the "validity" of
:attr:`input` tensor elements: if :attr:`mask` element is True then
the corresponding element in :attr:`input` tensor will be included in
log_softmax computation, otherwise the element is ignored.

The values of masked-out elements of the output tensor have undefined
value: it may or may not be set to zero or nan; the choice may correspond to
the value that leads to the most efficient storage of :attr:`output`
tensor.

The mask of the log_softmax output tensor can be computed as
``torch.broadcast_to(mask, input.shape)``.

The shapes of the :attr:`mask` tensor and the :attr:`input` tensor
don't need to match, but they must be :ref:`broadcastable
<broadcasting-semantics>` and the dimensionality of the :attr:`mask`
tensor must not be greater than of the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor
    dim (int): the dimension along which log_softmax is computed.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type
      of returned tensor.  If specified, the input tensor is
      casted to :attr:`dtype` before the operation is
      performed. Default: None.
"""


注释：
    # `mask`是一个可选参数，应为:class:`torch.Tensor`类型，它是一个布尔值张量，
    # 表示输入张量元素有效性的二进制掩码。
    # 默认情况下为None，相当于使用`torch.ones(input.shape, dtype=torch.bool)`。
# 定义一个多行字符串，描述了 logsumexp 函数的文档字符串，包括函数名称、参数、返回值以及功能说明
logsumexp_docstring = """logsumexp(input, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor

Returns logsumexp of all the elements in the :attr:`input`
tensor along the given dimension(s) :attr:`dim` while the :attr:`input`
elements are masked out according to the boolean tensor
:attr:`mask`.

The identity value of logsumexp operation, which is used to start the reduction, is ``-2147483648``.

If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of
size 1. Otherwise, :attr:`dim` is squeezed (see
:func:`torch.squeeze`), resulting in the output tensor having 1 (or
``len(dim)``) fewer dimension(s).

The boolean tensor :attr:`mask` defines the "validity" of
:attr:`input` tensor elements: if :attr:`mask` element is True
then the corresponding element in :attr:`input` tensor will be
included in logsumexp computation, otherwise the element is
ignored.

When all elements of :attr:`input` along the given dimension
:attr:`dim` are ignored (fully masked-out), the corresponding element
of the output tensor will have undefined value: it may or may not
correspond to the identity value of logsumexp operation; the
choice may correspond to the value that leads to the most efficient
storage of :attr:`output` tensor.

The mask of the output tensor can be computed as
``torch.any(torch.broadcast_to(mask, input.shape), dim, keepdim=keepdim,
dtype=torch.bool)``.

The shapes of the :attr:`mask` tensor and the :attr:`input` tensor
don't need to match, but they must be :ref:`broadcastable
<broadcasting-semantics>` and the dimensionality of the :attr:`mask`
tensor must not be greater than of the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor
    dim (int or tuple of ints, optional): the dimension or dimensions to reduce.
      Default: None that is equivalent to ``tuple(range(input.ndim))``.

Keyword args:
    keepdim (bool, optional): whether the output tensor has
      :attr:`dim` retained or not. Default: False.
    dtype (:class:`torch.dtype`, optional): the desired data type
      of returned tensor.  If specified, the input tensor is
      casted to :attr:`dtype` before the operation is
      performed. Default: None.
    mask (:class:`torch.Tensor`, optional): the boolean tensor
      containing the binary mask of validity of input tensor
      elements.
      Default: None that is equivalent to ``torch.ones(input.shape, dtype=torch.bool)``.

Example::

    >>> input = tensor([[-3, -2, -1], [ 0, 1, 2]])
    >>> input
"""
    # 创建一个二维张量，包含两行三列的整数数据
    tensor([[-3, -2, -1],
            [ 0,  1,  2]])
    # 创建一个与上面张量相同形状的布尔掩码张量
    mask = tensor([[ True, False, True], [False, False, False]])
    # 打印显示布尔掩码张量的值
    mask
    # 对输入张量进行按行的对数和指数运算，使用给定的布尔掩码进行掩码操作
    torch.masked._ops.logsumexp(input, 1, mask=mask)
    # 返回一个张量，其中第一个元素为 0，第二个元素为 Python 中的最小负整数
    tensor([                   0, -9223372036854775808])
"""
mean_docstring = """mean(input, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor

Returns mean of all the elements in the :attr:`input`
tensor along the given dimension(s) :attr:`dim` while the :attr:`input`
elements are masked out according to the boolean tensor
:attr:`mask`.

By definition, the identity value of a mean operation is the mean
value of the tensor. If all elements of the input tensor along given
dimension(s) :attr:`dim` are masked-out, the identity value of the
mean is undefined.  Due to this ambiguity, the elements of output
tensor with strided layout, that correspond to fully masked-out
elements, have ``nan`` values.

If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of
size 1. Otherwise, :attr:`dim` is squeezed (see
:func:`torch.squeeze`), resulting in the output tensor having 1 (or
``len(dim)``) fewer dimension(s).

The boolean tensor :attr:`mask` defines the "validity" of
:attr:`input` tensor elements: if :attr:`mask` element is True
then the corresponding element in :attr:`input` tensor will be
included in mean computation, otherwise the element is
ignored.

When all elements of :attr:`input` along the given dimension
:attr:`dim` are ignored (fully masked-out), the corresponding element
of the output tensor will have undefined value: it may or may not
correspond to the identity value of mean operation; the
choice may correspond to the value that leads to the most efficient
storage of :attr:`output` tensor.

The mask of the output tensor can be computed as
``torch.any(torch.broadcast_to(mask, input.shape), dim, keepdim=keepdim,
dtype=torch.bool)``.

The shapes of the :attr:`mask` tensor and the :attr:`input` tensor
don't need to match, but they must be :ref:`broadcastable
<broadcasting-semantics>` and the dimensionality of the :attr:`mask`
tensor must not be greater than of the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor
    dim (int or tuple of ints, optional): the dimension or dimensions to reduce.
      Default: None that is equivalent to ``tuple(range(input.ndim))``.

Keyword args:
    keepdim (bool, optional): whether the output tensor has
      :attr:`dim` retained or not. Default: False.
    dtype (:class:`torch.dtype`, optional): the desired data type
      of returned tensor.  If specified, the input tensor is
      casted to :attr:`dtype` before the operation is
      performed. Default: None.
    mask (:class:`torch.Tensor`, optional): the boolean tensor
      containing the binary mask of validity of input tensor
      elements.
      Default: None that is equivalent to ``torch.ones(input.shape, dtype=torch.bool)``.

Example::

    >>> input = tensor([[-3, -2, -1], [ 0, 1, 2]])
    >>> input
    tensor([[-3, -2, -1],
            [ 0,  1,  2]])
    >>> mask = tensor([[ True, False, True], [False, False, False]])
    >>> mask
    tensor([[ True, False,  True],
            [False, False, False]])

"""

# 定义函数 mean_docstring，描述了一个求平均值的函数的详细文档字符串
mean_docstring = """mean(input, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor

Returns mean of all the elements in the :attr:`input`
tensor along the given dimension(s) :attr:`dim` while the :attr:`input`
elements are masked out according to the boolean tensor
:attr:`mask`.

By definition, the identity value of a mean operation is the mean
value of the tensor. If all elements of the input tensor along given
dimension(s) :attr:`dim` are masked-out, the identity value of the
mean is undefined.  Due to this ambiguity, the elements of output
tensor with strided layout, that correspond to fully masked-out
elements, have ``nan`` values.

If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of
size 1. Otherwise, :attr:`dim` is squeezed (see
:func:`torch.squeeze`), resulting in the output tensor having 1 (or
``len(dim)``) fewer dimension(s).

The boolean tensor :attr:`mask` defines the "validity" of
:attr:`input` tensor elements: if :attr:`mask` element is True
then the corresponding element in :attr:`input` tensor will be
included in mean computation, otherwise the element is
ignored.

When all elements of :attr:`input` along the given dimension
:attr:`dim` are ignored (fully masked-out), the corresponding element
of the output tensor will have undefined value: it may or may not
correspond to the identity value of mean operation; the
choice may correspond to the value that leads to the most efficient
storage of :attr:`output` tensor.

The mask of the output tensor can be computed as
``torch.any(torch.broadcast_to(mask, input.shape), dim, keepdim=keepdim,
dtype=torch.bool)``.

The shapes of the :attr:`mask` tensor and the :attr:`input` tensor
don't need to match, but they must be :ref:`broadcastable
<broadcasting-semantics>` and the dimensionality of the :attr:`mask`
tensor must not be greater than of the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor
    dim (int or tuple of ints, optional): the dimension or dimensions to reduce.
      Default: None that is equivalent to ``tuple(range(input.ndim))``.

Keyword args:
    keepdim (bool, optional): whether the output tensor has
      :attr:`dim` retained or not. Default: False.
    dtype (:class:`torch.dtype`, optional): the desired data type
      of returned tensor.  If specified, the input tensor is
      casted to :attr:`dtype` before the operation is
      performed. Default: None.
    mask (:class:`torch.Tensor`, optional): the boolean tensor
      containing the binary mask of validity of input tensor
      elements.
      Default: None that is equivalent to ``torch.ones(input.shape, dtype=torch.bool)``.

Example::

    >>> input = tensor([[-3, -2, -1], [ 0, 1, 2]])
    >>> input
    tensor([[-3, -2, -1],
            [ 0,  1,  2]])
    >>> mask = tensor([[ True, False, True], [False, False, False]])
    >>> mask
    tensor([[ True, False,  True],
            [False, False, False]])

"""
    # 调用 PyTorch 中的 masked 操作，计算给定输入张量在第一维上的均值。
    # 参数 input 是输入张量，1 表示沿第一维进行均值计算。
    # 参数 mask 是一个掩码张量，用于指定哪些元素参与均值计算，哪些元素被忽略。
    # 返回一个张量，包含沿指定维度计算得到的均值结果。在这个例子中，返回的结果是 [-2., nan]。
    >>> torch.masked._ops.mean(input, 1, mask=mask)
    tensor([-2., nan])
"""

median_docstring = """median(input, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor
Returns median of all the elements in the :attr:`input`
tensor along the given dimension(s) :attr:`dim` while the :attr:`input`
elements are masked out according to the boolean tensor
:attr:`mask`.
By definition, the identity value of a median operation is the median
value of the tensor. If all elements of the input tensor along given
dimension(s) :attr:`dim` are masked-out, the identity value of the
median is undefined.  Due to this ambiguity, the elements of output
tensor with strided layout, that correspond to fully masked-out
elements, have ``nan`` values.
If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of
size 1. Otherwise, :attr:`dim` is squeezed (see
:func:`torch.squeeze`), resulting in the output tensor having 1 (or
``len(dim)``) fewer dimension(s).

The boolean tensor :attr:`mask` defines the "validity" of
:attr:`input` tensor elements: if :attr:`mask` element is True
then the corresponding element in :attr:`input` tensor will be
included in median computation, otherwise the element is
ignored.

When all elements of :attr:`input` along the given dimension
:attr:`dim` are ignored (fully masked-out), the corresponding element
of the output tensor will have undefined value: it may or may not
correspond to the identity value of median operation; the
choice may correspond to the value that leads to the most efficient
storage of :attr:`output` tensor.

The mask of the output tensor can be computed as
``torch.any(torch.broadcast_to(mask, input.shape), dim, keepdim=keepdim,
dtype=torch.bool)``.

The shapes of the :attr:`mask` tensor and the :attr:`input` tensor
don't need to match, but they must be :ref:`broadcastable
<broadcasting-semantics>` and the dimensionality of the :attr:`mask`
tensor must not be greater than of the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor
    dim (int): the dimension along which median is computed.

Keyword args:
    keepdim (bool, optional): whether the output tensor has
      :attr:`dim` retained or not. Default: False.
    dtype (:class:`torch.dtype`, optional): the desired data type
      of returned tensor.  If specified, the input tensor is
      casted to :attr:`dtype` before the operation is
      performed. Default: None.
    mask (:class:`torch.Tensor`, optional): the boolean tensor
      containing the binary mask of validity of input tensor
      elements.
      Default: None that is equivalent to ``torch.ones(input.shape, dtype=torch.bool)``.
Example::

    >>> input = tensor([[-3., -2., -1.], [ 0., 1., 2.]])
    >>> input
    tensor([[-3., -2., -1.],
            [ 0.,  1.,  2.]])
    >>> mask = tensor([[ True, False, True], [False, False, False]])
    >>> mask
    tensor([[ True, False,  True],
            [False, False, False]])
    >>> torch.masked._ops.median(input, 1, mask=mask)
"""

# 注释：定义了一个多行字符串变量，包含了关于 `median` 函数的详细文档字符串，描述了函数的参数、返回值和用法示例。
    # 创建一个张量（tensor），其中包含两个元素：-3.0 和 NaN（Not a Number，不是一个数字）。
    tensor([-3., nan])
# 定义函数 norm，计算输入张量 input 沿指定维度 dim 的范数，根据布尔张量 mask 掩盖部分元素
def norm(input, ord, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor:
    """
    Returns norm of all the elements in the :attr:`input`
    tensor along the given dimension(s) :attr:`dim` while the :attr:`input`
    elements are masked out according to the boolean tensor
    :attr:`mask`.

    The identity value of norm operation, which is used to start the
    reduction, is ``0.0``, except for ``ord=-inf`` it is
    ``inf``.

    If :attr:`keepdim` is ``True``, the output tensor is of the same size
    as :attr:`input` except in the dimension(s) :attr:`dim` where it is of
    size 1. Otherwise, :attr:`dim` is squeezed (see
    :func:`torch.squeeze`), resulting in the output tensor having 1 (or
    ``len(dim)``) fewer dimension(s).

    The boolean tensor :attr:`mask` defines the "validity" of
    :attr:`input` tensor elements: if :attr:`mask` element is True
    then the corresponding element in :attr:`input` tensor will be
    included in norm computation, otherwise the element is
    ignored.

    When all elements of :attr:`input` along the given dimension
    :attr:`dim` are ignored (fully masked-out), the corresponding element
    of the output tensor will have undefined value: it may or may not
    correspond to the identity value of norm operation; the
    choice may correspond to the value that leads to the most efficient
    storage of :attr:`output` tensor.

    The mask of the output tensor can be computed as
    ``torch.any(torch.broadcast_to(mask, input.shape), dim, keepdim=keepdim,
    dtype=torch.bool)``.

    The shapes of the :attr:`mask` tensor and the :attr:`input` tensor
    don't need to match, but they must be :ref:`broadcastable
    <broadcasting-semantics>` and the dimensionality of the :attr:`mask`
    tensor must not be greater than of the :attr:`input` tensor.

    Args:
        input (Tensor): the input tensor
        ord (int, float, optional): the order of vector norm. Default: 2.
          See :func:`torch.linalg.vector_norm` for a list of supported norms.
        dim (int or tuple of ints, optional): the dimension or dimensions to reduce.
          Default: None that is equivalent to ``tuple(range(input.ndim))``.

    Keyword args:
        keepdim (bool, optional): whether the output tensor has
          :attr:`dim` retained or not. Default: False.
        dtype (:class:`torch.dtype`, optional): the desired data type
          of returned tensor.  If specified, the input tensor is
          casted to :attr:`dtype` before the operation is
          performed. Default: None.
        mask (:class:`torch.Tensor`, optional): the boolean tensor
          containing the binary mask of validity of input tensor
          elements.
          Default: None that is equivalent to ``torch.ones(input.shape, dtype=torch.bool)``.

    Example::

        >>> input = tensor([[-3., -2., -1.], [ 0., 1., 2.]])
        >>> input
        tensor([[-3., -2., -1.],
                [ 0.,  1.,  2.]])
        >>> mask = tensor([[ True, False, True], [False, False, False]])
        >>> mask
        tensor([[ True, False,  True],
                [False, False, False]])
        >>> torch.masked._ops.norm(input, 2.0, 1, mask=mask)
    """
    pass
    # 创建一个包含两个元素的张量，数值分别为 3.1623 和 0.0000
    tensor([3.1623, 0.0000])
"""
normalize_docstring = """normalize(input, ord, dim, *, eps=1e-12, dtype=None, mask=None) -> Tensor

Returns normalize of all the slices in the :attr:`input` tensor
along :attr:`dim` while the :attr:`input` elements are masked out
according to the boolean tensor :attr:`mask`.

Let ``x`` be a sequence of unmasked elements of one-dimensional slice
of the :attr:`input` tensor. Normalize of i-th element in ``x`` is
defined as ``x[i]/max(norm(x, p), eps)``.

The boolean tensor :attr:`mask` defines the "validity" of
:attr:`input` tensor elements: if :attr:`mask` element is True then
the corresponding element in :attr:`input` tensor will be included in
normalize computation, otherwise the element is ignored.

The values of masked-out elements of the output tensor have undefined
value: it may or may not be set to zero or nan; the choice may correspond to
the value that leads to the most efficient storage of :attr:`output`
tensor.

The mask of the normalize output tensor can be computed as
``torch.broadcast_to(mask, input.shape)``.

The shapes of the :attr:`mask` tensor and the :attr:`input` tensor
don't need to match, but they must be :ref:`broadcastable
<broadcasting-semantics>` and the dimensionality of the :attr:`mask`
tensor must not be greater than of the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor
    ord (int, float): the order of vector norm. Default: 2.
      See :func:`torch.linalg.vector_norm` for a list of supported norms.
    dim (int): the dimension along which normalize is computed.

Keyword args:
    eps (float, optional): small value to avoid division by zero. Default: 1e-12.
    dtype (:class:`torch.dtype`, optional): the desired data type
      of returned tensor.  If specified, the input tensor is
      casted to :attr:`dtype` before the operation is
      performed. Default: None.
    mask (:class:`torch.Tensor`, optional): the boolean tensor
      containing the binary mask of validity of input tensor
      elements.
      Default: None that is equivalent to ``torch.ones(input.shape, dtype=torch.bool)``.

Example::

    >>> input = tensor([[-3., -2., -1.], [ 0., 1., 2.]])
    >>> input
    tensor([[-3., -2., -1.],
            [ 0.,  1.,  2.]])
    >>> mask = tensor([[ True, False, True], [False, False, False]])
    >>> mask
    tensor([[ True, False,  True],
            [False, False, False]])
    >>> torch.masked._ops.normalize(input, 2.0, 1, mask=mask)
    tensor([[-0.9487,  0.0000, -0.3162],
            [ 0.0000,  0.0000,  0.0000]])
"""

prod_docstring = """prod(input, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor

Returns product of all the elements in the :attr:`input`
tensor along the given dimension(s) :attr:`dim` while the :attr:`input`
elements are masked out according to the boolean tensor
:attr:`mask`.

The identity value of product operation, which is used to start the reduction, is ``1``.

If :attr:`keepdim` is ``True``, the output tensor is of the same size
"""

# 定义函数 prod，计算输入张量 input 沿指定维度 dim 的元素乘积，根据布尔张量 mask 掩码控制元素的有效性
def prod(input, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor:
    """
    Returns product of all the elements in the :attr:`input`
    tensor along the given dimension(s) :attr:`dim` while the :attr:`input`
    elements are masked out according to the boolean tensor
    :attr:`mask`.

    The identity value of product operation, which is used to start the reduction, is ``1``.

    If :attr:`keepdim` is ``True``, the output tensor is of the same size
    """
# 定义了一个函数文档字符串，描述了 softmax 函数的作用及参数说明
softmax(input, dim, *, dtype=None, mask=None) -> Tensor

Returns softmax of all the slices in the :attr:`input` tensor
along :attr:`dim` while the :attr:`input` elements are masked out
according to the boolean tensor :attr:`mask`.

Let ``x`` be a sequence of unmasked elements of one-dimensional slice
of the :attr:`input` tensor. Softmax of i-th element in ``x`` is
defined as ``exp(x[i])/sum(exp(x))``.

The boolean tensor :attr:`mask` defines the "validity" of
:attr:`input` tensor elements: if :attr:`mask` element is True then
the corresponding element in :attr:`input` tensor will be included in
softmax 计算中除了维度 :attr:`dim` 外，其它维度都保持不变。若维度 :attr:`dim` 的尺寸为 1，则会进行挤压操作（参见 :func:`torch.squeeze`），返回的张量将比输入张量少 1（或 ``len(dim)`` 个）维度。

The boolean tensor :attr:`mask` defines the "validity" of
:attr:`input` tensor elements: if :attr:`mask` element is True
then the corresponding element in :attr:`input` tensor will be
included in product computation, otherwise the element is
ignored.

When all elements of :attr:`input` along the given dimension
:attr:`dim` are ignored (fully masked-out), the corresponding element
of the output tensor will have undefined value: it may or may not
correspond to the identity value of product operation; the
choice may correspond to the value that leads to the most efficient
storage of :attr:`output` tensor.

The mask of the output tensor can be computed as
``torch.any(torch.broadcast_to(mask, input.shape), dim, keepdim=keepdim,
dtype=torch.bool)``.

The shapes of the :attr:`mask` tensor and the :attr:`input` tensor
don't need to match, but they must be :ref:`broadcastable
<broadcasting-semantics>` and the dimensionality of the :attr:`mask`
tensor must not be greater than of the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor
    dim (int or tuple of ints, optional): the dimension or dimensions to reduce.
      Default: None that is equivalent to ``tuple(range(input.ndim))``.

Keyword args:
    keepdim (bool, optional): whether the output tensor has
      :attr:`dim` retained or not. Default: False.
    dtype (:class:`torch.dtype`, optional): the desired data type
      of returned tensor.  If specified, the input tensor is
      casted to :attr:`dtype` before the operation is
      performed. Default: None.
    mask (:class:`torch.Tensor`, optional): the boolean tensor
      containing the binary mask of validity of input tensor
      elements.
      Default: None that is equivalent to ``torch.ones(input.shape, dtype=torch.bool)``.

Example::

    >>> input = tensor([[-3, -2, -1], [ 0, 1, 2]])
    >>> input
    tensor([[-3, -2, -1],
            [ 0,  1,  2]])
    >>> mask = tensor([[ True, False, True], [False, False, False]])
    >>> mask
    tensor([[ True, False,  True],
            [False, False, False]])
    >>> torch.masked._ops.prod(input, 1, mask=mask)
    tensor([3, 1])
"""
softmin_docstring = """softmin(input, dim, *, dtype=None, mask=None) -> Tensor
"""

"""
Returns softmin of all the slices in the :attr:`input` tensor
along :attr:`dim` while the :attr:`input` elements are masked out
according to the boolean tensor :attr:`mask`.

Let ``x`` be a sequence of unmasked elements of one-dimensional slice
of the :attr:`input` tensor. Softmin of i-th element in ``x`` is
defined as ``exp(-x[i])/sum(exp(-x))``.

The boolean tensor :attr:`mask` defines the "validity" of
:attr:`input` tensor elements: if :attr:`mask` element is True then
the corresponding element in :attr:`input` tensor will be included in
softmin computation, otherwise the element is ignored.

The values of masked-out elements of the output tensor have undefined
value: it may or may not be set to zero or nan; the choice may correspond to
the value that leads to the most efficient storage of :attr:`output`
tensor.

The mask of the softmin output tensor can be computed as
``torch.broadcast_to(mask, input.shape)``.

The shapes of the :attr:`mask` tensor and the :attr:`input` tensor
don't need to match, but they must be :ref:`broadcastable
<broadcasting-semantics>` and the dimensionality of the :attr:`mask`
tensor must not be greater than of the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor
"""
    dim (int): the dimension along which softmin is computed.
"""
std(input, dim, unbiased, *, keepdim=False, dtype=None, mask=None) -> Tensor
返回沿着给定维度 :attr:`dim` 的输入张量 :attr:`input` 所有元素的标准差，
在此操作中，根据布尔张量 :attr:`mask` 掩盖掉 :attr:`input` 元素。
标准差操作的单位值的身份值未定义。具有步幅布局的输出张量的元素，对应于完全被掩盖的元素，具有 ``nan`` 值。
如果 :attr:`keepdim` 是 ``True``，输出张量的大小与 :attr:`input` 相同，
除了在维度 :attr:`dim`，它的大小为 1。否则，会挤压维度（参见 :func:`torch.squeeze`），
导致输出张量少 1（或 ``len(dim)``）个维度。

布尔张量 :attr:`mask` 定义了 :attr:`input` 张量元素的“有效性”：
如果 :attr:`mask` 元素为 True，则 :attr:`input` 张量中对应的元素将包含在标准差计算中，否则将被忽略。

当沿着给定维度 :attr:`dim` 的所有 :attr:`input` 元素都被忽略（完全被掩盖）时，
输出张量的相应元素将具有未定义的值：它可能或可能不对应于标准差操作的单位值；
选择可能对应于导致 :attr:`output` 张量最有效存储的值。

输出张量的掩码可以计算为
``torch.any(torch.broadcast_to(mask, input.shape), dim, keepdim=keepdim,
dtype=torch.bool)``。

:attr:`mask` 张量和 :attr:`input` 张量的形状不需要匹配，但它们必须是 :ref:`broadcastable
<broadcasting-semantics>`，并且 :attr:`mask` 张量的维度不能大于 :attr:`input` 张量的维度。

参数:
    input (Tensor): 输入张量
    dim (int or tuple of ints, optional): 要减少的维度或维度。默认为 None，相当于 ``tuple(range(input.ndim))``。
    unbiased (bool): 是否使用无偏标准差计算。
    keepdim (bool, optional): 如果为 True，则输出张量与 :attr:`input` 的大小相同，
      除了在维度 :attr:`dim`，它的大小为 1。默认为 False。
    dtype (:class:`torch.dtype`, optional): 返回张量的所需数据类型。
      如果指定，则在执行操作之前，将输入张量转换为 :attr:`dtype`。默认为 None。
    mask (:class:`torch.Tensor`, optional): 包含输入张量元素有效性的二进制掩码的布尔张量。
      默认为 None，相当于 ``torch.ones(input.shape, dtype=torch.bool)``。

示例::

    >>> input = tensor([[-3., -2., -1.], [ 0., 1., 2.]])
    >>> input
    tensor([[-3., -2., -1.],
            [ 0.,  1.,  2.]])
    >>> mask = tensor([[ True, False, True], [False, False, False]])
    >>> mask
    tensor([[ True, False,  True],
            [False, False, False]])
    >>> torch.masked._ops.softmin(input, 1, mask=mask)
    tensor([[0.8808, 0.0000, 0.1192],
            [   nan,    nan,    nan]])
"""
    unbiased (bool): when True, use Bessel's correction, otherwise, compute
      the uncorrected sample variance.
"""
sum_docstring = """sum(input, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor

Returns sum of all the elements in the :attr:`input`
tensor along the given dimension(s) :attr:`dim` while the :attr:`input`
elements are masked out according to the boolean tensor
:attr:`mask`.

The identity value of sum operation, which is used to start the reduction, is ``0``.

If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of
size 1. Otherwise, :attr:`dim` is squeezed (see
:func:`torch.squeeze`), resulting in the output tensor having 1 (or
``len(dim)``) fewer dimension(s).

The boolean tensor :attr:`mask` defines the "validity" of
:attr:`input` tensor elements: if :attr:`mask` element is True
then the corresponding element in :attr:`input` tensor will be
included in sum computation, otherwise the element is
ignored.

When all elements of :attr:`input` along the given dimension
:attr:`dim` are ignored (fully masked-out), the corresponding element
of the output tensor will have undefined value: it may or may not
correspond to the identity value of sum operation; the
choice may correspond to the value that leads to the most efficient
storage of :attr:`output` tensor.

The mask of the output tensor can be computed as
``torch.any(torch.broadcast_to(mask, input.shape), dim, keepdim=keepdim,
dtype=torch.bool)``.

The shapes of the :attr:`mask` tensor and the :attr:`input` tensor
don't need to match, but they must be :ref:`broadcastable
<broadcasting-semantics>` and the dimensionality of the :attr:`mask`
tensor must not be greater than of the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor
    dim (int or tuple of ints, optional): the dimension or dimensions to reduce.
      Default: None that is equivalent to ``tuple(range(input.ndim))``.

Keyword args:
    keepdim (bool, optional): whether the output tensor has
      :attr:`dim` retained or not. Default: False.

"""

"""
注释：
sum_docstring 是一个长文档字符串，描述了 sum 函数的详细说明，包括功能、参数说明、返回值和特殊情况下的行为。
该函数用于计算输入张量沿指定维度的元素总和，根据布尔张量 mask 控制输入元素的有效性。
如果 keepdim 为 True，则输出张量与输入张量大小相同，除了在维度 dim 处大小为 1。否则，dim 被压缩，导致输出张量维度减少。
"""
    # 定义参数 `dtype`，用于指定返回张量的数据类型
    dtype (:class:`torch.dtype`, optional): the desired data type
      of returned tensor.  If specified, the input tensor is
      casted to :attr:`dtype` before the operation is
      performed. Default: None.
    # 定义参数 `mask`，用于指定输入张量元素的有效性二进制掩码张量
    mask (:class:`torch.Tensor`, optional): the boolean tensor
      containing the binary mask of validity of input tensor
      elements.
      # 默认为 `None`，相当于 `torch.ones(input.shape, dtype=torch.bool)`
      Default: None that is equivalent to ``torch.ones(input.shape, dtype=torch.bool)``.
"""
Example::

    >>> input = tensor([[-3, -2, -1], [ 0, 1, 2]])
    >>> input
    tensor([[-3, -2, -1],
            [ 0,  1,  2]])
    >>> mask = tensor([[ True, False, True], [False, False, False]])
    >>> mask
    tensor([[ True, False,  True],
            [False, False, False]])
    >>> torch.masked._ops.sum(input, 1, mask=mask)
    tensor([-4,  0])
"""

var_docstring = """var(input, dim, unbiased, *, keepdim=False, dtype=None, mask=None) -> Tensor
Returns variance of all the elements in the :attr:`input`
tensor along the given dimension(s) :attr:`dim` while the :attr:`input`
elements are masked out according to the boolean tensor
:attr:`mask`.
The identity value of sample variance operation is undefined. The
elements of output tensor with strided layout, that correspond to
fully masked-out elements, have ``nan`` values.
If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of
size 1. Otherwise, :attr:`dim` is squeezed (see
:func:`torch.squeeze`), resulting in the output tensor having 1 (or
``len(dim)``) fewer dimension(s).

The boolean tensor :attr:`mask` defines the "validity" of
:attr:`input` tensor elements: if :attr:`mask` element is True
then the corresponding element in :attr:`input` tensor will be
included in variance computation, otherwise the element is
ignored.

When all elements of :attr:`input` along the given dimension
:attr:`dim` are ignored (fully masked-out), the corresponding element
of the output tensor will have undefined value: it may or may not
correspond to the identity value of variance operation; the
choice may correspond to the value that leads to the most efficient
storage of :attr:`output` tensor.

The mask of the output tensor can be computed as
``torch.any(torch.broadcast_to(mask, input.shape), dim, keepdim=keepdim,
dtype=torch.bool)``.

The shapes of the :attr:`mask` tensor and the :attr:`input` tensor
don't need to match, but they must be :ref:`broadcastable
<broadcasting-semantics>` and the dimensionality of the :attr:`mask`
tensor must not be greater than of the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor
    dim (int or tuple of ints, optional): the dimension or dimensions to reduce.
      Default: None that is equivalent to ``tuple(range(input.ndim))``.
    unbiased (bool): when True, use Bessel's correction, otherwise, compute
      the uncorrected sample variance.

Keyword args:
    keepdim (bool, optional): whether the output tensor has
      :attr:`dim` retained or not. Default: False.
    dtype (:class:`torch.dtype`, optional): the desired data type
      of returned tensor.  If specified, the input tensor is
      casted to :attr:`dtype` before the operation is
      performed. Default: None.

"""
    # 掩码（:class:`torch.Tensor`, optional）：布尔张量，用于表示输入张量元素的有效性的二进制掩码。
    # 默认值为 None，相当于 ``torch.ones(input.shape, dtype=torch.bool)``。
# 创建一个名为 input 的张量，包含两个行和三个列的数据
>>> input = tensor([[-3, -2, -1], [ 0, 1, 2]])
# 显示 input 张量的内容，展示两行三列的数据结构
>>> input
tensor([[-3, -2, -1],
        [ 0,  1,  2]])
# 创建一个名为 mask 的张量，用于指定哪些位置的数据应该被操作
>>> mask = tensor([[ True, False, True], [False, False, False]])
# 显示 mask 张量的内容，展示两行三列的布尔值结构
>>> mask
tensor([[ True, False,  True],
        [False, False, False]])
# 使用 torch.masked._ops.var 函数对 input 张量进行方差计算，沿着第一维度（列）进行计算，
# 不进行无偏估计（False），并且根据 mask 参数指定只计算 True 的位置
>>> torch.masked._ops.var(input, 1, False, mask=mask)
# 返回一个包含两个元素的张量，第一个元素是第一行在 mask 为 True 的位置上的方差，第二个元素因 mask 为 False 而为 NaN
tensor([1., nan])
```