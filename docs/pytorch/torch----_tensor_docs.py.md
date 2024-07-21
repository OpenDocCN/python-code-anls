# `.\pytorch\torch\_tensor_docs.py`

```
# 添加 docstring 到 Tensor 函数

import torch._C  # 导入 torch._C 模块
from torch._C import _add_docstr as add_docstr  # 导入 _add_docstr 函数
from torch._torch_docs import parse_kwargs, reproducibility_notes  # 导入 parse_kwargs 和 reproducibility_notes 函数

# 定义函数 add_docstr_all，用于给指定方法添加文档字符串
def add_docstr_all(method, docstr):
    add_docstr(getattr(torch._C.TensorBase, method), docstr)

# 解析常见参数
common_args = parse_kwargs(
    """
    memory_format (:class:`torch.memory_format`, optional): the desired memory format of
        returned Tensor. Default: ``torch.preserve_format``.
"""
)

# 解析新的常见参数
new_common_args = parse_kwargs(
    """
    size (int...): a list, tuple, or :class:`torch.Size` of integers defining the
        shape of the output tensor.
    dtype (:class:`torch.dtype`, optional): the desired type of returned tensor.
        Default: if None, same :class:`torch.dtype` as this tensor.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if None, same :class:`torch.device` as this tensor.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.
    pin_memory (bool, optional): If set, returned tensor would be allocated in
        the pinned memory. Works only for CPU tensors. Default: ``False``.
    layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
        Default: ``torch.strided``.
"""
)

# 给 new_tensor 方法添加文档字符串
add_docstr_all(
    "new_tensor",
    """
new_tensor(data, *, dtype=None, device=None, requires_grad=False, layout=torch.strided, \
pin_memory=False) -> Tensor
"""
    + r"""

Returns a new Tensor with :attr:`data` as the tensor data.
By default, the returned Tensor has the same :class:`torch.dtype` and
:class:`torch.device` as this tensor.

.. warning::

    :func:`new_tensor` always copies :attr:`data`. If you have a Tensor
    ``data`` and want to avoid a copy, use :func:`torch.Tensor.requires_grad_`
    or :func:`torch.Tensor.detach`.
    If you have a numpy array and want to avoid a copy, use
    :func:`torch.from_numpy`.

.. warning::

    When data is a tensor `x`, :func:`new_tensor()` reads out 'the data' from whatever it is passed,
    and constructs a leaf variable. Therefore ``tensor.new_tensor(x)`` is equivalent to ``x.clone().detach()``
    and ``tensor.new_tensor(x, requires_grad=True)`` is equivalent to ``x.clone().detach().requires_grad_(True)``.
    The equivalents using ``clone()`` and ``detach()`` are recommended.

Args:
    data (array_like): The returned Tensor copies :attr:`data`.

Keyword args:
    {dtype}  # 描述 dtype 参数
    {device}  # 描述 device 参数
    {requires_grad}  # 描述 requires_grad 参数
    {layout}  # 描述 layout 参数
    {pin_memory}  # 描述 pin_memory 参数

Example::

    >>> tensor = torch.ones((2,), dtype=torch.int8)
    >>> data = [[0, 1], [2, 3]]
    >>> tensor.new_tensor(data)
    tensor([[ 0,  1],
            [ 2,  3]], dtype=torch.int8)

""".format(
        **new_common_args
    ),
)

# 给 new_full 方法添加文档字符串
add_docstr_all(
    "new_full",
    """
new_full(size, fill_value, *, dtype=None, device=None, requires_grad=False, layout=torch.strided, \
pin_memory=False) -> Tensor
"""
    + r"""
# 定义函数 new_full，返回一个指定大小的 Tensor，用 fill_value 填充
def new_full(size, fill_value, *, dtype=None, device=None, requires_grad=False, layout=torch.strided, pin_memory=False):
    """
    new_full(size, fill_value, *, dtype=None, device=None, requires_grad=False, layout=torch.strided, pin_memory=False) -> Tensor

    Returns a Tensor of size :attr:`size` filled with :attr:`fill_value`.
    By default, the returned Tensor has the same :class:`torch.dtype` and
    :class:`torch.device` as this tensor.

    Args:
        fill_value (scalar): the number to fill the output tensor with.

    Keyword args:
        {dtype}
        {device}
        {requires_grad}
        {layout}
        {pin_memory}

    Example::

        >>> tensor = torch.ones((2,), dtype=torch.float64)
        >>> tensor.new_full((3, 4), 3.141592)
        tensor([[ 3.1416,  3.1416,  3.1416,  3.1416],
                [ 3.1416,  3.1416,  3.1416,  3.1416],
                [ 3.1416,  3.1416,  3.1416,  3.1416]], dtype=torch.float64)
    """
    pass

# 定义函数 new_empty，返回一个未初始化数据的 Tensor
def new_empty(size, *, dtype=None, device=None, requires_grad=False, layout=torch.strided, pin_memory=False):
    """
    new_empty(size, *, dtype=None, device=None, requires_grad=False, layout=torch.strided, pin_memory=False) -> Tensor

    Returns a Tensor of size :attr:`size` filled with uninitialized data.
    By default, the returned Tensor has the same :class:`torch.dtype` and
    :class:`torch.device` as this tensor.

    Args:
        size (int...): a list, tuple, or :class:`torch.Size` of integers defining the
            shape of the output tensor.

    Keyword args:
        {dtype}
        {device}
        {requires_grad}
        {layout}
        {pin_memory}

    Example::

        >>> tensor = torch.ones(())
        >>> tensor.new_empty((2, 3))
        tensor([[ 5.8182e-18,  4.5765e-41, -1.0545e+30],
                [ 3.0949e-41,  4.4842e-44,  0.0000e+00]])
    """
    pass

# 定义函数 new_empty_strided，返回一个指定大小和步幅的未初始化数据的 Tensor
def new_empty_strided(size, stride, *, dtype=None, device=None, requires_grad=False, layout=torch.strided, pin_memory=False):
    """
    new_empty_strided(size, stride, *, dtype=None, device=None, requires_grad=False, layout=torch.strided, pin_memory=False) -> Tensor

    Returns a Tensor of size :attr:`size` and strides :attr:`stride` filled with
    uninitialized data. By default, the returned Tensor has the same
    :class:`torch.dtype` and :class:`torch.device` as this tensor.

    Args:
        size (int...): a list, tuple, or :class:`torch.Size` of integers defining the
            shape of the output tensor.
        stride (int...): a list, tuple, or :class:`torch.Size` of integers defining the
            stride of the output tensor.

    Keyword args:
        {dtype}
        {device}
        {requires_grad}
        {layout}
        {pin_memory}

    Example::

        >>> tensor = torch.ones(())
        >>> tensor.new_empty_strided((2, 3), (3, 1))
        tensor([[ 5.8182e-18,  4.5765e-41, -1.0545e+30],
                [ 3.0949e-41,  4.4842e-44,  0.0000e+00]])
    """
    pass

# 定义函数 new_ones，返回一个指定大小的 Tensor，用 1 填充
def new_ones(size, *, dtype=None, device=None, requires_grad=False, layout=torch.strided, pin_memory=False):
    """
    new_ones(size, *, dtype=None, device=None, requires_grad=False, layout=torch.strided, pin_memory=False) -> Tensor

    Returns a Tensor of size :attr:`size` filled with ``1``.
    By default, the returned Tensor has the same :class:`torch.dtype` and
    :class:`torch.device` as this tensor.

    Args:
        size (int...): a list, tuple, or :class:`torch.Size` of integers defining the
            shape of the output tensor.

    Keyword args:
        {dtype}
        {device}
        {requires_grad}
        {layout}
        {pin_memory}

    Example::

        >>> tensor = torch.tensor((), dtype=torch.int32)
        >>> tensor.new_ones((2, 3))
    """
    pass
    # 创建一个二维张量，包含两行三列的整数数据，数据类型为32位整数
    tensor([[ 1,  1,  1],
            [ 1,  1,  1]], dtype=torch.int32)
"""
add_docstr_all(
    "new_zeros",
    """
    new_zeros(size, *, dtype=None, device=None, requires_grad=False, layout=torch.strided, \
    pin_memory=False) -> Tensor
    """
    + r"""
    
    Returns a Tensor of size :attr:`size` filled with ``0``.
    By default, the returned Tensor has the same :class:`torch.dtype` and
    :class:`torch.device` as this tensor.
    
    Args:
        size (int...): a list, tuple, or :class:`torch.Size` of integers defining the
            shape of the output tensor.
    
    Keyword args:
        {dtype}
        {device}
        {requires_grad}
        {layout}
        {pin_memory}
    
    Example::
    
        >>> tensor = torch.tensor((), dtype=torch.float64)
        >>> tensor.new_zeros((2, 3))
        tensor([[ 0.,  0.,  0.],
                [ 0.,  0.,  0.]], dtype=torch.float64)
    
    """.format(
        **new_common_args
    ),
)

add_docstr_all(
    "abs",
    r"""
    abs() -> Tensor
    
    See :func:`torch.abs`
    """,
)

add_docstr_all(
    "abs_",
    r"""
    abs_() -> Tensor
    
    In-place version of :meth:`~Tensor.abs`
    """,
)

add_docstr_all(
    "absolute",
    r"""
    absolute() -> Tensor
    
    Alias for :func:`abs`
    """,
)

add_docstr_all(
    "absolute_",
    r"""
    absolute_() -> Tensor
    
    In-place version of :meth:`~Tensor.absolute`
    Alias for :func:`abs_`
    """,
)

add_docstr_all(
    "acos",
    r"""
    acos() -> Tensor
    
    See :func:`torch.acos`
    """,
)

add_docstr_all(
    "acos_",
    r"""
    acos_() -> Tensor
    
    In-place version of :meth:`~Tensor.acos`
    """,
)

add_docstr_all(
    "arccos",
    r"""
    arccos() -> Tensor
    
    See :func:`torch.arccos`
    """,
)

add_docstr_all(
    "arccos_",
    r"""
    arccos_() -> Tensor
    
    In-place version of :meth:`~Tensor.arccos`
    """,
)

add_docstr_all(
    "acosh",
    r"""
    acosh() -> Tensor
    
    See :func:`torch.acosh`
    """,
)

add_docstr_all(
    "acosh_",
    r"""
    acosh_() -> Tensor
    
    In-place version of :meth:`~Tensor.acosh`
    """,
)

add_docstr_all(
    "arccosh",
    r"""
    acosh() -> Tensor
    
    See :func:`torch.arccosh`
    """,
)

add_docstr_all(
    "arccosh_",
    r"""
    acosh_() -> Tensor
    
    In-place version of :meth:`~Tensor.arccosh`
    """,
)

add_docstr_all(
    "add",
    r"""
    add(other, *, alpha=1) -> Tensor
    
    Add a scalar or tensor to :attr:`self` tensor. If both :attr:`alpha`
    and :attr:`other` are specified, each element of :attr:`other` is scaled by
    :attr:`alpha` before being used.
    
    When :attr:`other` is a tensor, the shape of :attr:`other` must be
    :ref:`broadcastable <broadcasting-semantics>` with the shape of the underlying
    tensor
    
    See :func:`torch.add`
    """,
)

add_docstr_all(
    "add_",
    r"""
    add_(other, *, alpha=1) -> Tensor
    
    In-place version of :meth:`~Tensor.add`
    """,
)

add_docstr_all(
    "addbmm",
    r"""
    addbmm(batch1, batch2, *, beta=1, alpha=1) -> Tensor
    
    See :func:`torch.addbmm`
    """,
)

add_docstr_all(
    "addbmm_",
    r"""
    addbmm_(batch1, batch2, *, beta=1, alpha=1) -> Tensor
    
    In-place version of :meth:`~Tensor.addbmm`
    """,
)

add_docstr_all(
    "addcdiv",
    r"""
    addcdiv(tensor1, tensor2, *, value=1) -> Tensor
    
    See :func:`torch.addcdiv`
    """,
)
add_docstr_all(
    "addcdiv_",
    r"""
addcdiv_(tensor1, tensor2, *, value=1) -> Tensor

In-place version of :meth:`~Tensor.addcdiv`
""",
)

add_docstr_all(
    "addcmul",
    r"""
addcmul(tensor1, tensor2, *, value=1) -> Tensor

See :func:`torch.addcmul`
""",
)

add_docstr_all(
    "addcmul_",
    r"""
addcmul_(tensor1, tensor2, *, value=1) -> Tensor

In-place version of :meth:`~Tensor.addcmul`
""",
)

add_docstr_all(
    "addmm",
    r"""
addmm(mat1, mat2, *, beta=1, alpha=1) -> Tensor

See :func:`torch.addmm`
""",
)

add_docstr_all(
    "addmm_",
    r"""
addmm_(mat1, mat2, *, beta=1, alpha=1) -> Tensor

In-place version of :meth:`~Tensor.addmm`
""",
)

add_docstr_all(
    "addmv",
    r"""
addmv(mat, vec, *, beta=1, alpha=1) -> Tensor

See :func:`torch.addmv`
""",
)

add_docstr_all(
    "addmv_",
    r"""
addmv_(mat, vec, *, beta=1, alpha=1) -> Tensor

In-place version of :meth:`~Tensor.addmv`
""",
)

add_docstr_all(
    "sspaddmm",
    r"""
sspaddmm(mat1, mat2, *, beta=1, alpha=1) -> Tensor

See :func:`torch.sspaddmm`
""",
)

add_docstr_all(
    "smm",
    r"""
smm(mat) -> Tensor

See :func:`torch.smm`
""",
)

add_docstr_all(
    "addr",
    r"""
addr(vec1, vec2, *, beta=1, alpha=1) -> Tensor

See :func:`torch.addr`
""",
)

add_docstr_all(
    "addr_",
    r"""
addr_(vec1, vec2, *, beta=1, alpha=1) -> Tensor

In-place version of :meth:`~Tensor.addr`
""",
)

add_docstr_all(
    "align_as",
    r"""
align_as(other) -> Tensor

Permutes the dimensions of the :attr:`self` tensor to match the dimension order
in the :attr:`other` tensor, adding size-one dims for any new names.

This operation is useful for explicit broadcasting by names (see examples).

All of the dims of :attr:`self` must be named in order to use this method.
The resulting tensor is a view on the original tensor.

All dimension names of :attr:`self` must be present in ``other.names``.
:attr:`other` may contain named dimensions that are not in ``self.names``;
the output tensor has a size-one dimension for each of those new names.

To align a tensor to a specific order, use :meth:`~Tensor.align_to`.

Examples::

    # Example 1: Applying a mask
    >>> mask = torch.randint(2, [127, 128], dtype=torch.bool).refine_names('W', 'H')
    >>> imgs = torch.randn(32, 128, 127, 3, names=('N', 'H', 'W', 'C'))
    >>> imgs.masked_fill_(mask.align_as(imgs), 0)


    # Example 2: Applying a per-channel-scale
    >>> def scale_channels(input, scale):
    >>>    scale = scale.refine_names('C')
    >>>    return input * scale.align_as(input)

    >>> num_channels = 3
    >>> scale = torch.randn(num_channels, names=('C',))
    >>> imgs = torch.rand(32, 128, 128, num_channels, names=('N', 'H', 'W', 'C'))
    >>> more_imgs = torch.rand(32, num_channels, 128, 128, names=('N', 'C', 'H', 'W'))
    >>> videos = torch.randn(3, num_channels, 128, 128, 128, names=('N', 'C', 'H', 'W', 'D'))

    # scale_channels is agnostic to the dimension order of the input
    >>> scale_channels(imgs, scale)

""",
)
    # 调用名为 scale_channels 的函数，并传入参数 more_imgs 和 scale
    scale_channels(more_imgs, scale)
    # 调用名为 scale_channels 的函数，并传入参数 videos 和 scale
    scale_channels(videos, scale)
# 为函数"all"添加文档字符串，描述其参数和功能，引用了torch中的torch.all函数
add_docstr_all(
    "all",
    r"""
all(dim=None, keepdim=False) -> Tensor

See :func:`torch.all`
""",
)

# 为函数"allclose"添加文档字符串，描述其参数和功能，引用了torch中的torch.allclose函数
add_docstr_all(
    "allclose",
    r"""
allclose(other, rtol=1e-05, atol=1e-08, equal_nan=False) -> Tensor

See :func:`torch.allclose`
""",
)

# 为函数"angle"添加文档字符串，描述其参数和功能，引用了torch中的torch.angle函数
add_docstr_all(
    "angle",
    r"""
angle() -> Tensor

See :func:`torch.angle`
""",
)

# 为函数"any"添加文档字符串，描述其参数和功能，引用了torch中的torch.any函数
add_docstr_all(
    "any",
    r"""
any(dim=None, keepdim=False) -> Tensor

See :func:`torch.any`
""",
)

# 为函数"apply_"添加文档字符串，描述其参数和功能，包括其只适用于CPU张量和性能注意事项
add_docstr_all(
    "apply_",
    r"""
apply_(callable) -> Tensor

Applies the function :attr:`callable` to each element in the tensor, replacing
each element with the value returned by :attr:`callable`.

.. note::

    This function only works with CPU tensors and should not be used in code
    sections that require high performance.
""",
)

# 为函数"asin"添加文档字符串，描述其参数和功能，引用了torch中的torch.asin函数
add_docstr_all(
    "asin",
    r"""
asin() -> Tensor

See :func:`torch.asin`
""",
)

# 为函数"asin_"添加文档字符串，描述其参数和功能，是torch.Tensor.asin的原地版本
add_docstr_all(
    "asin_",
    r"""
asin_() -> Tensor

In-place version of :meth:`~Tensor.asin`
""",
)

# 为函数"arcsin"添加文档字符串，描述其参数和功能，引用了torch中的torch.arcsin函数
add_docstr_all(
    "arcsin",
    r"""
arcsin() -> Tensor

See :func:`torch.arcsin`
""",
)

# 为函数"arcsin_"添加文档字符串，描述其参数和功能，是torch.Tensor.arcsin的原地版本
add_docstr_all(
    "arcsin_",
    r"""
arcsin_() -> Tensor

In-place version of :meth:`~Tensor.arcsin`
""",
)

# 为函数"asinh"添加文档字符串，描述其参数和功能，引用了torch中的torch.asinh函数
add_docstr_all(
    "asinh",
    r"""
asinh() -> Tensor

See :func:`torch.asinh`
""",
)

# 为函数"asinh_"添加文档字符串，描述其参数和功能，是torch.Tensor.asinh的原地版本
add_docstr_all(
    "asinh_",
    r"""
asinh_() -> Tensor

In-place version of :meth:`~Tensor.asinh`
""",
)

# 为函数"arcsinh"添加文档字符串，描述其参数和功能，引用了torch中的torch.arcsinh函数
add_docstr_all(
    "arcsinh",
    r"""
arcsinh() -> Tensor

See :func:`torch.arcsinh`
""",
)

# 为函数"arcsinh_"添加文档字符串，描述其参数和功能，是torch.Tensor.arcsinh的原地版本
add_docstr_all(
    "arcsinh_",
    r"""
arcsinh_() -> Tensor

In-place version of :meth:`~Tensor.arcsinh`
""",
)

# 为函数"as_strided"添加文档字符串，描述其参数和功能，引用了torch中的torch.as_strided函数
add_docstr_all(
    "as_strided",
    r"""
as_strided(size, stride, storage_offset=None) -> Tensor

See :func:`torch.as_strided`
""",
)

# 为函数"as_strided_"添加文档字符串，描述其参数和功能，是torch.Tensor.as_strided的原地版本
add_docstr_all(
    "as_strided_",
    r"""
as_strided_(size, stride, storage_offset=None) -> Tensor

In-place version of :meth:`~Tensor.as_strided`
""",
)

# 为函数"atan"添加文档字符串，描述其参数和功能，引用了torch中的torch.atan函数
add_docstr_all(
    "atan",
    r"""
atan() -> Tensor

See :func:`torch.atan`
""",
)

# 为函数"atan_"添加文档字符串，描述其参数和功能，是torch.Tensor.atan的原地版本
add_docstr_all(
    "atan_",
    r"""
atan_() -> Tensor

In-place version of :meth:`~Tensor.atan`
""",
)

# 为函数"arctan"添加文档字符串，描述其参数和功能，引用了torch中的torch.arctan函数
add_docstr_all(
    "arctan",
    r"""
arctan() -> Tensor

See :func:`torch.arctan`
""",
)

# 为函数"arctan_"添加文档字符串，描述其参数和功能，是torch.Tensor.arctan的原地版本
add_docstr_all(
    "arctan_",
    r"""
arctan_() -> Tensor

In-place version of :meth:`~Tensor.arctan`
""",
)

# 为函数"atan2"添加文档字符串，描述其参数和功能，引用了torch中的torch.atan2函数
add_docstr_all(
    "atan2",
    r"""
atan2(other) -> Tensor

See :func:`torch.atan2`
""",
)

# 为函数"atan2_"添加文档字符串，描述其参数和功能，是torch.Tensor.atan2的原地版本
add_docstr_all(
    "atan2_",
    r"""
atan2_(other) -> Tensor

In-place version of :meth:`~Tensor.atan2`
""",
)

# 为函数"arctan2"添加文档字符串，描述其参数和功能，引用了torch中的torch.arctan2函数
add_docstr_all(
    "arctan2",
    r"""
arctan2(other) -> Tensor

See :func:`torch.arctan2`
""",
)

# 为函数"arctan2_"添加文档字符串，描述其参数和功能，是torch.Tensor.arctan2的原地版本
add_docstr_all(
    "arctan2_",
    r"""
atan2_(other) -> Tensor

In-place version of :meth:`~Tensor.arctan2`
""",
)

# 为函数"atanh"添加文档字符串，描述其参数和功能，引用了torch中的torch.atanh函数
add_docstr_all(
    "atanh",
    r"""
atanh() -> Tensor

See :func:`torch.atanh`
""",
)

# 为函数"atanh_"添加文档字符串，描述其参数和功能，是torch.Tensor.atanh的原地版本
add_docstr_all(
    "atanh_",
    r"""
atanh_(other) -> Tensor

In-place version of :meth:`~Tensor.atanh`
""",
)
# 添加文档字符串到函数 "arctanh"
add_docstr_all(
    "arctanh",
    r"""
arctanh() -> Tensor

See :func:`torch.arctanh`
""",
)

# 添加文档字符串到方法 "arctanh_"
add_docstr_all(
    "arctanh_",
    r"""
arctanh_(other) -> Tensor

In-place version of :meth:`~Tensor.arctanh`
""",
)

# 添加文档字符串到函数 "baddbmm"
add_docstr_all(
    "baddbmm",
    r"""
baddbmm(batch1, batch2, *, beta=1, alpha=1) -> Tensor

See :func:`torch.baddbmm`
""",
)

# 添加文档字符串到方法 "baddbmm_"
add_docstr_all(
    "baddbmm_",
    r"""
baddbmm_(batch1, batch2, *, beta=1, alpha=1) -> Tensor

In-place version of :meth:`~Tensor.baddbmm`
""",
)

# 添加文档字符串到函数 "bernoulli"
add_docstr_all(
    "bernoulli",
    r"""
bernoulli(*, generator=None) -> Tensor

Returns a result tensor where each :math:`\texttt{result[i]}` is independently
sampled from :math:`\text{Bernoulli}(\texttt{self[i]})`. :attr:`self` must have
floating point ``dtype``, and the result will have the same ``dtype``.

See :func:`torch.bernoulli`
""",
)

# 添加文档字符串到方法 "bernoulli_"
add_docstr_all(
    "bernoulli_",
    r"""
bernoulli_(p=0.5, *, generator=None) -> Tensor

Fills each location of :attr:`self` with an independent sample from
:math:`\text{Bernoulli}(\texttt{p})`. :attr:`self` can have integral
``dtype``.

:attr:`p` should either be a scalar or tensor containing probabilities to be
used for drawing the binary random number.

If it is a tensor, the :math:`\text{i}^{th}` element of :attr:`self` tensor
will be set to a value sampled from
:math:`\text{Bernoulli}(\texttt{p\_tensor[i]})`. In this case `p` must have
floating point ``dtype``.

See also :meth:`~Tensor.bernoulli` and :func:`torch.bernoulli`
""",
)

# 添加文档字符串到函数 "bincount"
add_docstr_all(
    "bincount",
    r"""
bincount(weights=None, minlength=0) -> Tensor

See :func:`torch.bincount`
""",
)

# 添加文档字符串到函数 "bitwise_not"
add_docstr_all(
    "bitwise_not",
    r"""
bitwise_not() -> Tensor

See :func:`torch.bitwise_not`
""",
)

# 添加文档字符串到方法 "bitwise_not_"
add_docstr_all(
    "bitwise_not_",
    r"""
bitwise_not_() -> Tensor

In-place version of :meth:`~Tensor.bitwise_not`
""",
)

# 添加文档字符串到函数 "bitwise_and"
add_docstr_all(
    "bitwise_and",
    r"""
bitwise_and() -> Tensor

See :func:`torch.bitwise_and`
""",
)

# 添加文档字符串到方法 "bitwise_and_"
add_docstr_all(
    "bitwise_and_",
    r"""
bitwise_and_() -> Tensor

In-place version of :meth:`~Tensor.bitwise_and`
""",
)

# 添加文档字符串到函数 "bitwise_or"
add_docstr_all(
    "bitwise_or",
    r"""
bitwise_or() -> Tensor

See :func:`torch.bitwise_or`
""",
)

# 添加文档字符串到方法 "bitwise_or_"
add_docstr_all(
    "bitwise_or_",
    r"""
bitwise_or_() -> Tensor

In-place version of :meth:`~Tensor.bitwise_or`
""",
)

# 添加文档字符串到函数 "bitwise_xor"
add_docstr_all(
    "bitwise_xor",
    r"""
bitwise_xor() -> Tensor

See :func:`torch.bitwise_xor`
""",
)

# 添加文档字符串到方法 "bitwise_xor_"
add_docstr_all(
    "bitwise_xor_",
    r"""
bitwise_xor_() -> Tensor

In-place version of :meth:`~Tensor.bitwise_xor`
""",
)

# 添加文档字符串到函数 "bitwise_left_shift"
add_docstr_all(
    "bitwise_left_shift",
    r"""
bitwise_left_shift(other) -> Tensor

See :func:`torch.bitwise_left_shift`
""",
)

# 添加文档字符串到方法 "bitwise_left_shift_"
add_docstr_all(
    "bitwise_left_shift_",
    r"""
bitwise_left_shift_(other) -> Tensor

In-place version of :meth:`~Tensor.bitwise_left_shift`
""",
)

# 添加文档字符串到函数 "bitwise_right_shift"
add_docstr_all(
    "bitwise_right_shift",
    r"""
bitwise_right_shift(other) -> Tensor

See :func:`torch.bitwise_right_shift`
""",
)
    "bitwise_right_shift_",  # 定义一个字符串常量 "bitwise_right_shift_"
    r"""  # 开始一个原始字符串字面量的定义
add_docstr_all(
    "bitwise_right_shift_",
    r"""
bitwise_right_shift_(other) -> Tensor

In-place version of :meth:`~Tensor.bitwise_right_shift`
""",
)

add_docstr_all(
    "broadcast_to",
    r"""
broadcast_to(shape) -> Tensor

See :func:`torch.broadcast_to`.
""",
)

add_docstr_all(
    "logical_and",
    r"""
logical_and() -> Tensor

See :func:`torch.logical_and`
""",
)

add_docstr_all(
    "logical_and_",
    r"""
logical_and_() -> Tensor

In-place version of :meth:`~Tensor.logical_and`
""",
)

add_docstr_all(
    "logical_not",
    r"""
logical_not() -> Tensor

See :func:`torch.logical_not`
""",
)

add_docstr_all(
    "logical_not_",
    r"""
logical_not_() -> Tensor

In-place version of :meth:`~Tensor.logical_not`
""",
)

add_docstr_all(
    "logical_or",
    r"""
logical_or() -> Tensor

See :func:`torch.logical_or`
""",
)

add_docstr_all(
    "logical_or_",
    r"""
logical_or_() -> Tensor

In-place version of :meth:`~Tensor.logical_or`
""",
)

add_docstr_all(
    "logical_xor",
    r"""
logical_xor() -> Tensor

See :func:`torch.logical_xor`
""",
)

add_docstr_all(
    "logical_xor_",
    r"""
logical_xor_() -> Tensor

In-place version of :meth:`~Tensor.logical_xor`
""",
)

add_docstr_all(
    "bmm",
    r"""
bmm(batch2) -> Tensor

See :func:`torch.bmm`
""",
)

add_docstr_all(
    "cauchy_",
    r"""
cauchy_(median=0, sigma=1, *, generator=None) -> Tensor

Fills the tensor with numbers drawn from the Cauchy distribution:

.. math::

    f(x) = \dfrac{1}{\pi} \dfrac{\sigma}{(x - \text{median})^2 + \sigma^2}

.. note::
  Sigma (:math:`\sigma`) is used to denote the scale parameter in Cauchy distribution.
""",
)

add_docstr_all(
    "ceil",
    r"""
ceil() -> Tensor

See :func:`torch.ceil`
""",
)

add_docstr_all(
    "ceil_",
    r"""
ceil_() -> Tensor

In-place version of :meth:`~Tensor.ceil`
""",
)

add_docstr_all(
    "cholesky",
    r"""
cholesky(upper=False) -> Tensor

See :func:`torch.cholesky`
""",
)

add_docstr_all(
    "cholesky_solve",
    r"""
cholesky_solve(input2, upper=False) -> Tensor

See :func:`torch.cholesky_solve`
""",
)

add_docstr_all(
    "cholesky_inverse",
    r"""
cholesky_inverse(upper=False) -> Tensor

See :func:`torch.cholesky_inverse`
""",
)

add_docstr_all(
    "clamp",
    r"""
clamp(min=None, max=None) -> Tensor

See :func:`torch.clamp`
""",
)

add_docstr_all(
    "clamp_",
    r"""
clamp_(min=None, max=None) -> Tensor

In-place version of :meth:`~Tensor.clamp`
""",
)

add_docstr_all(
    "clip",
    r"""
clip(min=None, max=None) -> Tensor

Alias for :meth:`~Tensor.clamp`.
""",
)

add_docstr_all(
    "clip_",
    r"""
clip_(min=None, max=None) -> Tensor

Alias for :meth:`~Tensor.clamp_`.
""",
)

add_docstr_all(
    "clone",
    r"""
clone(*, memory_format=torch.preserve_format) -> Tensor

See :func:`torch.clone`
""",
)

add_docstr_all(
    "coalesce",
    r"""
coalesce() -> Tensor

Returns a coalesced copy of :attr:`self` if :attr:`self` is an
:ref:`uncoalesced tensor <sparse-uncoalesced-coo-docs>`.
""",
)
add_docstr_all(
    "contiguous",
    r"""
contiguous(memory_format=torch.contiguous_format) -> Tensor

Returns a contiguous in memory tensor containing the same data as :attr:`self` tensor. If
:attr:`self` tensor is already in the specified memory format, this function returns the
:attr:`self` tensor.

Args:
    memory_format (:class:`torch.memory_format`, optional): the desired memory format of
        returned Tensor. Default: ``torch.contiguous_format``.
""",
)

add_docstr_all(
    "copy_",
    r"""
copy_(src, non_blocking=False) -> Tensor

Copies the elements from :attr:`src` into :attr:`self` tensor and returns
:attr:`self`.

The :attr:`src` tensor must be :ref:`broadcastable <broadcasting-semantics>`
with the :attr:`self` tensor. It may be of a different data type or reside on a
different device.

Args:
    src (Tensor): the source tensor to copy from
    non_blocking (bool): if ``True`` and this copy is between CPU and GPU,
        the copy may occur asynchronously with respect to the host. For other
        cases, this argument has no effect.
""",
)

add_docstr_all(
    "conj",
    r"""
conj() -> Tensor

See :func:`torch.conj`
""",
)

add_docstr_all(
    "conj_physical",
    r"""
conj_physical() -> Tensor

See :func:`torch.conj_physical`
""",
)

add_docstr_all(
    "conj_physical_",
    r"""
conj_physical_() -> Tensor

In-place version of :meth:`~Tensor.conj_physical`
""",
)

add_docstr_all(
    "resolve_conj",
    r"""
resolve_conj() -> Tensor

See :func:`torch.resolve_conj`
""",
)

add_docstr_all(
    "resolve_neg",
    r"""
resolve_neg() -> Tensor

See :func:`torch.resolve_neg`
""",
)

add_docstr_all(
    "copysign",
    r"""
copysign(other) -> Tensor

See :func:`torch.copysign`
""",
)

add_docstr_all(
    "copysign_",
    r"""
copysign_(other) -> Tensor

In-place version of :meth:`~Tensor.copysign`
""",
)

add_docstr_all(
    "cos",
    r"""
cos() -> Tensor

See :func:`torch.cos`
""",
)

add_docstr_all(
    "cos_",
    r"""
cos_() -> Tensor

In-place version of :meth:`~Tensor.cos`
""",
)

add_docstr_all(
    "cosh",
    r"""
cosh() -> Tensor

See :func:`torch.cosh`
""",
)

add_docstr_all(
    "cosh_",
    r"""
cosh_() -> Tensor

In-place version of :meth:`~Tensor.cosh`
""",
)

add_docstr_all(
    "cpu",
    r"""
cpu(memory_format=torch.preserve_format) -> Tensor

Returns a copy of this object in CPU memory.

If this object is already in CPU memory and on the correct device,
then no copy is performed and the original object is returned.

Args:
    memory_format (:class:`torch.memory_format`, optional): the desired memory format of
        returned Tensor. Default: ``torch.preserve_format``.
""",
)

add_docstr_all(
    "count_nonzero",
    r"""
count_nonzero(dim=None) -> Tensor

See :func:`torch.count_nonzero`
""",
)

add_docstr_all(
    "cov",
    r"""
cov(*, correction=1, fweights=None, aweights=None) -> Tensor

See :func:`torch.cov`
""",
)

add_docstr_all(
    "corrcoef",
    r"""
corrcoef(*, correction=1) -> Tensor

See :func:`torch.corrcoef`
""",
)
# 定义了一个文档字符串，描述了 `corrcoef()` 函数的作用
"""
See :func:`torch.corrcoef`
"""



# 为 `cross` 方法添加了文档字符串，描述了其功能和使用方法
"""
cross(other, dim=None) -> Tensor

See :func:`torch.cross`
"""



# 为 `cuda` 方法添加了文档字符串，详细描述了其参数和功能
"""
cuda(device=None, non_blocking=False, memory_format=torch.preserve_format) -> Tensor

Returns a copy of this object in CUDA memory.

If this object is already in CUDA memory and on the correct device,
then no copy is performed and the original object is returned.

Args:
    device (:class:`torch.device`): The destination GPU device.
        Defaults to the current CUDA device.
    non_blocking (bool): If ``True`` and the source is in pinned memory,
        the copy will be asynchronous with respect to the host.
        Otherwise, the argument has no effect. Default: ``False``.
    memory_format (torch.memory_format): Desired memory format for the result Tensor.
        Default: ``torch.preserve_format``.
"""



# 为 `ipu` 方法添加了文档字符串，详细描述了其参数和功能
"""
ipu(device=None, non_blocking=False, memory_format=torch.preserve_format) -> Tensor

Returns a copy of this object in IPU memory.

If this object is already in IPU memory and on the correct device,
then no copy is performed and the original object is returned.

Args:
    device (:class:`torch.device`): The destination IPU device.
        Defaults to the current IPU device.
    non_blocking (bool): If ``True`` and the source is in pinned memory,
        the copy will be asynchronous with respect to the host.
        Otherwise, the argument has no effect. Default: ``False``.
    memory_format (torch.memory_format): Desired memory format for the result Tensor.
        Default: ``torch.preserve_format``.
"""



# 为 `xpu` 方法添加了文档字符串，详细描述了其参数和功能
"""
xpu(device=None, non_blocking=False, memory_format=torch.preserve_format) -> Tensor

Returns a copy of this object in XPU memory.

If this object is already in XPU memory and on the correct device,
then no copy is performed and the original object is returned.

Args:
    device (:class:`torch.device`): The destination XPU device.
        Defaults to the current XPU device.
    non_blocking (bool): If ``True`` and the source is in pinned memory,
        the copy will be asynchronous with respect to the host.
        Otherwise, the argument has no effect. Default: ``False``.
    memory_format (torch.memory_format): Desired memory format for the result Tensor.
        Default: ``torch.preserve_format``.
"""



# 为 `logcumsumexp` 方法添加了文档字符串，描述了其功能和使用方法
"""
logcumsumexp(dim) -> Tensor

See :func:`torch.logcumsumexp`
"""



# 为 `cummax` 方法添加了文档字符串，描述了其功能和使用方法
"""
cummax(dim) -> (Tensor, Tensor)

See :func:`torch.cummax`
"""



# 为 `cummin` 方法添加了文档字符串，描述了其功能和使用方法
"""
cummin(dim) -> (Tensor, Tensor)

See :func:`torch.cummin`
"""



# 为 `cumprod` 方法添加了文档字符串，详细描述了其参数和功能
"""
cumprod(dim, dtype=None) -> Tensor

See :func:`torch.cumprod`
"""



# 为 `cumprod_` 方法添加了文档字符串，描述了其作用是 `cumprod` 的原地版本
"""
cumprod_(dim, dtype=None) -> Tensor

In-place version of :meth:`~Tensor.cumprod`
"""



# 为 `cumsum` 方法添加了文档字符串，详细描述了其参数和功能
"""
cumsum(dim, dtype=None) -> Tensor

See :func:`torch.cumsum`
"""



# 为 `cumsum_` 方法添加了文档字符串，描述了其作用是 `cumsum` 的原地版本
"""
cumsum_(dim, dtype=None) -> Tensor

In-place version of :meth:`~Tensor.cumsum`
"""
# 为所有函数添加文档字符串
add_docstr_all(
    "data_ptr",
    r"""
    data_ptr() -> int

    返回 :attr:`self` 张量第一个元素的地址。
    """,
)

add_docstr_all(
    "dequantize",
    r"""
    dequantize() -> Tensor

    给定一个量化的张量，对其进行去量化并返回浮点数张量。
    """,
)

add_docstr_all(
    "dense_dim",
    r"""
    dense_dim() -> int

    返回 :attr:`self` 中的稠密维度数量。

    .. note::
      如果 :attr:`self` 不是稀疏张量，则返回 ``len(self.shape)``。

    参见 :meth:`Tensor.sparse_dim` 和 :ref:`混合张量 <sparse-hybrid-coo-docs>`。
    """,
)

add_docstr_all(
    "diag",
    r"""
    diag(diagonal=0) -> Tensor

    参见 :func:`torch.diag`
    """,
)

add_docstr_all(
    "diag_embed",
    r"""
    diag_embed(offset=0, dim1=-2, dim2=-1) -> Tensor

    参见 :func:`torch.diag_embed`
    """,
)

add_docstr_all(
    "diagflat",
    r"""
    diagflat(offset=0) -> Tensor

    参见 :func:`torch.diagflat`
    """,
)

add_docstr_all(
    "diagonal",
    r"""
    diagonal(offset=0, dim1=0, dim2=1) -> Tensor

    参见 :func:`torch.diagonal`
    """,
)

add_docstr_all(
    "diagonal_scatter",
    r"""
    diagonal_scatter(src, offset=0, dim1=0, dim2=1) -> Tensor

    参见 :func:`torch.diagonal_scatter`
    """,
)

add_docstr_all(
    "as_strided_scatter",
    r"""
    as_strided_scatter(src, size, stride, storage_offset=None) -> Tensor

    参见 :func:`torch.as_strided_scatter`
    """,
)

add_docstr_all(
    "fill_diagonal_",
    r"""
    fill_diagonal_(fill_value, wrap=False) -> Tensor

    填充至少有两个维度的张量的主对角线。
    当维度>2时，输入的所有维度必须具有相等长度。
    此函数会就地修改输入张量，并返回输入张量。

    参数:
        fill_value (标量): 填充值
        wrap (bool): 对于高矩阵，是否将对角线“环绕”在 N 列后。

    示例::

        >>> a = torch.zeros(3, 3)
        >>> a.fill_diagonal_(5)
        tensor([[5., 0., 0.],
                [0., 5., 0.],
                [0., 0., 5.]])
        >>> b = torch.zeros(7, 3)
        >>> b.fill_diagonal_(5)
        tensor([[5., 0., 0.],
                [0., 5., 0.],
                [0., 0., 5.],
                [0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.]])
        >>> c = torch.zeros(7, 3)
        >>> c.fill_diagonal_(5, wrap=True)
        tensor([[5., 0., 0.],
                [0., 5., 0.],
                [0., 0., 5.],
                [0., 0., 0.],
                [5., 0., 0.],
                [0., 5., 0.],
                [0., 0., 5.]])
    """,
)

add_docstr_all(
    "floor_divide",
    r"""
    floor_divide(value) -> Tensor

    参见 :func:`torch.floor_divide`
    """,
)

add_docstr_all(
    "floor_divide_",
    r"""
    floor_divide_(value) -> Tensor

    :meth:`~Tensor.floor_divide` 的原地版本。
    """,
)

add_docstr_all(
    "diff",
    r"""
    diff(n=1, dim=-1, prepend=None, append=None) -> Tensor

    参见 :func:`torch.diff`
    """,
)

add_docstr_all(
    "digamma",
    r"""
    digamma() -> Tensor

    参见 :func:`torch.digamma`
    """,
)

add_docstr_all(
    "digamma_",
    r"""
    """
# 返回 Tensor 的 digamma 函数的就地版本
digamma_() -> Tensor

# 为 dim() 方法添加文档字符串，描述返回 self 张量的维度数量
add_docstr_all(
    "dim",
    r"""
    dim() -> int
    
    返回张量 :attr:`self` 的维度数。
    """,
)

# 为 dist() 方法添加文档字符串，指向 torch.dist 函数
add_docstr_all(
    "dist",
    r"""
    dist(other, p=2) -> Tensor
    
    参见 :func:`torch.dist`
    """,
)

# 为 div() 方法添加文档字符串，指向 torch.div 函数
add_docstr_all(
    "div",
    r"""
    div(value, *, rounding_mode=None) -> Tensor
    
    参见 :func:`torch.div`
    """,
)

# 为 div_() 方法添加文档字符串，描述为 Tensor.div 的就地版本
add_docstr_all(
    "div_",
    r"""
    div_(value, *, rounding_mode=None) -> Tensor
    
    :meth:`~Tensor.div` 的就地版本
    """,
)

# 为 divide() 方法添加文档字符串，指向 torch.divide 函数
add_docstr_all(
    "divide",
    r"""
    divide(value, *, rounding_mode=None) -> Tensor
    
    参见 :func:`torch.divide`
    """,
)

# 为 divide_() 方法添加文档字符串，描述为 Tensor.divide 的就地版本
add_docstr_all(
    "divide_",
    r"""
    divide_(value, *, rounding_mode=None) -> Tensor
    
    :meth:`~Tensor.divide` 的就地版本
    """,
)

# 为 dot() 方法添加文档字符串，指向 torch.dot 函数
add_docstr_all(
    "dot",
    r"""
    dot(other) -> Tensor
    
    参见 :func:`torch.dot`
    """,
)

# 为 element_size() 方法添加文档字符串，描述返回每个元素的字节大小
add_docstr_all(
    "element_size",
    r"""
    element_size() -> int
    
    返回单个元素的字节大小。
    
    示例::
    
        >>> torch.tensor([]).element_size()
        4
        >>> torch.tensor([], dtype=torch.uint8).element_size()
        1
    """,
)

# 为 eq() 方法添加文档字符串，指向 torch.eq 函数
add_docstr_all(
    "eq",
    r"""
    eq(other) -> Tensor
    
    参见 :func:`torch.eq`
    """,
)

# 为 eq_() 方法添加文档字符串，描述为 Tensor.eq 的就地版本
add_docstr_all(
    "eq_",
    r"""
    eq_(other) -> Tensor
    
    :meth:`~Tensor.eq` 的就地版本
    """,
)

# 为 equal() 方法添加文档字符串，指向 torch.equal 函数
add_docstr_all(
    "equal",
    r"""
    equal(other) -> bool
    
    参见 :func:`torch.equal`
    """,
)

# 为 erf() 方法添加文档字符串，指向 torch.erf 函数
add_docstr_all(
    "erf",
    r"""
    erf() -> Tensor
    
    参见 :func:`torch.erf`
    """,
)

# 为 erf_() 方法添加文档字符串，描述为 Tensor.erf 的就地版本
add_docstr_all(
    "erf_",
    r"""
    erf_() -> Tensor
    
    :meth:`~Tensor.erf` 的就地版本
    """,
)

# 为 erfc() 方法添加文档字符串，指向 torch.erfc 函数
add_docstr_all(
    "erfc",
    r"""
    erfc() -> Tensor
    
    参见 :func:`torch.erfc`
    """,
)

# 为 erfc_() 方法添加文档字符串，描述为 Tensor.erfc 的就地版本
add_docstr_all(
    "erfc_",
    r"""
    erfc_() -> Tensor
    
    :meth:`~Tensor.erfc` 的就地版本
    """,
)

# 为 erfinv() 方法添加文档字符串，指向 torch.erfinv 函数
add_docstr_all(
    "erfinv",
    r"""
    erfinv() -> Tensor
    
    参见 :func:`torch.erfinv`
    """,
)

# 为 erfinv_() 方法添加文档字符串，描述为 Tensor.erfinv 的就地版本
add_docstr_all(
    "erfinv_",
    r"""
    erfinv_() -> Tensor
    
    :meth:`~Tensor.erfinv` 的就地版本
    """,
)

# 为 exp() 方法添加文档字符串，指向 torch.exp 函数
add_docstr_all(
    "exp",
    r"""
    exp() -> Tensor
    
    参见 :func:`torch.exp`
    """,
)

# 为 exp_() 方法添加文档字符串，描述为 Tensor.exp 的就地版本
add_docstr_all(
    "exp_",
    r"""
    exp_() -> Tensor
    
    :meth:`~Tensor.exp` 的就地版本
    """,
)

# 为 exp2() 方法添加文档字符串，指向 torch.exp2 函数
add_docstr_all(
    "exp2",
    r"""
    exp2() -> Tensor
    
    参见 :func:`torch.exp2`
    """,
)

# 为 exp2_() 方法添加文档字符串，描述为 Tensor.exp2 的就地版本
add_docstr_all(
    "exp2_",
    r"""
    exp2_() -> Tensor
    
    :meth:`~Tensor.exp2` 的就地版本
    """,
)

# 为 expm1() 方法添加文档字符串，指向 torch.expm1 函数
add_docstr_all(
    "expm1",
    r"""
    expm1() -> Tensor
    
    参见 :func:`torch.expm1`
    """,
)

# 为 expm1_() 方法添加文档字符串，描述为 Tensor.expm1 的就地版本
add_docstr_all(
    "expm1_",
    r"""
    expm1_() -> Tensor
    
    :meth:`~Tensor.expm1` 的就地版本
    """,
)

# 为 exponential_() 方法添加文档字符串，描述用指数分布填充张量的内容
add_docstr_all(
    "exponential_",
    r"""
    exponential_(lambd=1, *, generator=None) -> Tensor
    
    使用概率密度函数 (PDF) 填充 :attr:`self` 张量的元素：
    
    .. math::
    
        f(x) = \lambda e^{-\lambda x}, x > 0
    """,
)
# 填充当前张量（即调用此方法的张量）的所有元素为指定的值
fill_(value) -> Tensor


# 返回当前张量的向下取整结果
floor() -> Tensor

See :func:`torch.floor`


# 返回沿指定维度翻转后的张量
flip(dims) -> Tensor

See :func:`torch.flip`


# 返回沿第二维度（即列）翻转后的张量
fliplr() -> Tensor

See :func:`torch.fliplr`


# 返回沿第一维度（即行）翻转后的张量
flipud() -> Tensor

See :func:`torch.flipud`


# 返回沿指定维度进行循环移位后的张量
roll(shifts, dims) -> Tensor

See :func:`torch.roll`


# 将当前张量的所有元素取下界（即向下取整），并返回结果张量
floor_() -> Tensor

In-place version of :meth:`~Tensor.floor`


# 返回当前张量除以指定数值后的余数张量
fmod(divisor) -> Tensor

See :func:`torch.fmod`


# 将当前张量的所有元素取余数，并返回结果张量
fmod_(divisor) -> Tensor

In-place version of :meth:`~Tensor.fmod`


# 返回当前张量的所有元素的小数部分
frac() -> Tensor

See :func:`torch.frac`


# 将当前张量的所有元素取小数部分，并返回结果张量
frac_() -> Tensor

In-place version of :meth:`~Tensor.frac`


# 返回输入张量的尾数（mantissa）和指数（exponent）
frexp(input) -> (Tensor mantissa, Tensor exponent)

See :func:`torch.frexp`


# 返回从指定起始维度到结束维度（不包括）展平后的张量
flatten(start_dim=0, end_dim=-1) -> Tensor

See :func:`torch.flatten`


# 根据指定维度和索引张量，收集当前张量的元素，并返回结果张量
gather(dim, index) -> Tensor

See :func:`torch.gather`


# 返回当前张量和另一张量的最大公约数（gcd）
gcd(other) -> Tensor

See :func:`torch.gcd`


# 将当前张量和另一张量的所有元素计算最大公约数，并返回结果张量
gcd_(other) -> Tensor

In-place version of :meth:`~Tensor.gcd`


# 返回当前张量和另一张量进行逐元素比较，大于等于返回1，否则返回0的结果张量
ge(other) -> Tensor

See :func:`torch.ge`.


# 将当前张量和另一张量逐元素比较，大于等于的元素置1，否则置0，并返回结果张量
ge_(other) -> Tensor

In-place version of :meth:`~Tensor.ge`.


# 返回当前张量和另一张量进行逐元素比较，大于等于返回1，否则返回0的结果张量
greater_equal(other) -> Tensor

See :func:`torch.greater_equal`.


# 将当前张量和另一张量逐元素比较，大于等于的元素置1，否则置0，并返回结果张量
greater_equal_(other) -> Tensor

In-place version of :meth:`~Tensor.greater_equal`.


# 使用几何分布填充当前张量（即调用此方法的张量），参数 p 表示概率参数，generator 表示生成器，默认为 None
geometric_(p, *, generator=None) -> Tensor

Fills :attr:`self` tensor with elements drawn from the geometric distribution:

.. math::

    P(X=k) = (1 - p)^{k - 1} p, k = 1, 2, ...
"""
.. note::
   :func:`torch.Tensor.geometric_` `k`-th trial is the first success hence draws samples in :math:`\{1, 2, \ldots\}`, whereas
   :func:`torch.distributions.geometric.Geometric` :math:`(k+1)`-th trial is the first success
   hence draws samples in :math:`\{0, 1, \ldots\}`.
"""

add_docstr_all(
    "geqrf",
    r"""
    geqrf() -> (Tensor, Tensor)

    See :func:`torch.geqrf`
    """
)

add_docstr_all(
    "ger",
    r"""
    ger(vec2) -> Tensor

    See :func:`torch.ger`
    """
)

add_docstr_all(
    "inner",
    r"""
    inner(other) -> Tensor

    See :func:`torch.inner`.
    """
)

add_docstr_all(
    "outer",
    r"""
    outer(vec2) -> Tensor

    See :func:`torch.outer`.
    """
)

add_docstr_all(
    "hypot",
    r"""
    hypot(other) -> Tensor

    See :func:`torch.hypot`
    """
)

add_docstr_all(
    "hypot_",
    r"""
    hypot_(other) -> Tensor

    In-place version of :meth:`~Tensor.hypot`
    """
)

add_docstr_all(
    "i0",
    r"""
    i0() -> Tensor

    See :func:`torch.i0`
    """
)

add_docstr_all(
    "i0_",
    r"""
    i0_() -> Tensor

    In-place version of :meth:`~Tensor.i0`
    """
)

add_docstr_all(
    "igamma",
    r"""
    igamma(other) -> Tensor

    See :func:`torch.igamma`
    """
)

add_docstr_all(
    "igamma_",
    r"""
    igamma_(other) -> Tensor

    In-place version of :meth:`~Tensor.igamma`
    """
)

add_docstr_all(
    "igammac",
    r"""
    igammac(other) -> Tensor

    See :func:`torch.igammac`
    """
)

add_docstr_all(
    "igammac_",
    r"""
    igammac_(other) -> Tensor

    In-place version of :meth:`~Tensor.igammac`
    """
)

add_docstr_all(
    "indices",
    r"""
    indices() -> Tensor

    Return the indices tensor of a :ref:`sparse COO tensor <sparse-coo-docs>`.

    .. warning::
       Throws an error if :attr:`self` is not a sparse COO tensor.

    See also :meth:`Tensor.values`.

    .. note::
       This method can only be called on a coalesced sparse tensor. See
       :meth:`Tensor.coalesce` for details.
    """
)

add_docstr_all(
    "get_device",
    r"""
    get_device() -> Device ordinal (Integer)

    For CUDA tensors, this function returns the device ordinal of the GPU on which the tensor resides.
    For CPU tensors, this function returns `-1`.

    Example::

        >>> x = torch.randn(3, 4, 5, device='cuda:0')
        >>> x.get_device()
        0
        >>> x.cpu().get_device()
        -1
    """
)

add_docstr_all(
    "values",
    r"""
    values() -> Tensor

    Return the values tensor of a :ref:`sparse COO tensor <sparse-coo-docs>`.

    .. warning::
       Throws an error if :attr:`self` is not a sparse COO tensor.

    See also :meth:`Tensor.indices`.

    .. note::
       This method can only be called on a coalesced sparse tensor. See
       :meth:`Tensor.coalesce` for details.
    """
)

add_docstr_all(
    "gt",
    r"""
    gt(other) -> Tensor

    See :func:`torch.gt`.
    """
)

add_docstr_all(
    "gt_",
    r"""
    gt_(other) -> Tensor

    In-place version of :meth:`~Tensor.gt`.
    """
)

add_docstr_all(
    "greater",
    r"""
    greater(other) -> Tensor

    See :func:`torch.greater`.
    """
)

add_docstr_all(
    "greater_",
    r"""
    greater_(other) -> Tensor

    In-place version of :meth:`~Tensor.greater`.
    """
)
# 给指定函数添加文档字符串，描述其功能和用法
add_docstr_all(
    "has_names",
    r"""
    Is ``True`` if any of this tensor's dimensions are named. Otherwise, is ``False``.
    """,
)

# 给指定函数添加文档字符串，描述其功能和用法，引用了 torch.nn.functional.hardshrink 函数
add_docstr_all(
    "hardshrink",
    r"""
    hardshrink(lambd=0.5) -> Tensor

    See :func:`torch.nn.functional.hardshrink`
    """,
)

# 给指定函数添加文档字符串，描述其功能和用法，引用了 torch.heaviside 函数
add_docstr_all(
    "heaviside",
    r"""
    heaviside(values) -> Tensor

    See :func:`torch.heaviside`
    """,
)

# 给指定函数添加文档字符串，描述其功能和用法，是 Tensor 类的 inplace 方法的文档字符串
add_docstr_all(
    "heaviside_",
    r"""
    heaviside_(values) -> Tensor

    In-place version of :meth:`~Tensor.heaviside`
    """,
)

# 给指定函数添加文档字符串，描述其功能和用法，引用了 torch.histc 函数
add_docstr_all(
    "histc",
    r"""
    histc(bins=100, min=0, max=0) -> Tensor

    See :func:`torch.histc`
    """,
)

# 给指定函数添加文档字符串，描述其功能和用法，引用了 torch.histogram 函数
add_docstr_all(
    "histogram",
    r"""
    histogram(input, bins, *, range=None, weight=None, density=False) -> (Tensor, Tensor)

    See :func:`torch.histogram`
    """,
)

# 给指定函数添加文档字符串，描述其功能和用法，是 Tensor 类的 inplace 方法的文档字符串，包含了详细的用例和注意事项
add_docstr_all(
    "index_add_",
    r"""
    index_add_(dim, index, source, *, alpha=1) -> Tensor

    Accumulate the elements of :attr:`alpha` times ``source`` into the :attr:`self`
    tensor by adding to the indices in the order given in :attr:`index`. For example,
    if ``dim == 0``, ``index[i] == j``, and ``alpha=-1``, then the ``i``\ th row of
    ``source`` is subtracted from the ``j``\ th row of :attr:`self`.

    The :attr:`dim`\ th dimension of ``source`` must have the same size as the
    length of :attr:`index` (which must be a vector), and all other dimensions must
    match :attr:`self`, or an error will be raised.

    For a 3-D tensor the output is given as::

        self[index[i], :, :] += alpha * src[i, :, :]  # if dim == 0
        self[:, index[i], :] += alpha * src[:, i, :]  # if dim == 1
        self[:, :, index[i]] += alpha * src[:, :, i]  # if dim == 2

    Note:
        This operation might not be forward-reproducible. See notes on :ref:`reproducibility <reproducibility-notes>`.

    Args:
        dim (int): dimension along which to index
        index (Tensor): indices of ``source`` to select from,
                should have dtype either `torch.int64` or `torch.int32`
        source (Tensor): the tensor containing values to add

    Keyword args:
        alpha (Number): the scalar multiplier for ``source``

    Example::

        >>> x = torch.ones(5, 3)
        >>> t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
        >>> index = torch.tensor([0, 4, 2])
        >>> x.index_add_(0, index, t)
        tensor([[  2.,   3.,   4.],
                [  1.,   1.,   1.],
                [  8.,   9.,  10.],
                [  1.,   1.,   1.],
                [  5.,   6.,   7.]])
        >>> x.index_add_(0, index, t, alpha=-1)
        tensor([[  1.,   1.,   1.],
                [  1.,   1.,   1.],
                [  1.,   1.,   1.],
                [  1.,   1.,   1.],
                [  1.,   1.,   1.]])
    """,
)

# 给指定函数添加文档字符串，描述其功能和用法，是 Tensor 类的 inplace 方法的文档字符串
add_docstr_all(
    "index_copy_",
    r"""
    index_copy_(dim, index, tensor) -> Tensor

    Copies the elements of :attr:`tensor` into the :attr:`self` tensor by selecting
    the indices in the order given in :attr:`index`. For example, if ``dim == 0``
    and ``index[i] == j``, then the ``i``\ th row of :attr:`tensor` is copied to the
    ``j``\ th row of :attr:`self`.
    """,
)
# 基于给定索引从源张量中选择元素，并根据给定规则将其累加到调用此方法的张量中。
# 其中累加的规则由 `reduce` 参数确定，可以是 sum、prod 等。
# 注意：如果 `dim == 0`，`index[i] == j`，`reduce == prod` 并且 `include_self == True`，
# 则第 `i` 行的结果会将 `source[j]` 中的元素相乘后累加到调用方法的张量的第 `i` 行。

def index_reduce_(dim, index, source, reduce, *, include_self=True) -> Tensor:
    """
    Accumulate the elements of `source` into the :attr:`self`
    tensor by accumulating to the indices in the order given in :attr:`index`
    using the reduction given by the ``reduce`` argument. For example, if ``dim == 0``,
    ``index[i] == j``, ``reduce == prod`` and ``include_self == True`` then the ``i``\ th
    """
add_docstr_all(
    "index_select",
    r"""
index_select(dim, index) -> Tensor

See :func:`torch.index_select`
""",
)



add_docstr_all(
    "sparse_mask",
    r"""
sparse_mask(mask) -> Tensor

Returns a new :ref:`sparse tensor <sparse-docs>` with values from a
strided tensor :attr:`self` filtered by the indices of the sparse
tensor :attr:`mask`. The values of :attr:`mask` sparse tensor are
ignored. :attr:`self` and :attr:`mask` tensors must have the same
shape.

.. note::

  The returned sparse tensor might contain duplicate values if :attr:`mask`
  is not coalesced. It is therefore advisable to pass ``mask.coalesce()``
  if such behavior is not desired.

.. note::

  The returned sparse tensor has the same indices as the sparse tensor
  :attr:`mask`, even when the corresponding values in :attr:`self` are
  zeros.

Args:
    mask (SparseTensor): The sparse tensor indicating indices to be included

Returns:
    Tensor: A new sparse tensor with values from :attr:`self` at indices specified by :attr:`mask`
""",
)
    mask (Tensor): a sparse tensor whose indices are used as a filter
# Example::

>>> nse = 5
>>> dims = (5, 5, 2, 2)
>>> I = torch.cat([torch.randint(0, dims[0], size=(nse,)),
...                torch.randint(0, dims[1], size=(nse,))], 0).reshape(2, nse)
>>> V = torch.randn(nse, dims[2], dims[3])
>>> S = torch.sparse_coo_tensor(I, V, dims).coalesce()
>>> D = torch.randn(dims)
>>> D.sparse_mask(S)
tensor(indices=tensor([[0, 0, 0, 2],
                       [0, 1, 4, 3]]),
       values=tensor([[[ 1.6550,  0.2397],
                       [-0.1611, -0.0779]],

                      [[ 0.2326, -1.0558],
                       [ 1.4711,  1.9678]],

                      [[-0.5138, -0.0411],
                       [ 1.9417,  0.5158]],

                      [[ 0.0793,  0.0036],
                       [-0.2569, -0.1055]]]),
       size=(5, 5, 2, 2), nnz=4, layout=torch.sparse_coo)


add_docstr_all(
    "inverse",
    r"""
    inverse() -> Tensor
    
    See :func:`torch.inverse`
    """,
)

add_docstr_all(
    "isnan",
    r"""
    isnan() -> Tensor
    
    See :func:`torch.isnan`
    """,
)

add_docstr_all(
    "isinf",
    r"""
    isinf() -> Tensor
    
    See :func:`torch.isinf`
    """,
)

add_docstr_all(
    "isposinf",
    r"""
    isposinf() -> Tensor
    
    See :func:`torch.isposinf`
    """,
)

add_docstr_all(
    "isneginf",
    r"""
    isneginf() -> Tensor
    
    See :func:`torch.isneginf`
    """,
)

add_docstr_all(
    "isfinite",
    r"""
    isfinite() -> Tensor
    
    See :func:`torch.isfinite`
    """,
)

add_docstr_all(
    "isclose",
    r"""
    isclose(other, rtol=1e-05, atol=1e-08, equal_nan=False) -> Tensor
    
    See :func:`torch.isclose`
    """,
)

add_docstr_all(
    "isreal",
    r"""
    isreal() -> Tensor
    
    See :func:`torch.isreal`
    """,
)

add_docstr_all(
    "is_coalesced",
    r"""
    is_coalesced() -> bool
    
    Returns ``True`` if :attr:`self` is a :ref:`sparse COO tensor
    <sparse-coo-docs>` that is coalesced, ``False`` otherwise.
    
    .. warning::
      Throws an error if :attr:`self` is not a sparse COO tensor.
    
    See :meth:`coalesce` and :ref:`uncoalesced tensors <sparse-uncoalesced-coo-docs>`.
    """,
)

add_docstr_all(
    "is_contiguous",
    r"""
    is_contiguous(memory_format=torch.contiguous_format) -> bool
    
    Returns True if :attr:`self` tensor is contiguous in memory in the order specified
    by memory format.
    
    Args:
        memory_format (:class:`torch.memory_format`, optional): Specifies memory allocation
            order. Default: ``torch.contiguous_format``.
    """,
)

add_docstr_all(
    "is_pinned",
    r"""
    Returns true if this tensor resides in pinned memory.
    """,
)

add_docstr_all(
    "is_floating_point",
    r"""
    is_floating_point() -> bool
    
    Returns True if the data type of :attr:`self` is a floating point data type.
    """,
)

add_docstr_all(
    "is_complex",
    r"""
    is_complex() -> bool
    
    Returns True if the data type of :attr:`self` is a complex data type.
    """,
)

add_docstr_all(
    "is_inference",
    r"""
    is_inference() -> bool
    
    See :func:`torch.is_inference`
    """,
)

add_docstr_all(
    "is_conj",
    r"""
    is_conj
    """,
)
# 返回是否设置了 :attr:`self` 的共轭位为真。
is_conj() -> bool
"""

# 返回是否设置了 :attr:`self` 的负号位为真。
add_docstr_all(
    "is_neg",
    r"""
is_neg() -> bool

Returns True if the negative bit of :attr:`self` is set to true.
""",
)

# 返回是否 :attr:`self` 的数据类型是有符号数据类型。
add_docstr_all(
    "is_signed",
    r"""
is_signed() -> bool

Returns True if the data type of :attr:`self` is a signed data type.
""",
)

# 返回是否两个张量指向完全相同的内存（相同的存储、偏移、大小和步幅）。
add_docstr_all(
    "is_set_to",
    r"""
is_set_to(tensor) -> bool

Returns True if both tensors are pointing to the exact same memory (same
storage, offset, size and stride).
""",
)

# 返回此张量作为标准 Python 数字的值。仅适用于只包含一个元素的张量。对于其他情况，参见 :meth:`~Tensor.tolist`。
# 此操作不可微分。
add_docstr_all(
    "item",
    r"""
item() -> number

Returns the value of this tensor as a standard Python number. This only works
for tensors with one element. For other cases, see :meth:`~Tensor.tolist`.

This operation is not differentiable.

Example::

    >>> x = torch.tensor([1.0])
    >>> x.item()
    1.0
""",
)

# 参见 :func:`torch.kron`
add_docstr_all(
    "kron",
    r"""
kron(other) -> Tensor

See :func:`torch.kron`
""",
)

# 参见 :func:`torch.kthvalue`
add_docstr_all(
    "kthvalue",
    r"""
kthvalue(k, dim=None, keepdim=False) -> (Tensor, LongTensor)

See :func:`torch.kthvalue`
""",
)

# 参见 :func:`torch.ldexp`
add_docstr_all(
    "ldexp",
    r"""
ldexp(other) -> Tensor

See :func:`torch.ldexp`
""",
)

# :meth:`~Tensor.ldexp` 的原地版本。
add_docstr_all(
    "ldexp_",
    r"""
ldexp_(other) -> Tensor

In-place version of :meth:`~Tensor.ldexp`
""",
)

# 参见 :func:`torch.lcm`
add_docstr_all(
    "lcm",
    r"""
lcm(other) -> Tensor

See :func:`torch.lcm`
""",
)

# :meth:`~Tensor.lcm` 的原地版本。
add_docstr_all(
    "lcm_",
    r"""
lcm_(other) -> Tensor

In-place version of :meth:`~Tensor.lcm`
""",
)

# 参见 :func:`torch.le`
add_docstr_all(
    "le",
    r"""
le(other) -> Tensor

See :func:`torch.le`.
""",
)

# :meth:`~Tensor.le` 的原地版本。
add_docstr_all(
    "le_",
    r"""
le_(other) -> Tensor

In-place version of :meth:`~Tensor.le`.
""",
)

# 参见 :func:`torch.less_equal`
add_docstr_all(
    "less_equal",
    r"""
less_equal(other) -> Tensor

See :func:`torch.less_equal`.
""",
)

# :meth:`~Tensor.less_equal` 的原地版本。
add_docstr_all(
    "less_equal_",
    r"""
less_equal_(other) -> Tensor

In-place version of :meth:`~Tensor.less_equal`.
""",
)

# 参见 :func:`torch.lerp`
add_docstr_all(
    "lerp",
    r"""
lerp(end, weight) -> Tensor

See :func:`torch.lerp`
""",
)

# :meth:`~Tensor.lerp` 的原地版本。
add_docstr_all(
    "lerp_",
    r"""
lerp_(end, weight) -> Tensor

In-place version of :meth:`~Tensor.lerp`
""",
)

# 参见 :func:`torch.lgamma`
add_docstr_all(
    "lgamma",
    r"""
lgamma() -> Tensor

See :func:`torch.lgamma`
""",
)

# :meth:`~Tensor.lgamma` 的原地版本。
add_docstr_all(
    "lgamma_",
    r"""
lgamma_() -> Tensor

In-place version of :meth:`~Tensor.lgamma`
""",
)

# 参见 :func:`torch.log`
add_docstr_all(
    "log",
    r"""
log() -> Tensor

See :func:`torch.log`
""",
)

# :meth:`~Tensor.log` 的原地版本。
add_docstr_all(
    "log_",
    r"""
log_() -> Tensor

In-place version of :meth:`~Tensor.log`
""",
)

# 参见 :func:`torch.log10`
add_docstr_all(
    "log10",
    r"""
log10() -> Tensor

See :func:`torch.log10`
""",
)

# :meth:`~Tensor.log10` 的原地版本。
add_docstr_all(
    "log10_",
    r"""
log10_() -> Tensor

In-place version of :meth:`~Tensor.log10`
""",
)

# 参见 :func:`torch.log1p`
add_docstr_all(
    "log1p",
    r"""
log1p() -> Tensor

See :func:`torch.log1p`
""",
)

# :meth:`~Tensor.log1p` 的原地版本。
add_docstr_all(
    "log1p_",
    r"""
log1p_() -> Tensor

In-place version of :meth:`~Tensor.log1p`
""",
)

# 参见 :func:`torch.log2`
add_docstr_all(
    "log2",
    r"""
log2() -> Tensor
""",
)

# 参见 :func:`torch.log2`
# 为函数 torch.log2 添加文档字符串
add_docstr_all(
    "log2_",
    r"""
    log2_() -> Tensor

    In-place version of :meth:`~Tensor.log2`
    """,
)

# 为函数 torch.logaddexp 添加文档字符串
add_docstr_all(
    "logaddexp",
    r"""
    logaddexp(other) -> Tensor

    See :func:`torch.logaddexp`
    """,
)

# 为函数 torch.logaddexp2 添加文档字符串
add_docstr_all(
    "logaddexp2",
    r"""
    logaddexp2(other) -> Tensor

    See :func:`torch.logaddexp2`
    """,
)

# 为函数 torch.log_normal_ 添加文档字符串
add_docstr_all(
    "log_normal_",
    r"""
    log_normal_(mean=1, std=2, *, generator=None)

    Fills :attr:`self` tensor with numbers samples from the log-normal distribution
    parameterized by the given mean :math:`\mu` and standard deviation
    :math:`\sigma`. Note that :attr:`mean` and :attr:`std` are the mean and
    standard deviation of the underlying normal distribution, and not of the
    returned distribution:

    .. math::

        f(x) = \dfrac{1}{x \sigma \sqrt{2\pi}}\ e^{-\frac{(\ln x - \mu)^2}{2\sigma^2}}
    """,
)

# 为函数 torch.logsumexp 添加文档字符串
add_docstr_all(
    "logsumexp",
    r"""
    logsumexp(dim, keepdim=False) -> Tensor

    See :func:`torch.logsumexp`
    """,
)

# 为函数 torch.lt 添加文档字符串
add_docstr_all(
    "lt",
    r"""
    lt(other) -> Tensor

    See :func:`torch.lt`.
    """,
)

# 为函数 torch.lt_ 添加文档字符串
add_docstr_all(
    "lt_",
    r"""
    lt_(other) -> Tensor

    In-place version of :meth:`~Tensor.lt`.
    """,
)

# 为函数 torch.less 添加文档字符串
add_docstr_all(
    "less",
    r"""
    lt(other) -> Tensor

    See :func:`torch.less`.
    """,
)

# 为函数 torch.less_ 添加文档字符串
add_docstr_all(
    "less_",
    r"""
    less_(other) -> Tensor

    In-place version of :meth:`~Tensor.less`.
    """,
)

# 为函数 torch.lu_solve 添加文档字符串
add_docstr_all(
    "lu_solve",
    r"""
    lu_solve(LU_data, LU_pivots) -> Tensor

    See :func:`torch.lu_solve`
    """,
)

# 为函数 torch.map_ 添加文档字符串
add_docstr_all(
    "map_",
    r"""
    map_(tensor, callable)

    Applies :attr:`callable` for each element in :attr:`self` tensor and the given
    :attr:`tensor` and stores the results in :attr:`self` tensor. :attr:`self` tensor and
    the given :attr:`tensor` must be :ref:`broadcastable <broadcasting-semantics>`.

    The :attr:`callable` should have the signature::

        def callable(a, b) -> number
    """,
)

# 为函数 torch.masked_scatter_ 添加文档字符串
add_docstr_all(
    "masked_scatter_",
    r"""
    masked_scatter_(mask, source)

    Copies elements from :attr:`source` into :attr:`self` tensor at positions where
    the :attr:`mask` is True. Elements from :attr:`source` are copied into :attr:`self`
    starting at position 0 of :attr:`source` and continuing in order one-by-one for each
    occurrence of :attr:`mask` being True.
    The shape of :attr:`mask` must be :ref:`broadcastable <broadcasting-semantics>`
    with the shape of the underlying tensor. The :attr:`source` should have at least
    as many elements as the number of ones in :attr:`mask`.

    Args:
        mask (BoolTensor): the boolean mask
        source (Tensor): the tensor to copy from

    .. note::

        The :attr:`mask` operates on the :attr:`self` tensor, not on the given
        :attr:`source` tensor.

    Example:

        >>> self = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
        >>> mask = torch.tensor([[0, 0, 0, 1, 1], [1, 1, 0, 1, 1]], dtype=torch.bool)
        >>> source = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        >>> self.masked_scatter_(mask, source)
    """,
)
    # 创建一个包含两个行的张量（tensor），每行包含五个元素
    tensor([[0, 0, 0, 0, 1],
            [2, 3, 0, 4, 5]])
"""

add_docstr_all(
    "masked_fill_",
    r"""
masked_fill_(mask, value)

在条件mask为True时，将张量self的元素填充为value。mask的形状必须与底层张量的形状
符合广播语义。

参数:
    mask (BoolTensor): 布尔掩码
    value (float): 填充的值
""",
)

add_docstr_all(
    "masked_select",
    r"""
masked_select(mask) -> Tensor

参见 :func:`torch.masked_select`
""",
)

add_docstr_all(
    "matrix_power",
    r"""
matrix_power(n) -> Tensor

.. note:: :meth:`~Tensor.matrix_power` 已弃用，请使用 :func:`torch.linalg.matrix_power`

别名 :func:`torch.linalg.matrix_power`
""",
)

add_docstr_all(
    "matrix_exp",
    r"""
matrix_exp() -> Tensor

参见 :func:`torch.matrix_exp`
""",
)

add_docstr_all(
    "max",
    r"""
max(dim=None, keepdim=False) -> Tensor or (Tensor, Tensor)

参见 :func:`torch.max`
""",
)

add_docstr_all(
    "amax",
    r"""
amax(dim=None, keepdim=False) -> Tensor

参见 :func:`torch.amax`
""",
)

add_docstr_all(
    "maximum",
    r"""
maximum(other) -> Tensor

参见 :func:`torch.maximum`
""",
)

add_docstr_all(
    "fmax",
    r"""
fmax(other) -> Tensor

参见 :func:`torch.fmax`
""",
)

add_docstr_all(
    "argmax",
    r"""
argmax(dim=None, keepdim=False) -> LongTensor

参见 :func:`torch.argmax`
""",
)

add_docstr_all(
    "argwhere",
    r"""
argwhere() -> Tensor

参见 :func:`torch.argwhere`
""",
)

add_docstr_all(
    "mean",
    r"""
mean(dim=None, keepdim=False, *, dtype=None) -> Tensor

参见 :func:`torch.mean`
""",
)

add_docstr_all(
    "nanmean",
    r"""
nanmean(dim=None, keepdim=False, *, dtype=None) -> Tensor

参见 :func:`torch.nanmean`
""",
)

add_docstr_all(
    "median",
    r"""
median(dim=None, keepdim=False) -> (Tensor, LongTensor)

参见 :func:`torch.median`
""",
)

add_docstr_all(
    "nanmedian",
    r"""
nanmedian(dim=None, keepdim=False) -> (Tensor, LongTensor)

参见 :func:`torch.nanmedian`
""",
)

add_docstr_all(
    "min",
    r"""
min(dim=None, keepdim=False) -> Tensor or (Tensor, Tensor)

参见 :func:`torch.min`
""",
)

add_docstr_all(
    "amin",
    r"""
amin(dim=None, keepdim=False) -> Tensor

参见 :func:`torch.amin`
""",
)

add_docstr_all(
    "minimum",
    r"""
minimum(other) -> Tensor

参见 :func:`torch.minimum`
""",
)

add_docstr_all(
    "aminmax",
    r"""
aminmax(*, dim=None, keepdim=False) -> (Tensor min, Tensor max)

参见 :func:`torch.aminmax`
""",
)

add_docstr_all(
    "fmin",
    r"""
fmin(other) -> Tensor

参见 :func:`torch.fmin`
""",
)

add_docstr_all(
    "argmin",
    r"""
argmin(dim=None, keepdim=False) -> LongTensor

参见 :func:`torch.argmin`
""",
)

add_docstr_all(
    "mm",
    r"""
mm(mat2) -> Tensor

参见 :func:`torch.mm`
""",
)

add_docstr_all(
    "mode",
    r"""
mode(dim=None, keepdim=False) -> (Tensor, LongTensor)

参见 :func:`torch.mode`
""",
)

add_docstr_all(
    "movedim",
    r"""
movedim(source, destination) -> Tensor

参见 :func:`torch.movedim`
""",
)
add_docstr_all(
    "moveaxis",
    r"""
    moveaxis(source, destination) -> Tensor

    See :func:`torch.moveaxis`
    """,
)

add_docstr_all(
    "mul",
    r"""
    mul(value) -> Tensor

    See :func:`torch.mul`.
    """,
)

add_docstr_all(
    "mul_",
    r"""
    mul_(value) -> Tensor

    In-place version of :meth:`~Tensor.mul`.
    """,
)

add_docstr_all(
    "multiply",
    r"""
    multiply(value) -> Tensor

    See :func:`torch.multiply`.
    """,
)

add_docstr_all(
    "multiply_",
    r"""
    multiply_(value) -> Tensor

    In-place version of :meth:`~Tensor.multiply`.
    """,
)

add_docstr_all(
    "multinomial",
    r"""
    multinomial(num_samples, replacement=False, *, generator=None) -> Tensor

    See :func:`torch.multinomial`
    """,
)

add_docstr_all(
    "mv",
    r"""
    mv(vec) -> Tensor

    See :func:`torch.mv`
    """,
)

add_docstr_all(
    "mvlgamma",
    r"""
    mvlgamma(p) -> Tensor

    See :func:`torch.mvlgamma`
    """,
)

add_docstr_all(
    "mvlgamma_",
    r"""
    mvlgamma_(p) -> Tensor

    In-place version of :meth:`~Tensor.mvlgamma`
    """,
)

add_docstr_all(
    "narrow",
    r"""
    narrow(dimension, start, length) -> Tensor

    See :func:`torch.narrow`.
    """,
)

add_docstr_all(
    "narrow_copy",
    r"""
    narrow_copy(dimension, start, length) -> Tensor

    See :func:`torch.narrow_copy`.
    """,
)

add_docstr_all(
    "ndimension",
    r"""
    ndimension() -> int

    Alias for :meth:`~Tensor.dim()`
    """,
)

add_docstr_all(
    "nan_to_num",
    r"""
    nan_to_num(nan=0.0, posinf=None, neginf=None) -> Tensor

    See :func:`torch.nan_to_num`.
    """,
)

add_docstr_all(
    "nan_to_num_",
    r"""
    nan_to_num_(nan=0.0, posinf=None, neginf=None) -> Tensor

    In-place version of :meth:`~Tensor.nan_to_num`.
    """,
)

add_docstr_all(
    "ne",
    r"""
    ne(other) -> Tensor

    See :func:`torch.ne`.
    """,
)

add_docstr_all(
    "ne_",
    r"""
    ne_(other) -> Tensor

    In-place version of :meth:`~Tensor.ne`.
    """,
)

add_docstr_all(
    "not_equal",
    r"""
    not_equal(other) -> Tensor

    See :func:`torch.not_equal`.
    """,
)

add_docstr_all(
    "not_equal_",
    r"""
    not_equal_(other) -> Tensor

    In-place version of :meth:`~Tensor.not_equal`.
    """,
)

add_docstr_all(
    "neg",
    r"""
    neg() -> Tensor

    See :func:`torch.neg`
    """,
)

add_docstr_all(
    "negative",
    r"""
    negative() -> Tensor

    See :func:`torch.negative`
    """,
)

add_docstr_all(
    "neg_",
    r"""
    neg_() -> Tensor

    In-place version of :meth:`~Tensor.neg`
    """,
)

add_docstr_all(
    "negative_",
    r"""
    negative_() -> Tensor

    In-place version of :meth:`~Tensor.negative`
    """,
)

add_docstr_all(
    "nelement",
    r"""
    nelement() -> int

    Alias for :meth:`~Tensor.numel`
    """,
)

add_docstr_all(
    "nextafter",
    r"""
    nextafter(other) -> Tensor
    See :func:`torch.nextafter`
    """,
)

add_docstr_all(
    "nextafter_",
    r"""
    nextafter_(other) -> Tensor
    In-place version of :meth:`~Tensor.nextafter`
    """,
)

add_docstr_all(
    "nonzero",
    r"""
    nonzero() -> LongTensor

    See :func:`torch.nonzero`
    """,
)

add_docstr_all(
    "nonzero_static",
    r"""

    """,
)
# 返回一个二维张量，其中每行是非零值的索引。
# 返回的张量与 torch.nonzero() 相同的 torch.dtype。

def nonzero_static(input, *, size, fill_value=-1) -> Tensor:
    """
    Args:
        input (Tensor): 输入张量，用于计算非零元素。
    
    Keyword args:
        size (int): 预期包含在输出张量中的非零元素的大小。
            如果 `size` 大于非零元素的总数，则用 `fill_value` 填充输出张量。
            如果 `size` 小于等于非零元素的总数，则截断输出张量。
            `size` 必须是非负整数。
        fill_value (int): 当 `size` 大于非零元素的总数时，填充输出张量的值。
            默认值为 `-1`，表示无效索引。

    Example:

        # 示例 1：填充
        >>> input_tensor = torch.tensor([[1, 0], [3, 2]])
        >>> static_size = 4
        >>> t = torch.nonzero_static(input_tensor, size=static_size)
        tensor([[  0,   0],
                [  1,   0],
                [  1,   1],
                [  -1, -1]], dtype=torch.int64)

        # 示例 2：截断
        >>> input_tensor = torch.tensor([[1, 0], [3, 2]])
        >>> static_size = 2
        >>> t = torch.nonzero_static(input_tensor, size=static_size)
        tensor([[  0,   0],
                [  1,   0]], dtype=torch.int64)

        # 示例 3：大小为 0
        >>> input_tensor = torch.tensor([10])
        >>> static_size = 0
        >>> t = torch.nonzero_static(input_tensor, size=static_size)
        tensor([], size=(0, 1), dtype=torch.int64)

        # 示例 4：零维输入
        >>> input_tensor = torch.tensor(10)
        >>> static_size = 2
        >>> t = torch.nonzero_static(input_tensor, size=static_size)
        tensor([], size=(2, 0), dtype=torch.int64)
    """
    pass

# 参考 torch.norm
def norm(p=2, dim=None, keepdim=False) -> Tensor:
    """
    Args:
        p (int, float, inf, -inf, 'fro', 'nuc', optional): 范数的计算方式，默认为 2。
        dim (int or tuple of ints, optional): 沿指定维度计算，默认为整个张量。
        keepdim (bool, optional): 是否保持输出张量的维度与输入张量一致，默认为 False。

    See :func:`torch.norm`
    """
    pass

# 参考 torch.normal_
def normal_(mean=0, std=1, *, generator=None) -> Tensor:
    """
    Args:
        mean (float): 正态分布的均值，默认为 0。
        std (float): 正态分布的标准差，默认为 1。
        generator (torch.Generator, optional): 用于生成随机数的生成器。

    Fills :attr:`self` tensor with elements samples from the normal distribution
    parameterized by :attr:`mean` and :attr:`std`.
    """
    pass

# 参考 torch.numel
def numel() -> int:
    """
    Returns the total number of elements in the input tensor.

    See :func:`torch.numel`
    """
    pass

# 参考 torch.numpy
def numpy(*, force=False) -> numpy.ndarray:
    """
    Args:
        force (bool, optional): 如果为 True，则执行转换，即使张量不满足默认转换条件。
            默认为 False。

    Returns:
        numpy.ndarray: 返回张量的 NumPy 数组表示。

    Returns the tensor as a NumPy :class:`ndarray`.
    """
    pass
    force (bool): if ``True``, the ndarray may be a copy of the tensor
                   instead of always sharing memory, defaults to ``False``.
add_docstr_all(
    "orgqr",
    r"""
orgqr(input2) -> Tensor

See :func:`torch.orgqr`
""",
)

add_docstr_all(
    "ormqr",
    r"""
ormqr(input2, input3, left=True, transpose=False) -> Tensor

See :func:`torch.ormqr`
""",
)

add_docstr_all(
    "permute",
    r"""
permute(*dims) -> Tensor

See :func:`torch.permute`
""",
)

add_docstr_all(
    "polygamma",
    r"""
polygamma(n) -> Tensor

See :func:`torch.polygamma`
""",
)

add_docstr_all(
    "polygamma_",
    r"""
polygamma_(n) -> Tensor

In-place version of :meth:`~Tensor.polygamma`
""",
)

add_docstr_all(
    "positive",
    r"""
positive() -> Tensor

See :func:`torch.positive`
""",
)

add_docstr_all(
    "pow",
    r"""
pow(exponent) -> Tensor

See :func:`torch.pow`
""",
)

add_docstr_all(
    "pow_",
    r"""
pow_(exponent) -> Tensor

In-place version of :meth:`~Tensor.pow`
""",
)

add_docstr_all(
    "float_power",
    r"""
float_power(exponent) -> Tensor

See :func:`torch.float_power`
""",
)

add_docstr_all(
    "float_power_",
    r"""
float_power_(exponent) -> Tensor

In-place version of :meth:`~Tensor.float_power`
""",
)

add_docstr_all(
    "prod",
    r"""
prod(dim=None, keepdim=False, dtype=None) -> Tensor

See :func:`torch.prod`
""",
)

add_docstr_all(
    "put_",
    r"""
put_(index, source, accumulate=False) -> Tensor

Copies the elements from :attr:`source` into the positions specified by
:attr:`index`. For the purpose of indexing, the :attr:`self` tensor is treated as if
it were a 1-D tensor.

:attr:`index` and :attr:`source` need to have the same number of elements, but not necessarily
the same shape.

If :attr:`accumulate` is ``True``, the elements in :attr:`source` are added to
:attr:`self`. If accumulate is ``False``, the behavior is undefined if :attr:`index`
contain duplicate elements.

Args:
    index (LongTensor): the indices into self
    source (Tensor): the tensor containing values to copy from
    accumulate (bool): whether to accumulate into self

Example::

    >>> src = torch.tensor([[4, 3, 5],
    ...                     [6, 7, 8]])
    >>> src.put_(torch.tensor([1, 3]), torch.tensor([9, 10]))
    tensor([[  4,   9,   5],
            [ 10,   7,   8]])
""",
)

add_docstr_all(
    "put",
    r"""
put(input, index, source, accumulate=False) -> Tensor

Out-of-place version of :meth:`torch.Tensor.put_`.
`input` corresponds to `self` in :meth:`torch.Tensor.put_`.
""",
)

add_docstr_all(
    "qr",
    r"""
qr(some=True) -> (Tensor, Tensor)

See :func:`torch.qr`
""",
)

add_docstr_all(
    "qscheme",
    r"""
qscheme() -> torch.qscheme

Returns the quantization scheme of a given QTensor.
""",
)

add_docstr_all(
    "quantile",
    r"""
quantile(q, dim=None, keepdim=False, *, interpolation='linear') -> Tensor

See :func:`torch.quantile`
""",
)

add_docstr_all(
    "nanquantile",
    r"""
nanquantile(q, dim=None, keepdim=False, *, interpolation='linear') -> Tensor

See :func:`torch.nanquantile`
""",
)

add_docstr_all(
    "q_scale",
    r"""
q_scale() -> float

Returns the scale factor of a given QTensor.
""",
)
# Given a Tensor quantized by linear(affine) quantization,
# returns the scale of the underlying quantizer().
"""

add_docstr_all(
    "q_zero_point",
    r"""
    q_zero_point() -> int

    Given a Tensor quantized by linear(affine) quantization,
    returns the zero_point of the underlying quantizer().
    """,
)

add_docstr_all(
    "q_per_channel_scales",
    r"""
    q_per_channel_scales() -> Tensor

    Given a Tensor quantized by linear (affine) per-channel quantization,
    returns a Tensor of scales of the underlying quantizer. It has the number of
    elements that matches the corresponding dimensions (from q_per_channel_axis) of
    the tensor.
    """,
)

add_docstr_all(
    "q_per_channel_zero_points",
    r"""
    q_per_channel_zero_points() -> Tensor

    Given a Tensor quantized by linear (affine) per-channel quantization,
    returns a tensor of zero_points of the underlying quantizer. It has the number of
    elements that matches the corresponding dimensions (from q_per_channel_axis) of
    the tensor.
    """,
)

add_docstr_all(
    "q_per_channel_axis",
    r"""
    q_per_channel_axis() -> int

    Given a Tensor quantized by linear (affine) per-channel quantization,
    returns the index of dimension on which per-channel quantization is applied.
    """,
)

add_docstr_all(
    "random_",
    r"""
    random_(from=0, to=None, *, generator=None) -> Tensor

    Fills :attr:`self` tensor with numbers sampled from the discrete uniform
    distribution over ``[from, to - 1]``. If not specified, the values are usually
    only bounded by :attr:`self` tensor's data type. However, for floating point
    types, if unspecified, range will be ``[0, 2^mantissa]`` to ensure that every
    value is representable. For example, `torch.tensor(1, dtype=torch.double).random_()`
    will be uniform in ``[0, 2^53]``.
    """,
)

add_docstr_all(
    "rad2deg",
    r"""
    rad2deg() -> Tensor

    See :func:`torch.rad2deg`
    """,
)

add_docstr_all(
    "rad2deg_",
    r"""
    rad2deg_() -> Tensor

    In-place version of :meth:`~Tensor.rad2deg`
    """,
)

add_docstr_all(
    "deg2rad",
    r"""
    deg2rad() -> Tensor

    See :func:`torch.deg2rad`
    """,
)

add_docstr_all(
    "deg2rad_",
    r"""
    deg2rad_() -> Tensor

    In-place version of :meth:`~Tensor.deg2rad`
    """,
)

add_docstr_all(
    "ravel",
    r"""
    ravel() -> Tensor

    see :func:`torch.ravel`
    """,
)

add_docstr_all(
    "reciprocal",
    r"""
    reciprocal() -> Tensor

    See :func:`torch.reciprocal`
    """,
)

add_docstr_all(
    "reciprocal_",
    r"""
    reciprocal_() -> Tensor

    In-place version of :meth:`~Tensor.reciprocal`
    """,
)

add_docstr_all(
    "record_stream",
    r"""
    record_stream(stream)

    Marks the tensor as having been used by this stream.  When the tensor
    is deallocated, ensure the tensor memory is not reused for another tensor
    until all work queued on :attr:`stream` at the time of deallocation is
    complete.

    .. note::

        The caching allocator is aware of only the stream where a tensor was
        allocated. Due to the awareness, it already correctly manages the life
    """,
)
    # 将张量的使用信息告知内存分配器，以防止在不同流上重用内存
    # 当张量仅在一个流上周期性使用时，内存分配器可能会意外地重用内存
    # 调用此方法可以让内存分配器知道哪些流使用了该张量
# 该方法适合在需要从侧流（side stream）中创建张量并希望用户在使用张量时不必过多考虑流安全性的情况下使用。
# 这些安全性保证会带来一些性能和可预测性的代价（类似于垃圾回收和手动内存管理之间的权衡），因此，如果您能够管理张量的整个生命周期，
# 可能考虑手动管理 CUDA 事件，以避免必须调用此方法。
# 特别是在调用此方法后，在后续的内存分配中，分配器将轮询记录的流以查看所有操作是否已完成；
# 您可能会与侧流计算竞争，并且在分配内存时会出现非确定性的重用或无法重用。

# 您可以安全地使用在侧流上分配的张量而无需使用 :meth:`~Tensor.record_stream`；
# 您必须手动确保张量的任何非创建流使用在释放张量之前同步回创建流。
# 由于 CUDA 缓存分配器保证内存仅在相同创建流上重用，因此这足以确保将来重新分配内存的写操作将延迟到非创建流使用完成时。
# （与直觉相反，您可能会观察到在 CPU 端我们已经重新分配了张量，即使旧张量上的 CUDA 内核仍在进行中也是如此。
# 这是可以接受的，因为对新张量的 CUDA 操作将适当地等待旧操作完成，因为它们都在同一个流上。）

# 具体地，操作如下所示::

#     with torch.cuda.stream(s0):
#         x = torch.zeros(N)

#     s1.wait_stream(s0)
#     with torch.cuda.stream(s1):
#         y = some_comm_op(x)

#     ... 在流 s0 上进行一些计算 ...

# 在释放 x 之前，将创建流 s0 同步到侧流 s1
# s0.wait_stream(s1)
# del x

# 请注意，在决定何时执行 ``s0.wait_stream(s1)`` 时需要一些审慎。
# 特别是，如果我们在 ``some_comm_op`` 之后立即等待，那么使用侧流的意义就不存在了；
# 这相当于在 ``s0`` 上运行 ``some_comm_op``。相反，同步必须放置在某个适当的、稍后的时间点，
# 您预期侧流 ``s1`` 已完成工作。通常通过分析来标识此位置，例如使用生成的 Chrome 跟踪
# :meth:`torch.autograd.profiler.profile.export_chrome_trace`。如果等待时间点过早，
# s0 上的工作将阻塞，直到 ``s1`` 完成，从而防止进一步的通信和计算重叠。
# 如果等待时间点太晚，则会使用比严格
    necessary (as you are keeping ``x`` live for longer.)  For a concrete
    example of how this guidance can be applied in practice, see this post:
    `FSDP and CUDACachingAllocator
    <https://dev-discuss.pytorch.org/t/fsdp-cudacachingallocator-an-outsider-newb-perspective/1486>`_.



    # 这是一个多行注释的示例，它并不影响代码执行，主要用于提供额外的文档和说明
    # 本段文字提到了一个具体的实践示例，建议查看相关链接以了解更多细节
# 修改张量的形状，返回具有指定形状的新张量
def reshape(*shape) -> Tensor:
# 调整张量的形状以与指定的张量相同。等效于 self.resize_(tensor.size())
def resize_as_(tensor, memory_format=torch.contiguous_format) -> Tensor:
    """
    调整 self 张量的大小，使其与指定的 tensor 相同大小。

    Args:
        tensor (torch.Tensor): 用于指定大小的张量
        memory_format (torch.memory_format, 可选): 张量的内存格式。默认为 torch.contiguous_format。
            注意，如果 self.size() 与 sizes 匹配，则不会影响 self 的内存格式。

    Example::

        >>> x = torch.tensor([[1, 2], [3, 4], [5, 6]])
        >>> y = torch.tensor([[7, 8], [9, 10]])
        >>> x.resize_as_(y)
        tensor([[ 1,  2],
                [ 3,  4]])
    """
    pass
    memory_format (:class:`torch.memory_format`, optional): the desired memory format of
        Tensor. Default: ``torch.contiguous_format``. Note that memory format of
        :attr:`self` is going to be unaffected if ``self.size()`` matches ``tensor.size()``.
# 为函数 "rot90" 添加文档字符串
add_docstr_all(
    "rot90",
    r"""
rot90(k, dims) -> Tensor

See :func:`torch.rot90`
""",
)

# 为函数 "round" 添加文档字符串
add_docstr_all(
    "round",
    r"""
round(decimals=0) -> Tensor

See :func:`torch.round`
""",
)

# 为函数 "round_" 添加文档字符串
add_docstr_all(
    "round_",
    r"""
round_(decimals=0) -> Tensor

In-place version of :meth:`~Tensor.round`
""",
)

# 为函数 "rsqrt" 添加文档字符串
add_docstr_all(
    "rsqrt",
    r"""
rsqrt() -> Tensor

See :func:`torch.rsqrt`
""",
)

# 为函数 "rsqrt_" 添加文档字符串
add_docstr_all(
    "rsqrt_",
    r"""
rsqrt_() -> Tensor

In-place version of :meth:`~Tensor.rsqrt`
""",
)

# 为函数 "scatter_" 添加文档字符串
add_docstr_all(
    "scatter_",
    r"""
scatter_(dim, index, src, *, reduce=None) -> Tensor

Writes all values from the tensor :attr:`src` into :attr:`self` at the indices
specified in the :attr:`index` tensor. For each value in :attr:`src`, its output
index is specified by its index in :attr:`src` for ``dimension != dim`` and by
the corresponding value in :attr:`index` for ``dimension = dim``.

For a 3-D tensor, :attr:`self` is updated as::

    self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
    self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
    self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2

This is the reverse operation of the manner described in :meth:`~Tensor.gather`.

:attr:`self`, :attr:`index` and :attr:`src` (if it is a Tensor) should all have
the same number of dimensions. It is also required that
``index.size(d) <= src.size(d)`` for all dimensions ``d``, and that
``index.size(d) <= self.size(d)`` for all dimensions ``d != dim``.
Note that ``index`` and ``src`` do not broadcast.

Moreover, as for :meth:`~Tensor.gather`, the values of :attr:`index` must be
between ``0`` and ``self.size(dim) - 1`` inclusive.

.. warning::

    When indices are not unique, the behavior is non-deterministic (one of the
    values from ``src`` will be picked arbitrarily) and the gradient will be
    incorrect (it will be propagated to all locations in the source that
    correspond to the same index)!

.. note::

    The backward pass is implemented only for ``src.shape == index.shape``.

Additionally accepts an optional :attr:`reduce` argument that allows
specification of an optional reduction operation, which is applied to all
values in the tensor :attr:`src` into :attr:`self` at the indices
specified in the :attr:`index`. For each value in :attr:`src`, the reduction
operation is applied to an index in :attr:`self` which is specified by
its index in :attr:`src` for ``dimension != dim`` and by the corresponding
value in :attr:`index` for ``dimension = dim``.

Given a 3-D tensor and reduction using the multiplication operation, :attr:`self`
is updated as::

    self[index[i][j][k]][j][k] *= src[i][j][k]  # if dim == 0
    self[i][index[i][j][k]][k] *= src[i][j][k]  # if dim == 1
    self[i][j][index[i][j][k]] *= src[i][j][k]  # if dim == 2

Reducing with the addition operation is the same as using
:meth:`~torch.Tensor.scatter_add_`.

.. warning::
    # 使用 Tensor 的 reduce 参数在未来的 PyTorch 版本中将被弃用，建议使用 :meth:`~torch.Tensor.scatter_reduce_` 
    # 替代，以获得更多的减少选项。
    The reduce argument with Tensor ``src`` is deprecated and will be removed in
    a future PyTorch release. Please use :meth:`~torch.Tensor.scatter_reduce_`
    instead for more reduction options.
# 导入了scatter_add_方法
from typing import Any

# 定义scatter_add_方法，作用是将src中的值根据index张量指定的索引加到self张量中
def scatter_add_(dim: int, index: LongTensor, src: Tensor) -> Tensor:
    """
    Adds all values from the tensor `src` into `self` at the indices
    specified in the `index` tensor in a similar fashion as
    :meth:`~torch.Tensor.scatter_`. For each value in `src`, it is added to
    an index in `self` which is specified by its index in `src`
    for dimension != dim and by the corresponding value in `index` for
    dimension = dim.

    Args:
        dim (int): the axis along which to index
        index (LongTensor): the indices of elements to scatter, can be either empty
            or of the same dimensionality as `src`. When empty, the operation
            returns `self` unchanged.
        src (Tensor): the source element(s) to scatter.

    Returns:
        Tensor: updated tensor `self`.

    Example::

        >>> index = torch.tensor([[0, 1]])
        >>> src = torch.tensor([[1, 2]])
        >>> torch.zeros(3, 2).scatter_add_(0, index, src)
        tensor([[1., 2.],
                [0., 0.],
                [0., 0.]])
    """
    # 如果index张量为空，则直接返回self张量，不做任何修改
    if index.numel() == 0:
        return self

    # 确保index张量和src张量的维度相同
    assert index.shape == src.shape, "index tensor must have the same shape as src tensor"

    # 循环遍历src张量中的每个元素，并将其加到self张量对应位置上
    for i in range(src.size(dim)):
        # 构造索引张量，用于确定加法操作的位置
        index_slice = tuple(index.select(dim, i).squeeze(dim).tolist())
        # 执行加法操作，将src张量中的值加到self张量对应位置上
        self[index_slice] += src.select(dim, i)

    # 返回更新后的self张量
    return self
    # 将 src[i][j][k] 添加到 self[i][j][index[i][j][k]] 上，仅当 dim == 2 时有效
    self[i][j][index[i][j][k]] += src[i][j][k]  # if dim == 2
"""
scatter_reduce_(dim, index, src, reduce, *, include_self=True) -> Tensor

Reduces all values from the `src` tensor to the indices specified in
the `index` tensor in the `self` tensor using the applied reduction
defined via the `reduce` argument (`"sum"`, `"prod"`, `"mean"`,
`"amax"`, `"amin"`). For each value in `src`, it is reduced to an
index in `self` which is specified by its index in `src` for
dimension != dim and by the corresponding value in `index` for
dimension = dim. If `include_self=True`, the values in the `self`
tensor are included in the reduction.

`self`, `index` and `src` should all have
the same number of dimensions. It is also required that
index.size(d) <= src.size(d) for all dimensions d, and that
index.size(d) <= self.size(d) for all dimensions d != dim.
Note that `index` and `src` do not broadcast.

For a 3-D tensor with reduce="sum" and include_self=True the
output is given as::

    self[index[i][j][k]][j][k] += src[i][j][k]  # if dim == 0
    self[i][index[i][j][k]][k] += src[i][j][k]  # if dim == 1
    self[i][j][index[i][j][k]] += src[i][j][k]  # if dim == 2

Note:
    {forward_reproducibility_note}

.. note::

    The backward pass is implemented only for `src.shape == index.shape`.

.. warning::

    This function is in beta and may change in the near future.

Args:
    dim (int): the axis along which to index
    index (LongTensor): the indices of elements to scatter and reduce.
    src (Tensor): the source elements to scatter and reduce
"""
    reduce (str): the reduction operation to apply for non-unique indices
        (:obj:`"sum"`, :obj:`"prod"`, :obj:`"mean"`, :obj:`"amax"`, :obj:`"amin"`)
    include_self (bool): whether elements from the :attr:`self` tensor are
        included in the reduction
# Example::

# 创建一个包含浮点数的张量
>>> src = torch.tensor([1., 2., 3., 4., 5., 6.])
# 创建一个索引张量，指定了src张量中每个位置的分配目标
>>> index = torch.tensor([0, 1, 0, 1, 2, 1])
# 创建一个输入张量，用于接收散布归约操作的结果
>>> input = torch.tensor([1., 2., 3., 4.])
# 对输入张量进行散布归约操作，使用索引和src张量，并指定reduce操作为求和
>>> input.scatter_reduce(0, index, src, reduce="sum")
# 返回结果张量，其值为对应索引处散布归约的结果
tensor([5., 14., 8., 4.])
# 再次进行散布归约操作，不包含自身元素，结果如上
>>> input.scatter_reduce(0, index, src, reduce="sum", include_self=False)
tensor([4., 12., 5., 4.])
# 创建另一个输入张量
>>> input2 = torch.tensor([5., 4., 3., 2.])
# 对其进行散布归约操作，reduce操作为取最大值
>>> input2.scatter_reduce(0, index, src, reduce="amax")
# 返回结果张量，其值为对应索引处散布归约的最大值
tensor([5., 6., 5., 2.])
# 再次进行散布归约操作，不包含自身元素，结果如上
>>> input2.scatter_reduce(0, index, src, reduce="amax", include_self=False)
tensor([3., 6., 5., 2.])


""".format(
        **reproducibility_notes
    ),
)

# 下面是函数的文档字符串注释，用于torch的各个函数
add_docstr_all(
    "select",
    r"""
select(dim, index) -> Tensor

See :func:`torch.select`
""",
)

add_docstr_all(
    "select_scatter",
    r"""
select_scatter(src, dim, index) -> Tensor

See :func:`torch.select_scatter`
""",
)

add_docstr_all(
    "slice_scatter",
    r"""
slice_scatter(src, dim=0, start=None, end=None, step=1) -> Tensor

See :func:`torch.slice_scatter`
""",
)

add_docstr_all(
    "set_",
    r"""
set_(source=None, storage_offset=0, size=None, stride=None) -> Tensor

Sets the underlying storage, size, and strides. If :attr:`source` is a tensor,
:attr:`self` tensor will share the same storage and have the same size and
strides as :attr:`source`. Changes to elements in one tensor will be reflected
in the other.

If :attr:`source` is a :class:`~torch.Storage`, the method sets the underlying
storage, offset, size, and stride.

Args:
    source (Tensor or Storage): the tensor or storage to use
    storage_offset (int, optional): the offset in the storage
    size (torch.Size, optional): the desired size. Defaults to the size of the source.
    stride (tuple, optional): the desired stride. Defaults to C-contiguous strides.
""",
)

add_docstr_all(
    "sigmoid",
    r"""
sigmoid() -> Tensor

See :func:`torch.sigmoid`
""",
)

add_docstr_all(
    "sigmoid_",
    r"""
sigmoid_() -> Tensor

In-place version of :meth:`~Tensor.sigmoid`
""",
)

add_docstr_all(
    "logit",
    r"""
logit() -> Tensor

See :func:`torch.logit`
""",
)

add_docstr_all(
    "logit_",
    r"""
logit_() -> Tensor

In-place version of :meth:`~Tensor.logit`
""",
)

add_docstr_all(
    "sign",
    r"""
sign() -> Tensor

See :func:`torch.sign`
""",
)

add_docstr_all(
    "sign_",
    r"""
sign_() -> Tensor

In-place version of :meth:`~Tensor.sign`
""",
)

add_docstr_all(
    "signbit",
    r"""
signbit() -> Tensor

See :func:`torch.signbit`
""",
)

add_docstr_all(
    "sgn",
    r"""
sgn() -> Tensor

See :func:`torch.sgn`
""",
)

add_docstr_all(
    "sgn_",
    r"""
sgn_() -> Tensor

In-place version of :meth:`~Tensor.sgn`
""",
)

add_docstr_all(
    "sin",
    r"""
sin() -> Tensor

See :func:`torch.sin`
""",
)

add_docstr_all(
    "sin_",
    r"""
sin_() -> Tensor

In-place version of :meth:`~Tensor.sin`
""",
)

add_docstr_all(
    "sinc",
    r"""
sinc() -> Tensor

See :func:`torch.sinc`
""",
)

# 最后一个函数的文档字符串注释结束
    # 创建一个原始字符串（raw string），内容为 "sinc_"
    "sinc_",
# 调整稀疏张量的大小并清除内容，就地操作
def sparse_resize_and_clear_(size, sparse_dim, dense_dim):
    """
    Resizes the sparse tensor `self` to the desired size and number of sparse and dense dimensions.

    Args:
        size (torch.Size): Desired size of the tensor.
            If `self` is non-empty, the desired size cannot be smaller than the original size.
        sparse_dim (int): Number of sparse dimensions.
        dense_dim (int): Number of dense dimensions.

    Notes:
        - If `self` is empty (zero elements), `size`, `sparse_dim`, and `dense_dim` can be any
          positive integers such that `len(size) == sparse_dim + dense_dim`.
        - If `self` is non-empty, each dimension in `size` must not be smaller than the corresponding
          dimension of `self`. Also, `sparse_dim` must match the number of sparse dimensions in `self`,
          and `dense_dim` must match the number of dense dimensions in `self`.

    Raises:
        - RuntimeError: If `self` is not a sparse tensor.

    Example::

        >>> t = torch.sparse_coo_tensor(indices=[[0, 1], [2, 0]], values=[3.0, 4.0], size=(3, 4))
        >>> t.sparse_resize_and_clear_(size=(2, 3), sparse_dim=2, dense_dim=0)
    """
"""
Removes all specified elements from a :ref:`sparse tensor
<sparse-docs>` :attr:`self` and resizes :attr:`self` to the desired
size and the number of sparse and dense dimensions.

.. warning:
  Throws an error if :attr:`self` is not a sparse tensor.

Args:
    size (torch.Size): the desired size.
    sparse_dim (int): the number of sparse dimensions
    dense_dim (int): the number of dense dimensions
"""
)

"""
Adds docstring to the function named "sqrt".

sqrt() -> Tensor

See :func:`torch.sqrt`
"""
add_docstr_all(
    "sqrt",
    r"""
sqrt() -> Tensor

See :func:`torch.sqrt`
"""
)

"""
Adds docstring to the function named "sqrt_".

sqrt_() -> Tensor

In-place version of :meth:`~Tensor.sqrt`
"""
add_docstr_all(
    "sqrt_",
    r"""
sqrt_() -> Tensor

In-place version of :meth:`~Tensor.sqrt`
"""
)

"""
Adds docstring to the function named "square".

square() -> Tensor

See :func:`torch.square`
"""
add_docstr_all(
    "square",
    r"""
square() -> Tensor

See :func:`torch.square`
"""
)

"""
Adds docstring to the function named "square_".

square_() -> Tensor

In-place version of :meth:`~Tensor.square`
"""
add_docstr_all(
    "square_",
    r"""
square_() -> Tensor

In-place version of :meth:`~Tensor.square`
"""
)

"""
Adds docstring to the function named "squeeze".

squeeze(dim=None) -> Tensor

See :func:`torch.squeeze`
"""
add_docstr_all(
    "squeeze",
    r"""
squeeze(dim=None) -> Tensor

See :func:`torch.squeeze`
"""
)

"""
Adds docstring to the function named "squeeze_".

squeeze_(dim=None) -> Tensor

In-place version of :meth:`~Tensor.squeeze`
"""
add_docstr_all(
    "squeeze_",
    r"""
squeeze_(dim=None) -> Tensor

In-place version of :meth:`~Tensor.squeeze`
"""
)

"""
Adds docstring to the function named "std".

std(dim=None, *, correction=1, keepdim=False) -> Tensor

See :func:`torch.std`
"""
add_docstr_all(
    "std",
    r"""
std(dim=None, *, correction=1, keepdim=False) -> Tensor

See :func:`torch.std`
"""
)

"""
Adds docstring to the function named "storage_offset".

storage_offset() -> int

Returns :attr:`self` tensor's offset in the underlying storage in terms of
number of storage elements (not bytes).

Example::

    >>> x = torch.tensor([1, 2, 3, 4, 5])
    >>> x.storage_offset()
    0
    >>> x[3:].storage_offset()
    3
"""
add_docstr_all(
    "storage_offset",
    r"""
storage_offset() -> int

Returns :attr:`self` tensor's offset in the underlying storage in terms of
number of storage elements (not bytes).

Example::

    >>> x = torch.tensor([1, 2, 3, 4, 5])
    >>> x.storage_offset()
    0
    >>> x[3:].storage_offset()
    3
"""
)

"""
Adds docstring to the function named "untyped_storage".

untyped_storage() -> torch.UntypedStorage

Returns the underlying :class:`UntypedStorage`.
"""
add_docstr_all(
    "untyped_storage",
    r"""
untyped_storage() -> torch.UntypedStorage

Returns the underlying :class:`UntypedStorage`.
"""
)

"""
Adds docstring to the function named "stride".

stride(dim) -> tuple or int

Returns the stride of :attr:`self` tensor.

Stride is the jump necessary to go from one element to the next one in the
specified dimension :attr:`dim`. A tuple of all strides is returned when no
argument is passed in. Otherwise, an integer value is returned as the stride in
the particular dimension :attr:`dim`.

Args:
    dim (int, optional): the desired dimension in which stride is required

Example::

    >>> x = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    >>> x.stride()
    (5, 1)
    >>> x.stride(0)
    5
    >>> x.stride(-1)
    1
"""
add_docstr_all(
    "stride",
    r"""
stride(dim) -> tuple or int

Returns the stride of :attr:`self` tensor.

Stride is the jump necessary to go from one element to the next one in the
specified dimension :attr:`dim`. A tuple of all strides is returned when no
argument is passed in. Otherwise, an integer value is returned as the stride in
the particular dimension :attr:`dim`.

Args:
    dim (int, optional): the desired dimension in which stride is required

Example::

    >>> x = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    >>> x.stride()
    (5, 1)
    >>> x.stride(0)
    5
    >>> x.stride(-1)
    1
"""
)

"""
Adds docstring to the function named "sub".

sub(other, *, alpha=1) -> Tensor

See :func:`torch.sub`.
"""
add_docstr_all(
    "sub",
    r"""
sub(other, *, alpha=1) -> Tensor

See :func:`torch.sub`.
"""
)

"""
Adds docstring to the function named "sub_".

sub_(other, *, alpha=1) -> Tensor

In-place version of :meth:`~Tensor.sub`
"""
add_docstr_all(
    "sub_",
    r"""
sub_(other, *, alpha=1) -> Tensor

In-place version of :meth:`~Tensor.sub`
"""
)

"""
Adds docstring to the function named "subtract".

subtract(other, *, alpha=1) -> Tensor

See :func:`torch.subtract`.
"""
add_docstr_all(
    "subtract",
    r"""
subtract(other, *, alpha=1) -> Tensor

See :func:`torch.subtract`.
"""
)

"""
Adds docstring to the function named "subtract_".

subtract_(other, *, alpha=1) -> Tensor

In-place version of :meth:`~Tensor.subtract`.
"""
add_docstr_all(
    "subtract_",
    r"""
subtract_(other, *, alpha=1) -> Tensor

In-place version of :meth:`~Tensor.subtract`.
"""
)

"""
Adds docstring to the function named "sum".

sum(dim=None, keepdim=False, dtype=None) -> Tensor

See :func:`torch.sum`
"""
add_docstr_all(
    "sum",
    r"""
sum(dim=None, keepdim=False, dtype=None) -> Tensor

See :func:`torch.sum`
"""
)

"""
Adds docstring to the function named "nansum".

nansum(dim=None, keepdim=False, dtype=None) -> Tensor

See :func:`torch.nansum`
"""
add_docstr_all(
    "nansum",
    r"""
nansum(dim=None, keepdim=False, dtype=None) -> Tensor

See :func:`torch.nansum`
"""
)

"""
This is a placeholder for the next function documentation.

The actual function definition is not provided in the given code snippet.
"""
    r"""


注释：


# 定义一个原始字符串字面值，通常用于包含格式化字符或正则表达式的代码中，
# 这里的 r""" 表示一个多行的原始字符串，可以跨越多行而不需要转义特殊字符。
# 在这个特定的例子中，r""" 可能用于创建包含大量文本或格式化代码的字符串。
# 请注意，r""" 后面的三个双引号表示字符串的开始，它应该有一个匹配的三个双引号来结束字符串。
# 定义一个函数签名，说明函数接受的参数和返回值类型
svd(some=True, compute_uv=True) -> (Tensor, Tensor, Tensor)

# 详细文档见 torch.svd
"""

add_docstr_all(
    "swapdims",
    r"""
    swapdims(dim0, dim1) -> Tensor

    See :func:`torch.swapdims`
    """,
)

add_docstr_all(
    "swapdims_",
    r"""
    swapdims_(dim0, dim1) -> Tensor

    In-place version of :meth:`~Tensor.swapdims`
    """,
)

add_docstr_all(
    "swapaxes",
    r"""
    swapaxes(axis0, axis1) -> Tensor

    See :func:`torch.swapaxes`
    """,
)

add_docstr_all(
    "swapaxes_",
    r"""
    swapaxes_(axis0, axis1) -> Tensor

    In-place version of :meth:`~Tensor.swapaxes`
    """,
)

add_docstr_all(
    "t",
    r"""
    t() -> Tensor

    See :func:`torch.t`
    """,
)

add_docstr_all(
    "t_",
    r"""
    t_() -> Tensor

    In-place version of :meth:`~Tensor.t`
    """,
)

add_docstr_all(
    "tile",
    r"""
    tile(dims) -> Tensor

    See :func:`torch.tile`
    """,
)

add_docstr_all(
    "to",
    r"""
    to(*args, **kwargs) -> Tensor

    Performs Tensor dtype and/or device conversion. A :class:`torch.dtype` and :class:`torch.device` are
    inferred from the arguments of ``self.to(*args, **kwargs)``.

    .. note::

        If the ``self`` Tensor already
        has the correct :class:`torch.dtype` and :class:`torch.device`, then ``self`` is returned.
        Otherwise, the returned tensor is a copy of ``self`` with the desired
        :class:`torch.dtype` and :class:`torch.device`.

    Here are the ways to call ``to``:

    .. method:: to(dtype, non_blocking=False, copy=False, memory_format=torch.preserve_format) -> Tensor
       :noindex:

        Returns a Tensor with the specified :attr:`dtype`

        Args:
            {memory_format}

    .. method:: to(device=None, dtype=None, non_blocking=False, copy=False, memory_format=torch.preserve_format) -> Tensor
       :noindex:

        Returns a Tensor with the specified :attr:`device` and (optional)
        :attr:`dtype`. If :attr:`dtype` is ``None`` it is inferred to be ``self.dtype``.
        When :attr:`non_blocking`, tries to convert asynchronously with respect to
        the host if possible, e.g., converting a CPU Tensor with pinned memory to a
        CUDA Tensor.
        When :attr:`copy` is set, a new Tensor is created even when the Tensor
        already matches the desired conversion.

        Args:
            {memory_format}

    .. method:: to(other, non_blocking=False, copy=False) -> Tensor
       :noindex:

        Returns a Tensor with same :class:`torch.dtype` and :class:`torch.device` as
        the Tensor :attr:`other`. When :attr:`non_blocking`, tries to convert
        asynchronously with respect to the host if possible, e.g., converting a CPU
        Tensor with pinned memory to a CUDA Tensor.
        When :attr:`copy` is set, a new Tensor is created even when the Tensor
        already matches the desired conversion.

    Example::

        >>> tensor = torch.randn(2, 2)  # Initially dtype=float32, device=cpu
        >>> tensor.to(torch.float64)
        tensor([[-0.5044,  0.0005],
                [ 0.3310, -0.0584]], dtype=torch.float64)

        >>> cuda0 = torch.device('cuda:0')
        >>> tensor.to(cuda0)

    """
    # 创建一个包含两行两列数据的张量，存储在 CUDA 设备 cuda:0 上
    tensor([[-0.5044,  0.0005],
            [ 0.3310, -0.0584]], device='cuda:0')
    
    # 将张量 tensor 转换为指定的数据类型 torch.float64，并放置在 CUDA 设备 cuda:0 上
    >>> tensor.to(cuda0, dtype=torch.float64)
    tensor([[-0.5044,  0.0005],
            [ 0.3310, -0.0584]], dtype=torch.float64, device='cuda:0')
    
    # 使用指定的数据类型 torch.float64 和 CUDA 设备 cuda0 创建一个随机数张量
    # 使用 non_blocking=True 参数将张量 tensor 转换到张量 other 的设备，保证非阻塞操作
    >>> other = torch.randn((), dtype=torch.float64, device=cuda0)
    >>> tensor.to(other, non_blocking=True)
    tensor([[-0.5044,  0.0005],
            [ 0.3310, -0.0584]], dtype=torch.float64, device='cuda:0')
# 为方法 "long" 添加文档字符串，指定其作用和参数描述
add_docstr_all(
    "long",
    r"""
long(memory_format=torch.preserve_format) -> Tensor

``self.long()`` 等同于 ``self.to(torch.int64)``。参见 :func:`to`。

Args:
    {memory_format}
""".format(
        **common_args
    ),
)
# 定义 long 方法，返回一个 Tensor
long(memory_format=torch.preserve_format) -> Tensor
# 方法 self.long() 等同于 self.to(torch.int64)，参见 to 函数的文档

# 方法 short，返回一个 Tensor
def short(memory_format=torch.preserve_format) -> Tensor:
    # 方法 self.short() 等同于 self.to(torch.int16)，参见 to 函数的文档
    pass

# 方法 take，接受一个参数 indices，返回一个 Tensor
def take(indices) -> Tensor:
    # 参见 torch.take 函数的文档
    pass

# 方法 take_along_dim，接受两个参数 indices 和 dim，返回一个 Tensor
def take_along_dim(indices, dim) -> Tensor:
    # 参见 torch.take_along_dim 函数的文档
    pass

# 方法 tan，返回一个 Tensor
def tan() -> Tensor:
    # 参见 torch.tan 函数的文档
    pass

# 方法 tan_，返回一个 Tensor
def tan_() -> Tensor:
    # 原地执行方法，等同于 Tensor.tan() 的就地版本
    pass

# 方法 tanh，返回一个 Tensor
def tanh() -> Tensor:
    # 参见 torch.tanh 函数的文档
    pass

# 方法 softmax，接受一个参数 dim，返回一个 Tensor
def softmax(dim) -> Tensor:
    # 是 torch.nn.functional.softmax 的别名
    pass

# 方法 tanh_，返回一个 Tensor
def tanh_() -> Tensor:
    # 原地执行方法，等同于 Tensor.tanh() 的就地版本
    pass

# 方法 tolist，返回一个列表或数字
def tolist() -> Union[list, number]:
    # 将 Tensor 转换为（嵌套的）列表。对于标量，返回标准的 Python 数字，类似于 Tensor.item 方法
    # 如有必要，会自动将 Tensor 移动到 CPU 上
    # 此操作不可微分
    pass

# 方法 topk，接受参数 k, dim, largest, sorted，返回一个元组 (Tensor, LongTensor)
def topk(k, dim=None, largest=True, sorted=True) -> Tuple[Tensor, LongTensor]:
    # 参见 torch.topk 函数的文档
    pass

# 方法 to_dense，接受参数 dtype 和 masked_grad，返回一个 Tensor
def to_dense(dtype=None, *, masked_grad=True) -> Tensor:
    # 如果 self 不是分块张量，则创建 self 的分块副本，否则返回 self
    # 关键字参数：
    # dtype：指定的数据类型
    # masked_grad（bool，可选）：如果设置为 True（默认），并且 self 具有稀疏布局，则 to_dense 的反向传播返回 grad.sparse_mask(self)
    # 示例见文档
    pass

# 方法 to_sparse，接受参数 sparseDims，返回一个 Tensor
def to_sparse(sparseDims) -> Tensor:
    # 返回张量的稀疏副本。PyTorch 支持坐标格式的稀疏张量
    # 参数：
    # sparseDims：新稀疏张量中包含的稀疏维度数
    # 示例见文档
    pass
    # 将稠密张量转换为稀疏张量
    >>> d.to_sparse()
    # 返回稀疏张量对象，其中包含稀疏表示所需的索引、值和大小信息
    tensor(indices=tensor([[1, 1],    # 稀疏张量的非零元素索引，这里是 (1,1) 和 (0,2)
                           [0, 2]]),
           values=tensor([ 9, 10]),   # 稀疏张量的非零元素的值，这里是 9 和 10
           size=(3, 3),               # 稀疏张量的大小，这里是 (3,3)
           nnz=2,                     # 稀疏张量的非零元素数量，这里是 2
           layout=torch.sparse_coo)   # 稀疏张量的布局，这里是 COO (Coordinate) 布局
    
    # 将稠密张量按指定的维度转换为稀疏张量
    >>> d.to_sparse(1)
    # 返回按第1维度（列）压缩的稀疏张量对象，包含稀疏表示所需的索引、值和大小信息
    tensor(indices=tensor([[1]]),     # 稀疏张量的非零元素索引，这里是 (1,0)
           values=tensor([[ 9,  0, 10]]),  # 稀疏张量的非零元素的值，这里是 [9, 0, 10]
           size=(3, 3),               # 稀疏张量的大小，这里是 (3,3)
           nnz=1,                     # 稀疏张量的非零元素数量，这里是 1
           layout=torch.sparse_coo)   # 稀疏张量的布局，这里是 COO (Coordinate) 布局
.. method:: to_sparse(*, layout=None, blocksize=None, dense_dim=None) -> Tensor
   :noindex:

返回具有指定布局和块大小的稀疏张量。如果 :attr:`self` 是分步的，则可以指定密集维度，并且将创建一个混合稀疏张量，其中有 `dense_dim` 个密集维度和 `self.dim() - 2 - dense_dim` 个批次维度。

.. note:: 如果 :attr:`self` 的布局和块大小与指定的布局和块大小匹配，则返回 :attr:`self`。否则，返回 :attr:`self` 的稀疏张量副本。

Args:

    layout (:class:`torch.layout`, optional): 所需的稀疏布局。可以是 ``torch.sparse_coo``、``torch.sparse_csr``、``torch.sparse_csc``、``torch.sparse_bsr`` 或 ``torch.sparse_bsc`` 中的一个。默认为 ``torch.sparse_coo``。

    blocksize (list, tuple, :class:`torch.Size`, optional): 结果为 BSR 或 BSC 张量的块大小。对于其他布局，指定非 ``None`` 的块大小将导致 RuntimeError 异常。块大小必须是长度为两个的元组，其项均匀地划分两个稀疏维度。

    dense_dim (int, optional): 结果为 CSR、CSC、BSR 或 BSC 张量的密集维度数。仅当 :attr:`self` 是分步张量时才应使用此参数，并且必须是介于 0 和 :attr:`self` 张量维数减去两个的值之间。

Example::

    >>> x = torch.tensor([[1, 0], [0, 0], [2, 3]])
    >>> x.to_sparse(layout=torch.sparse_coo)
    tensor(indices=tensor([[0, 2, 2],
                           [0, 0, 1]]),
           values=tensor([1, 2, 3]),
           size=(3, 2), nnz=3, layout=torch.sparse_coo)
    >>> x.to_sparse(layout=torch.sparse_bsr, blocksize=(1, 2))
    tensor(crow_indices=tensor([0, 1, 1, 2]),
           col_indices=tensor([0, 0]),
           values=tensor([[[1, 0]],
                          [[2, 3]]]), size=(3, 2), nnz=2, layout=torch.sparse_bsr)
    >>> x.to_sparse(layout=torch.sparse_bsr, blocksize=(2, 1))
    RuntimeError: Tensor size(-2) 3 needs to be divisible by blocksize[0] 2
    >>> x.to_sparse(layout=torch.sparse_csr, blocksize=(3, 1))
    RuntimeError: to_sparse for Strided to SparseCsr conversion does not use specified blocksize

    >>> x = torch.tensor([[[1], [0]], [[0], [0]], [[2], [3]]])
    >>> x.to_sparse(layout=torch.sparse_csr, dense_dim=1)
    tensor(crow_indices=tensor([0, 1, 1, 3]),
           col_indices=tensor([0, 0, 1]),
           values=tensor([[1],
                          [2],
                          [3]]), size=(3, 2, 1), nnz=3, layout=torch.sparse_csr)

"""

add_docstr_all(
    "to_sparse_csr",
    r"""
to_sparse_csr(dense_dim=None) -> Tensor

将张量转换为压缩行存储格式（CSR）。除了分步张量外，仅适用于二维张量。如果 :attr:`self` 是分步的，则可以指定密集维度，并且一个



to_sparse_csr(dense_dim=None) -> Tensor
将张量转换为压缩行存储格式（CSR）。除非是分步张量，否则仅适用于二维张量。如果 :attr:`self` 是分步的，则可以指定密集维度，并且一个
# 转换为块稀疏行 (BSR) 存储格式的稀疏张量。如果 self 是分块张量，
# 则可以指定稠密维度，创建具有 dense_dim 个稠密维度和 self.dim() - 2 - dense_dim 个批量维度的混合 BSR 张量。

def to_sparse_bsr(blocksize, dense_dim):
    """
    Convert a tensor to a block sparse row (BSR) storage format of given
    blocksize.  If the :attr:`self` is strided, then the number of dense
    dimensions could be specified, and a hybrid BSR tensor will be
    created, with `dense_dim` dense dimensions and `self.dim() - 2 -
    dense_dim` batch dimension.

    Args:

        blocksize (list, tuple, :class:`torch.Size`, optional): Block size
          of the resulting BSR tensor. A block size must be a tuple of
          length two such that its items evenly divide the two sparse
          dimensions.

        dense_dim (int, optional): Number of dense dimensions of the
          resulting BSR tensor.  This argument should be used only if
          :attr:`self` is a strided tensor, and must be a value between 0
          and dimension of :attr:`self` tensor minus two.

    Example::

        >>> dense = torch.randn(10, 10)
    """
    # 将稠密张量转换为稀疏的 CSR 格式
    >>> sparse = dense.to_sparse_csr()
    
    # 基于稀疏的 CSR 格式张量创建稀疏的 BSR 格式张量，大小为 (5, 5)
    >>> sparse_bsr = sparse.to_sparse_bsr((5, 5))
    
    # 获取稀疏 BSR 格式张量的列索引
    >>> sparse_bsr.col_indices()
    tensor([0, 1, 0, 1])
    
    # 创建一个形状为 (4, 3, 1) 的全零稠密张量
    >>> dense = torch.zeros(4, 3, 1)
    
    # 对张量进行切片操作，设置部分元素为 1
    >>> dense[0:2, 0] = dense[0:2, 2] = dense[2:4, 1] = 1
    
    # 将稠密张量转换为稀疏的 BSR 格式张量，使用块大小为 (2, 1)，以及设定填充值为 1
    >>> dense.to_sparse_bsr((2, 1), 1)
    tensor(crow_indices=tensor([0, 2, 3]),
           col_indices=tensor([0, 2, 1]),
           values=tensor([[[[1.]],
    
                           [[1.]]],
    
    
                          [[[1.]],
    
                           [[1.]]],
    
    
                          [[[1.]],
    
                           [[1.]]]]), size=(4, 3, 1), nnz=3,
           layout=torch.sparse_bsr)
# 转换为块稀疏列（BSC）存储格式的张量
to_sparse_bsc(blocksize, dense_dim) -> Tensor
"""
Convert a tensor to a block sparse column (BSC) storage format of
given blocksize.  If the :attr:`self` is strided, then the number of
dense dimensions could be specified, and a hybrid BSC tensor will be
created, with `dense_dim` dense dimensions and `self.dim() - 2 -
dense_dim` batch dimension.
"""

# blocksize 参数: 结果 BSC 张量的块大小，必须是长度为两的元组，其元素必须均匀地划分两个稀疏维度。
# dense_dim 参数: 结果 BSC 张量的密集维度数，仅当 self 是分步张量时使用，必须是 0 到 self 张量维度减二之间的值。
"""

to_mkldnn() -> Tensor
"""
Returns a copy of the tensor in ``torch.mkldnn`` layout.
"""

trace() -> Tensor
"""
See :func:`torch.trace`
"""

transpose(dim0, dim1) -> Tensor
"""
See :func:`torch.transpose`
"""

transpose_(dim0, dim1) -> Tensor
"""
In-place version of :meth:`~Tensor.transpose`
"""

triangular_solve(A, upper=True, transpose=False, unitriangular=False) -> (Tensor, Tensor)
"""
See :func:`torch.triangular_solve`
"""

tril(diagonal=0) -> Tensor
"""
See :func:`torch.tril`
"""

tril_(diagonal=0) -> Tensor
"""
In-place version of :meth:`~Tensor.tril`
"""

triu(diagonal=0) -> Tensor
"""
See :func:`torch.triu`
"""

triu_(diagonal=0) -> Tensor
"""
In-place version of :meth:`~Tensor.triu`
"""

true_divide(value) -> Tensor
"""
See :func:`torch.true_divide`
"""

true_divide_(value) -> Tensor
"""
In-place version of :meth:`~Tensor.true_divide_`
"""

trunc() -> Tensor
"""
See :func:`torch.trunc`
"""
# 为函数 "fix" 添加文档字符串
add_docstr_all(
    "fix",
    r"""
fix() -> Tensor

See :func:`torch.fix`.
""",
)

# 为函数 "trunc_" 添加文档字符串
add_docstr_all(
    "trunc_",
    r"""
trunc_() -> Tensor

In-place version of :meth:`~Tensor.trunc`
""",
)

# 为函数 "fix_" 添加文档字符串
add_docstr_all(
    "fix_",
    r"""
fix_() -> Tensor

In-place version of :meth:`~Tensor.fix`
""",
)

# 为函数 "type" 添加文档字符串
add_docstr_all(
    "type",
    r"""
type(dtype=None, non_blocking=False, **kwargs) -> str or Tensor
Returns the type if `dtype` is not provided, else casts this object to
the specified type.

If this is already of the correct type, no copy is performed and the
original object is returned.

Args:
    dtype (dtype or string): The desired type
    non_blocking (bool): If ``True``, and the source is in pinned memory
        and destination is on the GPU or vice versa, the copy is performed
        asynchronously with respect to the host. Otherwise, the argument
        has no effect.
    **kwargs: For compatibility, may contain the key ``async`` in place of
        the ``non_blocking`` argument. The ``async`` arg is deprecated.
""",
)

# 为函数 "type_as" 添加文档字符串
add_docstr_all(
    "type_as",
    r"""
type_as(tensor) -> Tensor

Returns this tensor cast to the type of the given tensor.

This is a no-op if the tensor is already of the correct type. This is
equivalent to ``self.type(tensor.type())``

Args:
    tensor (Tensor): the tensor which has the desired type
""",
)

# 为函数 "unfold" 添加文档字符串
add_docstr_all(
    "unfold",
    r"""
unfold(dimension, size, step) -> Tensor

Returns a view of the original tensor which contains all slices of size :attr:`size` from
:attr:`self` tensor in the dimension :attr:`dimension`.

Step between two slices is given by :attr:`step`.

If `sizedim` is the size of dimension :attr:`dimension` for :attr:`self`, the size of
dimension :attr:`dimension` in the returned tensor will be
`(sizedim - size) / step + 1`.

An additional dimension of size :attr:`size` is appended in the returned tensor.

Args:
    dimension (int): dimension in which unfolding happens
    size (int): the size of each slice that is unfolded
    step (int): the step between each slice

Example::

    >>> x = torch.arange(1., 8)
    >>> x
    tensor([ 1.,  2.,  3.,  4.,  5.,  6.,  7.])
    >>> x.unfold(0, 2, 1)
    tensor([[ 1.,  2.],
            [ 2.,  3.],
            [ 3.,  4.],
            [ 4.,  5.],
            [ 5.,  6.],
            [ 6.,  7.]])
    >>> x.unfold(0, 2, 2)
    tensor([[ 1.,  2.],
            [ 3.,  4.],
            [ 5.,  6.]])
""",
)

# 为函数 "uniform_" 添加文档字符串
add_docstr_all(
    "uniform_",
    r"""
uniform_(from=0, to=1, *, generator=None) -> Tensor

Fills :attr:`self` tensor with numbers sampled from the continuous uniform
distribution:

.. math::
    f(x) = \dfrac{1}{\text{to} - \text{from}}
""",
)

# 为函数 "unsqueeze" 添加文档字符串
add_docstr_all(
    "unsqueeze",
    r"""
unsqueeze(dim) -> Tensor

See :func:`torch.unsqueeze`
""",
)

# 为函数 "unsqueeze_" 添加文档字符串
add_docstr_all(
    "unsqueeze_",
    r"""
unsqueeze_(dim) -> Tensor

In-place version of :meth:`~Tensor.unsqueeze`
""",
)

# 为函数 "var" 添加文档字符串
add_docstr_all(
    "var",
    r"""
# 定义一个变量维度为零或更多，具有可选参数修正为1和是否保持维度不变的标志，返回一个张量
var(dim=None, *, correction=1, keepdim=False) -> Tensor

See :func:`torch.var`



# 给 "vdot" 函数添加文档字符串，描述其接受一个张量并返回另一个张量的点积
add_docstr_all(
    "vdot",
    r"""
vdot(other) -> Tensor

See :func:`torch.vdot`
""",
)


# 给 "view" 函数添加文档字符串，解释它返回一个与输入张量相同数据但形状不同的新张量
add_docstr_all(
    "view",
    r"""
view(*shape) -> Tensor

Returns a new tensor with the same data as the :attr:`self` tensor but of a
different :attr:`shape`.

The returned tensor shares the same data and must have the same number
of elements, but may have a different size. For a tensor to be viewed, the new
view size must be compatible with its original size and stride, i.e., each new
view dimension must either be a subspace of an original dimension, or only span
across original dimensions :math:`d, d+1, \dots, d+k` that satisfy the following
contiguity-like condition that :math:`\forall i = d, \dots, d+k-1`,

.. math::

  \text{stride}[i] = \text{stride}[i+1] \times \text{size}[i+1]

Otherwise, it will not be possible to view :attr:`self` tensor as :attr:`shape`
without copying it (e.g., via :meth:`contiguous`). When it is unclear whether a
:meth:`view` can be performed, it is advisable to use :meth:`reshape`, which
returns a view if the shapes are compatible, and copies (equivalent to calling
:meth:`contiguous`) otherwise.

Args:
    shape (torch.Size or int...): the desired size

Example::

    >>> x = torch.randn(4, 4)
    >>> x.size()
    torch.Size([4, 4])
    >>> y = x.view(16)
    >>> y.size()
    torch.Size([16])
    >>> z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
    >>> z.size()
    torch.Size([2, 8])

    >>> a = torch.randn(1, 2, 3, 4)
    >>> a.size()
    torch.Size([1, 2, 3, 4])
    >>> b = a.transpose(1, 2)  # Swaps 2nd and 3rd dimension
    >>> b.size()
    torch.Size([1, 3, 2, 4])
    >>> c = a.view(1, 3, 2, 4)  # Does not change tensor layout in memory
    >>> c.size()
    torch.Size([1, 3, 2, 4])
    >>> torch.equal(b, c)
    False


.. method:: view(dtype) -> Tensor
   :noindex:

Returns a new tensor with the same data as the :attr:`self` tensor but of a
different :attr:`dtype`.

If the element size of :attr:`dtype` is different than that of ``self.dtype``,
then the size of the last dimension of the output will be scaled
proportionally.  For instance, if :attr:`dtype` element size is twice that of
``self.dtype``, then each pair of elements in the last dimension of
:attr:`self` will be combined, and the size of the last dimension of the output
will be half that of :attr:`self`. If :attr:`dtype` element size is half that
of ``self.dtype``, then each element in the last dimension of :attr:`self` will
be split in two, and the size of the last dimension of the output will be
double that of :attr:`self`. For this to be possible, the following conditions
must be true:

    * ``self.dim()`` must be greater than 0.
    * ``self.stride(-1)`` must be 1.

Additionally, if the element size of :attr:`dtype` is greater than that of
``self.dtype``, the following conditions must be true as well:

""",
)
    * ``self.size(-1)`` must be divisible by the ratio between the element
      sizes of the dtypes.
    * ``self.storage_offset()`` must be divisible by the ratio between the
      element sizes of the dtypes.
    * The strides of all dimensions, except the last dimension, must be
      divisible by the ratio between the element sizes of the dtypes.
# 将张量自身视图调整为与另一个张量相同大小
# `self.view_as(other)` 等同于 `self.view(other.size())`
def view_as(other) -> Tensor:
    """
    View this tensor as the same size as :attr:`other`.
    
    Args:
        other (:class:`torch.Tensor`): The result tensor has the same size
            as :attr:`other`.
    """
    pass

# 将张量的单例维度扩展到更大的尺寸，返回一个新的视图
# `-1` 表示该维度的尺寸保持不变
# 张量也可以扩展到更多维度，新的维度将附加在前面。对于新维度，尺寸不能设置为 `-1`。
# 扩展张量不会分配新内存，而只是在现有张量上创建新视图，其中尺寸为一的维度通过设置 `stride` 为 0 扩展到更大尺寸。
def expand(*sizes) -> Tensor:
    """
    Returns a new view of the :attr:`self` tensor with singleton dimensions expanded
    to a larger size.
    
    Args:
        *sizes (int...): Sizes to expand the tensor to.
    """
    pass
# 计算此张量的行列式
def det() -> Tensor
"""
该函数设置张量的梯度保持标志，以便在反向传播时保留梯度。

当调用 backward() 计算张量的梯度时，如果调用 retain_grad()，则张量将保留其梯度值。这使得可以在后续的计算中访问和使用这些梯度值。

注意：
- 这是一个原地操作，不会返回新的张量对象。
"""



"""
此属性默认为 None，在第一次调用 backward() 时，如果计算了 self 的梯度，该属性将成为一个 Tensor。
之后的 backward() 调用将会累积（添加）梯度到该属性中。
"""
r"""
Stores names for each of this tensor's dimensions.

``names[idx]`` corresponds to the name of tensor dimension ``idx``.
Names are either a string if the dimension is named or ``None`` if the
dimension is unnamed.

Dimension names may contain characters or underscore. Furthermore, a dimension
name must be a valid Python variable name (i.e., does not start with underscore).

Tensors may not have two named dimensions with the same name.

.. warning::
    The named tensor API is experimental and subject to change.

"""
# 返回是否张量存储在IPU设备上，如果是返回True，否则返回False
"""

add_docstr_all(
    "is_xpu",
    r"""
    返回是否张量存储在XPU设备上，如果是返回True，否则返回False
    """,
)

add_docstr_all(
    "is_quantized",
    r"""
    返回是否张量被量化，如果是返回True，否则返回False
    """,
)

add_docstr_all(
    "is_meta",
    r"""
    返回是否张量是元张量，如果是返回True，否则返回False。元张量类似于普通张量，但不携带数据。
    """,
)

add_docstr_all(
    "is_mps",
    r"""
    返回是否张量存储在MPS设备上，如果是返回True，否则返回False
    """,
)

add_docstr_all(
    "is_sparse",
    r"""
    返回是否张量使用稀疏COO存储布局，如果是返回True，否则返回False
    """,
)

add_docstr_all(
    "is_sparse_csr",
    r"""
    返回是否张量使用稀疏CSR存储布局，如果是返回True，否则返回False
    """,
)

add_docstr_all(
    "device",
    r"""
    返回张量所在的torch.device对象
    """,
)

add_docstr_all(
    "ndim",
    r"""
    别名，等同于Tensor.dim()
    """,
)

add_docstr_all(
    "itemsize",
    r"""
    别名，等同于Tensor.element_size()
    """,
)

add_docstr_all(
    "nbytes",
    r"""
    如果张量不使用稀疏存储布局，则返回"view"元素的字节数。定义为Tensor.numel() * Tensor.element_size()
    """,
)

add_docstr_all(
    "T",
    r"""
    返回张量维度反转后的视图。

    如果x有n个维度，x.T 等同于 x.permute(n-1, n-2, ..., 0)。

    警告：
    在张量维度不为2时使用Tensor.T翻转形状已被弃用，将在未来版本中报错。考虑使用.attr('mT')来转置矩阵批次或x.permute(*torch.arange(x.ndim - 1, -1, -1))来反转张量的维度。
    """,
)

add_docstr_all(
    "H",
    r"""
    返回共轭转置后的矩阵（2-D张量）视图。

    对于复数矩阵，x.H 等同于 x.transpose(0, 1).conj()，对于实数矩阵，x.H 等同于 x.transpose(0, 1)。

    参见：
        .attr('mH')：也适用于矩阵批次的属性。
    """,
)

add_docstr_all(
    "mT",
    r"""
    返回最后两个维度转置后的张量视图。

    x.mT 等同于 x.transpose(-2, -1)。
    """,
)

add_docstr_all(
    "mH",
    r"""
    访问此属性等同于调用adjoint函数。
    """,
)

add_docstr_all(
    "adjoint",
    r"""
    adjoint() -> Tensor

    别名，等同于adjoint函数。
    """,
)

add_docstr_all(
    "real",
    r"""
    返回复杂值输入张量的实部值的新张量。返回的张量与self共享相同的底层存储。

    如果self是实数张量，则返回self。

    示例：
        >>> x=torch.randn(4, dtype=torch.cfloat)
        >>> x
    """,
)
    # 创建一个复数张量，每个元素包含实部和虚部
    tensor([(0.3100+0.3553j), (-0.5445-0.7896j), (-1.6492-0.0633j), (-0.0638-0.8119j)])
    
    # 提取张量 x 的实部部分，返回一个包含所有元素实部的张量
    >>> x.real
    tensor([ 0.3100, -0.5445, -1.6492, -0.0638])
# 定义函数 add_docstr_all，为指定函数或方法添加文档字符串
def add_docstr_all(
    # 函数名或方法名，需要添加文档字符串的目标
    "imag",
    # 多行文档字符串，解释函数 imag 的作用和用法
    r"""
    Returns a new tensor containing imaginary values of the :attr:`self` tensor.
    The returned tensor and :attr:`self` share the same underlying storage.
    
    .. warning::
        :func:`imag` is only supported for tensors with complex dtypes.
    
    Example::
        >>> x=torch.randn(4, dtype=torch.cfloat)
        >>> x
        tensor([(0.3100+0.3553j), (-0.5445-0.7896j), (-1.6492-0.0633j), (-0.0638-0.8119j)])
        >>> x.imag
        tensor([ 0.3553, -0.7896, -0.0633, -0.8119])
    
    """,
)

# 定义函数 add_docstr_all，为指定函数或方法添加文档字符串
def add_docstr_all(
    # 函数名或方法名，需要添加文档字符串的目标
    "as_subclass",
    # 多行文档字符串，解释函数 as_subclass 的作用和用法
    r"""
    as_subclass(cls) -> Tensor
    
    Makes a ``cls`` instance with the same data pointer as ``self``. Changes
    in the output mirror changes in ``self``, and the output stays attached
    to the autograd graph. ``cls`` must be a subclass of ``Tensor``.
    """,
)

# 定义函数 add_docstr_all，为指定函数或方法添加文档字符串
def add_docstr_all(
    # 函数名或方法名，需要添加文档字符串的目标
    "crow_indices",
    # 多行文档字符串，解释函数 crow_indices 的作用和用法
    r"""
    crow_indices() -> IntTensor
    
    Returns the tensor containing the compressed row indices of the :attr:`self`
    tensor when :attr:`self` is a sparse CSR tensor of layout ``sparse_csr``.
    The ``crow_indices`` tensor is strictly of shape (:attr:`self`.size(0) + 1)
    and of type ``int32`` or ``int64``. When using MKL routines such as sparse
    matrix multiplication, it is necessary to use ``int32`` indexing in order
    to avoid downcasting and potentially losing information.
    
    Example::
        >>> csr = torch.eye(5,5).to_sparse_csr()
        >>> csr.crow_indices()
        tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32)
    
    """,
)

# 定义函数 add_docstr_all，为指定函数或方法添加文档字符串
def add_docstr_all(
    # 函数名或方法名，需要添加文档字符串的目标
    "col_indices",
    # 多行文档字符串，解释函数 col_indices 的作用和用法
    r"""
    col_indices() -> IntTensor
    
    Returns the tensor containing the column indices of the :attr:`self`
    tensor when :attr:`self` is a sparse CSR tensor of layout ``sparse_csr``.
    The ``col_indices`` tensor is strictly of shape (:attr:`self`.nnz())
    and of type ``int32`` or ``int64``.  When using MKL routines such as sparse
    matrix multiplication, it is necessary to use ``int32`` indexing in order
    to avoid downcasting and potentially losing information.
    
    Example::
        >>> csr = torch.eye(5,5).to_sparse_csr()
        >>> csr.col_indices()
        tensor([0, 1, 2, 3, 4], dtype=torch.int32)
    
    """,
)

# 定义函数 add_docstr_all，为指定函数或方法添加文档字符串
def add_docstr_all(
    # 函数名或方法名，需要添加文档字符串的目标
    "to_padded_tensor",
    # 多行文档字符串，解释函数 to_padded_tensor 的作用和用法
    r"""
    to_padded_tensor(padding, output_size=None) -> Tensor
    See :func:`to_padded_tensor`
    """,
)
```