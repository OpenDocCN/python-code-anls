# `.\pytorch\torch\_torch_docs.py`

```py
# mypy: allow-untyped-defs
"""Adds docstrings to functions defined in the torch._C module."""

import re  # 导入正则表达式模块

import torch._C  # 导入 torch._C 模块
from torch._C import _add_docstr as add_docstr  # 导入 _add_docstr 函数，并起别名为 add_docstr


def parse_kwargs(desc):
    r"""Map a description of args to a dictionary of {argname: description}.

    Input:
        ('    weight (Tensor): a weight tensor\n' +
         '        Some optional description')
    Output: {
        'weight': \
        'weight (Tensor): a weight tensor\n        Some optional description'
    }
    """
    # 使用正则表达式分割描述字符串，形成关键字参数列表
    regx = re.compile(r"\n\s{4}(?!\s)")
    kwargs = [section.strip() for section in regx.split(desc)]
    kwargs = [section for section in kwargs if len(section) > 0]
    return {desc.split(" ")[0]: desc for desc in kwargs}  # 返回关键字参数字典


def merge_dicts(*dicts):
    """Merge dictionaries into a single dictionary."""
    return {x: d[x] for d in dicts for x in d}  # 合并多个字典为一个字典


common_args = parse_kwargs(
    """
    input (Tensor): the input tensor.
    generator (:class:`torch.Generator`, optional): a pseudorandom number generator for sampling
    out (Tensor, optional): the output tensor.
    memory_format (:class:`torch.memory_format`, optional): the desired memory format of
        returned tensor. Default: ``torch.preserve_format``.
"""
)

reduceops_common_args = merge_dicts(
    common_args,
    parse_kwargs(
        """
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        If specified, the input tensor is casted to :attr:`dtype` before the operation
        is performed. This is useful for preventing data type overflows. Default: None.
    keepdim (bool): whether the output tensor has :attr:`dim` retained or not.
"""
    ),
)

multi_dim_common = merge_dicts(
    reduceops_common_args,
    parse_kwargs(
        """
    dim (int or tuple of ints): the dimension or dimensions to reduce.
"""
    ),
    {
        "keepdim_details": """
If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in the
output tensor having 1 (or ``len(dim)``) fewer dimension(s).
"""
    },
    {
        "opt_dim": """
    dim (int or tuple of ints, optional): the dimension or dimensions to reduce.
        If ``None``, all dimensions are reduced.
"""
    },
)

single_dim_common = merge_dicts(
    reduceops_common_args,
    parse_kwargs(
        """
    dim (int): the dimension to reduce.
"""
    ),
    {
        "keepdim_details": """If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension :attr:`dim` where it is of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in
the output tensor having 1 fewer dimension than :attr:`input`."""
    },
)

factory_common_args = merge_dicts(
    common_args,
    parse_kwargs(
        """
    # 指定返回张量的数据类型，默认使用全局默认值（参见 torch.set_default_dtype）
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
    # 指定返回张量的布局，默认为 torch.strided
    layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
        Default: ``torch.strided``.
    # 指定返回张量的设备，默认情况下根据张量类型选择当前设备
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, uses the current device for the default tensor type
        (see :func:`torch.set_default_device`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.
    # 指定是否记录返回张量上的自动求导操作，默认为 False
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.
    # 如果设置为 True，则返回的张量将分配在固定内存中，仅适用于 CPU 张量，默认为 False
    pin_memory (bool, optional): If set, returned tensor would be allocated in
        the pinned memory. Works only for CPU tensors. Default: ``False``.
    # 指定返回张量的内存格式，默认为 torch.contiguous_format
    memory_format (:class:`torch.memory_format`, optional): the desired memory format of
        returned Tensor. Default: ``torch.contiguous_format``.
    # 指定是否检查稀疏张量的不变性，默认根据 torch.sparse.check_sparse_tensor_invariants.is_enabled 返回值，最初为 False
    check_invariants (bool, optional): If sparse tensor invariants are checked.
        Default: as returned by :func:`torch.sparse.check_sparse_tensor_invariants.is_enabled`,
        initially False.
"""
),
{
    "sparse_factory_device_note": """\
.. note::

   如果未指定 ``device`` 参数，则给定的 :attr:`values` 和索引张量的设备必须匹配。
   但是，如果指定了该参数，则输入张量将转换为给定的设备，并且由此确定构建的稀疏张量的设备。"""
},
)

factory_like_common_args = parse_kwargs(
"""
input (Tensor): 输入的大小将决定输出张量的大小。
layout (:class:`torch.layout`, 可选): 返回张量的期望布局。默认情况下，如果 ``None``，则默认为 :attr:`input` 的布局。
dtype (:class:`torch.dtype`, 可选): 返回张量的期望数据类型。默认情况下，如果 ``None``，则默认为 :attr:`input` 的数据类型。
device (:class:`torch.device`, 可选): 返回张量的期望设备。默认情况下，如果 ``None``，则默认为 :attr:`input` 的设备。
requires_grad (bool, 可选): 是否应在返回的张量上记录 autograd 操作。默认为 ``False``。
pin_memory (bool, 可选): 如果设置，则返回的张量将在固定内存中分配。仅适用于 CPU 张量。默认为 ``False``。
memory_format (:class:`torch.memory_format`, 可选): 返回张量的期望内存格式。默认为 ``torch.preserve_format``。"""
)

factory_data_common_args = parse_kwargs(
"""
data (array_like): 张量的初始数据。可以是列表、元组、NumPy ``ndarray``、标量和其他类型。
dtype (:class:`torch.dtype`, 可选): 返回张量的期望数据类型。默认情况下，从 :attr:`data` 推断数据类型。
device (:class:`torch.device`, 可选): 返回张量的期望设备。默认情况下，使用当前设备作为默认张量类型的设备
(参见 :func:`torch.set_default_device`)。对于 CPU 张量类型， :attr:`device` 将是 CPU；
对于 CUDA 张量类型，将是当前 CUDA 设备。
requires_grad (bool, 可选): 是否应在返回的张量上记录 autograd 操作。默认为 ``False``。
pin_memory (bool, 可选): 如果设置，则返回的张量将在固定内存中分配。仅适用于 CPU 张量。默认为 ``False``。"""
)

tf32_notes = {
    "tf32_note": """此运算符支持 :ref:`TensorFloat32<tf32_on_ampere>`。"""
}

rocm_fp16_notes = {
    "rocm_fp16_note": """在某些 ROCm 设备上，当使用 float16 输入时，此模块将使用 \
:ref:`不同的精度<fp16_on_mi200>` 进行反向传播。"""
}

reproducibility_notes = {
    "forward_reproducibility_note": """当在 CUDA 设备上给定张量时，此操作可能在行为上表现为非确定性。
请参阅 :doc:`/notes/randomness` 获取更多信息。""",
    "backward_reproducibility_note": """当在 CUDA 设备上给定张量时，此操作可能产生非确定性梯度。
add_docstr(
    torch.abs,
    r"""
abs(input, *, out=None) -> Tensor

计算 :attr:`input` 中每个元素的绝对值。

.. math::
    \text{out}_{i} = |\text{input}_{i}
"""
    + r"""
Args:
    {input}

Keyword args:
    {out}

Example::

    >>> torch.abs(torch.tensor([-1, -2, 3]))
    tensor([ 1,  2,  3])
""".format(
        **common_args
    ),
)

add_docstr(
    torch.absolute,
    r"""
absolute(input, *, out=None) -> Tensor

:func:`torch.abs` 的别名
""",
)

add_docstr(
    torch.acos,
    r"""
acos(input, *, out=None) -> Tensor

计算 :attr:`input` 中每个元素的反余弦值。

.. math::
    \text{out}_{i} = \cos^{-1}(\text{input}_{i})
"""
    + r"""
Args:
    {input}

Keyword args:
    {out}

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.3348, -0.5889,  0.2005, -0.1584])
    >>> torch.acos(a)
    tensor([ 1.2294,  2.2004,  1.3690,  1.7298])
""".format(
        **common_args
    ),
)

add_docstr(
    torch.arccos,
    r"""
arccos(input, *, out=None) -> Tensor

:func:`torch.acos` 的别名
""",
)

add_docstr(
    torch.acosh,
    r"""
acosh(input, *, out=None) -> Tensor

返回一个新的张量，其中包含 :attr:`input` 元素的反双曲余弦值。

.. math::
    \text{out}_{i} = \cosh^{-1}(\text{input}_{i})

注意:
    反双曲余弦函数的定义域为 `[1, inf)`，超出此范围的值将映射为 ``NaN``，
    但对于 `+ INF`，输出将映射为 `+ INF`。
"""
    + r"""
Args:
    {input}

Keyword arguments:
    {out}

Example::

    >>> a = torch.randn(4).uniform_(1, 2)
    >>> a
    tensor([ 1.3192, 1.9915, 1.9674, 1.7151 ])
    >>> torch.acosh(a)
    tensor([ 0.7791, 1.3120, 1.2979, 1.1341 ])
""".format(
        **common_args
    ),
)

add_docstr(
    torch.arccosh,
    r"""
arccosh(input, *, out=None) -> Tensor

:func:`torch.acosh` 的别名
""",
)

add_docstr(
    torch.index_add,
    r"""
index_add(input, dim, index, source, *, alpha=1, out=None) -> Tensor

参见 :meth:`~Tensor.index_add_` 的功能描述。
""",
)

add_docstr(
    torch.index_copy,
    r"""
index_copy(input, dim, index, source, *, out=None) -> Tensor
# 为 torch.index_reduce 函数添加文档字符串
add_docstr(
    torch.index_reduce,
    r"""
index_reduce(input, dim, index, source, reduce, *, include_self=True, out=None) -> Tensor

See :meth:`~Tensor.index_reduce_` for function description.
""",
)

# 为 torch.add 函数添加文档字符串
add_docstr(
    torch.add,
    r"""
add(input, other, *, alpha=1, out=None) -> Tensor

Adds :attr:`other`, scaled by :attr:`alpha`, to :attr:`input`.

.. math::
    \text{{out}}_i = \text{{input}}_i + \text{{alpha}} \times \text{{other}}_i

Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`,
:ref:`type promotion <type-promotion-doc>`, and integer, float, and complex inputs.

Args:
    input (Tensor): the first operand
    other (Tensor or Number): the tensor or number to add to :attr:`input`.

Keyword arguments:
    alpha (Number): the multiplier for :attr:`other`.
    out (Tensor, optional): the output tensor.

Examples::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.0202,  1.0985,  1.3506, -0.6056])
    >>> torch.add(a, 20)
    tensor([ 20.0202,  21.0985,  21.3506,  19.3944])

    >>> b = torch.randn(4)
    >>> b
    tensor([-0.9732, -0.3497,  0.6245,  0.4022])
    >>> c = torch.randn(4, 1)
    >>> c
    tensor([[ 0.3743],
            [-1.7724],
            [-0.5811],
            [-0.8017]])
    >>> torch.add(b, c, alpha=10)
    tensor([[  2.7695,   3.3930,   4.3672,   4.1450],
            [-18.6971, -18.0736, -17.0994, -17.3216],
            [ -6.7845,  -6.1610,  -5.1868,  -5.4090],
            [ -8.9902,  -8.3667,  -7.3925,  -7.6147]])
""",
)

# 为 torch.addbmm 函数添加文档字符串
add_docstr(
    torch.addbmm,
    r"""
addbmm(input, batch1, batch2, *, beta=1, alpha=1, out=None) -> Tensor

Performs a batch matrix-matrix product of matrices stored
in :attr:`batch1` and :attr:`batch2`,
with a reduced add step (all matrix multiplications get accumulated
along the first dimension).
:attr:`input` is added to the final result.

:attr:`batch1` and :attr:`batch2` must be 3-D tensors each containing the
same number of matrices.

If :attr:`batch1` is a :math:`(b \times n \times m)` tensor, :attr:`batch2` is a
:math:`(b \times m \times p)` tensor, :attr:`input` must be
:ref:`broadcastable <broadcasting-semantics>` with a :math:`(n \times p)` tensor
and :attr:`out` will be a :math:`(n \times p)` tensor.

.. math::
    out = \beta\ \text{input} + \alpha\ (\sum_{i=0}^{b-1} \text{batch1}_i \mathbin{@} \text{batch2}_i)

If :attr:`beta` is 0, then :attr:`input` will be ignored, and `nan` and `inf` in
it will not be propagated.

Args:
    batch1 (Tensor): the first batch of matrices to be multiplied
    batch2 (Tensor): the second batch of matrices to be multiplied

Keyword args:
    beta (Number, optional): multiplier for :attr:`input` (:math:`\beta`)
    alpha (Number, optional): multiplier for the batch multiplication (:math:`\alpha`)
    out (Tensor, optional): the output tensor.
    alpha (Number, optional): multiplier for `batch1 @ batch2` (:math:`\alpha`)
    {out}
# 定义函数 addcdiv，实现 tensor1 与 tensor2 的逐元素除法，乘以标量 value 后加到 input 上
addcdiv(input, tensor1, tensor2, *, value=1, out=None) -> Tensor

# 函数功能说明和警告信息
Performs the element-wise division of :attr:`tensor1` by :attr:`tensor2`,
multiplies the result by the scalar :attr:`value` and adds it to :attr:`input`.

.. warning::
    Integer division with addcdiv is no longer supported, and in a future
    release addcdiv will perform a true division of tensor1 and tensor2.
    The historic addcdiv behavior can be implemented as
    (input + value * torch.trunc(tensor1 / tensor2)).to(input.dtype)
    for integer inputs and as (input + value * tensor1 / tensor2) for float inputs.
    The future addcdiv behavior is just the latter implementation:
    (input + value * tensor1 / tensor2), for all dtypes.

# 数学表达式说明
.. math::
    \text{out}_i = \text{input}_i + \text{value} \times \frac{\text{tensor1}_i}{\text{tensor2}_i}

# 参数要求和使用示例
The shapes of :attr:`input`, :attr:`tensor1`, and :attr:`tensor2` must be
:ref:`broadcastable <broadcasting-semantics>`.

For inputs of type `FloatTensor` or `DoubleTensor`, :attr:`value` must be
a real number, otherwise an integer.

Args:
    input (Tensor): the tensor to be added
    tensor1 (Tensor): the numerator tensor
    tensor2 (Tensor): the denominator tensor

Keyword args:
    value (Number, optional): multiplier for :math:`\text{{tensor1}} / \text{{tensor2}}`
    {out}

Example::

    >>> t = torch.randn(1, 3)
    >>> t1 = torch.randn(3, 1)
    >>> t2 = torch.randn(1, 3)
    >>> torch.addcdiv(t, t1, t2, value=0.1)
    tensor([[-0.2312, -3.6496,  0.1312],
            [-1.0428,  3.4292, -0.1030],
            [-0.5369, -0.9829,  0.0430]])
    # 创建一个形状为(1, 3)的张量t，其中的值是从标准正态分布中随机采样得到的
    t = torch.randn(1, 3)
    # 创建一个形状为(3, 1)的张量t1，其中的值是从标准正态分布中随机采样得到的
    t1 = torch.randn(3, 1)
    # 创建一个形状为(1, 3)的张量t2，其中的值是从标准正态分布中随机采样得到的
    t2 = torch.randn(1, 3)
    # 对张量t执行按元素相乘再相加的操作，结果存储在t中，其中t1和t2是输入张量，value=0.1是缩放因子
    torch.addcmul(t, t1, t2, value=0.1)
    # 打印结果张量，展示按元素相乘再相加后的值
    tensor([[-0.8635, -0.6391,  1.6174],
            [-0.7617, -0.5879,  1.7388],
            [-0.8353, -0.6249,  1.6511]])
add_docstr(
    torch.addmm,
    r"""
    addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None) -> Tensor
    
    Performs a matrix multiplication of the matrices :attr:`mat1` and :attr:`mat2`.
    The matrix :attr:`input` is added to the final result.
    
    If :attr:`mat1` is a :math:`(n \times m)` tensor, :attr:`mat2` is a
    :math:`(m \times p)` tensor, then :attr:`input` must be
    :ref:`broadcastable <broadcasting-semantics>` with a :math:`(n \times p)` tensor
    and :attr:`out` will be a :math:`(n \times p)` tensor.
    
    :attr:`alpha` and :attr:`beta` are scaling factors on matrix-vector product between
    :attr:`mat1` and :attr:`mat2` and the added matrix :attr:`input` respectively.
    
    .. math::
        \text{out} = \beta\ \text{input} + \alpha\ (\text{mat1}_i \mathbin{@} \text{mat2}_i)
    
    If :attr:`beta` is 0, then :attr:`input` will be ignored, and `nan` and `inf` in
    it will not be propagated.
    
    For inputs of type `FloatTensor` or `DoubleTensor`, arguments :attr:`beta` and
    :attr:`alpha` must be real numbers, otherwise they should be integers.
    
    This operation has support for arguments with :ref:`sparse layouts<sparse-docs>`. If
    :attr:`input` is sparse the result will have the same layout and if :attr:`out`
    is provided it must have the same layout as :attr:`input`.
    
    {sparse_beta_warning}
    
    {tf32_note}
    
    {rocm_fp16_note}
    
    Args:
        input (Tensor): matrix to be added
        mat1 (Tensor): the first matrix to be matrix multiplied
        mat2 (Tensor): the second matrix to be matrix multiplied
    
    Keyword args:
        beta (Number, optional): multiplier for :attr:`input` (:math:`\beta`)
        alpha (Number, optional): multiplier for :math:`mat1 @ mat2` (:math:`\alpha`)
        {out}
    
    Example::
    
        >>> M = torch.randn(2, 3)
        >>> mat1 = torch.randn(2, 3)
        >>> mat2 = torch.randn(3, 3)
        >>> torch.addmm(M, mat1, mat2)
        tensor([[-4.8716,  1.4671, -1.3746],
                [ 0.7573, -3.9555, -2.8681]])
    """.format(
        **common_args, **tf32_notes, **rocm_fp16_notes, **sparse_support_notes
    ),
)

add_docstr(
    torch.adjoint,
    r"""
    adjoint(Tensor) -> Tensor
    
    Returns a view of the tensor conjugated and with the last two dimensions transposed.
    
    ``x.adjoint()`` is equivalent to ``x.transpose(-2, -1).conj()`` for complex tensors and
    to ``x.transpose(-2, -1)`` for real tensors.
    
    Example::
        >>> x = torch.arange(4, dtype=torch.float)
        >>> A = torch.complex(x, x).reshape(2, 2)
        >>> A
        tensor([[0.+0.j, 1.+1.j],
                [2.+2.j, 3.+3.j]])
        >>> A.adjoint()
        tensor([[0.-0.j, 2.-2.j],
                [1.-1.j, 3.-3.j]])
        >>> (A.adjoint() == A.mH).all()
        tensor(True)
    """,
)

add_docstr(
    torch.sspaddmm,
    r"""
    sspaddmm(input, mat1, mat2, *, beta=1, alpha=1, out=None) -> Tensor
    
    Matrix multiplies a sparse tensor :attr:`mat1` with a dense tensor
    :attr:`mat2`, then adds the sparse tensor :attr:`input` to the result.
    
    Note: This function is equivalent to :func:`torch.addmm`, except
    :attr:`input` and :attr:`mat1` are sparse.
    """
)
add_docstr(
    torch.smm,
    r"""
smm(input, mat) -> Tensor

Performs a matrix multiplication of the sparse matrix :attr:`input`
with the dense matrix :attr:`mat`.

Args:
    input (Tensor): a sparse matrix to be matrix multiplied
    mat (Tensor): a dense matrix to be matrix multiplied
""",
)

add_docstr(
    torch.addmv,
    r"""
addmv(input, mat, vec, *, beta=1, alpha=1, out=None) -> Tensor

Performs a matrix-vector product of the matrix :attr:`mat` and
the vector :attr:`vec`.
The vector :attr:`input` is added to the final result.

If :attr:`mat` is a :math:`(n \times m)` tensor, :attr:`vec` is a 1-D tensor of
size `m`, then :attr:`input` must be
:ref:`broadcastable <broadcasting-semantics>` with a 1-D tensor of size `n` and
:attr:`out` will be 1-D tensor of size `n`.

:attr:`alpha` and :attr:`beta` are scaling factors on matrix-vector product between
:attr:`mat` and :attr:`vec` and the added tensor :attr:`input` respectively.

.. math::
    \text{out} = \beta\ \text{input} + \alpha\ (\text{mat} \mathbin{@} \text{vec})

If :attr:`beta` is 0, then :attr:`input` will be ignored, and `nan` and `inf` in
it will not be propagated.

For inputs of type `FloatTensor` or `DoubleTensor`, arguments :attr:`beta` and
:attr:`alpha` must be real numbers, otherwise they should be integers.

Args:
    input (Tensor): vector to be added
    mat (Tensor): matrix to be matrix multiplied
    vec (Tensor): vector to be matrix multiplied

Keyword args:
    beta (Number, optional): multiplier for :attr:`input` (:math:`\beta`)
    alpha (Number, optional): multiplier for :math:`mat @ vec` (:math:`\alpha`)
    {out}

Example::

    >>> M = torch.randn(2)
    >>> mat = torch.randn(2, 3)
    >>> vec = torch.randn(3)
    >>> torch.addmv(M, mat, vec)
    tensor([-0.3768, -5.5565])
""".format(
        **common_args
    ),
)

add_docstr(
    torch.addr,
    r"""
addr(input, vec1, vec2, *, beta=1, alpha=1, out=None) -> Tensor

Performs the outer-product of vectors :attr:`vec1` and :attr:`vec2`
and adds it to the matrix :attr:`input`.

Optional values :attr:`beta` and :attr:`alpha` are scaling factors on the
outer product between :attr:`vec1` and :attr:`vec2` and the added matrix
:attr:`input` respectively.

.. math::
    \text{out} = \beta\ \text{input} + \alpha\ (\text{vec1} \otimes \text{vec2})

If :attr:`beta` is 0, then :attr:`input` will be ignored, and `nan` and `inf` in
it will not be propagated.

If :attr:`vec1` is a vector of size `n` and :attr:`vec2` is a vector
of size `m`, then :attr:`input` must be a matrix of size `(n, m)`.

Args:
    input (Tensor): matrix to be added to
    vec1 (Tensor): first vector for outer product
    vec2 (Tensor): second vector for outer product

Keyword args:
    beta (Number, optional): multiplier for :attr:`input` (:math:`\beta`)
    alpha (Number, optional): multiplier for outer product (:math:`\alpha`)
    {out}
"""
    + r"""
Example::

    >>> M = torch.randn(2, 3)
    >>> v1 = torch.randn(2)
    >>> v2 = torch.randn(3)
    >>> torch.addr(M, v1, v2)
    tensor([[ 1.0635, -0.3917, -1.5572],
            [-1.3012,  0.5706,  0.5163]])
""".format(
        **common_args
    ),
)
# 为 torch.allclose 函数添加文档字符串和注释
add_docstr(
    torch.allclose,
    r"""
allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False) -> bool

This function checks if :attr:`input` and :attr:`other` satisfy the condition:

.. math::
    \lvert \text{input} - \text{other} \rvert \leq \texttt{atol} + \texttt{rtol} \times \lvert \text{other} \rvert

elementwise, for all elements of :attr:`input` and :attr:`other`. The behaviour of this function is analogous to
`numpy.allclose <https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html>`_

Args:
    input (Tensor): first tensor to compare
    other (Tensor): second tensor to compare
    atol (float, optional): absolute tolerance. Default: 1e-08
    rtol (float, optional): relative tolerance. Default: 1e-05
    equal_nan (bool, optional): if ``True``, then two ``NaN`` s will be considered equal. Default: ``False``

Example::

    >>> torch.allclose(torch.tensor([10000., 1e-07]), torch.tensor([10000.1, 1e-08]))
    False
    >>> torch.allclose(torch.tensor([10000., 1e-08]), torch.tensor([10000.1, 1e-09]))
    True
    >>> torch.allclose(torch.tensor([1.0, float('nan')]), torch.tensor([1.0, float('nan')]))
    False
    >>> torch.allclose(torch.tensor([1.0, float('nan')]), torch.tensor([1.0, float('nan')]), equal_nan=True)
    True
"""
)

# 为 torch.all 函数添加文档字符串和注释
add_docstr(
    torch.all,
    r"""
all(input) -> Tensor

Tests if all elements in :attr:`input` evaluate to `True`.

.. note:: This function matches the behaviour of NumPy in returning
          output of dtype `bool` for all supported dtypes except `uint8`.
          For `uint8` the dtype of output is `uint8` itself.

Example::

    >>> a = torch.rand(1, 2).bool()
    >>> a
    tensor([[False, True]], dtype=torch.bool)
    >>> torch.all(a)
    tensor(False, dtype=torch.bool)
    >>> a = torch.arange(0, 3)
    >>> a
    tensor([0, 1, 2])
    >>> torch.all(a)
    tensor(False)

.. function:: all(input, dim, keepdim=False, *, out=None) -> Tensor
   :noindex:

For each row of :attr:`input` in the given dimension :attr:`dim`,
returns `True` if all elements in the row evaluate to `True` and `False` otherwise.

Args:
    input (Tensor): the input tensor
    dim (int): the dimension to reduce. If None, reduce all dimensions.
    keepdim (bool, optional): whether the output tensor has dim retained or not. Default: False

Keyword args:
    out (Tensor, optional): the output tensor. If not provided, a new tensor is created.

"""
)
# 创建一个随机的布尔类型的张量，形状为 (4, 2)
a = torch.rand(4, 2).bool()

# 显示张量 a 的内容
a
tensor([[True, True],
        [True, False],
        [True, True],
        [True, True]], dtype=torch.bool)

# 对张量 a 按行进行逻辑全局运算，返回每行是否全为 True 的结果
torch.all(a, dim=1)
tensor([ True, False,  True,  True], dtype=torch.bool)

# 对张量 a 按列进行逻辑全局运算，返回每列是否全为 True 的结果
torch.all(a, dim=0)
tensor([ True, False], dtype=torch.bool)
    "overlapped" (with multiple indices referring to the same element in
    memory) its behavior is undefined.
Args:
    {input}
    size (tuple or ints): the shape of the output tensor
    stride (tuple or ints): the stride of the output tensor
    storage_offset (int, optional): the offset in the underlying storage of the output tensor.
        If ``None``, the storage_offset of the output tensor will match the input tensor.

Example::

    >>> x = torch.randn(3, 3)
    >>> x
    tensor([[ 0.9039,  0.6291,  1.0795],
            [ 0.1586,  2.1939, -0.4900],
            [-0.1909, -0.7503,  1.9355]])
    >>> t = torch.as_strided(x, (2, 2), (1, 2))
    >>> t
    tensor([[0.9039, 1.0795],
            [0.6291, 0.1586]])
    >>> t = torch.as_strided(x, (2, 2), (1, 2), 1)
    tensor([[0.6291, 0.1586],
            [1.0795, 2.1939]])
"""
).format(
    **common_args
),

add_docstr(
    torch.as_tensor,
    r"""
as_tensor(data, dtype=None, device=None) -> Tensor

Converts :attr:`data` into a tensor, sharing data and preserving autograd
history if possible.

If :attr:`data` is already a tensor with the requested dtype and device
then :attr:`data` itself is returned, but if :attr:`data` is a
tensor with a different dtype or device then it's copied as if using
`data.to(dtype=dtype, device=device)`.

If :attr:`data` is a NumPy array (an ndarray) with the same dtype and device then a
tensor is constructed using :func:`torch.from_numpy`.

.. seealso::

    :func:`torch.tensor` never shares its data and creates a new "leaf tensor" (see :doc:`/notes/autograd`).


Args:
    {data}
    {dtype}
    device (:class:`torch.device`, optional): the device of the constructed tensor. If None and data is a tensor
        then the device of data is used. If None and data is not a tensor then
        the result tensor is constructed on the current device.


Example::

    >>> a = numpy.array([1, 2, 3])
    >>> t = torch.as_tensor(a)
    >>> t
    tensor([ 1,  2,  3])
    >>> t[0] = -1
    >>> a
    array([-1,  2,  3])

    >>> a = numpy.array([1, 2, 3])
    >>> t = torch.as_tensor(a, device=torch.device('cuda'))
    >>> t
    tensor([ 1,  2,  3])
    >>> t[0] = -1
    >>> a
    array([1,  2,  3])
""".format(
    **factory_data_common_args
),

add_docstr(
    torch.asin,
    r"""
asin(input, *, out=None) -> Tensor

Returns a new tensor with the arcsine of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \sin^{-1}(\text{input}_{i})
"""
    + r"""
Args:
    {input}

Keyword args:
    {out}

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-0.5962,  1.4985, -0.4396,  1.4525])
    >>> torch.asin(a)
    tensor([-0.6387,     nan, -0.4552,     nan])
""".format(
    **common_args
),

add_docstr(
    torch.arcsin,
    r"""
arcsin(input, *, out=None) -> Tensor

Alias for :func:`torch.asin`.
""",
)

add_docstr(
    torch.asinh,
    r"""
asinh(input, *, out=None) -> Tensor

Returns a new tensor with the inverse hyperbolic sine of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \sinh^{-1}(\text{input}_{i})
"""
    + r"""
Args:

    {input}
    {out}
    # 定义一个名为 `process_data` 的函数，接收一个参数 `data`
    def process_data(data):
        # 使用列表推导式从参数 `data` 中获取每个元素的长度，生成一个长度列表
        lengths = [len(x) for x in data]
        # 将生成的长度列表按照从大到小排序
        lengths.sort(reverse=True)
        # 返回排序后的长度列表作为函数的结果
        return lengths
# 将给定的参数解包到字符串模板中，并为给定的torch函数添加文档字符串
add_docstr(
    torch.asarray,
    r"""
    asarray(data, *, out=None) -> Tensor

    将输入数据转换为张量（Tensor）。

    Args:
        data (array_like): 需要转换为张量的输入数据

    Keyword args:
        {out}

    Example::

        >>> a = [1, 2, 3]
        >>> torch.asarray(a)
        tensor([1, 2, 3])
    """.format(
        **common_args
    ),
)
# 将输入对象转换为张量（tensor）。

# 参数`obj`可以是以下之一：
# 1. 一个张量（tensor）
# 2. 一个 NumPy 数组或标量
# 3. 一个 DLPack capsule
# 4. 实现了 Python 缓冲协议的对象
# 5. 一个标量
# 6. 一系列标量

# 当`obj`是张量、NumPy 数组或 DLPack capsule 时，默认情况下返回的张量不需要梯度，
# 具有与`obj`相同的数据类型、在相同的设备上，并且与其共享内存。这些属性可以通过
# `dtype`、`device`、`copy` 和 `requires_grad` 关键字参数进行控制。如果返回的
# 张量具有不同的数据类型、在不同的设备上或需要复制，则不会与`obj`共享内存。如果
# `requires_grad` 设为 `True`，则返回的张量将需要梯度；如果`obj`也是具有自动求导
# 历史的张量，则返回的张量将具有相同的历史记录。

# 当`obj`不是张量、NumPy 数组或 DLPack capsule，但实现了 Python 缓冲协议时，
# 缓冲区将根据传递给`dtype`关键字参数的数据类型大小进行解释。（如果没有传递数据类型，
# 则使用默认的浮点数据类型。）返回的张量将具有指定的数据类型（如果未指定则使用默认的
# 浮点数据类型），并且默认在 CPU 设备上并与缓冲区共享内存。

# 当`obj`是 NumPy 标量时，返回的张量将是一个在 CPU 上的零维张量，不共享内存
# （即`copy=True`）。默认情况下，数据类型将是对应于 NumPy 标量数据类型的 PyTorch 数据类型。

# 当`obj`既不是上述情况，而是标量或标量序列时，返回的张量将默认从标量值中推断其数据类型，
# 在当前默认设备上，并且不共享内存。

# 参见：
# :func:`torch.tensor` - 创建一个总是从输入对象复制数据的张量。
# :func:`torch.from_numpy` - 创建一个总是从 NumPy 数组共享内存的张量。
# :func:`torch.frombuffer` - 创建一个总是从实现了缓冲协议的对象共享内存的张量。
# :func:`torch.from_dlpack` - 创建一个总是从 DLPack capsules 共享内存的张量。

def asarray(obj, *, dtype=None, device=None, copy=None, requires_grad=False) -> Tensor:
    # obj (object): 一个张量、NumPy 数组、DLPack Capsule、实现 Python 缓冲协议的对象、标量或标量序列。
    # dtype (:class:`torch.dtype`, optional): 返回张量的数据类型。默认值：`None`，导致从`obj`推断返回张量的数据类型。
    copy (bool, optional): controls whether the returned tensor shares memory with :attr:`obj`.
           Default: ``None``, which causes the returned tensor to share memory with :attr:`obj`
           whenever possible. If ``True`` then the returned tensor does not share its memory.
           If ``False`` then the returned tensor shares its memory with :attr:`obj` and an
           error is thrown if it cannot.
    device (:class:`torch.device`, optional): the device of the returned tensor.
           Default: ``None``, which causes the device of :attr:`obj` to be used. Or, if
           :attr:`obj` is a Python sequence, the current default device will be used.
    requires_grad (bool, optional): whether the returned tensor requires grad.
           Default: ``False``, which causes the returned tensor not to require a gradient.
           If ``True``, then the returned tensor will require a gradient, and if :attr:`obj`
           is also a tensor with an autograd history then the returned tensor will have
           the same history.


# 控制返回的张量是否与 :attr:`obj` 共享内存。
# 默认为 ``None``，这会使返回的张量在可能时与 :attr:`obj` 共享内存。
# 如果为 ``True``，则返回的张量不共享其内存。
# 如果为 ``False``，则返回的张量与 :attr:`obj` 共享其内存，如果无法共享，则会抛出错误。

# 返回张量的设备。
# 默认为 ``None``，这会使用 :attr:`obj` 的设备。或者，如果 :attr:`obj` 是 Python 序列，则使用当前默认设备。

# 返回张量是否需要梯度。
# 默认为 ``False``，这使得返回的张量不需要梯度。
# 如果为 ``True``，则返回的张量将需要梯度，并且如果 :attr:`obj` 也是具有自动求导历史的张量，则返回的张量将具有相同的历史。
# 执行 torch 操作的模块引入
import torch

# 函数签名及描述文档
"""
baddbmm(input, batch1, batch2, *, beta=1, alpha=1, out=None) -> Tensor

对在 batch1 和 batch2 中的矩阵进行批次矩阵乘积计算。
input 被加到最终结果中。

batch1 和 batch2 必须是包含相同数量矩阵的三维张量。

如果 batch1 是形状为 (b × n × m) 的张量，batch2 是形状为 (b × m × p) 的张量，
那么 input 必须与形状为 (b × n × p) 的张量具有广播兼容性，并且 out 将是形状为 (b × n × p) 的张量。
alpha 和 beta 表示与 torch.addbmm 中使用的缩放因子相同。

数学公式表示为：
out_i = beta * input_i + alpha * (batch1_i @ batch2_i)

如果 beta 为 0，则 input 将被忽略，并且其中的 nan 和 inf 不会传播。
"""

# 输入参数说明
"""
Args:
    input (Tensor): 要添加的张量
    batch1 (Tensor): 要相乘的第一批矩阵
    batch2 (Tensor): 要相乘的第二批矩阵

Keyword args:
    beta (Number, optional): input 的乘数 (β)
    alpha (Number, optional): batch1 @ batch2 的乘数 (α)
    out (Tensor, optional): 输出张量，用于存储结果
"""

# 使用示例
"""
Example::

    >>> M = torch.randn(10, 3, 5)
    >>> batch1 = torch.randn(10, 3, 4)
    >>> batch2 = torch.randn(10, 4, 5)
    >>> torch.baddbmm(M, batch1, batch2).size()
    torch.Size([10, 3, 5])
"""
""".format(
        **common_args, **tf32_notes, **rocm_fp16_notes
    ),
)

add_docstr(
    torch.bernoulli,
    r"""
bernoulli(input, *, generator=None, out=None) -> Tensor

Draws binary random numbers (0 or 1) from a Bernoulli distribution.

The :attr:`input` tensor should be a tensor containing probabilities
to be used for drawing the binary random number.
Hence, all values in :attr:`input` have to be in the range:
:math:`0 \leq \text{input}_i \leq 1`.

The :math:`\text{i}^{th}` element of the output tensor will draw a
value :math:`1` according to the :math:`\text{i}^{th}` probability value given
in :attr:`input`.

.. math::
    \text{out}_{i} \sim \mathrm{Bernoulli}(p = \text{input}_{i})

The returned :attr:`out` tensor only has values 0 or 1 and is of the same
shape as :attr:`input`.

:attr:`out` can have integral ``dtype``, but :attr:`input` must have floating
point ``dtype``.

Args:
    input (Tensor): the input tensor of probability values for the Bernoulli distribution

Keyword args:
    {generator}
    {out}

Example::

    >>> a = torch.empty(3, 3).uniform_(0, 1)  # generate a uniform random matrix with range [0, 1]
    >>> a
    tensor([[ 0.1737,  0.0950,  0.3609],
            [ 0.7148,  0.0289,  0.2676],
            [ 0.9456,  0.8937,  0.7202]])
    >>> torch.bernoulli(a)
    tensor([[ 1.,  0.,  0.],
            [ 0.,  0.,  0.],
            [ 1.,  1.,  1.]])

    >>> a = torch.ones(3, 3) # probability of drawing "1" is 1
    >>> torch.bernoulli(a)
    tensor([[ 1.,  1.,  1.],
            [ 1.,  1.,  1.],
            [ 1.,  1.,  1.]])
    >>> a = torch.zeros(3, 3) # probability of drawing "1" is 0
    >>> torch.bernoulli(a)
    tensor([[ 0.,  0.,  0.],
            [ 0.,  0.,  0.],
            [ 0.,  0.,  0.]])
""".format(
        **common_args
    ),
)

add_docstr(
    torch.bincount,
    r"""
bincount(input, weights=None, minlength=0) -> Tensor

Count the frequency of each value in an array of non-negative ints.

The number of bins (size 1) is one larger than the largest value in
:attr:`input` unless :attr:`input` is empty, in which case the result is a
tensor of size 0. If :attr:`minlength` is specified, the number of bins is at least
:attr:`minlength` and if :attr:`input` is empty, then the result is tensor of size
:attr:`minlength` filled with zeros. If ``n`` is the value at position ``i``,
``out[n] += weights[i]`` if :attr:`weights` is specified else
``out[n] += 1``.

Note:
    {backward_reproducibility_note}

Arguments:
    input (Tensor): 1-d int tensor
    weights (Tensor): optional, weight for each value in the input tensor.
        Should be of same size as input tensor.
    minlength (int): optional, minimum number of bins. Should be non-negative.

Returns:
    output (Tensor): a tensor of shape ``Size([max(input) + 1])`` if
    :attr:`input` is non-empty, else ``Size(0)``

Example::

    >>> input = torch.randint(0, 8, (5,), dtype=torch.int64)
    >>> weights = torch.linspace(0, 1, steps=5)
"""
)
    # 输入张量和权重张量
    >>> input, weights
    
    # input张量的直方图统计，返回每个值出现的次数
    >>> torch.bincount(input)
    tensor([0, 0, 0, 2, 2, 0, 1])
    
    # 使用input张量作为索引，统计weights张量中每个索引出现的和
    >>> input.bincount(weights)
    tensor([0.0000, 0.0000, 0.0000, 1.0000, 1.0000, 0.0000, 0.5000])
# 添加文档字符串给 torch.bitwise_not 函数
add_docstr(
    torch.bitwise_not,
    r"""
bitwise_not(input, *, out=None) -> Tensor

Computes the bitwise NOT of the given input tensor. The input tensor must be of
integral or Boolean types. For bool tensors, it computes the logical NOT.

Args:
    input (Tensor): the input tensor to apply bitwise NOT
    out (Tensor, optional): the output tensor (default: None)

Keyword args:
    {out}

Example::

    >>> torch.bitwise_not(torch.tensor([-1, -2, 3], dtype=torch.int8))
    tensor([ 0,  1, -4], dtype=torch.int8)
""".format(
        **common_args
    ),
)

# 添加文档字符串给 torch.bmm 函数
add_docstr(
    torch.bmm,
    r"""
bmm(input, mat2, *, out=None) -> Tensor

Performs a batch matrix-matrix product of matrices stored in :attr:`input`
and :attr:`mat2`.

:attr:`input` and :attr:`mat2` must be 3-D tensors each containing
the same number of matrices.

If :attr:`input` is a :math:`(b \times n \times m)` tensor, :attr:`mat2` is a
:math:`(b \times m \times p)` tensor, :attr:`out` will be a
:math:`(b \times n \times p)` tensor.

.. math::
    \text{out}_i = \text{input}_i \mathbin{@} \text{mat2}_i

{tf32_note}  # 添加 TF32 笔记（如果有）
{rocm_fp16_note}  # 添加 ROCm FP16 笔记（如果有）

.. note:: This function does not :ref:`broadcast <broadcasting-semantics>`.
          For broadcasting matrix products, see :func:`torch.matmul`.

Args:
    input (Tensor): the first batch of matrices to be multiplied
    mat2 (Tensor): the second batch of matrices to be multiplied

Keyword Args:
    {out}

Example::

    >>> input = torch.randn(10, 3, 4)
    >>> mat2 = torch.randn(10, 4, 5)
    >>> res = torch.bmm(input, mat2)
    >>> res.size()
    torch.Size([10, 3, 5])
""".format(
        **common_args, **tf32_notes, **rocm_fp16_notes
    ),
)

# 添加文档字符串给 torch.bitwise_and 函数
add_docstr(
    torch.bitwise_and,
    r"""
bitwise_and(input, other, *, out=None) -> Tensor

Computes the bitwise AND of :attr:`input` and :attr:`other`. The input tensor must be of
integral or Boolean types. For bool tensors, it computes the logical AND.

Args:
    input (Tensor): the first input tensor
    other (Tensor): the second input tensor

Keyword args:
    {out}

Example::

    >>> torch.bitwise_and(torch.tensor([-1, -2, 3], dtype=torch.int8), torch.tensor([1, 0, 3], dtype=torch.int8))
    tensor([1, 0,  3], dtype=torch.int8)
    >>> torch.bitwise_and(torch.tensor([True, True, False]), torch.tensor([False, True, False]))
    tensor([ False, True, False])
""".format(
        **common_args
    ),
)

# 添加文档字符串给 torch.bitwise_or 函数
add_docstr(
    torch.bitwise_or,
    r"""
bitwise_or(input, other, *, out=None) -> Tensor

Computes the bitwise OR of :attr:`input` and :attr:`other`. The input tensor must be of
integral or Boolean types. For bool tensors, it computes the logical OR.

Args:
    input (Tensor): the first input tensor
    other (Tensor): the second input tensor

Keyword args:
    {out}

Example::

    >>> torch.bitwise_or(torch.tensor([-1, -2, 3], dtype=torch.int8), torch.tensor([1, 0, 3], dtype=torch.int8))
    tensor([-1, -2,  3], dtype=torch.int8)
    >>> torch.bitwise_or(torch.tensor([True, True, False]), torch.tensor([False, True, False]))
    tensor([ True, True, False])
"""
)
# 根据给定的常见参数格式化文档字符串
""".format(
    **common_args
),

# 为 torch.bitwise_xor 函数添加文档字符串
add_docstr(
    torch.bitwise_xor,
    r"""
bitwise_xor(input, other, *, out=None) -> Tensor

Computes the bitwise XOR of :attr:`input` and :attr:`other`. The input tensor must be of
integral or Boolean types. For bool tensors, it computes the logical XOR.

Args:
    input: the first input tensor
    other: the second input tensor

Keyword args:
    {out}

Example::

    >>> torch.bitwise_xor(torch.tensor([-1, -2, 3], dtype=torch.int8), torch.tensor([1, 0, 3], dtype=torch.int8))
    tensor([-2, -2,  0], dtype=torch.int8)
    >>> torch.bitwise_xor(torch.tensor([True, True, False]), torch.tensor([False, True, False]))
    tensor([ True, False, False])
""".format(
    **common_args
),

# 为 torch.bitwise_left_shift 函数添加文档字符串
add_docstr(
    torch.bitwise_left_shift,
    r"""
bitwise_left_shift(input, other, *, out=None) -> Tensor

Computes the left arithmetic shift of :attr:`input` by :attr:`other` bits.
The input tensor must be of integral type. This operator supports
:ref:`broadcasting to a common shape <broadcasting-semantics>` and
:ref:`type promotion <type-promotion-doc>`.

The operation applied is:

.. math::
    \text{{out}}_i = \text{{input}}_i << \text{{other}}_i

Args:
    input (Tensor or Scalar): the first input tensor
    other (Tensor or Scalar): the second input tensor

Keyword args:
    {out}

Example::

    >>> torch.bitwise_left_shift(torch.tensor([-1, -2, 3], dtype=torch.int8), torch.tensor([1, 0, 3], dtype=torch.int8))
    tensor([-2, -2, 24], dtype=torch.int8)
""".format(
    **common_args
),

# 为 torch.bitwise_right_shift 函数添加文档字符串
add_docstr(
    torch.bitwise_right_shift,
    r"""
bitwise_right_shift(input, other, *, out=None) -> Tensor

Computes the right arithmetic shift of :attr:`input` by :attr:`other` bits.
The input tensor must be of integral type. This operator supports
:ref:`broadcasting to a common shape <broadcasting-semantics>` and
:ref:`type promotion <type-promotion-doc>`.
In any case, if the value of the right operand is negative or is greater
or equal to the number of bits in the promoted left operand, the behavior is undefined.

The operation applied is:

.. math::
    \text{{out}}_i = \text{{input}}_i >> \text{{other}}_i

Args:
    input (Tensor or Scalar): the first input tensor
    other (Tensor or Scalar): the second input tensor

Keyword args:
    {out}

Example::

    >>> torch.bitwise_right_shift(torch.tensor([-2, -7, 31], dtype=torch.int8), torch.tensor([1, 0, 3], dtype=torch.int8))
    tensor([-1, -7,  3], dtype=torch.int8)
""".format(
    **common_args
),

# 为 torch.broadcast_to 函数添加文档字符串
add_docstr(
    torch.broadcast_to,
    r"""
broadcast_to(input, shape) -> Tensor

Broadcasts :attr:`input` to the shape :attr:`\shape`.
Equivalent to calling ``input.expand(shape)``. See :meth:`~Tensor.expand` for details.

Args:
    {input}
    shape (list, tuple, or :class:`torch.Size`): the new shape.

Example::

    >>> x = torch.tensor([1, 2, 3])
    >>> torch.broadcast_to(x, (3, 3))

"""
    # 创建一个包含3行3列的张量
    tensor([[1, 2, 3],
            [1, 2, 3],
            [1, 2, 3]])
# 将格式化字符串插入到 torch.stack 的文档字符串中，替换公共参数
add_docstr(
    torch.stack,
    r"""
stack(tensors, dim=0, *, out=None) -> Tensor

Concatenates a sequence of tensors along a new dimension.

All tensors need to be of the same size.

.. seealso::

    :func:`torch.cat` concatenates the given sequence along an existing dimension.

Arguments:
    tensors (sequence of Tensors): sequence of tensors to concatenate
    dim (int, optional): dimension to insert. Has to be between 0 and the number
        of dimensions of concatenated tensors (inclusive). Default: 0

Keyword args:
    {out}  # 插入公共参数，用于指定输出结果的位置

Example::

    >>> x = torch.randn(2, 3)
    >>> x
    tensor([[ 0.3367,  0.1288,  0.2345],
            [ 0.2303, -1.1229, -0.1863]])
    >>> x = torch.stack((x, x)) # same as torch.stack((x, x), dim=0)
    >>> x
    tensor([[[ 0.3367,  0.1288,  0.2345],
             [ 0.2303, -1.1229, -0.1863]],

            [[ 0.3367,  0.1288,  0.2345],
             [ 0.2303, -1.1229, -0.1863]]])
    >>> x.size()
    torch.Size([2, 2, 3])
    >>> x = torch.stack((x, x), dim=1)
    tensor([[[ 0.3367,  0.1288,  0.2345],
             [ 0.3367,  0.1288,  0.2345]],

            [[ 0.2303, -1.1229, -0.1863],
             [ 0.2303, -1.1229, -0.1863]]])
    >>> x = torch.stack((x, x), dim=2)
    tensor([[[ 0.3367,  0.3367],
             [ 0.1288,  0.1288],
             [ 0.2345,  0.2345]],

            [[ 0.2303,  0.2303],
             [-1.1229, -1.1229],
             [-0.1863, -0.1863]]])
    >>> x = torch.stack((x, x), dim=-1)
    tensor([[[ 0.3367,  0.3367],
             [ 0.1288,  0.1288],
             [ 0.2345,  0.2345]],

            [[ 0.2303,  0.2303],
             [-1.1229, -1.1229],
             [-0.1863, -0.1863]]])
""".format(
        **common_args  # 格式化字符串，插入公共参数
    ),
)

# 将格式化字符串插入到 torch.hstack 的文档字符串中，替换公共参数
add_docstr(
    torch.hstack,
    r"""
hstack(tensors, *, out=None) -> Tensor

Stack tensors in sequence horizontally (column wise).

This is equivalent to concatenation along the first axis for 1-D tensors, and along the second axis for all other tensors.

Args:
    tensors (sequence of Tensors): sequence of tensors to concatenate

Keyword args:
    {out}  # 插入公共参数，用于指定输出结果的位置

Example::

    >>> a = torch.tensor([1, 2, 3])
    >>> b = torch.tensor([4, 5, 6])
    >>> torch.hstack((a,b))
    tensor([1, 2, 3, 4, 5, 6])
    >>> a = torch.tensor([[1],[2],[3]])
    >>> b = torch.tensor([[4],[5],[6]])
    >>> torch.hstack((a,b))
    tensor([[1, 4],
            [2, 5],
            [3, 6]])

""".format(
        **common_args  # 格式化字符串，插入公共参数
    ),
)

# 将格式化字符串插入到 torch.vstack 的文档字符串中，替换公共参数
add_docstr(
    torch.vstack,
    r"""
vstack(tensors, *, out=None) -> Tensor

Stack tensors in sequence vertically (row wise).

This is equivalent to concatenation along the first axis after all 1-D tensors have been reshaped by :func:`torch.atleast_2d`.

Args:
    tensors (sequence of Tensors): sequence of tensors to concatenate

Keyword args:
    {out}  # 插入公共参数，用于指定输出结果的位置

Example::

    >>> a = torch.tensor([1, 2, 3])
    >>> b = torch.tensor([4, 5, 6])
    >>> torch.vstack((a,b))
    tensor([[1, 2, 3],
            [4, 5, 6]])

""".format(
        **common_args  # 格式化字符串，插入公共参数
    ),
)
    # 创建一个张量 a，其中包含三行一列的数据
    a = torch.tensor([[1],[2],[3]])
    # 创建一个张量 b，其中包含三行一列的数据
    b = torch.tensor([[4],[5],[6]])
    # 使用 torch.vstack 函数将张量 a 和 b 垂直堆叠起来，形成一个新的张量
    stacked_tensor = torch.vstack((a,b))
    # 打印结果张量，显示垂直堆叠后的数据布局
    print(stacked_tensor)
# 为 torch.dstack 函数添加文档字符串
add_docstr(
    torch.dstack,
    r"""
dstack(tensors, *, out=None) -> Tensor

Stack tensors in sequence depthwise (along third axis).

This is equivalent to concatenation along the third axis after 1-D and 2-D tensors have been reshaped by :func:`torch.atleast_3d`.

Args:
    tensors (sequence of Tensors): sequence of tensors to concatenate

Keyword args:
    {out}  # 描述可选的输出参数，用于接收结果的张量

Example::

    >>> a = torch.tensor([1, 2, 3])
    >>> b = torch.tensor([4, 5, 6])
    >>> torch.dstack((a,b))  # 将张量 a 和 b 沿第三个轴深度堆叠
    tensor([[[1, 4],
             [2, 5],
             [3, 6]]])
    >>> a = torch.tensor([[1],[2],[3]])
    >>> b = torch.tensor([[4],[5],[6]])
    >>> torch.dstack((a,b))  # 将张量 a 和 b 沿第三个轴深度堆叠
    tensor([[[1, 4]],
            [[2, 5]],
            [[3, 6]]])


""".format(
        **common_args
    ),
)

# 为 torch.tensor_split 函数添加文档字符串
add_docstr(
    torch.tensor_split,
    r"""
tensor_split(input, indices_or_sections, dim=0) -> List of Tensors

Splits a tensor into multiple sub-tensors, all of which are views of :attr:`input`,
along dimension :attr:`dim` according to the indices or number of sections specified
by :attr:`indices_or_sections`. This function is based on NumPy's
:func:`numpy.array_split`.

Args:
    input (Tensor): the tensor to split  # 要分割的输入张量
    indices_or_sections (Tensor, int or list or tuple of ints):
        If :attr:`indices_or_sections` is an integer ``n`` or a zero dimensional long tensor
        with value ``n``, :attr:`input` is split into ``n`` sections along dimension :attr:`dim`.
        If :attr:`input` is divisible by ``n`` along dimension :attr:`dim`, each
        section will be of equal size, :code:`input.size(dim) / n`. If :attr:`input`
        is not divisible by ``n``, the sizes of the first :code:`int(input.size(dim) % n)`
        sections will have size :code:`int(input.size(dim) / n) + 1`, and the rest will
        have size :code:`int(input.size(dim) / n)`.

        If :attr:`indices_or_sections` is a list or tuple of ints, or a one-dimensional long
        tensor, then :attr:`input` is split along dimension :attr:`dim` at each of the indices
        in the list, tuple or tensor. For instance, :code:`indices_or_sections=[2, 3]` and :code:`dim=0`
        would result in the tensors :code:`input[:2]`, :code:`input[2:3]`, and :code:`input[3:]`.

        If :attr:`indices_or_sections` is a tensor, it must be a zero-dimensional or one-dimensional
        long tensor on the CPU.

    dim (int, optional): dimension along which to split the tensor. Default: ``0``  # 沿着哪个维度进行分割，默认为 0

Example::

    >>> x = torch.arange(8)
    >>> torch.tensor_split(x, 3)  # 将长度为 8 的张量 x 分割成 3 个子张量
    (tensor([0, 1, 2]), tensor([3, 4, 5]), tensor([6, 7]))

    >>> x = torch.arange(7)
    >>> torch.tensor_split(x, 3)  # 将长度为 7 的张量 x 分割成 3 个子张量
    (tensor([0, 1, 2]), tensor([3, 4]), tensor([5, 6]))
    >>> torch.tensor_split(x, (1, 6))  # 在指定索引处分割张量 x
    (tensor([0]), tensor([1, 2, 3, 4, 5]), tensor([6]))

    >>> x = torch.arange(14).reshape(2, 7)  # 创建一个 2x7 的张量 x
    >>> x
    tensor([[ 0,  1,  2,  3,  4,  5,  6],
            [ 7,  8,  9, 10, 11, 12, 13]])

""",
)
    # 使用 torch 库中的函数 tensor_split 对张量 x 进行分割，分割成 3 份，沿着第二维度（列）进行操作
    torch.tensor_split(x, 3, dim=1)
    (tensor([[0, 1, 2],   # 第一个张量：包含原始张量 x 第二维度上的前三列数据
            [7, 8, 9]]),  # 继续第一个张量：包含原始张量 x 第二维度上的后三列数据
     tensor([[ 3,  4],   # 第二个张量：包含原始张量 x 第二维度上的中间两列数据
            [10, 11]]),
     tensor([[ 5,  6],   # 第三个张量：包含原始张量 x 第二维度上的最后两列数据
            [12, 13]]))
    
    # 使用 torch 库中的函数 tensor_split 对张量 x 进行分割，分割成 3 份，分别在第 1 列和第 6 列的位置进行分割
    torch.tensor_split(x, (1, 6), dim=1)
    (tensor([[0],          # 第一个张量：包含原始张量 x 第二维度上的前 1 列数据
            [7]]),         # 继续第一个张量：包含原始张量 x 第二维度上的第 1 列数据
     tensor([[ 1,  2,  3,  4,  5],   # 第二个张量：包含原始张量 x 第二维度上的第 2 到第 6 列数据
            [ 8,  9, 10, 11, 12]]),
     tensor([[ 6],          # 第三个张量：包含原始张量 x 第二维度上的最后一列数据
            [13]]))
# 将 torch.chunk 函数添加文档字符串
add_docstr(
    torch.chunk,
    r"""
    chunk(input, chunks, dim=0) -> List of Tensors

    Attempts to split a tensor into the specified number of chunks. Each chunk is a view of
    the input tensor.

    .. note::

        This function may return fewer than the specified number of chunks!

    .. seealso::

        :func:`torch.tensor_split` a function that always returns exactly the specified number of chunks

    If the tensor size along the given dimension :attr:`dim` is divisible by :attr:`chunks`,
    all returned chunks will be the same size.
    If the tensor size along the given dimension :attr:`dim` is not divisible by :attr:`chunks`,
    all returned chunks will be the same size, except the last one.
    If such division is not possible, this function may return fewer
    than the specified number of chunks.

    Arguments:
        input (Tensor): the tensor to split
        chunks (int): number of chunks to return
        dim (int): dimension along which to split the tensor

    Example:
        >>> torch.arange(11).chunk(6)
        (tensor([0, 1]),
         tensor([2, 3]),
         tensor([4, 5]),
         tensor([6, 7]),
         tensor([8, 9]),
         tensor([10]))
        >>> torch.arange(12).chunk(6)
        (tensor([0, 1]),
         tensor([2, 3]),
         tensor([4, 5]),
         tensor([6, 7]),
         tensor([8, 9]),
         tensor([10, 11]))
        >>> torch.arange(13).chunk(6)
        (tensor([0, 1, 2]),
         tensor([3, 4, 5]),
         tensor([6, 7, 8]),
         tensor([ 9, 10, 11]),
         tensor([12]))
    """
)

# 将 torch.unsafe_chunk 函数添加文档字符串
add_docstr(
    torch.unsafe_chunk,
    r"""
    unsafe_chunk(input, chunks, dim=0) -> List of Tensors

    Works like :func:`torch.chunk` but without enforcing the autograd restrictions
    on inplace modification of the outputs.

    .. warning::
        This function is safe to use as long as only the input, or only the outputs
        are modified inplace after calling this function. It is user's
        responsibility to ensure that is the case. If both the input and one or more
        of the outputs are modified inplace, gradients computed by autograd will be
        silently incorrect.
    """
)

# 将 torch.unsafe_split 函数添加文档字符串
add_docstr(
    torch.unsafe_split,
    r"""
    unsafe_split(tensor, split_size_or_sections, dim=0) -> List of Tensors

    Works like :func:`torch.split` but without enforcing the autograd restrictions
    on inplace modification of the outputs.

    .. warning::
        This function is safe to use as long as only the input, or only the outputs
        are modified inplace after calling this function. It is user's
        responsibility to ensure that is the case. If both the input and one or more
        of the outputs are modified inplace, gradients computed by autograd will be
        silently incorrect.
    """
)

# 将 torch.hsplit 函数添加文档字符串
add_docstr(
    torch.hsplit,
    r"""
    hsplit(input, indices_or_sections) -> List of Tensors

    Splits :attr:`input`, a tensor with one or more dimensions, into multiple tensors
    horizontally according to :attr:`indices_or_sections`. Each split is a view of
    :attr:`input`.

    If :attr:`input` is one dimensional this is equivalent to calling
    ```
# 定义函数 torch.dsplit，用于深度分割三维及以上维度的张量
def dsplit(input, indices_or_sections):
    # 使用 torch.tensor_split 函数对输入张量进行深度分割，分割维度为 2
    return torch.tensor_split(input, indices_or_sections, dim=2)
# 计算给定输入张量的协方差矩阵。
# 
# Args:
#     input (Tensor): 包含多个变量和观察结果的二维矩阵，或表示单个变量的标量或一维向量。
#     bias (bool, optional): 如果为True，则使用偏差修正。默认为False。
# 
# Returns:
#     Tensor: 输入变量的协方差矩阵。
# 
# Example::
#     >>> x = torch.tensor([[0, 1, 2], [2, 1, 0]])
#     >>> torch.cov(x)
#     tensor([[ 1., -1.],
#             [-1.,  1.]])
#     >>> x = torch.randn(2, 4)
#     >>> torch.cov(x)
#     tensor([[ 0.2411, -0.0625],
#             [-0.0625,  0.4427]])
# 
# 注意：
#     如果输入是一个向量，则返回的是标量（0阶张量）而不是矩阵。
# 
# .. seealso::
#     :func:`torch.corrcoef` Pearson相关系数矩阵。
# ```py
# 估算给定输入矩阵的协方差矩阵，其中行是变量，列是观测值。
# 协方差矩阵是一个方阵，给出每对变量的协方差。对角线包含每个变量的方差（变量与自身的协方差）。
# 如果输入代表单个变量（标量或1D），则返回其方差。

cov(input, *, correction=1, fweights=None, aweights=None) -> Tensor
# 函数定义：cov函数接受input作为输入参数，并返回一个Tensor作为输出，表示变量的协方差矩阵。

# 样本协方差定义：
# 对于变量x和y的样本协方差由以下公式给出：
# cov(x,y) = (Σᵢ(xᵢ - x̄)(yᵢ - ȳ)) / max(0, N - correction * N)
# 其中x̄和ȳ分别是x和y的简单均值，correction是修正因子。

# 如果提供了fweights和/或aweights，则计算加权协方差，公式如下：
# cov_w(x,y) = (Σᵢwᵢ(xᵢ - μₓ⁎)(yᵢ - μy⁎)) / max(0, Σᵢwᵢ - (Σᵢwᵢaᵢ / Σᵢwᵢ) * correction * N)
# 其中w表示fweights或aweights，根据提供的哪一个，或者如果两者都提供了，则w = f * a。
# μₓ⁎ = Σᵢwᵢxᵢ / Σᵢwᵢ 是变量的加权均值。如果未提供，f和/或a可以视为适当大小的单位向量。

# 参数:
# input (Tensor): 包含多个变量和观测值的2D矩阵，或者表示单个变量的标量或1D向量。

# 关键字参数:
# correction (int, optional): 样本大小与样本自由度之差。默认为贝塞尔修正，correction = 1表示返回无偏估计，
# 即使同时指定了fweights和aweights。correction = 0将返回简单平均。默认为1。
# fweights (tensor, optional): 观测向量频率的标量或1D张量，表示每个观测应重复的次数。
# 其numel必须等于input的列数。如果为None，则忽略。默认为None。
# aweights (tensor, optional): 观测向量权重的标量或1D数组。
# 这些相对权重通常对于被认为“重要”的观测具有较大值，对于被认为“不重要”的观测具有较小值。
# 其numel必须等于input的列数。如果为None，则忽略。默认为None。

# 返回:
# (Tensor) 变量的协方差矩阵。

# 示例：
# >>> x = torch.tensor([[0, 2], [1, 1], [2, 0]]).T
# >>> x
    # 创建一个包含整数数据的 PyTorch 张量，形状为 (2, 3)
    tensor([[0, 1, 2],
            [2, 1, 0]])
    
    # 使用 torch.cov 函数计算输入张量 x 的协方差矩阵
    >>> torch.cov(x)
    tensor([[ 1., -1.],
            [-1.,  1.]])
    
    # 使用 correction=0 参数调用 torch.cov 函数计算 x 的无偏估计协方差矩阵
    >>> torch.cov(x, correction=0)
    tensor([[ 0.6667, -0.6667],
            [-0.6667,  0.6667]])
    
    # 创建一个包含随机整数数据的 PyTorch 张量 fw，形状为 (3,)
    >>> fw = torch.randint(1, 10, (3,))
    >>> fw
    tensor([1, 6, 9])
    
    # 创建一个包含随机浮点数数据的 PyTorch 张量 aw，形状为 (3,)
    >>> aw = torch.rand(3)
    >>> aw
    tensor([0.4282, 0.0255, 0.4144])
    
    # 使用 torch.cov 函数计算输入张量 x 的加权协方差矩阵，其中 fweights=fw 和 aweights=aw
    >>> torch.cov(x, fweights=fw, aweights=aw)
    tensor([[ 0.4169, -0.4169],
            [-0.4169,  0.4169]])
# 定义一个文档字符串，描述了 torch.cat 函数的功能和用法
add_docstr(
    torch.cat,
    r"""
cat(tensors, dim=0, *, out=None) -> Tensor

Concatenates the given sequence of :attr:`seq` tensors in the given dimension.
All tensors must either have the same shape (except in the concatenating
dimension) or be a 1-D empty tensor with size ``(0,)``.

:func:`torch.cat` can be seen as an inverse operation for :func:`torch.split`
and :func:`torch.chunk`.

:func:`torch.cat` can be best understood via examples.

.. seealso::

    :func:`torch.stack` concatenates the given sequence along a new dimension.

Args:
    tensors (sequence of Tensors): any python sequence of tensors of the same type.
        Non-empty tensors provided must have the same shape, except in the
        cat dimension.
    dim (int, optional): the dimension over which the tensors are concatenated

Keyword args:
    {out}

Example::

    >>> x = torch.randn(2, 3)
    >>> x
    tensor([[ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497]])
    >>> torch.cat((x, x, x), 0)
    tensor([[ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497],
            [ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497],
            [ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497]])
    >>> torch.cat((x, x, x), 1)
    tensor([[ 0.6580, -1.0969, -0.4614,  0.6580, -1.0969, -0.4614,  0.6580,
             -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497, -0.1034, -0.5790,  0.1497, -0.1034,
             -0.5790,  0.1497]])
""".format(
        **common_args
    ),
)

# 给 torch.concat 函数添加一个文档字符串，说明它是 torch.cat 的别名
add_docstr(
    torch.concat,
    r"""
concat(tensors, dim=0, *, out=None) -> Tensor

Alias of :func:`torch.cat`.
""",
)

# 给 torch.concatenate 函数添加一个文档字符串，说明它也是 torch.cat 的别名
add_docstr(
    torch.concatenate,
    r"""
concatenate(tensors, axis=0, out=None) -> Tensor

Alias of :func:`torch.cat`.
""",
)

# 给 torch.ceil 函数添加一个文档字符串，解释其功能和用法
add_docstr(
    torch.ceil,
    r"""
ceil(input, *, out=None) -> Tensor

Returns a new tensor with the ceil of the elements of :attr:`input`,
the smallest integer greater than or equal to each element.

For integer inputs, follows the array-api convention of returning a
copy of the input tensor.

.. math::
    \text{out}_{i} = \left\lceil \text{input}_{i} \right\rceil
"""
    + r"""
Args:
    {input}

Keyword args:
    {out}

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-0.6341, -1.4208, -1.0900,  0.5826])
    >>> torch.ceil(a)
    tensor([-0., -1., -1.,  1.])
""".format(
        **common_args
    ),
)

# 给 torch.real 函数添加一个文档字符串，解释其功能和用法
add_docstr(
    torch.real,
    r"""
real(input) -> Tensor

Returns a new tensor containing real values of the :attr:`self` tensor.
The returned tensor and :attr:`self` share the same underlying storage.

Args:
    {input}

Example::

    >>> x=torch.randn(4, dtype=torch.cfloat)
    >>> x
    tensor([(0.3100+0.3553j), (-0.5445-0.7896j), (-1.6492-0.0633j), (-0.0638-0.8119j)])
    >>> x.real
    tensor([ 0.3100, -0.5445, -1.6492, -0.0638])

""".format(
        **common_args
    ),
)

# 给 torch.imag 函数添加一个文档字符串，解释其功能和用法
add_docstr(
    torch.imag,
    r"""
imag(input) -> Tensor

Returns a new tensor containing the imaginary values of the :attr:`self` tensor.
The returned tensor and :attr:`self` share the same underlying storage.

Args:
    {input}

Example::

    >>> x=torch.randn(4, dtype=torch.cfloat)
    >>> x
    tensor([(0.3100+0.3553j), (-0.5445-0.7896j), (-1.6492-0.0633j), (-0.0638-0.8119j)])
    >>> x.imag
    tensor([ 0.3553, -0.7896, -0.0633, -0.8119])

""".format(
        **common_args
    ),
)
"""
reciprocal(input, *, out=None) -> Tensor

Returns a new tensor with the reciprocal of the elements of :attr:`input`

.. math::
    \text{out}_{i} = \frac{1}{\text{input}_{i}}

.. note::
    Unlike NumPy's reciprocal, torch.reciprocal supports integral inputs. Integral
    inputs to reciprocal are automatically :ref:`promoted <type-promotion-doc>` to
    the default scalar type.

Args:
    {input}  # 输入张量，其元素将被取倒数

Keyword args:
    {out}  # 可选参数，用于指定结果的输出张量

Example::

    >>> x = torch.tensor([2.0, 4.0, 0.5])
    >>> torch.reciprocal(x)
    tensor([0.5000, 0.2500, 2.0000])
"""
    # 创建一个包含4个随机数的张量 `a`
    >>> a = torch.randn(4)
    
    # 显示张量 `a` 的值
    >>> a
    tensor([-0.4595, -2.1219, -1.4314,  0.7298])
    
    # 计算张量 `a` 中每个元素的倒数，返回新的张量
    >>> torch.reciprocal(a)
    tensor([-2.1763, -0.4713, -0.6986,  1.3702])
# 为 torch.cholesky 函数添加文档字符串
add_docstr(
    torch.cholesky,
    r"""
cholesky(input, upper=False, *, out=None) -> Tensor

计算对称正定矩阵 :math:`A` 或批量对称正定矩阵的 Cholesky 分解。

如果 :attr:`upper` 为 ``True``，返回的矩阵 ``U`` 是上三角矩阵，分解形式为：

.. math::

  A = U^TU

如果 :attr:`upper` 为 ``False``，返回的矩阵 ``L`` 是下三角矩阵，分解形式为：

.. math::

    A = LL^T

如果 :attr:`upper` 为 ``True``，且 :math:`A` 是批量的对称正定矩阵，则返回的张量将由每个矩阵的上三角 Cholesky 因子组成。
类似地，当 :attr:`upper` 为 ``False`` 时，返回的张量将由每个矩阵的下三角 Cholesky 因子组成。

警告：

    :func:`torch.cholesky` 已被废弃，请使用 :func:`torch.linalg.cholesky` 替代，并将在未来的 PyTorch 版本中移除。

    ``L = torch.cholesky(A)`` 应替换为

    .. code:: python

        L = torch.linalg.cholesky(A)

    ``U = torch.cholesky(A, upper=True)`` 应替换为

    .. code:: python

        U = torch.linalg.cholesky(A).mH

    这种转换将对所有有效（对称正定）输入产生等效结果。

参数:
    input (Tensor): 大小为 :math:`(*, n, n)` 的输入张量 :math:`A`，其中 `*` 是零个或多个批次维度，包含对称正定矩阵。
    upper (bool, optional): 指示是否返回上三角或下三角矩阵的标志。默认为 ``False``

关键字参数:
    out (Tensor, optional): 输出矩阵

示例::

    >>> a = torch.randn(3, 3)
    >>> a = a @ a.mT + 1e-3 # 生成对称正定矩阵
    >>> l = torch.cholesky(a)
    >>> a
    tensor([[ 2.4112, -0.7486,  1.4551],
            [-0.7486,  1.3544,  0.1294],
            [ 1.4551,  0.1294,  1.6724]])
    >>> l
    tensor([[ 1.5528,  0.0000,  0.0000],
            [-0.4821,  1.0592,  0.0000],
            [ 0.9371,  0.5487,  0.7023]])
    >>> l @ l.mT
    tensor([[ 2.4112, -0.7486,  1.4551],
            [-0.7486,  1.3544,  0.1294],
            [ 1.4551,  0.1294,  1.6724]])
    >>> a = torch.randn(3, 2, 2) # 批量输入示例
    >>> a = a @ a.mT + 1e-03 # 生成对称正定矩阵
    >>> l = torch.cholesky(a)
    >>> z = l @ l.mT
    >>> torch.dist(z, a)
    tensor(2.3842e-07)
""",
)

# 为 torch.cholesky_solve 函数添加文档字符串
add_docstr(
    torch.cholesky_solve,
    r"""
cholesky_solve(B, L, upper=False, *, out=None) -> Tensor

使用 Cholesky 分解解决具有复共轭 Hermite 或实对称正定 lhs 的线性方程组。

设 :math:`A` 是一个复共轭 Hermite 或实对称正定矩阵，:math:`L` 是其 Cholesky 分解，如下所示：

.. math::

    A = L L^T
def cholesky_inverse(L, upper=False, *, out=None):
    """
    Computes the inverse of a complex Hermitian or real symmetric
    positive-definite matrix given its Cholesky decomposition.

    Let A be a complex Hermitian or real symmetric positive-definite matrix,
    and L its Cholesky decomposition such that:

    A = LL^H

    where L^H is the conjugate transpose when L is complex,
    and the transpose when L is real-valued.

    Computes the inverse matrix A^{-1}.

    Supports input of float, double, cfloat and cdouble dtypes.
    Also supports batches of matrices, and if A is a batch of matrices
    then the output has the same batch dimensions.

    Args:
        L (Tensor): tensor of shape (*, n, n) where * is zero or more batch dimensions
            consisting of lower or upper triangular Cholesky decompositions of
            symmetric or Hermitian positive-definite matrices.
        upper (bool, optional): flag that indicates whether L is lower triangular
            or upper triangular. Default: False

    Keyword args:
        out (Tensor, optional): output tensor. Ignored if None. Default: None
    """
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.
torch.column_stack(
    tensors,  # 由要水平堆叠的张量组成的列表或元组
    *, 
    out=None  # 可选，用于存储结果的输出张量
) -> Tensor  # 返回一个水平堆叠后的新张量

Creates a new tensor by horizontally stacking the tensors in :attr:`tensors`.
# 创建一个新张量，通过水平堆叠 :attr:`tensors` 中的张量而成。

Equivalent to ``torch.hstack(tensors)``, except each zero or one dimensional tensor ``t``
in :attr:`tensors` is first reshaped into a ``(t.numel(), 1)`` column before being stacked horizontally.
# 等效于 ``torch.hstack(tensors)``，但其中的每个零维或一维张量 ``t``
# 在水平堆叠之前会首先被重塑为 ``(t.numel(), 1)`` 列。

Args:
    tensors (list or tuple of Tensors): tensors to be horizontally stacked
        # 要进行水平堆叠的张量列表或元组
    out (Tensor, optional): output tensor to store the result
        # 可选，用于存储结果的输出张量

Returns:
    Tensor: a new tensor containing the horizontally stacked tensors
        # 包含水平堆叠张量的新张量
    # Import the necessary libraries for tensor operations
    from torch import tensor
add_docstr(
    torch.column_stack,
    r"""
    column_stack(tensors, *, out=None) -> Tensor
    
    Stack tensors in sequence along the column dimension (dim 1).

    Args:
        tensors (sequence of Tensors): Tensors to stack. Each tensor must have the same size in the
            first dimension.

    Keyword args:
        out (Tensor, optional): Output tensor. If provided, the result will be placed into this tensor.
    
    Returns:
        Tensor: The stacked tensor.

    Example::

        >>> a = torch.tensor([1, 2, 3])
        >>> b = torch.tensor([4, 5, 6])
        >>> torch.column_stack((a, b))
        tensor([[1, 4],
                [2, 5],
                [3, 6]])

        >>> a = torch.arange(5)
        >>> b = torch.arange(10).reshape(5, 2)
        >>> torch.column_stack((a, b, b))
        tensor([[0, 0, 1, 0, 1],
                [1, 2, 3, 2, 3],
                [2, 4, 5, 4, 5],
                [3, 6, 7, 6, 7],
                [4, 8, 9, 8, 9]])

    """.format(
        **common_args
    ),
)

add_docstr(
    torch.complex,
    r"""
    complex(real, imag, *, out=None) -> Tensor
    
    Constructs a complex tensor with its real part equal to :attr:`real` and its
    imaginary part equal to :attr:`imag`.
    
    Args:
        real (Tensor): The real part of the complex tensor. Must be half, float or double.
        imag (Tensor): The imaginary part of the complex tensor. Must be same dtype
            as :attr:`real`.
    
    Keyword args:
        out (Tensor): If the inputs are ``torch.float32``, must be
            ``torch.complex64``. If the inputs are ``torch.float64``, must be
            ``torch.complex128``.
    
    Example::
    
        >>> real = torch.tensor([1, 2], dtype=torch.float32)
        >>> imag = torch.tensor([3, 4], dtype=torch.float32)
        >>> z = torch.complex(real, imag)
        >>> z
        tensor([(1.+3.j), (2.+4.j)])
        >>> z.dtype
        torch.complex64
    
    """.format(
        **common_args
    ),
)

add_docstr(
    torch.polar,
    r"""
    polar(abs, angle, *, out=None) -> Tensor
    
    Constructs a complex tensor whose elements are Cartesian coordinates
    corresponding to the polar coordinates with absolute value :attr:`abs` and angle
    :attr:`angle`.
    
    .. math::
        \text{out} = \text{abs} \cdot \cos(\text{angle}) + \text{abs} \cdot \sin(\text{angle}) \cdot j
    
    .. note::
        `torch.polar` is similar to
        `std::polar <https://en.cppreference.com/w/cpp/numeric/complex/polar>`_
        and does not compute the polar decomposition
        of a complex tensor like Python's `cmath.polar` and SciPy's `linalg.polar` do.
        The behavior of this function is undefined if `abs` is negative or NaN, or if `angle` is
        infinite.
    
    Args:
        abs (Tensor): The absolute value the complex tensor. Must be float or double.
        angle (Tensor): The angle of the complex tensor. Must be same dtype as
            :attr:`abs`.
    
    Keyword args:
        out (Tensor): If the inputs are ``torch.float32``, must be
            ``torch.complex64``. If the inputs are ``torch.float64``, must be
            ``torch.complex128``.
    
    Example::
    
        >>> import numpy as np
        >>> abs = torch.tensor([1, 2], dtype=torch.float64)
        >>> angle = torch.tensor([np.pi / 2, 5 * np.pi / 4], dtype=torch.float64)
        >>> z = torch.polar(abs, angle)
        >>> z
        tensor([(0.0000+1.0000j), (-1.4142-1.4142j)], dtype=torch.complex128)
    
    """.format(
        **common_args
    ),
)

add_docstr(
    torch.conj_physical,
    r"""
    conj_physical(input, *, out=None) -> Tensor
    
    Computes the element-wise conjugate of the given :attr:`input` tensor.
    If :attr:`input` has a non-complex dtype, this function just returns :attr:`input`.
    
    Args:
        input (Tensor): The input tensor to compute the conjugate.
    
    Keyword args:
        out (Tensor): Output tensor. If provided, the result will be placed into this tensor.
    
    Returns:
        Tensor: The conjugate of the input tensor.
    
    """.format(
        **common_args
    ),
)
add_docstr(
    torch.copysign,
    r"""
copysign(input, other, *, out=None) -> Tensor

Create a new floating-point tensor with the magnitude of :attr:`input` and the sign of :attr:`other`, elementwise.

.. math::
    \text{{out}}_i = |\text{{input}}_i| \times \text{{sign}}(\text{{other}}_i)

Args:
    {input}
    other (Tensor): The tensor whose sign will be copied to `input`.

Keyword args:
    out (Tensor, optional): Output tensor. If provided, results will be written into this tensor.

Example::

    >>> x = torch.tensor([-1.0, 2.5, -3.7])
    >>> y = torch.tensor([1.0, -1.0, 0.0])
    >>> torch.copysign(x, y)
    tensor([-1.0, -2.5, 3.7])
""".format(
        **common_args
    ),
)
    # 根据条件对输入进行处理，生成输出值
    \text{out}_{i} = \begin{cases}
        -|\text{input}_{i}| & \text{if } \text{other}_{i} \leq -0.0 \\  # 如果条件满足：other_i 小于等于负零，则将 input_i 取负值
         |\text{input}_{i}| & \text{if } \text{other}_{i} \geq 0.0 \\  # 如果条件满足：other_i 大于等于零，则保持 input_i 不变
    \end{cases}
# 定义 cross 函数，计算输入张量 input 和 other 的向量叉乘
def cross(input, other, dim=None, *, out=None) -> Tensor:
    """
    Returns the cross product of vectors in dimension `dim` of `input`
    and `other`.

    Supports input of float, double, cfloat and cdouble dtypes. Also supports batches
    of vectors, for which it computes the product along the dimension `dim`.
    In this case, the output has the same batch dimensions as the inputs.
    """

    # 函数参数说明:
    # input (Tensor): 包含向量的张量
    # other (Tensor): 包含向量的张量，与 input 进行叉乘
    # dim (int, optional): 沿其进行叉乘的维度
    # out (Tensor, optional): 输出张量，存储结果的位置

    # 注意事项:
    # - 仅支持浮点、双精度、复数浮点和复数双精度类型输入
    # - 如果处理批量向量，沿指定维度 dim 进行计算，并且输出与输入具有相同的批量维度
    # - 未指定 dim 时，默认为最后一个维度进行叉乘

    # 返回值:
    # Tensor: 包含叉乘结果的张量
    # 如果未指定 `dim` 参数，则默认为找到的第一个维度大小为 3 的维度。
    # 注意，这可能会出乎意料。

    # 此行为已被弃用，并将在未来的发布版本中更改，以匹配 `torch.linalg.cross` 函数的行为。
"""
Returns a namedtuple ``(values, indices)`` where ``values`` is the cumulative minimum of
elements of :attr:`input` in the dimension :attr:`dim`. And ``indices`` is the index
location of each minimum value found in the dimension :attr:`dim`.

Args:
    {input} (Tensor): the input tensor
    dim (int): the dimension along which to compute the cumulative minimum

Keyword args:
    out (tuple, optional): the result tuple of two output tensors (values, indices)

Example::

    >>> a = torch.randn(10)
    >>> a
    tensor([-0.3449, -1.5447,  0.0685, -1.5104, -1.1706,  0.2259,  1.4696, -1.3284,
             1.9946, -0.8209])
    >>> torch.cummin(a, dim=0)
    torch.return_types.cummin(
        values=tensor([-0.3449, -1.5447, -1.5447, -1.5447, -1.5447, -1.5447, -1.5447, -1.5447,
                       -1.5447, -1.5447]),
        indices=tensor([0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    )
""".format(
    **reduceops_common_args
)
# 导入函数和方法的文档字符串增强工具
add_docstr(
    # 为 torch.cummin 函数添加文档字符串
    torch.cummin,
    r"""
cummin(input, dim, *, out=None) -> (Tensor, LongTensor)

Returns a namedtuple ``(values, indices)`` where ``values`` is the cumulative minimum of
elements of :attr:`input` in the dimension :attr:`dim`. And ``indices`` is the index
location of each maximum value found in the dimension :attr:`dim`.

.. math::
    y_i = min(x_1, x_2, x_3, \dots, x_i)

Args:
    {input}  # input: 要操作的输入张量
    dim  (int): the dimension to do the operation over  # dim: 执行操作的维度

Keyword args:
    out (tuple, optional): the result tuple of two output tensors (values, indices)  # out: 两个输出张量（values, indices）的结果元组，可选

Example::

    >>> a = torch.randn(10)  # 创建一个随机张量 a
    >>> a
    tensor([-0.2284, -0.6628,  0.0975,  0.2680, -1.3298, -0.4220, -0.3885,  1.1762,
         0.9165,  1.6684])
    >>> torch.cummin(a, dim=0)  # 在 dim=0 维度上计算累积最小值
    torch.return_types.cummin(
        values=tensor([-0.2284, -0.6628, -0.6628, -0.6628, -1.3298, -1.3298, -1.3298, -1.3298,
        -1.3298, -1.3298]),  # 返回累积最小值的张量
        indices=tensor([0, 1, 1, 1, 4, 4, 4, 4, 4, 4]))  # 返回每个最小值的索引位置的张量
""".format(
        **reduceops_common_args  # 使用 reduceops_common_args 中的参数填充文档字符串中的占位符
    ),
)

add_docstr(
    # 为 torch.cumprod 函数添加文档字符串
    torch.cumprod,
    r"""
cumprod(input, dim, *, dtype=None, out=None) -> Tensor

Returns the cumulative product of elements of :attr:`input` in the dimension
:attr:`dim`.

For example, if :attr:`input` is a vector of size N, the result will also be
a vector of size N, with elements.

.. math::
    y_i = x_1 \times x_2\times x_3\times \dots \times x_i

Args:
    {input}  # input: 要操作的输入张量
    dim  (int): the dimension to do the operation over  # dim: 执行操作的维度

Keyword args:
    {dtype}  # dtype: 输出张量的数据类型，可选
    {out}  # out: 输出张量，可选

Example::

    >>> a = torch.randn(10)  # 创建一个随机张量 a
    >>> a
    tensor([ 0.6001,  0.2069, -0.1919,  0.9792,  0.6727,  1.0062,  0.4126,
            -0.2129, -0.4206,  0.1968])
    >>> torch.cumprod(a, dim=0)  # 在 dim=0 维度上计算累积乘积
    tensor([ 0.6001,  0.1241, -0.0238, -0.0233, -0.0157, -0.0158, -0.0065,
             0.0014, -0.0006, -0.0001])

    >>> a[5] = 0.0  # 修改张量 a 中的第 5 个元素为 0
    >>> torch.cumprod(a, dim=0)  # 再次在 dim=0 维度上计算累积乘积
    tensor([ 0.6001,  0.1241, -0.0238, -0.0233, -0.0157, -0.0000, -0.0000,
             0.0000, -0.0000, -0.0000])
""".format(
        **reduceops_common_args  # 使用 reduceops_common_args 中的参数填充文档字符串中的占位符
    ),
)

add_docstr(
    # 为 torch.cumsum 函数添加文档字符串
    torch.cumsum,
    r"""
cumsum(input, dim, *, dtype=None, out=None) -> Tensor

Returns the cumulative sum of elements of :attr:`input` in the dimension
:attr:`dim`.

For example, if :attr:`input` is a vector of size N, the result will also be
a vector of size N, with elements.

.. math::
    y_i = x_1 + x_2 + x_3 + \dots + x_i

Args:
    {input}  # input: 要操作的输入张量
    dim  (int): the dimension to do the operation over  # dim: 执行操作的维度

Keyword args:
    {dtype}  # dtype: 输出张量的数据类型，可选
    {out}  # out: 输出张量，可选

Example::

    >>> a = torch.randint(1, 20, (10,))  # 创建一个随机整数张量 a，范围在 1 到 20 之间，大小为 10
    >>> a
    tensor([13,  7,  3, 10, 13,  3, 15, 10,  9, 10])
    >>> torch.cumsum(a, dim=0)  # 在 dim=0 维度上计算累积求和
    tensor([13, 20, 23, 33, 46, 49, 64, 74, 83, 93])
""".format(
        **reduceops_common_args  # 使用 reduceops_common_args 中的参数填充文档字符串中的占位符
    ),
)

add_docstr(
    # 为 torch.count_nonzero 函数添加文档字符串
    torch.count_nonzero,
    r"""
count_nonzero(input, dim=None) -> Tensor

Counts the number of non-zero values in the tensor :attr:`input` along the given :attr:`dim`.
If no dim is specified then all non-zeros in the tensor are counted.

Args:
    {input}  # input: 要操作的输入张量
    dim (int or tuple of ints, optional): Dim or tuple of dims along which to count non-zeros.
r"""
diag_embed(input, offset=0, dim1=-2, dim2=-1) -> Tensor

Creates a tensor whose diagonals of certain 2D planes (specified by
:attr:`dim1` and :attr:`dim2`) are filled by :attr:`input`.
To facilitate creating batched diagonal matrices, the 2D planes formed by
the last two dimensions of the returned tensor are chosen by default.

The argument :attr:`offset` controls which diagonal to consider:

- If :attr:`offset` = 0, it is the main diagonal.

Args:
    input (Tensor): A 1-D tensor whose elements will be placed on the specified diagonals.
    offset (int, optional): Offset for the diagonal. Default is 0 (main diagonal).
    dim1 (int, optional): First dimension index of the 2D planes. Default is -2.
    dim2 (int, optional): Second dimension index of the 2D planes. Default is -1.

Returns:
    Tensor: A tensor with diagonals filled with elements from `input` according to the specified dimensions.

Examples:

Create a batched diagonal matrix from a vector:

    >>> x = torch.tensor([1, 2, 3])
    >>> torch.diag_embed(x)
    tensor([[1, 0, 0],
            [0, 2, 0],
            [0, 0, 3]])

Create a batched diagonal matrix using different dimensions:

    >>> y = torch.tensor([10, 20, 30])
    >>> torch.diag_embed(y, dim1=-3, dim2=-2)
    tensor([[[10,  0,  0],
             [ 0,  0,  0],
             [ 0,  0,  0]],

            [[ 0,  0,  0],
             [ 0, 20,  0],
             [ 0,  0,  0]],

            [[ 0,  0,  0],
             [ 0,  0,  0],
             [ 0,  0, 30]]])
"""
def add_docstr(func, docstr):
    """
    Add a docstring to a given function or method.

    Args:
        func (function): Function or method to which the docstring will be added.
        docstr (str): The docstring to add to the function or method.
    """
    # 获取输入函数的文档字符串并附加新的文档字符串
    func.__doc__ = docstr.format(**common_args)
    # 创建一个 2x2 的张量 `a`，其中元素服从标准正态分布
    >>> a = torch.randn(2, 2)
    
    # 打印张量 `a` 的值
    >>> a
    tensor([[ 0.2094, -0.3018],
            [-0.1516,  1.9342]])
    
    # 将张量 `a` 转换为一个以其对角线元素为非零值的方阵，并将其余元素设为零
    >>> torch.diagflat(a)
    tensor([[ 0.2094,  0.0000,  0.0000,  0.0000],
            [ 0.0000, -0.3018,  0.0000,  0.0000],
            [ 0.0000,  0.0000, -0.1516,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  1.9342]])
# 导入所需模块
""".format(
    **common_args
),

# 给 torch.diagonal 函数添加文档字符串
add_docstr(
    torch.diagonal,
    r"""
diagonal(input, offset=0, dim1=0, dim2=1) -> Tensor

Returns a partial view of :attr:`input` with the its diagonal elements
with respect to :attr:`dim1` and :attr:`dim2` appended as a dimension
at the end of the shape.

The argument :attr:`offset` controls which diagonal to consider:

- If :attr:`offset` = 0, it is the main diagonal.
- If :attr:`offset` > 0, it is above the main diagonal.
- If :attr:`offset` < 0, it is below the main diagonal.

Applying :meth:`torch.diag_embed` to the output of this function with
the same arguments yields a diagonal matrix with the diagonal entries
of the input. However, :meth:`torch.diag_embed` has different default
dimensions, so those need to be explicitly specified.

Args:
    {input} Must be at least 2-dimensional.
    offset (int, optional): which diagonal to consider. Default: 0
        (main diagonal).
    dim1 (int, optional): first dimension with respect to which to
        take diagonal. Default: 0.
    dim2 (int, optional): second dimension with respect to which to
        take diagonal. Default: 1.

.. note::  To take a batch diagonal, pass in dim1=-2, dim2=-1.

Examples::

    >>> a = torch.randn(3, 3)
    >>> a
    tensor([[-1.0854,  1.1431, -0.1752],
            [ 0.8536, -0.0905,  0.0360],
            [ 0.6927, -0.3735, -0.4945]])


    >>> torch.diagonal(a, 0)
    tensor([-1.0854, -0.0905, -0.4945])


    >>> torch.diagonal(a, 1)
    tensor([ 1.1431,  0.0360])


    >>> x = torch.randn(2, 5, 4, 2)
    >>> torch.diagonal(x, offset=-1, dim1=1, dim2=2)
    tensor([[[-1.2631,  0.3755, -1.5977, -1.8172],
             [-1.1065,  1.0401, -0.2235, -0.7938]],

            [[-1.7325, -0.3081,  0.6166,  0.2335],
             [ 1.0500,  0.7336, -0.3836, -1.1015]]])
""".format(
    **common_args
),

# 给 torch.diagonal_scatter 函数添加文档字符串
add_docstr(
    torch.diagonal_scatter,
    r"""
diagonal_scatter(input, src, offset=0, dim1=0, dim2=1) -> Tensor

Embeds the values of the :attr:`src` tensor into :attr:`input` along
the diagonal elements of :attr:`input`, with respect to :attr:`dim1`
and :attr:`dim2`.

This function returns a tensor with fresh storage; it does not
return a view.

The argument :attr:`offset` controls which diagonal to consider:

- If :attr:`offset` = 0, it is the main diagonal.
- If :attr:`offset` > 0, it is above the main diagonal.
- If :attr:`offset` < 0, it is below the main diagonal.

Args:
    {input} Must be at least 2-dimensional.
    src (Tensor): the tensor to embed into :attr:`input`.
    offset (int, optional): which diagonal to consider. Default: 0
        (main diagonal).
    dim1 (int, optional): first dimension with respect to which to
        take diagonal. Default: 0.
    dim2 (int, optional): second dimension with respect to which to
        take diagonal. Default: 1.

.. note::

    :attr:`src` must be of the proper size in order to be embedded
    into :attr:`input`. Specifically, it should have the same shape as
    # 使用 torch.diagonal 函数从输入张量中提取对角线元素
    torch.diagonal(input, offset, dim1, dim2)
# 计算张量的 n 次前向差分，沿指定维度 dim 进行计算，默认为最后一个维度
# 第一阶差分由 out[i] = input[i + 1] - input[i] 给出。通过递归使用 torch.diff 可以计算更高阶的差分
def diff(input, n=1, dim=-1, prepend=None, append=None) -> Tensor:
    # input (Tensor): 要计算差分的张量
    # n (int, optional): 递归计算差分的次数
    # dim (int, optional): 要沿其计算差分的维度，默认为最后一个维度
    # prepend, append (Tensor, optional): 在计算差分前或后要添加到 input 的值
    # 它们的维度必须与 input 相同，并且除了 dim 维度外，它们的形状必须与 input 的形状相匹配

    # 返回计算结果的张量
    return torch.diff(input, n, dim, prepend, append)

# 计算输入张量的值按照给定的 size、stride 和 storage_offset 作为 as_strided 函数的结果，将 src 张量的值嵌入到 input 中
# 返回一个带有新存储的张量；它不是视图
def as_strided_scatter(input, src, size, stride, storage_offset=None) -> Tensor:
    # input (Tensor): 要在其中嵌入 src 张量值的张量
    # src (Tensor): 要嵌入的值
    # size (tuple or ints): 输出张量的形状
    # stride (tuple or ints): 输出张量的步幅
    # storage_offset (int, optional): 输出张量底层存储中的偏移量

    # 将 src 张量的值嵌入到 input 中，沿着调用 input.as_strided(size, stride, storage_offset) 的结果对应的元素
    # 返回一个带有新存储的张量
    return torch.as_strided_scatter(input, src, size, stride, storage_offset)

# 计算张量的 n 次前向差分，沿指定维度 dim 进行计算，默认为最后一个维度
def digamma(input) -> Tensor:
    # input (Tensor): 输入张量

    # 返回 digamma 函数应用于输入张量的结果
    return torch.digamma(input)
# 计算输入张量与另一张量之间的p范数
# 
# 参数:
#     input (Tensor): 左侧输入张量
#     other (Tensor): 右侧输入张量
#     p (float, optional): 要计算的范数，默认为2
# 
# 返回:
#     Tensor: 计算得到的p范数结果张量
# 
# 异常:
#     ValueError: 如果input和other的形状不可广播
# 
def dist(input, other, p=2):
    # 计算两个张量的差值
    return torch.special.digamma(input, other, p)

# 将输入张量的每个元素除以other张量对应的元素
# 
# Args:
#     input (Tensor): 被除数张量
#     other (Tensor or Number): 除数张量或数值
# 
# Keyword args:
#     rounding_mode (str, optional): 结果舍入模式:
# 
#         * None - 默认行为，执行真实除法，如果input和other都是整数类型，将它们提升为默认标量类型。等效于Python的真实除法和NumPy的np.true_divide。
#         * "trunc" - 向零舍入结果的除法。
#         * "floor" - 向下舍入结果的除法。
# 
# 返回:
#     Tensor: 计算得到的除法结果张量
# 
# 异常:
#     ValueError: 如果input和other的形状不可广播
# 
def div(input, other, *, rounding_mode=None, out=None):
    # 使用torch.div函数计算输入张量和其他张量的除法
    return torch.div(input, other, rounding_mode=rounding_mode, out=out)
    # 创建一个包含浮点数的张量，形状为 4x4
    tensor([[-0., -6.,  0.,  1.],
            [ 0., -3., -1.,  6.],
            [ 0.,  4., -0.,  5.],
            [-0., -0., -1.,  6.]])
    
    # 使用 torch.div 函数对张量 a 和 b 进行逐元素除法运算，指定向下取整的舍入模式
    # 返回一个新的张量，包含了 a 除以 b 后的结果，每个元素都被向下取整
    tensor([[-1., -7.,  0.,  1.],
            [ 0., -4., -2.,  6.],
            [ 0.,  4., -1.,  5.],
            [-1., -1., -2.,  6.]])
add_docstr(
    torch.divide,
    r"""
divide(input, other, *, rounding_mode=None, out=None) -> Tensor

Alias for :func:`torch.div`.
""",
)

add_docstr(
    torch.dot,
    r"""
dot(input, tensor, *, out=None) -> Tensor

Computes the dot product of two 1D tensors.

.. note::

    Unlike NumPy's dot, torch.dot intentionally only supports computing the dot product
    of two 1D tensors with the same number of elements.

Args:
    input (Tensor): first tensor in the dot product, must be 1D.
    tensor (Tensor): second tensor in the dot product, must be 1D.

Keyword args:
    {out}

Example::

    >>> torch.dot(torch.tensor([2, 3]), torch.tensor([2, 1]))
    tensor(7)

    >>> t1, t2 = torch.tensor([0, 1]), torch.tensor([2, 3])
    >>> torch.dot(t1, t2)
    tensor(3)
""".format(
        **common_args
    ),
)

add_docstr(
    torch.vdot,
    r"""
vdot(input, other, *, out=None) -> Tensor

Computes the dot product of two 1D vectors along a dimension.

In symbols, this function computes

.. math::

    \sum_{i=1}^n \overline{x_i}y_i.

where :math:`\overline{x_i}` denotes the conjugate for complex
vectors, and it is the identity for real vectors.

.. note::

    Unlike NumPy's vdot, torch.vdot intentionally only supports computing the dot product
    of two 1D tensors with the same number of elements.

.. seealso::

    :func:`torch.linalg.vecdot` computes the dot product of two batches of vectors along a dimension.

Args:
    input (Tensor): first tensor in the dot product, must be 1D. Its conjugate is used if it's complex.
    other (Tensor): second tensor in the dot product, must be 1D.

Keyword args:
    {out}

Example::

    >>> torch.vdot(torch.tensor([2, 3]), torch.tensor([2, 1]))
    tensor(7)
    >>> a = torch.tensor((1 +2j, 3 - 1j))
    >>> b = torch.tensor((2 +1j, 4 - 0j))
    >>> torch.vdot(a, b)
    tensor([16.+1.j])
    >>> torch.vdot(b, a)
    tensor([16.-1.j])
""".format(
        **common_args
    ),
)

add_docstr(
    torch.eq,
    r"""
eq(input, other, *, out=None) -> Tensor

Computes element-wise equality

The second argument can be a number or a tensor whose shape is
:ref:`broadcastable <broadcasting-semantics>` with the first argument.

Args:
    input (Tensor): the tensor to compare
    other (Tensor or float): the tensor or value to compare

Keyword args:
    {out}

Returns:
    A boolean tensor that is True where :attr:`input` is equal to :attr:`other` and False elsewhere

Example::

    >>> torch.eq(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
    tensor([[ True, False],
            [False, True]])
""".format(
        **common_args
    ),
)

add_docstr(
    torch.equal,
    r"""
equal(input, other) -> bool

``True`` if two tensors have the same size and elements, ``False`` otherwise.

Example::

    >>> torch.equal(torch.tensor([1, 2]), torch.tensor([1, 2]))
    True
""",
)

add_docstr(
    torch.erf,
    r"""
erf(input, *, out=None) -> Tensor

Computes the error function of each element of :attr:`input`.

Returns:
    A tensor of the same shape as :attr:`input` containing the error function of each element.

Example::

    >>> torch.erf(torch.tensor([0, -1., 10.]))
    tensor([ 0.0000, -0.8427,  1.0000])
"""
# 给 torch.erfc 函数添加文档字符串
add_docstr(
    torch.erfc,
    r"""
    erfc(input, *, out=None) -> Tensor

    别名：:func:`torch.special.erfc`.

    计算输入张量 :attr:`input` 的互补误差函数（complementary error function）。

    Args:
        input (Tensor): 输入张量

    Keyword args:
        {out}

    Example::

        >>> torch.erfc(torch.tensor([0, 1.0, 2.0]))
        tensor([ 1.0000,  0.1573,  0.0047])
    """.format(
        **common_args
    ),
)

# 给 torch.erfinv 函数添加文档字符串
add_docstr(
    torch.erfinv,
    r"""
    erfinv(input, *, out=None) -> Tensor

    别名：:func:`torch.special.erfinv`.

    计算输入张量 :attr:`input` 的逆误差函数（inverse error function）。

    Args:
        input (Tensor): 输入张量

    Keyword args:
        {out}

    Example::

        >>> torch.erfinv(torch.tensor([0, 0.5, 0.8]))
        tensor([ 0.0000,  0.4769,  0.9062])
    """.format(
        **common_args
    ),
)

# 给 torch.exp 函数添加文档字符串
add_docstr(
    torch.exp,
    r"""
    exp(input, *, out=None) -> Tensor

    返回一个新的张量，其中元素为输入张量 :attr:`input` 元素的指数值。

    .. math::
        y_{i} = e^{x_{i}}

    Args:
        {input}

    Keyword args:
        {out}

    Example::

        >>> torch.exp(torch.tensor([0, math.log(2.)]))
        tensor([ 1.,  2.])
    """.format(
        **common_args
    ),
)

# 给 torch.exp2 函数添加文档字符串
add_docstr(
    torch.exp2,
    r"""
    exp2(input, *, out=None) -> Tensor

    别名：:func:`torch.special.exp2`.

    计算输入张量 :attr:`input` 的二次幂。

    Args:
        input (Tensor): 输入张量

    Keyword args:
        {out}

    Example::

        >>> torch.exp2(torch.tensor([0, 1, 2]))
        tensor([ 1.,  2.,  4.])
    """.format(
        **common_args
    ),
)

# 给 torch.expm1 函数添加文档字符串
add_docstr(
    torch.expm1,
    r"""
    expm1(input, *, out=None) -> Tensor

    别名：:func:`torch.special.expm1`.

    计算输入张量 :attr:`input` 的 exp(x) - 1。

    Args:
        input (Tensor): 输入张量

    Keyword args:
        {out}

    Example::

        >>> torch.expm1(torch.tensor([0, math.log(2.)]))
        tensor([ 0.,  1.])
    """.format(
        **common_args
    ),
)

# 给 torch.eye 函数添加文档字符串
add_docstr(
    torch.eye,
    r"""
    eye(n, m=None, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

    返回一个二维张量，对角线上为1，其他位置为0。

    Args:
        n (int): 行数
        m (int, optional): 列数，默认为 :attr:`n`

    Keyword arguments:
        {out}
        {dtype}
        {layout}
        {device}
        {requires_grad}

    Returns:
        Tensor: 对角线上为1，其他位置为0的二维张量

    Example::

        >>> torch.eye(3)
        tensor([[ 1.,  0.,  0.],
                [ 0.,  1.,  0.],
                [ 0.,  0.,  1.]])
    """.format(
        **factory_common_args
    ),
)

# 给 torch.floor 函数添加文档字符串
add_docstr(
    torch.floor,
    r"""
    floor(input, *, out=None) -> Tensor

    返回一个新的张量，其中元素为 :attr:`input` 张量元素的向下取整结果，
    即不大于每个元素的最大整数。

    对于整数输入，遵循数组API约定，返回输入张量的副本。

    .. math::
        \text{out}_{i} = \left\lfloor \text{input}_{i} \right\rfloor

    Args:
        {input}

    Keyword args:
        {out}

    Example::

        >>> a = torch.randn(4)
        >>> a
        tensor([-0.8166,  1.5308, -0.2530, -0.2091])
        >>> torch.floor(a)
        tensor([-1.,  1., -1., -1.])
    """.format(
        **common_args
    ),
)

# 给 torch.floor_divide 函数添加文档字符串
add_docstr(
    torch.floor_divide,
    r"""
    floor_divide(input, other, *, out=None) -> Tensor

    计算 :attr:`input` 除以 :attr:`other` 的元素级除法，并对结果取整。

    .. note::

        在PyTorch 1.13之前，:func:`torch.floor_divide` 错误执行了截断除法。
        要恢复以前的行为，请使用 ``rounding_mode='trunc'`` 的 :func:`torch.div`。

    .. math::
        \text{{out}}_i = \text{floor} \left( \frac{{\text{{input}}_i}}{{\text{{other}}_i}} \right)

    支持广播到一个公共形状，类型提升，整数和浮点输入。

    Args:
        input (Tensor or Number): 被除数
        other (Tensor or Number): 除数

    Keyword args:
        {out}

    Example::

        >>> a = torch.tensor([4.0, 3.0])
    # 创建一个包含两个浮点数 2.0 的张量 b
    b = torch.tensor([2.0, 2.0])
    # 使用 torch.floor_divide 函数对张量 a 和张量 b 进行元素级地整除运算
    # 返回结果是一个张量，其中每个元素是 a 中对应位置元素除以 b 中对应位置元素的向下取整结果
    torch.floor_divide(a, b)
    # 返回结果张量 [2.0, 1.0]
    
    # 使用 torch.floor_divide 函数对张量 a 和标量 1.4 进行元素级地整除运算
    # 返回结果是一个张量，其中每个元素是 a 中对应位置元素除以 1.4 的向下取整结果
    torch.floor_divide(a, 1.4)
    # 返回结果张量 [2.0, 2.0]
add_docstr(
    torch.from_numpy,
    r"""
from_numpy(ndarray) -> Tensor

Creates a :class:`Tensor` from a :class:`numpy.ndarray`.

The returned tensor and :attr:`ndarray` share the same memory. Modifications to
the tensor will be reflected in the ndarray and vice versa. The returned tensor
is not resizable.

Args:
    ndarray (numpy.ndarray): a numpy array.

Returns:
    Tensor: a tensor created from the numpy array.

Example::

    >>> import numpy as np
    >>> arr = np.array([1, 2, 3])
    >>> t = torch.from_numpy(arr)
    >>> t
    tensor([1, 2, 3])
    >>> t[0] = 5
    >>> arr
    array([5, 2, 3])
""",
)
# 从缓冲区创建一个一维的张量对象，该缓冲区实现了 Python 的缓冲区协议
def frombuffer(buffer, *, dtype, count=-1, offset=0, requires_grad=False) -> Tensor:
    """
    Args:
        buffer (object): 一个实现了缓冲区接口的 Python 对象。
    
    Keyword args:
        dtype (:class:`torch.dtype`): 返回张量的所需数据类型。
        count (int, optional): 欲读取的元素个数。如果为负数，则读取所有元素直到缓冲区末尾。默认为 -1。
        offset (int, optional): 起始跳过的字节数。默认为 0。
        requires_grad (bool, optional): 是否需要梯度。默认为 False。

    Returns:
        Tensor: 从缓冲区创建的一维张量。

    Raises:
        Undefined behavior when passed an object with data not on the CPU.

    Notes:
        返回的张量与缓冲区共享内存。对张量的修改将反映在缓冲区中，反之亦然。返回的张量不可调整大小。
        此函数会增加共享内存对象的引用计数。因此，这样的内存在返回的张量超出作用域之前不会被释放。

    Warnings:
        当传递实现缓冲区协议但数据不在 CPU 上的对象时，此函数行为未定义，可能会导致段错误。
        此函数不会尝试推断 dtype（因此不是可选的）。传递不同于源的 dtype 可能会导致意外行为。

    Example::

        >>> import array
        >>> a = array.array('i', [1, 2, 3])
        >>> t = torch.frombuffer(a, dtype=torch.int32)
        >>> t
        tensor([ 1,  2,  3])
        >>> t[0] = -1
        >>> a
        array([-1,  2,  3])
    """
    pass
    # 导入array模块，用于操作数组
    import array
    # 创建一个array数组，类型为'signed char'，初始值为[-1, 0, 0, 0]
    a = array.array('b', [-1, 0, 0, 0])
    # 使用torch.frombuffer函数从数组a创建一个Tensor，数据类型为torch.int32
    torch.frombuffer(a, dtype=torch.int32)
    # 返回一个包含单个元素的Tensor，值为[255]，数据类型为torch.int32
    tensor([255], dtype=torch.int32)
# 为 torch.from_file 函数添加文档字符串
add_docstr(
    torch.from_file,
    r"""
from_file(filename, shared=None, size=0, *, dtype=None, layout=None, device=None, pin_memory=False)

Creates a CPU tensor with a storage backed by a memory-mapped file.

If ``shared`` is True, then memory is shared between processes. All changes are written to the file.
If ``shared`` is False, then changes to the tensor do not affect the file.

``size`` is the number of elements in the Tensor. If ``shared`` is ``False``, then the file must contain
at least ``size * sizeof(dtype)`` bytes. If ``shared`` is ``True`` the file will be created if needed.

.. note::
    Only CPU tensors can be mapped to files.

.. note::
    For now, tensors with storages backed by a memory-mapped file cannot be created in pinned memory.

Args:
    filename (str): file name to map
    shared (bool): whether to share memory (whether ``MAP_SHARED`` or ``MAP_PRIVATE`` is passed to the
                    underlying `mmap(2) call <https://man7.org/linux/man-pages/man2/mmap.2.html>`_)
    size (int): number of elements in the tensor

Keyword args:
    dtype (torch.dtype, optional): the desired data type of returned tensor. Default: inferred from data.
    layout (torch.layout, optional): the desired layout of returned tensor. Default: torch.strided.
    device (torch.device, optional): the desired device of returned tensor. Default: current device.
    pin_memory (bool, optional): whether to allocate memory for tensor on pinned memory. Default: False.

Example::
    >>> t = torch.randn(2, 5, dtype=torch.float64)
    >>> t.numpy().tofile('storage.pt')
    >>> t_mapped = torch.from_file('storage.pt', shared=False, size=10, dtype=torch.float64)
""",
)
    # 使用 torch.flatten 函数，将张量的多个维度压缩成一个维度。这是其逆函数。
    # 它将多维张量变成一维张量。
# 计算沿着指定维度 `dim` 对张量 `input` 进行展开操作后的视图，返回新的张量。
# 参数 `dim` 是一个索引，用来指定要展开的维度，应该是在 `input.shape` 中的索引。
# 参数 `sizes` 是一个元组，指定了展开后的新形状。元组中可以包含 `-1`，表示相应的输出维度由函数推断得出；否则，`sizes` 中各维度的乘积必须等于 `input.shape[dim]`。
def torch.unflatten(input, dim, sizes):
    pass

# 从张量 `input` 中按照 `dim` 指定的轴收集（gather）值。
# 对于一个三维张量，输出的规则如下：
# - 如果 `dim == 0`，则 `out[i][j][k] = input[index[i][j][k]][j][k]`
# - 如果 `dim == 1`，则 `out[i][j][k] = input[i][index[i][j][k]][k]`
# - 如果 `dim == 2`，则 `out[i][j][k] = input[i][j][index[i][j][k]]`
# 要求 `input` 和 `index` 的维度数相同，并且对于所有维度 `d != dim`，都要求 `index.size(d) <= input.size(d)`。
# 返回的张量 `out` 将具有与 `index` 相同的形状。
def torch.gather(input, dim, index, *, sparse_grad=False, out=None):
    pass

# 计算张量 `input` 和 `other` 逐元素的最大公约数（GCD）。
# `input` 和 `other` 必须都是整数类型的张量。
# 注意，定义了 `gcd(0, 0) = 0`。
def torch.gcd(input, other, *, out=None):
    pass

# 计算逐元素的大于等于比较，即 `input >= other`。
# 第二个参数可以是一个数值或者与第一个参数具有广播语义的张量。
def torch.ge(input, other, *, out=None):
    pass
Keyword args:
    {out}


Returns:
    A boolean tensor that is True where :attr:`input` is greater than or equal to :attr:`other` and False elsewhere


Example::

    >>> torch.ge(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
    tensor([[True, True], [False, True]])
"""
.format(
    **common_args
),


add_docstr(
    torch.greater_equal,
    r"""
greater_equal(input, other, *, out=None) -> Tensor

Alias for :func:`torch.ge`.
""",
)


add_docstr(
    torch.gradient,
    r"""
gradient(input, *, spacing=1, dim=None, edge_order=1) -> List of Tensors

Estimates the gradient of a function :math:`g : \mathbb{R}^n \rightarrow \mathbb{R}` in
one or more dimensions using the `second-order accurate central differences method
<https://www.ams.org/journals/mcom/1988-51-184/S0025-5718-1988-0935077-0/S0025-5718-1988-0935077-0.pdf>`_ and
either first or second order estimates at the boundaries.

The gradient of :math:`g` is estimated using samples. By default, when :attr:`spacing` is not
specified, the samples are entirely described by :attr:`input`, and the mapping of input coordinates
to an output is the same as the tensor's mapping of indices to values. For example, for a three-dimensional
:attr:`input` the function described is :math:`g : \mathbb{R}^3 \rightarrow \mathbb{R}`, and
:math:`g(1, 2, 3)\ == input[1, 2, 3]`.

When :attr:`spacing` is specified, it modifies the relationship between :attr:`input` and input coordinates.
This is detailed in the "Keyword Arguments" section below.

The gradient is estimated by estimating each partial derivative of :math:`g` independently. This estimation is
accurate if :math:`g` is in :math:`C^3` (it has at least 3 continuous derivatives), and the estimation can be
improved by providing closer samples. Mathematically, the value at each interior point of a partial derivative
is estimated using `Taylor's theorem with remainder <https://en.wikipedia.org/wiki/Taylor%27s_theorem>`_.
Letting :math:`x` be an interior point with :math:`x-h_l` and :math:`x+h_r` be points neighboring
it to the left and right respectively, :math:`f(x+h_r)` and :math:`f(x-h_l)` can be estimated using:

.. math::
    \begin{aligned}
        f(x+h_r) = f(x) + h_r f'(x) + {h_r}^2  \frac{f''(x)}{2} + {h_r}^3 \frac{f'''(\xi_1)}{6}, \xi_1 \in (x, x+h_r) \\
        f(x-h_l) = f(x) - h_l f'(x) + {h_l}^2  \frac{f''(x)}{2} - {h_l}^3 \frac{f'''(\xi_2)}{6}, \xi_2 \in (x, x-h_l) \\
    \end{aligned}

Using the fact that :math:`f \in C^3` and solving the linear system, we derive:

.. math::
    f'(x) \approx \frac{ {h_l}^2 f(x+h_r) - {h_r}^2 f(x-h_l)
          + ({h_r}^2-{h_l}^2 ) f(x) }{ {h_r} {h_l}^2 + {h_r}^2 {h_l} }

.. note::
    We estimate the gradient of functions in complex domain
    :math:`g : \mathbb{C}^n \rightarrow \mathbb{C}` in the same way.

The value of each partial derivative at the boundary points is computed differently. See edge_order below.

Args:


注释：
    # 输入参数 (``Tensor``): 表示函数数值的张量
    input (``Tensor``): the tensor that represents the values of the function
Keyword args:
    spacing (``scalar``, ``list of scalar``, ``list of Tensor``, optional): :attr:`spacing` 可以用来修改
        :attr:`input` 张量的索引如何与样本坐标相关联。如果 :attr:`spacing` 是一个标量，则将索引乘以该标量以生成坐标。
        例如，如果 :attr:`spacing=2`，则索引 (1, 2, 3) 变为坐标 (2, 4, 6)。
        如果 :attr:`spacing` 是一个标量列表，则对应的索引分别相乘。
        例如，如果 :attr:`spacing=(2, -1, 3)`，则索引 (1, 2, 3) 变为坐标 (2, -2, 9)。
        最后，如果 :attr:`spacing` 是一个一维张量列表，则每个张量指定相应维度的坐标。
        例如，如果索引是 (1, 2, 3)，张量是 (t0, t1, t2)，则坐标是 (t0[1], t1[2], t2[3])。

    dim (``int``, ``list of int``, optional): 要计算梯度的维度或维度列表。默认情况下，计算每个维度的偏导数。
        注意，当指定 :attr:`dim` 时，:attr:`spacing` 参数的元素必须与指定的维度对应。

    edge_order (``int``, optional): 1 或 2，用于边界值的 `一阶估计 <https://www.ams.org/journals/mcom/1988-51-184/S0025-5718-1988-0935077-0/S0025-5718-1988-0935077-0.pdf>`_
        或 `二阶估计 <https://www.ams.org/journals/mcom/1988-51-184/S0025-5718-1988-0935077-0/S0025-5718-1988-0935077-0.pdf>`_。

Examples::

    >>> # 对 f(x)=x^2 在点 [-2, -1, 2, 4] 处估计梯度
    >>> coordinates = (torch.tensor([-2., -1., 1., 4.]),)
    >>> values = torch.tensor([4., 1., 1., 16.], )
    >>> torch.gradient(values, spacing = coordinates)
    (tensor([-3., -2., 2., 5.]),)

    >>> # 估计 R^2 -> R 函数的梯度，其样本由张量 t 描述。隐式坐标为最外层维度的 [0, 1] 和最内层维度的 [0, 1, 2, 3]，并为两个维度估计偏导数。
    >>> t = torch.tensor([[1, 2, 4, 8], [10, 20, 40, 80]])
    >>> torch.gradient(t)
    (tensor([[ 9., 18., 36., 72.],
             [ 9., 18., 36., 72.]]),
     tensor([[ 1.0000, 1.5000, 3.0000, 4.0000],
             [10.0000, 15.0000, 30.0000, 40.0000]]))

    >>> # 标量值的 spacing 通过将索引乘以标量来修改张量索引和输入坐标之间的关系。例如，下面的最内层维度的索引 0, 1, 2, 3 转换为坐标 [0, 2, 4, 6]，
    >>> # 最外层维度的索引 0, 1 转换为坐标 [0, 2]。
    >>> torch.gradient(t, spacing = 2.0) # dim = None (implicitly [0, 1])
    # doubling the spacing between samples halves the estimated partial gradients.
    # 对样本之间的间距加倍会使估计的偏导数减半。

    # Estimates only the partial derivative for dimension 1
    # 仅估算维度 1 的偏导数
    torch.gradient(t, dim=1)  # spacing = None (implicitly 1.)
    (tensor([[ 1.0000, 1.5000, 3.0000, 4.0000],
             [10.0000, 15.0000, 30.0000, 40.0000]]),)

    # When spacing is a list of scalars, the relationship between the tensor
    # indices and input coordinates changes based on dimension.
    # 当 spacing 是一个标量列表时，张量索引和输入坐标之间的关系根据维度变化。
    # 例如，在下面的示例中，最内层维度的索引 0, 1, 2, 3 对应于坐标 [0, 3, 6, 9]，
    # 最外层维度的索引 0, 1 对应于坐标 [0, 2]。
    torch.gradient(t, spacing=[3., 2.])
    (tensor([[ 4.5000, 9.0000, 18.0000, 36.0000],
             [ 4.5000, 9.0000, 18.0000, 36.0000]]),
     tensor([[ 0.3333, 0.5000, 1.0000, 1.3333],
             [ 3.3333, 5.0000, 10.0000, 13.3333]]))

    # The following example is a replication of the previous one with explicit
    # coordinates.
    # 下面的示例与前面的示例是一样的，只是使用了显式的坐标。
    coords = (torch.tensor([0, 2]), torch.tensor([0, 3, 6, 9]))
    torch.gradient(t, spacing=coords)
    (tensor([[ 4.5000, 9.0000, 18.0000, 36.0000],
             [ 4.5000, 9.0000, 18.0000, 36.0000]]),
     tensor([[ 0.3333, 0.5000, 1.0000, 1.3333],
             [ 3.3333, 5.0000, 10.0000, 13.3333]]))
# 定义一个函数，用于为指定函数添加文档字符串
add_docstr(
    torch.geqrf,
    r"""
geqrf(input, *, out=None) -> (Tensor, Tensor)

This is a low-level function for calling LAPACK's geqrf directly. This function
returns a namedtuple (a, tau) as defined in `LAPACK documentation for geqrf`_ .

Computes a QR decomposition of :attr:`input`.
Both `Q` and `R` matrices are stored in the same output tensor `a`.
The elements of `R` are stored on and above the diagonal.
Elementary reflectors (or Householder vectors) implicitly defining matrix `Q`
are stored below the diagonal.
The results of this function can be used together with :func:`torch.linalg.householder_product`
to obtain the `Q` matrix or
with :func:`torch.ormqr`, which uses an implicit representation of the `Q` matrix,
for an efficient matrix-matrix multiplication.

See `LAPACK documentation for geqrf`_ for further details.

.. note::
    See also :func:`torch.linalg.qr`, which computes Q and R matrices, and :func:`torch.linalg.lstsq`
    with the ``driver="gels"`` option for a function that can solve matrix equations using a QR decomposition.

Args:
    input (Tensor): the input matrix

Keyword args:
    out (tuple, optional): the output tuple of (Tensor, Tensor). Ignored if `None`. Default: `None`.

.. _LAPACK documentation for geqrf:
    http://www.netlib.org/lapack/explore-html/df/dc5/group__variants_g_ecomputational_ga3766ea903391b5cf9008132f7440ec7b.html

""",
)

# 定义一个函数，用于计算一维张量的点积，或者在更高维度上沿最后一个维度求元素乘积的和
add_docstr(
    torch.inner,
    r"""
inner(input, other, *, out=None) -> Tensor

Computes the dot product for 1D tensors. For higher dimensions, sums the product
of elements from :attr:`input` and :attr:`other` along their last dimension.

.. note::

    If either :attr:`input` or :attr:`other` is a scalar, the result is equivalent
    to `torch.mul(input, other)`.

    If both :attr:`input` and :attr:`other` are non-scalars, the size of their last
    dimension must match and the result is equivalent to `torch.tensordot(input,
    other, dims=([-1], [-1]))`

Args:
    input (Tensor): First input tensor
    other (Tensor): Second input tensor

Keyword args:
    out (Tensor, optional): Optional output tensor to write result into. The output
                            shape is `input.shape[:-1] + other.shape[:-1]`.

Example::

    # Dot product
    >>> torch.inner(torch.tensor([1, 2, 3]), torch.tensor([0, 2, 1]))
    tensor(7)

    # Multidimensional input tensors
    >>> a = torch.randn(2, 3)
    >>> a
    tensor([[0.8173, 1.0874, 1.1784],
            [0.3279, 0.1234, 2.7894]])
    >>> b = torch.randn(2, 4, 3)
    >>> b
    tensor([[[-0.4682, -0.7159,  0.1506],
            [ 0.4034, -0.3657,  1.0387],
            [ 0.9892, -0.6684,  0.1774],
            [ 0.9482,  1.3261,  0.3917]],

            [[ 0.4537,  0.7493,  1.1724],
            [ 0.2291,  0.5749, -0.2267],
            [-0.7920,  0.3607, -0.3701],
            [ 1.3666, -0.5850, -1.7242]]])
    >>> torch.inner(a, b)

""",
)
    # 一个包含两个3x4张量的张量，每个元素为浮点数
    tensor([[[-0.9837,  1.1560,  0.2907,  2.6785],
             [ 2.5671,  0.5452, -0.6912, -1.5509]],
    
            [[ 0.1782,  2.9843,  0.7366,  1.5672],
             [ 3.5115, -0.4864, -1.2476, -4.4337]]])
    
    # 使用标量输入进行张量的内积运算
    >>> torch.inner(a, torch.tensor(2))
    tensor([[1.6347, 2.1748, 2.3567],
            [0.6558, 0.2469, 5.5787]])
# 计算张量的直方图
# 将元素分配到最小值和最大值之间的等宽箱中
# 如果最小值和最大值都为零，则使用数据的最小值和最大值
def histc(input, bins=100, min=0, max=0, *, out=None) -> Tensor:
"""
Elements lower than min and higher than max and NaN elements are ignored.

Args:
    {input}
    bins (int): number of histogram bins
    min (Scalar): lower end of the range (inclusive)
    max (Scalar): upper end of the range (inclusive)

Keyword args:
    {out}

Returns:
    Tensor: Histogram represented as a tensor

Example::

    >>> torch.histc(torch.tensor([1., 2, 1]), bins=4, min=0, max=3)
    tensor([ 0.,  2.,  1.,  0.])
""".format(
    **common_args
),
"""

add_docstr(
    torch.histogram,
    r"""
histogram(input, bins, *, range=None, weight=None, density=False, out=None) -> (Tensor, Tensor)

Computes a histogram of the values in a tensor.

:attr:`bins` can be an integer or a 1D tensor.

If :attr:`bins` is an int, it specifies the number of equal-width bins.
By default, the lower and upper range of the bins is determined by the
minimum and maximum elements of the input tensor. The :attr:`range`
argument can be provided to specify a range for the bins.

If :attr:`bins` is a 1D tensor, it specifies the sequence of bin edges
including the rightmost edge. It should contain at least 2 elements
and its elements should be increasing.

Args:
    {input}
    bins: int or 1D Tensor. If int, defines the number of equal-width bins. If tensor,
          defines the sequence of bin edges including the rightmost edge.
    range (tuple of float): Defines the range of the bins.
    weight (Tensor): If provided, weight should have the same shape as input. Each value in
                     input contributes its associated weight towards its bin's result.
    density (bool): If False, the result will contain the count (or total weight) in each bin.
                    If True, the result is the value of the probability density function over the bins,
                    normalized such that the integral over the range of the bins is 1.
    {out} (tuple, optional): The result tuple of two output tensors (hist, bin_edges).

Returns:
    hist (Tensor): 1D Tensor containing the values of the histogram.
    bin_edges(Tensor): 1D Tensor containing the edges of the histogram bins.

Example::

    >>> torch.histogram(torch.tensor([1., 2, 1]), bins=4, range=(0., 3.), weight=torch.tensor([1., 2., 4.]))
    (tensor([ 0.,  5.,  2.,  0.]), tensor([0., 0.75, 1.5, 2.25, 3.]))
    >>> torch.histogram(torch.tensor([1., 2, 1]), bins=4, range=(0., 3.), weight=torch.tensor([1., 2., 4.]), density=True)
    (tensor([ 0.,  0.9524,  0.3810,  0.]), tensor([0., 0.75, 1.5, 2.25, 3.]))
""".format(
    **common_args
),
"""

add_docstr(
    torch.histogramdd,
    r"""
histogramdd(input, bins, *, range=None, weight=None, density=False, out=None) -> (Tensor, Tensor[])

Computes a multi-dimensional histogram of the values in a tensor.

Interprets the elements of an input tensor whose innermost dimension has size N
as a collection of N-dimensional points. Maps each of the points into a set of
# 计算 N 维空间中的数据点落入各个箱子的数量（或总权重）。

# :attr:`input` 必须是至少有两个维度的张量。
# 如果 input 的形状为 (M, N)，则其每一行定义了 N 维空间中的一个点。
# 如果 input 的维度大于等于三，除了最后一个维度外的所有维度都会被展平。

# 每个维度独立地关联着其自己的严格递增的箱子边界序列。箱子边界可以通过传递一个 1 维张量序列来显式指定。
# 或者，可以通过传递一个指定每个维度中等宽箱子数量的整数序列来自动构造箱子边界。

# 对于 input 中的每个 N 维点：
#   - 其每个坐标独立地根据其维度对应的箱子边界进行分箱
#   - 分箱结果被组合以确定点落入的 N 维箱子（如果有的话）
#   - 如果点落入某个箱子，则增加该箱子的计数（或总权重）
#   - 不落入任何箱子的点不会对输出产生贡献

# :attr:`bins` 可以是 N 个 1 维张量的序列、N 个整数的序列或一个整数。

# 如果 :attr:`bins` 是 N 个 1 维张量的序列，则明确指定了 N 个箱子边界序列。
# 每个 1 维张量应该包含至少一个元素的严格递增序列。K 个箱子边界定义了 K-1 个箱子，显式指定了所有箱子的左右边界。每个箱子都不包括其左边界，只有最右边的箱子包括其右边界。

# 如果 :attr:`bins` 是 N 个整数的序列，则指定了每个维度中的等宽箱子数量。
# 默认情况下，每个维度中的最左边和最右边的箱子边界由相应维度中输入张量的最小和最大元素确定。可以提供 :attr:`range` 参数来手动指定每个维度中最左边和最右边的箱子边界。

# 如果 :attr:`bins` 是一个整数，则指定了所有维度中的等宽箱子数量。

# .. note::
#     另请参见 :func:`torch.histogram`，该函数专门计算 1 维直方图。
#     虽然 :func:`torch.histogramdd` 从 :attr:`input` 的形状推断其箱子的维度和分箱的值，
#     但 :func:`torch.histogram` 接受并展平任何形状的 :attr:`input`。

def histogramdd(input, bins, **kwargs):
    pass
    weight (Tensor): 默认情况下，输入中的每个值的权重为1。如果传入了一个权重张量，那么输入中的每个N维坐标将根据其相关的权重对其所在的箱子结果做出贡献。权重张量应该与 :attr:`input` 张量具有相同的形状，但不包括其最内部的维度N。

    density (bool): 如果为False（默认），结果将包含每个箱子中的计数（或总权重）。如果为True，则每个计数（权重）将除以总计数（总权重），然后再除以其关联箱子的体积。
# 返回一个新的张量，按照给定的索引在指定维度上对输入张量进行索引选择
def index_select(input, dim, index, *, out=None) -> Tensor:
    """
    Returns a new tensor which indexes the 'input' tensor along dimension
    'dim' using the entries in 'index' which is a `LongTensor`.

    The returned tensor has the same number of dimensions as the original tensor
    ('input'). The 'dim'-th dimension has the same size as the length
    of 'index'; other dimensions have the same size as in the original tensor.

    .. note:: The returned tensor does **not** use the same storage as the original
              tensor. If 'out' has a different shape than expected, we
              silently change it to the correct shape, reallocating the underlying
              storage if necessary.

    Args:
        {input}
        dim (int): the dimension in which we index
        index (LongTensor): the 1D tensor containing the indices to index along 'dim'

    Keyword Args:
        out (Tensor, optional): the output tensor

    """
    index (IntTensor or LongTensor): the 1-D tensor containing the indices to index
# 为 torch.inverse 函数添加文档字符串，作为 torch.linalg.inv 的别名
add_docstr(
    torch.inverse,
    r"""
    inverse(input, *, out=None) -> Tensor

    Alias for :func:`torch.linalg.inv`
    """,
)

# 为 torch.isin 函数添加文档字符串，说明其功能和用法
add_docstr(
    torch.isin,
    r"""
    isin(elements, test_elements, *, assume_unique=False, invert=False) -> Tensor

    Tests if each element of :attr:`elements` is in :attr:`test_elements`. Returns
    a boolean tensor of the same shape as :attr:`elements` that is True for elements
    in :attr:`test_elements` and False otherwise.

    .. note::
        One of :attr:`elements` or :attr:`test_elements` can be a scalar, but not both.

    Args:
        elements (Tensor or Scalar): Input elements
        test_elements (Tensor or Scalar): Values against which to test for each input element
        assume_unique (bool, optional): If True, assumes both :attr:`elements` and
            :attr:`test_elements` contain unique elements, which can speed up the
            calculation. Default: False
        invert (bool, optional): If True, inverts the boolean return tensor, resulting in True
            values for elements *not* in :attr:`test_elements`. Default: False

    Returns:
        A boolean tensor of the same shape as :attr:`elements` that is True for elements in
        :attr:`test_elements` and False otherwise

    Example:
        >>> torch.isin(torch.tensor([[1, 2], [3, 4]]), torch.tensor([2, 3]))
        tensor([[False,  True],
                [ True, False]])
    """,
)

# 为 torch.isinf 函数添加文档字符串，解释其用途和参数
add_docstr(
    torch.isinf,
    r"""
    isinf(input) -> Tensor

    Tests if each element of :attr:`input` is infinite
    (positive or negative infinity) or not.

    .. note::
        Complex values are infinite when their real or imaginary part is
        infinite.

    Args:
        input (Tensor): The input tensor

    Returns:
        A boolean tensor that is True where :attr:`input` is infinite and False elsewhere

    Example::

        >>> torch.isinf(torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')]))
        tensor([False,  True,  False,  True,  False])
    """,
)

# 为 torch.isposinf 函数添加文档字符串，详细说明其参数和示例
add_docstr(
    torch.isposinf,
    r"""
    isposinf(input, *, out=None) -> Tensor

    Tests if each element of :attr:`input` is positive infinity or not.

    Args:
        input (Tensor): The input tensor

    Keyword args:
        {out}

    Example::

        >>> a = torch.tensor([-float('inf'), float('inf'), 1.2])
        >>> torch.isposinf(a)
        tensor([False,  True, False])
    """.format(
        **common_args
    ),
)

# 为 torch.isneginf 函数添加文档字符串，说明其功能和使用方式
add_docstr(
    torch.isneginf,
    r"""
    isneginf(input, *, out=None) -> Tensor

    Tests if each element of :attr:`input` is negative infinity or not.

    Args:
        input (Tensor): The input tensor

    Keyword args:
        {out}
    """,
)
# 给 torch.isclose 函数添加文档字符串
add_docstr(
    torch.isclose,
    r"""
isclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False) -> Tensor

返回一个新的张量，其中的布尔元素表示每个 :attr:`input` 元素是否与对应的 :attr:`other` 元素“接近”。
接近定义如下：

.. math::
    \lvert \text{input} - \text{other} \rvert \leq \texttt{atol} + \texttt{rtol} \times \lvert \text{other} \rvert

其中 :attr:`input` 和 :attr:`other` 都是有限值。当 :attr:`input` 和/或 :attr:`other` 是非有限值时，
只有在 :attr:`equal_nan` 为 True 时它们才被视为相等，NaN 被认为与其它 NaN 相等。

Args:
    input (Tensor): 要比较的第一个张量
    other (Tensor): 要比较的第二个张量
    atol (float, optional): 绝对容差。默认为 1e-08
    rtol (float, optional): 相对容差。默认为 1e-05
    equal_nan (bool, optional): 如果为 ``True``，则两个 ``NaN`` 将被视为相等。默认为 ``False``

Examples::

    >>> torch.isclose(torch.tensor((1., 2, 3)), torch.tensor((1 + 1e-10, 3, 4)))
    tensor([ True, False, False])
    >>> torch.isclose(torch.tensor((float('inf'), 4)), torch.tensor((float('inf'), 6)), rtol=.5)
    tensor([True, True])
"""
    + r"""
    # 创建一个布尔张量，当 input 张量中的元素是真实值时为 True，其他情况为 False
    A boolean tensor that is True where :attr:`input` is real and False elsewhere
add_docstr(
    torch.isreal,
    r"""
isreal(input) -> Tensor

Returns a boolean tensor of the same shape as :attr:`input` with elements
True if the corresponding element in :attr:`input` is real (i.e., has zero imaginary part),
and False otherwise.

Args:
    {input}

Examples::

    >>> torch.isreal(torch.tensor([1, 1+1j, 2+0j]))
    tensor([True, False, True])
""".format(
        **common_args
    ),
)

add_docstr(
    torch.is_floating_point,
    r"""
is_floating_point(input) -> (bool)

Returns True if the data type of :attr:`input` is a floating point data type i.e.,
one of ``torch.float64``, ``torch.float32``, ``torch.float16``, and ``torch.bfloat16``.

Args:
    {input}
""".format(
        **common_args
    ),
)

add_docstr(
    torch.is_complex,
    r"""
is_complex(input) -> (bool)

Returns True if the data type of :attr:`input` is a complex data type i.e.,
one of ``torch.complex64``, and ``torch.complex128``.

Args:
    {input}
""".format(
        **common_args
    ),
)

add_docstr(
    torch.is_grad_enabled,
    r"""
is_grad_enabled() -> (bool)

Returns True if grad mode is currently enabled.
""".format(
        **common_args
    ),
)

add_docstr(
    torch.is_inference_mode_enabled,
    r"""
is_inference_mode_enabled() -> (bool)

Returns True if inference mode is currently enabled.
""".format(
        **common_args
    ),
)

add_docstr(
    torch.is_inference,
    r"""
is_inference(input) -> (bool)

Returns True if :attr:`input` is an inference tensor.

A non-view tensor is an inference tensor if and only if it was
allocated during inference mode. A view tensor is an inference
tensor if and only if the tensor it is a view of is an inference tensor.

For details on inference mode please see
`Inference Mode <https://pytorch.org/cppdocs/notes/inference_mode.html>`_.

Args:
    {input}
""".format(
        **common_args
    ),
)

add_docstr(
    torch.is_conj,
    r"""
is_conj(input) -> (bool)

Returns True if the :attr:`input` is a conjugated tensor, i.e. its conjugate bit is set to `True`.

Args:
    {input}
""".format(
        **common_args
    ),
)

add_docstr(
    torch.is_nonzero,
    r"""
is_nonzero(input) -> (bool)

Returns True if the :attr:`input` is a single element tensor which is not equal to zero
after type conversions.
i.e. not equal to ``torch.tensor([0.])`` or ``torch.tensor([0])`` or
``torch.tensor([False])``.
Throws a ``RuntimeError`` if ``torch.numel() != 1`` (even in case
of sparse tensors).

Args:
    {input}

Examples::

    >>> torch.is_nonzero(torch.tensor([0.]))
    False
    >>> torch.is_nonzero(torch.tensor([1.5]))
    True
    >>> torch.is_nonzero(torch.tensor([False]))
    False
    >>> torch.is_nonzero(torch.tensor([3]))
    True
    >>> torch.is_nonzero(torch.tensor([1, 3, 5]))
    Traceback (most recent call last):
    ...
    RuntimeError: bool value of Tensor with more than one value is ambiguous
    >>> torch.is_nonzero(torch.tensor([]))
    Traceback (most recent call last):
    ...
    RuntimeError: bool value of Tensor with no values is ambiguous
""".format(
        **common_args
    ),
)

add_docstr(
    torch.kron,
    r"""
kron(input, other, *, out=None) -> Tensor

Computes the Kronecker product, denoted by :math:`\otimes`, of :attr:`input` and :attr:`other`.
""".format(
        **common_args
    ),
)
# 导入必要的库，这里是torch模块
import torch

# 添加文档字符串给torch.kthvalue函数，指定其参数和返回值
def add_docstr(func, docstring):
    """
    将指定函数的文档字符串替换为给定的文档字符串。
    
    Args:
        func (function): 要添加文档字符串的函数对象
        docstring (str): 新的文档字符串
    """

    # 替换函数的文档字符串为给定的文档字符串
    func.__doc__ = docstring

# 替换torch.kthvalue函数的文档字符串为以下内容：
"""
Returns a namedtuple ``(values, indices)`` where ``values`` is the :attr:`k` th
smallest element of each row of the :attr:`input` tensor in the given dimension
:attr:`dim`. And ``indices`` is the index location of each element found.

If :attr:`dim` is not given, the last dimension of the `input` is chosen.

If :attr:`keepdim` is ``True``, both the :attr:`values` and :attr:`indices` tensors
are the same size as :attr:`input`, except in the dimension :attr:`dim` where
they are of size 1. Otherwise, :attr:`dim` is squeezed
(see :func:`torch.squeeze`), resulting in both the :attr:`values` and
:attr:`indices` tensors having 1 fewer dimension than the :attr:`input` tensor.

.. note::
    When :attr:`input` is a CUDA tensor and there are multiple valid
    :attr:`k` th values, this function may nondeterministically return
    :attr:`indices` for any of them.

Args:
    {input}
    k (int): k for the k-th smallest element
"""
    dim (int, optional): the dimension to find the kth value along
    {keepdim}
    r"""
    lerp(input, end, weight, *, out=None)

    Performs linear interpolation between two tensors :attr:`input` and :attr:`end`
    based on a scalar or tensor :attr:`weight`.

    .. math::
        \text{{out}}_i = \text{{input}}_i + \text{{weight}} \times (\text{{end}}_i - \text{{input}}_i)

    Args:
        input (Tensor): the starting point for the interpolation
        end (Tensor): the ending point for the interpolation
        weight (Tensor): the weight for the interpolation

    Keyword args:
        out (Tensor, optional): the output tensor

    Example::

        >>> start = torch.tensor([1.0, 2.0, 3.0])
        >>> end = torch.tensor([4.0, 5.0, 6.0])
        >>> weight = torch.tensor(0.5)
        >>> torch.lerp(start, end, weight)
        tensor([2.5000, 3.5000, 4.5000])

        >>> weight = torch.tensor([0.25, 0.5, 0.75])
        >>> torch.lerp(start, end, weight)
        tensor([1.7500, 3.5000, 5.2500])

    """
# 使用线性插值算法，根据权重参数对输入的起始点张量 `start` 和结束点张量 `end` 进行插值计算，并返回结果张量 `out`
"""
    + r"""
# 张量 `start` 和 `end` 的形状必须是可广播的。如果 `weight` 是张量，则 `weight`、`start` 和 `end` 的形状必须是可广播的。
Args:
    input (Tensor): 起始点张量
    end (Tensor): 结束点张量
    weight (float or tensor): 插值公式中的权重参数

Keyword args:
    {out}  # 输出张量的描述信息，根据具体函数的实现可能有不同的关键字参数

Example::

    >>> start = torch.arange(1., 5.)
    >>> end = torch.empty(4).fill_(10)
    >>> start
    tensor([ 1.,  2.,  3.,  4.])
    >>> end
    tensor([ 10.,  10.,  10.,  10.])
    >>> torch.lerp(start, end, 0.5)
    tensor([ 5.5000,  6.0000,  6.5000,  7.0000])
    >>> torch.lerp(start, end, torch.full_like(start, 0.5))
    tensor([ 5.5000,  6.0000,  6.5000,  7.0000])
""".format(
        **common_args
    ),
)

add_docstr(
    torch.lgamma,
    r"""
lgamma(input, *, out=None) -> Tensor

计算 :attr:`input` 上伽玛函数绝对值的自然对数。

.. math::
    \text{out}_{i} = \ln |\Gamma(\text{input}_{i})|
"""
    + """
Args:
    {input}

Keyword args:
    {out}

Example::

    >>> a = torch.arange(0.5, 2, 0.5)
    >>> torch.lgamma(a)
    tensor([ 0.5724,  0.0000, -0.1208])
""".format(
        **common_args
    ),
)

add_docstr(
    torch.linspace,
    r"""
linspace(start, end, steps, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

创建一个大小为 :attr:`steps` 的一维张量，其值从 :attr:`start` 到 :attr:`end` 均匀分布。即值为：

.. math::
    (\text{start},
    \text{start} + \frac{\text{end} - \text{start}}{\text{steps} - 1},
    \ldots,
    \text{start} + (\text{steps} - 2) * \frac{\text{end} - \text{start}}{\text{steps} - 1},
    \text{end})
"""
    + """

从 PyTorch 1.11 开始，linspace 需要 steps 参数。使用 steps=100 可以恢复之前的行为。

Args:
    start (float or Tensor): 点集的起始值。如果是 `Tensor`，必须是 0 维的。
    end (float or Tensor): 点集的结束值。如果是 `Tensor`，必须是 0 维的。
    steps (int): 构造张量的大小

Keyword arguments:
    {out}
    dtype (torch.dtype, optional): 计算时使用的数据类型。
        默认情况下，如果 `start` 和 `end` 都是实数，则使用全局默认的数据类型 (见 torch.get_default_dtype())；
        如果其中一个是复数，则使用相应的复数数据类型。
    {layout}
    {device}
    {requires_grad}

Example::

    >>> torch.linspace(3, 10, steps=5)
    # 创建一个包含指定范围内均匀间隔的数值的张量，包括起始值和结束值
    tensor([  3.0000,   4.7500,   6.5000,   8.2500,  10.0000])
    >>> torch.linspace(-10, 10, steps=5)
    # 创建一个包含指定范围内均匀间隔的数值的张量，起始值为-10，结束值为10，共5个步骤
    tensor([-10.,  -5.,   0.,   5.,  10.])
    >>> torch.linspace(start=-10, end=10, steps=5)
    # 同样是创建一个包含指定范围内均匀间隔的数值的张量，起始值为-10，结束值为10，共5个步骤
    tensor([-10.,  -5.,   0.,   5.,  10.])
    >>> torch.linspace(start=-10, end=10, steps=1)
    # 创建一个包含指定范围内均匀间隔的数值的张量，起始值为-10，结束值为10，但只有1个步骤，因此只包含起始值
    tensor([-10.])
# 将格式化的字符串添加为函数的文档字符串
add_docstr(
    torch.log,
    r"""
log(input, *, out=None) -> Tensor

Returns a new tensor with the natural logarithm of the elements
of :attr:`input`.

.. math::
    y_{i} = \log_{e} (x_{i})
"""
    + r"""

Args:
    {input}  # 描述输入参数 `input`

Keyword args:
    {out}  # 描述关键字参数 `out`

Example::

    >>> a = torch.rand(5) * 5
    >>> a
    tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739])
    >>> torch.log(a)
    tensor([ 1.5637,  1.4640,  0.1952, -1.4226,  1.5204])
""".format(
        **common_args  # 插入常见参数的值
    ),
)

# 将格式化的字符串添加为函数的文档字符串
add_docstr(
    torch.log10,
    r"""
log10(input, *, out=None) -> Tensor

Returns a new tensor with the logarithm to the base 10 of the elements
of :attr:`input`.

.. math::
    y_{i} = \log_{10} (x_{i})
"""
    + r"""

Args:
    {input}  # 描述输入参数 `input`

Keyword args:
    {out}  # 描述关键字参数 `out`

Example::

    >>> a = torch.rand(5)
    >>> a
    tensor([ 0.5224,  0.9354,  0.7257,  0.1301,  0.2251])


    >>> torch.log10(a)
    tensor([-0.2820, -0.0290, -0.1392, -0.8857, -0.6476])

""".format(
        **common_args  # 插入常见参数的值
    ),
)

# 将格式化的字符串添加为函数的文档字符串
add_docstr(
    torch.log1p,
    r"""
log1p(input, *, out=None) -> Tensor

Returns a new tensor with the natural logarithm of (1 + :attr:`input`).

.. math::
    y_i = \log_{e} (x_i + 1)
"""
    + r"""
.. note:: This function is more accurate than :func:`torch.log` for small
          values of :attr:`input`

Args:
    {input}  # 描述输入参数 `input`

Keyword args:
    {out}  # 描述关键字参数 `out`

Example::

    >>> a = torch.randn(5)
    >>> a
    tensor([-1.0090, -0.9923,  1.0249, -0.5372,  0.2492])
    >>> torch.log1p(a)
    tensor([    nan, -4.8653,  0.7055, -0.7705,  0.2225])
""".format(
        **common_args  # 插入常见参数的值
    ),
)

# 将格式化的字符串添加为函数的文档字符串
add_docstr(
    torch.log2,
    r"""
log2(input, *, out=None) -> Tensor

Returns a new tensor with the logarithm to the base 2 of the elements
of :attr:`input`.

.. math::
    y_{i} = \log_{2} (x_{i})
"""
    + r"""

Args:
    {input}  # 描述输入参数 `input`

Keyword args:
    {out}  # 描述关键字参数 `out`

Example::

    >>> a = torch.rand(5)
    >>> a
    tensor([ 0.8419,  0.8003,  0.9971,  0.5287,  0.0490])


    >>> torch.log2(a)
    tensor([-0.2483, -0.3213, -0.0042, -0.9196, -4.3504])

""".format(
        **common_args  # 插入常见参数的值
    ),
)

# 将格式化的字符串添加为函数的文档字符串
add_docstr(
    torch.logaddexp,
    r"""
logaddexp(input, other, *, out=None) -> Tensor

Logarithm of the sum of exponentiations of the inputs.

Calculates pointwise :math:`\log\left(e^x + e^y\right)`. This function is useful
in statistics where the calculated probabilities of events may be so small as to
exceed the range of normal floating point numbers. In such cases the logarithm
of the calculated probability is stored. This function allows adding
probabilities stored in such a fashion.

This op should be disambiguated with :func:`torch.logsumexp` which performs a
reduction on a single tensor.

Args:
    {input}  # 描述输入参数 `input`
    other (Tensor): the second input tensor

Keyword arguments:
    {out}  # 描述关键字参数 `out`

Example::

    >>> torch.logaddexp(torch.tensor([-1.0]), torch.tensor([-1.0, -2, -3]))
    tensor([-0.3069, -0.6867, -0.8731])

""".format(
        **common_args  # 插入常见参数的值
    ),
)
    # 使用 PyTorch 中的 logaddexp 函数计算两个张量的对数和指数和
    >>> torch.logaddexp(torch.tensor([-100.0, -200, -300]), torch.tensor([-1.0, -2, -3]))
    # 返回一个张量，其中包含每对输入张量元素的 logaddexp 结果
    tensor([-1., -2., -3.])
    
    >>> torch.logaddexp(torch.tensor([1.0, 2000, 30000]), torch.tensor([-1.0, -2, -3]))
    # 返回一个张量，其中包含每对输入张量元素的 logaddexp 结果
    tensor([1.1269e+00, 2.0000e+03, 3.0000e+04])
add_docstr(
    torch.logaddexp2,
    r"""
logaddexp2(input, other, *, out=None) -> Tensor

Logarithm of the sum of exponentiations of the inputs in base-2.

Calculates pointwise :math:`\log_2\left(2^x + 2^y\right)`. See
:func:`torch.logaddexp` for more details.

Args:
    {input}  # 输入张量，作为第一个输入
    other (Tensor): the second input tensor  # 第二个输入张量

Keyword arguments:
    {out}  # 可选，输出张量

""".format(
        **common_args
    ),
)

add_docstr(
    torch.xlogy,
    r"""
xlogy(input, other, *, out=None) -> Tensor

Alias for :func:`torch.special.xlogy`.
""",
)

add_docstr(
    torch.logical_and,
    r"""
logical_and(input, other, *, out=None) -> Tensor

Computes the element-wise logical AND of the given input tensors. Zeros are treated as ``False`` and nonzeros are
treated as ``True``.

Args:
    {input}  # 第一个输入张量
    other (Tensor): the tensor to compute AND with  # 第二个输入张量

Keyword args:
    {out}  # 可选，输出张量

Example::

    >>> torch.logical_and(torch.tensor([True, False, True]), torch.tensor([True, False, False]))
    tensor([ True, False, False])
    >>> a = torch.tensor([0, 1, 10, 0], dtype=torch.int8)
    >>> b = torch.tensor([4, 0, 1, 0], dtype=torch.int8)
    >>> torch.logical_and(a, b)
    tensor([False, False,  True, False])
    >>> torch.logical_and(a.double(), b.double())
    tensor([False, False,  True, False])
    >>> torch.logical_and(a.double(), b)
    tensor([False, False,  True, False])
    >>> torch.logical_and(a, b, out=torch.empty(4, dtype=torch.bool))
    tensor([False, False,  True, False])
""".format(
        **common_args
    ),
)

add_docstr(
    torch.logical_not,
    r"""
logical_not(input, *, out=None) -> Tensor

Computes the element-wise logical NOT of the given input tensor. If not specified, the output tensor will have the bool
dtype. If the input tensor is not a bool tensor, zeros are treated as ``False`` and non-zeros are treated as ``True``.

Args:
    {input}  # 输入张量

Keyword args:
    {out}  # 可选，输出张量

Example::

    >>> torch.logical_not(torch.tensor([True, False]))
    tensor([False,  True])
    >>> torch.logical_not(torch.tensor([0, 1, -10], dtype=torch.int8))
    tensor([ True, False, False])
    >>> torch.logical_not(torch.tensor([0., 1.5, -10.], dtype=torch.double))
    tensor([ True, False, False])
    >>> torch.logical_not(torch.tensor([0., 1., -10.], dtype=torch.double), out=torch.empty(3, dtype=torch.int16))
    tensor([1, 0, 0], dtype=torch.int16)
""".format(
        **common_args
    ),
)

add_docstr(
    torch.logical_or,
    r"""
logical_or(input, other, *, out=None) -> Tensor

Computes the element-wise logical OR of the given input tensors. Zeros are treated as ``False`` and nonzeros are
treated as ``True``.

Args:
    {input}  # 第一个输入张量
    other (Tensor): the tensor to compute OR with  # 第二个输入张量

Keyword args:
    {out}  # 可选，输出张量

Example::

    >>> torch.logical_or(torch.tensor([True, False, True]), torch.tensor([True, False, False]))
    tensor([ True, False,  True])
    >>> a = torch.tensor([0, 1, 10, 0], dtype=torch.int8)
    # 创建一个包含整数的 PyTorch 张量 `a`
    >>> a = torch.tensor([1, 1, 1, 0], dtype=torch.bool)
    # 使用 torch.logical_or 函数计算逐元素的逻辑或操作，返回一个张量
    >>> torch.logical_or(a, b)
    # 结果张量显示每个对应位置上的逻辑或结果
    tensor([ True,  True,  True, False])
    # 将张量 `a` 和 `b` 转换为双精度浮点数类型后进行逻辑或操作
    >>> torch.logical_or(a.double(), b.double())
    # 返回结果张量显示每个对应位置上的逻辑或结果
    tensor([ True,  True,  True, False])
    # 将张量 `a` 转换为双精度浮点数类型后与张量 `b` 进行逻辑或操作
    >>> torch.logical_or(a.double(), b)
    # 返回结果张量显示每个对应位置上的逻辑或结果
    tensor([ True,  True,  True, False])
    # 使用 torch.logical_or 函数计算逐元素的逻辑或操作，并将结果存储到预先创建的空张量中
    >>> torch.logical_or(a, b, out=torch.empty(4, dtype=torch.bool))
    # 返回结果张量显示每个对应位置上的逻辑或结果
    tensor([ True,  True,  True, False])
"""
add_docstr(
    torch.logical_xor,
    r"""
logical_xor(input, other, *, out=None) -> Tensor

Computes the element-wise logical XOR of the given input tensors. Zeros are treated as ``False`` and nonzeros are
treated as ``True``.

Args:
    {input}  # 描述第一个参数 input，接受的数据类型和含义
    other (Tensor): the tensor to compute XOR with  # 描述第二个参数 other，接受的数据类型和含义

Keyword args:
    {out}  # 描述关键字参数 out 的作用和使用方式

Example::

    >>> torch.logical_xor(torch.tensor([True, False, True]), torch.tensor([True, False, False]))
    tensor([False, False,  True])
    >>> a = torch.tensor([0, 1, 10, 0], dtype=torch.int8)
    >>> b = torch.tensor([4, 0, 1, 0], dtype=torch.int8)
    >>> torch.logical_xor(a, b)
    tensor([ True,  True, False, False])
    >>> torch.logical_xor(a.double(), b.double())
    tensor([ True,  True, False, False])
    >>> torch.logical_xor(a.double(), b)
    tensor([ True,  True, False, False])
    >>> torch.logical_xor(a, b, out=torch.empty(4, dtype=torch.bool))
    tensor([ True,  True, False, False])
""".format(
        **common_args
    ),
)

add_docstr(
    torch.logspace,
    """
logspace(start, end, steps, base=10.0, *, \
         out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
"""
    + r"""

Creates a one-dimensional tensor of size :attr:`steps` whose values are evenly
spaced from :math:`{{\text{{base}}}}^{{\text{{start}}}}` to
:math:`{{\text{{base}}}}^{{\text{{end}}}}`, inclusive, on a logarithmic scale
with base :attr:`base`. That is, the values are:

.. math::
    (\text{base}^{\text{start}},
    \text{base}^{(\text{start} + \frac{\text{end} - \text{start}}{ \text{steps} - 1})},
    \ldots,
    \text{base}^{(\text{start} + (\text{steps} - 2) * \frac{\text{end} - \text{start}}{ \text{steps} - 1})},
    \text{base}^{\text{end}})
"""


From PyTorch 1.11 logspace requires the steps argument. Use steps=100 to restore the previous behavior.

Args:
    start (float or Tensor): the starting value for the set of points. If `Tensor`, it must be 0-dimensional
    end (float or Tensor): the ending value for the set of points. If `Tensor`, it must be 0-dimensional
    steps (int): size of the constructed tensor
    base (float, optional): base of the logarithm function. Default: ``10.0``.

Keyword arguments:
    {out}  # 描述关键字参数 out 的作用和使用方式
    dtype (torch.dtype, optional): the data type to perform the computation in.
        Default: if None, uses the global default dtype (see torch.get_default_dtype())
        when both :attr:`start` and :attr:`end` are real,
        and corresponding complex dtype when either is complex.
    {layout}  # 描述关键字参数 layout 的作用和使用方式
    {device}  # 描述关键字参数 device 的作用和使用方式
    {requires_grad}  # 描述关键字参数 requires_grad 的作用和使用方式

Example::

    >>> torch.logspace(start=-10, end=10, steps=5)  # 示例用法及结果
    tensor([ 1.0000e-10,  1.0000e-05,  1.0000e+00,  1.0000e+05,  1.0000e+10])
    >>> torch.logspace(start=0.1, end=1.0, steps=5)
    tensor([  1.2589,   2.1135,   3.5481,   5.9566,  10.0000])
    >>> torch.logspace(start=0.1, end=1.0, steps=1)
    tensor([1.2589])
    >>> torch.logspace(start=2, end=2, steps=1, base=2)  # 针对特定参数 base 的示例用法
"""
    # 创建一个包含单个元素 4.0 的张量
    tensor([4.0])
""".format(
        **factory_common_args
    ),
)

add_docstr(
    torch.logsumexp,
    r"""
logsumexp(input, dim, keepdim=False, *, out=None)

Returns the log of summed exponentials of each row of the :attr:`input`
tensor in the given dimension :attr:`dim`. The computation is numerically
stabilized.

For summation index :math:`j` given by `dim` and other indices :math:`i`, the result is

    .. math::
        \text{{logsumexp}}(x)_{{i}} = \log \sum_j \exp(x_{{ij}})

{keepdim_details}

Args:
    {input}
    {opt_dim}
    {keepdim}

Keyword args:
    {out}

Example::

    >>> a = torch.randn(3, 3)
    >>> torch.logsumexp(a, 1)
    tensor([1.4907, 1.0593, 1.5696])
    >>> torch.dist(torch.logsumexp(a, 1), torch.log(torch.sum(torch.exp(a), 1)))
    tensor(1.6859e-07)
""".format(
        **multi_dim_common
    ),
)

add_docstr(
    torch.lt,
    r"""
lt(input, other, *, out=None) -> Tensor

Computes :math:`\text{input} < \text{other}` element-wise.
"""
    + r"""

The second argument can be a number or a tensor whose shape is
:ref:`broadcastable <broadcasting-semantics>` with the first argument.

Args:
    input (Tensor): the tensor to compare
    other (Tensor or float): the tensor or value to compare

Keyword args:
    {out}

Returns:
    A boolean tensor that is True where :attr:`input` is less than :attr:`other` and False elsewhere

Example::

    >>> torch.lt(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
    tensor([[False, False], [True, False]])
""".format(
        **common_args
    ),
)

add_docstr(
    torch.lu_unpack,
    r"""
lu_unpack(LU_data, LU_pivots, unpack_data=True, unpack_pivots=True, *, out=None) -> (Tensor, Tensor, Tensor)

Unpacks the LU decomposition returned by :func:`~linalg.lu_factor` into the `P, L, U` matrices.

.. seealso::

    :func:`~linalg.lu` returns the matrices from the LU decomposition. Its gradient formula is more efficient
    than that of doing :func:`~linalg.lu_factor` followed by :func:`~linalg.lu_unpack`.

Args:
    LU_data (Tensor): the packed LU factorization data
    LU_pivots (Tensor): the packed LU factorization pivots
    unpack_data (bool): flag indicating if the data should be unpacked.
                        If ``False``, then the returned ``L`` and ``U`` are empty tensors.
                        Default: ``True``
    unpack_pivots (bool): flag indicating if the pivots should be unpacked into a permutation matrix ``P``.
                          If ``False``, then the returned ``P`` is  an empty tensor.
                          Default: ``True``

Keyword args:
    out (tuple, optional): output tuple of three tensors. Ignored if `None`.

Returns:
    A namedtuple ``(P, L, U)``

Examples::

    >>> A = torch.randn(2, 3, 3)
    >>> LU, pivots = torch.linalg.lu_factor(A)
    >>> P, L, U = torch.lu_unpack(LU, pivots)
    >>> # We can recover A from the factorization
    >>> A_ = P @ L @ U
    >>> torch.allclose(A, A_)
    True

    >>> # LU factorization of a rectangular matrix:
    >>> A = torch.randn(2, 3, 2)
    # 创建一个大小为 (2, 3, 2) 的张量 A，其中元素为随机数

    >>> LU, pivots = torch.linalg.lu_factor(A)
    # 对张量 A 进行 LU 分解，返回 LU 分解结果 LU 和置换矩阵 pivots

    >>> P, L, U = torch.lu_unpack(LU, pivots)
    # 根据 LU 分解结果 LU 和置换矩阵 pivots，解包得到置换矩阵 P，下三角矩阵 L 和上三角矩阵 U

    >>> # P, L, U are the same as returned by linalg.lu
    # 检查使用 torch.linalg.lu 函数和 torch.lu_unpack 函数得到的结果是否与使用 torch.linalg.lu 函数直接得到的结果相同

    >>> P_, L_, U_ = torch.linalg.lu(A)
    # 直接使用 torch.linalg.lu 函数对张量 A 进行 LU 分解，得到置换矩阵 P_，下三角矩阵 L_ 和上三角矩阵 U_

    >>> torch.allclose(P, P_) and torch.allclose(L, L_) and torch.allclose(U, U_)
    # 检查解包后的 P、L、U 是否与直接分解得到的 P_、L_、U_ 在数值上是否相近（使用 allclose 函数）

    True
    # 返回 True，表示两种方法得到的 LU 分解结果 P、L、U 是相同的
add_docstr(
    torch.matrix_power,
    r"""
matrix_power(input, n, *, out=None) -> Tensor

Compute the matrix power of a square matrix :attr:`input` raised to the power :attr:`n`.

Arguments:
    input (Tensor): the input tensor of shape :math:`(*, m, m)` where :math:`*` is zero or more batch dimensions.
    n (int): the exponent to which the matrix is raised

Keyword args:
    {out}

Returns:
    Tensor: the matrix power of shape :math:`(*, m, m)`.

Example::

    >>> A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    >>> torch.matrix_power(A, 2)
    tensor([[ 7., 10.],
            [15., 22.]])
""",
)
add_docstr(
    torch.linalg.matrix_power,
    r"""
matrix_power(input, n) -> Tensor

Alias for :func:`torch.linalg.matrix_power`.
""",
)

add_docstr(
    torch.matrix_exp,
    r"""
matrix_exp(A) -> Tensor

Alias for :func:`torch.linalg.matrix_exp`.
""",
)

add_docstr(
    torch.max,
    r"""
max(input) -> Tensor

Returns the maximum value of all elements in the ``input`` tensor.

.. warning::
    This function produces deterministic (sub)gradients unlike ``max(dim=0)``

Args:
    {input}

Example::

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[ 0.6763,  0.7445, -2.2369]])
    >>> torch.max(a)
    tensor(0.7445)

""".format(
        **single_dim_common
    ),
)

add_docstr(
    torch.max,
    r"""
.. function:: max(input, dim, keepdim=False, *, out=None) -> (Tensor, LongTensor)
   :noindex:

Returns a namedtuple ``(values, indices)`` where ``values`` is the maximum
value of each row of the :attr:`input` tensor in the given dimension
:attr:`dim`. And ``indices`` is the index location of each maximum value found
(argmax).

If ``keepdim`` is ``True``, the output tensors are of the same size
as ``input`` except in the dimension ``dim`` where they are of size 1.
Otherwise, ``dim`` is squeezed (see :func:`torch.squeeze`), resulting
in the output tensors having 1 fewer dimension than ``input``.

.. note:: If there are multiple maximal values in a reduced row then
          the indices of the first maximal value are returned.

Args:
    {input}
    {dim}
    {keepdim} Default: ``False``.

Keyword args:
    out (tuple, optional): the result tuple of two output tensors (max, max_indices)

Example::

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[-1.2360, -0.2942, -0.1222,  0.8475],
            [ 1.1949, -1.1127, -2.2379, -0.6702],
            [ 1.5717, -0.9207,  0.1297, -1.8768],
            [-0.6172,  1.0036, -0.6060, -0.2432]])
    >>> torch.max(a, 1)
    torch.return_types.max(values=tensor([0.8475, 1.1949, 1.5717, 1.0036]), indices=tensor([3, 0, 0, 1]))

""".format(
        **single_dim_common
    ),
)

add_docstr(
    torch.max,
    r"""
.. function:: max(input, other, *, out=None) -> Tensor
   :noindex:

See :func:`torch.maximum`.

""".format(
        **single_dim_common
    ),
)

add_docstr(
    torch.maximum,
    r"""
maximum(input, other, *, out=None) -> Tensor

Computes the element-wise maximum of :attr:`input` and :attr:`other`.

.. note::
    If one of the elements being compared is a NaN, then that element is returned.
    :func:`maximum` is not supported for tensors with complex dtypes.

Args:
    {input}
    other (Tensor): the second input tensor

Keyword args:
    {out}

Example::

    >>> a = torch.tensor((1, 2, -1))
    >>> b = torch.tensor((3, 0, 4))
    >>> torch.maximum(a, b)
    tensor([3, 2, 4])
""".format(
        **common_args
    ),
)

add_docstr(
    torch.fmax,
    r"""
fmax(input, other, *, out=None) -> Tensor

Computes the element-wise maximum of :attr:`input` and :attr:`other`.

This is like :func:`torch.maximum` except it handles NaNs differently:
if exactly one of the two elements being compared is a NaN then the non-NaN element is taken as the maximum.
Only if both elements are NaN is NaN propagated.
""",
)
#`
# 为 torch.fmax 函数添加文档字符串，介绍其功能和使用方法
This function is a wrapper around C++'s ``std::fmax`` and is similar to NumPy's ``fmax`` function.

# 支持广播到公共形状，类型提升，以及整数和浮点输入
Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`,
:ref:`type promotion <type-promotion-doc>`, and integer and floating-point inputs.

# 参数说明
Args:
    {input}  # 第一个输入张量
    other (Tensor): the second input tensor  # 第二个输入张量

# 关键字参数说明
Keyword args:
    {out}  # 输出张量的指定位置

# 示例代码，展示了如何使用 torch.fmax 函数
Example::
    >>> a = torch.tensor([9.7, float('nan'), 3.1, float('nan')])
    >>> b = torch.tensor([-2.2, 0.5, float('nan'), float('nan')])
    >>> torch.fmax(a, b)
    tensor([9.7000, 0.5000, 3.1000,    nan])
""".format(
        **common_args  # 填充通用参数
    ),
)

# 为 torch.amax 函数添加文档字符串，详细描述其功能和使用方法
add_docstr(
    torch.amax,
    r"""
amax(input, dim, keepdim=False, *, out=None) -> Tensor

# 返回指定维度上的张量切片的最大值
Returns the maximum value of each slice of the :attr:`input` tensor in the given
dimension(s) :attr:`dim`.

.. note::
    The difference between ``max``/``min`` and ``amax``/``amin`` is:
        - ``amax``/``amin`` supports reducing on multiple dimensions,
        - ``amax``/``amin`` does not return indices,
        - ``amax``/``amin`` evenly distributes gradient between equal values,
          while ``max(dim)``/``min(dim)`` propagates gradient only to a single
          index in the source tensor.

{keepdim_details}  # 填充 keepdim 参数的详细说明

Args:
    {input}  # 输入张量
    {dim}  # 指定维度
    {keepdim}  # 保持维度

# 关键字参数说明
Keyword args:
  {out}  # 输出张量的指定位置

# 示例代码，展示了如何使用 torch.amax 函数
Example::
    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 0.8177,  1.4878, -0.2491,  0.9130],
            [-0.7158,  1.1775,  2.0992,  0.4817],
            [-0.0053,  0.0164, -1.3738, -0.0507],
            [ 1.9700,  1.1106, -1.0318, -1.0816]])
    >>> torch.amax(a, 1)
    tensor([1.4878, 2.0992, 0.0164, 1.9700])
""".format(
        **multi_dim_common  # 填充多维常用参数
    ),
)

# 为 torch.argmax 函数添加文档字符串，说明其功能和使用方法
add_docstr(
    torch.argmax,
    r"""
argmax(input) -> LongTensor

# 返回输入张量中所有元素的最大值的索引
Returns the indices of the maximum value of all elements in the :attr:`input` tensor.

This is the second value returned by :meth:`torch.max`. See its
documentation for the exact semantics of this method.

.. note:: If there are multiple maximal values then the indices of the first maximal value are returned.

Args:
    {input}  # 输入张量

# 示例代码，展示了如何使用 torch.argmax 函数
Example::
    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 1.3398,  0.2663, -0.2686,  0.2450],
            [-0.7401, -0.8805, -0.3402, -1.1936],
            [ 0.4907, -1.3948, -1.0691, -0.3132],
            [-1.6092,  0.5419, -0.2993,  0.3195]])
    >>> torch.argmax(a)
    tensor(0)

.. function:: argmax(input, dim, keepdim=False) -> LongTensor
   :noindex:

# 返回张量在指定维度上的最大值的索引
Returns the indices of the maximum values of a tensor across a dimension.

This is the second value returned by :meth:`torch.max`. See its
documentation for the exact semantics of this method.

Args:
    {input}  # 输入张量
    {dim} If ``None``, the argmax of the flattened input is returned.  # 指定维度，如果为 None，则返回扁平化输入的 argmax
    {keepdim}  # 是否保持维度

# 示例代码，展示了如何使用 torch.argmax 函数在指定维度上获取最大值索引
Example::
    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 1.3398,  0.2663, -0.2686,  0.2450],
            [-0.7401, -0.8805, -0.3402, -1.1936],
            [ 0.4907, -1.3948, -1.0691, -0.3132],
            [-1.6092,  0.5419, -0.2993,  0.3195]])
""")
    # 使用 PyTorch 中的 argmax 函数来计算张量 a 沿着第一个维度的最大值索引
    >>> torch.argmax(a, dim=1)
    # 返回一个张量，包含每行最大值的索引
    tensor([ 0,  2,  0,  1])
""".format(
        **single_dim_common
    ),
)

add_docstr(
    torch.argwhere,
    r"""
argwhere(input) -> Tensor

Returns a tensor containing the indices of all non-zero elements of
:attr:`input`.  Each row in the result contains the indices of a non-zero
element in :attr:`input`. The result is sorted lexicographically, with
the last index changing the fastest (C-style).

If :attr:`input` has :math:`n` dimensions, then the resulting indices tensor
:attr:`out` is of size :math:`(z \times n)`, where :math:`z` is the total number of
non-zero elements in the :attr:`input` tensor.

.. note::
    This function is similar to NumPy's `argwhere`.

    When :attr:`input` is on CUDA, this function causes host-device synchronization.

Args:
    input (Tensor):
      the input tensor, typically containing non-zero elements

Example::

    >>> t = torch.tensor([1, 0, 1])
    >>> torch.argwhere(t)
    tensor([[0],
            [2]])
    >>> t = torch.tensor([[1, 0, 1], [0, 1, 1]])
    >>> torch.argwhere(t)
    tensor([[0, 0],
            [0, 2],
            [1, 1],
            [1, 2]])
""",
)

add_docstr(
    torch.mean,
    r"""
mean(input, *, dtype=None) -> Tensor

Returns the mean value of all elements in the :attr:`input` tensor. Input must be floating point or complex.

Args:
    input (Tensor):
      the input tensor, either of floating point or complex dtype

Keyword args:
    dtype (torch.dtype, optional):
      the desired data type of the returned tensor. If specified, the input tensor is casted to :attr:`dtype` before the operation is performed.

Example::

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[ 0.2294, -0.5481,  1.3288]])
    >>> torch.mean(a)
    tensor(0.3367)

.. function:: mean(input, dim, keepdim=False, *, dtype=None, out=None) -> Tensor
   :noindex:

Returns the mean value of each row of the :attr:`input` tensor in the given
dimension :attr:`dim`. If :attr:`dim` is a list of dimensions,
reduce over all of them.

Args:
    input (Tensor):
      the input tensor, either of floating point or complex dtype
    dim (int or tuple of ints):
      the dimension or dimensions to reduce. If `dim` is a single integer, it specifies the dimension to reduce. If `dim` is a tuple of integers, reduce over all specified dimensions.
    keepdim (bool, optional):
      whether the output tensor has :attr:`dim` retained or not. Default: False

Keyword args:
    dtype (torch.dtype, optional):
      the desired data type of the returned tensor. If specified, the input tensor is casted to :attr:`dtype` before the operation is performed.
    out (Tensor, optional):
      the output tensor. If specified, the result will be stored in this tensor.

.. seealso::

    :func:`torch.nanmean` computes the mean value of `non-NaN` elements.

Example::

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[-0.3841,  0.6320,  0.4254, -0.7384],
            [-0.9644,  1.0131, -0.6549, -1.4279],
            [-0.2951, -1.3350, -0.7694,  0.5600],
            [ 1.0842, -0.9580,  0.3623,  0.2343]])
    >>> torch.mean(a, 1)
    tensor([-0.0163, -0.5085, -0.4599,  0.1807])
    >>> torch.mean(a, 1, True)
    tensor([[-0.0163],
            [-0.5085],
            [-0.4599],
            [ 0.1807]])
""".format(
        **multi_dim_common
    ),
)

add_docstr(
    torch.nanmean,
    r"""
nanmean(input, dim=None, keepdim=False, *, dtype=None, out=None) -> Tensor

Computes the mean of all `non-NaN` elements along the specified dimensions.
Input must be floating point or complex.

This function is identical to :func:`torch.mean` when there are no `NaN` values
in the :attr:`input` tensor. In the presence of `NaN`, :func:`torch.mean` will
propagate the `NaN` to the output whereas :func:`torch.nanmean` will ignore the
`NaN` values (`torch.nanmean(a)` is equivalent to `torch.mean(a[~a.isnan()])`).

Args:
    input (Tensor):
      the input tensor, typically containing floating point or complex data
    dim (int or tuple of ints, optional):
      the dimension or dimensions to reduce. If `dim` is None, the mean is computed over all dimensions of the input tensor.
    keepdim (bool, optional):
      whether the output tensor has :attr:`dim` retained or not. Default: False

Keyword args:
    dtype (torch.dtype, optional):
      the desired data type of the returned tensor. If specified, the input tensor is casted to :attr:`dtype` before the operation is performed.
    out (Tensor, optional):
      the output tensor. If specified, the result will be stored in this tensor.

Example::

    >>> a = torch.tensor([[1.0, float('nan')], [2.0, 3.0]])
    >>> torch.nanmean(a)
    tensor(2.0000)
    >>> torch.nanmean(a, dim=0)
    tensor([1.5000, 3.0000])
    >>> torch.nanmean(a, dim=1)
    tensor([1.0000, 2.5000])
""",
)
    # 接受一个张量作为输入，可以是浮点数或复数类型
    input (Tensor): the input tensor, either of floating point or complex dtype
    # 可选参数，表示操作的维度信息，通常用花括号 {} 表示
    {opt_dim}
    # 可选参数，表示是否保持操作后的维度信息不变
    {keepdim}
# 定义了关键字参数 {dtype} 和 {out}，用于文档字符串的格式化

# seealso 部分提到了与 torch.mean 相关的信息，可以查看该函数来了解如何处理 NaN 值

# 示例部分展示了 torch.mean 和 torch.median 的用法示例，展示了处理 NaN 值的情况

add_docstr(
    torch.median,
    r"""
median(input) -> Tensor

返回张量 :attr:`input` 中数值的中位数。

.. note::
    对于具有偶数个元素的 :attr:`input` 张量，中位数不唯一。在这种情况下，返回较小的中位数。
    若要计算两个中位数的平均值，请使用 :func:`torch.quantile` 并设定 ``q=0.5``。

.. warning::
    与 ``median(dim=0)`` 不同，此函数产生确定性的（子）梯度。

Args:
    {input}  # 输入参数，描述了输入张量的含义

Example::

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[ 1.5219, -1.5212,  0.2202]])
    >>> torch.median(a)
    tensor(0.2202)

# function:: median(input, dim=-1, keepdim=False, *, out=None) -> (Tensor, LongTensor)
# :noindex:

返回一个命名元组 ``(values, indices)``，其中 ``values`` 包含 :attr:`input` 张量在维度 :attr:`dim` 的每行的中位数，
而 ``indices`` 包含在维度 :attr:`dim` 找到的中位数值的索引。

默认情况下，:attr:`dim` 是 :attr:`input` 张量的最后一个维度。

如果 :attr:`keepdim` 为 ``True``，输出张量与 :attr:`input` 张量的尺寸相同，除了维度 :attr:`dim` 外，其尺寸为 1。
否则，会挤压 :attr:`dim`（参见 :func:`torch.squeeze`），结果是输出张量的维度比 :attr:`input` 少 1。

.. note::
    对于在维度 :attr:`dim` 中具有偶数个元素的 :attr:`input` 张量，中位数不唯一。在这种情况下，返回较小的中位数。
    若要计算 :attr:`input` 中两个中位数的平均值，请使用 :func:`torch.quantile` 并设定 ``q=0.5``。

.. warning::
    ``indices`` 并不一定包含每个找到的中位数值的第一次出现，除非它是唯一的。
    具体的实现细节是特定于设备的。
    通常情况下，不要期望在 CPU 和 GPU 上运行时得到相同的结果。
    由于同样的原因，不要期望梯度是确定性的。

Args:
    {input}  # 输入参数，描述了输入张量的含义
    {dim}    # 表示计算中位数的维度
    {keepdim}  # 是否保持输出张量的维度结构不变

Keyword args:
    out ((Tensor, Tensor), optional): 第一个张量将包含中位数的值，第二个张量必须具有 long 类型，包含 :attr:`input` 张量在维度 :attr:`dim` 的中位数的索引。

Example::

    >>> a = torch.randn(4, 5)
    >>> a
    # 定义一个名为 `tensor` 的张量，包含多行数据
    tensor([[ 0.2505, -0.3982, -0.9948,  0.3518, -1.3131],
            [ 0.3180, -0.6993,  1.0436,  0.0438,  0.2270],
            [-0.2751,  0.7303,  0.2192,  0.3321,  0.2488],
            [ 1.0778, -1.9510,  0.7048,  0.4742, -0.7125]])
    # 使用 `torch.median` 函数计算张量 `tensor` 每行的中位数
    >>> torch.median(a, 1)
    # 返回 `torch.return_types.median` 对象，包含中位数和其对应的索引
    torch.return_types.median(values=tensor([-0.3982,  0.2270,  0.2488,  0.4742]), indices=tensor([1, 4, 4, 3]))
# 添加文档字符串给 torch.quantile 函数
add_docstr(
    torch.quantile,
    r"""
quantile(input, q, dim=None, keepdim=False, *, interpolation='linear', out=None) -> Tensor

Computes the q-th quantiles of each row of the :attr:`input` tensor along the dimension :attr:`dim`.

To compute the quantile, we map q in [0, 1] to the range of indices [0, n] to find the location
of the quantile in the sorted input. If the quantile lies between two data points ``a < b`` with
indices ``i`` and ``j`` in the sorted order, result is computed according to the given
:attr:`interpolation` method as follows:

- ``linear``: ``a + (b - a) * fraction``, where ``fraction`` is the fractional part of the computed quantile index.
- ``lower``: ``a``.
- ``higher``: ``b``.
""",
)
"""
nanquantile(input, q, dim=None, keepdim=False, *, interpolation='linear', out=None) -> Tensor

This is a variant of :func:`torch.quantile` that "ignores" ``NaN`` values,
computing the quantiles :attr:`q` as if ``NaN`` values in :attr:`input` did
not exist. If all values in a reduced row are ``NaN`` then the quantiles for
that reduction will be ``NaN``. See the documentation for :func:`torch.quantile`.

Args:
    input (Tensor): the input tensor containing data to compute quantiles over.
    q (float or Tensor): a scalar or 1D tensor of quantile values in the range [0, 1]
    dim (int, optional): the dimension over which to compute quantiles. Default is None.
    keepdim (bool, optional): whether the output tensor has dim retained or not. Default is False.

Keyword arguments:
    interpolation (str): interpolation method to use when the desired quantile lies between two data points.
                            Can be ``linear``, ``lower``, ``higher``, ``midpoint`` and ``nearest``.
                            Default is ``linear``.
    out (Tensor, optional): the output tensor. Must be of the same shape as expected output.

Example::

    >>> t = torch.tensor([float('nan'), 1, 2])
    >>> t.quantile(0.5)
    tensor(nan)
    >>> t.nanquantile(0.5)
    tensor(1.5000)
    >>> t = torch.tensor([[float('nan'), float('nan')], [1, 2]])
    >>> t
    tensor([[nan, nan],
            [ 1.,  2.]])

"""
    # 创建一个包含 NaN（Not a Number，非数）的张量，形状为 (2, 2)
    tensor([[nan, nan],
            [1., 2.]])
    # 使用张量 t 的 nanquantile 方法，计算沿着 dim=0 维度的 0.5 分位数（忽略 NaN 值）
    # 返回沿着 dim=0 维度计算的分位数结果张量
    >>> t.nanquantile(0.5, dim=0)
    tensor([1., 2.])
    # 使用张量 t 的 nanquantile 方法，计算沿着 dim=1 维度的 0.5 分位数（忽略 NaN 值）
    # 返回沿着 dim=1 维度计算的分位数结果张量
    >>> t.nanquantile(0.5, dim=1)
    tensor([   nan, 1.5000])
# 定义函数 add_docstr，用于向指定函数或方法添加文档字符串
add_docstr(
    torch.min,
    r"""
min(input) -> Tensor

Returns the minimum value of all elements in the :attr:`input` tensor.

.. warning::
    This function produces deterministic (sub)gradients unlike ``min(dim=0)``

Args:
    {input}  # 描述 input 参数，表示输入的张量

Example::

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[ 0.6750,  1.0857,  1.7197]])
    >>> torch.min(a)
    tensor(0.6750)
""",
)

add_docstr(
    torch.min,
    r"""
min(input, dim, keepdim=False, *, out=None) -> (Tensor, LongTensor)
   :noindex:

Returns a namedtuple ``(values, indices)`` where ``values`` is the minimum
value of each row of the :attr:`input` tensor in the given dimension
:attr:`dim`. And ``indices`` is the index location of each minimum value found
(argmin).

If :attr:`keepdim` is ``True``, the output tensors are of the same size as
:attr:`input` except in the dimension :attr:`dim` where they are of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in
the output tensors having 1 fewer dimension than :attr:`input`.

.. note:: If there are multiple minimal values in a reduced row then
          the indices of the first minimal value are returned.

Args:
    {input}  # 描述 input 参数，表示输入的张量
    {dim}    # 描述 dim 参数，表示指定的维度
    {keepdim}  # 描述 keepdim 参数，指定是否保持维度

Keyword args:
    out (tuple, optional): the tuple of two output tensors (min, min_indices)

Example::

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[-0.6248,  1.1334, -1.1899, -0.2803],
            [-1.4644, -0.2635, -0.3651,  0.6134],
            [ 0.2457,  0.0384,  1.0128,  0.7015],
            [-0.1153,  2.9849,  2.1458,  0.5788]])
    >>> torch.min(a, 1)
    torch.return_types.min(values=tensor([-1.1899, -1.4644,  0.0384, -0.1153]), indices=tensor([2, 0, 1, 0]))
""",
)

add_docstr(
    torch.minimum,
    r"""
minimum(input, other, *, out=None) -> Tensor

Computes the element-wise minimum of :attr:`input` and :attr:`other`.

.. note::
    If one of the elements being compared is a NaN, then that element is returned.
    :func:`minimum` is not supported for tensors with complex dtypes.

Args:
    {input}  # 描述 input 参数，表示第一个输入的张量
    other (Tensor): the second input tensor

Keyword args:
    {out}  # 描述 out 参数，指定输出张量的位置

Example::

    >>> a = torch.tensor((1, 2, -1))
    >>> b = torch.tensor((3, 0, 4))
    >>> torch.minimum(a, b)
    tensor([1, 0, -1])
""",
)

add_docstr(
    torch.fmin,
    r"""
fmin(input, other, *, out=None) -> Tensor

Computes the element-wise minimum of :attr:`input` and :attr:`other`.

This is like :func:`torch.minimum` except it handles NaNs differently:
if exactly one of the two elements being compared is a NaN then the non-NaN element is taken as the minimum.
Only if both elements are NaN is NaN propagated.

This function is a wrapper around C++'s ``std::fmin`` and is similar to NumPy's ``fmin`` function.

Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`,

Args:
    {input}  # 描述 input 参数，表示第一个输入的张量
    other (Tensor): the second input tensor
""",
)
# 定义函数 add_docstr，用于为 torch.aminmax 函数添加文档字符串
add_docstr(
    torch.aminmax,
    r"""
aminmax(input, *, dim=None, keepdim=False, out=None) -> (Tensor min, Tensor max)

Computes the minimum and maximum values of the :attr:`input` tensor.

Args:
    input (Tensor):
        输入张量

Keyword Args:
    dim (Optional[int]):
        计算数值的维度。如果为 `None`，则在整个 :attr:`input` 张量上计算数值。默认为 `None`。
    keepdim (bool):
        如果为 `True`，则保留输出张量中的缩小维度，用于广播。否则，会移除这些维度，就像调用 (:func:`torch.squeeze`) 一样。默认为 `False`。
    out (Optional[Tuple[Tensor, Tensor]]):
        用于写入结果的可选张量。必须与预期输出具有相同的形状和数据类型。默认为 `None`。

Returns:
    包含最小值和最大值的命名元组 `(min, max)`。

Raises:
    RuntimeError:
        如果任何计算数值的维度大小为 0。

.. note::
    如果至少有一个值为 NaN，则 NaN 值会传播到输出。

Example::

    >>> torch.aminmax(torch.tensor([1, -3, 5]))
    torch.return_types.aminmax(
    min=tensor(-3),
    max=tensor(5))

    >>> # aminmax 会传播 NaN
    >>> torch.aminmax(torch.tensor([1, -3, 5, torch.nan]))
    # 创建一个包含最小值和最大值的 torch.return_types.aminmax 对象
    torch.return_types.aminmax(
        # 设置最小值为 NaN
        min=tensor(nan),
        # 设置最大值为 NaN
        max=tensor(nan))
    
    
    # 创建一个张量 t，其中包含从 0 到 9 的整数，并将其形状调整为 2 行 5 列的矩阵
    t = torch.arange(10).view(2, 5)
    # 打印张量 t
    t
    
    
    # 对张量 t 沿着第一个维度（即行）计算最小值和最大值，并保持结果的维度与输入张量相同
    t.aminmax(dim=0, keepdim=True)
    # 创建一个包含最小值和最大值的 torch.return_types.aminmax 对象，其中最小值是一个形状为 (1, 5) 的张量，最大值同样也是
    torch.return_types.aminmax(
        min=tensor([[0, 1, 2, 3, 4]]),
        max=tensor([[5, 6, 7, 8, 9]]))
"""
add_docstr(
    torch.argmin,
    r"""
    argmin(input, dim=None, keepdim=False) -> LongTensor
    
    返回张量中最小值的索引，可以是整个张量的最小值索引或沿着某个维度的最小值索引。
    
    这是 :meth:`torch.min` 的第二个返回值。详细语义请参考其文档。
    
    .. note::
        如果存在多个最小值，则返回第一个最小值的索引。
    
    Args:
        input (Tensor): 输入张量
        dim (int, optional): 沿着哪个维度计算最小值，默认为 None，即计算整个张量的最小值
        keepdim (bool, optional): 结果张量是否保留输入张量的维度，默认为 False
    
    Example::
    
        >>> a = torch.randn(4, 4)
        >>> a
        tensor([[ 0.1139,  0.2254, -0.1381,  0.3687],
                [ 1.0100, -1.1975, -0.0102, -0.4732],
                [-0.9240,  0.1207, -0.7506, -1.0213],
                [ 1.7809, -1.2960,  0.9384,  0.1438]])
        >>> torch.argmin(a)
        tensor(13)
        >>> torch.argmin(a, dim=1)
        tensor([ 2,  1,  3,  1])
        >>> torch.argmin(a, dim=1, keepdim=True)
        tensor([[2],
                [1],
                [3],
                [1]])
    """.format(
        **single_dim_common
    ),
)

add_docstr(
    torch.mm,
    r"""
    mm(input, mat2, *, out=None) -> Tensor
    
    计算两个矩阵的矩阵乘法。
    
    如果 `input` 是一个 `(n × m)` 的张量，`mat2` 是一个 `(m × p)` 的张量，
    那么 `out` 将是一个 `(n × p)` 的张量。
    
    .. note::
        此函数不支持广播操作矩阵乘法，如需广播操作，请参见 :func:`torch.matmul`。
    
    Args:
        input (Tensor): 第一个矩阵
        mat2 (Tensor): 第二个矩阵
    
    Keyword args:
        out (Tensor, optional): 输出张量，用于保存结果的张量
    
    Example::
    
        >>> mat1 = torch.randn(2, 3)
        >>> mat2 = torch.randn(3, 3)
        >>> torch.mm(mat1, mat2)
        tensor([[ 0.4851,  0.5037, -0.3633],
                [-0.0760, -3.6705,  2.4784]])
    """.format(
        **common_args, **tf32_notes, **rocm_fp16_notes, **sparse_support_notes
    ),
)

add_docstr(
    torch.hspmm,
    r"""
    hspmm(mat1, mat2, *, out=None) -> Tensor
    
    计算一个稀疏 COO 矩阵 `mat1` 和一个稠密矩阵 `mat2` 的矩阵乘法。
    结果是一个 (1 + 1)-维混合 COO 矩阵。
    
    Args:
        mat1 (Tensor): 第一个稀疏矩阵，采用 COO 格式
        mat2 (Tensor): 第二个稠密矩阵，采用普通张量格式
    
    Keyword args:
        out (Tensor, optional): 输出张量，用于保存结果的张量
    """.format(
        **common_args
    ),
)

add_docstr(
    torch.matmul,
    r"""
    matmul(input, other, *, out=None) -> Tensor
    
    计算两个张量的矩阵乘积。
    
    具体行为取决于张量的维度情况，如下所示：
    
    ```
    
    """.format(
        **common_args
    ),
)
# 如果两个张量都是一维的，则返回它们的点积（标量）。
# 如果两个张量都是二维的，则返回矩阵乘积。
# 如果第一个参数是一维的，第二个参数是二维的，
# 则在进行矩阵乘法之前，在第二个参数的维度前面添加一个维度1。
# 然后在乘法完成后移除添加的维度1。
# 如果第一个参数是二维的，第二个参数是一维的，
# 则返回矩阵和向量的乘积。
# 如果两个参数至少都是一维的，并且至少一个参数是N维的（其中N > 2），
# 则返回批量矩阵乘积。如果第一个参数是一维的，为了进行批量矩阵乘法，
# 在其维度前面添加一个维度1，并在乘法完成后移除。如果第二个参数是一维的，
# 为了进行批量矩阵乘法，在其维度后面添加一个维度1，并在乘法完成后移除。
# 非矩阵维度（即批量维度）会进行广播（因此必须可广播）。
# 例如，如果input是一个(j × 1 × n × n)的张量，other是一个(k × n × n)的张量，
# 则out将是一个(j × k × n × n)的张量。
# 
# 注意，广播逻辑仅在确定输入是否可广播时查看批量维度，而不是矩阵维度。
# 例如，如果input是一个(j × 1 × n × m)的张量，other是一个(k × m × p)的张量，
# 这些输入在广播时是有效的，尽管最后两个维度（即矩阵维度）是不同的。
# out将是一个(j × k × n × p)的张量。

# 此操作支持具有稀疏布局的参数。特别是矩阵-矩阵（两个参数都是二维的）支持稀疏参数，
# 其限制与torch.mm相同。

# {sparse_beta_warning} 占位符，表示稀疏参数的警告信息。

# {tf32_note} 占位符，可能是关于TF32精度的注意事项。

# {rocm_fp16_note} 占位符，可能是关于ROCM FP16精度的注意事项。

# .. note::
#     这个函数的一维点积版本不支持out参数。

# 参数:
#     input (Tensor): 第一个要相乘的张量。
#     other (Tensor): 第二个要相乘的张量。

# 关键字参数:
#     {out} 占位符，可能是关于out参数的详细信息。

# 示例::

#     >>> # 向量 x 向量
#     >>> tensor1 = torch.randn(3)
#     >>> tensor2 = torch.randn(3)
#     >>> torch.matmul(tensor1, tensor2).size()
#     torch.Size([])
#     >>> # 矩阵 x 向量
#     >>> tensor1 = torch.randn(3, 4)
#     >>> tensor2 = torch.randn(4)
#     >>> torch.matmul(tensor1, tensor2).size()
#     torch.Size([3])
#     >>> # 批量矩阵 x 广播向量
#     >>> tensor1 = torch.randn(10, 3, 4)
#     >>> tensor2 = torch.randn(4)
#     >>> torch.matmul(tensor1, tensor2).size()
#     torch.Size([10, 3])
#     >>> # 批量矩阵 x 批量矩阵
#     >>> tensor1 = torch.randn(10, 3, 4)
#     >>> tensor2 = torch.randn(10, 4, 5)
#     >>> torch.matmul(tensor1, tensor2).size()
    # 展示当前张量的形状，此处是一个 10x3x5 的张量
    torch.Size([10, 3, 5])
    >>> # 批量矩阵乘以广播矩阵
    >>> # 创建一个大小为 10x3x4 的随机张量
    >>> tensor1 = torch.randn(10, 3, 4)
    # 创建一个大小为 4x5 的随机张量
    >>> tensor2 = torch.randn(4, 5)
    # 使用 torch.matmul 函数计算 tensor1 和 tensor2 的矩阵乘法结果的形状
    >>> torch.matmul(tensor1, tensor2).size()
    # 返回结果张量的形状，应为 10x3x5
    torch.Size([10, 3, 5])
add_docstr(
    torch.mode,
    r"""
mode(input, dim=-1, keepdim=False, *, out=None) -> (Tensor, LongTensor)

Returns a namedtuple ``(values, indices)`` where ``values`` is the mode
value of each row of the :attr:`input` tensor in the given dimension
:attr:`dim`, i.e. a value which appears most often
in that row, and ``indices`` is the index location of each mode value found.

By default, :attr:`dim` is the last dimension of the :attr:`input` tensor.

If :attr:`keepdim` is ``True``, the output tensors are of the same size as
:attr:`input` except in the dimension :attr:`dim` where they are of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting
in the output tensors having 1 fewer dimension than :attr:`input`.

.. note:: This function is not defined for ``torch.cuda.Tensor`` yet.

Args:
    input (Tensor): the input tensor from which to compute the mode
    dim (int, optional): the dimension over which to compute the mode (default is -1)
    keepdim (bool, optional): whether to retain the reduced dimension in the output (default is False)

Keyword args:
    out (tuple, optional): the result tuple of two output tensors (values, indices)

Example::

    >>> b = torch.tensor(
           [[0, 0, 0, 2, 0, 0, 2],
            [0, 3, 0, 0, 2, 0, 1],
            [2, 2, 2, 0, 0, 0, 3],
            [2, 2, 3, 0, 1, 1, 0],
            [1, 1, 0, 0, 2, 0, 2]])
    >>> torch.mode(b, 0)
    torch.return_types.mode(
    values=tensor([0, 2, 0, 0, 0, 0, 2]),
    indices=tensor([1, 3, 4, 4, 2, 4, 4]))
""".format(
        **single_dim_common
    ),
)

add_docstr(
    torch.mul,
    r"""
mul(input, other, *, out=None) -> Tensor

Multiplies :attr:`input` by :attr:`other`.

.. math::
    \text{out}_i = \text{input}_i \times \text{other}_i

Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`,
:ref:`type promotion <type-promotion-doc>`, and integer, float, and complex inputs.

Args:
    input (Tensor): the first tensor to multiply
    other (Tensor or Number): the tensor or number to multiply input by

Keyword args:
    out (Tensor, optional): the output tensor (default is None)

Examples::

    >>> a = torch.randn(3)
    >>> a
    tensor([ 0.2015, -0.4255,  2.6087])
    >>> torch.mul(a, 100)
    tensor([  20.1494,  -42.5491,  260.8663])

    >>> b = torch.randn(4, 1)
    >>> b
    tensor([[ 1.1207],
            [-0.3137],
            [ 0.0700],
            [ 0.8378]])
    >>> c = torch.randn(1, 4)
    >>> c
    tensor([[ 0.5146,  0.1216, -0.5244,  2.2382]])
    >>> torch.mul(b, c)
    tensor([[ 0.5767,  0.1363, -0.5877,  2.5083],
            [-0.1614, -0.0382,  0.1645, -0.7021],
            [ 0.0360,  0.0085, -0.0367,  0.1567],
            [ 0.4312,  0.1019, -0.4394,  1.8753]])
""".format(
        **common_args
    ),
)

add_docstr(
    torch.multiply,
    r"""
multiply(input, other, *, out=None)

Alias for :func:`torch.mul`.
""",
)

add_docstr(
    torch.multinomial,
    r"""
multinomial(input, num_samples, replacement=False, *, generator=None, out=None) -> LongTensor

Returns a tensor where each row contains :attr:`num_samples` indices sampled
from the multinomial (a stricter definition would be multivariate,
"""
Movedim(input, source, destination) -> Tensor

Moves the dimension(s) of 'input' at the position(s) in 'source'
to the position(s) in 'destination'.

Other dimensions of 'input' that are not explicitly moved remain in
their original order and appear at the positions not specified in 'destination'.

Args:
    input (Tensor): the input tensor
    source (Tuple[int]): tuple of ints denoting the positions of dimensions to move
    destination (Tuple[int]): tuple of ints denoting the new positions of dimensions

Returns:
    Tensor: a tensor with dimensions moved according to the specified source and destination

Example::

    >>> x = torch.randn(3, 4, 5)
    >>> torch.movedim(x, (0, 2), (2, 0)).size()
    torch.Size([5, 4, 3])
"""
    # 定义函数参数source和destination，它们可以是单个整数或整数元组，表示要移动的原始维度的位置或位置集合。
    source (int or tuple of ints): Original positions of the dims to move. These must be unique.
    destination (int or tuple of ints): Destination positions for each of the original dims. These must also be unique.
# 返回一个窄化版本的张量，与给定维度和范围相关
torch.narrow(
    # 输入张量
    input,
    # 窄化的维度索引
    dim,
    # 起始索引
    start,
    # 窄化的长度
    length
)
# 用于创建返回一个张量，其在指定维度上缩小到指定长度的副本，而不是与输入张量共享存储空间
def narrow_copy(input, dim, start, length, *, out=None) -> Tensor:
    """
    narrow_copy(input, dim, start, length, *, out=None) -> Tensor

    Same as :meth:`Tensor.narrow` except this returns a copy rather
    than shared storage. This is primarily for sparse tensors, which
    do not have a shared-storage narrow method.

    Args:
        input (Tensor): the tensor to narrow
        dim (int): the dimension along which to narrow
        start (int): index of the element to start the narrowed dimension from. Can
            be negative, which means indexing from the end of `dim`
        length (int): length of the narrowed dimension, must be weakly positive

    Keyword args:
        {out}

    Example::

        >>> x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> torch.narrow_copy(x, 0, 0, 2)
        tensor([[ 1,  2,  3],
                [ 4,  5,  6]])
        >>> torch.narrow_copy(x, 1, 1, 2)
        tensor([[ 2,  3],
                [ 5,  6],
                [ 8,  9]])
        >>> s = torch.arange(16).reshape(2, 2, 2, 2).to_sparse(2)
        >>> torch.narrow_copy(s, 0, 0, 1)
        tensor(indices=tensor([[0, 0],
                               [0, 1]]),
               values=tensor([[[0, 1],
                               [2, 3]],

                              [[4, 5],
                               [6, 7]]]),
               size=(1, 2, 2, 2), nnz=2, layout=torch.sparse_coo)

    .. seealso::

            :func:`torch.narrow` for a non copy variant
    """
    posinf (Number, optional): 如果是一个数字，用于替换正无穷值的值。
        如果为 None，则正无穷值将替换为 :attr:`input` 的数据类型所能表示的最大有限值。
        默认为 None。
    neginf (Number, optional): 如果是一个数字，用于替换负无穷值的值。
        如果为 None，则负无穷值将替换为 :attr:`input` 的数据类型所能表示的最小有限值。
        默认为 None。
Keyword args:
    {out}

Example::

    >>> x = torch.tensor([float('nan'), float('inf'), -float('inf'), 3.14])
    >>> torch.nan_to_num(x)
    tensor([ 0.0000e+00,  3.4028e+38, -3.4028e+38,  3.1400e+00])
    >>> torch.nan_to_num(x, nan=2.0)
    tensor([ 2.0000e+00,  3.4028e+38, -3.4028e+38,  3.1400e+00])
    >>> torch.nan_to_num(x, nan=2.0, posinf=1.0)
    tensor([ 2.0000e+00,  1.0000e+00, -3.4028e+38,  3.1400e+00])

""".format(
        **common_args
    ),
)

add_docstr(
    torch.ne,
    r"""
ne(input, other, *, out=None) -> Tensor

Computes :math:`\text{input} \neq \text{other}` element-wise.
"""
    + r"""

The second argument can be a number or a tensor whose shape is
:ref:`broadcastable <broadcasting-semantics>` with the first argument.

Args:
    input (Tensor): the tensor to compare
    other (Tensor or float): the tensor or value to compare

Keyword args:
    {out}

Returns:
    A boolean tensor that is True where :attr:`input` is not equal to :attr:`other` and False elsewhere

Example::

    >>> torch.ne(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
    tensor([[False, True], [True, False]])
""".format(
        **common_args
    ),
)

add_docstr(
    torch.not_equal,
    r"""
not_equal(input, other, *, out=None) -> Tensor

Alias for :func:`torch.ne`.
""",
)

add_docstr(
    torch.neg,
    r"""
neg(input, *, out=None) -> Tensor

Returns a new tensor with the negative of the elements of :attr:`input`.

.. math::
    \text{out} = -1 \times \text{input}
"""
    + r"""
Args:
    {input}

Keyword args:
    {out}

Example::

    >>> a = torch.randn(5)
    >>> a
    tensor([ 0.0090, -0.2262, -0.0682, -0.2866,  0.3940])
    >>> torch.neg(a)
    tensor([-0.0090,  0.2262,  0.0682,  0.2866, -0.3940])
""".format(
        **common_args
    ),
)

add_docstr(
    torch.negative,
    r"""
negative(input, *, out=None) -> Tensor

Alias for :func:`torch.neg`
""",
)

add_docstr(
    torch.nextafter,
    r"""
nextafter(input, other, *, out=None) -> Tensor

Return the next floating-point value after :attr:`input` towards :attr:`other`, elementwise.

The shapes of ``input`` and ``other`` must be
:ref:`broadcastable <broadcasting-semantics>`.

Args:
    input (Tensor): the first input tensor
    other (Tensor): the second input tensor

Keyword args:
    {out}

Example::

    >>> eps = torch.finfo(torch.float32).eps
    >>> torch.nextafter(torch.tensor([1.0, 2.0]), torch.tensor([2.0, 1.0])) == torch.tensor([eps + 1, 2 - eps])
    tensor([True, True])

""".format(
        **common_args
    ),
)

add_docstr(
    torch.nonzero,
    r"""
nonzero(input, *, out=None, as_tuple=False) -> LongTensor or tuple of LongTensors

.. note::
    :func:`torch.nonzero(..., as_tuple=False) <torch.nonzero>` (default) returns a
    2-D tensor where each row is the index for a nonzero value.

    :func:`torch.nonzero(..., as_tuple=True) <torch.nonzero>` returns a tuple of 1-D
    index tensors, allowing for advanced indexing, so ``x[x.nonzero(as_tuple=True)]``

"""
    # 获取张量 `x` 的所有非零值。返回的元组中，每个索引张量包含某一维度上的非零索引。

    # 以下是两种行为的详细说明。

    # 当输入张量 `input` 存在于 CUDA 上时，调用 `torch.nonzero() <torch.nonzero>` 会导致主机和设备之间的同步。
# 导入 torch 库
import torch

# 定义 normal 函数，生成从多个正态分布中抽取的随机数张量
def normal(mean, std, *, generator=None, out=None):
    """
    Returns a tensor of random numbers drawn from separate normal distributions
    whose mean and standard deviation are given.

    The :attr:`mean` is a tensor with the mean of
    each output element's normal distribution

    The :attr:`std` is a tensor with the standard deviation of
    each output element's normal distribution

    The shapes of :attr:`mean` and :attr:`std` don't need to match, but the
    total number of elements in each tensor need to be the same.

    Args:
        mean (Tensor): A tensor containing the mean of each output element's normal distribution.
        std (Tensor): A tensor containing the standard deviation of each output element's normal distribution.

    Keyword Args:
        generator (torch.Generator, optional): A pseudorandom number generator for sampling. Default is `None`.
        out (Tensor, optional): The output tensor. If provided, must be of the same shape as `mean` or `std`.

    Returns:
        Tensor: A tensor containing random numbers drawn from the normal distributions defined by `mean` and `std`.
    """
    pass
"""
.. function:: normal(mean=0.0, std, *, out=None) -> Tensor
   :noindex:

   Similar to the function above, but the means are shared among all drawn
   elements.

   Args:
       mean (float, optional): the mean for all distributions
       std (Tensor): the tensor of per-element standard deviations

   Keyword args:
       {out}

   Example::

       >>> torch.normal(mean=0.5, std=torch.arange(1., 6.))
       tensor([-1.2793, -1.0732, -2.0687,  5.1177, -1.2303])
"""

"""
.. function:: normal(mean, std=1.0, *, out=None) -> Tensor
   :noindex:

   Similar to the function above, but the standard deviations are shared among
   all drawn elements.

   Args:
       mean (Tensor): the tensor of per-element means
       std (float, optional): the standard deviation for all distributions

   Keyword args:
       out (Tensor, optional): the output tensor

   Example::

       >>> torch.normal(mean=torch.arange(1., 6.))
       tensor([ 1.1552,  2.6148,  2.6535,  5.8318,  4.2361])
"""

"""
.. function:: normal(mean, std, size, *, out=None) -> Tensor
   :noindex:

   Similar to the function above, but the means and standard deviations are shared
   among all drawn elements. The resulting tensor has size given by :attr:`size`.

   Args:
       mean (float): the mean for all distributions
       std (float): the standard deviation for all distributions
       size (int...): a sequence of integers defining the shape of the output tensor.

   Keyword args:
       {out}

   Example::

       >>> torch.normal(2, 3, size=(1, 4))
       tensor([[-1.3987, -1.9544,  3.6048,  0.7909]])
"""

"""
.. function:: numel(input) -> int

   Returns the total number of elements in the :attr:`input` tensor.

   Args:
       {input}

   Example::

       >>> a = torch.randn(1, 2, 3, 4, 5)
       >>> torch.numel(a)
       120
       >>> a = torch.zeros(4,4)
       >>> torch.numel(a)
       16
"""

"""
.. function:: ones(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

   Returns a tensor filled with the scalar value `1`, with the shape defined
   by the variable argument :attr:`size`.

   Args:
       size (int...): a sequence of integers defining the shape of the output tensor.
           Can be a variable number of arguments or a collection like a list or tuple.

   Keyword arguments:
       {out}
       {dtype}
       {layout}
       {device}
       {requires_grad}

   Example::

       >>> torch.ones(3, 4, dtype=torch.float64)
       tensor([[1., 1., 1., 1.],
               [1., 1., 1., 1.],
               [1., 1., 1., 1.]], dtype=torch.float64)
"""
    # 导入 torch 库并调用 ones 函数，生成一个形状为 (2, 3) 的张量，所有元素的值均为 1
    >>> torch.ones(2, 3)
    tensor([[ 1.,  1.,  1.],
            [ 1.,  1.,  1.]])
    
    # 再次调用 ones 函数，生成一个长度为 5 的一维张量，所有元素的值均为 1
    >>> torch.ones(5)
    tensor([ 1.,  1.,  1.,  1.,  1.])
# 添加文档字符串给 torch.ones_like 函数
add_docstr(
    torch.ones_like,
    r"""
ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) -> Tensor

Returns a tensor filled with the scalar value `1`, with the same size as
:attr:`input`. ``torch.ones_like(input)`` is equivalent to
``torch.ones(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)``.

.. warning::
    As of 0.4, this function does not support an :attr:`out` keyword. As an alternative,
    the old ``torch.ones_like(input, out=output)`` is equivalent to
    ``torch.ones(input.size(), out=output)``.

Args:
    input (Tensor): the input tensor whose size will be used for the output tensor.

Keyword arguments:
    dtype (torch.dtype, optional): the desired data type of the output tensor.
    layout (torch.layout, optional): the desired layout of the output tensor.
    device (torch.device, optional): the desired device of the output tensor.
    requires_grad (bool, optional): if autograd should record operations on the returned tensor.
    memory_format (torch.memory_format, optional): the desired memory format of the output tensor.

Example::

    >>> input = torch.empty(2, 3)
    >>> torch.ones_like(input)
    tensor([[ 1.,  1.,  1.],
            [ 1.,  1.,  1.]])
""".format(
        **factory_like_common_args
    ),
)

# 添加文档字符串给 torch.orgqr 函数
add_docstr(
    torch.orgqr,
    r"""
orgqr(input, tau) -> Tensor

Alias for :func:`torch.linalg.householder_product`.
""",
)

# 添加文档字符串给 torch.ormqr 函数
add_docstr(
    torch.ormqr,
    r"""
ormqr(input, tau, other, left=True, transpose=False, *, out=None) -> Tensor

Computes the matrix-matrix multiplication of a product of Householder matrices with a general matrix.

Multiplies a :math:`m \times n` matrix `C` (given by :attr:`other`) with a matrix `Q`,
where `Q` is represented using Householder reflectors `(input, tau)`.
See `Representation of Orthogonal or Unitary Matrices`_ for further details.

If :attr:`left` is `True` then `op(Q)` times `C` is computed, otherwise the result is `C` times `op(Q)`.
When :attr:`left` is `True`, the implicit matrix `Q` has size :math:`m \times m`.
It has size :math:`n \times n` otherwise.
If :attr:`transpose` is `True` then `op` is the conjugate transpose operation, otherwise it's a no-op.

Supports inputs of float, double, cfloat and cdouble dtypes.
Also supports batched inputs, and, if the input is batched, the output is batched with the same dimensions.

.. seealso::
        :func:`torch.geqrf` can be used to form the Householder representation `(input, tau)` of matrix `Q`
        from the QR decomposition.

.. note::
        This function supports backward but it is only fast when ``(input, tau)`` do not require gradients
        and/or ``tau.size(-1)`` is very small.

Args:
    input (Tensor): tensor of shape `(*, mn, k)` where `*` is zero or more batch dimensions
                    and `mn` equals to `m` or `n` depending on the :attr:`left`.
    tau (Tensor): tensor of shape `(*, min(mn, k))` where `*` is zero or more batch dimensions.
    other (Tensor): tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions.
    left (bool): controls the order of multiplication.
    transpose (bool): controls whether the matrix `Q` is conjugate transposed or not.

Keyword args:
    out (Tensor, optional): the output Tensor. Ignored if `None`. Default: `None`.
""",
)
# 定义函数，为 torch.permute 函数添加文档字符串
add_docstr(
    torch.permute,
    r"""
    permute(input, dims) -> Tensor

    Returns a view of the original tensor :attr:`input` with its dimensions permuted.

    Args:
        {input}  # 输入参数，通常是一个张量

        dims (tuple of int): The desired ordering of dimensions  # dims 是一个元组，指定维度的顺序

    Example:
        >>> x = torch.randn(2, 3, 5)
        >>> x.size()
        torch.Size([2, 3, 5])
        >>> torch.permute(x, (2, 0, 1)).size()
        torch.Size([5, 2, 3])
    """.format(
        **common_args
    ),
)

# 定义函数，为 torch.poisson 函数添加文档字符串
add_docstr(
    torch.poisson,
    r"""
    poisson(input, generator=None) -> Tensor

    Returns a tensor of the same size as :attr:`input` with each element
    sampled from a Poisson distribution with rate parameter given by the corresponding
    element in :attr:`input` i.e.,

    .. math::
        \text{{out}}_i \sim \text{{Poisson}}(\text{{input}}_i)

    :attr:`input` must be non-negative.

    Args:
        input (Tensor): the input tensor containing the rates of the Poisson distribution

    Keyword args:
        {generator}  # 可选的关键字参数，通常用于生成器

    Example::

        >>> rates = torch.rand(4, 4) * 5  # rate parameter between 0 and 5
        >>> torch.poisson(rates)
        tensor([[9., 1., 3., 5.],
                [8., 6., 6., 0.],
                [0., 4., 5., 3.],
                [2., 1., 4., 2.]])
    """.format(
        **common_args
    ),
)

# 定义函数，为 torch.polygamma 函数添加文档字符串
add_docstr(
    torch.polygamma,
    r"""
    polygamma(n, input, *, out=None) -> Tensor

    Alias for :func:`torch.special.polygamma`.
    """,
)

# 定义函数，为 torch.positive 函数添加文档字符串
add_docstr(
    torch.positive,
    r"""
    positive(input) -> Tensor

    Returns :attr:`input`.
    Throws a runtime error if :attr:`input` is a bool tensor.

    Args:
        {input}  # 输入参数，通常是一个张量

    Example::

        >>> t = torch.randn(5)
        >>> t
        tensor([ 0.0090, -0.2262, -0.0682, -0.2866,  0.3940])
        >>> torch.positive(t)
        tensor([ 0.0090, -0.2262, -0.0682, -0.2866,  0.3940])
    """.format(
        **common_args
    ),
)

# 定义函数，为 torch.pow 函数添加文档字符串
add_docstr(
    torch.pow,
    r"""
    pow(input, exponent, *, out=None) -> Tensor

    Takes the power of each element in :attr:`input` with :attr:`exponent` and
    returns a tensor with the result.

    :attr:`exponent` can be either a single ``float`` number or a `Tensor`
    with the same number of elements as :attr:`input`.

    When :attr:`exponent` is a scalar value, the operation applied is:

    .. math::
        \text{{out}}_i = x_i ^ \text{{exponent}}

    When :attr:`exponent` is a tensor, the operation applied is:

    .. math::
        \text{{out}}_i = x_i ^ {{\text{{exponent}}_i}}

    When :attr:`exponent` is a tensor, the shapes of :attr:`input`
    and :attr:`exponent` must be :ref:`broadcastable <broadcasting-semantics>`.

    Args:
        {input}  # 输入参数，通常是一个张量

        exponent (float or tensor): the exponent value

    Keyword args:
        {out}  # 可选的关键字参数，通常是输出张量的选项

    Example::

        >>> a = torch.randn(4)
        >>> a
        tensor([ 0.4331,  1.2475,  0.6834, -0.2791])
        >>> torch.pow(a, 2)
        tensor([ 0.1875,  1.5561,  0.4670,  0.0779])
        >>> exp = torch.arange(1., 5.)

        >>> a = torch.arange(1., 5.)
        >>> a
        tensor([ 1.,  2.,  3.,  4.])
        >>> exp
        tensor([ 1.,  2.,  3.,  4.])
    """.format(
        **common_args
    ),
)
    # 使用 PyTorch 的 torch.pow 函数对张量 a 中的每个元素进行指数运算
    torch.pow(a, exp)
    # 返回一个新的张量，其中每个元素都是对应位置元素的 exp 次幂结果
    tensor([   1.,    4.,   27.,  256.])
# 定义函数 `promote_types`，用于在 Torch 中提升类型
def promote_types(
    # 输入参数 `tensors` 是一个可变长度的 Tensor 元组或列表
    tensors,
):
    """
    Promotes the types of the input tensors to a common type that can accommodate all input types.
    Returns the promoted dtype and the promoted tensors.
    
    Args:
        tensors (tuple of Tensors): Tensors to promote the types for.
        
    Returns:
        dtype (torch.dtype): The promoted dtype that can accommodate all input types.
        promoted_tensors (tuple of Tensors): Tensors with promoted types.
    """
    ...
# 计算两个数据类型 type1 和 type2 之间的类型提升结果，并返回最小尺寸和标量类型的 torch.dtype
def promote_types(type1, type2) -> dtype:
    """
    根据 type1 和 type2 的类型进行类型提升，返回一个 torch.dtype，该类型既不比 type1 或 type2 更小也不是更低级别的类型。
    详见类型提升文档：https://pytorch.org/docs/stable/torch.html#type-promotion-doc

    Args:
        type1 (:class:`torch.dtype`): 第一个输入的数据类型
        type2 (:class:`torch.dtype`): 第二个输入的数据类型

    Example::

        >>> torch.promote_types(torch.int32, torch.float32)
        torch.float32
        >>> torch.promote_types(torch.uint8, torch.long)
        torch.long
    """
    return dtype

# 定义 torch.qr 函数的文档字符串
add_docstr(
    torch.qr,
    r"""
    qr(input, some=True, *, out=None) -> (Tensor, Tensor)

    计算矩阵或批量矩阵 input 的 QR 分解，并返回一个命名元组 (Q, R)，
    满足 :math:`\text{input} = Q R`，其中 :math:`Q` 是正交矩阵或批量正交矩阵，
    :math:`R` 是上三角矩阵或批量上三角矩阵。

    如果 some 参数为 ``True``，则返回紧凑（减少的）QR分解结果；
    如果 some 参数为 ``False``，则返回完整的QR分解结果。

    .. warning::

        :func:`torch.qr` 已被弃用，推荐使用 :func:`torch.linalg.qr`，
        且将在未来的 PyTorch 版本中移除。参数 some 已被字符串参数 mode 替代。

        ``Q, R = torch.qr(A)`` 应替换为

        .. code:: python

            Q, R = torch.linalg.qr(A)

        ``Q, R = torch.qr(A, some=False)`` 应替换为

        .. code:: python

            Q, R = torch.linalg.qr(A, mode="complete")

    .. warning::

        如果计划通过 QR 分解进行反向传播，请注意当前的反向传播实现仅在
        :math:`\min(input.size(-1), input.size(-2))` 列是线性独立时才是良定义的。
        一旦 QR 支持枢轴操作，这种行为可能会发生变化。

    .. note::
        该函数在 CPU 输入上使用 LAPACK，在 CUDA 输入上使用 MAGMA，
        并且可能在不同设备类型或平台上产生不同但有效的分解结果。

    Args:
        input (Tensor): 大小为 :math:`(*, m, n)` 的输入张量，其中 `*` 是零个或多个批次维度，
                    包含尺寸为 :math:`m \times n` 的矩阵。
        some (bool, optional): 设置为 ``True`` 以获得紧凑的 QR 分解结果，设置为 ``False`` 以获得完整的 QR 分解结果。
                    如果 `k = min(m, n)`，则：

                      * ``some=True`` : 返回尺寸为 (m, k), (k, n) 的 `(Q, R)`
                      * ``'some=False'``: 返回尺寸为 (m, m), (m, n) 的 `(Q, R)`（默认）

    Keyword args:
        out (tuple, optional): `Q` 和 `R` 张量的元组。
                    `Q` 和 `R` 的尺寸详见上述 some 参数的描述。

    Example::

        >>> a = torch.tensor([[12., -51, 4], [6, 167, -68], [-4, 24, -41]])
        >>> q, r = torch.qr(a)
        >>> q
    """
)
    # 定义一个3x3的张量，表示一个矩阵
    tensor([[-0.8571,  0.3943,  0.3314],
            [-0.4286, -0.9029, -0.0343],
            [ 0.2857, -0.1714,  0.9429]])
    # 张量r，表示另一个3x3的矩阵
    >>> r
    tensor([[ -14.0000,  -21.0000,   14.0000],
            [   0.0000, -175.0000,   70.0000],
            [   0.0000,    0.0000,  -35.0000]])
    # 计算q和r的矩阵乘积并四舍五入
    >>> torch.mm(q, r).round()
    tensor([[  12.,  -51.,    4.],
            [   6.,  167.,  -68.],
            [  -4.,   24.,  -41.]])
    # 计算q的转置和q的矩阵乘积并四舍五入
    >>> torch.mm(q.t(), q).round()
    tensor([[ 1.,  0.,  0.],
            [ 0.,  1., -0.],
            [ 0., -0.,  1.]])
    # 生成一个形状为(3, 4, 5)的随机张量a
    >>> a = torch.randn(3, 4, 5)
    # 对张量a进行QR分解，返回正交矩阵q和上三角矩阵r
    >>> q, r = torch.qr(a, some=False)
    # 检查q和r的乘积是否接近原始张量a
    >>> torch.allclose(torch.matmul(q, r), a)
    True
    # 检查q的转置和q的乘积是否接近单位矩阵
    >>> torch.allclose(torch.matmul(q.t(), q), torch.eye(5))
    True
# 创建一个与输入张量形状相同的张量，其中的值是从均匀分布[0, 1)中随机抽样得到的

def rand_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) -> Tensor:
    """
    Args:
        input (Tensor): 用于确定形状的输入张量.
        
    Keyword arguments:
        dtype (:class:`torch.dtype`, optional): 输出张量的数据类型.
        layout (:class:`torch.layout`, optional): 输出张量的布局.
        device (:class:`torch.device`, optional): 输出张量的设备.
        requires_grad (bool, optional): 输出张量是否需要梯度.
        memory_format (:class:`torch.memory_format`, optional): 输出张量的内存格式.

    Returns:
        Tensor: 与输入张量形状相同的张量，其元素值从均匀分布[0, 1)中随机抽样得到.
    """
add_docstr(
    torch.rand_like,
    """
rand_like(input, *, dtype=None, layout=torch.strided, device=None, requires_grad=False, memory_format=torch.preserve_format) -> Tensor

Returns a tensor with the same size as :attr:`input` that is filled with
random numbers from a uniform distribution on the interval :math:`[0, 1)`.

``torch.rand_like(input)`` is equivalent to
``torch.rand(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)``.

Args:
    input (Tensor): the input tensor from which to take the size and optionally the dtype, layout, device, and memory_format.

Keyword args:
    dtype (`torch.dtype`, optional): the desired data type of returned tensor. If not provided, defaults to the dtype of `input`.
    layout (`torch.layout`, optional): the desired layout of returned tensor. Default is `torch.strided`.
    device (`torch.device`, optional): the desired device of returned tensor. Default is `None` (which means the current device).
    requires_grad (bool, optional): if autograd should record operations on the returned tensor. Default is `False`.
    memory_format (`torch.memory_format`, optional): the desired memory format of returned tensor. Default is `torch.preserve_format`.

""".format(
        **factory_like_common_args
    ),
)
"""
For complex dtypes, the tensor is i.i.d. sampled from a `complex normal distribution`_ with zero mean and
unit variance as

.. math::
    \text{{out}}_{{i}} \sim \mathcal{{CN}}(0, 1)

This is equivalent to separately sampling the real :math:`(\operatorname{{Re}})` and imaginary
:math:`(\operatorname{{Im}})` part of :math:`\text{{out}}_i` as

.. math::
    \operatorname{{Re}}(\text{{out}}_{{i}}) \sim \mathcal{{N}}(0, \frac{{1}}{{2}}),\quad
    \operatorname{{Im}}(\text{{out}}_{{i}}) \sim \mathcal{{N}}(0, \frac{{1}}{{2}})

The shape of the tensor is defined by the variable argument :attr:`size`.
"""

add_docstr(
    torch.randn_like,
    r"""
randn_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) -> Tensor

Returns a tensor with the same size as :attr:`input` that is filled with
random numbers from a normal distribution with mean 0 and variance 1. Please refer to :func:`torch.randn` for the
sampling process of complex dtypes. ``torch.randn_like(input)`` is equivalent to
``torch.randn(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)``.

Args:
    input (Tensor): the tensor of the same size as the desired output tensor.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: same as :attr:`input`.
    layout (:class:`torch.layout`, optional): the desired layout of returned tensor.
        Default: same as :attr:`input`.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: same as :attr:`input`.
    requires_grad (bool, optional): if autograd should record operations on the returned tensor.
        Default: ``False``.
    memory_format (:class:`torch.memory_format`, optional): the desired memory format of returned tensor.
        Default: ``torch.preserve_format``.

""".format(
        **factory_like_common_args
    ),
)

add_docstr(
    torch.randperm,
    """
randperm(n, *, generator=None, out=None, dtype=torch.int64,layout=torch.strided, \
device=None, requires_grad=False, pin_memory=False) -> Tensor
"""
    + r"""
Returns a random permutation of integers from ``0`` to ``n - 1``.

Args:
    n (int): the upper bound (exclusive)

Keyword args:
    {generator}
    {out}
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: ``torch.int64``.
    {layout}
    {device}
    {requires_grad}
    {pin_memory}

Example::

    >>> torch.randperm(4)
    tensor([2, 1, 0, 3])
""".format(
        **factory_common_args
    ),
)

add_docstr(
    torch.tensor,
    r"""
tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False) -> Tensor

Constructs a tensor with no autograd history (also known as a "leaf tensor", see :doc:`/notes/autograd`) by copying :attr:`data`.

.. warning::

    When working with tensors prefer using :func:`torch.Tensor.clone`,

""".format(
        **factory_common_args
    ),
)
    :func:`torch.Tensor.detach`, and :func:`torch.Tensor.requires_grad_` for
    readability. Letting `t` be a tensor, ``torch.tensor(t)`` is equivalent to
    ``t.clone().detach()``, and ``torch.tensor(t, requires_grad=True)``
    is equivalent to ``t.clone().detach().requires_grad_(True)``



    # 引用 torch.Tensor 的 detach 方法和 requires_grad_ 方法以提高可读性
    # 假设 t 是一个张量，``torch.tensor(t)`` 等价于 ``t.clone().detach()``
    # ``torch.tensor(t, requires_grad=True)`` 等价于 ``t.clone().detach().requires_grad_(True)``
r"""
arange(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

返回一个 1-D 张量，大小为 :math:`\left\lfloor \frac{\text{end} - \text{start}}{\text{step}} \right\rfloor + 1`，
其值从 :attr:`start` 到 :attr:`end`，步长为 :attr:`step`。步长是张量中两个相邻值之间的间隔。

.. math::
    \text{out}_{i+1} = \text{out}_i + \text{step}.

.. warning::
    此函数已被弃用，并将在未来版本中移除，因为其行为与 Python 的内置函数 range 不一致。建议使用 :func:`torch.arange`，
    它生成的值范围是 [start, end)。

Args:
    start (float): 起始值，默认为 ``0``.
    end (float): 结束值。
    step (float): 每对相邻点之间的间隔。默认为 ``1``。

Keyword args:
    {out}
    {dtype} 如果未提供 `dtype`，则从其他输入参数推断数据类型。如果 `start`，`end` 或 `stop` 中有任何浮点数，
        则推断 `dtype` 为默认的 dtype，请参阅 :meth:`~torch.get_default_dtype`。否则，推断 `dtype` 为 `torch.int64`。
    {layout}
    {device}
    {requires_grad}

Example::

    >>> torch.arange(1, 5)
    tensor([ 1,  2,  3,  4])
    >>> torch.arange(1, 5, 0.5)
    tensor([ 1.0000,  1.5000,  2.0000,  2.5000,  3.0000,  3.5000,  4.0000])
""".format(
    **factory_common_args
)
add_docstr(
    torch.remainder,
    r"""
remainder(input, other, *, out=None) -> Tensor

Computes
`Python's modulus operation <https://docs.python.org/3/reference/expressions.html#binary-arithmetic-operations>`_
entrywise.  The result has the same sign as the divisor :attr:`other` and its absolute value
is less than that of :attr:`other`.

It may also be defined in terms of :func:`torch.div` as

.. code:: python

    torch.remainder(a, b) == a - a.div(b, rounding_mode="floor") * b

Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`,
:ref:`type promotion <type-promotion-doc>`, and integer and float inputs.

.. note::
    Complex inputs are not supported. In some cases, it is not mathematically
    possible to satisfy the definition of a modulo operation with complex numbers.
    See :func:`torch.fmod` for how division by zero is handled.

.. seealso::

    :func:`torch.fmod` which implements C++'s `std::fmod <https://en.cppreference.com/w/cpp/numeric/math/fmod>`_.
    This one is defined in terms of division rounding towards zero.

Args:
    input (Tensor or Scalar): the dividend
    other (Tensor or Scalar): the divisor

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> torch.remainder(torch.tensor([-3., -2, -1, 0, 1, 2, 3]), 2)
    tensor([ 1.,  0,  1,  0,  1,  0,  1])

"""
)
    # 使用 PyTorch 库中的 remainder 函数计算给定张量与指定标量的余数
    >>> torch.remainder(torch.tensor([-3., -2, -1, 1, 2, 3]), 2)
    # 返回一个张量，其中每个元素是对应输入张量元素与2取余的结果
    tensor([ 1.,  0.,  1.,  1.,  0.,  1.])
    
    # 使用 PyTorch 库中的 remainder 函数计算给定张量与指定标量的余数
    >>> torch.remainder(torch.tensor([1, 2, 3, 4, 5]), -1.5)
    # 返回一个张量，其中每个元素是对应输入张量元素与-1.5取余的结果
    tensor([ -0.5000, -1.0000,  0.0000, -0.5000, -1.0000 ])
# 使用 format 方法填充字符串模板，生成函数文档字符串，将 common_args 的内容传入格式化字符串中
""".format(
        **common_args
    ),
)

# 添加函数文档字符串给 torch.renorm 函数
add_docstr(
    torch.renorm,
    r"""
renorm(input, p, dim, maxnorm, *, out=None) -> Tensor

Returns a tensor where each sub-tensor of :attr:`input` along dimension
:attr:`dim` is normalized such that the `p`-norm of the sub-tensor is lower
than the value :attr:`maxnorm`

.. note:: If the norm of a row is lower than `maxnorm`, the row is unchanged

Args:
    {input}  # 描述 input 参数，通常是一个 Tensor 对象
    p (float): the power for the norm computation  # p 是用于规范化计算的幂次数
    dim (int): the dimension to slice over to get the sub-tensors  # dim 是用于切片获取子张量的维度
    maxnorm (float): the maximum norm to keep each sub-tensor under  # maxnorm 是保持每个子张量在其下的最大规范化范数

Keyword args:
    {out}  # 描述 out 参数，通常是一个可选的 Tensor 对象，用于存储结果

Example::

    >>> x = torch.ones(3, 3)
    >>> x[1].fill_(2)
    tensor([ 2.,  2.,  2.])
    >>> x[2].fill_(3)
    tensor([ 3.,  3.,  3.])
    >>> x
    tensor([[ 1.,  1.,  1.],
            [ 2.,  2.,  2.],
            [ 3.,  3.,  3.]])
    >>> torch.renorm(x, 1, 0, 5)
    tensor([[ 1.0000,  1.0000,  1.0000],  # 示例中对 x 进行 renorm 操作，使得每行的 L1 范数不超过 5
            [ 1.6667,  1.6667,  1.6667],
            [ 1.6667,  1.6667,  1.6667]])
""".format(
        **common_args
    ),
)

# 添加函数文档字符串给 torch.reshape 函数
add_docstr(
    torch.reshape,
    r"""
reshape(input, shape) -> Tensor

Returns a tensor with the same data and number of elements as :attr:`input`,
but with the specified shape. When possible, the returned tensor will be a view
of :attr:`input`. Otherwise, it will be a copy. Contiguous inputs and inputs
with compatible strides can be reshaped without copying, but you should not
depend on the copying vs. viewing behavior.

See :meth:`torch.Tensor.view` on when it is possible to return a view.

A single dimension may be -1, in which case it's inferred from the remaining
dimensions and the number of elements in :attr:`input`.

Args:
    input (Tensor): the tensor to be reshaped  # 要重塑的输入张量
    shape (tuple of int): the new shape  # 新的张量形状的元组

Example::

    >>> a = torch.arange(4.)
    >>> torch.reshape(a, (2, 2))  # 将长度为4的张量 a 重塑为 2x2 的张量
    tensor([[ 0.,  1.],
            [ 2.,  3.]])
    >>> b = torch.tensor([[0, 1], [2, 3]])
    >>> torch.reshape(b, (-1,))  # 将2x2的张量 b 重塑为长度为4的张量
    tensor([ 0,  1,  2,  3])
""",
)


# 添加函数文档字符串给 torch.result_type 函数
add_docstr(
    torch.result_type,
    r"""
result_type(tensor1, tensor2) -> dtype

Returns the :class:`torch.dtype` that would result from performing an arithmetic
operation on the provided input tensors. See type promotion :ref:`documentation <type-promotion-doc>`
for more information on the type promotion logic.

Args:
    tensor1 (Tensor or Number): an input tensor or number  # 第一个输入张量或数字
    tensor2 (Tensor or Number): an input tensor or number  # 第二个输入张量或数字

Example::

    >>> torch.result_type(torch.tensor([1, 2], dtype=torch.int), 1.0)  # 返回计算 int 张量与 float 数字的结果类型
    torch.float32
    >>> torch.result_type(torch.tensor([1, 2], dtype=torch.uint8), torch.tensor(1))  # 返回计算 uint8 张量与 int 张量的结果类型
    torch.uint8
""",
)

# 添加函数文档字符串给 torch.row_stack 函数，其实是 torch.vstack 的别名
add_docstr(
    torch.row_stack,
    r"""
row_stack(tensors, *, out=None) -> Tensor

Alias of :func:`torch.vstack`.
""",
)

# 添加函数文档字符串给 torch.round 函数
add_docstr(
    torch.round,
    r"""
round(input, *, decimals=0, out=None) -> Tensor

Rounds elements of :attr:`input` to the nearest integer.

""",
)
# 返回一个新的张量，其中每个元素的平方根的倒数
def rsqrt(input, *, out=None):
    # 计算每个元素的平方根的倒数，存储到新的张量中
    out_i = 1 / sqrt(input_i)
    # 返回结果张量



# 在指定维度上，根据索引将源张量的值散布到输入张量中
def scatter(input, dim, index, src) -> Tensor:
    # out-of-place 版本的 scatter_ 方法



# 在指定维度上，根据索引将源张量的值相加散布到输入张量中
def scatter_add(input, dim, index, src) -> Tensor:
    # out-of-place 版本的 scatter_add_ 方法



# 在指定维度上，根据索引将源张量的值进行指定的 reduce 操作后散布到输入张量中
def scatter_reduce(input, dim, index, src, reduce, *, include_self=True) -> Tensor:
    # out-of-place 版本的 scatter_reduce_ 方法



# 在给定索引处沿选定维度切片输入张量
def select(input, dim, index) -> Tensor:
    # 返回原始张量的视图，删除给定维度
"""
Embeds the values of the `src` tensor into `input` at the given index using scatter operation.

Args:
    input (Tensor): The tensor into which `src` will be embedded.
    src (Tensor): The tensor to embed into `input`.
    dim (int): The dimension along which to scatter.
    index (int): The index to select with.

Returns:
    Tensor: A new tensor with embedded values.

Notes:
    This function creates a new tensor with fresh storage, rather than returning a view.

Example:
    >>> a = torch.zeros(2, 2)
    >>> b = torch.ones(2)
    >>> torch.select_scatter(a, b, 0, 0)
    tensor([[1., 1.],
            [0., 0.]])
"""

"""
Embeds the values of the `src` tensor into `input` at the specified dimension using slice scatter.

Args:
    input (Tensor): The tensor into which `src` will be embedded.
    src (Tensor): The tensor to embed into `input`.
    dim (int, optional): The dimension to insert the slice into (default is 0).
    start (int, optional): The start index of where to insert the slice (default is None).
    end (int, optional): The end index of where to insert the slice (default is None).
    step (int): The stride between elements (default is 1).

Returns:
    Tensor: A new tensor with embedded values.

Example:
    >>> a = torch.zeros(8, 8)
    >>> b = torch.ones(2, 8)
    >>> torch.slice_scatter(a, b, start=6)
    tensor([[0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1.]])
    
    >>> b = torch.ones(8, 2)
    >>> torch.slice_scatter(a, b, dim=1, start=2, end=6, step=2)
    tensor([[0., 0., 1., 0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 1., 0., 0., 0.]])
"""
    # 设置 Torch 库中的 flush denormal 行为，用于处理极小数值
    torch.set_flush_denormal,
    # 多行字符串的开始，用于引用包含多行注释或文本的区块
    r"""
# 设置 CPU 上的 denormal 浮点数处理模式
set_flush_denormal(mode) -> bool
# 禁用 CPU 上的 denormal 浮点数处理模式。

# 如果系统支持刷新 denormal 数，并且成功配置了刷新 denormal 模式，则返回 True。
# :meth:`~torch.set_flush_denormal` 在支持 SSE3 的 x86 架构和 AArch64 架构上受支持。

Args:
    mode (bool): 控制是否启用刷新 denormal 模式

Example::

    >>> torch.set_flush_denormal(True)
    True
    >>> torch.tensor([1e-323], dtype=torch.float64)
    tensor([ 0.], dtype=torch.float64)
    >>> torch.set_flush_denormal(False)
    True
    >>> torch.tensor([1e-323], dtype=torch.float64)
    tensor(9.88131e-324 *
           [ 1.0000], dtype=torch.float64)




# 设置 CPU 上的 intraop 并行运行的线程数
set_num_threads(int)
# 设置用于 CPU 上 intraop 并行运行的线程数。

.. warning::
    为确保使用正确的线程数，必须在运行 eager、JIT 或 autograd 代码之前调用 set_num_threads。




# 设置 CPU 上的 interop 并行运行的线程数
set_num_interop_threads(int)
# 设置用于 CPU 上 interop 并行运行（例如 JIT 解释器）的线程数。

.. warning::
    只能调用一次，并且必须在启动任何 inter-op 并行工作（例如 JIT 执行）之前调用。




# 计算输入张量每个元素的 sigmoid 函数值
sigmoid(input, *, out=None) -> Tensor
# 别名为 :func:`torch.special.expit`。




# 计算输入张量每个元素的 logit 函数值
logit(input, eps=None, *, out=None) -> Tensor
# 别名为 :func:`torch.special.logit`。




# 计算输入张量每个元素的符号
sign(input, *, out=None) -> Tensor
# 返回一个新的张量，其中元素为 :attr:`input` 元素的符号值。

.. math::
    \text{out}_{i} = \operatorname{sgn}(\text{input}_{i})

Args:
    {input}

Keyword args:
    {out}

Example::

    >>> a = torch.tensor([0.7, -1.2, 0., 2.3])
    >>> a
    tensor([ 0.7000, -1.2000,  0.0000,  2.3000])
    >>> torch.sign(a)
    tensor([ 1., -1.,  0.,  1.])




# 检测输入张量每个元素的符号位是否设置
signbit(input, *, out=None) -> Tensor
# 测试 :attr:`input` 的每个元素是否设置了其符号位。

Args:
  {input}

Keyword args:
  {out}

Example::

    >>> a = torch.tensor([0.7, -1.2, 0., 2.3])
    >>> torch.signbit(a)
    tensor([ False, True,  False,  False])
    >>> a = torch.tensor([-0.0, 0.0])
    >>> torch.signbit(a)
    tensor([ True,  False])

.. note::
    signbit 处理带符号的零，因此负零 (-0) 返回 True。




# 计算输入张量每个元素的符号
sgn(input, *, out=None) -> Tensor
# 该函数是 torch.sign() 对复杂张量的扩展。
# 它计算一个新的张量，其元素角度与 :attr:`input` 对应元素相同，
# 对于复杂张量，其绝对值为一，对于非复杂张量等效于 torch.sign()。

.. math::
    # 根据输入的input_i的长度进行条件判断，确定out_i的取值
    out_i = \begin{cases}
                    0 & |\text{input}_i| == 0 \\  # 如果input_i的长度为0，则out_i为0
                    \frac{\text{input}_i}{|\text{input}_i|} & \text{otherwise}  # 否则，out_i等于input_i除以其绝对值
                    \end{cases}
"""
add_docstr(
    torch.sin,
    r"""
    sin(input, *, out=None) -> Tensor
    
    Returns a new tensor with the sine of the elements of :attr:`input`.
    
    .. math::
        \text{out}_{i} = \sin(\text{input}_{i})
    """
    + r"""
    Args:
        {input}  # 描述输入参数 `input`，是一个张量，包含了需要计算正弦值的元素
    
    Keyword args:
        {out}  # 描述关键字参数 `out`，可选的输出张量，用于存放计算结果的张量
    
    Example::
    
        >>> a = torch.randn(4)
        >>> a
        tensor([-0.5461,  0.1347, -2.7266, -0.2746])
        >>> torch.sin(a)
        tensor([-0.5194,  0.1343, -0.4032, -0.2711])
    """.format(
        **common_args
    ),
)

add_docstr(
    torch.sinc,
    r"""
    sinc(input, *, out=None) -> Tensor
    
    Alias for :func:`torch.special.sinc`.
    """
)

add_docstr(
    torch.sinh,
    r"""
    sinh(input, *, out=None) -> Tensor
    
    Returns a new tensor with the hyperbolic sine of the elements of
    :attr:`input`.
    
    .. math::
        \text{out}_{i} = \sinh(\text{input}_{i})
    """
    + r"""
    Args:
        {input}  # 描述输入参数 `input`，是一个张量，包含了需要计算双曲正弦值的元素
    
    Keyword args:
        {out}  # 描述关键字参数 `out`，可选的输出张量，用于存放计算结果的张量
    
    Example::
    
        >>> a = torch.randn(4)
        >>> a
        tensor([ 0.5380, -0.8632, -0.1265,  0.9399])
        >>> torch.sinh(a)
        tensor([ 0.5644, -0.9744, -0.1268,  1.0845])
    
    .. note::
       当 :attr:`input` 在 CPU 上时，torch.sinh 的实现可能使用 Sleef 库，
       它会将非常大的结果四舍五入为无穷大或负无穷大。
       更多细节请参考 `这里 <https://sleef.org/purec.xhtml>`_。
    """.format(
        **common_args
    ),
)

add_docstr(
    torch.sort,
    r"""
    sort(input, dim=-1, descending=False, stable=False, *, out=None) -> (Tensor, LongTensor)
    
    Sorts the elements of the :attr:`input` tensor along a given dimension
    in ascending order by value.
    
    If :attr:`dim` is not given, the last dimension of the `input` is chosen.
    
    If :attr:`descending` is ``True`` then the elements are sorted in descending
    order by value.
    
    If :attr:`stable` is ``True`` then the sorting routine becomes stable, preserving
    the order of equivalent elements.
    
    A namedtuple of (values, indices) is returned, where the `values` are the
    sorted values and `indices` are the indices of the elements in the original
    `input` tensor.
    
    Args:
        {input}  # 描述输入参数 `input`，是一个张量，包含了需要排序的元素
    
        dim (int, optional): 沿着哪个维度进行排序，默认为最后一个维度
        descending (bool, optional): 控制排序顺序（升序或降序）
        stable (bool, optional): 使排序过程稳定，保持等价元素的顺序不变
    
    Keyword args:
        out (tuple, optional): (`Tensor`, `LongTensor`) 的输出元组，可选地用作输出缓冲区
    
    Example::
    
        >>> x = torch.randn(3, 4)
        >>> sorted, indices = torch.sort(x)
        >>> sorted
        tensor([[-0.2162,  0.0608,  0.6719,  2.3332],
                [-0.5793,  0.0061,  0.6058,  0.9497],
                [-0.5071,  0.3343,  0.9553,  1.0960]])
        >>> indices
    """.format(
        **common_args
    ),
)
    # 创建一个包含整数的二维张量
    tensor([[ 1,  0,  2,  3],
            [ 3,  1,  0,  2],
            [ 0,  3,  1,  2]])
    
    # 对张量 x 按列进行排序，返回排序后的张量和对应的索引
    sorted, indices = torch.sort(x, 0)
    # 输出排序后的张量 sorted
    tensor([[-0.5071, -0.2162,  0.6719, -0.5793],
            [ 0.0608,  0.0061,  0.9497,  0.3343],
            [ 0.6058,  0.9553,  1.0960,  2.3332]])
    # 输出每列排序后元素在原始张量中的索引 indices
    tensor([[ 2,  0,  0,  1],
            [ 0,  1,  1,  2],
            [ 1,  2,  2,  0]])
    
    # 创建一个张量 x，包含交替的 0 和 1，总共 18 个元素
    x = torch.tensor([0, 1] * 9)
    # 对张量 x 进行排序，返回排序后的结果（值和索引），注意这里返回的是一个命名元组类型
    torch.return_types.sort(
        values=tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        indices=tensor([ 2, 16,  4,  6, 14,  8,  0, 10, 12,  9, 17, 15, 13, 11,  7,  5,  3,  1]))
    
    # 对张量 x 进行排序，设置 stable=True 以确保稳定排序，返回排序后的结果（值和索引）
    torch.return_types.sort(
        values=tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        indices=tensor([ 0,  2,  4,  6,  8, 10, 12, 14, 16,  1,  3,  5,  7,  9, 11, 13, 15, 17]))
# 为 torch.argsort 函数添加文档字符串
add_docstr(
    torch.argsort,
    r"""
argsort(input, dim=-1, descending=False, stable=False) -> Tensor

Returns the indices that sort a tensor along a given dimension in ascending
order by value.

This is the second value returned by :meth:`torch.sort`.  See its documentation
for the exact semantics of this method.

If :attr:`stable` is ``True`` then the sorting routine becomes stable, preserving
the order of equivalent elements. If ``False``, the relative order of values
which compare equal is not guaranteed. ``True`` is slower.

Args:
    {input}  # 参数 input: 要排序的输入张量
    dim (int, optional): the dimension to sort along  # dim (int, 可选): 沿其排序的维度
    descending (bool, optional): controls the sorting order (ascending or descending)  # descending (bool, 可选): 控制排序顺序 (升序或降序)
    stable (bool, optional): controls the relative order of equivalent elements  # stable (bool, 可选): 控制等效元素的相对顺序

Example::

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 0.0785,  1.5267, -0.8521,  0.4065],
            [ 0.1598,  0.0788, -0.0745, -1.2700],
            [ 1.2208,  1.0722, -0.7064,  1.2564],
            [ 0.0669, -0.2318, -0.8229, -0.9280]])

    >>> torch.argsort(a, dim=1)
    tensor([[2, 0, 3, 1],
            [3, 2, 1, 0],
            [2, 1, 0, 3],
            [3, 2, 1, 0]])
""".format(
        **common_args
    ),
)

# 为 torch.msort 函数添加文档字符串
add_docstr(
    torch.msort,
    r"""
msort(input, *, out=None) -> Tensor

Sorts the elements of the :attr:`input` tensor along its first dimension
in ascending order by value.

.. note:: `torch.msort(t)` is equivalent to `torch.sort(t, dim=0)[0]`.
          See also :func:`torch.sort`.

Args:
    {input}  # 参数 input: 要排序的输入张量

Keyword args:
    {out}  # 关键字参数 out: 输出张量 (可选)

Example::

    >>> t = torch.randn(3, 4)
    >>> t
    tensor([[-0.1321,  0.4370, -1.2631, -1.1289],
            [-2.0527, -1.1250,  0.2275,  0.3077],
            [-0.0881, -0.1259, -0.5495,  1.0284]])
    >>> torch.msort(t)
    tensor([[-2.0527, -1.1250, -1.2631, -1.1289],
            [-0.1321, -0.1259, -0.5495,  0.3077],
            [-0.0881,  0.4370,  0.2275,  1.0284]])
""".format(
        **common_args
    ),
)

# 为 torch.sparse_compressed_tensor 函数添加文档字符串
add_docstr(
    torch.sparse_compressed_tensor,
    r"""sparse_compressed_tensor(compressed_indices, plain_indices, values, size=None, """
    r"""*, dtype=None, layout=None, device=None, requires_grad=False, check_invariants=None) -> Tensor

Constructs a :ref:`sparse tensor in Compressed Sparse format - CSR,
CSC, BSR, or BSC - <sparse-compressed-docs>` with specified values at
the given :attr:`compressed_indices` and :attr:`plain_indices`. Sparse
matrix multiplication operations in Compressed Sparse format are
typically faster than that for sparse tensors in COO format. Make you
have a look at :ref:`the note on the data type of the indices
<sparse-compressed-docs>`.

{sparse_factory_device_note}  # sparse_factory_device_note: 稀疏张量工厂函数说明

Args:
    # 压缩索引 (array_like): 大小为 (*batchsize, compressed_dim_size + 1) 的 (B+1) 维数组。
    # 每个批次的最后一个元素表示非零元素或块的数量。这个张量根据给定的压缩维度（行或列）的起始位置，
    # 编码了在 values 和 plain_indices 中的索引。张量中每个连续的数字减去前一个数字表示给定压缩维度中元素或块的数量。

    # plain_indices (array_like): values 中每个元素或块的普通维度（行或列）坐标。
    # 是一个与 values 长度相同的 (B+1) 维张量。

    # values (array_list): 张量的初始值。可以是列表、元组、NumPy ndarray、标量或其他类型。
    # 表示 CSR 和 CSC 布局的 (1+K) 维张量，或表示 BSR 和 BSC 布局的 (1+2+K) 维张量，其中 K 是稠密维度的数量。

    # size (list, tuple, :class:`torch.Size`, 可选): 稀疏张量的大小。
    # 形状为 (*batchsize, nrows * blocksize[0], ncols * blocksize[1], *densesize)，
    # 其中对于 CSR 和 CSC 格式，blocksize[0] == blocksize[1] == 1。
    # 如果未提供，大小将被推断为足以容纳所有非零元素或块的最小大小。
# 导入所需的torch库中的sparse_csr_tensor函数
add_docstr(
    torch.sparse_csr_tensor,
    r"""sparse_csr_tensor(crow_indices, col_indices, values, size=None, """
    r"""*, dtype=None, device=None, requires_grad=False, check_invariants=None) -> Tensor

Constructs a :ref:`sparse tensor in CSR (Compressed Sparse Row) <sparse-csr-docs>` with specified
values at the given :attr:`crow_indices` and :attr:`col_indices`. Sparse matrix multiplication operations
in CSR format are typically faster than that for sparse tensors in COO format. Make you have a look
at :ref:`the note on the data type of the indices <sparse-csr-docs>`.

{sparse_factory_device_note}

Args:
    crow_indices (array_like): (B+1)-dimensional array of size
        ``(*batchsize, nrows + 1)``.  The last element of each batch
        is the number of non-zeros. This tensor encodes the index in
        values and col_indices depending on where the given row
        starts. Each successive number in the tensor subtracted by the
        number before it denotes the number of elements in a given
        row.
    col_indices (array_like): Column co-ordinates of each element in
        values. (B+1)-dimensional tensor with the same length
        as values.
    values (array_list): Initial values for the tensor. Can be a list,
        tuple, NumPy ``ndarray``, scalar, and other types that
        represents a (1+K)-dimensional tensor where ``K`` is the number
        of dense dimensions.
"""
)
    size (list, tuple, :class:`torch.Size`, optional): Size of the
        sparse tensor: ``(*batchsize, nrows, ncols, *densesize)``. If
        not provided, the size will be inferred as the minimum size
        big enough to hold all non-zero elements.
# 定义了一个函数 add_docstr，用于为指定的函数或方法添加文档字符串
def add_docstr(
    # 为 torch.sparse_csc_tensor 函数添加文档字符串，描述其功能和用法
    torch.sparse_csc_tensor,
    r"""sparse_csc_tensor(ccol_indices, row_indices, values, size=None, """
    r"""*, dtype=None, device=None, requires_grad=False, check_invariants=None) -> Tensor
    # 构造一个稀疏的 CSC 格式的张量，使用给定的 ccol_indices、row_indices 和 values 参数
    Constructs a :ref:`sparse tensor in CSC (Compressed Sparse Column)
    <sparse-csc-docs>` with specified values at the given
    :attr:`ccol_indices` and :attr:`row_indices`. Sparse matrix
    multiplication operations in CSC format are typically faster than that
    for sparse tensors in COO format. Make you have a look at :ref:`the
    note on the data type of the indices <sparse-csc-docs>`.

    {sparse_factory_device_note}

    Args:
        # ccol_indices 参数：(B+1) 维数组，大小为 (*batchsize, ncols + 1)。每个批次的最后一个元素表示非零元素的数量。
        ccol_indices (array_like): (B+1)-dimensional array of size
            ``(*batchsize, ncols + 1)``.  The last element of each batch
            is the number of non-zeros. This tensor encodes the index in
            values and row_indices depending on where the given column
            starts. Each successive number in the tensor subtracted by the
            number before it denotes the number of elements in a given
            column.
        # row_indices 参数：values 中每个元素的行坐标。与 values 长度相同的 (B+1) 维张量。
        row_indices (array_like): Row co-ordinates of each element in
            values. (B+1)-dimensional tensor with the same length as
            values.
        # values 参数：张量的初始值。可以是列表、元组、NumPy ndarray、标量或其他类型，表示 (1+K) 维张量，其中 K 是密集维度的数量。
        values (array_list): Initial values for the tensor. Can be a list,
            tuple, NumPy ``ndarray``, scalar, and other types that
            represents a (1+K)-dimensional tensor where ``K`` is the number
            of dense dimensions.
        # size 参数：稀疏张量的尺寸，(*batchsize, nrows, ncols, *densesize)。如果未提供，则将推断为足以容纳所有非零元素的最小尺寸。
        size (list, tuple, :class:`torch.Size`, optional): Size of the
            sparse tensor: ``(*batchsize, nrows, ncols, *densesize)``. If
            not provided, the size will be inferred as the minimum size
            big enough to hold all non-zero elements.

    Keyword args:
        # dtype 参数：返回张量的期望数据类型。如果未指定，默认从 values 推断数据类型。
        dtype (:class:`torch.dtype`, optional): the desired data type of
            returned tensor.  Default: if None, infers data type from
            :attr:`values`.
        # device 参数：返回张量的期望设备。如果未指定，默认使用当前设备作为默认张量类型的设备。
        device (:class:`torch.device`, optional): the desired device of
            returned tensor.  Default: if None, uses the current device
            for the default tensor type (see
            :func:`torch.set_default_device`). :attr:`device` will be
            the CPU for CPU tensor types and the current CUDA device for
            CUDA tensor types.
        # requires_grad 参数：是否要求计算梯度。默认为 False。
        {requires_grad}
        # check_invariants 参数：检查不变量。默认为 None。
        {check_invariants}
"""
    ),
)
    # 定义函数参数 dtype，指定返回张量的数据类型
    dtype (:class:`torch.dtype`, optional): the desired data type of
        returned tensor.  Default: if None, infers data type from
        :attr:`values`.
    # 定义函数参数 device，指定返回张量的设备
    device (:class:`torch.device`, optional): the desired device of
        returned tensor.  Default: if None, uses the current device
        for the default tensor type (see
        :func:`torch.set_default_device`). :attr:`device` will be
        the CPU for CPU tensor types and the current CUDA device for
        CUDA tensor types.
    # 控制是否计算梯度，根据 {requires_grad} 参数确定
    {requires_grad}
    # 检查张量返回前的不变性条件，根据 {check_invariants} 参数确定
    {check_invariants}
# 构造一个稀疏的 BSR (Block Compressed Sparse Row) 格式的张量
# 使用给定的列块索引和行块索引，以及对应的数值，来创建张量
torch.sparse_bsr_tensor(
    torch.tensor(ccol_indices, dtype=torch.int64),  # 列块索引，数据类型为64位整数
    torch.tensor(row_indices, dtype=torch.int64),   # 行块索引，数据类型为64位整数
    torch.tensor(values),                           # 数值列表，用于初始化张量
    dtype=torch.double                             # 张量的数据类型为双精度浮点数
)
tensor(
    ccol_indices=tensor([0, 2, 4]),                 # 列块索引张量
    row_indices=tensor([0, 1, 0, 1]),               # 行块索引张量
    values=tensor([1., 2., 3., 4.]),                # 值张量，已转换为双精度浮点数
    size=(2, 2),                                    # 张量的形状为(2, 2)
    nnz=4,                                          # 非零元素的数量为4
    dtype=torch.float64,                            # 数据类型为双精度浮点数
    layout=torch.sparse_csc                         # 稀疏张量的布局为 CSC (Compressed Sparse Column)
)
    # 定义稀疏张量的行索引
    crow_indices = [0, 1, 2]
    # 定义稀疏张量的列索引
    col_indices = [0, 1]
    # 定义稀疏张量的数值，是一个三维列表
    values = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    # 使用 torch.sparse_bsr_tensor 函数创建稀疏块压缩行 (BSR) 格式的张量
    torch.sparse_bsr_tensor(
        # 将行索引转换为 torch 张量，并指定数据类型为 int64
        torch.tensor(crow_indices, dtype=torch.int64),
        # 将列索引转换为 torch 张量，并指定数据类型为 int64
        torch.tensor(col_indices, dtype=torch.int64),
        # 将数值列表转换为 torch 张量，并指定数据类型为 double
        torch.tensor(values), dtype=torch.double
    )
    # 返回的张量对象包含稀疏张量的行索引、列索引、数值，以及张量的形状、非零元素数量、数据类型和布局信息
    tensor(
        crow_indices=tensor([0, 1, 2]),
        col_indices=tensor([0, 1]),
        values=tensor([[[1., 2.],
                        [3., 4.]],
                       [[5., 6.],
                        [7., 8.]]]),
        size=(2, 2),  # 稀疏张量的形状为 (2, 2)
        nnz=2,        # 非零元素的数量为 2
        dtype=torch.float64,  # 数据类型为 double
        layout=torch.sparse_bsr  # 张量的布局为稀疏块压缩行 (BSR) 格式
    )
""".format(
        **factory_common_args
    ),
)

add_docstr(
    torch.sparse_bsc_tensor,
    r"""sparse_bsc_tensor(ccol_indices, row_indices, values, size=None, """
    r"""*, dtype=None, device=None, requires_grad=False, check_invariants=None) -> Tensor

Constructs a :ref:`sparse tensor in BSC (Block Compressed Sparse
Column)) <sparse-bsc-docs>` with specified 2-dimensional blocks at the
given :attr:`ccol_indices` and :attr:`row_indices`. Sparse matrix
multiplication operations in BSC format are typically faster than that
for sparse tensors in COO format. Make you have a look at :ref:`the
note on the data type of the indices <sparse-bsc-docs>`.

{sparse_factory_device_note}

Args:
    ccol_indices (array_like): (B+1)-dimensional array of size
        ``(*batchsize, ncolblocks + 1)``. The last element of each
        batch is the number of non-zeros. This tensor encodes the
        index in values and row_indices depending on where the given
        column starts. Each successive number in the tensor subtracted
        by the number before it denotes the number of elements in a
        given column.
    row_indices (array_like): Row block co-ordinates of each block in
        values. (B+1)-dimensional tensor with the same length
        as values.
    values (array_list): Initial blocks for the tensor. Can be a list,
        tuple, NumPy ``ndarray``, and other types that
        represents a (1 + 2 + K)-dimensional tensor where ``K`` is the
        number of dense dimensions.
    size (list, tuple, :class:`torch.Size`, optional): Size of the
        sparse tensor: ``(*batchsize, nrows * blocksize[0], ncols *
        blocksize[1], *densesize)`` If not provided, the size will be
        inferred as the minimum size big enough to hold all non-zero
        blocks.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of
        returned tensor.  Default: if None, infers data type from
        :attr:`values`.
    device (:class:`torch.device`, optional): the desired device of
        returned tensor.  Default: if None, uses the current device
        for the default tensor type (see
        :func:`torch.set_default_device`). :attr:`device` will be
        the CPU for CPU tensor types and the current CUDA device for
        CUDA tensor types.
    {requires_grad}
    {check_invariants}

Example::
    >>> ccol_indices = [0, 1, 2]
    >>> row_indices = [0, 1]
    >>> values = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    >>> torch.sparse_bsc_tensor(torch.tensor(ccol_indices, dtype=torch.int64),
    ...                         torch.tensor(row_indices, dtype=torch.int64),
    ...                         torch.tensor(values), dtype=torch.double)
"""


注释：
-   这段代码定义了一个函数 `sparse_bsc_tensor`，用于创建 BSC 格式的稀疏张量。
-   函数的参数说明包括 `ccol_indices`、`row_indices`、`values`、`size`等，详细描述了每个参数的含义和用法。
-   文档中还包含了关于 `dtype`、`device`、`requires_grad` 和 `check_invariants` 等关键字参数的说明。
-   例子部分展示了如何调用这个函数来创建稀疏张量的示例。
    # 创建稀疏张量对象，指定列索引和行索引，并提供数据值
    tensor(ccol_indices=tensor([0, 1, 2]),  # 列索引为 [0, 1, 2]
           row_indices=tensor([0, 1]),     # 行索引为 [0, 1]
           values=tensor([[[1., 2.],       # 数据值为 [[[1., 2.],
                           [3., 4.]],      #             [3., 4.]]
                          [[5., 6.],       #            [[5., 6.],
                           [7., 8.]]]),    #             [7., 8.]]]
           size=(2, 2),                    # 稀疏张量的尺寸为 (2, 2)
           nnz=2,                          # 非零元素的数量为 2
           dtype=torch.float64,            # 数据类型为双精度浮点数
           layout=torch.sparse_bsc)        # 稀疏张量的布局为 torch.sparse_bsc
# 调用`add_docstr`函数，为`torch.sparse_coo_tensor`添加文档字符串
add_docstr(
    # 调用`torch.sparse_coo_tensor`函数，构造一个COO格式的稀疏张量
    torch.sparse_coo_tensor,
    # 文档字符串描述函数签名及功能
    r"""sparse_coo_tensor(indices, values, size=None, """
    r"""*, dtype=None, device=None, requires_grad=False, check_invariants=None, is_coalesced=None) -> Tensor

Constructs a :ref:`sparse tensor in COO(rdinate) format
<sparse-coo-docs>` with specified values at the given
:attr:`indices`.

.. note::

   This function returns an :ref:`uncoalesced tensor
   <sparse-uncoalesced-coo-docs>` when :attr:`is_coalesced` is
   unspecified or ``None``.

{sparse_factory_device_note}

Args:
    indices (array_like): Initial data for the tensor. Can be a list, tuple,
        NumPy ``ndarray``, scalar, and other types. Will be cast to a :class:`torch.LongTensor`
        internally. The indices are the coordinates of the non-zero values in the matrix, and thus
        should be two-dimensional where the first dimension is the number of tensor dimensions and
        the second dimension is the number of non-zero values.
    values (array_like): Initial values for the tensor. Can be a list, tuple,
        NumPy ``ndarray``, scalar, and other types.
    size (list, tuple, or :class:`torch.Size`, optional): Size of the sparse tensor. If not
        provided the size will be inferred as the minimum size big enough to hold all non-zero
        elements.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if None, infers data type from :attr:`values`.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if None, uses the current device for the default tensor type
        (see :func:`torch.set_default_device`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.
    {requires_grad}
    {check_invariants}
    is_coalesced (bool, optional): When``True``, the caller is
        responsible for providing tensor indices that correspond to a
        coalesced tensor.  If the :attr:`check_invariants` flag is
        False, no error will be raised if the prerequisites are not
        met and this will lead to silently incorrect results. To force
        coalescion please use :meth:`coalesce` on the resulting
        Tensor.
        Default: None: except for trivial cases (e.g. nnz < 2) the
        resulting Tensor has is_coalesced set to ``False```py.

Example::

    >>> i = torch.tensor([[0, 1, 1],
    ...                   [2, 0, 2]])
    >>> v = torch.tensor([3, 4, 5], dtype=torch.float32)
    >>> torch.sparse_coo_tensor(i, v, [2, 4])
    tensor(indices=tensor([[0, 1, 1],
                           [2, 0, 2]]),
           values=tensor([3., 4., 5.]),
           size=(2, 4), nnz=3, layout=torch.sparse_coo)

    >>> torch.sparse_coo_tensor(i, v)  # Shape inference
"""
    # 创建稀疏 COO（Coordinate）张量，指定非零元素的索引、值和形状
    tensor(indices=tensor([[0, 1, 1],    # 索引张量，描述非零元素的位置
                           [2, 0, 2]]),   # 每列为一个非零元素的坐标
           values=tensor([3., 4., 5.]),  # 非零元素的值
           size=(2, 3),                  # 张量的形状
           nnz=3,                        # 非零元素的数量
           layout=torch.sparse_coo)      # 张量的布局，这里使用稀疏 COO 格式
    
    # 使用指定的索引张量、值张量和形状创建稀疏 COO 张量，指定了数据类型和设备
    >>> torch.sparse_coo_tensor(i, v, [2, 4],
    ...                         dtype=torch.float64,
    ...                         device=torch.device('cuda:0'))
    tensor(indices=tensor([[0, 1, 1],    # 索引张量，描述非零元素的位置
                           [2, 0, 2]]),   # 每列为一个非零元素的坐标
           values=tensor([3., 4., 5.]),  # 非零元素的值
           device='cuda:0',              # 张量所在的设备
           size=(2, 4),                  # 张量的形状
           nnz=3,                        # 非零元素的数量
           dtype=torch.float64,          # 张量的数据类型
           layout=torch.sparse_coo)      # 张量的布局，这里使用稀疏 COO 格式
    
    # 创建一个空的稀疏张量，满足以下不变性条件：
    #   1. 稀疏维度 + 密集维度 = SparseTensor.shape 的长度
    #   2. SparseTensor._indices().shape = (稀疏维度, nnz)
    #   3. SparseTensor._values().shape = (nnz, SparseTensor.shape[稀疏维度:])
    #
    # 例如，创建一个 nnz = 0，dense_dim = 0，sparse_dim = 1 的空稀疏张量
    >>> S = torch.sparse_coo_tensor(torch.empty([1, 0]), [], [1])
    tensor(indices=tensor([], size=(1, 0)),  # 索引张量为空，因为没有非零元素
           values=tensor([], size=(0,)),     # 值张量为空，因为没有非零元素
           size=(1,),                        # 张量的形状为 (1,)
           nnz=0,                            # 非零元素的数量为 0
           layout=torch.sparse_coo)          # 张量的布局，这里使用稀疏 COO 格式
    
    # 创建一个空的稀疏张量，满足以下不变性条件：
    #   1. 稀疏维度 + 密集维度 = SparseTensor.shape 的长度
    #   2. SparseTensor._indices().shape = (稀疏维度, nnz)
    #   3. SparseTensor._values().shape = (nnz, SparseTensor.shape[稀疏维度:])
    #
    # 例如，创建一个 nnz = 0，dense_dim = 1，sparse_dim = 1 的空稀疏张量
    >>> S = torch.sparse_coo_tensor(torch.empty([1, 0]), torch.empty([0, 2]), [1, 2])
    tensor(indices=tensor([], size=(1, 0)),  # 索引张量为空，因为没有非零元素
           values=tensor([], size=(0, 2)),   # 值张量为空，因为没有非零元素
           size=(1, 2),                      # 张量的形状为 (1, 2)
           nnz=0,                            # 非零元素的数量为 0
           layout=torch.sparse_coo)          # 张量的布局，这里使用稀疏 COO 格式
add_docstr(
    torch.std,
    r"""
std(input, dim=None, *, correction=1, keepdim=False, out=None) -> Tensor

计算沿着指定维度（由 :attr:`dim` 指定）的标准差。
:attr:`dim` 可以是单个维度、维度列表，或者 ``None`` 表示沿着所有维度求标准差。

如果输入是多维张量，标准差计算方式为：

.. math::
    \text{std}(X) = \sqrt{\frac{1}{N - \text{correction}} \sum_{i} (X_{i} - \bar{X})^2}

其中， \( \bar{X} \) 是输入的平均值， \( N \) 是输入的元素数量。如果 `correction` 设置为 0，
则计算方式为样本标准差；如果设置为 1，则计算总体标准差。

.. note:: 返回的张量与输入张量共享存储空间，因此修改一个会影响另一个的内容。

.. warning:: 如果输入张量在某些维度上大小为 1，则 `std(input)` 会移除这些维度，这可能导致意外的错误。建议只指定希望移除的维度。

Args:
    {input}
    dim (int or tuple of ints, optional): 如果给定，将仅在指定的维度上进行标准差计算。
           默认为 ``None``，表示在所有维度上计算标准差。

Keyword args:
    correction (int, optional): 用于计算标准差的修正因子，默认为 1，表示总体标准差。
                                设置为 0 表示样本标准差。
    keepdim (bool, optional): 如果为 ``True``，则保持输出张量的维度数不变。
    out (Tensor, optional): 输出张量的位置。

Example::

    >>> x = torch.randn(4, 4)
    >>> x
    tensor([[ 0.0206, -0.9997, -0.2594,  0.6605],
            [-0.0991,  0.3128, -0.9085, -0.6023],
            [ 0.4722, -0.6355,  0.8309, -0.2600],
            [-0.2215,  0.1585, -1.1811, -0.0837]])
    >>> torch.std(x)
    tensor(0.6078)
    >>> torch.std(x, dim=1)
    tensor([0.6021, 0.4992, 0.6774, 0.5772])
    >>> torch.std(x, dim=0, unbiased=False)
    tensor([0.2799, 0.5816, 0.8897, 0.4686])
""",
)
"""
Defines documentation for the function torch.std_mean.

std_mean(input, dim=None, *, correction=1, keepdim=False, out=None) -> (Tensor, Tensor)

Calculates the standard deviation and mean over the dimensions specified by
:attr:`dim`. :attr:`dim` can be a single dimension, list of dimensions, or
``None`` to reduce over all dimensions.

The standard deviation (:math:`\sigma`) is calculated as

.. math:: \sigma = \sqrt{\frac{1}{\max(0,~N - \delta N)}\sum_{i=0}^{N-1}(x_i-\bar{x})^2}

where :math:`x` is the sample set of elements, :math:`\bar{x}` is the
sample mean, :math:`N` is the number of samples and :math:`\delta N` is
the :attr:`correction`.

{keepdim_details}

Args:
    input (Tensor): input tensor containing data to be analyzed.
    opt_dim (int or list of ints, optional): dimensions over which to compute std and mean.
        Default is ``None``, meaning std and mean are computed over all dimensions.
    correction (int, optional): difference between the sample size and sample degrees of freedom.
        Defaults to `Bessel's correction`_, ``correction=1``.

        .. versionchanged:: 2.0
            Previously this argument was called ``unbiased`` and was a boolean
            with ``True`` corresponding to ``correction=1`` and ``False`` being
            ``correction=0``.
    keepdim (bool, optional): whether the output tensors have dim retained or not. Default is False.
    out (tuple, optional): tuple to place the output tensors into.

Returns:
    A tuple (std, mean) containing the standard deviation and mean.

Example:

    >>> a = torch.tensor(
    ...     [[ 0.2035,  1.2959,  1.8101, -0.4644],
    ...      [ 1.5027, -0.3270,  0.5905,  0.6538],
    ...      [-1.5745,  1.3330, -0.5596, -0.6548],
    ...      [ 0.1264, -0.5080,  1.6420,  0.1992]])
    >>> torch.std_mean(a, dim=0, keepdim=True)
    (tensor([[1.2620, 1.0028, 1.0957, 0.6038]]),
     tensor([[ 0.0645,  0.4485,  0.8707, -0.0665]]))

.. _Bessel's correction: https://en.wikipedia.org/wiki/Bessel%27s_correction
"""

add_docstr(
    torch.std_mean,
    """
Adds documentation to the torch.std_mean function.

This function calculates the standard deviation and mean over specified dimensions of the input tensor.

{keepdim_details}

Args:
    input (Tensor): Input tensor containing the data.
    dim (int or list of ints, optional): Dimensions over which to compute std and mean. Default is None.
    correction (int, optional): Difference between sample size and degrees of freedom. Default is 1.
    keepdim (bool, optional): Whether to retain the output tensor's dimensions or not. Default is False.
    out (tuple, optional): Tuple to place the output tensors into.

Keyword args:
    correction (int): Difference between the sample size and sample degrees of freedom.
        Defaults to `Bessel's correction`_, ``correction=1``.

        .. versionchanged:: 2.0
            Previously this argument was called ``unbiased`` and was a boolean
            with ``True`` corresponding to ``correction=1`` and ``False`` being
            ``correction=0``.
    {keepdim}
    {out}

Example:

    >>> a = torch.tensor(
    ...     [[ 0.2035,  1.2959,  1.8101, -0.4644],
    ...      [ 1.5027, -0.3270,  0.5905,  0.6538],
    ...      [-1.5745,  1.3330, -0.5596, -0.6548],
    ...      [ 0.1264, -0.5080,  1.6420,  0.1992]])
    >>> torch.std_mean(a, dim=0, keepdim=True)
    (tensor([[1.2620, 1.0028, 1.0957, 0.6038]]),
     tensor([[ 0.0645,  0.4485,  0.8707, -0.0665]]))

.. _Bessel's correction: https://en.wikipedia.org/wiki/Bessel%27s_correction
""".format(
        **multi_dim_common
    ),
)
# 添加函数文档字符串到 torch.sub 函数
add_docstr(
    torch.sub,
    r"""
sub(input, other, *, alpha=1, out=None) -> Tensor

Subtracts :attr:`other`, scaled by :attr:`alpha`, from :attr:`input`.

.. math::
    \text{{out}}_i = \text{{input}}_i - \text{{alpha}} \times \text{{other}}_i
"""
    + r"""

Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`,
:ref:`type promotion <type-promotion-doc>`, and integer, float, and complex inputs.

Args:
    {input}
    other (Tensor or Number): the tensor or number to subtract from :attr:`input`.

Keyword args:
    alpha (Number): the multiplier for :attr:`other`.
    {out}

Example::

    >>> a = torch.tensor((1, 2))
    >>> b = torch.tensor((0, 1))
    >>> torch.sub(a, b, alpha=2)
    tensor([1, 0])
""".format(
        **common_args
    ),
)

# 添加函数文档字符串到 torch.subtract 函数，作为 torch.sub 的别名
add_docstr(
    torch.subtract,
    r"""
subtract(input, other, *, alpha=1, out=None) -> Tensor

Alias for :func:`torch.sub`.
""",
)

# 添加函数文档字符串到 torch.sum 函数
add_docstr(
    torch.sum,
    r"""
sum(input, *, dtype=None) -> Tensor

Returns the sum of all elements in the :attr:`input` tensor.

Args:
    {input}

Keyword args:
    {dtype}

Example::

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[ 0.1133, -0.9567,  0.2958]])
    >>> torch.sum(a)
    tensor(-0.5475)

.. function:: sum(input, dim, keepdim=False, *, dtype=None) -> Tensor
   :noindex:

Returns the sum of each row of the :attr:`input` tensor in the given
dimension :attr:`dim`. If :attr:`dim` is a list of dimensions,
reduce over all of them.

{keepdim_details}

Args:
    {input}
    {opt_dim}
    {keepdim}

Keyword args:
    {dtype}

Example::

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 0.0569, -0.2475,  0.0737, -0.3429],
            [-0.2993,  0.9138,  0.9337, -1.6864],
            [ 0.1132,  0.7892, -0.1003,  0.5688],
            [ 0.3637, -0.9906, -0.4752, -1.5197]])
    >>> torch.sum(a, 1)
    tensor([-0.4598, -0.1381,  1.3708, -2.6217])
    >>> b = torch.arange(4 * 5 * 6).view(4, 5, 6)
    >>> torch.sum(b, (2, 1))
    tensor([  435.,  1335.,  2235.,  3135.])
""".format(
        **multi_dim_common
    ),
)

# 添加函数文档字符串到 torch.nansum 函数
add_docstr(
    torch.nansum,
    r"""
nansum(input, *, dtype=None) -> Tensor

Returns the sum of all elements, treating Not a Numbers (NaNs) as zero.

Args:
    {input}

Keyword args:
    {dtype}

Example::

    >>> a = torch.tensor([1., 2., float('nan'), 4.])
    >>> torch.nansum(a)
    tensor(7.)

.. function:: nansum(input, dim, keepdim=False, *, dtype=None) -> Tensor
   :noindex:

Returns the sum of each row of the :attr:`input` tensor in the given
dimension :attr:`dim`, treating Not a Numbers (NaNs) as zero.
If :attr:`dim` is a list of dimensions, reduce over all of them.

{keepdim_details}

Args:
    {input}
    {opt_dim}
    {keepdim}

Keyword args:
    {dtype}

Example::

    >>> torch.nansum(torch.tensor([1., float("nan")]))
    1.0
    >>> a = torch.tensor([[1, 2], [3., float("nan")]])
    >>> torch.nansum(a)
    tensor(6.)
    >>> torch.nansum(a, dim=0)
    tensor([4., 2.])
    >>> torch.nansum(a, dim=1)

"""
)
    # 创建一个张量，包含两个元素，值分别为 3.0 和 3.0
    tensor([3., 3.])
# 格式化字符串，使用 multi_dim_common 字典中的内容填充
""".format(
    **multi_dim_common
),

# 添加文档字符串给 torch.svd 函数
add_docstr(
    torch.svd,
    r"""
svd(input, some=True, compute_uv=True, *, out=None) -> (Tensor, Tensor, Tensor)

计算矩阵或批量矩阵 :attr:`input` 的奇异值分解。奇异值分解表示为命名元组 `(U, S, V)`，使得 :attr:`input` = U diag(S) V^H，
其中对于实数输入，:math:`V^H` 是 `V` 的转置，对于复数输入，是 `V` 的共轭转置。
如果 :attr:`input` 是批量矩阵，则 `U`、`S` 和 `V` 也具有与 :attr:`input` 相同的批量维度。

如果 :attr:`some` 是 `True`（默认），方法返回减少的奇异值分解。在这种情况下，如果 :attr:`input` 的最后两个维度为 `m` 和 `n`，
则返回的 `U` 和 `V` 矩阵将只包含 `min(n, m)` 个正交列。

如果 :attr:`compute_uv` 是 `False`，返回的 `U` 和 `V` 将是填充为零的形状为 `(m, m)` 和 `(n, n)` 的矩阵，
与 :attr:`input` 相同的设备。当 :attr:`compute_uv` 是 `False` 时，参数 :attr:`some` 不起作用。

支持 float、double、cfloat 和 cdouble 数据类型的 :attr:`input`。
`U` 和 `V` 的数据类型与 :attr:`input` 相同。`S` 总是实数，即使 :attr:`input` 是复数。

.. warning::

    :func:`torch.svd` 已被弃用，建议使用 :func:`torch.linalg.svd`，并将在将来的 PyTorch 版本中移除。

    ``U, S, V = torch.svd(A, some=some, compute_uv=True)``（默认）应替换为

    .. code:: python

        U, S, Vh = torch.linalg.svd(A, full_matrices=not some)
        V = Vh.mH

    ``_, S, _ = torch.svd(A, some=some, compute_uv=False)`` 应替换为

    .. code:: python

        S = torch.linalg.svdvals(A)

.. note:: 与 :func:`torch.linalg.svd` 的不同之处：

             * :attr:`some` 与 :func:`torch.linalg.svd` 的 :attr:`full_matrices` 相反。注意默认值都是 `True`，因此默认行为实际上是相反的。
             * :func:`torch.svd` 返回 `V`，而 :func:`torch.linalg.svd` 返回 `Vh`，即 :math:`V^H`。
             * 如果 :attr:`compute_uv` 是 `False`，:func:`torch.svd` 返回填充为零的 `U` 和 `Vh` 张量，而 :func:`torch.linalg.svd` 返回空张量。

.. note:: 奇异值按降序排列返回。如果 :attr:`input` 是批量矩阵，则每个批量中的矩阵的奇异值按降序返回。

.. note:: 当 :attr:`compute_uv` 是 `True` 时，只有通过 :attr:`compute_uv` 为 `True` 时才能使用 `S` 张量来计算梯度。
""",
)
.. note:: 当 `some` 属性为 `False` 时，在反向传播过程中会忽略 `U[..., :, min(m, n):]` 和 `V[..., :, min(m, n):]` 上的梯度，因为这些向量可以是相应子空间的任意基向量。

.. note:: 在 CPU 上，:func:`torch.linalg.svd` 的实现使用 LAPACK 的 `?gesdd` 算法（分治算法），而不是 `?gesvd`，以提高速度。类似地，在 GPU 上，对于 CUDA 10.1.243 及更高版本，使用 cuSOLVER 的 `gesvdj` 和 `gesvdjBatched`，对于较早版本的 CUDA，使用 MAGMA 的 `gesdd` 算法。

.. note:: 返回的 `U` 张量可能不是连续的。矩阵（或矩阵的批量）将被表示为列主序（Fortran 连续）的矩阵。

.. warning:: 只有当输入不具有零或重复奇异值时，对 `U` 和 `V` 的梯度才是有限的。

.. warning:: 如果任意两个奇异值之间的距离接近零，则对 `U` 和 `V` 的梯度将在数值上不稳定，因为它们依赖于 :math:`\frac{1}{\min_{i \neq j} \sigma_i^2 - \sigma_j^2}`。当矩阵具有小的奇异值时，这些梯度也依赖于 `S^{-1}`。

.. warning:: 对于复值的 `input`，奇异值分解不是唯一的，因为 `U` 和 `V` 可能在每列上都乘以任意相位因子 :math:`e^{i \phi}`。当 `input` 具有重复奇异值时，可以将 `U` 和 `V` 中的跨度子空间的列乘以旋转矩阵，得到相同的向量空间。不同平台（如 NumPy）或不同设备类型的输入可能会产生不同的 `U` 和 `V` 张量。

Args:
    input (Tensor): 大小为 `(*, m, n)` 的输入张量，其中 `*` 是零个或多个批量维度，包含 `(m, n)` 矩阵。
    some (bool, 可选): 控制是否计算简化或完整分解，因此决定返回的 `U` 和 `V` 的形状。默认为 `True`。
    compute_uv (bool, 可选): 控制是否计算 `U` 和 `V`。默认为 `True`。

Keyword args:
    out (tuple, 可选): 输出张量的元组

Example::

    >>> a = torch.randn(5, 3)
    >>> a
    tensor([[ 0.2364, -0.7752,  0.6372],
            [ 1.7201,  0.7394, -0.0504],
            [-0.3371, -1.0584,  0.5296],
            [ 0.3550, -0.4022,  1.5569],
            [ 0.2445, -0.0158,  1.1414]])
    >>> u, s, v = torch.svd(a)
    >>> u
    tensor([[ 0.4027,  0.0287,  0.5434],
            [-0.1946,  0.8833,  0.3679],
            [ 0.4296, -0.2890,  0.5261],
            [ 0.6604,  0.2717, -0.2618],
            [ 0.4234,  0.2481, -0.4733]])
    >>> s
    tensor([2.3289, 2.0315, 0.7806])
    >>> v
    # 创建一个 3x3 的张量，包含指定的数值
    tensor([[-0.0199,  0.8766,  0.4809],
            [-0.5080,  0.4054, -0.7600],
            [ 0.8611,  0.2594, -0.4373]])
    # 计算张量 a 与奇异值分解的重构结果之间的欧几里得距离
    >>> torch.dist(a, torch.mm(torch.mm(u, torch.diag(s)), v.t()))
    tensor(8.6531e-07)
    # 创建一个大小为 7x5x3 的随机张量
    >>> a_big = torch.randn(7, 5, 3)
    # 对 a_big 进行奇异值分解，得到 u, s, v
    >>> u, s, v = torch.svd(a_big)
    # 计算张量 a_big 与奇异值分解的重构结果之间的欧几里得距离
    >>> torch.dist(a_big, torch.matmul(torch.matmul(u, torch.diag_embed(s)), v.t()))
    tensor(2.6503e-06)
# 为 torch.t 函数添加注释，说明其功能和使用方法
def t(input) -> Tensor:
    """
    t(input) -> Tensor

    Expects :attr:`input` to be <= 2-D tensor and transposes dimensions 0
    and 1.

    0-D and 1-D tensors are returned as is. When input is a 2-D tensor this
    is equivalent to ``transpose(input, 0, 1)``.

    Args:
        input (Tensor): The input tensor to transpose.

    Example::

        >>> x = torch.randn(())
        >>> x
        tensor(0.1995)
        >>> torch.t(x)
        tensor(0.1995)
        >>> x = torch.randn(3)
        >>> x
        tensor([ 2.4320, -0.4608,  0.7702])
        >>> torch.t(x)
        tensor([ 2.4320, -0.4608,  0.7702])
        >>> x = torch.randn(2, 3)
        >>> x
        tensor([[ 0.4875,  0.9158, -0.5872],
                [ 0.3938, -0.6929,  0.6932]])
        >>> torch.t(x)
        tensor([[ 0.4875,  0.3938],
                [ 0.9158, -0.6929],
                [-0.5872,  0.6932]])

    See also :func:`torch.transpose`.
    """



# 为 torch.flip 函数添加注释，解释其功能和使用方法
def flip(input, dims) -> Tensor:
    """
    flip(input, dims) -> Tensor

    Reverse the order of an n-D tensor along given axis in dims.

    Args:
        input (Tensor): The input tensor to flip.
        dims (list or tuple): The axis or axes to flip on.

    Example::

        >>> x = torch.arange(8).view(2, 2, 2)
        >>> x
        tensor([[[ 0,  1],
                 [ 2,  3]],

                [[ 4,  5],
                 [ 6,  7]]])
        >>> torch.flip(x, [0, 1])
        tensor([[[ 6,  7],
                 [ 4,  5]],

                [[ 2,  3],
                 [ 0,  1]]])
    """



# 为 torch.fliplr 函数添加注释，说明其功能和使用方法
def fliplr(input) -> Tensor:
    """
    fliplr(input) -> Tensor

    Flip tensor in the left/right direction, returning a new tensor.

    Note:
        Requires the tensor to be at least 2-D.

    Args:
        input (Tensor): The input tensor to flip left/right.

    Example::

        >>> x = torch.arange(4).view(2, 2)
        >>> x
        tensor([[0, 1],
                [2, 3]])
        >>> torch.fliplr(x)
        tensor([[1, 0],
                [3, 2]])
    """



# 为 torch.flipud 函数添加注释，解释其功能和使用方法
def flipud(input) -> Tensor:
    """
    flipud(input) -> Tensor

    Flip tensor in the up/down direction, returning a new tensor.

    Note:
        Requires the tensor to be at least 1-D.

    Args:
        input (Tensor): The input tensor to flip up/down.

    Example::

        >>> x = torch.arange(4)
        >>> x
        tensor([0, 1, 2, 3])
        >>> torch.flipud(x)
        tensor([3, 2, 1, 0])
    """
    # `torch.flipud`对输入的数据进行上下翻转，并返回一个数据的副本。
    # 这与NumPy的`np.flipud`不同，后者返回一个常数时间内的视图而不是副本。
    # 由于复制张量数据比查看数据更耗费资源，
    # 所以`torch.flipud`预计比`np.flipud`更慢。
# 从 torch 模块导入 take 函数
from torch import take

# 为 take 函数添加文档字符串，说明其功能和使用示例
add_docstr(
    take,
    r"""
take(input, index) -> Tensor

Returns a new tensor with the elements of :attr:`input` at the given indices.
The input tensor is treated as if it were viewed as a 1-D tensor. The result
takes the same shape as the indices.

Args:
    {input}  # input 参数的说明，通常是一个张量
    index (LongTensor): the indices into tensor  # LongTensor 类型的索引张量

Example::

    >>> src = torch.tensor([[4, 3, 5],  # 创建一个源张量 src
    ...                     [6, 7, 8]])
    >>> torch.take(src, torch.tensor([0, 2, 5]))  # 使用 take 函数获取指定索引处的元素
    tensor([ 4,  5,  8])

""".format(
        **common_args
    ),  # 使用 common_args 插入通用参数
)
"""
add_docstr(
    torch.take_along_dim,
    r"""
take_along_dim(input, indices, dim=None, *, out=None) -> Tensor

Selects values from :attr:`input` at the 1-dimensional indices from :attr:`indices` along the given :attr:`dim`.

If :attr:`dim` is None, the input array is treated as if it has been flattened to 1d.

Functions that return indices along a dimension, like :func:`torch.argmax` and :func:`torch.argsort`,
are designed to work with this function. See the examples below.

.. note::
    This function is similar to NumPy's `take_along_axis`.
    See also :func:`torch.gather`.

Args:
    {input}
    indices (tensor): the indices into :attr:`input`. Must have long dtype.
    dim (int, optional): dimension to select along.

Keyword args:
    {out}

Example::

    >>> t = torch.tensor([[10, 30, 20], [60, 40, 50]])
    >>> max_idx = torch.argmax(t)
    >>> torch.take_along_dim(t, max_idx)
    tensor([60])
    >>> sorted_idx = torch.argsort(t, dim=1)
    >>> torch.take_along_dim(t, sorted_idx, dim=1)
    tensor([[10, 20, 30],
            [40, 50, 60]])
""".format(
        **common_args
    ),
)

add_docstr(
    torch.tan,
    r"""
tan(input, *, out=None) -> Tensor

Returns a new tensor with the tangent of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \tan(\text{input}_{i})
"""
    + r"""
Args:
    {input}

Keyword args:
    {out}

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-1.2027, -1.7687,  0.4412, -1.3856])
    >>> torch.tan(a)
    tensor([-2.5930,  4.9859,  0.4722, -5.3366])
""".format(
        **common_args
    ),
)

add_docstr(
    torch.tanh,
    r"""
tanh(input, *, out=None) -> Tensor

Returns a new tensor with the hyperbolic tangent of the elements
of :attr:`input`.

.. math::
    \text{out}_{i} = \tanh(\text{input}_{i})
"""
    + r"""
Args:
    {input}

Keyword args:
    {out}

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.8986, -0.7279,  1.1745,  0.2611])
    >>> torch.tanh(a)
    tensor([ 0.7156, -0.6218,  0.8257,  0.2553])
""".format(
        **common_args
    ),
)

add_docstr(
    # torch.softmax doc str. Point this to torch.nn.functional.softmax
    torch.softmax,
    r"""
softmax(input, dim, *, dtype=None) -> Tensor

Alias for :func:`torch.nn.functional.softmax`.
""",
)

add_docstr(
    torch.topk,
    r"""
topk(input, k, dim=None, largest=True, sorted=True, *, out=None) -> (Tensor, LongTensor)

Returns the :attr:`k` largest elements of the given :attr:`input` tensor along
a given dimension.

If :attr:`dim` is not given, the last dimension of the `input` is chosen.

If :attr:`largest` is ``False`` then the `k` smallest elements are returned.

A namedtuple of `(values, indices)` is returned with the `values` and
`indices` of the largest `k` elements of each row of the `input` tensor in the
given dimension `dim`.

The boolean option :attr:`sorted` if ``True``, will make sure that the returned
`k` elements are themselves sorted

Args:
    k (int): "top-k" 中的 k 值，表示返回前 k 个元素
    dim (int, optional): 需要排序的维度
    largest (bool, optional): 控制是否返回最大的元素，True 表示返回最大的元素，False 表示返回最小的元素
    sorted (bool, optional): 控制返回的元素是否按顺序排列，True 表示按顺序排列，False 表示不按顺序排列
Keyword args:
    out (tuple, optional): the output tuple of (Tensor, LongTensor) that can be
        optionally given to be used as output buffers



Example::

    >>> x = torch.arange(1., 6.)
    >>> x
    tensor([ 1.,  2.,  3.,  4.,  5.])
    >>> torch.topk(x, 3)
    torch.return_types.topk(values=tensor([5., 4., 3.]), indices=tensor([4, 3, 2]))



""".format(
        **common_args
    ),
)

add_docstr(
    torch.trace,
    r"""
trace(input) -> Tensor

Returns the sum of the elements of the diagonal of the input 2-D matrix.

Example::

    >>> x = torch.arange(1., 10.).view(3, 3)
    >>> x
    tensor([[ 1.,  2.,  3.],
            [ 4.,  5.,  6.],
            [ 7.,  8.,  9.]])
    >>> torch.trace(x)
    tensor(15.)
""",
)

add_docstr(
    torch.transpose,
    r"""
transpose(input, dim0, dim1) -> Tensor

Returns a tensor that is a transposed version of :attr:`input`.
The given dimensions :attr:`dim0` and :attr:`dim1` are swapped.

If :attr:`input` is a strided tensor then the resulting :attr:`out`
tensor shares its underlying storage with the :attr:`input` tensor, so
changing the content of one would change the content of the other.

If :attr:`input` is a :ref:`sparse tensor <sparse-docs>` then the
resulting :attr:`out` tensor *does not* share the underlying storage
with the :attr:`input` tensor.

If :attr:`input` is a :ref:`sparse tensor <sparse-docs>` with compressed
layout (SparseCSR, SparseBSR, SparseCSC or SparseBSC) the arguments
:attr:`dim0` and :attr:`dim1` must be both batch dimensions, or must
both be sparse dimensions. The batch dimensions of a sparse tensor are the
dimensions preceding the sparse dimensions.

.. note::
    Transpositions which interchange the sparse dimensions of a `SparseCSR`
    or `SparseCSC` layout tensor will result in the layout changing between
    the two options. Transposition of the sparse dimensions of a ` SparseBSR`
    or `SparseBSC` layout tensor will likewise generate a result with the
    opposite layout.


Args:
    input (Tensor): the input tensor
    dim0 (int): the first dimension to be transposed
    dim1 (int): the second dimension to be transposed

Example::

    >>> x = torch.randn(2, 3)
    >>> x
    tensor([[ 1.0028, -0.9893,  0.5809],
            [-0.1669,  0.7299,  0.4942]])
    >>> torch.transpose(x, 0, 1)
    tensor([[ 1.0028, -0.1669],
            [-0.9893,  0.7299],
            [ 0.5809,  0.4942]])

See also :func:`torch.t`.
""".format(
        **common_args
    ),
)

add_docstr(
    torch.triangular_solve,
    r"""
triangular_solve(b, A, upper=True, transpose=False, unitriangular=False, *, out=None) -> (Tensor, Tensor)

Solves a system of equations with a square upper or lower triangular invertible matrix :math:`A`
and multiple right-hand sides :math:`b`.

In symbols, it solves :math:`AX = b` and assumes :math:`A` is square upper-triangular
(or lower-triangular if :attr:`upper`\ `= False`) and does not have zeros on the diagonal.

`torch.triangular_solve(b, A)` can take in 2D inputs `b, A` or inputs that are
"""
Returns the lower triangular part of the matrix (2-D tensor) or batch of matrices
`input`, setting all elements above the specified `diagonal` to 0.

The lower triangular part of a matrix consists of all elements on and below
the specified diagonal. Elements above the diagonal are set to 0.

Args:
    input (Tensor): the input tensor of shape `(N, N)` or `(B, N, N)` where `B` is the batch size.
    diagonal (int, optional): the diagonal to consider:
        - `diagonal = 0` (default): main diagonal and below.
        - `diagonal > 0`: elements above the main diagonal.
        - `diagonal < 0`: elements below the main diagonal.

Keyword args:
    out (Tensor, optional): the output tensor. If not `None`, results are written to this tensor.

Returns:
    Tensor: the lower triangular part of the input tensor, with all elements above the specified diagonal set to 0.
"""
# 将 torch.tril_indices 函数添加文档字符串
add_docstr(
    torch.tril_indices,
    r"""
tril_indices(row, col, offset=0, *, dtype=torch.long, device='cpu', layout=torch.strided) -> Tensor

Returns the indices of the lower triangular part of a :attr:`row`-by-
:attr:`col` matrix in a 2-by-N Tensor, where the first row contains row
coordinates of all indices and the second row contains column coordinates.
Indices are ordered based on rows and then columns.

The lower triangular part of the matrix is defined as the elements on and
below the diagonal.

The argument :attr:`offset` controls which diagonal to consider. If
:attr:`offset` = 0, all elements on and below the main diagonal are
retained. A positive value includes just as many diagonals above the main
diagonal, and similarly a negative value excludes just as many diagonals below
the main diagonal. The main diagonal are the set of indices
:math:`\lbrace (i, i) \rbrace` for :math:`i \in [0, \min\{d_{1}, d_{2}\} - 1]`
where :math:`d_{1}, d_{2}` are the dimensions of the matrix.

.. note::
    When running on CUDA, ``row * col`` must be less than :math:`2^{59}` to
    prevent overflow during calculation.
""",
)
    # 导入所需的模块requests和json
    import requests
    import json

    # 定义一个函数fetch_data，接收一个url参数
    def fetch_data(url):
        # 发送GET请求到指定的URL，返回响应对象
        response = requests.get(url)
        # 解析响应对象的JSON内容，并转换成Python字典
        data = response.json()
        # 返回解析后的数据字典
        return data
"""
triu(input, diagonal=0, *, out=None) -> Tensor

Returns the upper triangular part of a matrix (2-D tensor) or batch of matrices
:attr:`input`, the other elements of the result tensor :attr:`out` are set to 0.

The upper triangular part of the matrix is defined as the elements on and
above the diagonal.

The argument :attr:`diagonal` controls which diagonal to consider. If
:attr:`diagonal` = 0, all elements on and above the main diagonal are
retained. A positive value excludes just as many diagonals above the main
diagonal, and similarly a negative value includes just as many diagonals below
the main diagonal. The main diagonal are the set of indices
:math:`\lbrace (i, i) \rbrace` for :math:`i \in [0, \min\{d_{1}, d_{2}\} - 1]` where
:math:`d_{1}, d_{2}` are the dimensions of the matrix.

Args:
    input (:class:`torch.Tensor`): the input tensor to compute the upper triangular part from.
    diagonal (int, optional): which diagonal to consider (default: 0)

Keyword args:
    out (:class:`torch.Tensor`, optional): the output tensor (default: None)

Example::

    >>> a = torch.randn(3, 3)
    >>> a
    tensor([[ 0.2309,  0.5207,  2.0049],
            [ 0.2072, -1.0680,  0.6602],
            [ 0.3480, -0.5211, -0.4573]])
    >>> torch.triu(a)
    tensor([[ 0.2309,  0.5207,  2.0049],
            [ 0.0000, -1.0680,  0.6602],
            [ 0.0000,  0.0000, -0.4573]])
    >>> torch.triu(a, diagonal=1)
    tensor([[ 0.0000,  0.5207,  2.0049],
            [ 0.0000,  0.0000,  0.6602],
            [ 0.0000,  0.0000,  0.0000]])
    >>> torch.triu(a, diagonal=-1)
    tensor([[ 0.2309,  0.5207,  2.0049],
            [ 0.2072, -1.0680,  0.6602],
            [ 0.0000, -0.5211, -0.4573]])

    >>> b = torch.randn(4, 6)
    >>> b
    tensor([[ 0.5876, -0.0794, -1.8373,  0.6654,  0.2604,  1.5235],
            [-0.2447,  0.9556, -1.2919,  1.3378, -0.1768, -1.0857],
            [ 0.4333,  0.3146,  0.6576, -1.0432,  0.9348, -0.4410],
            [-0.9888,  1.0679, -1.3337, -1.6556,  0.4798,  0.2830]])
    >>> torch.triu(b, diagonal=1)
    tensor([[ 0.0000, -0.0794, -1.8373,  0.6654,  0.2604,  1.5235],
            [ 0.0000,  0.0000, -1.2919,  1.3378, -0.1768, -1.0857],
            [ 0.0000,  0.0000,  0.0000, -1.0432,  0.9348, -0.4410],
            [ 0.0000,  0.0000,  0.0000,  0.0000,  0.4798,  0.2830]])
"""
    # 创建一个 4x6 的张量，内容如下：
    # [[ 0.0000, -0.0794, -1.8373,  0.6654,  0.2604,  1.5235],
    #  [ 0.0000,  0.0000, -1.2919,  1.3378, -0.1768, -1.0857],
    #  [ 0.0000,  0.0000,  0.0000, -1.0432,  0.9348, -0.4410],
    #  [ 0.0000,  0.0000,  0.0000,  0.0000,  0.4798,  0.2830]]
    >>> torch.triu(b, diagonal=-1)
    # 对给定的张量 b 进行上三角矩阵的操作，将主对角线以下（diagonal=-1）的元素置零后返回新张量：
    # [[ 0.5876, -0.0794, -1.8373,  0.6654,  0.2604,  1.5235],
    #  [-0.2447,  0.9556, -1.2919,  1.3378, -0.1768, -1.0857],
    #  [ 0.0000,  0.3146,  0.6576, -1.0432,  0.9348, -0.4410],
    #  [ 0.0000,  0.0000, -1.3337, -1.6556,  0.4798,  0.2830]]
# 为 torch.triu_indices 函数添加文档字符串
add_docstr(
    torch.triu_indices,
    r"""
triu_indices(row, col, offset=0, *, dtype=torch.long, device='cpu', layout=torch.strided) -> Tensor

Returns the indices of the upper triangular part of a :attr:`row` by
:attr:`col` matrix in a 2-by-N Tensor, where the first row contains row
coordinates of all indices and the second row contains column coordinates.
Indices are ordered based on rows and then columns.

The upper triangular part of the matrix is defined as the elements on and
above the diagonal.

The argument :attr:`offset` controls which diagonal to consider. If
:attr:`offset` = 0, all elements on and above the main diagonal are
retained. A positive value excludes just as many diagonals above the main
diagonal, and similarly a negative value includes just as many diagonals below
the main diagonal. The main diagonal are the set of indices
:math:`\lbrace (i, i) \rbrace` for :math:`i \in [0, \min\{d_{1}, d_{2}\} - 1]`
where :math:`d_{1}, d_{2}` are the dimensions of the matrix.

.. note::
    When running on CUDA, ``row * col`` must be less than :math:`2^{59}` to
    prevent overflow during calculation.

Args:
    row (``int``): number of rows in the 2-D matrix.
    col (``int``): number of columns in the 2-D matrix.
    offset (``int``): diagonal offset from the main diagonal.
        Default: if not provided, 0.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if ``None``, ``torch.long``.
    {device}
    layout (:class:`torch.layout`, optional): currently only support ``torch.strided``.

Example::

    >>> a = torch.triu_indices(3, 3)
    >>> a
    tensor([[0, 0, 0, 1, 1, 2],
            [0, 1, 2, 1, 2, 2]])

    >>> a = torch.triu_indices(4, 3, -1)
    >>> a
    tensor([[0, 0, 0, 1, 1, 1, 2, 2, 3],
            [0, 1, 2, 0, 1, 2, 1, 2, 2]])

    >>> a = torch.triu_indices(4, 3, 1)
    >>> a
    tensor([[0, 0, 1],
            [1, 2, 2]])
""".format(
        **factory_common_args
    ),
)
"""
Returns a new tensor with the data in :attr:`input` fake quantized using :attr:`scale`,
:attr:`zero_point`, :attr:`quant_min` and :attr:`quant_max`.

.. math::
    \text{output} = (
        min(
            \text{quant\_max},
            max(
                \text{quant\_min},
                \text{std::nearby\_int}(\text{input} / \text{scale}) + \text{zero\_point}
            )
        ) - \text{zero\_point}
    ) \times \text{scale}

Args:
    input (Tensor): the input value(s), ``torch.float32`` tensor
    scale (double scalar or ``float32`` Tensor): quantization scale
    zero_point (int64 scalar or ``int32`` Tensor): quantization zero_point
    quant_min (int64): lower bound of the quantized domain
    quant_max (int64): upper bound of the quantized domain

Returns:
    Tensor: A newly fake_quantized ``torch.float32`` tensor

Example::

    >>> x = torch.randn(4)
    >>> x
    tensor([ 0.0552,  0.9730,  0.3973, -1.0780])
    >>> torch.fake_quantize_per_tensor_affine(x, 0.1, 0, 0, 255)
    tensor([0.1000, 1.0000, 0.4000, 0.0000])
    >>> torch.fake_quantize_per_tensor_affine(x, torch.tensor(0.1), torch.tensor(0), 0, 255)
    tensor([0.1000, 1.0000, 0.4000, 0.0000])
"""


注释：
# 定义函数 var_mean，计算输入张量沿指定维度的方差和均值
var_mean(input, dim=None, *, correction=1, keepdim=False, out=None) -> (Tensor, Tensor)

# 计算方差和均值的公式说明，其中方差使用 Bessel's 修正
Calculates the variance and mean over the dimensions specified by :attr:`dim`.
:attr:`dim` can be a single dimension, list of dimensions, or ``None`` to
reduce over all dimensions.

# 具体方差的计算公式
The variance (:math:`\sigma^2`) is calculated as

.. math:: \sigma^2 = \frac{1}{\max(0,~N - \delta N)}\sum_{i=0}^{N-1}(x_i-\bar{x})^2

# 其中 x 是样本元素集合，\bar{x} 是样本均值，N 是样本数，\delta N 是修正值 correction

# 张量参数 input 的说明
Args:
    {input}
    {opt_dim}

# 关键字参数列表，包括 Bessel's 修正、是否保持维度和输出参数
Keyword args:
    correction (int): difference between the sample size and sample degrees of freedom.
        Defaults to `Bessel's correction`_, ``correction=1``.

        .. versionchanged:: 2.0
            Previously this argument was called ``unbiased`` and was a boolean
            with ``True`` corresponding to ``correction=1`` and ``False`` being
            ``correction=0``.
    {keepdim}
    {out}

# 示例展示
Example:

    >>> a = torch.tensor(
    ...     [[ 0.2035,  1.2959,  1.8101, -0.4644],
    ...      [ 1.5027, -0.3270,  0.5905,  0.6538],
    ...      [-1.5745,  1.3330, -0.5596, -0.6548],
    ...      [ 0.1264, -0.5080,  1.6420,  0.1992]])
    >>> torch.var_mean(a, dim=1, keepdim=True)
    tensor([[1.0631],
            [0.5590],
            [1.4893],
            [0.8258]])

# Bessel's 修正的链接说明
.. _Bessel's correction: https://en.wikipedia.org/wiki/Bessel%27s_correction
# 定义了一个名为torch.empty的函数，返回一个形状由可变参数size定义的张量，元素值未初始化

add_docstr(
    torch.empty,
    """
empty(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False, \\
    memory_format=torch.contiguous_format) -> Tensor

Returns an uninitialized tensor of size :attr:`size` filled with zeros,
with the specified :attr:`dtype`, :attr:`layout`, :attr:`device`,
:attr:`requires_grad`, and :attr:`pin_memory` options.

Args:
    size (int...): a sequence of integers defining the shape of the output tensor.
        Can be a variable number of arguments or a collection like a list or tuple.

Keyword args:
    {out}
    {dtype}
    {layout}
    {device}
    {requires_grad}
    pin_memory (bool, optional): If ``True``, the returned tensor will be pinned
        and use pinned memory. Defaults to ``False``.
    {memory_format}

Example::

    >>> torch.empty(2, 3)
    tensor([[0., 0., 0.],
            [0., 0., 0.]])

    >>> torch.empty(5, dtype=torch.int32)
    tensor([0, 0, 0, 0, 0], dtype=torch.int32)
""".format(
        **factory_common_args
    ),
)
# 定义一个函数 `empty_like`，返回一个未初始化的张量，其形状与输入张量 `input` 相同
def empty_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) -> Tensor:
    # 如果 `torch.use_deterministic_algorithms()` 和 `torch.utils.deterministic.fill_uninitialized_memory` 均设置为 `True`，
    # 则初始化输出张量以避免将数据作为操作的输入时可能引起的任何非确定性行为。
    
    # 如果是浮点数或复数张量，则用 NaN 填充；如果是整数张量，则用最大值填充。
    
    # 参数 `input`：输入张量，新建的张量将与其大小相同
    {input}

Keyword args:
    {dtype}  # 数据类型，默认为 `None`
    {layout}  # 张量布局，默认为 `None`
    {device}  # 设备选项，默认为 `None`
    {requires_grad}  # 是否需要梯度，默认为 `False`
    {memory_format}  # 存储格式，默认为 `torch.preserve_format`

Example::

    >>> a=torch.empty((2,3), dtype=torch.int32, device = 'cuda')
    >>> torch.empty_like(a)
    tensor([[0, 0, 0],
            [0, 0, 0]], device='cuda:0', dtype=torch.int32)
""".format(
        **factory_like_common_args
    ),
)

add_docstr(
    torch.empty_strided,
    r"""
# 创建一个具有指定 `size` 和 `stride` 的张量，并填充为未定义数据。

# 警告：
# 如果构造的张量“重叠”（多个索引引用同一内存元素），其行为是未定义的。

# 注意：
# 如果 `torch.use_deterministic_algorithms()` 和 `torch.utils.deterministic.fill_uninitialized_memory` 均设置为 `True`，
# 则初始化输出张量以避免将数据作为操作的输入时可能引起的任何非确定性行为。

def empty_strided(size, stride, *, dtype=None, layout=None, device=None, requires_grad=False, pin_memory=False) -> Tensor:
    # 返回一个未定义数据的张量，其大小和步长由 `size` 和 `stride` 参数指定

Args:
    size (int...): 定义输出张量的形状的整数序列。
        可以是可变数量的参数或类似列表或元组的集合。
    stride (int...): 定义输出张量的步长的整数序列。
        必须与 `size` 的长度相同。

Keyword args:
    {dtype}  # 数据类型，默认为 `None`
    {layout}  # 张量布局，默认为 `None`
    {device}  # 设备选项，默认为 `None`
    {requires_grad}  # 是否需要梯度，默认为 `False`
    pin_memory=False  # 是否用于固定内存，默认为 `False`

""".format(
        **factory_strided_common_args
    ),
)
    Floating point and complex tensors are filled with NaN, and integer tensors
    are filled with the maximum value.
"""
Creates a tensor of a specified size filled with a specified value.

Args:
    size (tuple of int): Shape of the output tensor.
    fill_value (scalar): Value to fill the tensor with.

Keyword args:
    out (Tensor, optional): Output tensor. If provided, fills this tensor rather than creating a new one.
    dtype (:class:`torch.dtype`, optional): Data type of the output tensor.
    layout (:class:`torch.layout`, optional): Layout of the output tensor.
    device (:class:`torch.device`, optional): Device location of the output tensor.
    requires_grad (bool, optional): If autograd should record operations on the output tensor.

Returns:
    Tensor: Tensor filled with the specified `fill_value`.

Example::

    >>> torch.full((2, 3), 5.0)
    tensor([[5.0, 5.0, 5.0],
            [5.0, 5.0, 5.0]])

    >>> torch.full((2, 3), 5.0, dtype=torch.float64)
    tensor([[5.0, 5.0, 5.0],
            [5.0, 5.0, 5.0]], dtype=torch.float64)

    >>> torch.full((2, 3), 5)
    tensor([[5, 5, 5],
            [5, 5, 5]])

    >>> torch.full((2, 3), 5, dtype=torch.int)
    tensor([[5, 5, 5],
            [5, 5, 5]], dtype=torch.int)
""".format(
        **factory_common_args
    ),
)
add_docstr(
    torch.full,
    """
full(size, fill_value, *, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Returns a tensor of size :attr:`size` filled with :attr:`fill_value`.

Args:
    size (int...): a list, tuple, or :class:`torch.Size` of integers defining the
        shape of the output tensor.
    fill_value (Scalar): the value to fill the output tensor with.

Keyword args:
    {dtype}
    {layout}
    {device}
    {requires_grad}

Example::

    >>> torch.full((2, 3), 3.141592)
    tensor([[ 3.1416,  3.1416,  3.1416],
            [ 3.1416,  3.1416,  3.1416]])
""".format(
        **factory_common_args
    ),
)

add_docstr(
    torch.full_like,
    """
full_like(input, fill_value, *, dtype=None, layout=torch.strided, device=None, requires_grad=False, \
memory_format=torch.preserve_format) -> Tensor

Returns a tensor with the same size as :attr:`input` filled with :attr:`fill_value`.
``torch.full_like(input, fill_value)`` is equivalent to
``torch.full(input.size(), fill_value, dtype=input.dtype, layout=input.layout, device=input.device)``.

Args:
    {input}
    fill_value: the number to fill the output tensor with.

Keyword args:
    {dtype}
    {layout}
    {device}
    {requires_grad}
    {memory_format}
""".format(
        **factory_like_common_args
    ),
)

add_docstr(
    torch.det,
    r"""
det(input) -> Tensor

Alias for :func:`torch.linalg.det`
""",
)

add_docstr(
    torch.where,
    r"""
where(condition, input, other, *, out=None) -> Tensor

Return a tensor of elements selected from either :attr:`input` or :attr:`other`, depending on :attr:`condition`.

The operation is defined as:

.. math::
    \text{out}_i = \begin{cases}
        \text{input}_i & \text{if } \text{condition}_i \\
        \text{other}_i & \text{otherwise} \\
    \end{cases}

.. note::
    The tensors :attr:`condition`, :attr:`input`, :attr:`other` must be :ref:`broadcastable <broadcasting-semantics>`.

Arguments:
    condition (BoolTensor): When True (nonzero), yield input, otherwise yield other
    input (Tensor or Scalar): value (if :attr:`input` is a scalar) or values selected at indices
                          where :attr:`condition` is ``True``
    other (Tensor or Scalar): value (if :attr:`other` is a scalar) or values selected at indices
                          where :attr:`condition` is ``False``

Keyword args:
    {out}

Returns:
    Tensor: A tensor of shape equal to the broadcasted shape of :attr:`condition`, :attr:`input`, :attr:`other`

Example::

    >>> x = torch.randn(3, 2)
    >>> y = torch.ones(3, 2)
    >>> x
    tensor([[-0.4620,  0.3139],
            [ 0.3898, -0.7197],
            [ 0.0478, -0.1657]])
    >>> torch.where(x > 0, 1.0, 0.0)
    tensor([[0., 1.],
            [1., 0.],
            [1., 0.]])
    >>> torch.where(x > 0, x, y)
    tensor([[ 1.0000,  0.3139],
            [ 0.3898,  1.0000],
            [ 0.0478,  1.0000]])
    >>> x = torch.randn(2, 2, dtype=torch.double)
    >>> x
    tensor([[ 1.0779,  0.0383],
            [-0.8785, -1.1089]], dtype=torch.float64)
    >>> torch.where(x > 0, x, 0.)
""",
)
    # 创建一个二维张量（Tensor），包含两行两列的数据
    tensor([[1.0779, 0.0383],
            [0.0000, 0.0000]], dtype=torch.float64)
# 添加文档字符串到 torch.where 函数，描述其功能和用法
add_docstr(
    torch.where(condition) -> tuple of LongTensor
    :noindex:



# 说明 torch.where(condition) 和 torch.nonzero(condition, as_tuple=True) 之间的等价性



# 描述 torch.logdet(input) 函数的功能，计算方形矩阵或批量方形矩阵的对数行列式
add_docstr(
    torch.logdet,
    r"""
logdet(input) -> Tensor

Calculates log determinant of a square matrix or batches of square matrices.

It returns ``-inf`` if the input has a determinant of zero, and ``NaN`` if it has
a negative determinant.

.. note::
    Backward through :meth:`logdet` internally uses SVD results when :attr:`input`
    is not invertible. In this case, double backward through :meth:`logdet` will
    be unstable in when :attr:`input` doesn't have distinct singular values. See
    :func:`torch.linalg.svd` for details.

.. seealso::

        :func:`torch.linalg.slogdet` computes the sign (resp. angle) and natural logarithm of the
        absolute value of the determinant of real-valued (resp. complex) square matrices.

Arguments:
    input (Tensor): the input tensor of size ``(*, n, n)`` where ``*`` is zero or more
                batch dimensions.

Example::

    >>> A = torch.randn(3, 3)
    >>> torch.det(A)
    tensor(0.2611)
    >>> torch.logdet(A)
    tensor(-1.3430)
    >>> A
    tensor([[[ 0.9254, -0.6213],
             [-0.5787,  1.6843]],

            [[ 0.3242, -0.9665],
             [ 0.4539, -0.0887]],

            [[ 1.1336, -0.4025],
             [-0.7089,  0.9032]]])
    >>> A.det()
    tensor([1.1990, 0.4099, 0.7386])
    >>> A.det().log()
    tensor([ 0.1815, -0.8917, -0.3031])
""",
)



# 描述 torch.slogdet(input) 函数的功能，为 torch.linalg.slogdet 的别名
add_docstr(
    torch.slogdet,
    r"""
slogdet(input) -> (Tensor, Tensor)

Alias for :func:`torch.linalg.slogdet`
""",
)



# 描述 torch.pinverse(input, rcond=1e-15) 函数的功能，为 torch.linalg.pinv 的别名
add_docstr(
    torch.pinverse,
    r"""
pinverse(input, rcond=1e-15) -> Tensor

Alias for :func:`torch.linalg.pinv`
""",
)



# 描述 torch.hann_window 函数的功能，生成汉宁窗口函数
add_docstr(
    torch.hann_window,
    """
hann_window(window_length, periodic=True, *, dtype=None, \
layout=torch.strided, device=None, requires_grad=False) -> Tensor
"""
    + r"""
Hann window function.

.. math::
    w[n] = \frac{1}{2}\ \left[1 - \cos \left( \frac{2 \pi n}{N - 1} \right)\right] =
            \sin^2 \left( \frac{\pi n}{N - 1} \right),

where :math:`N` is the full window size.

The input :attr:`window_length` is a positive integer controlling the
returned window size. :attr:`periodic` flag determines whether the returned
window trims off the last duplicate value from the symmetric window and is
ready to be used as a periodic window with functions like
:meth:`torch.stft`. Therefore, if :attr:`periodic` is true, the :math:`N` in
above formula is in fact :math:`\text{window\_length} + 1`. Also, we always have
``torch.hann_window(L, periodic=True)`` equal to
``torch.hann_window(L + 1, periodic=False)[:-1])``.

.. note::
    If :attr:`window_length` :math:`=1`, the returned window contains a single value 1.
"""
    + r"""
Arguments:
    window_length (int): the size of returned window
    periodic (bool, optional): If True, returns a window to be used as periodic
        function. If False, return a symmetric window.
# 添加函数文档字符串和注释
add_docstr(
    torch.hamming_window,
    """
hamming_window(window_length, periodic=True, alpha=0.54, beta=0.46, *, dtype=None, \
layout=torch.strided, device=None, requires_grad=False) -> Tensor
"""
    + r"""
Hamming window function.

.. math::
    w[n] = \alpha - \beta\ \cos \left( \frac{2 \pi n}{N - 1} \right),

where :math:`N` is the full window size.

The input :attr:`window_length` is a positive integer controlling the
returned window size. :attr:`periodic` flag determines whether the returned
window trims off the last duplicate value from the symmetric window and is
ready to be used as a periodic window with functions like
:meth:`torch.stft`. Therefore, if :attr:`periodic` is true, the :math:`N` in
above formula is in fact :math:`\text{window\_length} + 1`. Also, we always have
``torch.hamming_window(L, periodic=True)`` equal to
``torch.hamming_window(L + 1, periodic=False)[:-1])``.

.. note::
    If :attr:`window_length` :math:`=1`, the returned window contains a single value 1.

.. note::
    This is a generalized version of :meth:`torch.hann_window`.
"""
    + r"""
Arguments:
    window_length (int): the size of returned window
    periodic (bool, optional): If True, returns a window to be used as periodic
        function. If False, return a symmetric window.
    alpha (float, optional): The coefficient :math:`\alpha` in the equation above
    beta (float, optional): The coefficient :math:`\beta` in the equation above

Keyword args:
    {dtype} Only floating point types are supported.
    layout (:class:`torch.layout`, optional): the desired layout of returned window tensor. Only
          ``torch.strided`` (dense layout) is supported.
    {device}
    {requires_grad}

Returns:
    Tensor: A 1-D tensor of size :math:`(\text{{window\_length}},)` containing the window.

""".format(
        **factory_common_args
    ),
)


# 添加函数文档字符串和注释
add_docstr(
    torch.bartlett_window,
    """
bartlett_window(window_length, periodic=True, *, dtype=None, \
layout=torch.strided, device=None, requires_grad=False) -> Tensor
"""
    + r"""
Bartlett window function.

.. math::
    w[n] = 1 - \left| \frac{2n}{N-1} - 1 \right| = \begin{cases}
        \frac{2n}{N - 1} & \text{if } 0 \leq n \leq \frac{N - 1}{2} \\
        2 - \frac{2n}{N - 1} & \text{if } \frac{N - 1}{2} < n < N \\
    \end{cases},

where :math:`N` is the full window size.

The input :attr:`window_length` is a positive integer controlling the
returned window size. :attr:`periodic` flag determines whether the returned
window trims off the last duplicate value from the symmetric window and is
"""
# 定义了一个函数 add_docstr，用于为给定的函数对象添加文档字符串
def add_docstr(
    # 第一个参数是一个函数对象 torch.blackman_window，表示要为此函数添加文档字符串
    torch.blackman_window,
    """
    blackman_window(window_length, periodic=True, *, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
    """
    # 文档字符串的一部分，介绍了 Blackman 窗口函数的数学定义和用法
    + r"""
    Blackman window function.

    .. math::
        w[n] = 0.42 - 0.5 \cos \left( \frac{2 \pi n}{N - 1} \right) + 0.08 \cos \left( \frac{4 \pi n}{N - 1} \right)

    where :math:`N` is the full window size.

    # 描述了窗口长度和周期性参数的作用
    The input :attr:`window_length` is a positive integer controlling the
    returned window size. :attr:`periodic` flag determines whether the returned
    window trims off the last duplicate value from the symmetric window and is
    ready to be used as a periodic window with functions like
    :meth:`torch.stft`. Therefore, if :attr:`periodic` is true, the :math:`N` in
    above formula is in fact :math:`\text{window\_length} + 1`. Also, we always have
    ``torch.blackman_window(L, periodic=True)`` equal to
    ``torch.blackman_window(L + 1, periodic=False)[:-1])``.

    .. note::
        If :attr:`window_length` :math:`=1`, the returned window contains a single value 1.
    """
    # 描述了函数参数和返回值的信息，使用了 format 方法插入参数的详细描述
    + r"""
    Arguments:
        window_length (int): the size of returned window
        periodic (bool, optional): If True, returns a window to be used as periodic
            function. If False, return a symmetric window.

    Keyword args:
        {dtype} Only floating point types are supported.
        layout (:class:`torch.layout`, optional): the desired layout of returned window tensor. Only
              ``torch.strided`` (dense layout) is supported.
        {device}
        {requires_grad}

    Returns:
        Tensor: A 1-D tensor of size :math:`(\text{{window\_length}},)` containing the window

    """.format(
        **factory_common_args  # 插入参数的公共描述信息
    ),
)
Let I_0 be the zeroth order modified Bessel function of the first kind (see :func:`torch.i0`) and
N = L - 1 if :attr:`periodic` is False and L if :attr:`periodic` is True,
where L is the :attr:`window_length`. This function computes:

.. math::
    out_i = I_0 \left( \beta \sqrt{1 - \left( {\frac{i - N/2}{N/2}} \right) ^2 } \right) / I_0( \beta )

Calling ``torch.kaiser_window(L, B, periodic=True)`` is equivalent to calling
``torch.kaiser_window(L + 1, B, periodic=False)[:-1])``.
The :attr:`periodic` argument is intended as a helpful shorthand
to produce a periodic window as input to functions like :func:`torch.stft`.

.. note::
    If :attr:`window_length` is one, then the returned window is a single element tensor containing a one.

"""
    + r"""
Args:
    window_length (int): length of the window.
    periodic (bool, optional): If True, returns a periodic window suitable for use in spectral analysis.
        If False, returns a symmetric window suitable for use in filter design.
    beta (float, optional): shape parameter for the window.

Keyword args:
    {dtype}
    layout (:class:`torch.layout`, optional): the desired layout of returned window tensor. Only
          ``torch.strided`` (dense layout) is supported.
    {device}
    {requires_grad}

""".format(
        **factory_common_args
    ),

add_docstr(
    torch.vander,
    """
vander(x, N=None, increasing=False) -> Tensor
"""
    + r"""
Generates a Vandermonde matrix.

The columns of the output matrix are elementwise powers of the input vector :math:`x^{{(N-1)}}, x^{{(N-2)}}, ..., x^0`.
If increasing is True, the order of the columns is reversed :math:`x^0, x^1, ..., x^{{(N-1)}}`. Such a
matrix with a geometric progression in each row is named for Alexandre-Theophile Vandermonde.

Arguments:
    x (Tensor): 1-D input tensor.
    N (int, optional): Number of columns in the output. If N is not specified,
        a square array is returned :math:`(N = len(x))`.
    increasing (bool, optional): Order of the powers of the columns. If True,
        the powers increase from left to right, if False (the default) they are reversed.

Returns:
    Tensor: Vandermonde matrix. If increasing is False, the first column is :math:`x^{{(N-1)}}`,
    the second :math:`x^{{(N-2)}}` and so forth. If increasing is True, the columns
    are :math:`x^0, x^1, ..., x^{{(N-1)}}`.

Example::

    >>> x = torch.tensor([1, 2, 3, 5])
    >>> torch.vander(x)
    tensor([[  1,   1,   1,   1],
            [  8,   4,   2,   1],
            [ 27,   9,   3,   1],
            [125,  25,   5,   1]])
    >>> torch.vander(x, N=3)
    tensor([[ 1,  1,  1],
            [ 4,  2,  1],
            [ 9,  3,  1],
            [25,  5,  1]])
    >>> torch.vander(x, N=3, increasing=True)
    tensor([[ 1,  1,  1],
            [ 1,  2,  4],
            [ 1,  3,  9],
            [ 1,  5, 25]])

""".format(
        **factory_common_args
    ),

add_docstr(
    torch.unbind,
    r"""
unbind(input, dim=0) -> seq

Removes a tensor dimension along a given axis.

Args:
    input (Tensor): the input tensor.
    dim (int, optional): the dimension to remove (default is 0).

Returns:
    seq: tuple of tensors resulting from removing the specified dimension.

"""
# 计算沿指定维度的梯形积分值。
# 默认情况下，假设元素之间的间距为1，使用梯形规则计算沿维度的积分。
# 但是可以使用 dx 参数指定不同的常量间距，使用 x 参数可以指定维度上的任意间距。

torch.trapezoid(
    y,      # 输入张量，假设为一维张量，表示积分的函数值
    x=None, # 用于指定维度上的任意间距的张量，默认为 None
    *,     # 以下是关键字参数，后面必须使用关键字调用
    dx=None,    # 用于指定不同常量间距的标量，影响积分结果
    dim=-1      # 沿其计算梯形积分的维度，默认为最后一个维度
) -> Tensor:   # 返回一个张量，表示沿指定维度的梯形积分结果
# 当张量 :attr:`x` 和 :attr:`y` 的尺寸相同时，按照上述描述进行计算，无需进行广播。
# 当 :attr:`x` 和 :attr:`y` 的尺寸不同时，该函数的广播行为如下。对于 :attr:`x` 和 :attr:`y`，
# 函数计算沿着维度 :attr:`dim` 的相邻元素之间的差异。这实际上创建了两个张量 `x_diff` 和 `y_diff`，
# 它们的形状与原始张量相同，除了它们沿着维度 :attr:`dim` 的长度减少了1。
# 然后，这两个张量进行广播以计算最终输出，作为梯形法则的一部分。
# 详细示例请参见下面的例子。

.. note::
    梯形法则是一种通过平均左和右黎曼和来近似计算函数的定积分的技术。
    随着分区分辨率的增加，该近似变得更加精确。

Arguments:
    y (Tensor): 计算梯形法则时使用的值。
    x (Tensor): 如果指定，则定义了值之间的间距，如上所述。

Keyword arguments:
    dx (float): 值之间的常量间距。如果未指定 :attr:`x` 或 :attr:`dx`，则默认为1。实际上将结果乘以其值。
    dim (int): 沿其计算梯形法则的维度。默认为最后（最内部）的维度。

Examples::

    >>> # 计算1维中的梯形法则，间距隐含为1
    >>> y = torch.tensor([1, 5, 10])
    >>> torch.trapezoid(y)
    tensor(10.5)

    >>> # 直接验证同样的梯形法则计算
    >>> (1 + 10 + 10) / 2
    10.5

    >>> # 计算1维中间距为2的梯形法则
    >>> # 注意：结果与之前相同，但乘以2
    >>> torch.trapezoid(y, dx=2)
    21.0

    >>> # 计算具有任意间距的1维梯形法则
    >>> x = torch.tensor([1, 3, 6])
    >>> torch.trapezoid(y, x)
    28.5

    >>> # 直接验证同样的梯形法则计算
    >>> ((3 - 1) * (1 + 5) + (6 - 3) * (5 + 10)) / 2
    28.5

    >>> # 计算3x3矩阵每行的梯形法则
    >>> y = torch.arange(9).reshape(3, 3)
    tensor([[0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]])
    >>> torch.trapezoid(y)
    tensor([ 2., 8., 14.])

    >>> # 计算矩阵每列的梯形法则
    >>> torch.trapezoid(y, dim=0)
    tensor([ 6., 8., 10.])

    >>> # 计算3x3全1矩阵每行的梯形法则，使用相同的任意间距
    >>> y = torch.ones(3, 3)
    >>> x = torch.tensor([1, 3, 6])
    >>> torch.trapezoid(y, x)
    array([5., 5., 5.])

    >>> # 计算3x3全1矩阵每行的梯形法则，每行使用不同的任意间距
    >>> y = torch.ones(3, 3)
    >>> x = torch.tensor([[1, 2, 3], [1, 3, 5], [1, 4, 7]])
    >>> torch.trapezoid(y, x)
    array([2., 4., 6.])
# 添加文档字符串给 torch.trapz 函数
add_docstr(
    torch.trapz,
    r"""
trapz(y, x, *, dim=-1) -> Tensor

Alias for :func:`torch.trapezoid`.
""",
)

# 添加文档字符串给 torch.cumulative_trapezoid 函数
add_docstr(
    torch.cumulative_trapezoid,
    r"""
cumulative_trapezoid(y, x=None, *, dx=None, dim=-1) -> Tensor

Cumulatively computes the `trapezoidal rule <https://en.wikipedia.org/wiki/Trapezoidal_rule>`_
along :attr:`dim`. By default the spacing between elements is assumed to be 1, but
:attr:`dx` can be used to specify a different constant spacing, and :attr:`x` can be
used to specify arbitrary spacing along :attr:`dim`.

For more details, please read :func:`torch.trapezoid`. The difference between :func:`torch.trapezoid`
and this function is that, :func:`torch.trapezoid` returns a value for each integration,
whereas this function returns a cumulative value for every spacing within the integration. This
is analogous to how `.sum` returns a value and `.cumsum` returns a cumulative sum.

Arguments:
    y (Tensor): Values to use when computing the trapezoidal rule.
    x (Tensor): If specified, defines spacing between values as specified above.

Keyword arguments:
    dx (float): constant spacing between values. If neither :attr:`x` or :attr:`dx`
        are specified then this defaults to 1. Effectively multiplies the result by its value.
    dim (int): The dimension along which to compute the trapezoidal rule.
        The last (inner-most) dimension by default.

Examples::

    >>> # Cumulatively computes the trapezoidal rule in 1D, spacing is implicitly 1.
    >>> y = torch.tensor([1, 5, 10])
    >>> torch.cumulative_trapezoid(y)
    tensor([3., 10.5])

    >>> # Computes the same trapezoidal rule directly up to each element to verify
    >>> (1 + 5) / 2
    3.0
    >>> (1 + 10 + 10) / 2
    10.5

    >>> # Cumulatively computes the trapezoidal rule in 1D with constant spacing of 2
    >>> # NOTE: the result is the same as before, but multiplied by 2
    >>> torch.cumulative_trapezoid(y, dx=2)
    tensor([6., 21.])

    >>> # Cumulatively computes the trapezoidal rule in 1D with arbitrary spacing
    >>> x = torch.tensor([1, 3, 6])
    >>> torch.cumulative_trapezoid(y, x)
    tensor([6., 28.5])

    >>> # Computes the same trapezoidal rule directly up to each element to verify
    >>> ((3 - 1) * (1 + 5)) / 2
    6.0
    >>> ((3 - 1) * (1 + 5) + (6 - 3) * (5 + 10)) / 2
    28.5

    >>> # Cumulatively computes the trapezoidal rule for each row of a 3x3 matrix
    >>> y = torch.arange(9).reshape(3, 3)
    tensor([[0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]])
    >>> torch.cumulative_trapezoid(y)
    tensor([[ 0.5,  2.],
            [ 3.5,  8.],
            [ 6.5, 14.]])

    >>> # Cumulatively computes the trapezoidal rule for each column of the matrix
    >>> torch.cumulative_trapezoid(y, dim=0)
    tensor([[ 1.5,  2.5,  3.5],
            [ 6.0,  8.0, 10.0]])

    >>> # Cumulatively computes the trapezoidal rule for each row of a 3x3 ones matrix
    >>> #   with the same arbitrary spacing
""",
)
    >>> y = torch.ones(3, 3)
    >>> x = torch.tensor([1, 3, 6])
    >>> torch.cumulative_trapezoid(y, x)
    tensor([[2., 5.],
            [2., 5.],
            [2., 5.]])
    
    >>> # 创建一个 3x3 的全为1的张量 y
    >>> # 创建一个张量 x，包含元素 [1, 3, 6]
    >>> # 对 y 的每一行应用累积梯形法则，使用 x 的元素作为各行的不同间距
    >>> # 返回结果为一个张量，包含了每行的累积梯形法则计算结果
    
    >>> y = torch.ones(3, 3)
    >>> x = torch.tensor([[1, 2, 3], [1, 3, 5], [1, 4, 7]])
    >>> torch.cumulative_trapezoid(y, x)
    tensor([[1., 2.],
            [2., 4.],
            [3., 6.]])
    
    >>> # 创建一个 3x3 的全为1的张量 y
    >>> # 创建一个二维张量 x，包含不同行的各自间距 [1, 2, 3], [1, 3, 5], [1, 4, 7]
    >>> # 对 y 的每一行应用累积梯形法则，使用 x 的每行作为各行的不同间距
    >>> # 返回结果为一个张量，包含了每行的累积梯形法则计算结果
"""
tile(input, dims) -> Tensor

Constructs a tensor by repeating the elements of :attr:`input`.
The :attr:`dims` argument specifies the number of repetitions
in each dimension.

If :attr:`dims` specifies fewer dimensions than :attr:`input` has, then
ones are prepended to :attr:`dims` until all dimensions are specified.
For example, if :attr:`input` has shape (8, 6, 4, 2) and :attr:`dims`
is (2, 2), then :attr:`dims` is treated as (1, 1, 2, 2).

Analogously, if :attr:`input` has fewer dimensions than :attr:`dims`
specifies, then :attr:`input` is treated as if it were unsqueezed at
dimension zero until it has as many dimensions as :attr:`dims` specifies.
For example, if :attr:`input` has shape (4, 2) and :attr:`dims`
is (3, 3, 2, 2), then :attr:`input` is treated as if it had the
shape (1, 1, 4, 2).

.. note::

    The behavior of this function is analogous to numpy's `tile`.

Args:
    input (Tensor): The tensor to repeat.
    dims (tuple of ints): The desired shape of the output tensor.

Returns:
    Tensor: A new tensor with repeated elements of :attr:`input` according to :attr:`dims`.

Example::

    >>> x = torch.tensor([[1, 2], [3, 4]])
    >>> torch.tile(x, (2, 3))
    tensor([[1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4],
            [1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4]])

    >>> y = torch.tensor([1, 2, 3])
    >>> torch.tile(y, (2,))
    tensor([1, 2, 3, 1, 2, 3])
"""
    # 这个函数类似于 NumPy 的 tile 函数，用于将给定数组沿指定方向复制多次形成新的数组。
    
    def tile(arr, reps):
        # 获取数组的维度数
        ndim = arr.ndim
        # 如果维度数小于要复制的次数，则抛出异常
        if ndim < reps:
            raise ValueError("Need more dimensions to tile")
        # 初始化结果数组的形状为原始数组的形状乘以复制次数
        shape_out = tuple(np.array(arr.shape) * np.array(reps))
        # 创建一个全零的结果数组，形状为 shape_out
        result = np.zeros(shape_out)
        # 在指定维度上将原始数组复制 reps[i] 次
        for i in range(ndim):
            result = np.repeat(result, reps[i], axis=i)
        # 返回复制后的结果数组
        return result
# 将输入的浮点张量转换为具有给定比例和零点的量化张量
def quantize_per_tensor(input, scale, zero_point, dtype):
    # 输入可以是单个浮点张量或张量列表，进行量化
    # scale参数表示量化公式中的缩放比例
    # zero_point参数表示映射到浮点零值的整数偏移量
    # dtype参数指定返回张量的数据类型，必须是量化数据类型之一
    return torch.quantize_per_tensor(input, scale, zero_point, dtype)

# 将输入的浮点张量动态转换为量化张量，动态计算比例和零点基于输入值
def quantize_per_tensor_dynamic(input, dtype, reduce_range):
    # 输入可以是单个浮点张量或张量列表，进行动态量化
    # dtype参数指定返回张量的数据类型，必须是量化数据类型之一
    # reduce_range参数用于指示是否通过减少1位量化数据范围来避免某些硬件的指令溢出
    return torch.quantize_per_tensor_dynamic(input, dtype, reduce_range)
    # 使用动态量化将给定的浮点张量进行量化为torch.quint8类型，禁用对称量化
    t = torch.quantize_per_tensor_dynamic(torch.tensor([-1.0, 0.0, 1.0, 2.0]), torch.quint8, False)
    
    # 打印张量t的内容
    print(t)
    # 输出:
    # tensor([-1.,  0.,  1.,  2.], size=(4,), dtype=torch.quint8,
    #        quantization_scheme=torch.per_tensor_affine, scale=0.011764705882352941,
    #        zero_point=85)
    
    # 将张量t转换为其整数表示形式
    t.int_repr()
    # 输出:
    # tensor([  0,  85, 170, 255], dtype=torch.uint8)
# 添加文档字符串给 torch.quantize_per_channel 函数
add_docstr(
    torch.quantize_per_channel,
    r"""
    将输入的浮点张量转换为具有给定缩放因子和零点的按通道量化张量。

    参数:
        input (Tensor): 要量化的浮点张量
        scales (Tensor): 用于量化的浮点1D张量，大小应与 ``input.size(axis)`` 匹配
        zero_points (Tensor): 用于量化的整数1D张量偏移量，大小应与 ``input.size(axis)`` 匹配
        axis (int): 应用按通道量化的维度
        dtype (:class:`torch.dtype`): 返回张量的期望数据类型。
            必须是量化数据类型之一: ``torch.quint8``, ``torch.qint8``, ``torch.qint32``

    返回:
        Tensor: 新量化的张量

    示例::

        >>> x = torch.tensor([[-1.0, 0.0], [1.0, 2.0]])
        >>> torch.quantize_per_channel(x, torch.tensor([0.1, 0.01]), torch.tensor([10, 0]), 0, torch.quint8)
        tensor([[-1.,  0.],
                [ 1.,  2.]], size=(2, 2), dtype=torch.quint8,
               quantization_scheme=torch.per_channel_affine,
               scale=tensor([0.1000, 0.0100], dtype=torch.float64),
               zero_point=tensor([10,  0]), axis=0)
        >>> torch.quantize_per_channel(x, torch.tensor([0.1, 0.01]), torch.tensor([10, 0]), 0, torch.quint8).int_repr()
        tensor([[  0,  10],
                [100, 200]], dtype=torch.uint8)
    """,
)


# 添加文档字符串给 torch.quantized_batch_norm 函数
add_docstr(
    torch.quantized_batch_norm,
    r"""
    在4D（NCHW）量化张量上应用批量归一化。

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    参数:
        input (Tensor): 量化张量
        weight (Tensor): 对应于 gamma 的浮点张量，大小为 C
        bias (Tensor): 对应于 beta 的浮点张量，大小为 C
        mean (Tensor): 批量归一化的浮点均值，大小为 C
        var (Tensor): 方差的浮点张量，大小为 C
        eps (float): 分母上添加的值，用于数值稳定性
        output_scale (float): 输出量化张量的缩放因子
        output_zero_point (int): 输出量化张量的零点

    返回:
        Tensor: 应用批量归一化后的量化张量

    示例::

        >>> qx = torch.quantize_per_tensor(torch.rand(2, 2, 2, 2), 1.5, 3, torch.quint8)
        >>> torch.quantized_batch_norm(qx, torch.ones(2), torch.zeros(2), torch.rand(2), torch.rand(2), 0.00001, 0.2, 2)
        tensor([[[[-0.2000, -0.2000],
                  [ 1.6000, -0.2000]],
        
                 [[-0.4000, -0.4000],
                  [-0.4000,  0.6000]]],
        
        
                [[[-0.2000, -0.2000],
                  [-0.2000, -0.2000]],
        
                 [[ 0.6000, -0.4000],
                  [ 0.6000, -0.4000]]]], size=(2, 2, 2, 2), dtype=torch.quint8,
               quantization_scheme=torch.per_tensor_affine, scale=0.2, zero_point=2)
    """,
)


# 添加文档字符串给 torch.quantized_max_pool1d 函数
add_docstr(
    torch.quantized_max_pool1d,
    r"""
# 设置生成器的状态，使其与给定状态匹配
torch.Generator.set_state,
add_docstr(
    torch.Generator.set_state,
    r"""
    Generator.set_state(new_state) -> None

    Sets the Generator state.

    Arguments:
        new_state (torch.ByteTensor): The desired state.

    Example::

        >>> g_cpu = torch.Generator()
        >>> g_cpu_other = torch.Generator()
        >>> g_cpu.set_state(g_cpu_other.get_state())
    """,
)
    # 访问 torch 库中 Generator 类的 initial_seed 静态属性，返回生成器的初始种子值
    torch.Generator.initial_seed,
    # 在字符串前缀 r"""" 中，r 表示原始字符串，不对反斜杠进行转义处理
# 返回生成随机数的初始种子值
Generator.initial_seed() -> int

Example::

    >>> g_cpu = torch.Generator()
    >>> g_cpu.initial_seed()
    2147483647



# 使用非确定性随机数（从std::random_device或当前时间获得）来为Generator对象设置种子
Generator.seed() -> int

Example::

    >>> g_cpu = torch.Generator()
    >>> g_cpu.seed()
    1516516984916



# 返回当前生成器的设备信息
Generator.device -> device

Example::

    >>> g_cpu = torch.Generator()
    >>> g_cpu.device
    device(type='cpu')



# 异步检查张量内容是否非零。对于CPU张量，相当于“assert tensor”或“assert tensor.is_nonzero()”；
# 对于CUDA张量，不进行同步，可能只在稍后的CUDA内核启动时才发现断言失败。
# 异步断言对于在CUDA张量中测试不变量而不损失性能很有帮助。不适用于常规错误检查，因为如果断言失败将破坏CUDA上下文（强制重启PyTorch进程）。
# Args:
#     tensor (Tensor): 用于测试是否非零的单元素张量。零元素（包括布尔张量中的False）将引发断言失败。
torch._assert_async(tensor) -> void



# 在sorted_sequence的最内层维度中找到索引，使得如果将values中对应的值插入这些索引之前，
# 则排序后，sorted_sequence中对应的最内层维度的顺序将保持不变。返回与values大小相同的新张量。
# 更正式地说，返回的索引满足以下规则：
# - 如果right为False，1-D情况下：sorted_sequence[i-1] < values[m][n]...[l][x] <= sorted_sequence[i]
# - 如果right为True，1-D情况下：sorted_sequence[i-1] <= values[m][n]...[l][x] < sorted_sequence[i]
# - 如果right为False，N-D情况下：sorted_sequence[m][n]...[l][i-1] < values[m][n]...[l][x] <= sorted_sequence[m][n]...[l][i]
# - 如果right为True，N-D情况下：sorted_sequence[m][n]...[l][i-1] <= values[m][n]...[l][x] < sorted_sequence[m][n]...[l][i]
# Args:
#     sorted_sequence (Tensor): N-D或1-D张量，包含*最内层*维度上单调递增的序列，除非提供了sorter，否则序列不需要排序
torch.searchsorted(
    sorted_sequence,
    values,
    *,
    out_int32=False,
    right=False,
    side=None,
    out=None,
    sorter=None
) -> Tensor
    values (Tensor or Scalar): N-D tensor or a Scalar containing the search value(s).
def bucketize(input, boundaries, *, out_int32=False, right=False, out=None) -> Tensor:
    """
    返回每个输入值所属的桶的索引，桶的边界由 `boundaries` 定义。返回与 `input` 相同大小的新张量。
    如果 `right` 为 False（默认），则左边界是开放的。

    Parameters:
        input (Tensor): 输入张量，包含要分桶的值。
        boundaries (Tensor): 桶的边界值，用于定义每个桶的范围。
        out_int32 (bool, optional): 输出数据类型指示。如果为 True，则输出数据类型为 torch.int32；否则为 torch.int64。
                                    默认值为 False，即默认输出数据类型为 torch.int64。
        right (bool, optional): 如果为 False，则返回找到的第一个合适位置。如果为 True，则返回最后一个这样的索引。
                                如果未找到合适的索引，则返回0（对于非数值值，例如 nan、inf）或者
                                :attr:`sorted_sequence` 的 *innermost* 维度的大小（超过 *innermost* 维度的最后一个索引）。
                                换句话说，如果为 False，则获取 :attr:`values` 中每个值在 :attr:`sorted_sequence` 的
                                对应 *innermost* 维度上的下界索引。如果为 True，则获取上界索引。默认值为 False。
                                :attr:`side` 也执行相同的功能，且更为优先。如果设置了 :attr:`side` 为 "left"，而此时为 True，
                                则会报错。
        out (Tensor, optional): 输出张量，如果提供，必须与 :attr:`values` 的大小相同。

    Returns:
        Tensor: 每个值所属的桶的索引。

    Example::

        >>> sorted_sequence = torch.tensor([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]])
        >>> values = torch.tensor([[3, 6, 9], [3, 6, 9]])
        >>> torch.bucketize(sorted_sequence, values)
        tensor([[1, 3, 4],
                [1, 2, 4]])
        >>> torch.bucketize(sorted_sequence, values, side='right')
        tensor([[2, 3, 5],
                [1, 3, 4]])

        >>> sorted_sequence_1d = torch.tensor([1, 3, 5, 7, 9])
        >>> torch.bucketize(sorted_sequence_1d, values)
        tensor([[1, 3, 4],
                [1, 3, 4]])
    """
    pass
# 为了避免覆盖输入张量，执行与 torch.permute 相同的操作，但生成一个新的输出张量。
add_docstr(
    torch.permute_copy,
    r"""
Performs the same operation as :func:`torch.permute`, but all output tensors
are freshly created instead of aliasing the input.
""",
)
add_docstr(
    torch.select_copy,
    r"""
Performs the same operation as :func:`torch.select`, but all output tensors
are freshly created instead of aliasing the input.
""",
)

add_docstr(
    torch.detach_copy,
    r"""
Performs the same operation as :func:`torch.detach`, but all output tensors
are freshly created instead of aliasing the input.
""",
)

add_docstr(
    torch.slice_copy,
    r"""
Performs the same operation as :func:`torch.slice`, but all output tensors
are freshly created instead of aliasing the input.
""",
)

add_docstr(
    torch.split_copy,
    r"""
Performs the same operation as :func:`torch.split`, but all output tensors
are freshly created instead of aliasing the input.
""",
)

add_docstr(
    torch.split_with_sizes_copy,
    r"""
Performs the same operation as :func:`torch.split_with_sizes`, but all output tensors
are freshly created instead of aliasing the input.
""",
)

add_docstr(
    torch.squeeze_copy,
    r"""
Performs the same operation as :func:`torch.squeeze`, but all output tensors
are freshly created instead of aliasing the input.
""",
)

add_docstr(
    torch.t_copy,
    r"""
Performs the same operation as :func:`torch.t`, but all output tensors
are freshly created instead of aliasing the input.
""",
)

add_docstr(
    torch.transpose_copy,
    r"""
Performs the same operation as :func:`torch.transpose`, but all output tensors
are freshly created instead of aliasing the input.
""",
)

add_docstr(
    torch.unsqueeze_copy,
    r"""
Performs the same operation as :func:`torch.unsqueeze`, but all output tensors
are freshly created instead of aliasing the input.
""",
)

add_docstr(
    torch.indices_copy,
    r"""
Performs the same operation as :func:`torch.indices`, but all output tensors
are freshly created instead of aliasing the input.
""",
)

add_docstr(
    torch.values_copy,
    r"""
Performs the same operation as :func:`torch.values`, but all output tensors
are freshly created instead of aliasing the input.
""",
)

add_docstr(
    torch.crow_indices_copy,
    r"""
Performs the same operation as :func:`torch.crow_indices`, but all output tensors
are freshly created instead of aliasing the input.
""",
)

add_docstr(
    torch.col_indices_copy,
    r"""
Performs the same operation as :func:`torch.col_indices`, but all output tensors
are freshly created instead of aliasing the input.
""",
)

add_docstr(
    torch.unbind_copy,
    r"""
Performs the same operation as :func:`torch.unbind`, but all output tensors
are freshly created instead of aliasing the input.
""",
)

add_docstr(
    torch.view_copy,
    r"""
Performs the same operation as :func:`torch.view`, but all output tensors
are freshly created instead of aliasing the input.
""",
)

add_docstr(
    torch.unfold_copy,
    r"""
Performs the same operation as :func:`torch.unfold`, but all output tensors
are freshly created instead of aliasing the input.
""",
)
# 为 torch.alias_copy 函数添加文档字符串
add_docstr(
    torch.alias_copy,
    r"""
Performs the same operation as :func:`torch.alias`, but all output tensors
are freshly created instead of aliasing the input.
""",
)

# 遍历一组一元函数名
for unary_base_func_name in (
    "exp",
    "sqrt",
    "abs",
    "acos",
    "asin",
    "atan",
    "ceil",
    "cos",
    "cosh",
    "erf",
    "erfc",
    "expm1",
    "floor",
    "log",
    "log10",
    "log1p",
    "log2",
    "neg",
    "tan",
    "tanh",
    "sin",
    "sinh",
    "round",
    "lgamma",
    "frac",
    "reciprocal",
    "sigmoid",
    "trunc",
    "zero",
):
    # 构造对应的 _foreach 函数名
    unary_foreach_func_name = f"_foreach_{unary_base_func_name}"
    # 检查是否存在对应的 _foreach 函数，并添加文档字符串
    if hasattr(torch, unary_foreach_func_name):
        add_docstr(
            getattr(torch, unary_foreach_func_name),
            rf"""
{unary_foreach_func_name}(self: List[Tensor]) -> List[Tensor]

Apply :func:`torch.{unary_base_func_name}` to each Tensor of the input list.
            """,
        )
    
    # 构造对应的 _foreach_inplace 函数名
    unary_inplace_foreach_func_name = f"{unary_foreach_func_name}_"
    # 检查是否存在对应的 _foreach_inplace 函数，并添加文档字符串
    if hasattr(torch, unary_inplace_foreach_func_name):
        add_docstr(
            getattr(torch, unary_inplace_foreach_func_name),
            rf"""
{unary_inplace_foreach_func_name}(self: List[Tensor]) -> None

Apply :func:`torch.{unary_base_func_name}` to each Tensor of the input list.
        """,
        )
```