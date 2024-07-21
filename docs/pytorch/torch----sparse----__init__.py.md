# `.\pytorch\torch\sparse\__init__.py`

```py
# 指定不允许未类型化的函数定义
# 该模块由 python_tensor.cpp 添加了 Tensor 类
from typing import Optional, Tuple, List, Union, Any

import torch
from torch._C import _add_docstr, _sparse  # type: ignore[attr-defined]
from torch import Tensor

# 半结构化稀疏张量支持
from .semi_structured import (
    SparseSemiStructuredTensor,
    SparseSemiStructuredTensorCUSPARSELT,
    SparseSemiStructuredTensorCUTLASS,
    to_sparse_semi_structured
)

# 用于支持 TorchScript 和 MyPy 的解决方案
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torch.types import _dtype as DType
    DimOrDims = Optional[Union[int, Tuple[int, ...], List[int]]]
else:
    # JIT 不理解 Union 和 torch.dtype
    DType = int
    DimOrDims = Optional[Tuple[int]]

# 导出的符号列表
__all__ = [
    'addmm',
    'check_sparse_tensor_invariants',
    'mm',
    'sum',
    'softmax',
    'log_softmax',
    'SparseSemiStructuredTensor',
    'SparseSemiStructuredTensorCUTLASS',
    'SparseSemiStructuredTensorCUSPARSELT',
    'to_sparse_semi_structured',
    'as_sparse_gradcheck',
]

# 给 sparse._sparse_addmm 函数添加文档字符串
addmm = _add_docstr(_sparse._sparse_addmm, r"""
sparse.addmm(mat, mat1, mat2, *, beta=1., alpha=1.) -> Tensor

This function does exact same thing as :func:`torch.addmm` in the forward,
except that it supports backward for sparse COO matrix :attr:`mat1`.
When :attr:`mat1` is a COO tensor it must have `sparse_dim = 2`.
When inputs are COO tensors, this function also supports backward for both inputs.

Supports both CSR and COO storage formats.

.. note::
    This function doesn't support computing derivaties with respect to CSR matrices.

Args:
    mat (Tensor): a dense matrix to be added
    mat1 (Tensor): a sparse matrix to be multiplied
    mat2 (Tensor): a dense matrix to be multiplied
    beta (Number, optional): multiplier for :attr:`mat` (:math:`\beta`)
    alpha (Number, optional): multiplier for :math:`mat1 @ mat2` (:math:`\alpha`)
""")

# 给 sparse._sparse_mm 函数添加文档字符串
mm = _add_docstr(_sparse._sparse_mm, r"""
    Performs a matrix multiplication of the sparse matrix :attr:`mat1`
    and the (sparse or strided) matrix :attr:`mat2`. Similar to :func:`torch.mm`, if :attr:`mat1` is a
    :math:`(n \times m)` tensor, :attr:`mat2` is a :math:`(m \times p)` tensor, out will be a
    :math:`(n \times p)` tensor.
    When :attr:`mat1` is a COO tensor it must have `sparse_dim = 2`.
    When inputs are COO tensors, this function also supports backward for both inputs.

    Supports both CSR and COO storage formats.

.. note::
    This function doesn't support computing derivaties with respect to CSR matrices.

    This function also additionally accepts an optional :attr:`reduce` argument that allows
    specification of an optional reduction operation, mathematically performs the following operation:

.. math::

    z_{ij} = \bigoplus_{k = 0}^{K - 1} x_{ik} y_{kj}

where :math:`\bigoplus` defines the reduce operator. :attr:`reduce` is implemented only for
CSR storage format on CPU device.
sampled_addmm = _add_docstr(_sparse.sparse_sampled_addmm, r"""
sparse.sampled_addmm(input, mat1, mat2, *, beta=1., alpha=1., out=None) -> Tensor

Performs a matrix multiplication of the dense matrices :attr:`mat1` and :attr:`mat2` at the locations
specified by the sparsity pattern of :attr:`input`. The matrix :attr:`input` is added to the final result.

Mathematically this performs the following operation:

.. math::

    \text{out} = \alpha\ (\text{mat1} \mathbin{@} \text{mat2})*\text{spy}(\text{input}) + \beta\ \text{input}

where :math:`\text{spy}(\text{input})` is the sparsity pattern matrix of :attr:`input`, :attr:`alpha`
and :attr:`beta` are the scaling factors.
:math:`\text{spy}(\text{input})` has value 1 at the positions where :attr:`input` has non-zero values, and 0 elsewhere.

.. note::
    :attr:`input` must be a sparse CSR tensor. :attr:`mat1` and :attr:`mat2` must be dense tensors.

Args:
    input (Tensor): a sparse CSR matrix of shape `(m, n)` to be added and used to compute
        the sampled matrix multiplication
    mat1 (Tensor): a dense matrix of shape `(m, k)` to be multiplied
    mat2 (Tensor): a dense matrix of shape `(k, n)` to be multiplied

Keyword args:
    beta (float, optional): scaling factor for the sparse input matrix addition. Default is 1.0.
    alpha (float, optional): scaling factor for the result of dense matrix multiplication. Default is 1.0.
    out (Tensor, optional): output tensor to store the result. Default is None.

""")
    beta (Number, optional): multiplier for :attr:`input` (:math:`\beta`)
    alpha (Number, optional): multiplier for :math:`mat1 @ mat2` (:math:`\alpha`)
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.
# 返回给定稀疏张量每行的总和
def sum(input: Tensor, dim: DimOrDims = None,
        dtype: Optional[DType] = None) -> Tensor:
    r"""Return the sum of each row of the given sparse tensor.

    Returns the sum of each row of the sparse tensor :attr:`input` in the given
    dimensions :attr:`dim`. If :attr:`dim` is a list of dimensions,
    reduce over all of them. When sum over all ``sparse_dim``, this method
    returns a dense tensor instead of a sparse tensor.

    All summed :attr:`dim` are squeezed (see :func:`torch.squeeze`), resulting an output
    tensor having :attr:`dim` fewer dimensions than :attr:`input`.

    During backward, only gradients at ``nnz`` locations of :attr:`input`
    will propagate back. Note that the gradients of :attr:`input` is coalesced.

    Args:
        input (Tensor): the input sparse tensor  # 输入的稀疏张量
        dim (int or tuple of ints): a dimension or a list of dimensions to reduce. Default: reduce
            over all dims.  # 要减少的维度或维度列表。默认情况下减少所有维度。
        dtype (:class:`torch.dtype`, optional): the desired data type of returned Tensor.
            Default: dtype of :attr:`input`.  # 返回张量的所需数据类型，默认为输入张量的数据类型。
    # 如果未指定数据类型(dtype)，则根据维度(dim)是否存在选择不同的稀疏求和操作
    if dtype is None:
        # 如果指定了维度(dim)，则调用 torch._sparse_sum 对稀疏张量(input)进行求和
        if dim is not None:
            return torch._sparse_sum(input, dim)
        else:
            # 如果未指定维度(dim)，则对整个稀疏张量(input)进行求和
            return torch._sparse_sum(input)
    else:
        # 如果指定了数据类型(dtype)，则根据维度(dim)是否存在选择不同的稀疏求和操作
        if dim is not None:
            # 调用 torch._sparse_sum 对稀疏张量(input)进行求和，并指定数据类型(dtype)
            return torch._sparse_sum(input, dim, dtype=dtype)
        else:
            # 调用 torch._sparse_sum 对稀疏张量(input)进行求和，并指定数据类型(dtype)
            return torch._sparse_sum(input, dtype=dtype)
softmax = _add_docstr(_sparse._sparse_softmax, r"""
sparse.softmax(input, dim, *, dtype=None) -> Tensor

Applies a softmax function.

Softmax is defined as:

:math:`\text{Softmax}(x_{i}) = \frac{exp(x_i)}{\sum_j exp(x_j)}`

where :math:`i, j` run over sparse tensor indices and unspecified
entries are ignores. This is equivalent to defining unspecified
entries as negative infinity so that :math:`exp(x_k) = 0` when the
entry with index :math:`k` has not specified.

It is applied to all slices along `dim`, and will re-scale them so
that the elements lie in the range `[0, 1]` and sum to 1.

Args:
    input (Tensor): input
    dim (int): A dimension along which softmax will be computed.
    dtype (:class:`torch.dtype`, optional): the desired data type
        of returned tensor.  If specified, the input tensor is
        casted to :attr:`dtype` before the operation is
        performed. This is useful for preventing data type
        overflows. Default: None
""")

log_softmax = _add_docstr(_sparse._sparse_log_softmax, r"""
sparse.log_softmax(input, dim, *, dtype=None) -> Tensor

Applies a softmax function followed by logarithm.

See :class:`~torch.sparse.softmax` for more details.

Args:
    input (Tensor): input
    dim (int): A dimension along which softmax will be computed.
    dtype (:class:`torch.dtype`, optional): the desired data type
        of returned tensor.  If specified, the input tensor is
        casted to :attr:`dtype` before the operation is
        performed. This is useful for preventing data type
        overflows. Default: None
""")

spdiags = _add_docstr(
    _sparse._spdiags,
    r"""
sparse.spdiags(diagonals, offsets, shape, layout=None) -> Tensor

Creates a sparse 2D tensor by placing the values from rows of
:attr:`diagonals` along specified diagonals of the output

The :attr:`offsets` tensor controls which diagonals are set.

- If :attr:`offsets[i]` = 0, it is the main diagonal
- If :attr:`offsets[i]` < 0, it is below the main diagonal
- If :attr:`offsets[i]` > 0, it is above the main diagonal

The number of rows in :attr:`diagonals` must match the length of :attr:`offsets`,
and an offset may not be repeated.

Args:
    diagonals (Tensor): Matrix storing diagonals row-wise
    offsets (Tensor): The diagonals to be set, stored as a vector
    shape (2-tuple of ints): The desired shape of the result
Keyword args:
    layout (:class:`torch.layout`, optional): The desired layout of the
        returned tensor. ``torch.sparse_coo``, ``torch.sparse_csc`` and ``torch.sparse_csr``
        are supported. Default: ``torch.sparse_coo``

Examples:

Set the main and first two lower diagonals of a matrix::

    >>> diags = torch.arange(9).reshape(3, 3)
    >>> diags
    tensor([[0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]])
    >>> s = torch.sparse.spdiags(diags, torch.tensor([0, -1, -2]), (3, 3))
    >>> s

""")
    # 创建稀疏张量对象，指定其非零元素的索引和值，以及张量的大小和非零元素的数量
    tensor(indices=tensor([[0, 1, 2, 1, 2, 2],    # 稀疏张量的索引，表示非零元素在张量中的位置
                           [0, 1, 2, 0, 1, 0]]),   # 稀疏张量的索引，表示非零元素在张量中的位置
           values=tensor([0, 1, 2, 3, 4, 6]),    # 稀疏张量的非零元素值
           size=(3, 3),                         # 稀疏张量的大小，表示其形状为 (3, 3)
           nnz=6,                               # 稀疏张量的非零元素数量
           layout=torch.sparse_coo)             # 稀疏张量的布局，采用 COO 格式
    
    # 将稀疏张量转换为密集张量（即普通的二维张量），其中未出现的元素默认为0
    >>> s.to_dense()
    tensor([[0, 0, 0],    # 密集张量的第一行
            [3, 1, 0],    # 密集张量的第二行
            [6, 4, 2]])   # 密集张量的第三行
# 修改输出布局以生成稀疏对角矩阵的示例

>>> diags = torch.arange(9).reshape(3, 3)
# 创建一个3x3的张量，其值从0到8，reshape成3行3列
>>> diags
# 打印出刚刚创建的张量
tensor([[0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]])

>>> s = torch.sparse.spdiags(diags, torch.tensor([0, -1, -2]), (3, 3), layout=torch.sparse_csr)
# 使用给定的对角线数据和偏移量，创建一个稀疏矩阵，布局为CSR格式，大小为3x3
>>> s
# 打印出创建的稀疏矩阵
tensor(crow_indices=tensor([0, 1, 3, 6]),
       col_indices=tensor([0, 0, 1, 0, 1, 2]),
       values=tensor([0, 3, 1, 6, 4, 2]), size=(3, 3), nnz=6,
       layout=torch.sparse_csr)

>>> s.to_dense()
# 将稀疏矩阵转换为密集矩阵并打印出来
tensor([[0, 0, 0],
        [3, 1, 0],
        [6, 4, 2]])

# 设置大输出的部分对角线示例

>>> diags = torch.tensor([[1, 2], [3, 4]])
# 创建一个2x2的张量，其值为[[1, 2], [3, 4]]
>>> offsets = torch.tensor([0, -1])
# 创建一个偏移量张量，值为[0, -1]
>>> torch.sparse.spdiags(diags, offsets, (5, 5)).to_dense()
# 使用给定的对角线数据和偏移量，创建一个稀疏矩阵，大小为5x5，并将其转换为密集矩阵打印出来
tensor([[1, 0, 0, 0, 0],
        [3, 2, 0, 0, 0],
        [0, 4, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]])

# 指定正偏移量的示例

>>> diags = torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
# 创建一个3x3的张量，其值为[[1, 2, 3], [1, 2, 3], [1, 2, 3]]
>>> torch.sparse.spdiags(diags, torch.tensor([0, 1, 2]), (5, 5)).to_dense()
# 使用给定的对角线数据和偏移量，创建一个稀疏矩阵，大小为5x5，并将其转换为密集矩阵打印出来
tensor([[1, 2, 3, 0, 0],
        [0, 2, 3, 0, 0],
        [0, 0, 3, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]])

"""

class check_sparse_tensor_invariants:
    """用于控制稀疏张量不变性检查的工具类。

    以下选项可用于管理在稀疏张量构建中检查稀疏张量不变性的设置：

    1. 使用上下文管理器：

       .. code:: python

           with torch.sparse.check_sparse_tensor_invariants():
               run_my_model()

    2. 使用过程化方法：

       .. code:: python

           prev_checks_enabled = torch.sparse.check_sparse_tensor_invariants.is_enabled()
           torch.sparse.check_sparse_tensor_invariants.enable()

           run_my_model()

           if not prev_checks_enabled:
               torch.sparse.check_sparse_tensor_invariants.disable()

    3. 使用函数装饰器：

       .. code:: python

           @torch.sparse.check_sparse_tensor_invariants()
           def run_my_model():
               ...

           run_my_model()

    4. 在稀疏张量构造函数调用中使用 ``check_invariants`` 关键字参数。
       例如：

       >>> torch.sparse_csr_tensor([0, 1, 3], [0, 1], [1, 2], check_invariants=True)
       Traceback (most recent call last):
         File "<stdin>", line 1, in <module>
       RuntimeError: `crow_indices[..., -1] == nnz` is not satisfied.
    """
    # 检查稀疏张量不变性检查是否已启用，返回True或False
    def is_enabled():
        r"""Return True if the sparse tensor invariants checking is enabled.

        .. note::

            Use :func:`torch.sparse.check_sparse_tensor_invariants.enable` or
            :func:`torch.sparse.check_sparse_tensor_invariants.disable` to
            manage the state of the sparse tensor invariants checks.
        """
        return torch._C._check_sparse_tensor_invariants()

    @staticmethod
    # 启用稀疏张量不变性检查，影响稀疏张量的构造函数
    def enable():
        r"""Enable sparse tensor invariants checking in sparse tensor constructors.

        .. note::

            By default, the sparse tensor invariants checks are disabled. Use
            :func:`torch.sparse.check_sparse_tensor_invariants.is_enabled` to
            retrieve the current state of sparse tensor invariants checking.

        .. note::

            The sparse tensor invariants check flag is effective to all sparse
            tensor constructors, both in Python and ATen.

        The flag can be locally overridden by the ``check_invariants``
        optional argument of the sparse tensor constructor functions.
        """
        torch._C._set_check_sparse_tensor_invariants(True)

    @staticmethod
    # 禁用稀疏张量不变性检查，影响稀疏张量的构造函数
    def disable():
        r"""Disable sparse tensor invariants checking in sparse tensor constructors.

        See :func:`torch.sparse.check_sparse_tensor_invariants.enable` for more information.
        """
        torch._C._set_check_sparse_tensor_invariants(False)

    # 上下文管理器支持
    def __init__(self, enable=True):
        # 初始化对象状态，根据enable参数设置初始状态
        self.state = enable
        self.saved_state : Optional[bool] = None

    def __enter__(self):
        # 进入上下文时，保存当前状态并设置新状态
        if self.saved_state is not None:
            raise RuntimeError('This context manager instance is already activated.'
                               ' Use a different context manager instance for context nesting.')
        self.saved_state = self.is_enabled()
        torch._C._set_check_sparse_tensor_invariants(self.state)

    def __exit__(self, type, value, traceback):
        # 退出上下文时，恢复之前保存的状态
        assert self.saved_state is not None
        torch._C._set_check_sparse_tensor_invariants(self.saved_state)
        self.saved_state = None

    # 装饰器支持
    def __call__(self, mth):
        # 定义一个包装函数，使用新的上下文管理器实例调用原始方法
        def test_mth(*args, **kwargs):
            with type(self)(self.state):
                return mth(*args, **kwargs)

        return test_mth
def as_sparse_gradcheck(gradcheck):
    """
    Decorate function, to extend gradcheck for sparse tensors.

    Decorator for torch.autograd.gradcheck or its functools.partial
    variants that extends the gradcheck function with support to input
    functions that operate on or/and return sparse tensors.

    The specified gradcheck function itself is guaranteed to operate
    on strided tensors only.

    For example:

    >>> gradcheck = torch.sparse.as_sparse_gradcheck(torch.autograd.gradcheck)
    >>> x = torch.tensor([[0, 1], [2, 3]], dtype=torch.float64).to_sparse_coo().requires_grad_(True)
    >>> gradcheck(lambda x: x.to_sparse_csr(), x)
    True
    """

    # 返回一个支持稀疏张量的 gradcheck 函数的装饰器
    return gradcheck_with_sparse_support
```