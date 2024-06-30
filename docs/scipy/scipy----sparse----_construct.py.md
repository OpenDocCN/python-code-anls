# `D:\src\scipysrc\scipy\scipy\sparse\_construct.py`

```
"""Functions to construct sparse matrices and arrays
"""

__docformat__ = "restructuredtext en"

__all__ = ['spdiags', 'eye', 'identity', 'kron', 'kronsum',
           'hstack', 'vstack', 'bmat', 'rand', 'random', 'diags', 'block_diag',
           'diags_array', 'block_array', 'eye_array', 'random_array']

import numbers              # 导入用于数值判断的模块
import math                 # 导入数学函数模块
import numpy as np          # 导入 NumPy 库

from scipy._lib._util import check_random_state, rng_integers  # 导入一些工具函数
from ._sputils import upcast, get_index_dtype, isscalarlike     # 导入辅助函数
from ._sparsetools import csr_hstack     # 导入 CSR 格式的稀疏矩阵堆叠函数
from ._bsr import bsr_matrix, bsr_array   # 导入 BSR 格式的稀疏矩阵和数组构造函数
from ._coo import coo_matrix, coo_array   # 导入 COO 格式的稀疏矩阵和数组构造函数
from ._csc import csc_matrix, csc_array   # 导入 CSC 格式的稀疏矩阵和数组构造函数
from ._csr import csr_matrix, csr_array   # 导入 CSR 格式的稀疏矩阵和数组构造函数
from ._dia import dia_matrix, dia_array   # 导入 DIA 格式的稀疏矩阵和数组构造函数

from ._base import issparse, sparray   # 导入判断是否为稀疏矩阵和稀疏数组的函数


def spdiags(data, diags, m=None, n=None, format=None):
    """
    Return a sparse matrix from diagonals.

    Parameters
    ----------
    data : array_like
        Matrix diagonals stored row-wise
    diags : sequence of int or an int
        Diagonals to set:

        * k = 0  the main diagonal
        * k > 0  the kth upper diagonal
        * k < 0  the kth lower diagonal
    m, n : int, tuple, optional
        Shape of the result. If `n` is None and `m` is a given tuple,
        the shape is this tuple. If omitted, the matrix is square and
        its shape is len(data[0]).
    format : str, optional
        Format of the result. By default (format=None) an appropriate sparse
        matrix format is returned. This choice is subject to change.

    .. warning::

        This function returns a sparse matrix -- not a sparse array.
        You are encouraged to use ``diags_array`` to take advantage
        of the sparse array functionality.

    See Also
    --------
    diags_array : more convenient form of this function
    diags : matrix version of diags_array
    dia_matrix : the sparse DIAgonal format.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import spdiags
    >>> data = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    >>> diags = np.array([0, -1, 2])
    >>> spdiags(data, diags, 4, 4).toarray()
    array([[1, 0, 3, 0],
           [1, 2, 0, 4],
           [0, 2, 3, 0],
           [0, 0, 3, 4]])

    """
    if m is None and n is None:     # 如果 m 和 n 都未指定
        m = n = len(data[0])        # 则默认为正方形矩阵，其形状为 data[0] 的长度
    elif n is None:                 # 如果只有 n 未指定
        m, n = m                    # 则将 m 解包给 m，n
    return dia_matrix((data, diags), shape=(m, n)).asformat(format)   # 返回 DIA 格式的稀疏矩阵，并根据指定的格式进行格式化


def diags_array(diagonals, /, *, offsets=0, shape=None, format=None, dtype=None):
    """
    Construct a sparse array from diagonals.

    Parameters
    ----------
    diagonals : sequence of array_like
        Sequence of arrays containing the array diagonals,
        corresponding to `offsets`.
    offsets : sequence of int or an int, optional
        Diagonals to set:
          - k = 0  the main diagonal (default)
          - k > 0  the kth upper diagonal
          - k < 0  the kth lower diagonal

    """
    # shape : tuple of int, optional
    #     结果数组的形状。如果省略，则返回足够大以包含对角线的正方形数组。
    # format : {"dia", "csr", "csc", "lil", ...}, optional
    #     结果的稀疏矩阵格式。默认情况下（format=None），返回合适的稀疏数组格式。此选择可能会更改。
    # dtype : dtype, optional
    #     数组的数据类型。

    Notes
    -----
    # diags_array 的结果等效于以下稀疏对角线的总和：
    #
    #     np.diag(diagonals[0], offsets[0])
    #     + ...
    #     + np.diag(diagonals[k], offsets[k])
    #
    # 不允许重复的对角线偏移。

    .. versionadded:: 1.11

    Examples
    --------
    # 导入 diags_array 函数
    >>> from scipy.sparse import diags_array
    # 定义对角线列表
    >>> diagonals = [[1, 2, 3, 4], [1, 2, 3], [1, 2]]
    # 调用 diags_array 并将结果转换成数组输出
    >>> diags_array(diagonals, offsets=[0, -1, 2]).toarray()
    array([[1, 0, 1, 0],
           [1, 2, 0, 2],
           [0, 2, 3, 0],
           [0, 0, 3, 4]])

    # 支持标量的广播（但需要指定形状）
    >>> diags_array([1, -2, 1], offsets=[-1, 0, 1], shape=(4, 4)).toarray()
    array([[-2.,  1.,  0.,  0.],
           [ 1., -2.,  1.,  0.],
           [ 0.,  1., -2.,  1.],
           [ 0.,  0.,  1., -2.]])

    # 如果只需要一个对角线（类似于 `numpy.diag`），以下也有效：
    >>> diags_array([1, 2, 3], offsets=1).toarray()
    array([[ 0.,  1.,  0.,  0.],
           [ 0.,  0.,  2.,  0.],
           [ 0.,  0.,  0.,  3.],
           [ 0.,  0.,  0.,  0.]])
    """
    # 如果 offsets 不是序列，则假设只有一个对角线
    if isscalarlike(offsets):
        # 现在检查是否确实只有一个对角线
        if len(diagonals) == 0 or isscalarlike(diagonals[0]):
            diagonals = [np.atleast_1d(diagonals)]
        else:
            raise ValueError("Different number of diagonals and offsets.")
    else:
        diagonals = list(map(np.atleast_1d, diagonals))

    # 将 offsets 转换为至少是一维的 numpy 数组
    offsets = np.atleast_1d(offsets)

    # 基本检查
    if len(diagonals) != len(offsets):
        raise ValueError("Different number of diagonals and offsets.")

    # 确定形状，如果未指定
    if shape is None:
        m = len(diagonals[0]) + abs(int(offsets[0]))
        shape = (m, m)

    # 确定数据类型，如果未指定
    if dtype is None:
        dtype = np.common_type(*diagonals)

    # 构造数据数组
    m, n = shape

    M = max([min(m + offset, n - offset) + max(0, offset)
             for offset in offsets])
    M = max(0, M)
    data_arr = np.zeros((len(offsets), M), dtype=dtype)

    K = min(m, n)
    # 遍历对角线列表以及它们对应的偏移量
    for j, diagonal in enumerate(diagonals):
        # 获取当前对角线的偏移量
        offset = offsets[j]
        # 计算从偏移量开始的有效索引起点
        k = max(0, offset)
        # 计算当前对角线的有效长度，限制在数组边界以及最大对角线长度 K 内
        length = min(m + offset, n - offset, K)
        # 如果有效长度小于 0，抛出值错误异常
        if length < 0:
            raise ValueError("Offset %d (index %d) out of bounds" % (offset, j))
        try:
            # 将当前对角线的数据部分复制到数组的相应位置
            data_arr[j, k:k+length] = diagonal[...,:length]
        except ValueError as e:
            # 如果对角线的长度与数组的大小不符合预期，抛出值错误异常
            if len(diagonal) != length and len(diagonal) != 1:
                raise ValueError(
                    "Diagonal length (index %d: %d at offset %d) does not "
                    "agree with array size (%d, %d)." % (
                    j, len(diagonal), offset, m, n)) from e
            raise

    # 返回一个 dia_array 对象，包含填充好的数据数组和对角线偏移信息，以指定格式表示
    return dia_array((data_arr, offsets), shape=(m, n)).asformat(format)
# 构建稀疏矩阵，从给定的对角线数据中生成
def diags(diagonals, offsets=0, shape=None, format=None, dtype=None):
    """
    Construct a sparse matrix from diagonals.

    .. warning::

        This function returns a sparse matrix -- not a sparse array.
        You are encouraged to use ``diags_array`` to take advantage
        of the sparse array functionality.

    Parameters
    ----------
    diagonals : sequence of array_like
        Sequence of arrays containing the matrix diagonals,
        corresponding to `offsets`.
    offsets : sequence of int or an int, optional
        Diagonals to set:
          - k = 0  the main diagonal (default)
          - k > 0  the kth upper diagonal
          - k < 0  the kth lower diagonal
    shape : tuple of int, optional
        Shape of the result. If omitted, a square matrix large enough
        to contain the diagonals is returned.
    format : {"dia", "csr", "csc", "lil", ...}, optional
        Matrix format of the result. By default (format=None) an
        appropriate sparse matrix format is returned. This choice is
        subject to change.
    dtype : dtype, optional
        Data type of the matrix.

    See Also
    --------
    spdiags : construct matrix from diagonals
    diags_array : construct sparse array instead of sparse matrix

    Notes
    -----
    This function differs from `spdiags` in the way it handles
    off-diagonals.

    The result from `diags` is the sparse equivalent of::

        np.diag(diagonals[0], offsets[0])
        + ...
        + np.diag(diagonals[k], offsets[k])

    Repeated diagonal offsets are disallowed.

    .. versionadded:: 0.11

    Examples
    --------
    >>> from scipy.sparse import diags
    >>> diagonals = [[1, 2, 3, 4], [1, 2, 3], [1, 2]]
    >>> diags(diagonals, [0, -1, 2]).toarray()
    array([[1, 0, 1, 0],
           [1, 2, 0, 2],
           [0, 2, 3, 0],
           [0, 0, 3, 4]])

    Broadcasting of scalars is supported (but shape needs to be
    specified):

    >>> diags([1, -2, 1], [-1, 0, 1], shape=(4, 4)).toarray()
    array([[-2.,  1.,  0.,  0.],
           [ 1., -2.,  1.,  0.],
           [ 0.,  1., -2.,  1.],
           [ 0.,  0.,  1., -2.]])


    If only one diagonal is wanted (as in `numpy.diag`), the following
    works as well:

    >>> diags([1, 2, 3], 1).toarray()
    array([[ 0.,  1.,  0.,  0.],
           [ 0.,  0.,  2.,  0.],
           [ 0.,  0.,  0.,  3.],
           [ 0.,  0.,  0.,  0.]])
    """
    # 使用 diags_array 函数构建稀疏数组 A
    A = diags_array(diagonals, offsets=offsets, shape=shape, dtype=dtype)
    # 将稀疏数组 A 转换为 dia_matrix 对象，并按照指定格式格式化
    return dia_matrix(A).asformat(format)


def identity(n, dtype='d', format=None):
    """Identity matrix in sparse format

    Returns an identity matrix with shape (n,n) using a given
    sparse format and dtype. This differs from `eye_array` in
    that it has a square shape with ones only on the main diagonal.
    It is thus the multiplicative identity. `eye_array` allows
    rectangular shapes and the diagonal can be offset from the main one.
    """
    # 返回一个形状为 (n,n) 的单位稀疏矩阵，使用给定的稀疏格式和数据类型
    # 只在主对角线上有值为 1，是乘法单位矩阵
    return diags([1] * n, 0, shape=(n, n), dtype=dtype).asformat(format)
    .. warning::

        This function returns a sparse matrix -- not a sparse array.
        You are encouraged to use ``eye_array`` to take advantage
        of the sparse array functionality.

    Parameters
    ----------
    n : int
        Shape of the identity matrix.
    dtype : dtype, optional
        Data type of the matrix
    format : str, optional
        Sparse format of the result, e.g., format="csr", etc.

    Examples
    --------
    >>> import scipy as sp
    >>> sp.sparse.identity(3).toarray()
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> sp.sparse.identity(3, dtype='int8', format='dia')
    <DIAgonal sparse matrix of dtype 'int8'
        with 3 stored elements (1 diagonals) and shape (3, 3)>
    >>> sp.sparse.eye_array(3, dtype='int8', format='dia')
    <DIAgonal sparse array of dtype 'int8'
        with 3 stored elements (1 diagonals) and shape (3, 3)>

    """
    返回一个稀疏矩阵的单位对角阵，而不是稀疏数组。建议使用``eye_array``来利用稀疏数组的功能，生成指定格式和数据类型的单位对角阵。
    返回的矩阵形状为(n, n)，数据类型为dtype，稀疏格式为format。
# 创建一个稀疏数组表示的单位矩阵
def eye_array(m, n=None, *, k=0, dtype=float, format=None):
    """Identity matrix in sparse array format

    Return a sparse array with ones on diagonal.
    Specifically a sparse array (m x n) where the kth diagonal
    is all ones and everything else is zeros.

    Parameters
    ----------
    m : int or tuple of ints
        Number of rows requested.
    n : int, optional
        Number of columns. Default: `m`.
    k : int, optional
        Diagonal to place ones on. Default: 0 (main diagonal).
    dtype : dtype, optional
        Data type of the array
    format : str, optional (default: "dia")
        Sparse format of the result, e.g., format="csr", etc.

    Examples
    --------
    >>> import numpy as np
    >>> import scipy as sp
    >>> sp.sparse.eye_array(3).toarray()
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> sp.sparse.eye_array(3, dtype=np.int8)
    <DIAgonal sparse array of dtype 'int8'
        with 3 stored elements (1 diagonals) and shape (3, 3)>

    """
    # 调用内部函数 _eye()，返回稀疏矩阵
    return _eye(m, n, k, dtype, format)


# 内部函数：生成稀疏矩阵的主要逻辑
def _eye(m, n, k, dtype, format, as_sparray=True):
    # 根据 as_sparray 参数决定使用哪种稀疏矩阵类型
    if as_sparray:
        csr_sparse = csr_array
        csc_sparse = csc_array
        coo_sparse = coo_array
        diags_sparse = diags_array
    else:
        csr_sparse = csr_matrix
        csc_sparse = csc_matrix
        coo_sparse = coo_matrix
        diags_sparse = diags

    # 如果未指定列数 n，则将其设为行数 m（即生成方阵）
    if n is None:
        n = m
    m, n = int(m), int(n)

    # 对于正方形且主对角线上的情况，优化处理
    if m == n and k == 0:
        # 对于特定的稀疏格式，快速生成矩阵
        if format in ['csr', 'csc']:
            idx_dtype = get_index_dtype(maxval=n)
            indptr = np.arange(n+1, dtype=idx_dtype)
            indices = np.arange(n, dtype=idx_dtype)
            data = np.ones(n, dtype=dtype)
            cls = {'csr': csr_sparse, 'csc': csc_sparse}[format]
            return cls((data, indices, indptr), (n, n))

        elif format == 'coo':
            idx_dtype = get_index_dtype(maxval=n)
            row = np.arange(n, dtype=idx_dtype)
            col = np.arange(n, dtype=idx_dtype)
            data = np.ones(n, dtype=dtype)
            return coo_sparse((data, (row, col)), (n, n))

    # 对于一般情况，生成具有特定偏移的稀疏对角线矩阵
    data = np.ones((1, max(0, min(m + k, n))), dtype=dtype)
    return diags_sparse(data, offsets=[k], shape=(m, n), dtype=dtype).asformat(format)
    # 返回稀疏矩阵，而不是稀疏数组。建议使用 `eye_array` 函数来利用稀疏数组的功能。
    # 
    # 示例
    # --------
    # >>> import numpy as np
    # >>> import scipy as sp
    # >>> sp.sparse.eye(3).toarray()
    # array([[ 1.,  0.,  0.],
    #        [ 0.,  1.,  0.],
    #        [ 0.,  0.,  1.]])
    # >>> sp.sparse.eye(3, dtype=np.int8)
    # <DIAgonal sparse matrix of dtype 'int8'
    #     with 3 stored elements (1 diagonals) and shape (3, 3)>
    # 
    # """
    # 使用 _eye 函数返回一个稀疏矩阵。
    return _eye(m, n, k, dtype, format, False)
# 定义一个函数，计算稀疏矩阵 A 和 B 的克罗内克积
def kron(A, B, format=None):
    """kronecker product of sparse matrices A and B
    
    Parameters
    ----------
    A : sparse or dense matrix
        first matrix of the product
    B : sparse or dense matrix
        second matrix of the product
    format : str, optional (default: 'bsr' or 'coo')
        format of the result (e.g. "csr")
        If None, choose 'bsr' for relatively dense array and 'coo' for others
    
    Returns
    -------
    kronecker product in a sparse format.
    Returns a sparse matrix unless either A or B is a
    sparse array in which case returns a sparse array.
    
    Examples
    --------
    >>> import numpy as np
    >>> import scipy as sp
    >>> A = sp.sparse.csr_array(np.array([[0, 2], [5, 0]]))
    >>> B = sp.sparse.csr_array(np.array([[1, 2], [3, 4]]))
    >>> sp.sparse.kron(A, B).toarray()
    array([[ 0,  0,  2,  4],
           [ 0,  0,  6,  8],
           [ 5, 10,  0,  0],
           [15, 20,  0,  0]])
    
    >>> sp.sparse.kron(A, [[1, 2], [3, 4]]).toarray()
    array([[ 0,  0,  2,  4],
           [ 0,  0,  6,  8],
           [ 5, 10,  0,  0],
           [15, 20,  0,  0]])
    
    """
    # TODO: delete next 10 lines and replace _sparse with _array when spmatrix removed
    
    # 如果 A 或 B 是稀疏矩阵，则将对应的稀疏矩阵类型赋给本地变量
    if isinstance(A, sparray) or isinstance(B, sparray):
        bsr_sparse = bsr_array  # 使用 BSR 格式的稀疏矩阵
        csr_sparse = csr_array  # 使用 CSR 格式的稀疏矩阵
        coo_sparse = coo_array  # 使用 COO 格式的稀疏矩阵
    else:
        # 否则使用 spmatrix
        bsr_sparse = bsr_matrix  # 使用 BSR 格式的稀疏矩阵
        csr_sparse = csr_matrix  # 使用 CSR 格式的稀疏矩阵
        coo_sparse = coo_matrix  # 使用 COO 格式的稀疏矩阵
    
    # 将矩阵 B 转换为 COO 格式的稀疏矩阵
    B = coo_sparse(B)
    
    # 如果 B 相对密集，则选择 BSR 格式
    if (format is None or format == "bsr") and 2*B.nnz >= B.shape[0] * B.shape[1]:
        A = csr_sparse(A, copy=True)  # 将 A 转换为 CSR 格式的稀疏矩阵
        output_shape = (A.shape[0]*B.shape[0], A.shape[1]*B.shape[1])  # 计算输出的形状
        
        if A.nnz == 0 or B.nnz == 0:
            # 如果 A 或 B 的非零元素数为零，则返回格式化为指定格式的 COO 稀疏矩阵
            return coo_sparse(output_shape).asformat(format)
        
        B = B.toarray()  # 将 B 转换为密集数组
        data = A.data.repeat(B.size).reshape(-1, B.shape[0], B.shape[1])  # 重复 A 的数据并重新塑形
        data = data * B  # 对数据进行元素级乘法
        
        # 返回以 BSR 格式返回的稀疏矩阵，包括数据、列索引和行指针，以及指定的形状
        return bsr_sparse((data, A.indices, A.indptr), shape=output_shape)
    else:
        # 如果不是 CSR 或 CSC 格式的稀疏矩阵，则转换为 COO 格式
        A = coo_sparse(A)
        # 计算输出的矩阵形状
        output_shape = (A.shape[0]*B.shape[0], A.shape[1]*B.shape[1])

        # 如果 A 或者 B 中没有非零元素
        if A.nnz == 0 or B.nnz == 0:
            # 克罗内克积是零矩阵
            return coo_sparse(output_shape).asformat(format)

        # 将 A 的条目扩展成块
        row = A.row.repeat(B.nnz)
        col = A.col.repeat(B.nnz)
        data = A.data.repeat(B.nnz)

        # 如果需要使用大于 int32 最大值的行和列索引
        if max(A.shape[0]*B.shape[0], A.shape[1]*B.shape[1]) > np.iinfo('int32').max:
            row = row.astype(np.int64)
            col = col.astype(np.int64)

        # 计算行和列的增量
        row *= B.shape[0]
        col *= B.shape[1]

        # 增加块索引
        row,col = row.reshape(-1,B.nnz),col.reshape(-1,B.nnz)
        row += B.row
        col += B.col
        row,col = row.reshape(-1),col.reshape(-1)

        # 计算块的条目
        data = data.reshape(-1,B.nnz) * B.data
        data = data.reshape(-1)

        # 返回 COO 格式的稀疏矩阵，其数据由 data, (row, col) 组成，并使用指定的格式
        return coo_sparse((data,(row,col)), shape=output_shape).asformat(format)
def _compressed_sparse_stack(blocks, axis, return_spmatrix):
    """
    Stacking fast path for CSR/CSC matrices or arrays
    (i) vstack for CSR, (ii) hstack for CSC.
    """
    # 根据轴的方向确定另一个轴的索引
    other_axis = 1 if axis == 0 else 0
    # 提取所有块的数据并进行连接
    data = np.concatenate([b.data for b in blocks])
    # 获取第一个块的常数维度
    constant_dim = blocks[0].shape[other_axis]
    # 获取合适的索引数据类型
    idx_dtype = get_index_dtype(arrays=[b.indptr for b in blocks],
                                maxval=max(data.size, constant_dim))
    # 初始化索引数组和指针数组
    indices = np.empty(data.size, dtype=idx_dtype)
    indptr = np.empty(sum(b.shape[axis] for b in blocks) + 1, dtype=idx_dtype)
    last_indptr = idx_dtype(0)
    sum_dim = 0
    sum_indices = 0
    # 遍历所有块
    for b in blocks:
        # 检查轴的维度是否兼容
        if b.shape[other_axis] != constant_dim:
            raise ValueError(f'incompatible dimensions for axis {other_axis}')
        # 将块的索引数据复制到整体索引数组中
        indices[sum_indices:sum_indices+b.indices.size] = b.indices
        sum_indices += b.indices.size
        # 确定当前块在整体指针数组中的位置范围
        idxs = slice(sum_dim, sum_dim + b.shape[axis])
        indptr[idxs] = b.indptr[:-1]
        indptr[idxs] += last_indptr
        sum_dim += b.shape[axis]
        last_indptr += b.indptr[-1]
    # 最后一个指针位置
    indptr[-1] = last_indptr
    # TODO remove this if-structure when sparse matrices removed
    # 如果需要返回稀疏矩阵
    if return_spmatrix:
        if axis == 0:
            # 垂直堆叠，返回 CSR 矩阵
            return csr_matrix((data, indices, indptr),
                              shape=(sum_dim, constant_dim))
        else:
            # 水平堆叠，返回 CSC 矩阵
            return csc_matrix((data, indices, indptr),
                              shape=(constant_dim, sum_dim))

    # 不返回稀疏矩阵时的处理
    if axis == 0:
        # 垂直堆叠，返回 CSR 数组
        return csr_array((data, indices, indptr),
                         shape=(sum_dim, constant_dim))
    # 如果条件不满足（即 data 为空或 indices 和 indptr 为空），则执行以下操作
    else:
        # 调用 csc_array 函数，传入参数 (data, indices, indptr)，指定数组的形状为 (constant_dim, sum_dim)
        return csc_array((data, indices, indptr),
                          shape=(constant_dim, sum_dim))
def _stack_along_minor_axis(blocks, axis):
    """
    Stacking fast path for CSR/CSC matrices along the minor axis
    (i) hstack for CSR, (ii) vstack for CSC.
    """
    # 获取块的数量
    n_blocks = len(blocks)
    if n_blocks == 0:
        # 如果没有块，则抛出值错误异常
        raise ValueError('Missing block matrices')

    if n_blocks == 1:
        # 如果只有一个块，则直接返回该块
        return blocks[0]

    # 检查不兼容的维度
    other_axis = 1 if axis == 0 else 0
    other_axis_dims = {b.shape[other_axis] for b in blocks}
    if len(other_axis_dims) > 1:
        # 如果其他轴向的维度不匹配，则抛出值错误异常
        raise ValueError(f'Mismatching dimensions along axis {other_axis}: '
                         f'{other_axis_dims}')
    constant_dim, = other_axis_dims

    # 进行堆叠操作
    indptr_list = [b.indptr for b in blocks]
    data_cat = np.concatenate([b.data for b in blocks])

    # 需要检查是否在连接后的 np.int32 下索引/indptr 的任何值过大：
    # - indices 的最大值是输出数组的堆叠轴长度 - 1
    # - indptr 的最大值是非零条目的数量。这几乎不太可能需要 int64，但出于谨慎而进行检查。
    sum_dim = sum(b.shape[axis] for b in blocks)
    nnz = sum(len(b.indices) for b in blocks)
    idx_dtype = get_index_dtype(maxval=max(sum_dim - 1, nnz))
    stack_dim_cat = np.array([b.shape[axis] for b in blocks], dtype=idx_dtype)
    if data_cat.size > 0:
        indptr_cat = np.concatenate(indptr_list).astype(idx_dtype)
        indices_cat = (np.concatenate([b.indices for b in blocks])
                       .astype(idx_dtype))
        indptr = np.empty(constant_dim + 1, dtype=idx_dtype)
        indices = np.empty_like(indices_cat)
        data = np.empty_like(data_cat)
        # 调用 CSR 堆叠函数进行堆叠
        csr_hstack(n_blocks, constant_dim, stack_dim_cat,
                   indptr_cat, indices_cat, data_cat,
                   indptr, indices, data)
    else:
        # 如果没有数据，则创建空的数据结构
        indptr = np.zeros(constant_dim + 1, dtype=idx_dtype)
        indices = np.empty(0, dtype=idx_dtype)
        data = np.empty(0, dtype=data_cat.dtype)

    if axis == 0:
        # 如果是按列堆叠，则返回 CSC 格式的容器
        return blocks[0]._csc_container((data, indices, indptr),
                          shape=(sum_dim, constant_dim))
    else:
        # 如果是按行堆叠，则返回 CSR 格式的容器
        return blocks[0]._csr_container((data, indices, indptr),
                          shape=(constant_dim, sum_dim))
    # 将输入的 blocks 转换为 numpy 数组，元素类型为 'object'
    blocks = np.asarray(blocks, dtype='object')
    # 检查 blocks 中是否有任何元素是稀疏数组 (sparray)
    if any(isinstance(b, sparray) for b in blocks.flat):
        # 如果 blocks 中至少有一个稀疏数组，调用 _block 函数处理 blocks，并返回结果
        return _block([blocks], format, dtype)
    else:
        # 如果 blocks 中没有稀疏数组，调用 _block 函数处理 blocks，并指示返回稀疏矩阵格式
        return _block([blocks], format, dtype, return_spmatrix=True)
def vstack(blocks, format=None, dtype=None):
    """
    Stack sparse arrays vertically (row wise)

    Parameters
    ----------
    blocks
        sequence of sparse arrays with compatible shapes
    format : str, optional
        sparse format of the result (e.g., "csr")
        by default an appropriate sparse array format is returned.
        This choice is subject to change.
    dtype : dtype, optional
        The data-type of the output array. If not given, the dtype is
        determined from that of `blocks`.

    Returns
    -------
    new_array : sparse matrix or array
        If any block in blocks is a sparse array, return a sparse array.
        Otherwise return a sparse matrix.

        If you want a sparse array built from blocks that are not sparse
        arrays, use ``block(vstack(blocks))`` or convert one block
        e.g. `blocks[0] = csr_array(blocks[0])`.

    See Also
    --------
    hstack : stack sparse matrices horizontally (column wise)

    Examples
    --------
    >>> from scipy.sparse import coo_array, vstack
    >>> A = coo_array([[1, 2], [3, 4]])
    >>> B = coo_array([[5, 6]])
    >>> vstack([A, B]).toarray()
    array([[1, 2],
           [3, 4],
           [5, 6]])

    """
    # 将输入的 blocks 转换为 numpy 数组，数据类型为 'object'
    blocks = np.asarray(blocks, dtype='object')
    # 检查 blocks 中是否有任何一个元素是稀疏数组
    if any(isinstance(b, sparray) for b in blocks.flat):
        # 如果有稀疏数组，则调用 _block 函数处理，返回稀疏数组或矩阵
        return _block([[b] for b in blocks], format, dtype)
    else:
        # 如果没有稀疏数组，则同样调用 _block 函数，但指定返回为稀疏矩阵
        return _block([[b] for b in blocks], format, dtype, return_spmatrix=True)


def bmat(blocks, format=None, dtype=None):
    """
    Build a sparse array or matrix from sparse sub-blocks

    Note: `block_array` is preferred over `bmat`. They are the same function
    except that `bmat` can return a deprecated sparse matrix.
    `bmat` returns a coo_matrix if none of the inputs are a sparse array.

    .. warning::

        This function returns a sparse matrix -- not a sparse array.
        You are encouraged to use ``block_array`` to take advantage
        of the sparse array functionality.

    Parameters
    ----------
    blocks : array_like
        Grid of sparse matrices with compatible shapes.
        An entry of None implies an all-zero matrix.
    format : {'bsr', 'coo', 'csc', 'csr', 'dia', 'dok', 'lil'}, optional
        The sparse format of the result (e.g. "csr"). By default an
        appropriate sparse matrix format is returned.
        This choice is subject to change.
    dtype : dtype, optional
        The data-type of the output matrix. If not given, the dtype is
        determined from that of `blocks`.

    Returns
    -------
    bmat : sparse matrix or array
        If any block in blocks is a sparse array, return a sparse array.
        Otherwise return a sparse matrix.

        If you want a sparse array built from blocks that are not sparse
        arrays, use ``block_array()``.

    See Also
    --------
    block_array

    Examples
    --------
    >>> from scipy.sparse import coo_array, bmat
    """
    """
    将输入的块数组转换为 numpy 数组对象，元素类型为 'object'
    blocks = np.asarray(blocks, dtype='object')
    
    如果 blocks 中的任何元素是 sparray 类型的对象，则调用 _block 函数，并返回其结果。
    否则，调用 _block 函数，传递 format 和 dtype 参数，并设置 return_spmatrix=True。
    
    返回最终的处理结果。
    """
    blocks = np.asarray(blocks, dtype='object')
    if any(isinstance(b, sparray) for b in blocks.flat):
        return _block(blocks, format, dtype)
    else:
        return _block(blocks, format, dtype, return_spmatrix=True)
def block_array(blocks, *, format=None, dtype=None):
    """
    Build a sparse array from sparse sub-blocks

    Parameters
    ----------
    blocks : array_like
        Grid of sparse arrays with compatible shapes.
        An entry of None implies an all-zero array.
    format : {'bsr', 'coo', 'csc', 'csr', 'dia', 'dok', 'lil'}, optional
        The sparse format of the result (e.g. "csr"). By default an
        appropriate sparse array format is returned.
        This choice is subject to change.
    dtype : dtype, optional
        The data-type of the output array. If not given, the dtype is
        determined from that of `blocks`.

    Returns
    -------
    block : sparse array

    See Also
    --------
    block_diag : specify blocks along the main diagonals
    diags : specify (possibly offset) diagonals

    Examples
    --------
    >>> from scipy.sparse import coo_array, block_array
    >>> A = coo_array([[1, 2], [3, 4]])
    >>> B = coo_array([[5], [6]])
    >>> C = coo_array([[7]])
    >>> block_array([[A, B], [None, C]]).toarray()
    array([[1, 2, 5],
           [3, 4, 6],
           [0, 0, 7]])

    >>> block_array([[A, None], [None, C]]).toarray()
    array([[1, 2, 0],
           [3, 4, 0],
           [0, 0, 7]])

    """
    return _block(blocks, format, dtype)


def _block(blocks, format, dtype, return_spmatrix=False):
    blocks = np.asarray(blocks, dtype='object')  # Convert blocks to a NumPy array of objects

    if blocks.ndim != 2:
        raise ValueError('blocks must be 2-D')  # Raise an error if blocks is not 2-D

    M,N = blocks.shape  # Get dimensions M (rows) and N (columns) of blocks

    # check for fast path cases based on format
    if (format in (None, 'csr') and
        all(issparse(b) and b.format == 'csr' for b in blocks.flat)
    ):
        if N > 1:
            # stack along columns (axis 1): must have shape (M, 1)
            blocks = [[_stack_along_minor_axis(blocks[b, :], 1)] for b in range(M)]  # Stack blocks along axis 1
            blocks = np.asarray(blocks, dtype='object')

        # stack along rows (axis 0):
        A = _compressed_sparse_stack(blocks[:, 0], 0, return_spmatrix)  # Stack blocks along axis 0
        if dtype is not None:
            A = A.astype(dtype)  # Convert the stacked array to specified dtype if given
        return A
    elif (format in (None, 'csc') and
          all(issparse(b) and b.format == 'csc' for b in blocks.flat)
    ):
        if M > 1:
            # stack along rows (axis 0): must have shape (1, N)
            blocks = [[_stack_along_minor_axis(blocks[:, b], 0) for b in range(N)]]  # Stack blocks along axis 0
            blocks = np.asarray(blocks, dtype='object')

        # stack along columns (axis 1):
        A = _compressed_sparse_stack(blocks[0, :], 1, return_spmatrix)  # Stack blocks along axis 1
        if dtype is not None:
            A = A.astype(dtype)  # Convert the stacked array to specified dtype if given
        return A

    block_mask = np.zeros(blocks.shape, dtype=bool)  # Initialize a boolean mask array with zeros
    brow_lengths = np.zeros(M, dtype=np.int64)  # Initialize an array to store row lengths
    bcol_lengths = np.zeros(N, dtype=np.int64)  # Initialize an array to store column lengths

    # convert everything to COO format
    # 遍历每个块的行索引 i 和列索引 j
    for i in range(M):
        for j in range(N):
            # 检查块是否存在
            if blocks[i,j] is not None:
                # 将块转换为 COO 格式的稀疏数组
                A = coo_array(blocks[i,j])
                blocks[i,j] = A  # 更新块为 COO 格式的稀疏数组
                block_mask[i,j] = True  # 标记块为存在

                # 检查当前行的长度信息
                if brow_lengths[i] == 0:
                    brow_lengths[i] = A.shape[0]
                elif brow_lengths[i] != A.shape[0]:
                    # 抛出维度不匹配的错误信息
                    msg = (f'blocks[{i},:] has incompatible row dimensions. '
                           f'Got blocks[{i},{j}].shape[0] == {A.shape[0]}, '
                           f'expected {brow_lengths[i]}.')
                    raise ValueError(msg)

                # 检查当前列的长度信息
                if bcol_lengths[j] == 0:
                    bcol_lengths[j] = A.shape[1]
                elif bcol_lengths[j] != A.shape[1]:
                    # 抛出维度不匹配的错误信息
                    msg = (f'blocks[:,{j}] has incompatible column '
                           f'dimensions. '
                           f'Got blocks[{i},{j}].shape[1] == {A.shape[1]}, '
                           f'expected {bcol_lengths[j]}.')
                    raise ValueError(msg)

    # 计算所有标记块的总非零元素数目
    nnz = sum(block.nnz for block in blocks[block_mask])

    # 如果 dtype 未指定，则从所有块中推断出 dtype
    if dtype is None:
        all_dtypes = [blk.dtype for blk in blocks[block_mask]]
        dtype = upcast(*all_dtypes) if all_dtypes else None

    # 计算行和列的偏移量
    row_offsets = np.append(0, np.cumsum(brow_lengths))
    col_offsets = np.append(0, np.cumsum(bcol_lengths))

    # 计算结果矩阵的形状
    shape = (row_offsets[-1], col_offsets[-1])

    # 创建用于存储数据、行索引和列索引的数组
    data = np.empty(nnz, dtype=dtype)
    idx_dtype = get_index_dtype(maxval=max(shape))
    row = np.empty(nnz, dtype=idx_dtype)
    col = np.empty(nnz, dtype=idx_dtype)

    # 重新初始化 nnz 变量，用于迭代计算非零元素
    nnz = 0

    # 遍历所有标记的块，获取它们的数据、行索引和列索引
    ii, jj = np.nonzero(block_mask)
    for i, j in zip(ii, jj):
        B = blocks[i, j]
        idx = slice(nnz, nnz + B.nnz)
        data[idx] = B.data
        # 将 B 的行索引加上行偏移量，存储到 row 数组中
        np.add(B.row, row_offsets[i], out=row[idx], dtype=idx_dtype)
        # 将 B 的列索引加上列偏移量，存储到 col 数组中
        np.add(B.col, col_offsets[j], out=col[idx], dtype=idx_dtype)
        nnz += B.nnz

    # 如果指定返回稀疏矩阵对象，则返回 COO 格式的稀疏矩阵
    if return_spmatrix:
        return coo_matrix((data, (row, col)), shape=shape).asformat(format)
    # 否则，返回 COO 格式的稀疏数组对象
    return coo_array((data, (row, col)), shape=shape).asformat(format)
# 从提供的矩阵构建一个块对角稀疏矩阵或数组

def block_diag(mats, format=None, dtype=None):
    """
    Build a block diagonal sparse matrix or array from provided matrices.

    Parameters
    ----------
    mats : sequence of matrices or arrays
        输入的矩阵或数组。
    format : str, optional
        结果的稀疏格式（例如 "csr"）。如果未提供，则以 "coo" 格式返回结果。
    dtype : dtype specifier, optional
        输出的数据类型。如果未提供，则从 `blocks` 的数据类型确定。

    Returns
    -------
    res : sparse matrix or array
        如果至少一个输入是稀疏数组，则输出为稀疏数组。否则输出为稀疏矩阵。

    Notes
    -----
    .. versionadded:: 0.11.0

    See Also
    --------
    block_array
    diags_array

    Examples
    --------
    >>> from scipy.sparse import coo_array, block_diag
    >>> A = coo_array([[1, 2], [3, 4]])
    >>> B = coo_array([[5], [6]])
    >>> C = coo_array([[7]])
    >>> block_diag((A, B, C)).toarray()
    array([[1, 2, 0, 0],
           [3, 4, 0, 0],
           [0, 0, 5, 0],
           [0, 0, 6, 0],
           [0, 0, 0, 7]])

    """
    # 检查输入的矩阵或数组是否有稀疏数组
    if any(isinstance(a, sparray) for a in mats):
        container = coo_array  # 如果有稀疏数组，使用 COO 格式存储结果
    else:
        container = coo_matrix  # 如果没有稀疏数组，使用 COO 格式存储结果

    row = []  # 存储每个块的行索引
    col = []  # 存储每个块的列索引
    data = []  # 存储每个块的数据
    r_idx = 0  # 行索引的起始位置
    c_idx = 0  # 列索引的起始位置

    # 遍历每个输入的矩阵或数组
    for a in mats:
        if isinstance(a, (list, numbers.Number)):
            a = coo_array(np.atleast_2d(a))  # 如果是列表或数字，转换成 COO 格式
        if issparse(a):
            a = a.tocoo()  # 如果是稀疏数组，转换成 COO 格式
            nrows, ncols = a._shape_as_2d
            row.append(a.row + r_idx)  # 更新行索引并添加到 row 列表中
            col.append(a.col + c_idx)  # 更新列索引并添加到 col 列表中
            data.append(a.data)  # 添加数据到 data 列表中
        else:
            nrows, ncols = a.shape
            a_row, a_col = np.divmod(np.arange(nrows*ncols), ncols)
            row.append(a_row + r_idx)  # 更新行索引并添加到 row 列表中
            col.append(a_col + c_idx)  # 更新列索引并添加到 col 列表中
            data.append(a.ravel())  # 添加数据到 data 列表中
        r_idx += nrows  # 更新行索引起始位置
        c_idx += ncols  # 更新列索引起始位置

    # 将所有块的行索引、列索引和数据连接起来，构建结果的数据结构
    row = np.concatenate(row)
    col = np.concatenate(col)
    data = np.concatenate(data)

    # 使用 container 构造函数创建稀疏矩阵或数组，并以指定格式返回
    return container((data, (row, col)),
                     shape=(r_idx, c_idx),
                     dtype=dtype).asformat(format)
    # density: 实数，可选（默认值为 0.01）
    # 生成矩阵的密度：密度为 1 表示一个完全矩阵，密度为 0 表示一个没有非零元素的矩阵。
    format : str, optional (default: 'coo')
    # 稀疏矩阵的格式。
    dtype : dtype, optional (default: np.float64)
    # 返回矩阵值的类型。
    random_state : {None, int, `Generator`, `RandomState`}, optional
    # 用于确定非零结构的随机数生成器。建议每次调用手动提供 `numpy.random.Generator`，因为这比 RandomState 快得多。
    #
    # - 如果 `None`（或 `np.random`），则使用 `numpy.random.RandomState` 单例。
    # - 如果是一个整数，则使用一个新的 ``Generator`` 实例，并用该整数种子初始化。
    # - 如果是 ``Generator`` 或 ``RandomState`` 实例，则使用该实例。
    #
    # 此随机状态将用于采样 `indices`（稀疏结构），默认也用于数据值（参见 `data_sampler`）。
    data_sampler : callable, optional (default depends on dtype)
    # 随机数据值的采样器，关键字参数 `size`。此函数应该接受一个关键字参数 `size`，指定其返回的 ndarray 的长度。
    # 在选择这些值的位置后，它用于生成矩阵中的非零值。
    # 默认情况下，除非 `dtype` 是整数（默认情况下是该dtype的均匀整数）或复数（默认情况下是复平面上的单位正方形的均匀分布），
    # 否则使用均匀 [0, 1) 的随机值。对于这些情况，使用 `random_state` rng，例如 ``rng.uniform(size=size)``。
    
    Returns
    -------
    res : sparse array
    # 返回稀疏数组。

Examples
--------
# 示例

Passing a ``np.random.Generator`` instance for better performance:
# 使用 `np.random.Generator` 实例以获得更好的性能：

>>> import numpy as np
>>> import scipy as sp
>>> rng = np.random.default_rng()

Default sampling uniformly from [0, 1):
# 默认从 [0, 1) 均匀采样：

>>> S = sp.sparse.random_array((3, 4), density=0.25, random_state=rng)

Providing a sampler for the values:
# 为值提供一个采样器：

>>> rvs = sp.stats.poisson(25, loc=10).rvs
>>> S = sp.sparse.random_array((3, 4), density=0.25,
...                            random_state=rng, data_sampler=rvs)
>>> S.toarray()
array([[ 36.,   0.,  33.,   0.],   # random
       [  0.,   0.,   0.,   0.],
       [  0.,   0.,  36.,   0.]])

Building a custom distribution.
This example builds a squared normal from np.random:
# 构建自定义分布。此示例从 np.random 构建一个平方正态分布：

>>> def np_normal_squared(size=None, random_state=rng):
...     return random_state.standard_normal(size) ** 2
>>> S = sp.sparse.random_array((3, 4), density=0.25, random_state=rng,
...                      data_sampler=np_normal_squared)

Or we can build it from sp.stats style rvs functions:
# 或者我们可以从 sp.stats 风格的 rvs 函数构建它：

>>> def sp_stats_normal_squared(size=None, random_state=rng):
...     std_normal = sp.stats.distributions.norm_gen().rvs
    # 如果未指定随机数生成器，则使用更高效的默认随机数生成器 np.random.default_rng()
    if random_state is None:
        random_state = np.random.default_rng()
    
    # 使用给定的参数调用 _random 函数，获取稀疏矩阵的数据和索引
    data, ind = _random(shape, density, format, dtype, random_state, data_sampler)
    
    # 将获取到的数据和索引作为参数创建一个 COO 格式的稀疏矩阵
    # 并将其转换为指定的格式
    return coo_array((data, ind), shape=shape).asformat(format)
# 定义一个生成稀疏矩阵的函数，矩阵形状由参数 m 和 n 指定，稀疏程度由 density 参数指定，默认为 0.01
def random(m, n, density=0.01, format='coo', dtype=None,
           random_state=None, data_rvs=None):
    """Generate a sparse matrix of the given shape and density with randomly
    distributed values.

    .. warning::

        Since numpy 1.17, passing a ``np.random.Generator`` (e.g.
        ``np.random.default_rng``) for ``random_state`` will lead to much
        faster execution times.

        A much slower implementation is used by default for backwards
        compatibility.

        从 numpy 1.17 起，如果使用 ``np.random.Generator``（例如 ``np.random.default_rng``）作为 ``random_state`` 参数，将会显著提升执行速度。

        为了向后兼容，默认情况下会使用较慢的实现方式。

    .. warning::

        This function returns a sparse matrix -- not a sparse array.
        You are encouraged to use ``random_array`` to take advantage of the
        sparse array functionality.

        此函数返回稀疏矩阵，而不是稀疏数组。
        建议您使用 ``random_array`` 函数，以充分利用稀疏数组功能。

    Parameters
    ----------
    m, n : int
        shape of the matrix
        矩阵的形状
    density : real, optional
        density of the generated matrix: density equal to one means a full
        matrix, density of 0 means a matrix with no non-zero items.
        生成矩阵的密度：密度为1表示一个全满的矩阵，密度为0表示一个没有非零项的矩阵。
    format : str, optional
        sparse matrix format.
        稀疏矩阵的格式。
    dtype : dtype, optional
        type of the returned matrix values.
        返回矩阵值的数据类型。
    random_state : {None, int, `numpy.random.Generator`,
                    `numpy.random.RandomState`}, optional
    # 参数 `random_state` 控制随机数生成器的状态，可以是 None、整数、`numpy.random.Generator` 或 `numpy.random.RandomState`

        - If `seed` is None (or `np.random`), the `numpy.random.RandomState`
          singleton is used.
        # 如果 `seed` 是 None 或 `np.random`，则使用 `numpy.random.RandomState` 的单例对象

        - If `seed` is an int, a new ``RandomState`` instance is used,
          seeded with `seed`.
        # 如果 `seed` 是一个整数，则创建一个新的 `RandomState` 实例，并使用 `seed` 进行初始化

        - If `seed` is already a ``Generator`` or ``RandomState`` instance then
          that instance is used.
        # 如果 `seed` 已经是一个 `Generator` 或 `RandomState` 实例，则直接使用该实例

        This random state will be used for sampling the sparsity structure, but
        not necessarily for sampling the values of the structurally nonzero
        entries of the matrix.
        # 这个随机状态将用于采样稀疏矩阵的稀疏结构，但不一定用于采样矩阵中结构上非零的条目的值

    data_rvs : callable, optional
    # 参数 `data_rvs` 是一个可调用对象，用于生成随机值的样本

        Samples a requested number of random values.
        # 这个函数应该接受一个参数，指定返回的 ndarray 的长度，用于生成随机值的样本

        This function should take a single argument specifying the length
        of the ndarray that it will return. The structurally nonzero entries
        of the sparse random matrix will be taken from the array sampled
        by this function. By default, uniform [0, 1) random values will be
        sampled using the same random state as is used for sampling
        the sparsity structure.
        # 稀疏随机矩阵中的结构上非零条目将从由此函数生成的数组中取出。默认情况下，将使用与采样稀疏结构相同的随机状态来采样均匀分布的 [0, 1) 随机值

    Returns
    -------
    res : sparse matrix
    # 返回值 `res` 是一个稀疏矩阵对象

    See Also
    --------
    random_array : constructs sparse arrays instead of sparse matrices
    # 参见 `random_array` 函数，用于构造稀疏数组而非稀疏矩阵

    Examples
    --------

    Passing a ``np.random.Generator`` instance for better performance:

    >>> import scipy as sp
    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> S = sp.sparse.random(3, 4, density=0.25, random_state=rng)
    # 使用 `np.random.Generator` 实例 `rng` 以提升性能生成稀疏矩阵

    Providing a sampler for the values:

    >>> rvs = sp.stats.poisson(25, loc=10).rvs
    >>> S = sp.sparse.random(3, 4, density=0.25, random_state=rng, data_rvs=rvs)
    >>> S.toarray()
    array([[ 36.,   0.,  33.,   0.],   # random
           [  0.,   0.,   0.,   0.],
           [  0.,   0.,  36.,   0.]])
    # 使用自定义的值采样器 `rvs` 生成稀疏矩阵，并将其转换为数组显示

    Building a custom distribution.
    This example builds a squared normal from np.random:

    >>> def np_normal_squared(size=None, random_state=rng):
    ...     return random_state.standard_normal(size) ** 2
    >>> S = sp.sparse.random(3, 4, density=0.25, random_state=rng,
    ...                      data_rvs=np_normal_squared)
    # 构建一个基于 `np.random` 的平方正态分布的稀疏矩阵示例

    Or we can build it from sp.stats style rvs functions:

    >>> def sp_stats_normal_squared(size=None, random_state=rng):
    ...     std_normal = sp.stats.distributions.norm_gen().rvs
    ...     return std_normal(size=size, random_state=random_state) ** 2
    >>> S = sp.sparse.random(3, 4, density=0.25, random_state=rng,
    ...                      data_rvs=sp_stats_normal_squared)
    # 或者使用 `sp.stats` 风格的随机变量采样函数构建稀疏矩阵

    Or we can subclass sp.stats rv_continuous or rv_discrete:

    >>> class NormalSquared(sp.stats.rv_continuous):
    ...     def _rvs(self,  size=None, random_state=rng):
    ...         return random_state.standard_normal(size) ** 2
    >>> X = NormalSquared()
    >>> Y = X()  # get a frozen version of the distribution
    >>> S = sp.sparse.random(3, 4, density=0.25, random_state=rng, data_rvs=Y.rvs)
    # 或者可以从 `sp.stats` 的 `rv_continuous` 或 `rv_discrete` 子类中构建稀疏矩阵
    # 如果 n 为 None，则将 n 设为 m
    if n is None:
        n = m
    # 将 m 和 n 转换为整数类型
    m, n = int(m), int(n)
    # 创建一个函数 data_rvs_kw 用于处理 data_rvs 的关键字参数，例如 data_rvs(size=7)
    # 如果 data_rvs 不为 None，则定义 data_rvs_kw 函数来调用 data_rvs
    if data_rvs is not None:
        def data_rvs_kw(size):
            return data_rvs(size)
    else:
        # 否则将 data_rvs_kw 设为 None
        data_rvs_kw = None
    # 调用 _random 函数生成随机值 vals 和对应的索引 ind
    vals, ind = _random((m, n), density, format, dtype, random_state, data_rvs_kw)
    # 将 vals 和 ind 转换为 COO 格式的稀疏矩阵，并按照指定格式进行格式化
    return coo_matrix((vals, ind), shape=(m, n)).asformat(format)
# 生成给定形状和密度的稀疏矩阵，其中值均匀分布在 [0, 1) 区间内
def rand(m, n, density=0.01, format="coo", dtype=None, random_state=None):
    """Generate a sparse matrix of the given shape and density with uniformly
    distributed values.

    .. warning::

        This function returns a sparse matrix -- not a sparse array.
        You are encouraged to use ``random_array`` to take advantage
        of the sparse array functionality.

    Parameters
    ----------
    m, n : int
        shape of the matrix
    density : real, optional
        density of the generated matrix: density equal to one means a full
        matrix, density of 0 means a matrix with no non-zero items.
    format : str, optional
        sparse matrix format.
    dtype : dtype, optional
        type of the returned matrix values.
    random_state : {None, int, `numpy.random.Generator`,
                    `numpy.random.RandomState`}, optional

        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

    Returns
    -------
    res : sparse matrix

    Notes
    -----
    Only float types are supported for now.

    See Also
    --------
    random : Similar function allowing a custom random data sampler
    random_array : Similar to random() but returns a sparse array

    Examples
    --------
    >>> from scipy.sparse import rand
    >>> matrix = rand(3, 4, density=0.25, format="csr", random_state=42)
    >>> matrix
    <Compressed Sparse Row sparse matrix of dtype 'float64'
        with 3 stored elements and shape (3, 4)>
    >>> matrix.toarray()
    array([[0.05641158, 0.        , 0.        , 0.65088847],  # random
           [0.        , 0.        , 0.        , 0.14286682],
           [0.        , 0.        , 0.        , 0.        ]])

    """
    # 调用 random 函数生成稀疏矩阵，并返回结果
    return random(m, n, density, format, dtype, random_state)
```