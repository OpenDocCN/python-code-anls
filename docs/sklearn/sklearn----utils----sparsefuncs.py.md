# `D:\src\scipysrc\scikit-learn\sklearn\utils\sparsefuncs.py`

```
"""A collection of utilities to work with sparse matrices and arrays."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 导入必要的库
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

# 从本地导入修复和验证函数
from ..utils.fixes import _sparse_min_max, _sparse_nan_min_max
from ..utils.validation import _check_sample_weight

# 从本地快速稀疏函数中导入特定函数
from .sparsefuncs_fast import (
    csc_mean_variance_axis0 as _csc_mean_var_axis0,
)
from .sparsefuncs_fast import (
    csr_mean_variance_axis0 as _csr_mean_var_axis0,
)
from .sparsefuncs_fast import (
    incr_mean_variance_axis0 as _incr_mean_var_axis0,
)


def _raise_typeerror(X):
    """Raises a TypeError if X is not a CSR or CSC matrix"""
    # 检查输入 X 是否为稀疏矩阵，如果不是则引发 TypeError
    input_type = X.format if sp.issparse(X) else type(X)
    err = "Expected a CSR or CSC sparse matrix, got %s." % input_type
    raise TypeError(err)


def _raise_error_wrong_axis(axis):
    """Raises a ValueError if axis is not 0 or 1"""
    # 检查轴向参数是否为 0 或 1，否则引发 ValueError
    if axis not in (0, 1):
        raise ValueError(
            "Unknown axis value: %d. Use 0 for rows, or 1 for columns" % axis
        )


def inplace_csr_column_scale(X, scale):
    """Inplace column scaling of a CSR matrix.

    Scale each feature of the data matrix by multiplying with specific scale
    provided by the caller assuming a (n_samples, n_features) shape.

    Parameters
    ----------
    X : sparse matrix of shape (n_samples, n_features)
        Matrix to normalize using the variance of the features.
        It should be of CSR format.

    scale : ndarray of shape (n_features,), dtype={np.float32, np.float64}
        Array of precomputed feature-wise values to use for scaling.

    Examples
    --------
    >>> from sklearn.utils import sparsefuncs
    >>> from scipy import sparse
    >>> import numpy as np
    >>> indptr = np.array([0, 3, 4, 4, 4])
    >>> indices = np.array([0, 1, 2, 2])
    >>> data = np.array([8, 1, 2, 5])
    >>> scale = np.array([2, 3, 2])
    >>> csr = sparse.csr_matrix((data, indices, indptr))
    >>> csr.todense()
    matrix([[8, 1, 2],
            [0, 0, 5],
            [0, 0, 0],
            [0, 0, 0]])
    >>> sparsefuncs.inplace_csr_column_scale(csr, scale)
    >>> csr.todense()
    matrix([[16,  3,  4],
            [ 0,  0, 10],
            [ 0,  0,  0],
            [ 0,  0,  0]])
    """
    # 确保 scale 数组的长度与 X 的列数相匹配
    assert scale.shape[0] == X.shape[1]
    # 在 CSR 矩阵的列数据上进行原地缩放
    X.data *= scale.take(X.indices, mode="clip")


def inplace_csr_row_scale(X, scale):
    """Inplace row scaling of a CSR matrix.

    Scale each sample of the data matrix by multiplying with specific scale
    provided by the caller assuming a (n_samples, n_features) shape.

    Parameters
    ----------
    X : sparse matrix of shape (n_samples, n_features)
        Matrix to be scaled. It should be of CSR format.

    scale : ndarray of float of shape (n_samples,)
        Array of precomputed sample-wise values to use for scaling.
    """
    # 确保 scale 数组的长度与 X 的行数相匹配
    assert scale.shape[0] == X.shape[0]
    # 在 CSR 矩阵的行数据上进行原地缩放
    X.data *= np.repeat(scale, np.diff(X.indptr))
    # 引发错误，如果轴参数不是0或1
    _raise_error_wrong_axis(axis)

    # 检查输入矩阵是否稀疏矩阵且格式为CSR或CSC
    if sp.issparse(X) and X.format == "csr":
        # 如果计算轴为0，调用CSR格式的均值和方差计算函数
        if axis == 0:
            return _csr_mean_var_axis0(
                X, weights=weights, return_sum_weights=return_sum_weights
            )
        else:
            # 如果计算轴为1，将矩阵转置后调用CSC格式的均值和方差计算函数
            return _csc_mean_var_axis0(
                X.T, weights=weights, return_sum_weights=return_sum_weights
            )
    elif sp.issparse(X) and X.format == "csc":
        # 如果输入矩阵格式为CSC，根据轴参数调用相应的均值和方差计算函数
        if axis == 0:
            return _csc_mean_var_axis0(
                X, weights=weights, return_sum_weights=return_sum_weights
            )
        else:
            # 如果计算轴为1，将矩阵转置后调用CSR格式的均值和方差计算函数
            return _csr_mean_var_axis0(
                X.T, weights=weights, return_sum_weights=return_sum_weights
            )
    else:
        # 如果输入不是稀疏矩阵或格式不是CSR或CSC，引发类型错误
        _raise_typeerror(X)



def incr_mean_variance_axis(X, *, axis, last_mean, last_var, last_n, weights=None):
    """Compute incremental mean and variance along an axis on a CSR or CSC matrix.

    last_mean, last_var are the statistics computed at the last step by this
    function. Both must be initialized to 0-arrays of the proper size, i.e.
    the number of features in X. last_n is the number of samples encountered
    until now.

    Parameters
    ----------
    X : sparse matrix of shape (n_samples, n_features)
        Input data. It can be of CSR or CSC format.

    axis : {0, 1}
        Axis along which the axis should be computed.

    last_mean : ndarray of shape (n_features,), dtype=floating
        Array containing the previous mean values computed.

    last_var : ndarray of shape (n_features,), dtype=floating
        Array containing the previous variance values computed.

    last_n : int
        Number of samples encountered until the last computation.

    weights : ndarray of shape (n_samples,) or (n_features,), default=None
        If axis is set to 0 shape is (n_samples,) or
        if axis is set to 1 shape is (n_features,).
        If it is set to None, then samples are equally weighted.

    Returns
    -------
    current_mean : ndarray of shape (n_features,), dtype=floating
        Incremental mean computed along the specified axis.

    current_var : ndarray of shape (n_features,), dtype=floating
        Incremental variance computed along the specified axis.
    """
    # 函数体未完，继续添加注释
    # 检查轴参数是否有效，如果无效则引发错误
    _raise_error_wrong_axis(axis)
    
    # 检查输入数据 X 是否为稀疏矩阵，并且格式为 CSR 或 CSC，如果不是则引发类型错误
    if not (sp.issparse(X) and X.format in ("csc", "csr")):
        _raise_typeerror(X)
    
    # 如果 last_n 的大小为 1，则将其扩展为与 last_mean 相同形状的 ndarray，
    # 使用相同的数据类型作为 last_mean 的数据类型
    if np.size(last_n) == 1:
        last_n = np.full(last_mean.shape, last_n, dtype=last_mean.dtype)
    # 检查 last_mean, last_var, last_n 是否具有相同的大小
    if not (np.size(last_mean) == np.size(last_var) == np.size(last_n)):
        # 如果它们的大小不相同，引发值错误异常
        raise ValueError("last_mean, last_var, last_n do not have the same shapes.")

    # 如果 axis 等于 1
    if axis == 1:
        # 检查 last_mean 的大小是否与 X 的行数相同
        if np.size(last_mean) != X.shape[0]:
            # 如果不同，引发值错误异常，显示预期的大小和实际的大小
            raise ValueError(
                "If axis=1, then last_mean, last_n, last_var should be of "
                f"size n_samples {X.shape[0]} (Got {np.size(last_mean)})."
            )
    else:  # 如果 axis 等于 0
        # 检查 last_mean 的大小是否与 X 的列数相同
        if np.size(last_mean) != X.shape[1]:
            # 如果不同，引发值错误异常，显示预期的大小和实际的大小
            raise ValueError(
                "If axis=0, then last_mean, last_n, last_var should be of "
                f"size n_features {X.shape[1]} (Got {np.size(last_mean)})."
            )

    # 如果 axis 等于 1，则转置 X（交换行列），以便之后操作在列上进行
    X = X.T if axis == 1 else X

    # 如果给定了权重 weights，则调用 _check_sample_weight 函数检查并返回合法的权重数组
    if weights is not None:
        weights = _check_sample_weight(weights, X, dtype=X.dtype)

    # 调用 _incr_mean_var_axis0 函数，传入参数 X, last_mean, last_var, last_n, weights，
    # 返回计算后的新的均值和方差信息，具体计算方式依赖于 axis 的值
    return _incr_mean_var_axis0(
        X, last_mean=last_mean, last_var=last_var, last_n=last_n, weights=weights
    )
# 原地操作 CSC 格式稀疏矩阵中的列缩放

"""Inplace column scaling of a CSC/CSR matrix.

Scale each feature of the data matrix by multiplying with specific scale
provided by the caller assuming a (n_samples, n_features) shape.

Parameters
----------
X : sparse matrix of shape (n_samples, n_features)
    Matrix to normalize using the variance of the features. It should be
    of CSC or CSR format.

scale : ndarray of shape (n_features,), dtype={np.float32, np.float64}
    Array of precomputed feature-wise values to use for scaling.

Examples
--------
>>> from sklearn.utils import sparsefuncs
>>> from scipy import sparse
>>> import numpy as np
>>> indptr = np.array([0, 3, 4, 4, 4])
>>> indices = np.array([0, 1, 2, 2])
>>> data = np.array([8, 1, 2, 5])
>>> scale = np.array([2, 3, 2])
>>> csr = sparse.csr_matrix((data, indices, indptr))
>>> csr.todense()
matrix([[8, 1, 2],
        [0, 0, 5],
        [0, 0, 0],
        [0, 0, 0]])
>>> sparsefuncs.inplace_column_scale(csr, scale)
>>> csr.todense()
matrix([[16,  3,  4],
        [ 0,  0, 10],
        [ 0,  0,  0],
        [ 0,  0,  0]])
"""
if sp.issparse(X) and X.format == "csc":
    inplace_csr_row_scale(X.T, scale)  # 如果 X 是 CSC 格式，则转置后进行行缩放操作
elif sp.issparse(X) and X.format == "csr":
    inplace_csr_column_scale(X, scale)  # 如果 X 是 CSR 格式，则直接进行列缩放操作
else:
    _raise_typeerror(X)


def inplace_row_scale(X, scale):
    """Inplace row scaling of a CSR or CSC matrix.

    Scale each row of the data matrix by multiplying with specific scale
    provided by the caller assuming a (n_samples, n_features) shape.

    Parameters
    ----------
    X : sparse matrix of shape (n_samples, n_features)
        Matrix to be scaled. It should be of CSR or CSC format.

    scale : ndarray of shape (n_features,), dtype={np.float32, np.float64}
        Array of precomputed sample-wise values to use for scaling.

    Examples
    --------
    >>> from sklearn.utils import sparsefuncs
    >>> from scipy import sparse
    >>> import numpy as np
    >>> indptr = np.array([0, 2, 3, 4, 5])
    >>> indices = np.array([0, 1, 2, 3, 3])
    >>> data = np.array([8, 1, 2, 5, 6])
    >>> scale = np.array([2, 3, 4, 5])
    >>> csr = sparse.csr_matrix((data, indices, indptr))
    >>> csr.todense()
    matrix([[8, 1, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 5],
            [0, 0, 0, 6]])
    >>> sparsefuncs.inplace_row_scale(csr, scale)
    >>> csr.todense()
    matrix([[16,  2,  0,  0],
            [ 0,  0,  6,  0],
            [ 0,  0,  0, 20],
            [ 0,  0,  0, 30]])
    """
    if sp.issparse(X) and X.format == "csc":
        inplace_csr_column_scale(X.T, scale)  # 如果 X 是 CSC 格式，则转置后进行列缩放操作
    elif sp.issparse(X) and X.format == "csr":
        inplace_csr_row_scale(X, scale)  # 如果 X 是 CSR 格式，则直接进行行缩放操作
    else:
        _raise_typeerror(X)


def inplace_swap_row_csc(X, m, n):
    """Swap two rows of a CSC matrix in-place.

    Parameters
    ----------
    X : sparse matrix of shape (n_samples, n_features)
        CSC format matrix where rows need to be swapped.

    m, n : int
        Row indices to be swapped.

    Notes
    -----
    This function modifies the input matrix X in-place.
    """
    X : sparse matrix of shape (n_samples, n_features)
        Matrix whose two rows are to be swapped. It should be of
        CSC format.

    m : int
        Index of the row of X to be swapped.

    n : int
        Index of the row of X to be swapped.
    """
    # 检查 m 和 n 是否为 numpy 数组，若是则抛出类型错误
    for t in [m, n]:
        if isinstance(t, np.ndarray):
            raise TypeError("m and n should be valid integers")

    # 如果 m 或 n 是负数，则转换为对应的正数索引
    if m < 0:
        m += X.shape[0]
    if n < 0:
        n += X.shape[0]

    # 创建一个布尔掩码，找出所有与 m 对应的非零元素的位置
    m_mask = X.indices == m
    # 将所有与 n 对应的非零元素的位置改为 m
    X.indices[X.indices == n] = m
    # 将之前 m 对应位置的元素改为 n
    X.indices[m_mask] = n
def inplace_swap_column(X, m, n):
    """
    Swap two columns of a CSC/CSR matrix in-place.

    Parameters
    ----------
    X : sparse matrix of shape (n_samples, n_features)
        Matrix whose two columns are to be swapped. It should be of
        CSR or CSC format.

    m : int
        Index of the column of X to be swapped.

    n : int
        Index of the column of X to be swapped.
    """
    """
    Swap columns m and n in a sparse matrix X in-place.

    Parameters
    ----------
    m : int
        Index of the column of X to be swapped.

    n : int
        Index of the column of X to be swapped.

    Examples
    --------
    >>> from sklearn.utils import sparsefuncs
    >>> from scipy import sparse
    >>> import numpy as np
    >>> indptr = np.array([0, 2, 3, 3, 3])
    >>> indices = np.array([0, 2, 2])
    >>> data = np.array([8, 2, 5])
    >>> csr = sparse.csr_matrix((data, indices, indptr))
    >>> csr.todense()
    matrix([[8, 0, 2],
            [0, 0, 5],
            [0, 0, 0],
            [0, 0, 0]])
    >>> sparsefuncs.inplace_swap_column(csr, 0, 1)
    >>> csr.todense()
    matrix([[0, 8, 2],
            [0, 0, 5],
            [0, 0, 0],
            [0, 0, 0]])
    """
    # 如果 m 小于 0，则将其转换为正数索引
    if m < 0:
        m += X.shape[1]
    # 如果 n 小于 0，则将其转换为正数索引
    if n < 0:
        n += X.shape[1]
    # 如果 X 是稀疏矩阵并且格式为 "csc"，则调用 inplace_swap_row_csr 函数交换列 m 和 n
    elif sp.issparse(X) and X.format == "csc":
        inplace_swap_row_csr(X, m, n)
    # 如果 X 是稀疏矩阵并且格式为 "csr"，则调用 inplace_swap_row_csc 函数交换列 m 和 n
    elif sp.issparse(X) and X.format == "csr":
        inplace_swap_row_csc(X, m, n)
    # 如果 X 不是稀疏矩阵或格式不符合要求，则抛出类型错误
    else:
        _raise_typeerror(X)
def min_max_axis(X, axis, ignore_nan=False):
    """Compute minimum and maximum along an axis on a CSR or CSC matrix.

    Optionally ignore NaN values.

    Parameters
    ----------
    X : sparse matrix of shape (n_samples, n_features)
        Input data. It should be of CSR or CSC format.

    axis : {0, 1}
        Axis along which the axis should be computed.

    ignore_nan : bool, default=False
        Ignore or pass through NaN values.

        .. versionadded:: 0.20

    Returns
    -------
    mins : ndarray of shape (n_features,), dtype={np.float32, np.float64}
        Feature-wise minima.

    maxs : ndarray of shape (n_features,), dtype={np.float32, np.float64}
        Feature-wise maxima.
    """
    if sp.issparse(X) and X.format in ("csr", "csc"):
        if ignore_nan:
            return _sparse_nan_min_max(X, axis=axis)
        else:
            return _sparse_min_max(X, axis=axis)
    else:
        _raise_typeerror(X)


def count_nonzero(X, axis=None, sample_weight=None):
    """A variant of X.getnnz() with extension to weighting on axis 0.

    Useful in efficiently calculating multilabel metrics.

    Parameters
    ----------
    X : sparse matrix of shape (n_samples, n_labels)
        Input data. It should be of CSR format.

    axis : {0, 1}, default=None
        The axis on which the data is aggregated.

    sample_weight : array-like of shape (n_samples,), default=None
        Weight for each row of X.

    Returns
    -------
    nnz : int, float, ndarray of shape (n_samples,) or ndarray of shape (n_features,)
        Number of non-zero values in the array along a given axis. Otherwise,
        the total number of non-zero values in the array is returned.
    """
    if axis == -1:
        axis = 1
    elif axis == -2:
        axis = 0
    elif X.format != "csr":
        raise TypeError("Expected CSR sparse format, got {0}".format(X.format))

    # We rely here on the fact that np.diff(Y.indptr) for a CSR
    # will return the number of nonzero entries in each row.
    # A bincount over Y.indices will return the number of nonzeros
    # in each column. See ``csr_matrix.getnnz`` in scipy >= 0.14.
    if axis is None:
        if sample_weight is None:
            return X.nnz
        else:
            return np.dot(np.diff(X.indptr), sample_weight)
    elif axis == 1:
        out = np.diff(X.indptr)
        if sample_weight is None:
            # astype here is for consistency with axis=0 dtype
            return out.astype("intp")
        return out * sample_weight
    elif axis == 0:
        if sample_weight is None:
            return np.bincount(X.indices, minlength=X.shape[1])
        else:
            weights = np.repeat(sample_weight, np.diff(X.indptr))
            return np.bincount(X.indices, minlength=X.shape[1], weights=weights)
    else:
        raise ValueError("Unsupported axis: {0}".format(axis))


def _get_median(data, n_zeros):
    """Compute the median of data with n_zeros additional zeros.

    Placeholder for the function `_get_median`. Implementation details
    are missing in the provided code snippet.
    """
    This function is used to support sparse matrices; it modifies data
    in-place.
    """
    # 计算数组中非零元素和预期的零元素个数总和
    n_elems = len(data) + n_zeros
    # 如果总元素个数为0，则返回 NaN
    if not n_elems:
        return np.nan
    # 统计数组中小于0的元素个数
    n_negative = np.count_nonzero(data < 0)
    # 计算中位数的位置和是否为奇数个元素
    middle, is_odd = divmod(n_elems, 2)
    # 对数据进行排序
    data.sort()

    # 如果元素个数为奇数，返回中位数
    if is_odd:
        return _get_elem_at_rank(middle, data, n_negative, n_zeros)

    # 如果元素个数为偶数，返回中间两个元素的平均值
    return (
        _get_elem_at_rank(middle - 1, data, n_negative, n_zeros)
        + _get_elem_at_rank(middle, data, n_negative, n_zeros)
    ) / 2.0
def _get_elem_at_rank(rank, data, n_negative, n_zeros):
    """Find the value in data augmented with n_zeros for the given rank"""
    # 如果 rank 小于 n_negative，直接返回 data 中的值
    if rank < n_negative:
        return data[rank]
    # 如果 rank 减去 n_negative 小于 n_zeros，返回 0
    if rank - n_negative < n_zeros:
        return 0
    # 否则返回 data 中对应的值
    return data[rank - n_zeros]


def csc_median_axis_0(X):
    """Find the median across axis 0 of a CSC matrix.

    It is equivalent to doing np.median(X, axis=0).

    Parameters
    ----------
    X : sparse matrix of shape (n_samples, n_features)
        Input data. It should be of CSC format.

    Returns
    -------
    median : ndarray of shape (n_features,)
        Median.
    """
    # 检查输入是否为 CSC 格式的稀疏矩阵
    if not (sp.issparse(X) and X.format == "csc"):
        raise TypeError("Expected matrix of CSC format, got %s" % X.format)

    indptr = X.indptr
    n_samples, n_features = X.shape
    # 创建一个全零数组用于存储中位数
    median = np.zeros(n_features)

    for f_ind, (start, end) in enumerate(zip(indptr[:-1], indptr[1:])):
        # 复制 X.data[start:end]，以防止原地修改 X
        data = np.copy(X.data[start:end])
        # 计算非零元素的数量
        nz = n_samples - data.size
        # 计算当前列的中位数并存储在 median 中
        median[f_ind] = _get_median(data, nz)

    return median


def _implicit_column_offset(X, offset):
    """Create an implicitly offset linear operator.

    This is used by PCA on sparse data to avoid densifying the whole data
    matrix.

    Params
    ------
        X : sparse matrix of shape (n_samples, n_features)
        offset : ndarray of shape (n_features,)

    Returns
    -------
    centered : LinearOperator
    """
    # 将 offset 转换为行向量
    offset = offset[None, :]
    XT = X.T
    # 创建并返回一个 LinearOperator 对象
    return LinearOperator(
        matvec=lambda x: X @ x - offset @ x,
        matmat=lambda x: X @ x - offset @ x,
        rmatvec=lambda x: XT @ x - (offset * x.sum()),
        rmatmat=lambda x: XT @ x - offset.T @ x.sum(axis=0)[None, :],
        dtype=X.dtype,
        shape=X.shape,
    )
```