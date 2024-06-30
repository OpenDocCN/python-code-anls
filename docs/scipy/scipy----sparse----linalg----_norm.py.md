# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_norm.py`

```
# 导入必要的库
import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import svds
from scipy.sparse._sputils import convert_pydata_sparse_to_scipy
import scipy.sparse as sp

# 导入部分函数和常量
from numpy import sqrt, abs

__all__ = ['norm']

# 定义计算稀疏矩阵Frobenius范数的函数
def _sparse_frobenius_norm(x):
    # 转换为Scipy稀疏矩阵的内部表示
    data = sp._sputils._todata(x)
    # 计算转换后数据的Frobenius范数
    return np.linalg.norm(data)

# 定义计算稀疏矩阵的不同范数的函数
def norm(x, ord=None, axis=None):
    """
    Norm of a sparse matrix

    This function is able to return one of seven different matrix norms,
    depending on the value of the ``ord`` parameter.

    Parameters
    ----------
    x : a sparse matrix
        Input sparse matrix.
    ord : {non-zero int, inf, -inf, 'fro'}, optional
        Order of the norm (see table under ``Notes``). inf means numpy's
        `inf` object.
    axis : {int, 2-tuple of ints, None}, optional
        If `axis` is an integer, it specifies the axis of `x` along which to
        compute the vector norms.  If `axis` is a 2-tuple, it specifies the
        axes that hold 2-D matrices, and the matrix norms of these matrices
        are computed.  If `axis` is None then either a vector norm (when `x`
        is 1-D) or a matrix norm (when `x` is 2-D) is returned.

    Returns
    -------
    n : float or ndarray

    Notes
    -----
    Some of the ord are not implemented because some associated functions like,
    _multi_svd_norm, are not yet available for sparse matrix.

    This docstring is modified based on numpy.linalg.norm.
    https://github.com/numpy/numpy/blob/main/numpy/linalg/linalg.py

    The following norms can be calculated:

    =====  ============================
    ord    norm for sparse matrices
    =====  ============================
    None   Frobenius norm
    'fro'  Frobenius norm
    inf    max(sum(abs(x), axis=1))
    -inf   min(sum(abs(x), axis=1))
    0      abs(x).sum(axis=axis)
    1      max(sum(abs(x), axis=0))
    -1     min(sum(abs(x), axis=0))
    2      Spectral norm (the largest singular value)
    -2     Not implemented
    other  Not implemented
    =====  ============================

    The Frobenius norm is given by [1]_:

        :math:`||A||_F = [\\sum_{i,j} abs(a_{i,j})^2]^{1/2}`

    References
    ----------
    .. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*,
        Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15

    Examples
    --------
    >>> from scipy.sparse import *
    >>> import numpy as np
    >>> from scipy.sparse.linalg import norm
    >>> a = np.arange(9) - 4
    >>> a
    array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
    >>> b = a.reshape((3, 3))
    >>> b
    array([[-4, -3, -2],
           [-1,  0,  1],
           [ 2,  3,  4]])

    >>> b = csr_matrix(b)
    >>> norm(b)
    7.745966692414834
    >>> norm(b, 'fro')
    7.745966692414834
    >>> norm(b, np.inf)
    9
    >>> norm(b, -np.inf)
    2
    >>> norm(b, 1)
    7
    >>> norm(b, -1)
    6

    The matrix 2-norm or the spectral norm is the largest singular value.

    """
    x = convert_pydata_sparse_to_scipy(x, target_format="csr")
    # 将输入稀疏矩阵转换为 scipy 的 csr 格式

    if not issparse(x):
        # 检查是否为稀疏矩阵，如果不是则抛出类型错误
        raise TypeError("input is not sparse. use numpy.linalg.norm")

    # 检查默认情况并立即处理它
    if axis is None and ord in (None, 'fro', 'f'):
        # 如果轴为 None 并且 ord 在 None、'fro'、'f' 中，则返回 Frobenius 范数
        return _sparse_frobenius_norm(x)

    # 将稀疏矩阵转换为 csr 格式
    x = x.tocsr()

    if axis is None:
        axis = (0, 1)
    elif not isinstance(axis, tuple):
        # 如果 axis 不是元组，则尝试将其转换为整数，若失败则抛出类型错误
        msg = "'axis' must be None, an integer or a tuple of integers"
        try:
            int_axis = int(axis)
        except TypeError as e:
            raise TypeError(msg) from e
        if axis != int_axis:
            raise TypeError(msg)
        axis = (int_axis,)

    nd = 2
    if len(axis) == 2:
        # 如果 axis 的长度为 2
        row_axis, col_axis = axis
        if not (-nd <= row_axis < nd and -nd <= col_axis < nd):
            # 检查轴是否在有效范围内，否则抛出值错误
            message = f'Invalid axis {axis!r} for an array with shape {x.shape!r}'
            raise ValueError(message)
        if row_axis % nd == col_axis % nd:
            # 如果行轴和列轴相同，则抛出值错误
            raise ValueError('Duplicate axes given.')
        if ord == 2:
            # 如果 ord 为 2，则计算最大奇异值
            _, s, _ = svds(x, k=1, solver="lobpcg")
            return s[0]
        elif ord == -2:
            # 如果 ord 为 -2，暂未实现
            raise NotImplementedError
            #return _multi_svd_norm(x, row_axis, col_axis, amin)
        elif ord == 1:
            # 如果 ord 为 1，则计算行轴上绝对值的和的最大值
            return abs(x).sum(axis=row_axis).max(axis=col_axis)[0,0]
        elif ord == np.inf:
            # 如果 ord 为 np.inf，则计算列轴上绝对值的和的最大值
            return abs(x).sum(axis=col_axis).max(axis=row_axis)[0,0]
        elif ord == -1:
            # 如果 ord 为 -1，则计算行轴上绝对值的和的最小值
            return abs(x).sum(axis=row_axis).min(axis=col_axis)[0,0]
        elif ord == -np.inf:
            # 如果 ord 为 -np.inf，则计算列轴上绝对值的和的最小值
            return abs(x).sum(axis=col_axis).min(axis=row_axis)[0,0]
        elif ord in (None, 'f', 'fro'):
            # 如果 ord 在 None、'f'、'fro' 中，则返回 Frobenius 范数
            return _sparse_frobenius_norm(x)
        else:
            # 如果 ord 无效，则抛出值错误
            raise ValueError("Invalid norm order for matrices.")
    # 如果轴 `axis` 的长度为1，则执行以下操作
    elif len(axis) == 1:
        # 从 `axis` 中解包出变量 `a`
        a, = axis
        # 如果 `a` 不在有效范围内，则抛出值错误异常
        if not (-nd <= a < nd):
            message = f'Invalid axis {axis!r} for an array with shape {x.shape!r}'
            raise ValueError(message)
        # 根据不同的范数 `ord` 计算 `M`
        if ord == np.inf:
            # 计算无穷范数
            M = abs(x).max(axis=a)
        elif ord == -np.inf:
            # 计算负无穷范数
            M = abs(x).min(axis=a)
        elif ord == 0:
            # 计算零范数
            M = (x != 0).sum(axis=a)
        elif ord == 1:
            # 特殊情况的加速处理
            M = abs(x).sum(axis=a)
        elif ord in (2, None):
            # 计算二范数或默认范数
            M = sqrt(abs(x).power(2).sum(axis=a))
        else:
            # 尝试将 ord + 1，如果出现类型错误，则抛出值错误异常
            try:
                ord + 1
            except TypeError as e:
                raise ValueError('Invalid norm order for vectors.') from e
            # 计算给定 ord 范数
            M = np.power(abs(x).power(ord).sum(axis=a), 1 / ord)
        # 如果 `M` 具有 `toarray` 属性，则将其转换为数组并展平
        if hasattr(M, 'toarray'):
            return M.toarray().ravel()
        # 如果 `M` 具有 `A` 属性，则将其转换为数组并展平
        elif hasattr(M, 'A'):
            return M.A.ravel()
        else:
            # 否则直接展平 `M`
            return M.ravel()
    else:
        # 如果轴 `axis` 的长度不为1，则抛出值错误异常
        raise ValueError("Improper number of dimensions to norm.")
```