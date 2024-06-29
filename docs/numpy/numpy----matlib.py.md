# `.\numpy\numpy\matlib.py`

```py
import warnings

# 在导入 numpy.matlib 的过程中，如果使用 matrix 类别会出现弃用警告
# 此警告于 numpy 1.19.0 版本添加，并且在此处设置 stacklevel=2 以准确显示警告位置
warnings.warn("Importing from numpy.matlib is deprecated since 1.19.0. "
              "The matrix subclass is not the recommended way to represent "
              "matrices or deal with linear algebra (see "
              "https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html). "
              "Please adjust your code to use regular ndarray. ",
              PendingDeprecationWarning, stacklevel=2)

import numpy as np
from numpy.matrixlib.defmatrix import matrix, asmatrix

# Matlib.py 文件包含了 numpy 命名空间的所有函数，但做了一些替换
# 参见 doc/source/reference/routines.matlib.rst 了解详细信息
# 此处使用 * 从 numpy 中复制命名空间，包括了所有函数和对象
from numpy import *  # noqa: F403

__version__ = np.__version__

# 复制 numpy 命名空间到当前模块的 __all__ 中
__all__ = np.__all__[:]  # copy numpy namespace
__all__ += ['rand', 'randn', 'repmat']

def empty(shape, dtype=None, order='C'):
    """Return a new matrix of given shape and type, without initializing entries.

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the empty matrix.
    dtype : data-type, optional
        Desired output data-type.
    order : {'C', 'F'}, optional
        Whether to store multi-dimensional data in row-major
        (C-style) or column-major (Fortran-style) order in
        memory.

    See Also
    --------
    numpy.empty : Equivalent array function.
    matlib.zeros : Return a matrix of zeros.
    matlib.ones : Return a matrix of ones.

    Notes
    -----
    Unlike other matrix creation functions (e.g. `matlib.zeros`,
    `matlib.ones`), `matlib.empty` does not initialize the values of the
    matrix, and may therefore be marginally faster. However, the values
    stored in the newly allocated matrix are arbitrary. For reproducible
    behavior, be sure to set each element of the matrix before reading.

    Examples
    --------
    >>> import numpy.matlib
    >>> np.matlib.empty((2, 2))    # filled with random data
    matrix([[  6.76425276e-320,   9.79033856e-307], # random
            [  7.39337286e-309,   3.22135945e-309]])
    >>> np.matlib.empty((2, 2), dtype=int)
    matrix([[ 6600475,        0], # random
            [ 6586976, 22740995]])

    """
    # 创建一个新的矩阵对象，但不初始化条目的值
    return ndarray.__new__(matrix, shape, dtype, order=order)

def ones(shape, dtype=None, order='C'):
    """
    Matrix of ones.

    Return a matrix of given shape and type, filled with ones.

    Parameters
    ----------
    shape : {sequence of ints, int}
        Shape of the matrix
    dtype : data-type, optional
        The desired data-type for the matrix, default is np.float64.
    order : {'C', 'F'}, optional
        Whether to store matrix in C- or Fortran-contiguous order,
        default is 'C'.

    Returns
    -------
    out : matrix
        Matrix of ones of given shape, dtype, and order.

    See Also
    --------
    ones : Array of ones.

    """
    # 返回一个给定形状和类型的矩阵，填充为全部为 1
    return np.ones(shape, dtype=dtype, order=order)
    # 使用 matlib 模块中的 zeros 函数创建一个零矩阵。
    # 
    # Notes
    # -----
    # 如果 `shape` 的长度为一，即 ``(N,)``，或者是一个标量 `N`，则 `out` 将变成一个形状为 ``(1,N)`` 的单行矩阵。
    # 
    # Examples
    # --------
    # >>> np.matlib.ones((2,3))
    # matrix([[1.,  1.,  1.],
    #         [1.,  1.,  1.]])
    # 
    # >>> np.matlib.ones(2)
    # matrix([[1.,  1.]])
    # 
    def zeros(shape, dtype=None, order='C'):
        """
        a = ndarray.__new__(matrix, shape, dtype, order=order)
        使用给定的 shape、dtype 和 order 创建一个新的 matrix 对象 a。
        """
        a = ndarray.__new__(matrix, shape, dtype, order=order)
        # 用值 1 填充矩阵 a
        a.fill(1)
        return a
# 返回一个给定形状和类型的零矩阵。

def zeros(shape, dtype=None, order='C'):
    # 创建一个新的矩阵对象，继承自 ndarray 类型的 matrix，并初始化为零矩阵
    a = ndarray.__new__(matrix, shape, dtype, order=order)
    # 将矩阵中的所有元素填充为零
    a.fill(0)
    return a

# 返回给定大小的单位矩阵。

def identity(n,dtype=None):
    # 创建一个长度为 n 的列表，第一个元素为 1，其余为 0，并指定数据类型为 dtype
    a = array([1]+n*[0], dtype=dtype)
    # 创建一个 n x n 的空矩阵，并将其扁平化后赋值为列表 a，形成单位矩阵
    b = empty((n, n), dtype=dtype)
    b.flat = a
    return b

# 返回一个主对角线上元素为 1 其余为零的矩阵。

def eye(n,M=None, k=0, dtype=float, order='C'):
    # 创建一个 n x M 的零矩阵，默认情况下 M 等于 n
    I = zeros((n, M if M else n), dtype=dtype, order=order)
    # 根据参数 k，在矩阵中设置对应对角线上的元素为 1
    if k >= 0:
        I.flat[k::I.shape[1]+1] = 1
    else:
        I.flat[-k*I.shape[1]::I.shape[1]-1] = 1
    return I
    # 创建一个 n x M 的单位矩阵，其中从主对角线偏移 k 个位置，并指定数据类型为 float
    >>> np.matlib.eye(3, k=1, dtype=float)
    matrix([[0.,  1.,  0.],
            [0.,  0.,  1.],
            [0.,  0.,  0.]])
    """
    返回一个 n x M 的单位矩阵的 numpy 矩阵表示，其中对角线向上偏移 k 个位置。
def repmat(a, m, n):
    """
    Repeat a 0-D to 2-D array or matrix MxN times.

    Parameters
    ----------
    a : array_like
        The array or matrix to be repeated.
    m : int
        Number of times the input array `a` is repeated along the first dimension.
    n : int
        Number of times the input array `a` is repeated along the second dimension, if `a` is 2-D.

    Returns
    -------
    out : ndarray
        The tiled output array.

    See Also
    --------
    tile, numpy.matlib.repmat

    Notes
    -----
    If `a` is 0-D or 1-D, the result will be a 1-D array repeating `a` `m` times.
    If `a` is 2-D, the result will be a 2-D array repeating `a` `m` times along the first axis and `n` times along the second axis.

    Examples
    --------
    >>> np.matlib.repmat(np.array([[1, 2], [3, 4]]), 2, 3)
    matrix([[1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4],
            [1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4]])

    >>> np.matlib.repmat(np.array([1, 2]), 3, 2)
    matrix([[1, 2],
            [1, 2],
            [1, 2]])

    """
    # 使用 numpy 的 tile 函数实现重复数组 a 的操作
    return np.tile(np.asarray(a), (m, n))
    # 将输入参数 `a` 转换为多维数组
    a = asanyarray(a)
    
    # 获取数组 `a` 的维度
    ndim = a.ndim
    
    # 根据数组的维度确定原始行数和列数
    if ndim == 0:
        origrows, origcols = (1, 1)  # 对于零维数组，原始行数和列数均为1
    elif ndim == 1:
        origrows, origcols = (1, a.shape[0])  # 对于一维数组，原始行数为1，列数为数组长度
    else:
        origrows, origcols = a.shape  # 对于高维数组，原始行数和列数即为数组的形状
    
    # 计算重复后的总行数和总列数
    rows = origrows * m
    cols = origcols * n
    
    # 将数组 `a` 展平为一维，并按指定的行数 `m` 重复堆叠，然后按原始列数重新整形
    # 再按指定的列数 `n` 重复堆叠，形成最终的重复数组
    c = a.reshape(1, a.size).repeat(m, 0).reshape(rows, origcols).repeat(n, 0)
    
    # 返回重复后的数组，重新整形为指定的总行数和总列数
    return c.reshape(rows, cols)
```