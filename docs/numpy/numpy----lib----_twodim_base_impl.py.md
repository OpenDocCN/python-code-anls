# `.\numpy\numpy\lib\_twodim_base_impl.py`

```py
""" Basic functions for manipulating 2d arrays

"""
# 导入 functools 模块，用于创建偏函数
import functools
# 导入 operator 模块，用于操作符函数
import operator

# 导入以下 numpy 内部模块和函数
from numpy._core._multiarray_umath import _array_converter
from numpy._core.numeric import (
    asanyarray, arange, zeros, greater_equal, multiply, ones,
    asarray, where, int8, int16, int32, int64, intp, empty, promote_types,
    diagonal, nonzero, indices
    )
# 导入 numpy 的 overrides 模块和函数
from numpy._core.overrides import set_array_function_like_doc, set_module
from numpy._core import overrides
# 导入 numpy 的 iinfo 模块
from numpy._core import iinfo
# 导入 numpy 的 _stride_tricks_impl 模块中的 broadcast_to 函数
from numpy.lib._stride_tricks_impl import broadcast_to


# 定义公开的函数和类列表
__all__ = [
    'diag', 'diagflat', 'eye', 'fliplr', 'flipud', 'tri', 'triu',
    'tril', 'vander', 'histogram2d', 'mask_indices', 'tril_indices',
    'tril_indices_from', 'triu_indices', 'triu_indices_from', ]


# 创建 functools.partial 对象，用于将 overrides.array_function_dispatch 函数与特定模块名 'numpy' 绑定
array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')


# 获取 int8、int16 和 int32 类型的信息并存储在 i1、i2 和 i4 变量中
i1 = iinfo(int8)
i2 = iinfo(int16)
i4 = iinfo(int32)


def _min_int(low, high):
    """ get small int that fits the range """
    # 根据给定的范围返回适合的最小整数类型
    if high <= i1.max and low >= i1.min:
        return int8
    if high <= i2.max and low >= i2.min:
        return int16
    if high <= i4.max and low >= i4.min:
        return int32
    return int64


# 定义用于 fliplr 函数的分发器
def _flip_dispatcher(m):
    return (m,)


@array_function_dispatch(_flip_dispatcher)
def fliplr(m):
    """
    Reverse the order of elements along axis 1 (left/right).

    For a 2-D array, this flips the entries in each row in the left/right
    direction. Columns are preserved, but appear in a different order than
    before.

    Parameters
    ----------
    m : array_like
        Input array, must be at least 2-D.

    Returns
    -------
    f : ndarray
        A view of `m` with the columns reversed.  Since a view
        is returned, this operation is :math:`\\mathcal O(1)`.

    See Also
    --------
    flipud : Flip array in the up/down direction.
    flip : Flip array in one or more dimensions.
    rot90 : Rotate array counterclockwise.

    Notes
    -----
    Equivalent to ``m[:,::-1]`` or ``np.flip(m, axis=1)``.
    Requires the array to be at least 2-D.

    Examples
    --------
    >>> A = np.diag([1.,2.,3.])
    >>> A
    array([[1.,  0.,  0.],
           [0.,  2.,  0.],
           [0.,  0.,  3.]])
    >>> np.fliplr(A)
    array([[0.,  0.,  1.],
           [0.,  2.,  0.],
           [3.,  0.,  0.]])

    >>> rng = np.random.default_rng()
    >>> A = rng.normal(size=(2,3,5))
    >>> np.all(np.fliplr(A) == A[:,::-1,...])
    True

    """
    # 将输入 m 转换为 ndarray 类型
    m = asanyarray(m)
    # 如果 m 的维度小于 2，则引发 ValueError 异常
    if m.ndim < 2:
        raise ValueError("Input must be >= 2-d.")
    # 返回列反转后的 m 视图
    return m[:, ::-1]


@array_function_dispatch(_flip_dispatcher)
def flipud(m):
    """
    Reverse the order of elements along axis 0 (up/down).

    For a 2-D array, this flips the entries in each column in the up/down
    direction. Rows are preserved, but appear in a different order than before.

    Parameters
    ----------
    m : array_like
        Input array.

    Returns
    -------
    # 将输入的数组 m 转换为 numpy 数组，确保能够进行后续操作
    m = asanyarray(m)
    # 如果输入数组的维度小于 1，则抛出值错误异常
    if m.ndim < 1:
        raise ValueError("Input must be >= 1-d.")
    # 返回翻转行的视图数组，相当于 m[::-1, ...] 的操作
    return m[::-1, ...]
# 用于设置函数的数组功能文档样式
@set_array_function_like_doc
# 用于设置函数的模块为 'numpy'
@set_module('numpy')
# 定义一个名为 eye 的函数，生成一个二维数组，对角线为 1，其他位置为 0
def eye(N, M=None, k=0, dtype=float, order='C', *, device=None, like=None):
    """
    Return a 2-D array with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    N : int
      Number of rows in the output.
    M : int, optional
      Number of columns in the output. If None, defaults to `N`.
    k : int, optional
      Index of the diagonal: 0 (the default) refers to the main diagonal,
      a positive value refers to an upper diagonal, and a negative value
      to a lower diagonal.
    dtype : data-type, optional
      Data-type of the returned array.
    order : {'C', 'F'}, optional
        Whether the output should be stored in row-major (C-style) or
        column-major (Fortran-style) order in memory.

        .. versionadded:: 1.14.0
    device : str, optional
        The device on which to place the created array. Default: None.
        For Array-API interoperability only, so must be ``"cpu"`` if passed.

        .. versionadded:: 2.0.0
    ${ARRAY_FUNCTION_LIKE}

        .. versionadded:: 1.20.0

    Returns
    -------
    I : ndarray of shape (N,M)
      An array where all elements are equal to zero, except for the `k`-th
      diagonal, whose values are equal to one.

    See Also
    --------
    identity : (almost) equivalent function
    diag : diagonal 2-D array from a 1-D array specified by the user.

    Examples
    --------
    >>> np.eye(2, dtype=int)
    array([[1, 0],
           [0, 1]])
    >>> np.eye(3, k=1)
    array([[0.,  1.,  0.],
           [0.,  0.,  1.],
           [0.,  0.,  0.]])

    """
    # 如果 like 参数不为 None，则调用 _eye_with_like 函数处理
    if like is not None:
        return _eye_with_like(
            like, N, M=M, k=k, dtype=dtype, order=order, device=device
        )
    # 如果 M 为 None，则将 M 设为 N
    if M is None:
        M = N
    # 创建一个形状为 (N, M) 的零数组 m，数据类型为 dtype，存储顺序为 order，设备为 device
    m = zeros((N, M), dtype=dtype, order=order, device=device)
    # 如果 k 大于等于 M，则直接返回 m
    if k >= M:
        return m
    # 确保 M 和 k 是整数，以避免意外的类型转换结果
    M = operator.index(M)
    k = operator.index(k)
    # 如果 k 大于等于 0，则将 i 设为 k；否则将 i 设为 (-k) * M
    if k >= 0:
        i = k
    else:
        i = (-k) * M
    # 在 m 的前 M-k 行中，每隔 M+1 个元素设置为 1，从索引 i 开始
    m[:M-k].flat[i::M+1] = 1
    # 返回生成的二维数组 m
    return m


# 通过数组函数分发装饰器将 _eye_with_like 函数与 eye 函数关联起来
_eye_with_like = array_function_dispatch()(eye)


# 定义一个 _diag_dispatcher 函数，用于 diag 函数的分派
def _diag_dispatcher(v, k=None):
    return (v,)


# 通过数组函数分发装饰器将 _diag_dispatcher 函数与 diag 函数关联起来
@array_function_dispatch(_diag_dispatcher)
# 定义 diag 函数，用于提取对角线或构造对角线数组
def diag(v, k=0):
    """
    Extract a diagonal or construct a diagonal array.

    See the more detailed documentation for ``numpy.diagonal`` if you use this
    function to extract a diagonal and wish to write to the resulting array;
    whether it returns a copy or a view depends on what version of numpy you
    are using.

    Parameters
    ----------
    v : array_like
        If `v` is a 2-D array, return a copy of its `k`-th diagonal.
        If `v` is a 1-D array, return a 2-D array with `v` on the `k`-th
        diagonal.

    """
    v = asanyarray(v)
    s = v.shape
    # 将输入转换为任意数组，确保可以处理不同类型的输入
    if len(s) == 1:
        # 如果输入是一维数组
        n = s[0]+abs(k)
        # 计算输出数组的大小，包括需要的额外对角线长度
        res = zeros((n, n), v.dtype)
        # 创建一个全零数组作为输出，与输入数组类型相同
        if k >= 0:
            i = k
        else:
            i = (-k) * n
        # 计算对角线的起始索引
        res[:n-k].flat[i::n+1] = v
        # 将输入数组的值填充到输出数组的对角线上
        return res
    elif len(s) == 2:
        # 如果输入是二维数组
        return diagonal(v, k)
        # 调用 diagonal 函数返回指定位置的对角线数组
    else:
        # 处理不支持的输入维度
        raise ValueError("Input must be 1- or 2-d.")
        # 抛出值错误，要求输入必须是一维或二维数组
@array_function_dispatch(_diag_dispatcher)
# 使用装饰器将 _diag_dispatcher 函数与 diagflat 函数关联，用于数组函数分发
def diagflat(v, k=0):
    """
    Create a two-dimensional array with the flattened input as a diagonal.

    Parameters
    ----------
    v : array_like
        Input data, which is flattened and set as the `k`-th
        diagonal of the output.
    k : int, optional
        Diagonal to set; 0, the default, corresponds to the "main" diagonal,
        a positive (negative) `k` giving the number of the diagonal above
        (below) the main.

    Returns
    -------
    out : ndarray
        The 2-D output array.

    See Also
    --------
    diag : MATLAB work-alike for 1-D and 2-D arrays.
    diagonal : Return specified diagonals.
    trace : Sum along diagonals.

    Examples
    --------
    >>> np.diagflat([[1,2], [3,4]])
    array([[1, 0, 0, 0],
           [0, 2, 0, 0],
           [0, 0, 3, 0],
           [0, 0, 0, 4]])

    >>> np.diagflat([1,2], 1)
    array([[0, 1, 0],
           [0, 0, 2],
           [0, 0, 0]])

    """
    # 转换输入数据 v 并展平
    conv = _array_converter(v)
    v, = conv.as_arrays(subok=False)
    v = v.ravel()
    s = len(v)
    n = s + abs(k)
    # 创建一个形状为 (n, n) 的零数组，指定数据类型为 v 的数据类型
    res = zeros((n, n), v.dtype)
    if (k >= 0):
        # 对于 k >= 0 的情况，设置正对角线及其上的对角线
        i = arange(0, n-k, dtype=intp)
        fi = i+k+i*n
    else:
        # 对于 k < 0 的情况，设置负对角线及其下的对角线
        i = arange(0, n+k, dtype=intp)
        fi = i+(i-k)*n
    # 将展平的 v 数组数据设置到 res 的指定位置
    res.flat[fi] = v

    return conv.wrap(res)


@set_array_function_like_doc
@set_module('numpy')
# 使用装饰器设置 tri 函数的数组函数行为及所属模块为 numpy
def tri(N, M=None, k=0, dtype=float, *, like=None):
    """
    An array with ones at and below the given diagonal and zeros elsewhere.

    Parameters
    ----------
    N : int
        Number of rows in the array.
    M : int, optional
        Number of columns in the array.
        By default, `M` is taken equal to `N`.
    k : int, optional
        The sub-diagonal at and below which the array is filled.
        `k` = 0 is the main diagonal, while `k` < 0 is below it,
        and `k` > 0 is above.  The default is 0.
    dtype : dtype, optional
        Data type of the returned array.  The default is float.
    ${ARRAY_FUNCTION_LIKE}

        .. versionadded:: 1.20.0

    Returns
    -------
    tri : ndarray of shape (N, M)
        Array with its lower triangle filled with ones and zero elsewhere;
        in other words ``T[i,j] == 1`` for ``j <= i + k``, 0 otherwise.

    Examples
    --------
    >>> np.tri(3, 5, 2, dtype=int)
    array([[1, 1, 1, 0, 0],
           [1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1]])

    >>> np.tri(3, 5, -1)
    array([[0.,  0.,  0.,  0.,  0.],
           [1.,  0.,  0.,  0.,  0.],
           [1.,  1.,  0.,  0.,  0.]])

    """
    if like is not None:
        # 如果指定了 like 参数，返回与 like 相同类型的三角矩阵
        return _tri_with_like(like, N, M=M, k=k, dtype=dtype)

    if M is None:
        M = N

    # 创建一个布尔类型的 N x M 形状的矩阵，其中下三角（包括对角线）为 1，其余为 0
    m = greater_equal.outer(arange(N, dtype=_min_int(0, N)),
                            arange(-k, M-k, dtype=_min_int(-k, M - k)))

    # 避免在已经是布尔类型时复制矩阵
    m = m.astype(dtype, copy=False)

    return m


# 将 array_function_dispatch 装饰器应用到 tri 函数
_tri_with_like = array_function_dispatch()(tri)
# 根据输入参数生成一个 Vandermonde 矩阵
def vander(x, N=None, increasing=False):
    # 如果 N 为 None，则设置 N 为 x 的长度
    if N is None:
        N = len(x)
    # 根据 increasing 参数决定是否递增排序
    if increasing:
        # 生成递增排序的 Vandermonde 矩阵
        return array([x**(N-1-i) for i in range(N)]).T
    else:
        # 生成递减排序的 Vandermonde 矩阵
        return array([x**i for i in range(N)]).T
    """
    The columns of the output matrix are powers of the input vector. The
    order of the powers is determined by the `increasing` boolean argument.
    Specifically, when `increasing` is False, the `i`-th output column is
    the input vector raised element-wise to the power of ``N - i - 1``. Such
    a matrix with a geometric progression in each row is named for Alexandre-
    Theophile Vandermonde.
    
    Parameters
    ----------
    x : array_like
        1-D input array.
    N : int, optional
        Number of columns in the output.  If `N` is not specified, a square
        array is returned (``N = len(x)``).
    increasing : bool, optional
        Order of the powers of the columns.  If True, the powers increase
        from left to right, if False (the default) they are reversed.
    
        .. versionadded:: 1.9.0
    
    Returns
    -------
    out : ndarray
        Vandermonde matrix.  If `increasing` is False, the first column is
        ``x^(N-1)``, the second ``x^(N-2)`` and so forth. If `increasing` is
        True, the columns are ``x^0, x^1, ..., x^(N-1)``.
    
    See Also
    --------
    polynomial.polynomial.polyvander
    
    Examples
    --------
    >>> x = np.array([1, 2, 3, 5])
    >>> N = 3
    >>> np.vander(x, N)
    array([[ 1,  1,  1],
           [ 4,  2,  1],
           [ 9,  3,  1],
           [25,  5,  1]])
    
    >>> np.column_stack([x**(N-1-i) for i in range(N)])
    array([[ 1,  1,  1],
           [ 4,  2,  1],
           [ 9,  3,  1],
           [25,  5,  1]])
    
    >>> x = np.array([1, 2, 3, 5])
    >>> np.vander(x)
    array([[  1,   1,   1,   1],
           [  8,   4,   2,   1],
           [ 27,   9,   3,   1],
           [125,  25,   5,   1]])
    >>> np.vander(x, increasing=True)
    array([[  1,   1,   1,   1],
           [  1,   2,   4,   8],
           [  1,   3,   9,  27],
           [  1,   5,  25, 125]])
    
    The determinant of a square Vandermonde matrix is the product
    of the differences between the values of the input vector:
    
    >>> np.linalg.det(np.vander(x))
    48.000000000000043 # may vary
    >>> (5-3)*(5-2)*(5-1)*(3-2)*(3-1)*(2-1)
    48
    """
    
    # 将输入数组 x 转换为 ndarray 类型
    x = asarray(x)
    
    # 如果 x 不是一维数组，则抛出异常
    if x.ndim != 1:
        raise ValueError("x must be a one-dimensional array or sequence.")
    
    # 如果未指定 N，则将 N 设置为 x 的长度
    if N is None:
        N = len(x)
    
    # 创建一个空的数组 v，形状为 (len(x), N)，数据类型为 x 的数据类型和整数类型的广播类型
    v = empty((len(x), N), dtype=promote_types(x.dtype, int))
    
    # 根据 increasing 参数选择操作数组 tmp
    tmp = v[:, ::-1] if not increasing else v
    
    # 如果 N > 0，则将 tmp 的第一列设置为 1
    if N > 0:
        tmp[:, 0] = 1
    
    # 如果 N > 1，则将 tmp 的第二列到最后一列设置为 x 的列向量的累积乘积
    if N > 1:
        tmp[:, 1:] = x[:, None]
        multiply.accumulate(tmp[:, 1:], out=tmp[:, 1:], axis=1)
    
    # 返回生成的 Vandermonde 矩阵 v
    return v
def _histogram2d_dispatcher(x, y, bins=None, range=None, density=None,
                            weights=None):
    # 生成器函数，用于分派参数 x, y, bins, range, density, weights
    yield x  # 生成器返回 x
    yield y  # 生成器返回 y

    # 以下逻辑从 histogram2d 中的检查逻辑进行了糟糕的适应
    try:
        N = len(bins)  # 尝试获取 bins 的长度
    except TypeError:
        N = 1  # 如果无法获取长度，则默认 N 为 1
    if N == 2:
        yield from bins  # 如果 N 等于 2，则返回 bins 中的元素 [x, y]
    else:
        yield bins  # 否则返回 bins

    yield weights  # 返回 weights


@array_function_dispatch(_histogram2d_dispatcher)
def histogram2d(x, y, bins=10, range=None, density=None, weights=None):
    """
    Compute the bi-dimensional histogram of two data samples.

    Parameters
    ----------
    x : array_like, shape (N,)
        An array containing the x coordinates of the points to be
        histogrammed.
    y : array_like, shape (N,)
        An array containing the y coordinates of the points to be
        histogrammed.
    bins : int or array_like or [int, int] or [array, array], optional
        The bin specification:

        * If int, the number of bins for the two dimensions (nx=ny=bins).
        * If array_like, the bin edges for the two dimensions
          (x_edges=y_edges=bins).
        * If [int, int], the number of bins in each dimension
          (nx, ny = bins).
        * If [array, array], the bin edges in each dimension
          (x_edges, y_edges = bins).
        * A combination [int, array] or [array, int], where int
          is the number of bins and array is the bin edges.

    range : array_like, shape(2,2), optional
        The leftmost and rightmost edges of the bins along each dimension
        (if not specified explicitly in the `bins` parameters):
        ``[[xmin, xmax], [ymin, ymax]]``. All values outside of this range
        will be considered outliers and not tallied in the histogram.
    density : bool, optional
        If False, the default, returns the number of samples in each bin.
        If True, returns the probability *density* function at the bin,
        ``bin_count / sample_count / bin_area``.
    weights : array_like, shape(N,), optional
        An array of values ``w_i`` weighing each sample ``(x_i, y_i)``.
        Weights are normalized to 1 if `density` is True. If `density` is
        False, the values of the returned histogram are equal to the sum of
        the weights belonging to the samples falling into each bin.

    Returns
    -------
    H : ndarray, shape(nx, ny)
        The bi-dimensional histogram of samples `x` and `y`. Values in `x`
        are histogrammed along the first dimension and values in `y` are
        histogrammed along the second dimension.
    xedges : ndarray, shape(nx+1,)
        The bin edges along the first dimension.
    yedges : ndarray, shape(ny+1,)
        The bin edges along the second dimension.

    See Also
    --------
    histogram : 1D histogram
    histogramdd : Multidimensional histogram

    Notes
    -----
    When `density` is True, then the returned histogram is the sample
    density, defined such that the sum over bins of the product
    """
    pass  # 空函数体，因为函数文档字符串已经提供了函数功能的说明
    # `bin_value * bin_area` is 1.
    # 上述表达式中 `bin_value * bin_area` 的结果为1。

    # Please note that the histogram does not follow the Cartesian convention
    # where `x` values are on the abscissa and `y` values on the ordinate
    # axis.  Rather, `x` is histogrammed along the first dimension of the
    # array (vertical), and `y` along the second dimension of the array
    # (horizontal).  This ensures compatibility with `histogramdd`.
    # 请注意，直方图不遵循笛卡尔坐标系的传统规定，其中 `x` 值位于横坐标轴上，
    # `y` 值位于纵坐标轴上。相反，`x` 被直方图化沿着数组的第一个维度（垂直方向），
    # `y` 被直方图化沿着数组的第二个维度（水平方向）。这样做确保与 `histogramdd` 兼容。

    # Examples
    # --------
    # 例子

    # Import necessary modules
    # 导入必要的模块
    >>> from matplotlib.image import NonUniformImage
    >>> import matplotlib.pyplot as plt

    # Construct a 2-D histogram with variable bin width. First define the bin
    # edges:
    # 构建一个具有可变箱宽的二维直方图。首先定义箱子的边缘：

    >>> xedges = [0, 1, 3, 5]
    >>> yedges = [0, 2, 3, 4, 6]

    # Next we create a histogram H with random bin content:
    # 接下来，我们用随机箱内容创建直方图 H：

    >>> x = np.random.normal(2, 1, 100)
    >>> y = np.random.normal(1, 1, 100)
    >>> H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))

    # Histogram does not follow Cartesian convention (see Notes),
    # therefore transpose H for visualization purposes.
    # 直方图不遵循笛卡尔坐标系的传统规定（见注释），因此为了可视化目的转置 H。
    >>> H = H.T

    # :func:`imshow <matplotlib.pyplot.imshow>` can only display square bins:
    # :func:`imshow <matplotlib.pyplot.imshow>` 只能显示正方形的箱子：

    >>> fig = plt.figure(figsize=(7, 3))
    >>> ax = fig.add_subplot(131, title='imshow: square bins')
    >>> plt.imshow(H, interpolation='nearest', origin='lower',
    ...         extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

    # :func:`pcolormesh <matplotlib.pyplot.pcolormesh>` can display actual edges:
    # :func:`pcolormesh <matplotlib.pyplot.pcolormesh>` 可以显示实际的边缘：

    >>> ax = fig.add_subplot(132, title='pcolormesh: actual edges',
    ...         aspect='equal')
    >>> X, Y = np.meshgrid(xedges, yedges)
    >>> ax.pcolormesh(X, Y, H)

    # :class:`NonUniformImage <matplotlib.image.NonUniformImage>` can be used to
    # display actual bin edges with interpolation:
    # :class:`NonUniformImage <matplotlib.image.NonUniformImage>` 可以用来显示带插值的实际箱边缘：

    >>> ax = fig.add_subplot(133, title='NonUniformImage: interpolated',
    ...         aspect='equal', xlim=xedges[[0, -1]], ylim=yedges[[0, -1]])
    >>> im = NonUniformImage(ax, interpolation='bilinear')
    >>> xcenters = (xedges[:-1] + xedges[1:]) / 2
    >>> ycenters = (yedges[:-1] + yedges[1:]) / 2
    >>> im.set_data(xcenters, ycenters, H)
    >>> ax.add_image(im)

    # Show the plot
    # 显示图形
    >>> plt.show()

    # It is also possible to construct a 2-D histogram without specifying bin
    # edges:
    # 也可以构建一个没有指定箱边界的二维直方图：

    >>> # Generate non-symmetric test data
    >>> n = 10000
    >>> x = np.linspace(1, 100, n)
    >>> y = 2*np.log(x) + np.random.rand(n) - 0.5
    >>> # Compute 2d histogram. Note the order of x/y and xedges/yedges
    >>> H, yedges, xedges = np.histogram2d(y, x, bins=20)

    # Now we can plot the histogram using
    # :func:`pcolormesh <matplotlib.pyplot.pcolormesh>`, and a
    # :func:`hexbin <matplotlib.pyplot.hexbin>` for comparison.
    # 现在我们可以使用 :func:`pcolormesh <matplotlib.pyplot.pcolormesh>` 绘制直方图，
    # 并使用 :func:`hexbin <matplotlib.pyplot.hexbin>` 进行比较。

    >>> # Plot histogram using pcolormesh
    >>> fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
    >>> ax1.pcolormesh(xedges, yedges, H, cmap='rainbow')
    >>> ax1.plot(x, 2*np.log(x), 'k-')
    >>> ax1.set_xlim(x.min(), x.max())
    >>> ax1.set_ylim(y.min(), y.max())
    >>> ax1.set_xlabel('x')
    >>> ax1.set_ylabel('y')
    >>> ax1.set_title('histogram2d')
    >>> ax1.grid()

    >>> # 在 ax1 上创建二维直方图
    >>> # 设置 y 轴的范围为 y 的最小值到最大值
    >>> ax1.set_ylim(y.min(), y.max())
    >>> # 设置 x 轴标签为 'x'
    >>> ax1.set_xlabel('x')
    >>> # 设置 y 轴标签为 'y'
    >>> ax1.set_ylabel('y')
    >>> # 设置图表标题为 'histogram2d'
    >>> ax1.set_title('histogram2d')
    >>> # 在 ax1 上显示网格线
    >>> ax1.grid()

    >>> # 在 ax2 上创建 hexbin 图进行比较
    >>> ax2.hexbin(x, y, gridsize=20, cmap='rainbow')
    >>> # 绘制直线图，x 为自变量，2*np.log(x) 为因变量，线条为黑色实线
    >>> ax2.plot(x, 2*np.log(x), 'k-')
    >>> # 设置 ax2 的标题为 'hexbin'
    >>> ax2.set_title('hexbin')
    >>> # 设置 x 轴的范围为 x 的最小值到最大值
    >>> ax2.set_xlim(x.min(), x.max())
    >>> # 设置 x 轴标签为 'x'
    >>> ax2.set_xlabel('x')
    >>> # 在 ax2 上显示网格线
    >>> ax2.grid()

    >>> # 显示整个图形
    >>> plt.show()
    """
    from numpy import histogramdd

    # 检查 x 和 y 的长度是否相等，如果不相等则抛出 ValueError 异常
    if len(x) != len(y):
        raise ValueError('x and y must have the same length.')

    # 尝试获取 bins 的长度 N，如果 bins 不是一个数组则 N 为 1
    try:
        N = len(bins)
    except TypeError:
        N = 1

    # 如果 N 不是 1 也不是 2，则将 bins 转换为数组，并分别设置 xedges 和 yedges
    if N != 1 and N != 2:
        xedges = yedges = asarray(bins)
        bins = [xedges, yedges]
    
    # 计算二维直方图 hist 和对应的边界 edges
    hist, edges = histogramdd([x, y], bins, range, density, weights)
    # 返回直方图 hist，以及 x 和 y 的边界 edges[0] 和 edges[1]
    return hist, edges[0], edges[1]
# 设置模块为 'numpy'，这是一个装饰器，用于指定模块名称
@set_module('numpy')
# 定义函数 mask_indices，返回用于访问 (n, n) 数组的索引，基于给定的遮罩函数
def mask_indices(n, mask_func, k=0):
    """
    Return the indices to access (n, n) arrays, given a masking function.

    Assume `mask_func` is a function that, for a square array a of size
    ``(n, n)`` with a possible offset argument `k`, when called as
    ``mask_func(a, k)`` returns a new array with zeros in certain locations
    (functions like `triu` or `tril` do precisely this). Then this function
    returns the indices where the non-zero values would be located.

    Parameters
    ----------
    n : int
        The returned indices will be valid to access arrays of shape (n, n).
    mask_func : callable
        A function whose call signature is similar to that of `triu`, `tril`.
        That is, ``mask_func(x, k)`` returns a boolean array, shaped like `x`.
        `k` is an optional argument to the function.
    k : scalar
        An optional argument which is passed through to `mask_func`. Functions
        like `triu`, `tril` take a second argument that is interpreted as an
        offset.

    Returns
    -------
    indices : tuple of arrays.
        The `n` arrays of indices corresponding to the locations where
        ``mask_func(np.ones((n, n)), k)`` is True.

    See Also
    --------
    triu, tril, triu_indices, tril_indices

    Notes
    -----
    .. versionadded:: 1.4.0

    Examples
    --------
    These are the indices that would allow you to access the upper triangular
    part of any 3x3 array:

    >>> iu = np.mask_indices(3, np.triu)

    For example, if `a` is a 3x3 array:

    >>> a = np.arange(9).reshape(3, 3)
    >>> a
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    >>> a[iu]
    array([0, 1, 2, 4, 5, 8])

    An offset can be passed also to the masking function.  This gets us the
    indices starting on the first diagonal right of the main one:

    >>> iu1 = np.mask_indices(3, np.triu, 1)

    with which we now extract only three elements:

    >>> a[iu1]
    array([1, 2, 5])

    """
    # 创建一个大小为 (n, n) 的整数数组 m，所有元素为 1
    m = ones((n, n), int)
    # 调用 mask_func 函数，传入 m 和 k，返回一个布尔数组 a
    a = mask_func(m, k)
    # 返回 a 中非零元素的索引
    return nonzero(a != 0)


# 设置模块为 'numpy'，这是一个装饰器，用于指定模块名称
@set_module('numpy')
# 定义函数 tril_indices，返回一个 (n, m) 数组的下三角形部分的索引
def tril_indices(n, k=0, m=None):
    """
    Return the indices for the lower-triangle of an (n, m) array.

    Parameters
    ----------
    n : int
        The row dimension of the arrays for which the returned
        indices will be valid.
    k : int, optional
        Diagonal offset (see `tril` for details).
    m : int, optional
        .. versionadded:: 1.9.0

        The column dimension of the arrays for which the returned
        arrays will be valid.
        By default `m` is taken equal to `n`.

    Returns
    -------
    inds : tuple of arrays
        The indices for the triangle. The returned tuple contains two arrays,
        each with the indices along one dimension of the array.

    See also
    --------
    triu_indices : similar function, for upper-triangular.
    mask_indices : generic function accepting an arbitrary mask function.
    tril, triu

    Notes

    """
    # 根据文档，该函数是在 NumPy 1.4.0 版本中添加的新功能
    .. versionadded:: 1.4.0
    
    # 示例部分，展示如何计算两种不同的索引集合，用于访问4x4数组：
    # 一种是从主对角线开始的下三角部分，另一种是从右侧两个对角线开始。
    
    # 使用 np.tril_indices 函数计算出4x4数组的下三角部分的索引集合
    il1 = np.tril_indices(4)
    
    # 使用 np.tril_indices 函数计算出4x4数组中右侧两个对角线开始的索引集合
    il2 = np.tril_indices(4, 2)
    
    # 下面展示了如何将这些索引集合应用到示例数组 a 上：
    
    # 示例数组 a，形状为4x4
    a = np.arange(16).reshape(4, 4)
    >>> a
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    
    # 使用 il1 索引集合进行索引：
    >>> a[il1]
    array([ 0,  4,  5,  8,  9, 10, 12, 13, 14, 15])
    
    # 使用 il1 索引集合进行赋值：
    >>> a[il1] = -1
    >>> a
    array([[-1,  1,  2,  3],
           [-1, -1,  6,  7],
           [-1, -1, -1, 11],
           [-1, -1, -1, -1]])
    
    # 使用 il2 索引集合进行赋值，涵盖几乎整个数组（从主对角线右侧两个对角线开始）：
    >>> a[il2] = -10
    >>> a
    array([[-10, -10, -10,   3],
           [-10, -10, -10, -10],
           [-10, -10, -10, -10],
           [-10, -10, -10, -10]])
    
    """
    tri_ = tri(n, m, k=k, dtype=bool)
    
    # 返回一个元组，其中每个元素是根据布尔数组 tri_ 的形状广播后的 inds 的值
    return tuple(broadcast_to(inds, tri_.shape)[tri_]
                 for inds in indices(tri_.shape, sparse=True))
# 定义一个函数分派器，返回输入参数的元组
def _trilu_indices_form_dispatcher(arr, k=None):
    return (arr,)


# 使用装饰器将下面的函数注册到数组函数分派机制中
@array_function_dispatch(_trilu_indices_form_dispatcher)
# 定义函数 tril_indices_from，返回给定数组的下三角部分的索引
def tril_indices_from(arr, k=0):
    """
    返回数组的下三角部分的索引。

    查看 `tril_indices` 获取详细信息。

    Parameters
    ----------
    arr : array_like
        索引适用于维度与 arr 相同的方形数组。
    k : int, optional
        对角线偏移量（详见 `tril` 的说明）。

    Examples
    --------

    创建一个 4x4 的数组。

    >>> a = np.arange(16).reshape(4, 4)
    >>> a
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])

    将数组传递给函数以获取下三角元素的索引。

    >>> trili = np.tril_indices_from(a)
    >>> trili
    (array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3]), array([0, 0, 1, 0, 1, 2, 0, 1, 2, 3]))

    >>> a[trili]
    array([ 0,  4,  5,  8,  9, 10, 12, 13, 14, 15])

    这是 `tril_indices()` 的语法糖。

    >>> np.tril_indices(a.shape[0])
    (array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3]), array([0, 0, 1, 0, 1, 2, 0, 1, 2, 3]))

    使用 `k` 参数来返回到第 k 条对角线的下三角数组的索引。

    >>> trili1 = np.tril_indices_from(a, k=1)
    >>> a[trili1]
    array([ 0,  1,  4,  5,  6,  8,  9, 10, 11, 12, 13, 14, 15])

    See Also
    --------
    tril_indices, tril, triu_indices_from

    Notes
    -----
    .. versionadded:: 1.4.0

    """
    # 如果输入数组不是二维的，抛出 ValueError 异常
    if arr.ndim != 2:
        raise ValueError("input array must be 2-d")
    # 返回调用 tril_indices 函数得到的结果，指定数组的行数和列数以及对角线偏移量
    return tril_indices(arr.shape[-2], k=k, m=arr.shape[-1])


# 将下面的函数设置为 numpy 模块的一部分
@set_module('numpy')
# 定义 triu_indices 函数，返回一个 (n, m) 数组的上三角部分的索引
def triu_indices(n, k=0, m=None):
    """
    返回一个 (n, m) 数组的上三角部分的索引。

    Parameters
    ----------
    n : int
        返回的索引适用于数组的大小。
    k : int, optional
        对角线偏移量（详见 `triu` 的说明）。
    m : int, optional
        .. versionadded:: 1.9.0

        返回的数组的列维度。
        默认情况下，`m` 等于 `n`。

    Returns
    -------
    inds : tuple, shape(2) of ndarrays, shape(`n`)
        三角形的索引。返回的元组包含两个数组，每个数组沿数组的一个维度包含索引。可用于切片形状为 (`n`, `n`) 的 ndarray。

    See also
    --------
    tril_indices : 类似的函数，用于下三角。
    mask_indices : 接受任意掩码函数的通用函数。
    triu, tril

    Notes
    -----
    .. versionadded:: 1.4.0

    Examples
    --------
    计算两组不同的索引以访问 4x4 数组，一组从主对角线开始为上三角部分，另一组从右边开始两个对角线更远的地方：

    >>> iu1 = np.triu_indices(4)
    >>> iu2 = np.triu_indices(4, 2)

    """
    # 此处省略函数主体，因为它只包含了文档字符串和一些函数签名，没有实际的代码实现
    # 创建一个布尔类型的三角形掩码，表示矩阵中不同的三角形区域
    tri_ = ~tri(n, m, k=k - 1, dtype=bool)

    # 返回一个元组，包含通过广播和掩码选择的索引的稀疏矩阵中的元素
    return tuple(broadcast_to(inds, tri_.shape)[tri_]
                 for inds in indices(tri_.shape, sparse=True))
# 使用装饰器来分派函数，根据数组的形式调度
@array_function_dispatch(_trilu_indices_form_dispatcher)
# 定义一个函数，返回给定数组的上三角部分的索引
def triu_indices_from(arr, k=0):
    """
    Return the indices for the upper-triangle of arr.

    See `triu_indices` for full details.

    Parameters
    ----------
    arr : ndarray, shape(N, N)
        The indices will be valid for square arrays.
        输入数组，应为方形数组，返回的索引对其有效。
    k : int, optional
        Diagonal offset (see `triu` for details).
        对角线的偏移量（参见 `triu` 以了解详情）。

    Returns
    -------
    triu_indices_from : tuple, shape(2) of ndarray, shape(N)
        Indices for the upper-triangle of `arr`.
        返回数组 `arr` 上三角部分的索引。

    Examples
    --------

    Create a 4 by 4 array.

    >>> a = np.arange(16).reshape(4, 4)
    >>> a
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])

    Pass the array to get the indices of the upper triangular elements.

    >>> triui = np.triu_indices_from(a)
    >>> triui
    (array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3]), array([0, 1, 2, 3, 1, 2, 3, 2, 3, 3]))

    >>> a[triui]
    array([ 0,  1,  2,  3,  5,  6,  7, 10, 11, 15])

    This is syntactic sugar for triu_indices().

    >>> np.triu_indices(a.shape[0])
    (array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3]), array([0, 1, 2, 3, 1, 2, 3, 2, 3, 3]))

    Use the `k` parameter to return the indices for the upper triangular array
    from the k-th diagonal.

    >>> triuim1 = np.triu_indices_from(a, k=1)
    >>> a[triuim1]
    array([ 1,  2,  3,  6,  7, 11])


    See Also
    --------
    triu_indices, triu, tril_indices_from

    Notes
    -----
    .. versionadded:: 1.4.0

    """
    # 如果输入数组的维度不是2，抛出值错误
    if arr.ndim != 2:
        raise ValueError("input array must be 2-d")
    # 返回调用 triu_indices 函数得到的结果，传递给它数组的形状及 k 参数
    return triu_indices(arr.shape[-2], k=k, m=arr.shape[-1])
```