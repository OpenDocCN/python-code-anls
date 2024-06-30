# `D:\src\scipysrc\scipy\scipy\signal\_savitzky_golay.py`

```
import numpy as np
from scipy.linalg import lstsq
from scipy._lib._util import float_factorial
from scipy.ndimage import convolve1d
from ._arraytools import axis_slice

# 计算一维 Savitzky-Golay FIR 滤波器的系数
def savgol_coeffs(window_length, polyorder, deriv=0, delta=1.0, pos=None,
                  use="conv"):
    """Compute the coefficients for a 1-D Savitzky-Golay FIR filter.

    Parameters
    ----------
    window_length : int
        The length of the filter window (i.e., the number of coefficients).
    polyorder : int
        The order of the polynomial used to fit the samples.
        `polyorder` must be less than `window_length`.
    deriv : int, optional
        The order of the derivative to compute. This must be a
        nonnegative integer. The default is 0, which means to filter
        the data without differentiating.
    delta : float, optional
        The spacing of the samples to which the filter will be applied.
        This is only used if deriv > 0.
    pos : int or None, optional
        If pos is not None, it specifies evaluation position within the
        window. The default is the middle of the window.
    use : str, optional
        Either 'conv' or 'dot'. This argument chooses the order of the
        coefficients. The default is 'conv', which means that the
        coefficients are ordered to be used in a convolution. With
        use='dot', the order is reversed, so the filter is applied by
        dotting the coefficients with the data set.

    Returns
    -------
    coeffs : 1-D ndarray
        The filter coefficients.

    See Also
    --------
    savgol_filter

    Notes
    -----
    .. versionadded:: 0.14.0

    References
    ----------
    A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of Data by
    Simplified Least Squares Procedures. Analytical Chemistry, 1964, 36 (8),
    pp 1627-1639.
    Jianwen Luo, Kui Ying, and Jing Bai. 2005. Savitzky-Golay smoothing and
    differentiation filter for even number data. Signal Process.
    85, 7 (July 2005), 1429-1434.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.signal import savgol_coeffs
    >>> savgol_coeffs(5, 2)
    array([-0.08571429,  0.34285714,  0.48571429,  0.34285714, -0.08571429])
    >>> savgol_coeffs(5, 2, deriv=1)
    array([ 2.00000000e-01,  1.00000000e-01,  2.07548111e-16, -1.00000000e-01,
           -2.00000000e-01])

    Note that use='dot' simply reverses the coefficients.

    >>> savgol_coeffs(5, 2, pos=3)
    array([ 0.25714286,  0.37142857,  0.34285714,  0.17142857, -0.14285714])
    >>> savgol_coeffs(5, 2, pos=3, use='dot')
    array([-0.14285714,  0.17142857,  0.34285714,  0.37142857,  0.25714286])
    >>> savgol_coeffs(4, 2, pos=3, deriv=1, use='dot')
    array([0.45,  -0.85,  -0.65,  1.05])

    `x` contains data from the parabola x = t**2, sampled at
    t = -1, 0, 1, 2, 3.  `c` holds the coefficients that will compute the
    derivative at the last position.  When dotted with `x` the result should
    be 6.
    """
    # 根据参数创建滤波器的系数
    if use == "conv":
        # 使用卷积方式计算系数
        return axis_slice(convolve1d(np.eye(window_length), savgol_polynomial(polyorder, deriv, delta, pos)),
                          axis=0, start=0, stop=window_length)
    elif use == "dot":
        # 使用点积方式计算系数
        return savgol_polynomial(polyorder, deriv, delta, pos)[::-1]
    else:
        raise ValueError("`use` must be 'conv' or 'dot'.")

def savgol_polynomial(polyorder, deriv=0, delta=1.0, pos=None):
    """Compute the polynomial coefficients for a Savitzky-Golay filter.

    Parameters
    ----------
    polyorder : int
        The order of the polynomial to use.
    deriv : int, optional
        The order of the derivative to compute (default is 0).
    delta : float, optional
        The sample spacing (default is 1.0).
    pos : int or None, optional
        The position within the window to evaluate (default is None, which
        means the center of the window).

    Returns
    -------
    ndarray
        The polynomial coefficients.

    Notes
    -----
    This function is used internally by `savgol_coeffs` to compute filter
    coefficients.
    """
    # 计算 Savitzky-Golay 滤波器的多项式系数
    if pos is None:
        pos = (polyorder + 1) // 2
    if deriv > 0:
        return float_factorial(deriv) * np.polyval(np.polyder(np.poly1d(np.array(
                [((-1) ** k * float_factorial(2 * k + 2)) / (delta ** (2 * k + 2))
                 for k in range(polyorder + 1)]))), np.arange(-pos, polyorder - pos + 1))
    else:
        return np.poly1d(np.array([((-1) ** k * float_factorial(2 * k)) / (delta ** (2 * k))
                 for k in range(polyorder + 1)]))[::-1]
    # 创建一个 numpy 数组，包含指定的数据
    x = np.array([1, 0, 1, 4, 9])
    # 使用指定的参数调用 savgol_coeffs 函数，返回系数向量 c
    c = savgol_coeffs(5, 2, pos=4, deriv=1, use='dot')
    # 计算向量 c 和向量 x 的点积，返回结果
    c.dot(x)
    # 返回结果应为 6.0
    6.0
    """

    # 如果 deriv=0，则使用以下方法查找系数的替代方法：
    #    t = np.arange(window_length)
    #    unit = (t == pos).astype(int)
    #    coeffs = np.polyval(np.polyfit(t, unit, polyorder), t)
    # 这里实现的方法更快。

    # 要重新创建 Numerical Recipes 书中 Savitzy-Golay 滤波器章节中显示的示例系数表格，请使用：
    #    window_length = nL + nR + 1
    #    pos = nL + 1
    #    c = savgol_coeffs(window_length, M, pos=pos, use='dot')

    # 如果 polyorder 大于等于 window_length，则抛出 ValueError 异常
    if polyorder >= window_length:
        raise ValueError("polyorder must be less than window_length.")

    # 计算 window_length 的一半和余数
    halflen, rem = divmod(window_length, 2)

    # 如果 pos 为 None，则根据 window_length 的奇偶性选择合适的 pos 值
    if pos is None:
        if rem == 0:
            pos = halflen - 0.5
        else:
            pos = halflen

    # 如果 pos 不在 [0, window_length) 范围内，则抛出 ValueError 异常
    if not (0 <= pos < window_length):
        raise ValueError("pos must be nonnegative and less than "
                         "window_length.")

    # 如果 use 不是 'conv' 或 'dot'，则抛出 ValueError 异常
    if use not in ['conv', 'dot']:
        raise ValueError("`use` must be 'conv' or 'dot'")

    # 如果 deriv 大于 polyorder，则返回全零系数向量
    if deriv > polyorder:
        coeffs = np.zeros(window_length)
        return coeffs

    # 构建设计矩阵 A。A 的列是从 -pos 到 window_length - pos - 1 的整数幂。
    # 幂的范围从 0 到 polyorder。（即，A 是一个 Vandermonde 矩阵，但不一定是方阵。）
    x = np.arange(-pos, window_length - pos, dtype=float)

    # 如果 use 是 "conv"，则将 x 反转，以便在卷积中使用
    if use == "conv":
        x = x[::-1]

    # 创建 polyorder+1 行的二维数组，其中每行包含从 0 到 polyorder 的整数
    order = np.arange(polyorder + 1).reshape(-1, 1)
    A = x ** order

    # y 确定返回哪个阶数的导数
    y = np.zeros(polyorder + 1)
    # y[deriv] 的系数缩放结果，以考虑导数的阶数和样本间距
    y[deriv] = float_factorial(deriv) / (delta ** deriv)

    # 求解最小二乘法问题 A*c = y，返回系数向量 coeffs
    coeffs, _, _, _ = lstsq(A, y)

    return coeffs
# Differentiate polynomials represented with coefficients.
# p must be a 1-D or 2-D array. In the 2-D case, each column gives
# the coefficients of a polynomial; the first row holds the coefficients
# associated with the highest power. m must be a nonnegative integer.
# (numpy.polyder doesn't handle the 2-D case.)
def _polyder(p, m):
    if m == 0:
        # If no differentiation is needed, return the original polynomial coefficients.
        result = p
    else:
        n = len(p)
        if n <= m:
            # If the order of differentiation is greater than or equal to the number of coefficients,
            # return a zero array of the appropriate shape.
            result = np.zeros_like(p[:1, ...])
        else:
            # Perform polynomial differentiation by taking successive differences.
            dp = p[:-m].copy()
            for k in range(m):
                # Generate the factorial-like range for differentiation.
                rng = np.arange(n - k - 1, m - k - 1, -1)
                dp *= rng.reshape((n - m,) + (1,) * (p.ndim - 1))
            result = dp
    return result


# Given an N-d array `x` and the specification of a slice of `x` from
# `window_start` to `window_stop` along `axis`, create an interpolating
# polynomial of each 1-D slice, and evaluate that polynomial in the slice
# from `interp_start` to `interp_stop`. Put the result into the
# corresponding slice of `y`.
def _fit_edge(x, window_start, window_stop, interp_start, interp_stop,
              axis, polyorder, deriv, delta, y):
    # Get the edge into a (window_length, -1) array.
    x_edge = axis_slice(x, start=window_start, stop=window_stop, axis=axis)
    
    if axis == 0 or axis == -x.ndim:
        xx_edge = x_edge
        swapped = False
    else:
        xx_edge = x_edge.swapaxes(axis, 0)
        swapped = True
    
    # Reshape to prepare for polynomial fitting.
    xx_edge = xx_edge.reshape(xx_edge.shape[0], -1)
    
    # Fit the edges. poly_coeffs has shape (polyorder + 1, -1),
    # where '-1' is the same as in xx_edge.
    poly_coeffs = np.polyfit(np.arange(0, window_stop - window_start),
                             xx_edge, polyorder)
    
    if deriv > 0:
        # If derivative order > 0, differentiate the polynomial coefficients.
        poly_coeffs = _polyder(poly_coeffs, deriv)
    
    # Compute the interpolated values for the edge.
    i = np.arange(interp_start - window_start, interp_stop - window_start)
    values = np.polyval(poly_coeffs, i.reshape(-1, 1)) / (delta ** deriv)
    
    # Reshape values to match y and insert into the appropriate slice of y.
    shp = list(y.shape)
    shp[0], shp[axis] = shp[axis], shp[0]
    values = values.reshape(interp_stop - interp_start, *shp[1:])
    
    if swapped:
        values = values.swapaxes(0, axis)
    
    # Get a view of the data to be replaced by values.
    y_edge = axis_slice(y, start=interp_start, stop=interp_stop, axis=axis)
    y_edge[...] = values


# Use polynomial interpolation of x at the low and high ends of the axis
# to fill in the halflen values in y.
# This function just calls _fit_edge twice, once for each end of the axis.
def _fit_edges_polyfit(x, window_length, polyorder, deriv, delta, axis, y):
    halflen = window_length // 2
    _fit_edge(x, 0, window_length, 0, halflen, axis,
              polyorder, deriv, delta, y)
    n = x.shape[axis]
    # 调用 _fit_edge 函数，该函数用于边缘处理或拟合操作
    _fit_edge(x, n - window_length, n, n - halflen, n, axis,
              polyorder, deriv, delta, y)
def savgol_filter(x, window_length, polyorder, deriv=0, delta=1.0,
                  axis=-1, mode='interp', cval=0.0):
    """ Apply a Savitzky-Golay filter to an array.

    This is a 1-D filter. If `x`  has dimension greater than 1, `axis`
    determines the axis along which the filter is applied.

    Parameters
    ----------
    x : array_like
        The data to be filtered. If `x` is not a single or double precision
        floating point array, it will be converted to type ``numpy.float64``
        before filtering.
    window_length : int
        The length of the filter window (i.e., the number of coefficients).
        If `mode` is 'interp', `window_length` must be less than or equal
        to the size of `x`.
    polyorder : int
        The order of the polynomial used to fit the samples.
        `polyorder` must be less than `window_length`.
    deriv : int, optional
        The order of the derivative to compute. This must be a
        nonnegative integer. The default is 0, which means to filter
        the data without differentiating.
    delta : float, optional
        The spacing of the samples to which the filter will be applied.
        This is only used if deriv > 0. Default is 1.0.
    axis : int, optional
        The axis of the array `x` along which the filter is to be applied.
        Default is -1.
    mode : str, optional
        Must be 'mirror', 'constant', 'nearest', 'wrap' or 'interp'. This
        determines the type of extension to use for the padded signal to
        which the filter is applied.  When `mode` is 'constant', the padding
        value is given by `cval`.  See the Notes for more details on 'mirror',
        'constant', 'wrap', and 'nearest'.
        When the 'interp' mode is selected (the default), no extension
        is used.  Instead, a degree `polyorder` polynomial is fit to the
        last `window_length` values of the edges, and this polynomial is
        used to evaluate the last `window_length // 2` output values.
    cval : scalar, optional
        Value to fill past the edges of the input if `mode` is 'constant'.
        Default is 0.0.

    Returns
    -------
    y : ndarray, same shape as `x`
        The filtered data.

    See Also
    --------
    savgol_coeffs

    Notes
    -----
    Details on the `mode` options:

        'mirror':
            Repeats the values at the edges in reverse order. The value
            closest to the edge is not included.
        'nearest':
            The extension contains the nearest input value.
        'constant':
            The extension contains the value given by the `cval` argument.
        'wrap':
            The extension contains the values from the other end of the array.

    For example, if the input is [1, 2, 3, 4, 5, 6, 7, 8], and
    `window_length` is 7, the following shows the extended data for

    """
    # 转换输入数据为 numpy.float64 类型，以便进行滤波处理
    from scipy.signal import savgol_filter
    # 应用 Savitzky-Golay 滤波器进行数据滤波
    return savgol_filter(x, window_length, polyorder, deriv, delta,
                         axis, mode, cval)
    # 如果给定的 mode 不在预定义的选项列表 ["mirror", "constant", "nearest", "interp", "wrap"] 中，抛出 ValueError 异常。
    if mode not in ["mirror", "constant", "nearest", "interp", "wrap"]:
        raise ValueError("mode must be 'mirror', 'constant', 'nearest' "
                         "'wrap' or 'interp'.")
    
    # 将输入数组 x 转换为 NumPy 数组，确保其数据类型为单精度或双精度浮点数。
    x = np.asarray(x)
    if x.dtype != np.float64 and x.dtype != np.float32:
        x = x.astype(np.float64)
    
    # 调用 savgol_coeffs 函数，获取 Savitzky-Golay 滤波器的系数。
    coeffs = savgol_coeffs(window_length, polyorder, deriv=deriv, delta=delta)
    
    # 如果 mode 是 "interp"，则需要进行特殊处理。
    if mode == "interp":
        # 如果窗口长度 window_length 大于数组 x 沿指定轴的尺寸，则引发 ValueError 异常。
        if window_length > x.shape[axis]:
            raise ValueError("If mode is 'interp', window_length must be less "
                             "than or equal to the size of x.")
        
        # 在 'interp' 模式下，不填充边界。而是对序列末尾的 window_length // 2 范围内的元素使用拟合到最后 window_length 个元素的多项式。
        y = convolve1d(x, coeffs, axis=axis, mode="constant")
        _fit_edges_polyfit(x, window_length, polyorder, deriv, delta, axis, y)
    else:
        # 对除了 'interp' 之外的任何模式，都直接传递给 ndimage.convolve1d 函数进行卷积操作。
        y = convolve1d(x, coeffs, axis=axis, mode=mode, cval=cval)
    
    # 返回滤波后的结果数组 y。
    return y
```