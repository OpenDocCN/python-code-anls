# `D:\src\scipysrc\scipy\scipy\interpolate\_cubic.py`

```
"""Interpolation algorithms using piecewise cubic polynomials."""

# 导入必要的模块和类
from __future__ import annotations  # 用于支持类型注解的导入方式

from typing import TYPE_CHECKING  # 导入类型检查相关模块

import numpy as np  # 导入数值计算库numpy

from scipy.linalg import solve, solve_banded  # 导入求解线性方程组的函数

from . import PPoly  # 导入PPoly类，用于表示分段多项式
from ._polyint import _isscalar  # 导入_isscalar函数，用于判断是否为标量

if TYPE_CHECKING:
    from typing import Literal  # 引入Literal类型，用于指定字面量类型

__all__ = ["CubicHermiteSpline", "PchipInterpolator", "pchip_interpolate",
           "Akima1DInterpolator", "CubicSpline"]

# 函数定义，用于为三次样条插值器准备输入数据
def prepare_input(x, y, axis, dydx=None):
    """Prepare input for cubic spline interpolators.

    All data are converted to numpy arrays and checked for correctness.
    Axes equal to `axis` of arrays `y` and `dydx` are moved to be the 0th
    axis. The value of `axis` is converted to lie in
    [0, number of dimensions of `y`).

    Parameters
    ----------
    x : array_like
        1-D array containing values of the independent variable.
    y : array_like
        Array containing values of the dependent variable.
    axis : int
        Specifies the axis along which the interpolation is performed.
    dydx : array_like, optional
        Array containing values of the derivative of `y` with respect to `x`.

    Returns
    -------
    x : ndarray
        Converted 1-D numpy array of `x`.
    dx : ndarray
        Array of differences `dx` computed from `x`.
    y : ndarray
        Converted numpy array of `y`.
    axis : int
        Converted axis value.
    dydx : ndarray or None
        Converted numpy array of `dydx` or None if not provided.

    Raises
    ------
    ValueError
        If input arrays have incorrect shapes, types, or values.

    Notes
    -----
    This function ensures all inputs are correctly formatted for cubic spline
    interpolation algorithms. It checks for finite values, matching dimensions,
    and strictly increasing values in `x`.

    """
    # 将输入的x和y转换为numpy数组，并检查其正确性
    x, y = map(np.asarray, (x, y))

    # 检查x是否包含实数值
    if np.issubdtype(x.dtype, np.complexfloating):
        raise ValueError("`x` must contain real values.")
    x = x.astype(float)

    # 检查y是否包含复数值
    if np.issubdtype(y.dtype, np.complexfloating):
        dtype = complex
    else:
        dtype = float

    # 如果提供了dydx，则转换为numpy数组，并检查其形状是否与y相同
    if dydx is not None:
        dydx = np.asarray(dydx)
        if y.shape != dydx.shape:
            raise ValueError("The shapes of `y` and `dydx` must be identical.")
        if np.issubdtype(dydx.dtype, np.complexfloating):
            dtype = complex
        dydx = dydx.astype(dtype, copy=False)

    # 将y的指定轴(axis)移动到0轴，以便后续处理
    y = y.astype(dtype, copy=False)
    axis = axis % y.ndim

    # 检查x是否为1维数组
    if x.ndim != 1:
        raise ValueError("`x` must be 1-dimensional.")
    
    # 检查x至少包含两个元素
    if x.shape[0] < 2:
        raise ValueError("`x` must contain at least 2 elements.")
    
    # 检查y在指定轴(axis)上的长度是否与x相同
    if x.shape[0] != y.shape[axis]:
        raise ValueError(f"The length of `y` along `axis`={axis} doesn't "
                         "match the length of `x`")

    # 检查x、y和dydx是否包含有限值
    if not np.all(np.isfinite(x)):
        raise ValueError("`x` must contain only finite values.")
    if not np.all(np.isfinite(y)):
        raise ValueError("`y` must contain only finite values.")

    if dydx is not None and not np.all(np.isfinite(dydx)):
        raise ValueError("`dydx` must contain only finite values.")

    # 检查x是否为严格递增的序列
    dx = np.diff(x)
    if np.any(dx <= 0):
        raise ValueError("`x` must be strictly increasing sequence.")

    # 将y和dydx的指定轴(axis)移动到0轴，以便后续处理
    y = np.moveaxis(y, axis, 0)
    if dydx is not None:
        dydx = np.moveaxis(dydx, axis, 0)

    # 返回处理后的数组和参数
    return x, dx, y, axis, dydx


class CubicHermiteSpline(PPoly):
    """Piecewise-cubic interpolator matching values and first derivatives.

    The result is represented as a `PPoly` instance.

    Parameters
    ----------
    x : array_like, shape (n,)
        1-D array containing values of the independent variable.
        Values must be real, finite and in strictly increasing order.
    y : array_like
        Array containing values of the dependent variable. It can have
        arbitrary number of dimensions, but the length along ``axis``
        (see below) must match the length of ``x``. Values must be finite.
    """
    
    # 省略了构造函数及其说明部分，因为不在代码块内
    dydx : array_like
        包含依赖变量的导数的数组。它可以有任意多的维度，但沿着“axis”（见下文）的长度必须与“x”的长度匹配。值必须是有限的。
    axis : int, optional
        假设“y”变化的轴。这意味着对于“x[i]”，相应的值是“np.take(y, i, axis=axis)”。
        默认为0。
    extrapolate : {bool, 'periodic', None}, optional
        如果是bool类型，则确定是否根据第一个和最后一个区间对超出边界的点进行外推，或者返回NaN。如果是'periodic'，则使用周期性外推。如果是None（默认），则设置为True。

    Attributes
    ----------
    x : ndarray, shape (n,)
        断点。与构造函数中传递的相同的“x”。
    c : ndarray, shape (4, n-1, ...)
        每段多项式的系数。尾随的维度与“y”的维度匹配，不包括“axis”。
        例如，如果“y”是1-D，则“c[k, i]”是在“x[i]”和“x[i+1]”之间的段上“(x-x[i])**(3-k)”的系数。
    axis : int
        插值轴。与构造函数中传递的相同的轴。

    Methods
    -------
    __call__
        调用对象以执行插值。
    derivative
        计算导数。
    antiderivative
        计算反导数。
    integrate
        计算积分。
    roots
        计算多项式的根。

    See Also
    --------
    Akima1DInterpolator : Akima 1D 插值器。
    PchipInterpolator : PCHIP 1-D 单调立方插值器。
    CubicSpline : 三次样条数据插值器。
    PPoly : 分段多项式，由系数和断点定义。

    Notes
    -----
    如果要创建匹配更高阶导数的高阶样条曲线，请使用 `BPoly.from_derivatives`。

    References
    ----------
    .. [1] `三次埃尔米特样条
            <https://en.wikipedia.org/wiki/Cubic_Hermite_spline>`_
            在维基百科上的介绍。
    """

    def __init__(self, x, y, dydx, axis=0, extrapolate=None):
        if extrapolate is None:
            extrapolate = True

        x, dx, y, axis, dydx = prepare_input(x, y, axis, dydx)

        dxr = dx.reshape([dx.shape[0]] + [1] * (y.ndim - 1))
        slope = np.diff(y, axis=0) / dxr
        t = (dydx[:-1] + dydx[1:] - 2 * slope) / dxr

        c = np.empty((4, len(x) - 1) + y.shape[1:], dtype=t.dtype)
        c[0] = t / dxr
        c[1] = (slope - dydx[:-1]) / dxr - t
        c[2] = dydx[:-1]
        c[3] = y[:-1]

        super().__init__(c, x, extrapolate=extrapolate)
        self.axis = axis
class PchipInterpolator(CubicHermiteSpline):
    r"""PCHIP 1-D monotonic cubic interpolation.

    ``x`` and ``y`` are arrays of values used to approximate some function f,
    with ``y = f(x)``. The interpolant uses monotonic cubic splines
    to find the value of new points. (PCHIP stands for Piecewise Cubic
    Hermite Interpolating Polynomial).

    Parameters
    ----------
    x : ndarray, shape (npoints, )
        A 1-D array of monotonically increasing real values. ``x`` cannot
        include duplicate values (otherwise f is overspecified)
    y : ndarray, shape (..., npoints, ...)
        A N-D array of real values. ``y``'s length along the interpolation
        axis must be equal to the length of ``x``. Use the ``axis``
        parameter to select the interpolation axis.
    axis : int, optional
        Axis in the ``y`` array corresponding to the x-coordinate values. Defaults
        to ``axis=0``.
    extrapolate : bool, optional
        Whether to extrapolate to out-of-bounds points based on first
        and last intervals, or to return NaNs.

    Methods
    -------
    __call__
        Evaluate the interpolator at new points.
    derivative
        Compute the first derivative of the interpolant.
    antiderivative
        Compute the integral (antiderivative) of the interpolant.
    roots
        Find the roots (zero-crossings) of the interpolant.

    See Also
    --------
    CubicHermiteSpline : Piecewise-cubic interpolator.
    Akima1DInterpolator : Akima 1D interpolator.
    CubicSpline : Cubic spline data interpolator.
    PPoly : Piecewise polynomial in terms of coefficients and breakpoints.

    Notes
    -----
    The interpolator preserves monotonicity in the interpolation data and does
    not overshoot if the data is not smooth.

    The first derivatives are guaranteed to be continuous, but the second
    derivatives may jump at :math:`x_k`.

    Determines the derivatives at the points :math:`x_k`, :math:`f'_k`,
    by using PCHIP algorithm [1]_.

    Let :math:`h_k = x_{k+1} - x_k`, and  :math:`d_k = (y_{k+1} - y_k) / h_k`
    are the slopes at internal points :math:`x_k`.
    If the signs of :math:`d_k` and :math:`d_{k-1}` are different or either of
    them equals zero, then :math:`f'_k = 0`. Otherwise, it is given by the
    weighted harmonic mean

    .. math::

        \frac{w_1 + w_2}{f'_k} = \frac{w_1}{d_{k-1}} + \frac{w_2}{d_k}

    where :math:`w_1 = 2 h_k + h_{k-1}` and :math:`w_2 = h_k + 2 h_{k-1}`.

    The end slopes are set using a one-sided scheme [2]_.


    References
    ----------
    .. [1] F. N. Fritsch and J. Butland,
           A method for constructing local
           monotone piecewise cubic interpolants,
           SIAM J. Sci. Comput., 5(2), 300-304 (1984).
           :doi:`10.1137/0905021`.
    .. [2] see, e.g., C. Moler, Numerical Computing with Matlab, 2004.
           :doi:`10.1137/1.9780898717952`

    """
    def __init__(self, x, y, axis=0, extrapolate=None):
        # 准备输入数据，获取经处理后的输入变量
        x, _, y, axis, _ = prepare_input(x, y, axis)
        
        # 检查 y 是否为复数类型，如果是则抛出异常
        if np.iscomplexobj(y):
            msg = ("`PchipInterpolator` only works with real values for `y`. "
                   "If you are trying to use the real components of the passed array, "
                   "use `np.real` on the array before passing to `PchipInterpolator`.")
            raise ValueError(msg)
        
        # 将 x 重塑为符合 y 维度的形状
        xp = x.reshape((x.shape[0],) + (1,)*(y.ndim-1))
        
        # 调用 _find_derivatives 方法计算导数 dk
        dk = self._find_derivatives(xp, y)
        
        # 调用父类的初始化方法，传递 x, y, dk 以及其他参数
        super().__init__(x, y, dk, axis=0, extrapolate=extrapolate)
        
        # 设置对象的 axis 属性
        self.axis = axis

    @staticmethod
    def _edge_case(h0, h1, m0, m1):
        # 单侧三点估计法计算导数
        d = ((2*h0 + h1)*m0 - h0*m1) / (h0 + h1)

        # 尝试保持形状一致性
        mask = np.sign(d) != np.sign(m0)
        mask2 = (np.sign(m0) != np.sign(m1)) & (np.abs(d) > 3.*np.abs(m0))
        mmm = (~mask) & mask2

        d[mask] = 0.
        d[mmm] = 3.*m0[mmm]

        return d

    @staticmethod
    def _find_derivatives(x, y):
        # 计算在点 y_k 处的导数 d_k，使用 PCHIP 算法：
        # 我们通过如下方式选择 x_k 处的导数：
        # 令 m_k 为第 k 段（在 k 和 k+1 之间）的斜率
        # 如果 m_k=0 或者 m_{k-1}=0 或者 sgn(m_k) != sgn(m_{k-1}) 则 d_k == 0
        # 否则使用加权调和平均数：
        #   w_1 = 2h_k + h_{k-1}, w_2 = h_k + 2h_{k-1}
        #   1/d_k = 1/(w_1 + w_2)*(w_1 / m_k + w_2 / m_{k-1})
        # 其中 h_k 是 x_k 和 x_{k+1} 之间的间距
        y_shape = y.shape
        
        # 如果 y 是一维数组，则扩展为二维数组，以避免 _edge_case 方法将值分配给标量
        if y.ndim == 1:
            x = x[:, None]
            y = y[:, None]

        # 计算相邻点之间的距离 hk 和斜率 mk
        hk = x[1:] - x[:-1]
        mk = (y[1:] - y[:-1]) / hk

        # 处理特殊情况：只有两个点的情况，使用线性插值
        if y.shape[0] == 2:
            dk = np.zeros_like(y)
            dk[0] = mk
            dk[1] = mk
            return dk.reshape(y_shape)

        # 计算斜率的符号
        smk = np.sign(mk)
        
        # 判断条件，用于确定是否应该将导数设为零
        condition = (smk[1:] != smk[:-1]) | (mk[1:] == 0) | (mk[:-1] == 0)

        # 计算加权调和平均数的分母 w1 + w2
        w1 = 2*hk[1:] + hk[:-1]
        w2 = hk[1:] + 2*hk[:-1]

        # 忽略除以零的错误，将在后续通过 'condition' 排除相关值
        with np.errstate(divide='ignore', invalid='ignore'):
            whmean = (w1/mk[:-1] + w2/mk[1:]) / (w1 + w2)

        # 初始化导数数组 dk
        dk = np.zeros_like(y)
        
        # 根据条件设置导数值
        dk[1:-1][condition] = 0.0
        dk[1:-1][~condition] = 1.0 / whmean[~condition]

        # 处理端点特殊情况，参考 Cleve Moler 的建议
        dk[0] = PchipInterpolator._edge_case(hk[0], hk[1], mk[0], mk[1])
        dk[-1] = PchipInterpolator._edge_case(hk[-1], hk[-2], mk[-1], mk[-2])

        return dk.reshape(y_shape)
# 引入所需的类和函数
from scipy.interpolate import PchipInterpolator, CubicHermiteSpline

# 定义一个便捷函数，使用 PCHIP 插值法进行插值
def pchip_interpolate(xi, yi, x, der=0, axis=0):
    """
    Convenience function for pchip interpolation.

    xi and yi are arrays of values used to approximate some function f,
    with ``yi = f(xi)``. The interpolant uses monotonic cubic splines
    to find the value of new points x and the derivatives there.

    See `scipy.interpolate.PchipInterpolator` for details.

    Parameters
    ----------
    xi : array_like
        A sorted list of x-coordinates, of length N.
    yi : array_like
        A 1-D array of real values. `yi`'s length along the interpolation
        axis must be equal to the length of `xi`. If N-D array, use axis
        parameter to select correct axis.

        .. deprecated:: 1.13.0
            Complex data is deprecated and will raise an error in
            SciPy 1.15.0. If you are trying to use the real components of
            the passed array, use ``np.real`` on `yi`.

    x : scalar or array_like
        Of length M.
    der : int or list, optional
        Derivatives to extract. The 0th derivative can be included to
        return the function value.
    axis : int, optional
        Axis in the yi array corresponding to the x-coordinate values.

    Returns
    -------
    y : scalar or array_like
        The result, of length R or length M or M by R.

    See Also
    --------
    PchipInterpolator : PCHIP 1-D monotonic cubic interpolator.

    Examples
    --------
    We can interpolate 2D observed data using pchip interpolation:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.interpolate import pchip_interpolate
    >>> x_observed = np.linspace(0.0, 10.0, 11)
    >>> y_observed = np.sin(x_observed)
    >>> x = np.linspace(min(x_observed), max(x_observed), num=100)
    >>> y = pchip_interpolate(x_observed, y_observed, x)
    >>> plt.plot(x_observed, y_observed, "o", label="observation")
    >>> plt.plot(x, y, label="pchip interpolation")
    >>> plt.legend()
    >>> plt.show()

    """
    # 使用 PCHIP 插值法创建插值对象 P
    P = PchipInterpolator(xi, yi, axis=axis)

    # 根据参数 der 的不同取值，返回不同的插值结果
    if der == 0:
        return P(x)
    elif _isscalar(der):
        return P.derivative(der)(x)
    else:
        return [P.derivative(nu)(x) for nu in der]


# 定义 Akima1DInterpolator 类，继承自 CubicHermiteSpline
class Akima1DInterpolator(CubicHermiteSpline):
    r"""
    Akima interpolator

    Fit piecewise cubic polynomials, given vectors x and y. The interpolation
    method by Akima uses a continuously differentiable sub-spline built from
    piecewise cubic polynomials. The resultant curve passes through the given
    data points and will appear smooth and natural.

    Parameters
    ----------
    x : ndarray, shape (npoints, )
        1-D array of monotonically increasing real values.
    y : ndarray, shape (..., npoints, ...)
        N-D array of real values. The length of ``y`` along the interpolation axis
        must be equal to the length of ``x``. Use the ``axis`` parameter to
        select the interpolation axis.

    """
    # 空白，因为类的定义本身已经提供了描述和参数文档
    pass
    axis : int, optional
        Axis in the ``y`` array corresponding to the x-coordinate values. Defaults
        to ``axis=0``.
    method : {'akima', 'makima'}, optional
        If ``"makima"``, use the modified Akima interpolation [2]_.
        Defaults to ``"akima"``, use the Akima interpolation [1]_.

        .. versionadded:: 1.13.0

    extrapolate : {bool, None}, optional
        If bool, determines whether to extrapolate to out-of-bounds points 
        based on first and last intervals, or to return NaNs. If None, 
        ``extrapolate`` is set to False.
        
    Methods
    -------
    __call__
        Interpolates the value at a given point or array of points.
    derivative
        Computes the derivative of the interpolated function.
    antiderivative
        Computes the antiderivative (indefinite integral) of the interpolated function.
    roots
        Computes the roots of the interpolated function.

    See Also
    --------
    PchipInterpolator : PCHIP 1-D monotonic cubic interpolator.
    CubicSpline : Cubic spline data interpolator.
    PPoly : Piecewise polynomial in terms of coefficients and breakpoints

    Notes
    -----
    .. versionadded:: 0.14

    Use only for precise data, as the fitted curve passes through the given
    points exactly. This routine is useful for plotting a pleasingly smooth
    curve through a few given points for purposes of plotting.

    Let :math:`\delta_i = (y_{i+1} - y_i) / (x_{i+1} - x_i)` be the slopes of
    the interval :math:`\left[x_i, x_{i+1}\right)`. Akima's derivative at
    :math:`x_i` is defined as:

    .. math::

        d_i = \frac{w_1}{w_1 + w_2}\delta_{i-1} + \frac{w_2}{w_1 + w_2}\delta_i

    In the Akima interpolation [1]_ (``method="akima"``), the weights are:

    .. math::

        \begin{aligned}
        w_1 &= |\delta_{i+1} - \delta_i| \\
        w_2 &= |\delta_{i-1} - \delta_{i-2}|
        \end{aligned}

    In the modified Akima interpolation [2]_ (``method="makima"``),
    to eliminate overshoot and avoid edge cases of both numerator and
    denominator being equal to 0, the weights are modified as follows:

    .. math::

        \begin{align*}
        w_1 &= |\delta_{i+1} - \delta_i| + |\delta_{i+1} + \delta_i| / 2 \\
        w_2 &= |\delta_{i-1} - \delta_{i-2}| + |\delta_{i-1} + \delta_{i-2}| / 2
        \end{align*}

    Examples
    --------
    Comparison of ``method="akima"`` and ``method="makima"``:

    >>> import numpy as np
    >>> from scipy.interpolate import Akima1DInterpolator
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(1, 7, 7)
    >>> y = np.array([-1, -1, -1, 0, 1, 1, 1])
    >>> xs = np.linspace(min(x), max(x), num=100)
    >>> y_akima = Akima1DInterpolator(x, y, method="akima")(xs)
    >>> y_makima = Akima1DInterpolator(x, y, method="makima")(xs)

    >>> fig, ax = plt.subplots()
    >>> ax.plot(x, y, "o", label="data")
    >>> ax.plot(xs, y_akima, label="akima")
    >>> ax.plot(xs, y_makima, label="makima")
    >>> ax.legend()
    >>> fig.show()

    The overshoot that occurred in ``"akima"`` has been avoided in ``"makima"``.

    References
    ----------
    """
    .. [1] A new method of interpolation and smooth curve fitting based
           on local procedures. Hiroshi Akima, J. ACM, October 1970, 17(4),
           589-602. :doi:`10.1145/321607.321609`
    .. [2] Makima Piecewise Cubic Interpolation. Cleve Moler and Cosmin Ionita, 2019.
           https://blogs.mathworks.com/cleve/2019/04/29/makima-piecewise-cubic-interpolation/

    """

    # 初始化函数，用于创建 Akima1DInterpolator 对象
    def __init__(self, x, y, axis=0, *, method: Literal["akima", "makima"]="akima", 
                 extrapolate:bool | None = None):
        # 如果指定的插值方法不是 "akima" 或 "makima"，则抛出未实现错误
        if method not in {"akima", "makima"}:
            raise NotImplementedError(f"`method`={method} is unsupported.")

        # 准备输入数据，对 x 和 y 进行处理，获取轴向和间隔 dx
        x, dx, y, axis, _ = prepare_input(x, y, axis)

        # 如果 y 是复数类型，则抛出值错误，因为 Akima1DInterpolator 只适用于实数值
        if np.iscomplexobj(y):
            msg = ("`Akima1DInterpolator` only works with real values for `y`. "
                   "If you are trying to use the real components of the passed array, "
                   "use `np.real` on the array before passing to "
                   "`Akima1DInterpolator`.")
            raise ValueError(msg)

        # 默认 Akima 插值的外推设置为 False，如果未指定则为 None
        extrapolate = False if extrapolate is None else extrapolate

        # 计算断点间的斜率
        m = np.empty((x.size + 3, ) + y.shape[1:])
        dx = dx[(slice(None), ) + (None, ) * (y.ndim - 1)]
        m[2:-2] = np.diff(y, axis=0) / dx

        # 在左侧添加两个额外的点...
        m[1] = 2. * m[2] - m[3]
        m[0] = 2. * m[1] - m[2]
        # ... 和右侧添加两个额外的点
        m[-2] = 2. * m[-3] - m[-4]
        m[-1] = 2. * m[-2] - m[-3]

        # 如果 m1 == m2 != m3 == m4，说明断点处的斜率未定义，将其设置为填充值
        t = .5 * (m[3:] + m[:-3])
        # 获取斜率 t 的分母
        dm = np.abs(np.diff(m, axis=0))
        if method == "makima":
            pm = np.abs(m[1:] + m[:-1])
            f1 = dm[2:] + 0.5 * pm[2:]
            f2 = dm[:-2] + 0.5 * pm[:-2]
        else:
            f1 = dm[2:]
            f2 = dm[:-2]
        f12 = f1 + f2
        # 这些是斜率在断点处被定义的掩码：
        ind = np.nonzero(f12 > 1e-9 * np.max(f12, initial=-np.inf))
        x_ind, y_ind = ind[0], ind[1:]
        # 设置断点处的斜率
        t[ind] = (f1[ind] * m[(x_ind + 1,) + y_ind] +
                  f2[ind] * m[(x_ind + 2,) + y_ind]) / f12[ind]

        # 调用父类的初始化函数，传递处理后的参数以创建 Akima1DInterpolator 对象
        super().__init__(x, y, t, axis=0, extrapolate=extrapolate)
        # 设置对象的轴向属性
        self.axis = axis

    # 扩展函数，目前尚未实现对 1-D Akima 插值器的扩展功能
    def extend(self, c, x, right=True):
        raise NotImplementedError("Extending a 1-D Akima interpolator is not "
                                  "yet implemented")

    # 这些方法继承自 PPoly，但它们不会产生 Akima 插值器，因此只是定义了桩代码
    @classmethod
    # 定义一个类方法，用于从样条曲线的表示形式创建 Akima 插值器对象
    def from_spline(cls, tck, extrapolate=None):
        # 抛出未实现错误，因为从样条曲线创建 Akima 插值器在这个上下文中没有意义
        raise NotImplementedError("This method does not make sense for "
                                  "an Akima interpolator.")

    # 定义一个类方法，用于从 Bernstein 基础创建 Akima 插值器对象
    def from_bernstein_basis(cls, bp, extrapolate=None):
        # 抛出未实现错误，因为从 Bernstein 基础创建 Akima 插值器在这个上下文中没有意义
        raise NotImplementedError("This method does not make sense for "
                                  "an Akima interpolator.")
# 定义 CubicSpline 类，继承自 CubicHermiteSpline
class CubicSpline(CubicHermiteSpline):
    """Cubic spline data interpolator.

    Interpolate data with a piecewise cubic polynomial which is twice
    continuously differentiable [1]_. The result is represented as a `PPoly`
    instance with breakpoints matching the given data.

    Parameters
    ----------
    x : array_like, shape (n,)
        1-D array containing values of the independent variable.
        Values must be real, finite and in strictly increasing order.
    y : array_like
        Array containing values of the dependent variable. It can have
        arbitrary number of dimensions, but the length along ``axis``
        (see below) must match the length of ``x``. Values must be finite.
    axis : int, optional
        Axis along which `y` is assumed to be varying. Meaning that for
        ``x[i]`` the corresponding values are ``np.take(y, i, axis=axis)``.
        Default is 0.
    bc_type : string or 2-tuple, optional
        Boundary condition type. Two additional equations, given by the
        boundary conditions, are required to determine all coefficients of
        polynomials on each segment [2]_.

        If `bc_type` is a string, then the specified condition will be applied
        at both ends of a spline. Available conditions are:

        * 'not-a-knot' (default): The first and second segment at a curve end
          are the same polynomial. It is a good default when there is no
          information on boundary conditions.
        * 'periodic': The interpolated functions is assumed to be periodic
          of period ``x[-1] - x[0]``. The first and last value of `y` must be
          identical: ``y[0] == y[-1]``. This boundary condition will result in
          ``y'[0] == y'[-1]`` and ``y''[0] == y''[-1]``.
        * 'clamped': The first derivative at curves ends are zero. Assuming
          a 1D `y`, ``bc_type=((1, 0.0), (1, 0.0))`` is the same condition.
        * 'natural': The second derivative at curve ends are zero. Assuming
          a 1D `y`, ``bc_type=((2, 0.0), (2, 0.0))`` is the same condition.

        If `bc_type` is a 2-tuple, the first and the second value will be
        applied at the curve start and end respectively. The tuple values can
        be one of the previously mentioned strings (except 'periodic') or a
        tuple ``(order, deriv_values)`` allowing to specify arbitrary
        derivatives at curve ends:

        * `order`: the derivative order, 1 or 2.
        * `deriv_value`: array_like containing derivative values, shape must
          be the same as `y`, excluding ``axis`` dimension. For example, if
          `y` is 1-D, then `deriv_value` must be a scalar. If `y` is 3-D with
          the shape (n0, n1, n2) and axis=2, then `deriv_value` must be 2-D
          and have the shape (n0, n1).
    """
    extrapolate : {bool, 'periodic', None}, optional
        # 插值方法的可选参数，控制是否对超出边界的点进行外推
        If bool, determines whether to extrapolate to out-of-bounds points
        # 如果是布尔值，则根据第一个和最后一个间隔来进行外推，或者返回 NaN
        based on first and last intervals, or to return NaNs. If 'periodic',
        # 如果设置为 'periodic'，则使用周期性外推
        periodic extrapolation is used. If None (default), ``extrapolate`` is
        # 如果设置为 None（默认值），则对于 ``bc_type='periodic'`` 使用 'periodic'，否则使用 True
        set to 'periodic' for ``bc_type='periodic'`` and to True otherwise.

    Attributes
    ----------
    x : ndarray, shape (n,)
        # 断点数组，与构造函数中传递的 ``x`` 相同
        Breakpoints. The same ``x`` which was passed to the constructor.
    c : ndarray, shape (4, n-1, ...)
        # 每个分段多项式的系数。尾随的维度与 `y` 的维度匹配，不包括 ``axis``。
        Coefficients of the polynomials on each segment. The trailing
        dimensions match the dimensions of `y`, excluding ``axis``.
        # 例如，如果 `y` 是一维的，则 ``c[k, i]`` 是在 ``x[i]`` 和 ``x[i+1]`` 之间段上 ``(x-x[i])**(3-k)`` 的系数
        For example, if `y` is 1-d, then ``c[k, i]`` is a coefficient for
        ``(x-x[i])**(3-k)`` on the segment between ``x[i]`` and ``x[i+1]``.
    axis : int
        # 插值轴，与构造函数中传递的相同轴
        Interpolation axis. The same axis which was passed to the
        constructor.

    Methods
    -------
    __call__
        # 实例可调用方法
    derivative
        # 求导方法
    antiderivative
        # 求反导方法
    integrate
        # 积分方法
    roots
        # 求根方法

    See Also
    --------
    Akima1DInterpolator : Akima 1D interpolator.
    PchipInterpolator : PCHIP 1-D monotonic cubic interpolator.
    PPoly : Piecewise polynomial in terms of coefficients and breakpoints.

    Notes
    -----
    Parameters `bc_type` and ``extrapolate`` work independently, i.e. the
    former controls only construction of a spline, and the latter only
    evaluation.
        # 参数 `bc_type` 和 ``extrapolate`` 独立工作，即前者仅控制样条的构造，后者仅控制评估。

    When a boundary condition is 'not-a-knot' and n = 2, it is replaced by
    a condition that the first derivative is equal to the linear interpolant
    slope. When both boundary conditions are 'not-a-knot' and n = 3, the
    solution is sought as a parabola passing through given points.
        # 当边界条件为 'not-a-knot' 且 n = 2 时，用第一导数等于线性插值斜率的条件替换。
        # 当两个边界条件都是 'not-a-knot' 且 n = 3 时，解决方案被视为通过给定点的抛物线。

    When 'not-a-knot' boundary conditions is applied to both ends, the
    resulting spline will be the same as returned by `splrep` (with ``s=0``)
    and `InterpolatedUnivariateSpline`, but these two methods use a
    representation in B-spline basis.
        # 当两端应用 'not-a-knot' 边界条件时，结果样条将与 `splrep` （使用 ``s=0`` 返回）和 `InterpolatedUnivariateSpline` 相同，
        # 但这两种方法使用 B-样条基础的表示。

    .. versionadded:: 0.18.0
        # 添加版本信息：0.18.0

    Examples
    --------
    In this example the cubic spline is used to interpolate a sampled sinusoid.
    You can see that the spline continuity property holds for the first and
    second derivatives and violates only for the third derivative.
        # 在此示例中，使用三次样条插值对采样的正弦波进行插值。
        # 您可以看到样条的连续性属性对于一阶和二阶导数保持，并且仅对三阶导数违反。

    >>> import numpy as np
    >>> from scipy.interpolate import CubicSpline
    >>> import matplotlib.pyplot as plt
    >>> x = np.arange(10)
    >>> y = np.sin(x)
    >>> cs = CubicSpline(x, y)
    >>> xs = np.arange(-0.5, 9.6, 0.1)
    >>> fig, ax = plt.subplots(figsize=(6.5, 4))
    >>> ax.plot(x, y, 'o', label='data')
    >>> ax.plot(xs, np.sin(xs), label='true')
    >>> ax.plot(xs, cs(xs), label="S")
    >>> ax.plot(xs, cs(xs, 1), label="S'")
    >>> ax.plot(xs, cs(xs, 2), label="S''")
    >>> ax.plot(xs, cs(xs, 3), label="S'''")
    >>> ax.set_xlim(-0.5, 9.5)
    >>> ax.legend(loc='lower left', ncol=2)
    >>> plt.show()

    In the second example, the unit circle is interpolated with a spline. A
        # 在第二个示例中，使用样条插值对单位圆进行插值。
    # 这段代码演示了使用 CubicSpline 类进行三次样条插值的几个例子，包括周期性边界条件和精确表示多项式函数的示例。

    # 第一个例子：创建一个圆的样本点，然后用周期性边界条件创建三次样条插值，并打印在周期点 (1, 0) 处的导数值。
    >>> theta = 2 * np.pi * np.linspace(0, 1, 5)
    >>> y = np.c_[np.cos(theta), np.sin(theta)]
    >>> cs = CubicSpline(theta, y, bc_type='periodic')
    >>> print("ds/dx={:.1f} ds/dy={:.1f}".format(cs(0, 1)[0], cs(0, 1)[1]))

    # 第二个例子：在整个周期范围内创建更多样本点，并绘制数据点、真实曲线和样条插值曲线。
    >>> xs = 2 * np.pi * np.linspace(0, 1, 100)
    >>> fig, ax = plt.subplots(figsize=(6.5, 4))
    >>> ax.plot(y[:, 0], y[:, 1], 'o', label='data')
    >>> ax.plot(np.cos(xs), np.sin(xs), label='true')
    >>> ax.plot(cs(xs)[:, 0], cs(xs)[:, 1], label='spline')
    >>> ax.axes.set_aspect('equal')
    >>> ax.legend(loc='center')
    >>> plt.show()

    # 第三个例子：插值一个多项式 y = x**3 在区间 [0, 1] 上，并验证三次样条插值能够精确表示该函数。
    >>> cs = CubicSpline([0, 1], [0, 1], bc_type=((1, 0), (1, 3)))
    >>> x = np.linspace(0, 1)
    >>> np.allclose(x**3, cs(x))
    
    # 引用部分，提供了关于三次样条插值的参考文献。
    References
    ----------
    .. [1] `Cubic Spline Interpolation
            <https://en.wikiversity.org/wiki/Cubic_Spline_Interpolation>`_
            on Wikiversity.
    .. [2] Carl de Boor, "A Practical Guide to Splines", Springer-Verlag, 1978.
    def _validate_bc(bc_type, y, expected_deriv_shape, axis):
        """Validate and prepare boundary conditions.

        Returns
        -------
        validated_bc : 2-tuple
            Boundary conditions for a curve start and end.
        y : ndarray
            y casted to complex dtype if one of the boundary conditions has
            complex dtype.
        """
        # 如果 bc_type 是字符串类型
        if isinstance(bc_type, str):
            # 如果 bc_type 是 'periodic'
            if bc_type == 'periodic':
                # 检查 `y` 起始点和结束点在给定轴上是否几乎相等
                if not np.allclose(y[0], y[-1], rtol=1e-15, atol=1e-15):
                    # 如果不是，则抛出数值错误异常
                    raise ValueError(
                        f"The first and last `y` point along axis {axis} must "
                        "be identical (within machine precision) when "
                        "bc_type='periodic'.")
            
            # 将单一的 bc_type 转换为 (bc_type, bc_type)
            bc_type = (bc_type, bc_type)

        else:
            # 如果 bc_type 不是字符串类型，则应该是一个长度为2的元组
            if len(bc_type) != 2:
                # 如果不是长度为2，则抛出数值错误异常
                raise ValueError("`bc_type` must contain 2 elements to "
                                 "specify start and end conditions.")

            # 检查 bc_type 中是否含有 'periodic'，如果有，则抛出数值错误异常
            if 'periodic' in bc_type:
                raise ValueError("'periodic' `bc_type` is defined for both "
                                 "curve ends and cannot be used with other "
                                 "boundary conditions.")

        # 初始化一个空列表，用于存储验证后的边界条件
        validated_bc = []
        # 遍历 bc_type 中的每个 bc
        for bc in bc_type:
            # 如果 bc 是字符串类型
            if isinstance(bc, str):
                # 根据不同的字符串类型，选择相应的边界条件并添加到 validated_bc 列表中
                if bc == 'clamped':
                    validated_bc.append((1, np.zeros(expected_deriv_shape)))
                elif bc == 'natural':
                    validated_bc.append((2, np.zeros(expected_deriv_shape)))
                elif bc in ['not-a-knot', 'periodic']:
                    validated_bc.append(bc)
                else:
                    # 如果 bc 是不允许的字符串类型，则抛出数值错误异常
                    raise ValueError(f"bc_type={bc} is not allowed.")
            else:
                # 如果 bc 不是字符串类型，则应该是一个元组形式 (order, value)
                try:
                    deriv_order, deriv_value = bc
                except Exception as e:
                    # 如果无法解包，则抛出数值错误异常
                    raise ValueError(
                        "A specified derivative value must be "
                        "given in the form (order, value)."
                    ) from e

                # 检查导数的阶数是否为 1 或 2
                if deriv_order not in [1, 2]:
                    raise ValueError("The specified derivative order must "
                                     "be 1 or 2.")

                # 将 deriv_value 转换为 numpy 数组
                deriv_value = np.asarray(deriv_value)
                # 检查 deriv_value 的形状是否符合预期的导数形状
                if deriv_value.shape != expected_deriv_shape:
                    raise ValueError(
                        f"`deriv_value` shape {deriv_value.shape} is not "
                        f"the expected one {expected_deriv_shape}."
                    )

                # 如果 deriv_value 是复数类型，则将 y 转换为复数类型
                if np.issubdtype(deriv_value.dtype, np.complexfloating):
                    y = y.astype(complex, copy=False)

                # 将验证后的导数条件添加到 validated_bc 列表中
                validated_bc.append((deriv_order, deriv_value))

        # 返回验证后的边界条件和可能转换过的 y
        return validated_bc, y
```