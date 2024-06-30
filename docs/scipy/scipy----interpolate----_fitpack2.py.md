# `D:\src\scipysrc\scipy\scipy\interpolate\_fitpack2.py`

```
# fitpack --- curve and surface fitting with splines

# fitpack is based on a collection of Fortran routines DIERCKX
# by P. Dierckx (see http://www.netlib.org/dierckx/) transformed
# to double routines by Pearu Peterson.

# Created by Pearu Peterson, June,August 2003

__all__ = [
    'UnivariateSpline',
    'InterpolatedUnivariateSpline',
    'LSQUnivariateSpline',
    'BivariateSpline',
    'LSQBivariateSpline',
    'SmoothBivariateSpline',
    'LSQSphereBivariateSpline',
    'SmoothSphereBivariateSpline',
    'RectBivariateSpline',
    'RectSphereBivariateSpline']

import warnings

from numpy import zeros, concatenate, ravel, diff, array
import numpy as np

from . import _fitpack_impl
from . import _dfitpack as dfitpack

# Define a dtype from _dfitpack module for use in type annotations
dfitpack_int = dfitpack.types.intvar.dtype

# ############### Univariate spline ####################

# Messages corresponding to various error codes during curve fitting
_curfit_messages = {
    1: """
    The required storage space exceeds the available storage space, as
    specified by the parameter nest: nest too small. If nest is already
    large (say nest > m/2), it may also indicate that s is too small.
    The approximation returned is the weighted least-squares spline
    according to the knots t[0],t[1],...,t[n-1]. (n=nest) the parameter fp
    gives the corresponding weighted sum of squared residuals (fp>s).
    """,
    2: """
    A theoretically impossible result was found during the iteration
    process for finding a smoothing spline with fp = s: s too small.
    There is an approximation returned but the corresponding weighted sum
    of squared residuals does not satisfy the condition abs(fp-s)/s < tol.
    """,
    3: """
    The maximal number of iterations maxit (set to 20 by the program)
    allowed for finding a smoothing spline with fp=s has been reached: s
    too small.
    There is an approximation returned but the corresponding weighted sum
    of squared residuals does not satisfy the condition abs(fp-s)/s < tol.
    """,
    10: """
    Error on entry, no approximation returned. The following conditions
    must hold:
    xb<=x[0]<x[1]<...<x[m-1]<=xe, w[i]>0, i=0..m-1
    if iopt=-1:
      xb<t[k+1]<t[k+2]<...<t[n-k-2]<xe
    """
}

# Mapping of extrapolation modes for UnivariateSpline class
_extrap_modes = {
    0: 0, 'extrapolate': 0,
    1: 1, 'zeros': 1,
    2: 2, 'raise': 2,
    3: 3, 'const': 3
}


class UnivariateSpline:
    """
    1-D smoothing spline fit to a given set of data points.

    Fits a spline y = spl(x) of degree `k` to the provided `x`, `y` data.  `s`
    specifies the number of knots by specifying a smoothing condition.

    Parameters
    ----------
    x : (N,) array_like
        1-D array of independent input data. Must be increasing;
        must be strictly increasing if `s` is 0.
    y : (N,) array_like
        1-D array of dependent input data, of the same length as `x`.
    w : (N,) array_like, optional
        Weights for spline fitting.  Must be positive.  If `w` is None,
        weights are all 1. Default is None.
    bbox : (2,) array_like, optional
        # 定界框参数，用于指定逼近区间的边界。如果 `bbox` 为 None，则取 `bbox=[x[0], x[-1]]`。默认为 None.
    k : int, optional
        # 平滑样条的阶数。必须满足 1 <= `k` <= 5。当 `k = 3` 时为三次样条。默认为 3.
    s : float or None, optional
        # 正的平滑因子，用于确定节点数目。节点数目会增加直到满足平滑条件:
        #
        #     sum((w[i] * (y[i]-spl(x[i])))**2, axis=0) <= s
        #
        # 由于数值问题，实际条件为:
        #
        #     abs(sum((w[i] * (y[i]-spl(x[i])))**2, axis=0) - s) < 0.001 * s
        #
        # 如果 `s` 为 None，则 `s` 被设置为 `len(w)`，用于一个使用所有数据点的平滑样条。
        # 如果 `s` 为 0，则样条将通过所有数据点进行插值，相当于 `InterpolatedUnivariateSpline`。
        # 推荐的 `s` 值依赖于权重 `w`。如果权重表示 `y` 的标准偏差的倒数，则一个好的 `s` 值应当在区间 (m-sqrt(2*m), m+sqrt(2*m)) 内，
        # 其中 m 是 `x`, `y`, 和 `w` 中的数据点数目。这意味着 `s = len(w)` 如果 `1/w[i]` 是 `y[i]` 标准偏差的估计值。
        # 默认为 None.
        # 用户可以使用 `s` 来控制逼近的紧密程度和平滑度的权衡。较大的 `s` 表示更多的平滑，而较小的值表示较少的平滑。
    ext : int or str, optional
        # 控制对于不在结点序列定义的区间之外的元素的外推模式。
        #
        # * 如果 ext=0 或 'extrapolate'，返回外推的值。
        # * 如果 ext=1 或 'zeros'，返回 0。
        # * 如果 ext=2 或 'raise'，抛出 ValueError。
        # * 如果 ext=3 或 'const'，返回边界值。
        #
        # 默认为 0.
    check_finite : bool, optional
        # 是否检查输入数组只包含有限数字。禁用可能会带来性能提升，但如果输入包含无穷大或 NaN，则可能导致问题（崩溃、非终止或无意义的结果）。
        # 默认为 False.
    See Also
    --------
    BivariateSpline :
        # 双变量样条的基类。
    SmoothBivariateSpline :
        # 通过给定点创建平滑双变量样条。
    LSQBivariateSpline :
        # 使用加权最小二乘拟合的双变量样条。
    RectSphereBivariateSpline :
        # 球面上矩形网格的双变量样条。
    SmoothSphereBivariateSpline :
        # 球面坐标系中的平滑双变量样条。
    LSQSphereBivariateSpline :
        # 使用加权最小二乘拟合的球面坐标系中的双变量样条。
    RectBivariateSpline :
        # 矩形网格上的双变量样条。
    InterpolatedUnivariateSpline :
        # 使用给定数据点创建一维插值样条函数
        a interpolating univariate spline for a given set of data points.
    bisplrep :
        # 找到一个二维 B-样条曲面的表示
        a function to find a bivariate B-spline representation of a surface
    bisplev :
        # 评估二维 B-样条及其导数
        a function to evaluate a bivariate B-spline and its derivatives
    splrep :
        # 找到一维曲线的 B-样条表示
        a function to find the B-spline representation of a 1-D curve
    splev :
        # 评估一维 B-样条或其导数
        a function to evaluate a B-spline or its derivatives
    sproot :
        # 找到三次 B-样条的根
        a function to find the roots of a cubic B-spline
    splint :
        # 计算 B-样条在给定点之间的定积分
        a function to evaluate the definite integral of a B-spline between two
        given points
    spalde :
        # 评估 B-样条的所有导数
        a function to evaluate all derivatives of a B-spline

    Notes
    -----
    # 数据点数量必须大于样条的阶数 `k`。

    **NaN 处理**: 如果输入数组包含 ``nan`` 值，结果将无用，因为底层的样条拟合程序无法处理 ``nan``。一个解决方法是对不是数字的数据点使用零权重：

    >>> import numpy as np
    >>> from scipy.interpolate import UnivariateSpline
    >>> x, y = np.array([1, 2, 3, 4]), np.array([1, np.nan, 3, 4])
    >>> w = np.isnan(y)
    >>> y[w] = 0.
    >>> spl = UnivariateSpline(x, y, w=~w)

    注意需要将 ``nan`` 替换为数值（具体值无关紧要，只要对应的权重是零）。

    References
    ----------
    # 基于以下文献中描述的算法 [1]_, [2]_, [3]_, and [4]_:

    .. [1] P. Dierckx, "An algorithm for smoothing, differentiation and
       integration of experimental data using spline functions",
       J.Comp.Appl.Maths 1 (1975) 165-184.
    .. [2] P. Dierckx, "A fast algorithm for smoothing data on a rectangular
       grid while using spline functions", SIAM J.Numer.Anal. 19 (1982)
       1286-1304.
    .. [3] P. Dierckx, "An improved algorithm for curve fitting with spline
       functions", report tw54, Dept. Computer Science,K.U. Leuven, 1981.
    .. [4] P. Dierckx, "Curve and surface fitting with splines", Monographs on
       Numerical Analysis, Oxford University Press, 1993.

    Examples
    --------
    # 示例

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.interpolate import UnivariateSpline
    >>> rng = np.random.default_rng()
    >>> x = np.linspace(-3, 3, 50)
    >>> y = np.exp(-x**2) + 0.1 * rng.standard_normal(50)
    >>> plt.plot(x, y, 'ro', ms=5)

    使用默认的平滑参数值：

    >>> spl = UnivariateSpline(x, y)
    >>> xs = np.linspace(-3, 3, 1000)
    >>> plt.plot(xs, spl(xs), 'g', lw=3)

    手动调整平滑参数值：

    >>> spl.set_smoothing_factor(0.5)
    >>> plt.plot(xs, spl(xs), 'b', lw=3)
    >>> plt.show()

    """
    def __init__(self, x, y, w=None, bbox=[None]*2, k=3, s=None,
                 ext=0, check_finite=False):
        # 对输入参数进行验证和初始化，确保它们符合要求，并返回有效的参数值
        x, y, w, bbox, self.ext = self.validate_input(x, y, w, bbox, k, s, ext,
                                                      check_finite)

        # 使用输入的参数调用 dfpack.fpcurf0 函数，获取拟合数据
        # data 包含了拟合的相关信息，_data 是一个包含各种拟合参数的元组
        data = dfitpack.fpcurf0(x, y, k, w=w, xb=bbox[0],
                                xe=bbox[1], s=s)
        if data[-1] == 1:
            # 如果返回的数据最后一个元素为1，表示嵌套级别过小，将其重置为最大边界
            data = self._reset_nest(data)
        # 将获取的拟合数据保存在 _data 属性中
        self._data = data
        # 初始化类的其他属性
        self._reset_class()

    @staticmethod
    def validate_input(x, y, w, bbox, k, s, ext, check_finite):
        # 将输入参数 x, y, bbox 转换为 NumPy 数组
        x, y, bbox = np.asarray(x), np.asarray(y), np.asarray(bbox)
        # 如果 w 存在，也将其转换为 NumPy 数组
        if w is not None:
            w = np.asarray(w)
        # 如果 check_finite 为 True，确保 x, y, w 中没有 NaN 或 inf
        if check_finite:
            w_finite = np.isfinite(w).all() if w is not None else True
            if (not np.isfinite(x).all() or not np.isfinite(y).all() or
                    not w_finite):
                raise ValueError("x and y array must not contain "
                                 "NaNs or infs.")
        # 检查参数 s 是否为 None 或大于 0，若为 True，检查 x 是否严格递增
        if s is None or s > 0:
            if not np.all(np.diff(x) >= 0.0):
                raise ValueError("x must be increasing if s > 0")
        else:
            # 若 s <= 0，则要求 x 必须严格递增
            if not np.all(np.diff(x) > 0.0):
                raise ValueError("x must be strictly increasing if s = 0")
        # 检查 x, y 的长度是否相等
        if x.size != y.size:
            raise ValueError("x and y should have a same length")
        # 如果 w 存在，检查其长度与 x, y 是否相等
        elif w is not None and not x.size == y.size == w.size:
            raise ValueError("x, y, and w should have a same length")
        # 检查 bbox 的形状是否为 (2,)
        elif bbox.shape != (2,):
            raise ValueError("bbox shape should be (2,)")
        # 检查 k 的取值范围是否在 1 到 5 之间
        elif not (1 <= k <= 5):
            raise ValueError("k should be 1 <= k <= 5")
        # 检查 s 是否大于等于 0.0
        elif s is not None and not s >= 0.0:
            raise ValueError("s should be s >= 0.0")

        # 尝试从 _extrap_modes 字典中获取 ext 对应的值，若找不到则抛出异常
        try:
            ext = _extrap_modes[ext]
        except KeyError as e:
            raise ValueError("Unknown extrapolation mode %s." % ext) from e

        # 返回经过验证和处理后的参数
        return x, y, w, bbox, ext

    @classmethod
    def _from_tck(cls, tck, ext=0):
        """Construct a spline object from given tck"""
        # 创建一个新的类实例
        self = cls.__new__(cls)
        # 从给定的 tck 元组中获取参数 t, c, k
        t, c, k = tck
        # 将 tck 元组作为 _eval_args 属性保存
        self._eval_args = tck
        # 初始化 _data 属性，包含各种参数
        # _data == x,y,w,xb,xe,k,s,n,t,c,fp,fpint,nrdata,ier
        self._data = (None, None, None, None, None, k, None, len(t), t,
                      c, None, None, None, None)
        # 设置 ext 属性
        self.ext = ext
        # 返回新创建的类实例
        return self
    def _reset_class(self):
        # 从 self._data 中获取需要的参数
        data = self._data
        n, t, c, k, ier = data[7], data[8], data[9], data[5], data[-1]
        # 设置 _eval_args，用于评估参数
        self._eval_args = t[:n], c[:n], k
        if ier == 0:
            # 如果 ier 等于 0，说明样条曲线具有残差平方和 fp，
            # 满足 abs(fp-s)/s <= tol 的相对容差，tol 由程序设定为 0.001
            pass
        elif ier == -1:
            # 如果 ier 等于 -1，说明样条曲线是一个插值样条
            self._set_class(InterpolatedUnivariateSpline)
        elif ier == -2:
            # 如果 ier 等于 -2，说明样条曲线是最小二乘多项式，其次数为 k
            # 在这种极端情况下，fp 提供了平滑因子 s 的上限 fp0
            self._set_class(LSQUnivariateSpline)
        else:
            # 如果 ier 不属于上述情况，则发生错误
            if ier == 1:
                self._set_class(LSQUnivariateSpline)
            # 获取错误信息
            message = _curfit_messages.get(ier, 'ier=%s' % (ier))
            # 发出警告
            warnings.warn(message, stacklevel=3)

    def _set_class(self, cls):
        # 设置当前对象的样条类别为 cls
        self._spline_class = cls
        # 如果当前类是 UnivariateSpline、InterpolatedUnivariateSpline、LSQUnivariateSpline 中的一种
        # 则将当前类设置为 cls
        if self.__class__ in (UnivariateSpline, InterpolatedUnivariateSpline,
                              LSQUnivariateSpline):
            self.__class__ = cls
        else:
            # 如果当前类是未知的子类，则不改变类别，参见 issue #731
            pass

    def _reset_nest(self, data, nest=None):
        # 重置嵌套深度 nest
        n = data[10]
        if nest is None:
            k, m = data[5], len(data[0])
            nest = m+k+1  # 这是 nest 的最大上限
        else:
            # 如果 nest 不为 None，则检查是否需要增加 nest
            if not n <= nest:
                raise ValueError("`nest` can only be increased")
        # 调整 t, c, fpint, nrdata 的大小为 nest
        t, c, fpint, nrdata = (np.resize(data[j], nest) for j in
                               [8, 9, 11, 12])

        # 构建新的参数列表 args
        args = data[:8] + (t, c, n, fpint, nrdata, data[13])
        # 调用 dfitpack.fpcurf1 函数进行曲线拟合
        data = dfitpack.fpcurf1(*args)
        return data

    def set_smoothing_factor(self, s):
        """ 继续使用给定的平滑因子 s 进行样条曲线计算，并使用上次调用时找到的结点。

        此例程直接修改样条曲线。

        """
        # 获取当前的数据
        data = self._data
        # 如果 data[6] 等于 -1，则发出警告，LSQ 样条的结点是固定的，平滑因子不会改变
        if data[6] == -1:
            warnings.warn('smoothing factor unchanged for'
                          'LSQ spline with fixed knots',
                          stacklevel=2)
            return
        # 构建新的参数列表 args
        args = data[:6] + (s,) + data[7:]
        # 调用 dfitpack.fpcurf1 函数进行曲线拟合
        data = dfitpack.fpcurf1(*args)
        # 如果返回值中最后一项为 1，表示 nest 太小，将其设置为最大上限
        if data[-1] == 1:
            data = self._reset_nest(data)
        # 更新当前对象的数据
        self._data = data
        # 重置当前对象的类别
        self._reset_class()
    def __call__(self, x, nu=0, ext=None):
        """
        在位置 x 处评估样条（或其 nu 阶导数）。

        Parameters
        ----------
        x : array_like
            要返回平滑样条或其导数值的点的一维数组。注意：x 可以是无序的，但如果 x 是（部分）有序的，评估将更有效。
        nu  : int
            要计算的样条导数阶数。
        ext : int
            控制对于不在结点序列定义的区间内的 x 元素返回的值。

            * 如果 ext=0 或 'extrapolate'，返回外推值。
            * 如果 ext=1 或 'zeros'，返回 0
            * 如果 ext=2 或 'raise'，引发 ValueError
            * 如果 ext=3 或 'const'，返回边界值。

            默认值为 0，从 UnivariateSpline 的初始化传递而来。

        """
        x = np.asarray(x)
        # 空输入产生空输出
        if x.size == 0:
            return np.array([])
        if ext is None:
            ext = self.ext
        else:
            try:
                ext = _extrap_modes[ext]
            except KeyError as e:
                raise ValueError("Unknown extrapolation mode %s." % ext) from e
        return _fitpack_impl.splev(x, self._eval_args, der=nu, ext=ext)

    def get_knots(self):
        """ 返回样条内部结点的位置。

        在内部，结点向量包含额外的 ``2*k`` 边界结点。
        """
        data = self._data
        k, n = data[5], data[7]
        return data[8][k:n-k]

    def get_coeffs(self):
        """ 返回样条系数。"""
        data = self._data
        k, n = data[5], data[7]
        return data[9][:n-k-1]

    def get_residual(self):
        """ 返回样条逼近的加权残差平方和。

           这等价于::

                sum((w[i] * (y[i]-spl(x[i])))**2, axis=0)

        """
        return self._data[10]
    # 返回该样条在给定区间 [a, b] 上的定积分值。

    def integral(self, a, b):
        """ Return definite integral of the spline between two given points.

        Parameters
        ----------
        a : float
            Lower limit of integration.
        b : float
            Upper limit of integration.

        Returns
        -------
        integral : float
            The value of the definite integral of the spline between limits.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.interpolate import UnivariateSpline
        >>> x = np.linspace(0, 3, 11)
        >>> y = x**2
        >>> spl = UnivariateSpline(x, y)
        >>> spl.integral(0, 3)
        9.0

        which agrees with :math:`\\int x^2 dx = x^3 / 3` between the limits
        of 0 and 3.

        A caveat is that this routine assumes the spline to be zero outside of
        the data limits:

        >>> spl.integral(-1, 4)
        9.0
        >>> spl.integral(-1, 0)
        0.0

        """
        # 调用底层函数 _fitpack_impl.splint 计算样条在 [a, b] 区间上的积分
        return _fitpack_impl.splint(a, b, self._eval_args)

    # 返回样条在给定点 x 处的所有阶导数值。

    def derivatives(self, x):
        """ Return all derivatives of the spline at the point x.

        Parameters
        ----------
        x : float
            The point to evaluate the derivatives at.

        Returns
        -------
        der : ndarray, shape(k+1,)
            Derivatives of the orders 0 to k.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.interpolate import UnivariateSpline
        >>> x = np.linspace(0, 3, 11)
        >>> y = x**2
        >>> spl = UnivariateSpline(x, y)
        >>> spl.derivatives(1.5)
        array([2.25, 3.0, 2.0, 0])

        """
        # 调用底层函数 _fitpack_impl.spalde 计算样条在点 x 处的所有阶导数
        return _fitpack_impl.spalde(x, self._eval_args)
    def roots(self):
        """
        Return the zeros of the spline.

        Notes
        -----
        Restriction: only cubic splines are supported by FITPACK. For non-cubic
        splines, use `PPoly.root` (see below for an example).

        Examples
        --------

        For some data, this method may miss a root. This happens when one of
        the spline knots (which FITPACK places automatically) happens to
        coincide with the true root. A workaround is to convert to `PPoly`,
        which uses a different root-finding algorithm.

        For example,

        >>> x = [1.96, 1.97, 1.98, 1.99, 2.00, 2.01, 2.02, 2.03, 2.04, 2.05]
        >>> y = [-6.365470e-03, -4.790580e-03, -3.204320e-03, -1.607270e-03,
        ...      4.440892e-16,  1.616930e-03,  3.243000e-03,  4.877670e-03,
        ...      6.520430e-03,  8.170770e-03]
        >>> from scipy.interpolate import UnivariateSpline
        >>> spl = UnivariateSpline(x, y, s=0)
        >>> spl.roots()
        array([], dtype=float64)

        Converting to a PPoly object does find the roots at `x=2`:

        >>> from scipy.interpolate import splrep, PPoly
        >>> tck = splrep(x, y, s=0)
        >>> ppoly = PPoly.from_spline(tck)
        >>> ppoly.roots(extrapolate=False)
        array([2.])

        See Also
        --------
        sproot
        PPoly.roots

        """

        # 获取样条曲线的阶数
        k = self._data[5]
        # 如果阶数为3（即三次样条曲线）
        if k == 3:
            # 获取评估参数
            t = self._eval_args[0]
            # 计算估计的节点数
            mest = 3 * (len(t) - 7)
            # 调用 FITPACK 库中的 sproot 函数计算样条曲线的零点
            return _fitpack_impl.sproot(self._eval_args, mest=mest)
        # 如果阶数不为3，抛出未实现错误
        raise NotImplementedError('finding roots unsupported for '
                                  'non-cubic splines')
    # 定义一个方法用于计算该样条曲线的导数
    def derivative(self, n=1):
        """
        Construct a new spline representing the derivative of this spline.

        Parameters
        ----------
        n : int, optional
            Order of derivative to evaluate. Default: 1

        Returns
        -------
        spline : UnivariateSpline
            Spline of order k2=k-n representing the derivative of this
            spline.

        See Also
        --------
        splder, antiderivative

        Notes
        -----

        .. versionadded:: 0.13.0

        Examples
        --------
        This can be used for finding maxima of a curve:

        >>> import numpy as np
        >>> from scipy.interpolate import UnivariateSpline
        >>> x = np.linspace(0, 10, 70)
        >>> y = np.sin(x)
        >>> spl = UnivariateSpline(x, y, k=4, s=0)

        Now, differentiate the spline and find the zeros of the
        derivative. (NB: `sproot` only works for order 3 splines, so we
        fit an order 4 spline):

        >>> spl.derivative().roots() / np.pi
        array([ 0.50000001,  1.5       ,  2.49999998])

        This agrees well with roots :math:`\\pi/2 + n\\pi` of
        :math:`\\cos(x) = \\sin'(x)`.

        """
        # 调用底层函数 _fitpack_impl.splder 对当前样条曲线进行 n 阶导数计算，返回样条曲线的参数 tck
        tck = _fitpack_impl.splder(self._eval_args, n)
        # 根据当前样条曲线的边界条件 self.ext，设置导数样条曲线的边界条件 ext
        # 如果 self.ext 是 3，则导数的 ext 设为 1（表示边界条件是 'zeros'）
        ext = 1 if self.ext == 3 else self.ext
        # 根据参数 tck 和 ext 构造一个新的 UnivariateSpline 对象，表示当前样条曲线的导数
        return UnivariateSpline._from_tck(tck, ext=ext)
    # 定义一个方法，用于生成当前样条曲线的不定积分表示的新样条曲线

    """
    Parameters
    ----------
    n : int, optional
        计算不定积分的阶数。默认为 1

    Returns
    -------
    spline : UnivariateSpline
        表示当前样条曲线不定积分的阶数为 k2=k+n 的新样条曲线对象

    Notes
    -----
    0.13.0 版本中添加

    See Also
    --------
    splantider, derivative

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.interpolate import UnivariateSpline
    >>> x = np.linspace(0, np.pi/2, 70)
    >>> y = 1 / np.sqrt(1 - 0.8*np.sin(x)**2)
    >>> spl = UnivariateSpline(x, y, s=0)

    The derivative is the inverse operation of the antiderivative,
    although some floating point error accumulates:

    >>> spl(1.7), spl.antiderivative().derivative()(1.7)
    (array(2.1565429877197317), array(2.1565429877201865))

    Antiderivative can be used to evaluate definite integrals:

    >>> ispl = spl.antiderivative()
    >>> ispl(np.pi/2) - ispl(0)
    2.2572053588768486

    This is indeed an approximation to the complete elliptic integral
    :math:`K(m) = \\int_0^{\\pi/2} [1 - m\\sin^2 x]^{-1/2} dx`:

    >>> from scipy.special import ellipk
    >>> ellipk(0.8)
    2.2572053268208538

    """
    # 调用底层的函数实现计算当前样条曲线的不定积分，并返回结果
    tck = _fitpack_impl.splantider(self._eval_args, n)
    # 使用返回的系数创建一个新的 UnivariateSpline 对象，表示不定积分的样条曲线
    return UnivariateSpline._from_tck(tck, self.ext)
class InterpolatedUnivariateSpline(UnivariateSpline):
    """
    1-D interpolating spline for a given set of data points.

    Fits a spline y = spl(x) of degree `k` to the provided `x`, `y` data.
    Spline function passes through all provided points. Equivalent to
    `UnivariateSpline` with  `s` = 0.

    Parameters
    ----------
    x : (N,) array_like
        Input dimension of data points -- must be strictly increasing
    y : (N,) array_like
        input dimension of data points
    w : (N,) array_like, optional
        Weights for spline fitting.  Must be positive.  If None (default),
        weights are all 1.
    bbox : (2,) array_like, optional
        2-sequence specifying the boundary of the approximation interval. If
        None (default), ``bbox=[x[0], x[-1]]``.
    k : int, optional
        Degree of the smoothing spline.  Must be ``1 <= k <= 5``. Default is
        ``k = 3``, a cubic spline.
    ext : int or str, optional
        Controls the extrapolation mode for elements
        not in the interval defined by the knot sequence.

        * if ext=0 or 'extrapolate', return the extrapolated value.
        * if ext=1 or 'zeros', return 0
        * if ext=2 or 'raise', raise a ValueError
        * if ext=3 of 'const', return the boundary value.

        The default value is 0.

    check_finite : bool, optional
        Whether to check that the input arrays contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination or non-sensical results) if the inputs
        do contain infinities or NaNs.
        Default is False.

    See Also
    --------
    UnivariateSpline :
        a smooth univariate spline to fit a given set of data points.
    LSQUnivariateSpline :
        a spline for which knots are user-selected
    SmoothBivariateSpline :
        a smoothing bivariate spline through the given points
    LSQBivariateSpline :
        a bivariate spline using weighted least-squares fitting
    splrep :
        a function to find the B-spline representation of a 1-D curve
    splev :
        a function to evaluate a B-spline or its derivatives
    sproot :
        a function to find the roots of a cubic B-spline
    splint :
        a function to evaluate the definite integral of a B-spline between two
        given points
    spalde :
        a function to evaluate all derivatives of a B-spline

    Notes
    -----
    The number of data points must be larger than the spline degree `k`.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.interpolate import InterpolatedUnivariateSpline
    >>> rng = np.random.default_rng()
    >>> x = np.linspace(-3, 3, 50)
    >>> y = np.exp(-x**2) + 0.1 * rng.standard_normal(50)
    >>> spl = InterpolatedUnivariateSpline(x, y)
    >>> plt.plot(x, y, 'ro', ms=5)
    >>> xs = np.linspace(-3, 3, 1000)
    >>> plt.plot(xs, spl(xs), 'g', lw=3, alpha=0.7)
    """

    # 构造函数，初始化插值样条对象
    def __init__(self, x, y, w=None, bbox=[None, None], k=3, ext=0, check_finite=False):
        # 调用父类 UnivariateSpline 的构造函数，使用 spline 插值方法
        super().__init__(x, y, w=w, bbox=bbox, k=k, ext=ext, check_finite=check_finite)
    # 显示绘图窗口
    >>> plt.show()

    # 注意 `spl(x)` 对 `y` 的插值效果：
    # 获取插值残差
    >>> spl.get_residual()
    0.0

    """

    # 初始化函数，接受输入参数 x, y, w, bbox, k, ext, check_finite
    def __init__(self, x, y, w=None, bbox=[None]*2, k=3,
                 ext=0, check_finite=False):

        # 验证并修正输入参数
        x, y, w, bbox, self.ext = self.validate_input(x, y, w, bbox, k, None,
                                            ext, check_finite)
        
        # 检查 x 是否严格递增，否则抛出 ValueError 异常
        if not np.all(diff(x) > 0.0):
            raise ValueError('x must be strictly increasing')

        # 使用 dfitpack.fpcurf0 函数进行平滑曲线拟合
        # self._data 包含 x, y, w, xb, xe, k, s, n, t, c, fp, fpint, nrdata, ier
        self._data = dfitpack.fpcurf0(x, y, k, w=w, xb=bbox[0],
                                      xe=bbox[1], s=0)
        
        # 重置类的状态
        self._reset_class()
# 定义一个多行字符串，包含错误信息，当输入参数被 fpchec 拒绝时，输出此错误信息
_fpchec_error_string = """The input parameters have been rejected by fpchec. \
This means that at least one of the following conditions is violated:

1) k+1 <= n-k-1 <= m
2) t(1) <= t(2) <= ... <= t(k+1)
   t(n-k) <= t(n-k+1) <= ... <= t(n)
3) t(k+1) < t(k+2) < ... < t(n-k)
4) t(k+1) <= x(i) <= t(n-k)
5) The conditions specified by Schoenberg and Whitney must hold
   for at least one subset of data points, i.e., there must be a
   subset of data points y(j) such that
       t(j) < y(j) < t(j+k+1), j=1,2,...,n-k-1
"""

# 定义 LSQUnivariateSpline 类，继承自 UnivariateSpline 类
class LSQUnivariateSpline(UnivariateSpline):
    """
    1-D spline with explicit internal knots.

    Fits a spline y = spl(x) of degree `k` to the provided `x`, `y` data.  `t`
    specifies the internal knots of the spline

    Parameters
    ----------
    x : (N,) array_like
        Input dimension of data points -- must be increasing
    y : (N,) array_like
        Input dimension of data points
    t : (M,) array_like
        interior knots of the spline.  Must be in ascending order and::

            bbox[0] < t[0] < ... < t[-1] < bbox[-1]

    w : (N,) array_like, optional
        weights for spline fitting. Must be positive. If None (default),
        weights are all 1.
    bbox : (2,) array_like, optional
        2-sequence specifying the boundary of the approximation interval. If
        None (default), ``bbox = [x[0], x[-1]]``.
    k : int, optional
        Degree of the smoothing spline.  Must be 1 <= `k` <= 5.
        Default is `k` = 3, a cubic spline.
    ext : int or str, optional
        Controls the extrapolation mode for elements
        not in the interval defined by the knot sequence.

        * if ext=0 or 'extrapolate', return the extrapolated value.
        * if ext=1 or 'zeros', return 0
        * if ext=2 or 'raise', raise a ValueError
        * if ext=3 of 'const', return the boundary value.

        The default value is 0.

    check_finite : bool, optional
        Whether to check that the input arrays contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination or non-sensical results) if the inputs
        do contain infinities or NaNs.
        Default is False.

    Raises
    ------
    ValueError
        If the interior knots do not satisfy the Schoenberg-Whitney conditions

    See Also
    --------
    UnivariateSpline :
        a smooth univariate spline to fit a given set of data points.
    InterpolatedUnivariateSpline :
        a interpolating univariate spline for a given set of data points.
    splrep :
        a function to find the B-spline representation of a 1-D curve
    splev :
        a function to evaluate a B-spline or its derivatives
    sproot :
        a function to find the roots of a cubic B-spline
    splint :
        a function to evaluate the definite integral of a B-spline between two
        given points
    spalde :
        a function to evaluate all derivatives of a B-spline

    Notes
    """
    """
    The number of data points must be larger than the spline degree `k`.

    Knots `t` must satisfy the Schoenberg-Whitney conditions,
    i.e., there must be a subset of data points ``x[j]`` such that
    ``t[j] < x[j] < t[j+k+1]``, for ``j=0, 1,...,n-k-2``.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.interpolate import LSQUnivariateSpline, UnivariateSpline
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> x = np.linspace(-3, 3, 50)
    >>> y = np.exp(-x**2) + 0.1 * rng.standard_normal(50)

    Fit a smoothing spline with a pre-defined internal knots:

    >>> t = [-1, 0, 1]
    >>> spl = LSQUnivariateSpline(x, y, t)

    >>> xs = np.linspace(-3, 3, 1000)
    >>> plt.plot(x, y, 'ro', ms=5)
    >>> plt.plot(xs, spl(xs), 'g-', lw=3)
    >>> plt.show()

    Check the knot vector:

    >>> spl.get_knots()
    array([-3., -1., 0., 1., 3.])

    Constructing lsq spline using the knots from another spline:

    >>> x = np.arange(10)
    >>> s = UnivariateSpline(x, x, s=0)
    >>> s.get_knots()
    array([ 0.,  2.,  3.,  4.,  5.,  6.,  7.,  9.])
    >>> knt = s.get_knots()
    >>> s1 = LSQUnivariateSpline(x, x, knt[1:-1])    # Chop 1st and last knot
    >>> s1.get_knots()
    array([ 0.,  2.,  3.,  4.,  5.,  6.,  7.,  9.])

    """

    # 初始化函数，用于创建一个 LSQUnivariateSpline 对象
    def __init__(self, x, y, t, w=None, bbox=[None]*2, k=3,
                 ext=0, check_finite=False):
        # 验证输入数据的有效性并设置对象的初始状态
        x, y, w, bbox, self.ext = self.validate_input(x, y, w, bbox, k, None,
                                                      ext, check_finite)
        # 检查 x 是否是递增的，如果不是则抛出 ValueError
        if not np.all(np.diff(x) >= 0.0):
            raise ValueError('x must be increasing')

        # 确定边界值 xb 和 xe，如果未指定则使用 x 的首尾元素
        xb = bbox[0]
        xe = bbox[1]
        if xb is None:
            xb = x[0]
        if xe is None:
            xe = x[-1]
        
        # 扩展节点 t，确保节点满足需要的个数和位置
        t = np.concatenate(([xb]*(k+1), t, [xe]*(k+1)))
        n = len(t)
        
        # 检查内部节点 t 是否满足 Schoenberg-Whitney 条件
        if not np.all(t[k+1:n-k] - t[k:n-k-1] > 0):
            raise ValueError('Interior knots t must satisfy '
                             'Schoenberg-Whitney conditions')
        
        # 使用 dfitpack 库中的 fpchec 函数检查参数是否有效
        if not dfitpack.fpchec(x, t, k) == 0:
            raise ValueError(_fpchec_error_string)
        
        # 使用 dfitpack 库中的 fpcurfm1 函数计算平滑样条拟合数据
        data = dfitpack.fpcurfm1(x, y, k, t, w=w, xb=xb, xe=xe)
        
        # 将处理后的数据保存在对象的 _data 属性中
        self._data = data[:-3] + (None, None, data[-1])
        
        # 初始化对象的状态
        self._reset_class()
# ############### Bivariate spline ####################

# 定义 _BivariateSplineBase 类，用于在矩形区域 [xb, xe] x [yb, ye] 上进行双变量样条插值
# 使用给定数据点 (x, y, z) 计算插值

class _BivariateSplineBase:
    """ Base class for Bivariate spline s(x,y) interpolation on the rectangle
    [xb,xe] x [yb, ye] calculated from a given set of data points
    (x,y,z).

    See Also
    --------
    bisplrep :
        a function to find a bivariate B-spline representation of a surface
    bisplev :
        a function to evaluate a bivariate B-spline and its derivatives
    BivariateSpline :
        a base class for bivariate splines.
    SphereBivariateSpline :
        a bivariate spline on a spherical grid
    """

    @classmethod
    def _from_tck(cls, tck):
        """Construct a spline object from given tck and degree"""
        # 创建一个样条对象，基于给定的 tck（样条系数）、阶数信息
        self = cls.__new__(cls)
        # 检查 tck 是否为 5 元组
        if len(tck) != 5:
            raise ValueError("tck should be a 5 element tuple of tx,"
                             " ty, c, kx, ky")
        # 从 tck 中提取出样条系数和阶数信息
        self.tck = tck[:3]  # tck 的前三个元素是样条系数和节点位置
        self.degrees = tck[3:]  # tck 的后两个元素是 x 和 y 方向的阶数
        return self

    def get_residual(self):
        """ Return weighted sum of squared residuals of the spline
        approximation: sum ((w[i]*(z[i]-s(x[i],y[i])))**2,axis=0)
        """
        # 返回样条逼近的加权残差平方和
        return self.fp

    def get_knots(self):
        """ Return a tuple (tx,ty) where tx,ty contain knots positions
        of the spline with respect to x-, y-variable, respectively.
        The position of interior and additional knots are given as
        t[k+1:-k-1] and t[:k+1]=b, t[-k-1:]=e, respectively.
        """
        # 返回节点位置的元组 (tx, ty)，其中 tx, ty 是关于 x 和 y 变量的节点位置
        # 内部和额外节点的位置如下：
        # 内部节点位置为 t[k+1:-k-1]
        # 额外节点位置为 t[:k+1]=b, t[-k-1:]=e
        return self.tck[:2]

    def get_coeffs(self):
        """ Return spline coefficients."""
        # 返回样条系数
        return self.tck[2]
    # 构造一个新的样条表示此样条的偏导数。
    def partial_derivative(self, dx, dy):
        """Construct a new spline representing a partial derivative of this
        spline.

        Parameters
        ----------
        dx, dy : int
            Orders of the derivative in x and y respectively. They must be
            non-negative integers and less than the respective degree of the
            original spline (self) in that direction (``kx``, ``ky``).

        Returns
        -------
        spline :
            A new spline of degrees (``kx - dx``, ``ky - dy``) representing the
            derivative of this spline.

        Notes
        -----

        .. versionadded:: 1.9.0

        """
        # 如果 dx 和 dy 均为零，直接返回原始样条对象 self
        if dx == 0 and dy == 0:
            return self
        else:
            # 获取原始样条的阶数 kx 和 ky
            kx, ky = self.degrees
            # 检查 dx 和 dy 是否为非负整数
            if not (dx >= 0 and dy >= 0):
                raise ValueError("order of derivative must be positive or"
                                 " zero")
            # 检查 dx 和 dy 是否小于原始样条的阶数 kx 和 ky
            if not (dx < kx and dy < ky):
                raise ValueError("order of derivative must be less than"
                                 " degree of spline")
            # 从原始样条的参数化表示中提取出 tx, ty 和 c
            tx, ty, c = self.tck[:3]
            # 调用 dfitpack.pardtc 函数计算偏导数后的新系数
            newc, ier = dfitpack.pardtc(tx, ty, c, kx, ky, dx, dy)
            # 如果计算过程中返回了错误码 ier，抛出异常
            if ier != 0:
                # 正常情况下不应该发生此错误
                raise ValueError("Unexpected error code returned by"
                                 " pardtc: %d" % ier)
            # 计算新的参数化向量 newtx 和 newty
            nx = len(tx)
            ny = len(ty)
            newtx = tx[dx:nx - dx]
            newty = ty[dy:ny - dy]
            # 计算新的阶数 newkx 和 newky
            newkx, newky = kx - dx, ky - dy
            # 计算新系数 newc 的长度 newclen
            newclen = (nx - dx - kx - 1) * (ny - dy - ky - 1)
            # 使用新的参数化向量和系数创建一个新的 DerivedBivariateSpline 对象并返回
            return _DerivedBivariateSpline._from_tck((newtx, newty,
                                                      newc[:newclen],
                                                      newkx, newky))
# 错误消息字典，用于不同的错误代码对应的详细错误信息
_surfit_messages = {
    1: """
    The required storage space exceeds the available storage space: nxest
    or nyest too small, or s too small.
    The weighted least-squares spline corresponds to the current set of
    knots.""",
    2: """
    A theoretically impossible result was found during the iteration
    process for finding a smoothing spline with fp = s: s too small or
    badly chosen eps.
    Weighted sum of squared residuals does not satisfy abs(fp-s)/s < tol.""",
    3: """
    the maximal number of iterations maxit (set to 20 by the program)
    allowed for finding a smoothing spline with fp=s has been reached:
    s too small.
    Weighted sum of squared residuals does not satisfy abs(fp-s)/s < tol.""",
    4: """
    No more knots can be added because the number of b-spline coefficients
    (nx-kx-1)*(ny-ky-1) already exceeds the number of data points m:
    either s or m too small.
    The weighted least-squares spline corresponds to the current set of
    knots.""",
    5: """
    No more knots can be added because the additional knot would (quasi)
    coincide with an old one: s too small or too large a weight to an
    inaccurate data point.
    The weighted least-squares spline corresponds to the current set of
    knots.""",
    10: """
    Error on entry, no approximation returned. The following conditions
    must hold:
    xb<=x[i]<=xe, yb<=y[i]<=ye, w[i]>0, i=0..m-1
    If iopt==-1, then
      xb<tx[kx+1]<tx[kx+2]<...<tx[nx-kx-2]<xe
      yb<ty[ky+1]<ty[ky+2]<...<ty[ny-ky-2]<ye""",
    -3: """
    The coefficients of the spline returned have been computed as the
    minimal norm least-squares solution of a (numerically) rank deficient
    system (deficiency=%i). If deficiency is large, the results may be
    inaccurate. Deficiency may strongly depend on the value of eps."""
}


class BivariateSpline(_BivariateSplineBase):
    """
    Base class for bivariate splines.

    This describes a spline ``s(x, y)`` of degrees ``kx`` and ``ky`` on
    the rectangle ``[xb, xe] * [yb, ye]`` calculated from a given set
    of data points ``(x, y, z)``.

    This class is meant to be subclassed, not instantiated directly.
    To construct these splines, call either `SmoothBivariateSpline` or
    `LSQBivariateSpline` or `RectBivariateSpline`.

    See Also
    --------
    UnivariateSpline :
        a smooth univariate spline to fit a given set of data points.
    SmoothBivariateSpline :
        a smoothing bivariate spline through the given points
    LSQBivariateSpline :
        a bivariate spline using weighted least-squares fitting
    RectSphereBivariateSpline :
        a bivariate spline over a rectangular mesh on a sphere
    SmoothSphereBivariateSpline :
        a smoothing bivariate spline in spherical coordinates
    LSQSphereBivariateSpline :
        a bivariate spline in spherical coordinates using weighted
        least-squares fitting
    RectBivariateSpline :
        a bivariate spline over a rectangular mesh.
    
    """
    bisplrep :
        用于找到一个二维 B 样条曲面的表示的函数
    bisplev :
        用于评估二维 B 样条曲面及其导数的函数
    """

    def ev(self, xi, yi, dx=0, dy=0):
        """
        在给定点评估样条曲面

        返回在 ``(xi[i], yi[i]), i=0,...,len(xi)-1`` 处的插值值。

        Parameters
        ----------
        xi, yi : array_like
            输入坐标。遵循标准的 NumPy 广播规则。
            轴的顺序与 ``np.meshgrid(..., indexing="ij")`` 一致，
            与默认顺序 ``np.meshgrid(..., indexing="xy")`` 不一致。
        dx : int, optional
            x 方向导数的阶数

            .. versionadded:: 0.14.0
        dy : int, optional
            y 方向导数的阶数

            .. versionadded:: 0.14.0

        Examples
        --------
        假设我们要对二维指数衰减函数进行双线性插值。

        >>> import numpy as np
        >>> from scipy.interpolate import RectBivariateSpline
        >>> def f(x, y):
        ...     return np.exp(-np.sqrt((x / 2) ** 2 + y**2))

        我们在粗网格上采样函数并设置插值器。注意，默认的 ``indexing="xy"`` 可能会导致插值后出现意外的（转置）结果。

        >>> xarr = np.linspace(-3, 3, 21)
        >>> yarr = np.linspace(-3, 3, 21)
        >>> xgrid, ygrid = np.meshgrid(xarr, yarr, indexing="ij")
        >>> zdata = f(xgrid, ygrid)
        >>> rbs = RectBivariateSpline(xarr, yarr, zdata, kx=1, ky=1)

        接下来，我们在一个更细的网格上沿坐标空间的对角线切片上对函数进行插值采样。

        >>> xinterp = np.linspace(-3, 3, 201)
        >>> yinterp = np.linspace(3, -3, 201)
        >>> zinterp = rbs.ev(xinterp, yinterp)

        并检查插值是否通过从原点沿切片的距离作为函数评估的函数。

        >>> import matplotlib.pyplot as plt
        >>> fig = plt.figure()
        >>> ax1 = fig.add_subplot(1, 1, 1)
        >>> ax1.plot(np.sqrt(xarr**2 + yarr**2), np.diag(zdata), "or")
        >>> ax1.plot(np.sqrt(xinterp**2 + yinterp**2), zinterp, "-b")
        >>> plt.show()
        """
        return self.__call__(xi, yi, dx=dx, dy=dy, grid=False)
    def integral(self, xa, xb, ya, yb):
        """
        Evaluate the integral of the spline over area [xa,xb] x [ya,yb].

        Parameters
        ----------
        xa, xb : float
            The end-points of the x integration interval.
        ya, yb : float
            The end-points of the y integration interval.

        Returns
        -------
        integ : float
            The value of the resulting integral.

        """
        # 从已定义的样条曲线上评估在区域 [xa,xb] x [ya,yb] 上的积分
        tx, ty, c = self.tck[:3]  # 从样条曲线对象中获取节点和系数
        kx, ky = self.degrees  # 获取样条曲线的阶数
        return dfitpack.dblint(tx, ty, c, kx, ky, xa, xb, ya, yb)  # 调用底层函数计算双重积分值

    @staticmethod
    def _validate_input(x, y, z, w, kx, ky, eps):
        """
        Validate input data for spline interpolation.

        Parameters
        ----------
        x, y, z : array_like
            Input arrays for spline interpolation.
        w : array_like or None
            Optional array of weights for spline fitting.
        kx, ky : int
            Degrees of the spline in the x and y directions.
        eps : float or None
            Tolerance parameter for spline fitting.

        Returns
        -------
        x, y, z, w : tuple of arrays
            Validated input arrays.

        Raises
        ------
        ValueError
            If input validation fails.

        """
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)  # 将输入转换为 NumPy 数组
        if not x.size == y.size == z.size:
            raise ValueError('x, y, and z should have a same length')  # 检查输入数组长度一致性

        if w is not None:
            w = np.asarray(w)
            if x.size != w.size:
                raise ValueError('x, y, z, and w should have a same length')  # 检查权重数组长度与输入数组长度一致性
            elif not np.all(w >= 0.0):
                raise ValueError('w should be positive')  # 检查权重数组的所有值是否非负

        if (eps is not None) and (not 0.0 < eps < 1.0):
            raise ValueError('eps should be between (0, 1)')  # 检查 eps 是否在 (0, 1) 范围内

        if not x.size >= (kx + 1) * (ky + 1):
            raise ValueError('The length of x, y and z should be at least'
                             ' (kx+1) * (ky+1)')  # 检查输入数组的长度是否至少满足 (kx+1) * (ky+1)

        return x, y, z, w  # 返回经验证的输入数组
class _DerivedBivariateSpline(_BivariateSplineBase):
    """从另一个样条的系数和结点构造的双变量样条。

    Notes
    -----
    此类不应直接从要插值或平滑的数据实例化。因此，其“fp”属性和“get_residual”方法被继承但被覆盖；在访问它们时会引发“AttributeError”。

    其他继承的属性可以像通常一样使用。
    """
    _invalid_why = ("不可用，因为_DerivedBivariateSpline实例不是从要插值或平滑的数据构造的，而是从另一个样条对象的底层结点和系数导出的")

    @property
    def fp(self):
        raise AttributeError("属性“fp” %s" % self._invalid_why)

    def get_residual(self):
        raise AttributeError("方法“get_residual” %s" % self._invalid_why)


class SmoothBivariateSpline(BivariateSpline):
    """
    平滑的双变量样条逼近。

    Parameters
    ----------
    x, y, z : array_like
        数据点的一维序列（顺序不重要）。
    w : array_like, optional
        正的一维权重序列，与`x`、`y`和`z`长度相同。
    bbox : array_like, optional
        指定矩形逼近域的边界的长度为4的序列。默认为
        ``bbox=[min(x), max(x), min(y), max(y)]``。
    kx, ky : ints, optional
        双变量样条的阶数。默认为3。
    s : float, optional
        定义估计条件的正的平滑因子：
        ``sum((w[i]*(z[i]-s(x[i], y[i])))**2, axis=0) <= s``
        默认 ``s=len(w)``，如果 ``1/w[i]`` 是 ``z[i]`` 标准差的估计值，则这应该是一个好的值。
    eps : float, optional
        确定过决定线性方程组有效秩的阈值。`eps` 应该在开区间 ``(0, 1)`` 内， 默认为 1e-16。

    See Also
    --------
    BivariateSpline :
        双变量样条的基类。
    UnivariateSpline :
        平滑的一元样条，用于拟合给定的数据点。
    LSQBivariateSpline :
        使用加权最小二乘拟合的双变量样条
    RectSphereBivariateSpline :
        球面上矩形网格的双变量样条
    SmoothSphereBivariateSpline :
        球面坐标中的平滑双变量样条
    LSQSphereBivariateSpline :
        使用加权最小二乘拟合的球面坐标中的双变量样条
    RectBivariateSpline :
        矩形网格上的双变量样条
    bisplrep :
        找到表面的双变量B样条表示的函数
    bisplev :
        评估双变量B样条及其导数的函数
    """
    """
    Notes
    -----
    The length of `x`, `y` and `z` should be at least ``(kx+1) * (ky+1)``.

    If the input data is such that input dimensions have incommensurate
    units and differ by many orders of magnitude, the interpolant may have
    numerical artifacts. Consider rescaling the data before interpolating.

    This routine constructs spline knot vectors automatically via the FITPACK
    algorithm. The spline knots may be placed away from the data points. For
    some data sets, this routine may fail to construct an interpolating spline,
    even if one is requested via ``s=0`` parameter. In such situations, it is
    recommended to use `bisplrep` / `bisplev` directly instead of this routine
    and, if needed, increase the values of ``nxest`` and ``nyest`` parameters
    of `bisplrep`.

    For linear interpolation, prefer `LinearNDInterpolator`.
    See ``https://gist.github.com/ev-br/8544371b40f414b7eaf3fe6217209bff``
    for discussion.

    """

    # 初始化函数，接收输入的数据和参数进行初始化
    def __init__(self, x, y, z, w=None, bbox=[None] * 4, kx=3, ky=3, s=None,
                 eps=1e-16):

        # 调用_validate_input方法验证并获取合法的输入数据
        x, y, z, w = self._validate_input(x, y, z, w, kx, ky, eps)

        # 将bbox展平为一维数组，检查其形状是否为(4,)
        bbox = ravel(bbox)
        if not bbox.shape == (4,):
            raise ValueError('bbox shape should be (4,)')

        # 如果参数s被指定，确保其值大于等于0
        if s is not None and not s >= 0.0:
            raise ValueError("s should be s >= 0.0")

        # 拆解bbox的边界值
        xb, xe, yb, ye = bbox

        # 使用dfitpack库的surfit_smth函数进行平滑插值
        # 返回nx, tx, ny, ty, c, fp, wrk1, ier这些变量
        nx, tx, ny, ty, c, fp, wrk1, ier = dfitpack.surfit_smth(x, y, z, w,
                                                                xb, xe, yb,
                                                                ye, kx, ky,
                                                                s=s, eps=eps,
                                                                lwrk2=1)

        # 如果ier>10，表示lwrk2过小，重新运行surfit_smth函数
        if ier > 10:
            nx, tx, ny, ty, c, fp, wrk1, ier = dfitpack.surfit_smth(x, y, z, w,
                                                                    xb, xe, yb,
                                                                    ye, kx, ky,
                                                                    s=s,
                                                                    eps=eps,
                                                                    lwrk2=ier)

        # 根据不同的返回值ier进行处理
        if ier in [0, -1, -2]:  # normal return
            pass
        else:
            # 如果返回值不在正常范围内，发出警告
            message = _surfit_messages.get(ier, 'ier=%s' % (ier))
            warnings.warn(message, stacklevel=2)

        # 将平滑度参数fp和插值参数tx, ty, c保存为对象的属性
        self.fp = fp
        self.tck = tx[:nx], ty[:ny], c[:(nx-kx-1)*(ny-ky-1)]

        # 将插值的阶数kx, ky保存为对象的属性
        self.degrees = kx, ky
# 定义一个继承自 BivariateSpline 的加权最小二乘双变量样条插值类
class LSQBivariateSpline(BivariateSpline):
    """
    Weighted least-squares bivariate spline approximation.

    Parameters
    ----------
    x, y, z : array_like
        1-D sequences of data points (order is not important).
        数据点的一维序列，顺序无关紧要。
    tx, ty : array_like
        Strictly ordered 1-D sequences of knots coordinates.
        严格有序的节点坐标的一维序列。
    w : array_like, optional
        Positive 1-D array of weights, of the same length as `x`, `y` and `z`.
        正权重的一维数组，长度与 `x`, `y`, `z` 相同。
    bbox : (4,) array_like, optional
        Sequence of length 4 specifying the boundary of the rectangular
        approximation domain.  By default,
        ``bbox=[min(x,tx),max(x,tx), min(y,ty),max(y,ty)]``.
        长度为4的序列，指定矩形逼近域的边界。
        默认为 ``bbox=[min(x,tx),max(x,tx), min(y,ty),max(y,ty)]``。
    kx, ky : ints, optional
        Degrees of the bivariate spline. Default is 3.
        双变量样条的次数。默认为3。
    eps : float, optional
        A threshold for determining the effective rank of an over-determined
        linear system of equations. `eps` should have a value within the open
        interval ``(0, 1)``, the default is 1e-16.
        用于确定超定线性方程组有效秩的阈值。`eps` 应在开区间 ``(0, 1)`` 内，缺省为 1e-16。

    See Also
    --------
    BivariateSpline :
        a base class for bivariate splines.
        双变量样条的基类。
    UnivariateSpline :
        a smooth univariate spline to fit a given set of data points.
        平滑的单变量样条，用于拟合给定的数据点集。
    SmoothBivariateSpline :
        a smoothing bivariate spline through the given points
        经过给定点的平滑双变量样条。
    RectSphereBivariateSpline :
        a bivariate spline over a rectangular mesh on a sphere
        球面上矩形网格上的双变量样条。
    SmoothSphereBivariateSpline :
        a smoothing bivariate spline in spherical coordinates
        球坐标中的平滑双变量样条。
    LSQSphereBivariateSpline :
        a bivariate spline in spherical coordinates using weighted
        least-squares fitting
        使用加权最小二乘拟合的球坐标中的双变量样条。
    RectBivariateSpline :
        a bivariate spline over a rectangular mesh.
        矩形网格上的双变量样条。
    bisplrep :
        a function to find a bivariate B-spline representation of a surface
        找到表面的双变量 B 样条表示的函数。
    bisplev :
        a function to evaluate a bivariate B-spline and its derivatives
        评估双变量 B 样条及其导数的函数。

    Notes
    -----
    The length of `x`, `y` and `z` should be at least ``(kx+1) * (ky+1)``.
    `x`, `y` 和 `z` 的长度至少应为 ``(kx+1) * (ky+1)``。

    If the input data is such that input dimensions have incommensurate
    units and differ by many orders of magnitude, the interpolant may have
    numerical artifacts. Consider rescaling the data before interpolating.
    如果输入数据的输入维度具有不相称的单位，并且差异很大，则插值可能会产生数值伪影。
    考虑在插值之前重新缩放数据。
    """
    # 初始化函数，接受多个参数并进行验证后赋值给对应变量
    def __init__(self, x, y, z, tx, ty, w=None, bbox=[None]*4, kx=3, ky=3,
                 eps=None):
        
        # 使用类方法 _validate_input 进行参数验证和处理，并将结果赋给对应变量
        x, y, z, w = self._validate_input(x, y, z, w, kx, ky, eps)
        
        # 将 bbox 变量展平成一维数组
        bbox = ravel(bbox)
        
        # 检查 bbox 数组的形状是否为 (4,)，如果不是则抛出 ValueError 异常
        if not bbox.shape == (4,):
            raise ValueError('bbox shape should be (4,)')
        
        # 根据公式计算 nx 和 ny 的值
        nx = 2*kx + 2 + len(tx)
        ny = 2*ky + 2 + len(ty)
        
        # 准备空的数组 tx1 和 ty1，其长度为 nx 和 ny 中的最大值 nmax
        nmax = max(nx, ny)
        tx1 = zeros((nmax,), float)
        ty1 = zeros((nmax,), float)
        
        # 将 tx 数组中的值复制到 tx1 的指定位置
        tx1[kx+1:nx-kx-1] = tx
        
        # 将 ty 数组中的值复制到 ty1 的指定位置
        ty1[ky+1:ny-ky-1] = ty
        
        # 解构赋值，调用 dfitpack.surfit_lsq 函数进行曲面拟合操作
        xb, xe, yb, ye = bbox
        tx1, ty1, c, fp, ier = dfitpack.surfit_lsq(x, y, z, nx, tx1, ny, ty1,
                                                   w, xb, xe, yb, ye,
                                                   kx, ky, eps, lwrk2=1)
        
        # 如果返回的错误代码 ier 大于 10，则重新调用 surfit_lsq 函数
        if ier > 10:
            tx1, ty1, c, fp, ier = dfitpack.surfit_lsq(x, y, z,
                                                       nx, tx1, ny, ty1, w,
                                                       xb, xe, yb, ye,
                                                       kx, ky, eps, lwrk2=ier)
        
        # 根据返回的错误代码 ier 进行处理
        if ier in [0, -1, -2]:  # 如果 ier 是 0, -1, -2，则正常返回，不做处理
            pass
        else:
            # 如果 ier 小于 -2，则计算缺失值
            if ier < -2:
                deficiency = (nx-kx-1)*(ny-ky-1) + ier
                # 根据缺失值获取警告消息
                message = _surfit_messages.get(-3) % (deficiency)
            else:
                # 否则，根据 ier 获取默认的警告消息
                message = _surfit_messages.get(ier, 'ier=%s' % (ier))
            
            # 发出警告消息，告知警告发生的堆栈层级为 2
            warnings.warn(message, stacklevel=2)
        
        # 将 fp 和 tx1, ty1, c 的子集作为对象的属性
        self.fp = fp
        self.tck = tx1[:nx], ty1[:ny], c
        # 将 kx 和 ky 的值作为对象的 degrees 属性
        self.degrees = kx, ky
# 继承自 BivariateSpline 的矩形双变量样条逼近类
class RectBivariateSpline(BivariateSpline):
    """
    矩形网格上的双变量样条逼近。

    可用于数据的平滑处理和插值。

    Parameters
    ----------
    x,y : array_like
        严格升序的坐标数组。
        在数据范围之外评估的点将被外推。
    z : array_like
        形状为 (x.size, y.size) 的二维数据数组。
    bbox : array_like, optional
        长度为 4 的序列，指定矩形逼近域的边界，
        即每个维度的起始和结束样条结点由这些值设置。
        默认为 ``bbox=[min(x), max(x), min(y), max(y)]``。
    kx, ky : ints, optional
        双变量样条的次数。默认为 3。
    s : float, optional
        定义在估计条件下的正的平滑因子：
        ``sum((z[i]-f(x[i], y[i]))**2, axis=0) <= s``，其中 f 是样条函数。
        默认为 ``s=0``，用于插值。

    See Also
    --------
    BivariateSpline :
        双变量样条的基类。
    UnivariateSpline :
        平滑的单变量样条，用于拟合给定的数据点。
    SmoothBivariateSpline :
        通过给定点的平滑双变量样条。
    LSQBivariateSpline :
        使用加权最小二乘拟合的双变量样条。
    RectSphereBivariateSpline :
        球面上的矩形网格双变量样条。
    SmoothSphereBivariateSpline :
        球坐标系中的平滑双变量样条。
    LSQSphereBivariateSpline :
        使用加权最小二乘拟合的球坐标系中的双变量样条。
    bisplrep :
        找到一个表面的双变量 B 样条表示的函数。
    bisplev :
        评估双变量 B 样条及其导数的函数。

    Notes
    -----

    如果输入数据的输入维度具有不兼容的单位并且差异很大，
    插值结果可能会有数值伪影。考虑在插值之前重新缩放数据。

    """
    # 初始化函数，用于创建一个对象实例，接受多个参数和可选参数
    def __init__(self, x, y, z, bbox=[None] * 4, kx=3, ky=3, s=0):
        # 将 x, y, bbox 展平成一维数组
        x, y, bbox = ravel(x), ravel(y), ravel(bbox)
        # 将 z 转换为 NumPy 数组
        z = np.asarray(z)
        
        # 检查 x 是否严格递增，如果不是则抛出 ValueError 异常
        if not np.all(diff(x) > 0.0):
            raise ValueError('x must be strictly increasing')
        # 检查 y 是否严格递增，如果不是则抛出 ValueError 异常
        if not np.all(diff(y) > 0.0):
            raise ValueError('y must be strictly increasing')
        # 检查 z 的第一维长度是否与 x 的长度相同，不同则抛出 ValueError 异常
        if not x.size == z.shape[0]:
            raise ValueError('x dimension of z must have same number of '
                             'elements as x')
        # 检查 z 的第二维长度是否与 y 的长度相同，不同则抛出 ValueError 异常
        if not y.size == z.shape[1]:
            raise ValueError('y dimension of z must have same number of '
                             'elements as y')
        # 检查 bbox 的形状是否为 (4,)，如果不是则抛出 ValueError 异常
        if not bbox.shape == (4,):
            raise ValueError('bbox shape should be (4,)')
        # 检查 s 是否为 None 或者非负数，如果不是则抛出 ValueError 异常
        if s is not None and not s >= 0.0:
            raise ValueError("s should be s >= 0.0")

        # 将 z 展平成一维数组
        z = ravel(z)
        # 调用 dfp_regrid_smth 函数进行光滑重构
        xb, xe, yb, ye = bbox
        nx, tx, ny, ty, c, fp, ier = dfitpack.regrid_smth(x, y, z, xb, xe, yb,
                                                          ye, kx, ky, s)

        # 检查返回的 ier 是否在指定的列表中，否则抛出相应的 ValueError 异常
        if ier not in [0, -1, -2]:
            msg = _surfit_messages.get(ier, 'ier=%s' % (ier))
            raise ValueError(msg)

        # 将返回的结果存储在对象的属性中
        self.fp = fp
        self.tck = tx[:nx], ty[:ny], c[:(nx - kx - 1) * (ny - ky - 1)]
        self.degrees = kx, ky
# 创建一个副本，包含与 _surfit_messages 相同的内容
_spherefit_messages = _surfit_messages.copy()

# 将索引为 10 的条目设置为一个多行字符串，描述输入数据的有效性约束条件和错误信息
_spherefit_messages[10] = """
ERROR. On entry, the input data are controlled on validity. The following
       restrictions must be satisfied:
            -1<=iopt<=1,  m>=2, ntest>=8 ,npest >=8, 0<eps<1,
            0<=teta(i)<=pi, 0<=phi(i)<=2*pi, w(i)>0, i=1,...,m
            lwrk1 >= 185+52*v+10*u+14*u*v+8*(u-1)*v**2+8*m
            kwrk >= m+(ntest-7)*(npest-7)
            if iopt=-1: 8<=nt<=ntest , 9<=np<=npest
                        0<tt(5)<tt(6)<...<tt(nt-4)<pi
                        0<tp(5)<tp(6)<...<tp(np-4)<2*pi
            if iopt>=0: s>=0
            if one of these conditions is found to be violated,control
            is immediately repassed to the calling program. in that
            case there is no approximation returned."""

# 将索引为 -3 的条目设置为一个多行字符串，提供关于数值系统中可能存在的警告信息
_spherefit_messages[-3] = """
WARNING. The coefficients of the spline returned have been computed as the
         minimal norm least-squares solution of a (numerically) rank
         deficient system (deficiency=%i, rank=%i). Especially if the rank
         deficiency, which is computed by 6+(nt-8)*(np-7)+ier, is large,
         the results may be inaccurate. They could also seriously depend on
         the value of eps."""


class SphereBivariateSpline(_BivariateSplineBase):
    """
    表示在球面上的双变量三次样条插值 s(x,y)，由给定的数据点 (theta,phi,r) 计算得到。

    .. versionadded:: 0.11.0

    See Also
    --------
    bisplrep :
        用于找到表面的双变量 B 样条表示的函数
    bisplev :
        用于评估双变量 B 样条及其导数的函数
    UnivariateSpline :
        用于拟合给定数据点的平滑单变量样条
    SmoothBivariateSpline :
        通过给定点进行平滑的双变量样条
    LSQUnivariateSpline :
        使用加权最小二乘拟合的单变量样条
    """
    def ev(self, theta, phi, dtheta=0, dphi=0):
        """
        Evaluate the spline at points

        Returns the interpolated value at ``(theta[i], phi[i]),
        i=0,...,len(theta)-1``.

        Parameters
        ----------
        theta, phi : array_like
            Input coordinates. Standard Numpy broadcasting is obeyed.
            The ordering of axes is consistent with
            np.meshgrid(..., indexing="ij") and inconsistent with the
            default ordering np.meshgrid(..., indexing="xy").
        dtheta : int, optional
            Order of theta-derivative

            .. versionadded:: 0.14.0
        dphi : int, optional
            Order of phi-derivative

            .. versionadded:: 0.14.0

        Examples
        --------
        Suppose that we want to use splines to interpolate a bivariate function on a
        sphere. The value of the function is known on a grid of longitudes and
        colatitudes.

        >>> import numpy as np
        >>> from scipy.interpolate import RectSphereBivariateSpline
        >>> def f(theta, phi):
        ...     return np.sin(theta) * np.cos(phi)

        We evaluate the function on the grid. Note that the default indexing="xy"
        of meshgrid would result in an unexpected (transposed) result after
        interpolation.

        >>> thetaarr = np.linspace(0, np.pi, 22)[1:-1]
        >>> phiarr = np.linspace(0, 2 * np.pi, 21)[:-1]
        >>> thetagrid, phigrid = np.meshgrid(thetaarr, phiarr, indexing="ij")
        >>> zdata = f(thetagrid, phigrid)

        We next set up the interpolator and use it to evaluate the function
        at points not on the original grid.

        >>> rsbs = RectSphereBivariateSpline(thetaarr, phiarr, zdata)
        >>> thetainterp = np.linspace(thetaarr[0], thetaarr[-1], 200)
        >>> phiinterp = np.linspace(phiarr[0], phiarr[-1], 200)
        >>> zinterp = rsbs.ev(thetainterp, phiinterp)

        Finally we plot the original data for a diagonal slice through the
        initial grid, and the spline approximation along the same slice.

        >>> import matplotlib.pyplot as plt
        >>> fig = plt.figure()
        >>> ax1 = fig.add_subplot(1, 1, 1)
        >>> ax1.plot(np.sin(thetaarr) * np.sin(phiarr), np.diag(zdata), "or")
        >>> ax1.plot(np.sin(thetainterp) * np.sin(phiinterp), zinterp, "-b")
        >>> plt.show()

        Returns
        -------
        numpy.ndarray
            The interpolated values at the specified points.

        Notes
        -----
        This method uses the underlying __call__ method to perform the actual
        evaluation of the spline.

        See Also
        --------
        __call__ : Method that actually evaluates the spline.

        """
        return self.__call__(theta, phi, dtheta=dtheta, dphi=dphi, grid=False)
class SmoothSphereBivariateSpline(SphereBivariateSpline):
    """
    Smooth bivariate spline approximation in spherical coordinates.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    theta, phi, r : array_like
        1-D sequences of data points (order is not important). Coordinates
        must be given in radians. Theta must lie within the interval
        ``[0, pi]``, and phi must lie within the interval ``[0, 2pi]``.
    w : array_like, optional
        Positive 1-D sequence of weights.
    s : float, optional
        Positive smoothing factor defined for estimation condition:
        ``sum((w(i)*(r(i) - s(theta(i), phi(i))))**2, axis=0) <= s``
        Default ``s=len(w)`` which should be a good value if ``1/w[i]`` is an
        estimate of the standard deviation of ``r[i]``.
    eps : float, optional
        A threshold for determining the effective rank of an over-determined
        linear system of equations. `eps` should have a value within the open
        interval ``(0, 1)``, the default is 1e-16.

    See Also
    --------
    BivariateSpline :
        a base class for bivariate splines.
    UnivariateSpline :
        a smooth univariate spline to fit a given set of data points.
    SmoothBivariateSpline :
        a smoothing bivariate spline through the given points
    LSQBivariateSpline :
        a bivariate spline using weighted least-squares fitting
    RectSphereBivariateSpline :
        a bivariate spline over a rectangular mesh on a sphere
    LSQSphereBivariateSpline :
        a bivariate spline in spherical coordinates using weighted
        least-squares fitting
    RectBivariateSpline :
        a bivariate spline over a rectangular mesh.
    bisplrep :
        a function to find a bivariate B-spline representation of a surface
    bisplev :
        a function to evaluate a bivariate B-spline and its derivatives

    Notes
    -----
    For more information, see the FITPACK_ site about this function.

    .. _FITPACK: http://www.netlib.org/dierckx/sphere.f

    Examples
    --------
    Suppose we have global data on a coarse grid (the input data does not
    have to be on a grid):

    >>> import numpy as np
    >>> theta = np.linspace(0., np.pi, 7)
    >>> phi = np.linspace(0., 2*np.pi, 9)
    >>> data = np.empty((theta.shape[0], phi.shape[0]))
    >>> data[:,0], data[0,:], data[-1,:] = 0., 0., 0.
    >>> data[1:-1,1], data[1:-1,-1] = 1., 1.
    >>> data[1,1:-1], data[-2,1:-1] = 1., 1.
    >>> data[2:-2,2], data[2:-2,-2] = 2., 2.
    >>> data[2,2:-2], data[-3,2:-2] = 2., 2.
    >>> data[3,3:-2] = 3.
    >>> data = np.roll(data, 4, 1)

    We need to set up the interpolator object

    >>> lats, lons = np.meshgrid(theta, phi)
    >>> from scipy.interpolate import SmoothSphereBivariateSpline
    >>> lut = SmoothSphereBivariateSpline(lats.ravel(), lons.ravel(),
    ...                                   data.T.ravel(), s=3.5)
    """
    # 继承自SphereBivariateSpline，实现在球坐标系中的平滑双变量样条逼近
    # 版本 0.11.0 新增
    # 参数说明：
    # theta, phi, r: array_like，数据点的1维序列（顺序无关），必须以弧度给出
    # w: array_like, optional，正的1维权重序列
    # s: float, optional，平滑因子，用于估计条件的定义
    # eps: float, optional，决定超定线性方程组有效秩的阈值
    pass
    input coordinates
    # 定义一个函数调用示例

    >>> data_orig = lut(theta, phi)
    # 使用 lut 函数计算原始数据，将结果存储在 data_orig 中

    Finally we interpolate the data to a finer grid
    # 最终将数据插值到更细的网格上

    >>> fine_lats = np.linspace(0., np.pi, 70)
    # 生成一个包含70个元素的数组，表示纬度从0到π之间的均匀间隔

    >>> fine_lons = np.linspace(0., 2 * np.pi, 90)
    # 生成一个包含90个元素的数组，表示经度从0到2π之间的均匀间隔

    >>> data_smth = lut(fine_lats, fine_lons)
    # 使用 lut 函数对细分后的纬度和经度进行插值计算，结果存储在 data_smth 中

    >>> import matplotlib.pyplot as plt
    # 导入 matplotlib.pyplot 模块，用于绘图

    >>> fig = plt.figure()
    # 创建一个新的图形窗口

    >>> ax1 = fig.add_subplot(131)
    # 在图形窗口中添加一个子图，1行3列布局的第一个位置

    >>> ax1.imshow(data, interpolation='nearest')
    # 在第一个子图中显示 data 数据，使用最近邻插值方法进行插值

    >>> ax2 = fig.add_subplot(132)
    # 在同一图形窗口中添加第二个子图，1行3列布局的第二个位置

    >>> ax2.imshow(data_orig, interpolation='nearest')
    # 在第二个子图中显示 data_orig 数据，使用最近邻插值方法进行插值

    >>> ax3 = fig.add_subplot(133)
    # 在同一图形窗口中添加第三个子图，1行3列布局的第三个位置

    >>> ax3.imshow(data_smth, interpolation='nearest')
    # 在第三个子图中显示 data_smth 数据，使用最近邻插值方法进行插值

    >>> plt.show()
    # 显示绘制好的图形

    """

    def __init__(self, theta, phi, r, w=None, s=0., eps=1E-16):
        # 初始化方法，接收球面拟合所需的参数

        theta, phi, r = np.asarray(theta), np.asarray(phi), np.asarray(r)
        # 将输入的 theta、phi、r 转换为 NumPy 数组

        # input validation
        # 输入验证
        if not ((0.0 <= theta).all() and (theta <= np.pi).all()):
            raise ValueError('theta should be between [0, pi]')
        # 检查 theta 是否在 [0, pi] 范围内，否则引发 ValueError

        if not ((0.0 <= phi).all() and (phi <= 2.0 * np.pi).all()):
            raise ValueError('phi should be between [0, 2pi]')
        # 检查 phi 是否在 [0, 2pi] 范围内，否则引发 ValueError

        if w is not None:
            w = np.asarray(w)
            if not (w >= 0.0).all():
                raise ValueError('w should be positive')
        # 如果 w 不为 None，则将其转换为 NumPy 数组并检查是否非负，否则引发 ValueError

        if not s >= 0.0:
            raise ValueError('s should be positive')
        # 检查 s 是否非负，否则引发 ValueError

        if not 0.0 < eps < 1.0:
            raise ValueError('eps should be between (0, 1)')
        # 检查 eps 是否在 (0, 1) 之间，否则引发 ValueError

        # Perform spherical smoothing fit
        # 执行球面平滑拟合
        nt_, tt_, np_, tp_, c, fp, ier = dfitpack.spherfit_smth(theta, phi,
                                                                r, w=w, s=s,
                                                                eps=eps)
        # 调用 dfpack.spherfit_smth 函数进行球面拟合

        if ier not in [0, -1, -2]:
            message = _spherefit_messages.get(ier, 'ier=%s' % (ier))
            raise ValueError(message)
        # 如果返回值 ier 不在 [0, -1, -2] 中，则根据错误码生成相应的错误消息并引发 ValueError

        # Assign computed values to instance variables
        # 将计算得到的值赋给实例变量
        self.fp = fp
        self.tck = tt_[:nt_], tp_[:np_], c[:(nt_ - 4) * (np_ - 4)]
        self.degrees = (3, 3)

    def __call__(self, theta, phi, dtheta=0, dphi=0, grid=True):
        # 实现可调用对象的方法，用于进行插值计算

        theta = np.asarray(theta)
        phi = np.asarray(phi)

        if phi.size > 0 and (phi.min() < 0. or phi.max() > 2. * np.pi):
            raise ValueError("requested phi out of bounds.")
        # 检查 phi 是否超出范围，如果超出则引发 ValueError

        return SphereBivariateSpline.__call__(self, theta, phi, dtheta=dtheta,
                                              dphi=dphi, grid=grid)
        # 调用父类的 __call__ 方法进行插值计算，并返回结果
# 定义一个继承自 SphereBivariateSpline 的加权最小二乘双变量样条插值类，在球面坐标系中进行插值逼近。

class LSQSphereBivariateSpline(SphereBivariateSpline):
    """
    加权最小二乘双变量样条插值类，在球面坐标系中进行插值逼近。

    根据给定的 `theta` 和 `phi` 方向的结点，确定一个平滑的双三次样条。

    .. versionadded:: 0.11.0

    Parameters
    ----------
    theta, phi, r : array_like
        数据点的一维序列（顺序无关紧要）。坐标必须以弧度给出。`theta` 必须在区间 ``[0, pi]`` 内，
        而 `phi` 必须在区间 ``[0, 2pi]`` 内。
    tt, tp : array_like
        严格有序的结点坐标的一维序列。坐标必须满足 ``0 < tt[i] < pi``，``0 < tp[i] < 2*pi``。
    w : array_like, optional
        正的一维权重序列，长度与 `theta`, `phi` 和 `r` 相同。
    eps : float, optional
        用于确定超定线性方程组的有效秩的阈值。`eps` 应该在开区间 ``(0, 1)`` 内，默认值为 1e-16。

    See Also
    --------
    BivariateSpline :
        双变量样条的基类。
    UnivariateSpline :
        平滑的单变量样条，用于拟合给定的数据点。
    SmoothBivariateSpline :
        通过给定的点创建平滑的双变量样条。
    LSQBivariateSpline :
        使用加权最小二乘拟合的双变量样条。
    RectSphereBivariateSpline :
        球面上矩形网格上的双变量样条。
    SmoothSphereBivariateSpline :
        球面坐标系中的平滑双变量样条。
    RectBivariateSpline :
        矩形网格上的双变量样条。
    bisplrep :
        找到表面的双变量 B 样条表示的函数。
    bisplev :
        评估双变量 B 样条及其导数的函数。

    Notes
    -----
    更多信息，请参阅 FITPACK_ 网站上关于此函数的说明。

    .. _FITPACK: http://www.netlib.org/dierckx/sphere.f

    Examples
    --------
    假设我们有全局数据在一个粗网格上（输入数据不必在网格上）：

    >>> from scipy.interpolate import LSQSphereBivariateSpline
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> theta = np.linspace(0, np.pi, num=7)
    >>> phi = np.linspace(0, 2*np.pi, num=9)
    >>> data = np.empty((theta.shape[0], phi.shape[0]))
    >>> data[:,0], data[0,:], data[-1,:] = 0., 0., 0.
    >>> data[1:-1,1], data[1:-1,-1] = 1., 1.
    >>> data[1,1:-1], data[-2,1:-1] = 1., 1.
    >>> data[2:-2,2], data[2:-2,-2] = 2., 2.
    >>> data[2,2:-2], data[-3,2:-2] = 2., 2.
    >>> data[3,3:-2] = 3.
    >>> data = np.roll(data, 4, 1)

    我们需要设置插值器对象。在这里，我们还必须指定要使用的结点的坐标。

    >>> lats, lons = np.meshgrid(theta, phi)
    >>> knotst, knotsp = theta.copy(), phi.copy()
    >>> knotst[0] += .0001
    >>> knotst[-1] -= .0001
    # 减小节点数组 `knotst` 中的最后一个元素的值，减少其值为0.0001
    >>> knotsp[0] += .0001
    # 增加节点数组 `knotsp` 中的第一个元素的值，增加其值为0.0001
    >>> knotsp[-1] -= .0001
    # 减小节点数组 `knotsp` 中的最后一个元素的值，减少其值为0.0001
    >>> lut = LSQSphereBivariateSpline(lats.ravel(), lons.ravel(),
    ...                                data.T.ravel(), knotst, knotsp)
    # 使用节点数组 `knotst` 和 `knotsp` 初始化 `LSQSphereBivariateSpline` 对象 `lut`，
    # 用于球面双变量样条插值

    As a first test, we'll see what the algorithm returns when run on the
    input coordinates

    >>> data_orig = lut(theta, phi)
    # 使用 `lut` 对象进行球面双变量样条插值，得到在坐标 (theta, phi) 处的插值数据，并赋给 `data_orig`

    Finally we interpolate the data to a finer grid

    >>> fine_lats = np.linspace(0., np.pi, 70)
    # 创建一个细化的纬度网格 `fine_lats`，从0到π，共70个点
    >>> fine_lons = np.linspace(0., 2*np.pi, 90)
    # 创建一个细化的经度网格 `fine_lons`，从0到2π，共90个点
    >>> data_lsq = lut(fine_lats, fine_lons)
    # 使用 `lut` 对象进行球面双变量样条插值，得到在细化网格上的插值数据，并赋给 `data_lsq`

    >>> fig = plt.figure()
    # 创建一个新的图形窗口 `fig`
    >>> ax1 = fig.add_subplot(131)
    # 在图形窗口中添加一个子图 `ax1`，分成1行3列，当前选中第1列
    >>> ax1.imshow(data, interpolation='nearest')
    # 在 `ax1` 子图中显示 `data` 数据，使用最近邻插值方式显示

    >>> ax2 = fig.add_subplot(132)
    # 在图形窗口中添加一个子图 `ax2`，分成1行3列，当前选中第2列
    >>> ax2.imshow(data_orig, interpolation='nearest')
    # 在 `ax2` 子图中显示 `data_orig` 数据，使用最近邻插值方式显示

    >>> ax3 = fig.add_subplot(133)
    # 在图形窗口中添加一个子图 `ax3`，分成1行3列，当前选中第3列
    >>> ax3.imshow(data_lsq, interpolation='nearest')
    # 在 `ax3` 子图中显示 `data_lsq` 数据，使用最近邻插值方式显示

    >>> plt.show()
    # 显示整个图形窗口
# 复制 `_surfit_messages` 的内容到 `_spfit_messages`
_spfit_messages = _surfit_messages.copy()

# 在 `_spfit_messages` 字典中添加键为 10 的错误信息字符串
_spfit_messages[10] = """
ERROR: on entry, the input data are controlled on validity
       the following restrictions must be satisfied.
          -1<=iopt(1)<=1, 0<=iopt(2)<=1, 0<=iopt(3)<=1,
          -1<=ider(1)<=1, 0<=ider(2)<=1, ider(2)=0 if iopt(2)=0.
          -1<=ider(3)<=1, 0<=ider(4)<=1, ider(4)=0 if iopt(3)=0.
          mu >= mumin (see above), mv >= 4, nuest >=8, nvest >= 8,
          kwrk>=5+mu+mv+nuest+nvest,
          lwrk >= 12+nuest*(mv+nvest+3)+nvest*24+4*mu+8*mv+max(nuest,mv+nvest)
          0< u(i-1)<u(i)< pi,i=2,..,mu,
          -pi<=v(1)< pi, v(1)<v(i-1)<v(i)<v(1)+2*pi, i=3,...,mv
          if iopt(1)=-1: 8<=nu<=min(nuest,mu+6+iopt(2)+iopt(3))
                         0<tu(5)<tu(6)<...<tu(nu-4)< pi
                         8<=nv<=min(nvest,mv+7)
                         v(1)<tv(5)<tv(6)<...<tv(nv-4)<v(1)+2*pi
                         the schoenberg-whitney conditions, i.e. there must be
                         subset of grid coordinates uu(p) and vv(q) such that
                            tu(p) < uu(p) < tu(p+4) ,p=1,...,nu-4
                            (iopt(2)=1 and iopt(3)=1 also count for a uu-value
                            tv(q) < vv(q) < tv(q+4) ,q=1,...,nv-4
                            (vv(q) is either a value v(j) or v(j)+2*pi)
          if iopt(1)>=0: s>=0
          if s=0: nuest>=mu+6+iopt(2)+iopt(3), nvest>=mv+7
       if one of these conditions is found to be violated,control is
       immediately repassed to the calling program. in that case there is no
       approximation returned."""

# 定义一个继承自 SphereBivariateSpline 的类 RectSphereBivariateSpline
class RectSphereBivariateSpline(SphereBivariateSpline):
    """
    在球面上矩形网格上的双变量样条插值。

    可用于平滑数据。

    .. versionadded:: 0.11.0

    Parameters
    ----------
    u : array_like
        严格升序的极地角坐标的一维数组。
        坐标必须以弧度给出，并位于开区间 ``(0, pi)``
    v : array_like
        严格升序的经度坐标的一维数组。
        坐标必须以弧度给出。第一个元素（``v[0]``）必须位于区间 ``[-pi, pi)``
        最后一个元素（``v[-1]``）必须满足 ``v[-1] <= v[0] + 2*pi``
    r : array_like
        形状为 ``(u.size, v.size)`` 的二维数据数组。
    s : float, optional
        正的平滑因子，用于估计条件（``s=0`` 表示插值）。
    pole_continuity : bool or (bool, bool), optional
        极点处的连续性顺序 ``u=0``（``pole_continuity[0]``）和 ``u=pi``（``pole_continuity[1]``）。
        当为 True 或 False 时，极点处的连续性顺序将分别为 1 或 0。
        默认为 False。
    """
    # pole_values 表示在极点 u=0 和 u=pi 处的数据值，可以是一个浮点数或一个元组 (float, float)，每个元素可以为 None，默认为 None。
    pole_values : float or (float, float), optional
    # pole_exact 表示在极点 u=0 和 u=pi 处的数据值是否精确。如果为 True，则值被认为是准确的函数值，将被精确拟合；如果为 False，则被视为普通数据值。默认为 False。
    pole_exact : bool or (bool, bool), optional
    # pole_flat 指定极点 u=0 和 u=pi 处的逼近是否有消失的导数。默认为 False。
    pole_flat : bool or (bool, bool), optional

    See Also
    --------
    # BivariateSpline ：双变量样条的基类。
    BivariateSpline :
    # UnivariateSpline ：用于拟合给定数据点的平滑单变量样条。
    UnivariateSpline :
    # SmoothBivariateSpline ：通过给定点进行平滑的双变量样条。
    SmoothBivariateSpline :
    # LSQBivariateSpline ：使用加权最小二乘拟合的双变量样条。
    LSQBivariateSpline :
    # SmoothSphereBivariateSpline ：球坐标中的平滑双变量样条。
    SmoothSphereBivariateSpline :
    # LSQSphereBivariateSpline ：在球坐标中使用加权最小二乘拟合的双变量样条。
    LSQSphereBivariateSpline :
    # RectBivariateSpline ：在矩形网格上的双变量样条。
    RectBivariateSpline :
    # bisplrep ：找到一个曲面的双变量 B-样条表示的函数。
    bisplrep :
    # bisplev ：评估双变量 B-样条及其导数的函数。
    bisplev :

    Notes
    -----
    # 目前仅支持平滑样条逼近（FITPACK 程序中的 iopt[0] = 0 和 iopt[0] = 1）。精确的最小二乘样条逼近尚未实现。
    Currently, only the smoothing spline approximation (``iopt[0] = 0`` and
    ``iopt[0] = 1`` in the FITPACK routine) is supported.  The exact
    least-squares spline approximation is not implemented yet.
    
    # 在实际进行插值时，请求的 v 值必须位于原始 v 值选择的长度为 2pi 的同一区间内。
    When actually performing the interpolation, the requested `v` values must
    lie within the same length 2pi interval that the original `v` values were
    chosen from.
    
    # 更多信息，请参阅 FITPACK_ 网站关于此函数的介绍。
    For more information, see the FITPACK_ site about this function.

    .. _FITPACK: http://www.netlib.org/dierckx/spgrid.f

    Examples
    --------
    # 假设我们有全球数据的粗网格
    Suppose we have global data on a coarse grid

    # 导入必要的库
    >>> import numpy as np
    # 生成经度和纬度的粗略网格
    >>> lats = np.linspace(10, 170, 9) * np.pi / 180.
    >>> lons = np.linspace(0, 350, 18) * np.pi / 180.
    # 生成粗网格上的数据
    >>> data = np.dot(np.atleast_2d(90. - np.linspace(-80., 80., 18)).T,
    ...               np.atleast_2d(180. - np.abs(np.linspace(0., 350., 9)))).T

    # 我们想要将其插值到全球的一度网格
    We want to interpolate it to a global one-degree grid

    # 生成新的经度和纬度网格
    >>> new_lats = np.linspace(1, 180, 180) * np.pi / 180
    >>> new_lons = np.linspace(1, 360, 360) * np.pi / 180
    >>> new_lats, new_lons = np.meshgrid(new_lats, new_lons)

    # 需要设置插值器对象
    We need to set up the interpolator object

    # 导入 RectSphereBivariateSpline 类
    >>> from scipy.interpolate import RectSphereBivariateSpline
    # 创建 RectSphereBivariateSpline 对象 lut
    >>> lut = RectSphereBivariateSpline(lats, lons, data)

    # 最后，进行数据插值。RectSphereBivariateSpline 对象
    Finally we interpolate the data.  The `RectSphereBivariateSpline` object
    """

    定义一个类，继承自 SphereBivariateSpline 类，表示球面双变量样条插值器。

    def __call__(self, theta, phi, dtheta=0, dphi=0, grid=True):
        # 将输入的 theta 和 phi 转换为 NumPy 数组
        theta = np.asarray(theta)
        phi = np.asarray(phi)

        # 调用父类 SphereBivariateSpline 的 __call__ 方法，进行球面双变量样条插值
        return SphereBivariateSpline.__call__(self, theta, phi, dtheta=dtheta,
                                              dphi=dphi, grid=grid)
    ```
```