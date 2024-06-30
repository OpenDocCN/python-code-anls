# `D:\src\scipysrc\scipy\scipy\interpolate\_polyint.py`

```
# 导入警告模块，用于处理警告信息
import warnings

# 导入 NumPy 库，用于科学计算
import numpy as np

# 导入 SciPy 库中的阶乘函数
from scipy.special import factorial

# 导入 SciPy 库中的数组验证函数和浮点数阶乘函数
from scipy._lib._util import _asarray_validated, float_factorial, check_random_state

# 定义本模块中可以导出的公共接口列表
__all__ = ["KroghInterpolator", "krogh_interpolate",
           "BarycentricInterpolator", "barycentric_interpolate",
           "approximate_taylor_polynomial"]

# 定义一个函数，用于检查输入是否为标量类型或者零维
def _isscalar(x):
    """Check whether x is if a scalar type, or 0-dim"""
    return np.isscalar(x) or hasattr(x, 'shape') and x.shape == ()

# 定义一个类，用于处理一维插值的通用特性
class _Interpolator1D:
    """
    Common features in univariate interpolation

    Deal with input data type and interpolation axis rolling. The
    actual interpolator can assume the y-data is of shape (n, r) where
    `n` is the number of x-points, and `r` the number of variables,
    and use self.dtype as the y-data type.

    Attributes
    ----------
    _y_axis
        Axis along which the interpolation goes in the original array
    _y_extra_shape
        Additional trailing shape of the input arrays, excluding
        the interpolation axis.
    dtype
        Dtype of the y-data arrays. Can be set via _set_dtype, which
        forces it to be float or complex.

    Methods
    -------
    __call__
    _prepare_x
    _finish_y
    _reshape_yi
    _set_yi
    _set_dtype
    _evaluate

    """

    # 限制类的属性只能为以下几个
    __slots__ = ('_y_axis', '_y_extra_shape', 'dtype')

    # 类的初始化方法，设置插值的轴和数据类型
    def __init__(self, xi=None, yi=None, axis=None):
        self._y_axis = axis  # 设置插值轴
        self._y_extra_shape = None  # 额外的输入数组形状，不包括插值轴
        self.dtype = None  # y 数据数组的数据类型
        if yi is not None:
            self._set_yi(yi, xi=xi, axis=axis)  # 设置 y 数据

    # 类的调用方法，用于评估插值
    def __call__(self, x):
        """
        Evaluate the interpolant

        Parameters
        ----------
        x : array_like
            Point or points at which to evaluate the interpolant.

        Returns
        -------
        y : array_like
            Interpolated values. Shape is determined by replacing
            the interpolation axis in the original array with the shape of `x`.

        Notes
        -----
        Input values `x` must be convertible to `float` values like `int`
        or `float`.

        """
        x, x_shape = self._prepare_x(x)  # 准备输入的 x 数组并将其展平为一维
        y = self._evaluate(x)  # 调用 _evaluate 方法进行插值计算
        return self._finish_y(y, x_shape)  # 返回最终的插值结果

    # 实际执行插值计算的方法，需子类实现
    def _evaluate(self, x):
        """
        Actually evaluate the value of the interpolator.
        """
        raise NotImplementedError()

    # 准备输入 x 数组并将其展平为一维的方法
    def _prepare_x(self, x):
        """Reshape input x array to 1-D"""
        x = _asarray_validated(x, check_finite=False, as_inexact=True)  # 验证并转换输入 x 为精确浮点数数组
        x_shape = x.shape  # 记录原始 x 的形状
        return x.ravel(), x_shape  # 返回展平后的 x 数组及其原始形状
    # 将插值后的 y 重塑为类似初始 y 的 N 维数组
    def _finish_y(self, y, x_shape):
        y = y.reshape(x_shape + self._y_extra_shape)  # 将 y 重塑为 x_shape 加上额外形状 _y_extra_shape
        if self._y_axis != 0 and x_shape != ():
            nx = len(x_shape)
            ny = len(self._y_extra_shape)
            # 构建用于 transpose 的轴顺序 s
            s = (list(range(nx, nx + self._y_axis))
                 + list(range(nx)) + list(range(nx+self._y_axis, nx+ny)))
            y = y.transpose(s)  # 对 y 应用轴顺序 s 的转置操作
        return y

    # 将插值后的 yi 重新排列，如果需要检查，则检查形状
    def _reshape_yi(self, yi, check=False):
        yi = np.moveaxis(np.asarray(yi), self._y_axis, 0)  # 移动 yi 的轴，使其插值轴位于最前面
        if check and yi.shape[1:] != self._y_extra_shape:
            ok_shape = "{!r} + (N,) + {!r}".format(self._y_extra_shape[-self._y_axis:],
                                                   self._y_extra_shape[:-self._y_axis])
            raise ValueError("Data must be of shape %s" % ok_shape)  # 如果需要检查并且形状不匹配，则引发错误
        return yi.reshape((yi.shape[0], -1))  # 将 yi 重塑为形状 (yi.shape[0], -1)

    # 设置插值后的 yi，可选设置插值轴 axis
    def _set_yi(self, yi, xi=None, axis=None):
        if axis is None:
            axis = self._y_axis  # 如果未指定轴，则使用当前插值轴 _y_axis
        if axis is None:
            raise ValueError("no interpolation axis specified")  # 如果未指定插值轴，则引发错误

        yi = np.asarray(yi)  # 将 yi 转换为 ndarray

        shape = yi.shape
        if shape == ():
            shape = (1,)  # 如果 yi 是标量，则将其形状设置为 (1,)
        if xi is not None and shape[axis] != len(xi):
            raise ValueError("x and y arrays must be equal in length along "
                             "interpolation axis.")  # 如果 xi 存在且沿插值轴的长度与 yi 不匹配，则引发错误

        self._y_axis = (axis % yi.ndim)  # 设置 _y_axis 为插值轴的模数
        self._y_extra_shape = yi.shape[:self._y_axis] + yi.shape[self._y_axis+1:]  # 计算并设置 _y_extra_shape
        self.dtype = None  # 重置 dtype
        self._set_dtype(yi.dtype)  # 设置插值后的 yi 的数据类型

    # 设置数据类型 dtype，如果 union 为 True，则进行合并
    def _set_dtype(self, dtype, union=False):
        if np.issubdtype(dtype, np.complexfloating) \
               or np.issubdtype(self.dtype, np.complexfloating):
            self.dtype = np.complex128  # 如果 dtype 或 self.dtype 是复数浮点数，则设置 dtype 为复数浮点数 np.complex128
        else:
            if not union or self.dtype != np.complex128:
                self.dtype = np.float64  # 否则设置 dtype 为 np.float64
class _Interpolator1DWithDerivatives(_Interpolator1D):
    # 继承自 _Interpolator1D 的类 _Interpolator1DWithDerivatives，用于一维插值和求导数

    def derivatives(self, x, der=None):
        """
        Evaluate several derivatives of the polynomial at the point `x`

        Produce an array of derivatives evaluated at the point `x`.

        Parameters
        ----------
        x : array_like
            Point or points at which to evaluate the derivatives
        der : int or list or None, optional
            How many derivatives to evaluate, or None for all potentially
            nonzero derivatives (that is, a number equal to the number
            of points), or a list of derivatives to evaluate. This number
            includes the function value as the '0th' derivative.

        Returns
        -------
        d : ndarray
            Array with derivatives; ``d[j]`` contains the jth derivative.
            Shape of ``d[j]`` is determined by replacing the interpolation
            axis in the original array with the shape of `x`.

        Examples
        --------
        >>> from scipy.interpolate import KroghInterpolator
        >>> KroghInterpolator([0,0,0],[1,2,3]).derivatives(0)
        array([1.0,2.0,3.0])
        >>> KroghInterpolator([0,0,0],[1,2,3]).derivatives([0,0])
        array([[1.0,1.0],
               [2.0,2.0],
               [3.0,3.0]])

        """
        # 准备 x 和 x 的形状
        x, x_shape = self._prepare_x(x)
        # 计算在点 x 处的多个导数
        y = self._evaluate_derivatives(x, der)

        # 调整返回数组的形状以匹配输入 x 的形状
        y = y.reshape((y.shape[0],) + x_shape + self._y_extra_shape)
        
        # 根据插值轴和输入 x 的形状，调整返回数组的维度顺序
        if self._y_axis != 0 and x_shape != ():
            nx = len(x_shape)
            ny = len(self._y_extra_shape)
            s = ([0] + list(range(nx+1, nx + self._y_axis+1))
                 + list(range(1, nx+1)) +
                 list(range(nx+1+self._y_axis, nx+ny+1)))
            y = y.transpose(s)
        
        # 返回计算得到的导数数组
        return y

    def derivative(self, x, der=1):
        """
        Evaluate a single derivative of the polynomial at the point `x`.

        Parameters
        ----------
        x : array_like
            Point or points at which to evaluate the derivatives

        der : integer, optional
            Which derivative to evaluate (default: first derivative).
            This number includes the function value as 0th derivative.

        Returns
        -------
        d : ndarray
            Derivative interpolated at the x-points. Shape of `d` is
            determined by replacing the interpolation axis in the
            original array with the shape of `x`.

        Notes
        -----
        This may be computed by evaluating all derivatives up to the desired
        one (using self.derivatives()) and then discarding the rest.

        """
        # 准备 x 和 x 的形状
        x, x_shape = self._prepare_x(x)
        # 计算直到所需导数的所有导数
        y = self._evaluate_derivatives(x, der+1)
        
        # 返回所需导数在 x 点上的插值结果
        return self._finish_y(y[der], x_shape)
    # 定义一个方法 `_evaluate_derivatives`，用于计算导数值，但是这里只是抛出一个未实现的错误
    def _evaluate_derivatives(self, x, der=None):
        """
        Actually evaluate the derivatives.

        Parameters
        ----------
        x : array_like
            1D array of points at which to evaluate the derivatives
        der : integer, optional
            The number of derivatives to evaluate, from 'order 0' (der=1)
            to order der-1.  If omitted, return all possibly-non-zero
            derivatives, ie 0 to order n-1.

        Returns
        -------
        d : ndarray
            Array of shape ``(der, x.size, self.yi.shape[1])`` containing
            the derivatives from 0 to der-1
        """
        # 抛出一个未实现错误，提醒调用方需要在子类中实现这个方法
        raise NotImplementedError()
class KroghInterpolator(_Interpolator1DWithDerivatives):
    """
    Interpolating polynomial for a set of points.

    The polynomial passes through all the pairs ``(xi, yi)``. One may
    additionally specify a number of derivatives at each point `xi`;
    this is done by repeating the value `xi` and specifying the
    derivatives as successive `yi` values.

    Allows evaluation of the polynomial and all its derivatives.
    For reasons of numerical stability, this function does not compute
    the coefficients of the polynomial, although they can be obtained
    by evaluating all the derivatives.

    Parameters
    ----------
    xi : array_like, shape (npoints, )
        Known x-coordinates. Must be sorted in increasing order.
    yi : array_like, shape (..., npoints, ...)
        Known y-coordinates. When an xi occurs two or more times in
        a row, the corresponding yi's represent derivative values. The length of `yi`
        along the interpolation axis must be equal to the length of `xi`. Use the
        `axis` parameter to select the correct axis.
    axis : int, optional
        Axis in the `yi` array corresponding to the x-coordinate values. Defaults to
        ``axis=0``.

    Notes
    -----
    Be aware that the algorithms implemented here are not necessarily
    the most numerically stable known. Moreover, even in a world of
    exact computation, unless the x coordinates are chosen very
    carefully - Chebyshev zeros (e.g., cos(i*pi/n)) are a good choice -
    polynomial interpolation itself is a very ill-conditioned process
    due to the Runge phenomenon. In general, even with well-chosen
    x values, degrees higher than about thirty cause problems with
    numerical instability in this code.

    Based on [1]_.

    References
    ----------
    .. [1] Krogh, "Efficient Algorithms for Polynomial Interpolation
        and Numerical Differentiation", 1970.

    Examples
    --------
    To produce a polynomial that is zero at 0 and 1 and has
    derivative 2 at 0, call

    >>> from scipy.interpolate import KroghInterpolator
    >>> KroghInterpolator([0,0,1],[0,2,0])

    This constructs the quadratic :math:`2x^2-2x`. The derivative condition
    is indicated by the repeated zero in the `xi` array; the corresponding
    yi values are 0, the function value, and 2, the derivative value.

    For another example, given `xi`, `yi`, and a derivative `ypi` for each
    point, appropriate arrays can be constructed as:

    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> xi = np.linspace(0, 1, 5)
    >>> yi, ypi = rng.random((2, 5))
    >>> xi_k, yi_k = np.repeat(xi, 2), np.ravel(np.dstack((yi,ypi)))
    >>> KroghInterpolator(xi_k, yi_k)

    To produce a vector-valued polynomial, supply a higher-dimensional
    array for `yi`:

    >>> KroghInterpolator([0,1],[[2,3],[4,5]])

    This constructs a linear polynomial giving (2,3) at 0 and (4,5) at 1.
    """
    # 初始化函数，接受输入 xi, yi 以及可选参数 axis，默认调用父类的初始化方法
    def __init__(self, xi, yi, axis=0):
        # 调用父类的初始化方法
        super().__init__(xi, yi, axis)

        # 将输入的 xi 转换为 NumPy 数组
        self.xi = np.asarray(xi)
        # 将输入的 yi 调整为合适的形状，并存储在实例变量 yi 中
        self.yi = self._reshape_yi(yi)
        # 获取 yi 的形状信息，n 表示行数，r 表示列数
        self.n, self.r = self.yi.shape

        # 检查 xi 的长度是否大于30，若是则发出警告
        if (deg := self.xi.size) > 30:
            warnings.warn(f"{deg} degrees provided, degrees higher than about"
                          " thirty cause problems with numerical instability "
                          "with 'KroghInterpolator'", stacklevel=2)

        # 初始化系数矩阵 c，用零填充，形状为 (n+1, r)
        c = np.zeros((self.n+1, self.r), dtype=self.dtype)
        # 将第一行赋值为 yi 的第一行
        c[0] = self.yi[0]
        # 初始化 Vk 矩阵，用零填充，形状为 (n, r)
        Vk = np.zeros((self.n, self.r), dtype=self.dtype)
        
        # 开始迭代计算 Vk 和 c 的值
        for k in range(1, self.n):
            s = 0
            # 计算 s 的值，满足条件时递增 s
            while s <= k and xi[k-s] == xi[k]:
                s += 1
            s -= 1
            # 计算 Vk[0] 的值
            Vk[0] = self.yi[k] / float_factorial(s)
            # 根据公式计算 Vk 和 c 的其他值
            for i in range(k - s):
                if xi[i] == xi[k]:
                    # 如果 xi 中的元素相等，则抛出 ValueError 异常
                    raise ValueError("Elements of `xi` can't be equal.")
                if s == 0:
                    Vk[i + 1] = (c[i] - Vk[i]) / (xi[i] - xi[k])
                else:
                    Vk[i + 1] = (Vk[i + 1] - Vk[i]) / (xi[i] - xi[k])
            c[k] = Vk[k - s]
        
        # 将最终计算得到的系数矩阵 c 赋值给实例变量 self.c
        self.c = c

    # 私有方法，用于在给定输入 x 的情况下计算插值函数的值
    def _evaluate(self, x):
        # 初始化 pi 为单位矩阵，形状为 (len(x), r)
        pi = 1
        # 初始化 p 为零矩阵，形状为 (len(x), r)
        p = np.zeros((len(x), self.r), dtype=self.dtype)
        # 将 c 的第一行加到 p 上
        p += self.c[0, np.newaxis, :]

        # 开始迭代计算 p 的值
        for k in range(1, self.n):
            # 计算 w，即 x 减去 xi[k-1]
            w = x - self.xi[k - 1]
            # 更新 pi
            pi = w * pi
            # 更新 p
            p += pi[:, np.newaxis] * self.c[k]

        # 返回最终计算得到的 p
        return p

    # 私有方法，用于在给定输入 x 和可选参数 der 的情况下计算插值函数的导数
    def _evaluate_derivatives(self, x, der=None):
        # 获取 n 和 r 的值
        n = self.n
        r = self.r

        # 如果 der 为 None，则设置其值为 n
        if der is None:
            der = self.n

        # 初始化 pi 和 w 矩阵
        pi = np.zeros((n, len(x)))
        w = np.zeros((n, len(x)))
        pi[0] = 1
        # 初始化 p 为零矩阵，形状为 (len(x), r)
        p = np.zeros((len(x), self.r), dtype=self.dtype)
        # 将 c 的第一行加到 p 上
        p += self.c[0, np.newaxis, :]

        # 开始迭代计算 p 的值
        for k in range(1, n):
            # 计算 w[k-1]
            w[k - 1] = x - self.xi[k - 1]
            # 更新 pi[k]
            pi[k] = w[k - 1] * pi[k - 1]
            # 更新 p
            p += pi[k, :, np.newaxis] * self.c[k]

        # 初始化 cn 矩阵，形状为 (max(der, n+1), len(x), r)
        cn = np.zeros((max(der, n + 1), len(x), r), dtype=self.dtype)
        # 将 c 的前 n+1 行加到 cn 上，并扩展维度
        cn[:n + 1, :, :] += self.c[:n + 1, np.newaxis, :]
        # 将 p 赋值给 cn 的第一行
        cn[0] = p

        # 开始迭代计算 cn 的值
        for k in range(1, n):
            for i in range(1, n - k + 1):
                # 更新 pi[i]
                pi[i] = w[k + i - 1] * pi[i - 1] + pi[i]
                # 更新 cn[k]
                cn[k] = cn[k] + pi[i, :, np.newaxis] * cn[k + i]
            # 对 cn[k] 进行阶乘计算
            cn[k] *= float_factorial(k)

        # 将 cn[n] 的值设为零矩阵，并返回 cn[:der] 的值
        cn[n, :, :] = 0
        return cn[:der]
# 定义了一个函数用于计算 Krogh 插值多项式
def krogh_interpolate(xi, yi, x, der=0, axis=0):
    """
    Convenience function for polynomial interpolation.

    See `KroghInterpolator` for more details.

    Parameters
    ----------
    xi : array_like
        Interpolation points (known x-coordinates).
    yi : array_like
        Known y-coordinates, of shape ``(xi.size, R)``. Interpreted as
        vectors of length R, or scalars if R=1.
    x : array_like
        Point or points at which to evaluate the derivatives.
    der : int or list or None, optional
        How many derivatives to evaluate, or None for all potentially
        nonzero derivatives (that is, a number equal to the number
        of points), or a list of derivatives to evaluate. This number
        includes the function value as the '0th' derivative.
    axis : int, optional
        Axis in the `yi` array corresponding to the x-coordinate values.

    Returns
    -------
    d : ndarray
        If the interpolator's values are R-D then the
        returned array will be the number of derivatives by N by R.
        If `x` is a scalar, the middle dimension will be dropped; if
        the `yi` are scalars then the last dimension will be dropped.

    See Also
    --------
    KroghInterpolator : Krogh interpolator

    Notes
    -----
    Construction of the interpolating polynomial is a relatively expensive
    process. If you want to evaluate it repeatedly consider using the class
    KroghInterpolator (which is what this function uses).

    Examples
    --------
    We can interpolate 2D observed data using Krogh interpolation:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.interpolate import krogh_interpolate
    >>> x_observed = np.linspace(0.0, 10.0, 11)
    >>> y_observed = np.sin(x_observed)
    >>> x = np.linspace(min(x_observed), max(x_observed), num=100)
    >>> y = krogh_interpolate(x_observed, y_observed, x)
    >>> plt.plot(x_observed, y_observed, "o", label="observation")
    >>> plt.plot(x, y, label="krogh interpolation")
    >>> plt.legend()
    >>> plt.show()
    """

    # 使用 KroghInterpolator 类来创建一个插值对象 P
    P = KroghInterpolator(xi, yi, axis=axis)
    
    # 根据参数 der 的不同取值，计算并返回不同阶数的导数或插值结果
    if der == 0:
        return P(x)
    elif _isscalar(der):
        return P.derivative(x, der=der)
    else:
        return P.derivatives(x, der=np.amax(der)+1)[der]


# 定义了一个函数用于在给定点 x 处估计函数 f 的 Taylor 多项式
def approximate_taylor_polynomial(f, x, degree, scale, order=None):
    """
    Estimate the Taylor polynomial of f at x by polynomial fitting.

    Parameters
    ----------
    f : callable
        The function whose Taylor polynomial is sought. Should accept
        a vector of `x` values.
    x : scalar
        The point at which the polynomial is to be evaluated.
    degree : int
        The degree of the Taylor polynomial
    scale : scalar
        The width of the interval to use to evaluate the Taylor polynomial.
        Function values spread over a range this wide are used to fit the
        polynomial. Must be chosen carefully.
    """
    if order is None:
        # 如果未指定 order，则使用 degree
        order = degree

    n = order+1
    # 选择 n 个点，这些点靠近区间的端点，以避免朗格现象。确保其中一个点恰好落在 x 处。
    xs = scale*np.cos(np.linspace(0,np.pi,n,endpoint=n % 1)) + x

    # 使用 KroghInterpolator 插值器创建 P 对象，并计算在 x 处的 degree+1 阶导数
    P = KroghInterpolator(xs, f(xs))
    d = P.derivatives(x,der=degree+1)

    # 构造并返回一个 poly1d 对象，其系数由 d 除以阶乘数组后倒序得到
    return np.poly1d((d/factorial(np.arange(degree+1)))[::-1])
# 定义一个类 BarycentricInterpolator，继承自 _Interpolator1DWithDerivatives 类
# 这个类用于通过一组给定的点构造插值多项式，允许对多项式及其所有导数进行评估，
# 有效地更改要插值的 y 值，并通过添加更多的 x 和 y 值来更新插值
class BarycentricInterpolator(_Interpolator1DWithDerivatives):
    
    # 构造函数，初始化插值器对象
    def __init__(self, xi, yi=None, axis=0, wi=None, random_state=None):
        r"""Interpolating polynomial for a set of points.
        
        Constructs a polynomial that passes through a given set of points.
        Allows evaluation of the polynomial and all its derivatives,
        efficient changing of the y-values to be interpolated,
        and updating by adding more x- and y-values.

        For reasons of numerical stability, this function does not compute
        the coefficients of the polynomial.

        The values `yi` need to be provided before the function is
        evaluated, but none of the preprocessing depends on them, so rapid
        updates are possible.

        Parameters
        ----------
        xi : array_like, shape (npoints, )
            1-D array of x coordinates of the points the polynomial
            should pass through
        yi : array_like, shape (..., npoints, ...), optional
            N-D array of y coordinates of the points the polynomial should pass through.
            If None, the y values will be supplied later via the `set_y` method.
            The length of `yi` along the interpolation axis must be equal to the length
            of `xi`. Use the ``axis`` parameter to select correct axis.
        axis : int, optional
            Axis in the yi array corresponding to the x-coordinate values. Defaults
            to ``axis=0``.
        wi : array_like, optional
            The barycentric weights for the chosen interpolation points `xi`.
            If absent or None, the weights will be computed from `xi` (default).
            This allows for the reuse of the weights `wi` if several interpolants
            are being calculated using the same nodes `xi`, without re-computation.
        random_state : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
            If `seed` is None (or `np.random`), the `numpy.random.RandomState`
            singleton is used.
            If `seed` is an int, a new ``RandomState`` instance is used,
            seeded with `seed`.
            If `seed` is already a ``Generator`` or ``RandomState`` instance then
            that instance is used.

        Notes
        -----
        This class uses a "barycentric interpolation" method that treats
        the problem as a special case of rational function interpolation.
        This algorithm is quite stable, numerically, but even in a world of
        exact computation, unless the x coordinates are chosen very
        carefully - Chebyshev zeros (e.g., cos(i*pi/n)) are a good choice -
        polynomial interpolation itself is a very ill-conditioned process
        due to the Runge phenomenon.

        Based on Berrut and Trefethen 2004, "Barycentric Lagrange Interpolation".

        Examples
        --------
        To produce a quintic barycentric interpolant approximating the function
        :math:`\sin x`, and its first four derivatives, using six randomly-spaced
        nodes in :math:`(0, \frac{\pi}{2})`:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from scipy.interpolate import BarycentricInterpolator
        >>> rng = np.random.default_rng()

        """
        # 调用父类 _Interpolator1DWithDerivatives 的构造函数
        super().__init__()

        # 初始化属性 xi
        self.xi = xi

        # 如果提供了 yi，则初始化属性 yi
        if yi is not None:
            self.set_y(yi, axis=axis)

        # 初始化属性 wi，如果没有提供则根据 xi 计算权重
        if wi is None:
            self.compute_weights()
        else:
            self.wi = wi

        # 初始化随机数生成器
        self.random_state = random_state

    # ...
    >>> xi = rng.random(6) * np.pi/2
    # 生成一个包含 6 个随机数的数组 xi，每个随机数乘以 pi/2
    >>> f, f_d1, f_d2, f_d3, f_d4 = np.sin, np.cos, lambda x: -np.sin(x), lambda x: -np.cos(x), np.sin
    # 定义函数 f 为 sin 函数，f_d1 为 cos 函数，f_d2 和 f_d3 是对应的一阶和二阶导数的匿名函数，f_d4 为 sin 函数
    >>> P = BarycentricInterpolator(xi, f(xi), random_state=rng)
    # 使用 BarycentricInterpolator 类基于 xi 和 f(xi) 进行插值，random_state 参数使用了给定的随机数生成器 rng
    >>> fig, axs = plt.subplots(5, 1, sharex=True, layout='constrained', figsize=(7,10))
    # 创建包含 5 个子图的图形窗口，共享 x 轴，布局约束为 constrained，尺寸为 7x10
    >>> x = np.linspace(0, np.pi, 100)
    # 在区间 [0, pi] 上生成 100 个均匀间隔的点作为 x 值
    >>> axs[0].plot(x, P(x), 'r:', x, f(x), 'k--', xi, f(xi), 'xk')
    # 在第一个子图上绘制 P(x) 的插值曲线（红色虚线），f(x) 的真实函数曲线（黑色虚线），以及插值节点 xi 的散点（黑色 x 标记）
    >>> axs[1].plot(x, P.derivative(x), 'r:', x, f_d1(x), 'k--', xi, f_d1(xi), 'xk')
    # 在第二个子图上绘制 P.derivative(x) 的插值曲线的一阶导数（红色虚线），f_d1(x) 的一阶导数真实函数曲线（黑色虚线），以及插值节点 xi 的一阶导数散点（黑色 x 标记）
    >>> axs[2].plot(x, P.derivative(x, 2), 'r:', x, f_d2(x), 'k--', xi, f_d2(xi), 'xk')
    # 在第三个子图上绘制 P.derivative(x, 2) 的插值曲线的二阶导数（红色虚线），f_d2(x) 的二阶导数真实函数曲线（黑色虚线），以及插值节点 xi 的二阶导数散点（黑色 x 标记）
    >>> axs[3].plot(x, P.derivative(x, 3), 'r:', x, f_d3(x), 'k--', xi, f_d3(xi), 'xk')
    # 在第四个子图上绘制 P.derivative(x, 3) 的插值曲线的三阶导数（红色虚线），f_d3(x) 的三阶导数真实函数曲线（黑色虚线），以及插值节点 xi 的三阶导数散点（黑色 x 标记）
    >>> axs[4].plot(x, P.derivative(x, 4), 'r:', x, f_d4(x), 'k--', xi, f_d4(xi), 'xk')
    # 在第五个子图上绘制 P.derivative(x, 4) 的插值曲线的四阶导数（红色虚线），f_d4(x) 的四阶导数真实函数曲线（黑色虚线），以及插值节点 xi 的四阶导数散点（黑色 x 标记）
    >>> axs[0].set_xlim(0, np.pi)
    # 设置第一个子图的 x 轴范围为 [0, pi]
    >>> axs[4].set_xlabel(r"$x$")
    # 设置第五个子图的 x 轴标签为 "$x$"
    >>> axs[4].set_xticks([i * np.pi / 4 for i in range(5)],
    ...                   ["0", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$"])
    # 设置第五个子图的 x 轴刻度为 [0, pi/4, pi/2, 3pi/4, pi]，并对应地设置刻度标签
    >>> axs[0].set_ylabel("$f(x)$")
    # 设置第一个子图的 y 轴标签为 "$f(x)$"
    >>> axs[1].set_ylabel("$f'(x)$")
    # 设置第二个子图的 y 轴标签为 "$f'(x)$"
    >>> axs[2].set_ylabel("$f''(x)$")
    # 设置第三个子图的 y 轴标签为 "$f''(x)$"
    >>> axs[3].set_ylabel("$f^{(3)}(x)$")
    # 设置第四个子图的 y 轴标签为 "$f^{(3)}(x)$"
    >>> axs[4].set_ylabel("$f^{(4)}(x)$")
    # 设置第五个子图的 y 轴标签为 "$f^{(4)}(x)$"
    >>> labels = ['Interpolation nodes', 'True function $f$', 'Barycentric interpolation']
    # 创建一个标签列表，包括 'Interpolation nodes', 'True function $f$', 'Barycentric interpolation'
    >>> axs[0].legend(axs[0].get_lines()[::-1], labels, bbox_to_anchor=(0., 1.02, 1., .102),
    ...               loc='lower left', ncols=3, mode="expand", borderaxespad=0., frameon=False)
    # 在第一个子图上创建图例，图例包含在 axs[0].get_lines() 返回的线条对象，以 labels 列表为标签，位于图表的左下角，分为 3 列扩展显示，无边框
    >>> plt.show()
    # 显示绘制好的图形
    """ # numpy/numpydoc#87  # noqa: E501
    # 初始化函数，接受输入 xi, yi 可选，axis 默认为 0，以及其他参数 wi 和 random_state
    def __init__(self, xi, yi=None, axis=0, *, wi=None, random_state=None):
        # 调用父类的初始化函数，将 xi, yi, axis 传递给父类
        super().__init__(xi, yi, axis)
        
        # 确保 random_state 是有效的随机状态对象
        random_state = check_random_state(random_state)

        # 将输入 xi 转换为 numpy 数组，类型为 np.float64
        self.xi = np.asarray(xi, dtype=np.float64)
        # 设置 yi 的值
        self.set_yi(yi)
        # 计算 xi 的长度并存储在 self.n 中
        self.n = len(self.xi)

        # 缓存导数对象以避免每次调用时重新计算权重
        self._diff_cij = None

        # 如果提供了 wi，则使用提供的值，否则按照 Berrut 和 Trefethen 2004 年的建议计算
        if wi is not None:
            self.wi = wi
        else:
            # 计算容量比例的倒数，以用于加权计算
            self._inv_capacity = 4.0 / (np.max(self.xi) - np.min(self.xi))
            # 生成 xi 长度的随机排列
            permute = random_state.permutation(self.n, )
            # 创建一个逆置排列数组
            inv_permute = np.zeros(self.n, dtype=np.int32)
            inv_permute[permute] = np.arange(self.n)
            # 初始化权重数组
            self.wi = np.zeros(self.n)

            # 计算权重
            for i in range(self.n):
                # 计算 xi[i] 到 permute[i] 的距离
                dist = self._inv_capacity * (self.xi[i] - self.xi[permute])
                # 对于 inv_permute[i]，距离设置为 1.0
                dist[inv_permute[i]] = 1.0
                # 计算距离的乘积
                prod = np.prod(dist)
                # 如果乘积为零，抛出值错误
                if prod == 0.0:
                    raise ValueError("Interpolation points xi must be"
                                     " distinct.")
                # 计算最终的权重值
                self.wi[i] = 1.0 / prod


    # 设置 yi 的函数，用于更新要插值的 y 值
    def set_yi(self, yi, axis=None):
        """
        Update the y values to be interpolated

        The barycentric interpolation algorithm requires the calculation
        of weights, but these depend only on the `xi`. The `yi` can be changed
        at any time.

        Parameters
        ----------
        yi : array_like
            The y-coordinates of the points the polynomial will pass through.
            If None, the y values must be supplied later.
        axis : int, optional
            Axis in the `yi` array corresponding to the x-coordinate values.

        """
        # 如果 yi 为 None，则将 self.yi 设置为 None 并返回
        if yi is None:
            self.yi = None
            return
        
        # 调用内部函数 _set_yi 来设置 yi 的值，传递 xi 和 axis 参数
        self._set_yi(yi, xi=self.xi, axis=axis)
        # 将设置后的 yi 重新形状化并存储在 self.yi 中
        self.yi = self._reshape_yi(yi)
        # 计算并存储 yi 的形状信息 self.n 和 self.r
        self.n, self.r = self.yi.shape
        # 将内部差分插值对象设为 None，表示需要重新计算
        self._diff_baryint = None
    def add_xi(self, xi, yi=None):
        """
        Add more x values to the set to be interpolated
        
        The barycentric interpolation algorithm allows easy updating by
        adding more points for the polynomial to pass through.
        
        Parameters
        ----------
        xi : array_like
            The x coordinates of the points that the polynomial should pass
            through.
        yi : array_like, optional
            The y coordinates of the points the polynomial should pass through.
            Should have shape ``(xi.size, R)``; if R > 1 then the polynomial is
            vector-valued.
            If `yi` is not given, the y values will be supplied later. `yi`
            should be given if and only if the interpolator has y values
            specified.
        
        Notes
        -----
        The new points added by `add_xi` are not randomly permuted
        so there is potential for numerical instability,
        especially for a large number of points. If this
        happens, please reconstruct interpolation from scratch instead.
        """
        # 如果给定了 yi 参数
        if yi is not None:
            # 如果之前没有设置过 yi 值，则引发错误
            if self.yi is None:
                raise ValueError("No previous yi value to update!")
            # 调整 yi 的形状，并更新到 self.yi
            yi = self._reshape_yi(yi, check=True)
            self.yi = np.vstack((self.yi, yi))
        else:
            # 如果没有提供 yi 参数，则检查是否已经设置过 yi 值，如果是则引发错误
            if self.yi is not None:
                raise ValueError("No update to yi provided!")
        
        # 保存旧的点数
        old_n = self.n
        # 将新的 x 值连接到现有的 xi 中
        self.xi = np.concatenate((self.xi, xi))
        # 更新点数计数
        self.n = len(self.xi)
        # 更新权重的倒数
        self.wi **= -1
        old_wi = self.wi
        # 重新初始化权重数组
        self.wi = np.zeros(self.n)
        # 复制旧的权重值
        self.wi[:old_n] = old_wi
        
        # 更新新加入的权重值
        for j in range(old_n, self.n):
            self.wi[:j] *= self._inv_capacity * (self.xi[j] - self.xi[:j])
            self.wi[j] = np.multiply.reduce(
                self._inv_capacity * (self.xi[:j] - self.xi[j])
            )
        
        # 再次更新权重的倒数
        self.wi **= -1
        # 清空之前计算的差分值和插值多项式
        self._diff_cij = None
        self._diff_baryint = None

    def __call__(self, x):
        """Evaluate the interpolating polynomial at the points x
        
        Parameters
        ----------
        x : array_like
            Point or points at which to evaluate the interpolant.
        
        Returns
        -------
        y : array_like
            Interpolated values. Shape is determined by replacing
            the interpolation axis in the original array with the shape of `x`.
        
        Notes
        -----
        Currently the code computes an outer product between `x` and the
        weights, that is, it constructs an intermediate array of size
        ``(N, len(x))``, where N is the degree of the polynomial.
        """
        # 调用 _Interpolator1D 类的 __call__ 方法，返回插值多项式在 x 处的值
        return _Interpolator1D.__call__(self, x)
    def _evaluate(self, x):
        # 如果输入的 x 是空的，则创建一个空的数组 p
        if x.size == 0:
            p = np.zeros((0, self.r), dtype=self.dtype)
        else:
            # 计算输入 x 与内部数据 xi 的差值
            c = x[..., np.newaxis] - self.xi
            # 找出差值为零的位置
            z = c == 0
            # 将差值为零的位置置为1（避免除零错误）
            c[z] = 1
            # 计算加权系数，wi 是权重，c 是非零差值的倒数
            c = self.wi / c
            # 忽略除法可能产生的警告
            with np.errstate(divide='ignore'):
                # 计算加权后的值与相应 yi 的乘积，并按照 c 的加权求和
                p = np.dot(c, self.yi) / np.sum(c, axis=-1)[..., np.newaxis]
            # 纠正 x 与某些 xi 相等的情况
            r = np.nonzero(z)
            # 如果 r 只有一个维度，则表示是在标量处进行评估
            if len(r) == 1:
                # 如果 r[0] 的长度大于0，表示 x 等于其中的一个点
                if len(r[0]) > 0:
                    # 直接取对应的 yi 值作为输出
                    p = self.yi[r[0][0]]
            else:
                # 否则将对应的 yi 值填入 p 的相应位置
                p[r[:-1]] = self.yi[r[-1]]
        return p

    def derivative(self, x, der=1):
        """
        Evaluate a single derivative of the polynomial at the point x.

        Parameters
        ----------
        x : array_like
            Point or points at which to evaluate the derivatives
        der : integer, optional
            Which derivative to evaluate (default: first derivative).
            This number includes the function value as 0th derivative.

        Returns
        -------
        d : ndarray
            Derivative interpolated at the x-points. Shape of `d` is
            determined by replacing the interpolation axis in the
            original array with the shape of `x`.
        """
        # 准备 x 并获取其形状
        x, x_shape = self._prepare_x(x)
        # 评估在点 x 处的多项式的一个导数
        y = self._evaluate_derivatives(x, der+1, all_lower=False)
        # 完成评估并返回结果
        return self._finish_y(y, x_shape)
    def _evaluate_derivatives(self, x, der=None, all_lower=True):
        # NB: der here is not the order of the highest derivative;
        # instead, it is the size of the derivatives matrix that
        # would be returned with all_lower=True, including the
        # '0th' derivative (the undifferentiated function).
        # E.g. to evaluate the 5th derivative alone, call
        # _evaluate_derivatives(x, der=6, all_lower=False).

        # 如果 all_lower=False 且 x 的大小为 0 或者 self.r 为 0，则返回一个空的数组
        if (not all_lower) and (x.size == 0 or self.r == 0):
            return np.zeros((0, self.r), dtype=self.dtype)

        # 如果 all_lower=False 且 der == 1，则调用 self._evaluate(x) 并返回结果
        if (not all_lower) and der == 1:
            return self._evaluate(x)

        # 如果 all_lower=False 且 der 大于 self.n，则返回一个空的数组
        if (not all_lower) and (der > self.n):
            return np.zeros((len(x), self.r), dtype=self.dtype)

        # 如果 der 为 None，则将其设为 self.n
        if der is None:
            der = self.n

        # 如果 all_lower=True 且 x 的大小为 0 或者 self.r 为 0，则返回一个特定大小的零数组
        if all_lower and (x.size == 0 or self.r == 0):
            return np.zeros((der, len(x), self.r), dtype=self.dtype)

        # 如果 self._diff_cij 为 None，则进行以下计算
        if self._diff_cij is None:
            # c[i,j] = xi[i] - xi[j]
            c = self.xi[:, np.newaxis] - self.xi

            # 避免由于除以 0 导致的错误（通过构造，对角线上的条目目前都是零）
            np.fill_diagonal(c, 1)

            # c[i,j] = (w[j] / w[i]) / (xi[i] - xi[j]) (equation 9.4)
            c = self.wi / (c * self.wi[..., np.newaxis])

            # 填充正确的对角线条目：每列的和为 0
            np.fill_diagonal(c, 0)

            # 计算对角线
            # c[j,j] = -sum_{i != j} c[i,j] (equation 9.5)
            d = -c.sum(axis=1)
            # c[i,j] = l_j(x_i)
            np.fill_diagonal(c, d)

            # 将计算结果存储在 self._diff_cij 中
            self._diff_cij = c

        # 如果 self._diff_baryint 为 None，则进行以下计算
        if self._diff_baryint is None:
            # 初始化并缓存导数插值器和 cijs；
            # 重复使用权重 wi（仅依赖于插值点 xi），以避免不必要的重新计算
            self._diff_baryint = BarycentricInterpolator(xi=self.xi,
                                                         yi=self._diff_cij @ self.yi,
                                                         wi=self.wi)
            self._diff_baryint._diff_cij = self._diff_cij

        # 如果 all_lower=True，则从 0 到 der-1 组装导数矩阵，格式符合 _Interpolator1DWithDerivatives 的要求
        if all_lower:
            cn = np.zeros((der, len(x), self.r), dtype=self.dtype)
            for d in range(der):
                cn[d, :, :] = self._evaluate_derivatives(x, d+1, all_lower=False)
            return cn

        # 递归地仅计算所请求的导数
        return self._diff_baryint._evaluate_derivatives(x, der-1, all_lower=False)
# 便利函数用于多项式插值

# 使用巴里心插值方法构造通过给定点集的多项式，然后评估多项式
# 由于数值稳定性的原因，此函数不计算多项式的系数

# 参数：
# xi : array_like
#     应通过的多项式点的 x 坐标的 1-D 数组
# yi : array_like
#     应通过的多项式点的 y 坐标
# x : scalar or array_like
#     要评估插值的点或点集
# der : int or list or None, optional
#     要评估的导数数量，或 None 表示所有可能非零导数
# axis : int, optional
#     对应于 x 坐标值的 yi 数组中的轴

# 返回：
# y : scalar or array_like
#     插值得到的值。形状由将插值轴替换为 x 的形状决定

# 注意：
# 插值权重的构建过程相对较慢。如果您要多次使用相同的 xi 调用此函数
# （但可能是不同的 yi 或 x），应该使用类 `BarycentricInterpolator`。
# 这是此函数在内部使用的类。

# 示例：
# 我们可以使用巴里心插值来插值二维观察数据：

# >>> import numpy as np
# >>> import matplotlib.pyplot as plt
# >>> from scipy.interpolate import barycentric_interpolate
# >>> x_observed = np.linspace(0.0, 10.0, 11)
# >>> y_observed = np.sin(x_observed)
# >>> x = np.linspace(min(x_observed), max(x_observed), num=100)
# >>> y = barycentric_interpolate(x_observed, y_observed, x)
# >>> plt.plot(x_observed, y_observed, "o", label="observation")
# >>> plt.plot(x, y, label="barycentric interpolation")
# >>> plt.legend()
# >>> plt.show()
```