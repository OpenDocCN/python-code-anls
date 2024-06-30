# `D:\src\scipysrc\scipy\scipy\interpolate\_interpolate.py`

```
__all__ = ['interp1d', 'interp2d', 'lagrange', 'PPoly', 'BPoly', 'NdPPoly']

# 导入 math 模块中的 prod 函数
from math import prod

# 导入 numpy 库，并将 array, asarray, intp, poly1d, searchsorted 导入当前命名空间
import numpy as np
from numpy import array, asarray, intp, poly1d, searchsorted

# 导入 scipy.special 模块中的 spec 对象，以及从 scipy._lib._util 中导入 copy_if_needed 函数
import scipy.special as spec
from scipy._lib._util import copy_if_needed

# 从 scipy.special 中导入 comb 函数
from scipy.special import comb

# 导入当前包中的 _fitpack_py 模块
from . import _fitpack_py

# 从当前包中导入 _Interpolator1D 类
from ._polyint import _Interpolator1D

# 从当前包中导入 _ppoly 模块
from . import _ppoly

# 从当前包中导入 _ndim_coords_from_arrays 函数
from .interpnd import _ndim_coords_from_arrays

# 从当前包中导入 _bsplines 模块，并导入 make_interp_spline, BSpline 对象
from ._bsplines import make_interp_spline, BSpline


def lagrange(x, w):
    r"""
    Return a Lagrange interpolating polynomial.

    Given two 1-D arrays `x` and `w,` returns the Lagrange interpolating
    polynomial through the points ``(x, w)``.

    Warning: This implementation is numerically unstable. Do not expect to
    be able to use more than about 20 points even if they are chosen optimally.

    Parameters
    ----------
    x : array_like
        `x` represents the x-coordinates of a set of datapoints.
    w : array_like
        `w` represents the y-coordinates of a set of datapoints, i.e., f(`x`).

    Returns
    -------
    lagrange : `numpy.poly1d` instance
        The Lagrange interpolating polynomial.

    Examples
    --------
    Interpolate :math:`f(x) = x^3` by 3 points.

    >>> import numpy as np
    >>> from scipy.interpolate import lagrange
    >>> x = np.array([0, 1, 2])
    >>> y = x**3
    >>> poly = lagrange(x, y)

    Since there are only 3 points, Lagrange polynomial has degree 2. Explicitly,
    it is given by

    .. math::

        \begin{aligned}
            L(x) &= 1\times \frac{x (x - 2)}{-1} + 8\times \frac{x (x-1)}{2} \\
                 &= x (-2 + 3x)
        \end{aligned}

    >>> from numpy.polynomial.polynomial import Polynomial
    >>> Polynomial(poly.coef[::-1]).coef
    array([ 0., -2.,  3.])

    >>> import matplotlib.pyplot as plt
    >>> x_new = np.arange(0, 2.1, 0.1)
    >>> plt.scatter(x, y, label='data')
    >>> plt.plot(x_new, Polynomial(poly.coef[::-1])(x_new), label='Polynomial')
    >>> plt.plot(x_new, 3*x_new**2 - 2*x_new + 0*x_new,
    ...          label=r"$3 x^2 - 2 x$", linestyle='-.')
    >>> plt.legend()
    >>> plt.show()

    """

    M = len(x)
    p = poly1d(0.0)
    for j in range(M):
        pt = poly1d(w[j])
        for k in range(M):
            if k == j:
                continue
            fac = x[j]-x[k]
            pt *= poly1d([1.0, -x[k]])/fac
        p += pt
    return p


# 错误消息字符串，提醒用户 `interp2d` 已在 SciPy 1.14.0 版本中删除
err_mesg = """\
`interp2d` has been removed in SciPy 1.14.0.

For legacy code, nearly bug-for-bug compatible replacements are
`RectBivariateSpline` on regular grids, and `bisplrep`/`bisplev` for
scattered 2D data.

In new code, for regular grids use `RegularGridInterpolator` instead.
For scattered data, prefer `LinearNDInterpolator` or
`CloughTocher2DInterpolator`.

For more details see
https://scipy.github.io/devdocs/tutorial/interpolate/interp_transition_guide.html
"""

class interp2d:
    """
    二维插值类，已在 SciPy 1.14.0 中移除。
    """
    interp2d(x, y, z, kind='linear', copy=True, bounds_error=False,
             fill_value=None)

    .. versionremoved:: 1.14.0
    表明此函数在 SciPy 1.14.0 版本中已移除。

        `interp2d` has been removed in SciPy 1.14.0.
        `interp2d` 函数在 SciPy 1.14.0 版本中已移除。

        For legacy code, nearly bug-for-bug compatible replacements are
        `RectBivariateSpline` on regular grids, and `bisplrep`/`bisplev` for
        scattered 2D data.
        对于旧代码，几乎可以完全兼容的替代方案是在常规网格上使用 `RectBivariateSpline`，
        对于散点的二维数据则使用 `bisplrep`/`bisplev`。

        In new code, for regular grids use `RegularGridInterpolator` instead.
        对于新代码，在常规网格上使用 `RegularGridInterpolator` 替代。
        For scattered data, prefer `LinearNDInterpolator` or
        `CloughTocher2DInterpolator`.
        对于散点数据，推荐使用 `LinearNDInterpolator` 或 `CloughTocher2DInterpolator`。

        For more details see :ref:`interp-transition-guide`.
        更多详细信息请参阅 :ref:`interp-transition-guide`.
    """
    def __init__(self, x, y, z, kind='linear', copy=True, bounds_error=False,
                 fill_value=None):
        raise NotImplementedError(err_mesg)
    定义了一个未实现的初始化函数，抛出一个未实现错误消息。
def _check_broadcast_up_to(arr_from, shape_to, name):
    """Helper to check that arr_from broadcasts up to shape_to"""
    # 获取 arr_from 的形状
    shape_from = arr_from.shape
    # 检查 shape_to 的维度是否大于等于 shape_from 的维度
    if len(shape_to) >= len(shape_from):
        # 逆序遍历 shape_to 和 shape_from 的维度
        for t, f in zip(shape_to[::-1], shape_from[::-1]):
            # 如果 arr_from 的维度不是 1 且不等于 t，则跳出循环
            if f != 1 and f != t:
                break
        else:  # 如果所有检查通过，则执行后续的类型提升操作
            # 如果 arr_from 的元素个数不为 1 并且形状不等于 shape_to，则进行类型提升操作
            if arr_from.size != 1 and arr_from.shape != shape_to:
                arr_from = np.ones(shape_to, arr_from.dtype) * arr_from
            # 将 arr_from 展平并返回
            return arr_from.ravel()
    # 如果至少有一个检查未通过，则引发 ValueError 异常
    raise ValueError(f'{name} argument must be able to broadcast up '
                     f'to shape {shape_to} but had shape {shape_from}')


def _do_extrapolate(fill_value):
    """Helper to check if fill_value == "extrapolate" without warnings"""
    # 检查 fill_value 是否为字符串且等于 "extrapolate"
    return (isinstance(fill_value, str) and
            fill_value == 'extrapolate')


class interp1d(_Interpolator1D):
    """
    Interpolate a 1-D function.

    .. legacy:: class

        For a guide to the intended replacements for `interp1d` see
        :ref:`tutorial-interpolate_1Dsection`.

    `x` and `y` are arrays of values used to approximate some function f:
    ``y = f(x)``. This class returns a function whose call method uses
    interpolation to find the value of new points.

    Parameters
    ----------
    x : (npoints, ) array_like
        A 1-D array of real values.
    y : (..., npoints, ...) array_like
        A N-D array of real values. The length of `y` along the interpolation
        axis must be equal to the length of `x`. Use the ``axis`` parameter
        to select correct axis. Unlike other interpolators, the default
        interpolation axis is the last axis of `y`.
    kind : str or int, optional
        Specifies the kind of interpolation as a string or as an integer
        specifying the order of the spline interpolator to use.
        The string has to be one of 'linear', 'nearest', 'nearest-up', 'zero',
        'slinear', 'quadratic', 'cubic', 'previous', or 'next'. 'zero',
        'slinear', 'quadratic' and 'cubic' refer to a spline interpolation of
        zeroth, first, second or third order; 'previous' and 'next' simply
        return the previous or next value of the point; 'nearest-up' and
        'nearest' differ when interpolating half-integers (e.g. 0.5, 1.5)
        in that 'nearest-up' rounds up and 'nearest' rounds down. Default
        is 'linear'.
    axis : int, optional
        Axis in the ``y`` array corresponding to the x-coordinate values. Unlike
        other interpolators, defaults to ``axis=-1``.
    copy : bool, optional
        If ``True``, the class makes internal copies of x and y. If ``False``,
        references to ``x`` and ``y`` are used if possible. The default is to copy.
    """
    pass
    # bounds_error 参数，指定是否在插值超出 x 范围时引发 ValueError
    bounds_error : bool, optional
        If True, a ValueError is raised any time interpolation is attempted on
        a value outside of the range of x (where extrapolation is
        necessary). If False, out of bounds values are assigned `fill_value`.
        By default, an error is raised unless ``fill_value="extrapolate"``.
    
    # fill_value 参数，用于指定在数据范围外请求点时的填充值策略
    fill_value : array-like or (array-like, array_like) or "extrapolate", optional
        - if a ndarray (or float), this value will be used to fill in for
          requested points outside of the data range. If not provided, then
          the default is NaN. The array-like must broadcast properly to the
          dimensions of the non-interpolation axes.
        - If a two-element tuple, then the first element is used as a
          fill value for ``x_new < x[0]`` and the second element is used for
          ``x_new > x[-1]``. Anything that is not a 2-element tuple (e.g.,
          list or ndarray, regardless of shape) is taken to be a single
          array-like argument meant to be used for both bounds as
          ``below, above = fill_value, fill_value``. Using a two-element tuple
          or ndarray requires ``bounds_error=False``.
          .. versionadded:: 0.17.0
        - If "extrapolate", then points outside the data range will be
          extrapolated.
          .. versionadded:: 0.17.0
    
    # assume_sorted 参数，指定是否假定输入的 x 值是已排序的
    assume_sorted : bool, optional
        If False, values of `x` can be in any order and they are sorted first.
        If True, `x` has to be an array of monotonically increasing values.
    
    # 属性
    Attributes
    ----------
    fill_value
    
    # 方法
    Methods
    -------
    __call__
    
    # 参考
    See Also
    --------
    splrep, splev
        Spline interpolation/smoothing based on FITPACK.
    UnivariateSpline : An object-oriented wrapper of the FITPACK routines.
    interp2d : 2-D interpolation
    
    # 注意事项
    Notes
    -----
    Calling `interp1d` with NaNs present in input values results in
    undefined behaviour.
    
    Input values `x` and `y` must be convertible to `float` values like
    `int` or `float`.
    
    If the values in `x` are not unique, the resulting behavior is
    undefined and specific to the choice of `kind`, i.e., changing
    `kind` will change the behavior for duplicates.
    
    # 示例
    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy import interpolate
    >>> x = np.arange(0, 10)
    >>> y = np.exp(-x/3.0)
    >>> f = interpolate.interp1d(x, y)
    
    >>> xnew = np.arange(0, 9, 0.1)
    >>> ynew = f(xnew)   # use interpolation function returned by `interp1d`
    >>> plt.plot(x, y, 'o', xnew, ynew, '-')
    >>> plt.show()
    """

    @property
    def fill_value(self):
        """The fill value."""
        # backwards compat: mimic a public attribute
        return self._fill_value_orig

    @fill_value.setter
    # 为填充值设置方法，接受一个填充值作为参数
    def fill_value(self, fill_value):
        # 只有在最近邻和线性方法下才进行外推
        if _do_extrapolate(fill_value):
            # 检查并更新用于外推的边界错误
            self._check_and_update_bounds_error_for_extrapolation()
            # 启用外推标志
            self._extrapolate = True
        else:
            # 计算广播后的形状，剔除当前轴
            broadcast_shape = (self.y.shape[:self.axis] +
                               self.y.shape[self.axis + 1:])
            if len(broadcast_shape) == 0:
                broadcast_shape = (1,)
            # 填充值可以是一个元组 (_below_range, _above_range) 或者单个值
            if isinstance(fill_value, tuple) and len(fill_value) == 2:
                below_above = [np.asarray(fill_value[0]),
                               np.asarray(fill_value[1])]
                names = ('fill_value (below)', 'fill_value (above)')
                for ii in range(2):
                    below_above[ii] = _check_broadcast_up_to(
                        below_above[ii], broadcast_shape, names[ii])
            else:
                fill_value = np.asarray(fill_value)
                below_above = [_check_broadcast_up_to(
                    fill_value, broadcast_shape, 'fill_value')] * 2
            # 分别设置上下范围的填充值
            self._fill_value_below, self._fill_value_above = below_above
            # 禁用外推标志
            self._extrapolate = False
            # 如果未指定边界错误，设定默认值为 True
            if self.bounds_error is None:
                self.bounds_error = True
        # 为了向后兼容性：填充值曾是公共属性；使其可写入
        self._fill_value_orig = fill_value

    # 检查并更新用于外推的边界错误的方法
    def _check_and_update_bounds_error_for_extrapolation(self):
        if self.bounds_error:
            # 如果边界错误已启用，则抛出值错误异常
            raise ValueError("Cannot extrapolate and raise "
                             "at the same time.")
        # 禁用边界错误标志
        self.bounds_error = False

    # 使用 NumPy 中的线性插值方法进行插值计算
    def _call_linear_np(self, x_new):
        # 注意：超出范围的值由 self._evaluate 处理
        return np.interp(x_new, self.x, self.y)

    # 使用自定义线性插值方法进行插值计算
    def _call_linear(self, x_new):
        # 2. 确定在原始数据中应插值的位置
        #    注意：如果 x_new[n] == x[m]，则 searchsorted 返回 m
        x_new_indices = searchsorted(self.x, x_new)

        # 3. 将 x_new_indices 裁剪到 self.x 索引范围内，并至少为 1，避免 x_new[n] = x[0] 的错误插值
        x_new_indices = x_new_indices.clip(1, len(self.x)-1).astype(int)

        # 4. 计算每个 x_new 值所在区域的斜率
        lo = x_new_indices - 1
        hi = x_new_indices

        x_lo = self.x[lo]
        x_hi = self.x[hi]
        y_lo = self._y[lo]
        y_hi = self._y[hi]

        # 根据广播语义，计算斜率
        slope = (y_hi - y_lo) / (x_hi - x_lo)[:, None]

        # 5. 计算每个 x_new 入口处的实际值
        y_new = slope*(x_new - x_lo)[:, None] + y_lo

        return y_new
    def _call_nearest(self, x_new):
        """ Find nearest neighbor interpolated y_new = f(x_new)."""

        # 2. Find where in the averaged data the values to interpolate
        #    would be inserted.
        #    Note: use side='left' (right) to searchsorted() to define the
        #    halfway point to be nearest to the left (right) neighbor
        # 使用 searchsorted() 函数查找 x_new 在 self.x_bds 中的插入位置索引
        x_new_indices = searchsorted(self.x_bds, x_new, side=self._side)

        # 3. Clip x_new_indices so that they are within the range of x indices.
        # 将 x_new_indices 裁剪到 x 索引的范围内，确保不超出边界
        x_new_indices = x_new_indices.clip(0, len(self.x)-1).astype(intp)

        # 4. Calculate the actual value for each entry in x_new.
        # 计算 x_new 中每个条目的实际值，即插值后的 y 值
        y_new = self._y[x_new_indices]

        return y_new

    def _call_previousnext(self, x_new):
        """Use previous/next neighbor of x_new, y_new = f(x_new)."""

        # 1. Get index of left/right value
        # 获取 x_new 在 self._x_shift 中的左/右值的索引
        x_new_indices = searchsorted(self._x_shift, x_new, side=self._side)

        # 2. Clip x_new_indices so that they are within the range of x indices.
        # 将 x_new_indices 裁剪到 x 索引的范围内，确保不超出边界
        x_new_indices = x_new_indices.clip(1-self._ind,
                                           len(self.x)-self._ind).astype(intp)

        # 3. Calculate the actual value for each entry in x_new.
        # 计算 x_new 中每个条目的实际值，即使用前/后邻居的插值 y 值
        y_new = self._y[x_new_indices+self._ind-1]

        return y_new

    def _call_spline(self, x_new):
        # 使用样条插值函数 _spline 计算 x_new 的插值 y 值
        return self._spline(x_new)

    def _call_nan_spline(self, x_new):
        # 使用样条插值函数 _spline 计算 x_new 的插值 y 值，并将结果设为 NaN
        out = self._spline(x_new)
        out[...] = np.nan
        return out

    def _evaluate(self, x_new):
        # 1. Handle values in x_new that are outside of x. Throw error,
        #    or return a list of mask array indicating the outofbounds values.
        #    The behavior is set by the bounds_error variable.
        # 将 x_new 转换为数组 x_new，处理超出 x 范围的值。根据 bounds_error 变量的设定，
        # 可能会抛出错误或返回一个掩码数组列表，指示超出范围的值。
        x_new = asarray(x_new)
        
        # 调用 _call 方法计算插值 y_new
        y_new = self._call(self, x_new)
        
        if not self._extrapolate:
            # 检查 x_new 中超出上下边界的索引
            below_bounds, above_bounds = self._check_bounds(x_new)
            
            if len(y_new) > 0:
                # 注意 fill_value 必须广播到适当的大小并展平以便工作
                # 如果不进行外推，将边界处的 y_new 值设为 _fill_value_below 和 _fill_value_above
                y_new[below_bounds] = self._fill_value_below
                y_new[above_bounds] = self._fill_value_above
        
        return y_new
    # 定义一个方法，用于检查插值数据的输入是否在预设范围内

    def _check_bounds(self, x_new):
        """Check the inputs for being in the bounds of the interpolated data.

        Parameters
        ----------
        x_new : array
            新的输入数据数组，用于进行插值

        Returns
        -------
        out_of_bounds : bool array
            布尔数组，标记哪些输入数据超出了插值数据的边界范围
        """

        # 如果 self.bounds_error 为 True，当 x_new 中的任何值超出了 x 的范围时，抛出错误
        # 否则，返回一个数组，指示哪些值超出了边界区域
        below_bounds = x_new < self.x[0]
        above_bounds = x_new > self.x[-1]

        if self.bounds_error and below_bounds.any():
            below_bounds_value = x_new[np.argmax(below_bounds)]
            raise ValueError(f"A value ({below_bounds_value}) in x_new is below "
                             f"the interpolation range's minimum value ({self.x[0]}).")
        if self.bounds_error and above_bounds.any():
            above_bounds_value = x_new[np.argmax(above_bounds)]
            raise ValueError(f"A value ({above_bounds_value}) in x_new is above "
                             f"the interpolation range's maximum value ({self.x[-1]}).")

        # 返回布尔数组，表示哪些值超出了边界区域
        return below_bounds, above_bounds
class _PPolyBase:
    """Base class for piecewise polynomials."""
    __slots__ = ('c', 'x', 'extrapolate', 'axis')

    def __init__(self, c, x, extrapolate=None, axis=0):
        # 将 c 转换为 NumPy 数组，并确保其为连续的内存布局
        self.c = np.asarray(c)
        # 将 x 转换为 NumPy 数组，并确保其为 float64 类型的连续数组
        self.x = np.ascontiguousarray(x, dtype=np.float64)

        # 处理 extrapolate 参数，默认为 True
        if extrapolate is None:
            extrapolate = True
        elif extrapolate != 'periodic':
            extrapolate = bool(extrapolate)
        self.extrapolate = extrapolate

        # 检查 c 的维度是否至少为 2
        if self.c.ndim < 2:
            raise ValueError("Coefficients array must be at least "
                             "2-dimensional.")

        # 检查 axis 的取值范围是否合法
        if not (0 <= axis < self.c.ndim - 1):
            raise ValueError(f"axis={axis} must be between 0 and {self.c.ndim-1}")

        # 设置对象的 axis 属性
        self.axis = axis
        # 如果 axis 不为 0，则将 c 数组的轴移动到第一个位置
        if axis != 0:
            self.c = np.moveaxis(self.c, axis+1, 0)
            self.c = np.moveaxis(self.c, axis+1, 0)

        # 检查 x 是否为一维数组
        if self.x.ndim != 1:
            raise ValueError("x must be 1-dimensional")
        # 检查 x 的长度是否至少为 2
        if self.x.size < 2:
            raise ValueError("at least 2 breakpoints are needed")
        # 检查 c 的维度是否至少为 2
        if self.c.ndim < 2:
            raise ValueError("c must have at least 2 dimensions")
        # 检查 c 的第一维长度是否大于 0
        if self.c.shape[0] == 0:
            raise ValueError("polynomial must be at least of order 0")
        # 检查 c 的第二维长度是否等于 x 的长度减一
        if self.c.shape[1] != self.x.size-1:
            raise ValueError("number of coefficients != len(x)-1")
        # 检查 x 是否严格单调递增或递减
        dx = np.diff(self.x)
        if not (np.all(dx >= 0) or np.all(dx <= 0)):
            raise ValueError("`x` must be strictly increasing or decreasing.")

        # 根据 c 的数据类型获取相应的 dtype，并将 c 转换为连续的内存布局
        dtype = self._get_dtype(self.c.dtype)
        self.c = np.ascontiguousarray(self.c, dtype=dtype)

    def _get_dtype(self, dtype):
        # 根据 c 的数据类型决定返回的 dtype 类型
        if np.issubdtype(dtype, np.complexfloating) \
               or np.issubdtype(self.c.dtype, np.complexfloating):
            return np.complex128
        else:
            return np.float64

    @classmethod
    def construct_fast(cls, c, x, extrapolate=None, axis=0):
        """
        Construct the piecewise polynomial without making checks.

        Takes the same parameters as the constructor. Input arguments
        ``c`` and ``x`` must be arrays of the correct shape and type. The
        ``c`` array can only be of dtypes float and complex, and ``x``
        array must have dtype float.
        """
        # 使用类方法构造分段多项式对象，不进行任何检查
        self = object.__new__(cls)
        self.c = c
        self.x = x
        self.axis = axis
        # 处理 extrapolate 参数，默认为 True
        if extrapolate is None:
            extrapolate = True
        self.extrapolate = extrapolate
        return self
    # 确保数组 self.x 是 C 连续存储的
    if not self.x.flags.c_contiguous:
        # 如果 self.x 不是 C 连续存储的，则创建其副本，使其成为 C 连续存储
        self.x = self.x.copy()

    # 确保数组 self.c 是 C 连续存储的
    if not self.c.flags.c_contiguous:
        # 如果 self.c 不是 C 连续存储的，则创建其副本，使其成为 C 连续存储
        self.c = self.c.copy()
    def extend(self, c, x):
        """
        Add additional breakpoints and coefficients to the polynomial.

        Parameters
        ----------
        c : ndarray, size (k, m, ...)
            Additional coefficients for polynomials in intervals. Note that
            the first additional interval will be formed using one of the
            ``self.x`` end points.
        x : ndarray, size (m,)
            Additional breakpoints. Must be sorted in the same order as
            ``self.x`` and either to the right or to the left of the current
            breakpoints.
        """

        c = np.asarray(c)  # 将输入的系数转换为NumPy数组
        x = np.asarray(x)  # 将输入的断点转换为NumPy数组

        if c.ndim < 2:
            raise ValueError("invalid dimensions for c")  # 如果系数数组的维度小于2，抛出异常

        if x.ndim != 1:
            raise ValueError("invalid dimensions for x")  # 如果断点数组的维度不为1，抛出异常

        if x.shape[0] != c.shape[1]:
            raise ValueError(f"Shapes of x {x.shape} and c {c.shape} are incompatible")
            # 如果断点数组和系数数组的形状不兼容，抛出异常

        if c.shape[2:] != self.c.shape[2:] or c.ndim != self.c.ndim:
            raise ValueError(
                f"Shapes of c {c.shape} and self.c {self.c.shape} are incompatible"
            )
            # 如果输入的系数数组形状与当前对象的系数数组形状不兼容，抛出异常

        if c.size == 0:
            return  # 如果系数数组大小为0，直接返回

        dx = np.diff(x)  # 计算断点数组的差分
        if not (np.all(dx >= 0) or np.all(dx <= 0)):
            raise ValueError("`x` is not sorted.")
            # 如果断点数组不是有序的（既不完全升序也不完全降序），抛出异常

        if self.x[-1] >= self.x[0]:
            if not x[-1] >= x[0]:
                raise ValueError("`x` is in the different order "
                                 "than `self.x`.")
                # 如果断点数组的顺序与当前对象的断点数组顺序不同，抛出异常

            if x[0] >= self.x[-1]:
                action = 'append'  # 如果断点数组在当前对象的断点数组右侧，执行追加操作
            elif x[-1] <= self.x[0]:
                action = 'prepend'  # 如果断点数组在当前对象的断点数组左侧，执行前置操作
            else:
                raise ValueError("`x` is neither on the left or on the right "
                                 "from `self.x`.")
                # 如果断点数组既不在当前对象的左侧也不在右侧，抛出异常
        else:
            if not x[-1] <= x[0]:
                raise ValueError("`x` is in the different order "
                                 "than `self.x`.")
                # 如果断点数组的顺序与当前对象的断点数组顺序不同，抛出异常

            if x[0] <= self.x[-1]:
                action = 'append'  # 如果断点数组在当前对象的断点数组右侧，执行追加操作
            elif x[-1] >= self.x[0]:
                action = 'prepend'  # 如果断点数组在当前对象的断点数组左侧，执行前置操作
            else:
                raise ValueError("`x` is neither on the left or on the right "
                                 "from `self.x`.")
                # 如果断点数组既不在当前对象的左侧也不在右侧，抛出异常

        dtype = self._get_dtype(c.dtype)  # 获取用于系数的数据类型

        k2 = max(c.shape[0], self.c.shape[0])  # 计算新系数数组的第一维度大小
        c2 = np.zeros((k2, self.c.shape[1] + c.shape[1]) + self.c.shape[2:],
                      dtype=dtype)  # 创建新的系数数组

        if action == 'append':
            c2[k2-self.c.shape[0]:, :self.c.shape[1]] = self.c
            c2[k2-c.shape[0]:, self.c.shape[1]:] = c
            self.x = np.r_[self.x, x]  # 在当前对象的断点数组末尾追加新的断点数组
        elif action == 'prepend':
            c2[k2-self.c.shape[0]:, :c.shape[1]] = c
            c2[k2-c.shape[0]:, c.shape[1]:] = self.c
            self.x = np.r_[x, self.x]  # 在当前对象的断点数组开头前置新的断点数组

        self.c = c2  # 更新当前对象的系数数组
    def __call__(self, x, nu=0, extrapolate=None):
        """
        Evaluate the piecewise polynomial or its derivative.

        Parameters
        ----------
        x : array_like
            Points to evaluate the interpolant at.
        nu : int, optional
            Order of derivative to evaluate. Must be non-negative.
        extrapolate : {bool, 'periodic', None}, optional
            If bool, determines whether to extrapolate to out-of-bounds points
            based on first and last intervals, or to return NaNs.
            If 'periodic', periodic extrapolation is used.
            If None (default), use `self.extrapolate`.

        Returns
        -------
        y : array_like
            Interpolated values. Shape is determined by replacing
            the interpolation axis in the original array with the shape of x.

        Notes
        -----
        Derivatives are evaluated piecewise for each polynomial
        segment, even if the polynomial is not differentiable at the
        breakpoints. The polynomial intervals are considered half-open,
        ``[a, b)``, except for the last interval which is closed
        ``[a, b]``.
        """
        # 如果 extrapolate 参数为 None，则使用 self.extrapolate 的值
        if extrapolate is None:
            extrapolate = self.extrapolate
        # 将输入 x 转换为 numpy 数组
        x = np.asarray(x)
        # 记录 x 的原始形状和维度
        x_shape, x_ndim = x.shape, x.ndim
        # 将 x 转换为连续的内存布局，并转换为 np.float64 类型
        x = np.ascontiguousarray(x.ravel(), dtype=np.float64)

        # 如果 extrapolate 参数为 'periodic'，则将 x 映射到周期性插值段
        if extrapolate == 'periodic':
            x = self.x[0] + (x - self.x[0]) % (self.x[-1] - self.x[0])
            # 将 extrapolate 参数重置为 False，表示不再使用周期性插值
            extrapolate = False

        # 创建一个空数组 out，用于存储插值结果
        out = np.empty((len(x), prod(self.c.shape[2:])), dtype=self.c.dtype)
        # 确保 self.c 数组具有 C 连续的内存布局
        self._ensure_c_contiguous()
        # 调用 _evaluate 方法进行插值计算
        self._evaluate(x, nu, extrapolate, out)
        # 将 out 重新调整为原始形状，以匹配输入 x 的形状和插值数组的维度
        out = out.reshape(x_shape + self.c.shape[2:])
        
        # 如果插值轴不在第一个位置，则进行轴的转置操作
        if self.axis != 0:
            # 创建一个轴索引列表
            l = list(range(out.ndim))
            # 将插值计算结果移动到正确的插值轴位置
            l = l[x_ndim:x_ndim+self.axis] + l[:x_ndim] + l[x_ndim+self.axis:]
            out = out.transpose(l)
        
        # 返回插值结果
        return out
    """
    Piecewise polynomial in terms of coefficients and breakpoints

    The polynomial between ``x[i]`` and ``x[i + 1]`` is written in the
    local power basis::

        S = sum(c[m, i] * (xp - x[i])**(k-m) for m in range(k+1))

    where ``k`` is the degree of the polynomial.

    Parameters
    ----------
    c : ndarray, shape (k, m, ...)
        Polynomial coefficients, order `k` and `m` intervals.
    x : ndarray, shape (m+1,)
        Polynomial breakpoints. Must be sorted in either increasing or
        decreasing order.
    extrapolate : bool or 'periodic', optional
        If bool, determines whether to extrapolate to out-of-bounds points
        based on first and last intervals, or to return NaNs. If 'periodic',
        periodic extrapolation is used. Default is True.
    axis : int, optional
        Interpolation axis. Default is zero.

    Attributes
    ----------
    x : ndarray
        Breakpoints.
    c : ndarray
        Coefficients of the polynomials. They are reshaped
        to a 3-D array with the last dimension representing
        the trailing dimensions of the original coefficient array.
    axis : int
        Interpolation axis.

    Methods
    -------
    __call__
    derivative
    antiderivative
    integrate
    solve
    roots
    extend
    from_spline
    from_bernstein_basis
    construct_fast

    See also
    --------
    BPoly : piecewise polynomials in the Bernstein basis

    Notes
    -----
    High-order polynomials in the power basis can be numerically
    unstable. Precision problems can start to appear for orders
    larger than 20-30.
    """

    def _evaluate(self, x, nu, extrapolate, out):
        """
        Evaluate the piecewise polynomial at given points.

        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the polynomial.
        nu : int
            Number of derivatives to evaluate (0 <= nu <= k).
        extrapolate : bool
            Whether to extrapolate for out-of-bounds points or return NaNs.
        out : ndarray
            Array to store the evaluation results.

        Notes
        -----
        This method reshapes the coefficients and uses `_ppoly.evaluate`
        to compute the polynomial values at specified points `x`.
        """
        _ppoly.evaluate(self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
                        self.x, x, nu, bool(extrapolate), out)
    def derivative(self, nu=1):
        """
        构造一个新的分段多项式，表示导数。

        Parameters
        ----------
        nu : int, optional
            要计算的导数阶数。默认为1，即计算一阶导数。如果为负数，返回反导数。

        Returns
        -------
        pp : PPoly
            表示该多项式导数的分段多项式，阶数为 k2 = k - n。

        Notes
        -----
        对每个多项式片段进行导数计算，即使在断点处多项式不可导也是如此。
        多项式区间被认为是半开放的，``[a, b)``, 最后一个区间是闭合的 ``[a, b]``。
        """
        if nu < 0:
            # 如果 nu 小于 0，返回反导数
            return self.antiderivative(-nu)

        # 降低阶数
        if nu == 0:
            # 如果 nu 等于 0，直接复制系数 c
            c2 = self.c.copy()
        else:
            # 否则，从前向后切片删除 nu 行，并复制剩余部分
            c2 = self.c[:-nu, :].copy()

        if c2.shape[0] == 0:
            # 如果 c2 的行数为 0，表示导数阶数为 0，结果为全零数组
            c2 = np.zeros((1,) + c2.shape[1:], dtype=c2.dtype)

        # 乘以正确的上升阶乘因子
        factor = spec.poch(np.arange(c2.shape[0], 0, -1), nu)
        c2 *= factor[(slice(None),) + (None,)*(c2.ndim-1)]

        # 构造一个兼容的多项式
        return self.construct_fast(c2, self.x, self.extrapolate, self.axis)
    # 定义一个方法，用于计算多项式的不定积分（反导函数）
    def antiderivative(self, nu=1):
        """
        Construct a new piecewise polynomial representing the antiderivative.

        Antiderivative is also the indefinite integral of the function,
        and derivative is its inverse operation.

        Parameters
        ----------
        nu : int, optional
            Order of antiderivative to evaluate. Default is 1, i.e., compute
            the first integral. If negative, the derivative is returned.

        Returns
        -------
        pp : PPoly
            Piecewise polynomial of order k2 = k + n representing
            the antiderivative of this polynomial.

        Notes
        -----
        The antiderivative returned by this function is continuous and
        continuously differentiable to order n-1, up to floating point
        rounding error.

        If antiderivative is computed and ``self.extrapolate='periodic'``,
        it will be set to False for the returned instance. This is done because
        the antiderivative is no longer periodic and its correct evaluation
        outside of the initially given x interval is difficult.
        """
        # 如果 nu 小于等于 0，则返回对应的导数
        if nu <= 0:
            return self.derivative(-nu)

        # 创建一个新的系数数组 c，用于存储反导函数的多项式表示
        c = np.zeros((self.c.shape[0] + nu, self.c.shape[1]) + self.c.shape[2:],
                     dtype=self.c.dtype)
        # 将原始多项式的系数复制到 c 中
        c[:-nu] = self.c

        # 计算正确的升幂因子并应用于 c 中的每个分段
        factor = spec.poch(np.arange(self.c.shape[0], 0, -1), nu)
        c[:-nu] /= factor[(slice(None),) + (None,)*(c.ndim-1)]

        # 确保添加的自由度的连续性
        self._ensure_c_contiguous()
        # 调用特定函数修复添加自由度的连续性
        _ppoly.fix_continuity(c.reshape(c.shape[0], c.shape[1], -1),
                              self.x, nu - 1)

        # 根据 self.extrapolate 的设置确定是否设置 extrapolate 为 False
        if self.extrapolate == 'periodic':
            extrapolate = False
        else:
            extrapolate = self.extrapolate

        # 构造一个兼容的多项式并返回
        return self.construct_fast(c, self.x, extrapolate, self.axis)
    # 定义一个方法用于在分段多项式上计算定积分

    # a 是积分下界，b 是积分上界
    # extrapolate 可选参数，控制是否对超出边界的点进行外推
    # 如果为 bool 类型，根据首尾区间进行外推，或者返回 NaN
    # 如果为 'periodic'，使用周期性外推
    # 如果为 None（默认），使用 self.extrapolate 的设置
    def integrate(self, a, b, extrapolate=None):
        # 如果 extrapolate 为 None，则使用 self.extrapolate 的值
        if extrapolate is None:
            extrapolate = self.extrapolate

        # 如果 b < a，则交换积分上下界，并记录符号
        sign = 1
        if b < a:
            a, b = b, a
            sign = -1

        # 创建一个空数组，用于存储积分结果
        range_int = np.empty((np.prod(self.c.shape[2:]),), dtype=self.c.dtype)

        # 确保 self.c 是 C 连续的
        self._ensure_c_contiguous()

        # 计算积分
        if extrapolate == 'periodic':
            # 将积分分为周期内的部分和剩余部分

            # 获取首尾点和周期长度
            xs, xe = self.x[0], self.x[-1]
            period = xe - xs
            interval = b - a
            n_periods, left = divmod(interval, period)

            if n_periods > 0:
                # 对周期内部分进行积分
                _ppoly.integrate(
                    self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
                    self.x, xs, xe, False, out=range_int)
                range_int *= n_periods
            else:
                range_int.fill(0)

            # 将 a 映射到 [xs, xe] 区间，b 总是 a + left
            a = xs + (a - xs) % period
            b = a + left

            # 如果 b <= xe，则积分区间为 [a, b]，否则为 [a, xe] 和 xs 到剩余部分
            remainder_int = np.empty_like(range_int)
            if b <= xe:
                _ppoly.integrate(
                    self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
                    self.x, a, b, False, out=remainder_int)
                range_int += remainder_int
            else:
                _ppoly.integrate(
                    self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
                    self.x, a, xe, False, out=remainder_int)
                range_int += remainder_int

                _ppoly.integrate(
                    self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
                    self.x, xs, xs + left + a - xe, False, out=remainder_int)
                range_int += remainder_int
        else:
            # 普通情况下直接计算积分
            _ppoly.integrate(
                self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
                self.x, a, b, bool(extrapolate), out=range_int)

        # 返回积分结果，并乘以之前记录的符号
        range_int *= sign
        return range_int.reshape(self.c.shape[2:])
    def solve(self, y=0., discontinuity=True, extrapolate=None):
        """
        Find real solutions of the equation ``pp(x) == y``.

        Parameters
        ----------
        y : float, optional
            Right-hand side. Default is zero.
        discontinuity : bool, optional
            Whether to report sign changes across discontinuities at
            breakpoints as roots.
        extrapolate : {bool, 'periodic', None}, optional
            If bool, determines whether to return roots from the polynomial
            extrapolated based on first and last intervals, 'periodic' works
            the same as False. If None (default), use `self.extrapolate`.

        Returns
        -------
        roots : ndarray
            Roots of the polynomial(s).

            If the PPoly object describes multiple polynomials, the
            return value is an object array whose each element is an
            ndarray containing the roots.

        Notes
        -----
        This routine works only on real-valued polynomials.

        If the piecewise polynomial contains sections that are
        identically zero, the root list will contain the start point
        of the corresponding interval, followed by a ``nan`` value.

        If the polynomial is discontinuous across a breakpoint, and
        there is a sign change across the breakpoint, this is reported
        if the `discont` parameter is True.

        Examples
        --------

        Finding roots of ``[x**2 - 1, (x - 1)**2]`` defined on intervals
        ``[-2, 1], [1, 2]``:

        >>> import numpy as np
        >>> from scipy.interpolate import PPoly
        >>> pp = PPoly(np.array([[1, -4, 3], [1, 0, 0]]).T, [-2, 1, 2])
        >>> pp.solve()
        array([-1.,  1.])
        """
        # 如果 extrapolate 参数为 None，则使用对象本身的 extrapolate 属性
        if extrapolate is None:
            extrapolate = self.extrapolate

        # 确保系数数组是 C 连续的
        self._ensure_c_contiguous()

        # 如果系数数组是复数类型，抛出 ValueError 异常
        if np.issubdtype(self.c.dtype, np.complexfloating):
            raise ValueError("Root finding is only for "
                             "real-valued polynomials")

        # 将 y 转换为 float 类型
        y = float(y)
        # 调用 C 扩展的函数 _ppoly.real_roots 进行实根计算
        r = _ppoly.real_roots(self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
                              self.x, y, bool(discontinuity),
                              bool(extrapolate))
        # 如果系数数组的维度为 2，则返回结果数组的第一个元素
        if self.c.ndim == 2:
            return r[0]
        else:
            # 否则，将结果数组 r 转换为对象数组，以处理多多项式的情况
            r2 = np.empty(np.prod(self.c.shape[2:]), dtype=object)
            # 由于 NumPy 1.6.0 中的切片赋值有问题，使用循环来逐个赋值
            for ii, root in enumerate(r):
                r2[ii] = root

            # 返回重新整形后的结果数组
            return r2.reshape(self.c.shape[2:])
    # 定义一个方法用于计算分段多项式的实根
    def roots(self, discontinuity=True, extrapolate=None):
        """
        Find real roots of the piecewise polynomial.

        Parameters
        ----------
        discontinuity : bool, optional
            是否报告在断点处的间断处的符号变化作为根。
        extrapolate : {bool, 'periodic', None}, optional
            如果是布尔值，则确定是否基于第一个和最后一个区间来返回多项式外推的根，
            'periodic' 与 False 的行为相同。如果为 None（默认），则使用 `self.extrapolate`。

        Returns
        -------
        roots : ndarray
            多项式的根。

            如果 PPoly 对象描述多个多项式，则返回值是一个对象数组，
            其中每个元素是一个包含根的 ndarray。

        See Also
        --------
        PPoly.solve
        """
        # 调用 solve 方法，求解多项式的根，起始点为 0
        return self.solve(0, discontinuity, extrapolate)

    @classmethod
    def from_spline(cls, tck, extrapolate=None):
        """
        Construct a piecewise polynomial from a spline

        Parameters
        ----------
        tck
            A spline, as returned by `splrep` or a BSpline object.
        extrapolate : bool or 'periodic', optional
            If bool, determines whether to extrapolate to out-of-bounds points
            based on first and last intervals, or to return NaNs.
            If 'periodic', periodic extrapolation is used. Default is True.

        Examples
        --------
        Construct an interpolating spline and convert it to a `PPoly` instance 

        >>> import numpy as np
        >>> from scipy.interpolate import splrep, PPoly
        >>> x = np.linspace(0, 1, 11)
        >>> y = np.sin(2*np.pi*x)
        >>> tck = splrep(x, y, s=0)
        >>> p = PPoly.from_spline(tck)
        >>> isinstance(p, PPoly)
        True

        Note that this function only supports 1D splines out of the box.

        If the ``tck`` object represents a parametric spline (e.g. constructed
        by `splprep` or a `BSpline` with ``c.ndim > 1``), you will need to loop
        over the dimensions manually.

        >>> from scipy.interpolate import splprep, splev
        >>> t = np.linspace(0, 1, 11)
        >>> x = np.sin(2*np.pi*t)
        >>> y = np.cos(2*np.pi*t)
        >>> (t, c, k), u = splprep([x, y], s=0)

        Note that ``c`` is a list of two arrays of length 11.

        >>> unew = np.arange(0, 1.01, 0.01)
        >>> out = splev(unew, (t, c, k))

        To convert this spline to the power basis, we convert each
        component of the list of b-spline coefficients, ``c``, into the
        corresponding cubic polynomial.

        >>> polys = [PPoly.from_spline((t, cj, k)) for cj in c]
        >>> polys[0].c.shape
        (4, 14)

        Note that the coefficients of the polynomials `polys` are in the
        power basis and their dimensions reflect just that: here 4 is the order
        (degree+1), and 14 is the number of intervals---which is nothing but
        the length of the knot array of the original `tck` minus one.

        Optionally, we can stack the components into a single `PPoly` along
        the third dimension:

        >>> cc = np.dstack([p.c for p in polys])    # has shape = (4, 14, 2)
        >>> poly = PPoly(cc, polys[0].x)
        >>> np.allclose(poly(unew).T,     # note the transpose to match `splev`
        ...             out, atol=1e-15)
        True

        """
        # Check if the input spline `tck` is an instance of `BSpline`
        if isinstance(tck, BSpline):
            # Unpack the `tck` object into `t`, `c`, and `k` attributes of `BSpline`
            t, c, k = tck.tck
            # If `extrapolate` is not provided, use the `extrapolate` attribute of `BSpline`
            if extrapolate is None:
                extrapolate = tck.extrapolate
        else:
            # Unpack `tck` into `t`, `c`, and `k` for non-`BSpline` cases
            t, c, k = tck

        # Initialize an array `cvals` to store computed spline coefficients
        cvals = np.empty((k + 1, len(t)-1), dtype=c.dtype)
        # Loop over polynomial degrees `m` from `k` down to `0`
        for m in range(k, -1, -1):
            # Evaluate the spline at points `t[:-1]` and normalize by factorial of `m+1`
            y = _fitpack_py.splev(t[:-1], tck, der=m)
            cvals[k - m, :] = y / spec.gamma(m+1)

        # Construct a `PPoly` object from computed coefficients `cvals`, knots `t`, and `extrapolate` setting
        return cls.construct_fast(cvals, t, extrapolate)

    @classmethod
    def from_bernstein_basis(cls, bp, extrapolate=None):
        """
        Construct a piecewise polynomial in the power basis
        from a polynomial in Bernstein basis.

        Parameters
        ----------
        bp : BPoly
            A Bernstein basis polynomial, as created by BPoly
        extrapolate : bool or 'periodic', optional
            If bool, determines whether to extrapolate to out-of-bounds points
            based on first and last intervals, or to return NaNs.
            If 'periodic', periodic extrapolation is used. Default is True.
        """
        # 检查输入的 bp 是否为 BPoly 类型，否则抛出 TypeError 异常
        if not isinstance(bp, BPoly):
            raise TypeError(".from_bernstein_basis only accepts BPoly instances. "
                            "Got %s instead." % type(bp))

        # 计算输入 bp 的节点间距
        dx = np.diff(bp.x)
        # 确定多项式的阶数
        k = bp.c.shape[0] - 1  # polynomial order

        # 对于多维情况，创建一个与 bp.c 维度相同的元组 rest，其值为 None
        rest = (None,)*(bp.c.ndim-2)

        # 初始化一个与 bp.c 结构相同的全零数组 c
        c = np.zeros_like(bp.c)
        
        # 对每个 a，计算对应的多项式系数
        for a in range(k+1):
            # 计算当前项的系数因子
            factor = (-1)**a * comb(k, a) * bp.c[a]
            # 对每个 s，计算多项式的值
            for s in range(a, k+1):
                # 计算组合数乘以 (-1)^s
                val = comb(k-a, s-a) * (-1)**s
                # 更新 c[k-s] 的值，注意 dx[(slice(None),)+rest]**s 是节点间距的 s 次幂
                c[k-s] += factor * val / dx[(slice(None),)+rest]**s

        # 如果没有指定 extrapolate 参数，则使用 bp 对象的 extrapolate 属性
        if extrapolate is None:
            extrapolate = bp.extrapolate

        # 调用类方法 cls.construct_fast 构建基于系数 c 和节点 bp.x 的快速多项式对象
        return cls.construct_fast(c, bp.x, extrapolate, bp.axis)
class BPoly(_PPolyBase):
    """Piecewise polynomial in terms of coefficients and breakpoints.

    The polynomial between ``x[i]`` and ``x[i + 1]`` is written in the
    Bernstein polynomial basis::

        S = sum(c[a, i] * b(a, k; x) for a in range(k+1)),

    where ``k`` is the degree of the polynomial, and::

        b(a, k; x) = binom(k, a) * t**a * (1 - t)**(k - a),

    with ``t = (x - x[i]) / (x[i+1] - x[i])`` and ``binom`` is the binomial
    coefficient.

    Parameters
    ----------
    c : ndarray, shape (k, m, ...)
        Polynomial coefficients, order `k` and `m` intervals
    x : ndarray, shape (m+1,)
        Polynomial breakpoints. Must be sorted in either increasing or
        decreasing order.
    extrapolate : bool, optional
        If bool, determines whether to extrapolate to out-of-bounds points
        based on first and last intervals, or to return NaNs. If 'periodic',
        periodic extrapolation is used. Default is True.
    axis : int, optional
        Interpolation axis. Default is zero.

    Attributes
    ----------
    x : ndarray
        Breakpoints.
    c : ndarray
        Coefficients of the polynomials. They are reshaped
        to a 3-D array with the last dimension representing
        the trailing dimensions of the original coefficient array.
    axis : int
        Interpolation axis.

    Methods
    -------
    __call__
    extend
    derivative
    antiderivative
    integrate
    construct_fast
    from_power_basis
    from_derivatives

    See also
    --------
    PPoly : piecewise polynomials in the power basis

    Notes
    -----
    Properties of Bernstein polynomials are well documented in the literature,
    see for example [1]_ [2]_ [3]_.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Bernstein_polynomial

    .. [2] Kenneth I. Joy, Bernstein polynomials,
       http://www.idav.ucdavis.edu/education/CAGDNotes/Bernstein-Polynomials.pdf

    .. [3] E. H. Doha, A. H. Bhrawy, and M. A. Saker, Boundary Value Problems,
           vol 2011, article ID 829546, :doi:`10.1155/2011/829543`.

    Examples
    --------
    >>> from scipy.interpolate import BPoly
    >>> x = [0, 1]
    >>> c = [[1], [2], [3]]
    >>> bp = BPoly(c, x)

    This creates a 2nd order polynomial

    .. math::

        B(x) = 1 \\times b_{0, 2}(x) + 2 \\times b_{1, 2}(x) + 3
               \\times b_{2, 2}(x) \\\\
             = 1 \\times (1-x)^2 + 2 \\times 2 x (1 - x) + 3 \\times x^2

    """  # noqa: E501

    def _evaluate(self, x, nu, extrapolate, out):
        """
        Evaluate the piecewise polynomial using the Bernstein basis.

        Parameters
        ----------
        x : ndarray
            Points to evaluate the polynomial at.
        nu : int
            Derivative order to evaluate.
        extrapolate : bool
            If True, extrapolate to out-of-bounds points; if False, return NaNs.
        out : ndarray
            Array to store the evaluation results.

        """
        _ppoly.evaluate_bernstein(
            # Reshape coefficients to match the expected format
            self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
            # Breakpoints of the polynomial
            self.x,
            # Points to evaluate
            x,
            # Derivative order
            nu,
            # Boolean for extrapolation
            bool(extrapolate),
            # Output array for evaluation results
            out
        )
    # 定义一个方法，用于计算多项式的导数
    def derivative(self, nu=1):
        """
        Construct a new piecewise polynomial representing the derivative.

        Parameters
        ----------
        nu : int, optional
            Order of derivative to evaluate. Default is 1, i.e., compute the
            first derivative. If negative, the antiderivative is returned.

        Returns
        -------
        bp : BPoly
            Piecewise polynomial of order k - nu representing the derivative of
            this polynomial.

        """
        # 如果 nu 小于 0，则返回该多项式的反导数
        if nu < 0:
            return self.antiderivative(-nu)

        # 如果 nu 大于 1，则递归计算多次导数
        if nu > 1:
            bp = self
            for k in range(nu):
                bp = bp.derivative()
            return bp

        # 对于 nu == 0，计算导数的低阶情况
        # 多项式的导数公式：
        #   B'(x) = \sum_{a=0}^{k-1} (c_{a+1} - c_a) b_{a, k-1}
        # 其中 c 是多项式的系数，b 是基函数
        # dx 是自变量的差值
        # rest 用于处理多维情况下的额外维度

        # 计算导数系数 c2
        k = self.c.shape[0] - 1
        dx = np.diff(self.x)[(None, slice(None)) + (None,) * (self.c.ndim - 2)]
        c2 = k * np.diff(self.c, axis=0) / dx

        # 如果 c2 的行数为 0，表示导数为 0
        if c2.shape[0] == 0:
            c2 = np.zeros((1,) + c2.shape[1:], dtype=c2.dtype)

        # 构造一个兼容的多项式对象并返回
        return self.construct_fast(c2, self.x, self.extrapolate, self.axis)
    def antiderivative(self, nu=1):
        """
        构造一个新的分段多项式，表示原函数的不定积分。

        Parameters
        ----------
        nu : int, optional
            求解的不定积分的阶数。默认为 1，即计算第一次积分。如果为负数，则返回导数。

        Returns
        -------
        bp : BPoly
            阶数为 k + nu 的分段多项式，表示该多项式的不定积分。

        Notes
        -----
        如果计算不定积分且 self.extrapolate='periodic'，
        对返回的实例将设置 extrapolate=False。这是因为不定积分不再是周期性的，
        在给定的 x 区间之外正确计算它变得困难。
        """
        if nu <= 0:
            return self.derivative(-nu)  # 如果 nu 小于等于 0，则返回相应阶数的导数

        if nu > 1:
            bp = self
            for k in range(nu):
                bp = bp.antiderivative()
            return bp  # 对于 nu 大于 1 的情况，递归地计算多次不定积分并返回

        # 在各个区间上构造不定积分
        c, x = self.c, self.x
        k = c.shape[0]
        c2 = np.zeros((k+1,) + c.shape[1:], dtype=c.dtype)

        c2[1:, ...] = np.cumsum(c, axis=0) / k
        delta = x[1:] - x[:-1]
        c2 *= delta[(None, slice(None)) + (None,)*(c.ndim-2)]

        # 现在修正连续性：在第一个区间上，将积分常数设为零；
        # 在区间 [x_j, x_{j+1}) 上，其中 j>0，积分常数等于 `bp` 在 x_j 处的跃变。
        # 后者由前一个区间上的 B_{n+1, n+1} 系数给出（其他 B. 多项式在断点处为零）。
        # 最后，利用分段多项式构成单位分区的性质。
        c2[:,1:] += np.cumsum(c2[k, :], axis=0)[:-1]

        if self.extrapolate == 'periodic':
            extrapolate = False
        else:
            extrapolate = self.extrapolate

        return self.construct_fast(c2, x, extrapolate, axis=self.axis)
    # 定义一个方法用于计算分段多项式的定积分

    def integrate(self, a, b, extrapolate=None):
        """
        Compute a definite integral over a piecewise polynomial.

        Parameters
        ----------
        a : float
            Lower integration bound
        b : float
            Upper integration bound
        extrapolate : {bool, 'periodic', None}, optional
            Whether to extrapolate to out-of-bounds points based on first
            and last intervals, or to return NaNs. If 'periodic', periodic
            extrapolation is used. If None (default), use `self.extrapolate`.

        Returns
        -------
        array_like
            Definite integral of the piecewise polynomial over [a, b]

        """

        # 计算分段多项式的不定积分
        ib = self.antiderivative()

        # 如果未指定 extrapolate 参数，则使用对象自身的 extrapolate 设置
        if extrapolate is None:
            extrapolate = self.extrapolate

        # 如果 extrapolate 不是 'periodic'，则设置 ib.extrapolate 为 extrapolate
        if extrapolate != 'periodic':
            ib.extrapolate = extrapolate

        # 如果 extrapolate 是 'periodic'，则对积分进行周期性处理
        if extrapolate == 'periodic':
            # 将积分区间分解为周期内的部分和剩余部分

            # 为了简单和清晰起见，将 a <= b 的情况转换为 a <= b
            if a <= b:
                sign = 1
            else:
                a, b = b, a
                sign = -1

            xs, xe = self.x[0], self.x[-1]
            period = xe - xs
            interval = b - a
            n_periods, left = divmod(interval, period)
            res = n_periods * (ib(xe) - ib(xs))

            # 将 a 和 b 映射到 [xs, xe] 区间内
            a = xs + (a - xs) % period
            b = a + left

            # 如果 b <= xe，则需要在 [a, b] 区间上积分，否则在 [a, xe] 和 [xs, a + left - xe] 区间上积分
            if b <= xe:
                res += ib(b) - ib(a)
            else:
                res += ib(xe) - ib(a) + ib(xs + left + a - xe) - ib(xs)

            return sign * res
        else:
            # 如果不进行周期性处理，则直接计算积分值
            return ib(b) - ib(a)

    # 定义一个方法用于扩展多项式对象的次数和系数
    def extend(self, c, x):
        k = max(self.c.shape[0], c.shape[0])
        self.c = self._raise_degree(self.c, k - self.c.shape[0])
        c = self._raise_degree(c, k - c.shape[0])
        return _PPolyBase.extend(self, c, x)
    extend.__doc__ = _PPolyBase.extend.__doc__
    @classmethod
    @staticmethod
    @staticmethod
    def _raise_degree(c, d):
        """
        Raise a degree of a polynomial in the Bernstein basis.

        Given the coefficients of a polynomial degree `k`, return (the
        coefficients of) the equivalent polynomial of degree `k+d`.

        Parameters
        ----------
        c : array_like
            coefficient array, 1-D
        d : integer
            degree by which to raise the polynomial

        Returns
        -------
        array
            coefficient array, 1-D array of length `c.shape[0] + d`

        Notes
        -----
        This uses the fact that a Bernstein polynomial `b_{a, k}` can be
        identically represented as a linear combination of polynomials of
        a higher degree `k+d`:

            .. math:: b_{a, k} = comb(k, a) \sum_{j=0}^{d} b_{a+j, k+d} \
                                 comb(d, j) / comb(k+d, a+j)

        """
        # If no degree change is needed, return the coefficients as is
        if d == 0:
            return c

        # Determine the current degree of the polynomial
        k = c.shape[0] - 1
        # Initialize an array to store coefficients of the raised degree polynomial
        out = np.zeros((c.shape[0] + d,) + c.shape[1:], dtype=c.dtype)

        # Iterate over each coefficient of the original polynomial
        for a in range(c.shape[0]):
            # Calculate the factor for this coefficient
            f = c[a] * comb(k, a)
            # Iterate over the degrees to raise
            for j in range(d + 1):
                # Apply the formula to compute the new coefficients
                out[a + j] += f * comb(d, j) / comb(k + d, a + j)
        return out
    """
    Piecewise tensor product polynomial

    The value at point ``xp = (x', y', z', ...)`` is evaluated by first
    computing the interval indices `i` such that::

        x[0][i[0]] <= x' < x[0][i[0]+1]
        x[1][i[1]] <= y' < x[1][i[1]+1]
        ...

    and then computing::

        S = sum(c[k0-m0-1,...,kn-mn-1,i[0],...,i[n]]
                * (xp[0] - x[0][i[0]])**m0
                * ...
                * (xp[n] - x[n][i[n]])**mn
                for m0 in range(k[0]+1)
                ...
                for mn in range(k[n]+1))

    where ``k[j]`` is the degree of the polynomial in dimension j. This
    representation is the piecewise multivariate power basis.

    Parameters
    ----------
    c : ndarray, shape (k0, ..., kn, m0, ..., mn, ...)
        Polynomial coefficients, with polynomial order `kj` and
        `mj+1` intervals for each dimension `j`.
    x : ndim-tuple of ndarrays, shapes (mj+1,)
        Polynomial breakpoints for each dimension. These must be
        sorted in increasing order.
    extrapolate : bool, optional
        Whether to extrapolate to out-of-bounds points based on first
        and last intervals, or to return NaNs. Default: True.

    Attributes
    ----------
    x : tuple of ndarrays
        Breakpoints.
    c : ndarray
        Coefficients of the polynomials.

    Methods
    -------
    __call__
    derivative
    antiderivative
    integrate
    integrate_1d
    construct_fast

    See also
    --------
    PPoly : piecewise polynomials in 1D

    Notes
    -----
    High-order polynomials in the power basis can be numerically
    unstable.

    """

    def __init__(self, c, x, extrapolate=None):
        # 将输入的 x 数组转换为元组，每个数组的数据类型为 float64
        self.x = tuple(np.ascontiguousarray(v, dtype=np.float64) for v in x)
        # 将输入的 c 数组转换为 ndarray 类型的多维数组
        self.c = np.asarray(c)
        # 如果未提供 extrapolate 参数，默认为 True
        if extrapolate is None:
            extrapolate = True
        # 将 extrapolate 参数转换为布尔值
        self.extrapolate = bool(extrapolate)

        # 维度数为 x 元组的长度
        ndim = len(self.x)
        # 检查 x 数组中的每个数组是否都是一维的
        if any(v.ndim != 1 for v in self.x):
            raise ValueError("x arrays must all be 1-dimensional")
        # 检查 x 数组中的每个数组是否至少包含两个点
        if any(v.size < 2 for v in self.x):
            raise ValueError("x arrays must all contain at least 2 points")
        # 检查 c 数组的维度是否足够用来表示多项式
        if c.ndim < 2*ndim:
            raise ValueError("c must have at least 2*len(x) dimensions")
        # 检查 x 数组是否按增序排列
        if any(np.any(v[1:] - v[:-1] < 0) for v in self.x):
            raise ValueError("x-coordinates are not in increasing order")
        # 检查 c 和 x 是否在区间数上一致
        if any(a != b.size - 1 for a, b in zip(c.shape[ndim:2*ndim], self.x)):
            raise ValueError("x and c do not agree on the number of intervals")

        # 获取 c 数组的数据类型，并将 c 转换为相同的数据类型和内存布局
        dtype = self._get_dtype(self.c.dtype)
        self.c = np.ascontiguousarray(self.c, dtype=dtype)

    @classmethod
    # 构造快速访问对象的类方法，用于创建分段多项式对象，不进行检查。
    @classmethod
    def construct_fast(cls, c, x, extrapolate=None):
        """
        Construct the piecewise polynomial without making checks.

        Takes the same parameters as the constructor. Input arguments
        ``c`` and ``x`` must be arrays of the correct shape and type.  The
        ``c`` array can only be of dtypes float and complex, and ``x``
        array must have dtype float.

        """
        # 使用特殊方法创建新的对象实例
        self = object.__new__(cls)
        # 设置对象属性 c 为输入的系数数组
        self.c = c
        # 设置对象属性 x 为输入的节点数组
        self.x = x
        # 如果未指定 extrapolate 参数，则默认为 True
        if extrapolate is None:
            extrapolate = True
        # 设置对象属性 extrapolate 为指定的 extrapolate 参数值
        self.extrapolate = extrapolate
        # 返回创建的对象实例
        return self

    # 内部方法，根据给定的 dtype 返回正确的数据类型
    def _get_dtype(self, dtype):
        if np.issubdtype(dtype, np.complexfloating) \
               or np.issubdtype(self.c.dtype, np.complexfloating):
            # 如果 dtype 是复数类型或者 self.c 的数据类型是复数类型，则返回 np.complex128
            return np.complex128
        else:
            # 否则返回 np.float64
            return np.float64

    # 内部方法，确保 self.c 是 C 连续的数组
    def _ensure_c_contiguous(self):
        if not self.c.flags.c_contiguous:
            # 如果 self.c 不是 C 连续的，则创建其副本并赋值给 self.c
            self.c = self.c.copy()
        if not isinstance(self.x, tuple):
            # 如果 self.x 不是元组类型，则将其转换为元组
            self.x = tuple(self.x)
    def __call__(self, x, nu=None, extrapolate=None):
        """
        Evaluate the piecewise polynomial or its derivative

        Parameters
        ----------
        x : array-like
            Points to evaluate the interpolant at.
        nu : tuple, optional
            Orders of derivatives to evaluate. Each must be non-negative.
        extrapolate : bool, optional
            Whether to extrapolate to out-of-bounds points based on first
            and last intervals, or to return NaNs.

        Returns
        -------
        y : array-like
            Interpolated values. Shape is determined by replacing
            the interpolation axis in the original array with the shape of x.

        Notes
        -----
        Derivatives are evaluated piecewise for each polynomial
        segment, even if the polynomial is not differentiable at the
        breakpoints. The polynomial intervals are considered half-open,
        ``[a, b)``, except for the last interval which is closed
        ``[a, b]``.

        """
        # 如果 extrapolate 参数为 None，则使用对象的默认值 self.extrapolate
        if extrapolate is None:
            extrapolate = self.extrapolate
        else:
            # 将 extrapolate 转换为布尔值
            extrapolate = bool(extrapolate)

        # 确定输入数组 x 的维数
        ndim = len(self.x)

        # 将 x 转换为多维坐标数组
        x = _ndim_coords_from_arrays(x)
        # 记录 x 的原始形状
        x_shape = x.shape
        # 将 x 展平为二维数组，并转换为连续的浮点数数组
        x = np.ascontiguousarray(x.reshape(-1, x.shape[-1]), dtype=np.float64)

        # 如果 nu 未提供，则初始化为全零数组，长度为 ndim
        if nu is None:
            nu = np.zeros((ndim,), dtype=np.intc)
        else:
            # 将 nu 转换为整数数组，并验证其维度和长度是否正确
            nu = np.asarray(nu, dtype=np.intc)
            if nu.ndim != 1 or nu.shape[0] != ndim:
                raise ValueError("invalid number of derivative orders nu")

        # 计算三个维度的乘积，用于重塑系数数组
        dim1 = prod(self.c.shape[:ndim])
        dim2 = prod(self.c.shape[ndim:2*ndim])
        dim3 = prod(self.c.shape[2*ndim:])
        # 创建一个整数数组，记录各个维度的大小
        ks = np.array(self.c.shape[:ndim], dtype=np.intc)

        # 创建一个空数组，用于存储计算结果
        out = np.empty((x.shape[0], dim3), dtype=self.c.dtype)
        # 确保系数数组是 C 连续的
        self._ensure_c_contiguous()

        # 调用 C 扩展的函数 evaluate_nd 进行多维插值计算
        _ppoly.evaluate_nd(self.c.reshape(dim1, dim2, dim3),
                           self.x,
                           ks,
                           x,
                           nu,
                           bool(extrapolate),
                           out)

        # 将结果数组重新塑形为原始输入 x 的形状
        return out.reshape(x_shape[:-1] + self.c.shape[2*ndim:])
    def _derivative_inplace(self, nu, axis):
        """
        Compute 1-D derivative along a selected dimension in-place
        May result to non-contiguous c array.
        """
        # 如果 nu 小于 0，则调用反导数函数处理
        if nu < 0:
            return self._antiderivative_inplace(-nu, axis)

        # 获取数组的维度
        ndim = len(self.x)
        # 对轴进行归一化处理，确保在数组维度范围内
        axis = axis % ndim

        # 减少阶数
        if nu == 0:
            # 若阶数为 0，则无操作
            return
        else:
            # 构建切片对象
            sl = [slice(None)] * ndim
            sl[axis] = slice(None, -nu, None)
            # 从 self.c 中获取切片 c2
            c2 = self.c[tuple(sl)]

        # 若 c2 在指定轴上的形状为 0，则导数阶数为 0
        if c2.shape[axis] == 0:
            shp = list(c2.shape)
            shp[axis] = 1
            c2 = np.zeros(shp, dtype=c2.dtype)

        # 根据正确的上升阶乘因子进行乘法操作
        factor = spec.poch(np.arange(c2.shape[axis], 0, -1), nu)
        sl = [None] * c2.ndim
        sl[axis] = slice(None)
        c2 *= factor[tuple(sl)]

        # 将结果赋值给 self.c
        self.c = c2

    def _antiderivative_inplace(self, nu, axis):
        """
        Compute 1-D antiderivative along a selected dimension
        May result to non-contiguous c array.
        """
        # 如果 nu 小于等于 0，则调用导数函数处理
        if nu <= 0:
            return self._derivative_inplace(-nu, axis)

        # 获取数组的维度
        ndim = len(self.x)
        # 对轴进行归一化处理，确保在数组维度范围内
        axis = axis % ndim

        # 创建轴置换列表，将选定轴移至第一个位置
        perm = list(range(ndim))
        perm[0], perm[axis] = perm[axis], perm[0]
        perm = perm + list(range(ndim, self.c.ndim))

        # 对数组进行轴置换
        c = self.c.transpose(perm)

        # 创建新数组 c2，将原数组 c 的内容复制进去，并扩展 nu 个零元素
        c2 = np.zeros((c.shape[0] + nu,) + c.shape[1:], dtype=c.dtype)
        c2[:-nu] = c

        # 根据正确的上升阶乘因子进行除法操作
        factor = spec.poch(np.arange(c.shape[0], 0, -1), nu)
        c2[:-nu] /= factor[(slice(None),) + (None,) * (c.ndim - 1)]

        # 修正添加的自由度的连续性
        perm2 = list(range(c2.ndim))
        perm2[1], perm2[ndim + axis] = perm2[ndim + axis], perm2[1]

        # 对 c2 进行轴置换和形状重塑，并调用 _ppoly.fix_continuity 函数处理
        c2 = c2.transpose(perm2)
        c2 = c2.copy()
        _ppoly.fix_continuity(c2.reshape(c2.shape[0], c2.shape[1], -1),
                              self.x[axis], nu - 1)

        # 再次进行轴置换
        c2 = c2.transpose(perm2)
        c2 = c2.transpose(perm)

        # 完成操作，将结果赋值给 self.c
        self.c = c2
    def derivative(self, nu):
        """
        Construct a new piecewise polynomial representing the derivative.

        Parameters
        ----------
        nu : ndim-tuple of int
            Order of derivatives to evaluate for each dimension.
            If negative, the antiderivative is returned.

        Returns
        -------
        pp : NdPPoly
            Piecewise polynomial of orders (k[0] - nu[0], ..., k[n] - nu[n])
            representing the derivative of this polynomial.

        Notes
        -----
        Derivatives are evaluated piecewise for each polynomial
        segment, even if the polynomial is not differentiable at the
        breakpoints. The polynomial intervals in each dimension are
        considered half-open, ``[a, b)``, except for the last interval
        which is closed ``[a, b]``.

        """
        # 创建一个快速构建的新对象 p，基于当前对象的系数 c、节点 x 和外推方法 extrapolate
        p = self.construct_fast(self.c.copy(), self.x, self.extrapolate)

        # 对每个维度按照给定的 nu 值进行求导操作
        for axis, n in enumerate(nu):
            # 在指定的维度上进行就地求导操作
            p._derivative_inplace(n, axis)

        # 确保结果的系数是按照 C 连续的
        p._ensure_c_contiguous()
        # 返回构建好的导数多项式对象 p
        return p

    def antiderivative(self, nu):
        """
        Construct a new piecewise polynomial representing the antiderivative.

        Antiderivative is also the indefinite integral of the function,
        and derivative is its inverse operation.

        Parameters
        ----------
        nu : ndim-tuple of int
            Order of derivatives to evaluate for each dimension.
            If negative, the derivative is returned.

        Returns
        -------
        pp : PPoly
            Piecewise polynomial of order k2 = k + n representing
            the antiderivative of this polynomial.

        Notes
        -----
        The antiderivative returned by this function is continuous and
        continuously differentiable to order n-1, up to floating point
        rounding error.

        """
        # 创建一个快速构建的新对象 p，基于当前对象的系数 c、节点 x 和外推方法 extrapolate
        p = self.construct_fast(self.c.copy(), self.x, self.extrapolate)

        # 对每个维度按照给定的 nu 值进行反求导（积分）操作
        for axis, n in enumerate(nu):
            # 在指定的维度上进行就地反求导（积分）操作
            p._antiderivative_inplace(n, axis)

        # 确保结果的系数是按照 C 连续的
        p._ensure_c_contiguous()
        # 返回构建好的反导数多项式对象 p
        return p
    def integrate_1d(self, a, b, axis, extrapolate=None):
        r"""
        Compute NdPPoly representation for one dimensional definite integral

        The result is a piecewise polynomial representing the integral:

        .. math::

           p(y, z, ...) = \int_a^b dx\, p(x, y, z, ...)

        where the dimension integrated over is specified with the
        `axis` parameter.

        Parameters
        ----------
        a, b : float
            Lower and upper bound for integration.
        axis : int
            Dimension over which to compute the 1-D integrals
        extrapolate : bool, optional
            Whether to extrapolate to out-of-bounds points based on first
            and last intervals, or to return NaNs.

        Returns
        -------
        ig : NdPPoly or array-like
            Definite integral of the piecewise polynomial over [a, b].
            If the polynomial was 1D, an array is returned,
            otherwise, an NdPPoly object.

        """
        # Determine extrapolation behavior
        if extrapolate is None:
            extrapolate = self.extrapolate
        else:
            extrapolate = bool(extrapolate)

        # Determine the number of dimensions in the polynomial
        ndim = len(self.x)
        
        # Normalize the axis parameter to ensure it is within valid range
        axis = int(axis) % ndim

        # Reorder dimensions to prepare for integration
        c = self.c
        swap = list(range(c.ndim))
        swap.insert(0, swap[axis])
        del swap[axis + 1]
        swap.insert(1, swap[ndim + axis])
        del swap[ndim + axis + 1]

        # Transpose the coefficients array to align with integration dimension
        c = c.transpose(swap)

        # Construct a fast piecewise polynomial along the specified axis
        p = PPoly.construct_fast(c.reshape(c.shape[0], c.shape[1], -1),
                                 self.x[axis],
                                 extrapolate=extrapolate)
        
        # Integrate the polynomial over the specified bounds
        out = p.integrate(a, b, extrapolate=extrapolate)

        # Construct the final result based on the polynomial's dimensions
        if ndim == 1:
            return out.reshape(c.shape[2:])
        else:
            c = out.reshape(c.shape[2:])
            x = self.x[:axis] + self.x[axis+1:]
            return self.construct_fast(c, x, extrapolate=extrapolate)
    def integrate(self, ranges, extrapolate=None):
        """
        Compute a definite integral over a piecewise polynomial.

        Parameters
        ----------
        ranges : ndim-tuple of 2-tuples float
            Sequence of lower and upper bounds for each dimension,
            ``[(a[0], b[0]), ..., (a[ndim-1], b[ndim-1])]``
        extrapolate : bool, optional
            Whether to extrapolate to out-of-bounds points based on first
            and last intervals, or to return NaNs.

        Returns
        -------
        ig : array_like
            Definite integral of the piecewise polynomial over
            [a[0], b[0]] x ... x [a[ndim-1], b[ndim-1]]

        """

        ndim = len(self.x)  # 获取维度数量

        if extrapolate is None:
            extrapolate = self.extrapolate  # 如果未指定 extrapolate，则使用对象的默认值
        else:
            extrapolate = bool(extrapolate)  # 将 extrapolate 转换为布尔类型

        if not hasattr(ranges, '__len__') or len(ranges) != ndim:
            raise ValueError("Range not a sequence of correct length")  # 如果 ranges 不是正确长度的序列，则引发 ValueError

        self._ensure_c_contiguous()  # 确保数组是 C 连续的

        # 重用一维积分例程
        c = self.c  # 获取多项式系数数组
        for n, (a, b) in enumerate(ranges):
            swap = list(range(c.ndim))  # 创建维度序号的列表
            swap.insert(1, swap[ndim - n])  # 在特定位置插入交换索引
            del swap[ndim - n + 1]  # 删除原索引位置

            c = c.transpose(swap)  # 根据交换的索引重新排列系数数组

            p = PPoly.construct_fast(c, self.x[n], extrapolate=extrapolate)  # 构建快速评估的分段多项式对象
            out = p.integrate(a, b, extrapolate=extrapolate)  # 计算分段多项式在指定范围内的积分
            c = out.reshape(c.shape[2:])  # 调整积分结果的形状以匹配系数数组

        return c  # 返回最终的积分结果数组
```