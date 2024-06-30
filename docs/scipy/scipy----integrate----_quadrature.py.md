# `D:\src\scipysrc\scipy\scipy\integrate\_quadrature.py`

```
# 导入未来的注解功能
from __future__ import annotations
# 导入类型检查相关的模块
from typing import TYPE_CHECKING, Callable, Any, cast
# 导入 numpy 库
import numpy as np
# 导入 numpy 的类型定义
import numpy.typing as npt
# 导入 math 库
import math
# 导入警告模块
import warnings
# 导入 namedtuple 类型
from collections import namedtuple

# 导入 scipy 库中的函数和类
from scipy.special import roots_legendre
from scipy.special import gammaln, logsumexp
from scipy._lib._util import _rng_spawn

# 定义模块导出的符号列表
__all__ = ['fixed_quad', 'romb',
           'trapezoid', 'simpson',
           'cumulative_trapezoid', 'newton_cotes',
           'qmc_quad', 'cumulative_simpson']

# 定义使用梯形法则进行数值积分的函数
def trapezoid(y, x=None, dx=1.0, axis=-1):
    r"""
    使用复合梯形法则沿着指定轴积分。

    如果提供了 `x`，则沿着其元素顺序执行积分 - 它们不会被排序。

    对 `y`（`x`）沿着给定轴上的每个 1D 切片进行积分，计算 :math:`\int y(x) dx`。
    当指定 `x` 时，这将沿着参数曲线进行积分，计算 :math:`\int_t y(t) dt =
    \int_t y(t) \left.\frac{dx}{dt}\right|_{x=x(t)} dt`。

    Parameters
    ----------
    y : array_like
        要积分的输入数组。
    x : array_like, optional
        与 `y` 值对应的样本点。如果 `x` 是 None，则假定样本点均匀间隔为 `dx`。默认为 None。
    dx : scalar, optional
        当 `x` 是 None 时，样本点之间的间距。默认为 1。
    axis : int, optional
        要积分的轴。

    Returns
    -------
    trapezoid : float or ndarray
        使用梯形法则沿单个轴近似积分 `y` 的定积分 = n 维数组。如果 `y` 是一维数组，则结果是一个浮点数。
        如果 `n` 大于 1，则结果是一个 `n-1` 维数组。

    See Also
    --------
    cumulative_trapezoid, simpson, romb

    Notes
    -----
    图片 [2]_ 说明了梯形法则 -- 默认情况下，点之间的 x 轴距离为 1.0，点的 y 轴位置将从 `y` 数组中获取，
    或者可以用 `x` 数组或 `dx` 标量提供。返回值将等于红线下的组合面积。

    References
    ----------
    .. [1] Wikipedia 页面: https://en.wikipedia.org/wiki/Trapezoidal_rule

    .. [2] Illustration image:
           https://en.wikipedia.org/wiki/File:Composite_trapezoidal_rule_illustration.png

    Examples
    --------
    在均匀间隔点上使用梯形法则：

    >>> import numpy as np
    >>> from scipy import integrate
    >>> integrate.trapezoid([1, 2, 3])
    4.0

    样本点之间的间距可以由 `x` 或 `dx` 参数选择：

    >>> integrate.trapezoid([1, 2, 3], x=[4, 6, 8])
    8.0
    >>> integrate.trapezoid([1, 2, 3], dx=2)
    8.0

    使用递减的 `x` 对应于反向积分：

    >>> integrate.trapezoid([1, 2, 3], x=[8, 6, 4])
    -8.0
    # 将输入的 y 转换为 NumPy 数组（如果尚未是）
    y = np.asanyarray(y)
    
    # 如果输入的 x 为 None，则使用预定义的 dx 作为步长差异
    if x is None:
        d = dx
    else:
        # 将输入的 x 转换为 NumPy 数组
        x = np.asanyarray(x)
        
        # 如果 x 是一维数组，则计算其差分作为步长差异
        if x.ndim == 1:
            d = np.diff(x)
            
            # 将 d 重塑为正确的形状，以便与 y 的维度匹配
            shape = [1] * y.ndim
            shape[axis] = d.shape[0]
            d = d.reshape(shape)
        else:
            # 如果 x 是多维数组，则沿指定轴计算差分作为步长差异
            d = np.diff(x, axis=axis)
    
    # 确定 y 的维度数
    nd = y.ndim
    
    # 创建两个切片对象，用于获取 y 数组中相邻元素的切片
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    
    # 为切片对象指定轴，以便进行差分计算
    slice1[axis] = slice(1, None)   # 从第二个元素开始到最后一个元素
    slice2[axis] = slice(None, -1)   # 从第一个元素到倒数第二个元素
    
    # 尝试计算梯形积分的近似值
    try:
        # 使用梯形法则计算近似积分值并沿指定轴求和
        ret = (d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0).sum(axis)
    except ValueError:
        # 如果操作失败，将 d 和 y 强制转换为 ndarray 类型再进行计算
        d = np.asarray(d)
        y = np.asarray(y)
        ret = np.add.reduce(d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0, axis)
    
    # 返回计算得到的积分值
    return ret
if TYPE_CHECKING:
    # 如果 TYPE_CHECKING 为真，则需要引入 typing 模块中的 Protocol
    # 参考：https://github.com/python/mypy/issues/2087#issuecomment-462726600
    from typing import Protocol

    # 定义 CacheAttributes 协议类型，要求含有一个 cache 属性，其值为字典，键为整数，值为元组（任意类型，任意类型）
    class CacheAttributes(Protocol):
        cache: dict[int, tuple[Any, Any]]
else:
    # 否则，CacheAttributes 为 Callable 类型
    CacheAttributes = Callable


# cache_decorator 函数装饰器，接受一个函数 func 作为参数，并返回一个 CacheAttributes 类型的对象
def cache_decorator(func: Callable) -> CacheAttributes:
    return cast(CacheAttributes, func)


# _cached_roots_legendre 函数，用于缓存 roots_legendre 函数的结果，以加速 fixed_quad 函数的调用
@cache_decorator
def _cached_roots_legendre(n):
    """
    Cache roots_legendre results to speed up calls of the fixed_quad
    function.
    """
    # 如果 n 已经在 _cached_roots_legendre.cache 中，则直接返回缓存结果
    if n in _cached_roots_legendre.cache:
        return _cached_roots_legendre.cache[n]

    # 否则，计算 roots_legendre(n) 的结果，并将其缓存起来
    _cached_roots_legendre.cache[n] = roots_legendre(n)
    return _cached_roots_legendre.cache[n]


# 初始化 _cached_roots_legendre.cache 为一个空字典
_cached_roots_legendre.cache = dict()


# fixed_quad 函数，用固定阶段的 Gauss 积分法计算定义域在 [a, b] 上的定积分
def fixed_quad(func, a, b, args=(), n=5):
    """
    Compute a definite integral using fixed-order Gaussian quadrature.

    Integrate `func` from `a` to `b` using Gaussian quadrature of
    order `n`.

    Parameters
    ----------
    func : callable
        A Python function or method to integrate (must accept vector inputs).
        If integrating a vector-valued function, the returned array must have
        shape ``(..., len(x))``.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.
    args : tuple, optional
        Extra arguments to pass to function, if any.
    n : int, optional
        Order of quadrature integration. Default is 5.

    Returns
    -------
    val : float
        Gaussian quadrature approximation to the integral
    none : None
        Statically returned value of None

    See Also
    --------
    quad : adaptive quadrature using QUADPACK
    dblquad : double integrals
    tplquad : triple integrals
    romb : integrators for sampled data
    simpson : integrators for sampled data
    cumulative_trapezoid : cumulative integration for sampled data

    Examples
    --------
    >>> from scipy import integrate
    >>> import numpy as np
    >>> f = lambda x: x**8
    >>> integrate.fixed_quad(f, 0.0, 1.0, n=4)
    (0.1110884353741496, None)
    >>> integrate.fixed_quad(f, 0.0, 1.0, n=5)
    (0.11111111111111102, None)
    >>> print(1/9.0)  # analytical result
    0.1111111111111111

    >>> integrate.fixed_quad(np.cos, 0.0, np.pi/2, n=4)
    (0.9999999771971152, None)
    >>> integrate.fixed_quad(np.cos, 0.0, np.pi/2, n=5)
    (1.000000000039565, None)
    >>> np.sin(np.pi/2)-np.sin(0)  # analytical result
    1.0

    """
    # 调用 _cached_roots_legendre(n) 获取 Gauss 积分法的节点和权重
    x, w = _cached_roots_legendre(n)
    # 取实部，因为 roots_legendre 可能返回复数
    x = np.real(x)
    # 如果 a 或 b 是无穷大，则抛出 ValueError
    if np.isinf(a) or np.isinf(b):
        raise ValueError("Gaussian quadrature is only available for "
                         "finite limits.")
    # 计算 Gauss 积分节点对应的实际坐标
    y = (b-a)*(x+1)/2.0 + a
    # 返回定积分的近似值和 None（静态返回值）
    return (b-a)/2.0 * np.sum(w*func(y, *args), axis=-1), None


# tupleset 函数，用于修改元组 t 中第 i 个位置的值，并返回修改后的元组
def tupleset(t, i, value):
    l = list(t)
    l[i] = value
    return tuple(l)


# cumulative_trapezoid 函数，用梯形法计算累积的积分值
def cumulative_trapezoid(y, x=None, dx=1.0, axis=-1, initial=None):
    """
    Cumulatively integrate y(x) using the composite trapezoidal rule.

    Parameters
    ----------
    y : array_like
        Values to integrate.
    x : array_like, optional
        The coordinate to integrate along. If None (default), use spacing `dx`
        between consecutive elements in `y`.
    dx : float, optional
        Spacing between elements of `y`. Only used if `x` is None.
    axis : int, optional
        Specifies the axis to cumulate. Default is -1 (last axis).
    initial : scalar, optional
        If given, insert this value at the beginning of the returned result.
        0 or None are the only values accepted. Default is None, which means
        `res` has one element less than `y` along the axis of integration.

    Returns
    -------
    res : ndarray
        The result of cumulative integration of `y` along `axis`.
        If `initial` is None, the shape is such that the axis of integration
        has one less value than `y`. If `initial` is given, the shape is equal
        to that of `y`.

    See Also
    --------
    numpy.cumsum, numpy.cumprod
    cumulative_simpson : cumulative integration using Simpson's 1/3 rule
    quad : adaptive quadrature using QUADPACK
    fixed_quad : fixed-order Gaussian quadrature
    dblquad : double integrals
    tplquad : triple integrals
    romb : integrators for sampled data

    Examples
    --------
    >>> from scipy import integrate
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> x = np.linspace(-2, 2, num=20)
    >>> y = x
    >>> y_int = integrate.cumulative_trapezoid(y, x, initial=0)
    >>> plt.plot(x, y_int, 'ro', x, y[0] + 0.5 * x**2, 'b-')
    >>> plt.show()

    """
    # 将输入的 y 转换为 ndarray 格式
    y = np.asarray(y)
    # 检查在指定的轴上 y 的形状是否为零，如果是，则抛出错误
    if y.shape[axis] == 0:
        raise ValueError("At least one point is required along `axis`.")
    # 如果未提供 x，则使用 dx 计算间距 d
    if x is None:
        d = dx
    else:
        # 将输入的 x 转换为 ndarray 格式
        x = np.asarray(x)
        # 如果 x 的维度为 1，则计算差分 d，并进行形状修正
        if x.ndim == 1:
            d = np.diff(x)
            shape = [1] * y.ndim
            shape[axis] = -1
            d = d.reshape(shape)
        # 如果 x 和 y 的形状不匹配，则抛出错误
        elif len(x.shape) != len(y.shape):
            raise ValueError("If given, shape of x must be 1-D or the "
                             "same as y.")
        else:
            # 计算沿指定轴的差分 d
            d = np.diff(x, axis=axis)

        # 检查沿指定轴的差分长度是否与 y 的长度匹配，否则抛出错误
        if d.shape[axis] != y.shape[axis] - 1:
            raise ValueError("If given, length of x along axis must be the "
                             "same as y.")

    # 获取 y 的维度数
    nd = len(y.shape)
    # 创建用于切片的元组
    slice1 = tupleset((slice(None),)*nd, axis, slice(1, None))
    slice2 = tupleset((slice(None),)*nd, axis, slice(None, -1))
    # 使用复合梯形法则计算沿指定轴的累积积分结果 res
    res = np.cumsum(d * (y[slice1] + y[slice2]) / 2.0, axis=axis)
    # 如果 `initial` 参数不是 None，则进行以下检查和操作
    if initial is not None:
        # 如果 `initial` 参数不等于 0，抛出数值错误异常
        if initial != 0:
            raise ValueError("`initial` must be `None` or `0`.")
        # 如果 `initial` 参数不是标量（scalar），抛出数值错误异常
        if not np.isscalar(initial):
            raise ValueError("`initial` parameter should be a scalar.")

        # 获取结果数组 `res` 的形状，并转换为列表
        shape = list(res.shape)
        # 将指定轴 `axis` 的长度设置为 1
        shape[axis] = 1
        # 在指定轴 `axis` 上使用初始值 `initial` 创建全新数组，与原始数组 `res` 进行拼接
        res = np.concatenate([np.full(shape, initial, dtype=res.dtype), res],
                             axis=axis)

    # 返回处理后的结果数组 `res`
    return res
# 使用复合辛普森法则对数组 y 进行积分。如果 x 为 None，则假设间距为 dx。
def _basic_simpson(y, start, stop, x, dx, axis):
    # 确定数组 y 的维度
    nd = len(y.shape)
    # 如果未指定起始索引 start，则默认为 0
    if start is None:
        start = 0
    # 步长设定为 2
    step = 2
    # 创建一个全部为切片的元组，长度为数组 y 的维度
    slice_all = (slice(None),)*nd
    # 构建不同步长下的切片索引
    slice0 = tupleset(slice_all, axis, slice(start, stop, step))
    slice1 = tupleset(slice_all, axis, slice(start+1, stop+1, step))
    slice2 = tupleset(slice_all, axis, slice(start+2, stop+2, step))

    # 如果 x 为 None，则使用等间距辛普森法则
    if x is None:
        # 计算辛普森法则下的积分结果
        result = np.sum(y[slice0] + 4.0*y[slice1] + y[slice2], axis=axis)
        result *= dx / 3.0
    else:
        # 考虑可能存在不同间距的情况
        h = np.diff(x, axis=axis)
        sl0 = tupleset(slice_all, axis, slice(start, stop, step))
        sl1 = tupleset(slice_all, axis, slice(start+1, stop+1, step))
        h0 = h[sl0].astype(float, copy=False)
        h1 = h[sl1].astype(float, copy=False)
        hsum = h0 + h1
        hprod = h0 * h1
        h0divh1 = np.true_divide(h0, h1, out=np.zeros_like(h0), where=h1 != 0)
        tmp = hsum/6.0 * (y[slice0] *
                          (2.0 - np.true_divide(1.0, h0divh1,
                                                out=np.zeros_like(h0divh1),
                                                where=h0divh1 != 0)) +
                          y[slice1] * (hsum *
                                       np.true_divide(hsum, hprod,
                                                      out=np.zeros_like(hsum),
                                                      where=hprod != 0)) +
                          y[slice2] * (2.0 - h0divh1))
        result = np.sum(tmp, axis=axis)
    # 返回积分结果
    return result


def simpson(y, *, x=None, dx=1.0, axis=-1):
    """
    使用给定轴上的样本和复合辛普森法则对 y(x) 进行积分。如果 x 为 None，则假设间距为 dx。

    如果样本数为偶数 N，则有 N-1 个间隔，但辛普森法则要求偶数个间隔。参数 'even' 控制如何处理这种情况。

    Parameters
    ----------
    y : array_like
        待积分的数组。
    x : array_like, optional
        如果给定，则为 `y` 的采样点。
    dx : float, optional
        `x` 轴上的积分点间距。仅在 `x` 为 None 时使用。默认为 1。
    axis : int, optional
        进行积分的轴。默认为最后一个轴。

    Returns
    -------
    float
        使用复合辛普森法则计算的估计积分值。

    See Also
    --------
    quad : 使用 QUADPACK 的自适应积分
    fixed_quad : 固定阶数的高斯积分
    dblquad : 双重积分
    tplquad : 三重积分
    romb : 对采样数据的积分器
    cumulative_trapezoid : 采样数据的累积梯形积分
    cumulative_simpson : 使用辛普森 1/3 法则的累积积分

    Notes
    -----
    对于等间距的奇数样本数，结果为：
    """
    y = np.asarray(y)
    # 将输入的 y 转换为 NumPy 数组，确保 y 是可操作的数组形式
    nd = len(y.shape)
    # 获取数组 y 的维度数
    N = y.shape[axis]
    # 获取数组 y 在指定轴上的长度，axis 是一个预定义的变量，表示轴的索引
    last_dx = dx
    # 将 dx 赋值给 last_dx，这里假设 dx 是之前定义过的变量
    returnshape = 0
    # 初始化 returnshape 为 0，用于表示返回形状的标志位
    if x is not None:
        # 如果 x 不为空
        x = np.asarray(x)
        # 将输入的 x 转换为 NumPy 数组
        if len(x.shape) == 1:
            # 如果 x 是一维数组
            shapex = [1] * nd
            # 创建一个与 y 维度数相同的长度为 1 的列表 shapex
            shapex[axis] = x.shape[0]
            # 将 x 的长度设置为 shapex 在指定轴上的长度
            saveshape = x.shape
            # 保存 x 的原始形状
            returnshape = 1
            # 将 returnshape 设置为 1，表示返回值将具有与输入 x 相同的形状
            x = x.reshape(tuple(shapex))
            # 将 x 重新塑造成具有 shapex 形状的数组
        elif len(x.shape) != len(y.shape):
            # 如果 x 的维度数与 y 的维度数不同
            raise ValueError("If given, shape of x must be 1-D or the "
                             "same as y.")
            # 抛出值错误，要求 x 的形状必须是一维的或者与 y 相同
        if x.shape[axis] != N:
            # 如果 x 在指定轴上的长度与 y 不同
            raise ValueError("If given, length of x along axis must be the "
                             "same as y.")
            # 抛出值错误，要求 x 在指定轴上的长度必须与 y 相同
    # 如果 N 是偶数
    if N % 2 == 0:
        # 初始化 val 和 result 为浮点数 0.0
        val = 0.0
        result = 0.0
        # 使用 slice_all 构建一个完整的切片元组，长度为 nd
        slice_all = (slice(None),) * nd

        # 如果 N 等于 2
        if N == 2:
            # 需要至少有三个点在积分轴上形成抛物线段。如果只有两个点，任何 'avg', 'first', 'last' 的选择结果相同。
            # 设置两个切片，分别对应倒数第一和第二个点
            slice1 = tupleset(slice_all, axis, -1)
            slice2 = tupleset(slice_all, axis, -2)
            # 如果给定了 x 数组，则计算最后两个点之间的间距
            if x is not None:
                last_dx = x[slice1] - x[slice2]
            # 计算 Simpson 法则下的一阶逼近（对于只有两个点的情况）
            val += 0.5 * last_dx * (y[slice1] + y[slice2])
        else:
            # 使用 Simpson 法则对前面的区间进行计算
            result = _basic_simpson(y, 0, N-3, x, dx, axis)

            # 设置三个切片，分别对应倒数第一、第二和第三个点
            slice1 = tupleset(slice_all, axis, -1)
            slice2 = tupleset(slice_all, axis, -2)
            slice3 = tupleset(slice_all, axis, -3)

            # 从给定的轴上抽取最后两个间距，并转换为浮点数数组 h
            h = np.asarray([dx, dx], dtype=np.float64)
            if x is not None:
                # 抽取轴上最后两个间距
                hm2 = tupleset(slice_all, axis, slice(-2, -1, 1))
                hm1 = tupleset(slice_all, axis, slice(-1, None, 1))

                # 计算轴上的间距差异
                diffs = np.float64(np.diff(x, axis=axis))
                h = [np.squeeze(diffs[hm2], axis=axis),
                     np.squeeze(diffs[hm1], axis=axis)]

            # Cartwright 的最后一个区间的修正
            # 根据给定的 Wikipedia 上的方程，进行修正
            num = 2 * h[1] ** 2 + 3 * h[0] * h[1]
            den = 6 * (h[1] + h[0])
            alpha = np.true_divide(
                num,
                den,
                out=np.zeros_like(den),
                where=den != 0
            )

            num = h[1] ** 2 + 3.0 * h[0] * h[1]
            den = 6 * h[0]
            beta = np.true_divide(
                num,
                den,
                out=np.zeros_like(den),
                where=den != 0
            )

            num = 1 * h[1] ** 3
            den = 6 * h[0] * (h[0] + h[1])
            eta = np.true_divide(
                num,
                den,
                out=np.zeros_like(den),
                where=den != 0
            )

            # 将 Cartwright 修正的值加到 result 中
            result += alpha * y[slice1] + beta * y[slice2] - eta * y[slice3]

        # 将 val 的值加到 result 中
        result += val
    else:
        # 如果 N 是奇数，则直接使用 Simpson 法则进行计算
        result = _basic_simpson(y, 0, N-2, x, dx, axis)

    # 如果 returnshape 为 True，则将 x 重新 reshape 成 saveshape 的形状
    if returnshape:
        x = x.reshape(saveshape)

    # 返回计算得到的 result
    return result
def _cumulatively_sum_simpson_integrals(
    y: np.ndarray, 
    dx: np.ndarray, 
    integration_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Calculate cumulative sum of Simpson integrals.
    Takes as input the integration function to be used. 
    The integration_func is assumed to return the cumulative sum using
    composite Simpson's rule. Assumes the axis of summation is -1.
    """
    # Calculate Simpson integrals using the provided integration function
    sub_integrals_h1 = integration_func(y, dx)
    # Calculate Simpson integrals for reversed arrays (h2 intervals)
    sub_integrals_h2 = integration_func(y[..., ::-1], dx[..., ::-1])[..., ::-1]
    
    # Prepare shape for the output array to store cumulative integrals
    shape = list(sub_integrals_h1.shape)
    shape[-1] += 1
    sub_integrals = np.empty(shape)
    
    # Combine h1 and h2 intervals into sub_integrals array
    sub_integrals[..., :-1:2] = sub_integrals_h1[..., ::2]
    sub_integrals[..., 1::2] = sub_integrals_h2[..., ::2]
    # Integral over last subinterval can only be calculated from 
    # formula for h2
    sub_integrals[..., -1] = sub_integrals_h2[..., -1]
    
    # Compute cumulative sum along the specified axis
    res = np.cumsum(sub_integrals, axis=-1)
    return res


def _cumulative_simpson_equal_intervals(y: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """Calculate the Simpson integrals for all h1 intervals assuming equal interval
    widths. The function can also be used to calculate the integral for all
    h2 intervals by reversing the inputs, `y` and `dx`.
    """
    # Extract intervals and function values for h1 intervals
    d = dx[..., :-1]
    f1 = y[..., :-2]
    f2 = y[..., 1:-1]
    f3 = y[..., 2:]

    # Calculate Simpson's rule integrals for h1 intervals
    return d / 3 * (5 * f1 / 4 + 2 * f2 - f3 / 4)


def _cumulative_simpson_unequal_intervals(y: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """Calculate the Simpson integrals for all h1 intervals assuming unequal interval
    widths. The function can also be used to calculate the integral for all
    h2 intervals by reversing the inputs, `y` and `dx`.
    """
    # Extract intervals, function values, and combined intervals for h1 intervals
    x21 = dx[..., :-1]
    x32 = dx[..., 1:]
    f1 = y[..., :-2]
    f2 = y[..., 1:-1]
    f3 = y[..., 2:]

    x31 = x21 + x32
    x21_x31 = x21 / x31
    x21_x32 = x21 / x32
    x21x21_x31x32 = x21_x31 * x21_x32

    # Calculate Simpson's rule integrals for h1 intervals with unequal intervals
    coeff1 = 3 - x21_x31
    coeff2 = 3 + x21x21_x31x32 + x21_x31
    coeff3 = -x21x21_x31x32

    return x21 / 6 * (coeff1 * f1 + coeff2 * f2 + coeff3 * f3)


def _ensure_float_array(arr: npt.ArrayLike) -> np.ndarray:
    """Ensure input array is converted to float if it contains integers."""
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(float, copy=False)
    return arr
    y : array_like
        # 要进行积分的值。至少需要沿着 `axis` 方向有一个点。如果沿着 `axis` 方向提供的点少于两个，则无法使用 Simpson 积分，结果将使用 `cumulative_trapezoid` 计算。
    x : array_like, optional
        # 要沿着其进行积分的坐标。必须与 `y` 具有相同的形状，或者必须是与 `y` 在 `axis` 方向上具有相同长度的一维数组。`x` 在 `axis` 方向上也必须严格递增。
        # 如果 `x` 是 None（默认），则使用 `y` 中连续元素之间的间距 `dx` 执行积分。
    dx : scalar or array_like, optional
        # `y` 中元素之间的间距。仅在 `x` 是 None 时使用。可以是浮点数，也可以是与 `y` 具有相同形状但在 `axis` 方向上长度为一的数组。默认为 1.0。
    axis : int, optional
        # 指定要沿其进行积分的轴。默认为 -1（最后一个轴）。
    initial : scalar or array_like, optional
        # 如果给定，则在返回结果的开头插入此值，并将其添加到结果的其余部分。默认为 None，这意味着不返回 `x[0]` 处的值，并且 `res` 在积分轴上的长度比 `y` 少一个元素。可以是浮点数，也可以是与 `y` 具有相同形状但在 `axis` 方向上长度为一的数组。

    Returns
    -------
    res : ndarray
        # `y` 沿 `axis` 方向的累积积分结果。
        # 如果 `initial` 是 None，则形状使得积分轴上的长度比 `y` 少一个值。如果给定了 `initial`，则形状与 `y` 相同。

    See Also
    --------
    numpy.cumsum
    cumulative_trapezoid : 使用复合梯形规则进行累积积分
    simpson : 使用复合 Simpson 法则对采样数据进行积分

    Notes
    -----

    .. versionadded:: 1.12.0

    # 复合 Simpson's 1/3 方法可以用于近似采样输入函数 :math:`y(x)` 的定积分 [1]_。该方法假设在包含任意三个连续采样点的区间上有二次关系。

    # 考虑三个连续点：
    # :math:`(x_1, y_1), (x_2, y_2), (x_3, y_3)`。

    # 假设在这三个点上有二次关系，那么在 :math:`x_1` 到 :math:`x_2` 的子区间上的积分由 [2]_ 的公式 (8) 给出：

    # .. math::
    #    \int_{x_1}^{x_2} y(x) dx\ &= \frac{x_2-x_1}{6}\left[\
    #    \left\{3-\frac{x_2-x_1}{x_3-x_1}\right\} y_1 + \
    #    \left\{3 + \frac{(x_2-x_1)^2}{(x_3-x_2)(x_3-x_1)} + \
    #    \frac{x_2-x_1}{x_3-x_1}\right\} y_2\\
    #    - \frac{(x_2-x_1)^2}{(x_3-x_2)(x_3-x_1)} y_3\right]

    # 在 :math:`x_2` 到 :math:`x_3` 之间的积分通过交换 :math:`x_1` 和 :math:`x_3` 的位置来给出。
    # 对每个子区间分别估计积分，然后累积求和以获得最终结果。
    y = _ensure_float_array(y)



    # 将输入的 y 确保转换为浮点数数组
    y = _ensure_float_array(y)



    original_y = y
    original_shape = y.shape



    # 将原始的 y 数组和其形状保存下来
    original_y = y
    original_shape = y.shape



    try:
        # 尝试将 y 数组的指定轴与最后一个轴交换，以便在最后一个轴上进行操作
        y = np.swapaxes(y, axis, -1)
    except IndexError as e:
        # 如果发生索引错误，说明指定的轴超出了 y 的维度范围，抛出 ValueError 异常
        message = f"`axis={axis}` is not valid for `y` with `y.ndim={y.ndim}`."
        raise ValueError(message) from e



    # 验证并将 `axis` 参数标准化，使其在最后一个轴上操作
    original_y = y
    original_shape = y.shape
    try:
        y = np.swapaxes(y, axis, -1)
    except IndexError as e:
        message = f"`axis={axis}` is not valid for `y` with `y.ndim={y.ndim}`."
        raise ValueError(message) from e



    if y.shape[-1] < 3:
        # 如果 y 在最后一个轴上的长度小于 3，则调用 cumulative_trapezoid 函数进行积分估计
        res = cumulative_trapezoid(original_y, x, dx=dx, axis=axis, initial=None)
        # 将结果再次交换轴，恢复到操作前的形状
        res = np.swapaxes(res, axis, -1)



    # 如果 y 在最后一个轴上的长度小于 3，则使用 cumulative_trapezoid 函数进行积分估计
    if y.shape[-1] < 3:
        res = cumulative_trapezoid(original_y, x, dx=dx, axis=axis, initial=None)
        # 将结果再次交换轴，恢复到操作前的形状
        res = np.swapaxes(res, axis, -1)
    # 如果 x 不为 None，则确保 x 是一个浮点数数组
    x = _ensure_float_array(x)
    # 错误消息，用于报告 x 的形状必须与 y 相同或在指定轴上与 y 的长度相同（如果给定）
    message = ("If given, shape of `x` must be the same as `y` or 1-D with "
               "the same length as `y` along `axis`.")
    # 检查 x 的形状是否与原始形状相同，或者（如果 x 是 1-D 数组）其长度是否与指定轴上的 y 的长度相同
    if not (x.shape == original_shape
            or (x.ndim == 1 and len(x) == original_shape[axis])):
        # 如果不符合条件，抛出 ValueError 异常，显示错误消息
        raise ValueError(message)

    # 如果 x 是 1-D 数组，则使用 np.broadcast_to 将其广播到与 y 相同的形状；否则将 x 与指定轴交换
    x = np.broadcast_to(x, y.shape) if x.ndim == 1 else np.swapaxes(x, axis, -1)
    # 计算 x 在指定轴上的差分
    dx = np.diff(x, axis=-1)
    # 如果任何差分值小于等于零，则抛出 ValueError 异常，指出输入 x 必须严格递增
    if np.any(dx <= 0):
        raise ValueError("Input x must be strictly increasing.")
    # 调用 _cumulatively_sum_simpson_integrals 函数，使用不等间隔辛普森积分方法计算结果
    res = _cumulatively_sum_simpson_integrals(
        y, dx, _cumulative_simpson_unequal_intervals
    )

else:
    # 如果 x 是 None，则确保 dx 是一个浮点数数组
    dx = _ensure_float_array(dx)
    # 最终的 dx 形状，通过将指定轴上的点数减少 1 来设置
    final_dx_shape = tupleset(original_shape, axis, original_shape[axis] - 1)
    # 替代输入的 dx 形状，只在指定轴上有 1 个点的形状
    alt_input_dx_shape = tupleset(original_shape, axis, 1)
    # 错误消息，用于报告如果提供了 `dx`，则它必须是标量或者与 `y` 相同形状但在指定轴上只有 1 个点
    message = ("If provided, `dx` must either be a scalar or have the same "
               "shape as `y` but with only 1 point along `axis`.")
    # 检查 dx 是否是标量或者与替代输入形状相同
    if not (dx.ndim == 0 or dx.shape == alt_input_dx_shape):
        # 如果不符合条件，抛出 ValueError 异常，显示错误消息
        raise ValueError(message)
    # 使用 np.broadcast_to 将 dx 广播到最终的 dx 形状
    dx = np.broadcast_to(dx, final_dx_shape)
    # 将 dx 与指定轴交换
    dx = np.swapaxes(dx, axis, -1)
    # 调用 _cumulatively_sum_simpson_integrals 函数，使用等间隔辛普森积分方法计算结果
    res = _cumulatively_sum_simpson_integrals(
        y, dx, _cumulative_simpson_equal_intervals
    )

if initial is not None:
    # 如果提供了 initial，则确保 initial 是一个浮点数数组
    initial = _ensure_float_array(initial)
    # 替代输入的 initial 形状，只在指定轴上有 1 个点的形状
    alt_initial_input_shape = tupleset(original_shape, axis, 1)
    # 错误消息，用于报告如果提供了 `initial`，则它必须是标量或者与 `y` 相同形状但在指定轴上只有 1 个点
    message = ("If provided, `initial` must either be a scalar or have the "
               "same shape as `y` but with only 1 point along `axis`.")
    # 检查 initial 是否是标量或者与替代输入形状相同
    if not (initial.ndim == 0 or initial.shape == alt_initial_input_shape):
        # 如果不符合条件，抛出 ValueError 异常，显示错误消息
        raise ValueError(message)
    # 使用 np.broadcast_to 将 initial 广播到替代输入的 initial 形状
    initial = np.broadcast_to(initial, alt_initial_input_shape)
    # 将 initial 与指定轴交换
    initial = np.swapaxes(initial, axis, -1)

    # 将 initial 加到 res 中
    res += initial
    # 在指定轴上将 initial 和 res 连接起来
    res = np.concatenate((initial, res), axis=-1)

# 将 res 与指定轴交换
res = np.swapaxes(res, -1, axis)
# 返回计算结果
return res
# 使用 Romberg 积分方法对函数进行积分估算
def romb(y, dx=1.0, axis=-1, show=False):
    """
    Romberg integration using samples of a function.

    Parameters
    ----------
    y : array_like
        A vector of ``2**k + 1`` equally-spaced samples of a function.
    dx : float, optional
        The sample spacing. Default is 1.
    axis : int, optional
        The axis along which to integrate. Default is -1 (last axis).
    show : bool, optional
        When `y` is a single 1-D array, then if this argument is True
        print the table showing Richardson extrapolation from the
        samples. Default is False.

    Returns
    -------
    romb : ndarray
        The integrated result for `axis`.

    See Also
    --------
    quad : adaptive quadrature using QUADPACK
    fixed_quad : fixed-order Gaussian quadrature
    dblquad : double integrals
    tplquad : triple integrals
    simpson : integrators for sampled data
    cumulative_trapezoid : cumulative integration for sampled data

    Examples
    --------
    >>> from scipy import integrate
    >>> import numpy as np
    >>> x = np.arange(10, 14.25, 0.25)
    >>> y = np.arange(3, 12)

    >>> integrate.romb(y)
    56.0

    >>> y = np.sin(np.power(x, 2.5))
    >>> integrate.romb(y)
    -0.742561336672229

    >>> integrate.romb(y, show=True)
    Richardson Extrapolation Table for Romberg Integration
    ======================================================
    -0.81576
     4.63862  6.45674
    -1.10581 -3.02062 -3.65245
    -2.57379 -3.06311 -3.06595 -3.05664
    -1.34093 -0.92997 -0.78776 -0.75160 -0.74256
    ======================================================
    -0.742561336672229  # may vary

    """

    # 将输入的 y 转换为 NumPy 数组
    y = np.asarray(y)
    # 获取 y 的维度数
    nd = len(y.shape)
    # 获取 y 在指定轴上的样本数
    Nsamps = y.shape[axis]
    # 计算采样点之间的间隔数
    Ninterv = Nsamps - 1
    n = 1
    k = 0
    # 找到使 n 是 2 的幂的最小值
    while n < Ninterv:
        n <<= 1
        k += 1
    # 如果 n 不等于 Ninterv，抛出异常
    if n != Ninterv:
        raise ValueError("Number of samples must be one plus a "
                         "non-negative power of 2.")

    # 初始化 Richardson 外推表格
    R = {}
    slice_all = (slice(None),) * nd
    slice0 = tupleset(slice_all, axis, 0)
    slicem1 = tupleset(slice_all, axis, -1)
    h = Ninterv * np.asarray(dx, dtype=float)
    # 计算第一层 Richardson 外推
    R[(0, 0)] = (y[slice0] + y[slicem1]) / 2.0 * h
    slice_R = slice_all
    start = stop = step = Ninterv
    # 通过循环计算 Richardson 外推表格的每一层
    for i in range(1, k + 1):
        start >>= 1
        slice_R = tupleset(slice_R, axis, slice(start, stop, step))
        step >>= 1
        R[(i, 0)] = 0.5 * (R[(i - 1, 0)] + h * y[slice_R].sum(axis=axis))
        for j in range(1, i + 1):
            prev = R[(i, j - 1)]
            # 使用 Richardson 外推公式进行计算
            R[(i, j)] = prev + (prev - R[(i - 1, j - 1)]) / ((1 << (2 * j)) - 1)
        h /= 2.0

    # 返回最终的积分结果
    return R[(k, k)]
    # 如果 show 参数为真（非零、非空、非 None），则执行以下代码块
    if show:
        # 检查 R[(0, 0)] 是否为标量（单个值）
        if not np.isscalar(R[(0, 0)]):
            # 如果 R[(0, 0)] 不是标量，则打印错误信息，表明只支持单一数据集的积分
            print("*** Printing table only supported for integrals" +
                  " of a single data set.")
        else:
            # 如果 R[(0, 0)] 是标量，则继续执行以下代码块
            try:
                # 尝试获取显示精度（第一个元素）
                precis = show[0]
            except (TypeError, IndexError):
                # 如果获取失败，设置默认精度为 5
                precis = 5
            try:
                # 尝试获取显示宽度（第二个元素）
                width = show[1]
            except (TypeError, IndexError):
                # 如果获取失败，设置默认宽度为 8
                width = 8
            # 格式化字符串，用于打印每个数值
            formstr = "%%%d.%df" % (width, precis)

            # 打印标题
            title = "Richardson Extrapolation Table for Romberg Integration"
            print(title, "=" * len(title), sep="\n", end="\n")
            # 遍历 Richardson 矩阵的每一行
            for i in range(k+1):
                # 在每行中遍历列
                for j in range(i+1):
                    # 按格式打印 Richardson 矩阵中的每个元素
                    print(formstr % R[(i, j)], end=" ")
                # 换行，打印下一行
                print()
            # 打印分隔线，长度与标题相同
            print("=" * len(title))

    # 返回 Richardson 矩阵中第 (k, k) 位置的值，即积分结果的估计值
    return R[(k, k)]
# Coefficients for Newton-Cotes quadrature
#
# These are the points being used
#  to construct the local interpolating polynomial
#  a are the weights for Newton-Cotes integration
#  B is the error coefficient.
#  error in these coefficients grows as N gets larger.
#  or as samples are closer and closer together

# You can use maxima to find these rational coefficients
#  for equally spaced data using the commands
#  a(i,N) := (integrate(product(r-j,j,0,i-1) * product(r-j,j,i+1,N),r,0,N)
#             / ((N-i)! * i!) * (-1)^(N-i));
#  Be(N) := N^(N+2)/(N+2)! * (N/(N+3) - sum((i/N)^(N+2)*a(i,N),i,0,N));
#  Bo(N) := N^(N+1)/(N+1)! * (N/(N+2) - sum((i/N)^(N+1)*a(i,N),i,0,N));
#  B(N) := (if (mod(N,2)=0) then Be(N) else Bo(N));
#
# pre-computed for equally-spaced weights
#
# num_a, den_a, int_a, num_B, den_B = _builtincoeffs[N]
#
#  a = num_a*array(int_a)/den_a
#  B = num_B*1.0 / den_B
#
#  integrate(f(x),x,x_0,x_N) = dx*sum(a*f(x_i)) + B*(dx)^(2k+3) f^(2k+2)(x*)
#    where k = N // 2
#
_builtincoeffs = {
    1: (1,2,[1,1],-1,12),  # Coefficients for N=1
    2: (1,3,[1,4,1],-1,90),  # Coefficients for N=2
    3: (3,8,[1,3,3,1],-3,80),  # Coefficients for N=3
    4: (2,45,[7,32,12,32,7],-8,945),  # Coefficients for N=4
    5: (5,288,[19,75,50,50,75,19],-275,12096),  # Coefficients for N=5
    6: (1,140,[41,216,27,272,27,216,41],-9,1400),  # Coefficients for N=6
    7: (7,17280,[751,3577,1323,2989,2989,1323,3577,751],-8183,518400),  # Coefficients for N=7
    8: (4,14175,[989,5888,-928,10496,-4540,10496,-928,5888,989],
        -2368,467775),  # Coefficients for N=8
    9: (9,89600,[2857,15741,1080,19344,5778,5778,19344,1080,
                 15741,2857], -4671, 394240),  # Coefficients for N=9
    10: (5,299376,[16067,106300,-48525,272400,-260550,427368,
                   -260550,272400,-48525,106300,16067],
         -673175, 163459296),  # Coefficients for N=10
    11: (11,87091200,[2171465,13486539,-3237113, 25226685,-9595542,
                      15493566,15493566,-9595542,25226685,-3237113,
                      13486539,2171465], -2224234463, 237758976000),  # Coefficients for N=11
    12: (1, 5255250, [1364651,9903168,-7587864,35725120,-51491295,
                      87516288,-87797136,87516288,-51491295,35725120,
                      -7587864,9903168,1364651], -3012, 875875),  # Coefficients for N=12
    13: (13, 402361344000,[8181904909, 56280729661, -31268252574,
                           156074417954,-151659573325,206683437987,
                           -43111992612,-43111992612,206683437987,
                           -151659573325,156074417954,-31268252574,
                           56280729661,8181904909], -2639651053,
         344881152000),  # Coefficients for N=13
    14: (7, 2501928000, [90241897,710986864,-770720657,3501442784,
                         -6625093363,12630121616,-16802270373,19534438464,
                         -16802270373,12630121616,-6625093363,3501442784,
                         -770720657,710986864,90241897], -3740727473,
         1275983280000)  # Coefficients for N=14
    }


def newton_cotes(rn, equal=0):
    r"""
    Return weights and error coefficient for Newton-Cotes integration.

    Suppose we have (N+1) samples of f at the positions
    x_0, x_1, ..., x_N. Then an N-point Newton-Cotes formula for the
    """
    根据 Newton-Cotes 公式计算给定积分区间上的权重和误差系数。

    Parameters
    ----------
    rn : int
        等间距数据的整数阶数或样本相对位置，第一个样本为0，最后一个为N，其中 N+1 是 rn 的长度。
        N 是 Newton-Cotes 积分的阶数。
    equal : int, optional
        设置为 1 表示强制使用等间距数据。

    Returns
    -------
    an : ndarray
        应用于给定样本位置上函数的权重数组。
    B : float
        误差系数。

    Notes
    -----
    通常，Newton-Cotes 规则用于较小的积分区域，复合规则用于返回总积分。

    Examples
    --------
    计算在 [0, π] 上 sin(x) 的积分：

    >>> from scipy.integrate import newton_cotes
    >>> import numpy as np
    >>> def f(x):
    ...     return np.sin(x)
    >>> a = 0
    >>> b = np.pi
    >>> exact = 2
    >>> for N in [2, 4, 6, 8, 10]:
    ...     x = np.linspace(a, b, N + 1)
    ...     an, B = newton_cotes(N, 1)
    ...     dx = (b - a) / N
    ...     quad = dx * np.sum(an * f(x))
    ...     error = abs(quad - exact)
    ...     print('{:2d}  {:10.9f}  {:.5e}'.format(N, quad, error))
    ...
     2   2.094395102   9.43951e-02
     4   1.998570732   1.42927e-03
     6   2.000017814   1.78136e-05
     8   1.999999835   1.64725e-07
    10   2.000000001   1.14677e-09

    """
    try:
        N = len(rn)-1
        if equal:
            rn = np.arange(N+1)
        elif np.all(np.diff(rn) == 1):
            equal = 1
    except Exception:
        N = rn
        rn = np.arange(N+1)
        equal = 1

    if equal and N in _builtincoeffs:
        na, da, vi, nb, db = _builtincoeffs[N]
        an = na * np.array(vi, dtype=float) / da
        return an, float(nb)/db

    if (rn[0] != 0) or (rn[-1] != N):
        raise ValueError("The sample positions must start at 0"
                         " and end at N")
    
    # 计算归一化的样本位置
    yi = rn / float(N)
    
    # 计算与样本位置相关的 ti
    ti = 2 * yi - 1
    
    # 创建向量 nvec，长度为 N+1
    nvec = np.arange(N+1)
    
    # 创建矩阵 C，C 的每一列都是 ti 的幂
    C = ti ** nvec[:, np.newaxis]
    
    # 计算 C 的逆矩阵 Cinv
    Cinv = np.linalg.inv(C)
    
    # 改进结果的精度
    for i in range(2):
        Cinv = 2*Cinv - Cinv.dot(C).dot(Cinv)
    
    # 计算 ai 的系数向量
    vec = 2.0 / (nvec[::2]+1)
    ai = Cinv[:, ::2].dot(vec) * (N / 2.)
    
    if (N % 2 == 0) and equal:
        BN = N/(N+3.)
        power = N+2
    else:
        BN = N/(N+2.)
        power = N+1
    
    # 计算 BN 和 fac 的值
    BN = BN - np.dot(yi**power, ai)
    p1 = power+1
    fac = power*math.log(N) - gammaln(p1)
    fac = math.exp(fac)
    
    return ai, BN*fac
def _qmc_quad_iv(func, a, b, n_points, n_estimates, qrng, log):
    # lazy import to avoid issues with partially-initialized submodule
    if not hasattr(qmc_quad, 'qmc'):
        from scipy import stats
        qmc_quad.stats = stats
    else:
        stats = qmc_quad.stats

    if not callable(func):
        # 检查 func 是否可调用，如果不是则抛出类型错误
        message = "`func` must be callable."
        raise TypeError(message)

    # a, b will be modified, so copy. Oh well if it's copied twice.
    # 将 a 和 b 至少转化为一维数组并复制，保持广播兼容性
    a = np.atleast_1d(a).copy()
    b = np.atleast_1d(b).copy()
    a, b = np.broadcast_arrays(a, b)
    dim = a.shape[0]

    try:
        func((a + b) / 2)
    except Exception as e:
        # 检查 func 在积分范围内的表现，如果出错则抛出值错误
        message = ("`func` must evaluate the integrand at points within "
                   "the integration range; e.g. `func( (a + b) / 2)` "
                   "must return the integrand at the centroid of the "
                   "integration volume.")
        raise ValueError(message) from e

    try:
        func(np.array([a, b]).T)
        vfunc = func
    except Exception as e:
        # 尝试向 func 提供矢量化调用，如果失败则警告并提供备用函数
        message = ("Exception encountered when attempting vectorized call to "
                   f"`func`: {e}. For better performance, `func` should "
                   "accept two-dimensional array `x` with shape `(len(a), "
                   "n_points)` and return an array of the integrand value at "
                   "each of the `n_points.")
        warnings.warn(message, stacklevel=3)

        def vfunc(x):
            return np.apply_along_axis(func, axis=-1, arr=x)

    n_points_int = np.int64(n_points)
    if n_points != n_points_int:
        # 检查 n_points 是否为整数，如果不是则抛出类型错误
        message = "`n_points` must be an integer."
        raise TypeError(message)

    n_estimates_int = np.int64(n_estimates)
    if n_estimates != n_estimates_int:
        # 检查 n_estimates 是否为整数，如果不是则抛出类型错误
        message = "`n_estimates` must be an integer."
        raise TypeError(message)

    if qrng is None:
        # 如果 qrng 为 None，则使用默认的 Halton 序列
        qrng = stats.qmc.Halton(dim)
    elif not isinstance(qrng, stats.qmc.QMCEngine):
        # 检查 qrng 是否为 QMCEngine 的实例，如果不是则抛出类型错误
        message = "`qrng` must be an instance of scipy.stats.qmc.QMCEngine."
        raise TypeError(message)

    if qrng.d != a.shape[0]:
        # 检查 qrng 的维度是否与 a 的维度相匹配，如果不匹配则抛出值错误
        message = ("`qrng` must be initialized with dimensionality equal to "
                   "the number of variables in `a`, i.e., "
                   "`qrng.random().shape[-1]` must equal `a.shape[0]`.")
        raise ValueError(message)

    rng_seed = getattr(qrng, 'rng_seed', None)
    rng = stats._qmc.check_random_state(rng_seed)

    if log not in {True, False}:
        # 检查 log 是否为布尔值 True 或 False，如果不是则抛出类型错误
        message = "`log` must be boolean (`True` or `False`)."
        raise TypeError(message)

    # 返回包含所有参数和统计工具的元组，用于后续的积分计算
    return (vfunc, a, b, n_points_int, n_estimates_int, qrng, rng, log, stats)
    func : callable
        # 定义一个可调用的函数 `func`，接受一个参数 `x`，是一个数组，用于指定评估标量值积分被评估的点，并返回积分被评估点的值。
        # 为了效率，该函数应该是矢量化的，接受一个形状为 `(d, n_points)` 的数组，其中 `d` 是变量的数量（即函数定义域的维度），`n_points` 是积分点的数量，并返回形状为 `(n_points,)` 的数组，即每个积分点上的积分值。
        
    a, b : array-like
        # 一维数组，指定每个 `d` 变量的积分下限和上限。

    n_estimates, n_points : int, optional
        # 可选参数，默认 `n_estimates` 为 8，每个 `n_points` 为 1024 的统计独立 QMC 样本将由 `qrng` 生成。
        # `func` 将在总共 `n_points * n_estimates` 个点上进行评估。详见备注以获取细节信息。

    qrng : `~scipy.stats.qmc.QMCEngine`, optional
        # QMCEngine 的实例，用于抽样 QMC 点。
        # QMCEngine 必须初始化为与传递给 `func` 的 `x1, ..., xd` 变量数量 `d` 相对应的维度。
        # 提供的 QMCEngine 用于生成第一个积分估计值。
        # 如果 `n_estimates` 大于一，将从第一个 QMCEngine 中生成额外的 QMCEngine（启用混淆选项）。
        # 如果未提供 QMCEngine，则将使用默认的 `scipy.stats.qmc.Halton`，其维度由 `a` 的长度确定。

    log : boolean, default: False
        # 当设置为 True 时，`func` 返回积分被评估的对数，并且结果对象包含积分的对数。

    Returns
    -------
    result : object
        # 结果对象，具有以下属性：

        integral : float
            # 积分的估计值。

        standard_error :
            # 误差估计值。详见备注以获取解释信息。

    Notes
    -----
    # 对 QMC 样本的每个 `n_points` 点上的积分值用于生成积分的估计值。
    # 此估计值来自可能的积分估计值的一个群体，我们获得其值取决于评估积分的特定点。
    # 我们执行这个过程 `n_estimates` 次，每次在不同的混淆 QMC 点上评估积分，有效地从积分估计值的群体中抽取 i.i.d. 随机样本。
    # 这些积分估计值的样本均值 `m` 是真实积分值的无偏估计量，这些估计的均值标准误差 `s` 可用于生成
    """
    confidence intervals using the t distribution with ``n_estimates - 1``
    degrees of freedom. Perhaps counter-intuitively, increasing `n_points`
    while keeping the total number of function evaluation points
    ``n_points * n_estimates`` fixed tends to reduce the actual error, whereas
    increasing `n_estimates` tends to decrease the error estimate.

    Examples
    --------
    QMC quadrature is particularly useful for computing integrals in higher
    dimensions. An example integrand is the probability density function
    of a multivariate normal distribution.

    >>> import numpy as np
    >>> from scipy import stats
    >>> dim = 8
    >>> mean = np.zeros(dim)
    >>> cov = np.eye(dim)
    >>> def func(x):
    ...     # `multivariate_normal` expects the _last_ axis to correspond with
    ...     # the dimensionality of the space, so `x` must be transposed
    ...     return stats.multivariate_normal.pdf(x.T, mean, cov)

    To compute the integral over the unit hypercube:

    >>> from scipy.integrate import qmc_quad
    >>> a = np.zeros(dim)
    >>> b = np.ones(dim)
    >>> rng = np.random.default_rng()
    >>> qrng = stats.qmc.Halton(d=dim, seed=rng)
    >>> n_estimates = 8
    >>> res = qmc_quad(func, a, b, n_estimates=n_estimates, qrng=qrng)
    >>> res.integral, res.standard_error
    (0.00018429555666024108, 1.0389431116001344e-07)

    A two-sided, 99% confidence interval for the integral may be estimated
    as:

    >>> t = stats.t(df=n_estimates-1, loc=res.integral,
    ...             scale=res.standard_error)
    >>> t.interval(0.99)
    (0.0001839319802536469, 0.00018465913306683527)

    Indeed, the value reported by `scipy.stats.multivariate_normal` is
    within this range.

    >>> stats.multivariate_normal.cdf(b, mean, cov, lower_limit=a)
    0.00018430867675187443

    """
    # 调用 _qmc_quad_iv 函数计算积分估计及其相关参数
    args = _qmc_quad_iv(func, a, b, n_points, n_estimates, qrng, log)
    func, a, b, n_points, n_estimates, qrng, rng, log, stats = args

    # 定义函数 sum_product，计算积分估计和增量的加权和或对数和
    def sum_product(integrands, dA, log=False):
        if log:
            return logsumexp(integrands) + np.log(dA)
        else:
            return np.sum(integrands * dA)

    # 定义函数 mean，计算积分估计的均值或对数平均值
    def mean(estimates, log=False):
        if log:
            return logsumexp(estimates) - np.log(n_estimates)
        else:
            return np.mean(estimates)

    # 定义函数 std，计算积分估计的标准差或对数标准差
    def std(estimates, m=None, ddof=0, log=False):
        m = m or mean(estimates, log)
        if log:
            estimates, m = np.broadcast_arrays(estimates, m)
            temp = np.vstack((estimates, m + np.pi * 1j))
            diff = logsumexp(temp, axis=0)
            return np.real(0.5 * (logsumexp(2 * diff)
                                  - np.log(n_estimates - ddof)))
        else:
            return np.std(estimates, ddof=ddof)
    def sem(estimates, m=None, s=None, log=False):
        # Calculate the mean if not provided
        m = m or mean(estimates, log)
        # Calculate the standard deviation if not provided
        s = s or std(estimates, m, ddof=1, log=log)
        # Return standard error of the mean if log-transformed
        if log:
            return s - 0.5*np.log(n_estimates)
        else:
            return s / np.sqrt(n_estimates)

    # 如果任何一个下限等于上限，积分的符号将取决于积分限的顺序。通过确保
    # 下限确实比上限小，并手动设置积分的符号来修复这个问题。
    if np.any(a == b):
        # 如果有下限等于上限的情况，给出警告消息并返回积分结果为零
        message = ("A lower limit was equal to an upper limit, so the value "
                   "of the integral is zero by definition.")
        warnings.warn(message, stacklevel=2)
        return QMCQuadResult(-np.inf if log else 0, 0)

    # 交换下限和上限的位置，确保下限小于上限
    i_swap = b < a
    # 计算交换次数的奇偶性，用于确定积分的符号
    sign = (-1)**(i_swap.sum(axis=-1))  # odd # of swaps -> negative
    # 实际进行下限和上限的交换
    a[i_swap], b[i_swap] = b[i_swap], a[i_swap]

    # 计算积分区域的体积
    A = np.prod(b - a)
    # 计算积分的微元大小
    dA = A / n_points

    # 初始化积分估计结果数组
    estimates = np.zeros(n_estimates)
    # 使用独立的随机数生成器种子生成不相关的积分估计
    rngs = _rng_spawn(qrng.rng, n_estimates)
    for i in range(n_estimates):
        # 生成积分估计样本点
        sample = qrng.random(n_points)
        # 使用QMC方法生成在指定区间[a, b]上的样本点，同时转置以便用户可以方便地解包x为单独的变量
        x = stats.qmc.scale(sample, a, b).T  # (n_dim, n_points)
        # 计算积分被积函数的值
        integrands = func(x)
        # 计算积分估计值
        estimates[i] = sum_product(integrands, dA, log)

        # 为下一次迭代获取新的独立随机数生成器
        qrng = type(qrng)(seed=rngs[i], **qrng._init_quad)

    # 计算积分的均值
    integral = mean(estimates, log)
    # 计算积分估计值的标准误差
    standard_error = sem(estimates, m=integral, log=log)
    # 根据积分的符号修正积分结果
    integral = integral + np.pi*1j if (log and sign < 0) else integral*sign
    # 返回QMC积分结果和其标准误差
    return QMCQuadResult(integral, standard_error)
```