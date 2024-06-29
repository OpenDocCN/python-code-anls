# `.\numpy\numpy\lib\_function_base_impl.py`

```
# 导入内置模块和第三方库
import builtins  # 内置函数和异常
import collections.abc  # 抽象基类集合
import functools  # 函数工具
import re  # 正则表达式
import sys  # Python 系统相关的参数和函数
import warnings  # 警告控制

# 导入 NumPy 库及其子模块
import numpy as np  # 数组操作
import numpy._core.numeric as _nx  # NumPy 核心数值计算
from numpy._core import transpose, overrides  # 数组转置和函数覆盖
from numpy._core.numeric import (
    ones, zeros_like, arange, concatenate, array, asarray, asanyarray, empty,
    ndarray, take, dot, where, intp, integer, isscalar, absolute
    )  # NumPy 核心数值计算函数
from numpy._core.umath import (
    pi, add, arctan2, frompyfunc, cos, less_equal, sqrt, sin,
    mod, exp, not_equal, subtract, minimum
    )  # NumPy 核心数学函数
from numpy._core.fromnumeric import (
    ravel, nonzero, partition, mean, any, sum
    )  # NumPy 核心数组操作函数
from numpy._core.numerictypes import typecodes  # 数值类型码
from numpy.lib._twodim_base_impl import diag  # 二维数组基本操作实现
from numpy._core.multiarray import (
    _place, bincount, normalize_axis_index, _monotonicity,
    interp as compiled_interp, interp_complex as compiled_interp_complex
    )  # NumPy 多维数组操作
from numpy._core._multiarray_umath import _array_converter  # 数组转换器
from numpy._utils import set_module  # 设置模块信息

# 导入需要在模块中使用的函数
from numpy.lib._histograms_impl import histogram, histogramdd  # 直方图函数，用于统计频数

# 使用 functools.partial 创建一个部分应用的函数，用于数组函数分派
array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')

# __all__ 列表定义了模块中公开的函数和类的名称
__all__ = [
    'select', 'piecewise', 'trim_zeros', 'copy', 'iterable', 'percentile',
    'diff', 'gradient', 'angle', 'unwrap', 'sort_complex', 'flip',
    'rot90', 'extract', 'place', 'vectorize', 'asarray_chkfinite', 'average',
    'bincount', 'digitize', 'cov', 'corrcoef',
    'median', 'sinc', 'hamming', 'hanning', 'bartlett',
    'blackman', 'kaiser', 'trapezoid', 'trapz', 'i0',
    'meshgrid', 'delete', 'insert', 'append', 'interp',
    'quantile'
    ]

# _QuantileMethods 是一个字典，列出了所有支持的分位数/百分位数计算方法
#
# virtual_index 指的是在排序样本中找到百分位数的元素的索引
# 当样本中恰好有所需的百分位数时，virtual_index 是这个元素的整数索引
# 当所需的百分位数介于两个元素之间时，virtual_index 由整数部分（如 'i' 或 'left'）和小数部分（如 'g' 或 'gamma'）组成
#
# _QuantileMethods 中的每个方法都有两个属性
# get_virtual_index : Callable
#   用于计算 virtual_index 的函数
# fix_gamma : Callable
#   用于离散方法的函数，将索引强制到特定值
_QuantileMethods = dict(
    # --- HYNDMAN and FAN METHODS
    # 离散方法
    inverted_cdf=dict(
        get_virtual_index=lambda n, quantiles: _inverted_cdf(n, quantiles),
        fix_gamma=None,  # 不应该被调用
    ),
    averaged_inverted_cdf=dict(
        get_virtual_index=lambda n, quantiles: (n * quantiles) - 1,
        fix_gamma=lambda gamma, _: _get_gamma_mask(
            shape=gamma.shape,
            default_value=1.,
            conditioned_value=0.5,
            where=gamma == 0),
    ),
    # 创建一个包含不同方法的字典，每个方法都有两个键值对：'get_virtual_index' 和 'fix_gamma'。
    closest_observation=dict(
        # Lambda函数，用于获取虚拟索引，调用 _closest_observation 函数
        get_virtual_index=lambda n, quantiles: _closest_observation(n, quantiles),
        fix_gamma=None,  # 不应该被调用的占位符
    ),
    
    # 连续方法：interpolated_inverted_cdf 方法
    interpolated_inverted_cdf=dict(
        # Lambda函数，用于获取虚拟索引，调用 _compute_virtual_index 函数
        get_virtual_index=lambda n, quantiles: _compute_virtual_index(n, quantiles, 0, 1),
        # Lambda函数，用于修正 gamma 参数，直接返回 gamma 参数本身
        fix_gamma=lambda gamma, _: gamma,
    ),
    
    # hazen 方法
    hazen=dict(
        # Lambda函数，用于获取虚拟索引，调用 _compute_virtual_index 函数
        get_virtual_index=lambda n, quantiles: _compute_virtual_index(n, quantiles, 0.5, 0.5),
        # Lambda函数，用于修正 gamma 参数，直接返回 gamma 参数本身
        fix_gamma=lambda gamma, _: gamma,
    ),
    
    # weibull 方法
    weibull=dict(
        # Lambda函数，用于获取虚拟索引，调用 _compute_virtual_index 函数
        get_virtual_index=lambda n, quantiles: _compute_virtual_index(n, quantiles, 0, 0),
        # Lambda函数，用于修正 gamma 参数，直接返回 gamma 参数本身
        fix_gamma=lambda gamma, _: gamma,
    ),
    
    # linear 方法
    linear=dict(
        # Lambda函数，用于获取虚拟索引，通过线性计算得出
        get_virtual_index=lambda n, quantiles: (n - 1) * quantiles,
        # Lambda函数，用于修正 gamma 参数，直接返回 gamma 参数本身
        fix_gamma=lambda gamma, _: gamma,
    ),
    
    # median_unbiased 方法
    median_unbiased=dict(
        # Lambda函数，用于获取虚拟索引，调用 _compute_virtual_index 函数
        get_virtual_index=lambda n, quantiles: _compute_virtual_index(n, quantiles, 1 / 3.0, 1 / 3.0),
        # Lambda函数，用于修正 gamma 参数，直接返回 gamma 参数本身
        fix_gamma=lambda gamma, _: gamma,
    ),
    
    # normal_unbiased 方法
    normal_unbiased=dict(
        # Lambda函数，用于获取虚拟索引，调用 _compute_virtual_index 函数
        get_virtual_index=lambda n, quantiles: _compute_virtual_index(n, quantiles, 3 / 8.0, 3 / 8.0),
        # Lambda函数，用于修正 gamma 参数，直接返回 gamma 参数本身
        fix_gamma=lambda gamma, _: gamma,
    ),
    
    # lower 方法
    lower=dict(
        # Lambda函数，用于获取虚拟索引，通过向下取整操作
        get_virtual_index=lambda n, quantiles: np.floor((n - 1) * quantiles).astype(np.intp),
        fix_gamma=None,  # 不应该被调用的占位符，索引的数据类型为整数
    ),
    
    # higher 方法
    higher=dict(
        # Lambda函数，用于获取虚拟索引，通过向上取整操作
        get_virtual_index=lambda n, quantiles: np.ceil((n - 1) * quantiles).astype(np.intp),
        fix_gamma=None,  # 不应该被调用的占位符，索引的数据类型为整数
    ),
    
    # midpoint 方法
    midpoint=dict(
        # Lambda函数，用于获取虚拟索引，通过取两个取整操作的平均值
        get_virtual_index=lambda n, quantiles: 0.5 * (
                np.floor((n - 1) * quantiles)
                + np.ceil((n - 1) * quantiles)),
        # Lambda函数，用于修正 gamma 参数，根据条件对 gamma 进行掩码处理
        fix_gamma=lambda gamma, index: _get_gamma_mask(
            shape=gamma.shape,
            default_value=0.5,
            conditioned_value=0.,
            where=index % 1 == 0
        ),
    ),
    
    # nearest 方法
    nearest=dict(
        # Lambda函数，用于获取虚拟索引，通过四舍五入取整操作
        get_virtual_index=lambda n, quantiles: np.around((n - 1) * quantiles).astype(np.intp),
        fix_gamma=None,  # 不应该被调用的占位符，索引的数据类型为整数
    )
# 定义一个分发函数 _rot90_dispatcher，接收 m, k 和 axes 参数，返回一个包含 m 的元组
def _rot90_dispatcher(m, k=None, axes=None):
    return (m,)

# 使用 array_function_dispatch 装饰器，将 rot90 函数与 _rot90_dispatcher 分发函数关联
@array_function_dispatch(_rot90_dispatcher)
# 定义 rot90 函数，用于将数组 m 按 axes 指定的平面旋转 90 度
def rot90(m, k=1, axes=(0, 1)):
    """
    Rotate an array by 90 degrees in the plane specified by axes.

    Rotation direction is from the first towards the second axis.
    This means for a 2D array with the default `k` and `axes`, the
    rotation will be counterclockwise.

    Parameters
    ----------
    m : array_like
        Array of two or more dimensions.
    k : integer
        Number of times the array is rotated by 90 degrees.
    axes : (2,) array_like
        The array is rotated in the plane defined by the axes.
        Axes must be different.

        .. versionadded:: 1.12.0

    Returns
    -------
    y : ndarray
        A rotated view of `m`.

    See Also
    --------
    flip : Reverse the order of elements in an array along the given axis.
    fliplr : Flip an array horizontally.
    flipud : Flip an array vertically.

    Notes
    -----
    ``rot90(m, k=1, axes=(1,0))``  is the reverse of
    ``rot90(m, k=1, axes=(0,1))``

    ``rot90(m, k=1, axes=(1,0))`` is equivalent to
    ``rot90(m, k=-1, axes=(0,1))``

    Examples
    --------
    >>> m = np.array([[1,2],[3,4]], int)
    >>> m
    array([[1, 2],
           [3, 4]])
    >>> np.rot90(m)
    array([[2, 4],
           [1, 3]])
    >>> np.rot90(m, 2)
    array([[4, 3],
           [2, 1]])
    >>> m = np.arange(8).reshape((2,2,2))
    >>> np.rot90(m, 1, (1,2))
    array([[[1, 3],
            [0, 2]],
           [[5, 7],
            [4, 6]]])

    """
    # 将 axes 转换为元组形式
    axes = tuple(axes)
    # 检查 axes 的长度是否为 2，若不是则抛出 ValueError 异常
    if len(axes) != 2:
        raise ValueError("len(axes) must be 2.")

    # 将 m 转换为 ndarray 类型
    m = asanyarray(m)

    # 检查 axes[0] 和 axes[1] 是否相等，或者它们的绝对差是否等于 m 的维度数，若是则抛出 ValueError 异常
    if axes[0] == axes[1] or absolute(axes[0] - axes[1]) == m.ndim:
        raise ValueError("Axes must be different.")

    # 检查 axes[0] 和 axes[1] 是否超出 m 的维度范围，若是则抛出 ValueError 异常
    if (axes[0] >= m.ndim or axes[0] < -m.ndim
        or axes[1] >= m.ndim or axes[1] < -m.ndim):
        raise ValueError("Axes={} out of range for array of ndim={}."
            .format(axes, m.ndim))

    # 将 k 取模 4，确保在 0 到 3 的范围内
    k %= 4

    # 若 k 等于 0，则直接返回 m 的完整切片
    if k == 0:
        return m[:]
    # 若 k 等于 2，则先沿 axes[0] 反转 m，再沿 axes[1] 反转 m，然后返回结果
    if k == 2:
        return flip(flip(m, axes[0]), axes[1])

    # 创建 axes_list，包含 0 到 m.ndim-1 的所有索引值
    axes_list = arange(0, m.ndim)
    # 交换 axes_list 中索引 axes[0] 和 axes[1] 的值
    (axes_list[axes[0]], axes_list[axes[1]]) = (axes_list[axes[1]],
                                                axes_list[axes[0]])

    # 若 k 等于 1，则先沿 axes[1] 反转 m，再按 axes_list 重新排列 m，然后返回结果
    if k == 1:
        return transpose(flip(m, axes[1]), axes_list)
    else:
        # 否则 k 必定等于 3，先按 axes_list 重新排列 m，再沿 axes[1] 反转 m，然后返回结果
        return flip(transpose(m, axes_list), axes[1])


# 定义一个分发函数 _flip_dispatcher，接收 m 和 axis 参数，返回一个包含 m 的元组
def _flip_dispatcher(m, axis=None):
    return (m,)

# 使用 array_function_dispatch 装饰器，将 flip 函数与 _flip_dispatcher 分发函数关联
@array_function_dispatch(_flip_dispatcher)
# 定义 flip 函数，用于沿给定轴反转数组中元素的顺序
def flip(m, axis=None):
    """
    Reverse the order of elements in an array along the given axis.

    The shape of the array is preserved, but the elements are reordered.

    .. versionadded:: 1.12.0

    Parameters
    ----------
    m : array_like
        Input array.

    """
    # 函数主体中的代码在上面的注释中已经描述清楚，因此不再重复注释
    axis : None or int or tuple of ints, optional
         Axis or axes along which to flip over. The default,
         axis=None, will flip over all of the axes of the input array.
         If axis is negative it counts from the last to the first axis.

         If axis is a tuple of ints, flipping is performed on all of the axes
         specified in the tuple.

         .. versionchanged:: 1.15.0
            None and tuples of axes are supported

    Returns
    -------
    out : array_like
        A view of `m` with the entries of axis reversed.  Since a view is
        returned, this operation is done in constant time.

    See Also
    --------
    flipud : Flip an array vertically (axis=0).
    fliplr : Flip an array horizontally (axis=1).

    Notes
    -----
    flip(m, 0) is equivalent to flipud(m).

    flip(m, 1) is equivalent to fliplr(m).

    flip(m, n) corresponds to ``m[...,::-1,...]`` with ``::-1`` at position n.

    flip(m) corresponds to ``m[::-1,::-1,...,::-1]`` with ``::-1`` at all
    positions.

    flip(m, (0, 1)) corresponds to ``m[::-1,::-1,...]`` with ``::-1`` at
    position 0 and position 1.

    Examples
    --------
    >>> A = np.arange(8).reshape((2,2,2))
    >>> A
    array([[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]])
    >>> np.flip(A, 0)
    array([[[4, 5],
            [6, 7]],
           [[0, 1],
            [2, 3]]])
    >>> np.flip(A, 1)
    array([[[2, 3],
            [0, 1]],
           [[6, 7],
            [4, 5]]])
    >>> np.flip(A)
    array([[[7, 6],
            [5, 4]],
           [[3, 2],
            [1, 0]]])
    >>> np.flip(A, (0, 2))
    array([[[5, 4],
            [7, 6]],
           [[1, 0],
            [3, 2]]])
    >>> rng = np.random.default_rng()
    >>> A = rng.normal(size=(3,4,5))
    >>> np.all(np.flip(A,2) == A[:,:,::-1,...])
    True
    """

    # 如果输入的 m 对象没有 ndim 属性，则将其转换为 ndarray 对象
    if not hasattr(m, 'ndim'):
        m = asarray(m)
    
    # 如果 axis 参数为 None，则创建一个索引器 indexer，用于反转所有轴的顺序
    if axis is None:
        indexer = (np.s_[::-1],) * m.ndim
    else:
        # 否则，将 axis 转换成标准化的轴元组
        axis = _nx.normalize_axis_tuple(axis, m.ndim)
        # 初始化一个包含全切片 np.s_[:] 的索引器列表
        indexer = [np.s_[:]] * m.ndim
        # 对于每个需要反转的轴，将对应的索引器改为反转切片 np.s_[::-1]
        for ax in axis:
            indexer[ax] = np.s_[::-1]
        # 将列表转换为元组，作为最终的索引器
        indexer = tuple(indexer)
    
    # 返回根据索引器 indexer 得到的 m 的视图
    return m[indexer]
# 设置模块为 'numpy'，这是一个装饰器函数，用于将当前模块设置为 'numpy'
@set_module('numpy')
# 定义函数 iterable，用于检查对象是否可以迭代
def iterable(y):
    """
    Check whether or not an object can be iterated over.

    Parameters
    ----------
    y : object
      Input object.

    Returns
    -------
    b : bool
      Return ``True`` if the object has an iterator method or is a
      sequence and ``False`` otherwise.

    Examples
    --------
    >>> np.iterable([1, 2, 3])
    True
    >>> np.iterable(2)
    False

    Notes
    -----
    In most cases, the results of ``np.iterable(obj)`` are consistent with
    ``isinstance(obj, collections.abc.Iterable)``. One notable exception is
    the treatment of 0-dimensional arrays::

        >>> from collections.abc import Iterable
        >>> a = np.array(1.0)  # 0-dimensional numpy array
        >>> isinstance(a, Iterable)
        True
        >>> np.iterable(a)
        False

    """
    try:
        # 尝试迭代对象 y，如果成功则返回 True
        iter(y)
    except TypeError:
        # 如果迭代失败（TypeError），则返回 False
        return False
    # 默认返回 True，表示对象 y 可迭代
    return True


# 定义函数 _weights_are_valid，用于验证权重数组的有效性
def _weights_are_valid(weights, a, axis):
    """Validate weights array.
    
    We assume, weights is not None.
    """
    # 将权重数组转换为 numpy 数组
    wgt = np.asanyarray(weights)

    # 进行一些健全性检查
    if a.shape != wgt.shape:
        if axis is None:
            # 如果未指定轴（axis），则抛出 TypeError
            raise TypeError(
                "Axis must be specified when shapes of a and weights "
                "differ.")
        if wgt.shape != tuple(a.shape[ax] for ax in axis):
            # 如果权重数组的形状与指定轴（axis）的数组形状不一致，则抛出 ValueError
            raise ValueError(
                "Shape of weights must be consistent with "
                "shape of a along specified axis.")

        # 对权重数组进行广播以匹配指定的轴（axis）
        wgt = wgt.transpose(np.argsort(axis))
        wgt = wgt.reshape(tuple((s if ax in axis else 1)
                                for ax, s in enumerate(a.shape)))
    # 返回验证后的权重数组 wgt
    return wgt


# 定义函数 _average_dispatcher，用作 average 函数的分派器
def _average_dispatcher(a, axis=None, weights=None, returned=None, *,
                        keepdims=None):
    # 返回输入数组 a 和权重 weights
    return (a, weights)


# 使用装饰器 array_function_dispatch 将 _average_dispatcher 设置为 average 函数的分派器
@array_function_dispatch(_average_dispatcher)
# 定义函数 average，计算沿指定轴（axis）的加权平均值
def average(a, axis=None, weights=None, returned=False, *,
            keepdims=np._NoValue):
    """
    Compute the weighted average along the specified axis.

    Parameters
    ----------
    a : array_like
        Array containing data to be averaged. If `a` is not an array, a
        conversion is attempted.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to average `a`.  The default,
        `axis=None`, will average over all of the elements of the input array.
        If axis is negative it counts from the last to the first axis.

        .. versionadded:: 1.7.0

        If axis is a tuple of ints, averaging is performed on all of the axes
        specified in the tuple instead of a single axis or all the axes as
        before.
    weights : array_like, optional
        # 可选参数，用于指定每个 `a` 中值的权重数组。每个 `a` 中的值根据其权重对平均值的贡献不同。
        # 如果没有指定轴，则权重数组必须与 `a` 具有相同的形状；否则，权重必须在指定轴上具有一致的维度和形状。
        # 如果 `weights=None`，则假定所有 `a` 中的数据都具有权重为1。
        # 计算公式为：
        #     avg = sum(a * weights) / sum(weights)
        # 其中求和涵盖所有包括的元素。
        # `weights` 的唯一约束是 `sum(weights)` 不能为0。

    returned : bool, optional
        # 默认为 `False`。如果为 `True`，则返回一个元组 (`average`, `sum_of_weights`)，否则仅返回平均值。
        # 如果 `weights=None`，则 `sum_of_weights` 相当于平均值计算涉及的元素数量。

    keepdims : bool, optional
        # 如果设置为 `True`，则减少的轴将作为大小为1的维度保留在结果中。
        # 使用此选项时，结果将正确地对原始 `a` 进行广播。
        # 注意：`keepdims` 在 `numpy.matrix` 或其他不支持 `keepdims` 的类的实例中不起作用。
        # 
        # .. versionadded:: 1.23.0
        

    Returns
    -------
    retval, [sum_of_weights] : array_type or double
        # 返回沿指定轴的平均值。当 `returned` 为 `True` 时，返回一个元组，第一个元素为平均值，第二个元素为权重的总和。
        # `sum_of_weights` 的类型与 `retval` 相同。结果的数据类型遵循一般模式。
        # 如果 `weights` 为 None，则结果的数据类型将与 `a` 的类型相同，或者如果 `a` 是整数，则为 `float64`。
        # 否则，如果 `weights` 不为 None 且 `a` 是非整数，则结果类型将是能够表示 `a` 和 `weights` 值的最低精度类型。
        # 如果 `a` 是整数，则仍然适用前述规则，但结果的数据类型至少为 `float64`。

    Raises
    ------
    ZeroDivisionError
        # 当指定轴上所有权重为零时抛出。查看 `numpy.ma.average` 获取一个对这种类型错误健壮的版本。
    TypeError
        # 当 `weights` 与 `a` 的形状不相同时，并且 `axis=None` 时抛出。
    ValueError
        # 当 `weights` 在指定轴上的维度和形状与 `a` 不一致时抛出。

    See Also
    --------
    mean

    ma.average : 带有掩码数组的平均值 - 如果您的数据包含 "缺失" 值，则非常有用。
    numpy.result_type : 返回应用于参数的 numpy 类型推广规则的类型。

    Examples
    --------
    >>> data = np.arange(1, 5)
    >>> data
    array([1, 2, 3, 4])
    >>> np.average(data)
    2.5
    >>> np.average(np.arange(1, 11), weights=np.arange(10, 0, -1))
    4.0

    >>> data = np.arange(6).reshape((3, 2))
    >>> data
    array([[0, 1],
           [2, 3],
           [4, 5]])
    >>> np.average(data, axis=1, weights=[1./4, 3./4])
    array([0.75, 2.75, 4.75])
    >>> np.average(data, weights=[1./4, 3./4])
    Traceback (most recent call last):
        ...
    TypeError: Axis must be specified when shapes of a and weights differ.

    With ``keepdims=True``, the following result has shape (3, 1).

    >>> np.average(data, axis=1, keepdims=True)
    array([[0.5],
           [2.5],
           [4.5]])

    >>> data = np.arange(8).reshape((2, 2, 2))
    >>> data
    array([[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]])
    >>> np.average(data, axis=(0, 1), weights=[[1./4, 3./4], [1., 1./2]])
    array([3.4, 4.4])
    >>> np.average(data, axis=0, weights=[[1./4, 3./4], [1., 1./2]])
    Traceback (most recent call last):
        ...
    ValueError: Shape of weights must be consistent
    with shape of a along specified axis.
    """
    # 将输入数组转换为任意数组
    a = np.asanyarray(a)

    # 如果指定了轴，则对轴进行标准化处理
    if axis is not None:
        axis = _nx.normalize_axis_tuple(axis, a.ndim, argname="axis")

    # 如果没有传递 keepdims 参数，则设置为默认空值
    if keepdims is np._NoValue:
        keepdims_kw = {}
    else:
        keepdims_kw = {'keepdims': keepdims}

    # 如果未提供权重，则计算平均值
    if weights is None:
        avg = a.mean(axis, **keepdims_kw)
        avg_as_array = np.asanyarray(avg)
        scl = avg_as_array.dtype.type(a.size/avg_as_array.size)
    else:
        # 验证权重的有效性
        wgt = _weights_are_valid(weights=weights, a=a, axis=axis)

        # 根据数组和权重数据类型选择结果数据类型
        if issubclass(a.dtype.type, (np.integer, np.bool)):
            result_dtype = np.result_type(a.dtype, wgt.dtype, 'f8')
        else:
            result_dtype = np.result_type(a.dtype, wgt.dtype)

        # 计算权重的总和
        scl = wgt.sum(axis=axis, dtype=result_dtype, **keepdims_kw)
        if np.any(scl == 0.0):
            raise ZeroDivisionError(
                "Weights sum to zero, can't be normalized")

        # 计算加权平均值
        avg = avg_as_array = np.multiply(a, wgt,
                          dtype=result_dtype).sum(axis, **keepdims_kw) / scl

    # 如果需要返回权重，根据情况返回平均值和权重总和，否则仅返回平均值
    if returned:
        if scl.shape != avg_as_array.shape:
            scl = np.broadcast_to(scl, avg_as_array.shape).copy()
        return avg, scl
    else:
        return avg
# 设置模块名为'numpy'，用于标识本函数与numpy库的关联性
@set_module('numpy')
# 将输入数据转换为数组，同时检查是否包含NaN或Inf
def asarray_chkfinite(a, dtype=None, order=None):
    """Convert the input to an array, checking for NaNs or Infs.

    Parameters
    ----------
    a : array_like
        Input data, in any form that can be converted to an array.  This
        includes lists, lists of tuples, tuples, tuples of tuples, tuples
        of lists and ndarrays.  Success requires no NaNs or Infs.
    dtype : data-type, optional
        By default, the data-type is inferred from the input data.
    order : {'C', 'F', 'A', 'K'}, optional
        Memory layout.  'A' and 'K' depend on the order of input array a.
        'C' row-major (C-style),
        'F' column-major (Fortran-style) memory representation.
        'A' (any) means 'F' if `a` is Fortran contiguous, 'C' otherwise
        'K' (keep) preserve input order
        Defaults to 'C'.

    Returns
    -------
    out : ndarray
        Array interpretation of `a`.  No copy is performed if the input
        is already an ndarray.  If `a` is a subclass of ndarray, a base
        class ndarray is returned.

    Raises
    ------
    ValueError
        Raises ValueError if `a` contains NaN (Not a Number) or Inf (Infinity).

    See Also
    --------
    asarray : Create and array.
    asanyarray : Similar function which passes through subclasses.
    ascontiguousarray : Convert input to a contiguous array.
    asfortranarray : Convert input to an ndarray with column-major
                     memory order.
    fromiter : Create an array from an iterator.
    fromfunction : Construct an array by executing a function on grid
                   positions.

    Examples
    --------
    Convert a list into an array.  If all elements are finite
    ``asarray_chkfinite`` is identical to ``asarray``.

    >>> a = [1, 2]
    >>> np.asarray_chkfinite(a, dtype=float)
    array([1., 2.])

    Raises ValueError if array_like contains Nans or Infs.

    >>> a = [1, 2, np.inf]
    >>> try:
    ...     np.asarray_chkfinite(a)
    ... except ValueError:
    ...     print('ValueError')
    ...
    ValueError

    """
    # 将输入数据转换为数组a，根据指定的dtype和order
    a = asarray(a, dtype=dtype, order=order)
    # 如果数组a的数据类型是浮点数且包含无穷大或NaN，则抛出ValueError异常
    if a.dtype.char in typecodes['AllFloat'] and not np.isfinite(a).all():
        raise ValueError(
            "array must not contain infs or NaNs")
    # 返回转换后的数组a
    return a


# 创建一个生成器函数_piecewise_dispatcher，用于分派函数_piecewise的输入参数x
def _piecewise_dispatcher(x, condlist, funclist, *args, **kw):
    # 生成器函数的第一个值是x
    yield x
    # 支持condlist是可迭代对象的未记录的行为，允许标量
    if np.iterable(condlist):
        # 生成器函数继续产出condlist中的元素
        yield from condlist


# 注册_piecewise_dispatcher为array_function_dispatch的分派函数
@array_function_dispatch(_piecewise_dispatcher)
# 定义piecewise函数，用于评估分段定义的函数
def piecewise(x, condlist, funclist, *args, **kw):
    """
    Evaluate a piecewise-defined function.

    Given a set of conditions and corresponding functions, evaluate each
    function on the input data wherever its condition is true.

    Parameters
    ----------
    x : ndarray or scalar
        The input domain.

    """
    x = asanyarray(x)
    # 将输入参数 x 转换为一个 NumPy 数组，以确保能够处理数组输入

    n2 = len(funclist)
    # 获取 funclist 列表的长度，即条件和对应函数的数量

    # undocumented: single condition is promoted to a list of one condition
    # 如果 condlist 只包含一个条件而不是列表，则将其提升为包含一个条件的列表
    # 检查条件列表是否为标量或者第一个元素不是列表或数组且 x 不是零维时，
    # 将条件列表包装为一个单元素列表
    if isscalar(condlist) or (
            not isinstance(condlist[0], (list, ndarray)) and x.ndim != 0):
        condlist = [condlist]

    # 将条件列表转换为布尔类型的 NumPy 数组
    condlist = asarray(condlist, dtype=bool)
    # 获取条件列表的长度
    n = len(condlist)

    # 计算 "否则" 条件
    if n == n2 - 1:
        condelse = ~np.any(condlist, axis=0, keepdims=True)
        # 将 "否则" 条件与原条件列表拼接在一起
        condlist = np.concatenate([condlist, condelse], axis=0)
        n += 1
    # 如果条件列表长度与 n2 不符合，抛出 ValueError 异常
    elif n != n2:
        raise ValueError(
            "with {} condition(s), either {} or {} functions are expected"
            .format(n, n, n+1)
        )

    # 初始化输出数组 y，形状与 x 相同，但值全为零
    y = zeros_like(x)
    # 遍历条件列表和函数列表，根据条件应用相应的函数或值到 y 中
    for cond, func in zip(condlist, funclist):
        # 如果函数不可调用，直接将其值赋给满足条件的 y 元素
        if not isinstance(func, collections.abc.Callable):
            y[cond] = func
        else:
            # 否则，从 x 中取出满足条件的值，并将其与额外的参数一起传递给函数 func
            vals = x[cond]
            if vals.size > 0:
                y[cond] = func(vals, *args, **kw)

    # 返回计算后的输出数组 y
    return y
# 定义一个生成器函数，用于选择调度条件列表和选择列表
def _select_dispatcher(condlist, choicelist, default=None):
    yield from condlist  # 生成器函数中，先生成条件列表的内容
    yield from choicelist  # 然后生成选择列表的内容


# 使用装饰器将_select_dispatcher函数与select函数关联，作为数组函数的分派器
@array_function_dispatch(_select_dispatcher)
def select(condlist, choicelist, default=0):
    """
    Return an array drawn from elements in choicelist, depending on conditions.

    Parameters
    ----------
    condlist : list of bool ndarrays
        The list of conditions which determine from which array in `choicelist`
        the output elements are taken. When multiple conditions are satisfied,
        the first one encountered in `condlist` is used.
    choicelist : list of ndarrays
        The list of arrays from which the output elements are taken. It has
        to be of the same length as `condlist`.
    default : scalar, optional
        The element inserted in `output` when all conditions evaluate to False.

    Returns
    -------
    output : ndarray
        The output at position m is the m-th element of the array in
        `choicelist` where the m-th element of the corresponding array in
        `condlist` is True.

    See Also
    --------
    where : Return elements from one of two arrays depending on condition.
    take, choose, compress, diag, diagonal

    Examples
    --------
    Beginning with an array of integers from 0 to 5 (inclusive),
    elements less than ``3`` are negated, elements greater than ``3``
    are squared, and elements not meeting either of these conditions
    (exactly ``3``) are replaced with a `default` value of ``42``.

    >>> x = np.arange(6)
    >>> condlist = [x<3, x>3]
    >>> choicelist = [x, x**2]
    >>> np.select(condlist, choicelist, 42)
    array([ 0,  1,  2, 42, 16, 25])

    When multiple conditions are satisfied, the first one encountered in
    `condlist` is used.

    >>> condlist = [x<=4, x>3]
    >>> choicelist = [x, x**2]
    >>> np.select(condlist, choicelist, 55)
    array([ 0,  1,  2,  3,  4, 25])

    """
    # 检查条件列表和选择列表的长度是否相同，如果不同则抛出 ValueError 异常
    if len(condlist) != len(choicelist):
        raise ValueError(
            'list of cases must be same length as list of conditions')

    # 如果条件列表为空，则抛出 ValueError 异常，因为无法进行选择
    if len(condlist) == 0:
        raise ValueError("select with an empty condition list is not possible")

    # TODO: 这段代码手动处理Python的整数、浮点数和复数类型，以获取正确的 `result_type`，随着 NEP 50 的实施，可能会有更好的替代方法
    # 将 choicelist 中的元素转换为 numpy 数组，如果元素类型不是 int、float 或 complex
    choicelist = [
        choice if type(choice) in (int, float, complex) else np.asarray(choice)
        for choice in choicelist]
    # 将 default 转换为 numpy 数组，如果其类型不是 int、float 或 complex
    choicelist.append(default if type(default) in (int, float, complex)
                      else np.asarray(default))

    # 尝试确定 choicelist 中所有数组的共同 dtype
    try:
        dtype = np.result_type(*choicelist)
    except TypeError as e:
        msg = f'Choicelist and default value do not have a common dtype: {e}'
        raise TypeError(msg) from None
    # 将条件列表转换为数组，并广播条件和选择列表，
    # 以便得到结果需要的形状。分开处理优化了当所有选择都是标量的情况。
    condlist = np.broadcast_arrays(*condlist)
    choicelist = np.broadcast_arrays(*choicelist)

    # 检查条件数组是否是布尔型的 ndarray 或标量布尔值，否则中止并抛出异常。
    for i, cond in enumerate(condlist):
        if cond.dtype.type is not np.bool:
            raise TypeError(
                'invalid entry {} in condlist: should be boolean ndarray'.format(i))

    if choicelist[0].ndim == 0:
        # 如果第一个选择是标量，通常情况，避免调用。
        result_shape = condlist[0].shape
    else:
        # 计算结果的形状，通过广播第一个条件和第一个选择来确定。
        result_shape = np.broadcast_arrays(condlist[0], choicelist[0])[0].shape

    # 创建结果数组，用最后一个选择值填充，指定 dtype。
    result = np.full(result_shape, choicelist[-1], dtype)

    # 使用 np.copyto 将每个选择列表数组按照对应的条件列表作为布尔掩码，
    # 倒序处理以确保第一个选择优先生效。
    choicelist = choicelist[-2::-1]
    condlist = condlist[::-1]
    for choice, cond in zip(choicelist, condlist):
        np.copyto(result, choice, where=cond)

    # 返回最终结果数组。
    return result
# 定义一个私有函数 _copy_dispatcher，用于分发参数 a，order 和 subok
def _copy_dispatcher(a, order=None, subok=None):
    # 返回一个包含参数 a 的元组
    return (a,)

# 使用装饰器 array_function_dispatch 将 copy 函数注册为 _copy_dispatcher 的分发函数
@array_function_dispatch(_copy_dispatcher)
def copy(a, order='K', subok=False):
    """
    Return an array copy of the given object.

    Parameters
    ----------
    a : array_like
        Input data.
    order : {'C', 'F', 'A', 'K'}, optional
        Controls the memory layout of the copy. 'C' means C-order,
        'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
        'C' otherwise. 'K' means match the layout of `a` as closely
        as possible. (Note that this function and :meth:`ndarray.copy` are very
        similar, but have different default values for their order=
        arguments.)
    subok : bool, optional
        If True, then sub-classes will be passed-through, otherwise the
        returned array will be forced to be a base-class array (defaults to False).

        .. versionadded:: 1.19.0

    Returns
    -------
    arr : ndarray
        Array interpretation of `a`.

    See Also
    --------
    ndarray.copy : Preferred method for creating an array copy

    Notes
    -----
    This is equivalent to:

    >>> np.array(a, copy=True)  #doctest: +SKIP

    The copy made of the data is shallow, i.e., for arrays with object dtype,
    the new array will point to the same objects.
    See Examples from `ndarray.copy`.

    Examples
    --------
    Create an array x, with a reference y and a copy z:

    >>> x = np.array([1, 2, 3])
    >>> y = x
    >>> z = np.copy(x)

    Note that, when we modify x, y changes, but not z:

    >>> x[0] = 10
    >>> x[0] == y[0]
    True
    >>> x[0] == z[0]
    False

    Note that, np.copy clears previously set WRITEABLE=False flag.

    >>> a = np.array([1, 2, 3])
    >>> a.flags["WRITEABLE"] = False
    >>> b = np.copy(a)
    >>> b.flags["WRITEABLE"]
    True
    >>> b[0] = 3
    >>> b
    array([3, 2, 3])
    """
    # 调用 array 函数来创建 `a` 的数组副本，参数 order 和 subok 可选
    return array(a, order=order, subok=subok, copy=True)

# Basic operations

# 定义一个私有函数 _gradient_dispatcher，用于分发参数 f，*varargs，axis 和 edge_order
def _gradient_dispatcher(f, *varargs, axis=None, edge_order=None):
    # 返回参数 f 和 *varargs 的生成器
    yield f
    yield from varargs

# 使用装饰器 array_function_dispatch 将 gradient 函数注册为 _gradient_dispatcher 的分发函数
@array_function_dispatch(_gradient_dispatcher)
def gradient(f, *varargs, axis=None, edge_order=1):
    """
    Return the gradient of an N-dimensional array.

    The gradient is computed using second order accurate central differences
    in the interior points and either first or second order accurate one-sides
    (forward or backwards) differences at the boundaries.
    The returned gradient hence has the same shape as the input array.

    Parameters
    ----------
    f : array_like
        An N-dimensional array containing samples of a scalar function.
    """
    varargs : list of scalar or array, optional
        # 可变参数，可以是标量或数组，用于指定 f 值之间的间距。默认单位间距适用于所有维度。
        # 可以通过以下方式指定间距：
        # 1. 单个标量来指定所有维度的采样距离。
        # 2. N 个标量来指定每个维度的常量采样距离，例如 `dx`, `dy`, `dz`, ...
        # 3. N 个数组来指定沿着每个维度 F 的值的坐标。数组的长度必须与相应维度的大小相匹配。
        # 4. 任意组合的 N 个标量/数组，含义与 2 和 3 相同。
        # 如果指定了 `axis`，则可变参数的数量必须等于轴的数量。
        # 默认值为 1（参见下面的示例）。

    edge_order : {1, 2}, optional
        # 在边界处使用 N 阶精确差分计算梯度。默认值为 1。
        # .. versionadded:: 1.9.1

    axis : None or int or tuple of ints, optional
        # 只计算给定轴或轴上的梯度。
        # 默认情况下（axis = None），计算输入数组的所有轴的梯度。
        # 轴可以为负数，此时从最后一个轴开始计数。
        # .. versionadded:: 1.11.0

    Returns
    -------
    gradient : ndarray or tuple of ndarray
        # 梯度：ndarray 或者多个 ndarray 的元组，每个数组对应于 f 对每个维度的导数。
        # 每个导数的形状与 f 相同。

    Examples
    --------
    >>> f = np.array([1, 2, 4, 7, 11, 16])
    >>> np.gradient(f)
    array([1. , 1.5, 2.5, 3.5, 4.5, 5. ])
    >>> np.gradient(f, 2)
    array([0.5 ,  0.75,  1.25,  1.75,  2.25,  2.5 ])

    # 间距也可以使用代表 F 值沿维度的坐标数组来指定。
    # 例如，均匀间距：
    >>> x = np.arange(f.size)
    >>> np.gradient(f, x)
    array([1. ,  1.5,  2.5,  3.5,  4.5,  5. ])

    # 或者非均匀间距：
    >>> x = np.array([0., 1., 1.5, 3.5, 4., 6.])
    >>> np.gradient(f, x)
    array([1. ,  3. ,  3.5,  6.7,  6.9,  2.5])

    # 对于二维数组，返回的梯度将按轴排序为两个数组。在此示例中，第一个数组代表行的梯度，第二个数组代表列的梯度：
    >>> np.gradient(np.array([[1, 2, 6], [3, 4, 5]]))
    (array([[ 2.,  2., -1.],
            [ 2.,  2., -1.]]),
     array([[1. , 2.5, 4. ],
            [1. , 1. , 1. ]]))

    # 在此示例中也指定了间距：
    # axis=0 为均匀，axis=1 为非均匀
    >>> dx = 2.
    >>> y = [1., 1.5, 3.5]
    >>> np.gradient(np.array([[1, 2, 6], [3, 4, 5]]), dx, y)
    (array([[ 1. ,  1. , -0.5],
            [ 1. ,  1. , -0.5]]),
     array([[2. , 2. , 2. ],
            [2. , 1.7, 0.5]]))

    # 可以使用 `edge_order` 指定边界处理方式
    >>> x = np.array([0, 1, 2, 3, 4])
    # 计算变量 x 的平方
    >>> f = x**2
    
    # 使用 numpy 中的 gradient 函数计算 f 的梯度，边界条件使用一阶导数
    >>> np.gradient(f, edge_order=1)
    array([1.,  2.,  4.,  6.,  7.])
    
    # 使用 numpy 中的 gradient 函数计算 f 的梯度，边界条件使用二阶导数
    >>> np.gradient(f, edge_order=2)
    array([0., 2., 4., 6., 8.])
    
    # 使用 axis 关键字可以指定计算梯度的轴的子集
    >>> np.gradient(np.array([[1, 2, 6], [3, 4, 5]]), axis=0)
    array([[ 2.,  2., -1.],
           [ 2.,  2., -1.]])
    
    # varargs 参数定义输入数组中样本点之间的间距，可以有两种形式：
    
    # 第一种形式：数组形式，指定坐标，可以是不均匀间距
    >>> x = np.array([0., 2., 3., 6., 8.])
    >>> y = x ** 2
    >>> np.gradient(y, x, edge_order=2)
    array([ 0.,  4.,  6., 12., 16.])
    
    # 第二种形式：标量形式，表示固定的样本距离
    >>> dx = 2
    >>> x = np.array([0., 2., 4., 6., 8.])
    >>> y = x ** 2
    >>> np.gradient(y, dx, edge_order=2)
    array([ 0.,  4.,  8., 12., 16.])
    
    # 可以为每个维度提供不同的数据间距，参数数量必须与输入数据的维度数相匹配
    >>> dx = 2
    >>> dy = 3
    >>> x = np.arange(0, 6, dx)
    >>> y = np.arange(0, 9, dy)
    >>> xs, ys = np.meshgrid(x, y)
    >>> zs = xs + 2 * ys
    >>> np.gradient(zs, dy, dx)  # 传递两个标量
    (array([[2., 2., 2.],
            [2., 2., 2.],
            [2., 2., 2.]]),
     array([[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]]))
    
    # 可以混合使用标量和数组：
    
    >>> np.gradient(zs, y, dx)  # 传递一个数组和一个标量
    (array([[2., 2., 2.],
            [2., 2., 2.],
            [2., 2., 2.]]),
     array([[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]]))
    
    # 注意：
    # 假设 f 属于 C^3 类（即 f 至少具有三阶连续导数），并且 h_* 是非齐次步长，我们
    # 最小化真实梯度与从相邻网格点的线性组合估算的“一致性误差” η_i：
    
    # η_i = f_i^(1) - [α f(x_i) + β f(x_i + h_d) + γ f(x_i - h_s)]
    
    # 通过用 f(x_i + h_d) 和 f(x_i - h_s) 的 Taylor 级数展开替换，这转化为解决如下线性系统：
    
    # {
    #   α+β+γ=0
    #   β h_d-γ h_s=1
    #   β h_d^2+γ h_s^2=0
    # }
    
    # 得到的 f_i^(1) 的近似值如下：
    """
    计算给定函数在多维网格上的数值偏导数。
    
    Parameters
    ----------
    f : np.ndarray
        输入的多维数组，表示要计算偏导数的函数。
    axis : None or int or tuple of ints
        表示计算偏导数的轴或轴组合。
    
    Returns
    -------
    list
        包含计算出的偏导数的列表。
    
    Notes
    -----
    此函数通过有限差分方法计算偏导数，使用不同的间距（dx）来处理不同情况。
    
    References
    ----------
    .. [1]  Quarteroni A., Sacco R., Saleri F. (2007) Numerical Mathematics
            (Texts in Applied Mathematics). New York: Springer.
    .. [2]  Durran D. R. (1999) Numerical Methods for Wave Equations
            in Geophysical Fluid Dynamics. New York: Springer.
    .. [3]  Fornberg B. (1988) Generation of Finite Difference Formulas on
            Arbitrarily Spaced Grids,
            Mathematics of Computation 51, no. 184 : 699-706.
            `PDF <https://www.ams.org/journals/mcom/1988-51-184/
            S0025-5718-1988-0935077-0/S0025-5718-1988-0935077-0.pdf>`_.
    """
    
    # 将输入数组转换为numpy数组（如果不是的话）
    f = np.asanyarray(f)
    # 确定输入数组的维度数目
    N = f.ndim  # number of dimensions
    
    # 如果未指定轴，则使用所有轴
    if axis is None:
        axes = tuple(range(N))
    else:
        # 标准化轴元组以便处理
        axes = _nx.normalize_axis_tuple(axis, N)
    
    # 计算轴元组的长度
    len_axes = len(axes)
    # 获取额外参数的数量
    n = len(varargs)
    
    # 根据不同情况为每个轴计算相应的间距(dx)
    if n == 0:
        # 如果没有给定间距参数，则默认为1
        dx = [1.0] * len_axes
    elif n == 1 and np.ndim(varargs[0]) == 0:
        # 如果只提供了一个标量作为间距参数，则在所有轴上使用相同的值
        dx = varargs * len_axes
    elif n == len_axes:
        # 如果为每个轴提供了标量或一维数组作为间距参数
        dx = list(varargs)
        for i, distances in enumerate(dx):
            # 确保间距参数是numpy数组
            distances = np.asanyarray(distances)
            # 检查间距数组的维度
            if distances.ndim == 0:
                continue
            elif distances.ndim != 1:
                # 如果不是一维数组，则引发错误
                raise ValueError("distances must be either scalars or 1d")
            # 检查间距数组的长度是否与相应维度匹配
            if len(distances) != f.shape[axes[i]]:
                raise ValueError("when 1d, distances must match "
                                 "the length of the corresponding dimension")
            # 将numpy整数类型转换为float64，以避免在np.diff中进行模运算
            if np.issubdtype(distances.dtype, np.integer):
                distances = distances.astype(np.float64)
            # 计算间距数组的差分
            diffx = np.diff(distances)
            # 如果间距是常数，则将其减少为标量情况，以提高计算效率
            if (diffx == diffx[0]).all():
                diffx = diffx[0]
            dx[i] = diffx
    # 如果参数个数不正确，则抛出类型错误异常
    else:
        raise TypeError("invalid number of arguments")

    # 如果边缘阶数大于2，则抛出数值错误异常，因为不支持大于2的边缘阶数
    if edge_order > 2:
        raise ValueError("'edge_order' greater than 2 not supported")

    # 在内部使用中心差分，而在端点使用单边差分。
    # 这可以在整个定义域上保持二阶精度。
    outvals = []

    # 创建切片对象 --- 初始时所有切片都是 [:, :, ..., :]
    slice1 = [slice(None)]*N
    slice2 = [slice(None)]*N
    slice3 = [slice(None)]*N
    slice4 = [slice(None)]*N

    # 获取数组 f 的数据类型
    otype = f.dtype
    # 如果数据类型是 np.datetime64，则将其视图转换为同一单位的 np.timedelta64 类型
    if otype.type is np.datetime64:
        otype = np.dtype(otype.name.replace('datetime', 'timedelta'))
        f = f.view(otype)  # 允许进行时间增减的视图转换
    # 如果数据类型是 np.timedelta64，则不做任何处理
    elif otype.type is np.timedelta64:
        pass
    # 如果数据类型是浮点数或其子类型，则不做任何处理
    elif np.issubdtype(otype, np.inexact):
        pass
    else:
        # 所有其他类型都转换为浮点数。
        # 首先检查 f 是否是 numpy 整数类型；如果是，则将其转换为 float64，
        # 以避免在计算 f 的变化时进行模运算。
        if np.issubdtype(otype, np.integer):
            f = f.astype(np.float64)
        otype = np.float64

    # 如果 len_axes 为 1，则返回 outvals 列表的第一个元素
    if len_axes == 1:
        return outvals[0]
    # 否则返回 outvals 列表的元组
    return tuple(outvals)
# 定义一个函数 _diff_dispatcher，用于将参数 a, n, axis, prepend, append 封装成一个元组返回
def _diff_dispatcher(a, n=None, axis=None, prepend=None, append=None):
    return (a, prepend, append)

# 使用 array_function_dispatch 装饰器，将 _diff_dispatcher 函数注册为 diff 函数的分发函数
@array_function_dispatch(_diff_dispatcher)
def diff(a, n=1, axis=-1, prepend=np._NoValue, append=np._NoValue):
    """
    Calculate the n-th discrete difference along the given axis.

    The first difference is given by ``out[i] = a[i+1] - a[i]`` along
    the given axis, higher differences are calculated by using `diff`
    recursively.

    Parameters
    ----------
    a : array_like
        Input array
    n : int, optional
        The number of times values are differenced. If zero, the input
        is returned as-is.
    axis : int, optional
        The axis along which the difference is taken, default is the
        last axis.
    prepend, append : array_like, optional
        Values to prepend or append to `a` along axis prior to
        performing the difference.  Scalar values are expanded to
        arrays with length 1 in the direction of axis and the shape
        of the input array in along all other axes.  Otherwise the
        dimension and shape must match `a` except along axis.

        .. versionadded:: 1.16.0

    Returns
    -------
    diff : ndarray
        The n-th differences. The shape of the output is the same as `a`
        except along `axis` where the dimension is smaller by `n`. The
        type of the output is the same as the type of the difference
        between any two elements of `a`. This is the same as the type of
        `a` in most cases. A notable exception is `datetime64`, which
        results in a `timedelta64` output array.

    See Also
    --------
    gradient, ediff1d, cumsum

    Notes
    -----
    Type is preserved for boolean arrays, so the result will contain
    `False` when consecutive elements are the same and `True` when they
    differ.

    For unsigned integer arrays, the results will also be unsigned. This
    should not be surprising, as the result is consistent with
    calculating the difference directly:

    >>> u8_arr = np.array([1, 0], dtype=np.uint8)
    >>> np.diff(u8_arr)
    array([255], dtype=uint8)
    >>> u8_arr[1,...] - u8_arr[0,...]
    255

    If this is not desirable, then the array should be cast to a larger
    integer type first:

    >>> i16_arr = u8_arr.astype(np.int16)
    >>> np.diff(i16_arr)
    array([-1], dtype=int16)

    Examples
    --------
    >>> x = np.array([1, 2, 4, 7, 0])
    >>> np.diff(x)
    array([ 1,  2,  3, -7])
    >>> np.diff(x, n=2)
    array([  1,   1, -10])

    >>> x = np.array([[1, 3, 6, 10], [0, 5, 6, 8]])
    >>> np.diff(x)
    array([[2, 3, 4],
           [5, 1, 2]])
    >>> np.diff(x, axis=0)
    array([[-1,  2,  0, -2]])

    >>> x = np.arange('1066-10-13', '1066-10-16', dtype=np.datetime64)
    >>> np.diff(x)
    array([1, 1], dtype='timedelta64[D]')

    """
    # 如果 n 等于 0，直接返回输入数组 a
    if n == 0:
        return a
    # 如果 n 小于 0，抛出 ValueError 异常，要求 n 必须为非负数
    if n < 0:
        raise ValueError(
            "order must be non-negative but got " + repr(n))

    # 将输入数组 a 转换为任意数组的类型
    a = asanyarray(a)
    # 获取输入数组的维度
    nd = a.ndim
    # 如果数组维度为0，抛出数值错误异常，要求输入至少为一维
    if nd == 0:
        raise ValueError("diff requires input that is at least one dimensional")
    # 根据指定的轴对索引进行规范化处理
    axis = normalize_axis_index(axis, nd)
    
    # 创建一个空列表用于存储组合后的数组
    combined = []
    # 如果有指定要在前面添加的数组（prepend 不是 np._NoValue）
    if prepend is not np._NoValue:
        # 将 prepend 转换为数组
        prepend = np.asanyarray(prepend)
        # 如果 prepend 是0维数组，将其扩展为与指定轴相对应的形状
        if prepend.ndim == 0:
            shape = list(a.shape)
            shape[axis] = 1
            prepend = np.broadcast_to(prepend, tuple(shape))
        # 将 prepend 添加到 combined 列表中
        combined.append(prepend)
    
    # 将原始数组 a 添加到 combined 列表中
    combined.append(a)
    
    # 如果有指定要在后面添加的数组（append 不是 np._NoValue）
    if append is not np._NoValue:
        # 将 append 转换为数组
        append = np.asanyarray(append)
        # 如果 append 是0维数组，将其扩展为与指定轴相对应的形状
        if append.ndim == 0:
            shape = list(a.shape)
            shape[axis] = 1
            append = np.broadcast_to(append, tuple(shape))
        # 将 append 添加到 combined 列表中
        combined.append(append)
    
    # 如果 combined 列表中包含多个数组
    if len(combined) > 1:
        # 使用指定轴进行连接操作，更新原始数组 a
        a = np.concatenate(combined, axis)
    
    # 创建两个用于切片操作的列表，初始都是全切片
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    # 将 slice1 和 slice2 中对应指定轴的切片更新为指定范围
    slice1[axis] = slice(1, None)   # 第一个切片从索引1到末尾
    slice2[axis] = slice(None, -1)  # 第二个切片从开头到倒数第二个
    # 转换为元组形式，用于后续切片操作
    slice1 = tuple(slice1)
    slice2 = tuple(slice2)
    
    # 根据数组的数据类型选择操作（如果是布尔型则选择 not_equal 否则选择 subtract）
    op = not_equal if a.dtype == np.bool else subtract
    # 迭代执行 n 次操作
    for _ in range(n):
        # 执行切片后的操作并更新数组 a
        a = op(a[slice1], a[slice2])
    
    # 返回处理后的数组 a
    return a
# 定义一个辅助函数 _interp_dispatcher，用于分发参数给 interp 函数
def _interp_dispatcher(x, xp, fp, left=None, right=None, period=None):
    # 直接返回参数 x, xp, fp，作为分发的结果
    return (x, xp, fp)


# 使用 array_function_dispatch 装饰器将 _interp_dispatcher 注册为 interp 函数的分发器
@array_function_dispatch(_interp_dispatcher)
def interp(x, xp, fp, left=None, right=None, period=None):
    """
    One-dimensional linear interpolation for monotonically increasing sample points.

    Returns the one-dimensional piecewise linear interpolant to a function
    with given discrete data points (`xp`, `fp`), evaluated at `x`.

    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the interpolated values.

    xp : 1-D sequence of floats
        The x-coordinates of the data points, must be increasing if argument
        `period` is not specified. Otherwise, `xp` is internally sorted after
        normalizing the periodic boundaries with ``xp = xp % period``.

    fp : 1-D sequence of float or complex
        The y-coordinates of the data points, same length as `xp`.

    left : optional float or complex corresponding to fp
        Value to return for `x < xp[0]`, default is `fp[0]`.

    right : optional float or complex corresponding to fp
        Value to return for `x > xp[-1]`, default is `fp[-1]`.

    period : None or float, optional
        A period for the x-coordinates. This parameter allows the proper
        interpolation of angular x-coordinates. Parameters `left` and `right`
        are ignored if `period` is specified.

        .. versionadded:: 1.10.0

    Returns
    -------
    y : float or complex (corresponding to fp) or ndarray
        The interpolated values, same shape as `x`.

    Raises
    ------
    ValueError
        If `xp` and `fp` have different length
        If `xp` or `fp` are not 1-D sequences
        If `period == 0`

    See Also
    --------
    scipy.interpolate

    Warnings
    --------
    The x-coordinate sequence is expected to be increasing, but this is not
    explicitly enforced.  However, if the sequence `xp` is non-increasing,
    interpolation results are meaningless.

    Note that, since NaN is unsortable, `xp` also cannot contain NaNs.

    A simple check for `xp` being strictly increasing is::

        np.all(np.diff(xp) > 0)

    Examples
    --------
    >>> xp = [1, 2, 3]
    >>> fp = [3, 2, 0]
    >>> np.interp(2.5, xp, fp)
    1.0
    >>> np.interp([0, 1, 1.5, 2.72, 3.14], xp, fp)
    array([3.  , 3.  , 2.5 , 0.56, 0.  ])
    >>> UNDEF = -99.0
    >>> np.interp(3.14, xp, fp, right=UNDEF)
    -99.0

    Plot an interpolant to the sine function:

    >>> x = np.linspace(0, 2*np.pi, 10)
    >>> y = np.sin(x)
    >>> xvals = np.linspace(0, 2*np.pi, 50)
    >>> yinterp = np.interp(xvals, x, y)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(x, y, 'o')
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.plot(xvals, yinterp, '-x')
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.show()

    Interpolation with periodic x-coordinates:

    >>> x = [-180, -170, -185, 185, -10, -5, 0, 365]
    """

    # 函数文档字符串中包含了详细的参数描述、返回值说明和示例用法
    # 实现了对于给定的 `x` 在 `xp` 和 `fp` 给定的离散数据点上进行线性插值的功能
    # 可选参数 `left` 和 `right` 控制在 `x` 超出 `xp` 范围时的返回值
    # 可选参数 `period` 允许处理周期性的 `x` 坐标，用于角度坐标的正确插值
    pass
    fp = np.asarray(fp)
    """
    将输入的 fp 转换为 NumPy 数组，确保统一的数据结构和操作性能
    """

    if np.iscomplexobj(fp):
        interp_func = compiled_interp_complex
        input_dtype = np.complex128
    else:
        interp_func = compiled_interp
        input_dtype = np.float64
    """
    检查 fp 是否包含复数对象，选择相应的插值函数和数据类型
    如果 fp 包含复数对象，则选择编译后的复数插值函数和复数数据类型 np.complex128
    否则选择编译后的标准插值函数和浮点数据类型 np.float64
    """

    if period is not None:
        if period == 0:
            raise ValueError("period must be a non-zero value")
        period = abs(period)
        left = None
        right = None

        x = np.asarray(x, dtype=np.float64)
        xp = np.asarray(xp, dtype=np.float64)
        fp = np.asarray(fp, dtype=input_dtype)
        """
        处理周期性插值的情况：
        - 确保 period 是正数
        - 初始化 left 和 right 为 None
        - 将 x, xp, fp 转换为 NumPy 数组，并指定数据类型为 np.float64 或者根据 fp 的复杂性选择 np.complex128
        """

        if xp.ndim != 1 or fp.ndim != 1:
            raise ValueError("Data points must be 1-D sequences")
        if xp.shape[0] != fp.shape[0]:
            raise ValueError("fp and xp are not of the same length")
        """
        检查 xp 和 fp 是否是一维数组，并且长度相同，否则抛出 ValueError 异常
        """

        # normalizing periodic boundaries
        x = x % period
        xp = xp % period
        asort_xp = np.argsort(xp)
        xp = xp[asort_xp]
        fp = fp[asort_xp]
        xp = np.concatenate((xp[-1:]-period, xp, xp[0:1]+period))
        fp = np.concatenate((fp[-1:], fp, fp[0:1]))
        """
        标准化周期边界：
        - 计算 x, xp 取模后的结果
        - 对 xp 进行排序并重新排序 fp
        - 扩展 xp 和 fp，以处理周期性边界条件
        """

    return interp_func(x, xp, fp, left, right)
"""
调用选定的插值函数 interp_func 进行插值计算，使用标准化的输入数据和边界条件
返回插值结果
"""
# 定义一个辅助函数 _angle_dispatcher，用于返回传入的参数 z
def _angle_dispatcher(z, deg=None):
    return (z,)

# 使用 array_function_dispatch 装饰器注册 _angle_dispatcher 函数，使其能够根据不同输入分派到合适的函数上
@array_function_dispatch(_angle_dispatcher)
# 定义 angle 函数，计算复数参数的幅角
def angle(z, deg=False):
    """
    Return the angle of the complex argument.

    Parameters
    ----------
    z : array_like
        A complex number or sequence of complex numbers.
    deg : bool, optional
        Return angle in degrees if True, radians if False (default).

    Returns
    -------
    angle : ndarray or scalar
        The counterclockwise angle from the positive real axis on the complex
        plane in the range ``(-pi, pi]``, with dtype as numpy.float64.

        .. versionchanged:: 1.16.0
            This function works on subclasses of ndarray like `ma.array`.

    See Also
    --------
    arctan2
    absolute

    Notes
    -----
    This function passes the imaginary and real parts of the argument to
    `arctan2` to compute the result; consequently, it follows the convention
    of `arctan2` when the magnitude of the argument is zero. See example.

    Examples
    --------
    >>> np.angle([1.0, 1.0j, 1+1j])               # in radians
    array([ 0.        ,  1.57079633,  0.78539816]) # may vary
    >>> np.angle(1+1j, deg=True)                  # in degrees
    45.0
    >>> np.angle([0., -0., complex(0., -0.), complex(-0., -0.)])  # convention
    array([ 0.        ,  3.14159265, -0.        , -3.14159265])

    """
    # 将输入 z 转换为 ndarray 类型
    z = asanyarray(z)
    # 如果 z 的数据类型是复数浮点数类型的子类，则分别提取虚部和实部
    if issubclass(z.dtype.type, _nx.complexfloating):
        zimag = z.imag
        zreal = z.real
    else:
        # 否则，虚部设为0，实部设为 z
        zimag = 0
        zreal = z

    # 使用 arctan2 计算 zimag 和 zreal 的反正切值，得到角度 a
    a = arctan2(zimag, zreal)
    # 如果 deg 为 True，则将角度 a 转换为度数
    if deg:
        a *= 180/pi
    # 返回计算得到的角度 a
    return a


# 定义一个辅助函数 _unwrap_dispatcher，用于返回传入的参数 p
def _unwrap_dispatcher(p, discont=None, axis=None, *, period=None):
    return (p,)

# 使用 array_function_dispatch 装饰器注册 _unwrap_dispatcher 函数，使其能够根据不同输入分派到合适的函数上
@array_function_dispatch(_unwrap_dispatcher)
# 定义 unwrap 函数，用于信号解包
def unwrap(p, discont=None, axis=-1, *, period=2*pi):
    r"""
    Unwrap by taking the complement of large deltas with respect to the period.

    This unwraps a signal `p` by changing elements which have an absolute
    difference from their predecessor of more than ``max(discont, period/2)``
    to their `period`-complementary values.

    For the default case where `period` is :math:`2\pi` and `discont` is
    :math:`\pi`, this unwraps a radian phase `p` such that adjacent differences
    are never greater than :math:`\pi` by adding :math:`2k\pi` for some
    integer :math:`k`.

    Parameters
    ----------
    p : array_like
        Input array.
    discont : float, optional
        Maximum discontinuity between values, default is ``period/2``.
        Values below ``period/2`` are treated as if they were ``period/2``.
        To have an effect different from the default, `discont` should be
        larger than ``period/2``.
    axis : int, optional
        Axis along which unwrap will operate, default is the last axis.
    period : float, optional
        Size of the range over which the input wraps. By default, it is
        ``2 pi``.

        .. versionadded:: 1.21.0

    Returns
    -------
    """
    # 将输入参数 `p` 转换为 NumPy 数组
    p = asarray(p)
    # 确定输入数组 `p` 的维度
    nd = p.ndim
    # 计算 `p` 沿指定轴的差分，结果存储在 `dd` 中
    dd = diff(p, axis=axis)
    # 如果未提供 `discont` 参数，则将其设为 `period` 的一半
    if discont is None:
        discont = period/2
    # 创建一个包含全切片的元组 `slice1`，用于构建切片对象
    slice1 = [slice(None, None)]*nd     # full slices
    slice1[axis] = slice(1, None)
    slice1 = tuple(slice1)
    # 推断结果数据类型，考虑 `dd` 和 `period` 的类型
    dtype = np.result_type(dd, period)
    # 如果数据类型是整数类型，计算 `period` 的一半和余数
    if _nx.issubdtype(dtype, _nx.integer):
        interval_high, rem = divmod(period, 2)
        boundary_ambiguous = rem == 0
    else:
        # 否则，将 `interval_high` 设为 `period` 的一半，且边界不明确
        interval_high = period / 2
        boundary_ambiguous = True
    # 计算 `interval_low` 作为 `-interval_high`
    interval_low = -interval_high
    # 对 `dd` 应用模运算，将结果映射到区间 `[interval_low, interval_high]`
    ddmod = mod(dd - interval_low, period) + interval_low
    # 如果边界不明确，修正可能存在的问题
    if boundary_ambiguous:
        # 对于 `mask = (abs(dd) == period/2)` 的情况，修正 `ddmod`
        _nx.copyto(ddmod, interval_high,
                   where=(ddmod == interval_low) & (dd > 0))
    # 计算修正的相位差 `ph_correct`
    ph_correct = ddmod - dd
    # 将小于 `discont` 的绝对值的元素修正为零
    _nx.copyto(ph_correct, 0, where=abs(dd) < discont)
    # 创建 `up` 数组作为 `p` 的副本，数据类型为 `dtype`
    up = array(p, copy=True, dtype=dtype)
    # 对 `up` 应用累积和修正后的相位差 `ph_correct`
    up[slice1] = p[slice1] + ph_correct.cumsum(axis)
    # 返回修正后的数组 `up`
    return up
# 定义一个函数 _sort_complex，接受一个参数 a，返回一个包含 a 的元组
def _sort_complex(a):
    return (a,)


# 使用装饰器 array_function_dispatch，将 _sort_complex 注册为 sort_complex 函数的分派函数
@array_function_dispatch(_sort_complex)
# 定义 sort_complex 函数，对复数数组按照实部先排序，再按照虚部排序
def sort_complex(a):
    """
    Sort a complex array using the real part first, then the imaginary part.

    Parameters
    ----------
    a : array_like
        Input array

    Returns
    -------
    out : complex ndarray
        Always returns a sorted complex array.

    Examples
    --------
    >>> np.sort_complex([5, 3, 6, 2, 1])
    array([1.+0.j, 2.+0.j, 3.+0.j, 5.+0.j, 6.+0.j])

    >>> np.sort_complex([1 + 2j, 2 - 1j, 3 - 2j, 3 - 3j, 3 + 5j])
    array([1.+2.j,  2.-1.j,  3.-3.j,  3.-2.j,  3.+5.j])

    """
    # 复制输入数组 a 到 b
    b = array(a, copy=True)
    # 对 b 进行排序
    b.sort()
    # 如果 b 的数据类型不是复数类型，则根据数据类型进行转换
    if not issubclass(b.dtype.type, _nx.complexfloating):
        if b.dtype.char in 'bhBH':
            return b.astype('F')
        elif b.dtype.char == 'g':
            return b.astype('G')
        else:
            return b.astype('D')
    else:
        return b


# 定义一个函数 _trim_zeros，接受一个参数 filt，并返回一个包含 filt 的元组
def _trim_zeros(filt, trim=None):
    return (filt,)


# 使用装饰器 array_function_dispatch，将 _trim_zeros 注册为 trim_zeros 函数的分派函数
@array_function_dispatch(_trim_zeros)
# 定义 trim_zeros 函数，从 1-D 数组或序列中移除开头和/或结尾的零值
def trim_zeros(filt, trim='fb'):
    """
    Trim the leading and/or trailing zeros from a 1-D array or sequence.

    Parameters
    ----------
    filt : 1-D array or sequence
        Input array.
    trim : str, optional
        A string with 'f' representing trim from front and 'b' to trim from
        back. Default is 'fb', trim zeros from both front and back of the
        array.

    Returns
    -------
    trimmed : 1-D array or sequence
        The result of trimming the input. The input data type is preserved.

    Examples
    --------
    >>> a = np.array((0, 0, 0, 1, 2, 3, 0, 2, 1, 0))
    >>> np.trim_zeros(a)
    array([1, 2, 3, 0, 2, 1])

    >>> np.trim_zeros(a, 'b')
    array([0, 0, 0, ..., 0, 2, 1])

    The input data type is preserved, list/tuple in means list/tuple out.

    >>> np.trim_zeros([0, 1, 2, 0])
    [1, 2]

    """

    # 初始化 first 为 0，将 trim 参数转换为大写
    first = 0
    trim = trim.upper()
    # 如果 trim 包含 'F'，从前向后遍历 filt
    if 'F' in trim:
        for i in filt:
            if i != 0.:
                break
            else:
                first = first + 1
    # 初始化 last 为 filt 的长度
    last = len(filt)
    # 如果 trim 包含 'B'，从后向前遍历 filt
    if 'B' in trim:
        for i in filt[::-1]:
            if i != 0.:
                break
            else:
                last = last - 1
    # 返回从 first 到 last 范围内的 filt 切片
    return filt[first:last]


# 定义一个函数 _extract_dispatcher，接受两个参数 condition 和 arr，并返回一个包含这两个参数的元组
def _extract_dispatcher(condition, arr):
    return (condition, arr)


# 使用装饰器 array_function_dispatch，将 _extract_dispatcher 注册为 extract 函数的分派函数
@array_function_dispatch(_extract_dispatcher)
# 定义 extract 函数，返回满足条件的数组元素
def extract(condition, arr):
    """
    Return the elements of an array that satisfy some condition.

    This is equivalent to ``np.compress(ravel(condition), ravel(arr))``.  If
    `condition` is boolean ``np.extract`` is equivalent to ``arr[condition]``.

    Note that `place` does the exact opposite of `extract`.

    Parameters
    ----------
    condition : array_like
        An array whose nonzero or True entries indicate the elements of `arr`
        to extract.
    arr : array_like
        Input array of the same size as `condition`.

    Returns
    -------



注释：
    extract : ndarray
        # 返回一个ndarray，其中包含满足条件`condition`为True的`arr`的一维数组值。

    See Also
    --------
    take, put, copyto, compress, place
        # 相关函数：take, put, copyto, compress, place

    Examples
    --------
    >>> arr = np.arange(12).reshape((3, 4))
    >>> arr
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])
    >>> condition = np.mod(arr, 3)==0
    >>> condition
    array([[ True, False, False,  True],
           [False, False,  True, False],
           [False,  True, False, False]])
    >>> np.extract(condition, arr)
    array([0, 3, 6, 9])


    If `condition` is boolean:

    >>> arr[condition]
    array([0, 3, 6, 9])
        # 如果`condition`是布尔值数组，可以直接使用布尔索引来获得满足条件的值。

    """
    # 返回满足条件的`arr`中的值，`ravel(arr)`将多维数组扁平化，`nonzero(ravel(condition))[0]`找到第一个满足条件的索引并取出值。
    return _nx.take(ravel(arr), nonzero(ravel(condition))[0])
# 定义一个元组，包含三个元素的函数，用于分发数组函数
def _place_dispatcher(arr, mask, vals):
    return (arr, mask, vals)


# 使用装饰器将 _place_dispatcher 函数注册为 place 函数的分发器
@array_function_dispatch(_place_dispatcher)
def place(arr, mask, vals):
    """
    Change elements of an array based on conditional and input values.

    Similar to ``np.copyto(arr, vals, where=mask)``, the difference is that
    `place` uses the first N elements of `vals`, where N is the number of
    True values in `mask`, while `copyto` uses the elements where `mask`
    is True.

    Note that `extract` does the exact opposite of `place`.

    Parameters
    ----------
    arr : ndarray
        Array to put data into.
    mask : array_like
        Boolean mask array. Must have the same size as `a`.
    vals : 1-D sequence
        Values to put into `a`. Only the first N elements are used, where
        N is the number of True values in `mask`. If `vals` is smaller
        than N, it will be repeated, and if elements of `a` are to be masked,
        this sequence must be non-empty.

    See Also
    --------
    copyto, put, take, extract

    Examples
    --------
    >>> arr = np.arange(6).reshape(2, 3)
    >>> np.place(arr, arr>2, [44, 55])
    >>> arr
    array([[ 0,  1,  2],
           [44, 55, 44]])

    """
    # 调用 _place 函数，并返回结果
    return _place(arr, mask, vals)


# 显示消息在设备上的函数
def disp(mesg, device=None, linefeed=True):
    """
    Display a message on a device.

    .. deprecated:: 2.0
        Use your own printing function instead.

    Parameters
    ----------
    mesg : str
        Message to display.
    device : object
        Device to write message. If None, defaults to ``sys.stdout`` which is
        very similar to ``print``. `device` needs to have ``write()`` and
        ``flush()`` methods.
    linefeed : bool, optional
        Option whether to print a line feed or not. Defaults to True.

    Raises
    ------
    AttributeError
        If `device` does not have a ``write()`` or ``flush()`` method.

    Examples
    --------
    Besides ``sys.stdout``, a file-like object can also be used as it has
    both required methods:

    >>> from io import StringIO
    >>> buf = StringIO()
    >>> np.disp('"Display" in a file', device=buf)
    >>> buf.getvalue()
    '"Display" in a file\\n'

    """

    # 在 NumPy 2.0 版本开始废弃，2023-07-11
    # 发出警告信息，建议使用者停止使用 disp 函数
    warnings.warn(
        "`disp` is deprecated, "
        "use your own printing function instead. "
        "(deprecated in NumPy 2.0)",
        DeprecationWarning,
        stacklevel=2
    )    

    # 确保设备参数不为空，否则使用默认的 sys.stdout
    if device is None:
        device = sys.stdout
    # 根据 linefeed 参数决定是否输出换行符
    if linefeed:
        device.write('%s\n' % mesg)
    else:
        device.write('%s' % mesg)
    # 刷新设备输出
    device.flush()
    return


# 定义一个正则表达式模式，用于解析广义通用函数（gufunc）的签名
# 详细内容请参考：https://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html
_DIMENSION_NAME = r'\w+'
_CORE_DIMENSION_LIST = '(?:{0:}(?:,{0:})*)?'.format(_DIMENSION_NAME)
_ARGUMENT = r'\({}\)'.format(_CORE_DIMENSION_LIST)
_ARGUMENT_LIST = '{0:}(?:,{0:})*'.format(_ARGUMENT)
_SIGNATURE = '^{0:}->{0:}$'.format(_ARGUMENT_LIST)


def _parse_gufunc_signature(signature):
    """
    # 使用正则表达式去除签名中的所有空白字符
    signature = re.sub(r'\s+', '', signature)

    # 如果签名不符合预期的模式（正则表达式 _SIGNATURE），抛出数值错误异常
    if not re.match(_SIGNATURE, signature):
        raise ValueError(
            'not a valid gufunc signature: {}'.format(signature))
    
    # 将签名字符串按 '->' 分割成输入和输出部分，并解析每个部分的维度名称
    return tuple([tuple(re.findall(_DIMENSION_NAME, arg))
                  for arg in re.findall(_ARGUMENT, arg_list)]
                 for arg_list in signature.split('->'))
def _update_dim_sizes(dim_sizes, arg, core_dims):
    """
    Incrementally check and update core dimension sizes for a single argument.

    Arguments
    ---------
    dim_sizes : Dict[str, int]
        Sizes of existing core dimensions. Will be updated in-place.
    arg : ndarray
        Argument to examine.
    core_dims : Tuple[str, ...]
        Core dimensions for this argument.
    """
    # 如果没有核心维度，直接返回
    if not core_dims:
        return

    # 获取当前参数的核心维度数量
    num_core_dims = len(core_dims)
    # 如果参数的维度少于核心维度数量，则抛出异常
    if arg.ndim < num_core_dims:
        raise ValueError(
            '%d-dimensional argument does not have enough '
            'dimensions for all core dimensions %r'
            % (arg.ndim, core_dims))

    # 获取参数在核心维度上的形状
    core_shape = arg.shape[-num_core_dims:]
    # 逐一检查核心维度及其大小，并更新到 dim_sizes 中
    for dim, size in zip(core_dims, core_shape):
        if dim in dim_sizes:
            # 如果已存在该核心维度，检查其大小是否一致，否则抛出异常
            if size != dim_sizes[dim]:
                raise ValueError(
                    'inconsistent size for core dimension %r: %r vs %r'
                    % (dim, size, dim_sizes[dim]))
        else:
            # 如果不存在该核心维度，则将其大小添加到 dim_sizes 中
            dim_sizes[dim] = size


def _parse_input_dimensions(args, input_core_dims):
    """
    Parse broadcast and core dimensions for vectorize with a signature.

    Arguments
    ---------
    args : Tuple[ndarray, ...]
        Tuple of input arguments to examine.
    input_core_dims : List[Tuple[str, ...]]
        List of core dimensions corresponding to each input.

    Returns
    -------
    broadcast_shape : Tuple[int, ...]
        Common shape to broadcast all non-core dimensions to.
    dim_sizes : Dict[str, int]
        Common sizes for named core dimensions.
    """
    # 初始化广播参数列表和核心维度大小字典
    broadcast_args = []
    dim_sizes = {}
    # 遍历输入参数和其对应的核心维度
    for arg, core_dims in zip(args, input_core_dims):
        # 更新核心维度大小字典
        _update_dim_sizes(dim_sizes, arg, core_dims)
        # 计算非核心维度的广播形状，创建虚拟数组
        ndim = arg.ndim - len(core_dims)
        dummy_array = np.lib.stride_tricks.as_strided(0, arg.shape[:ndim])
        broadcast_args.append(dummy_array)
    # 计算所有广播参数的最终形状
    broadcast_shape = np.lib._stride_tricks_impl._broadcast_shape(
        *broadcast_args
    )
    return broadcast_shape, dim_sizes


def _calculate_shapes(broadcast_shape, dim_sizes, list_of_core_dims):
    """
    Helper for calculating broadcast shapes with core dimensions.

    Arguments
    ---------
    broadcast_shape : Tuple[int, ...]
        Common shape to broadcast all non-core dimensions to.
    dim_sizes : Dict[str, int]
        Common sizes for named core dimensions.
    list_of_core_dims : List[Tuple[str, ...]]
        List of tuples of core dimensions for each output.

    Returns
    -------
    List[Tuple[int, ...]]
        List of shapes for each output array.
    """
    # 根据输入的广播形状和核心维度大小字典，计算每个输出数组的形状列表
    return [broadcast_shape + tuple(dim_sizes[dim] for dim in core_dims)
            for core_dims in list_of_core_dims]


def _create_arrays(broadcast_shape, dim_sizes, list_of_core_dims, dtypes,
                   results=None):
    """
    Helper for creating output arrays in vectorize.

    Arguments
    ---------
    broadcast_shape : Tuple[int, ...]
        Common shape to broadcast all non-core dimensions to.
    dim_sizes : Dict[str, int]
        Common sizes for named core dimensions.
    list_of_core_dims : List[Tuple[str, ...]]
        List of tuples of core dimensions for each output.
    dtypes : List[Optional[np.dtype]]
        List of data types for each output array.
    results : Optional[List[ndarray]]
        List of existing arrays to use as template.

    Returns
    -------
    Tuple[ndarray, ...]
        Tuple of newly created output arrays.
    """
    # 计算输出数组的形状列表
    shapes = _calculate_shapes(broadcast_shape, dim_sizes, list_of_core_dims)
    # 如果没有指定数据类型，全部置为 None
    if dtypes is None:
        dtypes = [None] * len(shapes)
    # 如果没有指定结果数组，创建空数组
    if results is None:
        arrays = tuple(np.empty(shape=shape, dtype=dtype)
                       for shape, dtype in zip(shapes, dtypes))
    else:
        # 否则，根据结果数组创建相同形状的数组
        arrays = tuple(np.empty_like(result, shape=shape, dtype=dtype)
                       for result, shape, dtype
                       in zip(results, shapes, dtypes))
    return arrays


def _get_vectorize_dtype(dtype):
    """
    Placeholder for determining dtype in vectorize.

    Arguments
    ---------
    dtype : np.dtype
        Input data type.

    Returns
    -------
    np.dtype
        Output data type.
    """
    # 这是一个占位函数，用于在向量化过程中确定数据类型
    pass
    # 如果 dtype 的类型字符是 "S" 或 "U" 中的任意一个，则返回该字符
    if dtype.char in "SU":
        # 返回 dtype 的类型字符，即 "S" 或 "U"
        return dtype.char
    # 如果 dtype 的类型字符不是 "S" 或 "U"，则返回 dtype 本身
    return dtype
@set_module('numpy')
class vectorize:
    """
    vectorize(pyfunc=np._NoValue, otypes=None, doc=None, excluded=None,
    cache=False, signature=None)

    Returns an object that acts like pyfunc, but takes arrays as input.

    Define a vectorized function which takes a nested sequence of objects or
    numpy arrays as inputs and returns a single numpy array or a tuple of numpy
    arrays. The vectorized function evaluates `pyfunc` over successive tuples
    of the input arrays like the python map function, except it uses the
    broadcasting rules of numpy.

    The data type of the output of `vectorized` is determined by calling
    the function with the first element of the input.  This can be avoided
    by specifying the `otypes` argument.

    Parameters
    ----------
    pyfunc : callable, optional
        A python function or method.
        Can be omitted to produce a decorator with keyword arguments.
    otypes : str or list of dtypes, optional
        The output data type. It must be specified as either a string of
        typecode characters or a list of data type specifiers. There should
        be one data type specifier for each output.
    doc : str, optional
        The docstring for the function. If None, the docstring will be the
        ``pyfunc.__doc__``.
    excluded : set, optional
        Set of strings or integers representing the positional or keyword
        arguments for which the function will not be vectorized.  These will be
        passed directly to `pyfunc` unmodified.

        .. versionadded:: 1.7.0

    cache : bool, optional
        If `True`, then cache the first function call that determines the number
        of outputs if `otypes` is not provided.

        .. versionadded:: 1.7.0

    signature : string, optional
        Generalized universal function signature, e.g., ``(m,n),(n)->(m)`` for
        vectorized matrix-vector multiplication. If provided, ``pyfunc`` will
        be called with (and expected to return) arrays with shapes given by the
        size of corresponding core dimensions. By default, ``pyfunc`` is
        assumed to take scalars as input and output.

        .. versionadded:: 1.12.0

    Returns
    -------
    out : callable
        A vectorized function if ``pyfunc`` was provided,
        a decorator otherwise.

    See Also
    --------
    frompyfunc : Takes an arbitrary Python function and returns a ufunc

    Notes
    -----
    The `vectorize` function is provided primarily for convenience, not for
    performance. The implementation is essentially a for loop.

    If `otypes` is not specified, then a call to the function with the
    first argument will be used to determine the number of outputs.  The
    results of this call will be cached if `cache` is `True` to prevent
    calling the function twice.  However, to implement the cache, the
    original function must be wrapped which will slow down subsequent
    calls, so only do this if your function is expensive.
    """
    # 导入NumPy库
    import numpy as np

    # 函数np.vectorize允许将普通函数向量化，以便能够处理数组输入
    # 可以通过doc参数来指定文档字符串，否则将使用输入函数的文档字符串
    # 例如，向量化函数myfunc
    >>> vfunc = np.vectorize(myfunc)

    # 示例：向量化后的函数可以处理数组输入，返回对应的结果数组
    >>> vfunc([1, 2, 3, 4], 2)
    array([3, 4, 1, 2])

    # 向量化函数的文档字符串可以通过__doc__属性访问
    >>> vfunc.__doc__
    'Return a-b if a>b, otherwise return a+b'

    # 可以通过otypes参数指定输出类型，否则由第一个输入元素类型决定
    >>> vfunc = np.vectorize(myfunc, otypes=[float])
    >>> out = vfunc([1, 2, 3, 4], 2)

    # 输出类型可以通过type()函数获取
    >>> type(out[0])
    <class 'numpy.float64'>

    # excluded参数用于排除不希望向量化的函数参数，比如数组长度固定的多项式系数
    >>> vpolyval = np.vectorize(mypolyval, excluded=['p'])
    >>> vpolyval(p=[1, 2, 3], x=[0, 1])

    # 也可以通过位置来排除参数
    >>> vpolyval.excluded.add(0)
    >>> vpolyval([1, 2, 3], x=[0, 1])

    # signature参数允许在非标量数组上向量化函数，如计算Pearson相关系数及其p值
    >>> pearsonr = np.vectorize(scipy.stats.pearsonr, signature='(n),(n)->(),()')
    >>> pearsonr([[0, 1, 2, 3]], [[1, 2, 3, 4], [4, 3, 2, 1]])

    # 或者用于向量化卷积操作
    >>> convolve = np.vectorize(np.convolve, signature='(n),(m)->(k)')
    >>> convolve(np.eye(4), [1, 2, 1])

    # 还支持装饰器语法，可以用作函数或函数调用来提供关键字参数
    >>> @np.vectorize
    ... def identity(x):
    ...     return x
    >>> identity([0, 1, 2])

    >>> @np.vectorize(otypes=[float])
    ... def as_float(x):
    ...     return x
    >>> as_float([0, 1, 2])
    # 初始化函数，用于初始化 Vectorize 对象的各个属性
    def __init__(self, pyfunc=np._NoValue, otypes=None, doc=None,
                 excluded=None, cache=False, signature=None):

        # 如果 pyfunc 参数不等于 np._NoValue，并且不可调用，则抛出 TypeError 异常
        if (pyfunc != np._NoValue) and (not callable(pyfunc)):
            # 错误信息被拆分成两部分以保持长度不超过 79 个字符
            part1 = "When used as a decorator, "
            part2 = "only accepts keyword arguments."
            raise TypeError(part1 + part2)

        # 将 pyfunc、cache 和 signature 参数赋给对象的属性
        self.pyfunc = pyfunc
        self.cache = cache
        self.signature = signature

        # 如果 pyfunc 不等于 np._NoValue 并且有 __name__ 属性，则将其赋给对象的 __name__ 属性
        if pyfunc != np._NoValue and hasattr(pyfunc, '__name__'):
            self.__name__ = pyfunc.__name__

        # 初始化缓存属性和文档属性
        self._ufunc = {}    # 用于缓存以提高默认性能
        self._doc = None
        self.__doc__ = doc

        # 如果 doc 参数为 None 并且 pyfunc 有 __doc__ 属性，则将其赋给对象的 __doc__ 属性
        if doc is None and hasattr(pyfunc, '__doc__'):
            self.__doc__ = pyfunc.__doc__
        else:
            self._doc = doc

        # 检查 otypes 参数的类型，根据不同情况赋值给对象的 otypes 属性
        if isinstance(otypes, str):
            for char in otypes:
                if char not in typecodes['All']:
                    raise ValueError("Invalid otype specified: %s" % (char,))
        elif iterable(otypes):
            otypes = [_get_vectorize_dtype(_nx.dtype(x)) for x in otypes]
        elif otypes is not None:
            raise ValueError("Invalid otype specification")
        self.otypes = otypes

        # 初始化 excluded 属性，如果 excluded 参数为 None，则设置为空集合
        if excluded is None:
            excluded = set()
        self.excluded = set(excluded)

        # 如果 signature 参数不为 None，则解析其核心维度并赋给对象的 _in_and_out_core_dims 属性
        if signature is not None:
            self._in_and_out_core_dims = _parse_gufunc_signature(signature)
        else:
            self._in_and_out_core_dims = None

    # 第二阶段初始化函数，设置对象的名称和文档属性
    def _init_stage_2(self, pyfunc, *args, **kwargs):
        self.__name__ = pyfunc.__name__  # 设置对象的 __name__ 属性为 pyfunc 的名称
        self.pyfunc = pyfunc  # 设置对象的 pyfunc 属性为传入的 pyfunc 参数
        if self._doc is None:
            self.__doc__ = pyfunc.__doc__  # 如果 _doc 属性为 None，则设置对象的 __doc__ 属性为 pyfunc 的文档
        else:
            self.__doc__ = self._doc  # 否则设置对象的 __doc__ 属性为 _doc 属性的值

    # 普通调用函数，用于正常调用向量化的 pyfunc，并根据排除项返回结果数组
    def _call_as_normal(self, *args, **kwargs):
        """
        Return arrays with the results of `pyfunc` broadcast (vectorized) over
        `args` and `kwargs` not in `excluded`.
        """
        excluded = self.excluded  # 获取对象的 excluded 属性

        # 如果 kwargs 和 excluded 都为空，则直接使用 self.pyfunc 和 args 进行调用
        if not kwargs and not excluded:
            func = self.pyfunc
            vargs = args
        else:
            # 否则，处理排除项和 kwargs，以及调整参数进行调用
            nargs = len(args)

            # 获取不在排除项内的 kwargs 的名称和位置索引
            names = [_n for _n in kwargs if _n not in excluded]
            inds = [_i for _i in range(nargs) if _i not in excluded]
            the_args = list(args)

            # 定义一个包装函数 func，用于处理传入的参数和 kwargs，并调用 self.pyfunc
            def func(*vargs):
                for _n, _i in enumerate(inds):
                    the_args[_i] = vargs[_n]
                kwargs.update(zip(names, vargs[len(inds):]))
                return self.pyfunc(*the_args, **kwargs)

            # 获取最终调用 func 所需的参数列表 vargs
            vargs = [args[_i] for _i in inds]
            vargs.extend([kwargs[_n] for _n in names])

        return self._vectorize_call(func=func, args=vargs)
    # 定义一个特殊方法 __call__，使得实例对象可以像函数一样被调用
    def __call__(self, *args, **kwargs):
        # 如果 self.pyfunc 是 np._NoValue，表示未初始化，执行第二阶段的初始化操作
        if self.pyfunc is np._NoValue:
            self._init_stage_2(*args, **kwargs)
            # 返回当前对象自身，以支持方法链式调用
            return self
        
        # 否则，以普通方式调用 _call_as_normal 方法，并返回其结果
        return self._call_as_normal(*args, **kwargs)
    def _get_ufunc_and_otypes(self, func, args):
        """Return (ufunc, otypes)."""
        # 如果参数 args 为空，则抛出 ValueError 异常
        if not args:
            raise ValueError('args can not be empty')

        # 如果已经设置了 otypes 属性，则使用已有的 otypes
        if self.otypes is not None:
            otypes = self.otypes

            # self._ufunc 是一个字典，其键是参数个数 len(args)，值是使用 frompyfunc 创建的 ufunc。
            # 当 func 不等于 self.pyfunc 或者当前参数个数 nin 不在 self._ufunc 中时，创建新的 ufunc。
            # self.pyfunc 为函数自身时，表示调用仅使用位置参数且没有参数被排除。
            nin = len(args)
            nout = len(self.otypes)
            if func is not self.pyfunc or nin not in self._ufunc:
                ufunc = frompyfunc(func, nin, nout)
            else:
                ufunc = None  # 我们将从 self._ufunc 中获取 ufunc
            if func is self.pyfunc:
                ufunc = self._ufunc.setdefault(nin, ufunc)
        else:
            # 通过调用函数 func 的第一个参数来获取输出的数量和类型
            args = [asarray(arg) for arg in args]
            if builtins.any(arg.size == 0 for arg in args):
                raise ValueError('cannot call `vectorize` on size 0 inputs '
                                 'unless `otypes` is set')

            inputs = [arg.flat[0] for arg in args]
            outputs = func(*inputs)

            # 性能提示：分析表明，对于简单的函数，这种包装几乎可以将执行时间翻倍。
            # 因此我们将其作为可选项。
            if self.cache:
                _cache = [outputs]

                def _func(*vargs):
                    if _cache:
                        return _cache.pop()
                    else:
                        return func(*vargs)
            else:
                _func = func

            if isinstance(outputs, tuple):
                nout = len(outputs)
            else:
                nout = 1
                outputs = (outputs,)

            # 确定输出的数据类型
            otypes = ''.join([asarray(outputs[_k]).dtype.char
                              for _k in range(nout)])

            # 性能提示：分析表明，创建 ufunc 与包装相比并不是显著的成本，因此似乎不值得尝试缓存这个过程。
            ufunc = frompyfunc(_func, len(args), nout)

        return ufunc, otypes
    def _vectorize_call(self, func, args):
        """Vectorized call to `func` over positional `args`."""
        # 如果存在签名信息，则使用带签名的向量化调用函数
        if self.signature is not None:
            res = self._vectorize_call_with_signature(func, args)
        # 如果没有参数，则直接调用 func 函数
        elif not args:
            res = func()
        else:
            # 获取ufunc和输出类型
            ufunc, otypes = self._get_ufunc_and_otypes(func=func, args=args)

            # 首先将参数转换为对象数组
            inputs = [asanyarray(a, dtype=object) for a in args]

            # 调用ufunc进行计算
            outputs = ufunc(*inputs)

            # 如果输出只有一个，将其转换为指定类型的数组
            if ufunc.nout == 1:
                res = asanyarray(outputs, dtype=otypes[0])
            else:
                # 如果有多个输出，则分别转换为指定类型的数组
                res = tuple([asanyarray(x, dtype=t) for x, t in zip(outputs, otypes)])
        # 返回计算结果
        return res
    # 对带有签名的位置参数进行向量化调用。
    """Vectorized call over positional arguments with a signature."""
    # 获取输入和输出的核心维度
    input_core_dims, output_core_dims = self._in_and_out_core_dims

    # 检查传入参数个数是否与输入核心维度的数量相符
    if len(args) != len(input_core_dims):
        raise TypeError('wrong number of positional arguments: '
                        'expected %r, got %r'
                        % (len(input_core_dims), len(args)))
    
    # 将所有参数转换为数组形式
    args = tuple(asanyarray(arg) for arg in args)

    # 解析输入维度并计算输入形状
    broadcast_shape, dim_sizes = _parse_input_dimensions(
        args, input_core_dims)
    input_shapes = _calculate_shapes(broadcast_shape, dim_sizes,
                                     input_core_dims)

    # 将每个参数广播到对应的形状
    args = [np.broadcast_to(arg, shape, subok=True)
            for arg, shape in zip(args, input_shapes)]

    outputs = None
    otypes = self.otypes
    nout = len(output_core_dims)

    # 在广播的形状上遍历
    for index in np.ndindex(*broadcast_shape):
        # 调用函数并获取结果
        results = func(*(arg[index] for arg in args))

        # 确定结果数量
        n_results = len(results) if isinstance(results, tuple) else 1

        # 检查输出结果的数量是否与预期一致
        if nout != n_results:
            raise ValueError(
                'wrong number of outputs from pyfunc: expected %r, got %r'
                % (nout, n_results))

        # 如果只有一个输出，将其转换为元组形式
        if nout == 1:
            results = (results,)

        # 如果尚未创建输出数组，根据结果创建数组
        if outputs is None:
            for result, core_dims in zip(results, output_core_dims):
                _update_dim_sizes(dim_sizes, result, core_dims)

            outputs = _create_arrays(broadcast_shape, dim_sizes,
                                     output_core_dims, otypes, results)

        # 将结果分配到输出数组中
        for output, result in zip(outputs, results):
            output[index] = result

    # 如果没有调用函数，根据输入创建空数组
    if outputs is None:
        if otypes is None:
            raise ValueError('cannot call `vectorize` on size 0 inputs '
                             'unless `otypes` is set')
        if builtins.any(dim not in dim_sizes
                        for dims in output_core_dims
                        for dim in dims):
            raise ValueError('cannot call `vectorize` with a signature '
                             'including new output dimensions on size 0 '
                             'inputs')
        outputs = _create_arrays(broadcast_shape, dim_sizes,
                                 output_core_dims, otypes)

    # 返回输出数组的第一个元素或整个数组
    return outputs[0] if nout == 1 else outputs
# 定义一个内部调度函数 _cov_dispatcher，用于决定传递给 cov 函数的参数
def _cov_dispatcher(m, y=None, rowvar=None, bias=None, ddof=None,
                    fweights=None, aweights=None, *, dtype=None):
    # 返回参数 m, y, fweights, aweights 的元组
    return (m, y, fweights, aweights)


# 使用 array_function_dispatch 装饰器将 _cov_dispatcher 注册为 cov 函数的分派函数
@array_function_dispatch(_cov_dispatcher)
# 定义计算协方差矩阵的函数 cov
def cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None,
        aweights=None, *, dtype=None):
    """
    Estimate a covariance matrix, given data and weights.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, :math:`X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element :math:`C_{ij}` is the covariance of
    :math:`x_i` and :math:`x_j`. The element :math:`C_{ii}` is the variance
    of :math:`x_i`.

    See the notes for an outline of the algorithm.

    Parameters
    ----------
    m : array_like
        A 1-D or 2-D array containing multiple variables and observations.
        Each row of `m` represents a variable, and each column a single
        observation of all those variables. Also see `rowvar` below.
    y : array_like, optional
        An additional set of variables and observations. `y` has the same form
        as that of `m`.
    rowvar : bool, optional
        If `rowvar` is True (default), then each row represents a
        variable, with observations in the columns. Otherwise, the relationship
        is transposed: each column represents a variable, while the rows
        contain observations.
    bias : bool, optional
        Default normalization (False) is by ``(N - 1)``, where ``N`` is the
        number of observations given (unbiased estimate). If `bias` is True,
        then normalization is by ``N``. These values can be overridden by using
        the keyword ``ddof`` in numpy versions >= 1.5.
    ddof : int, optional
        If not ``None`` the default value implied by `bias` is overridden.
        Note that ``ddof=1`` will return the unbiased estimate, even if both
        `fweights` and `aweights` are specified, and ``ddof=0`` will return
        the simple average. See the notes for the details. The default value
        is ``None``.

        .. versionadded:: 1.5
    fweights : array_like, int, optional
        1-D array of integer frequency weights; the number of times each
        observation vector should be repeated.

        .. versionadded:: 1.10
    aweights : array_like, optional
        1-D array of observation vector weights. These relative weights are
        typically large for observations considered "important" and smaller for
        observations considered less "important". If ``ddof=0`` the array of
        weights can be used to assign probabilities to observation vectors.

        .. versionadded:: 1.10
    dtype : data-type, optional
        Data-type of the result. By default, the return data-type will have
        at least `numpy.float64` precision.

        .. versionadded:: 1.20

    Returns
    -------
    out : ndarray
        The covariance matrix of the variables.

    See Also
    --------
    ```
    # 初始化权重变量为 None
    w = None
    # 如果给定了频率权重（fweights），则将其转换为浮点数的 NumPy 数组
    if fweights is not None:
        fweights = np.asarray(fweights, dtype=float)
        # 检查是否所有的频率权重都是整数
        if not np.all(fweights == np.around(fweights)):
            raise TypeError(
                "fweights must be integer")
        # 检查频率权重是否是一维的
        if fweights.ndim > 1:
            raise RuntimeError(
                "cannot handle multidimensional fweights")
        # 检查频率权重的长度与 X 的列数是否相同
        if fweights.shape[0] != X.shape[1]:
            raise RuntimeError(
                "incompatible numbers of samples and fweights")
        # 检查是否有任何负数的频率权重
        if any(fweights < 0):
            raise ValueError(
                "fweights cannot be negative")
        # 将权重赋值给 w
        w = fweights
    
    # 如果给定了分析权重（aweights），则将其转换为浮点数的 NumPy 数组
    if aweights is not None:
        aweights = np.asarray(aweights, dtype=float)
        # 检查分析权重是否是一维的
        if aweights.ndim > 1:
            raise RuntimeError(
                "cannot handle multidimensional aweights")
        # 检查分析权重的长度与 X 的列数是否相同
        if aweights.shape[0] != X.shape[1]:
            raise RuntimeError(
                "incompatible numbers of samples and aweights")
        # 检查是否有任何负数的分析权重
        if any(aweights < 0):
            raise ValueError(
                "aweights cannot be negative")
        # 如果之前未定义过 w，则将分析权重赋值给 w；否则将两者相乘更新 w
        if w is None:
            w = aweights
        else:
            w *= aweights
    
    # 计算加权平均值和加权和
    avg, w_sum = average(X, axis=1, weights=w, returned=True)
    w_sum = w_sum[0]

    # 确定归一化因子（degrees of freedom）
    if w is None:
        fact = X.shape[1] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof*sum(w*aweights)/w_sum
    
    # 如果归一化因子小于等于零，则发出警告并将其设为零
    if fact <= 0:
        warnings.warn("Degrees of freedom <= 0 for slice",
                      RuntimeWarning, stacklevel=2)
        fact = 0.0

    # 对 X 进行均值中心化
    X -= avg[:, None]
    # 根据是否有权重 w，计算 X 的转置乘积
    if w is None:
        X_T = X.T
    else:
        X_T = (X*w).T
    # 计算协方差矩阵
    c = dot(X, X_T.conj())
    # 对协方差矩阵进行归一化
    c *= np.true_divide(1, fact)
    # 去除结果中多余的维度
    return c.squeeze()
# 定义函数 _corrcoef_dispatcher，用于分派参数 x, y 给下一个函数
def _corrcoef_dispatcher(x, y=None, rowvar=None, bias=None, ddof=None, *,
                         dtype=None):
    # 直接返回参数 x, y，不做其他处理
    return (x, y)


# 使用 array_function_dispatch 装饰器，将 _corrcoef_dispatcher 函数应用于 corrcoef 函数
@array_function_dispatch(_corrcoef_dispatcher)
def corrcoef(x, y=None, rowvar=True, bias=np._NoValue, ddof=np._NoValue, *,
             dtype=None):
    """
    Return Pearson product-moment correlation coefficients.

    Please refer to the documentation for `cov` for more detail.  The
    relationship between the correlation coefficient matrix, `R`, and the
    covariance matrix, `C`, is

    .. math:: R_{ij} = \\frac{ C_{ij} } { \\sqrt{ C_{ii} C_{jj} } }

    The values of `R` are between -1 and 1, inclusive.

    Parameters
    ----------
    x : array_like
        A 1-D or 2-D array containing multiple variables and observations.
        Each row of `x` represents a variable, and each column a single
        observation of all those variables. Also see `rowvar` below.
    y : array_like, optional
        An additional set of variables and observations. `y` has the same
        shape as `x`.
    rowvar : bool, optional
        If `rowvar` is True (default), then each row represents a
        variable, with observations in the columns. Otherwise, the relationship
        is transposed: each column represents a variable, while the rows
        contain observations.
    bias : _NoValue, optional
        Has no effect, do not use.

        .. deprecated:: 1.10.0
    ddof : _NoValue, optional
        Has no effect, do not use.

        .. deprecated:: 1.10.0
    dtype : data-type, optional
        Data-type of the result. By default, the return data-type will have
        at least `numpy.float64` precision.

        .. versionadded:: 1.20

    Returns
    -------
    R : ndarray
        The correlation coefficient matrix of the variables.

    See Also
    --------
    cov : Covariance matrix

    Notes
    -----
    Due to floating point rounding the resulting array may not be Hermitian,
    the diagonal elements may not be 1, and the elements may not satisfy the
    inequality abs(a) <= 1. The real and imaginary parts are clipped to the
    interval [-1,  1] in an attempt to improve on that situation but is not
    much help in the complex case.

    This function accepts but discards arguments `bias` and `ddof`.  This is
    for backwards compatibility with previous versions of this function.  These
    arguments had no effect on the return values of the function and can be
    safely ignored in this and previous versions of numpy.

    Examples
    --------
    In this example we generate two random arrays, ``xarr`` and ``yarr``, and
    compute the row-wise and column-wise Pearson correlation coefficients,
    ``R``. Since ``rowvar`` is  true by  default, we first find the row-wise
    Pearson correlation coefficients between the variables of ``xarr``.

    >>> import numpy as np
    >>> rng = np.random.default_rng(seed=42)
    >>> xarr = rng.random((3, 3))
    >>> xarr
    """
    如果存在偏差 `bias` 或自由度差 `ddof` 不等于 `np._NoValue`：
        # 发出警告，因为 `bias` 和 `ddof` 已不再起作用并且已被弃用
        warnings.warn('bias and ddof have no effect and are deprecated',
                      DeprecationWarning, stacklevel=2)
    计算协方差矩阵 `c`，使用 `cov` 函数，传入参数 `x`, `y`, `rowvar`，还有 `dtype`
    尝试获取 `c` 的对角线元素 `d`
    如果出现 `ValueError` 异常：
        # 如果是标量协方差，返回 `c` 除以 `c`，得到 NaN 或者 1 的结果
        return c / c
    计算标准差 `stddev`，对应于 `d` 的实部的平方根
    将 `c` 按照 `stddev` 的列向量进行归一化
    将 `c` 按照 `stddev` 的行向量进行归一化

    # 对于实部和虚部进行截断，使得其取值范围在 [-1, 1] 之间
    Clip real and imaginary parts to [-1, 1].  This does not guarantee
    abs(a[i,j]) <= 1 for complex arrays, but is the best we can do without
    excessive work.
    """
    # 将复数数组 c 的实部裁剪到 [-1, 1] 的范围内，结果存回 c 的实部
    np.clip(c.real, -1, 1, out=c.real)
    # 检查数组 c 是否包含复数对象（即是否为复数数组）
    if np.iscomplexobj(c):
        # 如果是复数数组，将其虚部裁剪到 [-1, 1] 的范围内，结果存回 c 的虚部
        np.clip(c.imag, -1, 1, out=c.imag)
    
    # 返回裁剪后的复数数组 c
    return c
# 设置模块名称为 'numpy'，用于将函数加入 numpy 模块中
@set_module('numpy')
# 定义 Blackman 窗口函数，用于信号处理中的平滑处理
def blackman(M):
    """
    Return the Blackman window.

    The Blackman window is a taper formed by using the first three
    terms of a summation of cosines. It was designed to have close to the
    minimal leakage possible.  It is close to optimal, only slightly worse
    than a Kaiser window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.

    Returns
    -------
    out : ndarray
        The window, with the maximum value normalized to one (the value one
        appears only if the number of samples is odd).

    See Also
    --------
    bartlett, hamming, hanning, kaiser

    Notes
    -----
    The Blackman window is defined as

    .. math::  w(n) = 0.42 - 0.5 \\cos(2\\pi n/M) + 0.08 \\cos(4\\pi n/M)

    Most references to the Blackman window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  It is also known as an apodization (which means
    "removing the foot", i.e. smoothing discontinuities at the beginning
    and end of the sampled signal) or tapering function. It is known as a
    "near optimal" tapering function, almost as good (by some measures)
    as the kaiser window.

    References
    ----------
    Blackman, R.B. and Tukey, J.W., (1958) The measurement of power spectra,
    Dover Publications, New York.

    Oppenheim, A.V., and R.W. Schafer. Discrete-Time Signal Processing.
    Upper Saddle River, NJ: Prentice-Hall, 1999, pp. 468-471.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> np.blackman(12)
    array([-1.38777878e-17,   3.26064346e-02,   1.59903635e-01, # may vary
            4.14397981e-01,   7.36045180e-01,   9.67046769e-01,
            9.67046769e-01,   7.36045180e-01,   4.14397981e-01,
            1.59903635e-01,   3.26064346e-02,  -1.38777878e-17])

    Plot the window and the frequency response.

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from numpy.fft import fft, fftshift
        window = np.blackman(51)
        plt.plot(window)
        plt.title("Blackman window")
        plt.ylabel("Amplitude")
        plt.xlabel("Sample")
        plt.show()  # doctest: +SKIP

        plt.figure()
        A = fft(window, 2048) / 25.5
        mag = np.abs(fftshift(A))
        freq = np.linspace(-0.5, 0.5, len(A))
        with np.errstate(divide='ignore', invalid='ignore'):
            response = 20 * np.log10(mag)
        response = np.clip(response, -100, 100)
        plt.plot(freq, response)
        plt.title("Frequency response of Blackman window")
        plt.ylabel("Magnitude [dB]")
        plt.xlabel("Normalized frequency [cycles per sample]")
        plt.axis('tight')
        plt.show()

    """
    # 确保 M 至少为 float64 类型的 0.0。M 应为整数，但转换为双精度浮点数是安全的。
    values = np.array([0.0, M])
    # 取得输入数组的第二个元素并赋值给变量 M
    M = values[1]

    # 如果 M 小于 1，返回一个空数组，数据类型与输入数组相同
    if M < 1:
        return array([], dtype=values.dtype)
    
    # 如果 M 等于 1，返回一个包含一个元素的数组，元素值为 1，数据类型与输入数组相同
    if M == 1:
        return ones(1, dtype=values.dtype)
    
    # 生成一个等差数组 n，范围从 1-M 到 M-1，步长为 2
    n = arange(1-M, M, 2)
    
    # 计算并返回一个窗函数的值，窗函数由三部分组成：0.42，0.5*cos(pi*n/(M-1))，0.08*cos(2.0*pi*n/(M-1))
    return 0.42 + 0.5*cos(pi*n/(M-1)) + 0.08*cos(2.0*pi*n/(M-1))
# 设置函数的模块信息为'numpy'
@set_module('numpy')
# 定义函数bartlett，返回Bartlett窗口函数
def bartlett(M):
    """
    Return the Bartlett window.

    The Bartlett window is very similar to a triangular window, except
    that the end points are at zero.  It is often used in signal
    processing for tapering a signal, without generating too much
    ripple in the frequency domain.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an
        empty array is returned.

    Returns
    -------
    out : array
        The triangular window, with the maximum value normalized to one
        (the value one appears only if the number of samples is odd), with
        the first and last samples equal to zero.

    See Also
    --------
    blackman, hamming, hanning, kaiser

    Notes
    -----
    The Bartlett window is defined as

    .. math:: w(n) = \\frac{2}{M-1} \\left(
              \\frac{M-1}{2} - \\left|n - \\frac{M-1}{2}\\right|
              \\right)

    Most references to the Bartlett window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  Note that convolution with this window produces linear
    interpolation.  It is also known as an apodization (which means "removing
    the foot", i.e. smoothing discontinuities at the beginning and end of the
    sampled signal) or tapering function. The Fourier transform of the
    Bartlett window is the product of two sinc functions. Note the excellent
    discussion in Kanasewich [2]_.

    References
    ----------
    .. [1] M.S. Bartlett, "Periodogram Analysis and Continuous Spectra",
           Biometrika 37, 1-16, 1950.
    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics",
           The University of Alberta Press, 1975, pp. 109-110.
    .. [3] A.V. Oppenheim and R.W. Schafer, "Discrete-Time Signal
           Processing", Prentice-Hall, 1999, pp. 468-471.
    .. [4] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function
    .. [5] W.H. Press,  B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling,
           "Numerical Recipes", Cambridge University Press, 1986, page 429.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> np.bartlett(12)
    array([ 0.        ,  0.18181818,  0.36363636,  0.54545455,  0.72727273, # may vary
            0.90909091,  0.90909091,  0.72727273,  0.54545455,  0.36363636,
            0.18181818,  0.        ])

    Plot the window and its frequency response (requires SciPy and matplotlib).
    """
    # 确保至少使用 float64 类型的 0.0。M 应为整数，但转换为双精度浮点数在一定范围内是安全的。
    values = np.array([0.0, M])
    # 将 values 数组的第二个元素赋给 M
    M = values[1]
    
    # 如果 M 小于 1，则返回一个空数组，其数据类型为 values 的数据类型
    if M < 1:
        return array([], dtype=values.dtype)
    # 如果 M 等于 1，则返回一个包含一个元素的数组，元素的数据类型与 values 相同
    if M == 1:
        return ones(1, dtype=values.dtype)
    # 生成一个范围从 1-M 到 M 的步长为 2 的整数数组
    n = arange(1-M, M, 2)
    # 根据条件选择数组中的元素，如果元素小于等于 0，则返回 1 + n/(M-1)，否则返回 1 - n/(M-1)
    return where(less_equal(n, 0), 1 + n/(M-1), 1 - n/(M-1))
# 设置函数的模块为 'numpy'，用于标识这是一个 numpy 模块的函数
@set_module('numpy')
# 定义 Hanning 窗口函数，接受一个整数 M 作为参数
def hanning(M):
    """
    Return the Hanning window.

    The Hanning window is a taper formed by using a weighted cosine.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an
        empty array is returned.

    Returns
    -------
    out : ndarray, shape(M,)
        The window, with the maximum value normalized to one (the value
        one appears only if `M` is odd).

    See Also
    --------
    bartlett, blackman, hamming, kaiser

    Notes
    -----
    The Hanning window is defined as

    .. math::  w(n) = 0.5 - 0.5\\cos\\left(\\frac{2\\pi{n}}{M-1}\\right)
               \\qquad 0 \\leq n \\leq M-1

    The Hanning was named for Julius von Hann, an Austrian meteorologist.
    It is also known as the Cosine Bell. Some authors prefer that it be
    called a Hann window, to help avoid confusion with the very similar
    Hamming window.

    Most references to the Hanning window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  It is also known as an apodization (which means
    "removing the foot", i.e. smoothing discontinuities at the beginning
    and end of the sampled signal) or tapering function.

    References
    ----------
    .. [1] Blackman, R.B. and Tukey, J.W., (1958) The measurement of power
           spectra, Dover Publications, New York.
    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics",
           The University of Alberta Press, 1975, pp. 106-108.
    .. [3] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function
    .. [4] W.H. Press,  B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling,
           "Numerical Recipes", Cambridge University Press, 1986, page 425.

    Examples
    --------
    >>> np.hanning(12)
    array([0.        , 0.07937323, 0.29229249, 0.57115742, 0.82743037,
           0.97974649, 0.97974649, 0.82743037, 0.57115742, 0.29229249,
           0.07937323, 0.        ])

    Plot the window and its frequency response.

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from numpy.fft import fft, fftshift
        window = np.hanning(51)
        plt.plot(window)
        plt.title("Hann window")
        plt.ylabel("Amplitude")
        plt.xlabel("Sample")
        plt.show()

        plt.figure()
        A = fft(window, 2048) / 25.5
        mag = np.abs(fftshift(A))
        freq = np.linspace(-0.5, 0.5, len(A))
        with np.errstate(divide='ignore', invalid='ignore'):
            response = 20 * np.log10(mag)
        response = np.clip(response, -100, 100)
        plt.plot(freq, response)
        plt.title("Frequency response of the Hann window")
        plt.ylabel("Magnitude [dB]")
        plt.xlabel("Normalized frequency [cycles per sample]")
        plt.axis('tight')
        plt.show()

    """
    # 确保通过 [0.0, M] 数组至少有 float64 类型的数据。M 应为整数，但转换为双精度浮点数对于范围是安全的。
    values = np.array([0.0, M])
    # 将 M 设定为数组中的第二个元素
    M = values[1]

    # 如果 M 小于 1，则返回一个空数组，其数据类型为 values 的数据类型
    if M < 1:
        return array([], dtype=values.dtype)
    # 如果 M 等于 1，则返回一个包含一个元素的数组，元素类型为 values 的数据类型
    if M == 1:
        return ones(1, dtype=values.dtype)
    # 生成一个从 1-M 到 M-1（不包括 M-1）的等差数组，步长为 2
    n = arange(1-M, M, 2)
    # 返回一个数组，计算每个元素的余弦值，再加上 0.5，乘以 0.5
    return 0.5 + 0.5*cos(pi*n/(M-1))
# 设置函数的模块为 'numpy'，用于标记该函数是与 numpy 模块相关联的
@set_module('numpy')
# 定义 Hamming 窗口函数
def hamming(M):
    """
    Return the Hamming window.

    The Hamming window is a taper formed by using a weighted cosine.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an
        empty array is returned.

    Returns
    -------
    out : ndarray
        The window, with the maximum value normalized to one (the value
        one appears only if the number of samples is odd).

    See Also
    --------
    bartlett, blackman, hanning, kaiser

    Notes
    -----
    The Hamming window is defined as

    .. math::  w(n) = 0.54 - 0.46\\cos\\left(\\frac{2\\pi{n}}{M-1}\\right)
               \\qquad 0 \\leq n \\leq M-1

    The Hamming was named for R. W. Hamming, an associate of J. W. Tukey
    and is described in Blackman and Tukey. It was recommended for
    smoothing the truncated autocovariance function in the time domain.
    Most references to the Hamming window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  It is also known as an apodization (which means
    "removing the foot", i.e. smoothing discontinuities at the beginning
    and end of the sampled signal) or tapering function.

    References
    ----------
    .. [1] Blackman, R.B. and Tukey, J.W., (1958) The measurement of power
           spectra, Dover Publications, New York.
    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics", The
           University of Alberta Press, 1975, pp. 109-110.
    .. [3] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function
    .. [4] W.H. Press,  B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling,
           "Numerical Recipes", Cambridge University Press, 1986, page 425.

    Examples
    --------
    >>> np.hamming(12)
    array([ 0.08      ,  0.15302337,  0.34890909,  0.60546483,  0.84123594, # may vary
            0.98136677,  0.98136677,  0.84123594,  0.60546483,  0.34890909,
            0.15302337,  0.08      ])

    Plot the window and the frequency response.

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from numpy.fft import fft, fftshift
        window = np.hamming(51)
        plt.plot(window)
        plt.title("Hamming window")
        plt.ylabel("Amplitude")
        plt.xlabel("Sample")
        plt.show()

        plt.figure()
        A = fft(window, 2048) / 25.5
        mag = np.abs(fftshift(A))
        freq = np.linspace(-0.5, 0.5, len(A))
        response = 20 * np.log10(mag)
        response = np.clip(response, -100, 100)
        plt.plot(freq, response)
        plt.title("Frequency response of Hamming window")
        plt.ylabel("Magnitude [dB]")
        plt.xlabel("Normalized frequency [cycles per sample]")
        plt.axis('tight')
        plt.show()

    """
    # 确保通过 0.0 至少是 float64 类型。M 应该是整数，但转换为双精度浮点数对于一定范围内是安全的。
    # 创建一个包含两个元素的NumPy数组，分别为0.0和M
    values = np.array([0.0, M])
    # 从数组values中获取第二个元素的值，赋给变量M
    M = values[1]
    
    # 如果M小于1，返回一个空的NumPy数组，其数据类型与values的dtype相同
    if M < 1:
        return array([], dtype=values.dtype)
    # 如果M等于1，返回一个包含一个元素的NumPy数组，元素值为1，数据类型与values的dtype相同
    if M == 1:
        return ones(1, dtype=values.dtype)
    
    # 创建一个包含从1-M到M-1，步长为2的整数数组，赋给变量n
    n = arange(1-M, M, 2)
    # 返回一个由数学公式计算得出的NumPy数组，数组元素由cos函数计算得出
    return 0.54 + 0.46*cos(pi*n/(M-1))
# Coefficients for the approximation of the modified Bessel function of the first kind, order 0,
# using Clenshaw's recurrence formula. These coefficients are part of the Cephes Mathematical Library.
_i0A = [
    -4.41534164647933937950E-18,  # Coefficients for small x
    3.33079451882223809783E-17,
    -2.43127984654795469359E-16,
    1.71539128555513303061E-15,
    -1.16853328779934516808E-14,
    7.67618549860493561688E-14,
    -4.85644678311192946090E-13,
    2.95505266312963983461E-12,
    -1.72682629144155570723E-11,
    9.67580903537323691224E-11,
    -5.18979560163526290666E-10,
    2.65982372468238665035E-9,
    -1.30002500998624804212E-8,
    6.04699502254191894932E-8,
    -2.67079385394061173391E-7,
    1.11738753912010371815E-6,
    -4.41673835845875056359E-6,
    1.64484480707288970893E-5,
    -5.75419501008210370398E-5,
    1.88502885095841655729E-4,
    -5.76375574538582365885E-4,
    1.63947561694133579842E-3,
    -4.32430999505057594430E-3,
    1.05464603945949983183E-2,
    -2.37374148058994688156E-2,
    4.93052842396707084878E-2,
    -9.49010970480476444210E-2,
    1.71620901522208775349E-1,
    -3.04682672343198398683E-1,
    6.76795274409476084995E-1
    ]

_i0B = [
    -7.23318048787475395456E-18,  # Coefficients for large x
    -4.83050448594418207126E-18,
    4.46562142029675999901E-17,
    3.46122286769746109310E-17,
    -2.82762398051658348494E-16,
    -3.42548561967721913462E-16,
    1.77256013305652638360E-15,
    3.81168066935262242075E-15,
    -9.55484669882830764870E-15,
    -4.15056934728722208663E-14,
    1.54008621752140982691E-14,
    3.85277838274214270114E-13,
    7.18012445138366623367E-13,
    -1.79417853150680611778E-12,
    -1.32158118404477131188E-11,
    -3.14991652796324136454E-11,
    1.18891471078464383424E-11,
    4.94060238822496958910E-10,
    3.39623202570838634515E-9,
    2.26666899049817806459E-8,
    2.04891858946906374183E-7,
    2.89137052083475648297E-6,
    6.88975834691682398426E-5,
    3.36911647825569408990E-3,
    8.04490411014108831608E-1
    ]

def _chbevl(x, vals):
    # Evaluate a Chebyshev series at a point x using Horner's method
    b0 = vals[0]
    b1 = 0.0

    # Iterate through the series coefficients and compute the series value
    for i in range(1, len(vals)):
        b2 = b1
        b1 = b0
        b0 = x * b1 - b2 + vals[i]

    # Return the evaluated series value
    return 0.5 * (b0 - b2)


def _i0_1(x):
    # Compute the modified Bessel function of the first kind, order 0, for small x
    return exp(x) * _chbevl(x/2.0-2, _i0A)


def _i0_2(x):
    # Compute the modified Bessel function of the first kind, order 0, for large x
    return exp(x) * _chbevl(32.0/x - 2.0, _i0B) / sqrt(x)


def _i0_dispatcher(x):
    # Function dispatcher for the modified Bessel function of the first kind, order 0
    return (x,)


@array_function_dispatch(_i0_dispatcher)
def i0(x):
    """
    Modified Bessel function of the first kind, order 0.

    Usually denoted :math:`I_0`.

    Parameters
    ----------
    x : array_like of float
        Argument of the Bessel function.

    Returns
    -------
    out : ndarray, shape = x.shape, dtype = float
        The modified Bessel function evaluated at each of the elements of `x`.

    See Also
    --------
    scipy.special.i0, scipy.special.iv, scipy.special.ive

    Notes
    -----
    The scipy implementation is recommended over this function: it is a
    proper ufunc written in C, and more than an order of magnitude faster.

    We use the algorithm published by Clenshaw [1]_ and referenced by
    Abramowitz and Stegun [2]_, for which the function domain is
    """
    # The actual implementation of the modified Bessel function of the first kind, order 0,
    # based on Clenshaw's recurrence formula for a specified range of domain
    pass
    """
    Compute the modified Bessel function of the first kind, order 0, for each
    element in the input array x using Chebyshev polynomial expansions.

    Parameters
    ----------
    x : array_like
        Input array. If not already floating point, it will be converted to float.
        Complex values are not supported.

    Returns
    -------
    array_like
        Array of modified Bessel function values for each element in x.

    Raises
    ------
    TypeError
        If x contains complex values.

    Notes
    -----
    The computation is partitioned into intervals [0,8] and (8,inf). Chebyshev
    polynomial expansions are used in each interval. Relative error over the
    domain [0,30] using IEEE arithmetic is documented [3]_ with peak relative
    error of 5.8e-16 and root mean square (rms) error of 1.4e-16 (n = 30000).

    References
    ----------
    .. [1] C. W. Clenshaw, "Chebyshev series for mathematical functions", in
           *National Physical Laboratory Mathematical Tables*, vol. 5, London:
           Her Majesty's Stationery Office, 1962.
    .. [2] M. Abramowitz and I. A. Stegun, *Handbook of Mathematical
           Functions*, 10th printing, New York: Dover, 1964, pp. 379.
           https://personal.math.ubc.ca/~cbm/aands/page_379.htm
    .. [3] https://metacpan.org/pod/distribution/Math-Cephes/lib/Math/Cephes.pod#i0:-Modified-Bessel-function-of-order-zero

    Examples
    --------
    >>> np.i0(0.)
    array(1.0)
    >>> np.i0([0, 1, 2, 3])
    array([1.        , 1.26606588, 2.2795853 , 4.88079259])

    """
    x = np.asanyarray(x)  # Convert input x to a numpy array, handling any type
    if x.dtype.kind == 'c':
        raise TypeError("i0 not supported for complex values")  # Raise error if x contains complex values
    if x.dtype.kind != 'f':
        x = x.astype(float)  # Convert x to float if it's not already of float type
    x = np.abs(x)  # Compute the absolute values of elements in x
    return piecewise(x, [x <= 8.0], [_i0_1, _i0_2])  # Apply piecewise function with conditions and functions _i0_1, _i0_2
# 使用numpy模块来设置本函数的模块名称为'numpy'
@set_module('numpy')
# 定义一个Kaiser窗口生成函数，接受两个参数：M表示输出窗口中的点数，如果为零或更小，则返回一个空数组；beta表示窗口的形状参数
def kaiser(M, beta):
    """
    返回Kaiser窗口。

    Kaiser窗口是通过使用贝塞尔函数形成的锥形窗口。

    Parameters
    ----------
    M : int
        输出窗口中的点数。如果为零或更小，则返回一个空数组。
    beta : float
        窗口的形状参数。

    Returns
    -------
    out : array
        窗口，最大值归一化为1（仅当样本数为奇数时，值为1）。

    See Also
    --------
    bartlett, blackman, hamming, hanning

    Notes
    -----
    Kaiser窗口定义为

    .. math::  w(n) = I_0\\left( \\beta \\sqrt{1-\\frac{4n^2}{(M-1)^2}}
               \\right)/I_0(\\beta)

    其中

    .. math:: \\quad -\\frac{M-1}{2} \\leq n \\leq \\frac{M-1}{2},

    这里的 :math:`I_0` 是修正零阶贝塞尔函数。

    Kaiser窗口以Jim Kaiser命名，他发现了基于贝塞尔函数的DPSS窗口的简单近似。Kaiser窗口非常接近数字椭球序列DPSS窗口或Slepian窗口，后者在主瓣中最大化能量与总能量的比率。

    通过变化beta参数，Kaiser窗口可以近似许多其他窗口。

    ====  =======================
    beta  窗口形状
    ====  =======================
    0     矩形窗口
    5     类似于Hamming窗口
    6     类似于Hanning窗口
    8.6   类似于Blackman窗口
    ====  =======================

    beta值为14可能是一个良好的起始点。请注意，随着beta变大，窗口变窄，因此样本数需要足够大，以便对越来越窄的峰进行采样，否则会返回NaN。

    Kaiser窗口的大多数参考文献来自信号处理文献，用作许多窗函数之一，用于平滑数值。它也被称为拔顶（apodization），意思是“去除脚部”，即平滑采样信号起始和结束处的不连续性或锥度函数。

    References
    ----------
    .. [1] J. F. Kaiser, "Digital Filters" - Ch 7 in "Systems analysis by
           digital computer", Editors: F.F. Kuo and J.F. Kaiser, p 218-285.
           John Wiley and Sons, New York, (1966).
    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics", The
           University of Alberta Press, 1975, pp. 177-178.
    .. [3] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    """
    # 确保至少使用 float64，通过 0.0。M 应为整数，但转换为双精度对于范围是安全的。
    # （使用简化的 result_type 和 0.0 强制类型化。result_type 不/少受顺序影响，但这主要对整数有影响。）
    values = np.array([0.0, M, beta])
    # 从 values 数组中取出 M 和 beta 的值
    M = values[1]
    beta = values[2]

    # 如果 M 等于 1，则返回一个包含一个元素的数组，元素类型与 values 的 dtype 一致
    if M == 1:
        return np.ones(1, dtype=values.dtype)
    
    # 生成一个从 0 到 M-1 的整数数组
    n = arange(0, M)
    # 计算 alpha 值
    alpha = (M-1)/2.0
    
    # 返回 Kaiser 函数的计算结果
    return i0(beta * sqrt(1-((n-alpha)/alpha)**2.0))/i0(beta)
    """
# 定义一个分发函数 _sinc_dispatcher，用于接收参数 x 并返回元组 (x,)
def _sinc_dispatcher(x):
    return (x,)

# 使用装饰器 array_function_dispatch 包装 sinc 函数，以便处理不同类型的输入 x
@array_function_dispatch(_sinc_dispatcher)
def sinc(x):
    r"""
    返回标准化的 sinc 函数。

    sinc 函数对于任何参数 x != 0 等于 sin(π x)/π x。当 x = 0 时，sinc(0) 的极限值为 1，
    使得 sinc 不仅在所有点上连续，而且具有无限可微性。

    .. note::

        注意在定义中使用的 π 作为归一化因子。这是信号处理中最常用的定义方式。
        使用 sinc(x / np.pi) 可以得到未归一化的 sinc 函数 sin(x)/x，这在数学中更为常见。

    Parameters
    ----------
    x : ndarray
        要计算 sinc(x) 的值数组（可能是多维的）。

    Returns
    -------
    out : ndarray
        具有与输入相同形状的 sinc(x)。

    Notes
    -----
    sinc 的名称源自 "sine cardinal" 或 "sinus cardinalis"。

    sinc 函数在各种信号处理应用中使用，包括抗混叠、Lanczos 重采样滤波器的构建和插值。

    References
    ----------
    .. [1] Weisstein, Eric W. "Sinc Function." From MathWorld--A Wolfram Web
           Resource. https://mathworld.wolfram.com/SincFunction.html
    .. [2] Wikipedia, "Sinc function",
           https://en.wikipedia.org/wiki/Sinc_function

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-4, 4, 41)
    >>> np.sinc(x)
     array([-3.89804309e-17,  -4.92362781e-02,  -8.40918587e-02, # may vary
            -8.90384387e-02,  -5.84680802e-02,   3.89804309e-17,
            6.68206631e-02,   1.16434881e-01,   1.26137788e-01,
            8.50444803e-02,  -3.89804309e-17,  -1.03943254e-01,
            -1.89206682e-01,  -2.16236208e-01,  -1.55914881e-01,
            3.89804309e-17,   2.33872321e-01,   5.04551152e-01,
            7.56826729e-01,   9.35489284e-01,   1.00000000e+00,
            9.35489284e-01,   7.56826729e-01,   5.04551152e-01,
            2.33872321e-01,   3.89804309e-17,  -1.55914881e-01,
           -2.16236208e-01,  -1.89206682e-01,  -1.03943254e-01,
           -3.89804309e-17,   8.50444803e-02,   1.26137788e-01,
            1.16434881e-01,   6.68206631e-02,   3.89804309e-17,
            -5.84680802e-02,  -8.90384387e-02,  -8.40918587e-02,
            -4.92362781e-02,  -3.89804309e-17])

    >>> plt.plot(x, np.sinc(x))
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.title("Sinc Function")
    Text(0.5, 1.0, 'Sinc Function')
    >>> plt.ylabel("Amplitude")
    Text(0, 0.5, 'Amplitude')
    >>> plt.xlabel("X")
    Text(0.5, 0, 'X')
    >>> plt.show()

    """
    # 将输入 x 转换为任意数组
    x = np.asanyarray(x)
    # 计算 y = π * x，但对于 x == 0，使用 1.0e-20 代替避免除以零错误
    y = pi * where(x == 0, 1.0e-20, x)
    # 返回 sin(y)/y 作为 sinc(x) 的计算结果
    return sin(y)/y
# 定义一个内部函数，用于在不支持扩展轴的函数中调用 `func`，并将 `a` 作为第一个参数传递进去
# 交换轴以使用扩展轴，返回结果及将轴维度设置为1后的 `a` 的形状
def _ureduce(a, func, keepdims=False, **kwargs):
    # 将输入转换为任意数组
    a = np.asanyarray(a)
    # 获取轴参数，如果未指定，默认为 None
    axis = kwargs.get('axis', None)
    # 获取输出数组参数，如果未指定，默认为 None
    out = kwargs.get('out', None)

    # 如果 keepdims 为 np._NoValue，则将其设为 False
    if keepdims is np._NoValue:
        keepdims = False

    # 获取数组的维度数
    nd = a.ndim

    # 如果指定了轴参数
    if axis is not None:
        # 规范化轴元组
        axis = _nx.normalize_axis_tuple(axis, nd)

        # 如果 keepdims 为 True
        if keepdims:
            # 如果指定了输出数组，则根据轴参数将其相应位置设为 0，其他位置保持不变
            if out is not None:
                index_out = tuple(
                    0 if i in axis else slice(None) for i in range(nd))
                kwargs['out'] = out[(Ellipsis, ) + index_out]

        # 如果只有一个轴需要处理
        if len(axis) == 1:
            kwargs['axis'] = axis[0]
        else:
            # 计算出需要保留的轴
            keep = set(range(nd)) - set(axis)
            nkeep = len(keep)
            # 将不需要缩减的轴交换到前面
            for i, s in enumerate(sorted(keep)):
                a = a.swapaxes(i, s)
            # 合并被缩减的轴
            a = a.reshape(a.shape[:nkeep] + (-1,))
            kwargs['axis'] = -1
    else:
        # 如果 keepdims 为 True
        if keepdims:
            # 如果指定了输出数组，则将其设为全零数组
            if out is not None:
                index_out = (0, ) * nd
                kwargs['out'] = out[(Ellipsis, ) + index_out]

    # 调用 func 函数计算结果
    r = func(a, **kwargs)

    # 如果指定了输出数组，则返回输出数组
    if out is not None:
        return out

    # 如果 keepdims 为 True
    if keepdims:
        # 根据轴参数设定结果数组的形状
        if axis is None:
            index_r = (np.newaxis, ) * nd
        else:
            index_r = tuple(
                np.newaxis if i in axis else slice(None)
                for i in range(nd))
        r = r[(Ellipsis, ) + index_r]

    # 返回计算得到的结果
    return r


# 定义 `_median_dispatcher` 函数，返回输入数组 `a` 和输出数组 `out` 的元组
def _median_dispatcher(
        a, axis=None, out=None, overwrite_input=None, keepdims=None):
    return (a, out)


# 使用 `_median_dispatcher` 作为装饰器，为 `median` 函数分派调度器
@array_function_dispatch(_median_dispatcher)
# 定义 `median` 函数，计算沿指定轴的中位数
def median(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    """
    Compute the median along the specified axis.

    Returns the median of the array elements.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    """
    axis : {int, sequence of int, None}, optional
        # 指定计算中位数的轴或轴。默认情况下，axis=None，将沿数组的展平版本计算中位数。

        .. versionadded:: 1.9.0
        # 版本新增功能：1.9.0

        如果指定了一组轴，数组首先沿指定的轴展平，然后在展平的轴上计算中位数。

    out : ndarray, optional
        # 可选参数，用于存放结果的替代输出数组。它必须与期望输出具有相同的形状和缓冲区长度，但如果需要，输出的类型将被强制转换。

    overwrite_input : bool, optional
       # 如果为True，则允许使用输入数组 `a` 的内存进行计算。调用 `median` 后，输入数组将被修改。当不需要保留输入数组内容时，这将节省内存。将输入视为未定义，但可能已完全或部分排序。默认为False。如果 `overwrite_input` 是 ``True`` 并且 `a` 不是 `ndarray`，则会引发错误。

    keepdims : bool, optional
        # 如果设置为True，则在结果中保留被减少的轴作为大小为一的维度。使用此选项，结果将正确广播到原始的 `arr`。

        .. versionadded:: 1.9.0
        # 版本新增功能：1.9.0

    Returns
    -------
    median : ndarray
        # 包含结果的新数组。如果输入包含小于 ``float64`` 的整数或浮点数，则输出数据类型为 ``np.float64``。否则，输出的数据类型与输入相同。如果指定了 `out`，则返回该数组。

    See Also
    --------
    mean, percentile

    Notes
    -----
    给定长度为 ``N`` 的向量 ``V``，其中位数是排序副本 ``V_sorted`` 的中间值，即 ``V_sorted[(N-1)/2]``（当 ``N`` 为奇数时），或是 ``V_sorted`` 的两个中间值的平均值（当 ``N`` 为偶数时）。

    Examples
    --------
    >>> a = np.array([[10, 7, 4], [3, 2, 1]])
    >>> a
    array([[10,  7,  4],
           [ 3,  2,  1]])
    >>> np.median(a)
    np.float64(3.5)
    >>> np.median(a, axis=0)
    array([6.5, 4.5, 2.5])
    >>> np.median(a, axis=1)
    array([7.,  2.])
    >>> np.median(a, axis=(0, 1))
    np.float64(3.5)
    >>> m = np.median(a, axis=0)
    >>> out = np.zeros_like(m)
    >>> np.median(a, axis=0, out=m)
    array([6.5,  4.5,  2.5])
    >>> m
    array([6.5,  4.5,  2.5])
    >>> b = a.copy()
    >>> np.median(b, axis=1, overwrite_input=True)
    array([7.,  2.])
    >>> assert not np.all(a==b)
    >>> b = a.copy()
    >>> np.median(b, axis=None, overwrite_input=True)
    np.float64(3.5)
    >>> assert not np.all(a==b)
    # 调用 _ureduce 函数，并传递相应的参数
    return _ureduce(a, func=_median, keepdims=keepdims, axis=axis, out=out,
                    overwrite_input=overwrite_input)
# 无法合理地使用百分位数实现中位数计算，因为我们必须调用均值以避免破坏 astropy 库的功能
def _median(a, axis=None, out=None, overwrite_input=False):
    # 将输入转换为 NumPy 的数组形式
    a = np.asanyarray(a)

    # 设置分区索引
    if axis is None:
        sz = a.size
    else:
        sz = a.shape[axis]
    if sz % 2 == 0:
        szh = sz // 2
        kth = [szh - 1, szh]
    else:
        kth = [(sz - 1) // 2]

    # 需要检查 NaN 值（目前 'M' 实际上不起作用）
    supports_nans = np.issubdtype(a.dtype, np.inexact) or a.dtype.kind in 'Mm'
    if supports_nans:
        kth.append(-1)

    # 如果选择覆盖输入数组
    if overwrite_input:
        if axis is None:
            part = a.ravel()
            part.partition(kth)
        else:
            a.partition(kth, axis=axis)
            part = a
    else:
        part = partition(a, kth, axis=axis)  # 使用分区函数对数组进行分区操作

    # 处理零维数组的情况
    if part.shape == ():
        return part.item()  # 返回零维数组的元素值
    if axis is None:
        axis = 0

    # 设置索引器以获取中位数位置的切片
    indexer = [slice(None)] * part.ndim
    index = part.shape[axis] // 2
    if part.shape[axis] % 2 == 1:
        indexer[axis] = slice(index, index+1)
    else:
        indexer[axis] = slice(index-1, index+1)
    indexer = tuple(indexer)

    # 使用均值处理奇数和偶数情况，以强制数据类型转换，可以使用输出数组
    rout = mean(part[indexer], axis=axis, out=out)

    # 如果可能存在 NaN 值，警告并用 NaN 替换，类似于均值的处理方式
    if supports_nans and sz > 0:
        rout = np.lib._utils_impl._median_nancheck(part, rout, axis)

    return rout  # 返回计算得到的中位数值


def _percentile_dispatcher(a, q, axis=None, out=None, overwrite_input=None,
                           method=None, keepdims=None, *, weights=None,
                           interpolation=None):
    return (a, q, out, weights)


@array_function_dispatch(_percentile_dispatcher)
def percentile(a,
               q,
               axis=None,
               out=None,
               overwrite_input=False,
               method="linear",
               keepdims=False,
               *,
               weights=None,
               interpolation=None):
    """
    Compute the q-th percentile of the data along the specified axis.

    Returns the q-th percentile(s) of the array elements.

    Parameters
    ----------
    a : array_like of real numbers
        Input array or object that can be converted to an array.
    q : array_like of float
        Percentage or sequence of percentages for the percentiles to compute.
        Values must be between 0 and 100 inclusive.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the percentiles are computed. The
        default is to compute the percentile(s) along a flattened
        version of the array.

        .. versionchanged:: 1.9.0
            A tuple of axes is supported
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape and buffer length as the expected output, but the
        type (of the output) will be cast if necessary.
    overwrite_input : bool, optional
        If True, then allow the input array a to be modified by intermediate
        calculations, to save memory. In this case, the contents of the input
        a after this function completes is undefined.
    method : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
        This optional parameter specifies the interpolation method to use when
        the desired quantile lies between two data points i < j:

        - 'linear': i + (j - i) * fraction, where fraction is the fractional
          part of the index surrounded by i and j.
        - 'lower': i.
        - 'higher': j.
        - 'nearest': i or j, whichever is nearest.
        - 'midpoint': (i + j) / 2.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original array `a`.
    weights : array_like, optional
        An array of the same shape as `a`, containing weights to apply to the
        values when calculating the weighted average of the percentiles.
    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
        This optional parameter specifies the interpolation method to use.

    Returns
    -------
    percentile : ndarray
        Array with the same shape as `a`, with the specified percentile values
        along the specified axis.

    Notes
    -----
    Given a vector ``V`` of length ``N``, the q-th percentile of ``V`` is the
    value ``q/100`` of the way from the minimum to the maximum in a sorted
    copy of ``V``. The values and distances of the two nearest neighbors as
    well as the interpolation parameter will determine the percentile if the
    normalized ranking does not match the location of ``q`` exactly.

    """
    pass
    # 输出参数，可选，用于存放结果的数组。必须与期望的输出具有相同的形状和缓冲区长度，但必要时会进行类型转换。
    out : ndarray, optional

    # 是否覆盖输入。如果为 True，则允许中间计算修改输入数组 `a`，以节省内存。在这种情况下，函数完成后输入 `a` 的内容是未定义的。
    overwrite_input : bool, optional

    # 百分位数估算方法的参数。有多种不同的方法，其中一些是 NumPy 特有的。
    # 参见注释以了解详细解释。以下是按照它们在 H&F 论文 [1]_ 中总结的 R 类型排序的选项：
    # 1. 'inverted_cdf'
    # 2. 'averaged_inverted_cdf'
    # 3. 'closest_observation'
    # 4. 'interpolated_inverted_cdf'
    # 5. 'hazen'
    # 6. 'weibull'
    # 7. 'linear'（默认）
    # 8. 'median_unbiased'
    # 9. 'normal_unbiased'
    # 前三种方法是不连续的。NumPy 还定义了默认 'linear'（7.）选项的以下不连续变体：
    # * 'lower'
    # * 'higher'
    # * 'midpoint'
    # * 'nearest'
    # 
    # .. versionchanged:: 1.22.0
    #     此参数之前称为 "interpolation"，只提供了 "linear" 默认选项和最后四个选项。
    method : str, optional

    # 是否保持维度。如果设置为 True，则被减少的轴会作为大小为一的维度保留在结果中。
    # 使用此选项，结果将正确地广播到原始数组 `a`。
    # 
    # .. versionadded:: 1.9.0
    keepdims : bool, optional

    # 与数组 `a` 中的值相关联的权重数组。`a` 中的每个值根据其相关的权重贡献于百分位数。
    # 权重数组可以是 1-D（在这种情况下，其长度必须沿给定轴与 `a` 的大小相同）或与 `a` 的形状相同。
    # 如果 `weights=None`，则假定 `a` 中的所有数据权重都为一。
    # 只有 `method="inverted_cdf"` 支持权重。
    # 详见注释以了解更多细节。
    # 
    # .. versionadded:: 2.0.0
    weights : array_like, optional

    # 方法关键字参数的弃用名称。
    # 
    # .. deprecated:: 1.22.0
    interpolation : str, optional
    percentile : scalar or ndarray
        # percentile参数：标量或者数组，表示要计算的百分位数或百分位数列表
        If `q` is a single percentile and `axis=None`, then the result
        # 如果q是单个百分位数且axis=None，返回一个标量
        is a scalar. If multiple percentiles are given, first axis of
        # 如果给出多个百分位数，则结果的第一个轴对应于这些百分位数
        the result corresponds to the percentiles. The other axes are
        # 其他轴是在对a进行降维后剩余的轴
        the axes that remain after the reduction of `a`. If the input
        # 如果输入包含小于float64的整数或浮点数，输出数据类型为float64
        contains integers or floats smaller than ``float64``, the output
        data-type is ``float64``. Otherwise, the output data-type is the
        same as that of the input. If `out` is specified, that array is
        returned instead.

    See Also
    --------
    mean
    # 参见mean函数
    median : equivalent to ``percentile(..., 50)``
    # median：等价于percentile函数的使用，即percentile(..., 50)
    nanpercentile
    # nanpercentile函数
    quantile : equivalent to percentile, except q in the range [0, 1].
    # quantile：与percentile函数等价，除了q的范围在[0, 1]

    Notes
    -----
    The behavior of `numpy.percentile` with percentage `q` is
    # numpy.percentile函数在百分比q的情况下的行为
    that of `numpy.quantile` with argument ``q/100``.
    # 与numpy.quantile函数参数为q/100的行为相同
    For more information, please see `numpy.quantile`.

    Examples
    --------
    >>> a = np.array([[10, 7, 4], [3, 2, 1]])
    >>> a
    array([[10,  7,  4],
           [ 3,  2,  1]])
    >>> np.percentile(a, 50)
    3.5
    >>> np.percentile(a, 50, axis=0)
    array([6.5, 4.5, 2.5])
    >>> np.percentile(a, 50, axis=1)
    array([7.,  2.])
    >>> np.percentile(a, 50, axis=1, keepdims=True)
    array([[7.],
           [2.]])

    >>> m = np.percentile(a, 50, axis=0)
    >>> out = np.zeros_like(m)
    >>> np.percentile(a, 50, axis=0, out=out)
    array([6.5, 4.5, 2.5])
    >>> m
    array([6.5, 4.5, 2.5])

    >>> b = a.copy()
    >>> np.percentile(b, 50, axis=1, overwrite_input=True)
    array([7.,  2.])
    >>> assert not np.all(a == b)

    The different methods can be visualized graphically:

    .. plot::

        import matplotlib.pyplot as plt

        a = np.arange(4)
        p = np.linspace(0, 100, 6001)
        ax = plt.gca()
        lines = [
            ('linear', '-', 'C0'),
            ('inverted_cdf', ':', 'C1'),
            # Almost the same as `inverted_cdf`:
            ('averaged_inverted_cdf', '-.', 'C1'),
            ('closest_observation', ':', 'C2'),
            ('interpolated_inverted_cdf', '--', 'C1'),
            ('hazen', '--', 'C3'),
            ('weibull', '-.', 'C4'),
            ('median_unbiased', '--', 'C5'),
            ('normal_unbiased', '-.', 'C6'),
            ]
        for method, style, color in lines:
            ax.plot(
                p, np.percentile(a, p, method=method),
                label=method, linestyle=style, color=color)
        ax.set(
            title='Percentiles for different methods and data: ' + str(a),
            xlabel='Percentile',
            ylabel='Estimated percentile value',
            yticks=a)
        ax.legend(bbox_to_anchor=(1.03, 1))
        plt.tight_layout()
        plt.show()

    References
    ----------
    .. [1] R. J. Hyndman and Y. Fan,
       "Sample quantiles in statistical packages,"
       The American Statistician, 50(4), pp. 361-365, 1996
    # 如果插值参数不为空，则调用辅助函数检查插值方法是否合法，并更新方法变量
    if interpolation is not None:
        method = _check_interpolation_as_method(
            method, interpolation, "percentile")

    # 将输入数组转换为 NumPy 的数组对象
    a = np.asanyarray(a)
    
    # 如果数组的数据类型是复数类型，则抛出类型错误
    if a.dtype.kind == "c":
        raise TypeError("a must be an array of real numbers")

    # 根据数组的数据类型调整百分位数 q 的类型，以匹配数据数组的数据类型
    q = np.true_divide(q, a.dtype.type(100) if a.dtype.kind == "f" else 100)
    
    # 恢复任何由 ufunc 执行的衰减效果，确保 q 是一个数组
    q = asanyarray(q)  # undo any decay that the ufunc performed (see gh-13105)
    
    # 检查百分位数 q 是否在合法范围内 [0, 100]
    if not _quantile_is_valid(q):
        raise ValueError("Percentiles must be in the range [0, 100]")

    # 如果权重参数不为空，则进一步检查和处理权重参数
    if weights is not None:
        # 如果方法不是 "inverted_cdf"，则抛出值错误，只有 "inverted_cdf" 方法支持权重
        if method != "inverted_cdf":
            msg = ("Only method 'inverted_cdf' supports weights. "
                   f"Got: {method}.")
            raise ValueError(msg)
        
        # 如果指定了轴参数，则将其标准化为轴元组
        if axis is not None:
            axis = _nx.normalize_axis_tuple(axis, a.ndim, argname="axis")
        
        # 检查权重数组是否有效，并与输入数组 a 相容
        weights = _weights_are_valid(weights=weights, a=a, axis=axis)
        
        # 如果权重数组中存在负数，则抛出值错误
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative.")

    # 调用未经检查的分位数计算函数，返回计算结果
    return _quantile_unchecked(
        a, q, axis, out, overwrite_input, method, keepdims, weights)
# 定义一个函数 _quantile_dispatcher，用于分发参数给 quantile 函数
def _quantile_dispatcher(a, q, axis=None, out=None, overwrite_input=None,
                         method=None, keepdims=None, *, weights=None,
                         interpolation=None):
    # 返回元组 (a, q, out, weights)，将这些参数作为元组返回
    return (a, q, out, weights)


# 使用 array_function_dispatch 装饰器将 _quantile_dispatcher 函数与 quantile 函数关联起来
@array_function_dispatch(_quantile_dispatcher)
def quantile(a,
             q,
             axis=None,
             out=None,
             overwrite_input=False,
             method="linear",
             keepdims=False,
             *,
             weights=None,
             interpolation=None):
    """
    计算沿指定轴的第 q 个分位数。

    .. versionadded:: 1.15.0

    Parameters
    ----------
    a : array_like of real numbers
        输入数组或可以转换为数组的对象。
    q : array_like of float
        要计算的分位数的概率或概率序列。值必须在 0 到 1 之间（包括边界值）。
    axis : {int, tuple of int, None}, optional
        计算分位数的轴或轴。默认是沿数组的展平版本计算分位数。
    out : ndarray, optional
        替代的输出数组，用于存放结果。它必须具有与预期输出相同的形状和缓冲区长度，但如果需要，输出的类型将被强制转换。
    overwrite_input : bool, optional
        如果为 True，则允许中间计算修改输入数组 `a`，以节省内存。在这种情况下，此函数完成后，输入 `a` 的内容是未定义的。
    method : str, optional
        指定用于估算分位数的方法。有许多不同的方法，其中一些是 NumPy 独有的。
        推荐的选项如下，按它们在 [1]_ 中出现的顺序编号：

        1. 'inverted_cdf'
        2. 'averaged_inverted_cdf'
        3. 'closest_observation'
        4. 'interpolated_inverted_cdf'
        5. 'hazen'
        6. 'weibull'
        7. 'linear'  (默认)
        8. 'median_unbiased'
        9. 'normal_unbiased'

        前三种方法是不连续的。为了向后兼容以前版本的 NumPy，还提供了以下默认值为 'linear' (7.) 的不连续变体：

        * 'lower'
        * 'higher'
        * 'midpoint'
        * 'nearest'

        详情请见 Notes。

        .. versionchanged:: 1.22.0
            此参数之前称为 "interpolation"，并且只提供了 "linear" 默认和最后四个选项。

    keepdims : bool, optional
        如果设置为 True，则保持被减少的轴作为大小为一的维度在结果中。使用此选项，结果将正确地对原始数组 `a` 进行广播。
    weights : array_like, optional
        # 可选参数，用于给定数组 `a` 中的值关联的权重数组。`a` 中的每个值根据其关联的权重来计算分位数。
        # 权重数组可以是一维的（此时其长度必须与给定轴向上 `a` 的大小相同），或者与 `a` 的形状相同。
        # 如果 `weights=None`，则假定 `a` 中的所有数据的权重都等于一。
        # 只有 `method="inverted_cdf"` 支持权重。
        # 更多细节请参见注释部分。

        .. versionadded:: 2.0.0
        # 添加于版本 2.0.0

    interpolation : str, optional
        # 方法关键字参数的已弃用名称。

        .. deprecated:: 1.22.0
        # 已弃用于版本 1.22.0

    Returns
    -------
    quantile : scalar or ndarray
        # 如果 `q` 是单个概率值且 `axis=None`，则结果是一个标量。
        # 如果给定了多个概率水平，结果的第一个轴对应于分位数。其他轴是在减少 `a` 后剩余的轴。
        # 如果输入包含小于 ``float64`` 的整数或浮点数，则输出数据类型为 ``float64``。
        # 否则，输出数据类型与输入相同。如果指定了 `out`，则返回该数组。

    See Also
    --------
    mean
    percentile : 等价于 `quantile`，但 `q` 范围在 [0, 100] 内。
    median : 等价于 ``quantile(..., 0.5)``
    nanquantile

    Notes
    -----
    # 给定来自潜在分布的样本 `a`，`quantile` 提供了反向累积分布函数的非参数估计。

    # 默认情况下，通过在 ``y`` 中相邻元素之间插值来完成：

    #   (1-g)*y[j] + g*y[j+1]

    # 其中索引 ``j`` 和系数 ``g`` 是 ``q * (n-1)`` 的整数部分和小数部分，``n`` 是样本中的元素数。

    # 这是 H&F [1]_ 方程 1 的特例。更一般地，

    # - ``j = (q*n + m - 1) // 1``,
    # - ``g = (q*n + m - 1) % 1``，

    # 其中 ``m`` 可能根据几种不同的约定来定义。可以使用 ``method`` 参数选择首选约定：

    =============================== =============== ===============
    ``method``                      H&F 中的编号     ``m``
    =============================== =============== ===============
    ``interpolated_inverted_cdf``   4               ``0``
    ``hazen``                       5               ``1/2``
    ``weibull``                     6               ``q``
    ``linear`` (默认)               7               ``1 - q``
    ``median_unbiased``             8               ``q/3 + 1/3``
    ``normal_unbiased``             9               ``q/4 + 3/8``
    =============================== =============== ===============

    # 注意，索引 ``j`` 和 ``j + 1`` 被限制在范围 ``0`` 到
    """
    ``n - 1`` when the results of the formula would be outside the allowed
    range of non-negative indices. The ``- 1`` in the formulas for ``j`` and
    ``g`` accounts for Python's 0-based indexing.

    The table above includes only the estimators from H&F that are continuous
    functions of probability `q` (estimators 4-9). NumPy also provides the
    three discontinuous estimators from H&F (estimators 1-3), where ``j`` is
    defined as above and ``m`` and ``g`` are defined as follows.

    1. ``inverted_cdf``: ``m = 0`` and ``g = int(q*n > 0)``
    2. ``averaged_inverted_cdf``: ``m = 0`` and ``g = (1 + int(q*n > 0)) / 2``
    3. ``closest_observation``: ``m = -1/2`` and
       ``1 - int((g == 0) & (j%2 == 0))``

    For backward compatibility with previous versions of NumPy, `quantile`
    provides four additional discontinuous estimators. Like
    ``method='linear'``, all have ``m = 1 - q`` so that ``j = q*(n-1) // 1``,
    but ``g`` is defined as follows.

    - ``lower``: ``g = 0``
    - ``midpoint``: ``g = 0.5``
    - ``higher``: ``g = 1``
    - ``nearest``: ``g = (q*(n-1) % 1) > 0.5``

    **Weighted quantiles:**
    More formally, the quantile at probability level :math:`q` of a cumulative
    distribution function :math:`F(y)=P(Y \\leq y)` with probability measure
    :math:`P` is defined as any number :math:`x` that fulfills the
    *coverage conditions*

    .. math:: P(Y < x) \\leq q \\quad\\text{and}\\quad P(Y \\leq x) \\geq q

    with random variable :math:`Y\\sim P`.
    Sample quantiles, the result of `quantile`, provide nonparametric
    estimation of the underlying population counterparts, represented by the
    unknown :math:`F`, given a data vector `a` of length ``n``.

    Some of the estimators above arise when one considers :math:`F` as the
    empirical distribution function of the data, i.e.
    :math:`F(y) = \\frac{1}{n} \\sum_i 1_{a_i \\leq y}`.
    Then, different methods correspond to different choices of :math:`x` that
    fulfill the above coverage conditions. Methods that follow this approach
    are ``inverted_cdf`` and ``averaged_inverted_cdf``.

    For weighted quantiles, the coverage conditions still hold. The
    empirical cumulative distribution is simply replaced by its weighted
    version, i.e. 
    :math:`P(Y \\leq t) = \\frac{1}{\\sum_i w_i} \\sum_i w_i 1_{x_i \\leq t}`.
    Only ``method="inverted_cdf"`` supports weights.

    Examples
    --------
    >>> a = np.array([[10, 7, 4], [3, 2, 1]])
    >>> a
    array([[10,  7,  4],
           [ 3,  2,  1]])
    >>> np.quantile(a, 0.5)
    3.5
    >>> np.quantile(a, 0.5, axis=0)
    array([6.5, 4.5, 2.5])
    >>> np.quantile(a, 0.5, axis=1)
    array([7.,  2.])
    >>> np.quantile(a, 0.5, axis=1, keepdims=True)
    array([[7.],
           [2.]])
    >>> m = np.quantile(a, 0.5, axis=0)
    >>> out = np.zeros_like(m)
    >>> np.quantile(a, 0.5, axis=0, out=out)
    array([6.5, 4.5, 2.5])
    >>> m
    array([6.5, 4.5, 2.5])
    >>> b = a.copy()
    """
    # 计算数组 b 在每行的中位数，覆盖原始输入数组
    >>> np.quantile(b, 0.5, axis=1, overwrite_input=True)
    array([7.,  2.])

    # 断言数组 a 和 b 不完全相等
    >>> assert not np.all(a == b)

    # 查看 `numpy.percentile` 函数以获得大多数方法的可视化方法。

    # 参考文献
    # ----------
    # .. [1] R. J. Hyndman 和 Y. Fan,
    #    "Sample quantiles in statistical packages,"
    #    The American Statistician, 50(4), pp. 361-365, 1996
    """
    # 如果插值参数不为 None，则验证并更新 method 变量
    if interpolation is not None:
        method = _check_interpolation_as_method(
            method, interpolation, "quantile")

    # 将输入数组 a 转换为任意数组
    a = np.asanyarray(a)

    # 如果数组 a 的数据类型为复数，则引发类型错误
    if a.dtype.kind == "c":
        raise TypeError("a must be an array of real numbers")

    # 如果 q 是 Python 的整数或浮点数，并且数组 a 的数据类型为浮点数，则将 q 转换为相同数据类型的数组
    if isinstance(q, (int, float)) and a.dtype.kind == "f":
        q = np.asanyarray(q, dtype=a.dtype)
    else:
        q = np.asanyarray(q)

    # 验证 q 数组的值必须在 [0, 1] 范围内
    if not _quantile_is_valid(q):
        raise ValueError("Quantiles must be in the range [0, 1]")

    # 如果有权重参数，则进行以下验证和处理
    if weights is not None:
        # 如果 method 不是 "inverted_cdf"，则引发值错误
        if method != "inverted_cdf":
            msg = ("Only method 'inverted_cdf' supports weights. "
                   f"Got: {method}.")
            raise ValueError(msg)
        
        # 如果指定了轴参数，则将其规范化为元组形式
        if axis is not None:
            axis = _nx.normalize_axis_tuple(axis, a.ndim, argname="axis")
        
        # 验证权重数组的有效性，并返回处理后的权重数组
        weights = _weights_are_valid(weights=weights, a=a, axis=axis)
        
        # 如果权重数组中存在负数，则引发值错误
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative.")

    # 调用未经检查的量化函数，返回计算结果
    return _quantile_unchecked(
        a, q, axis, out, overwrite_input, method, keepdims, weights)
# 对未检查的分位数计算函数，假定 q 在 [0, 1] 范围内，并且是一个 ndarray
def _quantile_unchecked(a,
                        q,
                        axis=None,
                        out=None,
                        overwrite_input=False,
                        method="linear",
                        keepdims=False,
                        weights=None):
    # 调用 _ureduce 函数进行分位数计算
    return _ureduce(a,
                    func=_quantile_ureduce_func,
                    q=q,
                    weights=weights,
                    keepdims=keepdims,
                    axis=axis,
                    out=out,
                    overwrite_input=overwrite_input,
                    method=method)


def _quantile_is_valid(q):
    # 避免对元素少于 O(1000) 的数组进行昂贵的归约操作
    if q.ndim == 1 and q.size < 10:
        for i in range(q.size):
            # 检查 q 中的每个元素是否在 [0, 1] 范围内
            if not (0.0 <= q[i] <= 1.0):
                return False
    else:
        # 检查整体的最小值和最大值是否在 [0, 1] 范围内
        if not (q.min() >= 0 and q.max() <= 1):
            return False
    return True


def _check_interpolation_as_method(method, interpolation, fname):
    # 警告：NumPy 1.22 中已弃用，2021-11-08
    warnings.warn(
        f"the `interpolation=` argument to {fname} was renamed to "
        "`method=`, which has additional options.\n"
        "Users of the modes 'nearest', 'lower', 'higher', or "
        "'midpoint' are encouraged to review the method they used. "
        "(Deprecated NumPy 1.22)",
        DeprecationWarning, stacklevel=4)
    if method != "linear":
        # 检查：不应同时传递 `method` 和 `interpolation`
        raise TypeError(
            "You shall not pass both `method` and `interpolation`!\n"
            "(`interpolation` is Deprecated in favor of `method`)")
    return interpolation


def _compute_virtual_index(n, quantiles, alpha: float, beta: float):
    """
    计算用于线性插值分位数的浮点索引。
    n : array_like
        样本大小。
    quantiles : array_like
        分位数值。
    alpha : float
        用于修正计算的索引的常数。
    beta : float
        用于修正计算的索引的常数。

    alpha 和 beta 的值取决于所选择的方法（见 quantile 文档）。

    参考：
    Hyndman&Fan 的论文 "Sample Quantiles in Statistical Packages"，
    DOI: 10.1080/00031305.1996.10473566
    """
    return n * quantiles + (
            alpha + quantiles * (1 - alpha - beta)
    ) - 1


def _get_gamma(virtual_indexes, previous_indexes, method):
    """
    计算用于线性插值分位数的 gamma（也称为 'm' 或 'weight'）。

    virtual_indexes : array_like
        排序样本中应该找到百分位数的索引位置。
    previous_indexes : array_like
        virtual_indexes 的下限值。
    method : str
        选择的插值方法，可能具有修改 gamma 的特定规则。
    """
    gamma is usually the fractional part of virtual_indexes but can be modified
    by the interpolation method.
    """
    # 将虚拟索引与上一索引的差异转换为 NumPy 数组
    gamma = np.asanyarray(virtual_indexes - previous_indexes)
    # 使用给定的方法修正 gamma 值，方法由 method 字典中的 "fix_gamma" 键提供
    gamma = method["fix_gamma"](gamma, virtual_indexes)
    # 确保 gamma 变量是一个数组，并且保持与虚拟索引相同的数据类型
    # （可能已经匹配了输入数组的数据类型）。
    return np.asanyarray(gamma, dtype=virtual_indexes.dtype)
# 定义一个函数，用于执行线性插值，根据输入的数组 a, b, t 来计算加权插值结果
def _lerp(a, b, t, out=None):
    """
    Compute the linear interpolation weighted by gamma on each point of
    two same shape array.

    a : array_like
        Left bound.
    b : array_like
        Right bound.
    t : array_like
        The interpolation weight.
    out : array_like
        Output array.
    """
    # 计算 b 和 a 之间的差值
    diff_b_a = subtract(b, a)
    # 计算线性插值的结果，使用 add 函数，结果存储在 out 参数中
    lerp_interpolation = asanyarray(add(a, diff_b_a * t, out=out))
    # 根据条件 t >= 0.5，计算并更新 lerp_interpolation 的值
    subtract(b, diff_b_a * (1 - t), out=lerp_interpolation, where=t >= 0.5,
             casting='unsafe', dtype=type(lerp_interpolation.dtype))
    # 如果 lerp_interpolation 是 0 维数组且 out 为 None，则将其解包成标量
    if lerp_interpolation.ndim == 0 and out is None:
        lerp_interpolation = lerp_interpolation[()]  # unpack 0d arrays
    # 返回插值结果
    return lerp_interpolation


# 定义一个函数，根据条件生成一个指定形状的掩码数组，用指定值填充默认值
def _get_gamma_mask(shape, default_value, conditioned_value, where):
    # 创建一个形状为 shape 的数组，填充默认值 default_value
    out = np.full(shape, default_value)
    # 根据条件 where，将 conditioned_value 的值复制到 out 中，不安全的强制转换
    np.copyto(out, conditioned_value, where=where, casting="unsafe")
    # 返回生成的掩码数组 out
    return out


# 定义一个函数，执行离散的插值到边界的操作，返回根据条件生成的数组
def _discret_interpolation_to_boundaries(index, gamma_condition_fun):
    # 向下取整得到 previous，向上取整得到 next
    previous = np.floor(index)
    next = previous + 1
    # 计算 gamma
    gamma = index - previous
    # 使用 gamma_condition_fun 函数生成 gamma 条件掩码，并将结果转换为 np.intp 类型
    res = _get_gamma_mask(shape=index.shape,
                          default_value=next,
                          conditioned_value=previous,
                          where=gamma_condition_fun(gamma, index)
                          ).astype(np.intp)
    # 修剪超出边界的整数值
    res[res < 0] = 0
    # 返回结果数组 res
    return res


# 定义一个函数，查找最接近观测值的索引位置，使用 gamma 函数作为条件生成器
def _closest_observation(n, quantiles):
    # 定义 gamma 函数，返回条件 (gamma == 0) & (np.floor(index) % 2 == 0)
    gamma_fun = lambda gamma, index: (gamma == 0) & (np.floor(index) % 2 == 0)
    # 调用 _discret_interpolation_to_boundaries 函数，返回最接近观测值的索引位置
    return _discret_interpolation_to_boundaries((n * quantiles) - 1 - 0.5,
                                                gamma_fun)


# 定义一个函数，执行反向的累积分布函数插值操作，使用 gamma 函数作为条件生成器
def _inverted_cdf(n, quantiles):
    # 定义 gamma 函数，返回条件 (gamma == 0)
    gamma_fun = lambda gamma, _: (gamma == 0)
    # 调用 _discret_interpolation_to_boundaries 函数，执行反向的累积分布函数插值
    return _discret_interpolation_to_boundaries((n * quantiles) - 1,
                                                gamma_fun)


# 定义一个函数，执行 quantile 的降维函数操作，返回降维后的数组
def _quantile_ureduce_func(
        a: np.array,
        q: np.array,
        weights: np.array,
        axis: int = None,
        out=None,
        overwrite_input: bool = False,
        method="linear",
) -> np.array:
    # 如果 q 的维度大于 2，抛出 ValueError 异常
    if q.ndim > 2:
        raise ValueError("q must be a scalar or 1d")
    # 如果 overwrite_input 为 True，修改输入参数 a 和 weights 的形状
    if overwrite_input:
        if axis is None:
            axis = 0
            arr = a.ravel()
            wgt = None if weights is None else weights.ravel()
        else:
            arr = a
            wgt = weights
    else:
        if axis is None:
            axis = 0
            arr = a.flatten()
            wgt = None if weights is None else weights.flatten()
        else:
            arr = a.copy()
            wgt = weights
    # 调用 _quantile 函数来计算给定数组的分位数
    result = _quantile(arr,
                       quantiles=q,    # 传入分位数列表 q，用于计算对应的分位数
                       axis=axis,      # 指定计算分位数的轴
                       method=method,  # 指定计算分位数的方法（如 'linear' 或 'lower' 等）
                       out=out,        # 指定输出数组，用于存储计算结果
                       weights=wgt)    # 可选参数，指定加权的权重数组
    # 返回 _quantile 函数计算得到的结果
    return result
# 获取数组 arr 中虚拟索引 virtual_indexes 附近的有效索引。
# 这是用于分位数线性插值的辅助函数。

def _get_indexes(arr, virtual_indexes, valid_values_count):
    """
    Get the valid indexes of arr neighbouring virtual_indexes.
    Note
    This is a companion function to linear interpolation of
    Quantiles

    Returns
    -------
    (previous_indexes, next_indexes): Tuple
        A Tuple of virtual_indexes neighbouring indexes
    """
    # 向下取整得到虚拟索引的上一个索引
    previous_indexes = np.asanyarray(np.floor(virtual_indexes))
    # 向上取整得到虚拟索引的下一个索引
    next_indexes = np.asanyarray(previous_indexes + 1)
    
    # 判断是否有超出有效值范围的索引
    indexes_above_bounds = virtual_indexes >= valid_values_count - 1
    # 当索引超出最大索引时，将上一个和下一个索引都设为 -1
    if indexes_above_bounds.any():
        previous_indexes[indexes_above_bounds] = -1
        next_indexes[indexes_above_bounds] = -1
    
    # 判断是否有小于零的索引
    indexes_below_bounds = virtual_indexes < 0
    # 当索引小于零时，将上一个和下一个索引都设为 0
    if indexes_below_bounds.any():
        previous_indexes[indexes_below_bounds] = 0
        next_indexes[indexes_below_bounds] = 0
    
    # 如果数组 arr 的数据类型是浮点数类型，处理包含 NaN 的情况
    if np.issubdtype(arr.dtype, np.inexact):
        # 确定虚拟索引中是否有 NaN
        virtual_indexes_nans = np.isnan(virtual_indexes)
        # 当虚拟索引中有 NaN 时，将上一个和下一个索引都设为 -1
        if virtual_indexes_nans.any():
            previous_indexes[virtual_indexes_nans] = -1
            next_indexes[virtual_indexes_nans] = -1
    
    # 将索引数组转换为整数类型
    previous_indexes = previous_indexes.astype(np.intp)
    next_indexes = next_indexes.astype(np.intp)
    
    # 返回上一个索引数组和下一个索引数组的元组
    return previous_indexes, next_indexes
    if weights is None:
        # --- Computation of indexes
        # weights 参数为空时执行以下代码块，计算索引值

        # Index where to find the value in the sorted array.
        # Virtual because it is a floating point value, not an valid index.
        # The nearest neighbours are used for interpolation
        # 在排序数组中找到值的索引位置。由于是浮点数值，所以是虚拟的索引。
        # 最近的邻居用于插值

        try:
            method_props = _QuantileMethods[method]
        except KeyError:
            raise ValueError(
                f"{method!r} is not a valid method. Use one of: "
                f"{_QuantileMethods.keys()}") from None
        # 尝试获取指定方法的属性，若未找到则抛出 ValueError 异常

        virtual_indexes = method_props["get_virtual_index"](values_count,
                                                            quantiles)
        virtual_indexes = np.asanyarray(virtual_indexes)
        # 调用方法获取虚拟索引，并将其转换为 NumPy 数组

        if method_props["fix_gamma"] is None:
            supports_integers = True
        else:
            int_virtual_indices = np.issubdtype(virtual_indexes.dtype,
                                                np.integer)
            supports_integers = method == 'linear' and int_virtual_indices
        # 根据方法属性判断是否支持整数索引

        if supports_integers:
            # No interpolation needed, take the points along axis
            # 不需要插值，沿轴取点
            if supports_nans:
                # may contain nan, which would sort to the end
                # 可能包含 NaN，这些值会被排序到最后
                arr.partition(
                    concatenate((virtual_indexes.ravel(), [-1])), axis=0,
                )
                slices_having_nans = np.isnan(arr[-1, ...])
            else:
                # cannot contain nan
                # 不包含 NaN
                arr.partition(virtual_indexes.ravel(), axis=0)
                slices_having_nans = np.array(False, dtype=bool)
            result = take(arr, virtual_indexes, axis=0, out=out)
            # 从 arr 中获取虚拟索引对应的值，存入 result

        else:
            previous_indexes, next_indexes = _get_indexes(arr,
                                                          virtual_indexes,
                                                          values_count)
            # 获取前后索引值

            # --- Sorting
            # 对数组进行排序
            arr.partition(
                np.unique(np.concatenate(([0, -1],
                                          previous_indexes.ravel(),
                                          next_indexes.ravel(),
                                          ))),
                axis=0)
            if supports_nans:
                slices_having_nans = np.isnan(arr[-1, ...])
            else:
                slices_having_nans = None

            # --- Get values from indexes
            # 从索引值获取对应的数值
            previous = arr[previous_indexes]
            next = arr[next_indexes]

            # --- Linear interpolation
            # 线性插值
            gamma = _get_gamma(virtual_indexes, previous_indexes, method_props)
            result_shape = virtual_indexes.shape + (1,) * (arr.ndim - 1)
            gamma = gamma.reshape(result_shape)
            result = _lerp(previous,
                        next,
                        gamma,
                        out=out)
    # 检查是否存在包含 NaN 的切片
    if np.any(slices_having_nans):
        # 如果存在 NaN 的切片，执行以下操作
        if result.ndim == 0 and out is None:
            # 如果结果是标量且没有指定输出位置，无法写入标量，但索引将是正确的
            result = arr[-1]
        else:
            # 否则，使用 np.copyto() 复制 arr 的最后一个元素到 result，仅复制到存在 NaN 的切片
            np.copyto(result, arr[-1, ...], where=slices_having_nans)
    # 返回处理后的结果
    return result
# 使用 `array_function_dispatch` 装饰器，将 `_trapezoid_dispatcher` 函数与 `trapezoid` 函数关联起来
@array_function_dispatch(_trapezoid_dispatcher)
# 定义函数 `trapezoid`，使用复合梯形法则沿指定轴积分
def trapezoid(y, x=None, dx=1.0, axis=-1):
    r"""
    使用复合梯形法则沿给定轴积分。

    如果提供了 `x`，则在其元素上依序积分 - 它们不会被排序。

    沿给定轴在每个 1 维切片上积分 `y` (`x`)，计算 :math:`\int y(x) dx`。
    当指定了 `x` 时，这将沿参数曲线进行积分，计算 :math:`\int_t y(t) dt =
    \int_t y(t) \left.\frac{dx}{dt}\right|_{x=x(t)} dt`。

    .. versionadded:: 2.0.0

    Parameters
    ----------
    y : array_like
        要积分的输入数组。
    x : array_like, optional
        与 `y` 值对应的样本点。如果 `x` 是 None，则假定样本点均匀分布，间距为 `dx`。默认为 None。
    dx : scalar, optional
        当 `x` 是 None 时样本点之间的间距。默认为 1。
    axis : int, optional
        要进行积分的轴。

    Returns
    -------
    trapezoid : float or ndarray
        用梯形法则沿单个轴近似计算的 `y` 的定积分 = n 维数组。如果 `y` 是 1 维数组，则结果是一个浮点数。
        如果 `n` 大于 1，则结果是一个 `n-1` 维数组。

    See Also
    --------
    sum, cumsum

    Notes
    -----
    图片 [2]_ 描述了梯形法则 - 默认从 `y` 数组中取出点的 y 轴位置，x 轴点之间的距离默认为 1.0，
    也可以使用 `x` 数组或 `dx` 标量提供。返回值将等于红线下的组合面积。

    References
    ----------
    .. [1] Wikipedia 页面: https://en.wikipedia.org/wiki/Trapezoidal_rule

    .. [2] Illustration image:
           https://en.wikipedia.org/wiki/File:Composite_trapezoidal_rule_illustration.png

    Examples
    --------
    在均匀间隔点上使用梯形法则：

    >>> np.trapezoid([1, 2, 3])
    4.0

    可以通过 `x` 或 `dx` 参数选择样本点之间的间距：

    >>> np.trapezoid([1, 2, 3], x=[4, 6, 8])
    8.0
    >>> np.trapezoid([1, 2, 3], dx=2)
    8.0

    使用递减的 `x` 对应于反向积分：

    >>> np.trapezoid([1, 2, 3], x=[8, 6, 4])
    -8.0

    更一般地，`x` 用于沿参数曲线进行积分。我们可以估计积分 :math:`\int_0^1 x^2 = 1/3` 使用：

    >>> x = np.linspace(0, 1, num=50)
    >>> y = x**2
    >>> np.trapezoid(y, x)
    0.33340274885464394

    或者估计一个圆的面积，注意我们重复样本以闭合曲线：

    >>> theta = np.linspace(0, 2 * np.pi, num=1000, endpoint=True)
    >>> np.trapezoid(np.cos(theta), x=np.sin(theta))
    3.141571941375841
    """
    ``np.trapezoid`` can be applied along a specified axis to do multiple
    computations in one call:

    >>> a = np.arange(6).reshape(2, 3)
    >>> a
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.trapezoid(a, axis=0)
    array([1.5, 2.5, 3.5])
    >>> np.trapezoid(a, axis=1)
    array([2.,  8.])
    """

    # 将输入的 y 转换为数组（如果不是的话）
    y = asanyarray(y)
    # 如果 x 为 None，则使用 dx 作为间距 d
    if x is None:
        d = dx
    else:
        # 将输入的 x 转换为数组
        x = asanyarray(x)
        # 如果 x 是一维数组
        if x.ndim == 1:
            # 计算 x 的差分
            d = diff(x)
            # 调整形状以匹配 y 的维度
            shape = [1]*y.ndim
            shape[axis] = d.shape[0]
            d = d.reshape(shape)
        else:
            # 计算 x 在指定轴上的差分
            d = diff(x, axis=axis)
    # 获取 y 的维度
    nd = y.ndim
    # 创建用于切片的索引列表
    slice1 = [slice(None)]*nd
    slice2 = [slice(None)]*nd
    # 配置切片列表，以便获取 y 的相邻元素
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    try:
        # 计算梯形公式的结果
        ret = (d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0).sum(axis)
    except ValueError:
        # 如果计算出错，转换 d 和 y 为 ndarray 类型并重新计算
        d = np.asarray(d)
        y = np.asarray(y)
        ret = add.reduce(d * (y[tuple(slice1)]+y[tuple(slice2)])/2.0, axis)
    # 返回计算结果
    return ret
# 设置模块名称为'numpy'
@set_module('numpy')
# 定义一个函数 trapz，用于数值积分，但在 NumPy 2.0 中已被弃用
def trapz(y, x=None, dx=1.0, axis=-1):
    """
    `trapz` is deprecated in NumPy 2.0.

    Please use `trapezoid` instead, or one of the numerical integration
    functions in `scipy.integrate`.
    """
    # 发出警告，提示 trapz 已在 NumPy 2.0 中弃用，建议使用 trapezoid 函数或者 scipy.integrate 中的数值积分函数
    warnings.warn(
        "`trapz` is deprecated. Use `trapezoid` instead, or one of the "
        "numerical integration functions in `scipy.integrate`.",
        DeprecationWarning,
        stacklevel=2
    )
    # 调用 trapezoid 函数进行数值积分计算，并返回结果
    return trapezoid(y, x=x, dx=dx, axis=axis)


# 根据输入参数分发函数调用，这里直接返回参数 xi，未进行实际处理
def _meshgrid_dispatcher(*xi, copy=None, sparse=None, indexing=None):
    return xi


# 基于 scitools 中的 meshgrid 函数实现
# 使用 array_function_dispatch 装饰器进行函数分发
@array_function_dispatch(_meshgrid_dispatcher)
# 定义 meshgrid 函数，用于生成坐标网格
def meshgrid(*xi, copy=True, sparse=False, indexing='xy'):
    """
    Return a tuple of coordinate matrices from coordinate vectors.

    Make N-D coordinate arrays for vectorized evaluations of
    N-D scalar/vector fields over N-D grids, given
    one-dimensional coordinate arrays x1, x2,..., xn.

    .. versionchanged:: 1.9
       1-D and 0-D cases are allowed.

    Parameters
    ----------
    x1, x2,..., xn : array_like
        1-D arrays representing the coordinates of a grid.
    indexing : {'xy', 'ij'}, optional
        Cartesian ('xy', default) or matrix ('ij') indexing of output.
        See Notes for more details.

        .. versionadded:: 1.7.0
    sparse : bool, optional
        If True the shape of the returned coordinate array for dimension *i*
        is reduced from ``(N1, ..., Ni, ... Nn)`` to
        ``(1, ..., 1, Ni, 1, ..., 1)``.  These sparse coordinate grids are
        intended to be use with :ref:`basics.broadcasting`.  When all
        coordinates are used in an expression, broadcasting still leads to a
        fully-dimensonal result array.

        Default is False.

        .. versionadded:: 1.7.0
    copy : bool, optional
        If False, a view into the original arrays are returned in order to
        conserve memory.  Default is True.  Please note that
        ``sparse=False, copy=False`` will likely return non-contiguous
        arrays.  Furthermore, more than one element of a broadcast array
        may refer to a single memory location.  If you need to write to the
        arrays, make copies first.

        .. versionadded:: 1.7.0

    Returns
    -------
    X1, X2,..., XN : tuple of ndarrays
        For vectors `x1`, `x2`,..., `xn` with lengths ``Ni=len(xi)``,
        returns ``(N1, N2, N3,..., Nn)`` shaped arrays if indexing='ij'
        or ``(N2, N1, N3,..., Nn)`` shaped arrays if indexing='xy'
        with the elements of `xi` repeated to fill the matrix along
        the first dimension for `x1`, the second for `x2` and so on.

    Notes
    -----
    This function supports both indexing conventions through the indexing
    keyword argument.  Giving the string 'ij' returns a meshgrid with
    matrix indexing, while 'xy' returns a meshgrid with Cartesian indexing.
    """
    # 返回输入参数 xi，未做任何实际计算
    return xi
    """
    根据输入的 xi 返回一个 N 维矩阵或向量的广播数组。

    Parameters
    ----------
    xi : list of array-like
        输入的数组列表，每个数组的形状可能不同。
    indexing : {'xy', 'ij'}, optional
        索引顺序，'xy' 表示轴顺序为 (N, M, P)，'ij' 表示轴顺序为 (M, N, P)。
        默认为 'xy'。
    sparse : bool, optional
        是否返回稀疏输出数组。默认为 False。

    Returns
    -------
    output : list of ndarray
        广播后的 N 维数组列表，根据 indexing 和 sparse 参数的设置返回不同的形状。

    Raises
    ------
    ValueError
        如果 indexing 参数不是 'xy' 或 'ij'。

    Notes
    -----
    在 1-D 和 0-D 情况下，indexing 和 sparse 参数不产生影响。

    See Also
    --------
    mgrid : 使用索引符号构造多维“网格”。
    ogrid : 使用索引符号构造多维“开放网格”。
    :ref:`how-to-index`

    Examples
    --------
    >>> xi = [np.array([1, 2, 3]), np.array([4, 5])]
    >>> np.broadcast_arrays(*xi)
    [array([[1, 2, 3],
            [4, 5, 5]]), array([[4, 4, 4],
            [5, 5, 5]])]

    在 'xy' 索引情况下，交换第一和第二轴的顺序：

    >>> np.broadcast_arrays(*xi, indexing='xy')
    [array([[1, 1, 1],
            [2, 2, 2],
            [3, 3, 3]]), array([[4, 5, 5],
            [4, 5, 5]])]

    如果 sparse=True，则返回稀疏的输出数组：

    >>> np.broadcast_arrays(*xi, sparse=True)
    [array([[1, 2, 3],
            [4, 5, 5]]), array([[4],
            [5]])]

    """
    ndim = len(xi)  # 获取输入数组列表的维度

    if indexing not in ['xy', 'ij']:  # 检查索引顺序参数是否有效
        raise ValueError(
            "Valid values for `indexing` are 'xy' and 'ij'.")

    s0 = (1,) * ndim  # 创建一个与输入数组维度相同的单位元组

    # 使用列表推导式对每个输入数组进行广播
    output = [np.asanyarray(x).reshape(s0[:i] + (-1,) + s0[i + 1:])
              for i, x in enumerate(xi)]

    if indexing == 'xy' and ndim > 1:
        # 如果索引顺序是 'xy'，且输入数组维度大于1，则交换第一和第二轴
        output[0].shape = (1, -1) + s0[2:]
        output[1].shape = (-1, 1) + s0[2:]

    if not sparse:
        # 如果 sparse=False，则返回完整的 N 维矩阵，而不是仅仅是 1-D 向量的广播
        output = np.broadcast_arrays(*output, subok=True)
    # 如果 copy 参数为真，执行下面的代码块
    if copy:
        # 使用生成器表达式，复制 output 中每个元素，形成一个元组
        output = tuple(x.copy() for x in output)

    # 返回处理后的 output 变量
    return output
# 定义一个删除操作的调度函数，返回输入的数组和对象
def _delete_dispatcher(arr, obj, axis=None):
    return (arr, obj)

# 使用装饰器，将_delete_dispatcher函数注册为array_function_dispatch的处理函数
@array_function_dispatch(_delete_dispatcher)
def delete(arr, obj, axis=None):
    """
    Return a new array with sub-arrays along an axis deleted. For a one
    dimensional array, this returns those entries not returned by
    `arr[obj]`.

    Parameters
    ----------
    arr : array_like
        Input array.
    obj : slice, int or array of ints
        Indicate indices of sub-arrays to remove along the specified axis.

        .. versionchanged:: 1.19.0
            Boolean indices are now treated as a mask of elements to remove,
            rather than being cast to the integers 0 and 1.

    axis : int, optional
        The axis along which to delete the subarray defined by `obj`.
        If `axis` is None, `obj` is applied to the flattened array.

    Returns
    -------
    out : ndarray
        A copy of `arr` with the elements specified by `obj` removed. Note
        that `delete` does not occur in-place. If `axis` is None, `out` is
        a flattened array.

    See Also
    --------
    insert : Insert elements into an array.
    append : Append elements at the end of an array.

    Notes
    -----
    Often it is preferable to use a boolean mask. For example:

    >>> arr = np.arange(12) + 1
    >>> mask = np.ones(len(arr), dtype=bool)
    >>> mask[[0,2,4]] = False
    >>> result = arr[mask,...]

    Is equivalent to ``np.delete(arr, [0,2,4], axis=0)``, but allows further
    use of `mask`.

    Examples
    --------
    >>> arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
    >>> arr
    array([[ 1,  2,  3,  4],
           [ 5,  6,  7,  8],
           [ 9, 10, 11, 12]])
    >>> np.delete(arr, 1, 0)
    array([[ 1,  2,  3,  4],
           [ 9, 10, 11, 12]])

    >>> np.delete(arr, np.s_[::2], 1)
    array([[ 2,  4],
           [ 6,  8],
           [10, 12]])
    >>> np.delete(arr, [1,3,5], None)
    array([ 1,  3,  5,  7,  8,  9, 10, 11, 12])

    """
    # 将输入数组转换为合适的数组对象
    conv = _array_converter(arr)
    # 将转换后的数组返回给arr
    arr, = conv.as_arrays(subok=False)

    # 获取数组的维数和存储顺序
    ndim = arr.ndim
    arrorder = 'F' if arr.flags.fnc else 'C'
    
    # 如果axis为None，将数组展平以应用obj
    if axis is None:
        if ndim != 1:
            arr = arr.ravel()
        # 对于np.matrix，即使ravel后仍不是1维，需要再次获取维数
        ndim = arr.ndim
        axis = ndim - 1
    else:
        # 将axis标准化为有效的轴索引
        axis = normalize_axis_index(axis, ndim)

    # 创建切片对象列表，用于确定要删除的元素
    slobj = [slice(None)]*ndim
    N = arr.shape[axis]
    newshape = list(arr.shape)
    # 如果 obj 是切片对象
    if isinstance(obj, slice):
        # 获取切片的起始、终止和步长
        start, stop, step = obj.indices(N)
        # 根据步长生成一个范围对象 xr
        xr = range(start, stop, step)
        # 需要删除的元素个数
        numtodel = len(xr)

        # 如果没有需要删除的元素，则直接返回复制后的数组
        if numtodel <= 0:
            return conv.wrap(arr.copy(order=arrorder), to_scalar=False)

        # 如果步长为负数，则反转切片方向
        if step < 0:
            step = -step
            start = xr[-1]
            stop = xr[0] + 1

        # 更新沿指定轴的新形状，减去需要删除的元素个数
        newshape[axis] -= numtodel
        # 创建一个新的空数组
        new = empty(newshape, arr.dtype, arrorder)

        # 复制切片的起始部分
        if start == 0:
            pass
        else:
            slobj[axis] = slice(None, start)
            new[tuple(slobj)] = arr[tuple(slobj)]

        # 复制切片的结束部分
        if stop == N:
            pass
        else:
            slobj[axis] = slice(stop-numtodel, None)
            slobj2 = [slice(None)]*ndim
            slobj2[axis] = slice(stop, None)
            new[tuple(slobj)] = arr[tuple(slobj2)]

        # 复制中间部分
        if step == 1:
            pass
        else:
            # 使用数组索引来处理
            keep = ones(stop-start, dtype=bool)
            keep[:stop-start:step] = False
            slobj[axis] = slice(start, stop-numtodel)
            slobj2 = [slice(None)]*ndim
            slobj2[axis] = slice(start, stop)
            arr = arr[tuple(slobj2)]
            slobj2[axis] = keep
            new[tuple(slobj)] = arr[tuple(slobj2)]

        # 返回处理后的数组
        return conv.wrap(new, to_scalar=False)

    # 如果 obj 是整数类型，且不是布尔类型
    if isinstance(obj, (int, integer)) and not isinstance(obj, bool):
        # 单个值优化标志
        single_value = True
    else:
        # 否则，不是单个值
        single_value = False
        # 暂存原始对象
        _obj = obj
        # 转换为 NumPy 数组
        obj = np.asarray(obj)
        
        # 对于空数组的特殊处理，允许类似索引空列表的情况
        if obj.size == 0 and not isinstance(_obj, np.ndarray):
            obj = obj.astype(intp)
        # 对于只有一个元素且类型为整数的数组，将其转换为单个值处理
        elif obj.size == 1 and obj.dtype.kind in "ui":
            obj = obj.item()
            single_value = True

    # 如果是单个值处理的情况
    if single_value:
        # 对单个值的优化处理
        if (obj < -N or obj >= N):
            raise IndexError(
                "index %i is out of bounds for axis %i with "
                "size %i" % (obj, axis, N))
        if (obj < 0):
            obj += N
        # 更新沿指定轴的新形状，减少一个元素
        newshape[axis] -= 1
        # 创建一个新的空数组
        new = empty(newshape, arr.dtype, arrorder)
        # 复制切片的前半部分
        slobj[axis] = slice(None, obj)
        new[tuple(slobj)] = arr[tuple(slobj)]
        # 复制切片的后半部分
        slobj[axis] = slice(obj, None)
        slobj2 = [slice(None)]*ndim
        slobj2[axis] = slice(obj+1, None)
        new[tuple(slobj)] = arr[tuple(slobj2)]
    else:
        # 如果 obj 的数据类型是布尔型
        if obj.dtype == bool:
            # 如果 obj 的形状不是 (N,)，则抛出数值错误
            if obj.shape != (N,):
                raise ValueError('boolean array argument obj to delete '
                                 'must be one dimensional and match the axis '
                                 'length of {}'.format(N))

            # 优化：使用位反转操作得到 keep 数组
            keep = ~obj
        else:
            # 如果 obj 的数据类型不是布尔型，则创建一个全为 True 的布尔型数组
            keep = ones(N, dtype=bool)
            # 将 obj 中指定的索引位置置为 False
            keep[obj,] = False

        # 根据指定的 axis 修改 slobj 对应的元素
        slobj[axis] = keep
        # 根据修改后的 slobj 对 arr 进行切片操作得到新的数组 new
        new = arr[tuple(slobj)]

    # 使用 conv.wrap() 方法将 new 包装成适当的对象并返回
    return conv.wrap(new, to_scalar=False)
# 定义一个插入操作的分派函数，接收参数 arr、obj、values 和 axis
def _insert_dispatcher(arr, obj, values, axis=None):
    # 返回元组 (arr, obj, values)，用于将此函数作为分派函数的返回结果
    return (arr, obj, values)


# 使用 array_function_dispatch 装饰器将 _insert_dispatcher 函数注册为 insert 函数的分派函数
@array_function_dispatch(_insert_dispatcher)
def insert(arr, obj, values, axis=None):
    """
    在给定索引之前的指定轴上插入值。

    Parameters
    ----------
    arr : array_like
        输入数组。
    obj : int, slice 或 int 序列
        定义插入值 `values` 前的索引或索引。

        .. versionadded:: 1.8.0

        当 `obj` 是单个标量或包含一个元素的序列时支持多次插入（类似于多次调用 insert）。
    values : array_like
        要插入到 `arr` 中的值。如果 `values` 的类型与 `arr` 不同，将转换为 `arr` 的类型。
        `values` 应该被形状化，以便 ``arr[...,obj,...] = values`` 是合法的。
    axis : int, optional
        插入 `values` 的轴。如果 `axis` 是 None，则首先将 `arr` 展平。

    Returns
    -------
    out : ndarray
        插入了 `values` 的 `arr` 的副本。注意，`insert` 不是原地插入：会返回一个新数组。如果
        `axis` 是 None，`out` 是一个展平的数组。

    See Also
    --------
    append : 在数组末尾添加元素。
    concatenate : 沿着现有轴连接一系列数组。
    delete : 从数组中删除元素。

    Notes
    -----
    高维插入的注意事项：``obj=0`` 与 ``obj=[0]`` 之间的行为差异非常大，就像 ``arr[:,0,:] = values``
    与 ``arr[:,[0],:] = values`` 之间的差异一样。这是因为基本和高级 :ref:`索引 <basics.indexing>` 之间的差异。

    Examples
    --------
    >>> a = np.arange(6).reshape(3, 2)
    >>> a
    array([[0, 1],
           [2, 3],
           [4, 5]])
    >>> np.insert(a, 1, 6)
    array([0, 6, 1, 2, 3, 4, 5])
    >>> np.insert(a, 1, 6, axis=1)
    array([[0, 6, 1],
           [2, 6, 3],
           [4, 6, 5]])

    序列与标量之间的区别，展示了 `obj=[1]` 与 `obj=1` 的不同行为：

    >>> np.insert(a, [1], [[7],[8],[9]], axis=1)
    array([[0, 7, 1],
           [2, 8, 3],
           [4, 9, 5]])
    >>> np.insert(a, 1, [[7],[8],[9]], axis=1)
    array([[0, 7, 8, 9, 1],
           [2, 7, 8, 9, 3],
           [4, 7, 8, 9, 5]])
    >>> np.array_equal(np.insert(a, 1, [7, 8, 9], axis=1),
    ...                np.insert(a, [1], [[7],[8],[9]], axis=1))
    True

    >>> b = a.flatten()
    >>> b
    array([0, 1, 2, 3, 4, 5])
    >>> np.insert(b, [2, 2], [6, 7])
    array([0, 1, 6, 7, 2, 3, 4, 5])

    >>> np.insert(b, slice(2, 4), [7, 8])
    array([0, 1, 7, 2, 8, 3, 4, 5])

    >>> np.insert(b, [2, 2], [7.13, False]) # 类型转换
    array([0, 1, 7, 0, 2, 3, 4, 5])

    >>> x = np.arange(8).reshape(2, 4)
    >>> idx = (1, 3)
    >>> np.insert(x, idx, 999, axis=1)
    """
    array([[  0, 999,   1,   2, 999,   3],
           [  4, 999,   5,   6, 999,   7]])

    """
    # 转换数组为适合处理的形式
    conv = _array_converter(arr)
    # 将数组转换为数组对象，同时确保不接受子类对象
    arr, = conv.as_arrays(subok=False)

    # 获取数组的维度
    ndim = arr.ndim
    # 确定数组的存储顺序，是行优先 ('F') 还是列优先 ('C')
    arrorder = 'F' if arr.flags.fnc else 'C'

    # 如果未指定轴向，则根据数组的维度调整数组形状
    if axis is None:
        if ndim != 1:
            arr = arr.ravel()  # 将多维数组展平为一维数组
        ndim = arr.ndim  # 重新获取数组的维度（可能已被展平）
        axis = ndim - 1  # 默认轴向为最后一个维度
    else:
        axis = normalize_axis_index(axis, ndim)  # 根据指定轴向规范化索引

    # 创建一个用于切片的对象列表，长度为数组的维度
    slobj = [slice(None)] * ndim
    N = arr.shape[axis]  # 获取指定轴向上的长度
    newshape = list(arr.shape)  # 将数组形状转换为列表形式

    # 如果 obj 是切片对象，则转换为 range 对象
    if isinstance(obj, slice):
        indices = arange(*obj.indices(N), dtype=intp)
    else:
        # 否则需要复制 obj，因为后续会就地修改 indices
        indices = np.array(obj)
        if indices.dtype == bool:
            # 如果 obj 是布尔数组，警告未来将布尔数组视为布尔索引
            warnings.warn(
                "in the future insert will treat boolean arrays and "
                "array-likes as a boolean index instead of casting it to "
                "integer", FutureWarning, stacklevel=2)
            indices = indices.astype(intp)  # 将布尔数组转换为整数索引
        elif indices.ndim > 1:
            # 如果 indices 不是一维数组或标量，抛出异常
            raise ValueError(
                "index array argument obj to insert must be one dimensional "
                "or scalar")

    # 如果 indices 大小为 1，处理单个索引情况
    if indices.size == 1:
        index = indices.item()  # 获取单个索引的值
        # 检查索引是否超出范围
        if index < -N or index > N:
            raise IndexError(f"index {obj} is out of bounds for axis {axis} "
                             f"with size {N}")
        if index < 0:
            index += N  # 将负索引转换为正索引

        # 将 values 转换为数组，确保维度和数据类型与 arr 相同
        values = array(values, copy=None, ndmin=arr.ndim, dtype=arr.dtype)

        # 处理广播情况，使得 broadcasting 的行为符合预期
        if indices.ndim == 0:
            values = np.moveaxis(values, 0, axis)

        # 计算新数组的长度
        numnew = values.shape[axis]
        newshape[axis] += numnew  # 调整新数组的指定轴向的长度
        new = empty(newshape, arr.dtype, arrorder)  # 创建新数组

        # 复制旧数组的部分到新数组
        slobj[axis] = slice(None, index)
        new[tuple(slobj)] = arr[tuple(slobj)]

        # 将 values 插入到新数组的指定位置
        slobj[axis] = slice(index, index + numnew)
        new[tuple(slobj)] = values

        # 将剩余部分的旧数组复制到新数组
        slobj[axis] = slice(index + numnew, None)
        slobj2 = [slice(None)] * ndim
        slobj2[axis] = slice(index, None)
        new[tuple(slobj)] = arr[tuple(slobj2)]

        return conv.wrap(new, to_scalar=False)  # 将新数组包装成适当的对象并返回
    # 如果 indices 是空数组且 obj 不是 ndarray 类型，则将空数组安全地转换为 intp 类型
    elif indices.size == 0 and not isinstance(obj, np.ndarray):
        indices = indices.astype(intp)

    # 将小于 0 的 indices 元素加上数组总长度 N，以确保 indices 中的索引非负
    indices[indices < 0] += N

    # 获取 indices 的长度作为新数据的元素数量
    numnew = len(indices)

    # 对 indices 进行稳定排序，即保持相等元素的相对位置不变
    order = indices.argsort(kind='mergesort')   # stable sort

    # 将排序后的 indices 中的元素依次加上对应的序号，以便在数组中进行正确的插入
    indices[order] += np.arange(numnew)

    # 在指定的轴上增加新的形状尺寸
    newshape[axis] += numnew

    # 创建一个新的布尔掩码，用于标记被替换的旧值
    old_mask = ones(newshape[axis], dtype=bool)
    old_mask[indices] = False

    # 创建一个指定形状和类型的新数组
    new = empty(newshape, arr.dtype, arrorder)

    # 创建切片对象的列表，并将 indices 和 old_mask 分别作为切片对象的一部分
    slobj2 = [slice(None)]*ndim
    slobj[axis] = indices
    slobj2[axis] = old_mask

    # 将 values 插入到 new 的 indices 所指定的位置
    new[tuple(slobj)] = values

    # 将原始数组 arr 插入到 new 的 old_mask 所指定的位置
    new[tuple(slobj2)] = arr

    # 使用 conv.wrap 函数将 new 数组包装并返回，确保返回的数据不是标量
    return conv.wrap(new, to_scalar=False)
# 定义一个辅助函数 _append_dispatcher，用于数组操作的派发
def _append_dispatcher(arr, values, axis=None):
    # 直接返回输入的数组 arr 和 values
    return (arr, values)


# 使用 array_function_dispatch 装饰器，将 _append_dispatcher 和 append 函数关联起来
@array_function_dispatch(_append_dispatcher)
def append(arr, values, axis=None):
    """
    Append values to the end of an array.

    Parameters
    ----------
    arr : array_like
        Values are appended to a copy of this array.
    values : array_like
        These values are appended to a copy of `arr`.  It must be of the
        correct shape (the same shape as `arr`, excluding `axis`).  If
        `axis` is not specified, `values` can be any shape and will be
        flattened before use.
    axis : int, optional
        The axis along which `values` are appended.  If `axis` is not
        given, both `arr` and `values` are flattened before use.

    Returns
    -------
    append : ndarray
        A copy of `arr` with `values` appended to `axis`.  Note that
        `append` does not occur in-place: a new array is allocated and
        filled.  If `axis` is None, `out` is a flattened array.

    See Also
    --------
    insert : Insert elements into an array.
    delete : Delete elements from an array.

    Examples
    --------
    >>> np.append([1, 2, 3], [[4, 5, 6], [7, 8, 9]])
    array([1, 2, 3, ..., 7, 8, 9])

    When `axis` is specified, `values` must have the correct shape.

    >>> np.append([[1, 2, 3], [4, 5, 6]], [[7, 8, 9]], axis=0)
    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])

    >>> np.append([[1, 2, 3], [4, 5, 6]], [7, 8, 9], axis=0)
    Traceback (most recent call last):
        ...
    ValueError: all the input arrays must have same number of dimensions, but
    the array at index 0 has 2 dimension(s) and the array at index 1 has 1
    dimension(s)

    >>> a = np.array([1, 2], dtype=int)
    >>> c = np.append(a, [])
    >>> c
    array([1., 2.])
    >>> c.dtype
    float64

    Default dtype for empty ndarrays is `float64` thus making the output of dtype
    `float64` when appended with dtype `int64`

    """
    # 将 arr 转换为数组对象
    arr = asanyarray(arr)
    # 如果未指定 axis
    if axis is None:
        # 如果 arr 的维度不为1，则将其展平为一维数组
        if arr.ndim != 1:
            arr = arr.ravel()
        # 将 values 展平为一维数组
        values = ravel(values)
        # axis 设为 arr 的维度数减一
        axis = arr.ndim - 1
    # 使用 concatenate 函数沿指定轴将 arr 和 values 连接起来，返回连接后的新数组
    return concatenate((arr, values), axis=axis)


# 定义一个辅助函数 _digitize_dispatcher，用于数字化函数的派发
def _digitize_dispatcher(x, bins, right=None):
    # 直接返回输入的 x 和 bins
    return (x, bins)


# 使用 array_function_dispatch 装饰器，将 _digitize_dispatcher 和 digitize 函数关联起来
@array_function_dispatch(_digitize_dispatcher)
def digitize(x, bins, right=False):
    """
    Return the indices of the bins to which each value in input array belongs.

    =========  =============  ============================
    `right`    order of bins  returned index `i` satisfies
    =========  =============  ============================
    ``False``  increasing     ``bins[i-1] <= x < bins[i]``
    ``True``   increasing     ``bins[i-1] < x <= bins[i]``
    ``False``  decreasing     ``bins[i-1] > x >= bins[i]``
    ``True``   decreasing     ``bins[i-1] >= x > bins[i]``
    =========  =============  ============================

    If values in `x` are beyond the bounds of `bins`, 0 or ``len(bins)`` is
    """
    # 函数体未完，未提供完整的代码块



# 定义一个辅助函数 _append_dispatcher，用于数组操作的派发
def _append_dispatcher(arr, values, axis=None):
    # 直接返回输入的数组 arr 和 values
    return (arr, values)


# 使用 array_function_dispatch 装饰器，将 _append_dispatcher 和 append 函数关联起来
@array_function_dispatch(_append_dispatcher)
def append(arr, values, axis=None):
    """
    Append values to the end of an array.

    Parameters
    ----------
    arr : array_like
        Values are appended to a copy of this array.
    values : array_like
        These values are appended to a copy of `arr`.  It must be of the
        correct shape (the same shape as `arr`, excluding `axis`).  If
        `axis` is not specified, `values` can be any shape and will be
        flattened before use.
    axis : int, optional
        The axis along which `values` are appended.  If `axis` is not
        given, both `arr` and `values` are flattened before use.

    Returns
    -------
    append : ndarray
        A copy of `arr` with `values` appended to `axis`.  Note that
        `append` does not occur in-place: a new array is allocated and
        filled.  If `axis` is None, `out` is a flattened array.

    See Also
    --------
    insert : Insert elements into an array.
    delete : Delete elements from an array.

    Examples
    --------
    >>> np.append([1, 2, 3], [[4, 5, 6], [7, 8, 9]])
    array([1, 2, 3, ..., 7, 8, 9])

    When `axis` is specified, `values` must have the correct shape.

    >>> np.append([[1, 2, 3], [4, 5, 6]], [[7, 8, 9]], axis=0)
    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])

    >>> np.append([[1, 2, 3], [4, 5, 6]], [7, 8, 9], axis=0)
    Traceback (most recent call last):
        ...
    ValueError: all the input arrays must have same number of dimensions, but
    the array at index 0 has 2 dimension(s) and the array at index 1 has 1
    dimension(s)

    >>> a = np.array([1, 2], dtype=int)
    >>> c = np.append(a, [])
    >>> c
    array([1., 2.])
    >>> c.dtype
    float64

    Default dtype for empty ndarrays is `float64` thus making the output of dtype
    `float64` when appended with dtype `int64`

    """
    # 将 arr 转换为数组对象
    arr = asanyarray(arr)
    # 如果未指定 axis
    if axis is None:
        # 如果 arr 的维度不为1，则将其展平为一维数组
        if arr.ndim != 1:
            arr = arr.ravel()
        # 将 values 展平为一维数组
        values = ravel(values)
        # axis 设为 arr 的维度数减一
        axis = arr.ndim - 1
    # 使用 concatenate 函数沿指定轴将 arr 和 values 连接起来，返回连接后的新数组
    return concatenate((arr, values), axis=axis)


# 定义一个辅助函数 _digitize_dispatcher，用于数字化函数的派发
def _digitize_dispatcher(x, bins, right=None):
    # 直接返回输入的 x 和 bins
    return (x, bins)


# 使用 array_function_dispatch 装饰器，将 _digitize_dispatcher 和 digitize 函数关联起来
@array_function_dispatch(_digitize_dispatcher)
def digitize(x, bins, right=False):
    """
    Return the indices of the bins to which each value in input array belongs.

    =========  =============  ============================
    `right`    order of bins  returned index `i` satisfies
    =========  =============  ============================
    ``False``  increasing     ``bins[i-1] <= x < bins[i]``
    ``True``   increasing     ``bins[i-1] < x <= bins[i]``
    ``False``  decreasing     ``bins[i-1] > x >= bins[i]``
    ``True``   decreasing     ``bins[i-1] >= x > bins[i]``
    =========  =============  ============================

    If values in `x` are beyond the bounds of `bins`, 0 or ``len(bins)`` is
    """
    # 函数体未完，未提供完整的代码块
    # 将输入参数 x 和 bins 转换为 NumPy 数组
    x = _nx.asarray(x)
    bins = _nx.asarray(bins)
    
    # 如果输入数组 x 的数据类型是复数，抛出类型错误异常
    if np.issubdtype(x.dtype, _nx.complexfloating):
        raise TypeError("x may not be complex")
    
    # 检查 bins 是否单调增加或单调减少，返回值为 0 表示不满足条件，抛出数值错误异常
    mono = _monotonicity(bins)
    if mono == 0:
        raise ValueError("bins must be monotonically increasing or decreasing")
    
    # 根据 right 参数确定 side 的取值，用于后续的 searchsorted 函数调用
    side = 'left' if right else 'right'
    # 如果 mono 等于 -1，执行以下操作：
    # 反转 bins 数组，并对结果取反
    return len(bins) - _nx.searchsorted(bins[::-1], x, side=side)
    # 如果 mono 不等于 -1，则执行正常的 searchsorted 操作
    return _nx.searchsorted(bins, x, side=side)
```