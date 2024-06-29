# `.\numpy\numpy\lib\_nanfunctions_impl.py`

```
"""
Functions that ignore NaN.

Functions
---------

- `nanmin` -- minimum non-NaN value
- `nanmax` -- maximum non-NaN value
- `nanargmin` -- index of minimum non-NaN value
- `nanargmax` -- index of maximum non-NaN value
- `nansum` -- sum of non-NaN values
- `nanprod` -- product of non-NaN values
- `nancumsum` -- cumulative sum of non-NaN values
- `nancumprod` -- cumulative product of non-NaN values
- `nanmean` -- mean of non-NaN values
- `nanvar` -- variance of non-NaN values
- `nanstd` -- standard deviation of non-NaN values
- `nanmedian` -- median of non-NaN values
- `nanquantile` -- qth quantile of non-NaN values
- `nanpercentile` -- qth percentile of non-NaN values

"""
import functools  # 导入 functools 模块
import warnings  # 导入 warnings 模块
import numpy as np  # 导入 NumPy 库，并简称为 np
import numpy._core.numeric as _nx  # 导入 NumPy 内部的 numeric 模块，简称为 _nx
from numpy.lib import _function_base_impl as fnb  # 导入 NumPy 库中的 _function_base_impl 模块，简称为 fnb
from numpy.lib._function_base_impl import _weights_are_valid  # 从 _function_base_impl 模块中导入 _weights_are_valid 函数
from numpy._core import overrides  # 导入 NumPy 内部的 overrides 模块

# 创建一个偏函数 array_function_dispatch，用于重载 NumPy 数组函数的调度
array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')

# 定义 __all__ 列表，列出公开的函数名
__all__ = [
    'nansum', 'nanmax', 'nanmin', 'nanargmax', 'nanargmin', 'nanmean',
    'nanmedian', 'nanpercentile', 'nanvar', 'nanstd', 'nanprod',
    'nancumsum', 'nancumprod', 'nanquantile'
    ]

# 定义私有函数 _nan_mask，用于生成 NaN 掩码
def _nan_mask(a, out=None):
    """
    Parameters
    ----------
    a : array-like
        Input array with at least 1 dimension.
    out : ndarray, optional
        Alternate output array in which to place the result.  The default
        is ``None``; if provided, it must have the same shape as the
        expected output and will prevent the allocation of a new array.

    Returns
    -------
    y : bool ndarray or True
        A bool array where ``np.nan`` positions are marked with ``False``
        and other positions are marked with ``True``. If the type of ``a``
        is such that it can't possibly contain ``np.nan``, returns ``True``.
    """
    # 假设 a 是一个数组类型，在这个私有函数中

    # 如果 a 的数据类型种类不是 'fc'，即不是复数或浮点数，返回 True
    if a.dtype.kind not in 'fc':
        return True

    # 使用 np.isnan 函数检查数组 a 中的 NaN 值，生成对应的布尔数组 y
    y = np.isnan(a, out=out)
    # 使用 np.invert 函数取反布尔数组 y，并赋值给 y
    y = np.invert(y, out=y)
    return y

# 定义私有函数 _replace_nan，用于替换数组中的 NaN 值
def _replace_nan(a, val):
    """
    If `a` is of inexact type, make a copy of `a`, replace NaNs with
    the `val` value, and return the copy together with a boolean mask
    marking the locations where NaNs were present. If `a` is not of
    inexact type, do nothing and return `a` together with a mask of None.

    Note that scalars will end up as array scalars, which is important
    for using the result as the value of the out argument in some
    operations.

    Parameters
    ----------
    a : array-like
        Input array.
    val : float
        NaN values are set to val before doing the operation.

    Returns
    -------
    y : ndarray
        If `a` is of inexact type, return a copy of `a` with the NaNs
        replaced by the fill value, otherwise return `a`.
    mask: {bool, None}
        If `a` is of inexact type, return a boolean mask marking locations of
        NaNs, otherwise return None.
    """
    # 如果 a 的数据类型不是精确类型（即不是整数），则复制 a 并替换其中的 NaN 值为 val
    # 返回替换后的数组和 NaN 值的布尔掩码
    if np.issubdtype(a.dtype, np.inexact):
        y = np.array(a, copy=True)
        mask = np.isnan(a)
        y[mask] = val
        return y, mask
    else:
        # 如果 a 的数据类型是精确类型（如整数），直接返回 a 和 None
        return a, None
    # 将输入的参数 a 转换为 NumPy 数组
    a = np.asanyarray(a)
    
    # 检查数组的数据类型是否为 np.object_
    if a.dtype == np.object_:
        # 对象数组不支持 `isnan` 操作（见 GitHub issue gh-9009），因此做出猜测
        # 创建一个布尔类型的掩码数组，用于标识数组中的 NaN 值
        mask = np.not_equal(a, a, dtype=bool)
    elif issubclass(a.dtype.type, np.inexact):
        # 如果数组的数据类型是浮点数或复数类型的子类，则生成一个掩码数组，标识 NaN 值
        mask = np.isnan(a)
    else:
        # 如果不满足上述条件，则不生成掩码数组
        mask = None
    
    if mask is not None:
        # 如果存在掩码数组，复制输入数组 a，并将 val 的值复制到掩码位置
        a = np.array(a, subok=True, copy=True)
        np.copyto(a, val, where=mask)
    
    # 返回处理后的数组 a 和掩码数组 mask（可能为 None）
    return a, mask
# 将数组 `a` 中 `mask` 为 True 的位置替换为 NaN。与 `np.copyto` 不同，此函数处理 `a` 是 numpy 标量的情况。
def _copyto(a, val, mask):
    if isinstance(a, np.ndarray):
        # 使用 `np.copyto` 将 `val` 复制到 `a` 中 `mask` 为 True 的位置，不安全类型转换。
        np.copyto(a, val, where=mask, casting='unsafe')
    else:
        # 如果 `a` 是标量，则将其类型转换为 `val` 的类型。
        a = a.dtype.type(val)
    return a


# 从一维数组 `arr1d` 中移除 NaN 值，返回结果数组。
def _remove_nan_1d(arr1d, second_arr1d=None, overwrite_input=False):
    if arr1d.dtype == object:
        # 对象数组不支持 `isnan`，因此用 `np.not_equal` 作为替代方案。
        c = np.not_equal(arr1d, arr1d, dtype=bool)
    else:
        # 判断数组 `arr1d` 中的 NaN 值。
        c = np.isnan(arr1d)

    # 找出所有 NaN 值的索引。
    s = np.nonzero(c)[0]

    # 如果所有元素都是 NaN，发出警告并返回空数组。
    if s.size == arr1d.size:
        warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=6)
        if second_arr1d is None:
            return arr1d[:0], None, True
        else:
            return arr1d[:0], second_arr1d[:0], True
    # 如果没有 NaN 值，直接返回原数组。
    elif s.size == 0:
        return arr1d, second_arr1d, overwrite_input
    else:
        if not overwrite_input:
            # 如果不允许在原地修改 `arr1d`，则复制一份。
            arr1d = arr1d.copy()

        # 从数组末尾选择非 NaN 值。
        enonan = arr1d[-s.size:][~c[-s.size:]]
        # 将末尾的非 NaN 值填充到数组开头的 NaN 位置。
        arr1d[s[:enonan.size]] = enonan

        if second_arr1d is None:
            return arr1d[:-s.size], None, True
        else:
            if not overwrite_input:
                # 如果不允许在原地修改 `second_arr1d`，则复制一份。
                second_arr1d = second_arr1d.copy()
            enonan = second_arr1d[-s.size:][~c[-s.size:]]
            second_arr1d[s[:enonan.size]] = enonan

            return arr1d[:-s.size], second_arr1d[:-s.size], True


# 计算数组 `a` 与 `b` 的逐元素除法，忽略无效结果。
def _divide_by_count(a, b, out=None):
    """
    Compute a/b ignoring invalid results. If `a` is an array the division
    is done in place. If `a` is a scalar, then its type is preserved in the
    """
    # 使用 numpy 的错误状态管理器，忽略无效和除零错误
    with np.errstate(invalid='ignore', divide='ignore'):
        # 如果 a 是 ndarray 类型
        if isinstance(a, np.ndarray):
            # 如果没有提供输出数组 out，则直接在 a 上执行不安全类型转换的除法
            if out is None:
                return np.divide(a, b, out=a, casting='unsafe')
            # 如果提供了输出数组 out，则在指定的输出数组上执行不安全类型转换的除法
            else:
                return np.divide(a, b, out=out, casting='unsafe')
        else:
            # 如果没有提供输出数组 out
            if out is None:
                # 针对减少的对象数组进行预防措施
                try:
                    return a.dtype.type(a / b)
                # 捕获 AttributeError 异常，直接执行标量的除法
                except AttributeError:
                    return a / b
            # 如果提供了输出数组 out
            else:
                # 目前 numpy 标量可以输出到零维数组，这种情况可能有问题
                return np.divide(a, b, out=out, casting='unsafe')
# 定义一个分发器函数 `_nanmin_dispatcher`，用于分发参数到真正的 `nanmin` 函数
def _nanmin_dispatcher(a, axis=None, out=None, keepdims=None,
                       initial=None, where=None):
    # 返回参数 `a` 和 `out`，这些参数将被传递给 `nanmin` 函数
    return (a, out)


# 使用 `array_function_dispatch` 装饰器，将 `_nanmin_dispatcher` 函数注册为 `nanmin` 函数的分发器
@array_function_dispatch(_nanmin_dispatcher)
def nanmin(a, axis=None, out=None, keepdims=np._NoValue, initial=np._NoValue,
           where=np._NoValue):
    """
    Return minimum of an array or minimum along an axis, ignoring any NaNs.
    When all-NaN slices are encountered a ``RuntimeWarning`` is raised and
    Nan is returned for that slice.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose minimum is desired. If `a` is not an
        array, a conversion is attempted.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the minimum is computed. The default is to compute
        the minimum of the flattened array.
    out : ndarray, optional
        Alternate output array in which to place the result.  The default
        is ``None``; if provided, it must have the same shape as the
        expected output, but the type will be cast if necessary. See
        :ref:`ufuncs-output-type` for more details.

        .. versionadded:: 1.8.0
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `a`.

        If the value is anything but the default, then
        `keepdims` will be passed through to the `min` method
        of sub-classes of `ndarray`.  If the sub-classes methods
        does not implement `keepdims` any exceptions will be raised.

        .. versionadded:: 1.8.0
    initial : scalar, optional
        The maximum value of an output element. Must be present to allow
        computation on empty slice. See `~numpy.ufunc.reduce` for details.

        .. versionadded:: 1.22.0
    where : array_like of bool, optional
        Elements to compare for the minimum. See `~numpy.ufunc.reduce`
        for details.

        .. versionadded:: 1.22.0

    Returns
    -------
    nanmin : ndarray
        An array with the same shape as `a`, with the specified axis
        removed.  If `a` is a 0-d array, or if axis is None, an ndarray
        scalar is returned.  The same dtype as `a` is returned.

    See Also
    --------
    nanmax :
        The maximum value of an array along a given axis, ignoring any NaNs.
    amin :
        The minimum value of an array along a given axis, propagating any NaNs.
    fmin :
        Element-wise minimum of two arrays, ignoring any NaNs.
    minimum :
        Element-wise minimum of two arrays, propagating any NaNs.
    isnan :
        Shows which elements are Not a Number (NaN).
    isfinite:
        Shows which elements are neither NaN nor infinity.

    amax, fmax, maximum

    Notes
    -----
    NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754). This means that Not a Number is not equivalent to infinity.
    """
    """
    Positive infinity is treated as a very large number and negative
    infinity is treated as a very small (i.e. negative) number.

    If the input has a integer type the function is equivalent to np.min.

    Examples
    --------
    >>> a = np.array([[1, 2], [3, np.nan]])
    >>> np.nanmin(a)
    1.0
    >>> np.nanmin(a, axis=0)
    array([1.,  2.])
    >>> np.nanmin(a, axis=1)
    array([1.,  3.])

    When positive infinity and negative infinity are present:

    >>> np.nanmin([1, 2, np.nan, np.inf])
    1.0
    >>> np.nanmin([1, 2, np.nan, -np.inf])
    -inf

    """
    # 初始化空的关键字参数字典
    kwargs = {}
    # 如果 keepdims 参数不是默认值 np._NoValue，则设置到 kwargs 字典中
    if keepdims is not np._NoValue:
        kwargs['keepdims'] = keepdims
    # 如果 initial 参数不是默认值 np._NoValue，则设置到 kwargs 字典中
    if initial is not np._NoValue:
        kwargs['initial'] = initial
    # 如果 where 参数不是默认值 np._NoValue，则设置到 kwargs 字典中
    if where is not np._NoValue:
        kwargs['where'] = where

    # 如果 a 是 ndarray 类型且其 dtype 不是 np.object_，使用快速的 np.fmin.reduce 函数
    if type(a) is np.ndarray and a.dtype != np.object_:
        # 使用 np.fmin.reduce 函数计算最小值，可以指定轴向和输出数组等参数，同时传入 kwargs 字典中的其他参数
        res = np.fmin.reduce(a, axis=axis, out=out, **kwargs)
        # 如果结果中包含 NaN，发出警告
        if np.isnan(res).any():
            warnings.warn("All-NaN slice encountered", RuntimeWarning,
                          stacklevel=2)
    else:
        # 对于非 ndarray 或者包含 object 类型的数组，使用更慢但更安全的方法处理 NaN
        # 将数组中的 NaN 替换为正无穷 +np.inf
        a, mask = _replace_nan(a, +np.inf)
        # 使用 np.amin 函数计算最小值，可以指定轴向和输出数组等参数，同时传入 kwargs 字典中的其他参数
        res = np.amin(a, axis=axis, out=out, **kwargs)
        # 如果 mask 为空，直接返回结果
        if mask is None:
            return res

        # 检查是否存在全为 NaN 的轴
        kwargs.pop("initial", None)
        mask = np.all(mask, axis=axis, **kwargs)
        # 如果存在全为 NaN 的轴，用 NaN 替换结果，并发出警告
        if np.any(mask):
            res = _copyto(res, np.nan, mask)
            warnings.warn("All-NaN axis encountered", RuntimeWarning,
                          stacklevel=2)
    # 返回计算得到的最小值结果
    return res
# 定义一个分派函数 `_nanmax_dispatcher`，用于确定 `nanmax` 函数的调度器
def _nanmax_dispatcher(a, axis=None, out=None, keepdims=None,
                       initial=None, where=None):
    # 返回输入的参数 `a` 和 `out`
    return (a, out)


# 使用装饰器 `array_function_dispatch` 将 `_nanmax_dispatcher` 函数注册为 `nanmax` 函数的分发函数
@array_function_dispatch(_nanmax_dispatcher)
# 定义函数 `nanmax`，计算数组或沿指定轴的最大值，忽略 NaN 值
def nanmax(a, axis=None, out=None, keepdims=np._NoValue, initial=np._NoValue,
           where=np._NoValue):
    """
    Return the maximum of an array or maximum along an axis, ignoring any
    NaNs.  When all-NaN slices are encountered a ``RuntimeWarning`` is
    raised and NaN is returned for that slice.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose maximum is desired. If `a` is not an
        array, a conversion is attempted.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the maximum is computed. The default is to compute
        the maximum of the flattened array.
    out : ndarray, optional
        Alternate output array in which to place the result.  The default
        is ``None``; if provided, it must have the same shape as the
        expected output, but the type will be cast if necessary. See
        :ref:`ufuncs-output-type` for more details.

        .. versionadded:: 1.8.0
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `a`.

        If the value is anything but the default, then
        `keepdims` will be passed through to the `max` method
        of sub-classes of `ndarray`.  If the sub-classes methods
        does not implement `keepdims` any exceptions will be raised.

        .. versionadded:: 1.8.0
    initial : scalar, optional
        The minimum value of an output element. Must be present to allow
        computation on empty slice. See `~numpy.ufunc.reduce` for details.

        .. versionadded:: 1.22.0
    where : array_like of bool, optional
        Elements to compare for the maximum. See `~numpy.ufunc.reduce`
        for details.

        .. versionadded:: 1.22.0

    Returns
    -------
    nanmax : ndarray
        An array with the same shape as `a`, with the specified axis removed.
        If `a` is a 0-d array, or if axis is None, an ndarray scalar is
        returned.  The same dtype as `a` is returned.

    See Also
    --------
    nanmin :
        The minimum value of an array along a given axis, ignoring any NaNs.
    amax :
        The maximum value of an array along a given axis, propagating any NaNs.
    fmax :
        Element-wise maximum of two arrays, ignoring any NaNs.
    maximum :
        Element-wise maximum of two arrays, propagating any NaNs.
    isnan :
        Shows which elements are Not a Number (NaN).
    isfinite:
        Shows which elements are neither NaN nor infinity.

    amin, fmin, minimum

    Notes
    -----
    NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    """
    # 函数np.nanmax返回数组中的最大值，忽略NaN。可以指定轴向进行计算。
    # 参数kwargs用于接收可选参数keepdims、initial、where，这些参数会传递给底层的最大值计算函数。
    kwargs = {}
    if keepdims is not np._NoValue:
        kwargs['keepdims'] = keepdims
    if initial is not np._NoValue:
        kwargs['initial'] = initial
    if where is not np._NoValue:
        kwargs['where'] = where

    if type(a) is np.ndarray and a.dtype != np.object_:
        # 如果输入数组a是ndarray类型且不是对象数组，可以使用np.fmax.reduce快速计算最大值。
        # 这种方法快速但不安全，可能对ndarray的子类或对象数组（不支持isnan或fmax的正确实现）有问题。
        res = np.fmax.reduce(a, axis=axis, out=out, **kwargs)
        # 检查结果中是否有NaN值，如果有则发出警告。
        if np.isnan(res).any():
            warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=2)
    else:
        # 如果输入数组a不是ndarray类型或是对象数组，使用更慢但更安全的方法。
        # 替换数组a中的NaN为-inf，然后使用np.amax计算最大值。
        a, mask = _replace_nan(a, -np.inf)
        res = np.amax(a, axis=axis, out=out, **kwargs)
        if mask is None:
            return res

        # 检查是否有完全由NaN组成的轴。
        kwargs.pop("initial", None)
        mask = np.all(mask, axis=axis, **kwargs)
        if np.any(mask):
            # 如果存在完全由NaN组成的轴，将对应位置的res替换为NaN，并发出警告。
            res = _copyto(res, np.nan, mask)
            warnings.warn("All-NaN axis encountered", RuntimeWarning, stacklevel=2)
    # 返回计算得到的最大值res。
    return res
# 定义一个分派函数，接受参数 a, axis, out，并返回一个元组 (a,)
def _nanargmin_dispatcher(a, axis=None, out=None, *, keepdims=None):
    return (a,)


# 使用 array_function_dispatch 装饰器，将 _nanargmin_dispatcher 分派给 nanargmin 函数
@array_function_dispatch(_nanargmin_dispatcher)
def nanargmin(a, axis=None, out=None, *, keepdims=np._NoValue):
    """
    返回忽略 NaN 值后，在指定轴上的最小值的索引。对于全为 NaN 的切片，抛出 ValueError。警告：如果切片包含仅为 NaN 和 Inf 的值，结果不可信。

    Parameters
    ----------
    a : array_like
        输入数据。
    axis : int, optional
        操作的轴。默认为扁平化的输入。
    out : array, optional
        如果提供，结果将插入到此数组中。应具有适当的形状和数据类型。

        .. versionadded:: 1.22.0
    keepdims : bool, optional
        如果设置为 True，则减少的轴将保留在结果中作为尺寸为一的维度。使用此选项，结果将正确广播到数组。

        .. versionadded:: 1.22.0

    Returns
    -------
    index_array : ndarray
        索引数组或单个索引值。

    See Also
    --------
    argmin, nanargmax

    Examples
    --------
    >>> a = np.array([[np.nan, 4], [2, 3]])
    >>> np.argmin(a)
    0
    >>> np.nanargmin(a)
    2
    >>> np.nanargmin(a, axis=0)
    array([1, 1])
    >>> np.nanargmin(a, axis=1)
    array([1, 0])

    """
    # 调用 _replace_nan 函数替换数组 a 中的 NaN 值为 np.inf，并返回替换后的结果和掩码
    a, mask = _replace_nan(a, np.inf)
    # 如果掩码不为空且存在元素
    if mask is not None and mask.size:
        # 沿着轴 axis 检查掩码是否全为 True
        mask = np.all(mask, axis=axis)
        # 如果有任何 True 存在，抛出 ValueError 异常
        if np.any(mask):
            raise ValueError("All-NaN slice encountered")
    # 调用 np.argmin 函数计算数组 a 在指定轴上的最小值的索引，并返回结果
    res = np.argmin(a, axis=axis, out=out, keepdims=keepdims)
    return res


# 定义一个分派函数，接受参数 a, axis, out，并返回一个元组 (a,)
def _nanargmax_dispatcher(a, axis=None, out=None, *, keepdims=None):
    return (a,)


# 使用 array_function_dispatch 装饰器，将 _nanargmax_dispatcher 分派给 nanargmax 函数
@array_function_dispatch(_nanargmax_dispatcher)
def nanargmax(a, axis=None, out=None, *, keepdims=np._NoValue):
    """
    返回忽略 NaN 值后，在指定轴上的最大值的索引。对于全为 NaN 的切片，抛出 ValueError。警告：如果切片包含仅为 NaN 和 -Inf 的值，结果不可信。

    Parameters
    ----------
    a : array_like
        输入数据。
    axis : int, optional
        操作的轴。默认为扁平化的输入。
    out : array, optional
        如果提供，结果将插入到此数组中。应具有适当的形状和数据类型。

        .. versionadded:: 1.22.0
    keepdims : bool, optional
        如果设置为 True，则减少的轴将保留在结果中作为尺寸为一的维度。使用此选项，结果将正确广播到数组。

        .. versionadded:: 1.22.0

    Returns
    -------
    index_array : ndarray
        索引数组或单个索引值。

    See Also
    --------
    argmax, nanargmin

    Examples
    --------
    >>> a = np.array([[np.nan, 4], [2, 3]])
    >>> np.argmax(a)
    0

    """
    # 调用 _replace_nan 函数替换数组 a 中的 NaN 值为 np.inf，并返回替换后的结果和掩码
    a, mask = _replace_nan(a, -np.inf)
    # 如果掩码不为空且存在元素
    if mask is not None and mask.size:
        # 沿着轴 axis 检查掩码是否全为 True
        mask = np.all(mask, axis=axis)
        # 如果有任何 True 存在，抛出 ValueError 异常
        if np.any(mask):
            raise ValueError("All-NaN slice encountered")
    # 调用 np.argmax 函数计算数组 a 在指定轴上的最大值的索引，并返回结果
    res = np.argmax(a, axis=axis, out=out, keepdims=keepdims)
    return res
    # 返回数组中忽略 NaN 值后的最大值的索引，沿指定轴。
    >>> np.nanargmax(a)
    # 返回值：1
    # 返回数组 a 中忽略 NaN 值后的最大值的索引。在这个示例中，a 是一个数组，结果是索引 1 处的元素最大。
    
    >>> np.nanargmax(a, axis=0)
    # 返回值：array([1, 0])
    # 返回数组 a 每列中忽略 NaN 值后的最大值的索引数组。在这个示例中，a 是一个二维数组，第一列最大值索引为 1，第二列最大值索引为 0。
    
    >>> np.nanargmax(a, axis=1)
    # 返回值：array([1, 1])
    # 返回数组 a 每行中忽略 NaN 值后的最大值的索引数组。在这个示例中，a 是一个二维数组，第一行最大值索引为 1，第二行最大值索引也为 1。
    
    """
    a, mask = _replace_nan(a, -np.inf)
    # 调用 _replace_nan 函数，将数组 a 中的 NaN 替换为 -∞，返回替换后的数组 a 和掩码 mask。
    
    if mask is not None and mask.size:
        # 如果掩码 mask 不为空且大小不为零：
        mask = np.all(mask, axis=axis)
        # 计算沿指定轴的掩码 mask 的所有元素是否都为 True。
        if np.any(mask):
            # 如果掩码 mask 中有任何 True 的值：
            raise ValueError("All-NaN slice encountered")
            # 抛出 ValueError 异常，指示遇到了全为 NaN 的切片。
    
    res = np.argmax(a, axis=axis, out=out, keepdims=keepdims)
    # 使用 np.argmax 函数计算数组 a 沿指定轴的最大值的索引，可以指定输出结果的存储位置及保持维度信息。
    
    return res
    # 返回计算得到的最大值索引结果。
# 定义一个分派函数 _nansum_dispatcher，接受多个参数并返回元组 (a, out)
def _nansum_dispatcher(a, axis=None, dtype=None, out=None, keepdims=None,
                       initial=None, where=None):
    return (a, out)


# 使用装饰器 array_function_dispatch 包装 nansum 函数，使其支持分派机制
@array_function_dispatch(_nansum_dispatcher)
def nansum(a, axis=None, dtype=None, out=None, keepdims=np._NoValue,
           initial=np._NoValue, where=np._NoValue):
    """
    Return the sum of array elements over a given axis treating Not a
    Numbers (NaNs) as zero.

    In NumPy versions <= 1.9.0 Nan is returned for slices that are all-NaN or
    empty. In later versions zero is returned.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose sum is desired. If `a` is not an
        array, a conversion is attempted.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the sum is computed. The default is to compute the
        sum of the flattened array.
    dtype : data-type, optional
        The type of the returned array and of the accumulator in which the
        elements are summed.  By default, the dtype of `a` is used.  An
        exception is when `a` has an integer type with less precision than
        the platform (u)intp. In that case, the default will be either
        (u)int32 or (u)int64 depending on whether the platform is 32 or 64
        bits. For inexact inputs, dtype must be inexact.

        .. versionadded:: 1.8.0
    out : ndarray, optional
        Alternate output array in which to place the result.  The default
        is ``None``. If provided, it must have the same shape as the
        expected output, but the type will be cast if necessary.  See
        :ref:`ufuncs-output-type` for more details. The casting of NaN to integer
        can yield unexpected results.

        .. versionadded:: 1.8.0
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `a`.


        If the value is anything but the default, then
        `keepdims` will be passed through to the `mean` or `sum` methods
        of sub-classes of `ndarray`.  If the sub-classes methods
        does not implement `keepdims` any exceptions will be raised.

        .. versionadded:: 1.8.0
    initial : scalar, optional
        Starting value for the sum. See `~numpy.ufunc.reduce` for details.

        .. versionadded:: 1.22.0
    where : array_like of bool, optional
        Elements to include in the sum. See `~numpy.ufunc.reduce` for details.

        .. versionadded:: 1.22.0

    Returns
    -------
    nansum : ndarray.
        A new array holding the result is returned unless `out` is
        specified, in which it is returned. The result has the same
        size as `a`, and the same shape as `a` if `axis` is not None
        or `a` is a 1-d array.

    See Also
    --------
    numpy.sum : Sum across array propagating NaNs.
    isnan : Show which elements are NaN.
    """
    """
    将数组中的 NaN 替换为指定的值后，计算数组的和，跳过 NaN 值。

    Parameters
    ----------
    a : array_like
        输入的数组。
    axis : None or int or tuple of ints, optional
        沿着哪个轴计算和，默认为 None，在整个数组上执行操作。
    dtype : dtype, optional
        返回数组的数据类型，默认为 None，表示保持输入数组的数据类型。
    out : ndarray, optional
        结果数组，用于存储计算结果。
    keepdims : bool, optional
        如果为 True，则保持轴的维度。
    initial : scalar, optional
        起始值，用于累加计算的初始值。
    where : array_like of bool, optional
        只在 where 为 True 的位置执行操作。

    Returns
    -------
    ndarray
        返回数组中非 NaN 和 +/-inf 值的和。

    Notes
    -----
    如果同时存在正负无穷大，那么结果将是 Not A Number (NaN)。

    Examples
    --------
    >>> np.nansum(1)
    1
    >>> np.nansum([1])
    1
    >>> np.nansum([1, np.nan])
    1.0
    >>> a = np.array([[1, 1], [1, np.nan]])
    >>> np.nansum(a)
    3.0
    >>> np.nansum(a, axis=0)
    array([2.,  1.])
    >>> np.nansum([1, np.nan, np.inf])
    inf
    >>> np.nansum([1, np.nan, -np.inf])
    -inf
    >>> from numpy.testing import suppress_warnings
    >>> with np.errstate(invalid="ignore"):
    ...     np.nansum([1, np.nan, np.inf, -np.inf]) # 同时存在 +/- 无穷大
    np.float64(nan)
    """
    # 将数组中的 NaN 替换为指定的值（这里替换为 0）
    a, mask = _replace_nan(a, 0)
    # 计算数组的和，跳过 NaN 值，根据给定的参数进行计算
    return np.sum(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims,
                  initial=initial, where=where)
# 创建一个分发函数 _nanprod_dispatcher，用于根据参数返回元组 (a, out)
def _nanprod_dispatcher(a, axis=None, dtype=None, out=None, keepdims=None,
                        initial=None, where=None):
    return (a, out)


# 使用 array_function_dispatch 装饰器将 nanprod 函数与 _nanprod_dispatcher 分发函数关联起来
@array_function_dispatch(_nanprod_dispatcher)
def nanprod(a, axis=None, dtype=None, out=None, keepdims=np._NoValue,
            initial=np._NoValue, where=np._NoValue):
    """
    Return the product of array elements over a given axis treating Not a
    Numbers (NaNs) as ones.

    One is returned for slices that are all-NaN or empty.

    .. versionadded:: 1.10.0

    Parameters
    ----------
    a : array_like
        Array containing numbers whose product is desired. If `a` is not an
        array, a conversion is attempted.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the product is computed. The default is to compute
        the product of the flattened array.
    dtype : data-type, optional
        The type of the returned array and of the accumulator in which the
        elements are summed.  By default, the dtype of `a` is used.  An
        exception is when `a` has an integer type with less precision than
        the platform (u)intp. In that case, the default will be either
        (u)int32 or (u)int64 depending on whether the platform is 32 or 64
        bits. For inexact inputs, dtype must be inexact.
    out : ndarray, optional
        Alternate output array in which to place the result.  The default
        is ``None``. If provided, it must have the same shape as the
        expected output, but the type will be cast if necessary. See
        :ref:`ufuncs-output-type` for more details. The casting of NaN to integer
        can yield unexpected results.
    keepdims : bool, optional
        If True, the axes which are reduced are left in the result as
        dimensions with size one. With this option, the result will
        broadcast correctly against the original `arr`.
    initial : scalar, optional
        The starting value for this product. See `~numpy.ufunc.reduce`
        for details.

        .. versionadded:: 1.22.0
    where : array_like of bool, optional
        Elements to include in the product. See `~numpy.ufunc.reduce`
        for details.

        .. versionadded:: 1.22.0

    Returns
    -------
    nanprod : ndarray
        A new array holding the result is returned unless `out` is
        specified, in which case it is returned.

    See Also
    --------
    numpy.prod : Product across array propagating NaNs.
    isnan : Show which elements are NaN.

    Examples
    --------
    >>> np.nanprod(1)
    1
    >>> np.nanprod([1])
    1
    >>> np.nanprod([1, np.nan])
    1.0
    >>> a = np.array([[1, 2], [3, np.nan]])
    >>> np.nanprod(a)
    6.0
    >>> np.nanprod(a, axis=0)
    array([3., 2.])

    """
    # 调用 _replace_nan 函数将数组 a 中的 NaN 替换为 1，并返回替换后的数组 a 和 NaN 掩码 mask
    a, mask = _replace_nan(a, 1)
    # 调用 np.prod 计算数组 a 沿指定轴的元素的乘积，返回结果
    return np.prod(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims,
                   initial=initial, where=where)


# 创建一个分发函数 _nancumsum_dispatcher，用于根据参数返回适当的元组 (a, axis, dtype, out)
def _nancumsum_dispatcher(a, axis=None, dtype=None, out=None):
    # 返回一个元组，包含变量 a 和 out 的值
    return (a, out)
# 用于处理 nancumsum 函数分派的装饰器，使其可以根据不同输入类型调用相应的处理函数
@array_function_dispatch(_nancumsum_dispatcher)
def nancumsum(a, axis=None, dtype=None, out=None):
    """
    返回沿给定轴计算的数组元素的累积和，将非数值 (NaN) 视为零处理。
    当遇到 NaN 时，累积和不会改变，并且在开头的 NaN 被替换为零。

    对于完全是 NaN 或空的切片，返回零。

    .. versionadded:: 1.12.0

    Parameters
    ----------
    a : array_like
        输入数组。
    axis : int, optional
        沿其计算累积和的轴。默认为 None，表示在扁平化的数组上计算累积和。
    dtype : dtype, optional
        返回数组和累加器的类型，用于对元素求和。如果未指定 `dtype`，默认为 `a` 的 dtype，除非 `a` 的整数 dtype 的精度低于默认平台整数的精度。在这种情况下，将使用默认平台整数。
    out : ndarray, optional
        替代输出数组，用于放置结果。它必须具有与预期输出相同的形状和缓冲区长度，但如果必要会进行类型转换。详见 :ref:`ufuncs-output-type` 获取更多详情。

    Returns
    -------
    nancumsum : ndarray
        返回一个新的数组，保存结果，除非指定了 `out`，此时返回 `out`。结果与 `a` 具有相同的大小，并且如果 `axis` 不是 None 或 `a` 是 1 维数组，则具有与 `a` 相同的形状。

    See Also
    --------
    numpy.cumsum : 沿数组累积和并传播 NaN。
    isnan : 显示哪些元素是 NaN。

    Examples
    --------
    >>> np.nancumsum(1)
    array([1])
    >>> np.nancumsum([1])
    array([1])
    >>> np.nancumsum([1, np.nan])
    array([1.,  1.])
    >>> a = np.array([[1, 2], [3, np.nan]])
    >>> np.nancumsum(a)
    array([1.,  3.,  6.,  6.])
    >>> np.nancumsum(a, axis=0)
    array([[1.,  2.],
           [4.,  2.]])
    >>> np.nancumsum(a, axis=1)
    array([[1.,  3.],
           [3.,  3.]])
    """
    # 调用 _replace_nan 函数，将数组中的 NaN 替换为零，并返回替换后的数组及掩码
    a, mask = _replace_nan(a, 0)
    # 调用 numpy 库中的 cumsum 函数，计算累积和，传入相应的参数并返回结果
    return np.cumsum(a, axis=axis, dtype=dtype, out=out)


# 用于处理 nancumprod 函数分派的装饰器，使其可以根据不同输入类型调用相应的处理函数
def _nancumprod_dispatcher(a, axis=None, dtype=None, out=None):
    return (a, out)


@array_function_dispatch(_nancumprod_dispatcher)
def nancumprod(a, axis=None, dtype=None, out=None):
    """
    返回沿给定轴计算的数组元素的累积乘积，将非数值 (NaN) 视为一处理。
    当遇到 NaN 时，累积乘积不会改变，并且在开头的 NaN 被替换为一。

    对于完全是 NaN 或空的切片，返回一。

    .. versionadded:: 1.12.0

    Parameters
    ----------
    a : array_like
        输入数组。
    axis : int, optional
        沿其计算累积乘积的轴。默认情况下对输入进行扁平化处理。

    """
    # 函数体尚未完全注释，需要进一步完成
    pass


这样注释后的代码段将详细解释每一行代码的作用和功能，符合注释的要求。
    # 参数 dtype：返回数组的数据类型，以及累积乘积的累加器的数据类型。
    # 如果未指定 dtype，则默认为 a 的数据类型，除非 a 的整数类型精度小于默认平台整数类型的精度。
    # 在这种情况下，将使用默认平台整数类型。
    dtype : dtype, optional
    
    # 参数 out：替代的输出数组，用于存放结果。它必须具有与预期输出相同的形状和缓冲区长度，
    # 但如果需要会进行结果值的类型转换。
    out : ndarray, optional
    
    # 返回结果
    # 返回一个新的数组，其中包含结果，除非指定了 out 参数，在这种情况下将返回 out。
    Returns
    -------
    nancumprod : ndarray
    
    # 参见
    # numpy.cumprod : 在数组上执行累积乘积，处理 NaN 值。
    # isnan : 显示哪些元素是 NaN。
    See Also
    --------
    numpy.cumprod : Cumulative product across array propagating NaNs.
    isnan : Show which elements are NaN.
    
    # 示例
    # >>> np.nancumprod(1)
    # array([1])
    # >>> np.nancumprod([1])
    # array([1])
    # >>> np.nancumprod([1, np.nan])
    # array([1.,  1.])
    # >>> a = np.array([[1, 2], [3, np.nan]])
    # >>> np.nancumprod(a)
    # array([1.,  2.,  6.,  6.])
    # >>> np.nancumprod(a, axis=0)
    # array([[1.,  2.],
    #        [3.,  2.]])
    # >>> np.nancumprod(a, axis=1)
    # array([[1.,  2.],
    #        [3.,  3.]])
    Examples
    --------
    
    # 替换 NaN 值为指定的值（这里是 1），并返回替换后的数组以及掩码数组。
    a, mask = _replace_nan(a, 1)
    
    # 调用 numpy 库的累积乘积函数 cumprod，对数组 a 按指定轴进行操作，
    # 可选地指定数据类型 dtype 和输出数组 out。
    return np.cumprod(a, axis=axis, dtype=dtype, out=out)
# 定义一个调度函数 `_nanmean_dispatcher`，用于分派参数 `a` 和 `out`
def _nanmean_dispatcher(a, axis=None, dtype=None, out=None, keepdims=None,
                        *, where=None):
    # 返回参数 `a` 和 `out` 的元组
    return (a, out)


# 使用 `array_function_dispatch` 装饰器来声明 `nanmean` 函数
@array_function_dispatch(_nanmean_dispatcher)
def nanmean(a, axis=None, dtype=None, out=None, keepdims=np._NoValue,
            *, where=np._NoValue):
    """
    Compute the arithmetic mean along the specified axis, ignoring NaNs.

    Returns the average of the array elements.  The average is taken over
    the flattened array by default, otherwise over the specified axis.
    `float64` intermediate and return values are used for integer inputs.

    For all-NaN slices, NaN is returned and a `RuntimeWarning` is raised.

    .. versionadded:: 1.8.0

    Parameters
    ----------
    a : array_like
        Array containing numbers whose mean is desired. If `a` is not an
        array, a conversion is attempted.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the means are computed. The default is to compute
        the mean of the flattened array.
    dtype : data-type, optional
        Type to use in computing the mean.  For integer inputs, the default
        is `float64`; for inexact inputs, it is the same as the input
        dtype.
    out : ndarray, optional
        Alternate output array in which to place the result.  The default
        is ``None``; if provided, it must have the same shape as the
        expected output, but the type will be cast if necessary.
        See :ref:`ufuncs-output-type` for more details.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `a`.

        If the value is anything but the default, then
        `keepdims` will be passed through to the `mean` or `sum` methods
        of sub-classes of `ndarray`.  If the sub-classes methods
        does not implement `keepdims` any exceptions will be raised.
    where : array_like of bool, optional
        Elements to include in the mean. See `~numpy.ufunc.reduce` for details.

        .. versionadded:: 1.22.0

    Returns
    -------
    m : ndarray, see dtype parameter above
        If `out=None`, returns a new array containing the mean values,
        otherwise a reference to the output array is returned. Nan is
        returned for slices that contain only NaNs.

    See Also
    --------
    average : Weighted average
    mean : Arithmetic mean taken while not ignoring NaNs
    var, nanvar

    Notes
    -----
    The arithmetic mean is the sum of the non-NaN elements along the axis
    divided by the number of non-NaN elements.

    Note that for floating-point input, the mean is computed using the same
    precision the input has.  Depending on the input data, this can cause
    the results to be inaccurate, especially for `float32`.  Specifying a
    """
    arr, mask = _replace_nan(a, 0)
    # 调用 _replace_nan 函数，用 0 替换数组 a 中的 NaN 值，返回替换后的数组 arr 和掩码 mask

    if mask is None:
        # 如果掩码为空，则表示数组中没有 NaN 值
        return np.mean(arr, axis=axis, dtype=dtype, out=out, keepdims=keepdims,
                       where=where)
        # 返回数组 arr 的均值，根据指定的轴和数据类型，可选地输出到 out 数组，保持维度 keepdims，条件 where 生效

    if dtype is not None:
        dtype = np.dtype(dtype)
        # 如果指定了数据类型 dtype，则将其转换为 numpy 的 dtype 对象

    if dtype is not None and not issubclass(dtype.type, np.inexact):
        # 如果指定了数据类型，并且该类型不是浮点数类型，则抛出类型错误
        raise TypeError("If a is inexact, then dtype must be inexact")

    if out is not None and not issubclass(out.dtype.type, np.inexact):
        # 如果指定了输出数组 out，并且其类型不是浮点数类型，则抛出类型错误
        raise TypeError("If a is inexact, then out must be inexact")

    cnt = np.sum(~mask, axis=axis, dtype=np.intp, keepdims=keepdims,
                 where=where)
    # 统计掩码中非 NaN 值的数量，按指定轴求和，返回整数类型的结果 cnt

    tot = np.sum(arr, axis=axis, dtype=dtype, out=out, keepdims=keepdims,
                 where=where)
    # 计算数组 arr 沿指定轴的总和，根据指定的数据类型、输出数组和条件 where 进行计算，返回总和 tot

    avg = _divide_by_count(tot, cnt, out=out)
    # 调用 _divide_by_count 函数，计算总和 tot 除以非 NaN 值的数量 cnt 的平均值，结果保存在 avg 中

    isbad = (cnt == 0)
    # 创建布尔数组 isbad，标记在计算平均值时出现的空切片（即 cnt 为零的情况）

    if isbad.any():
        # 如果存在空切片的情况
        warnings.warn("Mean of empty slice", RuntimeWarning, stacklevel=2)
        # 发出运行时警告，指示出现了空切片的平均值
        # NaN 是唯一可能的无效值，因此不需要进一步处理坏结果。
    
    return avg
    # 返回计算得到的平均值 avg
# 私有函数，用于一维数组。计算忽略 NaN 值的中位数。
# 查看 nanmedian 函数的参数用法
def _nanmedian1d(arr1d, overwrite_input=False):
    # 调用 _remove_nan_1d 函数，移除 arr1d 中的 NaN 值，返回处理后的数组和覆盖输入的标志
    arr1d_parsed, _, overwrite_input = _remove_nan_1d(
        arr1d, overwrite_input=overwrite_input,
    )

    # 如果 arr1d_parsed 为空数组
    if arr1d_parsed.size == 0:
        # 确保返回一个类似 NaN 的标量，类型和单位与输入的 `timedelta64` 和 `complexfloating` 相匹配
        return arr1d[-1]

    # 返回 arr1d_parsed 的中位数，可以选择覆盖输入的标志
    return np.median(arr1d_parsed, overwrite_input=overwrite_input)


# 私有函数，不支持扩展轴或保持维度。
# 这些方法使用 _ureduce 扩展到该函数中
# 查看 nanmedian 函数的参数用法
def _nanmedian(a, axis=None, out=None, overwrite_input=False):
    # 如果 axis 为 None 或数组是一维的
    if axis is None or a.ndim == 1:
        # 将数组展平为一维
        part = a.ravel()
        # 如果没有提供输出数组
        if out is None:
            # 返回 _nanmedian1d 处理后的结果
            return _nanmedian1d(part, overwrite_input)
        else:
            # 将 _nanmedian1d 处理后的结果复制到输出数组中
            out[...] = _nanmedian1d(part, overwrite_input)
            return out
    else:
        # 对于较小的中位数，使用排序 + 索引，仍然比 apply_along_axis 更快
        # 在包含少量 NaN 的 shuffled (50, 50, x) 数据上进行基准测试
        if a.shape[axis] < 600:
            # 返回 _nanmedian_small 处理后的结果
            return _nanmedian_small(a, axis, out, overwrite_input)
        # 沿指定轴应用 _nanmedian1d 函数，返回结果
        result = np.apply_along_axis(_nanmedian1d, axis, a, overwrite_input)
        # 如果提供了输出数组，则将结果复制到输出数组中
        if out is not None:
            out[...] = result
        return result


# 私有函数，用于较小的中位数，排序 + 索引中位数，
# 对于多个维度的小中位数由于 apply_along_axis 的高开销更快
# 查看 nanmedian 函数的参数用法
def _nanmedian_small(a, axis=None, out=None, overwrite_input=False):
    # 创建一个掩码数组，将 a 中的 NaN 值掩盖起来
    a = np.ma.masked_array(a, np.isnan(a))
    # 计算掩码数组的中位数，可以指定轴和是否覆盖输入
    m = np.ma.median(a, axis=axis, overwrite_input=overwrite_input)
    # 如果遇到全为 NaN 的切片，则发出警告
    for i in range(np.count_nonzero(m.mask.ravel())):
        warnings.warn("All-NaN slice encountered", RuntimeWarning,
                      stacklevel=5)

    # 如果 m 的数据类型是时间间隔或复数浮点数，填充值为 NaT 或 NaN
    fill_value = np.timedelta64("NaT") if m.dtype.kind == "m" else np.nan
    # 如果提供了输出数组，则将填充后的 m 复制到输出数组中
    if out is not None:
        out[...] = m.filled(fill_value)
        return out
    # 否则返回填充后的 m
    return m.filled(fill_value)


# 分发器函数，将参数传递给 _nanmedian 函数
def _nanmedian_dispatcher(
        a, axis=None, out=None, overwrite_input=None, keepdims=None):
    return (a, out)


# 使用 array_function_dispatch 装饰器将 _nanmedian_dispatcher 函数绑定到 nanmedian 函数上
@array_function_dispatch(_nanmedian_dispatcher)
def nanmedian(a, axis=None, out=None, overwrite_input=False, keepdims=np._NoValue):
    """
    计算沿指定轴的中位数，忽略 NaN 值。

    返回数组元素的中位数。

    .. versionadded:: 1.9.0

    参数
    ----------
    a : array_like
        输入数组或可以转换为数组的对象。
    axis : {int, sequence of int, None}, optional
        计算中位数的轴或轴组。默认是沿数组的展平版本计算中位数。
        从版本 1.9.0 开始支持一系列轴。

    """
    # 将输入参数 `a` 转换为一个 NumPy 数组，不论它是什么类型的数组
    a = np.asanyarray(a)
    
    # 如果数组 `a` 的大小为 0，即为空数组，则返回空数组的均值
    # 这里使用了 `np.nanmean` 函数来处理空数组的情况，保留指定的轴和维度
    if a.size == 0:
        return np.nanmean(a, axis, out=out, keepdims=keepdims)
    
    # 调用底层的 `_ureduce` 函数，使用 `_nanmedian` 函数进行中位数计算
    # 可以指定是否保持降维后的维度，以及输出结果到指定的数组 `out`
    return fnb._ureduce(a, func=_nanmedian, keepdims=keepdims,
                        axis=axis, out=out,
                        overwrite_input=overwrite_input)
# 定义一个分派函数，用于_nanpercentile_dispatcher，接收参数a, q, axis, out, overwrite_input, method, keepdims, *, weights, interpolation，并返回前三个参数和weights
def _nanpercentile_dispatcher(
        a, q, axis=None, out=None, overwrite_input=None,
        method=None, keepdims=None, *, weights=None, interpolation=None):
    return (a, q, out, weights)


# 使用array_function_dispatch装饰器，将_nanpercentile_dispatcher作为分派函数，用于nanpercentile函数
@array_function_dispatch(_nanpercentile_dispatcher)
# 定义nanpercentile函数，计算沿指定轴的数据的q分位数，忽略NaN值
def nanpercentile(
        a,
        q,
        axis=None,
        out=None,
        overwrite_input=False,
        method="linear",
        keepdims=np._NoValue,
        *,
        weights=None,
        interpolation=None,
):
    """
    Compute the qth percentile of the data along the specified axis,
    while ignoring nan values.

    Returns the qth percentile(s) of the array elements.

    .. versionadded:: 1.9.0

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array, containing
        nan values to be ignored.
    q : array_like of float
        Percentile or sequence of percentiles to compute, which must be
        between 0 and 100 inclusive.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the percentiles are computed. The default
        is to compute the percentile(s) along a flattened version of the
        array.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape and buffer length as the expected output, but the
        type (of the output) will be cast if necessary.
    overwrite_input : bool, optional
        If True, then allow the input array `a` to be modified by
        intermediate calculations, to save memory. In this case, the
        contents of the input `a` after this function completes is
        undefined.
    method : str, optional
        This parameter specifies the method to use for estimating the
        percentile.  There are many different methods, some unique to NumPy.
        See the notes for explanation.  The options sorted by their R type
        as summarized in the H&F paper [1]_ are:

        1. 'inverted_cdf'
        2. 'averaged_inverted_cdf'
        3. 'closest_observation'
        4. 'interpolated_inverted_cdf'
        5. 'hazen'
        6. 'weibull'
        7. 'linear'  (default)
        8. 'median_unbiased'
        9. 'normal_unbiased'

        The first three methods are discontinuous.  NumPy further defines the
        following discontinuous variations of the default 'linear' (7.) option:

        * 'lower'
        * 'higher',
        * 'midpoint'
        * 'nearest'

        .. versionchanged:: 1.22.0
            This argument was previously called "interpolation" and only
            offered the "linear" default and last four options.
    """
    keepdims : bool, optional
        # 如果设置为True，则对被减少的轴留在结果中作为大小为一的维度。此选项可以使结果正确地与原数组“a”广播。

        # 如果这个值不是默认值，它将被传递（在空数组的特殊情况下）到底层数组的“mean”函数中。如果数组是一个子类，并且“mean”没有kwarg“keepdims”，则会引发RuntimeError。

    weights : array_like, optional
        # 与“a”中的值相关联的权重数组。每个“a”中的值根据其关联的权重对百分位数做出贡献。权重数组可以是1-D（其长度必须是沿着给定轴的“a”的大小）或与“a”的形状相同。如果“weights=None”，则假定“a”中的所有数据的权重都等于一。只有“method="inverted_cdf"”支持权重。

        .. versionadded:: 2.0.0

    interpolation : str, optional
        # 方法关键字参数的弃用名称。

        .. deprecated:: 1.22.0

    Returns
    -------
    percentile : scalar or ndarray
        # 如果“q”是单个百分位数且“axis=None”，则结果是一个标量。如果给定多个百分位数，结果的第一个轴对应于百分位数。其他轴是在“a”减少后保留的轴。如果输入包含小于“float64”的整数或浮点数，则输出数据类型是“float64”。否则，输出数据类型与输入的类型相同。如果指定了“out”，则返回该数组。

    See Also
    --------
    nanmean
    nanmedian : 等同于“nanpercentile(..., 50)” 
    percentile, median, mean
    nanquantile : 等同于nanpercentile，只是q的范围是[0, 1]。

    Notes
    -----
    # 用百分比“q”的“numpy.nanpercentile”的行为是使用参数“q/100”的“numpy.quantile”的行为（忽略nan值）。更多信息，请参见“numpy.quantile”。

    Examples
    --------
    # Examples的一系列用法示例，解释了如何使用numpy.nanpercentile

    References
    ----------
    """
    Calculate the weighted quantiles of an array with optional weights.

    .. [1] R. J. Hyndman and Y. Fan,
       "Sample quantiles in statistical packages,"
       The American Statistician, 50(4), pp. 361-365, 1996

    """
    # 如果指定了插值方法，检查并调整为正确的方法
    if interpolation is not None:
        method = fnb._check_interpolation_as_method(
            method, interpolation, "nanpercentile")

    # 将输入数组转换为最通用的数组表示形式
    a = np.asanyarray(a)
    # 如果数组包含复数，则抛出类型错误
    if a.dtype.kind == "c":
        raise TypeError("a must be an array of real numbers")

    # 将百分位数转换为小数形式
    q = np.true_divide(q, a.dtype.type(100) if a.dtype.kind == "f" else 100)
    # 恢复由ufunc执行的任何衰减（参见gh-13105）
    q = np.asanyarray(q)
    # 检查百分位数的有效性，必须在[0, 100]范围内
    if not fnb._quantile_is_valid(q):
        raise ValueError("Percentiles must be in the range [0, 100]")

    # 如果指定了权重
    if weights is not None:
        # 如果方法不是'inverted_cdf'，则不支持权重，抛出错误
        if method != "inverted_cdf":
            msg = ("Only method 'inverted_cdf' supports weights. "
                   f"Got: {method}.")
            raise ValueError(msg)
        # 如果指定了轴参数，标准化轴参数
        if axis is not None:
            axis = _nx.normalize_axis_tuple(axis, a.ndim, argname="axis")
        # 检查权重的有效性
        weights = _weights_are_valid(weights=weights, a=a, axis=axis)
        # 权重必须是非负数
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative.")

    # 调用内部函数计算未检查的NaN百分位数
    return _nanquantile_unchecked(
        a, q, axis, out, overwrite_input, method, keepdims, weights)
# 定义一个调度器函数，用于 nanquantile 函数的分派
def _nanquantile_dispatcher(a, q, axis=None, out=None, overwrite_input=None,
                            method=None, keepdims=None, *, weights=None,
                            interpolation=None):
    # 返回传入的参数元组，用于 nanquantile 函数的调用
    return (a, q, out, weights)


# 通过 array_function_dispatch 装饰器声明 nanquantile 函数
@array_function_dispatch(_nanquantile_dispatcher)
def nanquantile(
        a,
        q,
        axis=None,
        out=None,
        overwrite_input=False,
        method="linear",
        keepdims=np._NoValue,
        *,
        weights=None,
        interpolation=None,
):
    """
    Compute the qth quantile of the data along the specified axis,
    while ignoring nan values.
    Returns the qth quantile(s) of the array elements.

    .. versionadded:: 1.15.0

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array, containing
        nan values to be ignored
    q : array_like of float
        Probability or sequence of probabilities for the quantiles to compute.
        Values must be between 0 and 1 inclusive.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the quantiles are computed. The
        default is to compute the quantile(s) along a flattened
        version of the array.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output,
        but the type (of the output) will be cast if necessary.
    overwrite_input : bool, optional
        If True, then allow the input array `a` to be modified by intermediate
        calculations, to save memory. In this case, the contents of the input
        `a` after this function completes is undefined.
    method : str, optional
        This parameter specifies the method to use for estimating the
        quantile.  There are many different methods, some unique to NumPy.
        See the notes for explanation.  The options sorted by their R type
        as summarized in the H&F paper [1]_ are:

        1. 'inverted_cdf'
        2. 'averaged_inverted_cdf'
        3. 'closest_observation'
        4. 'interpolated_inverted_cdf'
        5. 'hazen'
        6. 'weibull'
        7. 'linear'  (default)
        8. 'median_unbiased'
        9. 'normal_unbiased'

        The first three methods are discontinuous.  NumPy further defines the
        following discontinuous variations of the default 'linear' (7.) option:

        * 'lower'
        * 'higher',
        * 'midpoint'
        * 'nearest'

        .. versionchanged:: 1.22.0
            This argument was previously called "interpolation" and only
            offered the "linear" default and last four options.

    """
    # nanquantile 函数的主体部分，在指定轴上计算忽略 NaN 值的 q 分位数
    # 返回计算得到的分位数或分位数数组
    pass
    keepdims : bool, optional
        如果设置为 True，则减少的轴会作为大小为一的维度保留在结果中。使用此选项，结果将正确地与原始数组 `a` 进行广播。
        如果设为非默认值，在特殊情况下（空数组），将传递给底层数组的 `mean` 函数。如果数组是子类且 `mean` 函数没有 `keepdims` 关键字参数，将引发 RuntimeError。

    weights : array_like, optional
        与数组 `a` 中的值相关联的权重数组。`a` 中的每个值根据其相关的权重贡献于分位数的计算。权重数组可以是 1-D 数组（此时其长度必须与沿给定轴的 `a` 的大小相同），或者与 `a` 具有相同形状。如果 `weights=None`，则假定 `a` 中的所有数据权重均为1。
        仅 `method="inverted_cdf"` 支持权重。

        .. versionadded:: 2.0.0

    interpolation : str, optional
        方法关键字参数的过时名称。

        .. deprecated:: 1.22.0

    Returns
    -------
    quantile : scalar or ndarray
        如果 `q` 是单个概率且 `axis=None`，则结果是标量。如果给定多个概率水平，结果的第一轴对应于分位数。其他轴是减少 `a` 后保留的轴。如果输入包含小于 ``float64`` 的整数或浮点数，则输出数据类型为 ``float64``。否则，输出数据类型与输入相同。如果指定了 `out`，则返回该数组。

    See Also
    --------
    quantile
    nanmean, nanmedian
    nanmedian : 相当于 ``nanquantile(..., 0.5)``
    nanpercentile : 与 nanquantile 相同，但 q 在范围 [0, 100] 内。

    Notes
    -----
    `numpy.nanquantile` 的行为与 `numpy.quantile` 相同（忽略 NaN 值）。
    欲了解更多信息，请参阅 `numpy.quantile`。

    Examples
    --------
    >>> a = np.array([[10., 7., 4.], [3., 2., 1.]])
    >>> a[0][1] = np.nan
    >>> a
    array([[10.,  nan,   4.],
          [ 3.,   2.,   1.]])
    >>> np.quantile(a, 0.5)
    np.float64(nan)
    >>> np.nanquantile(a, 0.5)
    3.0
    >>> np.nanquantile(a, 0.5, axis=0)
    array([6.5, 2. , 2.5])
    >>> np.nanquantile(a, 0.5, axis=1, keepdims=True)
    array([[7.],
           [2.]])
    >>> m = np.nanquantile(a, 0.5, axis=0)
    >>> out = np.zeros_like(m)
    >>> np.nanquantile(a, 0.5, axis=0, out=out)
    array([6.5, 2. , 2.5])
    >>> m
    array([6.5,  2. ,  2.5])
    >>> b = a.copy()
    >>> np.nanquantile(b, 0.5, axis=1, overwrite_input=True)
    array([7., 2.])
    >>> assert not np.all(a==b)

    References
    ----------
    """
    转换引用文献信息，指向文献中描述的相关内容

    """

    # 如果指定了插值方法，则检查并设置为对应的插值方法
    if interpolation is not None:
        method = fnb._check_interpolation_as_method(
            method, interpolation, "nanquantile")

    # 将输入数组转换为任意数组，确保可以处理各种类型的输入
    a = np.asanyarray(a)

    # 如果数组的数据类型是复数，则引发类型错误
    if a.dtype.kind == "c":
        raise TypeError("a must be an array of real numbers")

    # 如果 q 是 Python 的整数或浮点数，并且数组 a 的数据类型是浮点数，则使用数组的数据类型
    if isinstance(q, (int, float)) and a.dtype.kind == "f":
        q = np.asanyarray(q, dtype=a.dtype)
    else:
        q = np.asanyarray(q)

    # 检查分位数 q 是否有效，必须在 [0, 1] 的范围内
    if not fnb._quantile_is_valid(q):
        raise ValueError("Quantiles must be in the range [0, 1]")

    # 如果指定了 weights，则进行相应的检查和处理
    if weights is not None:
        # 如果方法不是 "inverted_cdf"，则引发错误
        if method != "inverted_cdf":
            msg = ("Only method 'inverted_cdf' supports weights. "
                   f"Got: {method}.")
            raise ValueError(msg)

        # 如果指定了轴，则规范化轴元组
        if axis is not None:
            axis = _nx.normalize_axis_tuple(axis, a.ndim, argname="axis")

        # 检查权重的有效性，并确保非负性
        weights = _weights_are_valid(weights=weights, a=a, axis=axis)
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative.")

    # 调用未经检查的 _nanquantile_unchecked 函数，计算给定数据的分位数
    return _nanquantile_unchecked(
        a, q, axis, out, overwrite_input, method, keepdims, weights)
def _nanquantile_unchecked(
        a,
        q,
        axis=None,
        out=None,
        overwrite_input=False,
        method="linear",
        keepdims=np._NoValue,
        weights=None,
):
    """Assumes that q is in [0, 1], and is an ndarray"""
    # apply_along_axis in _nanpercentile doesn't handle empty arrays well,
    # so deal them upfront
    # 如果数组 a 是空的，则返回沿着指定轴的 NaN 均值
    if a.size == 0:
        return np.nanmean(a, axis, out=out, keepdims=keepdims)
    # 否则调用 _ureduce 函数处理 a 数组，返回计算的结果
    return fnb._ureduce(a,
                        func=_nanquantile_ureduce_func,
                        q=q,
                        weights=weights,
                        keepdims=keepdims,
                        axis=axis,
                        out=out,
                        overwrite_input=overwrite_input,
                        method=method)


def _nanquantile_ureduce_func(
        a: np.array,
        q: np.array,
        weights: np.array,
        axis: int = None,
        out=None,
        overwrite_input: bool = False,
        method="linear",
):
    """
    Private function that doesn't support extended axis or keepdims.
    These methods are extended to this function using _ureduce
    See nanpercentile for parameter usage
    """
    # 如果 axis 为 None 或者数组 a 的维度为 1，则将数组展平处理
    if axis is None or a.ndim == 1:
        part = a.ravel()
        wgt = None if weights is None else weights.ravel()
        # 调用 _nanquantile_1d 函数计算一维情况下的分位数
        result = _nanquantile_1d(part, q, overwrite_input, method, weights=wgt)
    else:
        # 否则，尝试在这里填充 `out`
        if weights is None:
            # 对数组 a 沿着指定轴应用 _nanquantile_1d 函数
            result = np.apply_along_axis(_nanquantile_1d, axis, a, q,
                                         overwrite_input, method, weights)
            # apply_along_axis 填充了折叠轴的结果。
            # 将这些轴移到开头以匹配百分位数的约定。
            if q.ndim != 0:
                from_ax = [axis + i for i in range(q.ndim)]
                result = np.moveaxis(result, from_ax, list(range(q.ndim)))
        else:
            # 我们需要在两个数组 a 和 weights 上应用 along axis
            # 为简单起见，将操作轴移到末尾：
            a = np.moveaxis(a, axis, -1)
            if weights is not None:
                weights = np.moveaxis(weights, axis, -1)
            if out is not None:
                result = out
            else:
                # weights 限制在 `inverted_cdf` 中，因此结果的数据类型与 `a` 相同:
                result = np.empty_like(a, shape=q.shape + a.shape[:-1])

            for ii in np.ndindex(a.shape[:-1]):
                # 对每个索引 ii，调用 _nanquantile_1d 函数计算分位数
                result[(...,) + ii] = _nanquantile_1d(
                        a[ii], q, weights=weights[ii],
                        overwrite_input=overwrite_input, method=method,
                )
            # 这条路径已经处理了 `out` ...
            return result

    # 如果指定了 `out`，则将结果赋给 `out`
    if out is not None:
        out[...] = result
    return result


def _nanquantile_1d(
    arr1d,  # 第一个参数：一维数组，是要插值的数据点的坐标
    q,  # 第二个参数：标量或一维数组，是要计算插值的位置或位置序列
    overwrite_input=False,  # 是否覆盖输入数组，设置为 False 表示不覆盖，默认为 False
    method="linear",  # 插值方法，默认为线性插值
    weights=None,  # 可选参数：用于加权的数组，可以控制每个点的插值权重，默认为 None
):
    """
    Private function for rank 1 arrays. Compute quantile ignoring NaNs.
    See nanpercentile for parameter usage
    """
    # TODO: What to do when arr1d = [1, np.nan] and weights = [0, 1]?
    # 调用 _remove_nan_1d 函数处理 arr1d 和 weights，移除 NaN 值并返回处理后的数组和标志 overwrite_input
    arr1d, weights, overwrite_input = _remove_nan_1d(arr1d,
        second_arr1d=weights, overwrite_input=overwrite_input)
    # 如果 arr1d 的大小为 0，则返回一个与 q 形状相同的全 NaN 数组的标量值
    if arr1d.size == 0:
        return np.full(q.shape, np.nan, dtype=arr1d.dtype)[()]

    # 调用 fnb._quantile_unchecked 函数计算未检查的分位数
    return fnb._quantile_unchecked(
        arr1d,
        q,
        overwrite_input=overwrite_input,
        method=method,
        weights=weights,
    )


def _nanvar_dispatcher(a, axis=None, dtype=None, out=None, ddof=None,
                       keepdims=None, *, where=None, mean=None,
                       correction=None):
    # 返回包含参数 a 和 out 的元组
    return (a, out)


@array_function_dispatch(_nanvar_dispatcher)
def nanvar(a, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue,
           *, where=np._NoValue, mean=np._NoValue, correction=np._NoValue):
    """
    Compute the variance along the specified axis, while ignoring NaNs.

    Returns the variance of the array elements, a measure of the spread of
    a distribution.  The variance is computed for the flattened array by
    default, otherwise over the specified axis.

    For all-NaN slices or slices with zero degrees of freedom, NaN is
    returned and a `RuntimeWarning` is raised.

    .. versionadded:: 1.8.0

    Parameters
    ----------
    a : array_like
        Array containing numbers whose variance is desired.  If `a` is not an
        array, a conversion is attempted.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the variance is computed.  The default is to compute
        the variance of the flattened array.
    dtype : data-type, optional
        Type to use in computing the variance.  For arrays of integer type
        the default is `float64`; for arrays of float types it is the same as
        the array type.
    out : ndarray, optional
        Alternate output array in which to place the result.  It must have
        the same shape as the expected output, but the type is cast if
        necessary.
    ddof : {int, float}, optional
        "Delta Degrees of Freedom": the divisor used in the calculation is
        ``N - ddof``, where ``N`` represents the number of non-NaN
        elements. By default `ddof` is zero.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `a`.
    where : array_like of bool, optional
        Elements to include in the variance. See `~numpy.ufunc.reduce` for
        details.

        .. versionadded:: 1.22.0
    """
    mean : array_like, optional
        # 可选参数，用于提供均值以避免重新计算。均值应当具有 `keepdims=True` 计算后的形状。
        # 在调用此函数时，用于计算均值的轴应与此 var 函数使用的轴相同。

        .. versionadded:: 1.26.0
        # 引入版本：1.26.0

    correction : {int, float}, optional
        # 可选参数，与 Array API 兼容的 `ddof` 参数名称。这两者只能同时提供其中一个。

        .. versionadded:: 2.0.0
        # 引入版本：2.0.0

    Returns
    -------
    variance : ndarray, see dtype parameter above
        # 如果 `out` 为 None，则返回一个包含方差的新数组；否则返回对输出数组的引用。
        # 如果 `ddof` 大于等于切片中非 NaN 元素的数量或切片仅包含 NaN，则该切片的结果为 NaN。

    See Also
    --------
    std : 标准差
    mean : 平均值
    var : 不忽略 NaN 的方差
    nanstd, nanmean
    :ref:`ufuncs-output-type`
        # 参见：ufunc 的输出类型

    Notes
    -----
    # 方差是平均平方偏差，即 `var = mean(abs(x - x.mean())**2)`。

    # 均值通常计算为 `x.sum() / N`，其中 `N = len(x)`。
    # 然而，如果指定了 `ddof`，则使用除数 `N - ddof`。
    # 在标准统计实践中，`ddof=1` 提供了一个对无限总体方差的无偏估计。
    # `ddof=0` 提供了一个正态分布变量方差的最大似然估计。

    # 注意，对于复数，先取绝对值再平方，因此结果总是实数且非负。

    # 对于浮点输入，方差使用与输入相同的精度计算。根据输入数据，这可能导致结果不准确，特别是对于 `float32`（参见下面的示例）。
    # 使用 `dtype` 关键字指定更高精度的累加器可以缓解此问题。

    # 为了使此函数在 ndarray 的子类上正常工作，它们必须定义带有 `keepdims` 关键字的 `sum`。

    Examples
    --------
    >>> a = np.array([[1, np.nan], [3, 4]])
    >>> np.nanvar(a)
    1.5555555555555554
    >>> np.nanvar(a, axis=0)
    array([1.,  0.])
    >>> np.nanvar(a, axis=1)
    array([0.,  0.25])  # 可能会有所不同
    ```
    # 如果给定了修正值 correction，则检查是否同时给定了 ddof，若是则引发 ValueError 异常
    if correction != np._NoValue:
        if ddof != 0:
            raise ValueError(
                "ddof and correction can't be provided simultaneously."
            )
        else:
            ddof = correction

    # 计算均值
    if type(arr) is np.matrix:
        _keepdims = np._NoValue  # 对于 np.matrix 类型，不保持维度信息
    else:
        _keepdims = True  # 对于其他类型的数组，保持维度信息

    # 计算非缺失值的数量
    cnt = np.sum(~mask, axis=axis, dtype=np.intp, keepdims=_keepdims,
                 where=where)

    # 如果指定了均值 mean，则使用指定的均值；否则计算数组的均值
    if mean is not np._NoValue:
        avg = mean
    else:
        # 对于数组类型为 np.matrix，需要特殊处理以保持与旧版本的兼容性
        avg = np.sum(arr, axis=axis, dtype=dtype,
                     keepdims=_keepdims, where=where)
        avg = _divide_by_count(avg, cnt)  # 按计数值除以均值

    # 计算与均值的平方差
    np.subtract(arr, avg, out=arr, casting='unsafe', where=where)  # 计算差值并存储到 arr 中
    arr = _copyto(arr, 0, mask)  # 将 arr 中的缺失值用 0 填充
    if issubclass(arr.dtype.type, np.complexfloating):
        sqr = np.multiply(arr, arr.conj(), out=arr, where=where).real  # 对复数数组进行乘法运算并取实部
    else:
        sqr = np.multiply(arr, arr, out=arr, where=where)  # 对非复数数组进行乘法运算

    # 计算方差
    var = np.sum(sqr, axis=axis, dtype=dtype, out=out, keepdims=keepdims,
                 where=where)

    # 防范对减少的对象数组
    try:
        var_ndim = var.ndim  # 尝试获取 var 的维度信息
    except AttributeError:
        var_ndim = np.ndim(var)  # 对于没有 ndim 属性的对象，使用 np.ndim 获取维度信息
    if var_ndim < cnt.ndim:
        # ndarray 的子类可能会忽略 keepdims 参数，这里进行检查和压缩
        cnt = cnt.squeeze(axis)
    dof = cnt - ddof  # 自由度计算
    var = _divide_by_count(var, dof)  # 按自由度数目除以方差

    isbad = (dof <= 0)  # 检查自由度是否小于等于 0
    if np.any(isbad):
        warnings.warn("Degrees of freedom <= 0 for slice.", RuntimeWarning,
                      stacklevel=2)
        # NaN、inf 或负数都是可能的无效数值，用 NaN 明确替换它们
        var = _copyto(var, np.nan, isbad)  # 将无效值用 NaN 替换
    return var
# 分发器函数，接受多个参数并返回前两个参数
def _nanstd_dispatcher(a, axis=None, dtype=None, out=None, ddof=None,
                       keepdims=None, *, where=None, mean=None,
                       correction=None):
    return (a, out)

# 定义 nanstd 函数，并使用分发器进行装饰
@array_function_dispatch(_nanstd_dispatcher)
def nanstd(a, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue,
           *, where=np._NoValue, mean=np._NoValue, correction=np._NoValue):
    """
    Compute the standard deviation along the specified axis, while
    ignoring NaNs.

    Returns the standard deviation, a measure of the spread of a
    distribution, of the non-NaN array elements. The standard deviation is
    computed for the flattened array by default, otherwise over the
    specified axis.

    For all-NaN slices or slices with zero degrees of freedom, NaN is
    returned and a `RuntimeWarning` is raised.

    .. versionadded:: 1.8.0

    Parameters
    ----------
    a : array_like
        Calculate the standard deviation of the non-NaN values.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the standard deviation is computed. The default is
        to compute the standard deviation of the flattened array.
    dtype : dtype, optional
        Type to use in computing the standard deviation. For arrays of
        integer type the default is float64, for arrays of float types it
        is the same as the array type.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output but the type (of the
        calculated values) will be cast if necessary.
    ddof : {int, float}, optional
        Means Delta Degrees of Freedom.  The divisor used in calculations
        is ``N - ddof``, where ``N`` represents the number of non-NaN
        elements.  By default `ddof` is zero.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `a`.

        If this value is anything but the default it is passed through
        as-is to the relevant functions of the sub-classes.  If these
        functions do not have a `keepdims` kwarg, a RuntimeError will
        be raised.
    where : array_like of bool, optional
        Elements to include in the standard deviation.
        See `~numpy.ufunc.reduce` for details.

        .. versionadded:: 1.22.0

    mean : array_like, optional
        Provide the mean to prevent its recalculation. The mean should have
        a shape as if it was calculated with ``keepdims=True``.
        The axis for the calculation of the mean should be the same as used in
        the call to this std function.

        .. versionadded:: 1.26.0
    """
    correction : {int, float}, optional
        Array API compatible name for the ``ddof`` parameter. Only one of them
        can be provided at the same time.

        .. versionadded:: 2.0.0

    Returns
    -------
    standard_deviation : ndarray, see dtype parameter above.
        If `out` is None, return a new array containing the standard
        deviation, otherwise return a reference to the output array. If
        ddof is >= the number of non-NaN elements in a slice or the slice
        contains only NaNs, then the result for that slice is NaN.

    See Also
    --------
    var, mean, std
    nanvar, nanmean
    :ref:`ufuncs-output-type`

    Notes
    -----
    The standard deviation is the square root of the average of the squared
    deviations from the mean: ``std = sqrt(mean(abs(x - x.mean())**2))``.

    The average squared deviation is normally calculated as
    ``x.sum() / N``, where ``N = len(x)``.  If, however, `ddof` is
    specified, the divisor ``N - ddof`` is used instead. In standard
    statistical practice, ``ddof=1`` provides an unbiased estimator of the
    variance of the infinite population. ``ddof=0`` provides a maximum
    likelihood estimate of the variance for normally distributed variables.
    The standard deviation computed in this function is the square root of
    the estimated variance, so even with ``ddof=1``, it will not be an
    unbiased estimate of the standard deviation per se.

    Note that, for complex numbers, `std` takes the absolute value before
    squaring, so that the result is always real and nonnegative.

    For floating-point input, the *std* is computed using the same
    precision the input has. Depending on the input data, this can cause
    the results to be inaccurate, especially for float32 (see example
    below).  Specifying a higher-accuracy accumulator using the `dtype`
    keyword can alleviate this issue.

    Examples
    --------
    >>> a = np.array([[1, np.nan], [3, 4]])
    >>> np.nanstd(a)
    1.247219128924647
    >>> np.nanstd(a, axis=0)
    array([1., 0.])
    >>> np.nanstd(a, axis=1)
    array([0.,  0.5]) # may vary

    """
    # Compute the variance of the input array 'a' considering NaN values,
    # along the specified axis and with optional parameters
    var = nanvar(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
                 keepdims=keepdims, where=where, mean=mean,
                 correction=correction)
    
    # If 'var' is an ndarray, compute the square root of 'var' in place
    if isinstance(var, np.ndarray):
        std = np.sqrt(var, out=var)
    # If 'var' has a dtype attribute, compute the square root using its type
    elif hasattr(var, 'dtype'):
        std = var.dtype.type(np.sqrt(var))
    # Otherwise, compute the square root of 'var' directly
    else:
        std = np.sqrt(var)
    
    # Return the computed standard deviation
    return std
```