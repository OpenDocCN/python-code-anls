# `.\numpy\numpy\lib\_arraysetops_impl.py`

```
"""
Set operations for arrays based on sorting.

Notes
-----

For floating point arrays, inaccurate results may appear due to usual round-off
and floating point comparison issues.

Speed could be gained in some operations by an implementation of
`numpy.sort`, that can provide directly the permutation vectors, thus avoiding
calls to `numpy.argsort`.

Original author: Robert Cimrman

"""
import functools  # 导入 functools 模块，用于高阶函数和函数装饰器
import warnings  # 导入 warnings 模块，用于警告处理
from typing import NamedTuple  # 导入 NamedTuple 类型提示

import numpy as np  # 导入 NumPy 库，约定用法是 np
from numpy._core import overrides  # 导入 overrides 模块
from numpy._core._multiarray_umath import _array_converter  # 导入 _array_converter 函数


array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')  # 创建 array_function_dispatch 函数，用于分派数组函数的装饰器


__all__ = [
    "ediff1d", "in1d", "intersect1d", "isin", "setdiff1d", "setxor1d",
    "union1d", "unique", "unique_all", "unique_counts", "unique_inverse",
    "unique_values"
]  # 定义模块中公开的函数列表


def _ediff1d_dispatcher(ary, to_end=None, to_begin=None):
    return (ary, to_end, to_begin)


@array_function_dispatch(_ediff1d_dispatcher)
def ediff1d(ary, to_end=None, to_begin=None):
    """
    The differences between consecutive elements of an array.

    Parameters
    ----------
    ary : array_like
        If necessary, will be flattened before the differences are taken.
    to_end : array_like, optional
        Number(s) to append at the end of the returned differences.
    to_begin : array_like, optional
        Number(s) to prepend at the beginning of the returned differences.

    Returns
    -------
    ediff1d : ndarray
        The differences. Loosely, this is ``ary.flat[1:] - ary.flat[:-1]``.

    See Also
    --------
    diff, gradient

    Notes
    -----
    When applied to masked arrays, this function drops the mask information
    if the `to_begin` and/or `to_end` parameters are used.

    Examples
    --------
    >>> x = np.array([1, 2, 4, 7, 0])
    >>> np.ediff1d(x)
    array([ 1,  2,  3, -7])

    >>> np.ediff1d(x, to_begin=-99, to_end=np.array([88, 99]))
    array([-99,   1,   2, ...,  -7,  88,  99])

    The returned array is always 1D.

    >>> y = [[1, 2, 4], [1, 6, 24]]
    >>> np.ediff1d(y)
    array([ 1,  2, -3,  5, 18])

    """
    conv = _array_converter(ary)  # 转换输入数组为通用数组
    ary = conv[0].ravel()  # 将数组展平

    dtype_req = ary.dtype  # 获取数组的数据类型作为输出要求的数据类型

    if to_begin is None and to_end is None:  # 如果未提供 to_begin 和 to_end 参数
        return ary[1:] - ary[:-1]  # 返回连续元素的差值数组

    if to_begin is None:  # 如果只提供了 to_end 参数
        l_begin = 0  # 设置起始值个数为 0
    else:
        to_begin = np.asanyarray(to_begin)  # 转换 to_begin 参数为 NumPy 数组
        if not np.can_cast(to_begin, dtype_req, casting="same_kind"):
            # 如果无法将 to_begin 的数据类型转换为与 ary 相同的数据类型
            raise TypeError("dtype of `to_begin` must be compatible "
                            "with input `ary` under the `same_kind` rule.")

        to_begin = to_begin.ravel()  # 展平 to_begin 数组
        l_begin = len(to_begin)  # 获取 to_begin 数组的长度

    if to_end is None:  # 如果只提供了 to_begin 参数
        l_end = 0  # 设置结束值个数为 0
    # 如果不满足条件，将 `to_end` 转换为 NumPy 数组
    else:
        to_end = np.asanyarray(to_end)
        # 检查 `to_end` 是否可以在 `same_kind` 规则下转换为 `dtype_req` 类型，否则抛出类型错误
        if not np.can_cast(to_end, dtype_req, casting="same_kind"):
            raise TypeError("dtype of `to_end` must be compatible "
                            "with input `ary` under the `same_kind` rule.")
        
        # 将 `to_end` 摊平为一维数组，并获取其长度
        to_end = to_end.ravel()
        l_end = len(to_end)

    # 计算生成结果数组的长度，至少为 0
    l_diff = max(len(ary) - 1, 0)
    # 根据 `ary` 的类型和形状创建一个和其相同的空数组
    result = np.empty_like(ary, shape=l_diff + l_begin + l_end)

    # 如果 `l_begin` 大于 0，则将 `to_begin` 的内容复制到 `result` 的前部
    if l_begin > 0:
        result[:l_begin] = to_begin
    # 如果 `l_end` 大于 0，则将 `to_end` 的内容复制到 `result` 的尾部
    if l_end > 0:
        result[l_begin + l_diff:] = to_end
    # 使用 NumPy 函数计算 `ary` 相邻元素之间的差，并将结果存入 `result` 的指定范围内
    np.subtract(ary[1:], ary[:-1], result[l_begin:l_begin + l_diff])

    # 将计算结果封装并返回
    return conv.wrap(result)
# 解包单元素元组，用于作为返回值
def _unpack_tuple(x):
    if len(x) == 1:
        return x[0]
    else:
        return x


# 唯一性调度器，返回输入的元组 ar
def _unique_dispatcher(ar, return_index=None, return_inverse=None,
                       return_counts=None, axis=None, *, equal_nan=None):
    return (ar,)


# 使用数组函数调度装饰器，将 _unique_dispatcher 函数作为分派函数
@array_function_dispatch(_unique_dispatcher)
def unique(ar, return_index=False, return_inverse=False,
           return_counts=False, axis=None, *, equal_nan=True):
    """
    查找数组的唯一元素。

    返回数组的排序唯一元素。除了唯一元素外，还有三个可选的输出：

    * 给出唯一值的输入数组的索引
    * 重构输入数组的唯一数组的索引
    * 每个唯一值在输入数组中出现的次数

    Parameters
    ----------
    ar : array_like
        输入数组。除非指定了 `axis`，否则如果它不是 1-D，则将其展平。
    return_index : bool, optional
        如果为 True，则返回 `ar` 的索引（沿指定的轴，如果提供了，则在展平的数组中），
        这些索引会导致唯一数组。
    return_inverse : bool, optional
        如果为 True，则还返回唯一数组的索引（对于指定的轴，如果提供的话），这些索引可用于重构 `ar`。
    return_counts : bool, optional
        如果为 True，则还返回 `ar` 中每个唯一项出现的次数。
    axis : int or None, optional
        要操作的轴。如果为 None，则将 `ar` 展平。如果为整数，则由给定轴索引的子数组将被展平，
        并视为具有给定轴维度的 1-D 数组元素，请参阅说明以获取更多详细信息。
        如果使用 `axis` kwarg，则不支持对象数组或包含对象的结构化数组。默认为 None。

        .. versionadded:: 1.13.0

    equal_nan : bool, optional
        如果为 True，则将返回数组中的多个 NaN 值合并为一个。

        .. versionadded:: 1.24

    Returns
    -------
    unique : ndarray
        排序的唯一值。
    unique_indices : ndarray, optional
        原始数组中唯一值的第一次出现的索引。仅在 `return_index` 为 True 时提供。
    unique_inverse : ndarray, optional
        用于从唯一数组重构原始数组的索引。仅在 `return_inverse` 为 True 时提供。
    unique_counts : ndarray, optional
        每个唯一值在原始数组中出现的次数。仅在 `return_counts` 为 True 时提供。

        .. versionadded:: 1.9.0

    See Also
    --------
    repeat : 重复数组元素。

    Notes
    -----
    当指定轴时，由轴索引的子数组被排序。
    """
    return (ar,)
    """
    将指定的轴移动到数组的第一个维度，保持其他轴的顺序，并按C顺序展平子数组。展平后的子数组视为结构化类型，每个元素都被赋予标签，
    使得最终得到一个结构化类型的一维数组，可以像任何其他一维数组一样处理。结果是展平的子数组按词典顺序排列，从第一个元素开始。
    
    .. versionchanged: 1.21
        如果输入数组中包含 NaN 值，将所有 NaN 值放在排序后的唯一值的末尾。
    
        对于复数数组，所有 NaN 值都被视为等效（无论 NaN 是否在实部或虚部）。返回数组中词典顺序最小的 NaN 作为代表值 -
        有关复数数组的词典顺序定义，请参见 np.sort。
    
    .. versionchanged: 2.0
        对于多维输入，当 axis=None 时，重新整形 unique_inverse，使得可以使用 `np.take(unique, unique_inverse)` 重构输入，
        在其他情况下使用 `np.take_along_axis(unique, unique_inverse, axis=axis)`。
    
    Examples
    --------
    >>> np.unique([1, 1, 2, 2, 3, 3])
    array([1, 2, 3])
    >>> a = np.array([[1, 1], [2, 3]])
    >>> np.unique(a)
    array([1, 2, 3])
    
    返回二维数组的唯一行
    
    >>> a = np.array([[1, 0, 0], [1, 0, 0], [2, 3, 4]])
    >>> np.unique(a, axis=0)
    array([[1, 0, 0], [2, 3, 4]])
    
    返回原始数组中给定唯一值的索引：
    
    >>> a = np.array(['a', 'b', 'b', 'c', 'a'])
    >>> u, indices = np.unique(a, return_index=True)
    >>> u
    array(['a', 'b', 'c'], dtype='<U1')
    >>> indices
    array([0, 1, 3])
    >>> a[indices]
    array(['a', 'b', 'c'], dtype='<U1')
    
    从唯一值和逆向索引重构输入数组：
    
    >>> a = np.array([1, 2, 6, 4, 2, 3, 2])
    >>> u, indices = np.unique(a, return_inverse=True)
    >>> u
    array([1, 2, 3, 4, 6])
    >>> indices
    array([0, 1, 4, 3, 1, 2, 1])
    >>> u[indices]
    array([1, 2, 6, 4, 2, 3, 2])
    
    从唯一值和计数重构输入数组的值：
    
    >>> a = np.array([1, 2, 6, 4, 2, 3, 2])
    >>> values, counts = np.unique(a, return_counts=True)
    >>> values
    array([1, 2, 3, 4, 6])
    >>> counts
    array([1, 3, 1, 1, 1])
    >>> np.repeat(values, counts)
    array([1, 2, 2, 2, 3, 4, 6])    # 原始顺序未保留
    """
    ar = np.asanyarray(ar)  # 将输入转换为 ndarray，如果已经是 ndarray，则不会复制
    if axis is None:
        # 如果 axis 为 None，则调用 _unique1d 处理一维情况
        ret = _unique1d(ar, return_index, return_inverse, return_counts, 
                        equal_nan=equal_nan, inverse_shape=ar.shape)
        return _unpack_tuple(ret)  # 返回解包后的结果
    
    # 如果指定了 axis 并且不为 None，则执行以下代码块
    try:
        ar = np.moveaxis(ar, axis, 0)  # 将指定的轴移动到数组的第一个维度
    except np.exceptions.AxisError:
        # 如果捕获到 AxisError 异常，则移除错误消息中的 "axis1" 或 "axis2" 前缀
        raise np.exceptions.AxisError(axis, ar.ndim) from None
    # 创建一个长度为 ar.ndim 的列表，每个元素都为 1
    inverse_shape = [1] * ar.ndim
    # 将 inverse_shape 列表中的第 axis 个元素设置为 ar 的第一维的长度
    inverse_shape[axis] = ar.shape[0]

    # 必须将数组重塑为连续的二维数组才能继续操作...
    # 保存原始的形状和数据类型信息
    orig_shape, orig_dtype = ar.shape, ar.dtype
    # 将 ar 重塑为形状为 (orig_shape[0], np.prod(orig_shape[1:], dtype=np.intp)) 的数组
    ar = ar.reshape(orig_shape[0], np.prod(orig_shape[1:], dtype=np.intp))
    # 确保 ar 是一个连续的数组
    ar = np.ascontiguousarray(ar)
    # 创建一个结构化数据类型，每个字段名为 'f{i}'，数据类型与 ar 相同
    dtype = [('f{i}'.format(i=i), ar.dtype) for i in range(ar.shape[1])]

    # 此时，`ar` 的形状为 `(n, m)`，`dtype` 是一个结构化数据类型，有 `m` 个字段，每个字段的数据类型与 `ar` 相同。
    # 在接下来的步骤中，我们创建数组 `consolidated`，形状为 `(n,)`，数据类型为 `dtype`。
    try:
        if ar.shape[1] > 0:
            # 如果 ar 的第二维度大于 0，则通过视图转换为 `dtype` 类型的数组
            consolidated = ar.view(dtype)
        else:
            # 如果 ar 的第二维度等于 0，则 dtype 将是 `np.dtype([])`，其大小为 0，
            # `ar.view(dtype)` 将失败。我们使用 `np.empty` 明确地创建形状为 `(len(ar),)` 的数组。
            # 在这种情况下，由于 `dtype` 的大小为 0，结果的总大小仍然为 0 字节。
            consolidated = np.empty(len(ar), dtype=dtype)
    except TypeError as e:
        # 对于对象数组等情况，无法执行这些操作...
        msg = 'The axis argument to unique is not supported for dtype {dt}'
        # 抛出 TypeError 异常，说明 unique 函数不支持当前的数据类型 `ar.dtype`
        raise TypeError(msg.format(dt=ar.dtype)) from e

    # 定义函数 reshape_uniq，用于重塑唯一化后的数组
    def reshape_uniq(uniq):
        n = len(uniq)
        # 将唯一化后的数组视图转换回原始的数据类型 `orig_dtype`
        uniq = uniq.view(orig_dtype)
        # 将 uniq 重塑为形状为 `(n, *orig_shape[1:])` 的数组
        uniq = uniq.reshape(n, *orig_shape[1:])
        # 将 uniq 数组的第 0 维移动到指定的 axis 位置
        uniq = np.moveaxis(uniq, 0, axis)
        return uniq

    # 调用 _unique1d 函数处理 consolidated 数组，并根据参数返回需要的结果
    output = _unique1d(consolidated, return_index,
                       return_inverse, return_counts,
                       equal_nan=equal_nan, inverse_shape=inverse_shape)
    # 对输出结果的第一个元素进行 reshape_uniq 处理，并将处理后的结果与其它元素组合成元组
    output = (reshape_uniq(output[0]),) + output[1:]
    # 返回解包后的元组 output
    return _unpack_tuple(output)
# 将输入数组转换为 NumPy 数组，并展平为一维数组
ar = np.asanyarray(ar).flatten()

# 检查是否需要返回索引或者逆向索引
optional_indices = return_index or return_inverse

# 如果需要返回索引或者逆向索引，则按排序后的顺序排列数组
if optional_indices:
    perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
    aux = ar[perm]
else:
    # 否则，直接对数组进行排序
    ar.sort()
    aux = ar

# 创建一个布尔掩码，标记唯一元素的位置
mask = np.empty(aux.shape, dtype=np.bool)
mask[:1] = True

# 如果数组中包含 NaN 并且需要考虑 NaN 相等性
if (equal_nan and aux.shape[0] > 0 and aux.dtype.kind in "cfmM" and
        np.isnan(aux[-1])):
    if aux.dtype.kind == "c":  # 对于复数，所有的 NaN 被视为等价的
        # 查找第一个 NaN 的位置
        aux_firstnan = np.searchsorted(np.isnan(aux), True, side='left')
    else:
        # 查找第一个 NaN 的位置
        aux_firstnan = np.searchsorted(aux, aux[-1], side='left')

    # 根据 NaN 的位置更新掩码
    if aux_firstnan > 0:
        mask[1:aux_firstnan] = (
            aux[1:aux_firstnan] != aux[:aux_firstnan - 1])
    mask[aux_firstnan] = True
    mask[aux_firstnan + 1:] = False
else:
    # 根据相邻元素的不同更新掩码
    mask[1:] = aux[1:] != aux[:-1]

# 返回唯一元素数组及其可能的附加信息
ret = (aux[mask],)
if return_index:
    ret += (perm[mask],)
if return_inverse:
    # 生成逆向索引，用于重建输入数组
    imask = np.cumsum(mask) - 1
    inv_idx = np.empty(mask.shape, dtype=np.intp)
    inv_idx[perm] = imask
    ret += (inv_idx.reshape(inverse_shape),)
if return_counts:
    # 计算每个唯一元素的出现次数
    idx = np.concatenate(np.nonzero(mask) + ([mask.size],))
    ret += (np.diff(idx),)
return ret
    # 调用 unique 函数，并返回其结果作为 UniqueAllResult 对象的参数
    result = unique(
        x,  # 输入的数组或序列 x
        return_index=True,      # 返回唯一值的索引数组
        return_inverse=True,    # 返回唯一值的逆映射数组
        return_counts=True,     # 返回唯一值的计数数组
        equal_nan=False         # 是否将 NaN 视为相等的值（默认为不相等）
    )
    # 将 unique 函数返回的结果解构为 UniqueAllResult 的构造函数参数，并返回该对象
    return UniqueAllResult(*result)
# 定义一个单参数调度函数，用于分派到唯一计数函数
def _unique_counts_dispatcher(x, /):
    return (x,)


# 使用数组函数分派装饰器，为唯一计数函数提供多种实现方式
@array_function_dispatch(_unique_counts_dispatcher)
def unique_counts(x):
    """
    Find the unique elements and counts of an input array `x`.

    This function is an Array API compatible alternative to:

    >>> x = np.array([1, 1, 2])
    >>> np.unique(x, return_counts=True, equal_nan=False)
    (array([1, 2]), array([2, 1]))

    Parameters
    ----------
    x : array_like
        Input array. It will be flattened if it is not already 1-D.

    Returns
    -------
    out : namedtuple
        The result containing:

        * values - The unique elements of an input array.
        * counts - The corresponding counts for each unique element.

    See Also
    --------
    unique : Find the unique elements of an array.

    Examples
    --------
    >>> np.unique_counts([1, 1, 2])
    UniqueCountsResult(values=array([1, 2]), counts=array([2, 1]))

    """
    # 调用通用唯一函数，请求返回唯一元素和对应计数，不返回索引和逆向索引，不处理 NaN 值
    result = unique(
        x,
        return_index=False,
        return_inverse=False,
        return_counts=True,
        equal_nan=False
    )
    # 返回结果为命名元组 UniqueCountsResult，包含唯一元素和计数数组
    return UniqueCountsResult(*result)


# 定义一个单参数调度函数，用于分派到唯一逆向索引函数
def _unique_inverse_dispatcher(x, /):
    return (x,)


# 使用数组函数分派装饰器，为唯一逆向索引函数提供多种实现方式
@array_function_dispatch(_unique_inverse_dispatcher)
def unique_inverse(x):
    """
    Find the unique elements of `x` and indices to reconstruct `x`.

    This function is Array API compatible alternative to:

    >>> x = np.array([1, 1, 2])
    >>> np.unique(x, return_inverse=True, equal_nan=False)
    (array([1, 2]), array([0, 0, 1]))

    Parameters
    ----------
    x : array_like
        Input array. It will be flattened if it is not already 1-D.

    Returns
    -------
    out : namedtuple
        The result containing:

        * values - The unique elements of an input array.
        * inverse_indices - The indices from the set of unique elements
          that reconstruct `x`.

    See Also
    --------
    unique : Find the unique elements of an array.

    Examples
    --------
    >>> np.unique_inverse([1, 1, 2])
    UniqueInverseResult(values=array([1, 2]), inverse_indices=array([0, 0, 1]))

    """
    # 调用通用唯一函数，请求返回唯一元素和逆向索引，不返回计数，不处理 NaN 值
    result = unique(
        x,
        return_index=False,
        return_inverse=True,
        return_counts=False,
        equal_nan=False
    )
    # 返回结果为命名元组 UniqueInverseResult，包含唯一元素和逆向索引数组
    return UniqueInverseResult(*result)


# 定义一个单参数调度函数，用于分派到唯一值函数
def _unique_values_dispatcher(x, /):
    return (x,)


# 使用数组函数分派装饰器，为唯一值函数提供多种实现方式
@array_function_dispatch(_unique_values_dispatcher)
def unique_values(x):
    """
    Returns the unique elements of an input array `x`.

    This function is Array API compatible alternative to:

    >>> x = np.array([1, 1, 2])
    >>> np.unique(x, equal_nan=False)
    array([1, 2])

    Parameters
    ----------
    x : array_like
        Input array. It will be flattened if it is not already 1-D.

    Returns
    -------
    out : ndarray
        The unique elements of an input array.

    See Also
    --------
    unique : Find the unique elements of an array.

    Examples
    --------
    >>> np.unique_values([1, 1, 2])
    """
    array([1, 2])


# 创建一个包含元素 [1, 2] 的 NumPy 数组



    """
    return unique(
        x,
        return_index=False,
        return_inverse=False,
        return_counts=False,
        equal_nan=False
    )


# 返回数组 x 的唯一元素
# - return_index: 是否返回唯一值在原始数组中的索引
# - return_inverse: 是否返回唯一值在原始数组中的逆映射
# - return_counts: 是否返回唯一值的计数
# - equal_nan: 是否将 NaN 视为相等值处理
# 分发函数，将参数传递给实际函数_intersect1d_dispatcher
def _intersect1d_dispatcher(
        ar1, ar2, assume_unique=None, return_indices=None):
    return (ar1, ar2)

# 使用装饰器进行分派，将参数传递给_intersect1d_dispatcher函数
@array_function_dispatch(_intersect1d_dispatcher)
def intersect1d(ar1, ar2, assume_unique=False, return_indices=False):
    """
    Find the intersection of two arrays.

    Return the sorted, unique values that are in both of the input arrays.

    Parameters
    ----------
    ar1, ar2 : array_like
        Input arrays. Will be flattened if not already 1D.
    assume_unique : bool
        If True, the input arrays are both assumed to be unique, which
        can speed up the calculation.  If True but ``ar1`` or ``ar2`` are not
        unique, incorrect results and out-of-bounds indices could result.
        Default is False.
    return_indices : bool
        If True, the indices which correspond to the intersection of the two
        arrays are returned. The first instance of a value is used if there are
        multiple. Default is False.

        .. versionadded:: 1.15.0

    Returns
    -------
    intersect1d : ndarray
        Sorted 1D array of common and unique elements.
    comm1 : ndarray
        The indices of the first occurrences of the common values in `ar1`.
        Only provided if `return_indices` is True.
    comm2 : ndarray
        The indices of the first occurrences of the common values in `ar2`.
        Only provided if `return_indices` is True.

    Examples
    --------
    >>> np.intersect1d([1, 3, 4, 3], [3, 1, 2, 1])
    array([1, 3])

    To intersect more than two arrays, use functools.reduce:

    >>> from functools import reduce
    >>> reduce(np.intersect1d, ([1, 3, 4, 3], [3, 1, 2, 1], [6, 3, 4, 2]))
    array([3])

    To return the indices of the values common to the input arrays
    along with the intersected values:

    >>> x = np.array([1, 1, 2, 3, 4])
    >>> y = np.array([2, 1, 4, 6])
    >>> xy, x_ind, y_ind = np.intersect1d(x, y, return_indices=True)
    >>> x_ind, y_ind
    (array([0, 2, 4]), array([1, 0, 2]))
    >>> xy, x[x_ind], y[y_ind]
    (array([1, 2, 4]), array([1, 2, 4]), array([1, 2, 4]))

    """
    # 将输入数组转换为任意数组
    ar1 = np.asanyarray(ar1)
    ar2 = np.asanyarray(ar2)

    # 如果不假定数组唯一性
    if not assume_unique:
        # 如果需要返回索引
        if return_indices:
            # 对 ar1 和 ar2 进行唯一值处理，并返回索引
            ar1, ind1 = unique(ar1, return_index=True)
            ar2, ind2 = unique(ar2, return_index=True)
        else:
            # 对 ar1 和 ar2 进行唯一值处理
            ar1 = unique(ar1)
            ar2 = unique(ar2)
    else:
        # 将 ar1 和 ar2 拉直成一维数组
        ar1 = ar1.ravel()
        ar2 = ar2.ravel()

    # 将两个数组合并
    aux = np.concatenate((ar1, ar2))

    # 如果需要返回索引
    if return_indices:
        # 对合并后的数组进行排序，并返回排序后的数组
        aux_sort_indices = np.argsort(aux, kind='mergesort')
        aux = aux[aux_sort_indices]
    else:
        # 对合并后的数组进行排序
        aux.sort()

    # 创建一个掩码，指示重复的元素
    mask = aux[1:] == aux[:-1]
    # 提取出现重复的元素
    int1d = aux[:-1][mask]
    # 如果需要返回索引
    if return_indices:
        # 从 aux_sort_indices 中选择符合条件的索引，并根据 mask 进行过滤
        ar1_indices = aux_sort_indices[:-1][mask]
        # 从 aux_sort_indices 中选择下一个索引，并根据 mask 进行过滤，并且计算相对于 ar1 的偏移量
        ar2_indices = aux_sort_indices[1:][mask] - ar1.size
        
        # 如果不假设唯一性，将 ar1_indices 和 ar2_indices 转换为原始索引
        if not assume_unique:
            ar1_indices = ind1[ar1_indices]
            ar2_indices = ind2[ar2_indices]

        # 返回排序后的整数数组以及计算得到的索引
        return int1d, ar1_indices, ar2_indices
    else:
        # 如果不需要返回索引，直接返回排序后的整数数组
        return int1d
# 根据 _setxor1d_dispatcher 的分派规则，返回输入的两个数组
def _setxor1d_dispatcher(ar1, ar2, assume_unique=None):
    return (ar1, ar2)


# 使用 array_function_dispatch 装饰器注册 _setxor1d_dispatcher 分派函数
@array_function_dispatch(_setxor1d_dispatcher)
# 定义 setxor1d 函数，计算两个数组的对称差集
def setxor1d(ar1, ar2, assume_unique=False):
    """
    Find the set exclusive-or of two arrays.

    Return the sorted, unique values that are in only one (not both) of the
    input arrays.

    Parameters
    ----------
    ar1, ar2 : array_like
        Input arrays.
    assume_unique : bool
        If True, the input arrays are both assumed to be unique, which
        can speed up the calculation. Default is False.

    Returns
    -------
    setxor1d : ndarray
        Sorted 1D array of unique values that are in only one of the input
        arrays.

    Examples
    --------
    >>> a = np.array([1, 2, 3, 2, 4])
    >>> b = np.array([2, 3, 5, 7, 5])
    >>> np.setxor1d(a,b)
    array([1, 4, 5, 7])

    """
    # 如果 assume_unique 参数为 False，则将输入数组 ar1 和 ar2 唯一化
    if not assume_unique:
        ar1 = unique(ar1)
        ar2 = unique(ar2)

    # 将 ar1 和 ar2 的内容合并成一个新数组 aux
    aux = np.concatenate((ar1, ar2), axis=None)
    # 如果 aux 的大小为 0，则直接返回 aux
    if aux.size == 0:
        return aux

    # 对 aux 进行排序
    aux.sort()
    # 创建一个布尔数组 flag，标记 aux 中每个元素是否与前一个元素不同
    flag = np.concatenate(([True], aux[1:] != aux[:-1], [True]))
    # 返回 aux 中在 flag 为 True 的位置上的元素，即计算的对称差集
    return aux[flag[1:] & flag[:-1]]


# 根据 _in1d_dispatcher 的分派规则，返回输入的两个数组
def _in1d_dispatcher(ar1, ar2, assume_unique=None, invert=None, *, kind=None):
    return (ar1, ar2)


# 使用 array_function_dispatch 装饰器注册 _in1d_dispatcher 分派函数
@array_function_dispatch(_in1d_dispatcher)
# 定义 in1d 函数，测试第一个数组的每个元素是否也出现在第二个数组中
def in1d(ar1, ar2, assume_unique=False, invert=False, *, kind=None):
    """
    Test whether each element of a 1-D array is also present in a second array.

    .. deprecated:: 2.0
        Use :func:`isin` instead of `in1d` for new code.

    Returns a boolean array the same length as `ar1` that is True
    where an element of `ar1` is in `ar2` and False otherwise.

    Parameters
    ----------
    ar1 : (M,) array_like
        Input array.
    ar2 : array_like
        The values against which to test each value of `ar1`.
    assume_unique : bool, optional
        If True, the input arrays are both assumed to be unique, which
        can speed up the calculation.  Default is False.
    invert : bool, optional
        If True, the values in the returned array are inverted (that is,
        False where an element of `ar1` is in `ar2` and True otherwise).
        Default is False. ``np.in1d(a, b, invert=True)`` is equivalent
        to (but is faster than) ``np.invert(in1d(a, b))``.

    """
    # 返回输入数组 ar1 和 ar2
    return (ar1, ar2)
    # 根据给定的条件判断数组 ar1 中的元素是否在数组 ar2 中，并返回布尔类型的数组
    # 该函数已在 NumPy 2.0 版本中被弃用，推荐使用 np.isin 替代
    # 如果 ar2 是集合或类似非序列容器，则将其转换为数组可能导致意外的行为
    # ar1: 待检查的数组
    # ar2: 目标数组或集合，用于检查 ar1 中的元素是否存在于其中
    # assume_unique: 是否假设 ar2 中的元素是唯一的，这个参数在 kind='table' 时不起作用
    # invert: 是否反转返回的布尔数组
    # kind: 选择算法的类型，影响函数的速度和内存使用，但不影响最终结果

    # 根据内存消耗自动选择算法类型：如果内存消耗低于或等于 6 倍 ar1 和 ar2 大小的总和，则选择 'table'，否则选择 'sort'
    # 'sort' 使用归并排序，大约需要的内存是 ar1 和 ar2 大小总和的 6 倍，不考虑 dtype 大小
    # 'table' 使用类似计数排序的查找表方法，仅适用于布尔和整数数组，内存消耗为 ar1 的大小加上 ar2 的最大值和最小值之差

    # 返回一个布尔数组，表示 ar1 中的元素是否在 ar2 中存在
    # 返回结果的形状为 (M,)
    return _in1d(ar1, ar2, assume_unique, invert, kind=kind)
# 将输入的 ar1 数组转换为 NumPy 数组，并展平为一维数组
ar1 = np.asarray(ar1).ravel()

# 将输入的 ar2 数组转换为 NumPy 数组，并展平为一维数组
ar2 = np.asarray(ar2).ravel()

# 如果 ar2 的数据类型为对象数组，则重新形状为列数为 1 的二维数组
if ar2.dtype == object:
    ar2 = ar2.reshape(-1, 1)

# 检查 kind 参数是否有效，只接受 None, 'sort' 或 'table' 三种取值
if kind not in {None, 'sort', 'table'}:
    raise ValueError(
        f"Invalid kind: '{kind}'. Please use None, 'sort' or 'table'.")

# 检查所有输入数组是否为整数或布尔类型，以决定是否可以使用 'table' 方法
is_int_arrays = all(ar.dtype.kind in ("u", "i", "b") for ar in (ar1, ar2))
use_table_method = is_int_arrays and kind in {None, 'table'}
    # 如果使用表格方法
    if use_table_method:
        # 如果 ar2 是空数组
        if ar2.size == 0:
            # 如果要求反转结果，返回与 ar1 同形状的全为 True 的数组
            if invert:
                return np.ones_like(ar1, dtype=bool)
            # 否则返回与 ar1 同形状的全为 False 的数组
            else:
                return np.zeros_like(ar1, dtype=bool)

        # 将布尔类型数组 ar1 转换为 uint8 类型，以便使用快速整数算法
        if ar1.dtype == bool:
            ar1 = ar1.astype(np.uint8)
        # 将布尔类型数组 ar2 转换为 uint8 类型，以便使用快速整数算法
        if ar2.dtype == bool:
            ar2 = ar2.astype(np.uint8)

        # 计算 ar2 数组的最小值和最大值，并转换为整数
        ar2_min = int(np.min(ar2))
        ar2_max = int(np.max(ar2))

        # 计算 ar2 数组数值范围
        ar2_range = ar2_max - ar2_min

        # 判断是否可以使用表格方法的限制条件：
        # 1. 内存使用量是否在可接受范围内
        below_memory_constraint = ar2_range <= 6 * (ar1.size + ar2.size)
        # 2. 检查 (ar2 - ar2_min) 是否会溢出，数据类型为 ar2.dtype
        range_safe_from_overflow = ar2_range <= np.iinfo(ar2.dtype).max

        # 根据性能优化条件判断是否使用表格方法
        if (
            range_safe_from_overflow and 
            (below_memory_constraint or kind == 'table')
        ):
            # 根据是否要求反转结果，创建初始输出数组
            if invert:
                outgoing_array = np.ones_like(ar1, dtype=bool)
            else:
                outgoing_array = np.zeros_like(ar1, dtype=bool)

            # 将 ar2 中存在的整数位置设为 1
            if invert:
                isin_helper_ar = np.ones(ar2_range + 1, dtype=bool)
                isin_helper_ar[ar2 - ar2_min] = 0
            else:
                isin_helper_ar = np.zeros(ar2_range + 1, dtype=bool)
                isin_helper_ar[ar2 - ar2_min] = 1

            # 创建基本掩码，标记 ar1 中在 ar2_min 和 ar2_max 范围内的元素
            basic_mask = (ar1 <= ar2_max) & (ar1 >= ar2_min)
            # 使用 isin_helper_ar 对符合条件的 ar1 元素进行标记
            outgoing_array[basic_mask] = isin_helper_ar[
                    np.subtract(ar1[basic_mask], ar2_min, dtype=np.intp)]

            # 返回最终结果数组
            return outgoing_array
        elif kind == 'table':  # 如果不满足 range_safe_from_overflow
            # 抛出运行时错误，提示 ar2 或 ar1 的值范围超出数据类型的最大整数值
            raise RuntimeError(
                "You have specified kind='table', "
                "but the range of values in `ar2` or `ar1` exceed the "
                "maximum integer of the datatype. "
                "Please set `kind` to None or 'sort'."
            )
    elif kind == 'table':
        # 如果 kind 参数为 'table' 但不满足使用表格方法的条件，抛出值错误
        raise ValueError(
            "The 'table' method is only "
            "supported for boolean or integer arrays. "
            "Please select 'sort' or None for kind."
        )

    # 检查 ar1 或 ar2 是否包含任意对象类型
    contains_object = ar1.dtype.hasobject or ar2.dtype.hasobject

    # 当满足以下条件时执行该段代码：
    # a) 第一个条件为真，使代码运行速度显著提高
    # b) 第二个条件为真，即 `ar1` 或 `ar2` 可能包含任意对象
    # 如果数组 ar2 的长度小于 ar1 的长度乘以 10 的 0.145 次方，或者包含对象（即数组元素不唯一），则执行以下代码块
    if len(ar2) < 10 * len(ar1) ** 0.145 or contains_object:
        # 如果需要反转结果，则创建一个所有元素为 True 的布尔掩码数组
        if invert:
            mask = np.ones(len(ar1), dtype=bool)
            # 对 ar2 中的每个元素 a，将 mask 中对应 ar1 中等于 a 的位置置为 False
            for a in ar2:
                mask &= (ar1 != a)
        else:
            # 否则创建一个所有元素为 False 的布尔掩码数组
            mask = np.zeros(len(ar1), dtype=bool)
            # 对 ar2 中的每个元素 a，将 mask 中对应 ar1 中等于 a 的位置置为 True
            for a in ar2:
                mask |= (ar1 == a)
        # 返回最终的布尔掩码数组 mask
        return mask

    # 否则，使用排序方法处理数组 ar1 和 ar2
    # 如果不假设数组元素唯一，则对 ar1 和 ar2 去重，并返回去重后的数组 ar1 和 ar2，以及 ar1 的逆序索引
    if not assume_unique:
        ar1, rev_idx = np.unique(ar1, return_inverse=True)
        ar2 = np.unique(ar2)

    # 将去重后的 ar1 和 ar2 数组连接起来形成新的数组 ar
    ar = np.concatenate((ar1, ar2))
    # 对新数组 ar 进行稳定排序，使用 'mergesort' 算法确保结果正确
    order = ar.argsort(kind='mergesort')
    # 根据排序后的顺序重新排列原始数组 ar，并存储在 sar 中
    sar = ar[order]
    # 如果需要反转结果，则比较 sar 数组中相邻元素是否不相等，生成布尔数组 bool_ar
    if invert:
        bool_ar = (sar[1:] != sar[:-1])
    else:
        # 否则比较 sar 数组中相邻元素是否相等，生成布尔数组 bool_ar
        bool_ar = (sar[1:] == sar[:-1])
    # 构建最终的布尔数组 flag，将 bool_ar 和 invert 连接起来
    flag = np.concatenate((bool_ar, [invert]))
    # 创建一个与 ar 形状相同的空布尔数组 ret
    ret = np.empty(ar.shape, dtype=bool)
    # 根据排序后的索引 order，将 flag 中的值赋给 ret
    ret[order] = flag

    # 如果假设数组元素唯一，则返回 ret 的前 len(ar1) 个元素
    if assume_unique:
        return ret[:len(ar1)]
    else:
        # 否则根据之前保存的逆序索引 rev_idx 返回 ret 的对应元素
        return ret[rev_idx]
# 定义一个派发器函数 `_isin_dispatcher`，用于决定如何分派参数到对应的实际函数
def _isin_dispatcher(element, test_elements, assume_unique=None, invert=None,
                     *, kind=None):
    # 返回一个包含输入参数的元组，用于后续处理
    return (element, test_elements)

# 使用装饰器 `array_function_dispatch` 将 `_isin_dispatcher` 函数与 `isin` 函数关联起来
@array_function_dispatch(_isin_dispatcher)
def isin(element, test_elements, assume_unique=False, invert=False, *,
         kind=None):
    """
    计算 `element` 是否在 `test_elements` 中，对 `element` 进行广播计算。
    返回一个布尔数组，形状与 `element` 相同，其中对应的元素为 True 表示在 `test_elements` 中，否则为 False。

    Parameters
    ----------
    element : array_like
        输入数组。
    test_elements : array_like
        用于测试每个 `element` 值的目标值。
        如果是数组或类数组，会被展平处理。参见非类数组参数的行为说明。
    assume_unique : bool, optional
        如果为 True，则假设输入数组都是唯一的，这可以加速计算。默认为 False。
    invert : bool, optional
        如果为 True，则返回的数组值取反，相当于计算 `element` 不在 `test_elements` 中的情况。默认为 False。
        `np.isin(a, b, invert=True)` 等效于（但比）`np.invert(np.isin(a, b))` 更快。
    kind : {None, 'sort', 'table'}, optional
        使用的算法类型。这不会影响最终结果，但会影响速度和内存使用。默认为 None，会根据内存考虑自动选择。

        * 如果为 'sort'，将使用基于归并排序的方法。这将大致使用内存量为 `element` 和 `test_elements` 大小总和的 6 倍，不考虑 dtype 的大小。
        * 如果为 'table'，将使用类似计数排序的查找表方法。仅适用于布尔和整数数组。这将使用的内存量为 `element` 大小加上 `test_elements` 的最大最小值之差。
          当使用 'table' 选项时，`assume_unique` 参数不起作用。
        * 如果为 None，会自动选择 'table'，如果需要的内存分配小于等于 `element` 和 `test_elements` 大小总和的 6 倍，则选择 'sort'。
          这样做是为了避免默认使用大量内存，尽管在大多数情况下 'table' 可能更快。如果选择 'table'，`assume_unique` 参数将不起作用。

    Returns
    -------
    isin : ndarray, bool
        形状与 `element` 相同的数组。`element[isin]` 的值属于 `test_elements`。

    Notes
    -----
    `isin` 是 `in` Python 关键字的逐元素函数版本。
    如果 `a` 和 `b` 是 1-D 序列，则 `isin(a, b)` 大致相当于 `np.array([item in b for item in a])`。

    `element` 和 `test_elements` 如果不是数组，会被转换为数组。

    """
    """
    Compare elements of `element` with `test_elements` and return a boolean array
    indicating whether each element of `element` is contained in `test_elements`.

    Parameters
    ----------
    element : array_like
        Input array to be tested against `test_elements`.
    test_elements : array_like or set
        The set of elements to test against.
    assume_unique : bool, optional
        If True, assumes `test_elements` are unique. Default is False.
    invert : bool, optional
        If True, the boolean array is inverted, i.e., `True` for elements not in
        `test_elements`. Default is False.
    kind : {'auto', 'sort', 'table'}, optional
        Method used to perform the comparison:
        - 'auto': Default mode, chooses between 'sort' or 'table' based on memory usage.
        - 'sort': Uses sorting algorithm for comparison.
        - 'table': Uses hash table for comparison, potentially faster for large inputs.
    
    Returns
    -------
    ndarray
        Boolean array of the same shape as `element`, indicating if each element
        is in `test_elements`.

    Notes
    -----
    If `test_elements` is a set (or other non-sequence collection), it will be
    converted to an object array with one element, rather than an array of the
    values contained in `test_elements`. This is a consequence of the `array`
    constructor's way of handling non-sequence collections. Converting the set
    to a list usually gives the desired behavior.

    Using ``kind='table'`` tends to be faster than `kind='sort'` if the following
    relationship is true:
    ``log10(len(test_elements)) >
    (log10(max(test_elements)-min(test_elements)) - 2.27) / 0.927``,
    but may use greater memory. The default value for `kind` will be automatically
    selected based only on memory usage, so one may manually set ``kind='table'``
    if memory constraints can be relaxed.

    Examples
    --------
    >>> element = 2*np.arange(4).reshape((2, 2))
    >>> element
    array([[0, 2],
           [4, 6]])
    >>> test_elements = [1, 2, 4, 8]
    >>> mask = np.isin(element, test_elements)
    >>> mask
    array([[False,  True],
           [ True, False]])
    >>> element[mask]
    array([2, 4])

    The indices of the matched values can be obtained with `nonzero`:

    >>> np.nonzero(mask)
    (array([0, 1]), array([1, 0]))

    The test can also be inverted:

    >>> mask = np.isin(element, test_elements, invert=True)
    >>> mask
    array([[ True, False],
           [False,  True]])
    >>> element[mask]
    array([0, 6])

    Because of how `array` handles sets, the following does not work as expected:

    >>> test_set = {1, 2, 4, 8}
    >>> np.isin(element, test_set)
    array([[False, False],
           [False, False]])

    Casting the set to a list gives the expected result:

    >>> np.isin(element, list(test_set))
    array([[False,  True],
           [ True, False]])
    """
    element = np.asarray(element)
    return _in1d(element, test_elements, assume_unique=assume_unique,
                 invert=invert, kind=kind).reshape(element.shape)
# 定义一个分派函数，用于确定 union1d 函数的参数类型
def _union1d_dispatcher(ar1, ar2):
    # 直接返回传入的两个数组，供后续处理使用
    return (ar1, ar2)


# 使用装饰器 array_function_dispatch 对 union1d 函数进行装饰，以实现多态分发
@array_function_dispatch(_union1d_dispatcher)
def union1d(ar1, ar2):
    """
    Find the union of two arrays.

    Return the unique, sorted array of values that are in either of the two
    input arrays.

    Parameters
    ----------
    ar1, ar2 : array_like
        Input arrays. They are flattened if they are not already 1D.

    Returns
    -------
    union1d : ndarray
        Unique, sorted union of the input arrays.

    Examples
    --------
    >>> np.union1d([-1, 0, 1], [-2, 0, 2])
    array([-2, -1,  0,  1,  2])

    To find the union of more than two arrays, use functools.reduce:

    >>> from functools import reduce
    >>> reduce(np.union1d, ([1, 3, 4, 3], [3, 1, 2, 1], [6, 3, 4, 2]))
    array([1, 2, 3, 4, 6])
    """
    # 调用 numpy 的 unique 函数，返回连接后的唯一值数组
    return unique(np.concatenate((ar1, ar2), axis=None))


# 定义一个分派函数，用于确定 setdiff1d 函数的参数类型
def _setdiff1d_dispatcher(ar1, ar2, assume_unique=None):
    # 直接返回传入的两个数组，供后续处理使用
    return (ar1, ar2)


# 使用装饰器 array_function_dispatch 对 setdiff1d 函数进行装饰，以实现多态分发
@array_function_dispatch(_setdiff1d_dispatcher)
def setdiff1d(ar1, ar2, assume_unique=False):
    """
    Find the set difference of two arrays.

    Return the unique values in `ar1` that are not in `ar2`.

    Parameters
    ----------
    ar1 : array_like
        Input array.
    ar2 : array_like
        Input comparison array.
    assume_unique : bool
        If True, the input arrays are both assumed to be unique, which
        can speed up the calculation.  Default is False.

    Returns
    -------
    setdiff1d : ndarray
        1D array of values in `ar1` that are not in `ar2`. The result
        is sorted when `assume_unique=False`, but otherwise only sorted
        if the input is sorted.

    Examples
    --------
    >>> a = np.array([1, 2, 3, 2, 4, 1])
    >>> b = np.array([3, 4, 5, 6])
    >>> np.setdiff1d(a, b)
    array([1, 2])

    """
    # 根据 assume_unique 参数决定是否对输入数组调用 unique 函数进行处理
    if assume_unique:
        ar1 = np.asarray(ar1).ravel()
    else:
        ar1 = unique(ar1)
        ar2 = unique(ar2)
    # 使用 _in1d 函数找出 ar1 中在 ar2 中不存在的元素，并返回
    return ar1[_in1d(ar1, ar2, assume_unique=True, invert=True)]
```