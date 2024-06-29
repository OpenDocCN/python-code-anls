# `.\numpy\numpy\_core\numeric.py`

```
# 导入 functools 模块，用于支持创建偏函数
import functools
# 导入 itertools 模块，用于高效的迭代器循环
import itertools
# 导入 operator 模块，用于函数操作符的函数
import operator
# 导入 sys 模块，提供对解释器相关的功能访问
import sys
# 导入 warnings 模块，用于警告控制
import warnings
# 导入 numbers 模块，提供数值抽象基类
import numbers
# 导入 builtins 模块，提供内置函数的访问
import builtins
# 导入 math 模块，提供基本的数学函数
import math

# 导入 numpy 库，并重命名为 np
import numpy as np
# 从当前包中导入 multiarray 模块
from . import multiarray
# 从当前包中导入 numerictypes 模块，并重命名为 nt
from . import numerictypes as nt
# 从 multiarray 模块中导入指定的函数和常量
from .multiarray import (
    ALLOW_THREADS, BUFSIZE, CLIP, MAXDIMS, MAY_SHARE_BOUNDS, MAY_SHARE_EXACT,
    RAISE, WRAP, arange, array, asarray, asanyarray, ascontiguousarray,
    asfortranarray, broadcast, can_cast, concatenate, copyto, dot, dtype,
    empty, empty_like, flatiter, frombuffer, from_dlpack, fromfile, fromiter,
    fromstring, inner, lexsort, matmul, may_share_memory, min_scalar_type,
    ndarray, nditer, nested_iters, promote_types, putmask, result_type,
    shares_memory, vdot, where, zeros, normalize_axis_index,
    _get_promotion_state, _set_promotion_state, vecdot
)
# 从当前包中导入 overrides 模块
from . import overrides
# 从当前包中导入 umath 模块
from . import umath
# 从当前包中导入 shape_base 模块
from . import shape_base
# 从 overrides 模块中导入指定的函数
from .overrides import set_array_function_like_doc, set_module
# 从 umath 模块中导入指定的函数和常量
from .umath import (multiply, invert, sin, PINF, NAN)
# 从 numerictypes 模块中导入全部内容
from . import numerictypes
# 从 ..exceptions 包中导入 AxisError 异常类
from ..exceptions import AxisError
# 从当前包中导入 _ufunc_config 模块，导入 errstate 和 _no_nep50_warning 两个对象
from ._ufunc_config import errstate, _no_nep50_warning

# 将 invert 函数重命名为 bitwise_not
bitwise_not = invert
# 获取 sin 函数的类型，并将结果赋值给 ufunc
ufunc = type(sin)
# 将 newaxis 设置为 None
newaxis = None

# 使用 functools.partial 创建 array_function_dispatch 函数，其中 module 参数设置为 'numpy'
array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')

# __all__ 列表定义了模块中所有需要导出的公共接口
__all__ = [
    'newaxis', 'ndarray', 'flatiter', 'nditer', 'nested_iters', 'ufunc',
    'arange', 'array', 'asarray', 'asanyarray', 'ascontiguousarray',
    'asfortranarray', 'zeros', 'count_nonzero', 'empty', 'broadcast', 'dtype',
    'fromstring', 'fromfile', 'frombuffer', 'from_dlpack', 'where',
    'argwhere', 'copyto', 'concatenate', 'lexsort', 'astype',
    'can_cast', 'promote_types', 'min_scalar_type',
    'result_type', 'isfortran', 'empty_like', 'zeros_like', 'ones_like',
    'correlate', 'convolve', 'inner', 'dot', 'outer', 'vdot', 'roll',
    'rollaxis', 'moveaxis', 'cross', 'tensordot', 'little_endian',
    'fromiter', 'array_equal', 'array_equiv', 'indices', 'fromfunction',
    'isclose', 'isscalar', 'binary_repr', 'base_repr', 'ones',
    'identity', 'allclose', 'putmask',
    'flatnonzero', 'inf', 'nan', 'False_', 'True_', 'bitwise_not',
    'full', 'full_like', 'matmul', 'vecdot', 'shares_memory',
    'may_share_memory', '_get_promotion_state', '_set_promotion_state']

# 定义 _zeros_like_dispatcher 函数，用于 zeros_like 函数的分派器
def _zeros_like_dispatcher(
    a, dtype=None, order=None, subok=None, shape=None, *, device=None
):
    # 返回参数 a 的元组形式
    return (a,)

# 使用 array_function_dispatch 装饰器注册 _zeros_like_dispatcher 函数
@array_function_dispatch(_zeros_like_dispatcher)
# 定义 zeros_like 函数，返回一个与给定数组 a 相同形状和类型的全零数组
def zeros_like(
    a, dtype=None, order='K', subok=True, shape=None, *, device=None
):
    """
    Return an array of zeros with the same shape and type as a given array.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define these same attributes of
        the returned array.
    dtype : data-type, optional
        Overrides the data type of the result.

        .. versionadded:: 1.6.0
    """
    # 函数文档字符串，描述了函数的功能和参数信息
    res = empty_like(
        a, dtype=dtype, order=order, subok=subok, shape=shape, device=device
    )
    # 使用 `empty_like` 函数创建一个与输入数组 `a` 相同形状和类型的全零数组 `res`。
    # 可选参数 `dtype` 指定数组的数据类型，`order` 指定数组的存储顺序，`subok` 指定是否使用子类类型。
    # `shape` 参数可以覆盖结果数组的形状，如果 `order='K'` 且维度数不变，则尝试保持存储顺序，否则默认为 'C'。
    # `device` 参数指定创建数组的设备，仅用于 Array-API 兼容性，如果传递了此参数，必须为 `"cpu"`。

    # needed instead of a 0 to get same result as zeros for string dtypes
    # 为了在处理字符串类型数据时获得与 `zeros` 函数相同的结果，需要使用 `zeros(1, dtype=res.dtype)`。
    z = zeros(1, dtype=res.dtype)
    # 创建一个与 `res` 相同数据类型的长度为 1 的全零数组 `z`。

    multiarray.copyto(res, z, casting='unsafe')
    # 将 `z` 的值复制到 `res`，允许不安全类型转换。

    return res
    # 返回结果数组 `res`。
# 将该函数标记为数组函数，以便于文档生成器生成相应的文档
@set_array_function_like_doc
# 将该函数的模块标记为 'numpy'，这样文档生成器可以正确归类和组织文档
@set_module('numpy')
# 定义一个创建指定形状和类型、填充为全1的新数组的函数
def ones(shape, dtype=None, order='C', *, device=None, like=None):
    """
    Return a new array of given shape and type, filled with ones.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        The desired data-type for the array, e.g., `numpy.int8`.  Default is
        `numpy.float64`.
    order : {'C', 'F'}, optional, default: C
        Whether to store multi-dimensional data in row-major
        (C-style) or column-major (Fortran-style) order in
        memory.
    device : str, optional
        The device on which to place the created array. Default: None.
        For Array-API interoperability only, so must be ``"cpu"`` if passed.

        .. versionadded:: 2.0.0
    ${ARRAY_FUNCTION_LIKE}

        .. versionadded:: 1.20.0

    Returns
    -------
    out : ndarray
        Array of ones with the given shape, dtype, and order.

    See Also
    --------
    ones_like : Return an array of ones with shape and type of input.
    empty : Return a new uninitialized array.
    zeros : Return a new array setting values to zero.
    full : Return a new array of given shape filled with value.

    Examples
    --------
    >>> np.ones(5)
    array([1., 1., 1., 1., 1.])

    >>> np.ones((5,), dtype=int)
    array([1, 1, 1, 1, 1])

    >>> np.ones((2, 1))
    array([[1.],
           [1.]])

    >>> s = (2,2)
    >>> np.ones(s)
    array([[1.,  1.],
           [1.,  1.]])

    """
    # 如果传入了 like 参数，则调用 _ones_with_like 函数，创建与 like 参数形状、类型一致的全1数组
    if like is not None:
        return _ones_with_like(
            like, shape, dtype=dtype, order=order, device=device
        )

    # 否则，创建一个空的数组 a，形状和类型由 shape 和 dtype 决定，顺序由 order 决定
    a = empty(shape, dtype, order, device=device)
    # 将数组 a 中所有元素的值设置为 1，这里使用 'unsafe' 模式进行强制类型转换
    multiarray.copyto(a, 1, casting='unsafe')
    # 返回填充为全1的数组 a
    return a


# 将 _ones_with_like 函数与 ones 函数关联，使得 _ones_with_like 能够根据数组函数分派机制被调用
_ones_with_like = array_function_dispatch()(ones)


# 定义 _ones_like_dispatcher 函数，用于根据参数分派给 ones_like 函数
def _ones_like_dispatcher(
    a, dtype=None, order=None, subok=None, shape=None, *, device=None
):
    return (a,)


# 使用 _ones_like_dispatcher 函数注册 ones_like 函数，使其能够根据数组函数分派机制被调用
@array_function_dispatch(_ones_like_dispatcher)
# 定义 ones_like 函数，返回一个与给定数组 a 形状和类型相同的全1数组
def ones_like(
    a, dtype=None, order='K', subok=True, shape=None, *, device=None
):
    """
    Return an array of ones with the same shape and type as a given array.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define these same attributes of
        the returned array.
    dtype : data-type, optional
        Overrides the data type of the result.

        .. versionadded:: 1.6.0
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result. 'C' means C-order,
        'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
        'C' otherwise. 'K' means match the layout of `a` as closely
        as possible.

        .. versionadded:: 1.6.0
    subok : bool, optional.
        If True, then the newly created array will use the sub-class
        type of `a`, otherwise it will be a base-class array. Defaults
        to True.
    """
    pass
    # shape参数用于指定返回数组的形状，可以是整数或整数序列，可选项。
    # 如果指定了shape，则覆盖结果数组的形状。如果order='K'并且维度未改变，则尝试保持顺序；否则，隐含使用order='C'。
    # device参数用于指定创建数组时使用的设备，默认为None。仅用于Array-API互操作性，如果传递了值，必须为"cpu"。
    # 返回值
    # -------
    # out : ndarray
    #     与输入数组`a`具有相同形状和类型的全1数组。
    #
    # 参见
    # --------
    # empty_like : 返回一个形状和类型与输入相同的空数组。
    # zeros_like : 返回一个形状和类型与输入相同的全0数组。
    # full_like : 返回一个形状与输入相同且填充了特定值的新数组。
    # ones : 返回一个设置值为1的新数组。
    #
    # 示例
    # --------
    # >>> x = np.arange(6)
    # >>> x = x.reshape((2, 3))
    # >>> x
    # array([[0, 1, 2],
    #        [3, 4, 5]])
    # >>> np.ones_like(x)
    # array([[1, 1, 1],
    #        [1, 1, 1]])
    #
    # >>> y = np.arange(3, dtype=float)
    # >>> y
    # array([0., 1., 2.])
    # >>> np.ones_like(y)
    # array([1.,  1.,  1.])
    """
    res = empty_like(
        a, dtype=dtype, order=order, subok=subok, shape=shape, device=device
    )
    # 将值1复制到res数组中，使用'unsafe'转换方式
    multiarray.copyto(res, 1, casting='unsafe')
    # 返回生成的数组res
    return res
# 定义一个函数 `_full_dispatcher`，用于创建包含指定元素的数组，返回值是一个元组
def _full_dispatcher(
    shape, fill_value, dtype=None, order=None, *, device=None, like=None
):
    return(like,)


# 修饰器，设置函数的行为类似于某个函数文档
@set_array_function_like_doc
# 设置函数所属的模块为 'numpy'
@set_module('numpy')
# 定义函数 `full`，返回一个指定形状和类型的新数组，用指定值填充
def full(shape, fill_value, dtype=None, order='C', *, device=None, like=None):
    """
    Return a new array of given shape and type, filled with `fill_value`.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    fill_value : scalar or array_like
        Fill value.
    dtype : data-type, optional
        The desired data-type for the array  The default, None, means
         ``np.array(fill_value).dtype``.
    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- or Fortran-contiguous
        (row- or column-wise) order in memory.
    device : str, optional
        The device on which to place the created array. Default: None.
        For Array-API interoperability only, so must be ``"cpu"`` if passed.

        .. versionadded:: 2.0.0
    ${ARRAY_FUNCTION_LIKE}

        .. versionadded:: 1.20.0

    Returns
    -------
    out : ndarray
        Array of `fill_value` with the given shape, dtype, and order.

    See Also
    --------
    full_like : Return a new array with shape of input filled with value.
    empty : Return a new uninitialized array.
    ones : Return a new array setting values to one.
    zeros : Return a new array setting values to zero.

    Examples
    --------
    >>> np.full((2, 2), np.inf)
    array([[inf, inf],
           [inf, inf]])
    >>> np.full((2, 2), 10)
    array([[10, 10],
           [10, 10]])

    >>> np.full((2, 2), [1, 2])
    array([[1, 2],
           [1, 2]])

    """
    # 如果指定了 `like` 参数，则返回根据 `like` 的形状和类型创建的新数组
    if like is not None:
        return _full_with_like(
            like, shape, fill_value, dtype=dtype, order=order, device=device
        )

    # 如果未指定 `dtype`，将 `fill_value` 转换成数组并获取其数据类型
    if dtype is None:
        fill_value = asarray(fill_value)
        dtype = fill_value.dtype
    # 创建一个未初始化的数组 `a`，指定形状、数据类型、存储顺序和设备
    a = empty(shape, dtype, order, device=device)
    # 将 `fill_value` 的值复制到数组 `a` 中
    multiarray.copyto(a, fill_value, casting='unsafe')
    return a


# 使用函数分发装饰器注册 `full` 函数
_full_with_like = array_function_dispatch()(full)


# 定义一个函数 `_full_like_dispatcher`，用于返回一个元组，指定其参数 `a`
def _full_like_dispatcher(
    a, fill_value, dtype=None, order=None, subok=None, shape=None,
    *, device=None
):
    return (a,)


# 使用函数分发装饰器注册 `full_like` 函数，并指定 `_full_like_dispatcher` 作为分发函数
@array_function_dispatch(_full_like_dispatcher)
# 定义函数 `full_like`，返回一个与给定数组 `a` 相同形状和类型的新数组，用指定值填充
def full_like(
    a, fill_value, dtype=None, order='K', subok=True, shape=None,
    *, device=None
):
    """
    Return a full array with the same shape and type as a given array.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define these same attributes of
        the returned array.
    fill_value : array_like
        Fill value.
    dtype : data-type, optional
        Overrides the data type of the result.

    """
    # 创建一个与输入数组 `a` 相同形状和类型的新数组，使用指定的填充值 `fill_value`
    res = empty_like(
        a, dtype=dtype, order=order, subok=subok, shape=shape, device=device
    )
    # 将 `fill_value` 的值复制到新创建的数组 `res` 中，使用不安全的类型转换
    multiarray.copyto(res, fill_value, casting='unsafe')
    # 返回填充后的新数组 `res`
    return res
# 定义一个分发器函数，用于向 `count_nonzero` 函数分发参数 `a`
def _count_nonzero_dispatcher(a, axis=None, *, keepdims=None):
    # 返回一个元组，包含参数 `a`，其他参数为默认值
    return (a,)


# 使用装饰器 `array_function_dispatch` 装饰的 `count_nonzero` 函数
@array_function_dispatch(_count_nonzero_dispatcher)
def count_nonzero(a, axis=None, *, keepdims=False):
    """
    Counts the number of non-zero values in the array `a`.

    The word "non-zero" is in reference to the Python 2.x
    built-in method `__nonzero__()` (renamed `__bool__()`
    in Python 3.x) of Python objects that tests an object's
    "truthfulness". For example, any number is considered
    truthful if it is nonzero, whereas any string is considered
    truthful if it is not the empty string. Thus, this function
    (recursively) counts how many elements in `a` (and in
    sub-arrays thereof) have their `__nonzero__()` or `__bool__()`
    method evaluated to `True`.

    Parameters
    ----------
    a : array_like
        The array for which to count non-zeros.
    axis : int or tuple, optional
        Axis or tuple of axes along which to count non-zeros.
        Default is None, meaning that non-zeros will be counted
        along a flattened version of `a`.

        .. versionadded:: 1.12.0

    keepdims : bool, optional
        If this is set to True, the axes that are counted are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        .. versionadded:: 1.19.0

    Returns
    -------
    count : int or array of int
        Number of non-zero values in the array along a given axis.
        Otherwise, the total number of non-zero values in the array
        is returned.

    See Also
    --------
    nonzero : Return the coordinates of all the non-zero values.

    Examples
    --------
    >>> np.count_nonzero(np.eye(4))
    4
    >>> a = np.array([[0, 1, 7, 0],
    ...               [3, 0, 2, 19]])
    >>> np.count_nonzero(a)
    5
    >>> np.count_nonzero(a, axis=0)
    array([1, 1, 2, 1])
    >>> np.count_nonzero(a, axis=1)
    array([2, 3])
    >>> np.count_nonzero(a, axis=1, keepdims=True)
    array([[2],
           [3]])
    """
    # 如果 `axis` 为 None 并且 `keepdims` 为 False，则调用 `multiarray.count_nonzero` 函数
    if axis is None and not keepdims:
        return multiarray.count_nonzero(a)

    # 将 `a` 转换为 `ndarray` 类型
    a = asanyarray(a)

    # TODO: this works around .astype(bool) not working properly (gh-9847)
    # 如果 `a` 的数据类型为字符类型，则创建一个布尔类型的数组 `a_bool`
    if np.issubdtype(a.dtype, np.character):
        a_bool = a != a.dtype.type()
    else:
        # 否则，将 `a` 转换为布尔类型的数组 `a_bool`
        a_bool = a.astype(np.bool, copy=False)

    # 返回沿指定轴（如果有）求和后的布尔类型数组 `a_bool`，结果的数据类型为 `np.intp`
    return a_bool.sum(axis=axis, dtype=np.intp, keepdims=keepdims)


# 使用装饰器 `set_module` 装饰的 `isfortran` 函数
@set_module('numpy')
def isfortran(a):
    """
    Check if the array is Fortran contiguous but *not* C contiguous.

    This function is obsolete. If you only want to check if an array is Fortran
    contiguous use `a.flags.f_contiguous` instead.

    Parameters
    ----------
    a : ndarray
        Input array.

    Returns
    -------
    isfortran : bool
        Returns True if the array is Fortran contiguous but *not* C contiguous.


    Examples
    --------
    np.array allows to specify whether the array is written in C-contiguous
    order (last index varies the fastest), or FORTRAN-contiguous order in
    memory (first index varies the fastest).

    >>> a = np.array([[1, 2, 3], [4, 5, 6]], order='C')
    >>> a
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> np.isfortran(a)
    False

    >>> b = np.array([[1, 2, 3], [4, 5, 6]], order='F')
    >>> b
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> np.isfortran(b)
    True


    The transpose of a C-ordered array is a FORTRAN-ordered array.

    >>> a = np.array([[1, 2, 3], [4, 5, 6]], order='C')
    >>> a
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> np.isfortran(a)
    False
    >>> b = a.T
    >>> b
    array([[1, 4],
           [2, 5],
           [3, 6]])
    >>> np.isfortran(b)
    True

    C-ordered arrays evaluate as False even if they are also FORTRAN-ordered.

    >>> np.isfortran(np.array([1, 2], order='F'))
    False

    """
    # 返回数组 a 的 flags 属性中的 fnc 属性值
    return a.flags.fnc
# 定义一个函数分派器，用于返回输入参数元组
def _argwhere_dispatcher(a):
    return (a,)

# 使用装饰器将_dispatcher函数与argwhere函数关联，用于根据输入类型调度函数
@array_function_dispatch(_argwhere_dispatcher)
def argwhere(a):
    """
    Find the indices of array elements that are non-zero, grouped by element.

    Parameters
    ----------
    a : array_like
        Input data.

    Returns
    -------
    index_array : (N, a.ndim) ndarray
        Indices of elements that are non-zero. Indices are grouped by element.
        This array will have shape ``(N, a.ndim)`` where ``N`` is the number of
        non-zero items.

    See Also
    --------
    where, nonzero

    Notes
    -----
    ``np.argwhere(a)`` is almost the same as ``np.transpose(np.nonzero(a))``,
    but produces a result of the correct shape for a 0D array.

    The output of ``argwhere`` is not suitable for indexing arrays.
    For this purpose use ``nonzero(a)`` instead.

    Examples
    --------
    >>> x = np.arange(6).reshape(2,3)
    >>> x
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.argwhere(x>1)
    array([[0, 2],
           [1, 0],
           [1, 1],
           [1, 2]])

    """
    # 如果输入数组a的维度为0，则提升为至少1维数组
    if np.ndim(a) == 0:
        a = shape_base.atleast_1d(a)
        # 然后去除添加的维度
        return argwhere(a)[:, :0]
    # 返回数组a中非零元素的索引数组，索引按元素分组
    return transpose(nonzero(a))


# 定义一个函数分派器，用于返回输入参数元组
def _flatnonzero_dispatcher(a):
    return (a,)

# 使用装饰器将_dispatcher函数与flatnonzero函数关联，用于根据输入类型调度函数
@array_function_dispatch(_flatnonzero_dispatcher)
def flatnonzero(a):
    """
    Return indices that are non-zero in the flattened version of a.

    This is equivalent to ``np.nonzero(np.ravel(a))[0]``.

    Parameters
    ----------
    a : array_like
        Input data.

    Returns
    -------
    res : ndarray
        Output array, containing the indices of the elements of ``a.ravel()``
        that are non-zero.

    See Also
    --------
    nonzero : Return the indices of the non-zero elements of the input array.
    ravel : Return a 1-D array containing the elements of the input array.

    Examples
    --------
    >>> x = np.arange(-2, 3)
    >>> x
    array([-2, -1,  0,  1,  2])
    >>> np.flatnonzero(x)
    array([0, 1, 3, 4])

    Use the indices of the non-zero elements as an index array to extract
    these elements:

    >>> x.ravel()[np.flatnonzero(x)]
    array([-2, -1,  1,  2])

    """
    # 返回数组a的扁平化版本中非零元素的索引数组
    return np.nonzero(np.ravel(a))[0]


# 定义一个函数分派器，用于返回输入参数元组
def _correlate_dispatcher(a, v, mode=None):
    return (a, v)

# 使用装饰器将_dispatcher函数与correlate函数关联，用于根据输入类型调度函数
@array_function_dispatch(_correlate_dispatcher)
def correlate(a, v, mode='valid'):
    r"""
    Cross-correlation of two 1-dimensional sequences.

    This function computes the correlation as generally defined in signal
    processing texts [1]_:

    .. math:: c_k = \sum_n a_{n+k} \cdot \overline{v}_n

    with a and v sequences being zero-padded where necessary and
    :math:`\overline v` denoting complex conjugation.

    Parameters
    ----------
    a, v : array_like
        Input sequences.
    mode : {'valid', 'same', 'full'}, optional
        # 模式参数，指定交叉相关计算的模式，可以是'valid'、'same'、'full'
        Refer to the `convolve` docstring.  Note that the default
        # 参考`convolve`函数的文档字符串，注意默认值为'valid'，而不是`convolve`函数的'full'
        is 'valid', unlike `convolve`, which uses 'full'.

    Returns
    -------
    out : ndarray
        # 返回值为 ndarray 类型，表示输入数组 a 和 v 的离散交叉相关结果
        Discrete cross-correlation of `a` and `v`.

    See Also
    --------
    convolve : Discrete, linear convolution of two one-dimensional sequences.
    scipy.signal.correlate : uses FFT which has superior performance
        on large arrays.
        # 参见函数 convolve：计算两个一维序列的离散线性卷积。
        # scipy.signal.correlate 使用 FFT 在大数组上有更优越的性能表现。

    Notes
    -----
    The definition of correlation above is not unique and sometimes
    correlation may be defined differently. Another common definition is [1]_:

    .. math:: c'_k = \sum_n a_{n} \cdot \overline{v_{n+k}}

    which is related to :math:`c_k` by :math:`c'_k = c_{-k}`.
        # 上述的相关定义不是唯一的，有时可能会有不同的定义。另一个常见的定义如 [1] 所示：
        # 这与 c_k 的关系由 c'_k = c_{-k} 给出。

    `numpy.correlate` may perform slowly in large arrays (i.e. n = 1e5)
    because it does not use the FFT to compute the convolution; in that case,
    `scipy.signal.correlate` might be preferable.
        # 在大数组（例如 n = 1e5）上，`numpy.correlate` 的性能可能较慢，
        # 因为它不使用 FFT 来计算卷积；在这种情况下，可能更倾向于使用 `scipy.signal.correlate`。

    References
    ----------
    .. [1] Wikipedia, "Cross-correlation",
           https://en.wikipedia.org/wiki/Cross-correlation
        # 参考文献 [1]：Wikipedia，"Cross-correlation"，链接地址为 https://en.wikipedia.org/wiki/Cross-correlation

    Examples
    --------
    >>> np.correlate([1, 2, 3], [0, 1, 0.5])
    array([3.5])
    >>> np.correlate([1, 2, 3], [0, 1, 0.5], "same")
    array([2. ,  3.5,  3. ])
    >>> np.correlate([1, 2, 3], [0, 1, 0.5], "full")
    array([0.5,  2. ,  3.5,  3. ,  0. ])
        # 使用示例：计算不同模式下的离散交叉相关结果

    Using complex sequences:

    >>> np.correlate([1+1j, 2, 3-1j], [0, 1, 0.5j], 'full')
    array([ 0.5-0.5j,  1.0+0.j ,  1.5-1.5j,  3.0-1.j ,  0.0+0.j ])
        # 使用复数序列的示例：计算复数序列的离散交叉相关结果

    Note that you get the time reversed, complex conjugated result
    (:math:`\overline{c_{-k}}`) when the two input sequences a and v change
    places:

    >>> np.correlate([0, 1, 0.5j], [1+1j, 2, 3-1j], 'full')
    array([ 0.0+0.j ,  3.0+1.j ,  1.5+1.5j,  1.0+0.j ,  0.5+0.5j])
        # 注意：当输入序列 a 和 v 位置交换时，得到的结果是时间反转、复共轭的结果 `c_{-k}`

    """
    return multiarray.correlate2(a, v, mode)
        # 调用 multiarray 模块的 correlate2 函数，计算输入数组 a 和 v 的离散交叉相关结果
# 定义一个调度函数，用于派发输入给卷积函数
def _convolve_dispatcher(a, v, mode=None):
    # 返回输入参数 a 和 v 的元组
    return (a, v)


# 使用装饰器 array_function_dispatch 将 _convolve_dispatcher 函数与 convolve 函数关联起来
@array_function_dispatch(_convolve_dispatcher)
def convolve(a, v, mode='full'):
    """
    Returns the discrete, linear convolution of two one-dimensional sequences.

    The convolution operator is often seen in signal processing, where it
    models the effect of a linear time-invariant system on a signal [1]_.  In
    probability theory, the sum of two independent random variables is
    distributed according to the convolution of their individual
    distributions.

    If `v` is longer than `a`, the arrays are swapped before computation.

    Parameters
    ----------
    a : (N,) array_like
        First one-dimensional input array.
    v : (M,) array_like
        Second one-dimensional input array.
    mode : {'full', 'valid', 'same'}, optional
        'full':
          By default, mode is 'full'.  This returns the convolution
          at each point of overlap, with an output shape of (N+M-1,). At
          the end-points of the convolution, the signals do not overlap
          completely, and boundary effects may be seen.

        'same':
          Mode 'same' returns output of length ``max(M, N)``.  Boundary
          effects are still visible.

        'valid':
          Mode 'valid' returns output of length
          ``max(M, N) - min(M, N) + 1``.  The convolution product is only given
          for points where the signals overlap completely.  Values outside
          the signal boundary have no effect.

    Returns
    -------
    out : ndarray
        Discrete, linear convolution of `a` and `v`.

    See Also
    --------
    scipy.signal.fftconvolve : Convolve two arrays using the Fast Fourier
                               Transform.
    scipy.linalg.toeplitz : Used to construct the convolution operator.
    polymul : Polynomial multiplication. Same output as convolve, but also
              accepts poly1d objects as input.

    Notes
    -----
    The discrete convolution operation is defined as

    .. math:: (a * v)_n = \\sum_{m = -\\infty}^{\\infty} a_m v_{n - m}

    It can be shown that a convolution :math:`x(t) * y(t)` in time/space
    is equivalent to the multiplication :math:`X(f) Y(f)` in the Fourier
    domain, after appropriate padding (padding is necessary to prevent
    circular convolution).  Since multiplication is more efficient (faster)
    than convolution, the function `scipy.signal.fftconvolve` exploits the
    FFT to calculate the convolution of large data-sets.

    References
    ----------
    .. [1] Wikipedia, "Convolution",
        https://en.wikipedia.org/wiki/Convolution

    Examples
    --------
    Note how the convolution operator flips the second array
    before "sliding" the two across one another:

    >>> np.convolve([1, 2, 3], [0, 1, 0.5])
    array([0. , 1. , 2.5, 4. , 1.5])

    Only return the middle values of the convolution.
    Contains boundary effects, where zeros are taken
    into account:
    """
    # 返回函数的主体部分，用于执行一维序列的线性离散卷积
    pass
    # 使用 NumPy 的卷积函数 convolve 对数组 a 和 v 进行卷积运算，并返回指定模式下的结果
    >>> np.convolve([1,2,3],[0,1,0.5], 'same')
    # 在 'same' 模式下，返回与输入数组相同长度的卷积结果
    array([1. ,  2.5,  4. ])

    # 由于两个数组长度相同，因此只有一个位置完全重叠：
    >>> np.convolve([1,2,3],[0,1,0.5], 'valid')
    # 在 'valid' 模式下，返回完全重叠部分的卷积结果
    array([2.5])

    """
    # 将输入的数组 a 和 v 转换为 NumPy 数组，并确保它们至少是一维的
    a, v = array(a, copy=None, ndmin=1), array(v, copy=None, ndmin=1)
    # 如果 v 的长度大于 a，则交换它们，确保 a 是长度较长的数组
    if (len(v) > len(a)):
        a, v = v, a
    # 如果 a 的长度为 0，则抛出值错误异常
    if len(a) == 0:
        raise ValueError('a cannot be empty')
    # 如果 v 的长度为 0，则抛出值错误异常
    if len(v) == 0:
        raise ValueError('v cannot be empty')
    # 调用 NumPy 的多维数组库的 correlate 函数，对 a 和 v 进行相关运算，使用反转的 v，并返回指定 mode 的结果
    return multiarray.correlate(a, v[::-1], mode)
# 创建一个元组，包含三个输入参数 a, b, out（可选）
def _outer_dispatcher(a, b, out=None):
    return (a, b, out)

# 使用 array_function_dispatch 装饰器注册 _outer_dispatcher 函数
@array_function_dispatch(_outer_dispatcher)
# 定义 outer 函数，计算两个向量的外积
def outer(a, b, out=None):
    """
    计算两个向量的外积。

    给定长度分别为 ``M`` 和 ``N`` 的两个向量 `a` 和 `b`，外积 [1]_ 定义为::

      [[a_0*b_0  a_0*b_1 ... a_0*b_{N-1} ]
       [a_1*b_0    .
       [ ...          .
       [a_{M-1}*b_0            a_{M-1}*b_{N-1} ]]

    Parameters
    ----------
    a : (M,) array_like
        第一个输入向量。如果不是已经是 1 维的，则会被扁平化。
    b : (N,) array_like
        第二个输入向量。如果不是已经是 1 维的，则会被扁平化。
    out : (M, N) ndarray, optional
        结果存储的位置

        .. versionadded:: 1.9.0

    Returns
    -------
    out : (M, N) ndarray
        ``out[i, j] = a[i] * b[j]``

    See also
    --------
    inner
    einsum : ``einsum('i,j->ij', a.ravel(), b.ravel())`` 的等价形式。
    ufunc.outer : 对于维度不限于 1D 和其他操作的泛化版本。``np.multiply.outer(a.ravel(), b.ravel())``
                  是其等价形式。
    linalg.outer : ``np.outer`` 的一个兼容 Array API 的变体，仅接受 1 维输入。
    tensordot : ``np.tensordot(a.ravel(), b.ravel(), axes=((), ()))``
                是其等价形式。

    References
    ----------
    .. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*, 3rd
           ed., Baltimore, MD, Johns Hopkins University Press, 1996,
           pg. 8.

    Examples
    --------
    创建一个 (*非常粗糙的*) 网格来计算 Mandelbrot 集合：

    >>> rl = np.outer(np.ones((5,)), np.linspace(-2, 2, 5))
    >>> rl
    array([[-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,  2.]])
    >>> im = np.outer(1j*np.linspace(2, -2, 5), np.ones((5,)))
    >>> im
    array([[0.+2.j, 0.+2.j, 0.+2.j, 0.+2.j, 0.+2.j],
           [0.+1.j, 0.+1.j, 0.+1.j, 0.+1.j, 0.+1.j],
           [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
           [0.-1.j, 0.-1.j, 0.-1.j, 0.-1.j, 0.-1.j],
           [0.-2.j, 0.-2.j, 0.-2.j, 0.-2.j, 0.-2.j]])
    >>> grid = rl + im
    >>> grid
    array([[-2.+2.j, -1.+2.j,  0.+2.j,  1.+2.j,  2.+2.j],
           [-2.+1.j, -1.+1.j,  0.+1.j,  1.+1.j,  2.+1.j],
           [-2.+0.j, -1.+0.j,  0.+0.j,  1.+0.j,  2.+0.j],
           [-2.-1.j, -1.-1.j,  0.-1.j,  1.-1.j,  2.-1.j],
           [-2.-2.j, -1.-2.j,  0.-2.j,  1.-2.j,  2.-2.j]])

    使用一个字母的 "向量" 的示例：

    >>> x = np.array(['a', 'b', 'c'], dtype=object)
    >>> np.outer(x, [1, 2, 3])
    array([['a', 'aa', 'aaa'],
           ['b', 'bb', 'bbb'],
           ['c', 'cc', 'ccc']], dtype=object)

    """
    # 将输入向量 a 和 b 转换为数组
    a = asarray(a)
    b = asarray(b)
    # 返回 a 和 b 的外积，结果存储在 out 中
    return multiply(a.ravel()[:, newaxis], b.ravel()[newaxis, :], out)
# 定义函数_tensordot_dispatcher，接收参数a, b, axes，并直接返回元组(a, b)
def _tensordot_dispatcher(a, b, axes=None):
    return (a, b)

# 使用array_function_dispatch装饰器将函数_tensordot_dispatcher注册为tensordot的分派函数
@array_function_dispatch(_tensordot_dispatcher)
# 定义函数tensordot，实现张量的点乘操作
def tensordot(a, b, axes=2):
    """
    Compute tensor dot product along specified axes.

    Given two tensors, `a` and `b`, and an array_like object containing
    two array_like objects, ``(a_axes, b_axes)``, sum the products of
    `a`'s and `b`'s elements (components) over the axes specified by
    ``a_axes`` and ``b_axes``. The third argument can be a single non-negative
    integer_like scalar, ``N``; if it is such, then the last ``N`` dimensions
    of `a` and the first ``N`` dimensions of `b` are summed over.

    Parameters
    ----------
    a, b : array_like
        Tensors to "dot".

    axes : int or (2,) array_like
        * integer_like
          If an int N, sum over the last N axes of `a` and the first N axes
          of `b` in order. The sizes of the corresponding axes must match.
        * (2,) array_like
          Or, a list of axes to be summed over, first sequence applying to `a`,
          second to `b`. Both elements array_like must be of the same length.

    Returns
    -------
    output : ndarray
        The tensor dot product of the input.

    See Also
    --------
    dot, einsum

    Notes
    -----
    Three common use cases are:

    * ``axes = 0`` : tensor product :math:`a\\otimes b`
    * ``axes = 1`` : tensor dot product :math:`a\\cdot b`
    * ``axes = 2`` : (default) tensor double contraction :math:`a:b`

    When `axes` is a positive integer ``N``, the operation starts with
    axis ``-N`` of `a` and axis ``0`` of `b`, and it continues through
    axis ``-1`` of `a` and axis ``N-1`` of `b` (inclusive).

    When there is more than one axis to sum over - and they are not the last
    (first) axes of `a` (`b`) - the argument `axes` should consist of
    two sequences of the same length, with the first axis to sum over given
    first in both sequences, the second axis second, and so forth.

    The shape of the result consists of the non-contracted axes of the
    first tensor, followed by the non-contracted axes of the second.

    Examples
    --------
    A "traditional" example:

    >>> a = np.arange(60.).reshape(3,4,5)
    >>> b = np.arange(24.).reshape(4,3,2)
    >>> c = np.tensordot(a,b, axes=([1,0],[0,1]))
    >>> c.shape
    (5, 2)
    >>> c
    array([[4400., 4730.],
           [4532., 4874.],
           [4664., 5018.],
           [4796., 5162.],
           [4928., 5306.]])
    >>> # A slower but equivalent way of computing the same...
    >>> d = np.zeros((5,2))
    >>> for i in range(5):
    ...   for j in range(2):
    ...     for k in range(3):
    ...       for n in range(4):
    ...         d[i,j] += a[k,n,i] * b[n,k,j]
    >>> c == d
    array([[ True,  True],
           [ True,  True],
           [ True,  True],
           [ True,  True],
           [ True,  True]])

    An extended example taking advantage of the overloading of + and \\*:

    >>> a = np.array(range(1, 9))
    """
    # 函数的具体实现在array_function_dispatch中定义的分派函数中完成，此处不作具体实现
    pass
    # 将数组 a 的形状改变为 (2, 2, 2)
    >>> a.shape = (2, 2, 2)
    # 创建一个包含对象类型数据 ('a', 'b', 'c', 'd') 的数组 A，将其形状改变为 (2, 2)
    >>> A = np.array(('a', 'b', 'c', 'd'), dtype=object)
    >>> A.shape = (2, 2)
    # 打印数组 a 和数组 A
    >>> a; A
    array([[[1, 2],
            [3, 4]],
           [[5, 6],
            [7, 8]]])
    array([['a', 'b'],
           ['c', 'd']], dtype=object)
    
    # 对数组 a 和数组 A 进行张量点乘，默认使用双重收缩的第三个参数为 2
    >>> np.tensordot(a, A)
    array(['abbcccdddd', 'aaaaabbbbbbcccccccdddddddd'], dtype=object)
    
    # 对数组 a 和数组 A 进行张量点乘，指定第三个参数为 1
    >>> np.tensordot(a, A, 1)
    array([[['acc', 'bdd'],
            ['aaacccc', 'bbbdddd']],
           [['aaaaacccccc', 'bbbbbdddddd'],
            ['aaaaaaacccccccc', 'bbbbbbbdddddddd']]], dtype=object)
    
    # 对数组 a 和数组 A 进行张量积，结果太长未包含在此处
    >>> np.tensordot(a, A, 0)
    array([[[[['a', 'b'],
              ['c', 'd']],
             ...
    
    # 对数组 a 和数组 A 进行张量点乘，指定收缩的轴为 (0, 1)
    >>> np.tensordot(a, A, (0, 1))
    array([[['abbbbb', 'cddddd'],
            ['aabbbbbb', 'ccdddddd']],
           [['aaabbbbbbb', 'cccddddddd'],
            ['aaaabbbbbbbb', 'ccccdddddddd']]], dtype=object)
    
    # 对数组 a 和数组 A 进行张量点乘，指定收缩的轴为 (2, 1)
    >>> np.tensordot(a, A, (2, 1))
    array([[['abb', 'cdd'],
            ['aaabbbb', 'cccdddd']],
           [['aaaaabbbbbb', 'cccccdddddd'],
            ['aaaaaaabbbbbbbb', 'cccccccdddddddd']]], dtype=object)
    
    # 对数组 a 和数组 A 进行张量点乘，同时指定多个收缩的轴为 ((0, 1), (0, 1))
    >>> np.tensordot(a, A, ((0, 1), (0, 1)))
    array(['abbbcccccddddddd', 'aabbbbccccccdddddddd'], dtype=object)
    
    # 对数组 a 和数组 A 进行张量点乘，同时指定多个收缩的轴为 ((2, 1), (1, 0))
    >>> np.tensordot(a, A, ((2, 1), (1, 0)))
    array(['acccbbdddd', 'aaaaacccccccbbbbbbdddddddd'], dtype=object)
    
    try:
        iter(axes)
    except Exception:
        # 如果 axes 不可迭代，说明是一个整数
        axes_a = list(range(-axes, 0))
        axes_b = list(range(0, axes))
    else:
        # 如果 axes 可迭代，分别赋值给 axes_a 和 axes_b
        axes_a, axes_b = axes
    
    try:
        # 尝试获取 axes_a 的长度，并转换为列表
        na = len(axes_a)
        axes_a = list(axes_a)
    except TypeError:
        # 如果无法获取长度，说明 axes_a 只包含一个整数，将其转为列表形式
        axes_a = [axes_a]
        na = 1
    
    try:
        # 尝试获取 axes_b 的长度，并转换为列表
        nb = len(axes_b)
        axes_b = list(axes_b)
    except TypeError:
        # 如果无法获取长度，说明 axes_b 只包含一个整数，将其转为列表形式
        axes_b = [axes_b]
        nb = 1
    
    # 将数组 a 和 b 转换为数组，并获取它们的形状和维度
    a, b = asarray(a), asarray(b)
    as_ = a.shape
    nda = a.ndim
    bs = b.shape
    ndb = b.ndim
    equal = True
    
    # 检查 axes_a 和 axes_b 的长度是否相等，如果不相等则抛出 ValueError
    if na != nb:
        equal = False
    else:
        # 遍历 axes_a 和 axes_b，检查对应轴的形状是否相等
        for k in range(na):
            if as_[axes_a[k]] != bs[axes_b[k]]:
                equal = False
                break
            # 处理负索引，将其转换为正索引
            if axes_a[k] < 0:
                axes_a[k] += nda
            if axes_b[k] < 0:
                axes_b[k] += ndb
    
    # 如果形状不匹配，则抛出 ValueError
    if not equal:
        raise ValueError("shape-mismatch for sum")
    
    # 将要进行收缩的轴移到数组 a 的末尾，移到数组 b 的前面
    notin = [k for k in range(nda) if k not in axes_a]
    newaxes_a = notin + axes_a
    # 计算新形状和旧形状
    N2 = math.prod(as_[axis] for axis in axes_a)
    newshape_a = (math.prod([as_[ax] for ax in notin]), N2)
    olda = [as_[axis] for axis in notin]
    
    # 将要进行收缩的轴移到数组 b 的前面，移到数组 b 的末尾
    notin = [k for k in range(ndb) if k not in axes_b]
    newaxes_b = axes_b + notin
    # 计算新形状和旧形状
    N2 = math.prod(bs[axis] for axis in axes_b)
    newshape_b = (N2, math.prod([bs[ax] for ax in notin]))
    oldb = [bs[axis] for axis in notin]
    
    # 对数组 a 根据 newaxes_a 进行转置，并重新调整形状为 newshape_a
    at = a.transpose(newaxes_a).reshape(newshape_a)
    # 将数组 b 按照 newaxes_b 指定的轴顺序进行转置，并按照 newshape_b 指定的形状进行重塑
    bt = b.transpose(newaxes_b).reshape(newshape_b)
    # 计算矩阵 a 和重塑后的矩阵 bt 的矩阵乘积
    res = dot(at, bt)
    # 将结果重新按照原来的形状 olda + oldb 进行重塑并返回
    return res.reshape(olda + oldb)
# 定义一个私有函数 _roll_dispatcher，用于分派数组滚动操作
def _roll_dispatcher(a, shift, axis=None):
    # 返回一个元组，包含参数 a
    return (a,)


# 使用装饰器 @array_function_dispatch 将 roll 函数与 _roll_dispatcher 关联起来，
# 用于数组滚动操作的分派
@array_function_dispatch(_roll_dispatcher)
def roll(a, shift, axis=None):
    """
    Roll array elements along a given axis.

    Elements that roll beyond the last position are re-introduced at
    the first.

    Parameters
    ----------
    a : array_like
        Input array.
    shift : int or tuple of ints
        The number of places by which elements are shifted.  If a tuple,
        then `axis` must be a tuple of the same size, and each of the
        given axes is shifted by the corresponding number.  If an int
        while `axis` is a tuple of ints, then the same value is used for
        all given axes.
    axis : int or tuple of ints, optional
        Axis or axes along which elements are shifted.  By default, the
        array is flattened before shifting, after which the original
        shape is restored.

    Returns
    -------
    res : ndarray
        Output array, with the same shape as `a`.

    See Also
    --------
    rollaxis : Roll the specified axis backwards, until it lies in a
               given position.

    Notes
    -----
    .. versionadded:: 1.12.0

    Supports rolling over multiple dimensions simultaneously.

    Examples
    --------
    >>> x = np.arange(10)
    >>> np.roll(x, 2)
    array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7])
    >>> np.roll(x, -2)
    array([2, 3, 4, 5, 6, 7, 8, 9, 0, 1])

    >>> x2 = np.reshape(x, (2, 5))
    >>> x2
    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])
    >>> np.roll(x2, 1)
    array([[9, 0, 1, 2, 3],
           [4, 5, 6, 7, 8]])
    >>> np.roll(x2, -1)
    array([[1, 2, 3, 4, 5],
           [6, 7, 8, 9, 0]])
    >>> np.roll(x2, 1, axis=0)
    array([[5, 6, 7, 8, 9],
           [0, 1, 2, 3, 4]])
    >>> np.roll(x2, -1, axis=0)
    array([[5, 6, 7, 8, 9],
           [0, 1, 2, 3, 4]])
    >>> np.roll(x2, 1, axis=1)
    array([[4, 0, 1, 2, 3],
           [9, 5, 6, 7, 8]])
    >>> np.roll(x2, -1, axis=1)
    array([[1, 2, 3, 4, 0],
           [6, 7, 8, 9, 5]])
    >>> np.roll(x2, (1, 1), axis=(1, 0))
    array([[9, 5, 6, 7, 8],
           [4, 0, 1, 2, 3]])
    >>> np.roll(x2, (2, 1), axis=(1, 0))
    array([[8, 9, 5, 6, 7],
           [3, 4, 0, 1, 2]])

    """
    # 将输入转换为数组
    a = asanyarray(a)
    # 如果未指定轴向，则将数组展平后再进行滚动操作，最后恢复原始形状
    if axis is None:
        return roll(a.ravel(), shift, 0).reshape(a.shape)
    else:
        # 规范化轴参数，允许重复，并返回规范化后的轴元组
        axis = normalize_axis_tuple(axis, a.ndim, allow_duplicate=True)
        # 对 shift 进行广播，使其与 axis 兼容
        broadcasted = broadcast(shift, axis)
        # 如果广播后的维度大于 1，抛出 ValueError 异常
        if broadcasted.ndim > 1:
            raise ValueError(
                "'shift' and 'axis' should be scalars or 1D sequences")
        # 初始化一个字典 shifts，用于存储每个轴的偏移量
        shifts = {ax: 0 for ax in range(a.ndim)}
        # 遍历广播后的偏移量和对应的轴，并累加到 shifts 字典中
        for sh, ax in broadcasted:
            shifts[ax] += sh

        # 初始化 rolls 列表，每个元素都是一个元组，用于描述切片操作
        rolls = [((slice(None), slice(None)),)] * a.ndim
        # 遍历 shifts 字典中的每个轴和对应的偏移量
        for ax, offset in shifts.items():
            # 计算偏移量取模后的值，如果 a 是空的，则取 1
            offset %= a.shape[ax] or 1  # If `a` is empty, nothing matters.
            # 如果偏移量不为 0，则更新 rolls 列表中对应轴的切片操作元组
            if offset:
                # 更新 rolls 中对应轴的切片操作
                rolls[ax] = ((slice(None, -offset), slice(offset, None)),
                             (slice(-offset, None), slice(None, offset)))

        # 初始化结果数组，形状与 a 相同
        result = empty_like(a)
        # 使用 itertools.product 遍历 rolls 中所有可能的索引组合
        for indices in itertools.product(*rolls):
            # 将 arr_index 和 res_index 分别解压缩为两个元组
            arr_index, res_index = zip(*indices)
            # 将 a 中的数据根据 arr_index 复制到 result 的对应位置 res_index
            result[res_index] = a[arr_index]

        # 返回结果数组
        return result
# 根据传入的参数 `a`、`axis` 和可选参数 `start` 确定派发函数的实现。
def _rollaxis_dispatcher(a, axis, start=None):
    # 返回包含 `a` 的元组，作为派发函数的结果
    return (a,)

# 使用装饰器 `array_function_dispatch` 来声明 `rollaxis` 函数的分派规则
@array_function_dispatch(_rollaxis_dispatcher)
def rollaxis(a, axis, start=0):
    """
    Roll the specified axis backwards, until it lies in a given position.

    This function continues to be supported for backward compatibility, but you
    should prefer `moveaxis`. The `moveaxis` function was added in NumPy
    1.11.

    Parameters
    ----------
    a : ndarray
        Input array.
    axis : int
        The axis to be rolled. The positions of the other axes do not
        change relative to one another.
    start : int, optional
        When ``start <= axis``, the axis is rolled back until it lies in
        this position. When ``start > axis``, the axis is rolled until it
        lies before this position. The default, 0, results in a "complete"
        roll. The following table describes how negative values of ``start``
        are interpreted:

        .. table::
           :align: left

           +-------------------+----------------------+
           |     ``start``     | Normalized ``start`` |
           +===================+======================+
           | ``-(arr.ndim+1)`` | raise ``AxisError``  |
           +-------------------+----------------------+
           | ``-arr.ndim``     | 0                    |
           +-------------------+----------------------+
           | |vdots|           | |vdots|              |
           +-------------------+----------------------+
           | ``-1``            | ``arr.ndim-1``       |
           +-------------------+----------------------+
           | ``0``             | ``0``                |
           +-------------------+----------------------+
           | |vdots|           | |vdots|              |
           +-------------------+----------------------+
           | ``arr.ndim``      | ``arr.ndim``         |
           +-------------------+----------------------+
           | ``arr.ndim + 1``  | raise ``AxisError``  |
           +-------------------+----------------------+

        .. |vdots|   unicode:: U+22EE .. Vertical Ellipsis

    Returns
    -------
    res : ndarray
        For NumPy >= 1.10.0 a view of `a` is always returned. For earlier
        NumPy versions a view of `a` is returned only if the order of the
        axes is changed, otherwise the input array is returned.

    See Also
    --------
    moveaxis : Move array axes to new positions.
    roll : Roll the elements of an array by a number of positions along a
        given axis.

    Examples
    --------
    >>> a = np.ones((3,4,5,6))
    >>> np.rollaxis(a, 3, 1).shape
    (3, 6, 4, 5)
    >>> np.rollaxis(a, 2).shape
    (5, 3, 4, 6)
    >>> np.rollaxis(a, 1, 4).shape
    (3, 5, 6, 4)

    """
    # 获取数组 `a` 的维度数
    n = a.ndim
    # 标准化 `axis`，确保其在合法范围内
    axis = normalize_axis_index(axis, n)
    # 处理 `start` 参数为负数的情况，将其转换为非负数索引
    if start < 0:
        start += n
    # 创建错误消息模板，用于验证 `start` 参数的合法性
    msg = "'%s' arg requires %d <= %s < %d, but %d was passed in"
    # 检查起始索引是否在合法范围内（0到n之间），如果不是则引发 AxisError 异常
    if not (0 <= start < n + 1):
        raise AxisError(msg % ('start', -n, 'start', n + 1, start))
    
    # 如果轴（axis）小于起始索引（start），说明轴已被移除，需要将起始索引减一
    if axis < start:
        # it's been removed
        start -= 1
    
    # 如果轴（axis）等于起始索引（start），直接返回整个数组 a
    if axis == start:
        return a[...]
    
    # 创建一个包含从 0 到 n-1 的整数列表，表示所有轴的索引
    axes = list(range(0, n))
    
    # 从 axes 列表中移除原始轴（axis）
    axes.remove(axis)
    
    # 将轴（axis）插入到起始索引（start）的位置
    axes.insert(start, axis)
    
    # 根据重新排列的轴顺序进行数组 a 的转置操作，并返回结果
    return a.transpose(axes)
# 设置函数的模块为 "numpy.lib.array_utils"
@set_module("numpy.lib.array_utils")
def normalize_axis_tuple(axis, ndim, argname=None, allow_duplicate=False):
    """
    Normalizes an axis argument into a tuple of non-negative integer axes.

    This handles shorthands such as ``1`` and converts them to ``(1,)``,
    as well as performing the handling of negative indices covered by
    `normalize_axis_index`.

    By default, this forbids axes from being specified multiple times.

    Used internally by multi-axis-checking logic.

    .. versionadded:: 1.13.0

    Parameters
    ----------
    axis : int, iterable of int
        The un-normalized index or indices of the axis.
    ndim : int
        The number of dimensions of the array that `axis` should be normalized
        against.
    argname : str, optional
        A prefix to put before the error message, typically the name of the
        argument.
    allow_duplicate : bool, optional
        If False, the default, disallow an axis from being specified twice.

    Returns
    -------
    normalized_axes : tuple of int
        The normalized axis index, such that `0 <= normalized_axis < ndim`

    Raises
    ------
    AxisError
        If any axis provided is out of range
    ValueError
        If an axis is repeated

    See also
    --------
    normalize_axis_index : normalizing a single scalar axis
    """
    # 如果 axis 不是 tuple 或 list 类型，尝试将其转换为包含单个元素的列表
    if type(axis) not in (tuple, list):
        try:
            axis = [operator.index(axis)]
        except TypeError:
            pass
    # 使用列表推导式对每个轴进行归一化处理，返回一个元组
    axis = tuple([normalize_axis_index(ax, ndim, argname) for ax in axis])
    # 如果不允许重复轴，并且存在重复的轴，则抛出 ValueError 异常
    if not allow_duplicate and len(set(axis)) != len(axis):
        if argname:
            raise ValueError('repeated axis in `{}` argument'.format(argname))
        else:
            raise ValueError('repeated axis')
    return axis


def _moveaxis_dispatcher(a, source, destination):
    return (a,)


@array_function_dispatch(_moveaxis_dispatcher)
def moveaxis(a, source, destination):
    """
    Move axes of an array to new positions.

    Other axes remain in their original order.

    .. versionadded:: 1.11.0

    Parameters
    ----------
    a : np.ndarray
        The array whose axes should be reordered.
    source : int or sequence of int
        Original positions of the axes to move. These must be unique.
    destination : int or sequence of int
        Destination positions for each of the original axes. These must also be
        unique.

    Returns
    -------
    result : np.ndarray
        Array with moved axes. This array is a view of the input array.

    See Also
    --------
    transpose : Permute the dimensions of an array.
    swapaxes : Interchange two axes of an array.

    Examples
    --------
    >>> x = np.zeros((3, 4, 5))
    >>> np.moveaxis(x, 0, -1).shape
    (4, 5, 3)
    >>> np.moveaxis(x, -1, 0).shape
    (5, 3, 4)

    These all achieve the same result:
    """
    >>> np.transpose(x).shape
    (5, 4, 3)
    >>> np.swapaxes(x, 0, -1).shape
    (5, 4, 3)
    >>> np.moveaxis(x, [0, 1], [-1, -2]).shape
    (5, 4, 3)
    >>> np.moveaxis(x, [0, 1, 2], [-1, -2, -3]).shape
    (5, 4, 3)

    """
    # 尝试从对象 `a` 中获取 `transpose` 属性，如果不存在则转换为数组再获取
    try:
        transpose = a.transpose
    except AttributeError:
        a = asarray(a)
        transpose = a.transpose

    # 规范化 `source` 和 `destination`，确保它们符合轴的范围
    source = normalize_axis_tuple(source, a.ndim, 'source')
    destination = normalize_axis_tuple(destination, a.ndim, 'destination')
    # 检查 `source` 和 `destination` 是否具有相同数量的元素，否则引发异常
    if len(source) != len(destination):
        raise ValueError('`source` and `destination` arguments must have '
                         'the same number of elements')

    # 创建一个排列 `order`，根据 `source` 和 `destination` 来重新排序轴
    order = [n for n in range(a.ndim) if n not in source]

    # 将 `destination` 和 `source` 组合并按 `destination` 的顺序插入 `order` 中
    for dest, src in sorted(zip(destination, source)):
        order.insert(dest, src)

    # 应用排列 `order` 到 `transpose` 函数，并返回结果
    result = transpose(order)
    return result
# 使用数组函数调度器将函数 _cross_dispatcher 设置为 cross 函数的调度器，负责分发不同的输入情况
@array_function_dispatch(_cross_dispatcher)
def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
    """
    Return the cross product of two (arrays of) vectors.

    The cross product of `a` and `b` in :math:`R^3` is a vector perpendicular
    to both `a` and `b`.  If `a` and `b` are arrays of vectors, the vectors
    are defined by the last axis of `a` and `b` by default, and these axes
    can have dimensions 2 or 3.  Where the dimension of either `a` or `b` is
    2, the third component of the input vector is assumed to be zero and the
    cross product calculated accordingly.  In cases where both input vectors
    have dimension 2, the z-component of the cross product is returned.

    Parameters
    ----------
    a : array_like
        Components of the first vector(s).
    b : array_like
        Components of the second vector(s).
    axisa : int, optional
        Axis of `a` that defines the vector(s).  By default, the last axis.
    axisb : int, optional
        Axis of `b` that defines the vector(s).  By default, the last axis.
    axisc : int, optional
        Axis of `c` containing the cross product vector(s).  Ignored if
        both input vectors have dimension 2, as the return is scalar.
        By default, the last axis.
    axis : int, optional
        If defined, the axis of `a`, `b` and `c` that defines the vector(s)
        and cross product(s).  Overrides `axisa`, `axisb` and `axisc`.

    Returns
    -------
    c : ndarray
        Vector cross product(s).

    Raises
    ------
    ValueError
        When the dimension of the vector(s) in `a` and/or `b` does not
        equal 2 or 3.

    See Also
    --------
    inner : Inner product
    outer : Outer product.
    linalg.cross : An Array API compatible variation of ``np.cross``,
                   which accepts (arrays of) 3-element vectors only.
    ix_ : Construct index arrays.

    Notes
    -----
    .. versionadded:: 1.9.0

    Supports full broadcasting of the inputs.

    Dimension-2 input arrays were deprecated in 2.0.0. If you do need this
    functionality, you can use::

        def cross2d(x, y):
            return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]

    Examples
    --------
    Vector cross-product.

    >>> x = [1, 2, 3]
    >>> y = [4, 5, 6]
    >>> np.cross(x, y)
    array([-3,  6, -3])

    One vector with dimension 2.

    >>> x = [1, 2]
    >>> y = [4, 5, 6]
    >>> np.cross(x, y)
    array([12, -6, -3])

    Equivalently:

    >>> x = [1, 2, 0]
    >>> y = [4, 5, 6]
    >>> np.cross(x, y)
    array([12, -6, -3])

    Both vectors with dimension 2.

    >>> x = [1,2]
    >>> y = [4,5]
    >>> np.cross(x, y)
    array(-3)

    Multiple vector cross-products. Note that the direction of the cross
    product vector is defined by the *right-hand rule*.

    >>> x = np.array([[1,2,3], [4,5,6]])
    >>> y = np.array([[4,5,6], [1,2,3]])
    """
    # 直接返回输入的两个数组 a 和 b，这是一个默认的实现，实际的计算和处理由 _cross_dispatcher 函数进行分派处理
    return (a, b)
    """
    Calculate the cross product of vectors `a` and `b` in NumPy.

    Parameters:
    ----------
    a, b : array_like
        Input arrays defining vectors.

    axisc : int, optional
        Specify the axis of `c` if provided (default is -1).

    axisa, axisb : int, optional
        Specify the axes of `a` and `b` respectively (default is -1).

    Returns:
    -------
    cp : ndarray
        Cross product of `a` and `b`.

    Raises:
    ------
    ValueError
        - If either `a` or `b` has zero dimension.
        - If the dimensions of `a` or `b` are not compatible for cross product (must be 2 or 3).
    DeprecationWarning
        - If either `a` or `b` are 2-dimensional arrays, as they are deprecated.

    Notes:
    ------
    The function calculates the cross product between vectors `a` and `b` along specified axes.
    If `a` or `b` are 2-dimensional, a warning is issued, recommending the use of 3-dimensional vectors.

    """

    # If `axis` is provided, assign its value to `axisa`, `axisb`, and `axisc`
    if axis is not None:
        axisa, axisb, axisc = (axis,) * 3

    # Ensure `a` and `b` are numpy arrays
    a = asarray(a)
    b = asarray(b)

    # Check if `a` or `b` has zero dimension
    if (a.ndim < 1) or (b.ndim < 1):
        raise ValueError("At least one array has zero dimension")

    # Normalize `axisa` and `axisb` to ensure they are within bounds
    axisa = normalize_axis_index(axisa, a.ndim, msg_prefix='axisa')
    axisb = normalize_axis_index(axisb, b.ndim, msg_prefix='axisb')

    # Move `axisa` and `axisb` to the end of the shape
    a = moveaxis(a, axisa, -1)
    b = moveaxis(b, axisb, -1)

    # Check dimensions are compatible for cross product (must be 2 or 3)
    msg = ("incompatible dimensions for cross product\n"
           "(dimension must be 2 or 3)")
    if a.shape[-1] not in (2, 3) or b.shape[-1] not in (2, 3):
        raise ValueError(msg)

    # Issue deprecation warning if `a` or `b` are 2-dimensional
    if a.shape[-1] == 2 or b.shape[-1] == 2:
        warnings.warn(
            "Arrays of 2-dimensional vectors are deprecated. Use arrays of "
            "3-dimensional vectors instead. (deprecated in NumPy 2.0)",
            DeprecationWarning, stacklevel=2
        )

    # Create the output array `cp` with appropriate shape and dtype
    shape = broadcast(a[..., 0], b[..., 0]).shape
    if a.shape[-1] == 3 or b.shape[-1] == 3:
        shape += (3,)
        axisc = normalize_axis_index(axisc, len(shape), msg_prefix='axisc')
    dtype = promote_types(a.dtype, b.dtype)
    cp = empty(shape, dtype)

    # Recast arrays `a` and `b` as `dtype` for uniform operations
    a = a.astype(dtype)
    b = b.astype(dtype)

    # Create local aliases for readability
    a0 = a[..., 0]
    a1 = a[..., 1]
    if a.shape[-1] == 3:
        a2 = a[..., 2]
    b0 = b[..., 0]
    b1 = b[..., 1]
    if b.shape[-1] == 3:
        b2 = b[..., 2]
    if cp.ndim != 0 and cp.shape[-1] == 3:
        cp0 = cp[..., 0]
        cp1 = cp[..., 1]
        cp2 = cp[..., 2]

    # Perform cross product calculation based on dimensions of `a` and `b`
    if a.shape[-1] == 2:
        if b.shape[-1] == 2:
            # For 2-dimensional `a` and `b`, compute cross product
            multiply(a0, b1, out=cp)
            cp -= a1 * b0
            return cp
        else:
            # For 2-dimensional `a` and 3-dimensional `b`, compute cross product
            multiply(a1, b2, out=cp0)
            multiply(a0, b2, out=cp1)
            negative(cp1, out=cp1)
            multiply(a0, b1, out=cp2)
            cp2 -= a1 * b0
    else:
        # 确保数组 a 的最后一个维度是 3
        assert a.shape[-1] == 3
        if b.shape[-1] == 3:
            # 如果数组 b 的最后一个维度也是 3，则计算叉乘结果
            # cp0 = a1 * b2 - a2 * b1
            # cp1 = a2 * b0 - a0 * b2
            # cp2 = a0 * b1 - a1 * b0
            multiply(a1, b2, out=cp0)
            # 计算临时变量 tmp = a2 * b1
            tmp = array(a2 * b1)
            cp0 -= tmp
            multiply(a2, b0, out=cp1)
            multiply(a0, b2, out=tmp)
            cp1 -= tmp
            multiply(a0, b1, out=cp2)
            multiply(a1, b0, out=tmp)
            cp2 -= tmp
        else:
            # 如果数组 b 的最后一个维度是 2，则按照特定情况计算叉乘结果
            assert b.shape[-1] == 2
            # cp0 = 0 - a2 * b1  (因为 b2 = 0)
            # cp1 = a2 * b0 - 0  (因为 b2 = 0)
            # cp2 = a0 * b1 - a1 * b0
            multiply(a2, b1, out=cp0)
            negative(cp0, out=cp0)
            multiply(a2, b0, out=cp1)
            multiply(a0, b1, out=cp2)
            cp2 -= a1 * b0

    # 将计算结果 cp 的最后一个维度移到指定的轴向 axisc 处并返回
    return moveaxis(cp, -1, axisc)
# 根据系统的字节顺序确定是否为小端序
little_endian = (sys.byteorder == 'little')


@set_module('numpy')
# 将当前函数注册为 numpy 模块的 indices 函数
def indices(dimensions, dtype=int, sparse=False):
    """
    Return an array representing the indices of a grid.

    Compute an array where the subarrays contain index values 0, 1, ...
    varying only along the corresponding axis.

    Parameters
    ----------
    dimensions : sequence of ints
        The shape of the grid.
    dtype : dtype, optional
        Data type of the result.
    sparse : boolean, optional
        Return a sparse representation of the grid instead of a dense
        representation. Default is False.

        .. versionadded:: 1.17

    Returns
    -------
    grid : one ndarray or tuple of ndarrays
        If sparse is False:
            Returns one array of grid indices,
            ``grid.shape = (len(dimensions),) + tuple(dimensions)``.
        If sparse is True:
            Returns a tuple of arrays, with
            ``grid[i].shape = (1, ..., 1, dimensions[i], 1, ..., 1)`` with
            dimensions[i] in the ith place

    See Also
    --------
    mgrid, ogrid, meshgrid

    Notes
    -----
    The output shape in the dense case is obtained by prepending the number
    of dimensions in front of the tuple of dimensions, i.e. if `dimensions`
    is a tuple ``(r0, ..., rN-1)`` of length ``N``, the output shape is
    ``(N, r0, ..., rN-1)``.

    The subarrays ``grid[k]`` contains the N-D array of indices along the
    ``k-th`` axis. Explicitly::

        grid[k, i0, i1, ..., iN-1] = ik

    Examples
    --------
    >>> grid = np.indices((2, 3))
    >>> grid.shape
    (2, 2, 3)
    >>> grid[0]        # row indices
    array([[0, 0, 0],
           [1, 1, 1]])
    >>> grid[1]        # column indices
    array([[0, 1, 2],
           [0, 1, 2]])

    The indices can be used as an index into an array.

    >>> x = np.arange(20).reshape(5, 4)
    >>> row, col = np.indices((2, 3))
    >>> x[row, col]
    array([[0, 1, 2],
           [4, 5, 6]])

    Note that it would be more straightforward in the above example to
    extract the required elements directly with ``x[:2, :3]``.

    If sparse is set to true, the grid will be returned in a sparse
    representation.

    >>> i, j = np.indices((2, 3), sparse=True)
    >>> i.shape
    (2, 1)
    >>> j.shape
    (1, 3)
    >>> i        # row indices
    array([[0],
           [1]])
    >>> j        # column indices
    array([[0, 1, 2]])

    """
    dimensions = tuple(dimensions)
    N = len(dimensions)
    shape = (1,)*N
    # 根据 sparse 参数选择返回稠密或稀疏表示的结果数组
    if sparse:
        res = tuple()
    else:
        res = empty((N,)+dimensions, dtype=dtype)
    for i, dim in enumerate(dimensions):
        # 创建包含索引值的数组，每个子数组沿相应轴变化
        idx = arange(dim, dtype=dtype).reshape(
            shape[:i] + (dim,) + shape[i+1:]
        )
        if sparse:
            res = res + (idx,)
        else:
            res[i] = idx
    # 返回结果数组或元组
    return res


@set_array_function_like_doc
@set_module('numpy')
def fromfunction(function, shape, *, dtype=float, like=None, **kwargs):
    """
    Construct an array by executing a function over each coordinate.

    The resulting array therefore has a value ``fn(x, y, z)`` at
    coordinate ``(x, y, z)``.

    Parameters
    ----------
    function : callable
        The function is called with N parameters, where N is the rank of
        `shape`.  Each parameter represents the coordinates of the array
        varying along a specific axis.  For example, if `shape`
        were ``(2, 2)``, then the parameters would be
        ``array([[0, 0], [1, 1]])`` and ``array([[0, 1], [0, 1]])``
    shape : (N,) tuple of ints
        Shape of the output array, which also determines the shape of
        the coordinate arrays passed to `function`.
    dtype : data-type, optional
        Data-type of the coordinate arrays passed to `function`.
        By default, `dtype` is float.
    ${ARRAY_FUNCTION_LIKE}

        .. versionadded:: 1.20.0

    Returns
    -------
    fromfunction : any
        The result of the call to `function` is passed back directly.
        Therefore the shape of `fromfunction` is completely determined by
        `function`.  If `function` returns a scalar value, the shape of
        `fromfunction` would not match the `shape` parameter.

    See Also
    --------
    indices, meshgrid

    Notes
    -----
    Keywords other than `dtype` and `like` are passed to `function`.

    Examples
    --------
    >>> np.fromfunction(lambda i, j: i, (2, 2), dtype=float)
    array([[0., 0.],
           [1., 1.]])

    >>> np.fromfunction(lambda i, j: j, (2, 2), dtype=float)
    array([[0., 1.],
           [0., 1.]])

    >>> np.fromfunction(lambda i, j: i == j, (3, 3), dtype=int)
    array([[ True, False, False],
           [False,  True, False],
           [False, False,  True]])

    >>> np.fromfunction(lambda i, j: i + j, (3, 3), dtype=int)
    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4]])

    """
    # 如果给定了 like 参数，则调用 _fromfunction_with_like 函数处理
    if like is not None:
        return _fromfunction_with_like(
                like, function, shape, dtype=dtype, **kwargs)

    # 根据 shape 和 dtype 参数生成坐标数组 args
    args = indices(shape, dtype=dtype)
    # 调用 function 函数，并传入生成的坐标数组 args 和其他关键字参数 kwargs
    return function(*args, **kwargs)
# 将 `fromfunction` 函数注册为具有数组功能分派的函数，并赋值给 `_fromfunction_with_like`
_fromfunction_with_like = array_function_dispatch()(fromfunction)

# 从给定的缓冲区 `buf` 中创建一个数组，指定数据类型 `dtype`、形状 `shape` 和存储顺序 `order`，
# 然后将其重新塑形为指定形状和顺序的数组
def _frombuffer(buf, dtype, shape, order):
    return frombuffer(buf, dtype=dtype).reshape(shape, order=order)

# 将当前函数标记为属于 'numpy' 模块的函数
@set_module('numpy')
def isscalar(element):
    """
    Returns True if the type of `element` is a scalar type.

    Parameters
    ----------
    element : any
        Input argument, can be of any type and shape.

    Returns
    -------
    val : bool
        True if `element` is a scalar type, False if it is not.

    See Also
    --------
    ndim : Get the number of dimensions of an array

    Notes
    -----
    If you need a stricter way to identify a *numerical* scalar, use
    ``isinstance(x, numbers.Number)``, as that returns ``False`` for most
    non-numerical elements such as strings.

    In most cases ``np.ndim(x) == 0`` should be used instead of this function,
    as that will also return true for 0d arrays. This is how numpy overloads
    functions in the style of the ``dx`` arguments to `gradient` and
    the ``bins`` argument to `histogram`. Some key differences:

    +------------------------------------+---------------+-------------------+
    | x                                  |``isscalar(x)``|``np.ndim(x) == 0``|
    +====================================+===============+===================+
    | PEP 3141 numeric objects           | ``True``      | ``True``          |
    | (including builtins)               |               |                   |
    +------------------------------------+---------------+-------------------+
    | builtin string and buffer objects  | ``True``      | ``True``          |
    +------------------------------------+---------------+-------------------+
    | other builtin objects, like        | ``False``     | ``True``          |
    | `pathlib.Path`, `Exception`,       |               |                   |
    | the result of `re.compile`         |               |                   |
    +------------------------------------+---------------+-------------------+
    | third-party objects like           | ``False``     | ``True``          |
    | `matplotlib.figure.Figure`         |               |                   |
    +------------------------------------+---------------+-------------------+
    | zero-dimensional numpy arrays      | ``False``     | ``True``          |
    +------------------------------------+---------------+-------------------+
    | other numpy arrays                 | ``False``     | ``False``         |
    +------------------------------------+---------------+-------------------+
    | `list`, `tuple`, and other         | ``False``     | ``False``         |
    | sequence objects                   |               |                   |
    +------------------------------------+---------------+-------------------+

    Examples
    --------
    >>> np.isscalar(3.1)
    True
    >>> np.isscalar(np.array(3.1))
    False
    >>> np.isscalar([3.1])
    False
    >>> np.isscalar(False)
    True
    """
    # 检查给定元素是否是标量（scalar）
    >>> np.isscalar('numpy')
    # 返回 True，因为字符串 'numpy' 是一个标量
    
    NumPy 支持 PEP 3141 标量数值：
    
    # 导入 Fraction 类
    >>> from fractions import Fraction
    # 检查 Fraction(5, 17) 是否是标量
    >>> np.isscalar(Fraction(5, 17))
    # 返回 True，因为 Fraction(5, 17) 是一个标量
    
    # 导入 Number 类
    >>> from numbers import Number
    # 检查 Number() 是否是标量
    >>> np.isscalar(Number())
    # 返回 True，因为 Number() 是一个标量
    
    """
    返回一个布尔值，判断 element 是否是标量类型 generic 中的实例，
    或者 element 的类型是否在 ScalarType 中，
    或者 element 是否是 numbers.Number 的实例。
    """
    return (isinstance(element, generic)
            or type(element) in ScalarType
            or isinstance(element, numbers.Number))
@set_module('numpy')
# 设置模块为 'numpy'，这是一个装饰器函数，用于在函数定义时设置模块信息

def binary_repr(num, width=None):
    """
    Return the binary representation of the input number as a string.

    For negative numbers, if width is not given, a minus sign is added to the
    front. If width is given, the two's complement of the number is
    returned, with respect to that width.

    In a two's-complement system negative numbers are represented by the two's
    complement of the absolute value. This is the most common method of
    representing signed integers on computers [1]_. A N-bit two's-complement
    system can represent every integer in the range
    :math:`-2^{N-1}` to :math:`+2^{N-1}-1`.

    Parameters
    ----------
    num : int
        Only an integer decimal number can be used.
    width : int, optional
        The length of the returned string if `num` is positive, or the length
        of the two's complement if `num` is negative, provided that `width` is
        at least a sufficient number of bits for `num` to be represented in
        the designated form. If the `width` value is insufficient, an error is
        raised.

    Returns
    -------
    bin : str
        Binary representation of `num` or two's complement of `num`.

    See Also
    --------
    base_repr: Return a string representation of a number in the given base
               system.
    bin: Python's built-in binary representation generator of an integer.

    Notes
    -----
    `binary_repr` is equivalent to using `base_repr` with base 2, but about 25x
    faster.

    References
    ----------
    .. [1] Wikipedia, "Two's complement",
        https://en.wikipedia.org/wiki/Two's_complement

    Examples
    --------
    >>> np.binary_repr(3)
    '11'
    >>> np.binary_repr(-3)
    '-11'
    >>> np.binary_repr(3, width=4)
    '0011'

    The two's complement is returned when the input number is negative and
    width is specified:

    >>> np.binary_repr(-3, width=3)
    '101'
    >>> np.binary_repr(-3, width=5)
    '11101'

    """
    # 内部函数，用于检查指定的宽度是否足够容纳二进制表示，如果不足则引发 ValueError 异常
    def err_if_insufficient(width, binwidth):
        if width is not None and width < binwidth:
            raise ValueError(
                f"Insufficient bit {width=} provided for {binwidth=}"
            )

    # 确保 num 是一个 Python 整数，避免溢出或不必要的浮点转换
    num = operator.index(num)

    if num == 0:
        return '0' * (width or 1)
    elif num > 0:
        # 获取 num 的二进制表示，并移除前缀 '0b'
        binary = bin(num)[2:]
        binwidth = len(binary)  # 计算二进制字符串的长度
        # 计算输出字符串的宽度，如果指定了 width，则取二者的最大值
        outwidth = (binwidth if width is None
                    else builtins.max(binwidth, width))
        # 检查宽度是否足够，如果不足则引发异常
        err_if_insufficient(width, binwidth)
        # 左侧填充二进制字符串，使其达到指定的宽度
        return binary.zfill(outwidth)
    else:
        if width is None:
            # 如果宽度未指定，生成补码表示的负数的二进制字符串
            return '-' + bin(-num)[2:]
        else:
            # 计算补码表示的负数的二进制字符串的长度
            poswidth = len(bin(-num)[2:])

            # 查看 GitHub 问题 #8679：移除边界处数字的额外位数
            if 2**(poswidth - 1) == -num:
                # 如果数字处于边界，减少位数
                poswidth -= 1

            # 计算补码表示的数值
            twocomp = 2**(poswidth + 1) + num
            binary = bin(twocomp)[2:]  # 转换为二进制字符串
            binwidth = len(binary)  # 计算二进制字符串的长度

            # 确定输出的宽度
            outwidth = builtins.max(binwidth, width)
            # 如果宽度不足，则引发错误
            err_if_insufficient(width, binwidth)
            # 返回二进制字符串，左侧用 '1' 填充至指定宽度
            return '1' * (outwidth - binwidth) + binary
# 将当前模块设置为'numpy'
@set_module('numpy')
# 定义函数base_repr，用于将整数number转换为指定base进制的字符串表示形式，可指定左侧填充的零数padding
def base_repr(number, base=2, padding=0):
    """
    Return a string representation of a number in the given base system.

    Parameters
    ----------
    number : int
        The value to convert. Positive and negative values are handled.
    base : int, optional
        Convert `number` to the `base` number system. The valid range is 2-36,
        the default value is 2.
    padding : int, optional
        Number of zeros padded on the left. Default is 0 (no padding).

    Returns
    -------
    out : str
        String representation of `number` in `base` system.

    See Also
    --------
    binary_repr : Faster version of `base_repr` for base 2.

    Examples
    --------
    >>> np.base_repr(5)
    '101'
    >>> np.base_repr(6, 5)
    '11'
    >>> np.base_repr(7, base=5, padding=3)
    '00012'

    >>> np.base_repr(10, base=16)
    'A'
    >>> np.base_repr(32, base=16)
    '20'

    """
    # 定义所有可能的数字字符，用于各进制的表示
    digits = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    # 如果指定的进制超过了digits中的字符数，抛出异常
    if base > len(digits):
        raise ValueError("Bases greater than 36 not handled in base_repr.")
    # 如果指定的进制小于2，抛出异常
    elif base < 2:
        raise ValueError("Bases less than 2 not handled in base_repr.")

    # 取number的绝对值
    num = abs(int(number))
    res = []
    # 将number转换为base进制的字符串表示形式
    while num:
        res.append(digits[num % base])
        num //= base
    # 如果需要填充左侧零，则添加
    if padding:
        res.append('0' * padding)
    # 如果number为负数，则在结果前加上负号
    if number < 0:
        res.append('-')
    # 将列表res反转并连接成字符串，作为结果返回
    return ''.join(reversed(res or '0'))


# 下面这些都基本上是缩写
# 这些可能最终会出现在一个特殊的缩写模块中


# 定义私有函数_maketup，根据描述符descr和值val创建元组
def _maketup(descr, val):
    # 根据描述符创建数据类型对象dt
    dt = dtype(descr)
    # 如果dt没有字段，则直接返回val
    fields = dt.fields
    if fields is None:
        return val
    else:
        # 否则，递归调用_maketup，为每个字段创建元组，最后返回元组形式的结果
        res = [_maketup(fields[name][0], val) for name in dt.names]
        return tuple(res)


# 将当前函数设置为数组函数的文档模式
@set_array_function_like_doc
# 将当前模块设置为'numpy'
@set_module('numpy')
# 定义函数identity，返回一个单位矩阵数组
def identity(n, dtype=None, *, like=None):
    """
    Return the identity array.

    The identity array is a square array with ones on
    the main diagonal.

    Parameters
    ----------
    n : int
        Number of rows (and columns) in `n` x `n` output.
    dtype : data-type, optional
        Data-type of the output.  Defaults to ``float``.
    ${ARRAY_FUNCTION_LIKE}

        .. versionadded:: 1.20.0

    Returns
    -------
    out : ndarray
        `n` x `n` array with its main diagonal set to one,
        and all other elements 0.

    Examples
    --------
    >>> np.identity(3)
    array([[1.,  0.,  0.],
           [0.,  1.,  0.],
           [0.,  0.,  1.]])

    """
    # 如果指定了like参数，调用_identity_with_like函数进行处理
    if like is not None:
        return _identity_with_like(like, n, dtype=dtype)

    # 从numpy模块导入eye函数，返回一个单位矩阵数组
    from numpy import eye
    return eye(n, dtype=dtype, like=like)


# 将_identity_with_like函数通过array_function_dispatch装饰器进行分发
_identity_with_like = array_function_dispatch()(identity)


# 定义_allclose_dispatcher函数，返回传入的a、b、rtol、atol参数元组
def _allclose_dispatcher(a, b, rtol=None, atol=None, equal_nan=None):
    return (a, b, rtol, atol)


# 将_allclose_dispatcher函数通过array_function_dispatch装饰器进行分发
@array_function_dispatch(_allclose_dispatcher)
# 定义函数allclose，用于比较两个数组的元素是否在公差范围内相等
def allclose(a, b, rtol=1.e-5, atol=1.e-8, equal_nan=False):
    """
    # 比较两个数组是否在容差范围内逐元素相等。
    
    # 容差值为正数，通常是非常小的数。相对差异 (`rtol` * abs(`b`)) 和绝对差异 `atol` 被加在一起，用来与 `a` 和 `b` 的绝对差异比较。
    
    # .. warning:: 默认的 `atol` 不适用于比较远小于一的数值（参见注释）。
    
    # 如果NaN在同一位置且 ``equal_nan=True``，则被视为相等。如果Inf在同一位置且在两个数组中的符号相同，则被视为相等。
    
    # 参数
    # ----------
    # a, b : array_like
    #     要比较的输入数组。
    # rtol : array_like
    #     相对容差参数（见注释）。
    # atol : array_like
    #     绝对容差参数（见注释）。
    # equal_nan : bool
    #     是否将NaN视为相等。如果为True，则 `a` 中的NaN将与 `b` 中的NaN在输出数组中被视为相等。
    
    #     .. versionadded:: 1.10.0
    
    # 返回
    # -------
    # allclose : bool
    #     如果两个数组在给定的容差内相等，则返回True；否则返回False。
    
    # 参见
    # --------
    # isclose, all, any, equal
    
    # 注释
    # -----
    # 如果以下方程逐元素为True，则 `allclose` 返回True。::
    
    #     absolute(a - b) <= (atol + rtol * absolute(b))
    
    # 上述方程在 `a` 和 `b` 中不对称，因此在某些罕见情况下，``allclose(a, b)`` 可能与 ``allclose(b, a)`` 不同。
    
    # 当参考值 `b` 的幅度小于一时，默认值 `atol` 是不合适的。例如， ``a = 1e-9`` 和 ``b = 2e-9`` 可能不应被视为 "接近"，但是 ``allclose(1e-9, 2e-9)`` 在默认设置下返回True。请确保根据具体情况选择 `atol`，特别是用于定义 `a` 中非零值与 `b` 中非常小或零值之间的阈值。
    
    # `a` 和 `b` 的比较使用标准的广播方式，这意味着 `a` 和 `b` 不需要具有相同的形状才能使 ``allclose(a, b)`` 评估为True。对于 `equal` 也是如此，但对于 `array_equal` 则不是。
    
    # `allclose` 对于非数值数据类型未定义。
    # 对于此目的，`bool` 被视为数值数据类型。
    
    # 示例
    # --------
    # >>> np.allclose([1e10,1e-7], [1.00001e10,1e-8])
    # False
    # >>> np.allclose([1e10,1e-8], [1.00001e10,1e-9])
    # True
    # >>> np.allclose([1e10,1e-8], [1.0001e10,1e-9])
    # False
    # >>> np.allclose([1.0, np.nan], [1.0, np.nan])
    # False
    # >>> np.allclose([1.0, np.nan], [1.0, np.nan], equal_nan=True)
    # True
# 根据输入参数创建一个调度器，用于分派到合适的 isclose 函数
def _isclose_dispatcher(a, b, rtol=None, atol=None, equal_nan=None):
    # 返回输入参数元组，用于后续的分派
    return (a, b, rtol, atol)

# 使用 array_function_dispatch 装饰器，将 _isclose_dispatcher 函数作为分派器，用于 isclose 函数
@array_function_dispatch(_isclose_dispatcher)
def isclose(a, b, rtol=1.e-5, atol=1.e-8, equal_nan=False):
    """
    Returns a boolean array where two arrays are element-wise equal within a
    tolerance.

    The tolerance values are positive, typically very small numbers.  The
    relative difference (`rtol` * abs(`b`)) and the absolute difference
    `atol` are added together to compare against the absolute difference
    between `a` and `b`.

    .. warning:: The default `atol` is not appropriate for comparing numbers
                 with magnitudes much smaller than one (see Notes).

    Parameters
    ----------
    a, b : array_like
        Input arrays to compare.
    rtol : array_like
        The relative tolerance parameter (see Notes).
    atol : array_like
        The absolute tolerance parameter (see Notes).
    equal_nan : bool
        Whether to compare NaN's as equal.  If True, NaN's in `a` will be
        considered equal to NaN's in `b` in the output array.

    Returns
    -------
    y : array_like
        Returns a boolean array of where `a` and `b` are equal within the
        given tolerance. If both `a` and `b` are scalars, returns a single
        boolean value.

    See Also
    --------
    allclose
    math.isclose

    Notes
    -----
    .. versionadded:: 1.7.0

    For finite values, isclose uses the following equation to test whether
    two floating point values are equivalent.::

     absolute(a - b) <= (atol + rtol * absolute(b))

    Unlike the built-in `math.isclose`, the above equation is not symmetric
    in `a` and `b` -- it assumes `b` is the reference value -- so that
    `isclose(a, b)` might be different from `isclose(b, a)`.

    The default value of `atol` is not appropriate when the reference value
    `b` has magnitude smaller than one. For example, it is unlikely that
    ``a = 1e-9`` and ``b = 2e-9`` should be considered "close", yet
    ``isclose(1e-9, 2e-9)`` is ``True`` with default settings. Be sure
    to select `atol` for the use case at hand, especially for defining the
    threshold below which a non-zero value in `a` will be considered "close"
    to a very small or zero value in `b`.

    `isclose` is not defined for non-numeric data types.
    :class:`bool` is considered a numeric data-type for this purpose.

    Examples
    --------
    >>> np.isclose([1e10,1e-7], [1.00001e10,1e-8])
    array([ True, False])
    >>> np.isclose([1e10,1e-8], [1.00001e10,1e-9])
    array([ True, True])
    >>> np.isclose([1e10,1e-8], [1.0001e10,1e-9])
    array([False,  True])
    >>> np.isclose([1.0, np.nan], [1.0, np.nan])
    array([ True, False])
    >>> np.isclose([1.0, np.nan], [1.0, np.nan], equal_nan=True)
    array([ True, True])
    >>> np.isclose([1e-8, 1e-7], [0.0, 0.0])
    array([ True, False])
    >>> np.isclose([1e-100, 1e-7], [0.0, 0.0], atol=0.0)
    array([False, False])
    """
    # 使用 NumPy 的 isclose 函数比较两个数组的元素是否在误差范围内相等
    >>> np.isclose([1e-10, 1e-10], [1e-20, 0.0])
    array([ True,  True])
    
    # 使用 NumPy 的 isclose 函数比较两个数组的元素是否在指定的绝对误差范围内相等
    >>> np.isclose([1e-10, 1e-10], [1e-20, 0.999999e-10], atol=0.0)
    array([False,  True])
    """
    # 将除了 Python 标量之外的对象转换为数组
    x, y, atol, rtol = (
        a if isinstance(a, (int, float, complex)) else asanyarray(a)
        for a in (a, b, atol, rtol))
    
    # 确保 y 是一个非精确类型，以避免在 abs(MIN_INT) 上出现错误行为
    # 这将导致稍后将 x 转换为该类型。同时，确保允许子类（例如 numpy.ma）
    # 注意：我们明确允许 timedelta，这在过去有效。这可能会被弃用。另请参见 gh-18286。
    # timedelta 在 `atol` 是整数或 timedelta 时有效。
    # 尽管如此，默认的公差可能不太有用
    if (dtype := getattr(y, "dtype", None)) is not None and dtype.kind != "m":
        dt = multiarray.result_type(y, 1.)
        y = asanyarray(y, dtype=dt)
    elif isinstance(y, int):
        y = float(y)
    
    # 使用 errstate 来忽略无效值的警告，并通过 _no_nep50_warning 确保不显示 NEP 50 的警告
    with errstate(invalid='ignore'), _no_nep50_warning():
        # 计算两数组之间的相等性
        result = (less_equal(abs(x-y), atol + rtol * abs(y))
                  & isfinite(y)
                  | (x == y))
        # 如果需要，处理 NaN 的情况
        if equal_nan:
            result |= isnan(x) & isnan(y)
    
    # 返回扁平化的零维数组结果作为标量
    return result[()]
def _array_equal_dispatcher(a1, a2, equal_nan=None):
    # 将输入的两个数组作为一个元组返回，用于数组相等性判断的调度器
    return (a1, a2)


_no_nan_types = {
    # 不包含可以容纳 NaN 值的数据类型集合，这些类型不支持 NaN 比较
    # 应该使用 np.dtype.BoolDType，但在写作时未通过重新加载测试
    type(dtype(nt.bool)),
    type(dtype(nt.int8)),
    type(dtype(nt.int16)),
    type(dtype(nt.int32)),
    type(dtype(nt.int64)),
}


def _dtype_cannot_hold_nan(dtype):
    # 判断给定的数据类型是否在 _no_nan_types 集合中，即是否不能容纳 NaN 值
    return type(dtype) in _no_nan_types


@array_function_dispatch(_array_equal_dispatcher)
def array_equal(a1, a2, equal_nan=False):
    """
    如果两个数组具有相同的形状和元素，则返回 True，否则返回 False。

    Parameters
    ----------
    a1, a2 : array_like
        输入数组。
    equal_nan : bool
        是否将 NaN 视为相等。如果 a1 和 a2 的 dtype 是复数，则如果给定值的实部或虚部为 NaN，则视为相等。

        .. versionadded:: 1.19.0

    Returns
    -------
    b : bool
        如果数组相等则返回 True。

    See Also
    --------
    allclose: 如果两个数组在容差范围内逐元素相等，则返回 True。
    array_equiv: 如果输入数组在形状上一致且所有元素相等，则返回 True。

    Examples
    --------
    >>> np.array_equal([1, 2], [1, 2])
    True
    >>> np.array_equal(np.array([1, 2]), np.array([1, 2]))
    True
    >>> np.array_equal([1, 2], [1, 2, 3])
    False
    >>> np.array_equal([1, 2], [1, 4])
    False
    >>> a = np.array([1, np.nan])
    >>> np.array_equal(a, a)
    False
    >>> np.array_equal(a, a, equal_nan=True)
    True

    当 equal_nan 为 True 时，如果复数值的组成部分为 NaN，则认为它们是相等的。

    >>> a = np.array([1 + 1j])
    >>> b = a.copy()
    >>> a.real = np.nan
    >>> b.imag = np.nan
    >>> np.array_equal(a, b, equal_nan=True)
    True
    """
    try:
        a1, a2 = asarray(a1), asarray(a2)
    except Exception:
        return False
    if a1.shape != a2.shape:
        return False
    if not equal_nan:
        return builtins.bool((a1 == a2).all())
    cannot_have_nan = (_dtype_cannot_hold_nan(a1.dtype)
                       and _dtype_cannot_hold_nan(a2.dtype))
    if cannot_have_nan:
        if a1 is a2:
            return True
        return builtins.bool((a1 == a2).all())

    if a1 is a2:
        # NaN 将被视为相等，因此数组将与自身比较相等。
        return True
    # 如果 equal_nan 为 True，则处理 NaN 值
    a1nan, a2nan = isnan(a1), isnan(a2)
    # NaN 出现在不同位置
    if not (a1nan == a2nan).all():
        return False
    # 到这一步，a1、a2 和掩码的形状保证一致
    return builtins.bool((a1[~a1nan] == a2[~a1nan]).all())


def _array_equiv_dispatcher(a1, a2):
    # 将输入的两个数组作为一个元组返回，用于数组等价性判断的调度器
    return (a1, a2)


@array_function_dispatch(_array_equiv_dispatcher)
def array_equiv(a1, a2):
    """
    Returns True if input arrays are shape consistent and all elements equal.

    Shape consistent means they are either the same shape, or one input array
    can be broadcasted to create the same shape as the other one.

    Parameters
    ----------
    a1, a2 : array_like
        Input arrays.

    Returns
    -------
    out : bool
        True if equivalent, False otherwise.

    Examples
    --------
    >>> np.array_equiv([1, 2], [1, 2])
    True
    >>> np.array_equiv([1, 2], [1, 3])
    False

    Showing the shape equivalence:

    >>> np.array_equiv([1, 2], [[1, 2], [1, 2]])
    True
    >>> np.array_equiv([1, 2], [[1, 2, 1, 2], [1, 2, 1, 2]])
    False

    >>> np.array_equiv([1, 2], [[1, 2], [1, 3]])
    False

    """
    # 尝试将输入的 a1 和 a2 转换为数组
    try:
        a1, a2 = asarray(a1), asarray(a2)
    except Exception:
        # 转换失败则返回 False
        return False
    # 尝试使用广播来匹配数组的形状
    try:
        multiarray.broadcast(a1, a2)
    except Exception:
        # 广播失败则返回 False
        return False

    # 检查数组 a1 和 a2 的所有元素是否完全相等，返回比较结果
    return builtins.bool((a1 == a2).all())
# 定义一个函数 _astype_dispatcher，用于类型分发，接收参数 x, dtype 和可选的 copy
def _astype_dispatcher(x, dtype, /, *, copy=None):
    # 返回一个元组，包含参数 x 和 dtype
    return (x, dtype)


# 使用 array_function_dispatch 装饰器将 _astype_dispatcher 函数注册为 astype 函数的分发器
@array_function_dispatch(_astype_dispatcher)
# 定义 astype 函数，用于将数组复制到指定的数据类型
def astype(x, dtype, /, *, copy = True):
    """
    Copies an array to a specified data type.

    This function is an Array API compatible alternative to
    `numpy.ndarray.astype`.

    Parameters
    ----------
    x : ndarray
        Input NumPy array to cast. ``array_likes`` are explicitly not
        supported here.
    dtype : dtype
        Data type of the result.
    copy : bool, optional
        Specifies whether to copy an array when the specified dtype matches
        the data type of the input array ``x``. If ``True``, a newly allocated
        array must always be returned. If ``False`` and the specified dtype
        matches the data type of the input array, the input array must be
        returned; otherwise, a newly allocated array must be returned.
        Defaults to ``True``.

    Returns
    -------
    out : ndarray
        An array having the specified data type.

    See Also
    --------
    ndarray.astype

    Examples
    --------
    >>> arr = np.array([1, 2, 3]); arr
    array([1, 2, 3])
    >>> np.astype(arr, np.float64)
    array([1., 2., 3.])

    Non-copy case:

    >>> arr = np.array([1, 2, 3])
    >>> arr_noncpy = np.astype(arr, arr.dtype, copy=False)
    >>> np.shares_memory(arr, arr_noncpy)
    True

    """
    # 如果输入不是 NumPy 数组，则抛出 TypeError 异常
    if not isinstance(x, np.ndarray):
        raise TypeError(
            f"Input should be a NumPy array. It is a {type(x)} instead."
        )
    # 调用输入数组 x 的 astype 方法，将其转换为指定的数据类型 dtype，并根据 copy 参数决定是否复制数组
    return x.astype(dtype, copy=copy)


# 定义 inf 为正无穷大常量 PINF
inf = PINF
# 定义 nan 为 NaN 常量 NAN
nan = NAN
# 定义 False_ 为布尔类型 False 的别名
False_ = nt.bool(False)
# 定义 True_ 为布尔类型 True 的别名
True_ = nt.bool(True)


# 定义 extend_all 函数，用于向 __all__ 中扩展模块的所有内容
def extend_all(module):
    # 获取当前 __all__ 的内容，并转换为集合
    existing = set(__all__)
    # 获取给定模块的 __all__ 属性
    mall = getattr(module, '__all__')
    # 遍历模块的 __all__ 中的每个元素
    for a in mall:
        # 如果元素不在现有的 __all__ 中，则将其添加到 __all__ 中
        if a not in existing:
            __all__.append(a)


# 导入 umath 模块中的所有内容
from .umath import *
# 导入 numerictypes 模块中的所有内容
from .numerictypes import *
# 导入 fromnumeric 模块
from . import fromnumeric
# 从 fromnumeric 模块中导入所有内容
from .fromnumeric import *
# 导入 arrayprint 模块
from . import arrayprint
# 从 arrayprint 模块中导入所有内容
from .arrayprint import *
# 导入 _asarray 模块
from . import _asarray
# 从 _asarray 模块中导入所有内容
from ._asarray import *
# 导入 _ufunc_config 模块
from . import _ufunc_config
# 从 _ufunc_config 模块中导入所有内容
from ._ufunc_config import *

# 扩展 __all__ 列表以包含来自不同模块的内容
extend_all(fromnumeric)
extend_all(umath)
extend_all(numerictypes)
extend_all(arrayprint)
extend_all(_asarray)
extend_all(_ufunc_config)
```